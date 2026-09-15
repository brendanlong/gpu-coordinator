"""Runs exactly one job: environment, preflight, watchdogs, sync, exit code.

Exit-code discipline (measured in the shell implementation this replaces):
the job's exit code is captured before *any* cleanup, and a failed final sync
turns an otherwise-green job into `failed: sync` so lost outputs are never
silent.
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import IO

from gpuc.host import gpus, jobs, paths, queue, sync
from gpuc.host.gpus import SmiRunner
from gpuc.host.jobs import JobSpec

KILL_GRACE_S = 15.0
SAMPLE_INTERVAL_S = 30.0
UTIL_SAMPLES_KEPT = 40  # 20 minutes at the default sample interval
POLL_INTERVAL_S = 0.5

UtilSampler = Callable[[Sequence[str]], float]

PREFLIGHT_SOURCE = """
import os, sys, torch
expected = int(os.environ["GPUC_EXPECTED_GPUS"])
if not torch.cuda.is_available():
    sys.exit(f"CUDA not available (torch {torch.__version__}, build {torch.version.cuda})")
probe = torch.zeros(8, device="cuda")
_ = (probe + 1.0).sum().item()
count = torch.cuda.device_count()
if count != expected:
    sys.exit(f"device_count()=={count}, expected {expected}")
print(f"gpu preflight ok: torch {torch.__version__} cuda {torch.version.cuda} "
      f"devices {count} {torch.cuda.get_device_name(0)}")
"""


def process_group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def kill_process_group(
    pgid: int,
    *,
    grace_s: float = KILL_GRACE_S,
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], float] = time.monotonic,
    reap: Callable[[], object] | None = None,
) -> None:
    """SIGTERM the whole group, then SIGKILL it after `grace_s`.

    The group, not the pid: jobs routinely spawn helper processes (dataloader
    workers, a `sleep` in a shell wrapper) that would otherwise survive and
    hold the GPU.

    `reap` must wait() our own direct child if we have one: an unreaped zombie
    is still a member of its process group, so without it every kill would burn
    the whole grace period before the group looked empty.
    """
    if pgid <= 1:
        return
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = now() + grace_s
    while now() < deadline:
        if reap is not None:
            reap()
        if not process_group_alive(pgid):
            return
        sleep(0.25)
    with contextlib.suppress(ProcessLookupError):
        os.killpg(pgid, signal.SIGKILL)


@dataclass
class RunnerDeps:
    smi: SmiRunner = gpus.run_nvidia_smi
    sampler: UtilSampler | None = None
    command_runner: sync.CommandRunner = sync.run_command
    sleep: Callable[[float], None] = time.sleep
    now: Callable[[], float] = time.monotonic
    poll_interval_s: float = POLL_INTERVAL_S
    sample_interval_s: float = SAMPLE_INTERVAL_S
    kill_grace_s: float = KILL_GRACE_S
    preflight: bool = True

    def util_sampler(self) -> UtilSampler:
        if self.sampler is not None:
            return self.sampler
        return lambda uuids: gpus.mean_utilization(uuids, self.smi)


@dataclass
class _Window:
    """Rolling mean over the trailing `window_s`, with a separate record of how
    long we have been sampling: the retained samples always span *less* than
    the window, so they cannot themselves tell us the window is covered."""

    window_s: float
    samples: deque[tuple[float, float]] = field(default_factory=deque)
    first_t: float | None = None

    def add(self, t: float, value: float) -> None:
        if self.first_t is None:
            self.first_t = t
        self.samples.append((t, value))
        while len(self.samples) > 1 and t - self.samples[0][0] > self.window_s:
            self.samples.popleft()

    def full(self, t: float) -> bool:
        return (
            self.first_t is not None
            and len(self.samples) >= 2
            and (t - self.first_t) >= self.window_s
        )

    def mean(self) -> float:
        return sum(v for _, v in self.samples) / len(self.samples)


def build_env(spec: JobSpec, assigned: Sequence[str]) -> dict[str, str]:
    env = dict(os.environ)
    env.update(jobs.parse_env_file(paths.job_env_file(spec.job_id)))
    env.update(spec.env)
    # Last, so a spec `env` typo cannot hand the job the wrong cards.
    env["CUDA_VISIBLE_DEVICES"] = ",".join(assigned)
    env["GPUC_JOB_ID"] = spec.job_id
    env["GPUC_JOB_DIR"] = str(paths.job_dir(spec.job_id))
    env["GPUC_OUTPUTS"] = str(paths.outputs_dir(spec.job_id))
    env["GPUC_EXPECTED_GPUS"] = str(len(assigned))
    return env


class JobRunner:
    def __init__(self, job_id: str, deps: RunnerDeps | None = None) -> None:
        self.job_id = job_id
        self.deps = deps or RunnerDeps()
        self.spec = jobs.read_spec(job_id)
        self.state = jobs.read_state(job_id)
        self.assigned: list[str] = list(self.state.gpus)
        self.config = jobs.read_config()
        self.kill_reason: str | None = None

    # -- logging ---------------------------------------------------------
    def _log(self, log: IO[bytes], message: str) -> None:
        log.write(f">>> {message}\n".encode())
        log.flush()

    # -- phases ----------------------------------------------------------
    def _spawn(self, command: str, env: dict[str, str], log: IO[bytes]) -> subprocess.Popen[bytes]:
        return subprocess.Popen(
            ["bash", "-eo", "pipefail", "-c", command],
            cwd=str(paths.workdir(self.job_id)),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )

    def _monitor(
        self, proc: subprocess.Popen[bytes], phase: str, log: IO[bytes], job_start: float
    ) -> int:
        deps = self.deps
        pgid = proc.pid
        jobs.update_state(self.job_id, phase=phase, pid=proc.pid, pgid=pgid)
        sampler = deps.util_sampler()
        low_util = self.spec.low_util
        record_util = phase == "main" and bool(self.assigned)
        watch_low_util = record_util and low_util.enabled
        window = _Window(low_util.window_min * 60.0)
        phase_start = deps.now()
        # Sampling starts immediately so `gpuc status --suspects` has data, but
        # the kill window only opens after grace_min: setup-like work at the top
        # of main (model download, compile) is legitimately at 0% util.
        next_sample = phase_start + deps.sample_interval_s
        watch_from = phase_start + low_util.grace_min * 60.0
        max_runtime_s = (
            None if self.spec.max_runtime_min is None else self.spec.max_runtime_min * 60.0
        )

        while proc.poll() is None:
            deps.sleep(deps.poll_interval_s)
            t = deps.now()
            if queue.is_cancelled(self.job_id):
                self._kill(proc, "cancelled", log)
                break
            if max_runtime_s is not None and (t - job_start) >= max_runtime_s:
                self._kill(proc, "timeout", log)
                break
            if record_util and t >= next_sample:
                next_sample = t + deps.sample_interval_s
                try:
                    util = sampler(self.assigned)
                except gpus.GpuError as exc:
                    self._log(log, f"utilization sample failed: {exc}")
                    continue
                self._record_util(util)
                if not watch_low_util or t < watch_from:
                    continue
                window.add(t, util)
                if window.full(t) and window.mean() < low_util.floor_pct:
                    self._log(
                        log,
                        f"low-util watchdog: mean {window.mean():.1f}% over "
                        f"{low_util.window_min:g} min is below {low_util.floor_pct:g}%",
                    )
                    self._kill(proc, "low-util", log)
                    break
        return proc.wait()

    def _record_util(self, util: float) -> None:
        recent = [*jobs.read_state(self.job_id).util_recent, round(util, 1)][-UTIL_SAMPLES_KEPT:]
        jobs.update_state(self.job_id, util_recent=recent, util_sampled_at=jobs.utc_now())

    def _kill(self, proc: subprocess.Popen[bytes], reason: str, log: IO[bytes]) -> None:
        self.kill_reason = reason
        self._log(log, f"killing process group {proc.pid}: {reason}")
        kill_process_group(
            proc.pid,
            grace_s=self.deps.kill_grace_s,
            sleep=self.deps.sleep,
            now=self.deps.now,
            reap=proc.poll,
        )

    def _run_phase(
        self, phase: str, command: str, env: dict[str, str], log: IO[bytes], job_start: float
    ) -> int:
        self._log(log, f"phase={phase}: {command}")
        proc = self._spawn(command, env, log)
        code = self._monitor(proc, phase, log, job_start)
        self._log(log, f"phase={phase} exited {code}")
        return code

    # -- entry point -----------------------------------------------------
    def run(self) -> int:
        paths.ensure_job_layout(self.job_id)
        job_start = self.deps.now()
        env = build_env(self.spec, self.assigned)
        sync_loop = sync.SyncLoop(
            self.spec,
            paths.workdir(self.job_id),
            self.config.s3_prefix,
            runner=self.deps.command_runner,
        )
        with paths.log_file(self.job_id).open("ab", buffering=0) as log:
            jobs.update_state(
                self.job_id,
                status="running",
                phase="setup",
                started_at=self.state.started_at or jobs.utc_now(),
                runner_pid=os.getpid(),
            )
            try:
                gpus.assert_uuids_present(self.assigned, self.deps.smi)
            except gpus.GpuError as exc:
                self._log(log, f"GPU assertion failed: {exc}")
                return self._finalize(1, "failed", "gpu-assert", sync_loop, log)

            self._log(
                log,
                f"job {self.job_id} on {self.config.host} "
                f"CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES'] or '(none)'}",
            )

            if self.spec.setup:
                code = self._run_phase("setup", self.spec.setup, env, log, job_start)
                if code != 0 or self.kill_reason:
                    return self._finalize(code, *self._classify(code, "setup"), sync_loop, log)

            if self.assigned and self.deps.preflight:
                code = self._run_phase(
                    "preflight",
                    f"uv run --no-sync python -c {_shell_quote(PREFLIGHT_SOURCE)}",
                    env,
                    log,
                    job_start,
                )
                if code != 0 or self.kill_reason:
                    return self._finalize(
                        code, *self._classify(code, "gpu-preflight"), sync_loop, log
                    )

            sync_loop.start()
            code = self._run_phase("main", self.spec.command, env, log, job_start)
            status, reason = self._classify(code, None)
            return self._finalize(code, status, reason, sync_loop, log)

    def _classify(self, code: int, failure_reason: str | None) -> tuple[str, str | None]:
        if self.kill_reason == "cancelled":
            return "cancelled", "cancelled"
        if self.kill_reason:
            return "failed", self.kill_reason
        if code == 0:
            return "succeeded", None
        return "failed", failure_reason or f"exit {code}"

    def _finalize(
        self,
        exit_code: int,
        status: str,
        reason: str | None,
        sync_loop: sync.SyncLoop,
        log: IO[bytes],
    ) -> int:
        jobs.update_state(self.job_id, phase="sync")
        try:
            sync_loop.final()
        except sync.SyncError as exc:
            self._log(log, f"final sync FAILED: {exc}")
            if status == "succeeded":
                status, reason, exit_code = "failed", "sync", 1
            elif reason:
                reason = f"{reason}+sync"
        if sync_loop.last_error and status == "succeeded":
            self._log(log, f"periodic sync had errors: {sync_loop.last_error}")

        jobs.update_state(
            self.job_id,
            status=status,
            reason=reason,
            exit_code=exit_code,
            ended_at=jobs.utc_now(),
            phase=None,
            pid=None,
            pgid=None,
        )
        self._log(log, f"job {self.job_id} {status}{f' ({reason})' if reason else ''}")
        try:
            sync.sync_job_meta(self.job_id, self.config.s3_prefix, runner=self.deps.command_runner)
        except sync.SyncError as exc:
            self._log(log, f"final state upload failed: {exc}")
        return exit_code


def _shell_quote(text: str) -> str:
    return "'" + text.replace("'", "'\"'\"'") + "'"


def run_job(job_id: str, deps: RunnerDeps | None = None) -> int:
    return JobRunner(job_id, deps).run()


def main(argv: Sequence[str] | None = None) -> int:
    args = list(argv) if argv is not None else sys.argv[1:]
    if len(args) != 1:
        print("usage: python -m gpuc.host run <job-id>", file=sys.stderr)
        return 2
    return run_job(args[0])
