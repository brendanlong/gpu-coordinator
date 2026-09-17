"""Runs exactly one job: environment, preflight, watchdogs, sync, exit code.

Exit-code discipline (measured in the shell implementation this replaces):
the job's exit code is captured before *any* cleanup, and a failed final sync
turns an otherwise-green job into `failed: sync` so lost outputs are never
silent.
"""

from __future__ import annotations

import contextlib
import os
import shlex
import signal
import subprocess
import time
from collections import deque
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO

from gpuc._version import user_agent
from gpuc.host import (
    baseline,
    cleanup,
    gpus,
    jobs,
    paths,
    preflight,
    progress,
    queue,
    scope,
    sync,
)
from gpuc.host.gpus import SmiRunner
from gpuc.host.jobs import JobSpec

KILL_GRACE_S = 15.0
SAMPLE_INTERVAL_S = 30.0
SPEC_REFRESH_S = 30.0
"""How often the monitor re-reads `spec.json` while a job runs.

`gpuc estimate` edits the spec of a job that is already running, and the copy
loaded at job start would never see it -- which is the job that most needs an
end time, since nobody can add one before it started."""
UTIL_SAMPLES_KEPT = 40  # 20 minutes at the default sample interval
POLL_INTERVAL_S = 0.5
TERMINATED_EXIT_CODE = 143  # 128 + SIGTERM, the shell convention

UtilSampler = Callable[[Sequence[str]], float]
ProgressPoller = Callable[[str, Path, dict[str, str]], float]

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


# -- process facts ------------------------------------------------------------
# A recorded pid alone proves nothing: pids are reused within a boot and reused
# from 1 again after a reboot, so "is the process that wrote this file still
# running?" needs the boot id and the process start time as well.

BOOT_ID_PATH = Path("/proc/sys/kernel/random/boot_id")
PROC_PATH = Path("/proc")


def boot_id() -> str | None:
    try:
        return BOOT_ID_PATH.read_text().strip() or None
    except OSError:
        return None


def parse_starttime(stat: str) -> str | None:
    """Field 22 of ``/proc/<pid>/stat``, as text.

    Split after the last ``)``: the comm field is parenthesised and may itself
    contain spaces and parentheses, which breaks a naive ``split()``.
    """
    _, sep, rest = stat.rpartition(")")
    if not sep:
        return None
    fields = rest.split()
    if len(fields) < 20:
        return None
    return fields[19]


def starttime(pid: int) -> str | None:
    try:
        return parse_starttime(Path(f"/proc/{pid}/stat").read_text())
    except OSError:
        return None


def cmdline(pid: int) -> str:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return ""
    return raw.decode("utf-8", "replace").replace("\0", " ").strip()


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def is_gpuc_process(pid: int) -> bool:
    return "gpuc.host" in cmdline(pid)


def find_runner_pid(job_id: str) -> int | None:
    """The pid of a live `gpuc.host run <job_id>`, if this host has one.

    The question a recorded pid cannot answer. `launch_ready` writes `running`
    before there is a process to name and the pid only after the spawn, so a
    dispatcher that died in between left a state naming no runner at all --
    and /proc is the only remaining record of the runner it did start.
    """
    try:
        pids = sorted(int(entry.name) for entry in PROC_PATH.iterdir() if entry.name.isdigit())
    except OSError:
        return None
    for pid in pids:
        argv = cmdline(pid).split()
        if "gpuc.host" in argv and argv[-2:] == ["run", job_id]:
            return pid
    return None


def recorded_process_alive(
    pid: int | None, recorded_boot_id: str | None = None, recorded_starttime: str | None = None
) -> bool:
    """Is the *same* process we recorded still running?"""
    if not pid or not pid_alive(pid):
        return False
    current_boot = boot_id()
    if recorded_boot_id and current_boot and recorded_boot_id != current_boot:
        return False
    current_start = starttime(pid)
    return not (recorded_starttime and current_start and recorded_starttime != current_start)


class _Terminated(BaseException):
    """The runner itself was signalled. A BaseException so that no `except
    Exception` in a phase can swallow the shutdown."""

    def __init__(self, signum: int) -> None:
        super().__init__(signum)
        self.signum = signum


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


def preflight_command() -> str:
    return f"uv run --no-sync python -c {shlex.quote(PREFLIGHT_SOURCE)}"


@dataclass
class RunnerDeps:
    smi: SmiRunner = gpus.run_nvidia_smi
    sampler: UtilSampler | None = None
    progress_poller: ProgressPoller = progress.poll
    command_runner: sync.CommandRunner = sync.run_command
    sleep: Callable[[float], None] = time.sleep
    now: Callable[[], float] = time.monotonic
    poll_interval_s: float = POLL_INTERVAL_S
    sample_interval_s: float = SAMPLE_INTERVAL_S
    spec_refresh_s: float = SPEC_REFRESH_S
    kill_grace_s: float = KILL_GRACE_S
    preflight: bool = True
    preflight_command: Callable[[], str] = preflight_command
    sync_preflight: bool = True
    isolation: str | None = None
    """`cgroup`, `pgid`, or None to ask `scope.isolation()` at job start."""

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


def build_env(
    spec: JobSpec, assigned: Sequence[str], config: jobs.HostConfig | None = None
) -> dict[str, str]:
    env = dict(os.environ)
    # The host's own env and PATH go first, so a job may still pin either
    # explicitly. The dispatcher normally passes these down already; doing it
    # here too means a runner started by hand, or by a dispatcher from before a
    # `gpuc host set`, still gets them.
    (config or jobs.read_config()).apply_env(env)
    env.update(jobs.parse_env_file(paths.job_env_file(spec.job_id)))
    env.update(spec.env)
    # Last, so a spec `env` typo cannot hand the job the wrong cards.
    env["CUDA_VISIBLE_DEVICES"] = ",".join(assigned)
    env["GPUC_JOB_ID"] = spec.job_id
    env["GPUC_JOB_DIR"] = str(paths.job_dir(spec.job_id))
    env["GPUC_OUTPUTS"] = str(paths.outputs_dir(spec.job_id))
    env["GPUC_EXPECTED_GPUS"] = str(len(assigned))
    # `hf` puts this in its own User-Agent, so a Hub-side question about our
    # traffic has something to point at. A job may still override it.
    env.setdefault("HF_HUB_USER_AGENT_ORIGIN", user_agent())
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
        self.env: dict[str, str] = {}
        self.isolation: str = self.deps.isolation or scope.isolation()
        self._current: subprocess.Popen[bytes] | None = None
        self._current_unit: str | None = None
        self._progress_error: str | None = None
        """The last progress failure we logged, so an interval-by-interval
        repeat of it does not bury the job's own output."""
        self._measured_eta = False
        """Whether a `progress_command` has produced an eta yet. Once one has,
        the spec's estimate is no longer published: it is a guess, and this is
        a measurement."""
        self._live = self.spec
        """The spec as the last re-read found it on disk, which outlives the
        phase that read it: an estimate added during `setup` must not be
        undone by `main` starting from the copy loaded at job start."""
        self._published_estimate: float | None = None
        """The `estimated_runtime_min` behind the eta now in the state file,
        null when that eta is not ours. Kept so the spec re-read only writes
        state when the estimate actually changed: `state.json` is a
        read-modify-write with the sync loop as a second writer, and an eta
        recomputed from the same estimate is the same instant anyway."""
        self._terminating = False
        self._preempting = False
        """Whether this job is going back in the queue rather than ending here.

        Snapshotted at the top of `_finalize`; see the note there."""
        self._finalizing = False
        """Set for the whole of `_finalize`, which must run exactly once.

        The dispatcher escalates a cancel to a SIGTERM at the runner itself
        after 30 s, and that lands squarely in the final sync of a long upload.
        Without this, the signal unwound `_finalize`, `run()` caught it, and
        finalize ran again -- rewriting a job that had already been recorded as
        `succeeded` into `failed: terminated` and uploading every output a
        second time."""

    # -- logging ---------------------------------------------------------
    def _log(self, log: IO[bytes], message: str) -> None:
        log.write(f">>> {message}\n".encode())
        log.flush()

    # -- phases ----------------------------------------------------------
    def _spawn(
        self, phase: str, command: str, env: dict[str, str], log: IO[bytes]
    ) -> subprocess.Popen[bytes]:
        unit = scope.unit_name(self.job_id, phase) if self.isolation == scope.CGROUP else None
        self._current_unit = unit
        return subprocess.Popen(
            scope.phase_argv(command, unit),
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
        jobs.update_state(
            self.job_id,
            phase=phase,
            pid=proc.pid,
            pgid=pgid,
            isolation=self.isolation,
            cgroup_unit=self._current_unit,
        )
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
        # `spec.json` is re-read on a timer, so an estimate (or a progress
        # command) added after the job started still takes effect. Only what it
        # *reports* -- `estimated_runtime_min`, `progress_command` and its
        # interval -- is taken from the re-read: a command, an env or an output
        # path changing mid-flight would leave the spec describing a run that
        # never happened.
        live = self._live
        next_spec = phase_start + deps.spec_refresh_s
        # The submitter's estimate, published from the first phase on: a job
        # still installing torch is exactly the one somebody wants an end time
        # for. A `progress_command` replaces it with a measured one below.
        self._publish_estimated_eta(live.estimated_runtime_min, phase_start - job_start)
        # Progress is a fraction of the job's own work, so only `main` can
        # report it: during setup the command would be reading a file the job
        # has not started writing.
        progress_command = live.progress_command if phase == "main" else None
        next_progress = phase_start + live.progress_interval_s

        while proc.poll() is None:
            deps.sleep(deps.poll_interval_s)
            t = deps.now()
            if queue.is_cancelled(self.job_id):
                self._kill(proc, "cancelled", log)
                break
            requested = queue.kill_reason(self.job_id)
            if requested:
                self._kill(proc, requested, log)
                break
            if max_runtime_s is not None and (t - job_start) >= max_runtime_s:
                self._kill(proc, "timeout", log)
                break
            if t >= next_spec:
                next_spec = t + deps.spec_refresh_s
                live = self._live = self._live_spec(live)
                self._publish_estimated_eta(live.estimated_runtime_min, t - job_start)
                progress_command = live.progress_command if phase == "main" else None
            if progress_command and t >= next_progress:
                next_progress = t + live.progress_interval_s
                self._record_progress(progress_command, t - phase_start, log)
            if record_util and t >= next_sample:
                next_sample = t + deps.sample_interval_s
                try:
                    util = sampler(self.assigned)
                except (gpus.GpuError, ValueError) as exc:
                    # A missing sample is not evidence of an idle GPU, so it is
                    # recorded as unknown and never feeds the watchdog window.
                    self._log(log, f"utilization sample failed: {exc}")
                    self._record_util(None)
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

    def _live_spec(self, previous: JobSpec) -> JobSpec:
        """The spec as `spec.json` holds it now, or `previous` if it cannot be
        read -- a spec being rewritten under us may not end a running job."""
        try:
            return jobs.read_spec(self.job_id)
        except (RuntimeError, OSError, ValueError):
            return previous

    def _publish_estimated_eta(self, estimate: float | None, elapsed_s: float) -> None:
        """The end time the submitter's estimate implies, while that is the
        best we have. A measured one, once there is one, is never overwritten
        by a guess -- including a guess edited in halfway through the job."""
        if self._measured_eta or estimate == self._published_estimate:
            return
        if estimate is None:
            # Only when we are the one who published it: an estimate cleared
            # from the spec should take its eta with it, but a job that never
            # had one must not have its state rewritten at all.
            self._published_estimate = None
            jobs.update_state(self.job_id, eta=None)
            return
        eta = jobs.utc_in(estimate * 60.0 - elapsed_s)
        if eta is not None:
            self._published_estimate = estimate
            jobs.update_state(self.job_id, eta=eta)

    def _record_progress(self, command: str, elapsed_s: float, log: IO[bytes]) -> None:
        """One poll of the spec's `progress_command`, and the end time it implies.

        `elapsed_s` is time in phase `main`, which is the work the percentage is
        a fraction of: measuring from the runner's start would charge a 20
        minute `uv sync` to the first epoch and put the estimate hours out.
        """
        try:
            percent = self.deps.progress_poller(command, paths.workdir(self.job_id), self.env)
        except progress.ProgressError as exc:
            message = f"progress command {exc}"
            if message != self._progress_error:
                # Once per distinct failure: this runs every interval for the
                # rest of the job, and a broken command would otherwise be the
                # only thing left in the log.
                self._log(log, message)
            self._progress_error = message
            jobs.update_state(self.job_id, progress_error=message)
            return
        self._progress_error = None
        fields: dict[str, object] = {
            "progress_pct": percent,
            "progress_error": None,
        }
        if percent > 0:
            # At 0% there is no rate yet, so the submitter's estimate (if any)
            # stays; overwriting it with an infinite one helps nobody.
            eta = jobs.utc_in(elapsed_s * (100.0 - percent) / percent)
            if eta is not None:
                fields["eta"] = eta
                self._measured_eta = True
        jobs.update_state(self.job_id, **fields)

    def _record_util(self, util: float | None) -> None:
        sample = None if util is None else round(util, 1)
        recent = [*jobs.read_state(self.job_id).util_recent, sample][-UTIL_SAMPLES_KEPT:]
        jobs.update_state(self.job_id, util_recent=recent)

    def _kill(self, proc: subprocess.Popen[bytes], reason: str, log: IO[bytes]) -> None:
        self.kill_reason = reason
        unit = self._current_unit
        if unit is not None:
            self._log(log, f"stopping scope {unit}: {reason}")
            if not scope.stop_unit(unit):
                self._log(log, f"systemctl --user stop {unit} failed; falling back to the group")
        else:
            self._log(log, f"killing process group {proc.pid}: {reason}")
        # Always, scope or not: the group kill is the fallback, and against a
        # cgroup that is already empty it costs one ProcessLookupError.
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
        proc = self._spawn(phase, command, env, log)
        # Left set if _monitor raises: a terminating runner needs the handle to
        # the group (and the scope) it must take down.
        self._current = proc
        code = self._monitor(proc, phase, log, job_start)
        self._current = None
        jobs.update_state(self.job_id, cgroup_unit=None)
        self._current_unit = None
        self._log(log, f"phase={phase} exited {code}")
        return code

    # -- signals ---------------------------------------------------------
    @contextlib.contextmanager
    def _term_handlers(self) -> Iterator[None]:
        """Turn SIGTERM/SIGINT into an exception on the main thread.

        Without this the runner dies where it stands: the job's process group
        survives holding a GPU the dispatcher is about to hand to someone else,
        and the job is left `running` forever.
        """

        def handle(signum: int, _frame: object) -> None:
            if self._finalizing or self._terminating:
                return
            self._terminating = True
            raise _Terminated(signum)

        try:
            previous = [
                (sig, signal.signal(sig, handle)) for sig in (signal.SIGTERM, signal.SIGINT)
            ]
        except ValueError:
            previous = []  # not the main thread; the caller owns the signals
        try:
            yield
        finally:
            for sig, handler in previous:
                with contextlib.suppress(ValueError):
                    signal.signal(sig, handler)

    # -- entry point -----------------------------------------------------
    def run(self) -> int:
        paths.ensure_job_layout(self.job_id)
        job_start = self.deps.now()
        gpu_error = self._resolve_assigned()
        env = build_env(self.spec, self.assigned, self.config)
        # The sync loop uploads as the *job*: its `secrets:` are in `env`, so
        # `secrets: [AWS_ACCESS_KEY_ID, ...]` is all an output needs, with no
        # credential file anywhere on the host.
        self.env = env
        sync_loop = sync.SyncLoop(
            self.spec,
            paths.workdir(self.job_id),
            self.config.s3_prefix,
            runner=self.deps.command_runner,
            env=env,
        )
        with paths.log_file(self.job_id).open("ab", buffering=0) as log, self._term_handlers():
            try:
                return self._run_phases(env, sync_loop, log, job_start, gpu_error)
            except _Terminated as exc:
                return self._finalize_terminated(exc, sync_loop, log)

    def _resolve_assigned(self) -> str | None:
        """Turn the assignment into UUIDs, or say why it cannot be.

        A job may have been assigned cards by nvidia-smi index -- that is how
        ownership of a shared box is written, and a dispatcher from before the
        assignment was resolved host-side hands the index straight through. An
        index is only meaningful against the host's numbering right now, so it
        is resolved here and everything after this -- `CUDA_VISIBLE_DEVICES`,
        the utilization watchdog -- sees UUIDs.
        """
        before = list(self.assigned)
        try:
            self.assigned = gpus.resolve_present(self.assigned, "assigned GPUs", smi=self.deps.smi)
        except gpus.GpuError as exc:
            return str(exc)
        if self.assigned != before:
            # So everything reading the state afterwards -- `gpuc status`, the
            # control side's free/busy view -- names the same cards the job is
            # actually on. The dispatcher resolves what it adopts either way.
            jobs.update_state(self.job_id, gpus=self.assigned)
        return None

    def _run_phases(
        self,
        env: dict[str, str],
        sync_loop: sync.SyncLoop,
        log: IO[bytes],
        job_start: float,
        gpu_error: str | None,
    ) -> int:
        jobs.update_state(
            self.job_id,
            status="running",
            phase="setup",
            started_at=self.state.started_at or jobs.utc_now(),
            isolation=self.isolation,
            runner_pid=os.getpid(),
            runner_boot_id=boot_id(),
            runner_starttime=starttime(os.getpid()),
        )
        if gpu_error:
            self._log(log, f"GPU assertion failed: {gpu_error}")
            # The job never ran, so its `outputs:` cannot exist and a second
            # failure would only add a confusing `+no-outputs`.
            return self._finalize(1, "failed", "gpu-assert", sync_loop, log, skip_output_sync=True)

        self._log(
            log,
            f"job {self.job_id} on {self.config.host} "
            f"CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES'] or '(none)'}",
        )

        self._capture_output_baseline(log)

        cancelled = self._cancelled_before("setup", sync_loop, log)
        if cancelled is not None:
            return cancelled

        if self.spec.setup:
            code = self._run_phase("setup", self.spec.setup, env, log, job_start)
            if code != 0 or self.kill_reason:
                return self._finalize(code, *self._classify(code, "setup"), sync_loop, log)

        if self.assigned and self.deps.preflight:
            cancelled = self._cancelled_before("preflight", sync_loop, log)
            if cancelled is not None:
                return cancelled
            code = self._run_phase("preflight", self.deps.preflight_command(), env, log, job_start)
            if code != 0 or self.kill_reason:
                return self._finalize(code, *self._classify(code, "gpu-preflight"), sync_loop, log)

        if self._sync_preflight(log) is not None:
            return self._finalize(
                1, "failed", "sync-preflight", sync_loop, log, skip_output_sync=True
            )

        cancelled = self._cancelled_before("main", sync_loop, log)
        if cancelled is not None:
            return cancelled

        sync_loop.start()
        code = self._run_phase("main", self.spec.command, env, log, job_start)
        status, reason = self._classify(code, None)
        return self._finalize(code, status, reason, sync_loop, log)

    def _capture_output_baseline(self, log: IO[bytes]) -> None:
        """Record what the checkout already had where the outputs go.

        Before `setup`, because a setup step may legitimately write into an
        output path and that *is* this job's doing.
        """
        if not self.spec.outputs:
            return
        if self.state.attempt > 1 and paths.outputs_baseline_file(self.job_id).exists():
            # A preempted job re-runs in the workdir the stopped attempt left,
            # so re-scanning now would record that attempt's own results as
            # files "the checkout arrived with" -- and this attempt would then
            # never upload them, nor count them towards `outputs:`. The
            # baseline is a fact about the checkout, and the checkout has not
            # changed. (`gpuc requeue` cannot reach this: it is a new job id,
            # with a new job dir and no baseline in it.)
            self._log(log, "outputs baseline: keeping the one taken before the first attempt")
            return
        found = baseline.capture(self.spec, paths.workdir(self.job_id), self.job_id)
        for line in baseline.describe(found):
            self._log(log, f"outputs baseline: {line}")
        for path, entries in sorted(found.items()):
            if baseline.too_many(entries):
                self._log(
                    log,
                    f"WARNING: {path} already holds more than {baseline.MAX_TRACKED} files, too "
                    f"many to exclude one by one, so pre-existing files there WILL be uploaded. "
                    f"Point `outputs:` at a directory this job creates.",
                )

    def _sync_preflight(self, log: IO[bytes]) -> str | None:
        """Prove the uploads work before the job spends hours producing outputs.

        The alternative is finding out at the final sync, when the only copy of
        a checkpoint is on a host that may be about to go away.
        """
        if not self.deps.sync_preflight:
            return None
        jobs.update_state(self.job_id, phase="preflight")
        try:
            destinations = preflight.run(
                self.spec,
                self.config,
                runner=self.deps.command_runner,
                env=self.env or None,
            )
        except preflight.PreflightFailed as exc:
            self._log(log, f"sync preflight FAILED: {exc}")
            return str(exc)
        self._log(log, preflight.describe(destinations))
        return None

    def _cancelled_before(self, phase: str, sync_loop: sync.SyncLoop, log: IO[bytes]) -> int | None:
        """A job cancelled during the launch window never starts a phase."""
        if not queue.is_cancelled(self.job_id):
            return None
        self._log(log, f"cancel marker present before phase={phase}; not starting it")
        return self._finalize(TERMINATED_EXIT_CODE, "cancelled", "cancelled", sync_loop, log)

    def _finalize_terminated(
        self, exc: _Terminated, sync_loop: sync.SyncLoop, log: IO[bytes]
    ) -> int:
        if self._finalizing:
            # Belt and braces with the signal handler: whatever `_finalize`
            # decided is this job's outcome, so report that rather than
            # finalizing a second time.
            try:
                return jobs.read_state(self.job_id).exit_code or 0
            except RuntimeError:
                return TERMINATED_EXIT_CODE
        name = signal.Signals(exc.signum).name
        self._log(log, f"runner received {name}; stopping the job")
        proc = self._current
        if proc is not None:
            self._kill(proc, "terminated", log)
            with contextlib.suppress(Exception):
                proc.wait(timeout=5)
        cancelled = queue.is_cancelled(self.job_id)
        status, reason = ("cancelled", "cancelled") if cancelled else ("failed", "terminated")
        return self._finalize(TERMINATED_EXIT_CODE, status, reason, sync_loop, log)

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
        skip_output_sync: bool = False,
    ) -> int:
        self._finalizing = True
        # Read once, and before the final state write below: from that write
        # on, a dispatcher starting up sees a finished job with a preempt
        # marker and is entitled to consume the marker. Asked again afterwards
        # -- as the workdir and secrets decisions used to -- the answer flips
        # to "no preempt" and this runner deletes the very workdir and secrets
        # file the next attempt was about to be dispatched with.
        self._preempting = queue.is_preempted(self.job_id)
        jobs.update_state(self.job_id, phase="sync")
        if skip_output_sync:
            # The job never ran (a failed preflight, a card that is not here),
            # so a second failure would only add a confusing `+no-outputs` to a
            # reason that is already exact.
            self._log(log, "skipping the final output sync: the job never ran")
        else:
            try:
                sync_loop.final()
            except sync.MissingOutput as exc:
                self._log(log, f"final sync found no outputs: {exc}")
                status, reason, exit_code = self._blame(status, reason, exit_code, "no-outputs")
            except sync.SyncError as exc:
                self._log(log, f"final sync FAILED: {exc}")
                status, reason, exit_code = self._blame(status, reason, exit_code, "sync")
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
            cgroup_unit=None,
            # The job is over, so there is nothing left to estimate; the last
            # `progress_pct` stays, because how far it had got when it died is
            # the useful part. A surviving `eta` would read as a promise the
            # job is still going.
            eta=None,
            outputs_synced_at=sync_loop.outputs_synced_at,
        )
        self._log(log, f"job {self.job_id} {status}{f' ({reason})' if reason else ''}")
        # After the final state write, and only then: the outputs the sync just
        # uploaded live *inside* the workdir, so anything earlier would delete
        # the run's results on the way past.
        removed = self._cleanup_workdir(status, log)
        # Measure what is left while we are still standing in it: this job is
        # over, so the figure will not change, and `status` should not have to
        # walk a 67k-file venv to find it out again. Exact, because once is
        # cheap -- see `cleanup.reclaimable_bytes`.
        jobs.update_state(
            self.job_id,
            workdir_removed=removed,
            workdir_bytes=0 if removed else (cleanup.workdir_size(self.job_id) or 0),
        )
        try:
            warning = sync.final_meta_sync(
                self.job_id,
                self.config.s3_prefix,
                runner=self.deps.command_runner,
                env=self.env or None,
            )
            if warning:
                self._log(log, f"WARNING: {warning}")
        except sync.SyncError as exc:
            # meta_synced_at stays null, so `purge` will refuse to delete this
            # job dir: the only copy of the log lives here.
            self._log(log, f"final state upload failed: {exc}")
        # Only now: the final sync and the state upload authenticate with the
        # secrets this file holds, so removing it earlier would break exactly
        # the upload that matters most. On an ephemeral host with outputs still
        # unconfirmed it stays: the drain gets one more go at uploading them,
        # and the file dies with the pod in minutes either way.
        if self._keep_secrets_for_drain(sync_loop):
            self._log(
                log,
                "outputs are not confirmed uploaded; keeping this job's secrets file so the "
                "host's drain can retry the upload before the pod goes away",
            )
        elif self._preempting:
            # The next attempt is this same job id, and nothing will deliver
            # its secrets a second time: `gpuc preempt` never goes near the
            # control machine that holds them.
            self._log(log, "preempted; keeping this job's secrets file for the next attempt")
        else:
            paths.job_env_file(self.job_id).unlink(missing_ok=True)
        return exit_code

    def _keep_secrets_for_drain(self, sync_loop: sync.SyncLoop) -> bool:
        if not (self.config.ephemeral and self.spec.outputs and not sync_loop.outputs_synced_at):
            return False
        # The same question the drain asks before it retries anything: a job
        # that wrote no outputs is skipped there, so keeping its credentials on
        # disk buys a retry that will never happen.
        return cleanup.produced_outputs(self.job_id, self.spec)

    def _cleanup_workdir(self, status: str, log: IO[bytes]) -> bool:
        """Apply the spec's `cleanup:` policy to `workdir/`, and nothing else.

        A failure to delete is logged and nothing more: the job's own outcome
        has already been decided and uploaded, and turning a green run red over
        leftover disk would be the wrong trade.
        """
        if not cleanup.should_remove(self.spec.cleanup, status):
            return False
        if self._preempting:
            # `cleanup: always` would take the code with it, and the workdir is
            # the only copy on this host: the control side rsynced it once, at
            # submit, and the next attempt re-runs from what is there.
            self._log(log, f"preempted; keeping workdir (cleanup={self.spec.cleanup})")
            return False
        try:
            freed = cleanup.remove_workdir(self.job_id)
        except OSError as exc:
            self._log(log, f"could not remove workdir (cleanup={self.spec.cleanup}): {exc}")
            return False
        self._log(
            log,
            f"removed workdir (cleanup={self.spec.cleanup}), freeing "
            f"{cleanup.human_bytes(freed)}; spec.json, state.json and log.txt are kept",
        )
        return True

    @staticmethod
    def _blame(
        status: str, reason: str | None, exit_code: int, sync_reason: str
    ) -> tuple[str, str | None, int]:
        if status == "succeeded":
            return "failed", sync_reason, 1
        return status, f"{reason}+{sync_reason}" if reason else sync_reason, exit_code


def run_job(job_id: str, deps: RunnerDeps | None = None) -> int:
    return JobRunner(job_id, deps).run()
