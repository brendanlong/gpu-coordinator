"""Runs exactly one job: claim, environment, preflight, wall-clock limit, sync,
cleanup, and the one write that ends it.

The runner owns every transition of its job. Its first act is the
compare-and-set that takes the job out of the queue, so a `running` state
always names a runner that existed; its last act is the write that ends the
attempt -- a terminal status, or `queued` again for a preempted job -- so a
job is finished exactly when its runner is gone, and nothing has to guess
whether the process behind a finished state is still cleaning up.

Exit-code discipline (measured in the shell implementation this replaces):
the job's exit code is captured before *any* cleanup, and a failed final sync
turns an otherwise-green job into `failed: sync` so lost outputs are never
silent.
"""

from __future__ import annotations

import contextlib
import dataclasses
import os
import shlex
import signal
import subprocess
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import IO

from gpuc._version import user_agent
from gpuc.host import (
    baseline,
    cleanup,
    destinations,
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
from gpuc.host.jobs import JobSpec, Outcome
from gpuc.host.procs import KILL_GRACE_S, JobProcesses, boot_id, starttime

SAMPLE_INTERVAL_S = 30.0
ESTIMATE_REFRESH_S = 30.0
"""How often the monitor re-reads the job's estimate and wall-clock limit
while it runs.

`gpuc estimate` and `gpuc max-runtime` change the state of a job that is
already running, and the values loaded at job start would never see them --
which is the job that most needs them: nobody can add an end time before it
started, and a limit that turns out too tight is found out while it runs."""
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


class _Terminated(BaseException):
    """The runner itself was signalled. A BaseException so that no `except
    Exception` in a phase can swallow the shutdown."""

    def __init__(self, signum: int) -> None:
        super().__init__(signum)
        self.signum = signum


def preflight_command(spec: JobSpec) -> str:
    """The GPU check, run through the interpreter the spec says is the job's."""
    return f"{spec.python} -c {shlex.quote(PREFLIGHT_SOURCE)}"


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
    estimate_refresh_s: float = ESTIMATE_REFRESH_S
    kill_grace_s: float = KILL_GRACE_S
    preflight: bool = True
    preflight_command: Callable[[JobSpec], str] = preflight_command
    sync_preflight: bool = True

    def util_sampler(self) -> UtilSampler:
        if self.sampler is not None:
            return self.sampler
        return lambda uuids: gpus.mean_utilization(uuids, self.smi)


def build_env(
    spec: JobSpec, assigned: Sequence[str], indices: Mapping[str, int] | None = None
) -> dict[str, str]:
    """The job's environment: the runner's own, the secrets file, the spec's,
    then the cards -- last, so a spec `env` typo cannot hand the job the wrong
    ones. The host's `env` and PATH are already the runner's: the dispatcher
    applied them to everything it spawns.

    `CUDA_VISIBLE_DEVICES` names the cards by nvidia-smi index when the runner
    has just resolved every one (vLLM and others `int()` each entry, and a
    UUID there fails inside a subprocess with an error that points at the
    model), pinned to nvidia-smi's numbering by `CUDA_DEVICE_ORDER=PCI_BUS_ID`
    so the index means the same card to CUDA. Without a full index map the
    UUIDs go through as they are, which every torch accepts.
    """
    env = dict(os.environ)
    env.update(jobs.parse_env_file(paths.job_env_file(spec.job_id)))
    env.update(spec.env)
    if indices is not None and assigned and all(uuid in indices for uuid in assigned):
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(indices[uuid]) for uuid in assigned)
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    else:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(assigned)
    env["GPUC_JOB_ID"] = spec.job_id
    env["GPUC_JOB_DIR"] = str(paths.job_dir(spec.job_id))
    env["GPUC_OUTPUTS"] = str(paths.outputs_dir(spec.job_id))
    env["GPUC_EXPECTED_GPUS"] = str(len(assigned))
    env["GPUC_DATA_DIR"] = str(paths.data_dir(env))
    # `hf` puts this in its own User-Agent, so a Hub-side question about our
    # traffic has something to point at. A job may still override it.
    env.setdefault("HF_HUB_USER_AGENT_ORIGIN", user_agent())
    return env


class JobRunner:
    def __init__(
        self, job_id: str, assigned: Sequence[str], attempt: int, deps: RunnerDeps | None = None
    ) -> None:
        self.job_id = job_id
        self.assigned: list[str] = list(assigned)
        self.attempt = attempt
        """The attempt the dispatcher launched this runner for; the claim is
        for exactly that one."""
        self.deps = deps or RunnerDeps()
        self.spec = jobs.read_spec(job_id)
        self.state = jobs.read_state(job_id)
        self.config = jobs.read_config()
        self.kill_reason: str | None = None
        self.env: dict[str, str] = {}
        self.isolation: str = scope.isolation()
        self._indices: dict[str, int] | None = None
        """uuid -> nvidia-smi index, from the one reading that verified the
        assignment; None if it could not be read."""
        self._current: subprocess.Popen[bytes] | None = None
        self._current_unit: str | None = None
        self._progress_error: str | None = None
        """The last progress failure we logged, so an interval-by-interval
        repeat of it does not bury the job's own output."""
        self._measured_eta = False
        """Whether a `progress_command` has produced an eta yet. Once one has,
        the spec's estimate is no longer published: it is a guess, and this is
        a measurement."""
        self._estimate = self.state.estimated_runtime_min
        """The estimate as the last re-read found it, which outlives the phase
        that read it: an estimate added during `setup` must not be undone by
        `main` starting from the value loaded at job start."""
        self._max_runtime = self.state.max_runtime(self.spec)
        """The wall-clock limit as the last re-read found it, like `_estimate`."""
        self._published_estimate: float | None = None
        """The `estimated_runtime_min` behind the eta now in the state file,
        null when that eta is not ours. Kept so the spec re-read only writes
        state when the estimate actually changed: `state.json` is a
        read-modify-write with the sync loop as a second writer, and an eta
        recomputed from the same estimate is the same instant anyway."""
        self._main_started = False
        """Whether `main` has begun, which is what `Outcome.ran` reports: the
        final upload and the no-outputs check are for a job that produced
        something, and only `main` does."""
        self._ending = False
        """Set once the attempt is on its way out: by the signal handler as it
        raises `_Terminated`, and by `_finalize` as it starts. A signal after
        that is ignored. The dispatcher escalates a stop to a SIGTERM at the
        runner itself, and that lands squarely in the final sync of a long
        upload; without this it unwound `_finalize`, `run()` caught it, and
        finalize ran again -- rewriting a job already recorded as `succeeded`
        into `failed: terminated` and uploading every output a second time."""

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
        jobs.update_state(self.job_id, phase=phase, pgid=proc.pid, cgroup_unit=self._current_unit)
        sampler = deps.util_sampler()
        # Utilization is sampled for `gpuc status` only, and only in `main`:
        # setup is downloads and compiles, and a 0% there says nothing.
        record_util = phase == "main" and bool(self.assigned)
        phase_start = deps.now()
        next_sample = phase_start + deps.sample_interval_s
        # The estimate and the limit are re-read on a timer, so a change made
        # after the job started still takes effect. They are the only things a
        # running job re-reads: the spec is never rewritten after enqueue.
        next_estimate = phase_start + deps.estimate_refresh_s
        # The submitter's estimate, published from the first phase on: a job
        # still installing torch is exactly the one somebody wants an end time
        # for. A `progress_command` replaces it with a measured one below.
        self._publish_estimated_eta(self._estimate, phase_start - job_start)
        # Progress is a fraction of the job's own work, so only `main` can
        # report it: during setup the command would be reading a file the job
        # has not started writing.
        progress_command = self.spec.progress_command if phase == "main" else None
        next_progress = phase_start + self.spec.progress_interval_s

        while proc.poll() is None:
            deps.sleep(deps.poll_interval_s)
            t = deps.now()
            requested = queue.stop_requested(self.job_id)
            if requested:
                self._kill(proc, requested, log)
                break
            if t >= next_estimate:
                next_estimate = t + deps.estimate_refresh_s
                self._reread_live_fields(log)
                self._publish_estimated_eta(self._estimate, t - job_start)
            if self._max_runtime is not None and (t - job_start) >= self._max_runtime * 60.0:
                self._kill(proc, "timeout", log)
                break
            if progress_command and t >= next_progress:
                next_progress = t + self.spec.progress_interval_s
                self._record_progress(progress_command, t - phase_start, log)
            if record_util and t >= next_sample:
                next_sample = t + deps.sample_interval_s
                try:
                    util = sampler(self.assigned)
                except (gpus.GpuError, ValueError) as exc:
                    # A missing sample is not evidence of an idle GPU, so it is
                    # recorded as unknown rather than as 0%.
                    self._log(log, f"utilization sample failed: {exc}")
                    self._record_util(None)
                    continue
                self._record_util(util)
        return proc.wait()

    def _reread_live_fields(self, log: IO[bytes]) -> None:
        """The estimate and the limit as the state holds them now. Both stay
        as last read if the state cannot be read -- a file being replaced under
        us may not end a running job."""
        try:
            state = jobs.read_state(self.job_id)
        except (RuntimeError, OSError, ValueError):
            return
        self._estimate = state.estimated_runtime_min
        limit = state.max_runtime(self.spec)
        if limit != self._max_runtime:
            said = "no limit" if limit is None else f"{limit:g} min"
            self._log(log, f"max_runtime_min is now {said}")
            self._max_runtime = limit

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
        JobProcesses(self._current_unit, proc.pid).stop(
            reason,
            grace_s=self.deps.kill_grace_s,
            sleep=self.deps.sleep,
            now=self.deps.now,
            reap=proc.poll,
            log=lambda message: self._log(log, message),
        )

    def _stop_leftovers(self, proc: subprocess.Popen[bytes], log: IO[bytes]) -> None:
        """Whatever the phase left running goes with it, before its cards are
        handed on.

        A scope made with `--collect` outlives its shell for as long as any
        process is in it, so a leaked DataLoader worker or a background server
        kept holding GPU memory after a clean exit, and the *next* job on the
        card died at startup with an out-of-memory error naming nobody else.
        A stop returns once the cgroup is empty, and the driver releases a
        process's memory as it exits, so there is nothing to poll -- least of
        all `memory.used`, which on a shared card moves for other reasons.
        Under `pgid` a `setsid` grandchild still escapes, as it does a cancel.
        """
        processes = JobProcesses(self._current_unit, proc.pid)
        leftovers = processes.members()
        if not leftovers:
            return
        self._log(
            log,
            f"stopping {len(leftovers)} leftover process(es) of the phase: "
            f"{' '.join(map(str, leftovers))}",
        )
        processes.stop(
            "the phase exited",
            grace_s=self.deps.kill_grace_s,
            sleep=self.deps.sleep,
            now=self.deps.now,
            log=lambda message: self._log(log, message),
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
        if not self.kill_reason:
            self._stop_leftovers(proc, log)
        self._current = None
        # Both, together: a group number outlives its processes, and a
        # `pgid` left naming a finished phase is what the dispatcher's ladder
        # would SIGKILL during the final sync, once somebody else had it.
        jobs.update_state(self.job_id, pgid=None, cgroup_unit=None)
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
            if self._ending:
                return
            self._ending = True
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
        """Claim the job, run it, end it. Zero, quietly, if the claim failed:
        the job was cancelled between the dispatcher's decision and this
        process starting, another runner got here first, or the attempt this
        runner was started for is over, and either way it is not ours to
        touch."""
        if not self._claim():
            return 0
        paths.ensure_job_layout(self.job_id)
        job_start = self.deps.now()
        gpu_error = self._verify_assigned()
        env = build_env(self.spec, self.assigned, self._indices)
        # A directory the job cannot create is the job's error to hit, in its
        # own log, when it first writes there; the runner has a job to finish.
        with contextlib.suppress(OSError):
            Path(env["GPUC_DATA_DIR"]).mkdir(mode=0o700, parents=True, exist_ok=True)
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
        with paths.log_file(self.job_id).open("ab") as log, self._term_handlers():
            try:
                return self._run_phases(env, sync_loop, log, job_start, gpu_error)
            except _Terminated as exc:
                return self._finalize_terminated(exc, sync_loop, log)

    def _claim(self) -> bool:
        """Take the job out of the queue, naming this process as its runner.

        One compare-and-set on the attempt this runner was started for,
        carrying the assignment the dispatcher decided and the identity a
        later dispatcher needs to tell this process from a reused pid. So a
        state that says `running` always names a runner that existed, and a
        cancel that landed first simply wins.
        """
        pid = os.getpid()
        return queue.claim(
            self.job_id,
            self.attempt,
            self.spec,
            status="running",
            gpus=self.assigned,
            phase="setup",
            started_at=jobs.utc_now(),
            isolation=self.isolation,
            runner_pid=pid,
            runner_boot_id=boot_id(),
            runner_starttime=starttime(pid),
        )

    def _verify_assigned(self) -> str | None:
        """Check every assigned UUID against the driver, or say why not.

        The assignment is UUIDs, resolved by the dispatcher; what can still go
        wrong is a card that the driver no longer reports. The same reading
        gives the nvidia-smi index of each card for `CUDA_VISIBLE_DEVICES`.
        """
        if not self.assigned:
            return None if self.spec.gpus == 0 else "no GPUs assigned to a job that asks for some"
        try:
            table = gpus.list_gpus(self.deps.smi)
        except gpus.GpuError as exc:
            return str(exc)
        cards = gpus.resolve(self.assigned, table)
        if cards.missing:
            return (
                f"assigned GPUs not present on this host: {', '.join(cards.missing)}; "
                f"nvidia-smi reports: {gpus.describe_table(table)}"
            )
        if cards.duplicates:
            # A promise of *n* cards that names one twice would run a two-GPU
            # job on one.
            return (
                f"assigned GPUs name one card twice: {', '.join(cards.duplicates)}; "
                f"nvidia-smi reports: {gpus.describe_table(table)}"
            )
        self._indices = {gpu.uuid: gpu.index for gpu in table if gpu.index is not None}
        return None

    def _run_phases(
        self,
        env: dict[str, str],
        sync_loop: sync.SyncLoop,
        log: IO[bytes],
        job_start: float,
        gpu_error: str | None,
    ) -> int:
        if gpu_error:
            self._log(log, f"GPU assertion failed: {gpu_error}")
            return self._finalize(Outcome("failed", "gpu-assert", 1, ran=False), sync_loop, log)

        self._log(
            log,
            f"job {self.job_id} on {self.config.host} "
            f"CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES'] or '(none)'}",
        )

        self._capture_output_baseline(log)

        stopped = self._stopped_before("setup", sync_loop, log)
        if stopped is not None:
            return stopped

        if self.spec.setup:
            code = self._run_phase("setup", self.spec.setup, env, log, job_start)
            if code != 0 or self.kill_reason:
                return self._finalize(self._classify(code, "setup"), sync_loop, log)

        if self.deps.preflight and self.assigned:
            stopped = self._stopped_before("preflight", sync_loop, log)
            if stopped is not None:
                return stopped
            # A phase of its own, not the tail of `setup`: `gpuc status` can
            # then tell "still installing torch" from "proving the card works".
            code = self._run_phase(
                "preflight", self.deps.preflight_command(self.spec), env, log, job_start
            )
            if code != 0 or self.kill_reason:
                return self._finalize(self._classify(code, "gpu-preflight"), sync_loop, log)

        if self._sync_preflight(log) is not None:
            return self._finalize(Outcome("failed", "sync-preflight", 1, ran=False), sync_loop, log)

        stopped = self._stopped_before("main", sync_loop, log)
        if stopped is not None:
            return stopped

        self._main_started = True
        jobs.update_state(self.job_id, ran=True)
        sync_loop.start()
        code = self._run_phase("main", self.spec.command, env, log, job_start)
        return self._finalize(self._classify(code, None), sync_loop, log)

    def _capture_output_baseline(self, log: IO[bytes]) -> None:
        """Record what the checkout already had where the outputs go.

        Before `setup`, because a setup step may legitimately write into an
        output path and that *is* this job's doing. Once per job, not per
        attempt: a preempted job re-runs in the workdir the stopped attempt
        left, and re-scanning would record that attempt's own results as
        files "the checkout arrived with" -- never uploaded, never counted.
        The baseline is a fact about the checkout, and the checkout has not
        changed.
        """
        if not self.spec.outputs:
            return
        if paths.outputs_baseline_file(self.job_id).exists():
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

    def _stopped_before(self, phase: str, sync_loop: sync.SyncLoop, log: IO[bytes]) -> int | None:
        """A job asked to stop between phases never starts the next one."""
        requested = queue.stop_requested(self.job_id)
        if requested is None:
            return None
        self.kill_reason = requested
        self._log(log, f"{requested} before phase={phase}; not starting it")
        return self._finalize(self._classify(TERMINATED_EXIT_CODE, None), sync_loop, log)

    def _finalize_terminated(
        self, exc: _Terminated, sync_loop: sync.SyncLoop, log: IO[bytes]
    ) -> int:
        """The runner itself was signalled: stop the job, then end the attempt
        as whatever was asked of it -- the dispatcher's ladder reaches the
        runner while it is cancelling or preempting, and that must not turn a
        preempt into a failure or a cancel into a `terminated`."""
        name = signal.Signals(exc.signum).name
        self._log(log, f"runner received {name}; stopping the job")
        proc = self._current
        if proc is not None:
            self._kill(proc, "terminated", log)
            with contextlib.suppress(Exception):
                proc.wait(timeout=5)
        requested = queue.stop_requested(self.job_id)
        ran = self._main_started
        if requested == "cancelled":
            outcome = Outcome("cancelled", "cancelled", TERMINATED_EXIT_CODE, ran)
        elif requested == queue.PREEMPTED:
            outcome = Outcome("failed", queue.PREEMPTED, TERMINATED_EXIT_CODE, ran)
        else:
            outcome = Outcome("failed", "terminated", TERMINATED_EXIT_CODE, ran)
        return self._finalize(outcome, sync_loop, log)

    def _classify(self, code: int, failure_reason: str | None) -> Outcome:
        """The outcome of the phase that just ended, or that a stop landed
        before. `ran` is whether `main` started, whatever the phase."""
        ran = self._main_started
        if self.kill_reason == "cancelled":
            return Outcome("cancelled", "cancelled", code, ran)
        if self.kill_reason:
            return Outcome("failed", self.kill_reason, code, ran)
        if code == 0:
            return Outcome("succeeded", None, 0, ran)
        return Outcome("failed", failure_reason or f"exit {code}", code, ran)

    def _finalize(self, outcome: Outcome, sync_loop: sync.SyncLoop, log: IO[bytes]) -> int:
        """End the attempt: final sync, workdir, mirror, the one write, secrets.

        The order is the contract. The outputs are uploaded first, from inside
        the workdir; the workdir goes (or stays) and is measured; the log and
        state are mirrored and the mirror recorded; then the write that ends
        the attempt -- and only then, so the status stays `running` with
        `phase=sync` until this process has nothing left to do, and the
        dispatcher's patience for a runner in its final sync covers all of it.
        The mirror's `state.json` is put once more after that write, so the
        copy that survives this host says how the job ended; the log is not
        uploaded twice, since nothing below writes to it that a reader of the
        mirror needs.
        """
        self._ending = True
        jobs.update_state(self.job_id, phase="sync")
        problems: list[str] = []
        if not outcome.ran:
            self._log(log, "skipping the final output sync: the job's main phase never started")
        else:
            try:
                sync_loop.final()
            except sync.MissingOutput as exc:
                self._log(log, f"final sync found no outputs: {exc}")
                outcome = self._blame(outcome, "no-outputs", problems)
            except sync.SyncError as exc:
                self._log(log, f"final sync FAILED: {exc}")
                outcome = self._blame(outcome, "sync", problems)
            if sync_loop.last_error and outcome.status == "succeeded":
                self._log(log, f"periodic sync had errors: {sync_loop.last_error}")
        why = f" ({outcome.reason})" if outcome.reason else ""
        self._log(log, f"job {self.job_id} {outcome.status}{why}")
        coming_back = self._coming_back(outcome)
        # After the final sync, and only then: the outputs it just uploaded
        # live *inside* the workdir. Measured while still standing in it, so
        # `status` never has to walk a 67k-file venv to find out.
        jobs.update_state(
            self.job_id,
            workdir_bytes=self._cleanup_workdir("queued" if coming_back else outcome.status, log),
        )
        mirror = self._mirror(log)
        written = self._end(outcome, coming_back, problems, log)
        if mirror is not None and written is not None:
            warning = sync.put_state(
                self.job_id, mirror, runner=self.deps.command_runner, env=self.env or None
            )
            if warning:
                self._log(log, f"WARNING: {warning}")
        self._settle_secrets(written, log)
        return outcome.exit_code

    def _coming_back(self, outcome: Outcome) -> bool:
        """Is this attempt going back in the queue rather than ending?

        Only an attempt the preempt itself stopped: a job that ended for a
        reason of its own before the kill landed asked for nothing, and
        re-running it would be a retry nobody requested (`gpuc requeue` is
        that). This reading decides what the workdir is kept for; the write
        is `queue.next_attempt`'s own compare-and-set, which refuses if a
        cancel has landed since, and `_end` then ends the job `cancelled`.
        Nothing here asks whether the host is draining: a drain starts only
        when nothing is running, and a preempt is refused once it has.
        """
        return outcome.reason == queue.PREEMPTED and queue.is_preempted(self.job_id)

    def _cleanup_workdir(self, status: str, log: IO[bytes]) -> int:
        """Apply the spec's `cleanup:` to `workdir/`, through the one delete
        predicate, and return what the workdir holds afterwards.

        Asked with the status this attempt is about to write, since the write
        comes last and the outputs the workdir holds would go with it. A
        failure to delete is logged and nothing more: the job's own outcome has
        already been decided and uploaded, and turning a green run red over
        leftover disk would be the wrong trade.
        """
        state = dataclasses.replace(jobs.read_state(self.job_id), status=status)
        why = cleanup.may_delete(self.job_id, state, cleanup.WORKDIR, cleanup.Evidence(policy=True))
        if why is None:
            kept = cleanup.kept_outputs(self.job_id, self.spec, state)
            try:
                freed = cleanup.remove_workdir(self.job_id)
            except OSError as exc:
                self._log(log, f"could not remove workdir (cleanup={self.spec.cleanup}): {exc}")
                return cleanup.workdir_size(self.job_id) or 0
            what = f"the checkout, keeping {', '.join(kept)}" if kept else "workdir"
            self._log(
                log,
                f"removed {what} (cleanup={self.spec.cleanup}), freeing "
                f"{cleanup.human_bytes(freed)}; spec.json, state.json and log.txt are kept",
            )
            return 0
        if status == "queued":
            self._log(log, f"keeping workdir for the next attempt (cleanup={self.spec.cleanup})")
        else:
            self._log(log, f"keeping workdir: {why}")
        return cleanup.workdir_size(self.job_id) or 0

    def _mirror(self, log: IO[bytes]) -> destinations.S3 | None:
        """Mirror the log and state and record it. None when there is no
        mirror to put the final state to afterwards -- the host has none, or
        the upload failed, in which case `purge` will refuse this job dir: the
        only copy of the log lives here."""
        try:
            return sync.mirror_meta(
                self.job_id,
                self.config.s3_prefix,
                runner=self.deps.command_runner,
                env=self.env or None,
            )
        except sync.SyncError as exc:
            self._log(log, f"final state upload failed: {exc}")
            return None

    def _end(
        self, outcome: Outcome, coming_back: bool, problems: list[str], log: IO[bytes]
    ) -> str | None:
        """The write that ends the attempt, and the status it wrote.

        `queued` again for a preempted job, else the outcome; each a
        compare-and-set from `running`. A preempt a cancel overrode in the
        meantime ends the job `cancelled` -- the later request wins. None when
        the state was no longer `running` at all, which is nothing this
        process can repair and is logged rather than written over.
        """
        if coming_back and queue.next_attempt(self.job_id, ran=outcome.ran) is not None:
            return "queued"
        if outcome.reason == queue.PREEMPTED and queue.stop_requested(self.job_id) == "cancelled":
            outcome = replace(outcome, status="cancelled", reason="cancelled")
        if jobs.finish(self.job_id, outcome, problems=problems) is None:
            self._log(log, "state was no longer `running` at the end; nothing written")
            return None
        return outcome.status

    def _settle_secrets(self, written: str | None, log: IO[bytes]) -> None:
        """Delete the job's secrets file, unless something still needs it.

        Only now: the final sync and the mirror authenticate with what it
        holds. It stays for the next attempt of a preempted job, since nothing
        delivers secrets a second time (`gpuc preempt` never goes near the
        machine that holds them); `cleanup.settle_secrets` decides the rest.
        """
        if written == "queued":
            self._log(log, "preempted; keeping this job's secrets file for the next attempt")
            return
        kept = cleanup.settle_secrets(self.job_id, jobs.read_state(self.job_id))
        if kept:
            self._log(
                log,
                f"{kept}; keeping this job's secrets file so the host's drain can retry "
                f"the upload before the pod goes away",
            )

    @staticmethod
    def _blame(outcome: Outcome, sync_reason: str, problems: list[str]) -> Outcome:
        """A job that succeeded and then lost its outputs failed, for that
        reason. One that was already over for a reason of its own keeps it,
        and the upload failure is a problem noted beside it."""
        if outcome.status == "succeeded":
            return Outcome("failed", sync_reason, 1)
        if outcome.reason is None:
            return dataclasses.replace(outcome, reason=sync_reason)
        problems.append(sync_reason)
        return outcome


def run_job(
    job_id: str, assigned: Sequence[str], attempt: int, deps: RunnerDeps | None = None
) -> int:
    return JobRunner(job_id, assigned, attempt, deps).run()
