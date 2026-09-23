"""Output upload: which files go to which `Destination`, and when.

Every upload is recorded in the job's `state.json` (`jobs.record_upload`), one
record per destination: the last time it succeeded and the last error. That
record is the only account of whether a job's outputs are safe -- `status`,
`purge` and a rental's drain all read it and nothing re-derives it.
"""

from __future__ import annotations

import contextlib
import threading
import time
import traceback
from collections.abc import Sequence
from pathlib import Path

from gpuc.host import baseline, destinations, jobs, paths, queue
from gpuc.host.destinations import (
    DEFAULT_TIMEOUT_S,
    CommandResult,
    CommandRunner,
    Env,
    MissingOutput,
    SyncError,
    run_command,
)
from gpuc.host.jobs import JobSpec, Output

__all__ = [
    "DEFAULT_TIMEOUT_S",
    "CommandResult",
    "CommandRunner",
    "Env",
    "MissingOutput",
    "SyncError",
    "SyncLoop",
    "TooManyRecentFiles",
    "final_meta_sync",
    "mirror_for",
    "mirror_meta",
    "put_state",
    "recently_modified",
    "run_command",
    "sync_job_meta",
    "sync_output",
    "sync_outputs",
]

MIN_AGE_S = 10.0
MAX_EXCLUDES = 200
MISSING_TICKS = 2
"""Periodic ticks an output path may be missing before that is recorded as the
destination's failure. A job that writes its first checkpoint late is not
failing at the first tick; one whose path is still missing at the third has
an `outputs:` that does not match what it writes (issue #72)."""


class TooManyRecentFiles(SyncError):
    """More files are in flight than we are willing to name on the command line."""


def recently_modified(root: Path, min_age_s: float = MIN_AGE_S) -> list[str]:
    """Relative paths under ``root`` written within the last ``min_age_s``.

    We exclude these by name rather than copying the tree to a staging dir:
    checkpoints are routinely tens of gigabytes, so a staging copy would cost
    more disk and IO than the upload itself, and the next tick picks the file
    up anyway once it has stopped changing.
    """
    if not root.is_dir():
        return []
    cutoff = time.time() - min_age_s
    out: list[str] = []
    for path in root.rglob("*"):
        if path.is_file():
            try:
                if path.stat().st_mtime > cutoff:
                    out.append(str(path.relative_to(root)))
            except OSError:
                continue
    return sorted(out)


def _in_flight(root: Path, min_age_s: float) -> list[str]:
    """Files still being written, to leave out of this tick.

    Capped: a job that writes thousands of small files in a tick would
    otherwise build an argv megabytes long (and blow E2BIG). Skipping the tick
    costs nothing -- the next one picks the files up.
    """
    names = recently_modified(root, min_age_s)
    if len(names) > MAX_EXCLUDES:
        raise TooManyRecentFiles(
            f"{len(names)} files under {root} were modified in the last {min_age_s:g}s "
            f"(cap {MAX_EXCLUDES}); skipping this sync tick"
        )
    return names


def mirror_for(job_id: str, s3_prefix: str | None) -> destinations.S3 | None:
    """The host's mirror of this job's log and state, if it has one."""
    if not s3_prefix:
        return None
    return destinations.S3(f"{s3_prefix.rstrip('/')}/jobs/{job_id}")


def sync_output(
    output: Output,
    workdir: Path,
    job_id: str,
    *,
    min_age_s: float = MIN_AGE_S,
    runner: CommandRunner = run_command,
    timeout: float | None = DEFAULT_TIMEOUT_S,
    env: Env = None,
    baseline_entries: baseline.Entries | None = None,
    record_missing: bool = True,
) -> None:
    """Upload one output to each of its destinations, recording each result.

    `record_missing` is whether a path that is not there yet goes in the
    record: the loop withholds it for the first few ticks, since a job that
    writes its first checkpoint an hour in has not failed anything.
    """
    local = workdir / output.path.format(job_id=job_id)
    entries = baseline_entries or {}
    # Files that were in the checkout and have not been touched are not this
    # job's output; uploading them would publish the last run's results under
    # this run's name.
    exclude = () if baseline.too_many(entries) else tuple(baseline.unchanged(local, entries))
    # Before anything is recorded: a tick with too much in flight is skipped,
    # and skipping is not a failure at any destination.
    in_flight = _in_flight(local, min_age_s)
    errors: list[SyncError] = []
    for destination in destinations.of(output, job_id):
        try:
            if not local.exists():
                raise MissingOutput(f"output path does not exist: {local}")
            if entries and not baseline.has_new_content(local, entries):
                raise MissingOutput(
                    f"nothing new under {local}: every file there was already in the checkout"
                )
            destination.upload_dir(
                local,
                exclude=[*exclude, *in_flight],
                runner=runner,
                timeout=timeout,
                env=env,
            )
        except SyncError as exc:
            errors.append(exc)
            if record_missing or not isinstance(exc, MissingOutput):
                jobs.record_upload(job_id, destination.uri, output.path, error=str(exc))
            continue
        jobs.record_upload(job_id, destination.uri, output.path, ok_at=jobs.utc_now())
    if errors:
        raise _combined(errors)


def _combined(errors: Sequence[SyncError]) -> SyncError:
    # Keep the specific kind when every failure agrees, so the runner can still
    # tell "produced nothing" from "upload broke".
    kind = type(errors[0]) if len({type(e) for e in errors}) == 1 else SyncError
    return kind("; ".join(str(e) for e in errors))


def sync_outputs(
    outputs: Sequence[Output],
    workdir: Path,
    job_id: str,
    *,
    min_age_s: float = MIN_AGE_S,
    runner: CommandRunner = run_command,
    timeout: float | None = DEFAULT_TIMEOUT_S,
    env: Env = None,
    baseline_map: baseline.Baseline | None = None,
    record_missing: bool = True,
) -> None:
    errors: list[SyncError] = []
    for output in outputs:
        try:
            sync_output(
                output,
                workdir,
                job_id,
                min_age_s=min_age_s,
                runner=runner,
                timeout=timeout,
                env=env,
                baseline_entries=(baseline_map or {}).get(baseline.output_key(output, job_id)),
                record_missing=record_missing,
            )
        except SyncError as exc:
            errors.append(exc)
    if errors:
        raise _combined(errors)


def sync_job_meta(
    job_id: str,
    s3_prefix: str | None,
    *,
    runner: CommandRunner = run_command,
    timeout: float | None = 300.0,
    env: Env = None,
) -> None:
    mirror = mirror_for(job_id, s3_prefix)
    if mirror is None:
        return
    for path in (paths.log_file(job_id), paths.state_file(job_id)):
        if path.exists():
            mirror.put_file(path, path.name, runner=runner, timeout=timeout, env=env)


def mirror_meta(
    job_id: str,
    s3_prefix: str | None,
    *,
    runner: CommandRunner = run_command,
    timeout: float | None = 300.0,
    env: Env = None,
) -> destinations.S3 | None:
    """Mirror a job's log and state, then record that it happened.

    The mirror's upload record is the precondition `purge` checks before
    deleting a job dir, so it is written only once the upload has actually
    succeeded -- a raised SyncError, or no `s3_prefix` at all, leaves no
    record and the job dir unpurgeable. Writing it changes `state.json`, and
    so does whatever the caller writes next, so the mirror is returned for
    `put_state` to bring its copy up to date once the caller is done.
    """
    mirror = mirror_for(job_id, s3_prefix)
    if mirror is None:
        return None
    try:
        sync_job_meta(job_id, s3_prefix, runner=runner, timeout=timeout, env=env)
    except SyncError as exc:
        jobs.record_upload(job_id, mirror.uri, None, error=str(exc))
        raise
    jobs.record_upload(job_id, mirror.uri, None, ok_at=jobs.utc_now())
    return mirror


def put_state(
    job_id: str,
    mirror: destinations.S3,
    *,
    runner: CommandRunner = run_command,
    timeout: float | None = 300.0,
    env: Env = None,
) -> str | None:
    """Upload `state.json` once more, so the mirror's copy carries what was
    written since `mirror_meta`: the record of the mirror itself, and the
    write that ended the job.

    A failure is a warning, not an error: the local state is the authority on
    "was this backed up", and log + state are already in S3, so a mirrored
    `state.json` one revision behind is not worth undoing the record.
    """
    state = paths.state_file(job_id)
    try:
        if state.exists():
            mirror.put_file(state, state.name, runner=runner, timeout=timeout, env=env)
    except SyncError as exc:
        return f"the mirrored state.json is one revision behind (re-upload failed): {exc}"
    return None


def final_meta_sync(
    job_id: str,
    s3_prefix: str | None,
    *,
    runner: CommandRunner = run_command,
    timeout: float | None = 300.0,
    env: Env = None,
) -> str | None:
    """`mirror_meta` then `put_state`, for a job that is already over: the
    drain's last mirror of every job on a host about to go away."""
    mirror = mirror_meta(job_id, s3_prefix, runner=runner, timeout=timeout, env=env)
    if mirror is None:
        return None
    return put_state(job_id, mirror, runner=runner, timeout=timeout, env=env)


class SyncLoop:
    """Background periodic sync; `final()` runs one last synchronous pass.

    The final pass has no wall-clock timeout: it is the last chance to save a
    run's outputs, and a 30-minute cap on a 200 GB checkpoint upload would
    throw away exactly the work that is most expensive to recompute.
    """

    def __init__(
        self,
        spec: JobSpec,
        workdir: Path,
        s3_prefix: str | None,
        *,
        runner: CommandRunner = run_command,
        min_age_s: float = MIN_AGE_S,
        env: Env = None,
    ) -> None:
        self._spec = spec
        self._workdir = workdir
        self._s3_prefix = s3_prefix
        self._runner = runner
        self._min_age_s = min_age_s
        # The job's environment, secrets included: uploads authenticate as the
        # job, not as whatever the dispatcher happened to inherit.
        self._env = env
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._tick_lock = threading.Lock()
        self.last_error: str | None = None
        self._missing_ticks = 0
        """Consecutive periodic ticks that found an output path missing."""

    def _note(self, message: str) -> None:
        queue.note(self._spec.job_id, f"sync: {message}")

    def _tick(
        self,
        min_age_s: float,
        timeout: float | None = DEFAULT_TIMEOUT_S,
        *,
        record_missing: bool = True,
        meta: bool = True,
    ) -> None:
        with self._tick_lock:
            sync_outputs(
                self._spec.outputs,
                self._workdir,
                self._spec.job_id,
                min_age_s=min_age_s,
                runner=self._runner,
                timeout=timeout,
                env=self._env,
                baseline_map=baseline.read(self._spec.job_id),
                record_missing=record_missing,
            )
            if meta:
                sync_job_meta(
                    self._spec.job_id,
                    self._s3_prefix,
                    runner=self._runner,
                    timeout=timeout,
                    env=self._env,
                )

    def _loop(self) -> None:
        interval = max(1, self._spec.sync_interval_s)
        while not self._stop.wait(interval):
            try:
                self._tick(self._min_age_s, record_missing=self._missing_ticks >= MISSING_TICKS)
            except MissingOutput as exc:
                # Not an error yet: the job may simply not have written the
                # path. Once it has stayed missing for `MISSING_TICKS` it goes
                # in the upload record, which is where `status` shows it.
                self._missing_ticks += 1
                self._note(f"WARNING: {exc}")
            except TooManyRecentFiles as exc:
                self._note(f"WARNING: {exc}")
            except SyncError as exc:
                self.last_error = str(exc)
                self._note(str(exc))
            except BaseException as exc:
                self.last_error = f"periodic sync raised {exc!r}"
                self._note(f"{self.last_error}\n{traceback.format_exc()}")
            else:
                self._missing_ticks = 0

    def start(self) -> None:
        if not self._spec.outputs and not self._s3_prefix:
            return
        self._thread = threading.Thread(target=self._loop, name="gpuc-sync", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the loop and wait for any tick already in flight."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def final(self) -> None:
        """Stop the loop and upload every output once more, files just
        written included.

        Every output's record is cleared first: an earlier tick having worked
        says nothing about the files the job wrote in its last minute, and
        `purge` reads the records as "everything this job produced is
        somewhere else". The log and state are not part of it: the runner
        mirrors those itself, once, after the workdir is settled and measured
        (`mirror_meta`), so uploading them here would be the same bytes twice.
        """
        self.stop()
        with contextlib.suppress(RuntimeError, OSError):
            jobs.clear_output_uploads(self._spec.job_id)
        self._tick(min_age_s=0.0, timeout=None, meta=False)
