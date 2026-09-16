"""The queue itself: empty marker files under ``~/.gpuc/queue`` whose lexical
order is the dispatch order.

Nothing here takes the dispatcher lock. A wedged dispatcher must never be able
to lose an enqueue, so enqueue only writes files.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from pathlib import Path

from gpuc.host import jobs, paths
from gpuc.host.jobs import JobSpec, JobState


def marker_name(priority: int, job_id: str) -> str:
    return f"{max(0, min(99, priority)):02d}-{job_id}"


def job_id_of_marker(name: str) -> str:
    return name.split("-", 1)[1]


@dataclass
class QueueEntry:
    priority: int
    job_id: str
    marker: Path


def enqueue(spec: JobSpec) -> str:
    paths.ensure_layout()
    paths.ensure_job_layout(spec.job_id)
    jobs.write_spec(spec)
    jobs.write_state(spec.job_id, JobState(status="queued", attempt=spec.attempt))
    paths.log_file(spec.job_id).touch()
    # Marker last: a job is only dispatchable once its spec and state exist.
    (paths.queue_dir() / marker_name(spec.priority, spec.job_id)).touch()
    return spec.job_id


def list_queued() -> list[QueueEntry]:
    directory = paths.queue_dir()
    if not directory.is_dir():
        return []
    entries: list[QueueEntry] = []
    for marker in sorted(directory.iterdir(), key=lambda p: p.name):
        name = marker.name
        if name.startswith(".") or "-" not in name:
            continue
        prefix = name.split("-", 1)[0]
        if not prefix.isdigit():
            continue
        entries.append(QueueEntry(int(prefix), job_id_of_marker(name), marker))
    return entries


def find_marker(job_id: str) -> Path | None:
    for entry in list_queued():
        if entry.job_id == job_id:
            return entry.marker
    return None


def remove_marker(job_id: str) -> bool:
    marker = find_marker(job_id)
    if marker is None:
        return False
    marker.unlink(missing_ok=True)
    return True


def reorder(job_id: str, priority: int) -> bool:
    """Move a queued job, and record the move in its spec.

    The marker name is what the dispatcher orders by, so renaming it is the
    move. The spec is updated too because it is the only place a priority
    survives dispatch: without it `gpuc status` could say what a *queued* job's
    priority is and nothing at all about a running one's. A spec that cannot be
    rewritten does not undo the move -- the queue is still in the order that
    was asked for, and only the report of it is stale.
    """
    marker = find_marker(job_id)
    if marker is None:
        return False
    marker.rename(marker.parent / marker_name(priority, job_id))
    with contextlib.suppress(OSError, RuntimeError, ValueError):
        jobs.update_spec(job_id, priority=priority)
    return True


def cancel(job_id: str) -> str:
    """Request cancellation. Returns the resulting status.

    A queued job is cancelled here and now, so cancel works with no dispatcher
    running. A running job gets a marker that the runner (and, as a backstop,
    the dispatcher) acts on.
    """
    if not paths.job_dir(job_id).is_dir():
        raise FileNotFoundError(f"no such job: {job_id}")
    paths.cancel_file(job_id).touch()
    was_queued = remove_marker(job_id)
    state = jobs.read_state(job_id)
    if state.finished:
        return state.status
    if was_queued or state.status == "queued":
        jobs.update_state(job_id, status="cancelled", reason="cancelled", ended_at=jobs.utc_now())
        return "cancelled"
    return state.status


def is_cancelled(job_id: str) -> bool:
    return paths.cancel_file(job_id).exists()


def request_kill(job_id: str, reason: str) -> None:
    """Ask the runner to stop this job and record `reason` as why."""
    jobs.atomic_write_text(paths.kill_file(job_id), f"{reason}\n")


def kill_reason(job_id: str) -> str | None:
    try:
        return paths.kill_file(job_id).read_text().strip() or None
    except OSError:
        return None


PREEMPTED = "preempted"
"""The kill reason of a job stopped so that something else can have its GPUs.

Its own reason, like `ttl` and `low-util-pause` are: a preempted job is not a
failure of the job, and the log of the attempt that was stopped should say
which of the three ended it.
"""


def preempt(job_id: str, priority: int | None = None) -> str:
    """Stop a running job and queue it again, from the start.

    Two steps, because the runner owns the kill: the marker says the job is
    coming back, and the kill request stops it. The dispatcher re-queues it
    once its runner has stopped it and synced whatever it had produced -- see
    `requeue_preempted`, which is also what decides the new attempt number.

    The marker is written first. A runner that stops between the two writes
    would otherwise end the job for good, with nothing left saying it was
    meant to come back.
    """
    if not paths.job_dir(job_id).is_dir():
        raise FileNotFoundError(f"no such job: {job_id}")
    state = jobs.read_state(job_id)
    if state.finished:
        raise ValueError(
            f"job {job_id} has already {state.status}, so there is nothing to preempt; "
            f"`gpuc requeue {job_id}` submits it again"
        )
    if state.status != "running":
        raise ValueError(
            f"job {job_id} is {state.status}, not running, so it is already waiting its turn; "
            f"`gpuc reorder {job_id} --priority N` moves it"
        )
    if is_cancelled(job_id):
        raise ValueError(f"job {job_id} is already being cancelled, so it is not coming back")
    if priority is not None:
        # Before the kill, and in the spec rather than a marker: the spec is
        # what `requeue_preempted` queues the job at, and the only copy of a
        # running job's priority. `reorder` writes it the same way.
        jobs.update_spec(job_id, priority=priority)
    paths.preempt_file(job_id).touch()
    request_kill(job_id, PREEMPTED)
    return "preempting"


def is_preempted(job_id: str) -> bool:
    return paths.preempt_file(job_id).exists()


def requeue_preempted(job_id: str) -> int | None:
    """Put a preempted job back in the queue, and say which attempt it is now.

    None means it is not going back: it finished before the kill reached it,
    it was cancelled while it stopped, or it has no workdir left to re-run
    from. Either way the marker goes, so nothing tries again on the next pass.

    The state is written fresh rather than patched. The job runs from the
    start, so the exit code, the end time and the GPUs of the attempt that was
    stopped would all be lies about a queued job.
    """
    paths.preempt_file(job_id).unlink(missing_ok=True)
    state = jobs.read_state(job_id)
    if state.status == "succeeded":
        # It beat the kill to the finish line. Running the work again is not
        # what "put it back in the queue" was asking for.
        return None
    if is_cancelled(job_id):
        return None
    if not paths.workdir(job_id).is_dir():
        return None
    # Before anything else: it is still the request the stopped attempt was
    # given, and a runner that found it would kill the new attempt on its
    # first poll.
    paths.kill_file(job_id).unlink(missing_ok=True)
    attempt = state.attempt + 1
    spec = jobs.update_spec(job_id, attempt=attempt)
    _log_requeue(job_id, attempt, spec.priority)
    jobs.write_state(job_id, JobState(status="queued", attempt=attempt))
    # Marker last, exactly as `enqueue` writes it: a job is only dispatchable
    # once its spec and state say it is queued.
    (paths.queue_dir() / marker_name(spec.priority, job_id)).touch()
    return attempt


def _log_requeue(job_id: str, attempt: int, priority: int) -> None:
    """Say in the job's own log why it is starting over.

    `gpuc logs` is where somebody looks at a job that has restarted, and
    without this the log simply runs two attempts together.
    """
    with contextlib.suppress(OSError), paths.log_file(job_id).open("ab") as log:
        log.write(
            f">>> preempted; queued again as attempt {attempt} at priority "
            f"{priority}, to run from the start\n".encode()
        )
