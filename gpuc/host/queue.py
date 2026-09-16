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
    meant to come back; a kill request that cannot be written takes the marker
    back off again, because a marker nothing will ever act on re-runs the job
    the next time it fails for any reason at all.
    """
    if not paths.job_dir(job_id).is_dir():
        raise FileNotFoundError(f"no such job: {job_id}")
    if priority is not None and not 0 <= priority <= 99:
        raise ValueError(f"priority must be 0-99 (lower dispatches first), got {priority}")
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
    wanted = jobs.read_spec(job_id).priority if priority is None else priority
    refuse_if_nothing_else_can_run(job_id, wanted)
    if priority is not None:
        # In the spec rather than a marker: the spec is what
        # `requeue_preempted` queues the job at, and the only copy of a
        # running job's priority. `reorder` writes it the same way.
        jobs.update_spec(job_id, priority=priority)
    paths.preempt_file(job_id).touch()
    try:
        request_kill(job_id, PREEMPTED)
    except OSError:
        paths.preempt_file(job_id).unlink(missing_ok=True)
        raise
    return "preempting"


def queued_ahead_of(job_id: str, priority: int) -> QueueEntry | None:
    """The first queued job that would be dispatched before this one if it came
    back at `priority`, or None if it would go straight to the head of the queue.

    Marker names, because that is what the dispatcher orders by: at the same
    priority the tie-break is the job id, and a preempted job's id is older
    than anything queued while it was running, so it sorts ahead of all of them.
    """
    marker = marker_name(priority, job_id)
    for entry in list_queued():
        if entry.marker.name >= marker:
            break  # sorted, so nothing after this sorts earlier either
        if entry.job_id != job_id and not is_cancelled(entry.job_id):
            return entry
    return None


def refuse_if_nothing_else_can_run(job_id: str, priority: int) -> None:
    """Refuse a preempt that would only stop this job and start it again.

    The command hands a host to a job that is waiting for it, and it costs the
    running job everything it has done so far. With nothing that could take its
    place that is a pure loss, so it is a refusal with the way out in it rather
    than a surprise discovered in the log afterwards.
    """
    if paths.draining_file().exists():
        raise ValueError(
            f"this host is draining, so it will not start anything else: preempting "
            f"job {job_id} would throw away what it has done for nothing"
        )
    if paths.paused_file().exists():
        raise ValueError(
            f"this host is paused, so it is dispatching nothing: preempting job {job_id} "
            f"would throw away what it has done for nothing. Clear the pause first "
            f"(`gpuc host resume <host>`)"
        )
    if queued_ahead_of(job_id, priority) is not None:
        return
    ahead = [e for e in list_queued() if e.job_id != job_id and not is_cancelled(e.job_id)]
    if not ahead:
        raise ValueError(
            f"nothing else is queued on this host, so preempting job {job_id} would stop it "
            f"and start it again from the beginning. Queue the job you want to run first, "
            f"then preempt this one"
        )
    first = ahead[0]
    raise ValueError(
        f"job {job_id} would come back at priority {priority} and be dispatched ahead of "
        f"every job now waiting (the next is {first.job_id} at priority {first.priority}), "
        f"so preempting it would only start it over. Dispatch order is <priority>-<job id> "
        f"and this job was submitted first, so it wins a tie: give it "
        f"`--priority` above {first.priority} to queue it behind"
    )


def is_preempted(job_id: str) -> bool:
    return paths.preempt_file(job_id).exists()


STOPPED_BY_US = ("preempted", "runner-died", "terminated")
"""Reasons that mean the attempt ended because something stopped it, rather
than because the job itself was over. Only these come back: a job that failed
on its own in the seconds before the kill reached it asked for nothing, and
re-running it would be a retry nobody requested (`gpuc requeue` is that)."""


def stopped_for_preempt(state: JobState) -> bool:
    # `_blame` composes a compound reason -- `preempted+sync` when the final
    # upload failed too -- and the first part is what ended the job.
    return (state.reason or "").split("+")[0] in STOPPED_BY_US


def requeue_preempted(job_id: str) -> int | None:
    """Put a preempted job back in the queue, and say which attempt it is now.

    None means it is not going back, and the marker goes with the decision so
    nothing tries again: the job is already queued, it finished under its own
    steam before the kill reached it, it was cancelled while it stopped, or it
    has no workdir left to re-run from.

    The state is written fresh rather than patched. The job runs from the
    start, so the exit code, the end time and the GPUs of the attempt that was
    stopped would all be lies about a queued job.

    Ordering is for a process that dies in the middle of this: the queue marker
    is written before the preempt marker is removed, so the worst interruption
    leaves a job that is queued *and* still asking to be queued, which the
    first branch below turns into a no-op. Removing the preempt marker first --
    as this did -- loses the job entirely if the write after it fails.
    """
    state = jobs.read_state(job_id)
    if state.status == "queued":
        return _finish_interrupted_requeue(job_id, state)
    if not state.finished or not stopped_for_preempt(state):
        paths.preempt_file(job_id).unlink(missing_ok=True)
        return None
    if is_cancelled(job_id) or not paths.workdir(job_id).is_dir():
        paths.preempt_file(job_id).unlink(missing_ok=True)
        return None
    # Before the new state: it is still the request the stopped attempt was
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
    paths.preempt_file(job_id).unlink(missing_ok=True)
    return attempt


def _finish_interrupted_requeue(job_id: str, state: JobState) -> int | None:
    """Complete a re-queue that got as far as the queued state and no further.

    The attempt has already been counted, so this finishes the move rather
    than making a second one: the marker if it is missing, then the preempt
    marker. Returns the attempt when there was something to finish, None when
    the job was simply already back.
    """
    if find_marker(job_id) is not None:
        paths.preempt_file(job_id).unlink(missing_ok=True)
        return None
    priority = jobs.read_spec(job_id).priority
    (paths.queue_dir() / marker_name(priority, job_id)).touch()
    paths.preempt_file(job_id).unlink(missing_ok=True)
    return state.attempt


def _log_requeue(job_id: str, attempt: int, priority: int) -> None:
    """Say in the job's own log why it is starting over.

    `gpuc logs` is where somebody looks at a job that has restarted, and
    without this the log simply runs two attempts together.
    """
    note(
        job_id,
        f"preempted; queued again as attempt {attempt} at priority {priority}, "
        f"to run from the start in this same workdir",
    )


def note(job_id: str, message: str) -> None:
    """Append one `>>>` line to the job's own log, as the runner does.

    Never raises, and never the only record of anything: the dispatcher log has
    its own line. This one is for whoever is reading `gpuc logs <id>` and needs
    to know why the output stops and starts again.
    """
    with contextlib.suppress(OSError), paths.log_file(job_id).open("ab") as log:
        log.write(f">>> {message}\n".encode())
