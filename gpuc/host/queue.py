"""The queue: every job whose state is `queued`, in `(priority, job_id)` order.

Nothing is stored about the queue except each job's own `state.json`. Nothing
here takes the dispatcher lock, so a wedged dispatcher can never lose an
enqueue; every change of a job's state goes through `jobs.transition`, whose
per-job lock is what keeps a cancel and a dispatch from both winning.
"""

from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass

from gpuc.host import jobs, paths
from gpuc.host.jobs import CANCEL, PREEMPT, JobSpec, JobState

PREEMPTED = "preempted"
"""The reason of an attempt stopped so that something else can have its GPUs.

Its own reason: a preempted job is not a failure of the job, and the log of
the attempt that was stopped should say what ended it.
"""

STOP_REASONS = {CANCEL: "cancelled", PREEMPT: PREEMPTED}
"""What each intent ends the attempt as, when the runner acts on it."""


@dataclass(order=True)
class QueueEntry:
    priority: int
    job_id: str


def enqueue(spec: JobSpec) -> str:
    """Accept a job: write its spec and initial state, then move its dir from
    `incoming/` into `jobs/` in one rename.

    The rename is the acceptance. `gpuc submit` builds the dir under
    `incoming/` -- the rsynced workdir lands there first -- so a submit that
    dies at any point before this leaves nothing under `jobs/` at all, and
    `cleanup.stale_incoming` sweeps what it left. There is no order of writes
    inside the dir to get right, because nothing reads it until it has moved.
    """
    paths.ensure_layout()
    job_id = spec.job_id
    accepted = paths.job_dir(job_id)
    if accepted.exists():
        raise FileExistsError(f"job {job_id} already exists on this host")
    staged = paths.incoming_job_dir(job_id)
    staged.mkdir(parents=True, exist_ok=True)
    (staged / "workdir").mkdir(exist_ok=True)
    (staged / "outputs").mkdir(exist_ok=True)
    jobs.atomic_write_json(staged / "spec.json", spec.to_dict())
    jobs.atomic_write_json(staged / "state.json", JobState.initial(spec).to_dict())
    (staged / "log.txt").touch()
    os.rename(staged, accepted)
    return job_id


def list_queued() -> list[QueueEntry]:
    entries: list[QueueEntry] = []
    for job_id in jobs.list_job_ids():
        try:
            state = jobs.read_state(job_id)
        except RuntimeError:
            continue
        if state.status == "queued":
            entries.append(QueueEntry(state.priority, job_id))
    return sorted(entries)


def is_queued(job_id: str) -> bool:
    return any(entry.job_id == job_id for entry in list_queued())


def claim(job_id: str, **state: object) -> bool:
    """Take a job out of the queue, recording what became of it.

    False when the job was no longer queued: cancelled, or claimed by another
    pass. The caller lists the queue and then claims each job it acts on, and
    the two are not one operation.
    """
    return jobs.transition(job_id, expect="queued", **state) is not None


def reorder(job_id: str, priority: int) -> bool:
    """Move a queued job. False for a job that is not queued, or not here."""
    if not paths.job_dir(job_id).is_dir():
        return False
    return jobs.transition(job_id, expect="queued", priority=priority) is not None


def cancel(job_id: str) -> str:
    """Request cancellation. Returns `cancelled` for a job that was queued,
    `cancelling` for a running one, and a finished job's own status.

    A queued job is cancelled here and now, so cancel works with no dispatcher
    running. A running job gets the intent, and its runner (or, as a backstop,
    the dispatcher) acts on it.
    """
    if not paths.job_dir(job_id).is_dir():
        raise FileNotFoundError(f"no such job: {job_id}")
    with jobs.locked(job_id):
        state = jobs.read_state(job_id)
        if state.finished:
            return state.status
        if state.status == "queued":
            state.status, state.reason, state.ended_at = "cancelled", "cancelled", jobs.utc_now()
            state.intent = None
            jobs.write_state(job_id, state)
            return "cancelled"
        # A cancel overrides a preempt: the job is not coming back.
        state.intent = CANCEL
        jobs.write_state(job_id, state)
        return "cancelling"


def is_cancelled(job_id: str) -> bool:
    try:
        return jobs.read_state(job_id).intent == CANCEL
    except RuntimeError:
        return False


def stop_requested(job_id: str) -> str | None:
    """The reason a running job's runner should stop it now, or None."""
    try:
        intent = jobs.read_state(job_id).intent
    except RuntimeError:
        return None
    return STOP_REASONS.get(intent or "")


def preempt(job_id: str, priority: int | None = None) -> str:
    """Stop a running job and queue it again, from the start.

    The intent says the job is coming back; the runner owns the kill and the
    final sync, and the dispatcher re-queues the job once its runner has
    stopped it (`requeue_preempted`, which also counts the next attempt).
    """
    if not paths.job_dir(job_id).is_dir():
        raise FileNotFoundError(f"no such job: {job_id}")
    if priority is not None and not 0 <= priority <= 99:
        raise ValueError(f"priority must be 0-99 (lower dispatches first), got {priority}")
    with jobs.locked(job_id):
        state = jobs.read_state(job_id)
        if state.finished:
            raise ValueError(
                f"job {job_id} has already {state.status}, so there is nothing to preempt; "
                f"`gpuc requeue {job_id}` submits it again"
            )
        if state.status != "running":
            raise ValueError(
                f"job {job_id} is {state.status}, not running, so it is already waiting its "
                f"turn; `gpuc reorder {job_id} --priority N` moves it"
            )
        if state.intent == CANCEL:
            raise ValueError(f"job {job_id} is already being cancelled, so it is not coming back")
        wanted = state.priority if priority is None else priority
        refuse_if_nothing_else_can_run(job_id, wanted)
        state.intent = PREEMPT
        state.priority = wanted
        jobs.write_state(job_id, state)
    return "preempting"


def queued_ahead_of(job_id: str, priority: int) -> QueueEntry | None:
    """The first queued job that would be dispatched before this one if it came
    back at `priority`, or None if it would go straight to the head of the queue.

    At the same priority the tie-break is the job id, and a preempted job's id
    is older than anything queued while it was running, so it sorts ahead of
    all of them.
    """
    mine = QueueEntry(priority, job_id)
    for entry in list_queued():
        if entry >= mine:
            break  # sorted, so nothing after this sorts earlier either
        if entry.job_id != job_id:
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
    if queued_ahead_of(job_id, priority) is not None:
        return
    ahead = [e for e in list_queued() if e.job_id != job_id]
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
        f"so preempting it would only start it over. Dispatch order is (priority, job id) "
        f"and this job was submitted first, so it wins a tie: give it "
        f"`--priority` above {first.priority} to queue it behind"
    )


def is_preempted(job_id: str) -> bool:
    try:
        return jobs.read_state(job_id).intent == PREEMPT
    except RuntimeError:
        return False


STOPPED_BY_US = (PREEMPTED, "runner-died", "terminated")
"""Reasons that mean the attempt ended because something stopped it, rather
than because the job itself was over. Only these come back: a job that failed
on its own in the seconds before the kill reached it asked for nothing, and
re-running it would be a retry nobody requested (`gpuc requeue` is that)."""


def stopped_for_preempt(state: JobState) -> bool:
    return state.reason in STOPPED_BY_US


def requeue_preempted(job_id: str) -> int | None:
    """Put a preempted job back in the queue, and say which attempt it is now.

    None means it is not going back, and the intent goes with the decision so
    nothing tries again: the job is already queued, it finished under its own
    steam before the kill reached it, it was cancelled while it stopped, or it
    has no workdir left to re-run from.

    The state is written fresh rather than patched. The job runs from the
    start, so the exit code, the end time and the GPUs of the attempt that was
    stopped would all be lies about a queued job. One write, under the job's
    lock: there is no half-done requeue to finish after a crash.
    """
    with jobs.locked(job_id):
        state = jobs.read_state(job_id)
        if state.intent != PREEMPT:
            return None
        if state.status == "queued":
            state.intent = None
            jobs.write_state(job_id, state)
            return None
        if (
            not state.finished
            or not stopped_for_preempt(state)
            or not paths.workdir(job_id).is_dir()
        ):
            state.intent = None
            jobs.write_state(job_id, state)
            return None
        attempt = state.attempt + 1
        jobs.write_state(
            job_id,
            JobState(
                status="queued",
                attempt=attempt,
                priority=state.priority,
                estimated_runtime_min=state.estimated_runtime_min,
                progress_pct=None,
                workdir_bytes=None,
            ),
        )
    _log_requeue(job_id, attempt, state.priority)
    return attempt


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
