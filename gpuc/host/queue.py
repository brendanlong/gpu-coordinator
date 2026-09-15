"""The queue itself: empty marker files under ``~/.gpuc/queue`` whose lexical
order is the dispatch order.

Nothing here takes the dispatcher lock. A wedged dispatcher must never be able
to lose an enqueue, so enqueue only writes files.
"""

from __future__ import annotations

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
    marker = find_marker(job_id)
    if marker is None:
        return False
    marker.rename(marker.parent / marker_name(priority, job_id))
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


def running_job_ids() -> list[str]:
    return [j for j in jobs.list_job_ids() if _status(j) == "running"]


def _status(job_id: str) -> str | None:
    try:
        return jobs.read_state(job_id).status
    except (RuntimeError, FileNotFoundError):
        return None
