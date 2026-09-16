from __future__ import annotations

from pathlib import Path

import pytest

from gpuc.host import jobs, paths, queue
from tests.conftest import make_spec


def test_enqueue_writes_spec_state_and_marker(gpuc_home: Path) -> None:
    spec = make_spec(name="a", priority=7)
    job_id = queue.enqueue(spec)
    assert jobs.read_spec(job_id).name == "a"
    assert jobs.read_state(job_id).status == "queued"
    assert paths.workdir(job_id).is_dir()
    assert paths.log_file(job_id).exists()
    assert [e.job_id for e in queue.list_queued()] == [job_id]
    assert queue.list_queued()[0].marker.name.startswith("07-")


def test_dispatch_order_is_lexical_by_priority_then_id(gpuc_home: Path) -> None:
    low = queue.enqueue(make_spec(priority=90))
    high = queue.enqueue(make_spec(priority=10))
    middle = queue.enqueue(make_spec(priority=50))
    assert [e.job_id for e in queue.list_queued()] == [high, middle, low]


def test_enqueue_does_not_need_the_dispatcher_lock(gpuc_home: Path) -> None:
    from gpuc.host.dispatcher import DispatcherLock

    lock = DispatcherLock()
    assert lock.acquire()
    try:
        job_id = queue.enqueue(make_spec())
        assert [e.job_id for e in queue.list_queued()] == [job_id]
    finally:
        lock.release()


def test_reorder_renames_the_marker_and_records_it_in_the_spec(gpuc_home: Path) -> None:
    """The marker is the move; the spec is the only copy that outlives it, and
    a running job's priority comes from nowhere else."""
    first = queue.enqueue(make_spec(priority=50))
    second = queue.enqueue(make_spec(priority=50))
    assert queue.reorder(second, 1)
    assert [e.job_id for e in queue.list_queued()] == [second, first]
    assert jobs.read_spec(second).priority == 1
    assert jobs.read_spec(first).priority == 50
    assert not queue.reorder("no-such-job", 1)


def test_a_spec_that_cannot_be_rewritten_does_not_undo_the_move(
    gpuc_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The queue is then in the order that was asked for and only the report of
    it is stale, which is the better of the two failures."""
    job_id = queue.enqueue(make_spec(priority=50))

    def unwritable(*_: object, **__: object) -> None:
        raise OSError("read-only file system")

    monkeypatch.setattr(queue.jobs, "update_spec", unwritable)
    assert queue.reorder(job_id, 3)
    assert queue.list_queued()[0].marker.name.startswith("03-")


def test_cancel_of_a_queued_job_is_immediate(gpuc_home: Path) -> None:
    job_id = queue.enqueue(make_spec())
    assert queue.cancel(job_id) == "cancelled"
    assert queue.list_queued() == []
    state = jobs.read_state(job_id)
    assert state.status == "cancelled"
    assert state.reason == "cancelled"
    assert queue.is_cancelled(job_id)


def test_cancel_of_a_running_job_only_marks(gpuc_home: Path) -> None:
    job_id = queue.enqueue(make_spec())
    queue.remove_marker(job_id)
    jobs.update_state(job_id, status="running")
    assert queue.cancel(job_id) == "running"
    assert queue.is_cancelled(job_id)
    assert jobs.read_state(job_id).status == "running"


def test_cancel_of_a_finished_job_is_a_noop(gpuc_home: Path) -> None:
    job_id = queue.enqueue(make_spec())
    queue.remove_marker(job_id)
    jobs.update_state(job_id, status="succeeded", ended_at=jobs.utc_now())
    assert queue.cancel(job_id) == "succeeded"


def test_cancel_of_an_unknown_job_raises(gpuc_home: Path) -> None:
    with pytest.raises(FileNotFoundError):
        queue.cancel("nope")


def test_stray_files_in_the_queue_dir_are_ignored(gpuc_home: Path) -> None:
    job_id = queue.enqueue(make_spec())
    (paths.queue_dir() / ".nfs0001").touch()
    (paths.queue_dir() / "README").touch()
    assert [e.job_id for e in queue.list_queued()] == [job_id]
