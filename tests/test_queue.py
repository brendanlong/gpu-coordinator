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


def running_job(**overrides: object) -> str:
    job_id = queue.enqueue(make_spec(**overrides))
    queue.remove_marker(job_id)
    jobs.update_state(job_id, status="running", gpus=["GPU-x"], started_at=jobs.utc_now())
    return job_id


def test_preempt_marks_the_job_and_asks_its_runner_to_stop(gpuc_home: Path) -> None:
    job_id = running_job()
    assert queue.preempt(job_id) == "preempting"
    assert queue.is_preempted(job_id)
    assert queue.kill_reason(job_id) == "preempted"
    # Nothing here ends the job: the runner owns the kill and the final sync.
    assert jobs.read_state(job_id).status == "running"
    assert queue.list_queued() == []


def test_preempt_can_lower_the_priority_it_comes_back_at(gpuc_home: Path) -> None:
    job_id = running_job(priority=50)
    queue.preempt(job_id, priority=80)
    assert jobs.read_spec(job_id).priority == 80
    assert queue.requeue_preempted(job_id) == 2
    assert queue.list_queued()[0].marker.name.startswith("80-")


def test_a_queued_or_finished_job_cannot_be_preempted(gpuc_home: Path) -> None:
    queued = queue.enqueue(make_spec())
    with pytest.raises(ValueError, match="reorder"):
        queue.preempt(queued)
    done = running_job()
    jobs.update_state(done, status="succeeded", ended_at=jobs.utc_now())
    with pytest.raises(ValueError, match="requeue"):
        queue.preempt(done)
    with pytest.raises(FileNotFoundError):
        queue.preempt("nope")


def test_a_job_already_being_cancelled_is_not_coming_back(gpuc_home: Path) -> None:
    job_id = running_job()
    queue.cancel(job_id)
    with pytest.raises(ValueError, match="cancelled"):
        queue.preempt(job_id)
    assert not queue.is_preempted(job_id)


def test_requeue_after_preempt_starts_the_job_over_as_the_next_attempt(gpuc_home: Path) -> None:
    """A queued job that still carried the stopped attempt's exit code, GPUs and
    end time would be a job that reads as finished to everything but the queue."""
    job_id = running_job(priority=20)
    queue.preempt(job_id)
    jobs.update_state(
        job_id, status="failed", reason="preempted", exit_code=143, ended_at=jobs.utc_now()
    )
    assert queue.requeue_preempted(job_id) == 2

    state = jobs.read_state(job_id)
    assert (state.status, state.attempt) == ("queued", 2)
    assert (state.exit_code, state.ended_at, state.started_at, state.gpus) == (None, None, None, [])
    assert jobs.read_spec(job_id).attempt == 2
    assert [e.job_id for e in queue.list_queued()] == [job_id]
    # The kill request was the stopped attempt's: left behind, the runner of
    # the new one would find it and stop that too.
    assert queue.kill_reason(job_id) is None
    assert not queue.is_preempted(job_id)
    assert "queued again as attempt 2" in paths.log_file(job_id).read_text()


def test_a_job_cancelled_while_it_was_stopping_does_not_come_back(gpuc_home: Path) -> None:
    job_id = running_job()
    queue.preempt(job_id)
    queue.cancel(job_id)
    assert queue.requeue_preempted(job_id) is None
    assert queue.list_queued() == []
    assert not queue.is_preempted(job_id)


def test_a_job_whose_workdir_is_gone_has_nothing_to_re_run(gpuc_home: Path) -> None:
    """Its code was rsynced here once, at submit, and `gpuc preempt` never goes
    near the machine that holds it."""
    import shutil

    job_id = running_job()
    queue.preempt(job_id)
    shutil.rmtree(paths.workdir(job_id))
    assert queue.requeue_preempted(job_id) is None
    assert queue.list_queued() == []


def test_a_job_that_finished_before_the_kill_reached_it_is_not_run_again(
    gpuc_home: Path,
) -> None:
    """ "Put it back in the queue" was about the work still to do; the job did
    it all in the seconds it took the kill to land."""
    job_id = running_job()
    queue.preempt(job_id)
    jobs.update_state(job_id, status="succeeded", exit_code=0, ended_at=jobs.utc_now())
    assert queue.requeue_preempted(job_id) is None
    assert queue.list_queued() == []
    assert jobs.read_state(job_id).status == "succeeded"
