from __future__ import annotations

import shutil
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
    assert queue.cancel(job_id) == "cancelling"
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


def waiting_job(priority: int = 1) -> str:
    """Something queued that would take the preempted job's place.

    Every preempt needs one: with nothing waiting, the command refuses rather
    than stop a job and start it again.
    """
    return queue.enqueue(make_spec(priority=priority))


def test_preempt_marks_the_job_and_asks_its_runner_to_stop(gpuc_home: Path) -> None:
    waiting = waiting_job()
    job_id = running_job()
    assert queue.preempt(job_id) == "preempting"
    assert queue.is_preempted(job_id)
    assert queue.kill_reason(job_id) == "preempted"
    # Nothing here ends the job: the runner owns the kill and the final sync.
    assert jobs.read_state(job_id).status == "running"
    assert [e.job_id for e in queue.list_queued()] == [waiting]


def test_preempt_can_lower_the_priority_it_comes_back_at(gpuc_home: Path) -> None:
    waiting_job(priority=40)
    job_id = running_job(priority=50)
    queue.preempt(job_id, priority=80)
    assert jobs.read_spec(job_id).priority == 80
    stopped(job_id)
    assert queue.requeue_preempted(job_id) == 2
    assert queue.find_marker(job_id) is not None
    assert queue.find_marker(job_id).name.startswith("80-")  # pyright: ignore[reportOptionalMemberAccess]


def test_a_queued_or_finished_job_cannot_be_preempted(gpuc_home: Path) -> None:
    waiting_job()
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
    waiting_job()
    job_id = running_job()
    queue.cancel(job_id)
    with pytest.raises(ValueError, match="cancelled"):
        queue.preempt(job_id)
    assert not queue.is_preempted(job_id)


def test_an_out_of_range_priority_is_refused_before_anything_is_written(gpuc_home: Path) -> None:
    """The queue marker clamps it; `spec.json` does not, and a spec saying
    priority 500 is a job whose reported priority means nothing."""
    waiting_job()
    job_id = running_job(priority=50)
    with pytest.raises(ValueError, match="0-99"):
        queue.preempt(job_id, priority=500)
    assert jobs.read_spec(job_id).priority == 50
    assert not queue.is_preempted(job_id)


# -- preempting must actually free the host for something ---------------------


def test_preempt_refuses_when_nothing_else_is_queued(gpuc_home: Path) -> None:
    """It would stop the job and start it again, losing everything it had done
    for nothing at all."""
    job_id = running_job()
    with pytest.raises(ValueError, match="nothing else is queued"):
        queue.preempt(job_id)
    assert not queue.is_preempted(job_id)
    assert queue.kill_reason(job_id) is None


def test_preempt_refuses_when_it_would_beat_every_waiting_job_to_the_gpus(
    gpuc_home: Path,
) -> None:
    """The tie nobody expects: at equal priority the marker is
    `<priority>-<job id>`, and the preempted job was submitted first, so it
    takes its own cards straight back and the waiting job waits again."""
    job_id = running_job(priority=50)
    later = queue.enqueue(make_spec(job_id="20991231-235959-ffffff", priority=50))
    with pytest.raises(ValueError, match="wins a tie") as refused:
        queue.preempt(job_id)
    assert later in str(refused.value)
    assert "--priority` above 50" in str(refused.value)
    assert not queue.is_preempted(job_id)
    # ...and the way out works, without the refusal having written the spec.
    assert jobs.read_spec(job_id).priority == 50
    assert queue.preempt(job_id, priority=51) == "preempting"


def test_a_job_queued_ahead_of_it_is_what_makes_a_preempt_worth_it(gpuc_home: Path) -> None:
    job_id = running_job(priority=50)
    queue.enqueue(make_spec(priority=10))
    assert queue.preempt(job_id) == "preempting"


def test_a_cancelled_job_in_the_queue_does_not_count_as_something_waiting(
    gpuc_home: Path,
) -> None:
    job_id = running_job(priority=50)
    doomed = queue.enqueue(make_spec(priority=10))
    paths.cancel_file(doomed).touch()
    with pytest.raises(ValueError, match="nothing else is queued"):
        queue.preempt(job_id)


@pytest.mark.parametrize(("marker", "match"), [("paused", "paused"), ("draining", "draining")])
def test_preempt_refuses_on_a_host_that_is_dispatching_nothing(
    gpuc_home: Path, marker: str, match: str
) -> None:
    waiting_job()
    job_id = running_job()
    (paths.paused_file() if marker == "paused" else paths.draining_file()).touch()
    with pytest.raises(ValueError, match=match):
        queue.preempt(job_id)
    assert not queue.is_preempted(job_id)


def test_a_failed_kill_request_takes_the_marker_back_off(
    gpuc_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A marker nothing will ever act on is a landmine: the job keeps running,
    and whenever it eventually fails on its own the dispatcher re-runs it."""
    waiting_job()
    job_id = running_job()

    def unwritable(*_: object, **__: object) -> None:
        raise OSError("read-only file system")

    monkeypatch.setattr(queue.jobs, "atomic_write_text", unwritable)
    with pytest.raises(OSError, match="read-only"):
        queue.preempt(job_id)
    assert not queue.is_preempted(job_id)


# -- coming back --------------------------------------------------------------


def stopped(job_id: str, reason: str = "preempted") -> None:
    """The state the runner leaves behind when the kill lands."""
    jobs.update_state(
        job_id, status="failed", reason=reason, exit_code=143, ended_at=jobs.utc_now()
    )


def test_requeue_after_preempt_starts_the_job_over_as_the_next_attempt(gpuc_home: Path) -> None:
    """A queued job that still carried the stopped attempt's exit code, GPUs and
    end time would be a job that reads as finished to everything but the queue."""
    waiting_job()
    job_id = running_job(priority=20)
    queue.preempt(job_id)
    stopped(job_id)
    assert queue.requeue_preempted(job_id) == 2

    state = jobs.read_state(job_id)
    assert (state.status, state.attempt) == ("queued", 2)
    assert (state.exit_code, state.ended_at, state.started_at, state.gpus) == (None, None, None, [])
    assert jobs.read_spec(job_id).attempt == 2
    assert queue.find_marker(job_id) is not None
    # The kill request was the stopped attempt's: left behind, the runner of
    # the new one would find it and stop that too.
    assert queue.kill_reason(job_id) is None
    assert not queue.is_preempted(job_id)
    assert "queued again as attempt 2" in paths.log_file(job_id).read_text()


def test_the_queue_marker_is_written_before_the_preempt_marker_is_removed(
    gpuc_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Interrupted the other way round, the job is neither queued nor asking to
    be: it is simply gone, and only the dispatcher log remembers it."""
    waiting_job()
    job_id = running_job()
    queue.preempt(job_id)
    stopped(job_id)

    real_touch = Path.touch
    failed: list[str] = []

    def fail_the_first_queue_marker(self: Path, *args: object, **kwargs: object) -> None:
        if self.parent == paths.queue_dir() and not failed:
            failed.append(self.name)
            raise OSError("no space left on device")
        real_touch(self, *args, **kwargs)  # pyright: ignore[reportArgumentType]

    # Not `monkeypatch.undo()` afterwards: this test's `monkeypatch` is the one
    # the `gpuc_home` fixture set GPUC_HOME with, so undoing it would point
    # every path below at the developer's real ~/.gpuc.
    monkeypatch.setattr(Path, "touch", fail_the_first_queue_marker)
    with pytest.raises(OSError):
        queue.requeue_preempted(job_id)
    # Still asking to come back, so the next pass finishes the job off --
    # as the *same* attempt, since that one was already counted.
    assert queue.is_preempted(job_id)
    assert queue.requeue_preempted(job_id) == 2
    assert queue.find_marker(job_id) is not None
    assert jobs.read_state(job_id).attempt == 2
    assert not queue.is_preempted(job_id)


def test_a_job_that_is_already_back_in_the_queue_is_not_queued_twice(gpuc_home: Path) -> None:
    """The retry above must be a no-op once the job is really back, or a second
    dispatcher pass bumps the attempt and leaves two markers behind."""
    waiting_job()
    job_id = running_job()
    queue.preempt(job_id)
    stopped(job_id)
    assert queue.requeue_preempted(job_id) == 2
    paths.preempt_file(job_id).touch()
    assert queue.requeue_preempted(job_id) is None
    assert jobs.read_state(job_id).attempt == 2
    assert not queue.is_preempted(job_id)


def test_a_job_cancelled_while_it_was_stopping_does_not_come_back(gpuc_home: Path) -> None:
    waiting = waiting_job()
    job_id = running_job()
    queue.preempt(job_id)
    queue.cancel(job_id)
    stopped(job_id)
    assert queue.requeue_preempted(job_id) is None
    assert [e.job_id for e in queue.list_queued()] == [waiting]
    assert not queue.is_preempted(job_id)


def test_a_job_whose_workdir_is_gone_has_nothing_to_re_run(gpuc_home: Path) -> None:
    """Its code was rsynced here once, at submit, and `gpuc preempt` never goes
    near the machine that holds it."""
    waiting = waiting_job()
    job_id = running_job()
    queue.preempt(job_id)
    stopped(job_id)
    shutil.rmtree(paths.workdir(job_id))
    assert queue.requeue_preempted(job_id) is None
    assert [e.job_id for e in queue.list_queued()] == [waiting]


@pytest.mark.parametrize(
    ("status", "reason"),
    [("succeeded", None), ("failed", "exit 1"), ("failed", "timeout"), ("failed", "low-util")],
)
def test_a_job_that_ended_on_its_own_before_the_kill_landed_is_not_re_run(
    gpuc_home: Path, status: str, reason: str | None
) -> None:
    """It asked for nothing: the work is over, or it failed for a reason of its
    own. Re-running it would be a retry nobody requested -- `gpuc requeue` is
    the command that does that, deliberately."""
    waiting_job()
    job_id = running_job()
    queue.preempt(job_id)
    jobs.update_state(job_id, status=status, reason=reason, ended_at=jobs.utc_now())
    assert queue.requeue_preempted(job_id) is None
    assert queue.find_marker(job_id) is None
    assert jobs.read_state(job_id).status == status
    assert not queue.is_preempted(job_id)


@pytest.mark.parametrize("reason", ["preempted", "preempted+sync", "runner-died", "terminated"])
def test_every_way_the_stop_itself_can_end_the_attempt_comes_back(
    gpuc_home: Path, reason: str
) -> None:
    """`preempted+sync` is a preempt whose final upload also failed, and the two
    others are the escalation ladder taking the runner down."""
    waiting_job()
    job_id = running_job()
    queue.preempt(job_id)
    stopped(job_id, reason=reason)
    assert queue.requeue_preempted(job_id) == 2


def _status_when_marker_went(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Records a job's status at the moment its queue marker is dropped.

    The order of those two writes is the whole design: interrupted between
    them, a job has to land in a shape distinguishable from a submit that never
    committed. `running` (or `cancelled`) with a stale marker is; `queued` with
    no marker is not, because that is exactly what an interrupted `gpuc submit`
    leaves behind.
    """
    seen: list[str] = []
    real = queue.remove_marker

    def spy(job_id: str) -> bool:
        seen.append(jobs.read_state(job_id).status)
        return real(job_id)

    monkeypatch.setattr(queue, "remove_marker", spy)
    return seen


def test_leave_queue_writes_the_state_before_it_drops_the_marker(
    gpuc_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job_id = queue.enqueue(make_spec())
    entry = queue.list_queued()[0]
    went = _status_when_marker_went(monkeypatch)

    queue.leave_queue(entry, status="running")

    assert went == ["running"]
    assert queue.find_marker(job_id) is None


def test_leave_queue_removes_a_marker_that_was_renamed_underneath_it(gpuc_home: Path) -> None:
    """`launch_ready` holds the entries it listed at the top of the pass, and
    `gpuc reorder` renames markers with no lock between them. Unlinking the
    path that was listed removes nothing, and the marker left behind dispatches
    a second runner into the workdir the first one is using."""
    job_id = queue.enqueue(make_spec(priority=50))
    entry = queue.list_queued()[0]
    queue.reorder(job_id, 10)

    queue.leave_queue(entry, status="running")

    assert queue.list_queued() == []
    assert jobs.read_state(job_id).status == "running"


def test_cancelling_a_queued_job_writes_the_state_before_dropping_the_marker(
    gpuc_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job_id = queue.enqueue(make_spec())
    went = _status_when_marker_went(monkeypatch)

    assert queue.cancel(job_id) == "cancelled"

    assert went == ["cancelled"]
    assert queue.find_marker(job_id) is None


def test_cancelling_a_job_that_is_already_being_dispatched_does_not_undo_it(
    gpuc_home: Path,
) -> None:
    """The dispatcher published `running` first; writing `cancelled` over it
    would leave a live runner behind a state that says the job is over. From
    here the runner owns the kill, which is what `cancelling` means."""
    job_id = queue.enqueue(make_spec())
    jobs.update_state(job_id, status="running")

    assert queue.cancel(job_id) == "cancelling"
    assert jobs.read_state(job_id).status == "running"
    assert queue.is_cancelled(job_id)
