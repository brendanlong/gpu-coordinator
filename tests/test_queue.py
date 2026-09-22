from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from gpuc.host import jobs, paths, queue
from tests.conftest import make_spec


def test_enqueue_writes_the_spec_and_a_queued_state(gpuc_home: Path) -> None:
    spec = make_spec(name="a", priority=7)
    job_id = queue.enqueue(spec)
    assert jobs.read_spec(job_id).name == "a"
    state = jobs.read_state(job_id)
    assert (state.status, state.priority, state.intent) == ("queued", 7, None)
    assert paths.workdir(job_id).is_dir()
    assert paths.outputs_dir(job_id).is_dir()
    assert paths.log_file(job_id).exists()
    assert queue.list_queued() == [queue.QueueEntry(7, job_id)]


def test_enqueue_accepts_a_job_by_renaming_its_dir_out_of_incoming(gpuc_home: Path) -> None:
    """The rename is the acceptance, so a dir under `jobs/` is a job this host
    was asked to run and a submit that died leaves nothing there at all."""
    job_id = queue.enqueue(make_spec())
    assert paths.job_dir(job_id).is_dir()
    assert not paths.incoming_job_dir(job_id).exists()
    assert list(paths.incoming_dir().iterdir()) == []


def test_enqueue_carries_in_what_submit_staged_under_incoming(gpuc_home: Path) -> None:
    """`gpuc submit` rsyncs the working tree to `incoming/<id>/workdir` before
    the spec ever reaches the host, and the rename is what makes it a job."""
    spec = make_spec()
    staged = paths.incoming_job_dir(spec.job_id) / "workdir"
    staged.mkdir(parents=True)
    (staged / "train.py").write_text("print('hi')\n")

    job_id = queue.enqueue(spec)

    assert (paths.workdir(job_id) / "train.py").read_text() == "print('hi')\n"


def test_enqueue_refuses_a_job_id_that_is_already_here(gpuc_home: Path) -> None:
    """A second job dir cannot be made for one id, so the one already there --
    which may be running -- cannot be overwritten by a replayed submit."""
    spec = make_spec()
    queue.enqueue(spec)
    jobs.update_state(spec.job_id, status="running")

    with pytest.raises(FileExistsError, match="already exists"):
        queue.enqueue(spec)

    assert jobs.read_state(spec.job_id).status == "running"
    assert queue.list_queued() == []


def test_dispatch_order_is_lexical_by_priority_then_id(gpuc_home: Path) -> None:
    low = queue.enqueue(make_spec(priority=90))
    high = queue.enqueue(make_spec(priority=10))
    middle = queue.enqueue(make_spec(priority=50))
    assert [e.job_id for e in queue.list_queued()] == [high, middle, low]


def test_the_queue_is_every_job_whose_state_says_queued(gpuc_home: Path) -> None:
    """There is no queue file to disagree with the states: a job leaves the
    queue by becoming something other than `queued`, and nothing else."""
    queued = queue.enqueue(make_spec())
    running = queue.enqueue(make_spec())
    jobs.update_state(running, status="running")
    assert [e.job_id for e in queue.list_queued()] == [queued]
    assert queue.is_queued(queued) and not queue.is_queued(running)


def test_a_job_whose_state_cannot_be_read_is_left_out_of_the_queue(gpuc_home: Path) -> None:
    """One job dir somebody deleted half of must not stop the rest dispatching."""
    good = queue.enqueue(make_spec())
    broken = queue.enqueue(make_spec())
    paths.state_file(broken).unlink()
    assert [e.job_id for e in queue.list_queued()] == [good]


def test_enqueue_does_not_need_the_dispatcher_lock(gpuc_home: Path) -> None:
    from gpuc.host.dispatcher import DispatcherLock

    lock = DispatcherLock()
    assert lock.acquire()
    try:
        job_id = queue.enqueue(make_spec())
        assert [e.job_id for e in queue.list_queued()] == [job_id]
    finally:
        lock.release()


# -- claiming a job out of the queue ------------------------------------------


def test_claim_takes_a_job_out_of_the_queue_and_records_what_became_of_it(
    gpuc_home: Path,
) -> None:
    job_id = queue.enqueue(make_spec())
    assert queue.claim(job_id, status="running", gpus=["GPU-x"], started_at=jobs.utc_now())
    state = jobs.read_state(job_id)
    assert (state.status, state.gpus) == ("running", ["GPU-x"])
    assert queue.list_queued() == []


def test_claim_loses_to_a_cancel_that_landed_first(gpuc_home: Path) -> None:
    """The dispatcher lists the queue and then claims each job it acts on, and
    the two are not one operation: a cancel in between must win, or the host
    starts a runner for a job it has already told the user is cancelled."""
    job_id = queue.enqueue(make_spec())
    listed = queue.list_queued()[0]
    assert queue.cancel(job_id) == "cancelled"

    assert not queue.claim(listed.job_id, status="running", gpus=["GPU-x"])

    state = jobs.read_state(job_id)
    assert (state.status, state.gpus) == ("cancelled", [])


def test_only_one_claim_of_a_job_can_win(gpuc_home: Path) -> None:
    job_id = queue.enqueue(make_spec())
    assert queue.claim(job_id, status="running")
    assert not queue.claim(job_id, status="failed", reason="bad-spec")
    assert jobs.read_state(job_id).status == "running"


# -- reordering ---------------------------------------------------------------


def test_reorder_moves_a_queued_job_and_records_the_new_priority(gpuc_home: Path) -> None:
    """The state holds the live priority; the spec keeps what was submitted."""
    first = queue.enqueue(make_spec(priority=50))
    second = queue.enqueue(make_spec(priority=50))
    assert queue.reorder(second, 1)
    assert [e.job_id for e in queue.list_queued()] == [second, first]
    assert jobs.read_state(second).priority == 1
    assert jobs.read_spec(second).priority == 50
    assert jobs.read_state(first).priority == 50


def test_reorder_of_a_job_that_is_not_queued_is_refused(gpuc_home: Path) -> None:
    """A running job's place in the queue is not a thing that exists; moving it
    would only change the priority it reports."""
    job_id = queue.enqueue(make_spec(priority=50))
    jobs.update_state(job_id, status="running")
    assert not queue.reorder(job_id, 1)
    assert jobs.read_state(job_id).priority == 50


def test_reorder_of_an_unknown_job_is_refused_rather_than_an_error(gpuc_home: Path) -> None:
    """`gpuc reorder` on a typo is an error document and exit 1, so the queue
    has to answer "no such job" as a False and leave nothing behind."""
    assert not queue.reorder("no-such-job", 1)
    assert jobs.list_job_ids() == []


# -- cancelling ---------------------------------------------------------------


def test_cancel_of_a_queued_job_is_immediate(gpuc_home: Path) -> None:
    """It works with no dispatcher running: nothing else has to act on it."""
    job_id = queue.enqueue(make_spec())
    assert queue.cancel(job_id) == "cancelled"
    assert queue.list_queued() == []
    state = jobs.read_state(job_id)
    assert (state.status, state.reason) == ("cancelled", "cancelled")
    assert state.ended_at is not None
    # No intent left behind: an intent is a request standing against a job that
    # is still running, and this one is over.
    assert state.intent is None
    assert not queue.is_cancelled(job_id)


def test_cancel_of_a_running_job_records_the_intent(gpuc_home: Path) -> None:
    job_id = queue.enqueue(make_spec())
    jobs.update_state(job_id, status="running")
    assert queue.cancel(job_id) == "cancelling"
    assert queue.is_cancelled(job_id)
    assert queue.stop_requested(job_id) == "cancelled"
    # Nothing here ends the job: the runner owns the kill and the final sync.
    assert jobs.read_state(job_id).status == "running"


def test_cancel_of_a_finished_job_is_a_noop(gpuc_home: Path) -> None:
    job_id = queue.enqueue(make_spec())
    jobs.update_state(job_id, status="succeeded", ended_at=jobs.utc_now())
    assert queue.cancel(job_id) == "succeeded"
    assert jobs.read_state(job_id).intent is None


def test_cancel_of_an_unknown_job_raises(gpuc_home: Path) -> None:
    with pytest.raises(FileNotFoundError):
        queue.cancel("nope")


def test_nothing_is_asked_of_a_job_nobody_has_touched(gpuc_home: Path) -> None:
    job_id = queue.enqueue(make_spec())
    jobs.update_state(job_id, status="running")
    assert queue.stop_requested(job_id) is None
    assert queue.stop_requested("no-such-job") is None


def running_job(**overrides: object) -> str:
    job_id = queue.enqueue(make_spec(**overrides))
    jobs.update_state(job_id, status="running", gpus=["GPU-x"], started_at=jobs.utc_now())
    return job_id


def waiting_job(priority: int = 1) -> str:
    """Something queued that would take the preempted job's place.

    Every preempt needs one: with nothing waiting, the command refuses rather
    than stop a job and start it again.
    """
    return queue.enqueue(make_spec(priority=priority))


def test_preempt_asks_the_runner_to_stop_and_leaves_the_job_running(gpuc_home: Path) -> None:
    waiting = waiting_job()
    job_id = running_job()
    assert queue.preempt(job_id) == "preempting"
    assert queue.is_preempted(job_id)
    assert queue.stop_requested(job_id) == "preempted"
    # Nothing here ends the job: the runner owns the kill and the final sync.
    assert jobs.read_state(job_id).status == "running"
    assert [e.job_id for e in queue.list_queued()] == [waiting]


def test_preempt_can_lower_the_priority_it_comes_back_at(gpuc_home: Path) -> None:
    waiting_job(priority=40)
    job_id = running_job(priority=50)
    queue.preempt(job_id, priority=80)
    assert jobs.read_state(job_id).priority == 80
    stopped(job_id)
    assert queue.requeue_preempted(job_id) == 2
    assert queue.QueueEntry(80, job_id) in queue.list_queued()


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
    assert queue.stop_requested(job_id) == "cancelled"


def test_an_out_of_range_priority_is_refused_before_anything_is_written(gpuc_home: Path) -> None:
    """A job whose reported priority is 500 means nothing to anybody reading
    the queue it is ordered in."""
    waiting_job()
    job_id = running_job(priority=50)
    with pytest.raises(ValueError, match="0-99"):
        queue.preempt(job_id, priority=500)
    assert jobs.read_state(job_id).priority == 50
    assert not queue.is_preempted(job_id)


# -- preempting must actually free the host for something ---------------------


def test_preempt_refuses_when_nothing_else_is_queued(gpuc_home: Path) -> None:
    """It would stop the job and start it again, losing everything it had done
    for nothing at all."""
    job_id = running_job()
    with pytest.raises(ValueError, match="nothing else is queued"):
        queue.preempt(job_id)
    assert not queue.is_preempted(job_id)
    assert queue.stop_requested(job_id) is None


def test_preempt_refuses_when_it_would_beat_every_waiting_job_to_the_gpus(
    gpuc_home: Path,
) -> None:
    """The tie nobody expects: dispatch order is `(priority, job id)`, and the
    preempted job was submitted first, so it takes its own cards straight back
    and the waiting job waits again."""
    job_id = running_job(priority=50)
    later = queue.enqueue(make_spec(job_id="20991231-235959-ffffff", priority=50))
    with pytest.raises(ValueError, match="wins a tie") as refused:
        queue.preempt(job_id)
    assert later in str(refused.value)
    assert "--priority` above 50" in str(refused.value)
    assert not queue.is_preempted(job_id)
    # ...and the way out works, without the refusal having moved the job.
    assert jobs.read_state(job_id).priority == 50
    assert queue.preempt(job_id, priority=51) == "preempting"


def test_a_job_queued_ahead_of_it_is_what_makes_a_preempt_worth_it(gpuc_home: Path) -> None:
    job_id = running_job(priority=50)
    queue.enqueue(make_spec(priority=10))
    assert queue.preempt(job_id) == "preempting"


def test_a_cancelled_job_does_not_count_as_something_waiting(gpuc_home: Path) -> None:
    job_id = running_job(priority=50)
    doomed = queue.enqueue(make_spec(priority=10))
    queue.cancel(doomed)
    with pytest.raises(ValueError, match="nothing else is queued"):
        queue.preempt(job_id)


def test_preempt_refuses_on_a_draining_host(gpuc_home: Path) -> None:
    waiting_job()
    job_id = running_job()
    paths.draining_file().touch()
    with pytest.raises(ValueError, match="draining"):
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
    job_id = running_job(priority=20, estimated_runtime_min=45.0)
    queue.preempt(job_id)
    stopped(job_id)
    assert queue.requeue_preempted(job_id) == 2

    state = jobs.read_state(job_id)
    assert (state.status, state.attempt) == ("queued", 2)
    assert (state.exit_code, state.ended_at, state.started_at, state.gpus) == (None, None, None, [])
    # What a queued job is still ordered and described by survives the rewrite.
    assert (state.priority, state.estimated_runtime_min) == (20, 45.0)
    assert queue.QueueEntry(20, job_id) in queue.list_queued()
    # The request was the stopped attempt's: left behind, the runner of the new
    # one would find it and stop that too.
    assert state.intent is None
    assert queue.stop_requested(job_id) is None
    assert "queued again as attempt 2" in paths.log_file(job_id).read_text()


def test_a_job_that_is_already_back_in_the_queue_is_not_queued_twice(gpuc_home: Path) -> None:
    """A second dispatcher pass on a stale intent would bump the attempt again
    and rewrite a job that is already waiting its turn."""
    waiting_job()
    job_id = running_job()
    queue.preempt(job_id)
    stopped(job_id)
    assert queue.requeue_preempted(job_id) == 2

    jobs.update_state(job_id, intent=jobs.PREEMPT)
    assert queue.requeue_preempted(job_id) is None
    assert jobs.read_state(job_id).attempt == 2
    assert not queue.is_preempted(job_id)


def test_a_job_cancelled_while_it_was_stopping_does_not_come_back(gpuc_home: Path) -> None:
    """One intent, and the later one wins: somebody who cancels a job that is
    already stopping is asking for it to be over, not to start again."""
    waiting = waiting_job()
    job_id = running_job()
    queue.preempt(job_id)
    assert queue.cancel(job_id) == "cancelling"
    assert queue.stop_requested(job_id) == "cancelled"
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
    assert not queue.is_preempted(job_id)


def test_a_job_nobody_preempted_is_left_alone(gpuc_home: Path) -> None:
    job_id = running_job()
    stopped(job_id)
    assert queue.requeue_preempted(job_id) is None
    assert jobs.read_state(job_id).status == "failed"


@pytest.mark.parametrize(
    ("status", "reason"),
    [("succeeded", None), ("failed", "exit 1"), ("failed", "timeout"), ("failed", "sync")],
)
def test_a_job_that_ended_on_its_own_before_the_kill_landed_is_not_re_run(
    gpuc_home: Path, status: str, reason: str | None
) -> None:
    """It asked for nothing: the work is over, or it failed for a reason of its
    own. Re-running it would be a retry nobody requested -- `gpuc requeue` is
    the command that does that, deliberately. The intent goes with the
    decision, so nothing tries again."""
    waiting_job()
    job_id = running_job()
    queue.preempt(job_id)
    jobs.update_state(job_id, status=status, reason=reason, ended_at=jobs.utc_now())
    assert queue.requeue_preempted(job_id) is None
    assert not queue.is_queued(job_id)
    assert jobs.read_state(job_id).status == status
    assert jobs.read_state(job_id).intent is None


@pytest.mark.parametrize(
    ("reason", "problems"),
    [("preempted", []), ("preempted", ["sync"]), ("runner-died", []), ("terminated", [])],
)
def test_every_way_the_stop_itself_can_end_the_attempt_comes_back(
    gpuc_home: Path, reason: str, problems: list[str]
) -> None:
    """A preempt whose final upload also failed is still a preempt: the upload
    failure is a problem beside the reason. The two other reasons are the
    escalation ladder taking the runner down."""
    waiting_job()
    job_id = running_job()
    queue.preempt(job_id)
    stopped(job_id, reason=reason)
    jobs.update_state(job_id, problems=problems)
    assert queue.requeue_preempted(job_id) == 2
