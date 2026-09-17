from __future__ import annotations

import os
import shutil
import signal
import subprocess
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import pytest

from gpuc.host import cleanup, jobs, paths, queue, sync, terminate
from gpuc.host import dispatcher as host_dispatcher
from gpuc.host import runner as procinfo
from gpuc.host.dispatcher import Dispatcher, DispatcherDeps
from gpuc.host.jobs import HostConfig
from tests.conftest import FAKE_GPUS, fake_smi, make_spec


class FakeRunnerProcess:
    """Stands in for a spawned runner: alive until the test finishes it."""

    _next_pid = 500000

    def __init__(self, job_id: str) -> None:
        FakeRunnerProcess._next_pid += 1
        self.pid = FakeRunnerProcess._next_pid
        self.job_id = job_id
        self.returncode: int | None = None

    def poll(self) -> int | None:
        return self.returncode

    def finish(self, status: str = "succeeded", reason: str | None = None) -> None:
        self.returncode = 0
        jobs.update_state(
            self.job_id,
            status=status,
            reason=reason,
            exit_code=0 if status == "succeeded" else 1,
            ended_at=jobs.utc_now(),
        )


class FakeClock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


def make_dispatcher(
    clock: FakeClock | None = None,
    terminate_call: terminate.TerminateCall | None = None,
    utcnow: Callable[[], datetime] | None = None,
) -> tuple[Dispatcher, dict[str, FakeRunnerProcess]]:
    spawned: dict[str, FakeRunnerProcess] = {}

    def spawn(job_id: str) -> subprocess.Popen[bytes]:
        proc = FakeRunnerProcess(job_id)
        spawned[job_id] = proc
        return cast("subprocess.Popen[bytes]", proc)

    deps = DispatcherDeps(
        spawn_runner=spawn,
        monotonic=clock or FakeClock(),
        terminate_call=terminate_call or (lambda pod, key: "{}"),
        command_runner=lambda argv, timeout=None, env=None: sync.CommandResult(argv, 0, ""),
        utcnow=utcnow or (lambda: datetime.now(UTC)),
        kill_grace_s=1.0,
    )
    return Dispatcher(deps=deps), spawned


def test_queued_job_is_launched_with_assigned_uuids(gpuc_home: Path) -> None:
    job_id = queue.enqueue(make_spec(gpus=1))
    dispatcher, spawned = make_dispatcher()
    dispatcher.run_once()
    state = jobs.read_state(job_id)
    assert state.status == "running"
    assert state.gpus == [FAKE_GPUS[0]]
    assert state.runner_pid == spawned[job_id].pid
    assert queue.list_queued() == []


def test_two_single_gpu_jobs_run_concurrently(gpuc_home: Path) -> None:
    first = queue.enqueue(make_spec(gpus=1, priority=10))
    second = queue.enqueue(make_spec(gpus=1, priority=20))
    dispatcher, _ = make_dispatcher()
    dispatcher.run_once()
    assert jobs.read_state(first).gpus == [FAKE_GPUS[0]]
    assert jobs.read_state(second).gpus == [FAKE_GPUS[1]]
    assert len(dispatcher.running) == 2


def test_a_job_waits_when_not_enough_gpus_are_free(gpuc_home: Path) -> None:
    big = queue.enqueue(make_spec(gpus=2, priority=10))
    dispatcher, spawned = make_dispatcher()
    dispatcher.run_once()
    assert jobs.read_state(big).gpus == FAKE_GPUS

    waiting = queue.enqueue(make_spec(gpus=1, priority=20))
    dispatcher.run_once()
    assert jobs.read_state(waiting).status == "queued"

    spawned[big].finish()
    dispatcher.run_once()
    assert jobs.read_state(waiting).status == "running"


def test_a_wide_job_is_not_starved_by_a_stream_of_narrow_ones(gpuc_home: Path) -> None:
    """The dispatch rule the preempt livelock was a special case of. Priority
    is advisory if a job that does not fit can be walked past for ever: each
    one-card job fits the single card the two-card job is waiting for, so the
    most important job in the queue never ran at all."""
    dispatcher, spawned = make_dispatcher()
    narrow = [queue.enqueue(make_spec(gpus=1, priority=50)) for _ in range(2)]
    dispatcher.run_once()
    wide = queue.enqueue(make_spec(gpus=2, priority=10))

    done = narrow.pop(0)
    spawned[done].finish()
    later = queue.enqueue(make_spec(gpus=1, priority=50))  # and another arrives
    dispatcher.run_once()
    # The freed card is held for the job that is waiting for it, not handed to
    # the job behind it.
    assert jobs.read_state(later).status == "queued"

    spawned[narrow[0]].finish()
    dispatcher.run_once()
    assert jobs.read_state(wide).status == "running"
    assert jobs.read_state(wide).gpus == FAKE_GPUS


def test_a_job_asking_for_no_gpus_fails_at_dispatch(gpuc_home: Path) -> None:
    """`gpuc submit` refuses `gpus: 0`, so a spec that has it was queued by an
    older build or written by hand. It is failed, with the reason, rather than
    run on no card -- and it holds nothing up on its way out."""
    none = queue.enqueue(make_spec(gpus=0, priority=10))
    runnable = queue.enqueue(make_spec(gpus=1, priority=20))
    dispatcher, spawned = make_dispatcher()
    dispatcher.run_once()
    state = jobs.read_state(none)
    assert (state.status, state.reason) == ("failed", "needs at least 1 GPU, asked for 0")
    assert none not in spawned
    assert jobs.read_state(runnable).status == "running"


def test_a_job_larger_than_the_host_fails_instead_of_blocking(gpuc_home: Path) -> None:
    impossible = queue.enqueue(make_spec(gpus=8, priority=10))
    runnable = queue.enqueue(make_spec(gpus=1, priority=20))
    dispatcher, _ = make_dispatcher()
    dispatcher.run_once()
    state = jobs.read_state(impossible)
    assert state.status == "failed"
    assert "host owns 2" in (state.reason or "")
    assert jobs.read_state(runnable).status == "running"


def test_a_job_cancelled_while_queued_is_never_launched(gpuc_home: Path) -> None:
    job_id = queue.enqueue(make_spec(gpus=1))
    paths.cancel_file(job_id).touch()
    dispatcher, spawned = make_dispatcher()
    dispatcher.run_once()
    assert spawned == {}
    assert jobs.read_state(job_id).status == "cancelled"


def test_cancel_signals_the_job_process_group_then_the_runner(
    gpuc_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = FakeClock()
    dispatcher, spawned = make_dispatcher(clock=clock)
    job_id = queue.enqueue(make_spec(gpus=1))
    dispatcher.run_once()
    jobs.update_state(job_id, pgid=123456)

    signals: list[tuple[int | None, int]] = []
    monkeypatch.setattr(dispatcher, "_signal_group", lambda pgid, sig: signals.append((pgid, sig)))
    monkeypatch.setattr(dispatcher, "_signal_pid", lambda pid, sig: signals.append((pid, sig)))
    monkeypatch.setattr(host_dispatcher, "process_group_alive", lambda _pgid: True)
    queue.cancel(job_id)
    dispatcher.handle_cancels()
    assert signals == [(123456, 15)]

    clock.advance(1.5)
    dispatcher.handle_cancels()
    assert signals[-1] == (123456, 9)

    clock.advance(1.0)
    dispatcher.handle_cancels()
    assert signals[-1] == (spawned[job_id].pid, signal.SIGTERM)

    clock.advance(1.0)
    dispatcher.handle_cancels()
    assert signals[-1] == (spawned[job_id].pid, signal.SIGKILL)


def test_a_runner_that_dies_without_final_state_fails_the_job(gpuc_home: Path) -> None:
    job_id = queue.enqueue(make_spec(gpus=1))
    dispatcher, spawned = make_dispatcher()
    dispatcher.run_once()
    spawned[job_id].returncode = -9
    dispatcher.run_once()
    state = jobs.read_state(job_id)
    assert (state.status, state.reason) == ("failed", "runner-died")
    assert dispatcher.running == {}


def test_orphans_from_a_dead_dispatcher_are_reconciled(gpuc_home: Path) -> None:
    job_id = queue.enqueue(make_spec(gpus=1))
    queue.remove_marker(job_id)
    jobs.update_state(job_id, status="running", gpus=[FAKE_GPUS[0]], runner_pid=2**30)
    dispatcher, _ = make_dispatcher()
    dispatcher.adopt_orphans()
    assert jobs.read_state(job_id).reason == "runner-died"


def test_a_live_orphan_is_adopted_not_failed(gpuc_home: Path) -> None:
    import os

    job_id = queue.enqueue(make_spec(gpus=1))
    queue.remove_marker(job_id)
    jobs.update_state(job_id, status="running", gpus=[FAKE_GPUS[0]], runner_pid=os.getpid())
    dispatcher, _ = make_dispatcher()
    dispatcher.adopt_orphans()
    assert jobs.read_state(job_id).status == "running"
    assert job_id in dispatcher.running
    assert dispatcher.free_gpus() == [FAKE_GPUS[1]]


def test_a_paused_marker_from_an_older_build_does_not_stop_dispatching(gpuc_home: Path) -> None:
    """Builds before the host pause was removed left `paused` under the home;
    a stale one is a file this build does not know, not a reason to sit idle."""
    (paths.home() / "paused").write_text("two consecutive jobs failed with reason low-util\n")
    job_id = queue.enqueue(make_spec(gpus=1))
    dispatcher, _ = make_dispatcher()
    dispatcher.run_once()
    assert jobs.read_state(job_id).status == "running"


def test_a_non_provider_host_never_self_terminates(gpuc_home: Path) -> None:
    clock = FakeClock()
    terminated: list[str] = []
    dispatcher, _ = make_dispatcher(
        clock=clock, terminate_call=lambda pod, key: terminated.append(pod) or ""
    )
    dispatcher.run_once()
    clock.advance(10 * 3600)
    dispatcher.run_once()
    assert terminated == []
    assert not paths.draining_file().exists()
    assert dispatcher.idle_and_not_ephemeral()


def configure_pod(idle_minutes: float = 15.0, age_h: float = 0.0) -> None:
    created = datetime.now(UTC) - timedelta(hours=age_h)
    jobs.write_config(
        HostConfig(
            host="pod",
            gpus=list(FAKE_GPUS),
            provider={"kind": "runpod", "pod_id": "pod-1"},
            idle_minutes=idle_minutes,
            s3_prefix="s3://b/gpuc/pod",
            created_at=created.isoformat(),
        )
    )


def test_idle_terminate_drains_syncs_and_calls_the_provider(
    gpuc_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sync, "aws_binary", lambda env=None: "/fake/aws")
    monkeypatch.setenv("RUNPOD_API_KEY", "key")
    monkeypatch.setenv("RUNPOD_POD_ID", "pod-1")
    configure_pod(idle_minutes=15.0)
    clock = FakeClock()
    terminated: list[str] = []
    dispatcher, _ = make_dispatcher(
        clock=clock, terminate_call=lambda pod, key: terminated.append(pod) or ""
    )
    dispatcher.run_once()
    assert terminated == []
    clock.advance(14 * 60)
    dispatcher.run_once()
    assert terminated == []
    clock.advance(2 * 60)
    dispatcher.run_once()
    assert terminated == ["pod-1"]
    assert paths.draining_file().exists()
    assert dispatcher.should_exit


def test_a_running_job_resets_the_idle_timer(
    gpuc_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RUNPOD_API_KEY", "key")
    configure_pod(idle_minutes=1.0)
    clock = FakeClock()
    terminated: list[str] = []
    dispatcher, spawned = make_dispatcher(
        clock=clock, terminate_call=lambda pod, key: terminated.append(pod) or ""
    )
    job_id = queue.enqueue(make_spec(gpus=1))
    dispatcher.run_once()
    clock.advance(600)
    dispatcher.run_once()
    assert terminated == []
    spawned[job_id].finish()
    dispatcher.run_once()
    clock.advance(30)
    dispatcher.run_once()
    assert terminated == []
    clock.advance(40)
    dispatcher.run_once()
    assert terminated == ["pod-1"]


def test_failed_terminate_removes_draining_and_keeps_dispatching(
    gpuc_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RUNPOD_API_KEY", "key")
    configure_pod(idle_minutes=0.0)
    clock = FakeClock()
    attempts: list[str] = []

    def failing(pod: str, key: str) -> str:
        attempts.append(pod)
        raise terminate.TerminateError("HTTP 500 from runpod")

    dispatcher, _ = make_dispatcher(clock=clock, terminate_call=failing)
    dispatcher.run_once()
    assert attempts == ["pod-1"]
    assert not paths.draining_file().exists()
    assert not dispatcher.should_exit

    job_id = queue.enqueue(make_spec(gpus=1))
    dispatcher.run_once()
    assert jobs.read_state(job_id).status == "running"
    assert len(attempts) == 1

    log = paths.dispatcher_log().read_text()
    assert "SELF-TERMINATE FAILED" in log
    assert "HTTP 500 from runpod" in log


def test_terminate_is_retried_after_ten_minutes(
    gpuc_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RUNPOD_API_KEY", "key")
    configure_pod(idle_minutes=0.0)
    clock = FakeClock()
    attempts: list[str] = []
    outcomes: list[Any] = [terminate.TerminateError("nope"), None]

    def flaky(pod: str, key: str) -> str:
        attempts.append(pod)
        outcome = outcomes.pop(0) if outcomes else None
        if isinstance(outcome, Exception):
            raise outcome
        return "{}"

    dispatcher, _ = make_dispatcher(clock=clock, terminate_call=flaky)
    dispatcher.run_once()
    clock.advance(60)
    dispatcher.run_once()
    assert len(attempts) == 1
    clock.advance(600)
    dispatcher.run_once()
    assert len(attempts) == 2
    assert dispatcher.should_exit


def test_a_failed_final_drain_sync_still_terminates(
    gpuc_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A mirror we cannot write is not a reason to keep a paid pod alive.

    It used to be: the final sync shared a try with `self_terminate`, so a
    SyncError read as "terminate failed" and the pod stayed up retrying every
    ten minutes -- with a fresh heartbeat -- for as long as the credentials
    stayed broken.
    """
    monkeypatch.setenv("RUNPOD_API_KEY", "key")
    monkeypatch.setattr(sync, "aws_binary", lambda env=None: None)
    configure_pod(idle_minutes=0.0)
    terminated: list[str] = []
    dispatcher, _ = make_dispatcher(terminate_call=lambda pod, key: terminated.append(pod) or "")
    queue.enqueue(make_spec(gpus=1))
    dispatcher.run_once()
    dispatcher.running.clear()
    dispatcher.run_once()
    assert terminated == ["pod-1"]
    assert paths.draining_file().exists()
    assert dispatcher.should_exit
    assert "final state mirror failed" in paths.dispatcher_log().read_text()


def test_dispatcher_does_not_launch_while_draining(gpuc_home: Path) -> None:
    paths.draining_file().write_text("idle\n")
    job_id = queue.enqueue(make_spec(gpus=1))
    dispatcher, spawned = make_dispatcher()
    dispatcher.launch_ready()
    assert spawned == {}
    assert jobs.read_state(job_id).status == "queued"


def test_a_job_with_an_unreadable_spec_is_dropped(gpuc_home: Path) -> None:
    job_id = queue.enqueue(make_spec(gpus=1))
    paths.spec_file(job_id).write_text("{not json")
    dispatcher, spawned = make_dispatcher()
    dispatcher.run_once()
    assert spawned == {}
    assert jobs.read_state(job_id).reason == "bad-spec"
    assert queue.list_queued() == []


def orphan_process_group() -> subprocess.Popen[bytes]:
    """A job-like process in its own group, as the runner would have spawned."""
    return subprocess.Popen(["sleep", "300"], start_new_session=True)


def test_a_dead_runner_never_leaves_a_job_holding_a_card(gpuc_home: Path) -> None:
    job_id = queue.enqueue(make_spec(gpus=1))
    dispatcher, spawned = make_dispatcher()
    dispatcher.run_once()
    orphan = orphan_process_group()
    try:
        jobs.update_state(job_id, pgid=orphan.pid)
        spawned[job_id].returncode = -9
        dispatcher.run_once()
        assert orphan.wait(timeout=30) == -9
    finally:
        if orphan.poll() is None:
            orphan.kill()
    assert jobs.read_state(job_id).reason == "runner-died"
    assert dispatcher.free_gpus() == FAKE_GPUS
    assert f"orphaned process group {orphan.pid}" in paths.dispatcher_log().read_text()


def test_an_unreadable_state_for_a_running_job_is_runner_died_not_a_crash(
    gpuc_home: Path,
) -> None:
    job_id = queue.enqueue(make_spec(gpus=1))
    dispatcher, spawned = make_dispatcher()
    dispatcher.run_once()
    spawned[job_id].returncode = 0
    paths.state_file(job_id).write_text("{ truncated")
    dispatcher.run_once()
    state = jobs.read_state(job_id)
    assert (state.status, state.reason) == ("failed", "runner-died")
    assert dispatcher.running == {}


def test_run_once_failures_are_logged_and_eventually_give_up(
    gpuc_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gpuc.host.dispatcher import MAX_CONSECUTIVE_FAILURES, DispatcherLock

    dispatcher, _ = make_dispatcher()
    dispatcher.deps.sleep = lambda _seconds: None
    attempts: list[int] = []

    def explode() -> None:
        attempts.append(1)
        raise RuntimeError("the disk went away")

    monkeypatch.setattr(dispatcher, "run_once", explode)
    lock = DispatcherLock()
    assert lock.acquire()
    assert dispatcher.run(lock) == 1
    assert len(attempts) == MAX_CONSECUTIVE_FAILURES
    log = paths.dispatcher_log().read_text()
    assert "the disk went away" in log
    assert "GIVING UP" in log


def test_an_occasional_failure_does_not_stop_the_loop(gpuc_home: Path) -> None:
    dispatcher, _ = make_dispatcher()
    calls: list[int] = []

    def flaky() -> None:
        calls.append(1)
        if len(calls) < 3:
            raise RuntimeError("transient")

    assert not dispatcher._guard(flaky)
    assert dispatcher.consecutive_failures == 1
    assert not dispatcher._guard(flaky)
    assert dispatcher._guard(flaky)
    assert dispatcher.consecutive_failures == 0


def test_an_orphan_from_a_previous_boot_is_not_adopted(gpuc_home: Path) -> None:
    job_id = queue.enqueue(make_spec(gpus=1))
    queue.remove_marker(job_id)
    jobs.update_state(
        job_id,
        status="running",
        gpus=[FAKE_GPUS[0]],
        runner_pid=os.getpid(),
        runner_boot_id="0000-a-previous-boot",
    )
    dispatcher, _ = make_dispatcher()
    dispatcher.adopt_orphans()
    assert jobs.read_state(job_id).reason == "runner-died"


def test_an_orphan_whose_pid_was_reused_is_not_adopted(gpuc_home: Path) -> None:
    job_id = queue.enqueue(make_spec(gpus=1))
    queue.remove_marker(job_id)
    jobs.update_state(
        job_id,
        status="running",
        gpus=[FAKE_GPUS[0]],
        runner_pid=os.getpid(),
        runner_boot_id=procinfo.boot_id(),
        runner_starttime="1",
    )
    dispatcher, _ = make_dispatcher()
    dispatcher.adopt_orphans()
    assert jobs.read_state(job_id).reason == "runner-died"


def test_a_marker_left_behind_does_not_launch_a_second_runner(gpuc_home: Path) -> None:
    """`leave_queue` writes the state and then drops the marker, so a
    dispatcher killed between the two leaves a marker for a job that is already
    running. Dispatching on the marker alone starts a second runner in the
    workdir the first one is using."""
    job_id = queue.enqueue(make_spec(gpus=1))
    jobs.update_state(job_id, status="running", gpus=[FAKE_GPUS[0]])  # the crash

    dispatcher, spawned = make_dispatcher()
    dispatcher.run_once()

    assert spawned == {}
    assert queue.list_queued() == []


def test_a_marker_whose_job_is_gone_is_dropped_rather_than_dispatched(gpuc_home: Path) -> None:
    """A purge interrupted before `remove_job_dir` reached the marker."""
    job_id = queue.enqueue(make_spec(gpus=1))
    shutil.rmtree(paths.job_dir(job_id))

    dispatcher, spawned = make_dispatcher()
    dispatcher.run_once()

    assert spawned == {}
    assert queue.list_queued() == []


def test_a_submit_that_never_committed_is_not_run_and_does_not_stay_queued(
    gpuc_home: Path,
) -> None:
    """`gpuc submit` writes the spec and state and *then* the queue marker, so
    an ssh that dies between them leaves a job this host was never asked to
    run. It must not be dispatched -- and it must not sit `queued` for ever
    either, which is all anybody could see of it."""
    job_id = queue.enqueue(make_spec(gpus=1))
    queue.remove_marker(job_id)  # the submit dies here
    old = datetime.now(UTC).timestamp() - cleanup.INCOMING_STALE_S - 1
    os.utime(paths.state_file(job_id), (old, old))

    dispatcher, spawned = make_dispatcher()
    dispatcher.run_once()

    assert spawned == {}
    state = jobs.read_state(job_id)
    assert (state.status, state.reason) == ("failed", "incomplete-submit")


def test_a_submit_still_in_flight_is_not_swept(gpuc_home: Path) -> None:
    """The gap between the two writes is two syscalls wide. A sweep that could
    not tell it from an abandoned one would fail perfectly good submits."""
    job_id = queue.enqueue(make_spec(gpus=1))
    queue.remove_marker(job_id)

    dispatcher, _ = make_dispatcher()
    dispatcher.run_once()

    assert jobs.read_state(job_id).status == "queued"


def test_a_preempted_job_being_put_back_is_not_swept(gpuc_home: Path) -> None:
    """`requeue_preempted` writes the queued state, then the marker, then drops
    the preempt marker -- so mid-move it looks exactly like an abandoned
    submit. The preempt marker is what says it is coming back."""
    job_id = queue.enqueue(make_spec(gpus=1))
    queue.remove_marker(job_id)
    paths.preempt_file(job_id).touch()
    old = datetime.now(UTC).timestamp() - cleanup.INCOMING_STALE_S - 1
    os.utime(paths.state_file(job_id), (old, old))

    dispatcher, _ = make_dispatcher()
    dispatcher.sweep_abandoned_submits()

    assert jobs.read_state(job_id).status == "queued"


def test_a_runner_left_unrecorded_by_a_dead_dispatcher_is_adopted(
    gpuc_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`launch_ready` writes `running` before it has a process to name. A
    dispatcher killed between that write and the runner pid leaves a job that
    reads as abandoned while its runner is training: failing it would lose the
    job and hand the card it is on to whatever starts next."""
    job_id = queue.enqueue(make_spec(gpus=1))
    queue.remove_marker(job_id)
    jobs.update_state(job_id, status="running", gpus=[FAKE_GPUS[0]], started_at=jobs.utc_now())
    monkeypatch.setattr(host_dispatcher, "live_runner_pids", lambda: {job_id: os.getpid()})

    dispatcher, _ = make_dispatcher()
    dispatcher.adopt_orphans()

    assert jobs.read_state(job_id).status == "running"
    assert dispatcher.running[job_id].pid == os.getpid()
    assert dispatcher.free_gpus() == [FAKE_GPUS[1]]


def test_a_preempted_job_syncing_under_an_unrecorded_runner_is_not_queued_again(
    gpuc_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same window on the way out: a runner signalled before it recorded
    itself still writes its final state and then spends the whole output sync
    alive. Queued on that evidence, attempt 2 would start in the workdir
    attempt 1 is still uploading from."""
    job_id = queue.enqueue(make_spec(gpus=1))
    queue.remove_marker(job_id)
    queue.enqueue(make_spec(gpus=2, priority=1))
    jobs.update_state(job_id, status="running", gpus=[FAKE_GPUS[0]])
    queue.preempt(job_id)
    jobs.update_state(job_id, status="failed", reason="preempted", ended_at=jobs.utc_now())
    monkeypatch.setattr(host_dispatcher, "live_runner_pids", lambda: {job_id: os.getpid()})

    dispatcher, _ = make_dispatcher()
    dispatcher.adopt_orphans()

    assert jobs.read_state(job_id).status == "failed"
    assert queue.is_preempted(job_id)
    assert job_id in dispatcher.running


def test_a_running_job_with_no_runner_anywhere_still_fails(
    gpuc_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job_id = queue.enqueue(make_spec(gpus=1))
    queue.remove_marker(job_id)
    jobs.update_state(job_id, status="running", gpus=[FAKE_GPUS[0]])
    monkeypatch.setattr(host_dispatcher, "live_runner_pids", dict)

    dispatcher, _ = make_dispatcher()
    dispatcher.adopt_orphans()

    assert jobs.read_state(job_id).reason == "runner-died"
    assert dispatcher.free_gpus() == [FAKE_GPUS[0], FAKE_GPUS[1]]


def test_the_runner_the_dispatcher_spawns_is_one_the_scan_recognises(
    gpuc_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The command is built here and read back in `runner`: two modules, one
    fact. A runner the spawner starts and the scan cannot name is adopted by
    nobody, and nothing else would notice the two had drifted apart."""
    job_id = "20250101-000000-abcdef"
    captured: list[list[str]] = []

    class FakePopen:
        pid = 4242

        def __init__(self, argv: list[str], **_kwargs: object) -> None:
            captured.append(argv)

    monkeypatch.setattr(host_dispatcher.subprocess, "Popen", FakePopen)
    host_dispatcher.default_spawn_runner(job_id)

    assert procinfo.runner_job_id(captured[0]) == job_id


def test_launch_records_the_runner_identity_but_no_job_pgid_yet(gpuc_home: Path) -> None:
    job_id = queue.enqueue(make_spec(gpus=1))
    dispatcher, spawned = make_dispatcher()
    dispatcher.run_once()
    state = jobs.read_state(job_id)
    assert state.pgid is None
    assert state.runner_pid == spawned[job_id].pid
    assert state.runner_boot_id == procinfo.boot_id()


def test_cancel_in_the_launch_window_never_signals_the_runners_own_group(
    gpuc_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = FakeClock()
    dispatcher, _ = make_dispatcher(clock=clock)
    job_id = queue.enqueue(make_spec(gpus=1))
    dispatcher.run_once()

    signals: list[tuple[int | None, int]] = []
    monkeypatch.setattr(dispatcher, "_signal_group", lambda pgid, sig: signals.append((pgid, sig)))
    monkeypatch.setattr(dispatcher, "_signal_pid", lambda pid, sig: signals.append((pid, sig)))
    queue.cancel(job_id)
    dispatcher.handle_cancels()
    assert signals == []
    assert "has not published a job process group yet" in paths.dispatcher_log().read_text()

    # The runner publishes the job's group, and only then is it signalled.
    jobs.update_state(job_id, pgid=123456)
    clock.advance(2.0)
    dispatcher.handle_cancels()
    assert signals == [(123456, signal.SIGKILL)]


def test_spawned_children_get_the_home_tool_dirs_on_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Everything a job runs inherits this PATH, and a pod's sshd does not
    include ~/.local/bin, where uv lives."""
    fake_home = tmp_path / "home"
    (fake_home / ".local/bin").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    env = host_dispatcher._child_env(tmp_path / "pkg")
    assert env["PATH"].split(":")[0] == str(fake_home / ".local/bin")
    assert env["PYTHONPATH"].startswith(str(tmp_path / "pkg"))


# -- automatic retention ------------------------------------------------------


def finished_job(days_old: float = 30.0, *, mirrored: bool = True) -> str:
    job_id = queue.enqueue(make_spec())
    queue.remove_marker(job_id)
    ended = (datetime.now(UTC) - timedelta(days=days_old)).isoformat()
    jobs.update_state(
        job_id,
        status="succeeded",
        ended_at=ended,
        meta_synced_at=ended if mirrored else None,
        meta_synced_to="s3://b/gpuc/h" if mirrored else None,
    )
    paths.workdir(job_id).mkdir(parents=True, exist_ok=True)
    (paths.workdir(job_id) / "venv.bin").write_bytes(b"x" * 4096)
    return job_id


def configure_retention(days: float | None, workdir_days: float | None = None) -> None:
    jobs.write_config(
        HostConfig(
            host="h",
            gpus=list(FAKE_GPUS),
            s3_prefix="s3://b/gpuc/h",
            retention_days=days,
            workdir_days=workdir_days,
        )
    )


def test_no_retention_setting_never_purges(gpuc_home: Path) -> None:
    configure_retention(None)
    job_id = finished_job()
    dispatcher, _ = make_dispatcher()
    dispatcher.run_once()
    assert paths.state_file(job_id).exists()


def test_neither_horizon_set_reclaims_nothing(gpuc_home: Path) -> None:
    configure_retention(None, None)
    job_id = finished_job()
    dispatcher, _ = make_dispatcher()
    dispatcher.run_once()
    assert paths.workdir(job_id).is_dir()


def test_the_workdir_horizon_sweeps_without_a_purge_horizon(gpuc_home: Path) -> None:
    configure_retention(None, 1.0)
    old = finished_job(days_old=2.0)
    young = finished_job(days_old=0.5)
    dispatcher, _ = make_dispatcher()
    dispatcher.run_once()
    assert not paths.workdir(old).exists()
    assert paths.workdir(young).is_dir()
    # Only the workdir: the record of the run is what `retention_days` takes.
    assert paths.state_file(old).exists()
    assert jobs.read_state(old).workdir_removed is True
    assert "workdirs (1 days): removed 1 workdir(s)" in paths.dispatcher_log().read_text()


def test_the_workdir_horizon_needs_no_mirror(gpuc_home: Path) -> None:
    configure_retention(None, 1.0)
    job_id = finished_job(days_old=2.0, mirrored=False)
    dispatcher, _ = make_dispatcher()
    dispatcher.run_once()
    assert not paths.workdir(job_id).exists()
    assert paths.state_file(job_id).exists()


def test_the_workdir_horizon_never_touches_a_running_job(gpuc_home: Path) -> None:
    configure_retention(None, 0.0)
    job_id = queue.enqueue(make_spec(gpus=1))
    dispatcher, _ = make_dispatcher()
    dispatcher.run_once()
    assert jobs.read_state(job_id).status == "running"
    dispatcher._last_reclaim_at = None
    dispatcher.run_once()
    assert paths.workdir(job_id).is_dir()


def test_the_two_horizons_run_together_without_double_counting(gpuc_home: Path) -> None:
    configure_retention(7.0, 1.0)
    ancient = finished_job(days_old=30.0)
    middling = finished_job(days_old=2.0)
    dispatcher, _ = make_dispatcher()
    dispatcher.run_once()
    assert not paths.job_dir(ancient).exists(), "the purge horizon takes the whole dir"
    assert paths.state_file(middling).exists()
    assert not paths.workdir(middling).exists()
    log = paths.dispatcher_log().read_text()
    assert "retention (7 days): purged 1 job dir" in log
    assert "workdirs (1 days): removed 1 workdir(s)" in log


def test_the_workdir_horizon_leaves_outputs_that_never_reached_the_mirror(
    gpuc_home: Path,
) -> None:
    """The sweep is on by default, so it may not be the thing that loses data."""
    configure_retention(None, 1.0)
    spec = make_spec(outputs=[{"path": "results", "s3": "s3://bucket/{job_id}"}])
    job_id = queue.enqueue(spec)
    queue.remove_marker(job_id)
    ended = (datetime.now(UTC) - timedelta(days=2)).isoformat()
    jobs.update_state(job_id, status="failed", ended_at=ended)
    results = paths.workdir(job_id) / "results"
    results.mkdir(parents=True, exist_ok=True)
    (results / "checkpoint.pt").write_bytes(b"w" * 4096)

    dispatcher, _ = make_dispatcher()
    dispatcher.run_once()
    assert (results / "checkpoint.pt").exists()


def test_the_workdir_horizon_leaves_a_job_that_asked_to_keep_its_workdir(
    gpuc_home: Path,
) -> None:
    configure_retention(None, 1.0)
    job_id = queue.enqueue(make_spec(cleanup="never"))
    queue.remove_marker(job_id)
    ended = (datetime.now(UTC) - timedelta(days=2)).isoformat()
    jobs.update_state(job_id, status="failed", ended_at=ended)
    paths.workdir(job_id).mkdir(parents=True, exist_ok=True)
    (paths.workdir(job_id) / "venv.bin").write_bytes(b"x" * 4096)

    dispatcher, _ = make_dispatcher()
    dispatcher.run_once()
    assert paths.workdir(job_id).is_dir()


def test_the_workdir_horizon_runs_at_most_once_an_hour(gpuc_home: Path) -> None:
    configure_retention(None, 1.0)
    clock = FakeClock()
    dispatcher, _ = make_dispatcher(clock=clock)
    dispatcher.run_once()

    later = finished_job(days_old=2.0)
    clock.advance(59 * 60)
    dispatcher.run_once()
    assert paths.workdir(later).is_dir(), "swept again inside the hour"

    clock.advance(2 * 60)
    dispatcher.run_once()
    assert not paths.workdir(later).exists()


def test_retention_purges_at_startup(gpuc_home: Path) -> None:
    configure_retention(7.0)
    old = finished_job(days_old=30.0)
    young = finished_job(days_old=1.0)
    dispatcher, _ = make_dispatcher()
    dispatcher.run_once()
    assert not paths.job_dir(old).exists()
    assert paths.state_file(young).exists()
    assert "retention (7 days): purged 1 job dir" in paths.dispatcher_log().read_text()


def test_retention_runs_at_most_once_an_hour(gpuc_home: Path) -> None:
    configure_retention(7.0)
    clock = FakeClock()
    dispatcher, _ = make_dispatcher(clock=clock)
    dispatcher.run_once()

    later = finished_job(days_old=30.0)
    clock.advance(59 * 60)
    dispatcher.run_once()
    assert paths.state_file(later).exists(), "purged again inside the hour"

    clock.advance(2 * 60)
    dispatcher.run_once()
    assert not paths.job_dir(later).exists()


def test_retention_never_forces(gpuc_home: Path) -> None:
    configure_retention(1.0)
    unmirrored = finished_job(days_old=30.0, mirrored=False)
    dispatcher, _ = make_dispatcher()
    dispatcher.run_once()
    assert paths.state_file(unmirrored).exists()
    # ...but the ordinary workdir sweep still ran over it.
    assert not paths.workdir(unmirrored).exists()


def test_retention_never_touches_a_running_job(gpuc_home: Path) -> None:
    configure_retention(0.0)
    job_id = queue.enqueue(make_spec(gpus=1))
    dispatcher, _ = make_dispatcher()
    dispatcher.run_once()
    assert jobs.read_state(job_id).status == "running"
    dispatcher._last_reclaim_at = None
    dispatcher.run_once()
    assert paths.job_dir(job_id).is_dir()


# -- the drain's last go at unconfirmed outputs -------------------------------


def job_with_pending_outputs() -> str:
    spec = make_spec(outputs=[{"path": "results", "s3": "s3://bucket/{job_id}"}])
    job_id = queue.enqueue(spec)
    queue.remove_marker(job_id)
    jobs.update_state(job_id, status="succeeded", reason="sync", ended_at=jobs.utc_now())
    (paths.workdir(job_id) / "results").mkdir(parents=True, exist_ok=True)
    (paths.workdir(job_id) / "results" / "a.txt").write_text("hi\n")
    return job_id


def test_the_drain_retries_unconfirmed_outputs_and_records_success(
    gpuc_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sync, "aws_binary", lambda env=None: "/fake/aws")
    configure_pod(idle_minutes=0.0)
    job_id = job_with_pending_outputs()
    dispatcher, _ = make_dispatcher()
    dispatcher.drain_and_terminate("test")
    state = jobs.read_state(job_id)
    assert state.outputs_synced_at is not None
    assert state.outputs_lost is False


def test_the_drain_gives_up_after_three_tries_and_marks_the_outputs_lost(
    gpuc_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sync, "aws_binary", lambda env=None: "/fake/aws")
    configure_pod(idle_minutes=0.0)
    job_id = job_with_pending_outputs()
    attempts: list[list[str]] = []
    slept: list[float] = []
    clock = FakeClock()

    def failing(argv: list[str], timeout: float | None = None, env: sync.Env = None):
        attempts.append(argv)
        # The meta upload has to keep working, or the drain aborts before it
        # can write down that the outputs are gone.
        ok = "state.json" in " ".join(argv) or "log.txt" in " ".join(argv)
        return sync.CommandResult(argv, 0 if ok else 1, "" if ok else "AccessDenied")

    dispatcher, _ = make_dispatcher(clock=clock)
    dispatcher.deps.command_runner = failing
    dispatcher.deps.sleep = lambda seconds: (slept.append(seconds), clock.advance(seconds))[0]
    terminated: list[str] = []
    dispatcher.deps.terminate_call = lambda pod, key: terminated.append(pod) or ""
    monkeypatch.setenv("RUNPOD_API_KEY", "key")
    dispatcher.drain_and_terminate("test")

    output_attempts = [a for a in attempts if "s3://bucket/" in " ".join(a)]
    assert len(output_attempts) == 3
    assert slept == [60.0, 60.0]
    state = jobs.read_state(job_id)
    assert state.outputs_lost is True
    assert state.sync_error and "AccessDenied" in state.sync_error
    # The pod still goes away: it is billing, and a bucket we cannot reach is
    # no reason to keep paying for it.
    assert terminated == ["pod-1"]


def test_the_drain_records_the_meta_backup_for_every_job(
    gpuc_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sync, "aws_binary", lambda env=None: "/fake/aws")
    monkeypatch.setenv("RUNPOD_API_KEY", "key")
    configure_pod(idle_minutes=0.0)
    job_id = queue.enqueue(make_spec())
    queue.remove_marker(job_id)
    jobs.update_state(job_id, status="succeeded", ended_at=jobs.utc_now())
    dispatcher, _ = make_dispatcher()
    dispatcher.drain_and_terminate("test")
    state = jobs.read_state(job_id)
    assert state.meta_synced_at and state.meta_synced_to == "s3://b/gpuc/pod"


def test_an_ancient_idle_pod_is_only_ever_stopped_by_its_idle_timer(
    gpuc_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RUNPOD_API_KEY", "key")
    configure_pod(idle_minutes=600.0, age_h=500.0)
    terminated: list[str] = []
    dispatcher, _ = make_dispatcher(terminate_call=lambda pod, key: terminated.append(pod) or "")

    dispatcher.run_once()

    assert terminated == []


def test_an_auto_preempt_kill_is_only_asked_for_once(gpuc_home: Path) -> None:
    """A job already stopping has its cards counted as coming free; a second
    request would say nothing new and reset the escalation clock."""
    cheap = queue.enqueue(make_spec(gpus=2, priority=80, auto_preempt=True))
    dispatcher, _ = make_dispatcher()
    dispatcher.run_once()
    queue.enqueue(make_spec(gpus=2, priority=10))
    dispatcher.run_once()
    assert queue.kill_reason(cheap) == "preempted"
    written = paths.kill_file(cheap).stat().st_mtime_ns
    dispatcher.run_once()
    assert paths.kill_file(cheap).stat().st_mtime_ns == written


def test_a_runner_that_cannot_be_spawned_fails_the_job_instead_of_phantom_running(
    gpuc_home: Path,
) -> None:
    """State says `running` a line before the spawn, so a spawn that raises
    used to leave a job nothing was running: no runner pid to miss, no ended_at,
    and its GPUs handed back while `gpuc status` still showed it live."""
    job_id = queue.enqueue(make_spec(gpus=1))
    dispatcher, _ = make_dispatcher()

    def refuse(_job_id: str) -> subprocess.Popen[bytes]:
        raise OSError("fork: Resource temporarily unavailable")

    dispatcher.deps.spawn_runner = refuse
    dispatcher.run_once()

    state = jobs.read_state(job_id)
    assert (state.status, state.reason, state.exit_code) == ("failed", "spawn-failed", 1)
    assert state.ended_at and state.phase is None
    assert dispatcher.running == {}
    assert dispatcher.free_gpus() == FAKE_GPUS
    assert "could not spawn a runner" in paths.dispatcher_log().read_text()


def test_a_preempt_kill_the_runner_ignores_is_escalated(
    gpuc_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The kill marker is an ask, and a wedged runner never answers it. Without
    an escalation the job it was told to stop kept running, and the one waiting
    for its cards never started."""
    clock = FakeClock()
    dispatcher, spawned = make_dispatcher(clock=clock)
    job_id = queue.enqueue(make_spec(gpus=2, command="sleep 600"))
    dispatcher.run_once()
    jobs.update_state(job_id, pgid=123456)
    waiting = queue.enqueue(make_spec(gpus=2, priority=10))

    signals: list[tuple[int | None, int]] = []
    monkeypatch.setattr(dispatcher, "_signal_group", lambda pgid, sig: signals.append((pgid, sig)))
    monkeypatch.setattr(dispatcher, "_signal_pid", lambda pid, sig: signals.append((pid, sig)))
    monkeypatch.setattr(host_dispatcher, "process_group_alive", lambda _pgid: True)

    queue.preempt(job_id)
    dispatcher.run_once()
    assert queue.kill_reason(job_id) == "preempted"
    assert signals == []

    clock.advance(1.5)  # kill_grace_s is 1.0 in these tests
    dispatcher.run_once()
    assert signals[-1] == (123456, signal.SIGKILL)

    clock.advance(1.0)
    dispatcher.run_once()
    assert signals[-1] == (spawned[job_id].pid, signal.SIGTERM)

    clock.advance(1.0)
    dispatcher.run_once()
    assert signals[-1] == (spawned[job_id].pid, signal.SIGKILL)
    assert "escalating" in paths.dispatcher_log().read_text()

    spawned[job_id].finish(status="failed", reason="preempted")
    dispatcher.run_once()
    assert jobs.read_state(job_id).status == "queued"
    assert jobs.read_state(waiting).status == "running"


def test_the_drain_bounds_each_upload_and_skips_sync_preflight_failures(
    gpuc_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unbounded upload can hang a billing pod for hours, and a job that
    failed its sync preflight proved before it ran that these uploads cannot
    work -- retrying it three times only burns the budget."""
    monkeypatch.setattr(sync, "aws_binary", lambda env=None: "/fake/aws")
    monkeypatch.setenv("RUNPOD_API_KEY", "key")
    configure_pod(idle_minutes=0.0)
    pending = job_with_pending_outputs()
    hopeless = job_with_pending_outputs()
    jobs.update_state(hopeless, status="failed", reason="sync-preflight", exit_code=1)

    calls: list[tuple[list[str], float | None]] = []

    def record(argv: list[str], timeout: float | None = None, env: sync.Env = None):
        calls.append((argv, timeout))
        return sync.CommandResult(argv, 0, "")

    dispatcher, _ = make_dispatcher()
    dispatcher.deps.command_runner = record
    dispatcher.drain_and_terminate("test")

    uploads = [(argv, timeout) for argv, timeout in calls if "s3://bucket/" in " ".join(argv)]
    assert uploads, "the pending job's outputs should have been retried"
    assert all(
        timeout is not None and 0 < timeout <= host_dispatcher.OUTPUT_RETRY_BUDGET_S
        for _argv, timeout in uploads
    )
    assert not any(hopeless in " ".join(argv) for argv, _timeout in uploads)
    assert jobs.read_state(pending).outputs_synced_at is not None
    assert jobs.read_state(hopeless).outputs_lost is False


# -- a host may own its share of a box by nvidia-smi index ---------------------


def configure_indices(owned: list[str]) -> None:
    jobs.write_config(HostConfig(host="test-host", gpus=owned))


def test_gpus_owned_by_index_are_dispatched_as_uuids(gpuc_home: Path) -> None:
    """Ownership of a shared box is an agreement in nvidia-smi numbering, but a
    job is always pinned to a UUID: an index is only a name for whichever card
    the driver is calling 1 today."""
    configure_indices(["1"])
    dispatcher, _ = make_dispatcher()
    dispatcher.deps.smi = fake_smi()
    job_id = queue.enqueue(make_spec(gpus=1))
    dispatcher.run_once()

    assert jobs.read_state(job_id).gpus == [FAKE_GPUS[1]]
    assert dispatcher.owned_gpus() == [FAKE_GPUS[1]]
    assert dispatcher.free_gpus() == []


def test_an_orphan_holding_an_index_is_adopted_as_the_uuid_that_index_names(
    gpuc_home: Path,
) -> None:
    """A job launched before assignments were resolved host-side has an index in
    its state. Adopted as-is it would match nothing owned, so its card would
    read free and be handed to a second job while the first is still on it."""
    configure_indices(["0", "1"])
    dispatcher, _ = make_dispatcher()
    dispatcher.deps.smi = fake_smi()
    job_id = queue.enqueue(make_spec(gpus=1))
    queue.remove_marker(job_id)
    jobs.update_state(job_id, status="running", gpus=["1"], runner_pid=os.getpid())
    dispatcher.adopt_orphans()

    assert job_id in dispatcher.running
    assert dispatcher.free_gpus() == [FAKE_GPUS[0]]


def test_an_owned_index_the_host_cannot_see_is_not_handed_out(gpuc_home: Path) -> None:
    configure_indices(["0", "7"])
    dispatcher, _ = make_dispatcher()
    dispatcher.deps.smi = fake_smi()
    first = queue.enqueue(make_spec(gpus=1, priority=10))
    waiting = queue.enqueue(make_spec(gpus=1, priority=20))
    dispatcher.run_once()

    assert jobs.read_state(first).gpus == [FAKE_GPUS[0]]
    # Still queued, not failed: the host is configured for two cards and one of
    # them may well come back; only a spec bigger than the whole host fails.
    assert jobs.read_state(waiting).status == "queued"
    assert "does not report" in paths.dispatcher_log().read_text()


def test_a_job_wider_than_what_the_host_can_see_holds_no_cards(gpuc_home: Path) -> None:
    """It is not waiting for a card that is coming back, so holding one for it
    would idle the host for as long as a card stays missing -- possibly for
    ever. The job is not failed either: `config.gpus` says the host owns
    enough, and the card may be back on the next pass."""
    configure_indices(["0", "7"])  # only index 0 is real
    dispatcher, _ = make_dispatcher()
    dispatcher.deps.smi = fake_smi()
    wider_than_visible = queue.enqueue(make_spec(gpus=2, priority=10))
    behind = queue.enqueue(make_spec(gpus=1, priority=50))
    dispatcher.run_once()
    assert jobs.read_state(wider_than_visible).status == "queued"
    assert jobs.read_state(behind).status == "running"


def test_a_preempted_job_goes_back_in_the_queue_when_its_runner_stops(gpuc_home: Path) -> None:
    """The point of the command: the GPUs go to the job that was waiting, and
    the one that was stopped is queued again rather than lost."""
    running = queue.enqueue(make_spec(gpus=2, priority=50))
    dispatcher, spawned = make_dispatcher()
    dispatcher.run_once()
    assert jobs.read_state(running).status == "running"

    urgent = queue.enqueue(make_spec(gpus=2, priority=1))
    queue.preempt(running)
    dispatcher.run_once()
    # Still holding its cards: the runner has not stopped it yet.
    assert jobs.read_state(urgent).status == "queued"

    spawned[running].finish(status="failed", reason="preempted")
    dispatcher.run_once()
    assert jobs.read_state(urgent).status == "running"
    state = jobs.read_state(running)
    assert (state.status, state.attempt) == ("queued", 2)
    assert [e.job_id for e in queue.list_queued()] == [running]

    spawned[urgent].finish()
    dispatcher.run_once()
    assert jobs.read_state(running).status == "running"
    assert jobs.read_state(running).attempt == 2


def test_a_preempted_job_whose_runner_died_still_comes_back(gpuc_home: Path) -> None:
    """`failed: runner-died` is the dispatcher's own verdict on the attempt that
    was stopping, and it must not be the last word on a job somebody asked to
    keep."""
    job_id = queue.enqueue(make_spec(gpus=2))
    dispatcher, spawned = make_dispatcher()
    dispatcher.run_once()
    queue.enqueue(make_spec(gpus=2, priority=1))
    queue.preempt(job_id)
    spawned[job_id].returncode = -9
    dispatcher.run_once()
    state = jobs.read_state(job_id)
    assert (state.status, state.attempt) == ("queued", 2)


def test_a_job_preempted_while_no_dispatcher_was_alive_is_picked_up_at_startup(
    gpuc_home: Path,
) -> None:
    """Nothing else would ever look at the marker: the dispatcher that would
    have reaped this job is the one that died."""
    job_id = queue.enqueue(make_spec(gpus=1))
    queue.remove_marker(job_id)
    jobs.update_state(job_id, status="running")
    waiting = queue.enqueue(make_spec(gpus=1, priority=1))
    queue.preempt(job_id)
    jobs.update_state(
        job_id, status="failed", reason="preempted", exit_code=143, ended_at=jobs.utc_now()
    )

    dispatcher, _ = make_dispatcher()
    dispatcher.adopt_orphans()
    assert jobs.read_state(job_id).status == "queued"
    assert [e.job_id for e in queue.list_queued()] == [waiting, job_id]


def test_a_preempted_job_still_finalizing_is_not_queued_under_its_own_runner(
    gpuc_home: Path,
) -> None:
    """The runner that is still uploading owns that workdir. Queueing the job
    now would launch the next attempt straight into it -- two processes, one
    directory -- so the runner is adopted and `reap` does it afterwards."""
    job_id = queue.enqueue(make_spec(gpus=1))
    queue.remove_marker(job_id)
    # Two cards, so it cannot start while the finalizing runner holds one.
    queue.enqueue(make_spec(gpus=2, priority=1))
    jobs.update_state(
        job_id,
        status="running",
        gpus=[FAKE_GPUS[0]],
        runner_pid=os.getpid(),
        runner_boot_id=procinfo.boot_id(),
        runner_starttime=procinfo.starttime(os.getpid()),
    )
    queue.preempt(job_id)
    # ...and now it writes its final state, while still syncing.
    jobs.update_state(job_id, status="failed", reason="preempted", ended_at=jobs.utc_now())

    dispatcher, spawned = make_dispatcher()
    dispatcher.adopt_orphans()
    assert jobs.read_state(job_id).status == "failed"
    assert queue.is_preempted(job_id)
    assert job_id in dispatcher.running
    # Nothing is launched: not the preempted job into the workdir its own
    # runner is still writing, and nothing onto the cards it still holds.
    dispatcher.run_once()
    assert spawned == {}


def test_a_preempted_job_is_not_queued_again_on_a_host_that_is_going_away(
    gpuc_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Queued onto a pod that is terminating, the job would be lost outright --
    and the drain would stop counting its outputs as unconfirmed, because that
    list is finished jobs. Left finished, it keeps both."""
    monkeypatch.setenv("RUNPOD_API_KEY", "key")
    configure_pod(idle_minutes=600.0)
    dispatcher, spawned = make_dispatcher()
    job_id = queue.enqueue(make_spec(gpus=1, command="sleep 600"))
    dispatcher.run_once()
    queue.enqueue(make_spec(gpus=1, priority=1))
    queue.preempt(job_id)
    spawned[job_id].finish(status="failed", reason="preempted")
    paths.draining_file().write_text("idle\n")
    dispatcher.run_once()

    state = jobs.read_state(job_id)
    assert (state.status, state.reason) == ("failed", "preempted")
    assert queue.find_marker(job_id) is None
    assert not queue.is_preempted(job_id)
    assert "this host is draining" in paths.dispatcher_log().read_text()


def test_a_preempt_the_runner_ignores_is_escalated(
    gpuc_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The dispatcher did not write this kill marker, and before it adopted one
    it had no clock for it: a wedged runner kept the job (and its GPUs) for ever."""
    clock = FakeClock()
    dispatcher, spawned = make_dispatcher(clock=clock)
    job_id = queue.enqueue(make_spec(gpus=1))
    dispatcher.run_once()
    jobs.update_state(job_id, pgid=123456)

    signals: list[tuple[int | None, int]] = []
    monkeypatch.setattr(dispatcher, "_signal_group", lambda pgid, sig: signals.append((pgid, sig)))
    monkeypatch.setattr(dispatcher, "_signal_pid", lambda pid, sig: signals.append((pid, sig)))
    monkeypatch.setattr(host_dispatcher, "process_group_alive", lambda _pgid: True)

    queue.enqueue(make_spec(gpus=1, priority=1))
    queue.preempt(job_id)
    dispatcher.run_once()
    assert signals == []

    clock.advance(1.5)  # kill_grace_s is 1.0 in these tests
    dispatcher.run_once()
    assert signals[-1] == (123456, signal.SIGKILL)

    clock.advance(1.0)
    dispatcher.run_once()
    assert signals[-1] == (spawned[job_id].pid, signal.SIGTERM)


def test_a_job_queued_again_after_a_preempt_still_counts_as_holding_outputs(
    gpuc_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The stopped attempt's results are still in that workdir, and the drain
    is the last thing that will ever look at them. Filtering the retry list on
    `finished` alone hid them the moment the job went back in the queue."""
    monkeypatch.setattr(sync, "aws_binary", lambda env=None: "/fake/aws")
    configure_pod(idle_minutes=0.0)
    job_id = job_with_pending_outputs()
    queue.enqueue(make_spec(priority=1))  # what the preempt is making room for
    jobs.update_state(job_id, status="running", ended_at=None)
    queue.preempt(job_id)
    jobs.update_state(job_id, status="failed", reason="preempted", ended_at=jobs.utc_now())
    assert queue.requeue_preempted(job_id) == 2

    dispatcher, _ = make_dispatcher()
    assert dispatcher.unconfirmed_output_jobs() == [job_id]
    dispatcher.drain_and_terminate("test")
    assert jobs.read_state(job_id).outputs_synced_at is not None


def test_a_running_jobs_outputs_are_left_to_its_own_runner(gpuc_home: Path) -> None:
    job_id = job_with_pending_outputs()
    jobs.update_state(job_id, status="running", ended_at=None)
    dispatcher, _ = make_dispatcher()
    assert dispatcher.unconfirmed_output_jobs() == []


def test_a_preempted_job_whose_runner_died_before_the_dispatcher_did_comes_back(
    gpuc_home: Path,
) -> None:
    """Startup finds it still marked `running` with nobody running it. Without
    the re-queue here nothing would look at the marker again until some later
    dispatcher started -- and then it would resurrect a job reported failed
    hours before."""
    job_id = queue.enqueue(make_spec(gpus=1))
    queue.remove_marker(job_id)
    jobs.update_state(job_id, status="running", gpus=[FAKE_GPUS[0]], runner_pid=2**30)
    queue.enqueue(make_spec(gpus=1, priority=1))
    queue.preempt(job_id)

    dispatcher, _ = make_dispatcher()
    dispatcher.adopt_orphans()
    state = jobs.read_state(job_id)
    assert (state.status, state.attempt) == ("queued", 2)
    assert not queue.is_preempted(job_id)


# -- shared GPUs: cards we borrow rather than own ------------------------------

SHARED_GPUS = [
    "GPU-00000000-0000-0000-0000-000000000003",
    "GPU-00000000-0000-0000-0000-000000000004",
]
ALL_GPUS = [*FAKE_GPUS, *SHARED_GPUS]


def shared_host(
    *,
    owned: list[str] | None = None,
    shared: list[str] | None = None,
    utilization: dict[str, float] | None = None,
    memory_used: dict[str, float] | None = None,
) -> tuple[Dispatcher, dict[str, FakeRunnerProcess]]:
    """A dispatcher on a box of four cards, two of which are somebody else's."""
    jobs.write_config(
        HostConfig(
            host="test-host",
            gpus=list(FAKE_GPUS if owned is None else owned),
            shared_gpus=list(SHARED_GPUS if shared is None else shared),
        )
    )
    dispatcher, spawned = make_dispatcher()
    dispatcher.deps.smi = fake_smi(ALL_GPUS, utilization=utilization, memory_used=memory_used)
    return dispatcher, spawned


def enqueue_in_order(*specs: dict[str, Any]) -> list[str]:
    """Queue these jobs in exactly this order, whatever their priorities say.

    Dispatch order is the marker name `<priority>-<job id>`, and a real job id
    carries a random suffix -- so two jobs queued in the same second at the
    same priority sort unpredictably, and a test about which card a *particular*
    job gets would pass or fail on that coin flip.
    """
    return [
        queue.enqueue(make_spec(job_id=f"20260101-000000-{index:06d}", **spec))
        for index, spec in enumerate(specs)
    ]


def filling_the_owned_cards(priority: int = 50) -> list[dict[str, Any]]:
    """One single-card job per owned card, to make a borrower the only way on.

    `priority` because dispatch order is priority first: a test about a job at
    priority 20 has to queue these ahead of it, or the job under test simply
    takes an owned card and proves nothing."""
    return [{"gpus": 1, "priority": priority} for _ in FAKE_GPUS]


def test_a_job_that_did_not_ask_never_touches_a_shared_card(gpuc_home: Path) -> None:
    """Taking somebody else's GPU is a decision about a box, not about a job,
    so it is off unless the spec said so."""
    dispatcher, _ = shared_host()
    *holding, waiting = enqueue_in_order(*filling_the_owned_cards(), {"gpus": 1})
    dispatcher.run_once()

    assert all(jobs.read_state(job_id).status == "running" for job_id in holding)
    assert jobs.read_state(waiting).status == "queued"
    assert [e.job_id for e in queue.list_queued()] == [waiting]


def test_a_job_that_asked_borrows_an_idle_shared_card(gpuc_home: Path) -> None:
    dispatcher, _ = shared_host()
    *_, borrower = enqueue_in_order(*filling_the_owned_cards(), {"gpus": 1, "use_shared": True})
    dispatcher.run_once()

    assert jobs.read_state(borrower).status == "running"
    assert jobs.read_state(borrower).gpus == [SHARED_GPUS[0]]


def test_owned_cards_are_always_taken_before_borrowed_ones(gpuc_home: Path) -> None:
    """A shared card is held for the shortest time that runs the job, so a
    borrower uses every free card of ours first and borrows only the shortfall."""
    dispatcher, _ = shared_host()
    _, borrower = enqueue_in_order({"gpus": 1}, {"gpus": 2, "use_shared": True})
    dispatcher.run_once()

    assert jobs.read_state(borrower).gpus == [FAKE_GPUS[1], SHARED_GPUS[0]]


def test_a_job_bigger_than_the_host_owns_waits_for_the_shared_cards(gpuc_home: Path) -> None:
    """The second thing shared cards are for: four cards on a host that owns
    two, once the other two go quiet -- rather than a job that can never run."""
    dispatcher, _ = shared_host(memory_used={SHARED_GPUS[1]: 4096.0})
    (job_id,) = enqueue_in_order({"gpus": 4, "use_shared": True})
    dispatcher.run_once()
    assert jobs.read_state(job_id).status == "queued"

    dispatcher.deps.smi = fake_smi(ALL_GPUS)
    dispatcher.run_once()
    assert jobs.read_state(job_id).status == "running"
    assert jobs.read_state(job_id).gpus == ALL_GPUS


def test_a_shared_card_somebody_else_is_on_is_not_borrowed(gpuc_home: Path) -> None:
    dispatcher, _ = shared_host(
        memory_used={SHARED_GPUS[0]: 1024.0}, utilization={SHARED_GPUS[1]: 55.0}
    )
    *_, borrower = enqueue_in_order(*filling_the_owned_cards(), {"gpus": 1, "use_shared": True})
    dispatcher.run_once()

    assert jobs.read_state(borrower).status == "queued"
    assert "is in use" in paths.dispatcher_log().read_text()


def test_two_borrowers_in_one_pass_do_not_get_the_same_card(gpuc_home: Path) -> None:
    dispatcher, _ = shared_host()
    *_, first, second = enqueue_in_order(
        *filling_the_owned_cards(),
        {"gpus": 1, "use_shared": True},
        {"gpus": 1, "use_shared": True},
    )
    dispatcher.run_once()

    assert jobs.read_state(first).gpus == [SHARED_GPUS[0]]
    assert jobs.read_state(second).gpus == [SHARED_GPUS[1]]


def test_a_borrowed_card_is_busy_until_its_job_ends(gpuc_home: Path) -> None:
    dispatcher, spawned = shared_host(owned=[])
    (first,) = enqueue_in_order({"gpus": 1, "use_shared": True})
    dispatcher.run_once()
    assert jobs.read_state(first).gpus == [SHARED_GPUS[0]]

    # nvidia-smi now reports our own job's memory on that card. It must not be
    # read as "somebody else has it" *or* as free: it is simply already ours.
    dispatcher.deps.smi = fake_smi(ALL_GPUS, memory_used={SHARED_GPUS[0]: 8192.0})
    second = queue.enqueue(make_spec(gpus=1, use_shared=True))
    dispatcher.run_once()
    assert jobs.read_state(second).gpus == [SHARED_GPUS[1]]

    spawned[first].finish()
    dispatcher.deps.smi = fake_smi(ALL_GPUS)
    dispatcher.run_once()
    assert dispatcher.borrowable_gpus() == [SHARED_GPUS[0]]


def test_a_job_too_big_even_with_shared_cards_fails_with_what_would_help(
    gpuc_home: Path,
) -> None:
    """Two ways not to fit on a 2-owned, 2-shared host, and the reason has to
    name which -- otherwise `needs 5 GPUs, host owns 2` sends somebody looking
    for a bigger host when `use_shared: true` was the answer."""
    dispatcher, _ = shared_host()
    asked, did_not_ask = enqueue_in_order(
        {"gpus": 5, "use_shared": True},
        {"gpus": 3},
    )
    dispatcher.run_once()

    assert all(jobs.read_state(job).status == "failed" for job in (asked, did_not_ask))
    assert "may borrow 2 shared" in (jobs.read_state(asked).reason or "")
    assert "use_shared: true" in (jobs.read_state(did_not_ask).reason or "")


def test_a_job_the_host_could_run_is_never_dropped_from_the_queue(gpuc_home: Path) -> None:
    """The dispatcher's "this can never run here" is a deletion, so it may only
    read what is fixed for a queued job's life. The card counts are configured
    rather than resolved, and `use_shared` cannot change after submit -- so a
    job that fits on paper waits however long the borrowing takes."""
    dispatcher, _ = shared_host(memory_used={SHARED_GPUS[1]: 4096.0})
    (job_id,) = enqueue_in_order({"gpus": 4, "use_shared": True})
    for _ in range(3):
        dispatcher.run_once()
    assert jobs.read_state(job_id).status == "queued"
    assert [e.job_id for e in queue.list_queued()] == [job_id]

    # Moving it later in the queue moves it; it does not end it.
    queue.reorder(job_id, 99)
    dispatcher.run_once()
    assert jobs.read_state(job_id).status == "queued"

    dispatcher.deps.smi = fake_smi(ALL_GPUS)
    dispatcher.run_once()
    assert jobs.read_state(job_id).gpus == ALL_GPUS


def test_a_card_listed_as_both_owned_and_shared_is_only_owned(gpuc_home: Path) -> None:
    """Owning it is the stronger claim: it is handed out freely, rather than
    handed out freely *and* second-guessed as somebody else's."""
    dispatcher, _ = shared_host(shared=[FAKE_GPUS[1], SHARED_GPUS[0]])

    assert dispatcher.shared_gpus() == [SHARED_GPUS[0]]
    (job_id,) = enqueue_in_order({"gpus": 3, "use_shared": True})
    dispatcher.run_once()
    assert jobs.read_state(job_id).gpus == [*FAKE_GPUS, SHARED_GPUS[0]]


def test_shared_cards_may_be_named_by_index(gpuc_home: Path) -> None:
    dispatcher, _ = shared_host(owned=["0"], shared=["2", "3"])
    assert dispatcher.shared_gpus() == SHARED_GPUS

    (job_id,) = enqueue_in_order({"gpus": 2, "use_shared": True})
    dispatcher.run_once()
    assert jobs.read_state(job_id).gpus == [FAKE_GPUS[0], SHARED_GPUS[0]]


def test_a_shared_entry_that_names_no_card_is_logged_and_skipped(gpuc_home: Path) -> None:
    """The same rule owned entries get: jobs wait, they are not failed, and the
    log says the card is not being handed out."""
    dispatcher, _ = shared_host(shared=["9"])
    assert dispatcher.shared_gpus() == []
    assert "config.shared_gpus lists 9" in paths.dispatcher_log().read_text()


def test_nothing_is_sampled_when_no_queued_job_wants_to_borrow(gpuc_home: Path) -> None:
    """The preflight would be an nvidia-smi exec every two seconds if it were
    taken unconditionally, and most passes have nothing that could use it."""
    asked: list[list[str]] = []
    dispatcher, _ = shared_host()
    inner = dispatcher.deps.smi

    def counting(args: list[str]) -> str:
        asked.append(args)
        return inner(args)

    dispatcher.deps.smi = counting
    enqueue_in_order({"gpus": 1})
    dispatcher.run_once()
    assert not any("memory.used" in " ".join(args) for args in asked)

    # Two borrowers, one pass: one sample, judged the same way for both.
    enqueue_in_order(*filling_the_owned_cards(), {"gpus": 1, "use_shared": True})
    queue.enqueue(make_spec(gpus=1, use_shared=True))
    dispatcher.run_once()
    assert sum("memory.used" in " ".join(args) for args in asked) == 1


# -- automatic preemption -----------------------------------------------------


def test_an_auto_preempt_job_gives_its_cards_to_a_more_important_one(gpuc_home: Path) -> None:
    """The whole feature: nobody ran a command, and the cheap job is queued
    again rather than lost."""
    cheap = queue.enqueue(make_spec(gpus=2, priority=80, auto_preempt=True))
    dispatcher, spawned = make_dispatcher()
    dispatcher.run_once()
    assert jobs.read_state(cheap).status == "running"

    urgent = queue.enqueue(make_spec(gpus=2, priority=10))
    dispatcher.run_once()
    assert queue.is_preempted(cheap)
    assert queue.kill_reason(cheap) == "preempted"
    assert f"auto_preempt: stopping so job {urgent}" in paths.log_file(cheap).read_text()

    spawned[cheap].finish(status="failed", reason="preempted")
    dispatcher.run_once()
    assert jobs.read_state(urgent).status == "running"
    state = jobs.read_state(cheap)
    assert (state.status, state.attempt) == ("queued", 2)


@pytest.mark.parametrize("priority", [80, 50])
def test_an_auto_preempt_job_keeps_its_cards_when_nothing_better_is_waiting(
    gpuc_home: Path, priority: int
) -> None:
    """Equal priority counts as "not better": the stopped job's id is the older
    one, so it would win the tie, take its own cards straight back, and be
    preempted again for ever without either job getting anywhere."""
    cheap = queue.enqueue(make_spec(gpus=2, priority=50, auto_preempt=True))
    dispatcher, _ = make_dispatcher()
    dispatcher.run_once()
    queue.enqueue(make_spec(gpus=2, priority=priority))
    dispatcher.run_once()
    assert not queue.is_preempted(cheap)
    assert queue.kill_reason(cheap) is None


def test_a_job_that_never_asked_for_it_is_not_preempted(gpuc_home: Path) -> None:
    job_id = queue.enqueue(make_spec(gpus=2, priority=80))
    dispatcher, _ = make_dispatcher()
    dispatcher.run_once()
    queue.enqueue(make_spec(gpus=2, priority=1))
    dispatcher.run_once()
    assert not queue.is_preempted(job_id)


def test_nothing_is_stopped_when_it_would_not_free_enough_cards(gpuc_home: Path) -> None:
    """One of the two cards the waiting job needs starts nothing, and the
    attempt it costs is thrown away for a job that still waits."""
    cheap = queue.enqueue(make_spec(gpus=1, priority=80, auto_preempt=True))
    queue.enqueue(make_spec(gpus=1, priority=80))
    dispatcher, _ = make_dispatcher()
    dispatcher.run_once()
    queue.enqueue(make_spec(gpus=2, priority=10))
    dispatcher.run_once()
    assert not queue.is_preempted(cheap)


def test_only_as_many_jobs_are_stopped_as_the_waiting_one_needs(gpuc_home: Path) -> None:
    """And the second pass, with the first job still stopping, does not take
    another: the cards it is about to hand back already cover the gap."""
    first = queue.enqueue(make_spec(gpus=1, priority=80, auto_preempt=True))
    second = queue.enqueue(make_spec(gpus=1, priority=70, auto_preempt=True))
    dispatcher, _ = make_dispatcher()
    dispatcher.run_once()
    queue.enqueue(make_spec(gpus=1, priority=10))
    dispatcher.run_once()
    # The least important of the two.
    assert queue.is_preempted(first)
    assert not queue.is_preempted(second)
    dispatcher.run_once()
    assert not queue.is_preempted(second)


def test_the_shortest_running_of_two_equals_is_the_one_that_gives_way(gpuc_home: Path) -> None:
    """What a preempt throws away is the work the attempt has already done."""
    older = queue.enqueue(
        make_spec(job_id="20260101-000000-aaaaaa", gpus=1, priority=80, auto_preempt=True)
    )
    dispatcher, _ = make_dispatcher()
    dispatcher.run_once()
    younger = queue.enqueue(
        make_spec(job_id="20260101-000001-bbbbbb", gpus=1, priority=80, auto_preempt=True)
    )
    dispatcher.run_once()
    queue.enqueue(make_spec(gpus=1, priority=10))
    dispatcher.run_once()
    assert queue.is_preempted(younger)
    assert not queue.is_preempted(older)


def test_two_waiting_jobs_are_given_a_card_each(gpuc_home: Path) -> None:
    first = queue.enqueue(make_spec(gpus=1, priority=80, auto_preempt=True))
    second = queue.enqueue(make_spec(gpus=1, priority=80, auto_preempt=True))
    dispatcher, _ = make_dispatcher()
    dispatcher.run_once()
    queue.enqueue(make_spec(gpus=1, priority=10))
    queue.enqueue(make_spec(gpus=1, priority=11))
    dispatcher.run_once()
    assert queue.is_preempted(first)
    assert queue.is_preempted(second)


def test_a_cancelled_job_in_the_queue_is_not_worth_preempting_for(gpuc_home: Path) -> None:
    """The marker alone, as `gpuc cancel` writes it before it takes the queue
    marker off: for that instant the job is both queued and not coming back."""
    cheap = queue.enqueue(make_spec(gpus=2, priority=80, auto_preempt=True))
    dispatcher, _ = make_dispatcher()
    dispatcher.run_once()
    doomed = queue.enqueue(make_spec(gpus=2, priority=10))
    paths.cancel_file(doomed).touch()
    dispatcher.preempt_for_waiting()
    assert not queue.is_preempted(cheap)


def test_a_draining_host_preempts_nothing(gpuc_home: Path) -> None:
    """The cards would go to nobody: the job is stopped and nothing replaces it."""
    cheap = queue.enqueue(make_spec(gpus=2, priority=80, auto_preempt=True))
    dispatcher, _ = make_dispatcher()
    dispatcher.run_once()
    queue.enqueue(make_spec(gpus=2, priority=10))
    paths.draining_file().touch()
    dispatcher.run_once()
    assert not queue.is_preempted(cheap)


def test_the_cards_freed_for_a_waiting_job_are_not_handed_back_to_the_stopped_ones(
    gpuc_home: Path,
) -> None:
    """The livelock this reserves cards to prevent. Two jobs give a card each
    up for one that needs both; their runners stop a pass apart, so without a
    reservation `launch_ready` -- which walks past a job that does not fit --
    hands the first card straight back to the job that just gave it up, and the
    next pass stops it again, for ever, with the waiting job still queued."""
    first = queue.enqueue(make_spec(gpus=1, priority=80, auto_preempt=True))
    second = queue.enqueue(make_spec(gpus=1, priority=70, auto_preempt=True))
    dispatcher, spawned = make_dispatcher()
    dispatcher.run_once()
    waiting = queue.enqueue(make_spec(gpus=2, priority=10))
    dispatcher.run_once()
    assert queue.is_preempted(first) and queue.is_preempted(second)

    spawned[first].finish(status="failed", reason="preempted")
    dispatcher.run_once()
    # Queued again, and *not* running on the card it just gave up: that card is
    # being held for the job it was freed for.
    assert jobs.read_state(first).status == "queued"
    assert jobs.read_state(waiting).status == "queued"

    spawned[second].finish(status="failed", reason="preempted")
    dispatcher.run_once()
    assert jobs.read_state(waiting).status == "running"
    assert jobs.read_state(waiting).gpus == FAKE_GPUS
    for job_id in (first, second):
        assert jobs.read_state(job_id).attempt == 2  # stopped once, not once a pass


def test_a_manual_preempt_in_flight_does_not_donate_its_cards_twice(gpuc_home: Path) -> None:
    """`gpuc preempt` puts a job back in the *queue*, so its card is not simply
    coming free: the job is about to compete for it at its own priority, and it
    stops a pass before the auto-preempted one does. Counting that card towards
    the waiting job's gap and then letting its old holder take it back is how
    an attempt gets spent on a job that still cannot start."""
    manual = queue.enqueue(make_spec(gpus=1, priority=20))
    cheap = queue.enqueue(make_spec(gpus=1, priority=80, auto_preempt=True))
    dispatcher, spawned = make_dispatcher()
    dispatcher.run_once()
    waiting = queue.enqueue(make_spec(gpus=2, priority=5))
    queue.preempt(manual)  # somebody wanted that one card back for the big job

    dispatcher.run_once()
    assert queue.is_preempted(cheap)  # the other card, so the big job can start
    spawned[manual].finish(status="failed", reason="preempted")
    dispatcher.run_once()
    assert jobs.read_state(manual).status == "queued"  # and not running again

    spawned[cheap].finish(status="failed", reason="preempted")
    dispatcher.run_once()
    assert jobs.read_state(waiting).status == "running"


def test_one_stop_that_fails_does_not_take_the_rest_of_the_set_with_it(
    gpuc_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`enough_to_start` picks a set that covers the whole gap; once one of them
    cannot be stopped the gap is not covered any more, and stopping the others
    would spend their attempts on a job that still cannot start."""
    first = queue.enqueue(make_spec(gpus=1, priority=80, auto_preempt=True))
    second = queue.enqueue(make_spec(gpus=1, priority=70, auto_preempt=True))
    dispatcher, _ = make_dispatcher()
    dispatcher.run_once()
    queue.enqueue(make_spec(gpus=2, priority=10))

    real = queue.preempt

    def refuse_the_first(job_id: str, priority: int | None = None) -> str:
        if job_id == first:  # the least important, so the one tried first
            raise OSError("read-only file system")
        return real(job_id, priority)

    monkeypatch.setattr(host_dispatcher.queue, "preempt", refuse_the_first)
    dispatcher.run_once()
    assert not queue.is_preempted(first)
    assert not queue.is_preempted(second)
    assert "was not stopped" in paths.dispatcher_log().read_text()


def test_cards_held_for_a_job_that_is_cancelled_are_handed_out_again(gpuc_home: Path) -> None:
    """Otherwise they idle for the life of the dispatcher, waiting for a job
    nobody is waiting on any more."""
    cheap = queue.enqueue(make_spec(gpus=2, priority=80, auto_preempt=True))
    dispatcher, spawned = make_dispatcher()
    dispatcher.run_once()
    waiting = queue.enqueue(make_spec(gpus=2, priority=10))
    dispatcher.run_once()
    assert queue.is_preempted(cheap)

    queue.cancel(waiting)
    spawned[cheap].finish(status="failed", reason="preempted")
    dispatcher.run_once()
    assert jobs.read_state(cheap).status == "running"
    assert jobs.read_state(cheap).attempt == 2


def test_a_borrowed_card_is_not_freed_for_a_job_that_may_not_borrow(gpuc_home: Path) -> None:
    """Stopping it would hand back somebody else's card, which the waiting job
    cannot be dispatched onto: an attempt spent to start nothing."""
    dispatcher, _ = shared_host(shared=[SHARED_GPUS[0]])
    *_, borrower = enqueue_in_order(
        *filling_the_owned_cards(),
        {"gpus": 1, "priority": 80, "auto_preempt": True, "use_shared": True},
    )
    dispatcher.run_once()
    assert jobs.read_state(borrower).gpus == [SHARED_GPUS[0]]

    # Queued once every card is taken, or it would simply be dispatched onto one.
    waiting = queue.enqueue(make_spec(gpus=1, priority=10))
    dispatcher.run_once()
    assert not queue.is_preempted(borrower)
    assert jobs.read_state(waiting).status == "queued"


def test_a_borrowed_card_is_freed_for_a_job_that_asked_to_borrow(gpuc_home: Path) -> None:
    dispatcher, _ = shared_host(shared=[SHARED_GPUS[0]])
    *_, borrower = enqueue_in_order(
        *filling_the_owned_cards(),
        {"gpus": 1, "priority": 80, "auto_preempt": True, "use_shared": True},
    )
    dispatcher.run_once()
    assert jobs.read_state(borrower).gpus == [SHARED_GPUS[0]]

    waiting = queue.enqueue(make_spec(gpus=1, priority=10, use_shared=True))
    dispatcher.run_once()
    assert queue.is_preempted(borrower)
    assert f"auto_preempt: stopping so job {waiting}" in paths.log_file(borrower).read_text()


def test_a_job_that_can_only_run_by_borrowing_holds_no_owned_card(gpuc_home: Path) -> None:
    """The card it is short of comes free when somebody else's job ends, which
    is not this host's to wait for -- so the queue behind it runs."""
    dispatcher, _ = shared_host(
        shared=[SHARED_GPUS[0]],
        utilization={SHARED_GPUS[0]: 90.0},
        memory_used={SHARED_GPUS[0]: 8000.0},
    )
    holding, wide, narrow = enqueue_in_order(
        {"gpus": 1, "priority": 50},
        {"gpus": 3, "priority": 10, "use_shared": True},  # needs the busy shared card
        {"gpus": 1, "priority": 90},
    )
    dispatcher.run_once()
    assert jobs.read_state(wide).status == "queued"
    assert jobs.read_state(holding).status == "running"
    # The free owned card goes to the job behind it rather than idling for a
    # card somebody else is training on.
    assert jobs.read_state(narrow).status == "running"


def test_the_first_reading_of_the_shared_cards_is_logged_even_when_all_are_free(
    gpuc_home: Path,
) -> None:
    """The log line that says a job was allowed onto somebody else's card is the
    first thing anybody looks for when one was not, and "they were all free" was
    the one reading that never produced it."""
    dispatcher, _ = shared_host()
    *_, borrower = enqueue_in_order(*filling_the_owned_cards(), {"gpus": 1, "use_shared": True})
    dispatcher.run_once()

    assert jobs.read_state(borrower).status == "running"
    assert f"shared GPU(s) free to borrow: {', '.join(SHARED_GPUS)}" in (
        paths.dispatcher_log().read_text()
    )
