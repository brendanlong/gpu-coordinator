from __future__ import annotations

import os
import signal
import subprocess
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import pytest

from gpuc.host import dispatcher as host_dispatcher
from gpuc.host import jobs, paths, queue, sync, terminate
from gpuc.host import runner as procinfo
from gpuc.host.dispatcher import Dispatcher, DispatcherDeps
from gpuc.host.jobs import HostConfig
from tests.conftest import FAKE_GPUS, make_spec


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


def test_a_zero_gpu_job_never_waits(gpuc_home: Path) -> None:
    hog = queue.enqueue(make_spec(gpus=2, priority=10))
    cpu_job = queue.enqueue(make_spec(gpus=0, priority=90))
    dispatcher, _ = make_dispatcher()
    dispatcher.run_once()
    assert jobs.read_state(hog).status == "running"
    assert jobs.read_state(cpu_job).status == "running"
    assert jobs.read_state(cpu_job).gpus == []


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


def _finish_low_util(dispatcher: Dispatcher, spawned: dict[str, FakeRunnerProcess]) -> None:
    job_id = queue.enqueue(make_spec(gpus=1))
    dispatcher.run_once()
    spawned[job_id].finish(status="failed", reason="low-util")
    dispatcher.run_once()


def test_two_consecutive_low_util_failures_pause_the_host(gpuc_home: Path) -> None:
    dispatcher, spawned = make_dispatcher()
    _finish_low_util(dispatcher, spawned)
    assert not dispatcher.paused()
    _finish_low_util(dispatcher, spawned)
    assert dispatcher.paused()
    assert "low-util" in paths.paused_file().read_text()

    blocked = queue.enqueue(make_spec(gpus=1))
    dispatcher.run_once()
    assert jobs.read_state(blocked).status == "queued"


def test_a_success_between_low_util_failures_does_not_pause(gpuc_home: Path) -> None:
    dispatcher, spawned = make_dispatcher()
    _finish_low_util(dispatcher, spawned)
    good = queue.enqueue(make_spec(gpus=1))
    dispatcher.run_once()
    spawned[good].finish()
    dispatcher.run_once()
    _finish_low_util(dispatcher, spawned)
    assert not dispatcher.paused()


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


def configure_pod(
    idle_minutes: float = 15.0, ttl_hours: float | None = None, age_h: float = 0.0
) -> None:
    created = datetime.now(UTC) - timedelta(hours=age_h)
    jobs.write_config(
        HostConfig(
            host="pod",
            gpus=list(FAKE_GPUS),
            provider={"kind": "runpod", "pod_id": "pod-1"},
            idle_minutes=idle_minutes,
            ttl_hours=ttl_hours,
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


def test_ttl_terminates_an_old_but_idle_pod(
    gpuc_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RUNPOD_API_KEY", "key")
    configure_pod(idle_minutes=600.0, ttl_hours=1.0, age_h=2.0)
    terminated: list[str] = []
    dispatcher, _ = make_dispatcher(terminate_call=lambda pod, key: terminated.append(pod) or "")
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


def test_a_failed_final_drain_sync_does_not_terminate(
    gpuc_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sync, "aws_binary", lambda env=None: None)
    configure_pod(idle_minutes=0.0)
    terminated: list[str] = []
    dispatcher, _ = make_dispatcher(terminate_call=lambda pod, key: terminated.append(pod) or "")
    queue.enqueue(make_spec(gpus=1))
    dispatcher.run_once()
    dispatcher.running.clear()
    queue.list_queued()
    dispatcher.run_once()
    assert terminated == []
    assert not paths.draining_file().exists()


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


def configure_retention(days: float | None) -> None:
    jobs.write_config(
        HostConfig(host="h", gpus=list(FAKE_GPUS), s3_prefix="s3://b/gpuc/h", retention_days=days)
    )


def test_no_retention_setting_never_purges(gpuc_home: Path) -> None:
    configure_retention(None)
    job_id = finished_job()
    dispatcher, _ = make_dispatcher()
    dispatcher.run_once()
    assert paths.state_file(job_id).exists()


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
    dispatcher._last_purge_at = None
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
    # The pod still goes away: it is billing, and the TTL that sent us here
    # does not pause for a bucket we cannot reach.
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


# -- the TTL is opt-in, and when it is on the dispatcher enforces it ------------


def test_without_a_ttl_an_ancient_idle_pod_waits_for_its_idle_timer(
    gpuc_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RUNPOD_API_KEY", "key")
    configure_pod(idle_minutes=600.0, ttl_hours=None, age_h=500.0)
    terminated: list[str] = []
    dispatcher, _ = make_dispatcher(terminate_call=lambda pod, key: terminated.append(pod) or "")

    dispatcher.run_once()

    assert terminated == []


def test_a_ttl_kills_the_running_job_with_reason_ttl_then_terminates(
    gpuc_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RUNPOD_API_KEY", "key")
    monkeypatch.setattr(sync, "aws_binary", lambda env=None: "/fake/aws")
    configure_pod(idle_minutes=600.0, ttl_hours=1.0, age_h=2.0)
    terminated: list[str] = []
    dispatcher, spawned = make_dispatcher(
        terminate_call=lambda pod, key: terminated.append(pod) or ""
    )
    job_id = queue.enqueue(make_spec(gpus=1, command="sleep 600"))
    dispatcher.run_once()
    assert job_id in dispatcher.running

    dispatcher.run_once()
    # The runner owns the kill: the dispatcher asks, with the reason recorded.
    assert queue.kill_reason(job_id) == "ttl"
    assert terminated == []

    spawned[job_id].finish(status="failed", reason="ttl")
    dispatcher.run_once()
    assert terminated == ["pod-1"]
    assert paths.draining_file().exists()


def test_a_ttl_kill_is_only_asked_for_once(
    gpuc_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RUNPOD_API_KEY", "key")
    configure_pod(idle_minutes=600.0, ttl_hours=1.0, age_h=2.0)
    dispatcher, _ = make_dispatcher()
    job_id = queue.enqueue(make_spec(gpus=1, command="sleep 600"))
    dispatcher.run_once()
    dispatcher.run_once()
    written = paths.kill_file(job_id).stat().st_mtime_ns
    dispatcher.run_once()
    assert paths.kill_file(job_id).stat().st_mtime_ns == written
