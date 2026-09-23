from __future__ import annotations

import itertools
import json
import os
import signal
import subprocess
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import pytest

from gpuc.host import cleanup, destinations, jobs, paths, queue, runner, sync, terminate
from gpuc.host import dispatcher as host_dispatcher
from gpuc.host import procs as procinfo
from gpuc.host.dispatcher import Dispatcher, DispatcherDeps
from gpuc.host.jobs import HostConfig, Outcome
from tests.conftest import FAKE_GPUS, fake_smi, make_spec
from tests.test_runner import deps as runner_deps


class FakeRunnerProcess:
    """Stands in for a spawned runner: claims its job as the real one does,
    then stays alive until the test ends it the way a runner would."""

    _next_pid = 500000

    def __init__(
        self, job_id: str, gpus: Sequence[str], attempt: int = 1, *, claim: bool = True
    ) -> None:
        FakeRunnerProcess._next_pid += 1
        self.pid = FakeRunnerProcess._next_pid
        self.job_id = job_id
        self.gpus = list(gpus)
        self.attempt = attempt
        self.returncode: int | None = None
        self.claimed = claim and self.claim()

    def claim(self) -> bool:
        return queue.claim(
            self.job_id,
            self.attempt,
            status="running",
            gpus=self.gpus,
            phase="setup",
            started_at=jobs.utc_now(),
            runner_pid=self.pid,
            runner_boot_id=procinfo.boot_id(),
        )

    def poll(self) -> int | None:
        return self.returncode

    def finish(self, status: str = "succeeded", reason: str | None = None) -> None:
        """The runner's last write: the terminal status, then it exits."""
        self.returncode = 0
        jobs.finish(self.job_id, Outcome(status, reason, 0 if status == "succeeded" else 1))

    def requeue(self) -> None:
        """A preempted runner's last write: `queued` at the next attempt."""
        self.returncode = runner.TERMINATED_EXIT_CODE
        assert queue.next_attempt(self.job_id) is not None


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
    *,
    claim: bool = True,
) -> tuple[Dispatcher, dict[str, FakeRunnerProcess]]:
    """A dispatcher whose runners are fakes. `claim=False` spawns runners that
    have not claimed their job yet, for the window between spawn and claim."""
    spawned: dict[str, FakeRunnerProcess] = {}

    def spawn(job_id: str, gpus: Sequence[str], attempt: int) -> subprocess.Popen[bytes]:
        proc = FakeRunnerProcess(job_id, gpus, attempt, claim=claim)
        spawned[job_id] = proc
        return cast("subprocess.Popen[bytes]", proc)

    deps = DispatcherDeps(
        smi=fake_smi(),
        spawn_runner=spawn,
        monotonic=clock or FakeClock(),
        terminate_call=terminate_call or (lambda pod, key: "{}"),
        command_runner=lambda argv, timeout=None, env=None: sync.CommandResult(argv, 0, ""),
        utcnow=utcnow or (lambda: datetime.now(UTC)),
        kill_grace_s=1.0,
    )
    return Dispatcher(deps=deps), spawned


def record_signals(monkeypatch: pytest.MonkeyPatch) -> list[tuple[int, int]]:
    """Every real signal sent to a pid or a group, as `(target, signal)`, with
    every target reading alive. Liveness probes (signal 0) are not recorded."""
    signals: list[tuple[int, int]] = []

    def send(target: int, sig: int) -> None:
        if sig != 0:
            signals.append((target, sig))

    monkeypatch.setattr(os, "kill", send)
    monkeypatch.setattr(os, "killpg", send)
    monkeypatch.setattr(os, "getpgid", lambda _pid: 1)
    return signals


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
    """`gpuc submit` refuses `gpus: 0`, so a spec that has it was written by
    hand. The reader refuses it too, which makes it `bad-spec` like any other
    spec that cannot be read -- and it holds nothing up on its way out."""
    none = queue.enqueue(make_spec(gpus=1, priority=10))
    document = json.loads(paths.spec_file(none).read_text())
    document["gpus"] = 0
    paths.spec_file(none).write_text(json.dumps(document))
    runnable = queue.enqueue(make_spec(gpus=1, priority=20))
    dispatcher, spawned = make_dispatcher()
    dispatcher.run_once()
    state = jobs.read_state(none)
    assert (state.status, state.reason, state.exit_code) == ("failed", "bad-spec", 1)
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
    assert queue.cancel(job_id) == "cancelled"
    dispatcher, spawned = make_dispatcher()
    dispatcher.run_once()
    assert spawned == {}
    assert jobs.read_state(job_id).status == "cancelled"


def test_a_job_cancelled_between_listing_and_launching_is_never_run(
    gpuc_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Listing the queue and launching a job out of it are not one operation,
    so a cancel landing in between has to win: the runner's claim is the
    compare-and-set that decides it, the runner exits, and the card the job
    would have taken is free again once it has."""
    job_id = queue.enqueue(make_spec(gpus=1))
    dispatcher, spawned = make_dispatcher()
    read_spec = jobs.read_spec
    listed: list[str] = []

    def cancel_it_first(wanted: str) -> jobs.JobSpec:
        listed.append(wanted)
        queue.cancel(wanted)
        return read_spec(wanted)

    monkeypatch.setattr(host_dispatcher.jobs, "read_spec", cancel_it_first)
    dispatcher.launch_ready()

    assert listed == [job_id], "the job was queued when the pass listed it"
    assert not spawned[job_id].claimed
    assert jobs.read_state(job_id).status == "cancelled"
    spawned[job_id].returncode = 0
    dispatcher.reap()
    assert dispatcher.free_gpus() == FAKE_GPUS


def test_a_stop_the_runner_never_acts_on_is_escalated_after_the_grace_period(
    gpuc_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The runner owns the kill: it polls its own state and stops the job. That
    is nothing at all when the runner is wedged, so an intent it has not
    honoured within the grace period gets the ladder -- the job's group, then
    the runner, then the runner's own group -- and nothing before it."""
    clock = FakeClock()
    dispatcher, spawned = make_dispatcher(clock=clock)
    job_id = queue.enqueue(make_spec(gpus=1))
    dispatcher.run_once()
    jobs.update_state(job_id, pgid=123456)

    signals = record_signals(monkeypatch)
    assert queue.cancel(job_id) == "cancelling"
    dispatcher.escalate_stops()
    assert signals == [], "inside the grace period the runner is left to do it"

    clock.advance(1.5)  # kill_grace_s is 1.0 in these tests
    dispatcher.escalate_stops()
    assert signals[-1] == (123456, signal.SIGKILL)

    clock.advance(1.0)
    dispatcher.escalate_stops()
    assert signals[-1] == (spawned[job_id].pid, signal.SIGTERM)

    clock.advance(1.0)
    dispatcher.escalate_stops()
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


def test_a_runner_that_dies_leaves_no_output_confirmed(gpuc_home: Path) -> None:
    """A periodic tick's success says nothing about what the job wrote after
    it, and the final upload never ran: the sweep must not take the workdir."""
    spec = make_spec(gpus=1, outputs=[{"path": "out", "s3": "s3://b/{job_id}"}])
    job_id = queue.enqueue(spec)
    dispatcher, spawned = make_dispatcher()
    dispatcher.run_once()
    jobs.record_upload(job_id, f"s3://b/{job_id}", "out", ok_at=jobs.utc_now())
    (paths.workdir(job_id) / "out").mkdir()
    (paths.workdir(job_id) / "out" / "late.pt").write_text("x")
    spawned[job_id].returncode = -9
    dispatcher.run_once()
    state = jobs.read_state(job_id)
    assert state.reason == "runner-died"
    assert not state.outputs_uploaded(jobs.read_spec(job_id))
    assert (
        cleanup.outputs_pending(job_id, jobs.read_spec(job_id), state)
        == "outputs not confirmed uploaded"
    )


def test_a_runner_that_dies_while_preempting_leaves_a_failed_job(gpuc_home: Path) -> None:
    """The runner is what queues a preempted job again, and a dead one queued
    nothing: `runner-died` is the one verdict for a dead runner, whatever was
    asked of the job, and the intent goes with it. `gpuc requeue` is the way
    back."""
    job_id = queue.enqueue(make_spec(gpus=1))
    dispatcher, spawned = make_dispatcher()
    dispatcher.run_once()
    queue.enqueue(make_spec(gpus=1, priority=1))
    queue.preempt(job_id)
    spawned[job_id].returncode = -9
    dispatcher.run_once()
    state = jobs.read_state(job_id)
    assert (state.status, state.reason, state.attempt, state.intent) == (
        "failed",
        "runner-died",
        1,
        None,
    )


def test_orphans_from_a_dead_dispatcher_are_reconciled(gpuc_home: Path) -> None:
    job_id = queue.enqueue(make_spec(gpus=1))
    jobs.update_state(job_id, status="running", gpus=[FAKE_GPUS[0]], runner_pid=2**30)
    dispatcher, _ = make_dispatcher()
    dispatcher.adopt_orphans()
    assert jobs.read_state(job_id).reason == "runner-died"


def test_a_live_orphan_is_adopted_not_failed(gpuc_home: Path) -> None:
    import os

    job_id = queue.enqueue(make_spec(gpus=1))
    jobs.update_state(job_id, status="running", gpus=[FAKE_GPUS[0]], runner_pid=os.getpid())
    dispatcher, _ = make_dispatcher()
    dispatcher.adopt_orphans()
    assert jobs.read_state(job_id).status == "running"
    assert job_id in dispatcher.running
    assert dispatcher.free_gpus() == [FAKE_GPUS[1]]


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
    monkeypatch.setattr(destinations, "find_binary", lambda name, env=None: f"/fake/{name}")
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
    monkeypatch.setattr(destinations, "find_binary", lambda name, env=None: None)
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
    assert (
        f"job {job_id}: SIGKILLing process group {orphan.pid} before freeing its GPUs"
        in paths.dispatcher_log().read_text()
    )


def test_a_dead_runners_leftover_scope_is_stopped_before_its_cards_are_freed(
    gpuc_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job_id = queue.enqueue(make_spec(gpus=1))
    dispatcher, spawned = make_dispatcher()
    dispatcher.run_once()
    stopped: list[str] = []
    monkeypatch.setattr(
        host_dispatcher.scope, "stop_unit", lambda unit: stopped.append(unit) or True
    )
    jobs.update_state(job_id, cgroup_unit="gpuc-job-x.scope")
    spawned[job_id].returncode = -9
    dispatcher.run_once()
    assert stopped == ["gpuc-job-x.scope"]
    assert jobs.read_state(job_id).reason == "runner-died"
    assert (
        f"job {job_id}: stopping leftover scope gpuc-job-x.scope before freeing its GPUs"
        in paths.dispatcher_log().read_text()
    )


def test_an_unreadable_state_for_a_running_job_is_logged_and_left_alone(
    gpuc_home: Path,
) -> None:
    """Writing defaults over a file that cannot be read would replace its
    priority and upload records with guesses; the job is dropped from the
    running set (its runner is gone) and the file is somebody's to look at."""
    job_id = queue.enqueue(make_spec(gpus=1))
    dispatcher, spawned = make_dispatcher()
    dispatcher.run_once()
    spawned[job_id].returncode = 0
    paths.state_file(job_id).write_text("{ truncated")
    dispatcher.run_once()
    assert paths.state_file(job_id).read_text() == "{ truncated"
    assert dispatcher.running == {}
    assert "state.json is unreadable" in paths.dispatcher_log().read_text()


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


def staged_submit(job_id: str, *, age_s: float = 0.0) -> Path:
    """A job dir a `gpuc submit` was building under `incoming/`, `age_s` old.

    Aged by mtime on every path in it, which is what `stale_incoming` measures:
    an rsync still adding files keeps refreshing it.
    """
    staging = paths.incoming_job_dir(job_id)
    (staging / "workdir").mkdir(parents=True)
    (staging / "workdir" / "train.py").write_text("print('hi')\n")
    stamp = datetime.now(UTC).timestamp() - age_s
    for path in (staging, staging / "workdir", staging / "workdir" / "train.py"):
        os.utime(path, (stamp, stamp))
    return staging


def test_a_dead_submits_secrets_go_with_its_staged_dir(gpuc_home: Path) -> None:
    """The secrets file was delivered before the enqueue that never came, and
    nothing else would ever unlink it."""
    abandoned = staged_submit("20260101-000000-secret", age_s=cleanup.INCOMING_STALE_S + 60)
    secrets = paths.job_env_file("20260101-000000-secret")
    secrets.parent.mkdir(parents=True, exist_ok=True)
    secrets.write_text("AWS_SECRET_ACCESS_KEY=hunter2\n")

    dispatcher, _ = make_dispatcher()
    dispatcher.run_once()

    assert not abandoned.exists()
    assert not secrets.exists()


def test_a_dir_left_under_incoming_is_removed_after_an_hour(gpuc_home: Path) -> None:
    """A job is accepted by renaming its dir out of `incoming/` into `jobs/`, so
    one still sitting there an hour later is a submit that died before the host
    was ever asked to run it -- and nothing else will ever clear it away."""
    abandoned = staged_submit("20260101-000000-abandon", age_s=cleanup.INCOMING_STALE_S + 60)

    dispatcher, _ = make_dispatcher()
    dispatcher.run_once()

    assert not abandoned.exists()
    assert "a submit that never finished enqueueing it" in paths.dispatcher_log().read_text()


def test_a_submit_still_rsyncing_into_incoming_is_left_alone(gpuc_home: Path) -> None:
    """The working tree lands there first and a big one takes a while. A sweep
    that could not tell it from a dead submit would delete a live one."""
    staging = staged_submit("20260101-000000-inflight")

    dispatcher, _ = make_dispatcher()
    dispatcher.run_once()

    assert (staging / "workdir" / "train.py").exists()


def test_an_accepted_job_is_never_swept_however_long_it_waits(gpuc_home: Path) -> None:
    """The sweep is about `incoming/`, and `jobs/` is where a job the host has
    accepted lives: an old queued job is waiting its turn, not abandoned."""
    job_id = queue.enqueue(make_spec(gpus=1))
    old = datetime.now(UTC).timestamp() - cleanup.INCOMING_STALE_S - 60
    os.utime(paths.state_file(job_id), (old, old))

    dispatcher, _ = make_dispatcher()
    dispatcher.sweep_stale_incoming()

    assert paths.job_dir(job_id).is_dir()


def test_a_running_state_naming_no_runner_is_runner_died(gpuc_home: Path) -> None:
    """A `running` state is written by the runner that claimed it, naming
    itself; one that names nobody has no process behind it to adopt."""
    job_id = queue.enqueue(make_spec(gpus=1))
    jobs.update_state(job_id, status="running", gpus=[FAKE_GPUS[0]])

    dispatcher, _ = make_dispatcher()
    dispatcher.adopt_orphans()

    assert jobs.read_state(job_id).reason == "runner-died"
    assert dispatcher.free_gpus() == [FAKE_GPUS[0], FAKE_GPUS[1]]


def test_the_runner_is_started_with_its_assignment(
    gpuc_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cards travel on the command line, and the runner claims the job
    with them; where the host has user systemd the runner gets a scope of its
    own, named without the assignment in it."""
    job_id = "20250101-000000-abcdef"
    captured: list[list[str]] = []

    class FakePopen:
        pid = 4242

        def __init__(self, argv: list[str], **_kwargs: object) -> None:
            captured.append(argv)

    monkeypatch.setattr(host_dispatcher.subprocess, "Popen", FakePopen)
    monkeypatch.setenv(host_dispatcher.scope.ISOLATION_ENV, host_dispatcher.scope.PGID)
    host_dispatcher.default_spawn_runner(job_id, FAKE_GPUS, 2)
    monkeypatch.setenv(host_dispatcher.scope.ISOLATION_ENV, host_dispatcher.scope.CGROUP)
    host_dispatcher.default_spawn_runner(job_id, FAKE_GPUS, 2)

    plain, scoped = captured
    assert plain[1:] == [
        "-m",
        "gpuc.host",
        "run",
        job_id,
        "--gpus",
        ",".join(FAKE_GPUS),
        "--attempt",
        "2",
    ]
    assert scoped[:3] == ["systemd-run", "--user", "--scope"]
    assert scoped[scoped.index("--") + 1 :] == plain
    unit = next(arg for arg in scoped if arg.startswith("--unit="))
    assert unit.startswith(f"--unit=gpuc-run-{job_id}-") and "," not in unit


def test_a_runner_in_its_final_sync_is_given_a_long_patience_before_the_ladder(
    gpuc_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The final upload has no wall-clock cap, and a runner in it is honouring
    the stop: a ladder that reached it after the ordinary grace would kill the
    upload that matters most. One hung there for half an hour holds cards
    somebody wants, so the ladder does start, from the first rung."""
    clock = FakeClock()
    dispatcher, spawned = make_dispatcher(clock=clock)
    job_id = queue.enqueue(make_spec(gpus=1))
    dispatcher.run_once()
    jobs.update_state(job_id, pgid=123456, phase="sync")
    signals = record_signals(monkeypatch)

    assert queue.cancel(job_id) == "cancelling"
    for _ in range(10):
        dispatcher.escalate_stops()
        clock.advance(10.0)  # kill_grace_s is 1.0 in these tests
    assert signals == []
    assert "escalating" not in paths.dispatcher_log().read_text()

    # The rungs count from the end of the patience, one grace period apart.
    clock.advance(host_dispatcher.SYNC_STOP_PATIENCE_S - 100.0 + 0.5)
    dispatcher.escalate_stops()
    assert signals == [(123456, signal.SIGKILL)]
    clock.advance(1.5)
    dispatcher.escalate_stops()
    assert (spawned[job_id].pid, signal.SIGTERM) in signals


def test_launch_writes_nothing_until_the_runner_claims(gpuc_home: Path) -> None:
    """The runner claims the job itself, so a `running` state always names a
    live runner. Until then the job is still `queued` on disk, and the entry
    in `running` is what keeps the next pass from launching it twice."""
    job_id = queue.enqueue(make_spec(gpus=1))
    dispatcher, spawned = make_dispatcher(claim=False)
    dispatcher.run_once()
    state = jobs.read_state(job_id)
    assert (state.status, state.runner_pid, state.gpus) == ("queued", None, [])
    assert job_id in dispatcher.running
    assert dispatcher.free_gpus() == [FAKE_GPUS[1]]

    dispatcher.run_once()
    assert list(spawned) == [job_id], "launched twice"

    assert spawned[job_id].claim()
    state = jobs.read_state(job_id)
    assert (state.status, state.runner_pid, state.gpus) == (
        "running",
        spawned[job_id].pid,
        [FAKE_GPUS[0]],
    )
    assert state.pgid is None


def test_a_cancel_before_the_claim_wins_and_the_runner_exits(gpuc_home: Path) -> None:
    """A queued job is cancelled on the spot, claim or no claim: the runner's
    compare-and-set fails, it exits quietly, and the card is free again."""
    job_id = queue.enqueue(make_spec(gpus=1))
    dispatcher, spawned = make_dispatcher(claim=False)
    dispatcher.run_once()
    assert queue.cancel(job_id) == "cancelled"
    assert not spawned[job_id].claim()
    spawned[job_id].returncode = 0
    dispatcher.run_once()
    assert jobs.read_state(job_id).status == "cancelled"
    assert dispatcher.running == {}
    assert dispatcher.free_gpus() == FAKE_GPUS


def test_a_runner_that_dies_before_claiming_fails_the_job(gpuc_home: Path) -> None:
    """Still `queued` at the attempt this dispatcher launched, with the runner
    gone: without this the job would be launched again every pass, for ever."""
    job_id = queue.enqueue(make_spec(gpus=1))
    dispatcher, spawned = make_dispatcher(claim=False)
    dispatcher.run_once()
    spawned[job_id].returncode = 1
    dispatcher.run_once()
    state = jobs.read_state(job_id)
    assert (state.status, state.reason) == ("failed", "runner-died")
    assert list(spawned) == [job_id]
    assert dispatcher.free_gpus() == FAKE_GPUS


def test_a_claim_lost_to_another_runner_is_adopted_not_failed(gpuc_home: Path) -> None:
    """A dispatcher killed mid-spawn leaves a runner its successor knows
    nothing about; the successor spawns its own, the two race for the one
    claim, and exactly one wins. When the winner is not ours, the job is
    running under a runner we did not start, and it is adopted."""
    job_id = queue.enqueue(make_spec(gpus=1))
    dispatcher, spawned = make_dispatcher(claim=False)
    dispatcher.run_once()
    other = FakeRunnerProcess(job_id, [FAKE_GPUS[0]], claim=False)
    other.pid = os.getpid()
    assert other.claim()
    assert not spawned[job_id].claim()
    spawned[job_id].returncode = 0
    dispatcher.run_once()
    assert jobs.read_state(job_id).status == "running"
    assert dispatcher.running[job_id].runner_pid == os.getpid()
    assert dispatcher.free_gpus() == [FAKE_GPUS[1]]
    assert "adopted" in paths.dispatcher_log().read_text()


def test_a_claim_lost_on_other_cards_keeps_both_sets_busy_until_it_is_adopted(
    gpuc_home: Path,
) -> None:
    """The runner we lost the claim to was given its cards by the dispatcher
    we took over from, and they need not be the ones we chose. From its claim
    until our runner is reaped and the job adopted, the cards on disk are in
    use and the cards in `running` are what stops a second launch; a job may
    be handed neither."""
    job_id = queue.enqueue(make_spec(gpus=1))
    dispatcher, spawned = make_dispatcher(claim=False)
    dispatcher.run_once()
    assert dispatcher.running[job_id].gpus == [FAKE_GPUS[0]]

    theirs = FakeRunnerProcess(job_id, [FAKE_GPUS[1]], claim=False)
    theirs.pid = os.getpid()
    assert theirs.claim()
    waiting = queue.enqueue(make_spec(gpus=1))
    dispatcher.run_once()
    assert dispatcher.free_gpus() == []
    assert jobs.read_state(waiting).status == "queued"

    assert not spawned[job_id].claim()
    spawned[job_id].returncode = 0
    dispatcher.run_once()
    assert dispatcher.running[job_id].gpus == [FAKE_GPUS[1]]
    assert spawned[waiting].gpus == [FAKE_GPUS[0]]


def test_a_long_final_sync_after_a_cancel_is_never_sigkilled(
    gpuc_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The bug this design exists for. The runner's terminal write is its last
    act, so a cancelled job stays `running` with `phase=sync` for the whole of
    its final upload and cleanup, and the dispatcher's patience for a runner
    in its final sync covers all of it. The old order wrote the terminal
    status first, cleared no intent, and the ladder then SIGKILLed the runner
    mid-cleanup once a cancel had taken over three grace periods: no mirror
    record, secrets left behind."""
    monkeypatch.setattr(destinations, "find_binary", lambda name, env=None: f"/fake/{name}")
    jobs.write_config(HostConfig(host="h", gpus=list(FAKE_GPUS), s3_prefix="s3://b/gpuc/h"))
    clock = FakeClock()
    dispatcher, spawned = make_dispatcher(clock=clock)
    job_id = queue.enqueue(make_spec(gpus=1, command="sleep 300"))
    paths.job_env_file(job_id).write_text("AWS_ACCESS_KEY_ID=AKIA\n")
    dispatcher.run_once()
    signals = record_signals(monkeypatch)
    assert queue.cancel(job_id) == "cancelling"

    def slow_final(_self: sync.SyncLoop) -> None:
        # Four grace periods pass while the upload runs, with a dispatcher
        # pass at each: every rung of the ladder would have fired by now.
        for _ in range(4):
            clock.advance(dispatcher.deps.kill_grace_s + 0.5)
            dispatcher.escalate_stops()

    monkeypatch.setattr(sync.SyncLoop, "final", slow_final)
    # The real runner, in this process, on the job the fake was spawned for:
    # the fake pid's claim is rewritten with our own.
    jobs.update_state(job_id, status="queued", runner_pid=None)
    code = runner.run_job(
        job_id,
        [FAKE_GPUS[0]],
        1,
        runner_deps(
            command_runner=lambda argv, timeout=None, env=None: sync.CommandResult(argv, 0, "")
        ),
    )
    assert code == runner.TERMINATED_EXIT_CODE
    assert not any(target == os.getpid() for target, _ in signals), signals
    state = jobs.read_state(job_id)
    assert (state.status, state.reason, state.intent, state.phase) == (
        "cancelled",
        "cancelled",
        None,
        None,
    )
    assert state.mirrored
    assert not paths.job_env_file(job_id).exists()
    spawned[job_id].returncode = code
    dispatcher.run_once()
    assert dispatcher.running == {}


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
    ended = (datetime.now(UTC) - timedelta(days=days_old)).isoformat()
    jobs.update_state(job_id, status="succeeded", ended_at=ended)
    if mirrored:
        jobs.record_upload(job_id, f"s3://b/gpuc/h/jobs/{job_id}", None, ok_at=ended)
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
    assert jobs.read_state(old).workdir_bytes == 0
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
    jobs.update_state(job_id, status="succeeded", reason="sync", ended_at=jobs.utc_now())
    (paths.workdir(job_id) / "results").mkdir(parents=True, exist_ok=True)
    (paths.workdir(job_id) / "results" / "a.txt").write_text("hi\n")
    return job_id


def outputs_uploaded(job_id: str) -> bool:
    return jobs.read_state(job_id).outputs_uploaded(jobs.read_spec(job_id))


def test_the_drain_retries_unconfirmed_outputs_and_records_success(
    gpuc_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(destinations, "find_binary", lambda name, env=None: f"/fake/{name}")
    configure_pod(idle_minutes=0.0)
    job_id = job_with_pending_outputs()
    dispatcher, _ = make_dispatcher()
    dispatcher.drain_and_terminate("test")
    assert outputs_uploaded(job_id)
    assert jobs.read_state(job_id).outputs_lost is False


def test_the_drain_gives_up_after_three_tries_and_marks_the_outputs_lost(
    gpuc_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(destinations, "find_binary", lambda name, env=None: f"/fake/{name}")
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
    assert any("AccessDenied" in error for error in state.upload_errors())
    # The pod still goes away: it is billing, and a bucket we cannot reach is
    # no reason to keep paying for it.
    assert terminated == ["pod-1"]


def test_giving_up_on_outputs_marks_them_lost_and_changes_nothing_else(
    gpuc_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`outputs_lost` is the drain's whole verdict. The job's outcome is the
    runner's, and the failure itself is already in the destination's upload
    record, where `status` reads it from."""
    monkeypatch.setattr(destinations, "find_binary", lambda name, env=None: f"/fake/{name}")
    monkeypatch.setenv("RUNPOD_API_KEY", "key")
    configure_pod(idle_minutes=0.0)
    job_id = job_with_pending_outputs()
    before = jobs.read_state(job_id).to_dict()

    def outputs_fail(argv: list[str], timeout: float | None = None, env: sync.Env = None):
        broken = "s3://bucket/" in " ".join(argv)
        return sync.CommandResult(argv, 1 if broken else 0, "AccessDenied" if broken else "")

    dispatcher, _ = make_dispatcher()
    dispatcher.deps.command_runner = outputs_fail
    dispatcher.deps.sleep = lambda seconds: None
    dispatcher.drain_and_terminate("test")

    after = jobs.read_state(job_id)
    assert after.outputs_lost is True
    changed = {k for k, v in after.to_dict().items() if before.get(k) != v}
    assert changed == {"outputs_lost", "uploads"}
    assert (after.status, after.reason, after.problems) == ("succeeded", "sync", [])
    output_records = after.output_uploads()
    assert [(u.output, u.to, u.ok_at) for u in output_records] == [
        ("results", f"s3://bucket/{job_id}", None)
    ]
    assert output_records[0].error and "AccessDenied" in output_records[0].error
    assert after.mirrored  # the mirror record is the final meta sync's, not the drain's


def test_the_drain_records_the_meta_backup_for_every_job(
    gpuc_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(destinations, "find_binary", lambda name, env=None: f"/fake/{name}")
    monkeypatch.setenv("RUNPOD_API_KEY", "key")
    configure_pod(idle_minutes=0.0)
    job_id = queue.enqueue(make_spec())
    jobs.update_state(job_id, status="succeeded", ended_at=jobs.utc_now())
    dispatcher, _ = make_dispatcher()
    dispatcher.drain_and_terminate("test")
    state = jobs.read_state(job_id)
    assert state.mirrored
    assert state.mirror is not None and state.mirror.to == f"s3://b/gpuc/pod/jobs/{job_id}"


def test_an_ancient_idle_pod_is_only_ever_stopped_by_its_idle_timer(
    gpuc_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RUNPOD_API_KEY", "key")
    configure_pod(idle_minutes=600.0, age_h=500.0)
    terminated: list[str] = []
    dispatcher, _ = make_dispatcher(terminate_call=lambda pod, key: terminated.append(pod) or "")

    dispatcher.run_once()

    assert terminated == []


def test_an_auto_preempt_stop_is_only_asked_for_once(
    gpuc_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A job already stopping has its cards counted as coming free; a second
    request would say nothing new and reset the escalation clock."""
    cheap = queue.enqueue(make_spec(gpus=2, priority=80, auto_preempt=True))
    dispatcher, _ = make_dispatcher()
    dispatcher.run_once()

    asked: list[str] = []
    real = queue.preempt

    def counting(job_id: str, priority: int | None = None) -> str:
        asked.append(job_id)
        return real(job_id, priority)

    monkeypatch.setattr(host_dispatcher.queue, "preempt", counting)
    queue.enqueue(make_spec(gpus=2, priority=10))
    dispatcher.run_once()
    assert asked == [cheap]
    assert queue.stop_requested(cheap) == "preempted"

    dispatcher.run_once()
    assert asked == [cheap]


def test_a_runner_that_cannot_be_spawned_fails_the_job_instead_of_phantom_running(
    gpuc_home: Path,
) -> None:
    """A job nothing is running must not sit in the queue being launched
    again every pass, nor hold cards; it is failed, with the reason."""
    job_id = queue.enqueue(make_spec(gpus=1))
    dispatcher, _ = make_dispatcher()

    def refuse(_job_id: str, _gpus: Sequence[str], _attempt: int) -> subprocess.Popen[bytes]:
        raise OSError("fork: Resource temporarily unavailable")

    dispatcher.deps.spawn_runner = refuse
    dispatcher.run_once()

    state = jobs.read_state(job_id)
    assert (state.status, state.reason, state.exit_code) == ("failed", "spawn-failed", 1)
    assert state.ended_at and state.phase is None
    assert dispatcher.running == {}
    assert dispatcher.free_gpus() == FAKE_GPUS
    assert "could not spawn a runner" in paths.dispatcher_log().read_text()


def test_a_preempt_the_runner_ignores_is_escalated_and_the_job_still_comes_back(
    gpuc_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stop intent is an ask, and a wedged runner never answers it. Without an
    escalation the job it was told to stop kept running, and the one waiting for
    its cards never started."""
    clock = FakeClock()
    dispatcher, spawned = make_dispatcher(clock=clock)
    job_id = queue.enqueue(make_spec(gpus=2, command="sleep 600"))
    dispatcher.run_once()
    jobs.update_state(job_id, pgid=123456)
    waiting = queue.enqueue(make_spec(gpus=2, priority=10))

    signals = record_signals(monkeypatch)

    queue.preempt(job_id)
    dispatcher.run_once()
    assert queue.stop_requested(job_id) == "preempted"
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

    spawned[job_id].requeue()
    dispatcher.run_once()
    assert jobs.read_state(job_id).status == "queued"
    assert jobs.read_state(waiting).status == "running"


def test_the_drain_bounds_each_upload_and_skips_jobs_that_produced_nothing(
    gpuc_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unbounded upload can hang a billing pod for hours, and a job that
    declared outputs and never wrote them -- one that failed its sync
    preflight, say -- has nothing to retry; three tries would only burn the
    budget the jobs with real outputs need."""
    monkeypatch.setattr(destinations, "find_binary", lambda name, env=None: f"/fake/{name}")
    monkeypatch.setenv("RUNPOD_API_KEY", "key")
    configure_pod(idle_minutes=0.0)
    pending = job_with_pending_outputs()
    hopeless = queue.enqueue(make_spec(outputs=[{"path": "results", "s3": "s3://bucket/{job_id}"}]))
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
    assert outputs_uploaded(pending)
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


def test_a_job_waiting_for_a_missing_owned_card_holds_like_any_other(gpuc_home: Path) -> None:
    """`config.gpus` says the host has the card, so a host that cannot see it
    is misconfigured or broken. The job holds the card it can see and the
    queue behind it waits, which is how that gets noticed; it is not failed,
    since the configured host is big enough, and it runs once the card is
    back."""
    configure_indices(["0", "1"])
    dispatcher, _ = make_dispatcher()
    dispatcher.deps.smi = fake_smi([FAKE_GPUS[0]])  # index 1 has dropped off
    wide = queue.enqueue(make_spec(gpus=2, priority=10))
    behind = queue.enqueue(make_spec(gpus=1, priority=50))
    dispatcher.run_once()
    assert jobs.read_state(wide).status == "queued"
    assert jobs.read_state(behind).status == "queued"
    assert "does not report" in paths.dispatcher_log().read_text()

    dispatcher.deps.smi = fake_smi()
    dispatcher.run_once()
    assert jobs.read_state(wide).status == "running"
    assert jobs.read_state(wide).gpus == FAKE_GPUS
    assert jobs.read_state(behind).status == "queued"


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

    spawned[running].requeue()
    dispatcher.run_once()
    assert jobs.read_state(urgent).status == "running"
    state = jobs.read_state(running)
    assert (state.status, state.attempt) == ("queued", 2)
    assert [e.job_id for e in queue.list_queued()] == [running]

    spawned[urgent].finish()
    dispatcher.run_once()
    assert jobs.read_state(running).status == "running"
    assert jobs.read_state(running).attempt == 2


def test_a_preempted_job_still_finalizing_is_adopted_and_nothing_is_launched_into_it(
    gpuc_home: Path,
) -> None:
    """A preempted job whose runner is still there is a running job like any
    other: its status says so until the runner's last write. The runner is
    adopted, and the next attempt waits for the write that queues it."""
    job_id = queue.enqueue(make_spec(gpus=1))
    # Two cards, so it cannot start while the finalizing runner holds one.
    queue.enqueue(make_spec(gpus=2, priority=1))
    jobs.update_state(
        job_id,
        status="running",
        phase="sync",
        gpus=[FAKE_GPUS[0]],
        runner_pid=os.getpid(),
        runner_boot_id=procinfo.boot_id(),
        runner_starttime=procinfo.starttime(os.getpid()),
    )
    queue.preempt(job_id)

    dispatcher, spawned = make_dispatcher()
    dispatcher.adopt_orphans()
    assert job_id in dispatcher.running
    assert dispatcher.free_gpus() == [FAKE_GPUS[1]]
    dispatcher.run_once()
    assert spawned == {}


def test_a_stop_another_process_asked_for_gets_an_escalation_clock_of_its_own(
    gpuc_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`gpuc preempt` writes the intent over ssh, so this dispatcher never sent
    it and has nothing to time it from until it first sees it. Without a clock
    of its own, a wedged runner kept the job (and its GPUs) for ever."""
    clock = FakeClock()
    dispatcher, spawned = make_dispatcher(clock=clock)
    job_id = queue.enqueue(make_spec(gpus=1))
    dispatcher.run_once()
    jobs.update_state(job_id, pgid=123456)

    signals = record_signals(monkeypatch)

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
    monkeypatch.setattr(destinations, "find_binary", lambda name, env=None: f"/fake/{name}")
    configure_pod(idle_minutes=0.0)
    job_id = job_with_pending_outputs()
    queue.enqueue(make_spec(priority=1))  # what the preempt is making room for
    jobs.update_state(job_id, status="running", ended_at=None)
    queue.preempt(job_id)
    assert queue.next_attempt(job_id) == 2

    dispatcher, _ = make_dispatcher()
    assert dispatcher.unconfirmed_output_jobs() == [job_id]
    dispatcher.drain_and_terminate("test")
    assert outputs_uploaded(job_id)


def test_a_running_jobs_outputs_are_left_to_its_own_runner(gpuc_home: Path) -> None:
    job_id = job_with_pending_outputs()
    jobs.update_state(job_id, status="running", ended_at=None)
    dispatcher, _ = make_dispatcher()
    assert dispatcher.unconfirmed_output_jobs() == []


def test_a_preempted_job_whose_runner_died_before_the_dispatcher_did_is_failed(
    gpuc_home: Path,
) -> None:
    """Startup finds it still marked `running` with nobody running it: the
    same verdict as at any other time, since the runner that would have
    queued it again is gone."""
    job_id = queue.enqueue(make_spec(gpus=1))
    jobs.update_state(job_id, status="running", gpus=[FAKE_GPUS[0]], runner_pid=2**30)
    queue.enqueue(make_spec(gpus=1, priority=1))
    queue.preempt(job_id)

    dispatcher, _ = make_dispatcher()
    dispatcher.adopt_orphans()
    state = jobs.read_state(job_id)
    assert (state.status, state.reason, state.attempt, state.intent) == (
        "failed",
        "runner-died",
        1,
        None,
    )


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


_ORDERED_IDS = itertools.count()
"""Ids for `enqueue_in_order`, never reused: `enqueue` refuses a job id that
already has a dir under `jobs/`, so two calls in one test need two ranges."""


def enqueue_in_order(*specs: dict[str, Any]) -> list[str]:
    """Queue these jobs in exactly this order, whatever their priorities say.

    Dispatch order is `(priority, job id)`, and a real job id carries a random
    suffix -- so two jobs queued in the same second at the same priority sort
    unpredictably, and a test about which card a *particular* job gets would
    pass or fail on that coin flip.
    """
    return [
        queue.enqueue(make_spec(job_id=f"20260101-000000-{next(_ORDERED_IDS):06d}", **spec))
        for spec in specs
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
    assert dispatcher.borrowable_gpus() == ([SHARED_GPUS[0]], 0)


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
    assert queue.stop_requested(cheap) == "preempted"
    assert f"auto_preempt: stopping so job {urgent}" in paths.log_file(cheap).read_text()

    spawned[cheap].requeue()
    dispatcher.run_once()
    assert jobs.read_state(urgent).status == "running"
    state = jobs.read_state(cheap)
    assert (state.status, state.attempt) == ("queued", 2)


def test_auto_preempt_judges_a_job_by_the_priority_it_runs_at(gpuc_home: Path) -> None:
    """Submitted at 10, moved to 60 while queued: it is a 60 job now, and a 30
    job waiting is strictly more important than it."""
    cheap = queue.enqueue(make_spec(gpus=2, priority=10, auto_preempt=True))
    assert queue.reorder(cheap, 60)
    dispatcher, _ = make_dispatcher()
    dispatcher.run_once()
    assert jobs.read_state(cheap).status == "running"

    queue.enqueue(make_spec(gpus=2, priority=30))
    dispatcher.run_once()
    assert queue.is_preempted(cheap)


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
    assert queue.stop_requested(cheap) is None


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


def test_two_waiting_jobs_are_given_a_card_each_one_pass_apart(gpuc_home: Path) -> None:
    """One stop a pass: the job at the front is the only one asked, because a
    card handed back would reach it first. The pass after, its card is on its
    way and the second job is the one the queue is stuck on."""
    first, second = enqueue_in_order(
        {"gpus": 1, "priority": 80, "auto_preempt": True},
        {"gpus": 1, "priority": 70, "auto_preempt": True},
    )
    dispatcher, _ = make_dispatcher()
    dispatcher.run_once()
    queue.enqueue(make_spec(gpus=1, priority=10))
    queue.enqueue(make_spec(gpus=1, priority=11))
    dispatcher.run_once()
    assert queue.is_preempted(first)  # the least important of the two
    assert not queue.is_preempted(second)

    dispatcher.run_once()
    assert queue.is_preempted(second)


def test_nothing_is_stopped_for_a_job_that_has_been_cancelled(gpuc_home: Path) -> None:
    """A cancelled job leaves the queue there and then, so nothing is waiting
    for the cards any more and an attempt spent freeing them buys nothing."""
    cheap = queue.enqueue(make_spec(gpus=2, priority=80, auto_preempt=True))
    dispatcher, _ = make_dispatcher()
    dispatcher.run_once()
    doomed = queue.enqueue(make_spec(gpus=2, priority=10))
    assert queue.cancel(doomed) == "cancelled"
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

    spawned[first].requeue()
    dispatcher.run_once()
    # Queued again, and *not* running on the card it just gave up: that card is
    # being held for the job it was freed for.
    assert jobs.read_state(first).status == "queued"
    assert jobs.read_state(waiting).status == "queued"

    spawned[second].requeue()
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
    spawned[manual].requeue()
    dispatcher.run_once()
    assert jobs.read_state(manual).status == "queued"  # and not running again

    spawned[cheap].requeue()
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
    spawned[cheap].requeue()
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
    """The one job the strict order steps over: it could not fit even once
    every job of ours ends, because the card it is short of is a shared one
    somebody else is on. That comes free when *their* job ends, which is not
    this host's to wait for -- so the queue behind it runs."""
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


def test_a_wide_borrower_short_of_an_owned_card_holds_like_any_other(gpuc_home: Path) -> None:
    """The step-over is for a shared card somebody else is on, not for width.
    This job asks for more than the host owns, but the shared card it wants is
    idle and the card it is short of is an owned one our own job will free:
    it holds what it took, or a steady stream of one-card jobs behind it takes
    that owned card every time it frees and the job never runs."""
    dispatcher, spawned = shared_host(shared=[SHARED_GPUS[0]])
    holding = queue.enqueue(make_spec(gpus=1, priority=50))
    dispatcher.run_once()
    assert jobs.read_state(holding).gpus == [FAKE_GPUS[0]]

    wide = queue.enqueue(make_spec(gpus=3, priority=10, use_shared=True))
    narrow = queue.enqueue(make_spec(gpus=1, priority=50))
    dispatcher.run_once()
    assert jobs.read_state(wide).status == "queued"
    assert jobs.read_state(narrow).status == "queued"

    spawned[holding].finish()
    dispatcher.run_once()
    assert jobs.read_state(wide).status == "running"
    assert jobs.read_state(wide).gpus == [*FAKE_GPUS, SHARED_GPUS[0]]
    assert jobs.read_state(narrow).status == "queued"


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


def test_nothing_is_stopped_for_a_job_the_queue_ahead_would_take_the_cards_from(
    gpuc_home: Path,
) -> None:
    """The card would never reach the job it was freed for. The job at the
    front of the queue needs both cards and only one can be stopped for, so it
    holds what comes back and still does not start -- and the one-card job
    behind it, the only job the stop could have been for, gets nothing. Work
    discarded and nothing started."""
    stuck, cheap = enqueue_in_order(
        {"gpus": 1, "priority": 5},
        {"gpus": 1, "priority": 50, "auto_preempt": True},
    )
    dispatcher, spawned = make_dispatcher()
    dispatcher.run_once()
    assert jobs.read_state(cheap).status == "running"

    wide = queue.enqueue(make_spec(gpus=2, priority=10))
    narrow = queue.enqueue(make_spec(gpus=1, priority=20))
    dispatcher.run_once()
    assert not queue.is_preempted(cheap)
    assert queue.stop_requested(cheap) is None

    # And once the job that was holding is gone, the stop is worth making.
    queue.cancel(wide)
    dispatcher.run_once()
    assert queue.is_preempted(cheap)
    spawned[cheap].requeue()
    dispatcher.run_once()
    assert jobs.read_state(narrow).status == "running"
    assert jobs.read_state(stuck).status == "running"


def test_a_job_the_queue_steps_over_does_not_hold_up_a_preempt_behind_it(
    gpuc_home: Path,
) -> None:
    """The one job that holds nothing: it is short of a shared card somebody
    else is on, so the queue walks past it and the card a stop hands back
    reaches the job behind it."""
    dispatcher, spawned = shared_host(
        shared=[SHARED_GPUS[0]],
        utilization={SHARED_GPUS[0]: 90.0},
        memory_used={SHARED_GPUS[0]: 8000.0},
    )
    stuck, cheap = enqueue_in_order(
        {"gpus": 1, "priority": 5},
        {"gpus": 1, "priority": 50, "auto_preempt": True},
    )
    dispatcher.run_once()
    assert jobs.read_state(cheap).status == "running"

    queue.enqueue(make_spec(gpus=3, priority=10, use_shared=True))  # needs the busy card
    narrow = queue.enqueue(make_spec(gpus=1, priority=20))
    dispatcher.run_once()
    assert queue.is_preempted(cheap)

    spawned[cheap].requeue()
    dispatcher.run_once()
    assert jobs.read_state(narrow).status == "running"
    assert jobs.read_state(stuck).status == "running"


def test_a_free_shared_card_counts_towards_the_gap_a_stop_has_to_cover(
    gpuc_home: Path,
) -> None:
    """`launch_ready` holds the idle shared card for the waiting job, so the
    only card left to stop for is the owned one it is still short of. Counting
    the gap without it stops a second job for a card the waiting one has
    already been given."""
    dispatcher, spawned = shared_host(shared=[SHARED_GPUS[0]])
    first, second = enqueue_in_order(
        {"gpus": 1, "priority": 80, "auto_preempt": True},
        {"gpus": 1, "priority": 70, "auto_preempt": True},
    )
    dispatcher.run_once()
    waiting = queue.enqueue(make_spec(gpus=2, priority=10, use_shared=True))
    dispatcher.run_once()
    assert queue.is_preempted(first)  # the least important of the two, and only it
    assert not queue.is_preempted(second)

    spawned[first].requeue()
    dispatcher.run_once()
    assert jobs.read_state(waiting).status == "running"
    assert jobs.read_state(waiting).gpus == [FAKE_GPUS[1], SHARED_GPUS[0]]


def test_a_card_handed_out_earlier_in_the_pass_is_not_offered_to_the_walk(
    gpuc_home: Path,
) -> None:
    """One nvidia-smi reading serves the whole pass, so the shared card it found
    free may have been given to a job `launch_ready` launched since. Counting it
    again for a job still waiting stops somebody for a card that is not there."""
    dispatcher, _ = shared_host(shared=[SHARED_GPUS[0]])
    stuck, cheap = enqueue_in_order(
        {"gpus": 1, "priority": 5},
        {"gpus": 1, "priority": 60, "auto_preempt": True},
    )
    dispatcher.run_once()
    assert jobs.read_state(cheap).gpus == [FAKE_GPUS[1]]

    borrower = queue.enqueue(make_spec(gpus=1, priority=10, use_shared=True))
    wide = queue.enqueue(make_spec(gpus=2, priority=20, use_shared=True))
    dispatcher.run_once()
    assert jobs.read_state(borrower).gpus == [SHARED_GPUS[0]]
    # Stopping the cheap job frees one owned card; the shared card the wide job
    # would need on top of it is the one the borrower just took.
    assert not queue.is_preempted(cheap)
    assert jobs.read_state(wide).status == "queued"
    assert jobs.read_state(stuck).status == "running"


def test_a_second_job_is_not_stopped_for_a_card_already_on_its_way(gpuc_home: Path) -> None:
    """Least important first is by priority alone, so a two-card job is stopped
    for a one-card gap. Its spare card goes to the queue behind it like any
    other card that comes free, and the pass after, both waiting jobs are
    holding cards that are coming: neither is a job the queue is stuck on, so
    the second auto-preemptable job keeps its attempt."""
    dispatcher, _ = shared_host(owned=list(ALL_GPUS), shared=[])
    _, big, small = enqueue_in_order(
        {"gpus": 1, "priority": 50},
        {"gpus": 2, "priority": 90, "auto_preempt": True},
        {"gpus": 1, "priority": 80, "auto_preempt": True},
    )
    dispatcher.run_once()
    assert jobs.read_state(big).status == "running"

    queue.enqueue(make_spec(gpus=1, priority=10))
    queue.enqueue(make_spec(gpus=1, priority=20))
    dispatcher.run_once()
    assert queue.is_preempted(big)
    assert not queue.is_preempted(small)

    dispatcher.run_once()
    assert not queue.is_preempted(small)


def test_the_job_the_queue_is_stuck_on_is_the_last_one_asked(gpuc_home: Path) -> None:
    """Even about a card it could not use itself. The owned cards are held by
    jobs nothing may stop, so the job at the front is stuck for good; the
    borrower behind it could be dispatched onto the shared card, and that card
    would indeed go straight past a job that may not borrow. Working that out
    means modelling the dispatch order a second time, which is what this walk
    is instead of -- so the opportunity is given up, and nothing is stopped."""
    dispatcher, _ = shared_host(shared=[SHARED_GPUS[0]])
    *_, borrower = enqueue_in_order(
        *filling_the_owned_cards(priority=5),
        {"gpus": 1, "priority": 60, "auto_preempt": True, "use_shared": True},
    )
    dispatcher.run_once()
    assert jobs.read_state(borrower).gpus == [SHARED_GPUS[0]]

    stuck = queue.enqueue(make_spec(gpus=2, priority=10))
    waiting = queue.enqueue(make_spec(gpus=1, priority=20, use_shared=True))
    dispatcher.run_once()
    assert not queue.is_preempted(borrower)
    assert jobs.read_state(stuck).status == "queued"
    assert jobs.read_state(waiting).status == "queued"
