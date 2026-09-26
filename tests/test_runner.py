from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

from gpuc.host import destinations, jobs, paths, procs, queue, runner, sync
from gpuc.host.gpus import SmiRunner
from gpuc.host.jobs import HostConfig, JobState
from gpuc.host.runner import RunnerDeps
from tests.conftest import (
    FAKE_GPUS,
    fake_smi,
    install_fake_nvidia_smi,
    install_fake_torch,
    make_spec,
)

ASSIGNED: dict[str, list[str]] = {}
"""What the dispatcher would hand each prepared job, by id: the runner takes
its assignment from the dispatcher and claims the job with it, so a prepared
job is a queued one plus the cards `run` will pass."""


def prepare(gpus: Sequence[str] = (FAKE_GPUS[0],), **overrides: object) -> str:
    """`gpus` is the assignment; `gpus_` is the spec's count, if not 1."""
    if "gpus_" in overrides:
        overrides["gpus"] = overrides.pop("gpus_")
    job_id = queue.enqueue(make_spec(**overrides))
    ASSIGNED[job_id] = list(gpus)
    return job_id


def run(job_id: str, deps_: RunnerDeps | None = None, attempt: int | None = None) -> int:
    """Run the job as the dispatcher would launch it: for the attempt its
    state is queued at, unless the test says otherwise."""
    if attempt is None:
        attempt = jobs.read_state(job_id).attempt
    return runner.run_job(job_id, ASSIGNED.get(job_id, [FAKE_GPUS[0]]), attempt, deps_ or deps())


def stopping_after_claim(job_id: str, how: Callable[[str], object]) -> SmiRunner:
    """An nvidia-smi whose first answer lands a stop request on the job.

    Verifying the assignment is the runner's first act after its claim, so a
    request made there reaches a job that is `running` and has not started a
    phase: the deterministic way to ask something of a job the runner owns,
    since a cancel or preempt before the claim is a different case entirely.
    """
    real = fake_smi()
    asked = False

    def smi(args: list[str]) -> str:
        nonlocal asked
        if not asked:
            asked = True
            how(job_id)
        return real(args)

    return smi


def deps(**overrides: object) -> RunnerDeps:
    base: dict[str, object] = {
        "smi": fake_smi(),
        "poll_interval_s": 0.02,
        "sample_interval_s": 0.02,
        "kill_grace_s": 2.0,
        "preflight": False,
    }
    base.update(overrides)
    return RunnerDeps(**base)  # type: ignore[arg-type]


def log_of(job_id: str) -> str:
    return paths.log_file(job_id).read_text()


def test_exit_code_propagates_to_the_runner(gpuc_home: Path) -> None:
    job_id = prepare(command="exit 42")
    assert run(job_id) == 42
    state = jobs.read_state(job_id)
    assert (state.status, state.exit_code, state.reason) == ("failed", 42, "exit 42")
    assert state.ended_at and state.pgid is None


def test_success_writes_succeeded_and_captures_output(gpuc_home: Path) -> None:
    job_id = prepare(command="echo hello-from-job; echo to-stderr >&2")
    assert run(job_id) == 0
    state = jobs.read_state(job_id)
    assert (state.status, state.exit_code, state.reason) == ("succeeded", 0, None)
    assert "hello-from-job" in log_of(job_id)
    assert "to-stderr" in log_of(job_id)


def test_setup_failure_short_circuits_the_command(gpuc_home: Path) -> None:
    job_id = prepare(setup="exit 3", command="echo SHOULD-NOT-RUN")
    assert run(job_id) == 3
    state = jobs.read_state(job_id)
    assert (state.status, state.reason) == ("failed", "setup")
    assert "SHOULD-NOT-RUN" not in log_of(job_id)


def test_setup_uses_pipefail(gpuc_home: Path) -> None:
    job_id = prepare(setup="false | cat", command="true")
    assert run(job_id) != 0
    assert jobs.read_state(job_id).reason == "setup"


def test_cuda_visible_devices_is_the_nvidia_smi_indices_of_the_assigned_cards(
    gpuc_home: Path,
) -> None:
    job_id = prepare(
        gpus=FAKE_GPUS, command='echo "CVD=$CUDA_VISIBLE_DEVICES ORDER=$CUDA_DEVICE_ORDER"'
    )
    assert run(job_id) == 0
    assert "CVD=0,1 ORDER=PCI_BUS_ID" in log_of(job_id)


def test_build_env_names_cards_by_index_when_every_one_has_an_index(gpuc_home: Path) -> None:
    spec = make_spec(job_id="j1")
    env = runner.build_env(spec, [FAKE_GPUS[1]], indices={FAKE_GPUS[0]: 0, FAKE_GPUS[1]: 1})
    assert env["CUDA_VISIBLE_DEVICES"] == "1"
    assert env["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"


@pytest.mark.parametrize("indices", [None, {FAKE_GPUS[0]: 0}], ids=["no-table", "partial-table"])
def test_build_env_falls_back_to_uuids_without_a_full_index_map(
    gpuc_home: Path, monkeypatch: pytest.MonkeyPatch, indices: dict[str, int] | None
) -> None:
    monkeypatch.delenv("CUDA_DEVICE_ORDER", raising=False)
    spec = make_spec(job_id="j1")
    env = runner.build_env(spec, FAKE_GPUS, indices=indices)
    assert env["CUDA_VISIBLE_DEVICES"] == f"{FAKE_GPUS[0]},{FAKE_GPUS[1]}"
    assert "CUDA_DEVICE_ORDER" not in env


def test_spec_env_cannot_override_the_gpu_assignment(gpuc_home: Path) -> None:
    job_id = prepare(
        gpus=[FAKE_GPUS[0]],
        env={"CUDA_VISIBLE_DEVICES": "0,1,2,3", "MY_VAR": "set"},
        command='echo "CVD=$CUDA_VISIBLE_DEVICES MY_VAR=$MY_VAR"',
    )
    assert run(job_id) == 0
    assert "CVD=0 MY_VAR=set" in log_of(job_id)


def test_secrets_file_is_sourced_into_the_job(gpuc_home: Path) -> None:
    job_id = prepare(command='echo "TOKEN=$HF_TOKEN"')
    paths.job_env_file(job_id).write_text('export HF_TOKEN="hf_abc"\n')
    assert run(job_id) == 0
    assert "TOKEN=hf_abc" in log_of(job_id)


def test_the_claim_records_the_assignment_and_the_runner_itself(gpuc_home: Path) -> None:
    """The runner's first act: one compare-and-set that takes the job out of
    the queue with the cards it was given and the identity a later dispatcher
    can check against the process table."""
    job_id = prepare(gpus=[FAKE_GPUS[1]], command="true")
    seen: list[tuple[str, list[str], int | None]] = []

    def watching(args: list[str]) -> str:
        state = jobs.read_state(job_id)
        seen.append((state.status, state.gpus, state.runner_pid))
        return fake_smi()(args)

    assert run(job_id, deps(smi=watching)) == 0
    assert seen[0] == ("running", [FAKE_GPUS[1]], os.getpid())
    state = jobs.read_state(job_id)
    assert (state.runner_pid, state.runner_boot_id) == (os.getpid(), procs.boot_id())
    assert state.runner_starttime == procs.starttime(os.getpid())
    assert state.started_at and state.isolation == "pgid"


def test_a_runner_whose_claim_fails_writes_nothing_and_exits_quietly(gpuc_home: Path) -> None:
    """Cancelled between the dispatcher's decision and the runner starting: the
    job is not the runner's to touch, so it neither runs nor reports."""
    job_id = prepare(command="touch RAN")
    assert queue.cancel(job_id) == "cancelled"
    assert run(job_id) == 0
    state = jobs.read_state(job_id)
    assert (state.status, state.reason, state.runner_pid) == ("cancelled", "cancelled", None)
    assert not (paths.workdir(job_id) / "RAN").exists()
    assert log_of(job_id) == ""


def test_a_runner_started_for_an_earlier_attempt_claims_nothing(gpuc_home: Path) -> None:
    """Spawned for attempt 1 and slow to start, it finds the job queued again
    at attempt 2 after a preempt: its cards were a decision about a pass that
    is over, and taking them now could put two runners on one job."""
    job_id = prepare(command="touch RAN")
    queue.enqueue(make_spec(priority=1))  # something waiting, or preempt refuses
    jobs.update_state(job_id, status="running")
    queue.preempt(job_id)
    assert queue.next_attempt(job_id, ran=False) == 2

    assert run(job_id, attempt=1) == 0
    state = jobs.read_state(job_id)
    assert (state.status, state.attempt, state.runner_pid) == ("queued", 2, None)
    assert not (paths.workdir(job_id) / "RAN").exists()

    assert run(job_id, attempt=2) == 0
    state = jobs.read_state(job_id)
    assert (state.status, state.attempt, state.runner_pid) == ("succeeded", 2, os.getpid())


def test_an_assigned_card_the_host_does_not_have_fails_the_job(gpuc_home: Path) -> None:
    job_id = prepare(gpus=["GPU-nope"], command="echo SHOULD-NOT-RUN")
    assert run(job_id) == 1
    assert jobs.read_state(job_id).reason == "gpu-assert"
    assert "SHOULD-NOT-RUN" not in log_of(job_id)
    assert "assigned GPUs not present on this host: GPU-nope" in log_of(job_id)


def test_a_stale_assigned_uuid_fails_the_job_before_it_starts(gpuc_home: Path) -> None:
    job_id = prepare(gpus=["GPU-stale"], command="echo SHOULD-NOT-RUN")
    assert run(job_id) == 1
    state = jobs.read_state(job_id)
    assert (state.status, state.reason) == ("failed", "gpu-assert")
    assert "SHOULD-NOT-RUN" not in log_of(job_id)


def test_a_job_that_asks_for_no_gpu_runs_with_none_visible(gpuc_home: Path) -> None:
    """No GPU check, since there is nothing to check, and an empty
    `CUDA_VISIBLE_DEVICES` rather than an absent one, which CUDA reads as
    every card on the machine."""
    job_id = prepare(gpus=[], gpus_=0, command='echo "CVD=[$CUDA_VISIBLE_DEVICES]"')
    assert run(job_id, deps(preflight=True)) == 0
    assert jobs.read_state(job_id).status == "succeeded"
    log = log_of(job_id)
    assert "CVD=[]" in log
    assert "phase=preflight" not in log


def test_a_job_that_asks_for_gpus_and_is_assigned_none_never_runs(gpuc_home: Path) -> None:
    """The dispatcher never does this; a runner started by hand with none
    fails the way a missing card does."""
    job_id = prepare(gpus=[], command="echo SHOULD-NOT-RUN")
    assert run(job_id) == 1
    state = jobs.read_state(job_id)
    assert (state.status, state.reason) == ("failed", "gpu-assert")
    assert "SHOULD-NOT-RUN" not in log_of(job_id)
    assert "no GPUs assigned" in log_of(job_id)


def test_failed_final_sync_turns_a_succeeded_job_into_failed_sync(
    gpuc_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(destinations, "find_binary", lambda name, env=None: None)
    job_id = prepare(
        command="mkdir -p out && echo x > out/x",
        outputs=[{"path": "out", "s3": "s3://bucket/{job_id}"}],
    )
    # The preflight would catch the missing binary first; this test is about the
    # final sync, which is the last line of defence behind it.
    assert run(job_id, deps(sync_preflight=False)) == 1
    state = jobs.read_state(job_id)
    assert (state.status, state.reason, state.exit_code) == ("failed", "sync", 1)
    # The upload failing *is* the reason, so it is not listed again beside it.
    assert state.problems == []
    assert "final sync FAILED" in log_of(job_id)


def test_failed_final_sync_does_not_mask_a_failed_job(
    gpuc_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(destinations, "find_binary", lambda name, env=None: None)
    job_id = prepare(
        command="mkdir -p out && exit 7",
        outputs=[{"path": "out", "s3": "s3://bucket/{job_id}"}],
    )
    assert run(job_id, deps(sync_preflight=False)) == 7
    state = jobs.read_state(job_id)
    assert (state.status, state.reason, state.exit_code) == ("failed", "exit 7", 7)
    assert state.problems == ["sync"]


def test_a_preempted_job_whose_final_upload_fails_still_comes_back(
    gpuc_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The upload failure is noted beside the reason, not folded into it: a
    `preempted+sync` reason was one the requeue did not recognise."""
    monkeypatch.setattr(destinations, "find_binary", lambda name, env=None: f"/fake/{name}")
    command_runner, _ = uploading(fail_dest="s3://bucket/")
    job_id = prepare(
        command="mkdir -p results && echo hi > results/a.txt && sleep 30",
        outputs=[{"path": "results", "s3": "s3://bucket/{job_id}"}],
    )
    queue.enqueue(make_spec(priority=1))  # something waiting, or preempt refuses
    written = paths.workdir(job_id) / "results/a.txt"

    def preempt_once_written() -> None:
        _wait_until(written.exists)
        queue.preempt(job_id)

    thread = threading.Thread(target=preempt_once_written, daemon=True)
    thread.start()
    assert run(job_id, deps(command_runner=command_runner, sync_preflight=False)) != 0
    thread.join(timeout=30)
    state = jobs.read_state(job_id)
    assert (state.status, state.attempt, state.intent) == ("queued", 2, None)
    assert "final sync FAILED" in log_of(job_id)
    assert "queued again as attempt 2" in log_of(job_id)


def test_max_runtime_kills_with_reason_timeout(gpuc_home: Path) -> None:
    job_id = prepare(command="sleep 60", max_runtime_min=0.02)
    start = time.monotonic()
    code = run(job_id)
    assert time.monotonic() - start < 20
    assert code != 0
    state = jobs.read_state(job_id)
    assert (state.status, state.reason) == ("failed", "timeout")


def test_utilization_is_sampled_in_main_only(gpuc_home: Path) -> None:
    """Setup is the hour spent downloading a checkpoint at 0% util, and a
    sample from there would show `gpuc status` an idle job that has not
    started its work yet. Nothing a setup phase does reaches `util_recent`."""
    job_id = prepare(gpus=[FAKE_GPUS[0]], setup="sleep 0.3", command="sleep 0.3")
    seen: list[tuple[str | None, int]] = []

    def idle(uuids: Sequence[str]) -> float:
        state = jobs.read_state(job_id)
        seen.append((state.phase, len(state.util_recent)))
        return 0.0

    assert run(job_id, deps(sampler=idle)) == 0
    state = jobs.read_state(job_id)
    assert (state.status, state.reason) == ("succeeded", None)

    assert seen, "the sampler was never called, so this proved nothing"
    assert {phase for phase, _ in seen} == {"main"}
    # The first sample of main found an empty history: setup recorded nothing.
    assert seen[0][1] == 0
    assert state.util_recent and set(state.util_recent) == {0.0}
    assert (state.util_sum, state.util_samples) == (0.0, len(state.util_recent))


def test_an_idle_gpu_is_reported_and_never_a_reason_to_kill(gpuc_home: Path) -> None:
    job_id = prepare(gpus=[FAKE_GPUS[0]], command="sleep 0.5")
    assert run(job_id, deps(sampler=lambda uuids: 0.0)) == 0
    state = jobs.read_state(job_id)
    assert state.status == "succeeded"
    assert state.util_recent and set(state.util_recent) == {0.0}


def test_cancel_kills_the_whole_process_group_including_grandchildren(
    gpuc_home: Path,
) -> None:
    # The grandchild records its pid in a file rather than the log, because the
    # log also contains the phase banner echoing this very command.
    job_id = prepare(command="bash -c 'sleep 300 & echo $! > gc.pid ; wait' & echo started; wait")
    pid_file = paths.workdir(job_id) / "gc.pid"
    result: dict[str, int] = {}

    def run_it() -> None:
        result["code"] = run(job_id)

    thread = threading.Thread(target=run_it, daemon=True)
    thread.start()
    try:
        _wait_until(lambda: pid_file.exists() and pid_file.read_text().strip().isdigit())
        grandchild = int(pid_file.read_text().strip())
        pgid = jobs.read_state(job_id).pgid
        assert pgid and os.getpgid(grandchild) == pgid

        queue.cancel(job_id)
        thread.join(timeout=30)
        assert not thread.is_alive()
        _wait_until(lambda: not _pid_exists(grandchild))
    finally:
        pgid = jobs.read_state(job_id).pgid
        if pgid:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(pgid, 9)

    state = jobs.read_state(job_id)
    assert (state.status, state.reason) == ("cancelled", "cancelled")
    assert result["code"] != 0


def _wait_until(predicate: Callable[[], bool], timeout: float = 30.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(0.05)
    raise AssertionError("condition never became true")


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def test_state_records_the_phase_and_pgid_while_running(gpuc_home: Path) -> None:
    job_id = prepare(command="sleep 5")
    observed: list[tuple[str | None, int | None]] = []

    def watcher() -> None:
        deadline = time.time() + 10
        while time.time() < deadline:
            state = jobs.read_state(job_id)
            if state.phase == "main" and state.pgid:
                observed.append((state.phase, state.pgid))
                queue.cancel(job_id)
                return
            time.sleep(0.05)

    thread = threading.Thread(target=watcher, daemon=True)
    thread.start()
    run(job_id)
    thread.join(timeout=10)
    assert observed and observed[0][0] == "main"


def test_the_pgid_goes_with_the_phase_that_owned_it(
    gpuc_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`pgid` names the phase now running and nothing else. Left set during
    the final sync it named a finished group, and the dispatcher's ladder,
    once its patience ran out, would SIGKILL whatever the kernel had reissued
    that number to."""
    during_sync: list[tuple[str | None, int | None, str | None]] = []

    def observe(_self: sync.SyncLoop) -> None:
        state = jobs.read_state(job_id)
        during_sync.append((state.phase, state.pgid, state.cgroup_unit))

    monkeypatch.setattr(sync.SyncLoop, "final", observe)
    job_id = prepare(command="true")
    assert run(job_id) == 0
    assert during_sync == [("sync", None, None)]


def test_runner_uses_the_s3_prefix_for_log_and_state(
    gpuc_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    jobs.write_config(HostConfig(host="h", gpus=list(FAKE_GPUS), s3_prefix="s3://b/gpuc/h"))
    monkeypatch.setattr(destinations, "find_binary", lambda name, env=None: f"/fake/{name}")
    calls: list[list[str]] = []

    def command_runner(
        argv: list[str], timeout: float | None = None, env: sync.Env = None
    ) -> sync.CommandResult:
        calls.append(argv)
        return sync.CommandResult(argv, 0, "")

    job_id = prepare(command="true")
    assert run(job_id, deps(command_runner=command_runner)) == 0
    targets = [argv[-2] for argv in calls]
    assert f"s3://b/gpuc/h/jobs/{job_id}/log.txt" in targets
    assert f"s3://b/gpuc/h/jobs/{job_id}/state.json" in targets


def test_build_env_exposes_job_paths(gpuc_home: Path) -> None:
    spec = make_spec(job_id="j1")
    jobs.write_state("j1", JobState())
    env = runner.build_env(spec, [FAKE_GPUS[0]])
    assert env["GPUC_JOB_ID"] == "j1"
    assert env["GPUC_EXPECTED_GPUS"] == "1"
    assert env["GPUC_OUTPUTS"] == str(paths.outputs_dir("j1"))


def run_detached(job_id: str, home: Path) -> subprocess.Popen[bytes]:
    """The runner as the dispatcher really starts it: its own session, so a
    signal to it is not also a signal to the test process. It asks the real
    `nvidia-smi` and runs the real preflight, so both are faked on the machine."""
    install_fake_nvidia_smi(home / "fake-bin")
    install_fake_torch(paths.workdir(job_id))
    env = dict(os.environ)
    env["GPUC_HOME"] = str(home)
    env["PATH"] = f"{home / 'fake-bin'}{os.pathsep}{env['PATH']}"
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "gpuc.host",
            "run",
            job_id,
            "--gpus",
            ",".join(ASSIGNED[job_id]),
            "--attempt",
            str(jobs.read_state(job_id).attempt),
        ],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def wait_for_job_pgid(job_id: str) -> int:
    _wait_until(lambda: jobs.read_state(job_id).pgid is not None)
    pgid = jobs.read_state(job_id).pgid
    assert pgid is not None
    return pgid


def test_sigterm_kills_the_job_group_and_writes_failed_terminated(gpuc_home: Path) -> None:
    job_id = prepare(command="sleep 300")
    proc = run_detached(job_id, gpuc_home)
    try:
        pgid = wait_for_job_pgid(job_id)
        proc.send_signal(signal.SIGTERM)
        assert proc.wait(timeout=60) == runner.TERMINATED_EXIT_CODE
        _wait_until(lambda: not procs.process_group_alive(pgid))
    finally:
        if proc.poll() is None:
            proc.kill()
    state = jobs.read_state(job_id)
    assert (state.status, state.reason, state.exit_code) == ("failed", "terminated", 143)
    assert state.ended_at and state.phase is None
    assert "received SIGTERM" in log_of(job_id)


def test_sigterm_after_a_cancel_request_ends_the_job_as_cancelled(gpuc_home: Path) -> None:
    job_id = prepare(command="sleep 300")
    proc = run_detached(job_id, gpuc_home)
    try:
        wait_for_job_pgid(job_id)
        queue.cancel(job_id)
        proc.send_signal(signal.SIGTERM)
        assert proc.wait(timeout=60) == runner.TERMINATED_EXIT_CODE
    finally:
        if proc.poll() is None:
            proc.kill()
    state = jobs.read_state(job_id)
    assert (state.status, state.reason) == ("cancelled", "cancelled")


def test_a_job_cancelled_before_its_first_phase_never_runs_its_command(
    gpuc_home: Path,
) -> None:
    job_id = prepare(command="touch RAN")
    code = run(job_id, deps(smi=stopping_after_claim(job_id, queue.cancel)))
    assert code == runner.TERMINATED_EXIT_CODE
    state = jobs.read_state(job_id)
    assert (state.status, state.reason, state.intent) == ("cancelled", "cancelled", None)
    assert not (paths.workdir(job_id) / "RAN").exists()
    assert "cancelled before phase=setup; not starting it" in log_of(job_id)


def test_a_job_stopped_before_main_skips_the_final_sync_and_is_never_no_outputs(
    gpuc_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cancel that lands before the first phase ends a job that produced
    nothing: an `outputs:` path it never had the chance to write is not a
    problem of the job's, and the state says its main never started."""
    monkeypatch.setattr(destinations, "find_binary", lambda name, env=None: f"/fake/{name}")
    uploads: list[list[str]] = []

    def recording(argv: list[str], timeout: float | None = None, env: sync.Env = None):
        uploads.append(argv)
        return sync.CommandResult(argv, 0, "")

    job_id = prepare(
        command="touch RAN", outputs=[{"path": "never-written", "s3": "s3://bucket/{job_id}"}]
    )
    deps_ = deps(
        smi=stopping_after_claim(job_id, queue.cancel),
        command_runner=recording,
        sync_preflight=False,
    )
    assert run(job_id, deps_) == runner.TERMINATED_EXIT_CODE
    state = jobs.read_state(job_id)
    assert (state.status, state.reason, state.problems, state.ran) == (
        "cancelled",
        "cancelled",
        [],
        False,
    )
    assert uploads == []
    assert "skipping the final output sync" in log_of(job_id)


def test_a_job_preempted_before_its_first_phase_goes_back_without_running_it(
    gpuc_home: Path,
) -> None:
    """The intent lands between phases as readily as mid-phase, and a job that
    is going back in the queue must not spend a single phase's work first."""
    job_id = prepare(command="touch RAN")
    queue.enqueue(make_spec(priority=1))  # something waiting, or preempt refuses
    code = run(job_id, deps(smi=stopping_after_claim(job_id, queue.preempt)))
    assert code == runner.TERMINATED_EXIT_CODE
    state = jobs.read_state(job_id)
    assert (state.status, state.attempt, state.intent) == ("queued", 2, None)
    assert not (paths.workdir(job_id) / "RAN").exists()
    assert "preempted before phase=setup; not starting it" in log_of(job_id)


def test_the_secrets_file_is_removed_once_the_job_has_finished(gpuc_home: Path) -> None:
    job_id = prepare(command='test -n "$HF_TOKEN"')
    paths.job_env_file(job_id).write_text('HF_TOKEN="hf_abc"\n')
    assert run(job_id) == 0
    assert not paths.job_env_file(job_id).exists()


def test_a_failed_utilization_sample_is_recorded_as_unknown_not_as_idle(
    gpuc_home: Path,
) -> None:
    calls: list[int] = []

    def flaky(uuids: Sequence[str]) -> float:
        calls.append(1)
        if len(calls) == 1:
            raise runner.gpus.GpuError("nvidia-smi reported utilization.gpu='[N/A]'")
        return 90.0

    job_id = prepare(gpus=[FAKE_GPUS[0]], command="sleep 0.6")
    assert run(job_id, deps(sampler=flaky)) == 0
    state = jobs.read_state(job_id)
    recent = state.util_recent
    assert recent and recent[0] is None
    # The mean is over the readings that exist: the failure is not a 0%.
    assert state.util_samples == len(recent) - 1 >= 1
    assert state.util_sum == 90.0 * state.util_samples
    assert "utilization sample failed" in log_of(job_id)


def test_the_generated_preflight_asks_torch_whether_the_card_is_there() -> None:
    """The command the injected `echo` below stands in for.

    It runs under `uv run --no-sync` because the job's workdir is a uv project
    and the repo's own venv has no torch, and it has to actually *count*
    devices: importing torch proves nothing about a host whose driver is gone.
    """
    command = runner.preflight_command(make_spec())
    assert command.startswith("uv run --no-sync python -c ")
    assert "import os, sys, torch" in command
    assert "torch.cuda.device_count()" in command
    assert 'os.environ["GPUC_EXPECTED_GPUS"]' in command
    assert "device_count()=={count}, expected {expected}" in command


def test_the_jobs_python_is_the_interpreter_the_preflight_runs_under() -> None:
    command = runner.preflight_command(make_spec(python=".venv/bin/python"))
    assert command.startswith(".venv/bin/python -c ")
    assert "torch.cuda.device_count()" in command


def test_a_job_that_names_its_python_runs_its_preflight_through_it(gpuc_home: Path) -> None:
    job_id = prepare(gpus=[FAKE_GPUS[0]], command="true", python="echo VIA-THE-JOBS-PYTHON")
    assert run(job_id, deps(preflight=True)) == 0
    assert "phase=preflight: echo VIA-THE-JOBS-PYTHON -c " in log_of(job_id)


def test_preflight_is_a_real_phase(gpuc_home: Path) -> None:
    job_id = prepare(gpus=[FAKE_GPUS[0]], command="true")
    assert (
        run(job_id, deps(preflight=True, preflight_command=lambda spec: "echo gpu preflight ok"))
        == 0
    )
    log = log_of(job_id)
    assert "phase=preflight: echo gpu preflight ok" in log
    assert log.index("phase=preflight") < log.index("phase=main")


def test_a_missing_output_dir_fails_the_job_as_no_outputs_not_as_sync(
    gpuc_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(destinations, "find_binary", lambda name, env=None: f"/fake/{name}")
    monkeypatch.setattr(
        sync, "run_command", lambda argv, timeout=None, env=None: sync.CommandResult(argv, 0, "")
    )
    job_id = prepare(
        command="true", outputs=[{"path": "never-written", "s3": "s3://bucket/{job_id}"}]
    )
    assert run(job_id, deps(sync_preflight=False)) == 1
    state = jobs.read_state(job_id)
    assert (state.status, state.reason) == ("failed", "no-outputs")


# -- PATH, secrets and the sync environment ------------------------------------


def test_path_with_user_bins_skips_missing_dirs_and_never_duplicates(
    tmp_path: Path,
) -> None:
    (tmp_path / ".local/bin").mkdir(parents=True)
    environ = {"HOME": str(tmp_path), "PATH": f"{tmp_path / '.local/bin'}:/usr/bin"}
    assert paths.path_with_user_bins(environ) == f"{tmp_path / '.local/bin'}:/usr/bin"
    assert paths.path_with_user_bins({"HOME": str(tmp_path), "PATH": "/usr/bin"}) == (
        f"{tmp_path / '.local/bin'}:/usr/bin"
    )


def test_a_jobs_secrets_reach_the_sync_loop(
    gpuc_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`secrets: [AWS_ACCESS_KEY_ID]` must be enough: no ~/.aws on the host."""
    monkeypatch.setattr(destinations, "find_binary", lambda name, env=None: f"/fake/{name}")
    jobs.write_config(HostConfig(host="h", gpus=[], s3_prefix="s3://b/gpuc/h"))
    seen: list[sync.Env] = []

    def command_runner(
        argv: list[str], timeout: float | None = None, env: sync.Env = None
    ) -> sync.CommandResult:
        seen.append(env)
        return sync.CommandResult(argv, 0, "")

    job_id = prepare(
        command="mkdir -p results && echo hi > results/a.txt",
        outputs=[{"path": "results", "s3": "s3://bucket/{job_id}"}],
        secrets=["AWS_ACCESS_KEY_ID"],
    )
    paths.job_env_file(job_id).write_text("AWS_ACCESS_KEY_ID=AKIAFROMTHEJOB\n")
    assert run(job_id, deps(command_runner=command_runner)) == 0
    assert seen, "the sync loop never ran a command"
    assert all(env is not None and env["AWS_ACCESS_KEY_ID"] == "AKIAFROMTHEJOB" for env in seen)


def test_the_secrets_file_outlives_the_job_until_the_final_sync_is_done(
    gpuc_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unlinking it before the final sync would break exactly the upload that
    matters most: the one carrying the finished run's outputs."""
    monkeypatch.setattr(destinations, "find_binary", lambda name, env=None: f"/fake/{name}")
    jobs.write_config(HostConfig(host="h", gpus=[], s3_prefix="s3://b/gpuc/h"))
    present_during_sync: list[bool] = []

    def command_runner(
        argv: list[str], timeout: float | None = None, env: sync.Env = None
    ) -> sync.CommandResult:
        present_during_sync.append(paths.job_env_file(job_id).exists())
        return sync.CommandResult(argv, 0, "")

    job_id = prepare(
        command="mkdir -p results && echo hi > results/a.txt",
        outputs=[{"path": "results", "s3": "s3://bucket/{job_id}"}],
        secrets=["AWS_ACCESS_KEY_ID"],
    )
    paths.job_env_file(job_id).write_text("AWS_ACCESS_KEY_ID=AKIAFROMTHEJOB\n")
    assert run(job_id, deps(command_runner=command_runner)) == 0
    assert all(present_during_sync)
    assert not paths.job_env_file(job_id).exists()


def test_no_secret_value_is_ever_written_to_a_log(gpuc_home: Path) -> None:
    secret = "gpuc-canary-must-not-appear-0123456789"
    job_id = prepare(command="echo the job ran", env={"HARMLESS": "1"}, secrets=["WANDB_API_KEY"])
    paths.job_env_file(job_id).write_text(f"WANDB_API_KEY={secret}\n")
    assert run(job_id) == 0
    assert secret not in log_of(job_id)
    assert secret not in paths.state_file(job_id).read_text()
    dispatcher_log = paths.dispatcher_log()
    if dispatcher_log.exists():
        assert secret not in dispatcher_log.read_text()


# -- the backup record the purge depends on -----------------------------------


def uploading(
    ok: bool = True, fail_dest: str | None = None
) -> tuple[sync.CommandRunner, list[list[str]]]:
    calls: list[list[str]] = []

    def command_runner(
        argv: list[str], timeout: float | None = None, env: sync.Env = None
    ) -> sync.CommandResult:
        calls.append(argv)
        failed = not ok or (fail_dest is not None and fail_dest in " ".join(argv))
        return sync.CommandResult(argv, 1 if failed else 0, "AccessDenied" if failed else "")

    return command_runner, calls


def test_a_successful_final_sync_records_the_backup(
    gpuc_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(destinations, "find_binary", lambda name, env=None: f"/fake/{name}")
    jobs.write_config(HostConfig(host="h", gpus=[], s3_prefix="s3://b/gpuc/h"))
    command_runner, _ = uploading()
    job_id = prepare(command="true")
    assert run(job_id, deps(command_runner=command_runner)) == 0
    state = jobs.read_state(job_id)
    assert state.mirrored
    assert state.mirror is not None and state.mirror.to == f"s3://b/gpuc/h/jobs/{job_id}"


def test_a_failed_final_meta_sync_records_no_backup(
    gpuc_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(destinations, "find_binary", lambda name, env=None: f"/fake/{name}")
    jobs.write_config(HostConfig(host="h", gpus=[], s3_prefix="s3://b/gpuc/h"))
    command_runner, _ = uploading(ok=False)
    job_id = prepare(command="true")
    run(job_id, deps(command_runner=command_runner))
    state = jobs.read_state(job_id)
    assert not state.mirrored
    assert state.mirror is not None and state.mirror.ok_at is None and state.mirror.error
    assert "final state upload failed" in log_of(job_id)


def test_a_host_with_no_prefix_records_no_backup(gpuc_home: Path) -> None:
    jobs.write_config(HostConfig(host="h", gpus=[]))
    job_id = prepare(command="true")
    assert run(job_id) == 0
    state = jobs.read_state(job_id)
    assert state.uploads == []
    assert not state.mirrored


def test_confirmed_outputs_are_recorded(gpuc_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(destinations, "find_binary", lambda name, env=None: f"/fake/{name}")
    jobs.write_config(HostConfig(host="h", gpus=[], s3_prefix="s3://b/gpuc/h"))
    command_runner, _ = uploading()
    job_id = prepare(
        command="mkdir -p results && echo hi > results/a.txt",
        outputs=[{"path": "results", "s3": "s3://bucket/{job_id}"}],
    )
    assert run(job_id, deps(command_runner=command_runner)) == 0
    assert jobs.read_state(job_id).outputs_uploaded(jobs.read_spec(job_id))


def test_an_output_upload_that_fails_leaves_outputs_unconfirmed(
    gpuc_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(destinations, "find_binary", lambda name, env=None: f"/fake/{name}")
    jobs.write_config(HostConfig(host="h", gpus=[], s3_prefix="s3://b/gpuc/h"))
    command_runner, _ = uploading(fail_dest="s3://bucket/")
    job_id = prepare(
        command="mkdir -p results && echo hi > results/a.txt",
        outputs=[{"path": "results", "s3": "s3://bucket/{job_id}"}],
    )
    run(job_id, deps(command_runner=command_runner, sync_preflight=False))
    state = jobs.read_state(job_id)
    assert not state.outputs_uploaded(jobs.read_spec(job_id))
    assert state.upload_errors() and "AccessDenied" in state.upload_errors()[0]
    assert (state.status, state.reason) == ("failed", "sync")
    # The log and state still made it, so the record itself is backed up.
    assert state.mirrored


def test_an_ephemeral_host_keeps_the_secrets_file_for_the_drain(
    gpuc_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(destinations, "find_binary", lambda name, env=None: f"/fake/{name}")
    jobs.write_config(
        HostConfig(
            host="pod",
            gpus=[],
            s3_prefix="s3://b/gpuc/pod",
            provider={"kind": "runpod", "pod_id": "p1"},
        )
    )
    command_runner, _ = uploading(fail_dest="s3://bucket/")
    job_id = prepare(
        command="mkdir -p results && echo hi > results/a.txt",
        outputs=[{"path": "results", "s3": "s3://bucket/{job_id}"}],
        secrets=["AWS_ACCESS_KEY_ID"],
    )
    paths.job_env_file(job_id).write_text("AWS_ACCESS_KEY_ID=AKIA\n")
    # Past the sync preflight: a job that fails *that* never runs, and the
    # drain skips it, so it is not the shape this is about.
    run(job_id, deps(command_runner=command_runner, sync_preflight=False))
    assert jobs.read_state(job_id).reason == "sync"
    assert paths.job_env_file(job_id).exists()


def test_a_job_that_produced_nothing_does_not_keep_its_secrets_file(
    gpuc_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The drain only retries jobs that are holding something. Keeping a job's
    credentials on disk for a retry that will never come is pure exposure."""
    monkeypatch.setattr(destinations, "find_binary", lambda name, env=None: f"/fake/{name}")
    jobs.write_config(
        HostConfig(
            host="pod",
            gpus=[],
            s3_prefix="s3://b/gpuc/pod",
            provider={"kind": "runpod", "pod_id": "p1"},
        )
    )
    command_runner, _ = uploading(fail_dest="s3://bucket/")
    job_id = prepare(
        command="echo nothing-under-results",
        outputs=[{"path": "results", "s3": "s3://bucket/{job_id}"}],
        secrets=["AWS_ACCESS_KEY_ID"],
    )
    paths.job_env_file(job_id).write_text("AWS_ACCESS_KEY_ID=AKIA\n")
    run(job_id, deps(command_runner=command_runner, sync_preflight=False))
    assert not paths.job_env_file(job_id).exists()


def test_a_shared_host_still_removes_the_secrets_file(
    gpuc_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(destinations, "find_binary", lambda name, env=None: f"/fake/{name}")
    jobs.write_config(HostConfig(host="h", gpus=[], s3_prefix="s3://b/gpuc/h"))
    command_runner, _ = uploading(fail_dest="s3://bucket/")
    job_id = prepare(
        command="mkdir -p results && echo hi > results/a.txt",
        outputs=[{"path": "results", "s3": "s3://bucket/{job_id}"}],
        secrets=["AWS_ACCESS_KEY_ID"],
    )
    paths.job_env_file(job_id).write_text("AWS_ACCESS_KEY_ID=AKIA\n")
    run(job_id, deps(command_runner=command_runner))
    assert not paths.job_env_file(job_id).exists()


def test_a_sigterm_during_the_final_sync_does_not_rerun_finalize(
    gpuc_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The dispatcher escalates a cancel to the runner after 30 s, and that
    lands in the middle of a long final upload. It used to unwind `_finalize`,
    which `run()` caught and finalized again: a job already recorded as
    `succeeded` was rewritten as `failed: terminated` and every output was
    uploaded a second time.
    """
    job_id = prepare(command="true")
    calls: list[str] = []

    def final(_self: sync.SyncLoop) -> None:
        calls.append("final")
        os.kill(os.getpid(), signal.SIGTERM)
        time.sleep(0.05)

    monkeypatch.setattr(sync.SyncLoop, "final", final)
    assert run(job_id) == 0
    state = jobs.read_state(job_id)
    assert (state.status, state.reason, state.exit_code) == ("succeeded", None, 0)
    assert calls == ["final"]


def test_a_preempted_job_keeps_its_secrets_for_the_next_attempt(gpuc_home: Path) -> None:
    """Nothing delivers them a second time: the next attempt is this same job
    id, dispatched by the host, and the secrets only ever arrive at submit."""
    job_id = prepare(command="sleep 30")
    paths.job_env_file(job_id).write_text('HF_TOKEN="hf_abc"\n')
    queue.enqueue(make_spec(priority=1))  # something waiting, or preempt refuses
    assert run(job_id, deps(smi=stopping_after_claim(job_id, queue.preempt))) != 0
    assert jobs.read_state(job_id).status == "queued"
    assert paths.job_env_file(job_id).exists()
    assert "keeping this job's secrets file for the next attempt" in log_of(job_id)


# -- the runner owns every transition -----------------------------------------


def watch_statuses(job_id: str, seen: list[tuple[str, int]], stop: threading.Event) -> None:
    """Append every distinct (status, attempt) a poller of the state sees,
    with one more look after `stop` so the last write is never missed."""
    while True:
        state = jobs.read_state(job_id)
        if not seen or seen[-1] != (state.status, state.attempt):
            seen.append((state.status, state.attempt))
        if stop.is_set():
            return
        time.sleep(0.005)


def test_a_preempted_job_goes_straight_from_running_to_queued(gpuc_home: Path) -> None:
    """No terminal state in between: `status` polled throughout never sees the
    attempt finished, only running at attempt 1 and then queued at attempt 2."""
    job_id = prepare(command="sleep 30")
    queue.enqueue(make_spec(priority=1))  # something waiting, or preempt refuses
    stop = threading.Event()
    seen: list[tuple[str, int]] = []
    watcher = threading.Thread(target=watch_statuses, args=(job_id, seen, stop))
    watcher.start()

    def preempt_once_seen_running(wanted: str) -> None:
        _wait_until(lambda: ("running", 1) in seen)
        queue.preempt(wanted)

    try:
        assert run(job_id, deps(smi=stopping_after_claim(job_id, preempt_once_seen_running))) != 0
    finally:
        stop.set()
        watcher.join(timeout=10)
    assert seen[-2:] == [("running", 1), ("queued", 2)], seen
    assert all(status in ("queued", "running") for status, _ in seen)
    state = jobs.read_state(job_id)
    assert (state.status, state.attempt, state.intent, state.ended_at) == ("queued", 2, None, None)


def test_the_status_stays_running_until_the_mirror_has_been_written(
    gpuc_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A job is finished exactly when its runner has nothing left to do: the
    final sync, the workdir and the mirror all happen under `running` with
    `phase=sync`, and the terminal write comes after them. So a dispatcher
    escalating a stop keys its patience on the phase, and nothing has to
    guess whether the process behind a finished state is still cleaning up."""
    monkeypatch.setattr(destinations, "find_binary", lambda name, env=None: f"/fake/{name}")
    jobs.write_config(HostConfig(host="h", gpus=[], s3_prefix="s3://b/gpuc/h"))
    at_mirror: list[tuple[str, str | None, int | None]] = []
    command_runner, calls = uploading()

    def recording(argv: list[str], timeout: float | None = None, env: sync.Env = None):
        if argv[-2].endswith((f"/jobs/{job_id}/log.txt", f"/jobs/{job_id}/state.json")):
            state = jobs.read_state(job_id)
            at_mirror.append((state.status, state.phase, state.workdir_bytes))
        return command_runner(argv, timeout, env)

    job_id = prepare(command="true")
    assert run(job_id, deps(command_runner=recording)) == 0
    # The log and state go up under `running`/`sync` with the workdir already
    # measured; the one upload after that is the state, saying how it ended.
    assert [entry[:2] for entry in at_mirror] == [("running", "sync")] * 2 + [("succeeded", None)]
    assert all(entry[2] is not None for entry in at_mirror), "measured before the mirror"
    state = jobs.read_state(job_id)
    assert (state.status, state.phase, state.intent) == ("succeeded", None, None)
    assert state.mirrored
    # The mirror's own state.json is put once more after the terminal write,
    # so the copy that survives this host says how the job ended.
    state_puts = [argv for argv in calls if argv[-2].endswith("/state.json")]
    log_puts = [argv for argv in calls if argv[-2].endswith("/log.txt")]
    assert (len(state_puts), len(log_puts)) == (2, 1)


def preempt_in_main(job_id: str) -> None:
    """Preempt the job once its `main` phase is running, from a thread, so
    the attempt has something to stop and a final sync to run afterwards."""

    def once_in_main() -> None:
        _wait_until(lambda: jobs.read_state(job_id).phase == "main")
        queue.preempt(job_id)

    threading.Thread(target=once_in_main, daemon=True).start()


def test_a_cancel_that_lands_while_a_preempt_is_stopping_wins(
    gpuc_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The later request wins: somebody who cancels a job that is already
    stopping wants it over, not started again."""
    job_id = prepare(command="sleep 30")
    queue.enqueue(make_spec(priority=1))  # something waiting, or preempt refuses

    def cancel_during_final_sync(_self: sync.SyncLoop) -> None:
        assert queue.cancel(job_id) == "cancelling"

    monkeypatch.setattr(sync.SyncLoop, "final", cancel_during_final_sync)
    assert run(job_id, deps(smi=stopping_after_claim(job_id, preempt_in_main))) != 0
    state = jobs.read_state(job_id)
    assert (state.status, state.reason, state.attempt, state.intent) == (
        "cancelled",
        "cancelled",
        1,
        None,
    )
    assert not paths.job_env_file(job_id).exists()


def test_a_preempt_that_lands_in_the_final_sync_changes_nothing(
    gpuc_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A job that ended for a reason of its own before the kill landed asked
    for nothing: re-running it would be a retry nobody requested."""
    job_id = prepare(command="true")
    queue.enqueue(make_spec(priority=1))  # something waiting, or preempt refuses

    def preempt_during_final_sync(_self: sync.SyncLoop) -> None:
        assert queue.preempt(job_id) == "preempting"

    monkeypatch.setattr(sync.SyncLoop, "final", preempt_during_final_sync)
    assert run(job_id) == 0
    state = jobs.read_state(job_id)
    assert (state.status, state.reason, state.attempt, state.intent) == ("succeeded", None, 1, None)


def test_a_sigterm_to_a_preempting_runner_still_queues_the_job_again(gpuc_home: Path) -> None:
    """The dispatcher's ladder reaches the runner itself while it is stopping
    the job for a preempt; that must not turn the preempt into a failure."""
    job_id = prepare(command="sleep 300")
    queue.enqueue(make_spec(priority=1))  # something waiting, or preempt refuses
    proc = run_detached(job_id, gpuc_home)
    try:
        wait_for_job_pgid(job_id)
        assert queue.preempt(job_id) == "preempting"
        proc.send_signal(signal.SIGTERM)
        assert proc.wait(timeout=60) == runner.TERMINATED_EXIT_CODE
    finally:
        if proc.poll() is None:
            proc.kill()
    state = jobs.read_state(job_id)
    assert (state.status, state.attempt, state.intent) == ("queued", 2, None)
    assert "queued again as attempt 2" in log_of(job_id)


def test_a_cancel_overriding_a_preempt_before_main_keeps_that_main_never_started(
    gpuc_home: Path,
) -> None:
    """The later request wins and the job ends `cancelled`; what it must not
    do on the way is claim `main` ran, or a pod's drain would hold the
    checkout's files under `outputs:` as this job's results."""
    job_id = prepare(command="touch RAN", outputs=[{"path": "out", "s3": "s3://b/{job_id}"}])
    queue.enqueue(make_spec(priority=1))  # something waiting, or preempt refuses

    def preempt_then_cancel(job: str) -> None:
        queue.preempt(job)
        queue.cancel(job)

    code = run(job_id, deps(smi=stopping_after_claim(job_id, preempt_then_cancel)))
    assert code == runner.TERMINATED_EXIT_CODE
    state = jobs.read_state(job_id)
    assert (state.status, state.reason, state.ran) == ("cancelled", "cancelled", False)
    assert not (paths.workdir(job_id) / "RAN").exists()
