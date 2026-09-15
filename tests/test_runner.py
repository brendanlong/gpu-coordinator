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

from gpuc.host import jobs, paths, queue, runner, sync
from gpuc.host.jobs import HostConfig, JobState
from gpuc.host.runner import RunnerDeps
from tests.conftest import FAKE_GPUS, fake_smi, make_spec


def prepare(gpus: Sequence[str] = (), **overrides: object) -> str:
    spec = make_spec(**overrides)
    job_id = queue.enqueue(spec)
    queue.remove_marker(job_id)
    jobs.update_state(job_id, status="running", gpus=list(gpus))
    return job_id


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
    assert runner.run_job(job_id, deps()) == 42
    state = jobs.read_state(job_id)
    assert (state.status, state.exit_code, state.reason) == ("failed", 42, "exit 42")
    assert state.ended_at and state.pid is None


def test_success_writes_succeeded_and_captures_output(gpuc_home: Path) -> None:
    job_id = prepare(command="echo hello-from-job; echo to-stderr >&2")
    assert runner.run_job(job_id, deps()) == 0
    state = jobs.read_state(job_id)
    assert (state.status, state.exit_code, state.reason) == ("succeeded", 0, None)
    assert "hello-from-job" in log_of(job_id)
    assert "to-stderr" in log_of(job_id)


def test_setup_failure_short_circuits_the_command(gpuc_home: Path) -> None:
    job_id = prepare(setup="exit 3", command="echo SHOULD-NOT-RUN")
    assert runner.run_job(job_id, deps()) == 3
    state = jobs.read_state(job_id)
    assert (state.status, state.reason) == ("failed", "setup")
    assert "SHOULD-NOT-RUN" not in log_of(job_id)


def test_setup_uses_pipefail(gpuc_home: Path) -> None:
    job_id = prepare(setup="false | cat", command="true")
    assert runner.run_job(job_id, deps()) != 0
    assert jobs.read_state(job_id).reason == "setup"


def test_cuda_visible_devices_is_the_assigned_uuids(gpuc_home: Path) -> None:
    job_id = prepare(gpus=FAKE_GPUS, command='echo "CVD=$CUDA_VISIBLE_DEVICES"')
    assert runner.run_job(job_id, deps()) == 0
    assert f"CVD={FAKE_GPUS[0]},{FAKE_GPUS[1]}" in log_of(job_id)


def test_zero_gpu_job_gets_an_empty_cuda_visible_devices(gpuc_home: Path) -> None:
    job_id = prepare(command='echo "CVD=[$CUDA_VISIBLE_DEVICES]"')
    assert runner.run_job(job_id, deps()) == 0
    assert "CVD=[]" in log_of(job_id)


def test_spec_env_cannot_override_the_gpu_assignment(gpuc_home: Path) -> None:
    job_id = prepare(
        gpus=[FAKE_GPUS[0]],
        env={"CUDA_VISIBLE_DEVICES": "0,1,2,3", "MY_VAR": "set"},
        command='echo "CVD=$CUDA_VISIBLE_DEVICES MY_VAR=$MY_VAR"',
    )
    assert runner.run_job(job_id, deps()) == 0
    assert f"CVD={FAKE_GPUS[0]} MY_VAR=set" in log_of(job_id)


def test_secrets_file_is_sourced_into_the_job(gpuc_home: Path) -> None:
    job_id = prepare(command='echo "TOKEN=$HF_TOKEN"')
    paths.job_env_file(job_id).write_text('export HF_TOKEN="hf_abc"\n')
    assert runner.run_job(job_id, deps()) == 0
    assert "TOKEN=hf_abc" in log_of(job_id)


def test_a_stale_assigned_uuid_fails_the_job_before_it_starts(gpuc_home: Path) -> None:
    job_id = prepare(gpus=["GPU-stale"], command="echo SHOULD-NOT-RUN")
    assert runner.run_job(job_id, deps()) == 1
    state = jobs.read_state(job_id)
    assert (state.status, state.reason) == ("failed", "gpu-assert")
    assert "SHOULD-NOT-RUN" not in log_of(job_id)


def test_failed_final_sync_turns_a_succeeded_job_into_failed_sync(
    gpuc_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sync, "aws_binary", lambda: None)
    job_id = prepare(
        command="mkdir -p out && echo x > out/x",
        outputs=[{"path": "out", "s3": "s3://bucket/{job_id}"}],
    )
    assert runner.run_job(job_id, deps()) == 1
    state = jobs.read_state(job_id)
    assert (state.status, state.reason, state.exit_code) == ("failed", "sync", 1)
    assert "final sync FAILED" in log_of(job_id)


def test_failed_final_sync_does_not_mask_a_failed_job(
    gpuc_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sync, "aws_binary", lambda: None)
    job_id = prepare(
        command="mkdir -p out && exit 7",
        outputs=[{"path": "out", "s3": "s3://bucket/{job_id}"}],
    )
    assert runner.run_job(job_id, deps()) == 7
    state = jobs.read_state(job_id)
    assert state.status == "failed"
    assert state.reason == "exit 7+sync"
    assert state.exit_code == 7


def test_max_runtime_kills_with_reason_timeout(gpuc_home: Path) -> None:
    job_id = prepare(command="sleep 60", max_runtime_min=0.02)
    start = time.monotonic()
    code = runner.run_job(job_id, deps())
    assert time.monotonic() - start < 20
    assert code != 0
    state = jobs.read_state(job_id)
    assert (state.status, state.reason) == ("failed", "timeout")


def test_low_util_watchdog_kills_only_after_grace_and_window(gpuc_home: Path) -> None:
    samples: list[float] = []

    def sampler(uuids: Sequence[str]) -> float:
        samples.append(time.monotonic())
        return 1.0

    job_id = prepare(
        gpus=[FAKE_GPUS[0]],
        command="sleep 60",
        low_util={"enabled": True, "window_min": 0.005, "floor_pct": 5, "grace_min": 0.002},
    )
    code = runner.run_job(job_id, deps(sampler=sampler))
    assert code != 0
    state = jobs.read_state(job_id)
    assert (state.status, state.reason) == ("failed", "low-util")
    assert len(samples) >= 2
    assert "low-util watchdog" in log_of(job_id)


def test_busy_gpu_is_never_killed_by_the_watchdog(gpuc_home: Path) -> None:
    job_id = prepare(
        gpus=[FAKE_GPUS[0]],
        command="sleep 0.5",
        low_util={"enabled": True, "window_min": 0.002, "floor_pct": 5, "grace_min": 0.0},
    )
    assert runner.run_job(job_id, deps(sampler=lambda uuids: 97.0)) == 0
    assert jobs.read_state(job_id).status == "succeeded"


def test_watchdog_can_be_disabled_per_job(gpuc_home: Path) -> None:
    job_id = prepare(
        gpus=[FAKE_GPUS[0]],
        command="sleep 0.5",
        low_util={"enabled": False, "window_min": 0.002, "floor_pct": 5, "grace_min": 0.0},
    )
    assert runner.run_job(job_id, deps(sampler=lambda uuids: 0.0)) == 0


def test_watchdog_never_runs_for_a_zero_gpu_job(gpuc_home: Path) -> None:
    def explode(uuids: Sequence[str]) -> float:
        raise AssertionError("sampled utilization for a gpus:0 job")

    job_id = prepare(
        command="sleep 0.4",
        low_util={"enabled": True, "window_min": 0.001, "floor_pct": 99, "grace_min": 0.0},
    )
    assert runner.run_job(job_id, deps(sampler=explode)) == 0


def test_cancel_kills_the_whole_process_group_including_grandchildren(
    gpuc_home: Path,
) -> None:
    # The grandchild records its pid in a file rather than the log, because the
    # log also contains the phase banner echoing this very command.
    job_id = prepare(command="bash -c 'sleep 300 & echo $! > gc.pid ; wait' & echo started; wait")
    pid_file = paths.workdir(job_id) / "gc.pid"
    result: dict[str, int] = {}

    def run() -> None:
        result["code"] = runner.run_job(job_id, deps())

    thread = threading.Thread(target=run, daemon=True)
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


def test_kill_process_group_escalates_to_sigkill(tmp_path: Path) -> None:
    ready = tmp_path / "ready"
    proc = subprocess.Popen(
        ["bash", "-c", f"trap '' TERM; touch {ready}; sleep 60"], start_new_session=True
    )
    try:
        _wait_until(ready.exists, timeout=10)
        start = time.monotonic()
        runner.kill_process_group(proc.pid, grace_s=1.0, reap=proc.poll)
        assert proc.wait(timeout=10) == -9
        assert 1.0 <= time.monotonic() - start < 10
    finally:
        if proc.poll() is None:
            proc.kill()


def test_kill_process_group_ignores_a_dead_group() -> None:
    runner.kill_process_group(0)
    runner.kill_process_group(2**30)


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
    runner.run_job(job_id, deps())
    thread.join(timeout=10)
    assert observed and observed[0][0] == "main"


def test_runner_uses_the_s3_prefix_for_log_and_state(
    gpuc_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    jobs.write_config(HostConfig(host="h", gpus=list(FAKE_GPUS), s3_prefix="s3://b/gpuc/h"))
    monkeypatch.setattr(sync, "aws_binary", lambda: "/fake/aws")
    calls: list[list[str]] = []

    def command_runner(argv: list[str], timeout: float) -> sync.CommandResult:
        calls.append(argv)
        return sync.CommandResult(argv, 0, "")

    job_id = prepare(command="true")
    assert runner.run_job(job_id, deps(command_runner=command_runner)) == 0
    destinations = [argv[-2] for argv in calls]
    assert f"s3://b/gpuc/h/jobs/{job_id}/log.txt" in destinations
    assert f"s3://b/gpuc/h/jobs/{job_id}/state.json" in destinations


def test_runner_main_requires_a_job_id(gpuc_home: Path) -> None:
    assert runner.main([]) == 2


def test_window_needs_a_full_window_before_it_fires() -> None:
    window = runner._Window(window_s=10.0)
    window.add(0.0, 1.0)
    assert not window.full(0.0)
    window.add(5.0, 1.0)
    assert not window.full(5.0)
    window.add(10.0, 1.0)
    assert window.full(10.0)
    assert window.mean() == 1.0


def test_build_env_exposes_job_paths(gpuc_home: Path) -> None:
    spec = make_spec(job_id="j1")
    jobs.write_state("j1", JobState())
    env = runner.build_env(spec, [FAKE_GPUS[0]])
    assert env["GPUC_JOB_ID"] == "j1"
    assert env["GPUC_EXPECTED_GPUS"] == "1"
    assert env["GPUC_OUTPUTS"] == str(paths.outputs_dir("j1"))


def run_detached(job_id: str, home: Path) -> subprocess.Popen[bytes]:
    """The runner as the dispatcher really starts it: its own session, so a
    signal to it is not also a signal to the test process."""
    env = dict(os.environ)
    env["GPUC_HOME"] = str(home)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    return subprocess.Popen(
        [sys.executable, "-m", "gpuc.host", "run", job_id],
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
        _wait_until(lambda: not runner.process_group_alive(pgid))
    finally:
        if proc.poll() is None:
            proc.kill()
    state = jobs.read_state(job_id)
    assert (state.status, state.reason, state.exit_code) == ("failed", "terminated", 143)
    assert state.ended_at and state.phase is None
    assert "received SIGTERM" in log_of(job_id)


def test_sigterm_after_a_cancel_marker_ends_the_job_as_cancelled(gpuc_home: Path) -> None:
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


def test_a_job_cancelled_during_the_launch_window_never_runs_its_command(
    gpuc_home: Path,
) -> None:
    job_id = prepare(command="touch RAN")
    queue.cancel(job_id)
    assert runner.run_job(job_id, deps()) == runner.TERMINATED_EXIT_CODE
    state = jobs.read_state(job_id)
    assert (state.status, state.reason) == ("cancelled", "cancelled")
    assert not (paths.workdir(job_id) / "RAN").exists()
    assert "cancel marker present before phase=setup" in log_of(job_id)


def test_the_secrets_file_is_removed_once_the_job_has_finished(gpuc_home: Path) -> None:
    job_id = prepare(command='test -n "$HF_TOKEN"')
    paths.job_env_file(job_id).write_text('HF_TOKEN="hf_abc"\n')
    assert runner.run_job(job_id, deps()) == 0
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

    job_id = prepare(
        gpus=[FAKE_GPUS[0]],
        command="sleep 0.6",
        low_util={"enabled": True, "window_min": 0.001, "floor_pct": 50, "grace_min": 0.0},
    )
    assert runner.run_job(job_id, deps(sampler=flaky)) == 0
    recent = jobs.read_state(job_id).util_recent
    assert recent and recent[0] is None
    assert "utilization sample failed" in log_of(job_id)


def test_preflight_is_a_real_phase(gpuc_home: Path) -> None:
    assert "preflight" in jobs.PHASES
    job_id = prepare(gpus=[FAKE_GPUS[0]], command="true")
    assert (
        runner.run_job(
            job_id,
            deps(preflight=True, preflight_command=lambda: "echo gpu preflight ok"),
        )
        == 0
    )
    log = log_of(job_id)
    assert "phase=preflight: echo gpu preflight ok" in log
    assert log.index("phase=preflight") < log.index("phase=main")


def test_a_missing_output_dir_fails_the_job_as_no_outputs_not_as_sync(
    gpuc_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sync, "aws_binary", lambda: "/fake/aws")
    monkeypatch.setattr(
        sync, "run_command", lambda argv, timeout=None: sync.CommandResult(argv, 0, "")
    )
    job_id = prepare(
        command="true", outputs=[{"path": "never-written", "s3": "s3://bucket/{job_id}"}]
    )
    assert runner.run_job(job_id, deps()) == 1
    state = jobs.read_state(job_id)
    assert (state.status, state.reason) == ("failed", "no-outputs")
