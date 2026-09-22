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
from gpuc.host.jobs import HostConfig, JobState
from gpuc.host.runner import RunnerDeps
from tests.conftest import (
    FAKE_GPUS,
    fake_smi,
    install_fake_nvidia_smi,
    install_fake_torch,
    make_spec,
)


def prepare(gpus: Sequence[str] = (FAKE_GPUS[0],), **overrides: object) -> str:
    spec = make_spec(**overrides)
    job_id = queue.enqueue(spec)
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


def test_cuda_visible_devices_is_the_nvidia_smi_indices_of_the_assigned_cards(
    gpuc_home: Path,
) -> None:
    job_id = prepare(
        gpus=FAKE_GPUS, command='echo "CVD=$CUDA_VISIBLE_DEVICES ORDER=$CUDA_DEVICE_ORDER"'
    )
    assert runner.run_job(job_id, deps()) == 0
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
    assert runner.run_job(job_id, deps()) == 0
    assert "CVD=0 MY_VAR=set" in log_of(job_id)


def test_secrets_file_is_sourced_into_the_job(gpuc_home: Path) -> None:
    job_id = prepare(command='echo "TOKEN=$HF_TOKEN"')
    paths.job_env_file(job_id).write_text('export HF_TOKEN="hf_abc"\n')
    assert runner.run_job(job_id, deps()) == 0
    assert "TOKEN=hf_abc" in log_of(job_id)


def test_a_job_assigned_gpus_by_index_runs_on_the_card_that_index_names(
    gpuc_home: Path,
) -> None:
    """A position-pinned host assigns `1`, and the preflight must resolve that
    rather than compare it against UUIDs and fail every job."""
    job_id = prepare(gpus=["1"], command='echo "CVD=$CUDA_VISIBLE_DEVICES"')
    assert runner.run_job(job_id, deps()) == 0
    assert "CVD=1" in log_of(job_id)
    # Written back so a dispatcher that restarts adopts the card as busy.
    assert jobs.read_state(job_id).gpus == [FAKE_GPUS[1]]


def test_an_assigned_index_the_host_does_not_have_fails_the_job(gpuc_home: Path) -> None:
    job_id = prepare(gpus=["7"], command="echo SHOULD-NOT-RUN")
    assert runner.run_job(job_id, deps()) == 1
    assert jobs.read_state(job_id).reason == "gpu-assert"
    assert "SHOULD-NOT-RUN" not in log_of(job_id)
    assert "assigned GPUs not present on this host: 7" in log_of(job_id)


def test_a_stale_assigned_uuid_fails_the_job_before_it_starts(gpuc_home: Path) -> None:
    job_id = prepare(gpus=["GPU-stale"], command="echo SHOULD-NOT-RUN")
    assert runner.run_job(job_id, deps()) == 1
    state = jobs.read_state(job_id)
    assert (state.status, state.reason) == ("failed", "gpu-assert")
    assert "SHOULD-NOT-RUN" not in log_of(job_id)


def test_a_job_with_no_gpu_assigned_never_runs(gpuc_home: Path) -> None:
    """Every job runs on at least one card. A state with none was written by a
    build that still allowed it, and the job fails the way a missing card does."""
    job_id = prepare(gpus=[], command="echo SHOULD-NOT-RUN")
    assert runner.run_job(job_id, deps()) == 1
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
    assert runner.run_job(job_id, deps(sync_preflight=False)) == 1
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
    assert runner.run_job(job_id, deps(sync_preflight=False)) == 7
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
    assert runner.run_job(job_id, deps(command_runner=command_runner, sync_preflight=False)) != 0
    thread.join(timeout=30)
    state = jobs.read_state(job_id)
    assert (state.status, state.reason, state.problems) == ("failed", "preempted", ["sync"])
    assert queue.requeue_preempted(job_id) == 2
    assert jobs.read_state(job_id).status == "queued"


def test_max_runtime_kills_with_reason_timeout(gpuc_home: Path) -> None:
    job_id = prepare(command="sleep 60", max_runtime_min=0.02)
    start = time.monotonic()
    code = runner.run_job(job_id, deps())
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

    assert runner.run_job(job_id, deps(sampler=idle)) == 0
    state = jobs.read_state(job_id)
    assert (state.status, state.reason) == ("succeeded", None)

    assert seen, "the sampler was never called, so this proved nothing"
    assert {phase for phase, _ in seen} == {"main"}
    # The first sample of main found an empty history: setup recorded nothing.
    assert seen[0][1] == 0
    assert state.util_recent and set(state.util_recent) == {0.0}


def test_an_idle_gpu_is_reported_and_never_a_reason_to_kill(gpuc_home: Path) -> None:
    job_id = prepare(gpus=[FAKE_GPUS[0]], command="sleep 0.5")
    assert runner.run_job(job_id, deps(sampler=lambda uuids: 0.0)) == 0
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
    monkeypatch.setattr(destinations, "find_binary", lambda name, env=None: f"/fake/{name}")
    calls: list[list[str]] = []

    def command_runner(
        argv: list[str], timeout: float | None = None, env: sync.Env = None
    ) -> sync.CommandResult:
        calls.append(argv)
        return sync.CommandResult(argv, 0, "")

    job_id = prepare(command="true")
    assert runner.run_job(job_id, deps(command_runner=command_runner)) == 0
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


def test_a_job_cancelled_during_the_launch_window_never_runs_its_command(
    gpuc_home: Path,
) -> None:
    job_id = prepare(command="touch RAN")
    queue.cancel(job_id)
    assert runner.run_job(job_id, deps()) == runner.TERMINATED_EXIT_CODE
    state = jobs.read_state(job_id)
    assert (state.status, state.reason) == ("cancelled", "cancelled")
    assert not (paths.workdir(job_id) / "RAN").exists()
    assert "cancelled before phase=setup; not starting it" in log_of(job_id)


def test_a_job_preempted_during_the_launch_window_never_runs_its_command(
    gpuc_home: Path,
) -> None:
    """The intent lands between phases as readily as mid-phase, and a job that
    is going back in the queue must not spend a single phase's work first."""
    job_id = prepare(command="touch RAN")
    queue.enqueue(make_spec(priority=1))  # something waiting, or preempt refuses
    queue.preempt(job_id)
    assert runner.run_job(job_id, deps()) == runner.TERMINATED_EXIT_CODE
    state = jobs.read_state(job_id)
    assert (state.status, state.reason) == ("failed", "preempted")
    assert not (paths.workdir(job_id) / "RAN").exists()
    assert "preempted before phase=setup; not starting it" in log_of(job_id)


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

    job_id = prepare(gpus=[FAKE_GPUS[0]], command="sleep 0.6")
    assert runner.run_job(job_id, deps(sampler=flaky)) == 0
    recent = jobs.read_state(job_id).util_recent
    assert recent and recent[0] is None
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
    assert runner.run_job(job_id, deps(preflight=True)) == 0
    assert "phase=preflight: echo VIA-THE-JOBS-PYTHON -c " in log_of(job_id)


def test_preflight_is_a_real_phase(gpuc_home: Path) -> None:
    job_id = prepare(gpus=[FAKE_GPUS[0]], command="true")
    assert (
        runner.run_job(
            job_id,
            deps(preflight=True, preflight_command=lambda spec: "echo gpu preflight ok"),
        )
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
    assert runner.run_job(job_id, deps(sync_preflight=False)) == 1
    state = jobs.read_state(job_id)
    assert (state.status, state.reason) == ("failed", "no-outputs")


# -- PATH, secrets and the sync environment ------------------------------------


def test_build_env_puts_the_home_tool_dirs_in_front_of_path(
    gpuc_home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The pod's sshd PATH has no ~/.local/bin, so `uv run --no-sync` -- which
    the runner's own GPU preflight uses -- would not resolve."""
    fake_home = tmp_path / "home"
    (fake_home / ".local/bin").mkdir(parents=True)
    (fake_home / ".cargo/bin").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    env = runner.build_env(make_spec(), [])
    assert env["PATH"].split(os.pathsep)[:2] == [
        str(fake_home / ".local/bin"),
        str(fake_home / ".cargo/bin"),
    ]
    assert env["PATH"].endswith("/usr/bin:/bin")


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
    assert runner.run_job(job_id, deps(command_runner=command_runner)) == 0
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
    assert runner.run_job(job_id, deps(command_runner=command_runner)) == 0
    assert all(present_during_sync)
    assert not paths.job_env_file(job_id).exists()


def test_no_secret_value_is_ever_written_to_a_log(gpuc_home: Path) -> None:
    secret = "gpuc-canary-must-not-appear-0123456789"
    job_id = prepare(command="echo the job ran", env={"HARMLESS": "1"}, secrets=["WANDB_API_KEY"])
    paths.job_env_file(job_id).write_text(f"WANDB_API_KEY={secret}\n")
    assert runner.run_job(job_id, deps()) == 0
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
    assert runner.run_job(job_id, deps(command_runner=command_runner)) == 0
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
    runner.run_job(job_id, deps(command_runner=command_runner))
    state = jobs.read_state(job_id)
    assert not state.mirrored
    assert state.mirror is not None and state.mirror.ok_at is None and state.mirror.error
    assert "final state upload failed" in log_of(job_id)


def test_a_host_with_no_prefix_records_no_backup(gpuc_home: Path) -> None:
    jobs.write_config(HostConfig(host="h", gpus=[]))
    job_id = prepare(command="true")
    assert runner.run_job(job_id, deps()) == 0
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
    assert runner.run_job(job_id, deps(command_runner=command_runner)) == 0
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
    runner.run_job(job_id, deps(command_runner=command_runner, sync_preflight=False))
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
    runner.run_job(job_id, deps(command_runner=command_runner, sync_preflight=False))
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
    runner.run_job(job_id, deps(command_runner=command_runner, sync_preflight=False))
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
    runner.run_job(job_id, deps(command_runner=command_runner))
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
    assert runner.run_job(job_id, deps()) == 0
    state = jobs.read_state(job_id)
    assert (state.status, state.reason, state.exit_code) == ("succeeded", None, 0)
    assert calls == ["final"]


def test_a_preempted_job_keeps_its_secrets_for_the_next_attempt(gpuc_home: Path) -> None:
    """Nothing delivers them a second time: the next attempt is this same job
    id, dispatched by the host, and the secrets only ever arrive at submit."""
    job_id = prepare(command="sleep 30")
    paths.job_env_file(job_id).write_text('HF_TOKEN="hf_abc"\n')
    queue.enqueue(make_spec(priority=1))  # something waiting, or preempt refuses
    queue.preempt(job_id)
    assert runner.run_job(job_id, deps()) != 0
    assert paths.job_env_file(job_id).exists()
    assert "keeping this job's secrets file for the next attempt" in log_of(job_id)
