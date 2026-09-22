"""End-to-end on the real local GPU, through the real detached dispatcher.

Tiny tensors only (torch.zeros(8)): the card is shared with other people's jobs.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import time
from collections.abc import Callable
from pathlib import Path

import pytest

from gpuc.host import dispatcher, gpus, jobs, paths, queue
from gpuc.host.jobs import HostConfig, JobSpec
from tests.conftest import LOCAL_GPU_UUID, requires_gpu

pytestmark = [pytest.mark.gpu, requires_gpu]

PROBE = (
    "uv run --no-sync python -c "
    "'import os, torch; "
    'print("CVD=" + os.environ["CUDA_VISIBLE_DEVICES"]); '
    'print("COUNT=%d" % torch.cuda.device_count()); '
    'print("NAME=" + torch.cuda.get_device_name(0)); '
    'print("SUM=%d" % (torch.zeros(8, device="cuda") + 1).sum().item())\''
)


@pytest.fixture
def gpu_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "gpuc-home"
    monkeypatch.setenv("GPUC_HOME", str(home))
    paths.ensure_layout()
    jobs.write_config(HostConfig(host="local-gpu-test", gpus=[LOCAL_GPU_UUID]))
    return home


def wait_until(predicate: Callable[[], bool], timeout: float, what: str) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(0.2)
    raise AssertionError(f"timed out after {timeout}s waiting for {what}")


def finished(job_id: str) -> bool:
    try:
        return jobs.read_state(job_id).finished
    except (RuntimeError, FileNotFoundError):
        return False


def enqueue_in_project(spec: JobSpec, project: Path | None) -> str:
    job_id = queue.enqueue(spec)
    if project is not None:
        for name in ("pyproject.toml", "uv.lock"):
            shutil.copy(project / name, paths.workdir(job_id) / name)
    return job_id


def start_dispatcher() -> int:
    pid = dispatcher.spawn_detached_dispatcher()
    wait_until(lambda: paths.heartbeat_file().exists(), 60, "the dispatcher to take the lock")
    return pid


def stop_dispatcher(pid: int) -> None:
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(pid, 9)


def spec_for(command: str, **overrides: object) -> JobSpec:
    document: dict[str, object] = {
        "job_id": jobs.new_job_id(),
        "name": "gpu-integration",
        "command": command,
        "gpus": 1,
        "setup": "uv sync --frozen --quiet",
    }
    document.update(overrides)
    return JobSpec.from_dict(document)


def test_a_real_gpu_job_runs_on_the_assigned_card_and_succeeds(
    gpu_home: Path, torch_project: Path
) -> None:
    job_id = enqueue_in_project(spec_for(PROBE), torch_project)
    pid = start_dispatcher()
    try:
        wait_until(lambda: finished(job_id), 900, f"job {job_id} to finish")
    finally:
        stop_dispatcher(pid)

    log = paths.log_file(job_id).read_text()
    state = jobs.read_state(job_id)
    assert state.status == "succeeded", log[-3000:]
    assert state.exit_code == 0
    assert state.gpus == [LOCAL_GPU_UUID]
    # By nvidia-smi index, not UUID: vLLM and others `int()` each entry.
    index = next(gpu.index for gpu in gpus.list_gpus() if gpu.uuid == LOCAL_GPU_UUID)
    assert f"CVD={index}" in log
    assert "COUNT=1" in log
    assert "SUM=8" in log
    assert "gpu preflight ok" in log
    assert state.reason is None and state.pgid is None


def test_a_failing_gpu_job_propagates_its_exit_code(gpu_home: Path, torch_project: Path) -> None:
    job_id = enqueue_in_project(
        spec_for('uv run --no-sync python -c "import torch, sys; sys.exit(17)"'),
        torch_project,
    )
    pid = start_dispatcher()
    try:
        wait_until(lambda: finished(job_id), 900, f"job {job_id} to finish")
    finally:
        stop_dispatcher(pid)
    state = jobs.read_state(job_id)
    assert (state.status, state.exit_code) == ("failed", 17)
    assert state.reason == "exit 17"


def test_cancel_on_the_real_dispatcher_kills_a_grandchild(
    gpu_home: Path, torch_project: Path
) -> None:
    job_id = enqueue_in_project(
        spec_for("bash -c 'sleep 600 & echo $! > gc.pid ; wait' & echo started; wait"),
        torch_project,
    )
    pid_file = paths.workdir(job_id) / "gc.pid"
    pid = start_dispatcher()
    try:
        wait_until(
            lambda: pid_file.exists() and pid_file.read_text().strip().isdigit(),
            900,
            "the grandchild to record its pid",
        )
        grandchild = int(pid_file.read_text().strip())
        assert os.getpgid(grandchild) == jobs.read_state(job_id).pgid

        queue.cancel(job_id)
        wait_until(lambda: finished(job_id), 120, "the job to be cancelled")
        wait_until(lambda: not _alive(grandchild), 60, "the grandchild to die")
    finally:
        with contextlib.suppress(OSError, ValueError):
            os.kill(int(pid_file.read_text().strip()), 9)
        stop_dispatcher(pid)

    state = jobs.read_state(job_id)
    assert (state.status, state.reason) == ("cancelled", "cancelled")


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True
