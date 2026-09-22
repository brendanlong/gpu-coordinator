"""`gpuc` end to end on the real local GPU: add, bootstrap, submit, status, logs.

Tiny tensors only (torch.zeros(8)); the card is shared with other people's jobs.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path

import pytest

from gpuc.control.cli import main
from gpuc.host import gpus
from tests.conftest import LOCAL_GPU_UUID, requires_gpu
from tests.test_control_e2e import (
    HEALTH_ARGS,
    SHARED_UV_CACHE,
    _stop_dispatcher,
    finished,
    log_tail,
    state_of,
    submit,
    wait_for_main_phase,
    wait_until,
)

pytestmark = [pytest.mark.gpu, requires_gpu]

PROBE = (
    "uv run --no-sync python -c "
    "'import os, torch; "
    'print("CVD=" + os.environ["CUDA_VISIBLE_DEVICES"]); '
    'print("COUNT=%d" % torch.cuda.device_count()); '
    'print("SUM=%d" % (torch.zeros(8, device="cuda") + 1).sum().item())\''
)


@pytest.fixture
def torch_workdir(tmp_path: Path, torch_project: Path) -> Path:
    root = tmp_path / "project"
    root.mkdir()
    for name in ("pyproject.toml", "uv.lock"):
        shutil.copy(torch_project / name, root / name)
    (root / "README.md").write_text("test project\n")
    for argv in (
        ["git", "init", "-q"],
        ["git", "config", "user.email", "t@example.com"],
        ["git", "config", "user.name", "t"],
        ["git", "add", "-A"],
        ["git", "commit", "-qm", "init"],
    ):
        subprocess.run(argv, cwd=root, check=True, capture_output=True)
    return root


@pytest.fixture
def gpu_host(control_env: Path, tmp_path: Path) -> Iterator[Path]:
    home = tmp_path / "gpuc-home"
    assert (
        main(
            [
                "host",
                "add",
                "local",
                "--gpus",
                LOCAL_GPU_UUID,
                "--gpuc-home",
                str(home),
                "--cache-dir",
                SHARED_UV_CACHE,
            ]
        )
        == 0
    )
    assert main(["host", "bootstrap", "local", "--health-args", HEALTH_ARGS]) == 0
    yield home
    _stop_dispatcher(home)


def queued_ids(status: str) -> list[str]:
    """The job ids on `gpuc status`'s queued lines, in the order printed.

    A line is `  queued  <name> (<job-id>) prio=NN ...`, with the name and its
    parentheses there only when the job has one -- so the id is a field whose
    position moves, which is what these assertions used to get wrong.
    """
    ids: list[str] = []
    for line in status.splitlines():
        if not line.strip().startswith("queued"):
            continue
        labelled = re.search(r"\(([^)]+)\)", line)
        ids.append(labelled.group(1) if labelled else line.split()[1])
    return ids


def gpu_job(command: str, name: str = "gpu-e2e", priority: int = 50) -> str:
    return (
        f"name: {name}\n"
        f"command: {command}\n"
        f"priority: {priority}\n"
        "gpus: 1\n"
        "setup: uv sync --frozen --quiet\n"
    )


def test_a_submitted_gpu_job_runs_on_the_owned_uuid(
    gpu_host: Path, torch_workdir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    home = gpu_host
    job_id = submit(torch_workdir, gpu_job(PROBE))
    capsys.readouterr()
    wait_until(lambda: finished(home, job_id), 900, f"job {job_id} to finish")

    log = (home / "jobs" / job_id / "log.txt").read_text()
    state = state_of(home, job_id)
    assert state["status"] == "succeeded", log[-3000:]
    assert state["gpus"] == [LOCAL_GPU_UUID]
    # The card by its nvidia-smi index, which is what every CUDA stack accepts.
    index = next(gpu.index for gpu in gpus.list_gpus() if gpu.uuid == LOCAL_GPU_UUID)
    assert f"CVD={index}" in log
    assert "COUNT=1" in log
    assert "SUM=8" in log
    assert "gpu preflight ok" in log

    assert main(["logs", job_id, "-n", "500"]) == 0
    assert f"CVD={index}" in capsys.readouterr().out

    assert main(["status", "--host", "local"]) == 0
    status = capsys.readouterr().out
    assert f"done    gpu-e2e ({job_id}) succeeded" in status
    assert "gpus 1/1 free" in status


def test_cancelling_a_running_gpu_job_frees_the_card(
    gpu_host: Path, torch_workdir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    home = gpu_host
    job_id = submit(torch_workdir, gpu_job("sleep 600", name="gpu-cancel"))
    capsys.readouterr()
    wait_for_main_phase(home, job_id)
    assert main(["cancel", job_id]) == 0
    assert job_id in capsys.readouterr().out

    wait_until(lambda: finished(home, job_id), 120, "the job to be cancelled")
    assert state_of(home, job_id)["status"] == "cancelled"

    assert main(["status", "--host", "local"]) == 0
    assert "gpus 1/1 free" in capsys.readouterr().out


def test_queued_jobs_reorder_and_cancel_while_the_card_is_busy(
    gpu_host: Path, torch_workdir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    home = gpu_host
    hog = submit(torch_workdir, gpu_job("sleep 600", name="hog"), "hog.yaml")
    wait_for_main_phase(home, hog)
    first = submit(torch_workdir, gpu_job("echo first", name="first"), "first.yaml")
    second = submit(torch_workdir, gpu_job("echo second", name="second"), "second.yaml")
    capsys.readouterr()

    assert main(["status", "--host", "local"]) == 0
    # Set, not sequence: job ids are second-granular, so two submits inside one
    # second tie-break on their random suffix, not on submission order.
    assert {first, second} == set(queued_ids(capsys.readouterr().out))

    assert main(["reorder", second, "--priority", "10"]) == 0
    capsys.readouterr()
    assert main(["status", "--host", "local"]) == 0
    status = capsys.readouterr().out
    assert [second, first] == queued_ids(status)
    head = next(line for line in status.splitlines() if line.strip().startswith("queued"))
    assert "prio=10" in head
    assert "gpus 0/1 free" in status

    for job_id in (first, second, hog):
        assert main(["cancel", job_id]) == 0
        wait_until(lambda job_id=job_id: finished(home, job_id), 120, f"{job_id} to be cancelled")
        assert state_of(home, job_id)["status"] == "cancelled", log_tail(home, job_id)


def test_preempting_a_gpu_job_hands_the_card_to_the_one_waiting(
    gpu_host: Path, torch_workdir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """What `gpuc preempt` is for, with a real card in the middle of it: the
    running job lets go, the job that was waiting gets the GPU, and the one
    that let go is queued again under its own id with its workdir intact."""
    home = gpu_host
    hog = submit(torch_workdir, gpu_job("sleep 600", name="hog"), "hog.yaml")
    wait_for_main_phase(home, hog)
    waiting = submit(
        torch_workdir, gpu_job("sleep 600", name="waiting", priority=10), "waiting.yaml"
    )
    wait_until(
        lambda: state_of(home, waiting).get("status") == "queued", 60, "the second job to queue"
    )
    capsys.readouterr()

    assert main(["preempt", hog]) == 0
    assert "queued again at priority" in capsys.readouterr().out
    wait_until(
        lambda: state_of(home, waiting).get("status") == "running",
        300,
        "the waiting job to get the card",
    )
    stopped = state_of(home, hog)
    assert (stopped["status"], stopped["attempt"]) == ("queued", 2), log_tail(home, hog)
    assert stopped["gpus"] == []
    # The workdir the next attempt re-runs from, and its venv, are still here.
    assert (home / "jobs" / hog / "workdir" / "pyproject.toml").exists()
    assert "queued again as attempt 2" in log_tail(home, hog, lines=200)

    for job_id in (hog, waiting):
        assert main(["cancel", job_id]) == 0
        wait_until(lambda job_id=job_id: finished(home, job_id), 120, f"{job_id} to be cancelled")
