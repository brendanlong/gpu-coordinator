from __future__ import annotations

import os
import shutil
import subprocess
import sys
import textwrap
from collections.abc import Iterator
from pathlib import Path

import pytest

from gpuc.control.config import HostEntry
from gpuc.host import jobs, paths
from gpuc.host.jobs import HostConfig, JobSpec

FAKE_GPUS = [
    "GPU-00000000-0000-0000-0000-000000000001",
    "GPU-00000000-0000-0000-0000-000000000002",
]

LOCAL_GPU_UUID = "GPU-2a4bad3b-9fe3-7031-914d-384254e92908"


@pytest.fixture
def gpuc_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    home = tmp_path / "gpuc-home"
    monkeypatch.setenv("GPUC_HOME", str(home))
    paths.ensure_layout()
    jobs.write_config(HostConfig(host="test-host", gpus=list(FAKE_GPUS)))
    yield home


def make_spec(**overrides: object) -> JobSpec:
    document: dict[str, object] = {
        "job_id": jobs.new_job_id(),
        "name": "t",
        "command": "true",
        "gpus": 0,
    }
    document.update(overrides)
    return JobSpec.from_dict(document)


def fake_smi(uuids: list[str] | None = None, utilization: dict[str, float] | None = None):
    """A stand-in for `nvidia-smi` that answers the two queries gpus.py makes."""
    listed = FAKE_GPUS if uuids is None else uuids

    def run(args: list[str]) -> str:
        query = next(a for a in args if a.startswith("--query-gpu="))
        fields = query.split("=", 1)[1].split(",")
        selected = listed
        if "-i" in args:
            wanted = args[args.index("-i") + 1].split(",")
            selected = [u for u in listed if u in wanted]
        rows: list[str] = []
        for index, uuid in enumerate(listed):
            if uuid not in selected:
                continue
            cells: list[str] = []
            for field in fields:
                if field == "index":
                    cells.append(str(index))
                elif field == "uuid":
                    cells.append(uuid)
                elif field == "driver_version":
                    cells.append("580.173.02")
                elif field == "utilization.gpu":
                    cells.append(str((utilization or {}).get(uuid, 0.0)))
                else:
                    cells.append("")
            rows.append(", ".join(cells))
        return "\n".join(rows) + "\n"

    return run


def nvidia_smi_available() -> bool:
    if shutil.which("nvidia-smi") is None:
        return False
    proc = subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode == 0 and LOCAL_GPU_UUID in proc.stdout


requires_gpu = pytest.mark.skipif(
    not nvidia_smi_available(), reason=f"no local NVIDIA GPU {LOCAL_GPU_UUID}"
)


@pytest.fixture(scope="session")
def torch_project() -> Path:
    """A minimal uv project with torch, reused across runs via the uv cache.

    The repo's own venv has no torch, and the runner's GPU preflight is
    specified as `uv run --no-sync python -c ...` from the job's workdir, so a
    real GPU job needs a workdir that is a uv project.
    """
    # Every job syncs its own venv under /tmp; copying ~3 GB of torch per job
    # fills the shared tmpfs, so link the cache instead.
    os.environ.setdefault("UV_LINK_MODE", "symlink")
    root = Path(os.environ.get("GPUC_TEST_TORCH_PROJECT", "/tmp/gpuc-test-torch-project"))
    root.mkdir(parents=True, exist_ok=True)
    (root / "pyproject.toml").write_text(
        textwrap.dedent(
            """
            [project]
            name = "gpuc-test-job"
            version = "0.0.0"
            requires-python = ">=3.11"
            dependencies = ["torch"]
            """
        ).strip()
        + "\n"
    )
    (root / "README.md").write_text("test project\n")
    proc = subprocess.run(
        ["uv", "sync", "--quiet"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
        timeout=1800,
    )
    if proc.returncode != 0:
        pytest.skip(f"could not build a torch project: {proc.stdout}\n{proc.stderr}")
    return root


@pytest.fixture
def control_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """A control side whose config, state and on-host GPUC_HOME are all temporary."""
    root = tmp_path / "control"
    monkeypatch.setenv("GPUC_CONFIG_DIR", str(root / "config"))
    monkeypatch.setenv("GPUC_STATE_DIR", str(root / "state"))
    monkeypatch.delenv("GPUC_HOME", raising=False)
    (root / "config").mkdir(parents=True)
    (root / "state").mkdir(parents=True)
    yield root


def local_host_entry(name: str, home: Path, gpus: list[str] | None = None) -> HostEntry:
    """A `local` host whose ~/.gpuc is redirected, so tests never touch the real one."""
    return HostEntry(
        name=name,
        kind="local",
        gpus=gpus or [],
        gpuc_home=str(home),
        python=sys.executable,
    )
