from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from gpuc.control.config import HostCache, HostEntry, HostKind, registry_transaction
from gpuc.control.gpuinfo import GpuInfo
from gpuc.host import jobs, paths, scope
from gpuc.host.jobs import HostConfig, JobSpec

pytest_plugins = ["tests.fakehost"]
"""`fake_host`: the in-memory host `gpuc host add|set` talk to in these tests."""

FAKE_GPUS = [
    "GPU-00000000-0000-0000-0000-000000000001",
    "GPU-00000000-0000-0000-0000-000000000002",
]

LOCAL_GPU_UUID = "GPU-2a4bad3b-9fe3-7031-914d-384254e92908"

SEEN_AT = "2026-09-15T12:00:00+00:00"
"""When a test entry's cache was filled. A fixed stamp, so a listing that says
how old the cache is has something stable to say."""


@pytest.fixture
def gpuc_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    home = tmp_path / "gpuc-home"
    monkeypatch.setenv("GPUC_HOME", str(home))
    # No test may inherit a real provider key from the developer running it: a
    # test that needs one sets its own, and one that does not must behave the
    # same on a laptop with `RUNPOD_API_KEY` exported and on a bare CI runner.
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    # Pin the isolation mode: whether *this* machine can make systemd scopes is
    # not something a unit test should depend on. The cgroup path has its own
    # tests, which set this to `cgroup` themselves.
    monkeypatch.setenv(scope.ISOLATION_ENV, scope.PGID)
    paths.ensure_layout()
    jobs.write_config(HostConfig(host="test-host", gpus=list(FAKE_GPUS)))
    yield home


def host_entry(
    *,
    name: str = "h",
    kind: HostKind = "local",
    ssh: str | None = None,
    port: int = 22,
    gpuc_home: str | None = None,
    persistent_root: str | None = None,
    pod_id: str | None = None,
    bootstrapped_at: str | None = SEEN_AT,
    python: str | None = None,
    uv: str | None = None,
    gpu_info: dict[str, GpuInfo] | None = None,
    driver_version: str | None = None,
    read_at: str | None = SEEN_AT,
    **config: Any,
) -> HostEntry:
    """A registry entry whose cache already holds the host's own config.

    The address is the entry's; everything in `**config` is a key of the
    `config.json` that lives on the host, which a real entry only ever gets by
    reading it. Tests that are not about connecting say what the host holds
    here instead of staging an ssh round trip for it.

    Bootstrapped by default, because "a registered host" in a test almost
    always means one that is set up and can be submitted to; a test about a
    host nobody has installed gpuc on passes `bootstrapped_at=None`.
    """
    cache_dir = config.pop("cache_dir", None)
    if cache_dir:
        config["env"] = {**(config.get("env") or {}), "UV_CACHE_DIR": cache_dir}
    document: dict[str, Any] = {"host": name, **config}
    if kind == "runpod":
        # A rental is an address with a pod behind it; `kind` alone is not one.
        pod_id = pod_id or f"pod-{name}"
        document.setdefault("provider", {"kind": "runpod", "pod_id": pod_id})
    return HostEntry(
        name=name,
        ssh=ssh,
        port=port,
        gpuc_home=gpuc_home,
        persistent_root=persistent_root,
        pod_id=pod_id,
        bootstrapped_at=bootstrapped_at,
        cache=HostCache(
            read_at=read_at,
            python=python,
            uv=uv,
            gpu_info=dict(gpu_info or {}),
            driver_version=driver_version,
            config=HostConfig.from_dict(document).to_dict(),
        ),
    )


def register_host(*, gpus: str | list[str] | None = None, **fields: Any) -> HostEntry:
    """Put a host in the registry without connecting to it.

    `gpuc host add` reads the host's own config now, so a test that only needs
    a host to exist says what that host holds here rather than staging an ssh
    round trip for it. `gpus` takes the flag's spelling as well as a list.
    """
    entry = host_entry(gpus=_gpus(gpus), **fields)
    with registry_transaction() as registry:
        registry.put(entry)
    return entry


def _gpus(gpus: str | list[str] | None) -> list[str]:
    if isinstance(gpus, str):
        return [part for part in gpus.split(",") if part]
    return list(gpus or [])


def make_spec(**overrides: object) -> JobSpec:
    document: dict[str, object] = {
        "job_id": jobs.new_job_id(),
        "name": "t",
        "command": "true",
        "gpus": 1,
    }
    document.update(overrides)
    return JobSpec.from_dict(document)


FAKE_SMI_SCRIPT = """\
import sys

UUIDS = {uuids!r}
VALUES = {{"name": "Fake A40", "memory.total": "46068", "memory.used": "0",
          "utilization.gpu": "0", "driver_version": "580.173.02"}}
UNITS = {{"memory.total": " MiB", "memory.used": " MiB", "utilization.gpu": " %"}}

args = sys.argv[1:]
query = next((a for a in args if a.startswith("--query-gpu=")), None)
if query is None:
    sys.exit("fake nvidia-smi: only --query-gpu is supported")
fields = query.split("=", 1)[1].split(",")
fmt = next((a for a in args if a.startswith("--format=")), "--format=csv")
nounits = "nounits" in fmt
wanted = args[args.index("-i") + 1].split(",") if "-i" in args else UUIDS
for index, uuid in enumerate(UUIDS):
    if uuid not in wanted:
        continue
    cells = []
    for field in fields:
        if field == "index":
            cells.append(str(index))
        elif field == "uuid":
            cells.append(uuid)
        else:
            cells.append(VALUES.get(field, "") + ("" if nounits else UNITS.get(field, "")))
    print(", ".join(cells))
"""


def install_fake_nvidia_smi(bin_dir: Path, uuids: list[str] | None = None) -> None:
    """Put an `nvidia-smi` for `uuids` (default `FAKE_GPUS`) in `bin_dir`.

    For the tests that run the real dispatcher and runner as subprocesses on a
    machine with no card: those find nvidia-smi on PATH, so `fake_smi` cannot
    reach them. It answers the `--query-gpu` queries gpus.py makes and nothing
    else.
    """
    bin_dir.mkdir(parents=True, exist_ok=True)
    script = bin_dir / "fake-nvidia-smi.py"
    script.write_text(FAKE_SMI_SCRIPT.format(uuids=list(FAKE_GPUS if uuids is None else uuids)))
    wrapper = bin_dir / "nvidia-smi"
    wrapper.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n')
    wrapper.chmod(0o755)


FAKE_TORCH = """\
# A torch that passes the runner's GPU preflight on a machine with no card. It
# sees as many devices as CUDA_VISIBLE_DEVICES names, as the real one does.
import os

__version__ = "0.0-fake"


class version:
    cuda = "0.0"


class _Tensor:
    def __add__(self, other):
        return self

    def sum(self):
        return self

    def item(self):
        return 8.0


def zeros(n, device=None):
    return _Tensor()


class cuda:
    @staticmethod
    def is_available():
        return True

    @staticmethod
    def device_count():
        return len([d for d in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if d])

    @staticmethod
    def get_device_name(index):
        return "Fake A40"
"""


def install_fake_torch(workdir: Path) -> None:
    """A `torch.py` the preflight's `python -c` finds first, being run from `workdir`."""
    (workdir / "torch.py").write_text(FAKE_TORCH)


def fake_smi(
    uuids: list[str] | None = None,
    utilization: dict[str, float] | None = None,
    memory_used: dict[str, float] | None = None,
):
    """A stand-in for `nvidia-smi` that answers the queries gpus.py makes.

    `--format=` is honoured rather than assumed: real nvidia-smi prints a header
    row unless `noheader` is asked for, and a caller that forgets it gets a
    field-name line where it expected data. A fake that never emits one would
    hide exactly that bug.
    """
    listed = FAKE_GPUS if uuids is None else uuids

    def run(args: list[str]) -> str:
        query = next(a for a in args if a.startswith("--query-gpu="))
        fields = query.split("=", 1)[1].split(",")
        fmt = next((a for a in args if a.startswith("--format=")), "--format=csv")
        options = fmt.split("=", 1)[1].split(",")
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
                elif field == "memory.used":
                    cells.append(str((memory_used or {}).get(uuid, 0.0)))
                else:
                    cells.append("")
            rows.append(", ".join(cells))
        if "noheader" not in options:
            rows.insert(0, ", ".join(fields))
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
def session_monkeypatch() -> Iterator[pytest.MonkeyPatch]:
    """`monkeypatch` is function-scoped; session fixtures need their own."""
    patch = pytest.MonkeyPatch()
    yield patch
    patch.undo()


def default_torch_project() -> Path:
    """A per-user path: /tmp is shared, and two users must not collide there."""
    return Path(tempfile.gettempdir()) / f"gpuc-test-torch-project-{os.getuid()}"


@pytest.fixture(scope="session")
def torch_project(session_monkeypatch: pytest.MonkeyPatch) -> Path:
    """A minimal uv project with torch, reused across runs via the uv cache.

    The repo's own venv has no torch, and the runner's GPU preflight is
    specified as `uv run --no-sync python -c ...` from the job's workdir, so a
    real GPU job needs a workdir that is a uv project.
    """
    # Every job syncs its own venv under /tmp; copying ~3 GB of torch per job
    # fills the shared tmpfs, so link the cache instead. Set through monkeypatch
    # so the variable does not outlive the session that wanted it.
    if "UV_LINK_MODE" not in os.environ:
        session_monkeypatch.setenv("UV_LINK_MODE", "symlink")
    override = os.environ.get("GPUC_TEST_TORCH_PROJECT")
    root = Path(override) if override else default_torch_project()
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
    # $HOME as well as the two gpuc directories: a `local` host's gpuc home is
    # under it, and a test that reaches one must not write to the real ~/.gpuc.
    (root / "home").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(root / "home"))
    monkeypatch.setenv("GPUC_CONFIG_DIR", str(root / "config"))
    monkeypatch.setenv("GPUC_STATE_DIR", str(root / "state"))
    monkeypatch.delenv("GPUC_HOME", raising=False)
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    monkeypatch.setenv(scope.ISOLATION_ENV, scope.PGID)
    (root / "config").mkdir(parents=True)
    (root / "state").mkdir(parents=True)
    yield root
