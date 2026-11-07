from __future__ import annotations

import fcntl
import hashlib
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
from collections.abc import Callable, Iterator, Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gpuc.control import wait as wait_mod
from gpuc.control.config import (
    HostCache,
    HostEntry,
    HostKind,
    Registry,
    Rental,
    open_registry,
    registry_transaction,
)
from gpuc.control.gpuinfo import GpuInfo
from gpuc.host import jobs, paths, queue, scope
from gpuc.host.jobs import HostConfig, JobSpec, JobState
from tests import fake_nvidia_smi

pytest_plugins = ["tests.fakehost"]
"""`fake_host`: a host in a temporary home that `gpuc host add|set` and the
provisioning flow really talk to."""

FAKE_GPUS = [
    "GPU-00000000-0000-0000-0000-000000000001",
    "GPU-00000000-0000-0000-0000-000000000002",
]

LOCAL_GPU_UUID = "GPU-2a4bad3b-9fe3-7031-914d-384254e92908"

SEEN_AT = "2026-09-15T12:00:00+00:00"
"""When a test entry's cache was filled. A fixed stamp, so a listing that says
how old the cache is has something stable to say."""


@pytest.fixture(autouse=True)
def no_aws_keys(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """No test but the live one sees the developer's AWS keys: a submit to a
    rental mints narrowed keys from them, and a unit test must never reach STS."""
    if request.node.get_closest_marker("rental") is None:
        for name in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"):
            monkeypatch.delenv(name, raising=False)


def ago(**delta: float) -> str:
    """An ISO timestamp `delta` before now; a negative delta is in the future."""
    return (datetime.now(UTC) - timedelta(**delta)).isoformat()


@pytest.fixture
def empty_caches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the uv and Hugging Face caches the health check sizes at empty
    directories: sizing the developer's own took seconds per test, and made
    the result depend on the machine."""
    monkeypatch.setenv("UV_CACHE_DIR", str(tmp_path / "uv-cache"))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf-home"))
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)


@pytest.fixture
def gpuc_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    home = tmp_path / "gpuc-home"
    monkeypatch.setenv("GPUC_HOME", str(home))
    # No test may inherit a real provider key from the developer running it: a
    # test that needs one sets its own, and one that does not must behave the
    # same on a laptop with a provider key exported and on a bare CI runner.
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    monkeypatch.delenv("VAST_API_KEY", raising=False)
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
    rental: Rental | None = None
    if kind == "rental" or pod_id is not None:
        # A rental is an address with a pod behind it; `kind` alone is not one.
        pod_id = pod_id or f"pod-{name}"
        rental = Rental(provider="runpod", pod_id=pod_id)
        document.setdefault("provider", {"kind": "runpod", "pod_id": pod_id})
    return HostEntry(
        name=name,
        ssh=ssh,
        port=port,
        gpuc_home=gpuc_home,
        persistent_root=persistent_root,
        rental=rental,
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


def load_registry() -> Registry:
    """The registry as a command would read it, for a test's assertions."""
    return open_registry().registry


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


def accept_job(spec: JobSpec, **state: Any) -> str:
    """A job on the host, accepted the way `gpuc submit` gets one accepted
    (`queue.enqueue`: spec and initial state written under `incoming/`, then
    the rename), then in whatever state the test needs."""
    job_id = queue.enqueue(spec)
    if state:
        jobs.write_state(job_id, JobState(**state))
    return job_id


def make_spec(**overrides: object) -> JobSpec:
    document: dict[str, object] = {
        "job_id": jobs.new_job_id(),
        "name": "t",
        "command": "true",
        "gpus": 1,
    }
    document.update(overrides)
    return JobSpec.from_dict(document)


def wait_until(
    predicate: Callable[[], bool],
    timeout: float = 30.0,
    what: str = "condition",
    interval: float = 0.05,
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(interval)
    raise AssertionError(f"timed out after {timeout}s waiting for {what}")


def install_fake_nvidia_smi(bin_dir: Path, uuids: list[str] | None = None) -> None:
    """Put an `nvidia-smi` for `uuids` (default `FAKE_GPUS`) in `bin_dir`.

    For everything that execs the real binary: the dispatcher and runner as
    subprocesses, the probe script, the health check. It runs
    `tests/fake_nvidia_smi.py`, the same rows `fake_smi` answers in-process.
    """
    bin_dir.mkdir(parents=True, exist_ok=True)
    wrapper = bin_dir / "nvidia-smi"
    script = Path(__file__).with_name("fake_nvidia_smi.py")
    listed = ",".join(FAKE_GPUS if uuids is None else uuids)
    wrapper.write_text(
        f"#!/bin/sh\n{fake_nvidia_smi.UUIDS_ENV}={shlex.quote(listed)} "
        f'exec "{sys.executable}" "{script}" "$@"\n'
    )
    wrapper.chmod(0o755)


FAKE_LIBCUDA_SOURCE = r"""
/* The few CUDA driver calls the runner's GPU preflight makes, over a driver
   that sees as many devices as CUDA_VISIBLE_DEVICES names, as the real one
   does. */
#include <stdlib.h>
#include <string.h>

#define NO_DEVICE 100

static int visible(void) {
    const char *list = getenv("CUDA_VISIBLE_DEVICES");
    int count = 0;
    if (list == NULL || *list == '\0') return 0;
    for (count = 1; *list; list++) count += *list == ',';
    return count;
}

int cuInit(unsigned flags) { return visible() ? 0 : NO_DEVICE; }
int cuGetErrorName(int status, const char **name) {
    *name = status == NO_DEVICE ? "CUDA_ERROR_NO_DEVICE" : NULL;
    return 0;
}
int cuDeviceGetCount(int *count) { *count = visible(); return 0; }
int cuDeviceGet(int *device, int index) { *device = index; return 0; }
int cuDeviceGetName(char *name, int size, int device) {
    strncpy(name, "Fake A40", size);
    return 0;
}
int cuDevicePrimaryCtxRetain(void **context, int device) { *context = context; return 0; }
int cuDevicePrimaryCtxRelease_v2(int device) { return 0; }
"""


def fake_libcuda_dir() -> Path:
    """A directory holding a fake `libcuda.so.1`, for `LD_LIBRARY_PATH`:
    searched before the system's, so it stands in for a real driver too.
    Built once per source and uid; a machine without `cc` skips the test."""
    digest = hashlib.sha256(FAKE_LIBCUDA_SOURCE.encode()).hexdigest()[:12]
    lib_dir = Path(tempfile.gettempdir()) / f"gpuc-test-libcuda-{os.getuid()}-{digest}"
    lib = lib_dir / "libcuda.so.1"
    if lib.exists():
        return lib_dir
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("no C compiler to build a fake libcuda")
    lib_dir.mkdir(exist_ok=True)
    source = lib_dir / f"libcuda-{os.getpid()}.c"
    source.write_text(FAKE_LIBCUDA_SOURCE)
    built = lib_dir / f"libcuda-{os.getpid()}.so"
    subprocess.run(
        [compiler, "-shared", "-fPIC", "-o", str(built), str(source)],
        check=True,
        capture_output=True,
    )
    built.replace(lib)
    source.unlink()
    return lib_dir


def fake_cuda_env(env: Mapping[str, str]) -> dict[str, str]:
    """`env` with the fake driver ahead of any real one."""
    rest = env.get("LD_LIBRARY_PATH")
    path = str(fake_libcuda_dir()) + (os.pathsep + rest if rest else "")
    return {**env, "LD_LIBRARY_PATH": path}


def fake_torch(cuda: bool = True) -> str:
    """A `torch.py` for the preflight's optional torch step: a CUDA build
    that sees what the fake driver sees, or a CPU-only one."""
    return f"""\
import os

__version__ = "0.0-fake"


class version:
    cuda = {"'0.0'" if cuda else None}


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
        return {cuda}

    @staticmethod
    def device_count():
        return len([d for d in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if d])
"""


def fake_smi(
    uuids: list[str] | None = None,
    utilization: dict[str, float] | None = None,
    memory_used: dict[str, float] | None = None,
) -> Callable[[list[str]], str]:
    """A stand-in for `nvidia-smi` for code that takes an injectable `smi`:
    `fake_nvidia_smi.render` over `uuids` (default `FAKE_GPUS`)."""
    listed = FAKE_GPUS if uuids is None else uuids

    def run(args: list[str]) -> str:
        return fake_nvidia_smi.render(args, listed, utilization, memory_used)

    return run


requires_gpu = pytest.mark.skipif(
    shutil.which("nvidia-smi") is None
    or LOCAL_GPU_UUID
    not in subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout,
    reason=f"no local NVIDIA GPU {LOCAL_GPU_UUID}",
)


@pytest.fixture(scope="session")
def session_monkeypatch() -> Iterator[pytest.MonkeyPatch]:
    """`monkeypatch` is function-scoped; session fixtures need their own."""
    patch = pytest.MonkeyPatch()
    yield patch
    patch.undo()


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
    # Per user by default: /tmp is shared, and two users must not collide there.
    default = Path(tempfile.gettempdir()) / f"gpuc-test-torch-project-{os.getuid()}"
    root = Path(os.environ.get("GPUC_TEST_TORCH_PROJECT") or default)
    root.mkdir(parents=True, exist_ok=True)
    # Every xdist worker builds its own session fixtures, all on this one path.
    with (root.parent / f"{root.name}.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _sync_torch_project(root)


def _sync_torch_project(root: Path) -> Path:
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
    monkeypatch.delenv("VAST_API_KEY", raising=False)
    monkeypatch.setenv(scope.ISOLATION_ENV, scope.PGID)
    (root / "config").mkdir(parents=True)
    (root / "state").mkdir(parents=True)
    yield root


def stub_wait_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Make `wait`'s poll sleeps instant, and return the list they are recorded in.

    Only `wait`'s own view of `time` is replaced: stubbing `time.sleep` itself
    also stubbed the loop `subprocess` waits for a child in, so a host asked in
    a thread counted thousands of sleeps on a busy machine.
    """
    slept: list[float] = []
    monkeypatch.setattr(
        wait_mod, "time", SimpleNamespace(monotonic=time.monotonic, sleep=slept.append)
    )
    return slept
