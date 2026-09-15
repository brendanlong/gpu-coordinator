"""The live test: one real A40 pod, one real job, real teardown.

Costs money (about $0.10). Everything it creates is terminated in a `finally`
that verifies against the provider, not against our own state. Pods without the
`gpuc-` prefix belong to other people and are only ever read.

Run it with:  uv run pytest -m runpod -s
"""

from __future__ import annotations

import configparser
import json
import os
import shutil
import subprocess
import tempfile
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from gpuc.control import status as status_mod
from gpuc.control.cli import main
from gpuc.control.config import (
    HostEntry,
    Settings,
    desired_dir,
    load_desired,
    load_registry,
    load_settings,
)
from gpuc.control.providers.base import Caps, Offer, Pod
from gpuc.control.providers.runpod import RunPodProvider
from gpuc.control.s3index import LocalIndex

pytestmark = pytest.mark.runpod

BUCKET = "brendanlong-experiments"
OUTPUT_PREFIX = "gpuc-e2e"
NAME_HINT = "e2e"
PROVISION_TIMEOUT_S = 900.0
JOB_TIMEOUT_S = 1500.0
IDLE_TIMEOUT_S = 900.0

PYPROJECT = """\
[project]
name = "gpuc-e2e"
version = "0.0.0"
requires-python = ">=3.11"
dependencies = ["torch"]

[tool.uv.sources]
torch = { index = "pytorch-cu128" }

[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true
"""

PROBE = (
    'import torch; x = torch.zeros(8, device="cuda"); '
    "print(torch.cuda.get_device_name(0), (x + 1).sum().item())"
)

JOB = f"""\
name: gpuc-e2e
setup: uv sync
command: mkdir -p results && uv run --no-sync python -c '{PROBE}' | tee results/gpu.txt
gpus: 1
secrets: [AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY]
outputs:
  - path: results
    s3: s3://{BUCKET}/{OUTPUT_PREFIX}/{{job_id}}/results
sync_interval_s: 60
max_runtime_min: 20
requires:
  cuda_min: "12.8"
"""


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def wait_until(
    predicate: Callable[[], bool], timeout_s: float, what: str, interval_s: float = 10.0
) -> float:
    start = time.monotonic()
    while time.monotonic() - start < timeout_s:
        if predicate():
            return time.monotonic() - start
        time.sleep(interval_s)
    raise AssertionError(f"timed out after {timeout_s:.0f}s waiting for {what}")


class RecordingProvider(RunPodProvider):
    """A real provider that remembers what it created, so `finally` can undo it."""

    def __init__(self, caps: Caps) -> None:
        super().__init__(caps=caps)
        self.created_ids: list[str] = []

    def create(self, offer: Offer, name: str, **kwargs: Any) -> Pod:
        pod = super().create(offer, name, **kwargs)
        self.created_ids.append(pod.id)
        return pod


@pytest.fixture
def aws_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """The job's `secrets:` and the pod's log mirror both read these from our env.

    The job's own S3 output needs nothing but the `secrets:` line above: the
    runner hands its secrets to the sync loop.
    """
    if os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY"):
        return
    parser = configparser.ConfigParser()
    parser.read(Path.home() / ".aws/credentials")
    if not parser.has_section("default"):
        pytest.skip("no AWS credentials in the environment or ~/.aws/credentials")
    for key in ("aws_access_key_id", "aws_secret_access_key", "aws_session_token"):
        if parser.has_option("default", key):
            monkeypatch.setenv(key.upper(), parser.get("default", key))


@pytest.fixture
def workdir(tmp_path: Path) -> Path:
    root = tmp_path / "gpuc-e2e"
    root.mkdir()
    (root / "pyproject.toml").write_text(PYPROJECT)
    (root / "README.md").write_text("gpuc live test project\n")
    (root / "job.yaml").write_text(JOB)
    for argv in (
        ["git", "init", "-q"],
        ["git", "config", "user.email", "t@example.com"],
        ["git", "config", "user.name", "gpuc"],
        ["git", "add", "-A"],
        ["git", "commit", "-qm", "e2e"],
    ):
        subprocess.run(argv, cwd=root, check=True, capture_output=True)
    return root


@pytest.fixture
def live_settings(monkeypatch: pytest.MonkeyPatch) -> Iterator[Settings]:
    """A short state dir, not pytest's tmp_path.

    ssh's ControlPath is a unix socket, so the whole path must fit in 108
    bytes; `/tmp/pytest-of-user/pytest-57/test_.../control/cm-<hash>` does not.
    """
    root = Path(tempfile.mkdtemp(prefix="gpuc-e2e-", dir="/tmp"))
    monkeypatch.setenv("GPUC_CONFIG_DIR", str(root / "config"))
    monkeypatch.setenv("GPUC_STATE_DIR", str(root / "state"))
    monkeypatch.delenv("GPUC_HOME", raising=False)
    (root / "config").mkdir()
    (root / "config/config.toml").write_text(
        f's3_bucket = "{BUCKET}"\nmax_pods = 2\nmax_total_usd_per_hour = 1.5\n'
    )
    yield load_settings()
    shutil.rmtree(root, ignore_errors=True)


def s3_client() -> Any:
    import boto3

    return boto3.client("s3")


def job_state(entry: HostEntry, settings: Settings, job_id: str) -> dict[str, Any]:
    view = status_mod.gather(entry, settings)
    for job in view.queue + view.running + view.finished:
        if job.job_id == job_id:
            return {"status": job.status, "phase": job.phase, "reason": job.reason}
    return {"status": "unknown", "phase": None, "reason": None}


def test_submit_to_a_real_pod_runs_a_gpu_job_and_tears_itself_down(
    live_settings: Settings,
    workdir: Path,
    aws_env: None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    if not os.environ.get("RUNPOD_API_KEY"):
        pytest.skip("RUNPOD_API_KEY is not set")
    provider = RecordingProvider(live_settings.caps())
    monkeypatch.setattr("gpuc.control.cli.make_provider", lambda settings: provider)
    monkeypatch.chdir(workdir)
    started = time.monotonic()

    try:
        with capsys.disabled():
            log("submitting; provisioning starts now")
            assert (
                main(
                    [
                        "submit",
                        "job.yaml",
                        "--runpod",
                        "--gpu",
                        "A40",
                        "--max-price",
                        "0.60",
                        "--cloud",
                        "any",
                        "--idle-min",
                        "2",
                        "--ttl-hours",
                        "1",
                        "--disk",
                        "20",
                        "--name-hint",
                        NAME_HINT,
                        "--no-reuse",
                    ]
                )
                == 0
            )
            log(f"submit returned after {time.monotonic() - started:.0f}s")

            entries = [e for e in load_registry().hosts.values() if e.kind == "runpod"]
            assert len(entries) == 1, entries
            entry = entries[0]
            assert entry.pod_id in provider.created_ids
            job_ids = [e.job_id for e in LocalIndex().list()]
            assert len(job_ids) == 1
            job_id = job_ids[0]

            pod = provider.get(entry.pod_id)
            assert pod is not None
            log(
                f"pod {pod.id} {pod.name}: {pod.gpu_name} cuda {pod.cuda_version} "
                f"${pod.cost_usd_hr:.3f}/h, {len(entry.gpus)} GPU(s) {entry.gpus}"
            )
            assert [d.pod_id for d in load_desired()] == [pod.id]

            assert main(["status"]) == 0
            assert main(["pods"]) == 0

            log(f"waiting for job {job_id}")
            wait_until(
                lambda: job_state(entry, live_settings, job_id)["status"] == "running",
                JOB_TIMEOUT_S,
                "the job to start running",
            )
            assert main(["logs", job_id, "-n", "20"]) == 0
            elapsed = wait_until(
                lambda: (
                    job_state(entry, live_settings, job_id)["status"]
                    in ("succeeded", "failed", "cancelled")
                ),
                JOB_TIMEOUT_S,
                "the job to finish",
            )
            final = job_state(entry, live_settings, job_id)
            log(f"job finished after {elapsed:.0f}s: {final}")
            if final["status"] != "succeeded":
                main(["logs", job_id, "-n", "80"])
            assert final["status"] == "succeeded", final

            objects = s3_client().list_objects_v2(
                Bucket=BUCKET, Prefix=f"{OUTPUT_PREFIX}/{job_id}/"
            )
            keys = [item["Key"] for item in objects.get("Contents", [])]
            log(f"s3://{BUCKET}/{OUTPUT_PREFIX}/{job_id}/ holds {keys}")
            assert any(key.endswith("results/gpu.txt") for key in keys)

            log("waiting for the host to terminate itself after 2 idle minutes")
            idle = wait_until(
                lambda: _gone(provider, pod.id), IDLE_TIMEOUT_S, "the pod to self-terminate"
            )
            log(f"pod gone {idle:.0f}s after the job finished")
            assert pod.id not in [p.id for p in provider.list_ours()]

            assert main(["reconcile", "--once"]) == 0
            assert load_registry().hosts == {}
            assert list(desired_dir().glob("*.json")) == []
            log(f"billing: {json.dumps(provider.billing(pod.id))[:400]}")
            log(f"total wall time {time.monotonic() - started:.0f}s")
    finally:
        with capsys.disabled():
            for pod_id in provider.created_ids:
                try:
                    provider.terminate(pod_id)
                    log(f"finally: {pod_id} terminated")
                # Teardown must never mask the real failure, whatever went wrong.
                except Exception as exc:
                    log(f"finally: terminate of {pod_id} failed: {exc}")
            live = [p for p in provider.list_ours() if p.id in provider.created_ids]
            log(f"finally: our live pods = {[p.name for p in live]}")
            everything = [(p.name, p.status) for p in provider.list_ours()]
            log(f"finally: all prefixed pods = {everything}")
            assert live == []


def _gone(provider: RunPodProvider, pod_id: str) -> bool:
    pod = provider.get(pod_id)
    return pod is None or pod.status == "TERMINATED"
