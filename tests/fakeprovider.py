"""An in-memory provider and transport, so the whole provisioning flow is testable offline.

Pods follow a script: how many `get` polls before `ssh.direct` appears, what
the pod log says, whether `create` fails with a capacity error. That is enough
to reproduce every failure the real flow has to survive.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from gpuc.control.bootstrap import BootstrapResult
from gpuc.control.config import HostEntry, Settings, utc_now
from gpuc.control.providers.base import (
    DEFAULT_IMAGE,
    Caps,
    Cloud,
    Constraints,
    Offer,
    Pod,
    PodStatus,
    Provider,
    ProviderError,
    SshEndpoint,
)
from gpuc.control.provision import offer_satisfies
from gpuc.control.transport import CommandResult, Transport, TransportError

CAPACITY_ERROR = "no capacity for this gpu type right now"
BROKEN_LOG = "system: error: failed to create shim task: OCI runtime create failed"


def make_offer(
    name: str = "A40",
    price: float = 0.49,
    cloud: Cloud = "SECURE",
    vram_gb: int = 48,
    cuda_versions: tuple[str, ...] = ("12.8",),
    gpu_id: str | None = None,
) -> Offer:
    """The catalog's `name` is short ("A40") and its `id` is long ("NVIDIA A40")."""
    return Offer(
        gpu_id=gpu_id or ("NVIDIA A40" if name == "A40" else name),
        name=name,
        vram_gb=vram_gb,
        price_usd_hr=price,
        cloud=cloud,
        availability="HIGH",
        cuda_versions=list(cuda_versions),
    )


@dataclass
class PodScript:
    """How the pod created for one offer behaves."""

    ssh_after_polls: int = 1
    log_text: str = ""
    create_error: str | None = None
    status_after_polls: dict[int, PodStatus] = field(default_factory=dict)
    cuda_version: str = "12.8"


@dataclass
class _FakePod:
    pod: Pod
    script: PodScript
    polls: int = 0


class FakeProvider(Provider):
    def __init__(
        self,
        offers: list[Offer] | None = None,
        *,
        caps: Caps | None = None,
        scripts: list[PodScript] | None = None,
        existing: list[Pod] | None = None,
    ) -> None:
        self.caps = caps or Caps()
        self._offers = offers if offers is not None else [make_offer()]
        self._scripts = iter(scripts or [])
        self._ids = itertools.count(1)
        self._pods: dict[str, _FakePod] = {}
        self.foreign = list(existing or [])
        self.created: list[dict[str, Any]] = []
        self.terminated: list[str] = []
        self.registered_keys: list[str] = []

    # -- Provider ---------------------------------------------------------
    def offers(self, constraints: Constraints) -> list[Offer]:
        """Filtered, like a real catalog query.

        A fake that hands back everything lets a selection test pass on an
        offer the real provider would never have shown it -- which is the one
        thing those tests exist to check. `offer_satisfies` is the same
        predicate `provision` applies, and `availability` is the one further
        ground `RunpodProvider._offers_from_catalog` drops a GPU on.
        """
        return sorted(
            (
                offer
                for offer in self._offers
                if offer.availability != "NONE" and offer_satisfies(offer, constraints)
            ),
            key=lambda o: (o.price_usd_hr, o.gpu_id, o.cloud),
        )

    def create(
        self,
        offer: Offer,
        name: str,
        *,
        image: str = DEFAULT_IMAGE,
        disk_gb: int = 20,
        env: dict[str, str] | None = None,
        cuda_min: str | None = None,
        gpu_count: int = 1,
    ) -> Pod:
        if not name.startswith(self.caps.prefix):
            raise ProviderError(f"pod name {name!r} must start with {self.caps.prefix!r}")
        script = next(self._scripts, PodScript())
        self.created.append(
            {
                "name": name,
                "offer": offer,
                "image": image,
                "disk_gb": disk_gb,
                "env": env,
                "cuda_min": cuda_min,
                "gpu_count": gpu_count,
            }
        )
        if script.create_error:
            raise ProviderError(f"{script.create_error} ({offer.gpu_id})")
        pod = Pod(
            id=f"pod{next(self._ids)}",
            name=name,
            status="PROVISIONING",
            cost_usd_hr=offer.price_usd_hr,
            gpu_name=offer.gpu_id,
            gpu_count=gpu_count,
            cuda_version=script.cuda_version,
            created_at=datetime.now(UTC),
        )
        self._pods[pod.id] = _FakePod(pod=pod, script=script)
        return pod

    def get(self, pod_id: str) -> Pod | None:
        entry = self._pods.get(pod_id)
        if entry is None:
            return next((p for p in self.foreign if p.id == pod_id), None)
        if entry.pod.status == "TERMINATED":
            return entry.pod
        entry.polls += 1
        status = entry.script.status_after_polls.get(entry.polls)
        if status is not None:
            entry.pod = entry.pod.model_copy(update={"status": status})
        elif entry.pod.status == "PROVISIONING":
            entry.pod = entry.pod.model_copy(update={"status": "RUNNING"})
        ready = entry.script.ssh_after_polls >= 0 and entry.polls >= entry.script.ssh_after_polls
        if ready and entry.pod.ssh_direct is None and entry.pod.status == "RUNNING":
            entry.pod = entry.pod.model_copy(
                update={
                    "ssh_direct": SshEndpoint(host="1.2.3.4", port=22000, username="root"),
                    "gpu_utils": [0],
                }
            )
        return entry.pod

    def logs(self, pod_id: str, tail: int = 100) -> str:
        entry = self._pods.get(pod_id)
        return entry.script.log_text if entry else ""

    def terminate(self, pod_id: str) -> None:
        entry = self._pods.get(pod_id)
        if entry is None:
            raise ProviderError(f"unknown pod {pod_id}")
        self.terminated.append(pod_id)
        entry.pod = entry.pod.model_copy(update={"status": "TERMINATED"})

    def list(self) -> list[Pod]:
        return [entry.pod for entry in self._pods.values()] + list(self.foreign)

    def ensure_ssh_key(self, public_key: str) -> bool:
        if public_key in self.registered_keys:
            return False
        self.registered_keys.append(public_key)
        return True

    # -- test helpers -----------------------------------------------------
    def adopt(self, pod: Pod, script: PodScript | None = None) -> Pod:
        """Put an already-running pod in the provider, as another session would."""
        self._pods[pod.id] = _FakePod(pod=pod, script=script or PodScript(ssh_after_polls=0))
        return pod

    def live_names(self) -> list[str]:
        return sorted(p.name for p in self.list() if p.status != "TERMINATED")


def running_pod(name: str, pod_id: str, *, cost: float = 0.49, age_minutes: float = 30.0) -> Pod:
    return Pod(
        id=pod_id,
        name=name,
        status="RUNNING",
        cost_usd_hr=cost,
        gpu_name="NVIDIA A40",
        gpu_count=1,
        cuda_version="12.8",
        created_at=datetime.now(UTC) - timedelta(minutes=age_minutes),
        ssh_direct=SshEndpoint(host="1.2.3.4", port=22000, username="root"),
    )


@dataclass
class FakeTransport:
    """Answers the handful of commands provisioning runs, and can refuse ssh N times."""

    host: str = "fake"
    ssh_failures: int = 0
    gpu_uuids: tuple[str, ...] = ("GPU-1111", "GPU-2222")
    home: str = "/root"
    commands: list[str] = field(default_factory=list)
    files: dict[str, str] = field(default_factory=dict)

    def run(self, command: str, *, timeout: float = 120.0, check: bool = True) -> CommandResult:
        """`check` means the same thing here as in Local/SshTransport: raise.

        A fake that quietly returned the failure would let caller code that
        forgot `check=False` look correct in tests and blow up in production.
        """
        self.commands.append(command)
        result = self._answer(command)
        if check and result.returncode != 0:
            raise TransportError(result)
        return result

    def _answer(self, command: str) -> CommandResult:
        if command == "true":
            if self.ssh_failures > 0:
                self.ssh_failures -= 1
                return CommandResult(self.host, ["ssh", command], 255, "", "connection refused")
            return CommandResult(self.host, ["ssh", command], 0, "", "")
        if command.startswith("nvidia-smi"):
            return CommandResult(
                self.host, ["ssh", command], 0, "".join(f"{u}\n" for u in self.gpu_uuids), ""
            )
        if "$HOME" in command:
            return CommandResult(self.host, ["ssh", command], 0, self.home, "")
        return CommandResult(self.host, ["ssh", command], 0, "", "")

    def put_file(self, content: str | bytes, remote_path: str, mode: int = 0o600) -> None:
        self.files[remote_path] = content if isinstance(content, str) else content.decode()

    def rsync(
        self,
        local_root: Path,
        remote_path: str,
        files: Any = None,
        excludes: Sequence[str] = (),
    ) -> CommandResult:
        return CommandResult(self.host, ["rsync"], 0, "", "")

    def tail(self, remote_path: str, lines: int = 200, follow: bool = False) -> CommandResult:
        return CommandResult(self.host, ["tail"], 0, "", "")


def fake_bootstrap(
    entry: HostEntry,
    settings: Settings | None = None,
    *,
    transport: Transport | None = None,
    report: Any = print,
    health_args: str = "",
) -> tuple[HostEntry, BootstrapResult]:
    report(f"fake bootstrap of {entry.name}")
    updated = entry.model_copy(
        update={
            "python": "/root/.venv/bin/python",
            "uv": "/root/.local/bin/uv",
            "bootstrapped_at": utc_now(),
        }
    )
    return updated, BootstrapResult(
        host=entry.name,
        uv="/root/.local/bin/uv",
        python="/root/.venv/bin/python",
        home="/root/.gpuc",
        files=20,
        health={"ok": True, "checks": []},
        dispatcher_pid=4242,
    )
