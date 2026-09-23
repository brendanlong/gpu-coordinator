from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Literal

from pydantic import BaseModel, Field

from gpuc.control.tolerant import TolerantModel

Cloud = Literal["SECURE", "COMMUNITY"]
Availability = Literal["NONE", "LOW", "MEDIUM", "HIGH"]
PodStatus = Literal["PROVISIONING", "STARTING", "RUNNING", "EXITED", "ERROR", "TERMINATED"]

DEFAULT_IMAGE = "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"
DEFAULT_PREFIX = "gpuc-"
DEFAULT_CUDA_MIN = "12.8"
"""The CUDA floor a request carries unless `--cuda-min` says otherwise: the
one default, so the catalog query and the create ask for the same thing."""


def cuda_key(version: str) -> tuple[int, ...]:
    """CUDA versions compare numerically per component, so 12.11 is above 12.2."""
    return tuple(int(part) for part in version.split("."))


class ProviderError(RuntimeError):
    pass


class Constraints(BaseModel):
    gpu_names: list[str] = Field(default_factory=list)
    min_vram_gb: int | None = None
    max_price_usd_hr: float | None = None
    clouds: list[Cloud] = Field(default_factory=lambda: ["SECURE"])
    cuda_min: str = DEFAULT_CUDA_MIN
    gpu_count: int = 1


class Offer(TolerantModel):
    """One catalog entry. Read tolerantly because the `offer` a pod's own
    config records was written by whichever build rented it: a pod we cannot
    reuse is a smaller failure than one we cannot read, and one we cannot read
    is one `submit --runpod` buys a second of."""

    gpu_id: str = ""
    name: str = ""
    vram_gb: int = 0
    price_usd_hr: float = 0.0
    cloud: Cloud = "SECURE"
    availability: Availability = "NONE"
    cuda_versions: list[str] = Field(default_factory=list)

    def matches_cuda_floor(self, cuda_min: str | None) -> bool:
        if cuda_min is None:
            return True
        floor = cuda_key(cuda_min)
        return any(cuda_key(v) >= floor for v in self.cuda_versions)


class SshEndpoint(BaseModel):
    host: str
    port: int
    username: str


class Pod(BaseModel):
    id: str
    name: str
    status: PodStatus
    cost_usd_hr: float
    gpu_name: str | None = None
    gpu_count: int = 0
    cuda_version: str | None = None
    ssh_direct: SshEndpoint | None = None
    gpu_utils: list[int] = Field(default_factory=list)
    created_at: datetime | None = None

    @property
    def age(self) -> timedelta | None:
        if self.created_at is None:
            return None
        return datetime.now(UTC) - self.created_at


def owned_pods(pods: list[Pod], prefix: str = DEFAULT_PREFIX) -> list[Pod]:
    return [p for p in pods if p.name.startswith(prefix)]


class Provider(ABC):
    """One rental provider. Everything the control side needs to know about a
    provider's vocabulary lives on the instance, so the provisioning and
    teardown flows are written once and read it from here.

    The defaults below are RunPod's, which every provider so far shares; a
    provider with another vocabulary overrides them, and nothing outside
    `providers/` names a status.
    """

    prefix: str
    dead_statuses: tuple[str, ...] = ("EXITED", "ERROR", "TERMINATED")
    """Pod statuses nothing can run on. A pod in one is a failed host."""
    gone_statuses: tuple[str, ...] = ("TERMINATED",)
    """Statuses that mean the rental has ended: forget the pod, never terminate it."""
    running_statuses: tuple[str, ...] = ("RUNNING",)
    """Statuses under which the pod can be dialled and a job enqueued."""
    broken_host: re.Pattern[str] = re.compile(
        r"card[0-9]|device nodes|OCI runtime|runc create|failed to create shim", re.IGNORECASE
    )
    """Log signatures of a host whose GPU device nodes are broken: re-place, never retry."""
    terminate_attempts: int = 3
    terminate_retry_s: float = 5.0
    confirm_polls: int = 60
    confirm_poll_s: float = 5.0

    def is_dead(self, pod: Pod | None) -> bool:
        return pod is None or pod.status in self.dead_statuses

    def is_gone(self, pod: Pod | None) -> bool:
        return pod is None or pod.status in self.gone_statuses

    def is_running(self, pod: Pod | None) -> bool:
        return pod is not None and pod.status in self.running_statuses

    def list_ours(self) -> list[Pod]:
        return owned_pods(self.list(), self.prefix)

    def terminate_confirmed(
        self,
        pod_id: str,
        *,
        report: Callable[[str], None],
        sleep: Callable[[float], None],
    ) -> None:
        """End `pod_id` and return only once the provider says it is gone.

        The one place a terminate is retried, for the provisioning failure
        path and `gpuc host terminate` alike: a 5xx or a rate limit on the
        call that stops the bill is the worst place to give up after one try,
        and once the caller has moved on nothing tries again. `terminate`
        itself is the bare API call; a pod the provider no longer has counts
        as ended. Raises `ProviderError` when the pod could not be confirmed
        gone, so the caller can say it is still billing.
        """
        for attempt in range(1, self.terminate_attempts + 1):
            try:
                self.terminate(pod_id)
                break
            except ProviderError as exc:
                if attempt == self.terminate_attempts:
                    raise ProviderError(
                        f"could not terminate pod {pod_id} in {attempt} attempts: {exc}"
                    ) from exc
                report(
                    f"terminate {pod_id} failed ({exc}); retrying in {self.terminate_retry_s:g}s"
                )
                sleep(self.terminate_retry_s)
        status = ""
        for _ in range(self.confirm_polls):
            pod = self.get(pod_id)
            if self.is_gone(pod):
                return
            assert pod is not None
            status = pod.status
            sleep(self.confirm_poll_s)
        raise ProviderError(
            f"pod {pod_id} still {status} {self.confirm_polls * self.confirm_poll_s:.0f}s "
            f"after terminate"
        )

    @abstractmethod
    def offers(self, constraints: Constraints) -> list[Offer]: ...

    @abstractmethod
    def create(
        self,
        offer: Offer,
        name: str,
        *,
        image: str = DEFAULT_IMAGE,
        disk_gb: int = 20,
        env: dict[str, str] | None = None,
        cuda_min: str = DEFAULT_CUDA_MIN,
        gpu_count: int = 1,
    ) -> Pod: ...

    @abstractmethod
    def get(self, pod_id: str) -> Pod | None: ...

    @abstractmethod
    def logs(self, pod_id: str, tail: int = 100) -> str: ...

    @abstractmethod
    def terminate(self, pod_id: str) -> None:
        """The bare terminate call, once. `terminate_confirmed` is what callers use."""

    @abstractmethod
    def list(self) -> list[Pod]: ...

    @abstractmethod
    def ensure_ssh_key(self, public_key: str) -> bool: ...
