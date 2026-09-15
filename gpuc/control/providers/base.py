from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import UTC, datetime, timedelta
from typing import Literal

from pydantic import BaseModel, Field

Cloud = Literal["SECURE", "COMMUNITY"]
Availability = Literal["NONE", "LOW", "MEDIUM", "HIGH"]
PodStatus = Literal["PROVISIONING", "STARTING", "RUNNING", "EXITED", "ERROR", "TERMINATED"]

DEFAULT_IMAGE = "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"
DEFAULT_PREFIX = "gpuc-"


def cuda_key(version: str) -> tuple[int, ...]:
    """CUDA versions compare numerically per component, so 12.11 is above 12.2."""
    return tuple(int(part) for part in version.split("."))


class ProviderError(RuntimeError):
    pass


class CapsExceeded(ProviderError):
    pass


class Constraints(BaseModel):
    gpu_names: list[str] = Field(default_factory=list)
    min_vram_gb: int | None = None
    max_price_usd_hr: float | None = None
    clouds: list[Cloud] = Field(default_factory=lambda: ["SECURE"])
    cuda_min: str | None = None
    gpu_count: int = 1


class Offer(BaseModel):
    gpu_id: str
    name: str
    vram_gb: int
    price_usd_hr: float
    cloud: Cloud
    availability: Availability
    cuda_versions: list[str]

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


class Caps(BaseModel):
    prefix: str = DEFAULT_PREFIX
    max_pods: int = 3
    max_total_usd_per_hour: float = 3.0


def owned_pods(pods: list[Pod], prefix: str = DEFAULT_PREFIX) -> list[Pod]:
    return [p for p in pods if p.name.startswith(prefix)]


def check_caps(caps: Caps, pods: list[Pod], new_price_usd_hr: float) -> None:
    ours = [p for p in owned_pods(pods, caps.prefix) if p.status != "TERMINATED"]
    if len(ours) + 1 > caps.max_pods:
        raise CapsExceeded(
            f"max_pods={caps.max_pods} would be exceeded: "
            f"{len(ours)} pods with prefix {caps.prefix!r} already exist "
            f"({', '.join(p.name for p in ours)})"
        )
    total = sum(p.cost_usd_hr for p in ours) + new_price_usd_hr
    if total > caps.max_total_usd_per_hour:
        raise CapsExceeded(
            f"max_total_usd_per_hour={caps.max_total_usd_per_hour} would be exceeded: "
            f"${sum(p.cost_usd_hr for p in ours):.2f}/h running + ${new_price_usd_hr:.2f}/h new "
            f"= ${total:.2f}/h"
        )


class Provider(ABC):
    caps: Caps

    def list_ours(self) -> list[Pod]:
        return owned_pods(self.list(), self.caps.prefix)

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
        cuda_min: str | None = None,
        gpu_count: int = 1,
    ) -> Pod: ...

    @abstractmethod
    def get(self, pod_id: str) -> Pod | None: ...

    @abstractmethod
    def logs(self, pod_id: str, tail: int = 100) -> str: ...

    @abstractmethod
    def terminate(self, pod_id: str) -> None: ...

    @abstractmethod
    def list(self) -> list[Pod]: ...

    @abstractmethod
    def ensure_ssh_key(self, public_key: str) -> bool: ...
