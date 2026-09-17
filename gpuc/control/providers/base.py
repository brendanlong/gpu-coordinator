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


class Constraints(BaseModel):
    gpu_names: list[str] = Field(default_factory=list)
    min_vram_gb: int | None = None
    max_price_usd_hr: float | None = None
    clouds: list[Cloud] = Field(default_factory=lambda: ["SECURE"])
    cuda_min: str | None = None
    gpu_count: int = 1


class Offer(BaseModel):
    """One catalog entry. Every field has a default so the `offer` a pod's own
    config records, written by another build, still parses: a pod we cannot
    reuse is a smaller failure than one we cannot read."""

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
    prefix: str

    def list_ours(self) -> list[Pod]:
        return owned_pods(self.list(), self.prefix)

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
