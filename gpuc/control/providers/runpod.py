from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from datetime import datetime
from http.client import HTTPResponse
from typing import Any

from gpuc._version import user_agent

from .base import (
    DEFAULT_CUDA_MIN,
    DEFAULT_IMAGE,
    DEFAULT_PREFIX,
    Cloud,
    Constraints,
    Offer,
    Pod,
    Provider,
    ProviderError,
    SshEndpoint,
    cuda_key,
)

LOG_READ_S = 10.0
"""How long a log tail reads the SSE stream: it stays open after the backfill."""
DEFAULT_POD_ENV = {"HF_HUB_ENABLE_HF_TRANSFER": "0"}
"""hf_transfer is not installed in the image, and `hf` fails loudly when told to use it."""
BASE_URL = "https://api.runpod.io/v2"
# Cloudflare in front of api.runpod.io rejects the default urllib User-Agent with a 1010.
USER_AGENT = user_agent()
MAX_RATE_LIMIT_SLEEP_S = 60.0


class RunPodError(ProviderError):
    def __init__(self, method: str, url: str, status: int, body: str) -> None:
        super().__init__(f"{method} {url} -> HTTP {status}: {body[:500]}")
        self.status = status
        self.body = body


class PodNotFound(RunPodError):
    pass


def _rate_limit_pause(header: str | None) -> float:
    """`RateLimit: "minute";r=0;t=12, ...` — wait out any window with no requests left."""
    if not header:
        return 0.0
    waits = [float(t) for r, t in re.findall(r"r=(\d+);t=(\d+)", header) if int(r) == 0]
    return min(max(waits), MAX_RATE_LIMIT_SLEEP_S) if waits else 0.0


class RunPodProvider(Provider):
    """The v2 REST API. `sleep` is every pause the client takes on the API's
    behalf -- a `Retry-After`, a rate-limit window -- so a test can run the
    whole flow without waiting them out."""

    name = "runpod"

    def __init__(
        self,
        api_key: str | None = None,
        *,
        prefix: str = DEFAULT_PREFIX,
        timeout_s: float = 30.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        key = api_key or os.environ.get("RUNPOD_API_KEY")
        if not key:
            raise ProviderError("RUNPOD_API_KEY is not set")
        self._api_key = key
        self._base_url = BASE_URL.rstrip("/")
        self.prefix = prefix
        self._timeout_s = timeout_s
        self._sleep = sleep

    def _open(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        body: Any = None,
        accept: str = "application/json",
        timeout_s: float | None = None,
    ) -> HTTPResponse:
        url = self._base_url + path
        if params:
            url += "?" + urllib.parse.urlencode(params)
        data = json.dumps(body).encode() if body is not None else None
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Accept": accept,
            "User-Agent": USER_AGENT,
        }
        if data is not None:
            headers["Content-Type"] = "application/json"

        for attempt in range(5):
            request = urllib.request.Request(url, data=data, headers=headers, method=method)
            try:
                return urllib.request.urlopen(request, timeout=timeout_s or self._timeout_s)
            except urllib.error.HTTPError as error:
                text = error.read().decode(errors="replace")
                if error.code == 429 and attempt < 4:
                    retry_after = error.headers.get("Retry-After")
                    self._sleep(float(retry_after) if retry_after else 2.0 * (attempt + 1))
                    continue
                if error.code == 404:
                    raise PodNotFound(method, url, error.code, text) from error
                raise RunPodError(method, url, error.code, text) from error
            except OSError as error:
                # A `URLError` (DNS, refused), a socket timeout or a reset:
                # every caller that catches `ProviderError` -- the terminate
                # that must say "still billing", the confirm loop -- would
                # otherwise let it through as a traceback and say nothing.
                raise ProviderError(f"{method} {url}: {error}") from error
        raise AssertionError("unreachable: the last attempt raises")

    def _json(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        body: Any = None,
    ) -> Any:
        with self._open(method, path, params=params, body=body) as response:
            try:
                raw = response.read()
            except OSError as error:
                raise ProviderError(f"{method} {path}: {error}") from error
            pause = _rate_limit_pause(response.headers.get("RateLimit"))
            if pause:
                self._sleep(pause)
        return json.loads(raw) if raw else None

    def offers(self, constraints: Constraints) -> list[Offer]:
        wanted = {name.casefold() for name in constraints.gpu_names}
        found: list[Offer] = []
        for cloud in constraints.clouds:
            params: dict[str, Any] = {
                "include": "AVAILABILITY",
                "product": "POD",
                "cloud": cloud,
                "count": constraints.gpu_count,
                "minCudaVersion": constraints.cuda_min,
            }
            payload = self._json("GET", "/catalog/gpus", params=params)
            found.extend(self._offers_from_catalog(payload["gpus"], cloud, wanted, constraints))
        found.sort(key=lambda offer: (offer.price_usd_hr, offer.gpu_id, offer.cloud))
        return found

    @staticmethod
    def _offers_from_catalog(
        gpus: list[dict[str, Any]],
        cloud: Cloud,
        wanted: set[str],
        constraints: Constraints,
    ) -> list[Offer]:
        tier = cloud.lower()
        offers: list[Offer] = []
        for gpu in gpus:
            if wanted and not wanted & {gpu["id"].casefold(), gpu["name"].casefold()}:
                continue
            if gpu.get("availability", "NONE") == "NONE":
                continue
            if constraints.min_vram_gb is not None and gpu["memory"] < constraints.min_vram_gb:
                continue
            price = gpu["price"].get(tier)
            if price is None or price <= 0:
                continue
            # --max-price caps the whole pod, so compare the pod's price, not
            # one GPU's: at --gpu-count 4 the per-GPU figure is off by 4x.
            total_price = price * constraints.gpu_count
            if (
                constraints.max_price_usd_hr is not None
                and total_price > constraints.max_price_usd_hr
            ):
                continue
            if gpu["maxCount"].get(tier, 0) < constraints.gpu_count:
                continue
            cuda_versions = sorted(
                (v["version"] for v in gpu.get("cudaVersions", []) if v["available"]),
                key=cuda_key,
            )
            offer = Offer(
                gpu_id=gpu["id"],
                name=gpu["name"],
                vram_gb=gpu["memory"],
                price_usd_hr=total_price,
                cloud=cloud,
                availability=gpu["availability"],
                cuda_versions=cuda_versions,
            )
            if not offer.cuda_versions or not offer.matches_cuda_floor(constraints.cuda_min):
                continue
            offers.append(offer)
        return offers

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
    ) -> Pod:
        if not name.startswith(self.prefix):
            raise ProviderError(f"pod name {name!r} must start with {self.prefix!r}")
        gpu: dict[str, Any] = {"id": offer.gpu_id, "count": gpu_count, "minCudaVersion": cuda_min}
        body = {
            "name": name,
            "image": image,
            "gpu": gpu,
            "cloud": offer.cloud,
            "disk": disk_gb,
            "ports": ["22/tcp"],
            "startSsh": True,
            "env": env or DEFAULT_POD_ENV,
        }
        return _pod_from_api(self._json("POST", "/pods", body=body))

    def get(self, pod_id: str) -> Pod | None:
        try:
            return _pod_from_api(self._json("GET", f"/pods/{pod_id}"))
        except PodNotFound:
            return None

    def logs(self, pod_id: str, tail: int = 100) -> str:
        lines: list[str] = []
        deadline = time.monotonic() + LOG_READ_S
        try:
            stream = self._open(
                "GET",
                f"/pods/{pod_id}/logs",
                params={"tail": tail},
                accept="text/event-stream",
                timeout_s=LOG_READ_S,
            )
            with stream:
                for raw in stream:
                    text = raw.decode(errors="replace").strip()
                    if text.startswith("data:"):
                        event = json.loads(text[len("data:") :])
                        lines.append(f"{event.get('source', '?')}: {event.get('line', '')}")
                    if time.monotonic() >= deadline:
                        break
        except (TimeoutError, OSError):
            pass  # SSE stays open after the backfill; the read timeout is the end of the tail
        return "\n".join(lines)

    def terminate(self, pod_id: str) -> None:
        """One POST. A 404 is a pod already gone and a 409 one already ending,
        and both are what `terminate_confirmed`'s poll then confirms."""
        try:
            self._json("POST", f"/pods/{pod_id}/action", body={"action": "terminate"})
        except PodNotFound:
            return
        except RunPodError as error:
            if error.status != 409:
                raise

    def list(self) -> list[Pod]:
        payload = self._json("GET", "/pods", params={"includeClusterPods": "true"})
        return [_pod_from_api(pod) for pod in payload["pods"]]

    def ensure_ssh_key(self, public_key: str) -> bool:
        entry = public_key.strip()
        blob = entry.split()[1]
        registered: list[str] = self._json("GET", "/account/ssh-keys")["keys"]
        if any(len(k.split()) > 1 and k.split()[1] == blob for k in registered):
            return False
        self._json("PUT", "/account/ssh-keys", body={"keys": [*registered, entry]})
        return True

    def billing(self, pod_id: str) -> dict[str, Any]:
        return self._json(
            "GET", "/billing/pods", params={"podId": pod_id, "bucketSize": "hour", "lastN": 24}
        )


def _pod_from_api(payload: dict[str, Any]) -> Pod:
    direct = (payload.get("ssh") or {}).get("direct")
    runtime = payload.get("runtime") or {}
    created_at = payload.get("createdAt")
    gpu = payload.get("gpu") or {}
    machine = payload.get("machine") or {}
    return Pod(
        id=payload["id"],
        name=payload["name"],
        status=payload["status"],
        cost_usd_hr=payload.get("cost") or 0.0,
        gpu_name=gpu.get("id") or machine.get("gpuTypeId") or payload.get("gpuTypeId"),
        gpu_count=int(gpu.get("count") or payload.get("gpuCount") or 0),
        cuda_version=payload.get("cudaVersion"),
        ssh_direct=SshEndpoint(**{k: direct[k] for k in ("host", "port", "username")})
        if direct
        else None,
        gpu_utils=[gpu.get("util", 0) for gpu in runtime.get("gpus") or []],
        created_at=datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        if created_at
        else None,
    )
