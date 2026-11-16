from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from http.client import HTTPResponse
from typing import Any

from gpuc._version import user_agent

from .base import (
    LOG_TAIL_LINES,
    Cloud,
    Constraints,
    CreateRefused,
    HttpError,
    Offer,
    Pod,
    Provider,
    ProviderError,
    SshEndpoint,
    Unauthorized,
    cuda_key,
    gpu_name_key,
    matches_gpu_names,
    offer_satisfies,
)

IMAGE = "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"
LOG_READ_S = 10.0
"""How long a log tail reads the SSE stream: it stays open after the backfill."""
DEFAULT_POD_ENV = {"HF_HUB_ENABLE_HF_TRANSFER": "0"}
"""hf_transfer is not installed in the image, and `hf` fails loudly when told to use it."""
BASE_URL = "https://api.runpod.io/v2"
REQUEST_TIMEOUT_S = 30.0
# Cloudflare in front of api.runpod.io rejects the default urllib User-Agent with a 1010.
USER_AGENT = user_agent()
MAX_RATE_LIMIT_SLEEP_S = 60.0


class PodNotFound(HttpError):
    pass


def _rate_limit_pause(header: str | None) -> float:
    """`RateLimit: "minute";r=0;t=12, ...` — wait out any window with no requests left."""
    if not header:
        return 0.0
    waits = [float(t) for r, t in re.findall(r"r=(\d+);t=(\d+)", header) if int(r) == 0]
    return min(max(waits), MAX_RATE_LIMIT_SLEEP_S) if waits else 0.0


def _check_names_exist(names: list[str], gpus: list[dict[str, Any]]) -> None:
    """The catalog lists every GPU type, stocked or not, so a name that
    matches none of them is misspelled, and saying so beats the "no offers"
    error that reads as an availability problem (#143)."""
    catalog = {gpu_name_key(key): gpu["name"] for gpu in gpus for key in (gpu["id"], gpu["name"])}
    unknown = [name for name in names if gpu_name_key(name) not in catalog]
    if not unknown:
        return
    lines = []
    for name in unknown:
        near = sorted({full for key, full in catalog.items() if gpu_name_key(name) in key})
        hint = f"did you mean {', '.join(near)}?" if near else "no similar name"
        lines.append(f"--gpu {name!r} is not a RunPod GPU type ({hint})")
    known = ", ".join(sorted(set(catalog.values())))
    raise ProviderError("\n".join(lines) + f"\nRunPod GPU types: {known}")


class RunPodProvider(Provider):
    """The v2 REST API."""

    name = "runpod"
    api_key_env = "RUNPOD_API_KEY"
    default_image = IMAGE
    clouds = ("SECURE", "COMMUNITY")

    def _open(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        body: Any = None,
        accept: str = "application/json",
        timeout_s: float = REQUEST_TIMEOUT_S,
    ) -> HTTPResponse:
        url = BASE_URL + path
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
                return urllib.request.urlopen(request, timeout=timeout_s)
            except urllib.error.HTTPError as error:
                text = error.read().decode(errors="replace")
                if error.code == 429 and attempt < 4:
                    retry_after = error.headers.get("Retry-After")
                    self._sleep(float(retry_after) if retry_after else 2.0 * (attempt + 1))
                    continue
                if error.code == 404:
                    raise PodNotFound(method, url, error.code, text) from error
                raise HttpError(method, url, error.code, text) from error
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
        found: list[Offer] = []
        for cloud in constraints.clouds or self.clouds[:1]:
            params: dict[str, Any] = {
                "include": "AVAILABILITY",
                "product": "POD",
                "cloud": cloud,
                "count": constraints.gpu_count,
                "minCudaVersion": constraints.cuda_min,
            }
            payload = self._json("GET", "/catalog/gpus", params=params)
            _check_names_exist(constraints.gpu_names, payload["gpus"])
            found.extend(self._offers_from_catalog(payload["gpus"], cloud, constraints))
        found.sort(key=lambda offer: (offer.price_usd_hr, offer.gpu_id, offer.cloud))
        return found

    @staticmethod
    def _offers_from_catalog(
        gpus: list[dict[str, Any]],
        cloud: Cloud,
        constraints: Constraints,
    ) -> list[Offer]:
        tier = cloud.lower()
        offers: list[Offer] = []
        for gpu in gpus:
            # Before parsing, so a malformed row for a GPU nobody asked for
            # cannot fail the search.
            if not matches_gpu_names(constraints, gpu["id"], gpu["name"]):
                continue
            if gpu.get("availability", "NONE") == "NONE":
                continue
            price = gpu["price"].get(tier)
            if price is None or price <= 0:
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
                # --max-price caps the whole pod, so the offer carries the
                # pod's price, not one GPU's: at --gpu-count 4 that is 4x.
                price_usd_hr=price * constraints.gpu_count,
                cloud=cloud,
                availability=gpu["availability"],
                cuda_versions=cuda_versions,
            )
            if offer_satisfies(offer, constraints):
                offers.append(offer)
        return offers

    def create(
        self,
        offer: Offer,
        name: str,
        *,
        image: str,
        constraints: Constraints,
    ) -> Pod:
        if not name.startswith(self.prefix):
            raise ProviderError(f"pod name {name!r} must start with {self.prefix!r}")
        gpu: dict[str, Any] = {
            "id": offer.gpu_id,
            "count": constraints.gpu_count,
            "minCudaVersion": constraints.cuda_min,
        }
        body = {
            "name": name,
            "image": image,
            "gpu": gpu,
            "cloud": offer.cloud,
            "disk": constraints.disk_gb,
            "ports": ["22/tcp"],
            "startSsh": True,
            "env": DEFAULT_POD_ENV,
        }
        try:
            created = self._json("POST", "/pods", body=body)
        except HttpError as error:
            if error.status in (401, 403):
                raise Unauthorized(str(error)) from error
            if 400 <= error.status < 500:
                raise CreateRefused(str(error)) from error
            raise
        except ValueError as error:
            raise ProviderError(f"POST /pods: unreadable answer: {error}") from error
        try:
            return _pod_from_api(created)
        except (AttributeError, KeyError, TypeError, ValueError) as error:
            raise ProviderError(f"POST /pods: unexpected answer: {error!r}") from error

    def get(self, pod_id: str) -> Pod | None:
        try:
            return _pod_from_api(self._json("GET", f"/pods/{pod_id}"))
        except PodNotFound:
            return None

    def logs(self, pod_id: str) -> str:
        lines: list[str] = []
        deadline = time.monotonic() + LOG_READ_S
        try:
            stream = self._open(
                "GET",
                f"/pods/{pod_id}/logs",
                params={"tail": LOG_TAIL_LINES},
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
        except HttpError as error:
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
