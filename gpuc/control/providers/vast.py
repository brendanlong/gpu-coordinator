"""Vast.ai, through its REST API (`https://console.vast.ai/api/v0`).

Vast is a marketplace: an offer is one machine, not a GPU type, so the search
does the filtering (`Constraints.minimums` included) and every offer carries
the id `create` rents it by. Machines are of uneven quality, which is what the
floors and the health check before handoff are for.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from gpuc._version import user_agent

from .base import (
    DEFAULT_CUDA_MIN,
    DEFAULT_PREFIX,
    Constraints,
    CreateRefused,
    Offer,
    Pod,
    Provider,
    ProviderError,
    SshEndpoint,
    Unauthorized,
    offer_satisfies,
)

BASE_URL = "https://console.vast.ai/api/v0"
IMAGE = "vastai/base-image:@vastai-automatic-tag"
"""Vast's own base image; the automatic tag picks a CUDA build the machine's
driver runs. gpuc's bootstrap brings uv and a Python, and each job its own
torch, so nothing more is needed in the image."""
REQUEST_TIMEOUT_S = 30.0
RATE_LIMIT_RETRIES = 5
OFFER_LIMIT = 10_000
"""Every offer, in effect: the search cannot match GPU names loosely, so
names are matched here, over all of them (~600 for one GPU, 1.7 MB)."""
MB_PER_GB = 1000
"""`gpu_ram` is in MB, and a little under the card's name in MiB: a 24 GB
RTX 4090 is 24564 and an 80 GB H100 81559, so `--min-gpu-ram 24` would miss
the card it names at 1024."""
LOG_FETCH_ATTEMPTS = 5
LOG_FETCH_WAIT_S = 2.0

SEARCH_FIELDS: dict[str, str] = {
    "gpu_ram_gb": "gpu_ram",
    "tflops": "total_flops",
    "inet_down_mbps": "inet_down",
    "inet_up_mbps": "inet_up",
    "disk_bw_mbps": "disk_bw",
    "reliability": "reliability",
}
"""Each `MINIMUMS` key, by the search field it is a floor on."""


class VastError(ProviderError):
    def __init__(self, method: str, url: str, status: int, body: str) -> None:
        super().__init__(f"{method} {url} -> HTTP {status}: {body[:500]}")
        self.status = status


class VastProvider(Provider):
    """`sleep` is every pause taken on the API's behalf (a 429's backoff), so
    a test can run the whole flow without waiting them out."""

    name = "vast"
    api_key_env = "VAST_API_KEY"
    default_image = IMAGE
    filters = frozenset(SEARCH_FIELDS)
    running_statuses = ("running",)
    dead_statuses = ("exited", "stopped", "destroyed")
    """Not `offline` or `unknown`: those are Vast losing touch with the
    machine, which may still be running a job over ssh."""
    gone_statuses = ("destroyed",)

    def __init__(
        self,
        api_key: str | None = None,
        *,
        prefix: str = DEFAULT_PREFIX,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        key = api_key or os.environ.get(self.api_key_env)
        if not key:
            raise ProviderError(f"{self.api_key_env} is not set")
        self._api_key = key
        self.prefix = prefix
        self._sleep = sleep

    def _json(self, method: str, path: str, body: Any = None) -> Any:
        url = BASE_URL + path
        data = json.dumps(body).encode() if body is not None else None
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Accept": "application/json",
            "User-Agent": user_agent(),
        }
        if data is not None:
            headers["Content-Type"] = "application/json"
        for attempt in range(RATE_LIMIT_RETRIES):
            request = urllib.request.Request(url, data=data, headers=headers, method=method)
            try:
                with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_S) as response:
                    raw = response.read()
            except urllib.error.HTTPError as error:
                text = error.read().decode(errors="replace")
                if error.code == 429 and attempt < RATE_LIMIT_RETRIES - 1:
                    self._sleep(2.0 * (attempt + 1))
                    continue
                if error.code == 401 or (error.code == 404 and '"auth_error"' in text):
                    # Vast answers a key it does not know with a 404, and a
                    # 404 read as "no such instance" forgets a rental that is
                    # still billing.
                    raise Unauthorized(
                        f"{method} {url} -> HTTP {error.code}: {text[:500]}"
                    ) from error
                raise VastError(method, url, error.code, text) from error
            except OSError as error:
                raise ProviderError(f"{method} {url}: {error}") from error
            try:
                return json.loads(raw) if raw else None
            except ValueError as error:
                raise ProviderError(f"{method} {url}: unreadable answer: {error}") from error
        raise AssertionError("unreachable: the last attempt raises")

    def offers(self, constraints: Constraints) -> list[Offer]:
        """Search on everything but the GPU names, which Vast matches only
        as it spells them (`RTX 4090`, never `rtx4090`), then keep the offers
        `offer_satisfies` takes: the same rule a registered rental is
        matched by, so the two never disagree."""
        payload = self._json("POST", "/bundles/", search_query(constraints))
        found = [offer_from_api(row) for row in (payload or {}).get("offers") or []]
        kept = [offer for offer in found if offer_satisfies(offer, constraints)]
        if constraints.gpu_names and not kept:
            on_offer = sorted({offer.name for offer in found})
            raise ProviderError(
                f"no Vast offer that meets the other constraints is a "
                f"{' or '.join(repr(n) for n in constraints.gpu_names)} (the GPUs that do: "
                f"{', '.join(on_offer) or 'none'})"
            )
        kept.sort(key=lambda offer: (offer.price_usd_hr, offer.offer_id))
        return kept

    def create(
        self,
        offer: Offer,
        name: str,
        *,
        image: str,
        disk_gb: int,
        cuda_min: str = DEFAULT_CUDA_MIN,
        gpu_count: int = 1,
    ) -> Pod:
        """Rent the machine behind `offer`. `cuda_min` and `gpu_count` were
        the search's: a Vast offer is already one machine with them."""
        if not name.startswith(self.prefix):
            raise ProviderError(f"instance label {name!r} must start with {self.prefix!r}")
        body = {
            "image": image,
            "disk": disk_gb,
            "label": name,
            "runtype": "ssh_direct",
            "cancel_unavail": True,
        }
        try:
            created = self._json("PUT", f"/asks/{offer.offer_id}/", body)
        except VastError as error:
            # An offer someone else rented first is a 404 or a 410, and the
            # ordinary outcome on a marketplace: nothing was created.
            if 400 <= error.status < 500 and error.status != 429:
                raise CreateRefused(str(error)) from error
            raise
        contract = (created or {}).get("new_contract")
        if not contract:
            raise ProviderError(f"PUT /asks/{offer.offer_id}/: unexpected answer: {created!r}")
        return Pod(
            id=str(contract),
            name=name,
            status="created",
            cost_usd_hr=offer.price_usd_hr,
            gpu_name=offer.name,
            gpu_count=gpu_count,
        )

    def get(self, pod_id: str) -> Pod | None:
        """None only when Vast answers that the account has no such
        instance (`instances: null`); every error raises, since a rental
        read as gone over one is forgotten while it bills."""
        payload = self._json("GET", f"/instances/{pod_id}/")
        if not isinstance(payload, dict) or "instances" not in payload:
            raise ProviderError(f"GET /instances/{pod_id}/: unexpected answer: {payload!r}")
        instance = payload["instances"]
        if instance is None:
            return None
        if not isinstance(instance, dict):
            raise ProviderError(f"GET /instances/{pod_id}/: unexpected answer: {payload!r}")
        return pod_from_api(instance)

    def logs(self, pod_id: str, tail: int = 100) -> str:
        """The container log, which Vast writes to a URL it hands back and
        fills a moment later."""
        answer = self._json("PUT", f"/instances/request_logs/{pod_id}/", {"tail": str(tail)})
        url = (answer or {}).get("result_url")
        if not url:
            return ""
        for _ in range(LOG_FETCH_ATTEMPTS):
            request = urllib.request.Request(url, headers={"User-Agent": user_agent()})
            try:
                with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_S) as response:
                    return response.read().decode(errors="replace")
            except urllib.error.HTTPError as error:
                if error.code not in (403, 404):
                    raise ProviderError(f"GET {url} -> HTTP {error.code}") from error
            except OSError as error:
                raise ProviderError(f"GET {url}: {error}") from error
            self._sleep(LOG_FETCH_WAIT_S)
        return ""

    def terminate(self, pod_id: str) -> None:
        """Destroy, not stop: a stopped instance goes on billing for its disk."""
        try:
            self._json("DELETE", f"/instances/{pod_id}/")
        except VastError as error:
            if error.status != 404:
                raise

    def list(self) -> list[Pod]:
        payload = self._json("GET", "/instances/")
        return [pod_from_api(row) for row in (payload or {}).get("instances") or []]

    def ensure_ssh_key(self, public_key: str) -> bool:
        """The account's keys go into every instance it creates."""
        entry = public_key.strip()
        blob = entry.split()[1]
        registered = self._json("GET", "/ssh/") or []
        keys = registered if isinstance(registered, list) else registered.get("keys") or []
        for key in keys:
            if key.get("deleted_at"):
                continue
            # `public_key` is what the API answered when this was written;
            # its documentation says `key`.
            parts = str(key.get("public_key") or key.get("key") or "").split()
            if len(parts) > 1 and parts[1] == blob:
                return False
        self._json("POST", "/ssh/", {"ssh_key": entry})
        return True


def search_query(constraints: Constraints) -> dict[str, Any]:
    """The `bundles` search for these constraints: on-demand, verified
    machines with a directly reachable port (rsync cannot go through Vast's
    ssh proxy), room for the disk, and every floor asked for."""
    query: dict[str, Any] = {
        "type": "on-demand",
        "limit": OFFER_LIMIT,
        "order": [["dph_total", "asc"]],
        "verified": {"eq": True},
        "rentable": {"eq": True},
        "rented": {"eq": False},
        "num_gpus": {"eq": constraints.gpu_count},
        "direct_port_count": {"gte": 1},
        "disk_space": {"gte": constraints.disk_gb},
        "cuda_max_good": {"gte": float(constraints.cuda_min)},
        "cpu_arch": {"eq": "amd64"},
        # Prices include storage, at Vast's default size unless told the disk.
        "allocated_storage": constraints.disk_gb,
    }
    if constraints.max_price_usd_hr is not None:
        query["dph_total"] = {"lte": constraints.max_price_usd_hr}
    for key, floor in constraints.minimums.items():
        query[SEARCH_FIELDS[key]] = {"gte": floor * MB_PER_GB if key == "gpu_ram_gb" else floor}
    return query


def offer_from_api(row: dict[str, Any]) -> Offer:
    gpu_ram = row.get("gpu_ram")
    cuda = row.get("cuda_max_good")
    return Offer(
        gpu_id=str(row.get("gpu_name") or ""),
        name=str(row.get("gpu_name") or ""),
        price_usd_hr=float(row.get("dph_total") or 0.0),
        availability="HIGH",
        cuda_versions=[f"{float(cuda):g}"] if cuda else [],
        offer_id=str(row.get("id") or ""),
        gpu_ram_gb=round(float(gpu_ram) / MB_PER_GB, 1) if gpu_ram else None,
        tflops=_number(row.get("total_flops")),
        inet_down_mbps=_number(row.get("inet_down")),
        inet_up_mbps=_number(row.get("inet_up")),
        disk_bw_mbps=_number(row.get("disk_bw")),
        reliability=_number(row.get("reliability2", row.get("reliability"))),
    )


def _number(value: Any) -> float | None:
    return float(value) if isinstance(value, int | float) else None


def pod_status(instance: dict[str, Any]) -> str:
    """One word for an instance: `destroyed` once its contract is released,
    else what its container is doing (`created` before Vast says)."""
    if instance.get("cur_state") == "unloaded":
        return "destroyed"
    return str(instance.get("actual_status") or "created")


def pod_from_api(instance: dict[str, Any]) -> Pod:
    started = instance.get("start_date")
    utils = instance.get("gpu_util")
    return Pod(
        id=str(instance["id"]),
        name=str(instance.get("label") or ""),
        status=pod_status(instance),
        cost_usd_hr=float(instance.get("dph_total") or 0.0),
        gpu_name=instance.get("gpu_name"),
        gpu_count=int(instance.get("num_gpus") or 0),
        cuda_version=f"{float(instance['cuda_max_good']):g}"
        if instance.get("cuda_max_good")
        else None,
        ssh_direct=_direct_ssh(instance),
        gpu_utils=[round(float(utils))] if isinstance(utils, int | float) else [],
        created_at=datetime.fromtimestamp(float(started), UTC)
        if isinstance(started, int | float)
        else None,
    )


def _direct_ssh(instance: dict[str, Any]) -> SshEndpoint | None:
    """The machine's own address and the host port its 22 is mapped to; None
    until Vast has mapped it."""
    address = instance.get("public_ipaddr")
    ports = instance.get("ports")
    bindings = ports.get("22/tcp") if isinstance(ports, dict) else None
    mapped = bindings[0].get("HostPort") if isinstance(bindings, list) and bindings else None
    if not address or not mapped:
        return None
    return SshEndpoint(host=str(address).strip(), port=int(mapped), username="root")
