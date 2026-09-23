"""Self-termination of an ephemeral host via the provider REST API.

urllib only: this runs on a host whose venv may be broken, and leaking a paid
pod is the single most expensive failure mode in the system.

A pod-scoped RUNPOD_API_KEY can terminate its own pod (verified: HTTP 204, and
the pod 404s within two seconds), so no account key needs to be delivered. The
catch is that RunPod puts that key and RUNPOD_POD_ID in /etc/rp_environment,
which only *interactive* shells source -- and the dispatcher is started over a
non-interactive SSH session, so we read that file ourselves.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gpuc.host import USER_AGENT, paths
from gpuc.host.jobs import HostConfig, parse_env_file

RUNPOD_API = "https://api.runpod.io/v2"
DEFAULT_TIMEOUT_S = 60.0

TerminateCall = Callable[[str, str], str]
"""(pod_id, api_key) -> response body. Injectable so tests never hit RunPod."""


class TerminateError(RuntimeError):
    pass


def rp_environment_path() -> Path:
    return Path(os.environ.get("GPUC_RP_ENVIRONMENT", "/etc/rp_environment"))


def rp_environment() -> dict[str, str]:
    return parse_env_file(rp_environment_path())


@dataclass(frozen=True)
class Provider:
    """What ending a rental needs from one provider: the call, and where the
    pod finds its own id and key when the config does not say."""

    terminate: TerminateCall
    api_key_env: str
    pod_id_env: str
    environment_file: Callable[[], dict[str, str]]
    """The provider's own on-pod environment file, sourced by interactive
    shells only, so read here rather than inherited."""


def _provider(kind: str) -> Provider:
    provider = PROVIDERS.get(kind)
    if provider is None:
        raise TerminateError(f"unsupported provider kind: {kind!r}")
    return provider


def read_api_key(kind: str = "runpod") -> str:
    provider = _provider(kind)
    path = paths.secrets_file(kind)
    if path.exists():
        key = path.read_text().strip()
        if key:
            return key
    env_key = os.environ.get(provider.api_key_env, "").strip()
    if env_key:
        return env_key
    pod_key = provider.environment_file().get(provider.api_key_env, "").strip()
    if pod_key:
        return pod_key
    raise TerminateError(
        f"no provider API key: {path} is missing or empty, {provider.api_key_env} is unset, "
        f"and {rp_environment_path()} has no {provider.api_key_env}"
    )


def read_pod_id(config: dict[str, Any] | None = None) -> str:
    """The pod to terminate: the config's first, since that is the rental the
    client made and this host is; the pod's own environment says the same
    thing when the config was written without it."""
    kind = str((config or {}).get("kind", "runpod"))
    provider = _provider(kind)
    configured = str((config or {}).get("pod_id", "")).strip()
    if configured:
        return configured
    env_id = os.environ.get(provider.pod_id_env, "").strip()
    if env_id:
        return env_id
    pod_id = provider.environment_file().get(provider.pod_id_env, "").strip()
    if pod_id:
        return pod_id
    raise TerminateError(
        f"no pod id: config.json has no provider.pod_id, {provider.pod_id_env} is unset, "
        f"and {rp_environment_path()} has no {provider.pod_id_env}"
    )


def runpod_terminate(pod_id: str, api_key: str) -> str:
    url = f"{RUNPOD_API}/pods/{pod_id}/action"
    body = json.dumps({"action": "terminate"}).encode()
    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": USER_AGENT,
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=DEFAULT_TIMEOUT_S) as response:
            return response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[-500:]
        raise TerminateError(f"POST {url} -> HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise TerminateError(f"POST {url} failed: {exc.reason}") from exc


PROVIDERS: dict[str, Provider] = {
    "runpod": Provider(runpod_terminate, "RUNPOD_API_KEY", "RUNPOD_POD_ID", rp_environment),
}


def self_terminate(config: HostConfig, *, terminate_call: TerminateCall | None = None) -> str:
    provider: dict[str, Any] | None = config.provider
    if not provider:
        raise TerminateError("host has no provider configured; nothing to terminate")
    kind = str(provider.get("kind", ""))
    call = terminate_call or _provider(kind).terminate
    return call(read_pod_id(provider), read_api_key(kind))
