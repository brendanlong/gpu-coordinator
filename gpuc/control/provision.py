"""Turn constraints into a bootstrapped RunPod host, or give the money back.

The local side owns an ephemeral host until it has proven healthy, so every
failure path here ends in `terminate` plus a wait for TERMINATED before the
next offer is tried: a `draining` marker or an unreachable pod is never enough
to justify a second create (that is how you double-bill).

Nothing outside this process watches a pod it is bringing up: if anything
goes wrong before the host is registered -- including a Ctrl-C -- the pod is
terminated on the way out, and a terminate that fails is reported loudly for
`gpuc pods` to show, because from then on it bills until a person ends it.
"""

from __future__ import annotations

import os
import re
import secrets
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from gpuc.control import rented
from gpuc.control.bootstrap import BootstrapError, BootstrapResult, bootstrap_host
from gpuc.control.config import (
    DEFAULT_DISK_GB,
    ConfigError,
    HostEntry,
    Reporter,
    Settings,
    config_file,
    forget_host_locked,
    load_registry,
    registry_transaction,
    transport_for,
    utc_now,
)
from gpuc.control.connect import Connection, connect_host
from gpuc.control.gpuinfo import GpuInfo, discover, summarize
from gpuc.control.providers.base import (
    DEFAULT_IMAGE,
    Constraints,
    Offer,
    Pod,
    Provider,
    ProviderError,
)
from gpuc.control.remote import RemoteError, open_session
from gpuc.control.s3index import default_s3_prefix
from gpuc.control.transport import SshUnusable, Transport, TransportError

CEILING_MINUTES = 15.0
DEFAULT_CUDA_MIN = "12.8"
POLL_INTERVAL_S = 5.0
SSH_MAX_INTERVAL_S = 15.0
LOG_CHECK_INTERVAL_S = 30.0
SSH_REPORT_INTERVAL_S = 60.0
REUSE_HEARTBEAT_MAX_S = 30.0
"""How stale a pod's heartbeat may be for `submit` to reuse it rather than buy another."""
AWS_KEY_VARS = ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN")

SSH_MISCONFIGURED = re.compile(
    r"Bad configuration option|no such identity file|WARNING: UNPROTECTED PRIVATE KEY",
    re.IGNORECASE,
)
"""Local ssh problems that no amount of waiting can fix: fail before the ceiling.
(A ControlMaster socket that cannot bind is `transport.SshUnusable`, raised
before this is consulted.)"""

BROKEN_HOST = re.compile(
    r"card[0-9]|device nodes|OCI runtime|runc create|failed to create shim", re.IGNORECASE
)
"""Signatures of a host whose GPU device nodes are broken: re-place, never retry."""


class ProvisionError(RuntimeError):
    pass


class ConnectFn(Protocol):
    def __call__(
        self,
        address: HostEntry,
        settings: Settings | None = ...,
        *,
        fields: Mapping[str, Any] | None = ...,
        env_updates: Mapping[str, str | None] | None = ...,
        transport: Transport | None = ...,
        force: bool = ...,
    ) -> Connection: ...


class BootstrapFn(Protocol):
    def __call__(
        self,
        entry: HostEntry,
        settings: Settings | None = ...,
        *,
        transport: Transport | None = ...,
        report: Reporter = ...,
        health_args: str = ...,
    ) -> tuple[HostEntry, BootstrapResult]: ...


@dataclass
class ProvisionDeps:
    """Everything that talks to the world, so tests can run the flow offline."""

    sleep: Callable[[float], None] = time.sleep
    now: Callable[[], float] = time.monotonic
    bootstrap: BootstrapFn = bootstrap_host
    connect: ConnectFn = connect_host
    transport_factory: Callable[[HostEntry, Settings], Transport] = transport_for
    poll_interval_s: float = POLL_INTERVAL_S
    log_check_interval_s: float = LOG_CHECK_INTERVAL_S
    ceiling_minutes: float = CEILING_MINUTES


class _Progress:
    """Timestamps every line: provisioning is slow and the timings are the diagnosis."""

    def __init__(self, report: Reporter, now: Callable[[], float]) -> None:
        self._report = report
        self._now = now
        self._start = now()

    def __call__(self, message: str) -> None:
        clock = datetime.now(UTC).strftime("%H:%M:%S")
        self._report(f"[{clock} +{self._now() - self._start:5.0f}s] {message}")


def public_key_path(settings: Settings) -> Path:
    if settings.ssh_key:
        # Appended, not `with_suffix`: a key called `my.key` has a public half
        # called `my.key.pub`, and `with_suffix` would ask for `my.pub`.
        candidate = Path(f"{settings.ssh_key_path or ''}.pub")
        if not candidate.exists():
            raise ProvisionError(
                f"ssh_key is {settings.ssh_key} in {config_file()}, but its public half "
                f"{candidate} does not exist.\nRunPod needs the public key to authorise the "
                f"pod; create it with `ssh-keygen -y -f {settings.ssh_key_path} > {candidate}`."
            )
        return candidate
    for name in ("id_ed25519.pub", "id_ecdsa.pub", "id_rsa.pub"):
        candidate = Path.home() / ".ssh" / name
        if candidate.exists():
            return candidate
    raise ProvisionError(
        "no SSH public key found in ~/.ssh (looked for id_ed25519.pub, id_ecdsa.pub, "
        "id_rsa.pub).\nCreate one with `ssh-keygen -t ed25519`, or set ssh_key in "
        "~/.config/gpu-coordinator/config.toml."
    )


def pod_name(prefix: str, name_hint: str) -> str:
    hint = re.sub(r"[^a-z0-9-]+", "-", name_hint.lower()).strip("-") or "job"
    return f"{prefix}{hint}-{secrets.token_hex(3)}"


def offer_satisfies(offer: Offer, constraints: Constraints) -> bool:
    wanted = {name.casefold() for name in constraints.gpu_names}
    if wanted and not wanted & {offer.gpu_id.casefold(), offer.name.casefold()}:
        return False
    if constraints.min_vram_gb is not None and offer.vram_gb < constraints.min_vram_gb:
        return False
    cap = constraints.max_price_usd_hr
    if cap is not None and offer.price_usd_hr > cap:
        return False
    if constraints.clouds and offer.cloud not in constraints.clouds:
        return False
    return offer.matches_cuda_floor(constraints.cuda_min)


def address_for(name: str, pod: Pod) -> HostEntry:
    """How to reach this pod, and nothing about what it is."""
    address = rented.address_for(name, pod)
    if address is None:
        raise ProvisionError(f"pod {pod.id} has no direct SSH endpoint")
    return address


def initial_config(
    name: str,
    settings: Settings,
    *,
    idle_minutes: float,
    created_at: str,
    provider: dict[str, Any],
) -> dict[str, Any]:
    """What a pod we just bought is: the config `connect_host` gives it.

    A fresh pod has no `config.json`, so this is the one case where the machine
    that created a host also decides what it is. Everything after this reads
    the host's copy, including the next machine to connect to it -- which is
    why the `provider` block (`rented.pod_record`) is written here rather than
    kept on this machine: it is the pod's own record of what it was bought as,
    and it is what any other machine reads it from. The cards are not here:
    the pod owns every one it has, which is what `connect_host` gives a host
    with no config, from the `gpu_info` the address carries.
    """
    return {
        "idle_minutes": idle_minutes,
        "s3_prefix": default_s3_prefix(settings, name),
        "created_at": created_at,
        "provider": provider,
    }


def gpu_info_for(transport: Transport, uuids: list[str], offer: Offer) -> dict[str, GpuInfo]:
    """What the pod's cards are: nvidia-smi if it answers, else the offer.

    A pod whose driver is still coming up would otherwise list its GPUs as
    unknown forever, and the offer already says exactly what was bought.
    """
    discovered = discover(transport)
    fallback = GpuInfo(name=offer.name, vram_mib=int(offer.vram_gb * 1024))
    return {uuid: discovered.get(uuid) or fallback for uuid in uuids}


def discover_gpu_uuids(transport: Transport) -> list[str]:
    result = transport.run("nvidia-smi --query-gpu=uuid --format=csv,noheader", check=False)
    uuids = [line.strip() for line in result.stdout.splitlines() if line.strip().startswith("GPU-")]
    if not uuids:
        raise ProvisionError(
            f"`nvidia-smi --query-gpu=uuid` returned no GPU UUIDs on {transport.host} "
            f"(exit {result.returncode}): {result.output.strip()[-400:]}"
        )
    return uuids


def deliver_s3_credentials(
    transport: Transport,
    entry: HostEntry,
    progress: _Progress,
    environ: dict[str, str] | None = None,
) -> bool:
    """Give the *host* S3 credentials, not just each job.

    Still needed even though the runner now hands each job's own ``secrets:``
    to its sync loop: the dispatcher's drain path mirrors every job's log.txt
    and state.json to ``s3_prefix`` after the jobs (and their secrets files)
    are gone, and it has only its own environment to do it with. Written from
    stdin at 0600, never via argv and never in the pod env (``GET /pods``
    returns the env to any holder of an account key).
    """
    environ = environ if environ is not None else dict(os.environ)
    if not entry.s3_prefix:
        return False
    if not (environ.get("AWS_ACCESS_KEY_ID") and environ.get("AWS_SECRET_ACCESS_KEY")):
        progress(
            "no AWS credentials in this environment, so the pod cannot mirror logs to "
            f"{entry.s3_prefix}; export AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY before "
            "submitting if you want the mirror"
        )
        return False
    region = environ.get("AWS_REGION") or environ.get("AWS_DEFAULT_REGION") or "us-east-1"
    body = "[default]\n" + f"region = {region}\n"
    for name in AWS_KEY_VARS:
        if environ.get(name):
            body += f"{name.lower()} = {environ[name]}\n"
    home = transport.run('printf %s "$HOME"', check=True).stdout.strip()
    transport.put_file(body, f"{home}/.aws/credentials", 0o600)
    progress(f"delivered S3 credentials to {home}/.aws/credentials (0600) for the log mirror")
    return True


def provision(
    constraints: Constraints,
    settings: Settings,
    *,
    name_hint: str = "job",
    idle_minutes: float = 15.0,
    disk_gb: int = DEFAULT_DISK_GB,
    image: str = DEFAULT_IMAGE,
    provider: Provider,
    report: Reporter = print,
    health_args: str = "",
    deps: ProvisionDeps | None = None,
) -> HostEntry:
    """Create, wait for, bootstrap and register one pod. Returns its registry entry."""
    deps = deps or ProvisionDeps()
    progress = _Progress(report, deps.now)

    key = public_key_path(settings)
    added = provider.ensure_ssh_key(key.read_text())
    progress(f"ssh key {key}: {'registered now' if added else 'already registered'} on the account")

    cuda_min = constraints.cuda_min or DEFAULT_CUDA_MIN
    offers = provider.offers(constraints)
    if not offers:
        raise ProvisionError(
            f"no offers match {_describe(constraints)}.\n"
            f"Relax --max-price, add another --gpu, or try --cloud any; "
            f"availability moves hour to hour."
        )
    progress(
        f"{len(offers)} offer(s): "
        + ", ".join(f"{o.name}/{o.cloud.lower()} ${o.price_usd_hr:.3f}/h" for o in offers[:6])
    )

    failures: list[str] = []
    billing: list[str] = []
    for offer in offers:
        label = f"{offer.name}/{offer.cloud.lower()} ${offer.price_usd_hr:.3f}/h"
        try:
            return _try_offer(
                offer,
                constraints,
                settings,
                provider=provider,
                name_hint=name_hint,
                idle_minutes=idle_minutes,
                disk_gb=disk_gb,
                image=image,
                cuda_min=cuda_min,
                health_args=health_args,
                progress=progress,
                deps=deps,
                billing=billing,
            )
        except (ProvisionError, ProviderError, BootstrapError, RemoteError, TransportError) as exc:
            first = str(exc).splitlines()[0]
            failures.append(f"  - {label}: {first}")
            progress(f"offer {label} failed: {first}")
    cleanup = (
        f"Pod(s) {', '.join(billing)} could NOT be terminated and are still billing: "
        f"`gpuc pods` shows them, and `gpuc host terminate <pod-id> --force` (or the "
        f"provider's console) ends them."
        if billing
        else "All pods created here were terminated."
    )
    raise ProvisionError(
        "every offer failed to produce a healthy pod:\n"
        + "\n".join(failures)
        + f"\n{cleanup} Try again later, widen --gpu, or raise --max-price."
    )


def _describe(constraints: Constraints) -> str:
    parts = [f"gpu={','.join(constraints.gpu_names) or 'any'}"]
    if constraints.min_vram_gb is not None:
        parts.append(f"min-vram={constraints.min_vram_gb}GB")
    if constraints.max_price_usd_hr is not None:
        parts.append(f"max-price=${constraints.max_price_usd_hr:.2f}/h")
    parts.append(f"cloud={'+'.join(c.lower() for c in constraints.clouds)}")
    if constraints.cuda_min:
        parts.append(f"cuda>={constraints.cuda_min}")
    return " ".join(parts)


def _try_offer(
    offer: Offer,
    constraints: Constraints,
    settings: Settings,
    *,
    provider: Provider,
    name_hint: str,
    idle_minutes: float,
    disk_gb: int,
    image: str,
    cuda_min: str,
    health_args: str,
    progress: _Progress,
    deps: ProvisionDeps,
    billing: list[str],
) -> HostEntry:
    """One offer, start to finish. A pod this could not terminate on the way
    out is appended to `billing`, so the caller's report can name it."""
    name = pod_name(provider.prefix, name_hint)
    progress(
        f"creating {name}: {offer.name}/{offer.cloud.lower()} ${offer.price_usd_hr:.3f}/h "
        f"x{constraints.gpu_count}, disk {disk_gb}GB, cuda>={cuda_min}, image {image}"
    )
    pod = provider.create(
        offer,
        name,
        image=image,
        disk_gb=disk_gb,
        cuda_min=cuda_min,
        gpu_count=constraints.gpu_count,
    )
    created_at = utc_now()
    progress(
        f"pod {pod.id} created ({pod.status}); ceiling {deps.ceiling_minutes:.0f} min from now"
    )

    deadline = deps.now() + deps.ceiling_minutes * 60.0
    try:
        pod = _wait_for_ssh_direct(provider, pod, deadline, progress, deps)
        assert pod.ssh_direct is not None
        progress(
            f"ssh.direct {pod.ssh_direct.username}@{pod.ssh_direct.host}:{pod.ssh_direct.port} "
            f"(cuda {pod.cuda_version or '?'}, ${pod.cost_usd_hr:.3f}/h)"
        )
        address = address_for(name, pod)
        transport = deps.transport_factory(address, settings)
        _wait_for_ssh(transport, deadline, progress, deps)
        uuids = discover_gpu_uuids(transport)
        address = address.with_cache(gpu_info=gpu_info_for(transport, uuids, offer))
        progress(f"host GPUs: {summarize(uuids, address.gpu_info)} ({', '.join(uuids)})")
        # The same connect path `gpuc host add` takes, with the config a pod
        # nobody has configured yet needs: written to the pod, which owns it
        # from here on, and read back into the registry.
        entry = deps.connect(
            address,
            settings,
            fields=initial_config(
                name,
                settings,
                idle_minutes=idle_minutes,
                created_at=created_at,
                provider=rented.pod_record(address, offer, created_at),
            ),
            transport=transport,
        ).entry
        deliver_s3_credentials(transport, entry, progress)
        with registry_transaction() as registry:
            registry.put(entry)

        entry, result = deps.bootstrap(
            entry, settings, transport=transport, report=progress, health_args=health_args
        )
        with registry_transaction() as registry:
            registry.put(entry)
        progress(
            f"host {name} ready: dispatcher pid {result.dispatcher_pid}, "
            f"idle terminate {idle_minutes:g} min"
        )
        return entry
    except BaseException as exc:
        # Everything from `create` to the last registry write owns a live pod, so
        # *every* way out of here terminates first: a Ctrl-C, a full disk, or a
        # bug none of the narrow except clauses name still costs money otherwise.
        if not _abandon(provider, name, pod.id, progress, deps, _first_line(exc)):
            billing.append(pod.id)
        raise


def _first_line(exc: BaseException) -> str:
    lines = str(exc).strip().splitlines()
    return lines[0] if lines else f"{type(exc).__name__} (interrupted)"


def _terminate_now(
    provider: Provider,
    name: str,
    pod_id: str,
    progress: _Progress,
    deps: ProvisionDeps,
    reason: str,
) -> bool:
    """True only when the provider confirmed the pod is gone.

    Retried a few times: a 5xx or a rate limit on the one call that stops the
    bill is the worst place to give up after one try, and nothing else will
    try again once this process has moved on to the next offer.
    """
    progress(f"terminating {name} ({pod_id}): {reason}")
    for attempt in range(1, provider.terminate_attempts + 1):
        try:
            provider.terminate(pod_id)
        except ProviderError as exc:
            if attempt < provider.terminate_attempts:
                retry = provider.terminate_retry_s
                progress(f"terminate {pod_id} failed ({exc}); retrying in {retry:g}s")
                deps.sleep(retry)
                continue
            progress(
                f"WARNING: could not terminate {name} ({pod_id}) in {attempt} attempts: {exc}\n"
                f"  It is still billing until you end it: `gpuc pods` shows it, and "
                f"`gpuc host terminate {pod_id} --force` ends it."
            )
            return False
        progress(f"{name} terminated and confirmed gone")
        return True
    return False


def _abandon(
    provider: Provider,
    name: str,
    pod_id: str,
    progress: _Progress,
    deps: ProvisionDeps,
    reason: str,
) -> bool:
    """Terminate and forget a pod this run gave up on; False if it still bills."""
    if not _terminate_now(provider, name, pod_id, progress, deps, reason):
        # A pod that is still billing must stay visible: its registry entry, if
        # it got one, is what `gpuc status` shows a POD line for.
        progress(f"keeping the registry entry for {name} until {pod_id} is confirmed gone")
        return False
    forget_host_locked(name, pod_id, progress)
    return True


def _wait_for_ssh_direct(
    provider: Provider, pod: Pod, deadline: float, progress: _Progress, deps: ProvisionDeps
) -> Pod:
    """RUNNING arrives in seconds and means nothing; `ssh.direct` is readiness."""
    last_state = ""
    next_log_check = deps.now() + deps.log_check_interval_s
    while True:
        try:
            current = provider.get(pod.id)
        except ProviderError as exc:
            # A 5xx or a rate limit is not a placement failure: terminating a
            # healthy pod over one bad response costs the create again. Keep
            # polling; the ceiling is still the backstop.
            if deps.now() >= deadline:
                raise ProvisionError(
                    f"pod {pod.id} could not be read from the provider inside the "
                    f"{deps.ceiling_minutes:.0f} min ceiling: {_first_line(exc)}"
                ) from exc
            progress(f"pod {pod.id}: provider read failed, retrying: {_first_line(exc)}")
            deps.sleep(deps.poll_interval_s)
            continue
        if current is None:
            raise ProvisionError(f"pod {pod.id} vanished from the provider before it was ready")
        state = f"{current.status} ssh.direct={'yes' if current.ssh_direct else 'no'}"
        if state != last_state:
            progress(f"pod {pod.id}: {state}")
            last_state = state
        if provider.is_dead(current):
            raise ProvisionError(
                f"pod {pod.id} reached {current.status} before it was ready:\n"
                f"{_log_tail(provider, pod.id)}"
            )
        if current.ssh_direct is not None:
            return current
        if deps.now() >= next_log_check:
            next_log_check = deps.now() + deps.log_check_interval_s
            text = _safe_logs(provider, pod.id)
            match = BROKEN_HOST.search(text)
            if match:
                raise ProvisionError(
                    f"pod {pod.id} is on a broken host (log matched {match.group(0)!r}):\n"
                    f"{_tail(text)}"
                )
        if deps.now() >= deadline:
            raise ProvisionError(
                f"pod {pod.id} had no direct SSH endpoint {deps.ceiling_minutes:.0f} min after "
                f"create (last state {state}). Treating it as a placement failure.\n"
                f"{_log_tail(provider, pod.id)}"
            )
        deps.sleep(deps.poll_interval_s)


def _wait_for_ssh(
    transport: Transport, deadline: float, progress: _Progress, deps: ProvisionDeps
) -> None:
    interval = 2.0
    attempts = 0
    last = ""
    next_report = 0.0
    while True:
        attempts += 1
        try:
            result = transport.run("true", timeout=30.0, check=False)
            if result.returncode == 0:
                progress(f"ssh answered after {attempts} attempt(s)")
                return
            last = result.output.strip().splitlines()[-1] if result.output.strip() else "no output"
        except SshUnusable as exc:
            # Never retried: the socket path cannot get shorter while we wait.
            last, misconfigured = str(exc), True
        except TransportError as exc:
            last = str(exc).splitlines()[-1]
            misconfigured = bool(SSH_MISCONFIGURED.search(last))
        else:
            misconfigured = bool(SSH_MISCONFIGURED.search(last))
        if misconfigured:
            raise ProvisionError(
                f"ssh to {transport.host} cannot work as configured, so waiting would only "
                f"burn the pod's clock: {last}"
            )
        # A silent 15-minute wait hides the reason; say it early and then rarely.
        if deps.now() >= next_report:
            next_report = deps.now() + SSH_REPORT_INTERVAL_S
            progress(f"ssh not up yet (attempt {attempts}): {last}")
        if deps.now() >= deadline:
            raise ProvisionError(
                f"ssh to {transport.host} never succeeded within the "
                f"{deps.ceiling_minutes:.0f} min ceiling ({attempts} attempts); last error: {last}"
            )
        deps.sleep(interval)
        interval = min(interval * 1.5, SSH_MAX_INTERVAL_S)


def _safe_logs(provider: Provider, pod_id: str) -> str:
    try:
        return provider.logs(pod_id)
    except ProviderError:
        return ""


def _log_tail(provider: Provider, pod_id: str, lines: int = 15) -> str:
    text = _safe_logs(provider, pod_id)
    return _tail(text, lines) if text else "  (no pod logs available)"


def _tail(text: str, lines: int = 15) -> str:
    return "\n".join(f"  {line}" for line in text.strip().splitlines()[-lines:])


def host_status(
    entry: HostEntry, settings: Settings | None = None, *, timeout: float = 60.0
) -> dict[str, Any] | None:
    """The host's own status document, or None if it cannot be reached."""
    try:
        payload = open_session(entry, settings).host_json("status", timeout=timeout)
    except (RemoteError, TransportError, ConfigError):
        return None
    return payload if isinstance(payload, dict) else None


def dispatcher_heartbeat_age(entry: HostEntry, settings: Settings) -> float | None:
    payload = host_status(entry, settings)
    age = payload.get("dispatcher_heartbeat_age_s") if payload else None
    return float(age) if isinstance(age, (int, float)) else None


def pick_reusable_host(
    constraints: Constraints,
    settings: Settings,
    *,
    provider: Provider,
    report: Reporter = print,
) -> HostEntry | None:
    """An existing gpuc pod that is RUNNING, matches the constraints, and dispatches."""
    for entry in list(load_registry().hosts.values()):
        if not entry.pod_id:
            continue
        offer = rented.offer_of(entry.config.provider)
        if offer is None:
            report(
                f"reuse: skipping {entry.name}, its config records no offer to compare "
                f"with this request"
            )
            continue
        if not offer_satisfies(offer, constraints):
            report(
                f"reuse: skipping {entry.name}, its {offer.name or 'unrecorded'}/"
                f"{offer.cloud.lower()} ${offer.price_usd_hr:.3f}/h offer does not match "
                f"this request"
            )
            continue
        if len(entry.gpus) < constraints.gpu_count:
            report(
                f"reuse: skipping {entry.name}, it owns {len(entry.gpus)} GPU(s) and this "
                f"request needs {constraints.gpu_count}"
            )
            continue
        pod = provider.get(entry.pod_id)
        if pod is None or provider.is_gone(pod):
            # The pod is gone for good, so the entry can only mislead `status`,
            # `logs` and the next reuse pass. Drop it here rather than leaving
            # submit to fail on an ssh to an address someone else now owns.
            report(
                f"reuse: forgetting {entry.name}, its pod "
                f"{'is gone' if pod is None else f'is {pod.status}'}"
            )
            forget_host_locked(entry.name, entry.pod_id, report)
            continue
        if pod.status != "RUNNING":
            report(f"reuse: skipping {entry.name}, its pod is {pod.status}")
            continue
        status = host_status(entry, settings)
        age = status.get("dispatcher_heartbeat_age_s") if status else None
        if not isinstance(age, (int, float)) or age >= REUSE_HEARTBEAT_MAX_S:
            report(
                f"reuse: skipping {entry.name}, dispatcher heartbeat is "
                f"{'unreachable' if not isinstance(age, (int, float)) else f'{age:.0f}s old'}"
            )
            continue
        assert status is not None
        if status.get("draining"):
            # It is terminating itself; a job enqueued now dies with the pod.
            report(f"reuse: skipping {entry.name}, it is draining (terminating itself)")
            continue
        report(f"reusing host {entry.name} ({pod.id}, heartbeat {age:.0f}s old)")
        return entry
    return None


def runpod_host(
    constraints: Constraints,
    settings: Settings,
    *,
    provider: Provider,
    reuse: bool = True,
    name_hint: str = "job",
    idle_minutes: float = 15.0,
    disk_gb: int = DEFAULT_DISK_GB,
    image: str = DEFAULT_IMAGE,
    report: Reporter = print,
    health_args: str = "",
    deps: ProvisionDeps | None = None,
) -> HostEntry:
    if reuse:
        existing = pick_reusable_host(constraints, settings, provider=provider, report=report)
        if existing is not None:
            return existing
    return provision(
        constraints,
        settings,
        provider=provider,
        name_hint=name_hint,
        idle_minutes=idle_minutes,
        disk_gb=disk_gb,
        image=image,
        report=report,
        health_args=health_args,
        deps=deps,
    )
