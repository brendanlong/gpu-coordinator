"""Turn constraints into a bootstrapped RunPod host, or give the money back.

The local side owns an ephemeral host until it has proven healthy, so every
failure path here ends in a terminate the provider has confirmed before the
next offer is tried: a `draining` marker or an unreachable pod is never enough
to justify a second create (that is how you double-bill).

Nothing outside this process watches a pod it is bringing up: if anything
goes wrong before the host is registered -- including a Ctrl-C -- the pod is
terminated on the way out, and a terminate that fails is reported loudly for
`gpuc pods` to show, because from then on it bills until a person ends it.

One ceiling bounds the whole attempt, every offer included. A failure is one
of three things (`Verdict`): this offer is bad and the next may not be; the
attempt cannot succeed however many pods it buys; or nothing is wrong yet and
the poll should go on. The first buys another pod, the second stops before
it can, and the third is the only one that costs nothing.
"""

from __future__ import annotations

import re
import secrets
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

from gpuc.control import rented
from gpuc.control.bootstrap import (
    DEFAULT_HEALTH,
    BootstrapError,
    BootstrapResult,
    HealthOptions,
    bootstrap_host,
)
from gpuc.control.config import (
    DEFAULT_DISK_GB,
    HostEntry,
    Reporter,
    Settings,
    config_file,
    forget_host,
    open_registry,
    registry_transaction,
    transport_for,
    utc_now,
)
from gpuc.control.connect import Connection, connect_host
from gpuc.control.gpuinfo import summarize
from gpuc.control.probe import probe_host
from gpuc.control.providers.base import (
    DEFAULT_IMAGE,
    Constraints,
    Offer,
    Pod,
    Provider,
    ProviderError,
)
from gpuc.control.remote import (
    Answered,
    Gone,
    RemoteError,
    Unaskable,
    ask,
    reason_of,
)
from gpuc.control.status import HOST_ONLY, parse_status
from gpuc.control.transport import LocalToolMissing, SshUnusable, Transport, TransportError

CEILING_MINUTES = 15.0
"""How long one `submit --runpod` may spend buying, waiting for and proving a
host, every offer it tries included. Per attempt, not per offer: with N
offers a per-offer ceiling was N x 15 minutes of somebody's evening."""
POLL_INTERVAL_S = 5.0
SSH_MAX_INTERVAL_S = 15.0
LOG_CHECK_INTERVAL_S = 30.0
SSH_REPORT_INTERVAL_S = 60.0

SSH_MISCONFIGURED = re.compile(
    r"Bad configuration option|no such identity file|Identity file .* not accessible"
    r"|WARNING: UNPROTECTED PRIVATE KEY",
    re.IGNORECASE,
)
"""Local ssh problems that no amount of waiting -- and no other offer -- can fix.
(A ControlMaster socket that cannot bind is `transport.SshUnusable`, the same
verdict from the transport itself.)"""


class ProvisionError(RuntimeError):
    pass


class Unprovisionable(ProvisionError):
    """Nothing another offer could fix: the attempt stops at this pod."""


class Verdict(Enum):
    """What a failure while bringing up a pod means for the attempt."""

    KEEP_WAITING = "keep waiting"
    NEXT_OFFER = "next offer"
    ABORT = "abort"


RECOVERABLE = (ProvisionError, ProviderError, BootstrapError, RemoteError, TransportError)
"""Failures that are about the pod or the provider, not this program: the
attempt goes on to the next offer unless `verdict` says otherwise. Anything
else is a bug, and propagates once the pod has been terminated."""


def verdict(exc: BaseException, *, polling: bool = False) -> Verdict:
    """The one rule for what a failure costs.

    `polling` is the wait for a pod's endpoint or its sshd, where a provider
    read that failed or an ssh that was refused is the ordinary state of a
    pod still booting. The same failures at any other point are the offer's.
    A local ssh misconfiguration, or an `ssh`/`rsync` this machine does not
    have, is never the offer's: every pod would be bought, waited on and
    terminated identically, so it ends the attempt at the first one.
    """
    if isinstance(exc, (SshUnusable, LocalToolMissing, Unprovisionable)):
        return Verdict.ABORT
    if isinstance(exc, TransportError) and SSH_MISCONFIGURED.search(str(exc)):
        return Verdict.ABORT
    if polling and isinstance(exc, (ProviderError, TransportError)):
        return Verdict.KEEP_WAITING
    if isinstance(exc, RECOVERABLE):
        return Verdict.NEXT_OFFER
    return Verdict.ABORT


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
        health_options: HealthOptions = ...,
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


def address_for(name: str, pod: Pod, provider: Provider) -> HostEntry:
    """How to reach this pod, and nothing about what it is."""
    address = rented.address_for(name, pod, provider.name)
    if address is None:
        raise ProvisionError(f"pod {pod.id} has no direct SSH endpoint")
    return address


def _label(offer: Offer) -> str:
    return f"{offer.name}/{offer.cloud.lower()} ${offer.price_usd_hr:.3f}/h"


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
    health_options: HealthOptions = DEFAULT_HEALTH,
    deps: ProvisionDeps | None = None,
) -> HostEntry:
    """Create, wait for, bootstrap and register one pod. Returns its registry entry."""
    deps = deps or ProvisionDeps()
    progress = _Progress(report, deps.now)

    key = public_key_path(settings)
    added = provider.ensure_ssh_key(key.read_text())
    progress(f"ssh key {key}: {'registered now' if added else 'already registered'} on the account")

    offers = provider.offers(constraints)
    if not offers:
        raise ProvisionError(
            f"no offers match {_describe(constraints)}.\n"
            f"Relax --max-price, add another --gpu, or try --cloud any; "
            f"availability moves hour to hour."
        )
    deadline = deps.now() + deps.ceiling_minutes * 60.0
    progress(
        f"{len(offers)} offer(s): {', '.join(_label(o) for o in offers[:6])}; "
        f"ceiling {deps.ceiling_minutes:.0f} min for the whole attempt"
    )

    failures: list[str] = []
    billing: list[str] = []
    for offer in offers:
        if deps.now() >= deadline:
            failures.append(f"  - {_label(offer)}: not tried, the ceiling had passed")
            continue
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
                health_options=health_options,
                progress=progress,
                deps=deps,
                deadline=deadline,
                billing=billing,
            )
        except RECOVERABLE as exc:
            first = _first_line(exc)
            failures.append(f"  - {_label(offer)}: {first}")
            if verdict(exc) is Verdict.ABORT:
                raise ProvisionError(
                    f"provisioning stopped, no other offer could fix this: {first}\n"
                    + "\n".join(failures)
                    + f"\n{_cleanup(billing)}"
                ) from exc
            progress(f"offer {_label(offer)} failed: {first}")
    raise ProvisionError(
        "every offer failed to produce a healthy pod:\n"
        + "\n".join(failures)
        + f"\n{_cleanup(billing)} Try again later, widen --gpu, or raise --max-price."
    )


def _cleanup(billing: list[str]) -> str:
    if not billing:
        return "All pods created here were terminated."
    return (
        f"Pod(s) {', '.join(billing)} could NOT be terminated and are still billing: "
        f"`gpuc pods` shows them, and `gpuc host terminate <pod-id> --force` (or the "
        f"provider's console) ends them."
    )


def _describe(constraints: Constraints) -> str:
    parts = [f"gpu={','.join(constraints.gpu_names) or 'any'}"]
    if constraints.min_vram_gb is not None:
        parts.append(f"min-vram={constraints.min_vram_gb}GB")
    if constraints.max_price_usd_hr is not None:
        parts.append(f"max-price=${constraints.max_price_usd_hr:.2f}/h")
    parts.append(f"cloud={'+'.join(c.lower() for c in constraints.clouds)}")
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
    health_options: HealthOptions,
    progress: _Progress,
    deps: ProvisionDeps,
    deadline: float,
    billing: list[str],
) -> HostEntry:
    """One offer, start to finish. A pod this could not terminate on the way
    out is appended to `billing`, so the caller's report can name it."""
    name = pod_name(provider.prefix, name_hint)
    progress(
        f"creating {name}: {_label(offer)} x{constraints.gpu_count}, disk {disk_gb}GB, "
        f"cuda>={constraints.cuda_min}, image {image}"
    )
    pod = provider.create(
        offer,
        name,
        image=image,
        disk_gb=disk_gb,
        cuda_min=constraints.cuda_min,
        gpu_count=constraints.gpu_count,
    )
    created_at = utc_now()
    progress(
        f"pod {pod.id} created ({pod.status}); {_left(deadline, deps):.0f}s of the ceiling left"
    )

    try:
        pod = _wait_for_ssh_direct(provider, pod, deadline, progress, deps)
        assert pod.ssh_direct is not None
        progress(
            f"ssh.direct {pod.ssh_direct.username}@{pod.ssh_direct.host}:{pod.ssh_direct.port} "
            f"(cuda {pod.cuda_version or '?'}, ${pod.cost_usd_hr:.3f}/h)"
        )
        address = address_for(name, pod, provider)
        transport = deps.transport_factory(address, settings)
        _wait_for_ssh(transport, deadline, progress, deps)
        # The one look at the pod's cards, the same probe `gpuc host add`
        # takes; the connect below owns every card it saw.
        probed = probe_host(address, settings, transport=transport)
        if not probed.gpu_info:
            raise ProvisionError(
                f"nvidia-smi on {transport.host} reported no cards "
                f"(driver {probed.driver_version or 'missing'}): "
                f"{probed.sections.get('gpus', '').strip()[-400:] or '(no output)'}"
            )
        address = address.with_cache(
            gpu_info=probed.gpu_info,
            driver_version=probed.driver_version,
            python=probed.host_python,
        )
        uuids = list(probed.gpu_info)
        progress(f"host GPUs: {summarize(uuids, probed.gpu_info)} ({', '.join(uuids)})")
        # The same connect path `gpuc host add` takes: a pod nobody has
        # configured yet gets `first_config`, with what only this create knows
        # on top -- the idle timer asked for, and the pod's own record of what
        # it was bought as (`rented.pod_record`), written to the pod so any
        # other machine reads it from there.
        entry = deps.connect(
            address,
            settings,
            fields={
                "idle_minutes": idle_minutes,
                "created_at": created_at,
                "provider": rented.pod_record(address, offer, created_at),
            },
            transport=transport,
        ).entry
        with registry_transaction() as registry:
            registry.put(entry)

        entry, result = deps.bootstrap(
            entry, settings, transport=transport, report=progress, health_options=health_options
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


def _left(deadline: float, deps: ProvisionDeps) -> float:
    return max(deadline - deps.now(), 0.0)


def _first_line(exc: BaseException) -> str:
    """One line for a failure: what ssh said last, else the first line of the
    message (`remote.reason_of`); an interrupt has no message at all."""
    if not str(exc).strip():
        return f"{type(exc).__name__} (interrupted)"
    return reason_of(exc)


def _abandon(
    provider: Provider,
    name: str,
    pod_id: str,
    progress: _Progress,
    deps: ProvisionDeps,
    reason: str,
) -> bool:
    """Terminate and forget a pod this run gave up on; False if it still bills."""
    progress(f"terminating {name} ({pod_id}): {reason}")
    try:
        provider.terminate_confirmed(pod_id, report=progress, sleep=deps.sleep)
    except ProviderError as exc:
        # A pod that is still billing must stay visible: its registry entry, if
        # it got one, is what `gpuc status` shows a POD line for.
        progress(
            f"WARNING: {exc}\n"
            f"  {name} is still billing until you end it: `gpuc pods` shows it, and "
            f"`gpuc host terminate {pod_id} --force` ends it; its registry entry is kept "
            f"until then."
        )
        return False
    progress(f"{name} terminated and confirmed gone")
    forget_host(name, pod_id, progress)
    return True


def _poll_failure(exc: Exception, what: str, progress: _Progress) -> None:
    """A failure inside a wait: carry on, or end the attempt -- never the offer."""
    if verdict(exc, polling=True) is Verdict.ABORT:
        raise Unprovisionable(f"{what} cannot succeed as configured: {_first_line(exc)}") from exc
    progress(f"{what} failed, retrying: {_first_line(exc)}")


def _ceiling(deadline: float, what: str, deps: ProvisionDeps) -> None:
    if deps.now() >= deadline:
        raise Unprovisionable(
            f"the {deps.ceiling_minutes:.0f} min ceiling passed while {what}; treating the "
            f"attempt as failed"
        )


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
            # healthy pod over one bad response costs the create again.
            _poll_failure(exc, f"pod {pod.id}: provider read", progress)
            _ceiling(deadline, f"waiting for the provider to answer about pod {pod.id}", deps)
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
            match = provider.broken_host.search(text)
            if match:
                raise ProvisionError(
                    f"pod {pod.id} is on a broken host (log matched {match.group(0)!r}):\n"
                    f"{_tail(text)}"
                )
        _ceiling(
            deadline,
            f"pod {pod.id} had no direct SSH endpoint (last state {state})",
            deps,
        )
        deps.sleep(deps.poll_interval_s)


def _wait_for_ssh(
    transport: Transport, deadline: float, progress: _Progress, deps: ProvisionDeps
) -> None:
    interval = 2.0
    attempts = 0
    next_report = 0.0
    while True:
        attempts += 1
        try:
            result = transport.run("true", timeout=30.0, check=False)
            if result.returncode == 0:
                progress(f"ssh answered after {attempts} attempt(s)")
                return
            raise TransportError(result)
        except TransportError as exc:
            last = str(exc).strip().splitlines()[-1] if str(exc).strip() else "no output"
            if verdict(exc, polling=True) is Verdict.ABORT:
                raise Unprovisionable(
                    f"ssh to {transport.host} cannot work as configured, so waiting would only "
                    f"burn the pod's clock: {last}"
                ) from exc
        # A silent 15-minute wait hides the reason; say it early and then rarely.
        if deps.now() >= next_report:
            next_report = deps.now() + SSH_REPORT_INTERVAL_S
            progress(f"ssh not up yet (attempt {attempts}): {last}")
        _ceiling(
            deadline,
            f"waiting for ssh to {transport.host} ({attempts} attempts; last error: {last})",
            deps,
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


def pick_reusable_host(
    constraints: Constraints,
    settings: Settings,
    *,
    provider: Provider,
    report: Reporter = print,
) -> HostEntry | None:
    """An existing gpuc pod that is running, matches the constraints, and dispatches.

    Every fact is the host's own: its pod is looked up at the provider and
    its `status` asked over ssh (`ask`), and the offer it was bought on is
    read from the config that answer came with. Nothing here is decided on
    the registry's cache of any of that.
    """
    for entry in open_registry().registry.listing():
        if entry.rental is None:
            continue
        asked = ask(entry, HOST_ONLY, settings, provider=provider)
        if isinstance(asked, Gone):
            # The pod is gone for good, so the entry can only mislead `status`,
            # `logs` and the next reuse pass. Drop it here rather than leaving
            # submit to fail on an ssh to an address someone else now owns.
            report(f"reuse: forgetting {entry.name}, its {asked.reason}")
            forget_host(entry.name, entry.rental.pod_id, report)
            continue
        if isinstance(asked, Unaskable):
            report(f"reuse: skipping {entry.name}, it could not be asked: {asked.reason}")
            continue
        if not provider.is_running(asked.pod):
            status = asked.pod.status if asked.pod else "unknown to the provider"
            report(f"reuse: skipping {entry.name}, its pod is {status}")
            continue
        skip = _unreusable(entry, asked, constraints, provider)
        if skip:
            report(f"reuse: skipping {entry.name}, {skip}")
            continue
        age = parse_status(entry, asked).heartbeat_age_s or 0.0
        report(f"reusing host {entry.name} ({entry.rental.pod_id}, heartbeat {age:.0f}s old)")
        return entry
    return None


def _unreusable(
    entry: HostEntry, asked: Answered, constraints: Constraints, provider: Provider
) -> str | None:
    """Why this answering, running pod is not the one to enqueue on, or None."""
    offer = rented.offer_of(asked.session.config.provider)
    if offer is None:
        return "its config records no offer to compare with this request"
    if not offer_satisfies(offer, constraints):
        return (
            f"its {offer.name or 'unrecorded'}/{offer.cloud.lower()} ${offer.price_usd_hr:.3f}/h "
            f"offer does not match this request"
        )
    view = parse_status(entry, asked)
    if len(view.owned) < constraints.gpu_count:
        return f"it owns {len(view.owned)} GPU(s) and this request needs {constraints.gpu_count}"
    if not view.dispatcher_alive:
        age = "unknown" if view.heartbeat_age_s is None else f"{view.heartbeat_age_s:.0f}s old"
        return f"its dispatcher heartbeat is {age}"
    if view.draining:
        # It is terminating itself; a job enqueued now dies with the pod.
        return "it is draining (terminating itself)"
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
    health_options: HealthOptions = DEFAULT_HEALTH,
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
        health_options=health_options,
        deps=deps,
    )
