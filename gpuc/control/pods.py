"""`gpuc pods`: every provider's view, so what is billing is visible without trusting our state.

Pods without our prefix are counted and named and nothing else: they belong to
someone else and this command is the place that habit is most easily broken.
Nothing here terminates anything: a pod with a dispatcher ends itself when its
queue goes idle (`gpuc host set <host> --idle-min 0` hurries it), and
`gpuc host terminate <name-or-pod-id>` ends any of ours outright, including one
with no dispatcher to ask.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from gpuc.control.config import HostEntry, Settings, open_registry
from gpuc.control.providers.base import Pod, Provider, ProviderError, Providers, owned_pods
from gpuc.control.provision import CEILING_MINUTES
from gpuc.control.remote import ask
from gpuc.control.status import format_duration, parse_status

COLUMNS = (
    "NAME",
    "PROVIDER",
    "ID",
    "STATUS",
    "GPU",
    "$/H",
    "CUDA",
    "AGE",
    "UTIL",
    "HOST",
    "HEARTBEAT",
)


@dataclass
class PodRow:
    pod: Pod
    provider: Provider
    """Whose vocabulary says whether this pod still bills."""
    host: str | None
    """The registry name this machine drives the pod under; None if it does not."""
    heartbeat_s: float | None = None

    def cells(self) -> list[str]:
        gpu = self.pod.gpu_name or "?"
        if self.pod.gpu_count > 1:
            gpu = f"{gpu} x{self.pod.gpu_count}"
        util = ",".join(f"{u}%" for u in self.pod.gpu_utils) if self.pod.gpu_utils else "-"
        heartbeat = "-" if self.heartbeat_s is None else f"{self.heartbeat_s:.0f}s"
        return [
            self.pod.name,
            self.provider.name,
            self.pod.id,
            self.pod.status,
            gpu,
            f"{self.pod.cost_usd_hr:.3f}",
            self.pod.cuda_version or "?",
            "?" if self.pod.age is None else format_duration(self.pod.age.total_seconds()),
            util,
            self.host or "-",
            heartbeat,
        ]

    def document(self) -> dict[str, Any]:
        pod = self.pod
        age = pod.age
        return {
            "provider": self.provider.name,
            "id": pod.id,
            "name": pod.name,
            "status": pod.status,
            "gpu_name": pod.gpu_name,
            "gpu_count": pod.gpu_count,
            "cost_usd_hr": pod.cost_usd_hr,
            "cuda_version": pod.cuda_version,
            "age_s": None if age is None else round(age.total_seconds(), 1),
            "created_at": None if pod.created_at is None else pod.created_at.isoformat(),
            "gpu_utils": list(pod.gpu_utils),
            "host": self.host,
            "heartbeat_age_s": self.heartbeat_s,
        }


@dataclass
class PodsView:
    rows: list[PodRow] = field(default_factory=list)
    others: list[tuple[str, Pod]] = field(default_factory=list)
    """Pods without our prefix, with the provider each is at."""
    notes: list[str] = field(default_factory=list)
    """Providers not asked, for want of a key."""
    failures: list[str] = field(default_factory=list)
    """Providers that were asked and did not answer."""

    @property
    def hourly(self) -> float:
        return sum(row.pod.cost_usd_hr for row in self.rows if not row.provider.is_gone(row.pod))

    def document(self) -> dict[str, Any]:
        """`gpuc pods --json`: the provider's answer, ours and everyone else's.

        `others` are pods without our prefix: an id, a name and a status, and
        nothing else, because this command never touches them. `host` on one
        of ours is the name this machine's registry drives it under, null for
        a pod registered nowhere here -- `gpuc host add <name> --pod <id>`
        adopts it. `heartbeat_age_s` is null under `--no-heartbeat` and for a
        pod that could not be asked.
        """
        return {
            "pods": [row.document() for row in self.rows],
            "hourly_usd": round(self.hourly, 4),
            "others": [
                {"provider": provider, "id": pod.id, "name": pod.name, "status": pod.status}
                for provider, pod in self.others
            ],
            "notes": list(self.notes),
            "failures": list(self.failures),
        }


def gather(
    settings: Settings,
    providers: Providers,
    *,
    missing: list[str] | None = None,
    heartbeats: bool = True,
) -> PodsView:
    """Every provider in `providers` is listed; one that fails is a failure
    beside the others' rows, never instead of them. `missing` is why each
    provider not in `providers` was not asked."""
    view = PodsView(notes=[f"not asked, {why}" for why in missing or []])
    registry = open_registry().registry
    by_pod_id = {
        (e.rental.provider, e.rental.pod_id): e
        for e in registry.hosts.values()
        if e.rental is not None
    }
    for provider in providers.values():
        try:
            pods = provider.list()
        except ProviderError as exc:
            view.failures.append(f"{provider.name}: {exc}")
            continue
        ours = owned_pods(pods, provider.prefix)
        ours_ids = {pod.id for pod in ours}
        view.others += [(provider.name, pod) for pod in pods if pod.id not in ours_ids]
        for pod in ours:
            entry = by_pod_id.get((provider.name, pod.id))
            # Only bootstrapped, running hosts can answer; anything else costs an ssh timeout.
            age = (
                heartbeat_age(entry, settings)
                if heartbeats and entry is not None and provider.is_running(pod)
                else None
            )
            host = entry.name if entry is not None else None
            view.rows.append(PodRow(pod=pod, provider=provider, host=host, heartbeat_s=age))
    view.rows.sort(key=lambda row: row.pod.name)
    view.others.sort(key=lambda other: other[1].name)
    return view


def heartbeat_age(entry: HostEntry, settings: Settings) -> float | None:
    """How long ago the pod's dispatcher last beat, by the host's own `status`;
    the pod was just listed, so the provider is not asked about it again."""
    return parse_status(entry, ask(entry, "status", settings)).heartbeat_age_s


def _may_be_provisioning(pod: Pod) -> bool:
    age = pod.age
    return age is not None and age.total_seconds() < CEILING_MINUTES * 60.0


def render(view: PodsView) -> str:
    lines: list[str] = []
    rows = [list(COLUMNS)] + [row.cells() for row in view.rows]
    widths = [max(len(row[i]) for row in rows) for i in range(len(COLUMNS))]
    for row in rows:
        lines.append("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)).rstrip())
    if not view.rows:
        lines.append("(no pods with our prefix)")
    else:
        lines.append(f"{len(view.rows)} pod(s) with our prefix, ${view.hourly:.2f}/h total")
    unregistered = [
        row.pod for row in view.rows if row.host is None and not row.provider.is_gone(row.pod)
    ]
    if unregistered:
        names = ", ".join(f"{pod.name} ({pod.id})" for pod in unregistered)
        lines.append(
            f"not registered here: {names}. `gpuc host add <name> --pod <id>` drives one "
            f"from this machine; one whose dispatcher is gone will never idle out and bills "
            f"until `gpuc host terminate <id> --force` (or the provider's console) ends it."
        )
        young = [pod for pod in unregistered if _may_be_provisioning(pod)]
        if young:
            names = ", ".join(pod.name for pod in young)
            lines.append(
                f"{names}: younger than the {CEILING_MINUTES:.0f} min provisioning ceiling, "
                f"so a `gpuc host rent` elsewhere may still be setting it up; "
                f"ending it now would cost that submit its pod."
            )
    if view.others:
        names = ", ".join(f"{pod.name} ({provider}, {pod.status})" for provider, pod in view.others)
        lines.append(f"{len(view.others)} other pod(s) in the accounts, never touched: {names}")
    lines += [f"note: {note}" for note in view.notes]
    lines += [f"error: could not list the pods at {failure}" for failure in view.failures]
    return "\n".join(lines)
