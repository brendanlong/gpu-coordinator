"""`gpuc pods`: the provider's view, so what is billing is visible without trusting our state.

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
from gpuc.control.providers.base import Pod, Provider, owned_pods
from gpuc.control.provision import CEILING_MINUTES
from gpuc.control.remote import ask
from gpuc.control.status import HOST_ONLY, format_duration, parse_status

COLUMNS = ("NAME", "ID", "STATUS", "GPU", "$/H", "CUDA", "AGE", "UTIL", "HOST", "HEARTBEAT")


@dataclass
class PodRow:
    pod: Pod
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
    provider: Provider
    """Whose vocabulary says which of these pods still bill."""
    rows: list[PodRow] = field(default_factory=list)
    others: list[Pod] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def hourly(self) -> float:
        return sum(row.pod.cost_usd_hr for row in self.rows if not self.provider.is_gone(row.pod))

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
                {"id": pod.id, "name": pod.name, "status": pod.status} for pod in self.others
            ],
            "notes": list(self.notes),
        }


def gather(
    settings: Settings,
    provider: Provider,
    *,
    heartbeats: bool = True,
) -> PodsView:
    view = PodsView(provider)
    pods = provider.list()
    ours = owned_pods(pods, provider.prefix)
    ours_ids = {pod.id for pod in ours}
    others = [pod for pod in pods if pod.id not in ours_ids]
    registry = open_registry().registry
    by_pod_id = {e.rental.pod_id: e for e in registry.hosts.values() if e.rental is not None}

    for pod in sorted(ours, key=lambda p: p.name):
        entry = by_pod_id.get(pod.id)
        # Only bootstrapped, running hosts can answer; anything else costs an ssh timeout.
        age = (
            heartbeat_age(entry, settings)
            if heartbeats and entry is not None and provider.is_running(pod)
            else None
        )
        view.rows.append(
            PodRow(pod=pod, host=entry.name if entry is not None else None, heartbeat_s=age)
        )
    view.others = sorted(others, key=lambda p: p.name)
    return view


def heartbeat_age(entry: HostEntry, settings: Settings) -> float | None:
    """How long ago the pod's dispatcher last beat, by the host's own `status`;
    the pod was just listed, so the provider is not asked about it again."""
    return parse_status(entry, ask(entry, HOST_ONLY, settings)).heartbeat_age_s


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
        row.pod for row in view.rows if row.host is None and not view.provider.is_gone(row.pod)
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
                f"so a `gpuc submit --runpod` elsewhere may still be setting it up; "
                f"ending it now would cost that submit its pod."
            )
    if view.others:
        names = ", ".join(f"{pod.name} ({pod.status})" for pod in view.others)
        lines.append(f"{len(view.others)} other pod(s) in the account, never touched: {names}")
    lines += [f"note: {note}" for note in view.notes]
    return "\n".join(lines)
