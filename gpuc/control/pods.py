"""`gpuc pods`: the provider's view, so a leak is visible without trusting our state.

Pods without our prefix are counted and named and nothing else: they belong to
someone else and this command is the place that habit is most easily broken.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta

from gpuc.control.config import (
    DesiredUnreadable,
    Registry,
    Settings,
    load_desired,
    load_registry,
)
from gpuc.control.providers.base import Pod, Provider, owned_pods
from gpuc.control.provision import dispatcher_heartbeat_age
from gpuc.control.reconcile import STRAY_GRACE_MINUTES

COLUMNS = ("NAME", "ID", "STATUS", "GPU", "$/H", "CUDA", "AGE", "UTIL", "DESIRED", "HEARTBEAT")


def format_age(age: timedelta | None) -> str:
    if age is None:
        return "?"
    minutes = age.total_seconds() / 60.0
    if minutes < 90:
        return f"{minutes:.0f}m"
    return f"{minutes / 60.0:.1f}h"


@dataclass
class PodRow:
    pod: Pod
    desired: bool
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
            format_age(self.pod.age),
            util,
            "yes" if self.desired else "NO",
            heartbeat,
        ]


@dataclass
class PodsView:
    rows: list[PodRow] = field(default_factory=list)
    others: list[Pod] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def hourly(self) -> float:
        return sum(row.pod.cost_usd_hr for row in self.rows if row.pod.status != "TERMINATED")


def gather(
    settings: Settings,
    provider: Provider,
    *,
    heartbeats: bool = True,
    registry: Registry | None = None,
) -> PodsView:
    view = PodsView()
    pods = provider.list()
    ours = owned_pods(pods, provider.caps.prefix)
    ours_ids = {pod.id for pod in ours}
    others = [pod for pod in pods if pod.id not in ours_ids]
    registry = registry if registry is not None else load_registry()
    try:
        desired_ids = {host.pod_id for host in load_desired()}
    except DesiredUnreadable as exc:
        desired_ids = set()
        view.notes.append(f"desired state is unreadable, so every pod shows DESIRED=NO: {exc}")
    by_pod_id = {e.pod_id: e for e in registry.hosts.values() if e.kind == "runpod" and e.pod_id}

    for pod in sorted(ours, key=lambda p: p.name):
        entry = by_pod_id.get(pod.id)
        # Only bootstrapped, running hosts can answer; anything else costs an ssh timeout.
        age = (
            dispatcher_heartbeat_age(entry, settings)
            if heartbeats and entry is not None and entry.python and pod.status == "RUNNING"
            else None
        )
        view.rows.append(PodRow(pod=pod, desired=pod.id in desired_ids, heartbeat_s=age))
    view.others = sorted(others, key=lambda p: p.name)
    return view


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
    stray = [
        row.pod.name for row in view.rows if not row.desired and row.pod.status != "TERMINATED"
    ]
    if stray:
        lines.append(
            f"DESIRED=NO on {', '.join(stray)}: nothing local wants these. "
            f"`gpuc reconcile --once` terminates them once they are over "
            f"{STRAY_GRACE_MINUTES:.0f} min old."
        )
    if view.others:
        names = ", ".join(f"{pod.name} ({pod.status})" for pod in view.others)
        lines.append(f"{len(view.others)} other pod(s) in the account, never touched: {names}")
    lines += [f"note: {note}" for note in view.notes]
    return "\n".join(lines)
