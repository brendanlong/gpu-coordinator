"""`gpuc status`: one compact block per host, and the phase-aware suspect rule.

Reads the host over the transport (the host is authoritative); the S3 index
only fills in jobs whose host is gone. Never kills anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from gpuc.control.config import HostEntry, Settings
from gpuc.control.remote import HostSession, RemoteError, open_session
from gpuc.control.transport import TransportError

HEARTBEAT_STALE_S = 30.0
SUSPECT_FLOOR_PCT = 5.0
SUSPECT_SAMPLES = 20  # 10 minutes of main-phase samples at the 30 s runner cadence
RECENT_FINISHED = 5


@dataclass
class JobView:
    job_id: str
    name: str = ""
    status: str = "queued"
    phase: str | None = None
    priority: int | None = None
    gpus: list[str] = field(default_factory=list)
    reason: str | None = None
    exit_code: int | None = None
    attempt: int = 1
    started_at: str | None = None
    ended_at: str | None = None
    util_recent: list[float] = field(default_factory=list)

    @property
    def minutes(self) -> float | None:
        if not self.started_at:
            return None
        end = _parse(self.ended_at) if self.ended_at else datetime.now(UTC)
        start = _parse(self.started_at)
        if start is None or end is None:
            return None
        return (end - start).total_seconds() / 60.0

    @property
    def last_util(self) -> float | None:
        return self.util_recent[-1] if self.util_recent else None

    @property
    def suspect(self) -> bool:
        """Billing, main phase, and utilization flat on the floor for 10 minutes.

        Phase-aware by construction: the runner only records samples during
        `main`, so setup, download and compile can never look suspicious.
        """
        if self.status != "running" or self.phase != "main" or not self.gpus:
            return False
        window = self.util_recent[-SUSPECT_SAMPLES:]
        if len(window) < SUSPECT_SAMPLES:
            return False
        return sum(window) / len(window) < SUSPECT_FLOOR_PCT


def _parse(stamp: str | None) -> datetime | None:
    if not stamp:
        return None
    try:
        parsed = datetime.fromisoformat(stamp)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


@dataclass
class HostView:
    entry: HostEntry
    reachable: bool = False
    error: str | None = None
    heartbeat_age_s: float | None = None
    draining: bool = False
    paused: bool = False
    owned: list[str] = field(default_factory=list)
    queue: list[JobView] = field(default_factory=list)
    running: list[JobView] = field(default_factory=list)
    finished: list[JobView] = field(default_factory=list)

    @property
    def dispatcher_alive(self) -> bool:
        return self.heartbeat_age_s is not None and self.heartbeat_age_s < HEARTBEAT_STALE_S

    @property
    def free(self) -> list[str]:
        busy = {uuid for job in self.running for uuid in job.gpus}
        return [uuid for uuid in self.owned if uuid not in busy]

    @property
    def suspects(self) -> list[JobView]:
        return [job for job in self.running if job.suspect]

    @property
    def past_ttl(self) -> bool:
        if not self.entry.ephemeral or not self.entry.created_at:
            return False
        created = _parse(self.entry.created_at)
        if created is None:
            return False
        return (datetime.now(UTC) - created).total_seconds() / 3600.0 > self.entry.ttl_hours


def job_views(payload: dict[str, Any]) -> tuple[list[JobView], list[JobView], list[JobView]]:
    priorities = {e["job_id"]: e["priority"] for e in payload.get("queue", [])}
    queued: list[JobView] = []
    running: list[JobView] = []
    finished: list[JobView] = []
    for entry in payload.get("jobs", []):
        view = JobView(
            job_id=entry["job_id"],
            name=entry.get("name", ""),
            status=entry.get("status", "queued"),
            phase=entry.get("phase"),
            priority=priorities.get(entry["job_id"]),
            gpus=list(entry.get("gpus") or []),
            reason=entry.get("reason"),
            exit_code=entry.get("exit_code"),
            attempt=entry.get("attempt", 1),
            started_at=entry.get("started_at"),
            ended_at=entry.get("ended_at"),
            util_recent=[float(u) for u in entry.get("util_recent") or []],
        )
        if view.status == "running":
            running.append(view)
        elif view.status == "queued":
            queued.append(view)
        else:
            finished.append(view)
    queued.sort(key=lambda v: (v.priority if v.priority is not None else 99, v.job_id))
    finished.sort(key=lambda v: v.ended_at or "", reverse=True)
    return queued, running, finished


def gather(
    entry: HostEntry, settings: Settings | None = None, *, session: HostSession | None = None
) -> HostView:
    view = HostView(entry=entry, owned=list(entry.gpus))
    try:
        session = session or open_session(entry, settings)
        payload = session.host_json("status", timeout=60.0)
    except (RemoteError, TransportError) as exc:
        view.error = str(exc).splitlines()[0]
        return view
    view.reachable = True
    view.owned = list(payload.get("gpus") or entry.gpus)
    view.heartbeat_age_s = payload.get("dispatcher_heartbeat_age_s")
    view.draining = bool(payload.get("draining"))
    view.paused = bool(payload.get("paused"))
    view.queue, view.running, view.finished = job_views(payload)
    return view


def _fmt_util(job: JobView) -> str:
    return "util --" if job.last_util is None else f"util {job.last_util:.0f}%"


def _fmt_minutes(job: JobView) -> str:
    return "--" if job.minutes is None else f"{job.minutes:.1f}m"


def render(view: HostView, *, recent: int = RECENT_FINISHED, suspects_only: bool = False) -> str:
    entry = view.entry
    target = entry.ssh or "this machine"
    if not view.reachable:
        return (
            f"host {entry.name} [{entry.kind}] {target}: UNREACHABLE\n"
            f"  {view.error}\n"
            f"  try: gpuc host probe {entry.name}"
        )
    flags = []
    if view.draining:
        flags.append("DRAINING")
    if view.paused:
        flags.append("PAUSED (low-util); resume with `gpuc host bootstrap`")
    dispatcher = (
        f"dispatcher {view.heartbeat_age_s:.0f}s ago"
        if view.dispatcher_alive
        else "dispatcher DOWN (submit or bootstrap restarts it)"
    )
    header = (
        f"host {entry.name} [{entry.kind}] {target}  {dispatcher}  "
        f"gpus {len(view.free)}/{len(view.owned)} free"
    )
    if flags:
        header += "  " + " ".join(flags)
    lines = [header]

    if suspects_only:
        for job in view.suspects:
            lines.append(
                f"  SUSPECT {job.job_id} {job.name or '-'} phase={job.phase} "
                f"{_fmt_minutes(job)} {_fmt_util(job)}"
            )
        if view.past_ttl:
            lines.append(f"  SUSPECT pod for host {entry.name} is older than {entry.ttl_hours}h")
        if len(lines) == 1:
            lines.append("  no suspects")
        return "\n".join(lines)

    for job in view.running:
        mark = "  running" if not job.suspect else "  running!"
        lines.append(
            f"{mark} {job.job_id} {job.name or '-'} phase={job.phase or '-'} "
            f"{_fmt_minutes(job)} {_fmt_util(job)} gpus={len(job.gpus)}"
        )
    for job in view.queue:
        lines.append(f"  queued  {job.job_id} {job.name or '-'} prio={job.priority}")
    for job in view.finished[:recent]:
        detail = job.reason or (f"exit {job.exit_code}" if job.exit_code else "")
        lines.append(
            f"  done    {job.job_id} {job.name or '-'} {job.status}"
            f"{f' ({detail})' if detail else ''}"
        )
    if len(lines) == 1:
        lines.append("  idle; nothing queued, running or finished")
    return "\n".join(lines)
