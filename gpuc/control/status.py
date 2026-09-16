"""`gpuc status`: one compact block per host, and the phase-aware suspect rule.

Reads the host over the transport (the host is authoritative); the S3 index
only fills in jobs whose host is gone. Never kills anything.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from gpuc.control import version
from gpuc.control.config import HostEntry, Settings
from gpuc.control.gpuinfo import GpuInfo, summarize
from gpuc.control.gpuinfo import rows as gpu_rows
from gpuc.control.providers.base import Pod, Provider, ProviderError
from gpuc.control.remote import HostSession, RemoteError, open_session
from gpuc.control.transport import TransportError
from gpuc.host.cleanup import human_bytes
from gpuc.host.jobs import SCHEMA_VERSION

HEARTBEAT_STALE_S = 30.0
DEAD_POD_STATUSES = ("EXITED", "ERROR", "TERMINATED")
UTIL_SAMPLE_INTERVAL_S = 30.0
"""The runner's sampling cadence, which is what `util_recent` is measured in."""
UTIL_SAMPLES_KEPT = 40
"""How many samples the host keeps (20 min). A `window_min` longer than this is
judged on what there is rather than never firing at all."""
RECENT_FINISHED = 5
LEFTOVER_FLOOR_BYTES = 1 << 30
"""Only mention finished jobs' workdirs once they add up to something worth a
command. A job dir under a gigabyte is noise next to a 6.5 GB torch venv."""


@dataclass
class LowUtilView:
    """A job's own low-util watchdog settings, as the host reports them.

    The defaults are the spec's, so a host that does not report them yet is
    judged by exactly the rule its watchdog is running.
    """

    enabled: bool = True
    window_min: float = 25.0
    floor_pct: float = 5.0
    grace_min: float = 10.0

    @staticmethod
    def from_payload(raw: Any) -> LowUtilView:
        if not isinstance(raw, dict):
            return LowUtilView()
        default = LowUtilView()

        def number(key: str, fallback: float) -> float:
            value = raw.get(key)
            return float(value) if isinstance(value, (int, float)) else fallback

        return LowUtilView(
            enabled=bool(raw.get("enabled", True)),
            window_min=number("window_min", default.window_min),
            floor_pct=number("floor_pct", default.floor_pct),
            grace_min=number("grace_min", default.grace_min),
        )

    def samples(self, minutes: float) -> int:
        return max(1, round(minutes * 60.0 / UTIL_SAMPLE_INTERVAL_S))


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
    progress_pct: float | None = None
    """How far along the job's own `progress_command` last said it was."""
    eta: str | None = None
    """When the host expects this job to finish: measured from `progress_pct`
    where there is one, from the spec's `estimated_runtime_min` otherwise."""
    estimated_runtime_min: float | None = None
    """The submitter's own estimate, which is all a *queued* job has."""
    progress_error: str | None = None
    """Why this job's `progress_command` last produced nothing.

    Surfaced because the failure is logged into a `log.txt` that then fills up
    with hours of training output: without this, a typo'd progress command is
    indistinguishable from a job that never had one."""
    workdir_bytes: int | None = None
    """Disk still held by this job's `workdir/`; the host only measures it for
    finished jobs."""
    outputs_pending: bool = False
    """The job declared `outputs:` and no final upload of them was confirmed, so
    what it produced may exist only on this host. The host works this out; the
    control side never sees the spec."""
    outputs_lost: bool = False
    """An ephemeral host's drain retried the upload to the end and gave up."""
    isolation: str | None = None
    """`cgroup` if this job's phases run in a systemd scope (a cancel reaps the
    whole tree), `pgid` if only a process group (a daemonised grandchild
    escapes). Shown on running jobs because it changes what a kill guarantees."""
    low_util: LowUtilView = field(default_factory=LowUtilView)
    """This job's own watchdog settings, so `--suspects` names the jobs the host
    is actually about to kill -- and stays quiet about the ones that turned the
    watchdog off on purpose."""

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
    def eta_seconds(self) -> float | None:
        """Seconds until the estimated finish; negative once it is overdue.

        Read here rather than on the host so a `status` of a host whose clock
        or whose last report is minutes old still counts down.
        """
        when = _parse(self.eta)
        return None if when is None else (when - datetime.now(UTC)).total_seconds()

    @property
    def suspect(self) -> bool:
        """Billing, in `main`, and flat on this job's own low-util floor.

        Phase-aware by construction: the runner only records samples during
        `main`, so setup, download and compile can never look suspicious. The
        thresholds are the job's, not a constant here, so a job that raised its
        floor or turned the watchdog off is judged by what it asked for -- and
        the ones this flags are the ones the host is about to kill.
        """
        if self.status != "running" or self.phase != "main" or not self.gpus:
            return False
        rule = self.low_util
        if not rule.enabled:
            return False
        # grace_min of main phase has to have gone by before the host's own
        # watchdog even starts watching, and its window is what it averages.
        need = min(rule.samples(rule.grace_min + rule.window_min), UTIL_SAMPLES_KEPT)
        window = self.util_recent[-min(rule.samples(rule.window_min), UTIL_SAMPLES_KEPT) :]
        if len(self.util_recent) < need or not window:
            return False
        return sum(window) / len(window) < rule.floor_pct


DURATION_UNITS = {"s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0, "w": 604800.0}


def parse_duration(text: str) -> float:
    """`24h`, `7d`, `90m`, `45s`, or a bare number of hours. Seconds out."""
    raw = text.strip().lower()
    if not raw:
        raise ValueError("empty duration")
    unit = DURATION_UNITS.get(raw[-1])
    number = raw[:-1] if unit else raw
    try:
        value = float(number)
    except ValueError as exc:
        raise ValueError(
            f"{text!r} is not a duration: use 30m, 24h, 7d, or a bare number of hours"
        ) from exc
    if value < 0:
        raise ValueError(f"{text!r} is negative")
    return value * (unit or 3600.0)


def format_age(stamp: str | None, now: datetime | None = None) -> str:
    """`3m ago`, `2d ago`: enough to tell last night's run from last month's."""
    when = _parse(stamp)
    if when is None:
        return "age unknown"
    seconds = ((now or datetime.now(UTC)) - when).total_seconds()
    if seconds < 0:
        return "just now"
    for unit, size in (("d", 86400.0), ("h", 3600.0), ("m", 60.0)):
        if seconds >= size:
            return f"{int(seconds // size)}{unit} ago"
    return f"{int(seconds)}s ago"


def format_duration(seconds: float) -> str:
    """`2h10m`, `35m`, `50s`: how much longer, not a wall-clock time.

    Relative on purpose. Every consumer of this is deciding "do I wait or do I
    go somewhere else", and a clock time makes them do the subtraction -- in
    whichever timezone the host happens to think it is in.
    """
    # Rounded, not truncated, unlike `format_age`: an age of 59 minutes really
    # is "not an hour yet", but an estimate 45 minutes out printed as `44m` is
    # just wrong by the width of the arithmetic.
    seconds = max(0.0, seconds)
    minutes = round(seconds / 60.0)
    if minutes == 0:
        return f"{round(seconds)}s"
    if minutes < 60:
        return f"{minutes}m"
    hours, rest = divmod(minutes, 60)
    if hours < 24:
        return f"{hours}h{rest:02d}m"
    # A sweep really does run for days, and `est 72h00m` beside a `2d ago` on
    # the line above is the kind of unit mismatch a reader has to stop and do
    # arithmetic on.
    return f"{hours // 24}d{hours % 24:02d}h"


def _as_float(value: Any) -> float | None:
    """A number the host sent, or None for anything else (including a bool)."""
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _as_int(value: Any) -> int | None:
    number = _as_float(value)
    return None if number is None else int(number)


def _as_str(value: Any) -> str | None:
    return value if isinstance(value, str) else None


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
    pod_gone: bool = False
    draining: bool = False
    paused: bool = False
    owned: list[str] = field(default_factory=list)
    """The UUIDs this host owns, as the host itself resolved them: `config.gpus`
    may name cards by nvidia-smi index, and only the host knows today's
    numbering. Everything here -- free, busy, the per-card lines -- is UUIDs."""
    indices: dict[str, int] = field(default_factory=dict)
    """uuid -> the index the host is calling that card right now."""
    unavailable: list[str] = field(default_factory=list)
    """Owned entries the host could not resolve to a card it can see."""
    pod: Pod | None = None
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

    def gpu_holder(self, uuid: str) -> str | None:
        """Which running job has this card, per the host's own state."""
        for job in self.running:
            if uuid in job.gpus:
                return job.job_id
        return None

    @property
    def outputs_at_risk(self) -> list[JobView]:
        """Finished jobs holding the only copy of what they produced."""
        return [job for job in self.finished if job.outputs_pending]

    @property
    def leftover_bytes(self) -> int:
        """Disk held by workdirs of jobs that are over: reclaimable by `gpuc clean`."""
        return sum(job.workdir_bytes or 0 for job in self.finished)

    @property
    def past_ttl(self) -> bool:
        """Age from the provider's own createdAt when we have it.

        The registry's created_at is when *this* machine recorded the host,
        which is not the same clock the reaper's TTL uses; a pod adopted or
        re-registered later would read as young here and be terminated there.
        """
        if not self.entry.ephemeral or self.entry.ttl_hours is None:
            return False
        if self.pod is not None and self.pod.age is not None:
            return self.pod.age.total_seconds() / 3600.0 > self.entry.ttl_hours
        created = _parse(self.entry.created_at)
        if created is None:
            return False
        return (datetime.now(UTC) - created).total_seconds() / 3600.0 > self.entry.ttl_hours


def job_views(payload: dict[str, Any]) -> tuple[list[JobView], list[JobView], list[JobView]]:
    """Every field is treated as untrusted: this is another build's JSON.

    A host on a different commit -- or a half-written state file -- must cost
    one missing row, never a traceback out of `gpuc status` for every host.
    """
    priorities = {
        e["job_id"]: _as_int(e.get("priority"))
        for e in payload.get("queue") or []
        if isinstance(e, dict) and isinstance(e.get("job_id"), str)
    }
    queued: list[JobView] = []
    running: list[JobView] = []
    finished: list[JobView] = []
    for entry in payload.get("jobs") or []:
        if not isinstance(entry, dict) or not isinstance(entry.get("job_id"), str):
            continue
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
            # A sample is null when nvidia-smi failed; drop it rather than
            # counting a missing reading as 0% and calling the job a suspect.
            util_recent=[
                float(u) for u in entry.get("util_recent") or [] if isinstance(u, (int, float))
            ],
            progress_pct=_as_float(entry.get("progress_pct")),
            eta=_as_str(entry.get("eta")),
            estimated_runtime_min=_as_float(entry.get("estimated_runtime_min")),
            progress_error=_as_str(entry.get("progress_error")),
            workdir_bytes=entry.get("workdir_bytes"),
            outputs_pending=bool(entry.get("outputs_pending")),
            outputs_lost=bool(entry.get("outputs_lost")),
            isolation=entry.get("isolation"),
            low_util=LowUtilView.from_payload(entry.get("low_util")),
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
    entry: HostEntry,
    settings: Settings | None = None,
    *,
    session: HostSession | None = None,
    provider: Provider | None = None,
) -> HostView:
    view = HostView(entry=entry, owned=list(entry.gpus))
    if provider is not None and entry.kind == "runpod" and entry.pod_id:
        try:
            view.pod = provider.get(entry.pod_id)
            view.pod_gone = view.pod is None or view.pod.status in DEAD_POD_STATUSES
        except ProviderError as exc:
            view.error = f"could not read pod {entry.pod_id}: {exc}"
    if view.pod_gone:
        # The pod is gone but the registry still lists it. SSH would hang and
        # then print a stack about a refused connection, which tells nobody
        # anything: say what happened and what removes the entry.
        status = "missing" if view.pod is None else view.pod.status
        view.error = (
            f"pod {entry.pod_id} is {status}; the registry entry is stale. "
            f"Run `gpuc reconcile --once` to forget it."
        )
        return view
    try:
        session = session or open_session(entry, settings)
        payload = session.host_json("status", timeout=60.0)
    except (RemoteError, TransportError) as exc:
        view.error = str(exc).splitlines()[0]
        return view
    if not isinstance(payload, dict):
        view.error = f"host {entry.name} answered `status` with {type(payload).__name__}, not JSON"
        return view
    view.reachable = True
    view.owned, view.indices = owned_gpus(payload, entry)
    view.unavailable = [g for g in payload.get("gpus_unavailable") or [] if isinstance(g, str)]
    # Validated like `reconcile.probe_liveness` does: a host on another build
    # could answer with a string here, and formatting it would take out the
    # whole `gpuc status`, not just this host's line.
    view.heartbeat_age_s = _as_float(payload.get("dispatcher_heartbeat_age_s"))
    view.draining = bool(payload.get("draining"))
    view.paused = bool(payload.get("paused"))
    view.queue, view.running, view.finished = job_views(payload)
    return view


def pod_line(pod: Pod | None) -> str | None:
    """The provider's own view of the pod.

    Its utilization is labelled `provider util` because it is not the number on
    the running line: that one is the host's nvidia-smi sampler, averaged over
    the job's own cards. The two legitimately differ -- different sampler,
    different instant, and a pod may hold cards this host does not own -- and
    an unlabelled pair of percentages reads as a bug.
    """
    if pod is None:
        return None
    age = "age ?" if pod.age is None else f"age {pod.age.total_seconds() / 60.0:.0f}m"
    util = ",".join(f"{u}%" for u in pod.gpu_utils) if pod.gpu_utils else "--"
    return (
        f"  pod     {pod.id} {pod.status} {pod.gpu_name or '?'} "
        f"${pod.cost_usd_hr:.3f}/h cuda {pod.cuda_version or '?'} {age} provider util {util}"
    )


def _fmt_util(job: JobView) -> str:
    """`(host)` names the source: the host's sampler, not the provider's."""
    return "util -- (host)" if job.last_util is None else f"util {job.last_util:.0f}% (host)"


def _fmt_minutes(job: JobView) -> str:
    return "--" if job.minutes is None else f"{job.minutes:.1f}m"


def _fmt_eta(job: JobView) -> str:
    """` eta 2h10m (42%)` when the job measures its own progress, ` eta 2h10m
    (est)` when all it has is the submitter's guess, and nothing at all when it
    has neither. The tag matters: one of those numbers is evidence.

    A running job whose host published no `eta` falls back to the estimate the
    same host reports, rendered as a total rather than a remaining time. The
    one thing that may not happen is `--json` carrying an estimate the text
    does not show, and there are two ways to reach that: the window after
    `gpuc estimate` before the runner next re-reads the spec, and a host on a
    build old enough not to re-read it at all."""
    remaining = job.eta_seconds
    if remaining is None:
        return _fmt_estimate(job, total=True)
    # `not job.progress_pct` and not `is None`: at 0% the runner deliberately
    # leaves the submitter's estimate in place, so the eta being shown is the
    # guess and tagging it `(0%)` would claim evidence that is not there.
    source = f"{job.progress_pct:.0f}%" if job.progress_pct else "est"
    if remaining < 0:
        return f" eta overdue ({source})"
    return f" eta {format_duration(remaining)} ({source})"


def _fmt_estimate(job: JobView, *, total: bool = False) -> str:
    """` est 2h30m`: the whole run, not what is left of it. `total` says so out
    loud, for the lines that also carry elapsed or remaining times."""
    if job.estimated_runtime_min is None:
        return ""
    return f" est {format_duration(job.estimated_runtime_min * 60.0)}{' total' if total else ''}"


def next_free_line(view: HostView) -> str | None:
    """When a card on a fully-busy host is expected to come free.

    The whole point of the estimates: whether to queue behind what is running
    or go and pay for a pod. It says how many of the running jobs offered
    nothing to estimate from, because the real answer can only be *sooner* than
    this -- one of those could finish in a minute.

    It is a time or it is nothing. A host where no running job estimated
    anything has no answer to give, and saying so under a `free` label, next to
    a gpu list that already says every card is busy, is a line to scan past.
    """
    # `gpus: 0` jobs are running but hold no card, so they can never be the
    # reason one comes free -- and naming a five-minute CPU job as the next
    # card would answer the one question this line exists for with a lie.
    holding = [job for job in view.running if job.gpus]
    if not view.owned or view.free or not holding:
        return None
    known: list[tuple[float, JobView]] = []
    for job in holding:
        remaining = job.eta_seconds
        if remaining is not None:
            known.append((remaining, job))
    if not known:
        return None
    silent = len(holding) - len(known)
    remaining, job = min(known, key=lambda pair: pair[0])
    when = "overdue" if remaining < 0 else f"in ~{format_duration(remaining)}"
    # "no end time", not "no estimate": a job whose host has an estimate it has
    # not turned into an eta yet is counted here, and its own line above says
    # `est ...`. Two lines of one host block may not contradict each other.
    note = f"; {silent} other running job(s) gave no end time" if silent else ""
    return f"  free    next card {when} ({job.job_id}){note}"


def owned_gpus(payload: dict[str, Any], entry: HostEntry) -> tuple[list[str], dict[str, int]]:
    """The host's resolved cards, falling back to what it was configured with.

    A host running a build from before the resolution existed reports only
    `gpus`, which on that build could only ever have been UUIDs.
    """
    owned: list[str] = []
    indices: dict[str, int] = {}
    for row in payload.get("gpus_resolved") or []:
        if not isinstance(row, dict) or not isinstance(row.get("uuid"), str):
            continue
        uuid = row["uuid"]
        owned.append(uuid)
        index = _as_int(row.get("index"))
        if index is not None:
            indices[uuid] = index
    if owned or payload.get("gpus_resolved") is not None:
        return owned, indices
    return [g for g in (payload.get("gpus") or entry.gpus) if isinstance(g, str)], indices


def _gpu_lines(view: HostView) -> list[str]:
    """One line per owned card: which one it is, what it is, and who has it."""
    lines: list[str] = []
    for index, name, vram, uuid in gpu_rows(view.owned, view.entry.gpu_info, view.indices):
        holder = view.gpu_holder(uuid)
        lines.append(
            f"  gpu     [{index}] {name} {vram}".rstrip()
            + f"  {uuid}  {f'busy {holder}' if holder else 'free'}"
        )
    for missing in view.unavailable:
        lines.append(
            f"  gpu     [{missing}] UNAVAILABLE  nvidia-smi does not report this card on the "
            f"host, so nothing is dispatched to it"
        )
    return lines


def within(job: JobView, since_s: float | None, now: datetime | None = None) -> bool:
    if since_s is None:
        return True
    ended = _parse(job.ended_at)
    if ended is None:
        return False
    return ((now or datetime.now(UTC)) - ended).total_seconds() <= since_s


def render(
    view: HostView,
    *,
    recent: int = RECENT_FINISHED,
    suspects_only: bool = False,
    since_s: float | None = None,
) -> str:
    entry = view.entry
    target = entry.ssh or "this machine"
    if not view.reachable:
        state = "POD GONE" if view.pod_gone else "UNREACHABLE"
        lines = [
            f"host {entry.name} [{entry.kind}] {target}: {state}",
            f"  {view.error}",
        ]
        pod = pod_line(view.pod)
        if pod:
            lines.append(pod)
        if not view.pod_gone:
            lines.append(f"  try: gpuc host probe {entry.name}")
        return "\n".join(lines)
    flags = []
    if view.draining:
        flags.append("DRAINING")
    if view.paused:
        flags.append(f"PAUSED (low-util); resume with `gpuc host resume {entry.name}`")
    dispatcher = (
        f"dispatcher {view.heartbeat_age_s:.0f}s ago"
        if view.dispatcher_alive
        else "dispatcher DOWN (submit or bootstrap restarts it)"
    )
    summary = summarize(view.owned, entry.gpu_info) if view.owned else "no GPUs"
    driver = f", driver {entry.driver_version}" if entry.driver_version else ""
    header = (
        f"host {entry.name} [{entry.kind}] {target}  {dispatcher}  "
        f"gpus {len(view.free)}/{len(view.owned)} free ({summary}{driver})"
    )
    if flags:
        header += "  " + " ".join(flags)
    lines = [header]
    stale = stale_warning(entry)
    if stale:
        lines.append(f"  WARNING {stale}")
    if not suspects_only:
        lines += _gpu_lines(view)
    pod = pod_line(view.pod)
    if pod:
        lines.append(pod + ("  PAST TTL" if view.past_ttl else ""))
    # Where the per-job body starts. The header, the gpu lines and the pod line
    # are about the host, not about what is on it, so "nothing here" has to be
    # measured from here -- a host with GPUs printed neither `idle` nor `no
    # suspects` while this was compared against the whole list.
    body_start = len(lines)

    if suspects_only:
        for job in view.suspects:
            lines.append(
                f"  SUSPECT {job.job_id} {job.name or '-'} phase={job.phase} "
                f"{_fmt_minutes(job)} {_fmt_util(job)}"
            )
        if view.past_ttl:
            lines.append(f"  SUSPECT pod for host {entry.name} is older than {entry.ttl_hours}h")
        if len(lines) == body_start:
            lines.append("  no suspects")
        return "\n".join(lines)

    for job in view.running:
        mark = "  running" if not job.suspect else "  running!"
        lines.append(
            f"{mark} {job.job_id} {job.name or '-'} phase={job.phase or '-'} "
            f"{_fmt_minutes(job)} {_fmt_util(job)} gpus={len(job.gpus)} "
            f"iso={job.isolation or '?'}{_fmt_eta(job)}"
        )
    for job in view.queue:
        lines.append(
            f"  queued  {job.job_id} {job.name or '-'} prio={job.priority}{_fmt_estimate(job)}"
        )
    free = next_free_line(view)
    if free:
        lines.append(free)
    finished = [job for job in view.finished if within(job, since_s)]
    for job in finished[:recent]:
        detail = job.reason or (f"exit {job.exit_code}" if job.exit_code else "")
        flag = ""
        if job.outputs_lost and job.outputs_pending:
            # `outputs_lost` is written once and never cleared, so it outlives
            # the thing it describes: a job the host now reports as holding
            # nothing is not a lost result, whatever a past drain concluded.
            flag = "  OUTPUTS LOST"
        elif job.outputs_pending:
            flag = "  outputs not uploaded"
        lines.append(
            f"  done    {job.job_id} {job.name or '-'} {job.status}"
            f"{f' ({detail})' if detail else ''} {format_age(job.ended_at)}{flag}"
        )
    if since_s is not None and not finished and view.finished:
        lines.append(
            f"  done    none in the last {int(since_s // 60)} min ({len(view.finished)} older)"
        )
    at_risk = view.outputs_at_risk
    if at_risk:
        # Worth a line of its own: these are the jobs whose results a purge (or
        # a pod going away) would take with them, and only a human can decide
        # whether to requeue them or copy them off.
        lines.append(
            f"  outputs {len(at_risk)} finished job(s) produced outputs that never reached "
            f"S3/HF: {', '.join(job.job_id for job in at_risk[:3])}"
            f"{' ...' if len(at_risk) > 3 else ''}; `gpuc requeue` them or copy them off "
            f"before they are purged"
        )
    leftover = view.leftover_bytes
    if leftover > LEFTOVER_FLOOR_BYTES:
        held = [job for job in view.finished if (job.workdir_bytes or 0) > 0]
        lines.append(
            f"  disk    {human_bytes(leftover)} still in {len(held)} finished job workdir(s); "
            f"free it with: gpuc clean --host {entry.name} --all-finished"
        )
    if len(lines) == body_start:
        lines.append("  idle; nothing queued, running or finished")
    return "\n".join(lines)


def stale_warning(entry: HostEntry) -> str | None:
    """One line when this host's package is not the build running here.

    Two sessions of the same user on different commits, writing one shared
    registry and one on-host config, is what turned a field becoming optional
    into an hour of broken CLI. The cheap half of noticing is free: bootstrap
    already recorded the commit it shipped.
    """
    return version.stale_host_warning(entry.name, entry.pkg_commit, version.local_commit())


def job_json(job: JobView) -> dict[str, Any]:
    """The text view's fields, named the same, with nothing rendered.

    `running` is the list automation should key on. It is the host's own
    answer, so an empty list here means the host said "nothing is running" --
    never "we could not ask", which is `reachable: false` and an `errors` entry.
    """
    return {
        "job_id": job.job_id,
        "name": job.name,
        "status": job.status,
        "reason": job.reason,
        "phase": job.phase,
        "elapsed_s": None if job.minutes is None else round(job.minutes * 60.0, 1),
        "util": job.last_util,
        "progress_pct": job.progress_pct,
        "eta": job.eta,
        "eta_s": None if job.eta_seconds is None else round(job.eta_seconds, 1),
        "estimated_runtime_min": job.estimated_runtime_min,
        "progress_error": job.progress_error,
        "gpus": list(job.gpus),
        "iso": job.isolation,
        "ended_at": job.ended_at,
        "outputs_pending": job.outputs_pending,
    }


def gpu_json(view: HostView) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for uuid in view.owned:
        info = view.entry.gpu_info.get(uuid) or GpuInfo()
        out.append(
            {
                "index": view.indices.get(uuid, info.index),
                "uuid": uuid,
                "name": info.name,
                "vram_mib": info.vram_mib,
                "busy_job": view.gpu_holder(uuid),
            }
        )
    out += [{"owned_as": item, "available": False} for item in view.unavailable]
    return out


def host_json(
    view: HostView, *, recent: int = RECENT_FINISHED, since_s: float | None = None
) -> dict[str, Any]:
    """`provider_util` is the provider's per-GPU reading for an ephemeral host's
    pod (null for every other host); each job's `util` is the host's own
    sampler. They are two different measurements and are named as such."""
    entry = view.entry
    errors = [view.error] if view.error else []
    stale = stale_warning(entry)
    if stale:
        errors.append(stale)
    finished = [job for job in view.finished if within(job, since_s)][:recent]
    return {
        "name": entry.name,
        "kind": entry.kind,
        "reachable": view.reachable,
        "pkg_commit": entry.pkg_commit,
        "dispatcher": {
            "alive": view.dispatcher_alive,
            "heartbeat_age_s": view.heartbeat_age_s,
        },
        "provider_util": list(view.pod.gpu_utils) if view.pod is not None else None,
        "gpus": gpu_json(view),
        "queued": [job_json(job) for job in view.queue],
        "running": [job_json(job) for job in view.running],
        "finished": [job_json(job) for job in finished],
        "errors": errors,
    }


def document(
    views: Sequence[HostView],
    *,
    errors: Sequence[str] = (),
    recent: int = RECENT_FINISHED,
    since_s: float | None = None,
) -> dict[str, Any]:
    """The whole of `gpuc status --json`: one object, always this shape.

    Top-level `errors` are the ones that belong to no host -- an unreadable
    registry, a skipped entry -- and they are the reason exit 3 exists: a
    consumer that sees them must not read `hosts` as the whole truth.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "hosts": [host_json(view, recent=recent, since_s=since_s) for view in views],
        "errors": list(errors),
    }
