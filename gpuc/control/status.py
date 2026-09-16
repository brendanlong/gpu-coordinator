"""`gpuc status`: one compact block per host, and the phase-aware suspect rule.

Reads the host over the transport (the host is authoritative); the S3 index
only fills in jobs whose host is gone. Never kills anything.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.parse import quote

from gpuc.control import version
from gpuc.control.config import HostEntry, Settings
from gpuc.control.gpuinfo import GpuInfo
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
class SharedGpu:
    """One card this host may borrow, and what the host last saw on it.

    The memory and utilization are somebody else's, not ours: a shared card we
    are running on is `busy` here through the ordinary holder lookup, and these
    numbers exist to answer "why is my job still queued" when it is not.
    """

    uuid: str
    index: int | None = None
    memory_mib: float | None = None
    utilization_pct: float | None = None
    unused: bool = False
    """The host's own verdict: no memory held and no work running, so gpuc
    would borrow it. Never inferred here -- a host that could not read a card
    reports it unused=false, which is what keeps a failed read off the card."""

    def describe(self) -> str:
        memory = "?" if self.memory_mib is None else f"{self.memory_mib:.0f}"
        util = "?" if self.utilization_pct is None else f"{self.utilization_pct:.0f}"
        return f"{memory} MiB, {util}% util"

    @staticmethod
    def from_payload(raw: Any) -> SharedGpu | None:
        if not isinstance(raw, dict) or not isinstance(raw.get("uuid"), str):
            return None
        return SharedGpu(
            uuid=raw["uuid"],
            index=_as_int(raw.get("index")),
            memory_mib=_as_float(raw.get("memory_mib")),
            utilization_pct=_as_float(raw.get("utilization_pct")),
            unused=bool(raw.get("unused")),
        )


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
    gpus_requested: int | None = None
    """How many cards the spec asked for. A queued job holds none yet, so this
    is the only thing that says whether it is waiting for one card or eight.
    None from a host on a build that does not report it."""
    use_shared: bool = False
    """The spec said this job may borrow the host's shared cards, so the ones
    it is waiting for are not only the ones the host owns."""
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
    outputs: list[dict[str, Any]] = field(default_factory=list)
    """The spec's `outputs:` as the host reports them, `{job_id}` already
    expanded: where this job's results went, or were meant to go."""
    wandb: dict[str, str] = field(default_factory=dict)
    """`entity`, `project` and `run_id` from the job's `WANDB_*` environment,
    when it set them; enough to link to the run and nothing more."""

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


def _first_int(marker: int | None, reported: Any) -> int | None:
    """`0` is a real priority -- the highest one -- so this cannot be an `or`."""
    return marker if marker is not None else _as_int(reported)


def _as_str(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _str_dict(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    return {k: v for k, v in value.items() if isinstance(k, str) and isinstance(v, str)}


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
    shared: list[SharedGpu] = field(default_factory=list)
    """Cards this host may borrow, and what the host just saw on each."""
    shared_unavailable: list[str] = field(default_factory=list)
    """Shared entries the host could not resolve to a card it can see."""
    shared_min_priority: int | None = None
    """The floor a job's priority must clear to borrow one. None means no floor."""
    pod: Pod | None = None
    pkg_commit: str | None = None
    """The commit the *host* says its package came from, not the one this
    machine's registry remembers shipping. Null when the host was not asked or
    was bootstrapped by a build too old to record it."""
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

    def gpu_label(self, uuid: str) -> str:
        """What to call this card on a job's line: its index where we know it.

        The index is what the gpu lines above are numbered by, so `gpu=2,3`
        points at two of them. A card the host never resolved falls back to its
        UUID rather than a `?` that could be any of several.
        """
        index = self.indices.get(uuid)
        if index is None:
            info = self.entry.gpu_info.get(uuid)
            index = info.index if info else None
        return uuid if index is None else str(index)

    def gpu_holder(self, uuid: str) -> str | None:
        """Which running job has this card, per the host's own state."""
        for job in self.running:
            if uuid in job.gpus:
                return job.job_id
        return None

    def may_borrow(self, job: JobView) -> bool:
        """Could this job be dispatched to a shared card at all?

        The host's rule (`HostConfig.may_borrow`) repeated over what the host
        reported, so the two cannot disagree about why a job is waiting. A
        priority the host did not report counts as not clearing a floor: the
        answer this feeds is an explanation, and guessing one is worse than the
        general line it falls back to.
        """
        # `shared_unavailable` counts: the host judges a job against the cards
        # it is *configured* with, and makes it wait for one that is missing
        # this minute rather than failing it.
        if not job.use_shared or not (self.shared or self.shared_unavailable):
            return False
        if self.shared_min_priority is None:
            return True
        return job.priority is not None and job.priority <= self.shared_min_priority

    @property
    def borrowable(self) -> list[SharedGpu]:
        """Shared cards nobody is on: neither one of ours nor anybody else's."""
        return [c for c in self.shared if c.unused and not self.gpu_holder(c.uuid)]

    @property
    def outputs_at_risk(self) -> list[JobView]:
        """Finished jobs holding the only copy of what they produced."""
        return [job for job in self.finished if job.outputs_pending]

    @property
    def leftover_bytes(self) -> int:
        """Disk held by workdirs of jobs that are over: reclaimable by `gpuc clean`.

        The host answers for all but the finished job it ran out of measuring
        budget for, and that one is measured by the next call. A null that
        never resolves means a host too old to answer, which is what the build
        warning in `host_warnings` is for.
        """
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
    # The queue marker is the priority the dispatcher is actually ordering by,
    # so it wins while a job is queued; the job's own `priority` (from its
    # spec) is what is left once the marker is gone, and is all a running job
    # ever has.
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
            priority=_first_int(priorities.get(entry["job_id"]), entry.get("priority")),
            gpus=list(entry.get("gpus") or []),
            gpus_requested=_as_int(entry.get("gpus_requested")),
            use_shared=bool(entry.get("use_shared")),
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
            workdir_bytes=_as_int(entry.get("workdir_bytes")),
            outputs_pending=bool(entry.get("outputs_pending")),
            outputs_lost=bool(entry.get("outputs_lost")),
            isolation=entry.get("isolation"),
            low_util=LowUtilView.from_payload(entry.get("low_util")),
            outputs=[o for o in entry.get("outputs") or [] if isinstance(o, dict)],
            wandb=_str_dict(entry.get("wandb")),
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
    view.pkg_commit = _as_str(payload.get("pkg_commit"))
    view.owned, view.indices = owned_gpus(payload, entry)
    view.unavailable = [g for g in payload.get("gpus_unavailable") or [] if isinstance(g, str)]
    view.shared = [
        card
        for card in (
            SharedGpu.from_payload(row) for row in payload.get("shared_gpus_resolved") or []
        )
        if card is not None
    ]
    view.shared_unavailable = [
        g for g in payload.get("shared_gpus_unavailable") or [] if isinstance(g, str)
    ]
    # Into the one numbering table, because it is what names a card everywhere
    # it is mentioned -- including `gpu=4` on the line of a job that borrowed it.
    view.indices.update({c.uuid: c.index for c in view.shared if c.index is not None})
    view.shared_min_priority = _as_int(payload.get("shared_min_priority"))
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


def _fmt_util(job: JobView, *, source: bool = False) -> str:
    """`(host)` names the source: the host's sampler, not the provider's.

    Only worth saying where a provider's own reading is on screen -- on a host
    with no pod line there is no second percentage to confuse it with, and the
    tag is then a word on every running line that answers nothing.
    """
    tag = " (host)" if source else ""
    return f"util --{tag}" if job.last_util is None else f"util {job.last_util:.0f}%{tag}"


def _job_label(job: JobView) -> str:
    """`name (job-id)`: the name is what a reader is looking for, the id is
    what every other command takes as an argument."""
    return f"{job.name} ({job.job_id})" if job.name else job.job_id


def _fmt_gpus(view: HostView, job: JobView) -> str:
    """Which cards this job holds, by the index the gpu lines above use.

    The other direction from the gpu lines' old `busy <job-id>`: a job holding
    four cards was four lines repeating its id, and this is one field.
    """
    if not job.gpus:
        return "gpu=none"
    return "gpu=" + ",".join(view.gpu_label(uuid) for uuid in job.gpus)


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


def _fmt_cards(job: JobView) -> str:
    """` needs 3 gpus` on a queued job that wants more than one card.

    The usual job wants exactly one and saying so on every line is noise, but a
    job waiting for three is the answer to "there is a card free, why is it
    still queued".
    """
    # `<= 1` and not `== 1`: a `gpus: 0` job holds no card and never waits for
    # one, which is why it is ignored everywhere else here too.
    if job.gpus_requested is None or job.gpus_requested <= 1:
        return ""
    return f" needs {job.gpus_requested} gpus"


def _fmt_starts(job: JobView, starts: dict[str, float]) -> str:
    """` starts in ~2h10m`: when this job's turn comes, where that is known."""
    seconds = starts.get(job.job_id)
    return "" if seconds is None else f" starts {_fmt_wait(seconds)}"


def _fmt_estimate(job: JobView, *, total: bool = False) -> str:
    """` est 2h30m`: the whole run, not what is left of it. `total` says so out
    loud, for the lines that also carry elapsed or remaining times."""
    if job.estimated_runtime_min is None:
        return ""
    return f" est {format_duration(job.estimated_runtime_min * 60.0)}{' total' if total else ''}"


def queue_start_estimates(view: HostView) -> dict[str, float]:
    """Seconds until each queued job is expected to start, by job id.

    The host's own dispatch rule run forward over the estimates it has: a card
    comes free at the eta of the job holding it, the queue is walked in
    priority order, and a job that fits into what is free before the job ahead
    of it does starts first -- which is what the dispatcher does, since it
    walks the whole queue on every pass rather than blocking on the head of it.

    A job is in the answer or it is not: one whose turn depends on a job that
    gave no estimate is absent, never guessed at. That is why a *later* job can
    have a start time when an earlier one does not -- it fits in cards the
    unestimated job is not holding.

    Shared cards are in the model, but only the ones that are idle *now* and
    only for the jobs allowed onto them. A card somebody else is using is left
    out entirely rather than given a release time: when they will stop is the
    one thing this host cannot know. Leaving the idle ones out instead was the
    other option and is worse -- it told a job that would borrow on the next
    pass that it starts in six hours, which is the exact question this whole
    machinery exists to answer correctly.
    """
    # Nothing is dispatched on a paused or draining host, so every start time
    # here would be an answer to a question nobody asked: when it would have
    # started if the host were taking work.
    if view.paused or view.draining or not view.queue:
        return {}
    cards = _card_releases(view)
    starts: dict[str, float] = {}
    pending = list(view.queue)
    clock = 0.0
    blocked = False
    while pending and not blocked:
        for job in list(pending):
            if job.gpus_requested is None:
                # A host too old to say what a queued job asked for. It is
                # ahead in the queue and will take cards we cannot count, so
                # nothing behind it can be estimated either.
                blocked = True
                break
            borrows = view.may_borrow(job)
            # Owned first, exactly as the dispatcher assigns them, so a job
            # borrows only the shortfall and holds a shared card no longer
            # than it has to.
            free = [
                i
                for i, (release, shared) in enumerate(cards)
                if release is not None and release <= clock and (borrows or not shared)
            ]
            if len(free) < job.gpus_requested:
                continue
            done = (
                None
                if job.estimated_runtime_min is None
                else clock + job.estimated_runtime_min * 60.0
            )
            for index in free[: job.gpus_requested]:
                cards[index] = (done, cards[index][1])
            starts[job.job_id] = clock
            pending.remove(job)
        later = [release for release, _ in cards if release is not None and release > clock]
        if blocked or not later:
            break
        clock = min(later)
    return starts


def _card_releases(view: HostView) -> list[tuple[float | None, bool]]:
    """`(seconds until this card is free, is it a shared one)` per card.

    Now if nothing holds it, the holder's eta if one of our jobs does, and None
    when the card is not one anything can be scheduled onto -- either its
    holder offered no end time, or it is a shared card somebody else is on and
    nothing here can say when they will stop.

    Owned cards come first so that the walk above prefers them.
    """
    running = {job.job_id: job for job in view.running}

    def release(uuid: str) -> float | None:
        holder = running.get(view.gpu_holder(uuid) or "")
        if holder is None:
            return 0.0
        remaining = holder.eta_seconds
        return None if remaining is None else max(0.0, remaining)

    cards: list[tuple[float | None, bool]] = [(release(uuid), False) for uuid in view.owned]
    for card in view.shared:
        held = view.gpu_holder(card.uuid)
        if held or card.unused:
            cards.append((release(card.uuid), True))
    return cards


def queue_placement(view: HostView, job_id: str) -> dict[str, Any]:
    """Where one job sits in its host's queue, for `submit` and `reorder` to
    print: the answer to "so when does it run".

    Every field is null when the host could not be asked, which is not the same
    as "not queued": the job was enqueued before this was ever looked up.
    """
    if not view.reachable:
        return placement_unknown()
    queued = {job.job_id: job for job in view.queue}
    job = queued.get(job_id)
    seconds = queue_start_estimates(view).get(job_id) if job is not None else None
    return {
        "queue_position": list(queued).index(job_id) + 1 if job is not None else None,
        "queue_length": len(queued),
        # Already running: it left the queue between the enqueue and this call,
        # so `queue_position: null` here means dispatched, not unknown.
        "dispatched": any(running.job_id == job_id for running in view.running),
        "starts_in_s": None if seconds is None else round(seconds, 1),
        "starts_at": _at(seconds),
        "starts_unknown": None
        if job is None or seconds is not None
        else no_start_reason(view, job),
    }


def no_start_reason(view: HostView, job: JobView) -> str:
    """Why this queued job has no projected start time.

    There are several reasons and they are not interchangeable: a submit to a
    paused host is an ordinary mistake, and this is the moment the submitter is
    looking. Saying "a job ahead of it gave no estimate" about the only job in
    the queue of a host that is not dispatching at all would be a lie told at
    exactly the wrong time.
    """
    if view.paused:
        return f"host {view.entry.name} is paused, so nothing is being dispatched"
    if view.draining:
        return f"host {view.entry.name} is draining, so nothing more will be dispatched"
    borrows = view.may_borrow(job)
    # Cards the host cannot see this minute are counted in: the host makes a
    # job wait for one of those, it does not fail it (see the dispatcher's
    # `_capacity_failure`), and "it will never be dispatched" is an absolute
    # this may not say about a job the host is perfectly well set up to run.
    capacity = len(view.owned) + len(view.unavailable)
    if borrows:
        capacity += len(view.shared) + len(view.shared_unavailable)
    if job.gpus_requested is not None and job.gpus_requested > capacity:
        shared = " (shared included)" if borrows else ""
        return (
            f"it asks for {job.gpus_requested} card(s) and the host has "
            f"{capacity}{shared}, so it will never be dispatched"
        )
    if any(ahead.gpus_requested is None for ahead in view.queue):
        return "this host does not report how many cards a queued job asked for"
    if borrows and job.gpus_requested is not None and job.gpus_requested > len(view.owned):
        # Not an omission: a shared card comes free when its real owner stops
        # using it, and nothing here can know when that is. Saying so is the
        # honest answer, and the only alternative is a number we made up.
        return (
            f"it needs {job.gpus_requested - len(view.owned)} shared card(s), and when "
            f"somebody else stops using one is not something this host can predict"
        )
    # Running or queued: either way, the cards this job is waiting for are
    # spoken for by something that never said when it would be done with them.
    return "the jobs holding the cards it needs gave no end time"


def placement_unknown() -> dict[str, Any]:
    """Nobody could be asked. Not the same as "not queued": the enqueue already
    happened, and every field being null is what says we do not know."""
    return {
        "queue_position": None,
        "queue_length": None,
        "dispatched": None,
        "starts_in_s": None,
        "starts_at": None,
        "starts_unknown": None,
    }


def queue_note(placement: dict[str, Any]) -> str | None:
    """The one line `submit` and `reorder` print about the queue, or nothing
    when the host could not be asked (their own output already says so)."""
    if placement.get("dispatched"):
        return "  queue: dispatched already; it is running now"
    position = placement.get("queue_position")
    if position is None:
        return None
    when = placement.get("starts_in_s")
    starts = (
        f"start time unknown ({placement.get('starts_unknown')})"
        if when is None
        else f"starts {_fmt_wait(when)}"
    )
    return f"  queue: position {position} of {placement['queue_length']}; {starts}"


def _at(seconds: float | None) -> str | None:
    """A wait, as the wall-clock instant it lands on."""
    if seconds is None:
        return None
    return (datetime.now(UTC) + timedelta(seconds=seconds)).isoformat(timespec="seconds")


def _fmt_wait(seconds: float) -> str:
    """`now` for a job the next dispatcher pass will take, `in ~2h10m` beyond that."""
    return "now" if seconds < 60.0 else f"in ~{format_duration(seconds)}"


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
    """One line per owned card: free or busy first, then what the card is.

    No UUID and no holder: the question asked of this block is "is there a card
    for my job", and the running lines below name their own cards. `gpuc host
    list` is where UUIDs live, because that is where they are copied from.
    """
    lines: list[str] = []
    for index, name, vram, uuid in gpu_rows(view.owned, view.entry.gpu_info, view.indices):
        state = "busy" if view.gpu_holder(uuid) else "free"
        lines.append(f"  gpu     [{index}] {state} {name} {vram}".rstrip())
    for missing in view.unavailable:
        lines.append(
            f"  gpu     [{missing}] UNAVAILABLE  nvidia-smi does not report this card on the "
            f"host, so nothing is dispatched to it"
        )
    return lines + _shared_gpu_lines(view)


def _shared_gpu_lines(view: HostView) -> list[str]:
    """One line per card this host borrows, and who is on it.

    `free` and `busy` mean what they do above -- nobody is on it, one of our
    jobs is -- and the third state is the one these cards exist for: somebody
    else is on it, and the numbers say how much, because "my job is queued and
    there is a free-looking card" is the question this block gets asked.
    """
    if not view.shared and not view.shared_unavailable:
        return []
    lines: list[str] = []
    rows = gpu_rows([card.uuid for card in view.shared], view.entry.gpu_info, view.indices)
    for card, (index, name, vram, _uuid) in zip(view.shared, rows, strict=True):
        if view.gpu_holder(card.uuid):
            state, note = "busy", ""
        elif card.unused:
            state, note = "free", ""
        else:
            state, note = "IN USE", f" (somebody else: {card.describe()})"
        lines.append(f"  shared  [{index}] {state} {name} {vram}".rstrip() + note)
    for missing in view.shared_unavailable:
        lines.append(
            f"  shared  [{missing}] UNAVAILABLE  nvidia-smi does not report this card on the "
            f"host, so nothing is borrowed from it"
        )
    if view.shared and view.shared_min_priority is not None:
        lines.append(
            f"  shared  only jobs at priority {view.shared_min_priority} or better "
            f"(a lower number) may borrow these"
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
    # The cards are named one per line below, so the header carries only the
    # count a reader is deciding on: how many are free, right now.
    # A host that owns nothing and borrows something is a real configuration,
    # and `no GPUs` above a list of shared cards contradicts itself.
    if view.owned:
        cards = f"gpus {len(view.free)}/{len(view.owned)} free"
    elif view.shared:
        cards = f"shared {len(view.borrowable)}/{len(view.shared)} free, none owned"
    else:
        cards = "no GPUs"
    driver = f" (driver {entry.driver_version})" if entry.driver_version else ""
    header = f"host {entry.name} [{entry.kind}]  {cards}{driver}"
    if flags:
        header += "  " + " ".join(flags)
    lines = [header]
    lines += [f"  WARNING {warning}" for warning in host_warnings(view)]
    lines.append(f"  {dispatcher}")
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
                f"  SUSPECT {_job_label(job)} phase={job.phase} {_fmt_minutes(job)} "
                f"{_fmt_util(job, source=view.pod is not None)} {_fmt_gpus(view, job)}"
            )
        if view.past_ttl:
            lines.append(f"  SUSPECT pod for host {entry.name} is older than {entry.ttl_hours}h")
        if len(lines) == body_start:
            lines.append("  no suspects")
        return "\n".join(lines)

    for job in view.running:
        mark = "  running" if not job.suspect else "  running!"
        lines.append(
            f"{mark} {_job_label(job)} phase={job.phase or '-'} {_fmt_minutes(job)} "
            f"{_fmt_util(job, source=view.pod is not None)} {_fmt_gpus(view, job)}"
            f"{_fmt_eta(job)}"
        )
    starts = queue_start_estimates(view)
    for job in view.queue:
        lines.append(
            f"  queued  {_job_label(job)} prio={job.priority}{_fmt_cards(job)}"
            f"{_fmt_estimate(job)}{_fmt_starts(job, starts)}"
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
            f"  done    {_job_label(job)} {job.status}"
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
            f"  disk    {human_bytes(leftover)} still in {len(held)} finished job "
            f"workdir(s); free it with: gpuc clean --host {entry.name} --all-finished"
        )
    if len(lines) == body_start:
        lines.append("  idle; nothing queued, running or finished")
    return "\n".join(lines)


def host_warnings(view: HostView) -> list[str]:
    """What the host says about itself that this machine has not caught up with.

    Only the build: the host owns its config, so a config here that differs
    from the host's is a stale cache and not a disagreement -- everything that
    acts on a host reads `config.json` first, and this block already prints the
    host's own answer for the cards. What the host cannot fix by itself is the
    package: whichever machine bootstrapped it last is what it runs, and that
    may be a laptop on a newer build as easily as this machine on an older one.

    A host that was not reached says nothing: "we could not ask" is not
    evidence of a mismatch, and the unreachable block already says so.
    """
    if not view.reachable:
        return []
    stale = version.host_build_warning(view.entry.name, view.pkg_commit, version.local_commit())
    return [stale] if stale else []


def job_json(
    job: JobView, mirror_prefix: str | None = None, *, starts_in_s: float | None = None
) -> dict[str, Any]:
    """The text view's fields, named the same, with nothing rendered.

    `running` is the list automation should key on. It is the host's own
    answer, so an empty list here means the host said "nothing is running" --
    never "we could not ask", which is `reachable: false` and an `errors` entry.

    `queued` is in dispatch order and every job carries the `priority` that
    put it there, so sorting the list by `priority` reproduces the order the
    host will actually take them in. `starts_in_s` and `starts_at` are when a
    queued job's turn is expected to come, and are null for anything that is
    not queued -- or whose turn depends on a job that gave no estimate.

    `links` is the one thing here the text view has no room for: where the
    job's outputs, its W&B run and its mirrored log can be opened, for a
    dashboard to render as anchors. `mirror_prefix` is the host's `s3_prefix`.
    """
    return {
        "job_id": job.job_id,
        "name": job.name,
        "status": job.status,
        "reason": job.reason,
        "phase": job.phase,
        "priority": job.priority,
        "attempt": job.attempt,
        "started_at": job.started_at,
        "elapsed_s": None if job.minutes is None else round(job.minutes * 60.0, 1),
        "util": job.last_util,
        "progress_pct": job.progress_pct,
        "eta": job.eta,
        "eta_s": None if job.eta_seconds is None else round(job.eta_seconds, 1),
        "estimated_runtime_min": job.estimated_runtime_min,
        "progress_error": job.progress_error,
        "gpus": list(job.gpus),
        "gpus_requested": job.gpus_requested,
        "starts_in_s": None if starts_in_s is None else round(starts_in_s, 1),
        "starts_at": _at(starts_in_s),
        "iso": job.isolation,
        "ended_at": job.ended_at,
        "outputs_pending": job.outputs_pending,
        "outputs_lost": job.outputs_lost,
        "suspect": job.suspect,
        "workdir_bytes": job.workdir_bytes,
        "outputs": [dict(o) for o in job.outputs],
        "links": job_links(job, mirror_prefix),
    }


S3_CONSOLE = "https://s3.console.aws.amazon.com/s3/buckets/{bucket}?prefix={prefix}"
HF_TREE = "https://huggingface.co/{repo}/tree/main/{path}"
WANDB_RUN = "https://wandb.ai/{entity}/{project}/runs/{run_id}"
WANDB_PROJECT = "https://wandb.ai/{entity}/{project}"


def s3_console_url(uri: str) -> str | None:
    """The console page listing an `s3://bucket/key` prefix, or None for a non-S3 uri."""
    if not uri.startswith("s3://"):
        return None
    bucket, _, key = uri[len("s3://") :].partition("/")
    if not bucket:
        return None
    key = key.strip("/")
    return S3_CONSOLE.format(
        bucket=quote(bucket, safe=""), prefix=quote(key + "/" if key else "", safe="/")
    )


def job_links(job: JobView, mirror_prefix: str | None = None) -> list[dict[str, str | None]]:
    """Where to open what this job wrote: one entry per destination it declared.

    Every link is derived from what the job *said* it would do; none of them
    is checked. An S3 output that the final sync never uploaded still gets a
    link, and `outputs_pending` beside it is what says the link is empty.
    """
    links: list[dict[str, str | None]] = []
    for output in job.outputs:
        path = output.get("path") if isinstance(output.get("path"), str) else None
        s3 = output.get("s3")
        if isinstance(s3, str) and s3:
            links.append({"kind": "s3", "path": path, "target": s3, "url": s3_console_url(s3)})
        repo = output.get("hf")
        if isinstance(repo, str) and repo:
            sub = output.get("hf_path") if isinstance(output.get("hf_path"), str) else ""
            target = f"{repo}/{sub}" if sub else repo
            safe_repo = quote(repo, safe="/")
            url = (
                HF_TREE.format(repo=safe_repo, path=quote(sub))
                if sub
                else f"https://huggingface.co/{safe_repo}"
            )
            links.append({"kind": "hf", "path": path, "target": target, "url": url})
    entity, project, run_id = (job.wandb.get(k) for k in ("entity", "project", "run_id"))
    if entity and project:
        template = WANDB_RUN if run_id else WANDB_PROJECT
        url = template.format(
            entity=quote(entity, safe=""),
            project=quote(project, safe=""),
            run_id=quote(run_id or "", safe=""),
        )
        target = f"{entity}/{project}" + (f"/{run_id}" if run_id else "")
        links.append({"kind": "wandb", "path": None, "target": target, "url": url})
    if mirror_prefix and job.status != "queued":
        # The host's *current* prefix. A job mirrored under a prefix the host
        # has since been re-registered without gets a link to an empty
        # listing; `gpuc logs` reads the job's own index entry, this does not.
        mirror = f"{mirror_prefix.rstrip('/')}/jobs/{job.job_id}"
        links.append(
            {"kind": "mirror", "path": None, "target": mirror, "url": s3_console_url(mirror)}
        )
    return links


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


def shared_gpu_json(view: HostView) -> list[dict[str, Any]]:
    """The cards this host borrows. `unused` is the host's own verdict -- no
    memory held, no work running -- and is what decides whether gpuc takes one;
    `busy_job` means one of ours already has it."""
    out: list[dict[str, Any]] = []
    for card in view.shared:
        info = view.entry.gpu_info.get(card.uuid) or GpuInfo()
        out.append(
            {
                "index": card.index if card.index is not None else info.index,
                "uuid": card.uuid,
                "name": info.name,
                "vram_mib": info.vram_mib,
                "busy_job": view.gpu_holder(card.uuid),
                "memory_mib": card.memory_mib,
                "utilization_pct": card.utilization_pct,
                "unused": card.unused,
            }
        )
    out += [{"shared_as": item, "available": False} for item in view.shared_unavailable]
    return out


def host_json(
    view: HostView, *, recent: int = RECENT_FINISHED, since_s: float | None = None
) -> dict[str, Any]:
    """`provider_util` is the provider's per-GPU reading for an ephemeral host's
    pod (null for every other host); each job's `util` is the host's own
    sampler. They are two different measurements and are named as such."""
    entry = view.entry
    errors = [view.error] if view.error else []
    errors += host_warnings(view)
    finished = [job for job in view.finished if within(job, since_s)][:recent]
    starts = queue_start_estimates(view)
    return {
        "name": entry.name,
        "kind": entry.kind,
        "target": entry.ssh,
        "reachable": view.reachable,
        "pod_gone": view.pod_gone,
        "draining": view.draining,
        "paused": view.paused,
        # The host's own answer, so null means the host did not say, never
        # "current".
        "pkg_commit": view.pkg_commit,
        "dispatcher": {
            "alive": view.dispatcher_alive,
            "heartbeat_age_s": view.heartbeat_age_s,
        },
        "provider_util": list(view.pod.gpu_utils) if view.pod is not None else None,
        "pod": pod_json(view),
        "gpus": gpu_json(view),
        "shared_gpus": shared_gpu_json(view),
        "shared_min_priority": view.shared_min_priority,
        "queued": [
            job_json(job, entry.s3_prefix, starts_in_s=starts.get(job.job_id)) for job in view.queue
        ],
        "running": [job_json(job, entry.s3_prefix) for job in view.running],
        "finished": [job_json(job, entry.s3_prefix) for job in finished],
        "errors": errors,
    }


def pod_json(view: HostView) -> dict[str, Any] | None:
    """The provider's view of an ephemeral host's pod: the `pod` line, as data."""
    pod = view.pod
    if pod is None:
        return None
    return {
        "id": pod.id,
        "status": pod.status,
        "gpu_name": pod.gpu_name,
        "gpu_count": pod.gpu_count,
        "cost_usd_hr": pod.cost_usd_hr,
        "cuda_version": pod.cuda_version,
        "age_s": None if pod.age is None else round(pod.age.total_seconds(), 1),
        "past_ttl": view.past_ttl,
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
