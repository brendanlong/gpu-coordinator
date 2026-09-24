"""`gpuc status`: one compact block per host.

Asks the host (the host is authoritative) and renders what it said; the jobs
only the index knows are `actions.unhosted_jobs`. Never kills anything.
"""

from __future__ import annotations

import shlex
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any
from urllib.parse import quote

from gpuc.control import version
from gpuc.control.config import HostEntry, Settings, parse_timestamp
from gpuc.control.gpuinfo import GpuInfo
from gpuc.control.gpuinfo import rows as gpu_rows
from gpuc.control.providers.base import Pod, Provider
from gpuc.control.remote import Asked, Gone, HostSession, Unaskable, ask
from gpuc.control.s3index import IndexEntry, job_uri
from gpuc.host.cleanup import human_bytes
from gpuc.host.jobs import SCHEMA_VERSION

HEARTBEAT_STALE_S = 30.0
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
    use_shared: bool | None = None
    """The spec said this job may borrow the host's shared cards, so the ones
    it is waiting for are not only the ones the host owns. None from a host
    that did not say -- a build older than shared cards, or a spec it could
    not read."""
    reason: str | None = None
    problems: list[str] = field(default_factory=list)
    """What else went wrong on the way out, beside `reason`: `sync`, `no-outputs`."""
    upload_errors: list[str] = field(default_factory=list)
    """The last failure standing at each of the job's upload destinations. A
    running job with one here has an output path that is not being uploaded,
    which is the thing to notice before it runs for four hours."""
    exit_code: int | None = None
    attempt: int = 1
    """How many times the host has launched this id: a preempt adds one."""
    requeued_from: str | None = None
    """The job this one was requeued from, when the host reports it."""
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
    starts_in_s: float | None = None
    """When the host expects this queued job's turn to come, by replaying its
    own dispatch rule over the running jobs' etas. Null when it cannot say."""
    starts_unknown: str | None = None
    """Why the host could not say, for a queued job with no `starts_in_s`."""
    auto_preempt: bool | None = None
    """This job asked to be stopped and queued again whenever that lets a more
    important one start, so a `running` line for it is not a promise that it
    will still be running in a minute. None from a host that did not say."""
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
    escapes). In `--json` only: what a kill reaps is asked while debugging one
    job, not while scanning a host."""
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
        end = parse_timestamp(self.ended_at) if self.ended_at else datetime.now(UTC)
        start = parse_timestamp(self.started_at)
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
        when = parse_timestamp(self.eta)
        return None if when is None else (when - datetime.now(UTC)).total_seconds()


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
    when = parse_timestamp(stamp)
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


def _as_bool(value: Any) -> bool | None:
    """A flag the host sent, or None for a host that did not send one.

    Not `bool(value)`: the host sends null for a spec it could not read, and
    reporting that as `false` would answer a question we cannot answer -- "no,
    this job did not ask for that" -- rather than saying we do not know.
    """
    return value if isinstance(value, bool) else None


def _str_dict(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    return {k: v for k, v in value.items() if isinstance(k, str) and isinstance(v, str)}


class HostState(Enum):
    """The three answers of `remote.ask`, as the view carries them; everything
    that prints, exits or forgets reads this rather than re-deriving it."""

    ANSWERED = "answered"
    UNASKABLE = "unaskable"
    """Could not be asked and may still hold its jobs: a failure, with the
    reason in `error`."""
    GONE = "gone"
    """Does not exist any more. Not a failure: its finished jobs are the
    mirror's, and `actions.forget_gone_rentals` forgets the entry."""


@dataclass
class HostView:
    entry: HostEntry
    state: HostState = HostState.UNASKABLE
    error: str | None = None
    registered: bool = True
    """False for a name this machine has no entry for, whose `entry` is only
    the name: there is no address to print and none to forget."""
    lost: list[str] = field(default_factory=list)
    """A gone host's jobs the mirror has no final state for."""
    lost_reason: str | None = None
    """Why those jobs went with the host."""
    heartbeat_age_s: float | None = None
    draining: bool = False
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
    pod: Pod | None = None
    session: HostSession | None = None
    """The session the answer came over, for a caller with a follow-up
    question (`submit` and `reorder` ask where the job landed)."""
    mirror_prefix: str | None = None
    """The host's own `s3_prefix`, from the config the session read."""
    pkg_commit: str | None = None
    """The commit the *host* says its package came from, not the one this
    machine's registry remembers shipping. Null when the host was not asked or
    did not say."""
    dispatcher_pkg_commit: str | None = None
    """The commit recorded by whoever last took the host's dispatcher lock.

    A dispatcher imports its code once, so re-shipping the package under a live
    one changes nothing about what it dispatches with. One started from the new
    package replaces it, and this is how a host where that did not happen says
    so. Read from a lock file that outlives its writer, so it is only a fact
    about what is *running* next to a live heartbeat -- which is what
    `host_warnings` checks before saying anything. Null when the host did not
    say."""
    queue: list[JobView] = field(default_factory=list)
    running: list[JobView] = field(default_factory=list)
    finished: list[JobView] = field(default_factory=list)
    finished_count: int | None = None
    """How many finished jobs the host has, of which `finished` is the ones
    it sent; None where nobody counted."""

    @property
    def finished_total(self) -> int:
        return len(self.finished) if self.finished_count is None else self.finished_count

    @property
    def reachable(self) -> bool:
        return self.state is HostState.ANSWERED

    @property
    def gone(self) -> bool:
        return self.state is HostState.GONE

    @property
    def failure(self) -> str | None:
        """Why this host could not be read, if that is what happened. A host
        that is gone is not that."""
        return None if self.gone else self.error

    @property
    def dispatcher_alive(self) -> bool:
        return self.heartbeat_age_s is not None and self.heartbeat_age_s < HEARTBEAT_STALE_S

    @property
    def free(self) -> list[str]:
        busy = {uuid for job in self.running for uuid in job.gpus}
        return [uuid for uuid in self.owned if uuid not in busy]

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
        never resolves is a host that does not answer the question, which is
        what the build warning in `host_warnings` is for.
        """
        return sum(job.workdir_bytes or 0 for job in self.finished)


def job_views(payload: dict[str, Any]) -> tuple[list[JobView], list[JobView], list[JobView]]:
    """Every field is treated as untrusted: this is another build's JSON.

    A host on a different commit -- or a half-written state file -- must cost
    one missing row, never a traceback out of `gpuc status` for every host.
    """
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
            priority=_as_int(entry.get("priority")),
            gpus=list(entry.get("gpus") or []),
            gpus_requested=_as_int(entry.get("gpus_requested")),
            use_shared=_as_bool(entry.get("use_shared")),
            reason=entry.get("reason"),
            problems=[p for p in entry.get("problems") or [] if isinstance(p, str)],
            upload_errors=[
                u["error"]
                for u in entry.get("uploads") or []
                if isinstance(u, dict) and isinstance(u.get("error"), str)
            ],
            exit_code=entry.get("exit_code"),
            attempt=_as_int(entry.get("attempt")) or 1,
            requeued_from=_as_str(entry.get("requeued_from")),
            started_at=entry.get("started_at"),
            ended_at=entry.get("ended_at"),
            # A sample is null when nvidia-smi failed; drop it rather than
            # showing a missing reading as 0%.
            util_recent=[
                float(u) for u in entry.get("util_recent") or [] if isinstance(u, (int, float))
            ],
            progress_pct=_as_float(entry.get("progress_pct")),
            eta=_as_str(entry.get("eta")),
            estimated_runtime_min=_as_float(entry.get("estimated_runtime_min")),
            starts_in_s=_as_float(entry.get("starts_in_s")),
            starts_unknown=_as_str(entry.get("starts_unknown")),
            auto_preempt=_as_bool(entry.get("auto_preempt")),
            progress_error=_as_str(entry.get("progress_error")),
            workdir_bytes=_as_int(entry.get("workdir_bytes")),
            outputs_pending=bool(entry.get("outputs_pending")),
            outputs_lost=bool(entry.get("outputs_lost")),
            isolation=entry.get("isolation"),
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


def status_request(
    job_ids: Sequence[str] = (), *, recent: int | None = None, since_s: float | None = None
) -> str:
    """The host's `status`: for exactly these jobs, or naming none, for the
    finished jobs inside the window and everything that is not finished.
    Neither ids nor a window asks for every job the host has."""
    words = ["status", *(shlex.quote(job_id) for job_id in job_ids)]
    if recent is not None:
        words += ["--recent", str(recent)]
    if since_s is not None:
        words += ["--since", repr(float(since_s))]
    return " ".join(words)


def gather(
    entry: HostEntry,
    settings: Settings | None = None,
    *,
    request: str = "status",
    session: HostSession | None = None,
    provider: Provider | None = None,
) -> HostView:
    """One host's status: `ask` it, and read the answer.

    A bare `status` unless the caller needs the window or particular jobs:
    it is the one request every build of the host understands, so what only
    needs the host's cards, dispatcher and queue -- reuse, teardown, the
    queue placement after a submit -- keeps working against a host that has
    not been bootstrapped since."""
    return parse_status(entry, ask(entry, request, settings, provider=provider, session=session))


def parse_status(entry: HostEntry, asked: Asked) -> HostView:
    """A host's `status` answer as a view, every field read as untrusted:
    this is another build's JSON. A host nobody could ask produces no claim
    about its cards, its build or its jobs -- only the reason."""
    view = HostView(entry=entry, pod=asked.pod)
    if isinstance(asked, Gone):
        view.state, view.error = HostState.GONE, asked.reason
        return view
    if isinstance(asked, Unaskable):
        view.error = asked.reason
        return view
    view.state = HostState.ANSWERED
    view.session = asked.session
    view.mirror_prefix = asked.session.config.s3_prefix
    # The provider could not be asked about the pod: the host answered, and
    # this is still the command's exit code.
    view.error = asked.pod_error
    payload = asked.payload or {}
    view.pkg_commit = _as_str(payload.get("pkg_commit"))
    view.dispatcher_pkg_commit = _as_str(payload.get("dispatcher_pkg_commit"))
    view.owned, view.indices = owned_gpus(payload)
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
    # Validated rather than trusted: a host on another build could answer with
    # a string here, and formatting it would take out the whole `gpuc status`,
    # not just this host's line.
    view.heartbeat_age_s = _as_float(payload.get("dispatcher_heartbeat_age_s"))
    view.draining = bool(payload.get("draining"))
    view.queue, view.running, view.finished = job_views(payload)
    view.finished_count = _as_int(payload.get("finished_count"))
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
    age = "age ?" if pod.age is None else f"age {format_duration(pod.age.total_seconds())}"
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


def job_label(job: JobView) -> str:
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


def _fmt_elapsed(job: JobView) -> str:
    return "--" if job.minutes is None else format_duration(job.minutes * 60.0)


def _fmt_eta(job: JobView) -> str:
    """` eta 2h10m (42%)` when the job measures its own progress, ` eta 2h10m
    (est)` when all it has is the submitter's guess, and nothing at all when it
    has neither. The tag matters: one of those numbers is evidence.

    A running job whose host published no `eta` falls back to the estimate the
    same host reports, rendered as a total rather than a remaining time. The
    runner re-reads the estimate every `ESTIMATE_REFRESH_S`, which bounds the
    window after `gpuc estimate` but does not close it, and inside it `--json`
    carries an estimate the text would otherwise not show -- a scripted caller
    seeing what the operator cannot, which `tests/test_control_e2e.py` pins."""
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
    if job.gpus_requested is None or job.gpus_requested == 1:
        return ""
    return f" needs {job.gpus_requested} gpus"


def _fmt_starts(job: JobView) -> str:
    """` starts in ~2h10m`: when this job's turn comes, where the host said."""
    seconds = job.starts_in_s
    return "" if seconds is None else f" starts {_fmt_wait(seconds)}"


def _fmt_upload_error(job: JobView) -> str:
    """` UPLOAD FAILING: <why>`: an output that is not reaching its destination."""
    if not job.upload_errors:
        return ""
    first = job.upload_errors[0].splitlines()[0]
    return f" UPLOAD FAILING: {first[:80]}"


def _fmt_auto_preempt(job: JobView) -> str:
    """` auto-preempt`: this job gives its cards up to anything more important."""
    return " auto-preempt" if job.auto_preempt else ""


def _fmt_estimate(job: JobView, *, total: bool = False) -> str:
    """` est 2h30m`: the whole run, not what is left of it. `total` says so out
    loud, for the lines that also carry elapsed or remaining times."""
    if job.estimated_runtime_min is None:
        return ""
    return f" est {format_duration(job.estimated_runtime_min * 60.0)}{' total' if total else ''}"


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
    seconds = job.starts_in_s if job is not None else None
    return {
        "queue_position": list(queued).index(job_id) + 1 if job is not None else None,
        "queue_length": len(queued),
        # Already running: it left the queue between the enqueue and this call,
        # so `queue_position: null` here means dispatched, not unknown.
        "dispatched": any(running.job_id == job_id for running in view.running),
        "starts_in_s": None if seconds is None else round(seconds, 1),
        "starts_at": _at(seconds),
        "starts_unknown": None if job is None or seconds is not None else job.starts_unknown,
    }


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
    if when is not None:
        starts = f"starts {_fmt_wait(when)}"
    elif placement.get("starts_unknown"):
        starts = f"start time unknown ({placement['starts_unknown']})"
    else:
        starts = "start time unknown"
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
    if not view.owned or view.free or not view.running:
        return None
    known: list[tuple[float, JobView]] = []
    for job in view.running:
        remaining = job.eta_seconds
        if remaining is not None:
            known.append((remaining, job))
    if not known:
        return None
    silent = len(view.running) - len(known)
    remaining, job = min(known, key=lambda pair: pair[0])
    when = "overdue" if remaining < 0 else f"in ~{format_duration(remaining)}"
    # "no end time", not "no estimate": a job whose host has an estimate it has
    # not turned into an eta yet is counted here, and its own line above says
    # `est ...`. Two lines of one host block may not contradict each other.
    note = f"; {silent} other running job(s) gave no end time" if silent else ""
    return f"  free    next card {when} ({job.job_id}){note}"


def owned_gpus(payload: dict[str, Any]) -> tuple[list[str], dict[str, int]]:
    """The host's owned cards as it resolved them this pass, and their indices."""
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
    return owned, indices


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
            f"host; nothing is dispatched to it, and a job waiting for it holds the queue"
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
    return lines


def within(job: JobView, since_s: float | None, now: datetime | None = None) -> bool:
    if since_s is None:
        return True
    ended = parse_timestamp(job.ended_at)
    if ended is None:
        return False
    return ((now or datetime.now(UTC)) - ended).total_seconds() <= since_s


def _finished_lines(view: HostView, *, recent: int, since_s: float | None) -> list[str]:
    lines: list[str] = []
    finished = [job for job in view.finished if within(job, since_s)]
    for job in finished[:recent]:
        # `cancelled (cancelled)` says nothing twice: only a reason that adds
        # to the status is worth the parenthesis.
        reason = job.reason if job.reason != job.status else None
        detail = reason or (f"exit {job.exit_code}" if job.exit_code else "")
        if job.problems:
            detail = ", ".join(filter(None, [detail, *job.problems]))
        flag = ""
        if job.outputs_lost and job.outputs_pending:
            # `outputs_lost` is written once and never cleared, so it outlives
            # the thing it describes: a job the host now reports as holding
            # nothing is not a lost result, whatever a past drain concluded.
            flag = "  OUTPUTS LOST"
        elif job.outputs_pending:
            flag = "  outputs not uploaded"
        lines.append(
            f"  done    {job_label(job)} {job.status}"
            f"{f' ({detail})' if detail else ''} {format_age(job.ended_at)}{flag}"
        )
    if since_s is not None and not finished and view.finished_total:
        lines.append(
            f"  done    none in the last {int(since_s // 60)} min ({view.finished_total} older)"
        )
    return lines


def render(
    view: HostView,
    *,
    recent: int = RECENT_FINISHED,
    since_s: float | None = None,
) -> str:
    entry = view.entry
    target = entry.ssh or "this machine"
    if not view.reachable:
        # A gone host's reason is not a failure, so it is not an ERROR line.
        state, prefix = ("GONE", "") if view.gone else ("UNASKABLE", "ERROR ")
        where = f" [{entry.kind}] {target}" if view.registered else ""
        lines = [f"host {entry.name}{where}: {state}", f"  {prefix}{view.error}"]
        pod = pod_line(view.pod)
        if pod:
            lines.append(pod)
        elif not view.gone and view.registered:
            # No provider's word to read next, so the probe is the next step;
            # a pod line is what to read when there is one, and a stopped
            # pod's reason already names the commands that end or forget it.
            lines.append(f"  try: gpuc host probe {entry.name}")
        if view.gone:
            finished = _finished_lines(view, recent=recent, since_s=since_s)
            if finished:
                lines.append("  from the S3 mirror:")
            lines += finished
            if view.lost_reason:
                shown = ", ".join(view.lost[:3]) + (" ..." if len(view.lost) > 3 else "")
                ids = f"{len(view.lost)} job(s) ({shown}): " if view.lost else ""
                lines.append(f"  lost    {ids}{view.lost_reason}")
        return "\n".join(lines)
    flags = []
    if view.draining:
        flags.append("DRAINING")
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
    if view.error:
        # A host that answered can still carry one -- the provider could not be
        # asked about its pod -- and it is this command's exit code.
        lines.append(f"  ERROR {view.error}")
    lines += [f"  WARNING {warning}" for warning in host_warnings(view)]
    lines.append(f"  {dispatcher}")
    lines += _gpu_lines(view)
    pod = pod_line(view.pod)
    if pod:
        lines.append(pod)
    # Where the per-job body starts. The header, the gpu lines and the pod line
    # are about the host, not about what is on it, so "nothing here" has to be
    # measured from here -- a host with GPUs never printed `idle` while this
    # was compared against the whole list.
    body_start = len(lines)

    for job in view.running:
        lines.append(
            f"  running {job_label(job)} phase={job.phase or '-'} {_fmt_elapsed(job)} "
            f"{_fmt_util(job, source=view.pod is not None)} {_fmt_gpus(view, job)}"
            f"{_fmt_eta(job)}{_fmt_auto_preempt(job)}{_fmt_upload_error(job)}"
        )
    for job in view.queue:
        lines.append(
            f"  queued  {job_label(job)} prio={job.priority}{_fmt_cards(job)}"
            f"{_fmt_estimate(job)}{_fmt_starts(job)}{_fmt_auto_preempt(job)}"
        )
    free = next_free_line(view)
    if free:
        lines.append(free)
    lines += _finished_lines(view, recent=recent, since_s=since_s)
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
    if stale:
        # One problem, one fix. A host whose *package* is behind is already
        # being told to re-bootstrap, and which of the two builds its
        # dispatcher happens to be on does not change that.
        return [stale]
    if not view.dispatcher_alive:
        return []
    running = version.dispatcher_build_warning(
        view.entry.name, view.dispatcher_pkg_commit, view.pkg_commit
    )
    return [running] if running else []


def job_json(job: JobView, mirror_prefix: str | None = None) -> dict[str, Any]:
    """The text view's fields, named the same, with nothing rendered.

    `running` is the list automation should key on. It is the host's own
    answer, so an empty list here means the host said "nothing is running" --
    never "we could not ask", which is `reachable: false` and an `errors` entry.

    `queued` is in dispatch order and every job carries the `priority` that
    put it there, so sorting the list by `priority` reproduces the order the
    host will actually take them in. `starts_in_s` and `starts_at` are when a
    queued job's turn is expected to come, and are null for anything that is
    not queued -- or whose turn depends on a job that gave no estimate.
    `shared_gpus` says which cards the host may borrow and `use_shared` says
    which jobs may have them, which is the rest of what dates a queued job's
    turn. Like `auto_preempt` it is null rather than false when the host did
    not say: "this job did not ask for that" is a different answer from "we
    could not find out".

    `links` is the one thing here the text view has no room for: where the
    job's outputs, its W&B run and its mirrored log can be opened, for a
    dashboard to render as anchors. `mirror_prefix` is the host's `s3_prefix`.
    """
    return {
        "job_id": job.job_id,
        "name": job.name,
        "status": job.status,
        "reason": job.reason,
        "problems": list(job.problems),
        "upload_errors": list(job.upload_errors),
        "exit_code": job.exit_code,
        "phase": job.phase,
        "priority": job.priority,
        "attempt": job.attempt,
        "requeued_from": job.requeued_from,
        "started_at": job.started_at,
        "elapsed_s": None if job.minutes is None else round(job.minutes * 60.0, 1),
        "util": job.last_util,
        "progress_pct": job.progress_pct,
        "eta": job.eta,
        "eta_s": None if job.eta_seconds is None else round(job.eta_seconds, 1),
        "estimated_runtime_min": job.estimated_runtime_min,
        "auto_preempt": job.auto_preempt,
        "progress_error": job.progress_error,
        "gpus": list(job.gpus),
        "gpus_requested": job.gpus_requested,
        "use_shared": job.use_shared,
        "starts_in_s": None if job.starts_in_s is None else round(job.starts_in_s, 1),
        "starts_at": _at(job.starts_in_s),
        "starts_unknown": job.starts_unknown,
        "iso": job.isolation,
        "ended_at": job.ended_at,
        "outputs_pending": job.outputs_pending,
        "outputs_lost": job.outputs_lost,
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
        mirror = job_uri(mirror_prefix, job.job_id)
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
    finished = [job for job in view.finished if within(job, since_s)][:recent]
    return {
        "name": entry.name,
        "kind": entry.kind if view.registered else None,
        "target": entry.ssh,
        "state": view.state.value,
        "reachable": view.reachable,
        "draining": view.draining,
        # The host's own answer, so null means the host did not say, never
        # "current".
        "pkg_commit": view.pkg_commit,
        "dispatcher": {
            "alive": view.dispatcher_alive,
            "heartbeat_age_s": view.heartbeat_age_s,
            "pkg_commit": view.dispatcher_pkg_commit,
        },
        "provider_util": list(view.pod.gpu_utils) if view.pod is not None else None,
        "pod": pod_json(view),
        "gpus": gpu_json(view),
        "shared_gpus": shared_gpu_json(view),
        "queued": [job_json(job, view.mirror_prefix) for job in view.queue],
        "running": [job_json(job, view.mirror_prefix) for job in view.running],
        "finished": [job_json(job, view.mirror_prefix) for job in finished],
        "errors": [view.error] if view.error else [],
        "warnings": host_warnings(view),
        "lost": {"jobs": view.lost, "reason": view.lost_reason} if view.lost_reason else None,
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
    }


UNHOSTED_LABELS: dict[HostState, str] = {
    HostState.ANSWERED: "its host answered and does not have it (lost its state, or purged)",
    HostState.UNASKABLE: "its host could not be asked, so it may still be running there",
    HostState.GONE: "its host is gone",
}
"""What each `host_state` means for a job only the index knows."""


def requeue_offered(state: HostState) -> bool:
    """Whether `gpuc requeue` is the way back for a job only the index knows:
    its host cannot still be running it. A host that could not be asked may
    hold the job -- running, or finished with its outputs -- and a requeue
    offered on the strength of a connection error is how a run ends up done
    twice."""
    return state in (HostState.ANSWERED, HostState.GONE)


def unhosted_json(
    entry: IndexEntry, *, lost: bool, state: HostState, final: str | None = None
) -> dict[str, Any]:
    """One job only the index knows, as `status --all --json` lists it.
    `final` is the mirror's final status for a job whose host is gone."""
    return {
        "job_id": entry.job_id,
        "name": entry.name,
        "host": entry.host,
        "host_state": state.value,
        "status": final,
        "requeue": requeue_offered(state),
        "requeued_from": entry.requeued_from,
        "submitted_at": entry.submitted_at,
        "s3_prefix": entry.s3_prefix,
        "outputs_lost": lost,
    }


def unhosted_line(
    entry: IndexEntry, *, lost: bool, state: HostState, final: str | None = None
) -> str:
    flag = " OUTPUTS LOST (the host went away before they uploaded)" if lost else ""
    label = f"{entry.name} ({entry.job_id})" if entry.name else entry.job_id
    origin = f" requeued from {entry.requeued_from}" if entry.requeued_from else ""
    ended = f", and ended {final} per the S3 mirror" if final else ""
    return (
        f"  {label} host={entry.host}{origin} submitted {format_age(entry.submitted_at)}: "
        f"{UNHOSTED_LABELS[state]}{ended}{flag}"
    )


def document(
    views: Sequence[HostView],
    *,
    errors: Sequence[str] = (),
    unhosted: Sequence[dict[str, Any]] = (),
    recent: int = RECENT_FINISHED,
    since_s: float | None = None,
) -> dict[str, Any]:
    """The whole of `gpuc status --json`: one object, always this shape.

    Top-level `errors` are the ones that belong to no host -- an unreadable
    registry (exit 3), a skipped entry or an index that could not be read
    (exit 1) -- and a consumer that sees any of them must not read `hosts` as
    the whole truth. `unhosted` is `--all`'s list of jobs only the index
    knows, empty without the flag.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "hosts": [host_json(view, recent=recent, since_s=since_s) for view in views],
        "unhosted": list(unhosted),
        "errors": list(errors),
    }
