"""Job identity, spec/state serialisation, host config, atomic file IO."""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import secrets
import time
from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from gpuc.host import paths, progress

SCHEMA_VERSION = 1
"""The shape of the JSON files two builds of gpuc share (`~/.gpuc/config.json`
and the control side's `hosts.json`). Written always, accepted missing: a file
from before the field existed is version 1 by definition. It exists so a future
incompatible change has something to branch on -- the readers here are
deliberately tolerant enough that it has not had to."""

FINISHED_STATUSES = ("succeeded", "failed", "cancelled")

CANCEL = "cancel"
PREEMPT = "preempt"
INTENTS = (CANCEL, PREEMPT)
"""What somebody has asked of a job, as distinct from what it is doing.

`status` is the job's own progress; `intent` is the request standing against
it. A queued job is cancelled outright, so the only intents are ones a running
job's runner acts on: `cancel` ends it as `cancelled`, `preempt` ends the
attempt and the runner queues the job again itself. Both live in the same file
as `status`, written in the same atomic replace, so there is no order of
writes to get right and nothing to reconcile after a crash. The write that
ends a job clears the intent with it: an intent only ever stands against a
running job.
"""

DEFAULT_PYTHON = "uv run --no-sync python"

ON_SUCCESS = "on_success"
ALWAYS = "always"
NEVER = "never"
CLEANUP_POLICIES = (ON_SUCCESS, ALWAYS, NEVER)
DEFAULT_CLEANUP = ON_SUCCESS


def _polling_interval(fields: Any) -> float:
    """`progress_interval_s`, with anything unusable meaning the default.

    A zero, a negative or a NaN reaches here from a hand-edited `spec.json`, a
    staged `incoming/<id>/spec.json`, or another build -- the control side's
    validation is not in that path. Zero and negative would poll on every pass
    of the runner's loop, forking a shell twice a second for the life of the
    job; NaN would silently never poll at all.
    """
    interval = as_float(fields, "progress_interval_s", progress.DEFAULT_INTERVAL_S)
    return interval if interval > 0.0 else progress.DEFAULT_INTERVAL_S


def normalize_cleanup(value: object, *, origin: str = "cleanup") -> str:
    """Validate a `cleanup:` value.

    A typo has to fail loudly here: the two ways of being wrong are keeping
    6.5 GB per job forever and deleting a failed run's workdir before anyone
    could look at it, and neither should be reachable by misspelling a word.
    """
    if value is None:
        return DEFAULT_CLEANUP
    text = str(value)
    if text not in CLEANUP_POLICIES:
        raise ValueError(f"{origin} must be one of {', '.join(CLEANUP_POLICIES)}, got {text!r}")
    return text


def new_job_id() -> str:
    return f"{datetime.now(UTC):%Y%m%d-%H%M%S}-{secrets.token_hex(3)}"


def utc_now() -> str:
    # Microseconds, not seconds: `ended_at` is what orders "the last two jobs
    # to finish", and jobs on a multi-GPU host routinely end in the same second.
    return datetime.now(UTC).isoformat(timespec="microseconds")


def utc_in(seconds: float) -> str | None:
    """A timestamp `seconds` from now, or None if that is not a date.

    Seconds granularity, because everything that uses it is an estimate and a
    microsecond ETA reads as a promise.

    None rather than an exception: the only callers are estimates, and
    `estimated_runtime_min: .inf` (or a units typo of `1e10`) reaching
    `timedelta` raises OverflowError from inside the runner's monitor loop,
    which would kill the job as `runner-died`. An estimate may not decide a
    job's outcome, so an unrepresentable one is simply no estimate.
    """
    try:
        return (datetime.now(UTC) + timedelta(seconds=seconds)).isoformat(timespec="seconds")
    except (OverflowError, ValueError, OSError):
        return None


def atomic_write_text(path: Path, text: str, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp"
    try:
        tmp.write_text(text)
        tmp.chmod(mode)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def atomic_write_json(path: Path, obj: Any, mode: int = 0o644) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, sort_keys=True) + "\n", mode=mode)


def read_json(path: Path, attempts: int = 5) -> Any:
    """Read JSON, tolerating a writer that is replacing the file underneath us.

    os.replace is atomic, but a reader can still lose the race on filesystems
    where the old inode is unlinked between open() and read(), and a
    hand-edited file can be transiently truncated. Retrying briefly is far
    cheaper than making every reader take a lock.
    """
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, FileNotFoundError, OSError) as exc:
            last = exc
            time.sleep(0.05 * (attempt + 1))
    raise RuntimeError(f"could not read {path}: {last}")


def fields_of(d: Any) -> dict[str, Any]:
    """The mapping, or an empty one. A file that holds a list or `null` has
    nothing to say about any field, and every field here has a default."""
    return d if isinstance(d, dict) else {}


def as_float(d: Any, key: str, default: float) -> float:
    """`float(None)` is the bug this exists to make unreachable.

    A field another build made optional arrives as an explicit `null`; a
    hand-edited file arrives as a string. Neither may take the dispatcher down
    -- it crashed 20 times on one `"retention_days": null` and gave up -- so an
    unusable value means the default, which is what the field meant before the
    key existed at all.
    """
    value = fields_of(d).get(key)
    if value is None or isinstance(value, bool):
        return default
    try:
        return float(value)  # pyright: ignore[reportArgumentType]
    except (TypeError, ValueError):
        return default


def as_opt_float(d: Any, key: str) -> float | None:
    """The same, for a field whose `null` is a real value ("no cap at all")."""
    if fields_of(d).get(key) is None:
        return None
    parsed = as_float(d, key, float("nan"))
    return None if parsed != parsed else parsed


def as_int(d: Any, key: str, default: int) -> int:
    value = as_float(d, key, float(default))
    try:
        return int(value)
    except (OverflowError, ValueError):
        return default


def as_opt_int(d: Any, key: str) -> int | None:
    """An int field whose `null` is a real value ("no pid recorded").

    A pid that arrives as `"1234"` is the bug this exists to make unreachable:
    `os.killpg` on a string raises, and the dispatcher would then fail to clean
    up the very job whose state file is odd.
    """
    value = as_opt_float(d, key)
    if value is None:
        return None
    try:
        return int(value)
    except (OverflowError, ValueError):
        return None


def as_str(d: Any, key: str, default: str = "") -> str:
    value = fields_of(d).get(key)
    return default if value is None else str(value)


def as_opt_str(d: Any, key: str) -> str | None:
    value = fields_of(d).get(key)
    return None if value is None else str(value)


def as_bool(d: Any, key: str, default: bool = False) -> bool:
    value = fields_of(d).get(key)
    return default if value is None else bool(value)


def as_list(d: Any, key: str) -> list[Any]:
    value = fields_of(d).get(key)
    return list(value) if isinstance(value, list) else []


def as_str_list(d: Any, key: str) -> list[str]:
    value = fields_of(d).get(key)
    return [str(item) for item in value] if isinstance(value, list) else []


def as_opt_float_list(d: Any, key: str) -> list[float | None]:
    """A list of samples where `null` means "we could not read one"."""
    value = fields_of(d).get(key)
    if not isinstance(value, list):
        return []
    return [as_opt_float({key: item}, key) for item in value]


def as_str_dict(d: Any, key: str) -> dict[str, str]:
    value = fields_of(d).get(key)
    return {str(k): str(v) for k, v in value.items()} if isinstance(value, dict) else {}


def parse_env_file(path: Path) -> dict[str, str]:
    """Parse `KEY=value` / `export KEY=value` lines (secrets files, RunPod's
    /etc/rp_environment). Not a shell: no expansion, no continuations."""
    out: dict[str, str] = {}
    try:
        text = path.read_text()
    except OSError:
        return out
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :]
        key, _, value = line.partition("=")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        out[key.strip()] = value
    return out


@dataclass
class Output:
    path: str
    s3: str | None = None
    hf: str | None = None
    hf_path: str | None = None
    hf_create: bool = False
    """Create the Hugging Face repo if the sync preflight finds it missing.

    Off by default: a typo in a repo name should fail the job in seconds, not
    quietly create `org/lego-s4-typo` and upload a run into it."""

    @property
    def kept(self) -> bool:
        """No destination: what the job writes here stays on the host, in the
        workdir, when the rest of the checkout is swept."""
        return not self.s3 and not self.hf

    @staticmethod
    def from_dict(d: Any) -> Output:
        return Output(
            path=as_str(d, "path"),
            s3=as_opt_str(d, "s3"),
            hf=as_opt_str(d, "hf"),
            hf_path=as_opt_str(d, "hf_path"),
            hf_create=as_bool(d, "hf_create"),
        )


@dataclass
class JobSpec:
    job_id: str
    command: str
    name: str = ""
    setup: str | None = None
    python: str = DEFAULT_PYTHON
    """How to run Python inside the job's own environment, for the GPU check
    before `main`. The default assumes a uv project; a repo that keeps its
    stack elsewhere names its own interpreter (`.venv/bin/python`, `python`)."""
    gpus: int = 1
    use_shared: bool = False
    """May this job be dispatched to the host's `shared_gpus` -- cards gpuc
    does not own and may only borrow while nobody else is on them?

    Off by default, because a shared card is somebody else's and taking one is
    a decision about a box, not about a job. A job that opts in is dispatched
    to owned cards first and borrows only what it could not get there; see the
    dispatcher's `launch_ready`."""
    env: dict[str, str] = field(default_factory=dict)
    secrets: list[str] = field(default_factory=list)
    outputs: list[Output] = field(default_factory=list)
    sync_interval_s: int = 180
    priority: int = 50
    max_runtime_min: float | None = None
    estimated_runtime_min: float | None = None
    """Roughly how long this job expects to take, measured from the runner's
    start exactly as `max_runtime_min` is. Purely informational: nothing kills a
    job for running past it. It is how somebody else deciding between queueing
    behind this job and paying for another host finds out what they are waiting
    for, which nothing on the host can work out for them."""
    progress_command: str | None = None
    """An optional shell command, run in `workdir/` with the job's own
    environment every `progress_interval_s` of phase `main`, whose last line of
    stdout is how far along the job is -- a fraction of one (`0.42`) or a
    percentage written with a `%` (`42%`). It replaces the submitter's estimate
    with a measured one. A failure is recorded and ignored; see `progress.py`."""
    progress_interval_s: float = progress.DEFAULT_INTERVAL_S
    auto_preempt: bool = False
    """Let the dispatcher stop this job whenever that starts a more important
    one right away, as often as it takes: see `dispatcher.preempt_for_waiting`.

    It costs everything the attempt has done, so it is opt-in and belongs to
    jobs that are cheap to re-run from the start.
    """
    requires: dict[str, Any] = field(default_factory=dict)
    cleanup: str = DEFAULT_CLEANUP
    """`on_success` | `always` | `never`: when the runner deletes `workdir/`.

    The default keeps a failed or cancelled workdir so it can be inspected, and
    reclaims the (usually venv-dominated) space of a run that worked.
    """
    requeued_from: str | None = None
    """The job this one was resubmitted from by `gpuc requeue`, if any.

    Written by the control side and carried through untouched: the host never
    reads it, and publishes it in `status` so a client can follow the chain.
    Distinct from the state's `attempt`, which counts the launches of *this*
    id."""

    @staticmethod
    def from_dict(d: Any) -> JobSpec:
        """Unknown keys are ignored and a null means the default -- except for
        `command` and `gpus`, which have no default that could be right: a
        spec with nothing to run, or nothing to run it on, is a mistake to
        report, not one to paper over."""
        fields = fields_of(d)
        command = as_str(fields, "command")
        if not command:
            raise ValueError("a job spec needs a `command`")
        gpus = as_int(fields, "gpus", 1)
        if gpus < 1:
            raise ValueError(f"a job spec needs at least one GPU, got `gpus: {gpus}`")
        outputs = fields.get("outputs")
        return JobSpec(
            job_id=as_str(fields, "job_id") or new_job_id(),
            command=command,
            name=as_str(fields, "name"),
            setup=as_opt_str(fields, "setup"),
            python=as_str(fields, "python", DEFAULT_PYTHON) or DEFAULT_PYTHON,
            gpus=gpus,
            use_shared=as_bool(fields, "use_shared"),
            env=as_str_dict(fields, "env"),
            secrets=as_str_list(fields, "secrets"),
            outputs=[Output.from_dict(o) for o in (outputs if isinstance(outputs, list) else [])],
            sync_interval_s=as_int(fields, "sync_interval_s", 180),
            priority=as_int(fields, "priority", 50),
            max_runtime_min=as_opt_float(fields, "max_runtime_min"),
            estimated_runtime_min=as_opt_float(fields, "estimated_runtime_min"),
            progress_command=as_opt_str(fields, "progress_command"),
            progress_interval_s=_polling_interval(fields),
            auto_preempt=as_bool(fields, "auto_preempt"),
            requires=dict(fields.get("requires") or {})
            if isinstance(fields.get("requires"), dict)
            else {},
            cleanup=normalize_cleanup(fields.get("cleanup")),
            requeued_from=as_opt_str(fields, "requeued_from"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Upload:
    """The last thing that happened to one destination of one job.

    One record per destination, kept in the job's state: `ok_at` is the last
    successful upload, `error` the last failure, and the newer of the two is
    what stands. `output` is the spec path the files came from, or null for
    the host's mirror of the job's own log and state.
    """

    to: str
    output: str | None = None
    ok_at: str | None = None
    error: str | None = None

    @staticmethod
    def from_dict(d: Any) -> Upload | None:
        to = as_opt_str(d, "to")
        if not to:
            return None
        return Upload(
            to=to,
            output=as_opt_str(d, "output"),
            ok_at=as_opt_str(d, "ok_at"),
            error=as_opt_str(d, "error"),
        )


@dataclass
class JobState:
    status: str = "queued"
    """queued | running | succeeded | failed | cancelled."""
    intent: str | None = None
    """`cancel` | `preempt` | null: see `INTENTS`."""
    attempt: int = 1
    """How many times this job id has been started: 1 from the first launch,
    one more each time a preempted attempt is queued again. A resubmission is
    a new job id and starts at 1 again; see `JobSpec.requeued_from`."""
    priority: int = 50
    """The priority the job is (or was) ordered by. Starts as the spec's and
    is what `gpuc reorder` and `gpuc preempt --priority` change; the queue is
    every `queued` state sorted by `(priority, job_id)`, so this is the one
    copy of it."""
    estimated_runtime_min: float | None = None
    """The submitter's estimate, as `gpuc estimate` last left it. The spec is
    never rewritten after enqueue, so the live value is here and the runner
    re-reads it from here."""
    reason: str | None = None
    """What ended the job, one word: see usage.md's table."""
    problems: list[str] = field(default_factory=list)
    """What else went wrong on the way out -- `sync`, `no-outputs` -- for a
    job that already had a reason. A succeeded job with a failed upload is
    `failed: sync` outright, since its result never arrived."""
    exit_code: int | None = None
    gpus: list[str] = field(default_factory=list)
    started_at: str | None = None
    ended_at: str | None = None
    phase: str | None = None
    """setup | preflight | main | sync, or null between phases and once the
    job is over. `sync` is the runner's final upload and cleanup: the status
    stays `running` until its very last write, so a job is finished exactly
    when its runner is gone."""
    pgid: int | None = None
    """The process group of the phase now running, which the runner publishes
    when it spawns one. Never the runner's own group."""
    isolation: str | None = None
    """`cgroup` when the phase runs in a transient systemd scope, `pgid` when it
    is only a process group. A `pgid` job's daemonised grandchildren survive a
    kill; a `cgroup` job's cannot."""
    cgroup_unit: str | None = None
    """The scope unit of the phase now running, which `systemctl --user stop`
    reaps whole. Null between phases and on a `pgid` host."""
    runner_pid: int | None = None
    """The runner's own pid, with the boot id and start time that make it
    provably the same process later. Written by the runner in the same
    compare-and-set that takes the job out of the queue, so a `running` state
    always names a runner that existed."""
    runner_boot_id: str | None = None
    runner_starttime: str | None = None
    util_recent: list[float | None] = field(default_factory=list)
    progress_pct: float | None = None
    """The last percentage the spec's `progress_command` reported, 0-100. Null
    on a job that has no progress command, or has not answered yet."""
    progress_error: str | None = None
    """Why the last progress poll produced nothing. Kept because the alternative
    is a job that silently never estimates and nobody knowing the command is
    broken; it never affects the job's outcome."""
    eta: str | None = None
    """When this job is expected to finish, from `progress_pct` if there is one
    and from the spec's `estimated_runtime_min` otherwise. Null while the job is
    queued, once it is finished, and whenever it has nothing to estimate from."""
    uploads: list[Upload] = field(default_factory=list)
    """One record per destination this job uploads to; see `Upload`. The
    whole account of whether its outputs, its log and its state are
    somewhere other than this host."""
    workdir_bytes: int | None = None
    """What deleting `workdir/` would free, measured once when the job ended.

    A finished job's workdir does not change, and measuring it is the most
    expensive thing `status` can do -- a torch venv is ~67k files, and a host
    holding sixty of them made every `status` call walk half a million. So it
    is measured where the runner is already standing in the tree, and `status`
    reads the number. Null means nobody has measured it yet (a job that ended
    before this was recorded, or a runner that died before writing it), which
    matters only while the workdir is still there: the first `status` after
    that walks it and writes the figure here. Zero is an answer and not an
    absence -- usually the workdir being gone, sometimes a tree all of whose
    extents are shared with something else. See
    `cleanup.reported_workdir_bytes`, which is what decides what `status`
    reports and needs this populated only for a workdir still on disk.

    Not the number `du` gives: see `cleanup.reclaimable_bytes`. `gpuc clean`
    measures afresh rather than trusting this, because it is about to delete
    what it is quoting."""
    outputs_lost: bool = False
    """The drain retried this job's output upload to the end and it still
    failed, so an ephemeral host is about to take the only copy with it. The
    record's `error` holds the last failure. Surfaced by `status` because
    nothing fixes it afterwards except re-running the job."""
    ran: bool = True
    """Whether `main` has started, in this attempt or an earlier one: false
    from enqueue, recorded by the runner as `main` begins, kept by the write
    that ends an attempt and by the one that queues it again. Only a job
    that got that far can have written anything under `outputs:`, so nothing
    of a job that never did is pending, and its secrets are not kept for a
    drain that would upload nothing. True when a state file does not say
    (`from_dict`), and on a bare `JobState`: the point of asking is to keep
    the only copy of a result, so not knowing counts as having run."""
    checkout_removed_at: str | None = None
    """When the checkout was deleted from a `workdir/` that still holds the
    job's kept outputs. With it set, the workdir is those outputs and nothing
    else: no sweep has anything left to take, and only a purge removes them."""
    kept_bytes: int | None = None
    """What those kept outputs hold, measured as the checkout went: disk no
    sweep will ever free, which `status` reports so it is not forgotten."""

    @staticmethod
    def from_dict(d: Any) -> JobState:
        """Unknown keys are dropped and a null leaves the field at its default:
        every field here is additive, and a state file written by another build
        must still tell this one whether the job is running.

        Coerced field by field, like every other reader here. Copying the raw
        JSON in put a `"1234"` where a pid belongs, and the dispatcher then
        crashed in `os.killpg` on every pass -- a state file is the one input
        that can be written by another build, or by hand.
        """
        fields = fields_of(d)
        intent = as_opt_str(fields, "intent")
        return JobState(
            status=as_str(fields, "status", "queued") or "queued",
            intent=intent if intent in INTENTS else None,
            attempt=as_int(fields, "attempt", 1),
            priority=as_int(fields, "priority", 50),
            estimated_runtime_min=as_opt_float(fields, "estimated_runtime_min"),
            reason=as_opt_str(fields, "reason"),
            problems=as_str_list(fields, "problems"),
            exit_code=as_opt_int(fields, "exit_code"),
            gpus=as_str_list(fields, "gpus"),
            started_at=as_opt_str(fields, "started_at"),
            ended_at=as_opt_str(fields, "ended_at"),
            phase=as_opt_str(fields, "phase"),
            pgid=as_opt_int(fields, "pgid"),
            isolation=as_opt_str(fields, "isolation"),
            cgroup_unit=as_opt_str(fields, "cgroup_unit"),
            runner_pid=as_opt_int(fields, "runner_pid"),
            runner_boot_id=as_opt_str(fields, "runner_boot_id"),
            runner_starttime=as_opt_str(fields, "runner_starttime"),
            util_recent=as_opt_float_list(fields, "util_recent"),
            progress_pct=as_opt_float(fields, "progress_pct"),
            progress_error=as_opt_str(fields, "progress_error"),
            eta=as_opt_str(fields, "eta"),
            uploads=[
                upload
                for upload in (Upload.from_dict(u) for u in as_list(fields, "uploads"))
                if upload is not None
            ],
            workdir_bytes=as_opt_int(fields, "workdir_bytes"),
            outputs_lost=as_bool(fields, "outputs_lost"),
            ran=as_bool(fields, "ran", True),
            checkout_removed_at=as_opt_str(fields, "checkout_removed_at"),
            kept_bytes=as_opt_int(fields, "kept_bytes"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def finished(self) -> bool:
        return self.status in FINISHED_STATUSES

    @property
    def mirror(self) -> Upload | None:
        """The record of this job's log and state being mirrored, if any."""
        return next((u for u in self.uploads if u.output is None), None)

    @property
    def mirrored(self) -> bool:
        """Is the record of this job somewhere other than this host? The
        precondition for deleting its job dir."""
        mirror = self.mirror
        return mirror is not None and mirror.ok_at is not None and mirror.error is None

    def output_uploads(self) -> list[Upload]:
        return [u for u in self.uploads if u.output is not None]

    def upload_errors(self) -> list[str]:
        """The last failure at each destination that has one standing."""
        return [u.error for u in self.uploads if u.error]

    def outputs_uploaded(self, spec: JobSpec) -> bool:
        """Did the last upload of every declared output reach every one of its
        destinations? False for a job with outputs and no record at all."""
        wanted = {(o.path, d) for o in spec.outputs for d in _destination_uris(o, spec.job_id)}
        done = {(u.output, u.to) for u in self.uploads if u.ok_at and not u.error}
        return wanted <= done

    def forget_output_uploads(self) -> None:
        """Drop every output's success: a periodic tick having worked says
        nothing about the files written after it, so only an upload that ran
        after the job stopped writing may vouch for its outputs."""
        for record in self.output_uploads():
            record.ok_at = None

    @staticmethod
    def initial(spec: JobSpec) -> JobState:
        """The state a job is enqueued with: everything the spec said that can
        change afterwards, copied out of it once."""
        return JobState(
            status="queued",
            priority=spec.priority,
            estimated_runtime_min=spec.estimated_runtime_min,
            ran=False,
        )


@dataclass(frozen=True)
class Outcome:
    """How a job ended, in the one shape every terminal write takes.

    `reason` is one of the words usage.md's table lists, and `ran` says whether
    `main` started. Outputs are what `main` produces: a job that never got
    there -- a card that is not here, a setup that failed, a sync preflight
    that refused, a cancel before the first phase, a spec the dispatcher could
    not read -- has no result to upload, so the final upload and the
    no-outputs check are skipped rather than reported as a problem beside the
    reason the job actually ended for.
    """

    status: str
    reason: str | None = None
    exit_code: int = 1
    ran: bool = True

    def __post_init__(self) -> None:
        if self.status not in FINISHED_STATUSES:
            raise ValueError(f"an outcome is a finished status, not {self.status!r}")


def finish(
    job_id: str,
    outcome: Outcome,
    *,
    expect: str | tuple[str, ...] = "running",
    forget_output_uploads: bool = False,
    **extra: Any,
) -> JobState | None:
    """The one write that ends a job, as a compare-and-set on `status`.

    Every terminal status goes through here -- the runner's, and the
    dispatcher's for a job it failed before or instead of a runner -- so the
    fields a finished job must not carry are cleared in one place: the
    `intent` that stood against it, the phase and the processes of the attempt,
    the `eta` that would read as a promise the job is still going. None means
    the status was no longer what `expect` says and nothing was written: a
    runner that lost its job to a cancel, or a dispatcher whose runner finished
    after all, learns it from that rather than from a job written twice.

    `forget_output_uploads` is for a runner that died: its periodic uploads
    were recorded, its final one never ran, and the sweep must not read the
    former as the latter. An unreadable state raises, and the caller leaves the
    job alone: writing defaults over a file that cannot be read would replace
    its priority and upload records with guesses.

    `ran` is never cleared, only set: the runner records it as `main`
    begins, and an attempt that never reached `main` may be the second of a
    job whose first did, with that attempt's outputs still in the workdir. A
    dispatcher failing a job whose runner died after claiming it reports
    `ran` rather than knowing, which is the fail-closed answer.
    """
    wanted = (expect,) if isinstance(expect, str) else expect
    with locked(job_id):
        state = read_state(job_id)
        if state.status not in wanted:
            return None
        state.status = outcome.status
        state.reason = outcome.reason
        state.exit_code = outcome.exit_code
        state.ended_at = utc_now()
        state.intent = None
        state.phase = None
        state.pgid = None
        state.cgroup_unit = None
        state.eta = None
        state.ran = state.ran or outcome.ran
        if forget_output_uploads:
            state.forget_output_uploads()
        write_state(job_id, _apply(state, extra))
        return state


def _destination_uris(output: Output, job_id: str) -> list[str]:
    from gpuc.host import destinations  # it imports this module

    return [d.uri for d in destinations.of(output, job_id)]


def record_upload(
    job_id: str, to: str, output: str | None, *, ok_at: str | None = None, error: str | None = None
) -> None:
    """Note what just happened at one destination. The newer of success and
    failure stands: a success clears the error, a failure keeps the last
    success time so a reader can see when it last worked."""
    with locked(job_id):
        state = read_state(job_id)
        record = next((u for u in state.uploads if u.to == to and u.output == output), None)
        if record is None:
            record = Upload(to=to, output=output)
            state.uploads.append(record)
        if ok_at is not None:
            record.ok_at, record.error = ok_at, None
        else:
            record.error = error
        write_state(job_id, state)


def clear_output_uploads(job_id: str) -> None:
    """Forget every output's success: the final upload is the only one that
    proves the files written since the last tick are safe."""
    with locked(job_id):
        state = read_state(job_id)
        state.forget_output_uploads()
        write_state(job_id, state)


def read_spec(job_id: str) -> JobSpec:
    return JobSpec.from_dict(read_json(paths.spec_file(job_id)))


def write_state(job_id: str, state: JobState) -> None:
    atomic_write_json(paths.state_file(job_id), state.to_dict())


def read_state(job_id: str) -> JobState:
    return JobState.from_dict(read_json(paths.state_file(job_id)))


@contextlib.contextmanager
def locked(job_id: str) -> Iterator[None]:
    """Hold the job's lock for a read-modify-write of its state.

    Three processes update one job's `state.json`: the dispatcher, the job's
    runner (from two threads), and whatever `python -m gpuc.host` command an
    ssh session runs. Each write is an atomic replace, so readers never see a
    torn file, but two read-modify-writes that overlap lose one of them --
    which, when the lost one was `intent: cancel`, is a job that keeps running.
    """
    path = paths.job_lock_file(job_id)
    if not path.parent.is_dir():
        raise FileNotFoundError(f"no such job: {job_id}")
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


def _apply(state: JobState, fields: Mapping[str, Any]) -> JobState:
    for key, value in fields.items():
        if key not in JobState.__dataclass_fields__:
            raise KeyError(f"unknown JobState field: {key}")
        setattr(state, key, value)
    return state


def update_state(job_id: str, **fields: Any) -> JobState:
    with locked(job_id):
        state = _apply(read_state(job_id), fields)
        write_state(job_id, state)
        return state


def transition(
    job_id: str, *, expect: str | tuple[str, ...], attempt: int | None = None, **fields: Any
) -> JobState | None:
    """Update the state only if its `status` is still one of `expect` -- and,
    when `attempt` is given, only if that is still the attempt.

    The compare-and-set every change of ownership goes through: the runner
    claims a queued job, a cancel ends a queued job, a requeue puts a finished
    one back. Two of those racing on one job -- a cancel landing as the
    dispatcher launches it -- cannot both win, and the loser learns it from
    the None rather than from a job that is both running and cancelled.
    """
    wanted = (expect,) if isinstance(expect, str) else expect
    with locked(job_id):
        state = read_state(job_id)
        if state.status not in wanted or (attempt is not None and state.attempt != attempt):
            return None
        write_state(job_id, _apply(state, fields))
        return state


def list_job_ids() -> list[str]:
    root = paths.jobs_dir()
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir())


@dataclass(frozen=True)
class ManagedKey:
    """A key of `HostConfig.env` the tool itself has an opinion about.

    `env` is otherwise opaque, hand-set, and replaced wholesale by `--env`.
    The keys here are the exceptions, and this table is the whole list of
    them: `sticky` ones survive an `--env` that does not name them, because
    bootstrap derived them from the host's own filesystem and a user
    restating their `HF_TOKEN` did not mean to lose the uv cache; `on_path`
    ones name directories holding binaries, which go on every child's PATH;
    `beside_home` ones are derived by bootstrap as `<parent of gpuc
    home>/.cache/<name>` when the host names nothing -- beside gpuc home so a
    persistent root keeps them and an `rm -rf` of gpuc home does not.
    """

    sticky: bool = False
    on_path: bool = False
    beside_home: str | None = None


MANAGED_ENV: dict[str, ManagedKey] = {
    "UV_CACHE_DIR": ManagedKey(sticky=True, beside_home="uv"),
    "HF_HOME": ManagedKey(sticky=True, beside_home="huggingface"),
    "GPUC_DATA_DIR": ManagedKey(sticky=True),
    "UV_INSTALL_DIR": ManagedKey(on_path=True),
    "UV_TOOL_BIN_DIR": ManagedKey(on_path=True),
}


def sticky_env(theirs: Mapping[str, str], env: dict[str, str]) -> dict[str, str]:
    """`env` with the sticky keys the host already had carried over."""
    for key, managed in MANAGED_ENV.items():
        if managed.sticky and key in theirs:
            env.setdefault(key, theirs[key])
    return env


def cache_beside(home: str, name: str) -> str:
    """`<parent of gpuc home>/.cache/<name>`; `<home>/<name>-cache` for a home
    with no parent to speak of."""
    parent = home.rstrip("/").rsplit("/", 1)[0] if "/" in home.rstrip("/") else ""
    return f"{parent}/.cache/{name}" if parent else f"{home.rstrip('/')}/{name}-cache"


@dataclass
class HostConfig:
    schema_version: int = SCHEMA_VERSION
    host: str = "local"
    gpus: list[str] = field(default_factory=list)
    """The cards this host owns, stored exactly as they were given: nvidia-smi
    indices, UUIDs, or a mix.

    An index is how a share of a shared box is agreed ("you get 2 and 3"), and
    resolving it to a UUID at registration would freeze one boot's numbering
    into a file nobody looks at again. So the entry stays as typed and every
    reader resolves it against the live table (`gpus.resolve`); everything
    downstream of that is UUIDs.
    """
    shared_gpus: list[str] = field(default_factory=list)
    """Cards on this box that gpuc may *borrow*, spelled like `gpus`.

    Not ours: somebody else owns them, and the only time we may run on one is
    while nothing at all is. So a job reaches them only if it asked to
    (`use_shared`), only after the owned cards are full, and only while
    nvidia-smi says the card holds no memory and is doing no work. Nothing
    here is counted as capacity for a job that did not ask.

    There is deliberately no per-host floor on which jobs may borrow (a
    minimum priority, say): borrowing is not a reservation, so a floor would
    never protect an important job from a trivial one, only keep the card
    idle.
    """
    provider: dict[str, Any] | None = None
    idle_minutes: float = 15.0
    s3_prefix: str | None = None
    created_at: str | None = None
    retention_days: float | None = None
    """Automatic purge horizon, in days. Null (the default) never auto-purges.

    When set, the dispatcher purges finished, mirrored job dirs older than this
    at startup and at most once an hour while it lives. Never forced: a job
    without a confirmed mirror is kept however old it is."""
    workdir_days: float | None = None
    """Automatic workdir horizon, in days. Null (the default) never sweeps.

    The shorter of the two horizons, and the one worth setting: it reclaims a
    finished job's `workdir/` -- code and venv, the part `gpuc requeue` rebuilds
    from git -- and leaves the record alone, so it needs no mirror and asks
    nothing of the caller. `retention_days` is the one that deletes `log.txt`.

    Null here, and `cleanup.DEFAULT_WORKDIR_DAYS` only in a host's very first
    config (the control side's `first_config`): nothing that merely reads a
    config may turn
    a sweep on, so a host configured with the key unset stays that way however
    many packages are shipped to it."""
    env: dict[str, str] = field(default_factory=dict)
    """Host-wide environment, applied to every job before the job's own `env`.

    Hand-set per host (`gpuc host add|set --env K=V`) for the paths that belong
    somewhere other than this host's defaults -- an `HF_HOME` on a big volume,
    say. Nothing populates it automatically.
    """
    pkg_commit: str | None = None
    """The gpuc commit bootstrap shipped to this host, for `gpuc version` and
    the `status` warning that a host is running another build than this one.
    None is a host nothing has bootstrapped: `submit` refuses it and `status`
    warns."""

    @staticmethod
    def from_dict(d: Any) -> HostConfig:
        """The reader that took the dispatcher down. Nothing here may raise on
        a config.json written by a different build of gpuc: an unknown key is
        ignored, and a null for a field that is not nullable means its default.
        """
        fields = fields_of(d)
        provider = fields.get("provider")
        return HostConfig(
            schema_version=as_int(fields, "schema_version", SCHEMA_VERSION),
            host=as_str(fields, "host", "local") or "local",
            gpus=as_str_list(fields, "gpus"),
            shared_gpus=as_str_list(fields, "shared_gpus"),
            provider=provider if isinstance(provider, dict) else None,
            idle_minutes=as_float(fields, "idle_minutes", 15.0),
            s3_prefix=as_opt_str(fields, "s3_prefix"),
            created_at=as_opt_str(fields, "created_at"),
            retention_days=as_opt_float(fields, "retention_days"),
            workdir_days=as_opt_float(fields, "workdir_days"),
            env=as_str_dict(fields, "env"),
            pkg_commit=as_opt_str(fields, "pkg_commit"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def ephemeral(self) -> bool:
        return self.provider is not None

    def may_borrow(self, spec: JobSpec) -> bool:
        """May this job be given one of this host's shared cards?

        About the job, not about what the cards are doing this second --
        whether one is free is the dispatcher's question, asked only once this
        has said yes. And fixed for the life of a queued job: nothing changes
        a spec's `use_shared` after submit, which is what lets the dispatcher
        *fail* a job this says no to rather than leaving it to wait forever.
        """
        return spec.use_shared and bool(self.shared_gpus)

    def bin_dirs(self) -> list[str]:
        """Directories from `env` to put on PATH, in order, without duplicates."""
        out: list[str] = []
        for key, managed in MANAGED_ENV.items():
            value = self.env.get(key) if managed.on_path else None
            if value and value not in out:
                out.append(value)
        return out

    def apply_env(self, env: dict[str, str]) -> dict[str, str]:
        """Merge the host env into ``env`` and re-front PATH with its bin dirs.

        Mutates and returns ``env``. Callers apply this *before* a job's own
        ``env``, so a job can still override any of it.
        """
        env.update(self.env)
        env["PATH"] = paths.path_with_user_bins(env, extra=self.bin_dirs())
        return env


def read_config() -> HostConfig:
    path = paths.config_file()
    if not path.exists():
        return HostConfig()
    return HostConfig.from_dict(read_json(path))


def write_config(config: HostConfig) -> None:
    atomic_write_json(paths.config_file(), config.to_dict())


def merged_config(existing: Any, patch: Mapping[str, Any]) -> dict[str, Any]:
    """`existing` with `patch`'s keys replaced, normalised, unknown keys kept.

    The one rule for changing a host's config, which the control side applies
    by read-merge-rename over the transport. A key this build does not know is
    carried through untouched -- it belongs to whichever build wrote it -- and
    the keys it does know are normalised, so a hand-written
    `"idle_minutes": "30"` cannot leave a string where the dispatcher reads a
    number. `env` is replaced wholesale rather than merged: "set it to exactly
    this" is the only rule that can also express "set it to nothing".
    """
    merged = {**fields_of(existing), **patch}
    return {**merged, **HostConfig.from_dict(merged).to_dict()}
