"""Job identity, spec/state serialisation, host config, atomic file IO."""

from __future__ import annotations

import json
import os
import secrets
import time
from collections.abc import Mapping
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

ON_SUCCESS = "on_success"
ALWAYS = "always"
NEVER = "never"
CLEANUP_POLICIES = (ON_SUCCESS, ALWAYS, NEVER)
DEFAULT_CLEANUP = ON_SUCCESS


def _polling_interval(fields: Any) -> float:
    """`progress_interval_s`, with anything unusable meaning the default.

    A zero, a negative or a NaN reaches here from a hand-edited `spec.json`, a
    staged `incoming/<id>.json`, or another build -- the control side's
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
    -- it crashed 20 times on one `"ttl_hours": null` and gave up -- so an
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
class LowUtil:
    enabled: bool = True
    window_min: float = 25.0
    floor_pct: float = 5.0
    grace_min: float = 10.0

    @staticmethod
    def from_dict(d: Any) -> LowUtil:
        return LowUtil(
            enabled=as_bool(d, "enabled", True),
            window_min=as_float(d, "window_min", 25.0),
            floor_pct=as_float(d, "floor_pct", 5.0),
            grace_min=as_float(d, "grace_min", 10.0),
        )


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
    gpus: int = 1
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
    low_util: LowUtil = field(default_factory=LowUtil)
    requires: dict[str, Any] = field(default_factory=dict)
    cleanup: str = DEFAULT_CLEANUP
    """`on_success` | `always` | `never`: when the runner deletes `workdir/`.

    The default keeps a failed or cancelled workdir so it can be inspected, and
    reclaims the (usually venv-dominated) space of a run that worked.
    """
    attempt: int = 1

    @staticmethod
    def from_dict(d: Any) -> JobSpec:
        """Unknown keys are ignored and a null means the default -- except for
        `command`, which has no default that could be right: a spec with
        nothing to run is a mistake to report, not one to paper over."""
        fields = fields_of(d)
        command = as_str(fields, "command")
        if not command:
            raise ValueError("a job spec needs a `command`")
        outputs = fields.get("outputs")
        return JobSpec(
            job_id=as_str(fields, "job_id") or new_job_id(),
            command=command,
            name=as_str(fields, "name"),
            setup=as_opt_str(fields, "setup"),
            gpus=as_int(fields, "gpus", 1),
            env=as_str_dict(fields, "env"),
            secrets=as_str_list(fields, "secrets"),
            outputs=[Output.from_dict(o) for o in (outputs if isinstance(outputs, list) else [])],
            sync_interval_s=as_int(fields, "sync_interval_s", 180),
            priority=as_int(fields, "priority", 50),
            max_runtime_min=as_opt_float(fields, "max_runtime_min"),
            estimated_runtime_min=as_opt_float(fields, "estimated_runtime_min"),
            progress_command=as_opt_str(fields, "progress_command"),
            progress_interval_s=_polling_interval(fields),
            low_util=LowUtil.from_dict(fields.get("low_util")),
            requires=dict(fields.get("requires") or {})
            if isinstance(fields.get("requires"), dict)
            else {},
            cleanup=normalize_cleanup(fields.get("cleanup")),
            attempt=as_int(fields, "attempt", 1),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class JobState:
    status: str = "queued"
    """queued | running | succeeded | failed | cancelled."""
    attempt: int = 1
    reason: str | None = None
    exit_code: int | None = None
    gpus: list[str] = field(default_factory=list)
    started_at: str | None = None
    ended_at: str | None = None
    phase: str | None = None
    """setup | preflight | main | sync, or null between phases."""
    pid: int | None = None
    pgid: int | None = None
    isolation: str | None = None
    """`cgroup` when the phase runs in a transient systemd scope, `pgid` when it
    is only a process group. A `pgid` job's daemonised grandchildren survive a
    kill; a `cgroup` job's cannot."""
    cgroup_unit: str | None = None
    """The scope unit of the phase now running, which `systemctl --user stop`
    reaps whole. Null between phases and on a `pgid` host."""
    runner_pid: int | None = None
    runner_boot_id: str | None = None
    runner_starttime: str | None = None
    util_recent: list[float | None] = field(default_factory=list)
    util_sampled_at: str | None = None
    progress_pct: float | None = None
    """The last percentage the spec's `progress_command` reported, 0-100. Null
    on a job that has no progress command, or has not answered yet."""
    progress_at: str | None = None
    progress_error: str | None = None
    """Why the last progress poll produced nothing. Kept because the alternative
    is a job that silently never estimates and nobody knowing the command is
    broken; it never affects the job's outcome."""
    eta: str | None = None
    """When this job is expected to finish, from `progress_pct` if there is one
    and from the spec's `estimated_runtime_min` otherwise. Null while the job is
    queued, once it is finished, and whenever it has nothing to estimate from."""
    sync_error: str | None = None
    workdir_removed: bool = False
    """Whether `workdir/` has been deleted, by the job's `cleanup:` policy or by
    `gpuc clean`. Recorded so `status` and `logs` can say "gone on purpose"
    rather than leaving an empty job dir to look like data loss."""
    meta_synced_at: str | None = None
    """When this job's `log.txt` and `state.json` were last confirmed mirrored.

    Written only after a *successful* final `sync_job_meta`, and the whole
    precondition for `purge`: the local state is the authority on whether a job
    dir may be deleted, because the mirror cannot be consulted from the host
    without credentials the host may not have. Null means "no confirmed backup"
    -- either the upload failed or this host has no `s3_prefix` at all."""
    meta_synced_to: str | None = None
    """The `s3_prefix` the confirmed mirror went to, so a purge can name it."""
    outputs_synced_at: str | None = None
    """When the *final* upload of this job's `outputs:` finished without error.

    `outputs:` paths live inside `workdir/`, so a job that ended `failed: sync`
    -- or was killed between periodic ticks -- may hold the only copy of what it
    produced. Null with a declared `outputs:` and a workdir still on disk means
    `purge` must not delete it. A spec that declares no outputs leaves this null
    too; there is simply nothing to confirm."""
    outputs_lost: bool = False
    """The drain retried this job's output upload to the end and it still
    failed, so an ephemeral host is about to take the only copy with it.
    `sync_error` holds the last failure. Surfaced by `status` because nothing
    fixes it afterwards except re-running the job."""

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
        return JobState(
            status=as_str(fields, "status", "queued") or "queued",
            attempt=as_int(fields, "attempt", 1),
            reason=as_opt_str(fields, "reason"),
            exit_code=as_opt_int(fields, "exit_code"),
            gpus=as_str_list(fields, "gpus"),
            started_at=as_opt_str(fields, "started_at"),
            ended_at=as_opt_str(fields, "ended_at"),
            phase=as_opt_str(fields, "phase"),
            pid=as_opt_int(fields, "pid"),
            pgid=as_opt_int(fields, "pgid"),
            isolation=as_opt_str(fields, "isolation"),
            cgroup_unit=as_opt_str(fields, "cgroup_unit"),
            runner_pid=as_opt_int(fields, "runner_pid"),
            runner_boot_id=as_opt_str(fields, "runner_boot_id"),
            runner_starttime=as_opt_str(fields, "runner_starttime"),
            util_recent=as_opt_float_list(fields, "util_recent"),
            util_sampled_at=as_opt_str(fields, "util_sampled_at"),
            progress_pct=as_opt_float(fields, "progress_pct"),
            progress_at=as_opt_str(fields, "progress_at"),
            progress_error=as_opt_str(fields, "progress_error"),
            eta=as_opt_str(fields, "eta"),
            sync_error=as_opt_str(fields, "sync_error"),
            workdir_removed=as_bool(fields, "workdir_removed"),
            meta_synced_at=as_opt_str(fields, "meta_synced_at"),
            meta_synced_to=as_opt_str(fields, "meta_synced_to"),
            outputs_synced_at=as_opt_str(fields, "outputs_synced_at"),
            outputs_lost=as_bool(fields, "outputs_lost"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def finished(self) -> bool:
        return self.status in FINISHED_STATUSES


def write_spec(spec: JobSpec) -> None:
    atomic_write_json(paths.spec_file(spec.job_id), spec.to_dict())


def read_spec(job_id: str) -> JobSpec:
    return JobSpec.from_dict(read_json(paths.spec_file(job_id)))


def update_spec(job_id: str, **fields: Any) -> JobSpec:
    """Change fields of a job's spec, in place on disk.

    Raw JSON in and raw JSON out rather than a `JobSpec` round trip: a spec
    written by another build holds keys `from_dict` drops, and setting an
    estimate is not a reason to lose them.
    """
    document = dict(fields_of(read_json(paths.spec_file(job_id))))
    for key, value in fields.items():
        if key not in JobSpec.__dataclass_fields__:
            raise KeyError(f"unknown JobSpec field: {key}")
        document[key] = value
    atomic_write_json(paths.spec_file(job_id), document)
    return JobSpec.from_dict(document)


def write_state(job_id: str, state: JobState) -> None:
    atomic_write_json(paths.state_file(job_id), state.to_dict())


def read_state(job_id: str) -> JobState:
    return JobState.from_dict(read_json(paths.state_file(job_id)))


def update_state(job_id: str, **fields: Any) -> JobState:
    state = read_state(job_id)
    for key, value in fields.items():
        if key not in JobState.__dataclass_fields__:
            raise KeyError(f"unknown JobState field: {key}")
        setattr(state, key, value)
    write_state(job_id, state)
    return state


def list_job_ids() -> list[str]:
    root = paths.jobs_dir()
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir())


HOST_ENV_BIN_KEYS = ("UV_INSTALL_DIR", "UV_TOOL_BIN_DIR")
"""Keys in ``HostConfig.env`` whose values are directories holding binaries.

A host told to keep uv (or its tools) somewhere other than ``$HOME`` needs
that directory on ``PATH`` too, or the runner's own ``uv run --no-sync``
preflight cannot find it. Deriving the extra PATH entries from the env dict
keeps one source of truth: the config.
"""


@dataclass
class HostConfig:
    schema_version: int = SCHEMA_VERSION
    host: str = "local"
    gpus: list[str] = field(default_factory=list)
    provider: dict[str, Any] | None = None
    idle_minutes: float = 15.0
    ttl_hours: float | None = None
    """Hard cap on this host's life, in hours. Null (the default) never
    terminates on age: the idle timer is what stops an ephemeral host, and a
    wall clock that kills a running job at hour 24 is a worse failure than a
    pod that idles for fifteen minutes. When set, the dispatcher kills the
    running job with reason `ttl`, syncs, and terminates."""
    s3_prefix: str | None = None
    created_at: str | None = None
    retention_days: float | None = None
    """Automatic purge horizon, in days. Null (the default) never auto-purges.

    When set, the dispatcher purges finished, mirrored job dirs older than this
    at startup and at most once an hour while it lives. Never forced: a job
    without a confirmed mirror is kept however old it is."""
    env: dict[str, str] = field(default_factory=dict)
    """Host-wide environment, applied to every job before the job's own `env`.

    Hand-set per host (`gpuc host add|set --env K=V`) for the paths that belong
    somewhere other than this host's defaults -- an `HF_HOME` on a big volume,
    say. Nothing populates it automatically.
    """
    pkg_commit: str | None = None
    """The gpuc commit bootstrap shipped to this host, for `gpuc version` and
    the `status` warning that a host is running an older build than this one."""

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
            provider=provider if isinstance(provider, dict) else None,
            idle_minutes=as_float(fields, "idle_minutes", 15.0),
            ttl_hours=as_opt_float(fields, "ttl_hours"),
            s3_prefix=as_opt_str(fields, "s3_prefix"),
            created_at=as_opt_str(fields, "created_at"),
            retention_days=as_opt_float(fields, "retention_days"),
            env=as_str_dict(fields, "env"),
            pkg_commit=as_opt_str(fields, "pkg_commit"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def ephemeral(self) -> bool:
        return self.provider is not None

    def bin_dirs(self) -> list[str]:
        """Directories from `env` to put on PATH, in order, without duplicates."""
        out: list[str] = []
        for key in HOST_ENV_BIN_KEYS:
            value = self.env.get(key)
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


def merge_config(patch: dict[str, Any]) -> dict[str, Any]:
    """Apply `patch` to this host's config.json and return what it now holds.

    The host owns its config, so every control machine changes it the same way:
    read what is there, replace the named keys, write the whole file back
    atomically. A key this build does not know is carried through untouched --
    it belongs to whichever build wrote it, not to us -- and the keys it does
    know are normalised, so a hand-written `"ttl_hours": "24"` cannot leave a
    string where the dispatcher reads a number.

    `env` is replaced wholesale rather than merged: "set it to exactly this" is
    the only rule that can also express "set it to nothing".
    """
    path = paths.config_file()
    document = merged_config(read_json(path) if path.exists() else {}, patch)
    atomic_write_json(path, document)
    return document


def merged_config(existing: Any, patch: Mapping[str, Any]) -> dict[str, Any]:
    """`existing` with `patch`'s keys replaced, normalised, unknown keys kept.

    Split out from `merge_config` because the control side applies the same
    rule by hand on a host that has no gpuc package to run yet, and the two
    must not be able to disagree about what a patch means.
    """
    merged = {**fields_of(existing), **patch}
    return {**merged, **HostConfig.from_dict(merged).to_dict()}
