"""Job identity, spec/state serialisation, host config, atomic file IO."""

from __future__ import annotations

import json
import os
import secrets
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gpuc.host import paths

Status = str  # queued | running | succeeded | failed | cancelled
Phase = str
PHASES = ("setup", "preflight", "main", "sync")


def new_job_id() -> str:
    return f"{datetime.now(UTC):%Y%m%d-%H%M%S}-{secrets.token_hex(3)}"


def utc_now() -> str:
    # Microseconds, not seconds: `ended_at` is what orders "the last two jobs
    # to finish", and jobs on a multi-GPU host routinely end in the same second.
    return datetime.now(UTC).isoformat(timespec="microseconds")


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
    def from_dict(d: dict[str, Any] | None) -> LowUtil:
        if not d:
            return LowUtil()
        return LowUtil(
            enabled=bool(d.get("enabled", True)),
            window_min=float(d.get("window_min", 25.0)),
            floor_pct=float(d.get("floor_pct", 5.0)),
            grace_min=float(d.get("grace_min", 10.0)),
        )


@dataclass
class Output:
    path: str
    s3: str | None = None
    hf: str | None = None
    hf_path: str | None = None

    @staticmethod
    def from_dict(d: dict[str, Any]) -> Output:
        return Output(
            path=str(d["path"]),
            s3=d.get("s3"),
            hf=d.get("hf"),
            hf_path=d.get("hf_path"),
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
    low_util: LowUtil = field(default_factory=LowUtil)
    requires: dict[str, Any] = field(default_factory=dict)
    attempt: int = 1

    @staticmethod
    def from_dict(d: dict[str, Any]) -> JobSpec:
        return JobSpec(
            job_id=str(d.get("job_id") or new_job_id()),
            command=str(d["command"]),
            name=str(d.get("name", "")),
            setup=d.get("setup"),
            gpus=int(d.get("gpus", 1)),
            env={str(k): str(v) for k, v in (d.get("env") or {}).items()},
            secrets=[str(s) for s in (d.get("secrets") or [])],
            outputs=[Output.from_dict(o) for o in (d.get("outputs") or [])],
            sync_interval_s=int(d.get("sync_interval_s", 180)),
            priority=int(d.get("priority", 50)),
            max_runtime_min=(
                None if d.get("max_runtime_min") is None else float(d["max_runtime_min"])
            ),
            low_util=LowUtil.from_dict(d.get("low_util")),
            requires=dict(d.get("requires") or {}),
            attempt=int(d.get("attempt", 1)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class JobState:
    status: Status = "queued"
    attempt: int = 1
    reason: str | None = None
    exit_code: int | None = None
    gpus: list[str] = field(default_factory=list)
    started_at: str | None = None
    ended_at: str | None = None
    phase: Phase | None = None
    pid: int | None = None
    pgid: int | None = None
    runner_pid: int | None = None
    runner_boot_id: str | None = None
    runner_starttime: str | None = None
    util_recent: list[float | None] = field(default_factory=list)
    util_sampled_at: str | None = None
    sync_error: str | None = None

    @staticmethod
    def from_dict(d: dict[str, Any]) -> JobState:
        known = {f: d.get(f) for f in JobState.__dataclass_fields__}
        state = JobState()
        for key, value in known.items():
            if value is not None:
                setattr(state, key, value)
        return state

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def finished(self) -> bool:
        return self.status in ("succeeded", "failed", "cancelled")


def write_spec(spec: JobSpec) -> None:
    atomic_write_json(paths.spec_file(spec.job_id), spec.to_dict())


def read_spec(job_id: str) -> JobSpec:
    return JobSpec.from_dict(read_json(paths.spec_file(job_id)))


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


@dataclass
class HostConfig:
    host: str = "local"
    gpus: list[str] = field(default_factory=list)
    provider: dict[str, Any] | None = None
    idle_minutes: float = 15.0
    ttl_hours: float = 24.0
    s3_prefix: str | None = None
    created_at: str | None = None

    @staticmethod
    def from_dict(d: dict[str, Any]) -> HostConfig:
        return HostConfig(
            host=str(d.get("host", "local")),
            gpus=[str(g) for g in (d.get("gpus") or [])],
            provider=d.get("provider"),
            idle_minutes=float(d.get("idle_minutes", 15.0)),
            ttl_hours=float(d.get("ttl_hours", 24.0)),
            s3_prefix=d.get("s3_prefix"),
            created_at=d.get("created_at"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def ephemeral(self) -> bool:
        return self.provider is not None


def read_config() -> HostConfig:
    path = paths.config_file()
    if not path.exists():
        return HostConfig()
    return HostConfig.from_dict(read_json(path))


def write_config(config: HostConfig) -> None:
    atomic_write_json(paths.config_file(), config.to_dict())
