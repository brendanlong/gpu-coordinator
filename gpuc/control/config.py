"""Local settings, the host registry, and the state-directory lock.

Two directories, both XDG-overridable so tests never touch the real ones:
``~/.config/gpu-coordinator`` (hand-edited settings) and
``~/.local/share/gpu-coordinator`` (state this tool owns).
"""

from __future__ import annotations

import fcntl
import os
import tomllib
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from gpuc.control.transport import Transport, make_transport
from gpuc.host.jobs import HostConfig

HostKind = Literal["local", "ssh", "runpod"]

DEFAULT_POD_PREFIX = "gpuc-"


class ConfigError(RuntimeError):
    pass


def config_dir() -> Path:
    override = os.environ.get("GPUC_CONFIG_DIR")
    if override:
        return Path(override).expanduser()
    return Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "gpu-coordinator"


def state_dir() -> Path:
    override = os.environ.get("GPUC_STATE_DIR")
    if override:
        return Path(override).expanduser()
    return Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share")) / "gpu-coordinator"


def config_file() -> Path:
    return config_dir() / "config.toml"


def hosts_file() -> Path:
    return state_dir() / "hosts.json"


def known_hosts_file() -> Path:
    return state_dir() / "known_hosts"


def lock_file() -> Path:
    return state_dir() / "state.lock"


def desired_dir() -> Path:
    return state_dir() / "desired"


def index_dir() -> Path:
    return state_dir() / "jobs"


def ensure_state_dir() -> Path:
    directory = state_dir()
    directory.mkdir(parents=True, exist_ok=True)
    return directory


class Settings(BaseModel):
    s3_bucket: str | None = None
    runpod_pod_prefix: str = DEFAULT_POD_PREFIX
    max_pods: int = 3
    max_total_usd_per_hour: float = 3.0
    ssh_key: str | None = None

    @property
    def ssh_key_path(self) -> str | None:
        return str(Path(self.ssh_key).expanduser()) if self.ssh_key else None


def load_settings() -> Settings:
    path = config_file()
    if not path.exists():
        return Settings()
    try:
        document = tomllib.loads(path.read_text())
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ConfigError(f"{path} is not readable TOML: {exc}\nFix or delete the file.") from exc
    try:
        return Settings.model_validate(document)
    except ValidationError as exc:
        raise ConfigError(f"{path} has bad values:\n{exc}") from exc


class HostEntry(BaseModel):
    name: str
    kind: HostKind = "local"
    ssh: str | None = None
    port: int = 22
    gpus: list[str] = Field(default_factory=list)
    pod_id: str | None = None
    python: str | None = None
    uv: str | None = None
    gpuc_home: str | None = None
    idle_minutes: float = 15.0
    ttl_hours: float = 24.0
    s3_prefix: str | None = None
    created_at: str | None = None
    bootstrapped_at: str | None = None

    @property
    def remote_home(self) -> str:
        """The host's ``$GPUC_HOME``; unexpanded ``$HOME`` so the host resolves it."""
        return self.gpuc_home or "$HOME/.gpuc"

    @property
    def ephemeral(self) -> bool:
        return self.kind == "runpod"

    def provider(self) -> dict[str, Any] | None:
        if self.kind != "runpod":
            return None
        return {"kind": "runpod", "pod_id": self.pod_id}

    def host_config(self) -> HostConfig:
        return HostConfig(
            host=self.name,
            gpus=list(self.gpus),
            provider=self.provider(),
            idle_minutes=self.idle_minutes,
            ttl_hours=self.ttl_hours,
            s3_prefix=self.s3_prefix,
            created_at=self.created_at,
        )


class Registry(BaseModel):
    hosts: dict[str, HostEntry] = Field(default_factory=dict)

    def require(self, name: str) -> HostEntry:
        entry = self.hosts.get(name)
        if entry is None:
            known = ", ".join(sorted(self.hosts)) or "(none)"
            raise ConfigError(
                f"no host named {name!r}. Known hosts: {known}.\n"
                f"Add it with: gpuc host add {name} --ssh user@host --gpus GPU-uuid"
            )
        return entry

    def put(self, entry: HostEntry) -> None:
        self.hosts[entry.name] = entry


def load_registry() -> Registry:
    path = hosts_file()
    if not path.exists():
        return Registry()
    try:
        return Registry.model_validate_json(path.read_text())
    except (OSError, ValidationError) as exc:
        raise ConfigError(
            f"{path} is not a readable host registry: {exc}\n"
            f"Fix it by hand, or remove it and re-add your hosts with `gpuc host add`."
        ) from exc


def save_registry(registry: Registry) -> None:
    ensure_state_dir()
    path = hosts_file()
    tmp = path.parent / f".{path.name}.{os.getpid()}.tmp"
    tmp.write_text(registry.model_dump_json(indent=2) + "\n")
    os.replace(tmp, path)


@contextmanager
def state_lock(timeout_s: float = 30.0) -> Iterator[None]:
    """Serialise registry read-modify-write across concurrent agent sessions."""
    ensure_state_dir()
    fd = os.open(lock_file(), os.O_RDWR | os.O_CREAT, 0o644)
    deadline = datetime.now(UTC).timestamp() + timeout_s
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as exc:
                if datetime.now(UTC).timestamp() >= deadline:
                    raise ConfigError(
                        f"another gpuc process has held {lock_file()} for more than "
                        f"{timeout_s:.0f}s. Wait for it, or delete the file if nothing is running."
                    ) from exc
                os.sched_yield()
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


@contextmanager
def registry_transaction() -> Iterator[Registry]:
    with state_lock():
        registry = load_registry()
        yield registry
        save_registry(registry)


def transport_for(entry: HostEntry, settings: Settings | None = None) -> Transport:
    settings = settings or load_settings()
    ensure_state_dir()
    if entry.kind == "local":
        return make_transport(entry.name)
    if not entry.ssh:
        raise ConfigError(
            f"host {entry.name!r} has kind {entry.kind!r} but no ssh target.\n"
            f"Re-add it with: gpuc host add {entry.name} --ssh user@host --gpus ..."
        )
    return make_transport(
        entry.name,
        ssh=entry.ssh,
        port=entry.port,
        key=settings.ssh_key_path,
        state_dir=state_dir(),
    )


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")
