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

from gpuc.control.providers.base import DEFAULT_IMAGE, Caps, Offer
from gpuc.control.transport import Transport, make_transport
from gpuc.host.jobs import HostConfig

HostKind = Literal["local", "ssh", "runpod"]

DEFAULT_POD_PREFIX = "gpuc-"
DEFAULT_DISK_GB = 50


class ConfigError(RuntimeError):
    pass


class DesiredUnreadable(ConfigError):
    """The reaper's fail-closed signal: we cannot tell which pods are ours."""


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


def pod_known_hosts_file(name: str) -> Path:
    """One known_hosts per ephemeral host.

    RunPod recycles ``host:port`` between pods, so a shared file plus
    StrictHostKeyChecking=accept-new wedges the *second* pod to land on a
    reused endpoint with a host key mismatch.
    """
    return state_dir() / "known_hosts.d" / f"{name}"


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
    """Every key has a working default: gpuc runs with no config file at all.

    Without ``s3_bucket`` there is simply no S3 mirror, so specs, logs and
    state live only on the host they ran on.
    """

    s3_bucket: str | None = None
    runpod_pod_prefix: str = DEFAULT_POD_PREFIX
    max_pods: int = 3
    max_total_usd_per_hour: float = 3.0
    ssh_key: str | None = None
    image: str = DEFAULT_IMAGE
    disk_gb: int = DEFAULT_DISK_GB

    @property
    def ssh_key_path(self) -> str | None:
        return str(Path(self.ssh_key).expanduser()) if self.ssh_key else None

    def caps(self) -> Caps:
        return Caps(
            prefix=self.runpod_pod_prefix,
            max_pods=self.max_pods,
            max_total_usd_per_hour=self.max_total_usd_per_hour,
        )


CONFIG_TEMPLATE = f"""\
# gpu-coordinator settings. Every key here is optional; the values shown are
# the defaults. Delete this file to go back to all of them.

# Where specs, logs and job state are mirrored. Unset means no S3 mirror at
# all, which also means `gpuc requeue` and `gpuc logs` after a pod is gone
# cannot work.
# s3_bucket = "my-experiments"

# Only pods whose name starts with this are ever read, reaped or terminated.
runpod_pod_prefix = "{DEFAULT_POD_PREFIX}"

# Refuse to create a pod that would push us past either cap.
max_pods = 3
max_total_usd_per_hour = 3.0

# Private key for ssh and rsync to hosts and pods; its ".pub" is uploaded to
# the RunPod account. Unset means ssh picks its own.
# ssh_key = "~/.ssh/id_ed25519"

# Defaults for `gpuc submit --runpod`; override per submit with --disk.
image = "{DEFAULT_IMAGE}"
disk_gb = {DEFAULT_DISK_GB}
"""


def write_config_template(*, force: bool = False) -> Path:
    """Write a commented config.toml. Never clobbers an existing one silently."""
    path = config_file()
    if path.exists() and not force:
        raise ConfigError(f"{path} already exists. Edit it, or pass --force to overwrite it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.{os.getpid()}.tmp"
    tmp.write_text(CONFIG_TEMPLATE)
    os.replace(tmp, path)
    return path


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
    persistent_root: str | None = None
    env: dict[str, str] = Field(default_factory=dict)
    """Extra environment for every job on this host, set by hand with
    `gpuc host add|set --env K=V`. Nothing populates it automatically."""
    cache_dir: str | None = None
    """uv's cache for this host, surfaced to jobs as `UV_CACHE_DIR`.

    Unlike `env`, bootstrap *does* populate this: uv materialises a venv by
    reflinking or hardlinking out of its cache, which only works within one
    filesystem, so a host whose gpuc home is on a different volume from `$HOME`
    gets a cache next to gpuc home instead of copying every wheel. `--cache-dir`
    pins it by hand; an explicit `--env UV_CACHE_DIR=...` still wins."""
    idle_minutes: float = 15.0
    ttl_hours: float = 24.0
    s3_prefix: str | None = None
    created_at: str | None = None
    bootstrapped_at: str | None = None

    @property
    def root(self) -> str | None:
        """The persistent root, without a trailing slash."""
        return self.persistent_root.rstrip("/") or "/" if self.persistent_root else None

    @property
    def remote_home(self) -> str:
        """The host's ``$GPUC_HOME``; unexpanded ``$HOME`` so the host resolves it.

        A persistent root moves it off ``$HOME``: on a host whose home is wiped
        on restart, the queue, specs, state, logs and workdirs are the things
        that cannot be reinstalled, so they go on the volume that survives.
        Deliberately *only* those: uv, its caches and the aws bundle stay in
        ``$HOME``, both because bootstrap can reinstall them in seconds and
        because these shared volumes are much slower than the local disk.
        """
        if self.gpuc_home:
            return self.gpuc_home
        root = self.root
        return f"{root}/gpuc" if root else "$HOME/.gpuc"

    @property
    def ephemeral(self) -> bool:
        return self.kind == "runpod"

    def provider(self) -> dict[str, Any] | None:
        if self.kind != "runpod":
            return None
        return {"kind": "runpod", "pod_id": self.pod_id}

    def job_env(self) -> dict[str, str]:
        """The host env as jobs see it: `env`, plus `cache_dir` as a default.

        One source of truth for the dispatcher's config.json, bootstrap's own
        uv calls and every `HostSession` invocation, so they cannot disagree
        about which uv cache this host uses.
        """
        env = dict(self.env)
        if self.cache_dir:
            env.setdefault("UV_CACHE_DIR", self.cache_dir)
        return env

    def host_config(self) -> HostConfig:
        return HostConfig(
            host=self.name,
            gpus=list(self.gpus),
            provider=self.provider(),
            idle_minutes=self.idle_minutes,
            ttl_hours=self.ttl_hours,
            s3_prefix=self.s3_prefix,
            created_at=self.created_at,
            env=self.job_env(),
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


class DesiredHost(BaseModel):
    """What we asked the provider for, written before the pod can be lost.

    This file is the only thing that distinguishes a pod we are waiting on from
    a leaked one, so it is written immediately after `create` returns and
    removed only once the pod is gone.
    """

    name: str
    pod_id: str
    offer: Offer
    created_at: str
    ceiling_at: str
    idle_minutes: float = 15.0
    ttl_hours: float = 24.0
    image: str | None = None
    bootstrapped_at: str | None = None

    @property
    def bootstrapped(self) -> bool:
        return self.bootstrapped_at is not None


def desired_file(name: str) -> Path:
    return desired_dir() / f"{name}.json"


def write_desired(desired: DesiredHost) -> Path:
    directory = desired_dir()
    directory.mkdir(parents=True, exist_ok=True)
    path = desired_file(desired.name)
    tmp = path.parent / f".{path.name}.{os.getpid()}.tmp"
    tmp.write_text(desired.model_dump_json(indent=2) + "\n")
    os.replace(tmp, path)
    return path


def read_desired(name: str) -> DesiredHost | None:
    path = desired_file(name)
    if not path.exists():
        return None
    try:
        return DesiredHost.model_validate_json(path.read_text())
    except (OSError, ValidationError):
        return None


def remove_desired(name: str) -> None:
    desired_file(name).unlink(missing_ok=True)


def forget_host(name: str) -> None:
    """Drop every local trace of one host. The caller must hold the state lock.

    Deliberately not a `registry_transaction`: both callers already hold the
    lock, and flock is per open file description, so re-taking it in the same
    process would deadlock until the timeout.
    """
    remove_desired(name)
    pod_known_hosts_file(name).unlink(missing_ok=True)
    registry = load_registry()
    if registry.hosts.pop(name, None) is not None:
        save_registry(registry)


def load_desired() -> list[DesiredHost]:
    """Every desired host, or raise: a partial answer would reap live pods."""
    directory = desired_dir()
    if not directory.is_dir():
        raise DesiredUnreadable(
            f"{directory} does not exist, so nothing is known about which pods are ours.\n"
            f"That is not the same as `no pods`, so nothing will be terminated. "
            f"It is created by `gpuc submit --runpod`."
        )
    try:
        paths = sorted(directory.glob("*.json"))
    except OSError as exc:
        raise DesiredUnreadable(
            f"cannot list {directory}: {exc}\nFix its permissions; nothing was terminated."
        ) from exc
    hosts: list[DesiredHost] = []
    for path in paths:
        try:
            hosts.append(DesiredHost.model_validate_json(path.read_text()))
        except (OSError, ValidationError) as exc:
            raise DesiredUnreadable(
                f"{path} is not a readable desired-host record: {exc}\n"
                f"Nothing was terminated. Check `gpuc pods`, then fix or delete that file."
            ) from exc
    return hosts


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
        known_hosts=pod_known_hosts_file(entry.name) if entry.kind == "runpod" else None,
    )


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")
