"""Local settings, the host registry, and the state-directory lock.

Two directories, both XDG-overridable so tests never touch the real ones:
``~/.config/gpu-coordinator`` (hand-edited settings) and
``~/.local/share/gpu-coordinator`` (state this tool owns).
"""

from __future__ import annotations

import fcntl
import json
import os
import shutil
import sys
import time
import tomllib
import types
import typing
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from gpuc.control.gpuinfo import GpuInfo
from gpuc.control.providers.base import DEFAULT_IMAGE, Caps, Offer
from gpuc.control.transport import Transport, make_transport
from gpuc.host.jobs import SCHEMA_VERSION, HostConfig

HostKind = Literal["local", "ssh", "runpod"]

DEFAULT_POD_PREFIX = "gpuc-"
DEFAULT_DISK_GB = 50


class ConfigError(RuntimeError):
    pass


class LocalStateUnreadable(ConfigError):
    """Local state could not be read at all, so we know nothing rather than nothing-is-there.

    Its own class because the two are not the same answer: automation that
    reads "no hosts" as "no jobs running" is exactly how a broken registry
    turned into a wrong answer instead of an error.
    """


class HostNotFound(ConfigError):
    """A named host is not in the registry (CLI exit 4, not a generic failure)."""


class DesiredUnreadable(ConfigError):
    """The reaper's fail-closed signal: we cannot tell which pods are ours."""


def _allows_none(annotation: Any) -> bool:
    if annotation is None or annotation is type(None) or annotation is Any:
        return True
    if typing.get_origin(annotation) in (typing.Union, types.UnionType):
        return any(_allows_none(arg) for arg in typing.get_args(annotation))
    return False


class TolerantModel(BaseModel):
    """Read shared state the way Postel would: every field optional, nulls inert.

    Every model here parses a file that another version of gpuc -- an older
    build still installed in a second session, a newer one from `uv tool
    upgrade` -- may have written. Two rules make that safe in both directions:
    an unknown key is ignored (a newer writer may add fields), and an explicit
    `null` for a field that is not nullable is dropped so the field's default
    applies (a newer writer may make a field optional). Without the second
    rule, one `"ttl_hours": null` in the shared registry made every subcommand
    of the other session -- including `status` and `logs` -- fail validation.
    """

    model_config = ConfigDict(extra="ignore")

    @model_validator(mode="before")
    @classmethod
    def _null_means_default(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        document: dict[Any, Any] = data
        drop = [
            key
            for key, value in document.items()
            if value is None
            and isinstance(key, str)
            and key in cls.model_fields
            and not _allows_none(cls.model_fields[key].annotation)
        ]
        if not drop:
            return document
        return {key: value for key, value in document.items() if key not in drop}


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


class Settings(TolerantModel):
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
    dead_dispatcher_minutes: float = 30.0
    """How long an ephemeral host may be silent before the reaper terminates it.

    With no overall TTL, this is what stops a pod nobody is watching: a
    dispatcher that has not beaten -- or a pod that has not answered ssh -- for
    this long, with nothing running, is billing for nothing."""

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

# An ephemeral host whose dispatcher has not beaten (or whose ssh has not
# answered) for this long, with nothing running, is terminated by
# `gpuc reconcile`. There is no overall TTL unless you pass --ttl-hours.
dead_dispatcher_minutes = 30.0

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
        raise LocalStateUnreadable(
            f"{path} is not readable TOML: {exc}\nFix or delete the file."
        ) from exc
    try:
        return Settings.model_validate(document)
    except ValidationError as exc:
        raise LocalStateUnreadable(f"{path} has bad values:\n{exc}") from exc


class HostEntry(TolerantModel):
    name: str = ""
    kind: HostKind = "local"
    ssh: str | None = None
    port: int = 22
    gpus: list[str] = Field(default_factory=list)
    gpu_info: dict[str, GpuInfo] = Field(default_factory=dict)
    """What each UUID is: name and VRAM, recorded by bootstrap and `host probe`.

    Additive and best effort -- a host registered before this existed, or one
    with no nvidia-smi, simply lists its UUIDs without names."""
    driver_version: str | None = None
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
    ttl_hours: float | None = None
    """Hard cap on this host's life, in hours; None (the default) never expires.

    An opt-in cap, not a safety net: killing a training run at hour 24 is worse
    than the idle timer taking a little longer. The reaper's safety net is
    `Settings.dead_dispatcher_minutes` instead."""
    s3_prefix: str | None = None
    retention_days: float | None = None
    """Auto-purge horizon for this host, in days; None never auto-purges.

    Only ever acts on jobs whose log and state are confirmed mirrored, so a
    host with no `s3_prefix` (and no `s3_bucket` to derive one from) can set
    this and nothing will ever be deleted."""
    created_at: str | None = None
    bootstrapped_at: str | None = None
    pkg_commit: str | None = None
    """The gpuc commit bootstrap last shipped to this host, as `gpuc version`
    and `gpuc host list` report it. Null means "bootstrapped before this was
    recorded", which is not the same as "up to date"."""

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
            retention_days=self.retention_days,
            env=self.job_env(),
            pkg_commit=self.pkg_commit,
        )


class Registry(TolerantModel):
    schema_version: int = SCHEMA_VERSION
    """The shape of hosts.json. Written always, accepted missing: a registry
    from before it existed is version 1 by definition."""
    hosts: dict[str, HostEntry] = Field(default_factory=dict)

    def require(self, name: str) -> HostEntry:
        entry = self.hosts.get(name)
        if entry is None:
            known = ", ".join(sorted(self.hosts)) or "(none)"
            raise HostNotFound(
                f"no host named {name!r}. Known hosts: {known}.\n"
                f"Add it with: gpuc host add {name} --ssh user@host --gpus GPU-uuid"
            )
        return entry

    def put(self, entry: HostEntry) -> None:
        self.hosts[entry.name] = entry


class RegistryRead(BaseModel):
    """What one read of hosts.json produced, including what it could not read."""

    registry: Registry = Field(default_factory=Registry)
    errors: list[str] = Field(default_factory=list)
    unreadable: bool = False
    """The file itself could not be parsed, so `registry` is empty because we
    know nothing -- not because there are no hosts. Commands say so and exit 3
    rather than reporting an empty world."""
    skipped: dict[str, Any] = Field(default_factory=dict)
    """Host entries that did not validate, kept verbatim so a later write puts
    them back: they are another session's hosts, not ours to delete."""


def backup_path(path: Path) -> Path:
    return path.with_name(path.name + ".bak")


def _take_backup(path: Path) -> Path | None:
    """Copy a file we cannot parse aside, before anything here rewrites it."""
    target = backup_path(path)
    try:
        shutil.copy2(path, target)
    except OSError:
        return None
    return target


def read_registry() -> RegistryRead:
    """Parse hosts.json as far as it parses. Never raises on content.

    One unreadable host entry must not take the CLI with it: `status` and
    `logs` on the other hosts are exactly what someone needs while they fix it.
    So the file is parsed twice -- once as a document, then one host at a time
    -- and only the entries that fail are dropped, each with a line saying so.
    """
    path = hosts_file()
    if not path.exists():
        return RegistryRead()
    try:
        document = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        saved = _take_backup(path)
        return RegistryRead(
            unreadable=True,
            errors=[
                f"{path} is not a readable host registry: {exc}\n"
                f"  a copy is kept at {saved or backup_path(path)} before anything rewrites it\n"
                f"  fix it by hand, or remove it and re-add your hosts with `gpuc host add`"
            ],
        )
    if not isinstance(document, dict):
        saved = _take_backup(path)
        return RegistryRead(
            unreadable=True,
            errors=[
                f"{path} holds {type(document).__name__}, not a host registry object\n"
                f"  a copy is kept at {saved or backup_path(path)} before anything rewrites it"
            ],
        )
    hosts = document.get("hosts")
    if not isinstance(hosts, dict):
        saved = _take_backup(path)
        return RegistryRead(
            unreadable=True,
            errors=[
                f"{path} has no `hosts` object, so no host is known\n"
                f"  a copy is kept at {saved or backup_path(path)} before anything rewrites it"
            ],
        )
    registry = Registry.model_validate({**document, "hosts": {}})
    errors: list[str] = []
    skipped: dict[str, Any] = {}
    for name, raw in sorted(hosts.items()):
        try:
            entry = HostEntry.model_validate(raw)
        except ValidationError as exc:
            skipped[str(name)] = raw
            errors.append(
                f"skipping host {name!r} in {path}: {exc}\n"
                f"  every other host still works; fix that entry or re-add it with "
                f"`gpuc host add {name} ...`"
            )
            continue
        registry.hosts[str(name)] = (
            entry if entry.name else entry.model_copy(update={"name": str(name)})
        )
    return RegistryRead(registry=registry, errors=errors, skipped=skipped)


def warn_stderr(message: str) -> None:
    print(f"warning: {message}", file=sys.stderr)


def load_registry(warn: Callable[[str], None] = warn_stderr) -> Registry:
    """The salvaged registry, with every problem reported and none of them fatal.

    Callers that need to tell "nothing registered" from "nothing readable" use
    `read_registry()` instead; everything else can work with what parsed.
    """
    read = read_registry()
    for error in read.errors:
        warn(error)
    return read.registry


def save_registry(registry: Registry, keep: dict[str, Any] | None = None) -> None:
    """Write hosts.json, putting back any entry this build could not parse.

    `keep` is what `read_registry` skipped. Those entries belong to whoever
    wrote them -- very likely another session on a different build -- and
    dropping them on the first `gpuc host set` would turn one bad entry into
    someone else's missing host.
    """
    ensure_state_dir()
    path = hosts_file()
    document = registry.model_dump(mode="json")
    hosts = document.setdefault("hosts", {})
    for name, raw in (keep or {}).items():
        hosts.setdefault(name, raw)
    tmp = path.parent / f".{path.name}.{os.getpid()}.tmp"
    tmp.write_text(json.dumps(document, indent=2) + "\n")
    os.replace(tmp, path)


LOCK_POLL_S = 0.05
"""How often a blocked `state_lock` retries. Short enough to be invisible next
to the ssh round trip the lock protects, long enough not to spin."""


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
                # Waits here are measured in seconds (a provision holds the lock
                # across a create), so sleeping is free; spinning on sched_yield
                # burns a core for the whole wait and slows the holder down.
                time.sleep(LOCK_POLL_S)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


@contextmanager
def registry_transaction() -> Iterator[Registry]:
    """Read-modify-write under the lock. Refuses to write over a registry we
    could not read at all, because that write would delete every host in it."""
    with state_lock():
        read = read_registry()
        if read.unreadable:
            raise LocalStateUnreadable(
                "\n".join(read.errors)
                + f"\nNothing was written: replacing {hosts_file()} would drop every host it "
                f"still holds."
            )
        for error in read.errors:
            warn_stderr(error)
        yield read.registry
        save_registry(read.registry, read.skipped)


class DesiredHost(TolerantModel):
    """What we asked the provider for, written before the pod can be lost.

    This file is the only thing that distinguishes a pod we are waiting on from
    a leaked one, so it is written immediately after `create` returns and
    removed only once the pod is gone.
    """

    name: str = ""
    pod_id: str = ""
    offer: Offer = Field(default_factory=Offer)
    created_at: str = ""
    ceiling_at: str = ""
    idle_minutes: float = 15.0
    ttl_hours: float | None = None
    image: str | None = None
    bootstrapped_at: str | None = None
    last_seen_at: str | None = None
    """When this host last proved it was alive: a fresh dispatcher heartbeat, or
    a job running on it. The reaper terminates a pod that has not managed either
    for `dead_dispatcher_minutes`, which is also how an unreachable pod is
    caught -- an ssh that never answers never updates this."""

    @property
    def bootstrapped(self) -> bool:
        return self.bootstrapped_at is not None

    def silent_since(self) -> str | None:
        """The most recent moment we know this host was alive."""
        return self.last_seen_at or self.bootstrapped_at or self.created_at


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
    read = read_registry()
    if read.unreadable:
        return
    if read.registry.hosts.pop(name, None) is not None:
        save_registry(read.registry, read.skipped)


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
