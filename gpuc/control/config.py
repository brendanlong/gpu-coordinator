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
from collections.abc import Callable, Collection, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from gpuc.control.gpuinfo import GpuInfo
from gpuc.control.providers.base import DEFAULT_IMAGE, DEFAULT_PREFIX
from gpuc.control.transport import Transport, make_transport
from gpuc.host.jobs import SCHEMA_VERSION, HostConfig

HostKind = Literal["local", "ssh", "runpod"]

DEFAULT_DISK_GB = 50


Reporter = Callable[[str], None]
"""Where a step's progress goes: `print`, or stderr when stdout is a document."""


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
    rule, one `"retention_days": null` in the shared registry made every subcommand
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
    runpod_pod_prefix: str = DEFAULT_PREFIX
    ssh_key: str | None = None
    image: str = DEFAULT_IMAGE
    disk_gb: int = DEFAULT_DISK_GB

    @property
    def ssh_key_path(self) -> str | None:
        return str(Path(self.ssh_key).expanduser()) if self.ssh_key else None


CONFIG_TEMPLATE = f"""\
# gpu-coordinator settings. Every key here is optional; the values shown are
# the defaults. Delete this file to go back to all of them.

# Where specs, logs and job state are mirrored. Unset means no S3 mirror at
# all, which also means `gpuc requeue` and `gpuc logs` after a pod is gone
# cannot work.
# s3_bucket = "my-experiments"

# Only pods whose name starts with this are ever read or terminated.
runpod_pod_prefix = "{DEFAULT_PREFIX}"

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
        raise LocalStateUnreadable(
            f"{path} is not readable TOML: {exc}\nFix or delete the file."
        ) from exc
    try:
        return Settings.model_validate(document)
    except ValidationError as exc:
        raise LocalStateUnreadable(f"{path} has bad values:\n{exc}") from exc


class HostCache(TolerantModel):
    """What this machine last read off a host, and when it read it.

    Every field here is a copy of something the *host* owns, kept so the
    commands that ask nothing (`gpuc host list`, `gpuc version`) still have
    something to print -- labelled "last seen", because that is what it is.
    Nothing that decides anything reads it: a command that acts on a host
    (submit, bootstrap, set) asks the host, and refreshes this on the way past.
    """

    read_at: str | None = None
    """When this cache was filled, so a listing can say how old it is."""
    python: str | None = None
    uv: str | None = None
    gpu_info: dict[str, GpuInfo] = Field(default_factory=dict)
    """What each UUID on the host is: name and VRAM, from bootstrap or `host probe`.

    Additive and best effort -- a host with no nvidia-smi simply lists its
    UUIDs without names."""
    driver_version: str | None = None
    config: dict[str, Any] = Field(default_factory=dict)
    """The host's `config.json`, verbatim, as last read.

    Verbatim so a key some newer build wrote survives the round trip through
    this build's registry; `HostEntry.config` is the parsed view of it.
    """


# The address/config split landed on 2026-09-16 (518bc9b). A registry written
# before it carries each host's config flat beside its address; `save_registry`
# rewrites the folded shape, so this can go once every control machine has run
# a writing command on a build past that commit.
LEGACY_CACHE_KEYS = ("python", "uv", "gpu_info", "driver_version")
LEGACY_CONFIG_KEYS = (
    "gpus",
    "s3_prefix",
    "env",
    "idle_minutes",
    "retention_days",
    "created_at",
    "pkg_commit",
)


class HostEntry(TolerantModel):
    """How to reach one host, plus what this machine last saw on it.

    Two kinds of thing, and only two:

    - the **address** -- `ssh`, `port`, `gpuc_home` / `persistent_root`,
      `pod_id` -- hand-entered, local to this machine, and saying nothing about
      how the host behaves;
    - the **cache**, a copy of what the host said the last time we asked.

    What the host *is* -- its cards, its mirror, its env, its timers -- lives in
    `config.json` on the host and nowhere else, so two control machines driving
    one box cannot each believe their own version of it. `gpuc host add` reads
    that file, `gpuc host set` writes through to it, and bootstrapping a host
    never rewrites the rest of it.
    """

    name: str = ""
    kind: HostKind = "local"
    ssh: str | None = None
    port: int = 22
    gpuc_home: str | None = None
    persistent_root: str | None = None
    pod_id: str | None = None
    bootstrapped_at: str | None = None
    """When *this* machine last bootstrapped the host. Another machine's
    bootstrap is invisible here, which is why nothing decides on it."""
    cache: HostCache = Field(default_factory=HostCache)

    @model_validator(mode="before")
    @classmethod
    def _fold_pre_split_entry(cls, data: Any) -> Any:
        """Read a registry written before the address and the config were split.

        Such an entry holds a copy of the host's config flat beside the
        address, because the machine that wrote it believed it owned that
        config. It does not, so those fields become the cache, and the next
        connect, `gpuc host set` or bootstrap works from the host's own copy.
        """
        if not isinstance(data, dict):
            return data
        document: dict[Any, Any] = data
        if not any(
            key in document for key in (*LEGACY_CACHE_KEYS, *LEGACY_CONFIG_KEYS, "cache_dir")
        ):
            return document
        cache = dict(document.get("cache") or {})
        config = dict(cache.get("config") or {})
        for key in LEGACY_CACHE_KEYS:
            if key in document and key not in cache:
                cache[key] = document[key]
        for key in LEGACY_CONFIG_KEYS:
            if key in document and key not in config:
                config[key] = document[key]
        # `cache_dir` was the one config key the registry kept outside `env`.
        # On the host it has only ever been `env["UV_CACHE_DIR"]`.
        if document.get("cache_dir"):
            env = dict(config.get("env") or {})
            env.setdefault("UV_CACHE_DIR", str(document["cache_dir"]))
            config["env"] = env
        if document.get("name") and not config.get("host"):
            config["host"] = document["name"]
        cache["config"] = config
        return {**document, "cache": cache}

    # -- the address ---------------------------------------------------------

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
        Deliberately *only* those: uv, its managed Pythons and the aws bundle
        stay in ``$HOME``, both because bootstrap can reinstall them in seconds
        and because these shared volumes are much slower than the local disk.
        uv's *cache* is the exception bootstrap makes (`resolve_cache_dir`),
        and only to keep it on this home's filesystem, where uv can link a venv
        out of it instead of copying.
        """
        if self.gpuc_home:
            return self.gpuc_home
        root = self.root
        return f"{root}/gpuc" if root else "$HOME/.gpuc"

    @property
    def ephemeral(self) -> bool:
        return self.kind == "runpod"

    def provider(self) -> dict[str, Any] | None:
        """The `provider` block this address implies, for a config we initialise."""
        if self.kind != "runpod":
            return None
        return {"kind": "runpod", "pod_id": self.pod_id}

    # -- what the host last said about itself --------------------------------

    @property
    def python(self) -> str | None:
        return self.cache.python

    @property
    def uv(self) -> str | None:
        return self.cache.uv

    @property
    def gpu_info(self) -> dict[str, GpuInfo]:
        return self.cache.gpu_info

    @property
    def driver_version(self) -> str | None:
        return self.cache.driver_version

    @property
    def seen_at(self) -> str | None:
        """When the cache below was last filled. None means never."""
        return self.cache.read_at

    @property
    def config(self) -> HostConfig:
        """The host's own config as this machine last read it.

        Read it for a listing and say how old it is; never act on it without
        asking the host first. Everything that does talk to a host refreshes it
        (`with_config`), so in practice it is one round trip old.
        """
        return HostConfig.from_dict(self.cache.config)

    @property
    def gpus(self) -> list[str]:
        return self.config.gpus

    @property
    def env(self) -> dict[str, str]:
        """Extra environment for every job on this host (`--env K=V`), plus the
        `UV_CACHE_DIR` bootstrap derives from the host's own filesystem."""
        return self.config.env

    @property
    def cache_dir(self) -> str | None:
        """uv's cache for this host, which reaches jobs as `UV_CACHE_DIR`.

        uv materialises a venv by reflinking or hardlinking out of its cache,
        which only works within one filesystem, so a host whose gpuc home is on
        a different volume from `$HOME` gets a cache next to gpuc home instead
        of copying every wheel. Bootstrap fills it in; `--cache-dir` pins it.
        """
        return self.env.get("UV_CACHE_DIR")

    @property
    def s3_prefix(self) -> str | None:
        return self.config.s3_prefix

    @property
    def idle_minutes(self) -> float:
        return self.config.idle_minutes

    @property
    def retention_days(self) -> float | None:
        return self.config.retention_days

    @property
    def workdir_days(self) -> float | None:
        return self.config.workdir_days

    @property
    def created_at(self) -> str | None:
        return self.config.created_at

    @property
    def pkg_commit(self) -> str | None:
        """The gpuc commit the host's config says its package came from.

        Whoever bootstrapped last wrote it, which is the point: this machine's
        own record of what it shipped cannot answer the question."""
        return self.config.pkg_commit

    def initial_config(self) -> HostConfig:
        """The config to give a host that has none of its own.

        Two cases reach it: a host being bootstrapped from a registry written
        before the split (whose cached config is this machine's old record of
        it), and one whose gpuc home was wiped and has to be rebuilt. Both want
        the same thing -- what we last saw, with the facts only this address
        knows filled in.
        """
        config = self.config
        return replace(
            config,
            host=config.host if self.cache.config.get("host") else self.name or config.host,
            provider=config.provider or self.provider(),
            created_at=config.created_at or utc_now(),
        )

    def with_config(
        self, config: HostConfig | Mapping[str, Any], *, read_at: str | None = None
    ) -> HostEntry:
        """A copy whose cache holds `config`, stamped with when it was read."""
        document = config.to_dict() if isinstance(config, HostConfig) else dict(config)
        cache = self.cache.model_copy(update={"config": document, "read_at": read_at or utc_now()})
        return self.model_copy(update={"cache": cache})

    def with_cache(
        self,
        *,
        python: str | None = None,
        uv: str | None = None,
        gpu_info: Mapping[str, GpuInfo] | None = None,
        driver_version: str | None = None,
        read_at: str | None = None,
    ) -> HostEntry:
        """A copy carrying what a probe or a bootstrap just found; None leaves
        a field alone.

        `gpu_info` is merged, not replaced: a probe sees every card in the box
        and a later one may see fewer (a container handed a subset), and a UUID
        we already have a name for is worth keeping.
        """
        changes: dict[str, Any] = {
            key: value
            for key, value in (("python", python), ("uv", uv), ("driver_version", driver_version))
            if value is not None
        }
        if gpu_info is not None:
            changes["gpu_info"] = {**self.cache.gpu_info, **gpu_info}
        changes["read_at"] = read_at or utc_now()
        return self.model_copy(update={"cache": self.cache.model_copy(update=changes)})


NOT_DRIFT = {"schema_version", "pkg_commit", "created_at"}
"""Config keys a difference says nothing about: the shape of the file, the
commit (which moves on every re-ship, and is reported on its own), and when
whoever first registered the host did so."""


def _show(value: Any) -> str:
    if value is None:
        return "none"
    if isinstance(value, list):
        return ",".join(str(item) for item in value) or "none"
    if isinstance(value, dict):
        # `provider` is the one of these, and `{'kind': 'runpod', 'pod_id':
        # 'p1'}` in the middle of a sentence reads as punctuation. `env` never
        # gets here: its values are not printed at all.
        items: dict[Any, Any] = value
        return " ".join(f"{k}={items[k]}" for k in sorted(map(str, items))) or "none"
    return str(value)


def config_drift(
    existing: Any, incoming: HostConfig, keys: Collection[str] | None = None
) -> list[str]:
    """How the config a host holds differs from the one we are about to give it.

    Only the keys `existing` actually has are compared, so this takes a whole
    `config.json` read off the host or a subset of one, and `keys` narrows it
    further for a caller that only cares about some of them. It is how `gpuc
    host add` and `gpuc host set` name, field by field, what a flag is about to
    change on a host somebody already configured.

    `env` reports the names that differ and never the values -- it is
    free-form, it is where somebody hand-sets an HF_TOKEN, and this text ends
    up in transcripts. That holds however odd the host's own `env` is: a
    `config.json` from another build may have anything at all there, including
    a `null`, and formatting our side of the comparison would print the token
    we are trying not to print.
    """
    if not isinstance(existing, dict):
        return []
    drift: list[str] = []
    for key, ours in incoming.to_dict().items():
        if key in NOT_DRIFT or key not in existing or (keys is not None and key not in keys):
            continue
        theirs = existing[key]
        if theirs == ours:
            continue
        if key == "env":
            # A `null` env is the default env, per the tolerant-read rules, so
            # only named differences are differences.
            theirs_env = theirs if isinstance(theirs, dict) else {}
            names = sorted(
                k for k in (set(theirs_env) | set(ours)) if theirs_env.get(k) != ours.get(k)
            )
            if names:
                drift.append(f"env differs in {', '.join(names)}")
        else:
            drift.append(f"{key} {_show(theirs)} -> {_show(ours)}")
    return drift


def config_changes(existing: Any, patch: Mapping[str, Any]) -> list[str]:
    """One line per key `patch` would actually change on a host, as words.

    What `gpuc host add` and `gpuc host set` print: the host's config is the
    only copy of it, so a flag that touches it is an edit of somebody's host
    and says so, field by field, rather than being applied in silence.
    """
    fields = existing if isinstance(existing, dict) else {}
    # Compared against what the host *effectively* has, defaults included, so
    # a key it has never written does not read as a change to its own default.
    held = HostConfig.from_dict(fields).to_dict()
    base = {key: held.get(key) for key in patch}
    return config_drift(base, HostConfig.from_dict({**fields, **patch}), keys=set(patch))


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
                f"Add it with: gpuc host add {name} --ssh user@host"
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


def load_registry() -> Registry:
    """The salvaged registry, with every problem reported and none of them fatal.

    Callers that need to tell "nothing registered" from "nothing readable" use
    `read_registry()` instead; everything else can work with what parsed.
    """
    read = read_registry()
    for error in read.errors:
        warn_stderr(error)
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
                # Sleeping is free next to the ssh round trip the lock protects;
                # spinning on sched_yield burns a core for the whole wait and
                # slows the holder down.
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


def forget_host(name: str, pod_id: str | None = None) -> None:
    """Drop every local trace of one host. The caller must hold the state lock.

    `pod_id` names the pod the caller is forgetting, and the registry entry is
    only removed if it is that pod's. A pod and a registry entry can disagree
    about what a name means -- a hand-set `host`, a pod that answers to a name
    this machine already uses for a box of its own -- and dropping somebody's
    registered host because a *pod* under that name went away is not something
    this should be able to do.

    Deliberately not a `registry_transaction`: both callers already hold the
    lock, and flock is per open file description, so re-taking it in the same
    process would deadlock until the timeout.
    """
    pod_known_hosts_file(name).unlink(missing_ok=True)
    read = read_registry()
    if read.unreadable:
        return
    entry = read.registry.hosts.get(name)
    if entry is None:
        return
    if pod_id is not None and entry.pod_id != pod_id:
        # Not this pod's entry -- a box of this machine's that answers to the
        # same name, or another pod under it.
        return
    del read.registry.hosts[name]
    save_registry(read.registry, read.skipped)


def forget_host_locked(name: str, pod_id: str | None, report: Reporter) -> None:
    """`forget_host` under the state lock, taken for just that mutation.

    Never held across the provider and ssh calls that decide *whether* to
    forget: a terminate polls for up to five minutes, and every other command
    that touches the registry gives up on the lock after thirty seconds.
    """
    try:
        with state_lock():
            forget_host(name, pod_id)
    except ConfigError as exc:
        report(f"WARNING: could not remove host {name} from the registry: {exc}")


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
        known_hosts=(
            pod_known_hosts_file(entry.name) if entry.kind == "runpod" else known_hosts_file()
        ),
    )


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def parse_timestamp(stamp: str | None) -> datetime | None:
    """An ISO stamp from any file the two halves share, as an aware datetime, or None."""
    if not stamp:
        return None
    try:
        parsed = datetime.fromisoformat(stamp)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
