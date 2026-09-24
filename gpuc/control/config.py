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
import time
import tomllib
from collections.abc import Callable, Collection, Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError, model_validator

from gpuc.control.gpuinfo import GpuInfo, table_of
from gpuc.control.jsonout import warn
from gpuc.control.providers.base import DEFAULT_IMAGE, DEFAULT_PREFIX
from gpuc.control.tolerant import TolerantModel
from gpuc.control.transport import Transport, make_transport
from gpuc.host import gpus
from gpuc.host.cleanup import DEFAULT_WORKDIR_DAYS
from gpuc.host.jobs import SCHEMA_VERSION, HostConfig

HostKind = Literal["local", "ssh", "rental"]

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
    """One known_hosts per rental.

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


def default_s3_prefix(settings: Settings, host: str) -> str | None:
    """Where a host mirrors its jobs unless its config names somewhere else."""
    if not settings.s3_bucket:
        return None
    return f"s3://{settings.s3_bucket}/gpuc/{host}"


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
    opens a session, which reads the host's own `config.json` first and
    refreshes this on the way past.
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


class Rental(TolerantModel):
    """The pod behind an address: which provider is billing for it, and as what.

    The one spelling of "this host is rented": `kind` and `pod_id` are read
    off it, and "is this a rental" is `rental is not None`.
    """

    provider: str = "runpod"
    pod_id: str = ""


class HostEntry(TolerantModel):
    """How to reach one host, plus what this machine last saw on it.

    Two kinds of thing, and only two:

    - the **address** -- `ssh`, `port`, `gpuc_home` / `persistent_root`,
      `rental` -- hand-entered, local to this machine, and saying nothing
      about how the host behaves;
    - the **cache**, a copy of what the host said the last time we asked.

    What the host *is* -- its cards, its mirror, its env, its timers -- lives in
    `config.json` on the host and nowhere else, so two control machines driving
    one box cannot each believe their own version of it. `gpuc host add` reads
    that file, `gpuc host set` writes through to it, and bootstrapping a host
    never rewrites the rest of it.
    """

    name: str = ""
    ssh: str | None = None
    port: int = 22
    gpuc_home: str | None = None
    persistent_root: str | None = None
    rental: Rental | None = None
    bootstrapped_at: str | None = None
    """When *this* machine last bootstrapped the host. Another machine's
    bootstrap is invisible here, which is why nothing decides on it: it is a
    label on `gpuc host list` and nothing more."""
    cache: HostCache = Field(default_factory=HostCache)

    @model_validator(mode="before")
    @classmethod
    def _refuse_an_earlier_builds_rental(cls, data: Any) -> Any:
        """An entry with a top-level `pod_id` and no `rental` was written by
        a build that spelled a rental that way. Read tolerantly it would be
        an ssh host -- never terminated, never reused, never forgotten when
        its pod ends -- so it is refused instead, and the registry's usual
        rule for an entry that does not validate (skip it, warn, exit 1)
        prints the way back.
        """
        if isinstance(data, dict) and data.get("pod_id") and data.get("rental") is None:
            document: dict[Any, Any] = data
            raise ValueError(
                f"a rental registered by an earlier build (pod {document['pod_id']}); "
                f"run `gpuc host add <name> --pod {document['pod_id']}` again"
            )
        return data

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
    def kind(self) -> HostKind:
        """What the address says: a rental has a pod behind it, an ssh target
        is a remote box, neither is this machine. Derived, never stored, so it
        cannot disagree with the address."""
        if self.rental is not None:
            return "rental"
        return "ssh" if self.ssh else "local"

    @property
    def pod_id(self) -> str | None:
        return self.rental.pod_id if self.rental is not None else None

    def provider_block(self) -> dict[str, Any] | None:
        """The `provider` block a rental's own config names itself by."""
        if self.rental is None:
            return None
        return {"kind": self.rental.provider, "pod_id": self.rental.pod_id}

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

        For a listing, labelled with `seen_at`; never for a decision. A
        session (`remote.open_session`) reads the host's copy and is what
        anything that acts on a host works from.
        """
        return HostConfig.from_dict(self.cache.config)

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


def first_config(
    entry: HostEntry, settings: Settings, overrides: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """The `config.json` a host that has none is given: the one constructor.

    Used by `gpuc host add` on a box nobody has set up, by provisioning for a
    pod just bought, and by bootstrap restoring a wiped home (with the last
    config seen as `overrides`). By default the host owns every card the probe
    saw (`entry.gpu_info`) less any it was asked to share, sweeps workdirs
    after `DEFAULT_WORKDIR_DAYS`, and mirrors under `default_s3_prefix` when
    there is a bucket -- the same defaults whatever kind of host it is, so
    the machine that connects first has no say the next one cannot see in the
    file.
    """
    overrides = dict(overrides or {})
    shared_entries = [str(item) for item in overrides.get("shared_gpus") or []]
    shared = set(gpus.resolve(shared_entries, table_of(entry.gpu_info)).owned)
    document: dict[str, Any] = {
        "host": entry.name,
        "gpus": [uuid for uuid in entry.gpu_info if uuid not in shared],
        "workdir_days": DEFAULT_WORKDIR_DAYS,
        "s3_prefix": default_s3_prefix(settings, entry.name),
        "created_at": utc_now(),
        "provider": entry.provider_block(),
    }
    return {**document, **overrides}


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
    """The shape of hosts.json. Written always, accepted missing."""
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

    def listing(self) -> list[HostEntry]:
        return list(self.hosts.values())

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

    def named(self) -> Registry:
        """The registry, for a command given a host or job name to find.

        A registry that could not be parsed is exit 3 (unknown), not exit 4
        (does not exist): "no host named gpubox" would be a lie when the file
        holding gpubox is the thing that is broken. Listing commands do not use
        this -- they can honestly show what parsed.
        """
        if self.unreadable:
            raise LocalStateUnreadable("\n".join(self.errors))
        return self.registry

    def require(self, name: str) -> HostEntry:
        return self.named().require(name)


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
    """Parse hosts.json as far as it parses. Never raises on content, never prints.

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


def open_registry() -> RegistryRead:
    """The registry, with each entry it could not parse warned about on stderr.

    The one reader commands use. `.named()` is for a command given a name to
    look up; the rest work with what parsed and report `errors` as their own.
    """
    read = read_registry()
    for error in read.errors:
        warn(error)
    return read


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
            warn(error)
        yield read.registry
        save_registry(read.registry, read.skipped)


def update_cache(
    name: str,
    *,
    config: HostConfig | Mapping[str, Any] | None = None,
    python: str | None = None,
    uv: str | None = None,
    gpu_info: Mapping[str, GpuInfo] | None = None,
    driver_version: str | None = None,
    bootstrapped_at: str | None = None,
) -> HostEntry | None:
    """Record what a host just said about itself: re-read under the lock and
    write only these fields back.

    The one way the cache is written. Re-read rather than written from the
    entry a command started with, because every submit and probe does this
    and writing back a whole entry read before an ssh round trip would undo
    whatever a concurrent `gpuc host probe` learned about the same host in
    between. None is the host that was removed while the command ran.
    """
    with registry_transaction() as registry:
        current = registry.hosts.get(name)
        if current is None:
            return None
        updated = current.with_cache(
            python=python, uv=uv, gpu_info=gpu_info, driver_version=driver_version
        )
        if config is not None:
            updated = updated.with_config(config)
        if bootstrapped_at is not None:
            updated = updated.model_copy(update={"bootstrapped_at": bootstrapped_at})
        registry.put(updated)
        return updated


def forget_host(name: str, pod_id: str | None = None, report: Reporter = warn) -> bool:
    """Drop every local trace of one host, and say whether the entry went.

    False is every reason the registry still lists the host -- it was never
    there, it is another pod's, the file could not be read, the lock is held
    -- because a caller that reports "forgotten" has to be reporting what
    happened rather than what it asked for.

    `pod_id` names the pod the caller is forgetting, and the registry entry is
    only removed if it is that pod's. A pod and a registry entry can disagree
    about what a name means -- a hand-set `host`, a pod that answers to a name
    this machine already uses for a box of its own -- and dropping somebody's
    registered host because a *pod* under that name went away is not something
    this should be able to do.

    The lock is taken for just this mutation, never across the provider and
    ssh calls that decide *whether* to forget: a terminate polls for up to
    five minutes, and every other command gives up on the lock after thirty
    seconds. A lock another session is holding is a warning and a False, not
    a failure: the pod is already gone by the time anything calls this.
    """
    try:
        with state_lock():
            read = read_registry()
            if read.unreadable:
                return False
            entry = read.registry.hosts.get(name)
            if entry is None or (pod_id is not None and entry.pod_id != pod_id):
                return False
            del read.registry.hosts[name]
            save_registry(read.registry, read.skipped)
            # Only once the entry has gone: the pinned host key belongs to the
            # pod the entry names, which a request about another pod leaves.
            pod_known_hosts_file(name).unlink(missing_ok=True)
            return True
    except ConfigError as exc:
        report(f"could not remove host {name} from the registry: {exc}")
        return False


def transport_for(entry: HostEntry, settings: Settings | None = None) -> Transport:
    settings = settings or load_settings()
    ensure_state_dir()
    if entry.ssh is None:
        if entry.rental is not None:
            raise ConfigError(
                f"host {entry.name!r} is a rental with no ssh target recorded.\n"
                f"Re-add it with: gpuc host add {entry.name} --pod {entry.rental.pod_id}"
            )
        return make_transport(entry.name)
    return make_transport(
        entry.name,
        ssh=entry.ssh,
        port=entry.port,
        key=settings.ssh_key_path,
        known_hosts=(
            pod_known_hosts_file(entry.name) if entry.rental is not None else known_hosts_file()
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
