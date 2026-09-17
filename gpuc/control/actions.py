"""What `gpuc` and the web dashboard both do.

Each function here does one thing a command does and returns the document its
`--json` form prints. The CLI renders that document as text or JSON; the
dashboard serves it over HTTP. Neither adds judgement of its own, so a job the
CLI would refuse to cancel is one the dashboard refuses too, with the same
words.
"""

from __future__ import annotations

import json
import shlex
import sys
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from gpuc.control import status as status_mod
from gpuc.control import version as version_mod
from gpuc.control.bootstrap import BootstrapError
from gpuc.control.clean import CleanError
from gpuc.control.config import (
    ConfigError,
    HostEntry,
    HostNotFound,
    LocalStateUnreadable,
    Registry,
    RegistryRead,
    Settings,
    config_file,
    hosts_file,
    read_registry,
    registry_transaction,
    state_dir,
    write_config_template,
)
from gpuc.control.connect import Connection
from gpuc.control.providers.base import Provider, ProviderError
from gpuc.control.providers.runpod import RunPodProvider
from gpuc.control.provision import ProvisionError
from gpuc.control.remote import HostSession, RemoteError, open_session
from gpuc.control.s3index import (
    IndexEntry,
    LocalIndex,
    S3Index,
    S3IndexError,
    S3ObjectMissing,
    job_log_uri,
)
from gpuc.control.skill import SkillError
from gpuc.control.submit import SubmitError
from gpuc.control.transport import TransportError
from gpuc.host import jobs

EXIT_OK = 0
"""Everything the command was asked to do happened, including reporting that a
host is unreachable: that is data about a host, not a failure of the command."""
EXIT_ERROR = 1
"""The command failed: a transport error, a provider error, a refused submit."""
EXIT_USAGE = 2
"""The command line itself was wrong (argparse uses this too)."""
EXIT_LOCAL_STATE = 3
"""Local state -- the registry or the config file -- could not be read, so the
answer is unknown. Automation must not read this as `nothing is running`."""
EXIT_NOT_FOUND = 4
"""The named job or host does not exist."""


class CliError(RuntimeError):
    exit_code = EXIT_ERROR


class UsageError(CliError):
    """The invocation was wrong, not the world."""

    exit_code = EXIT_USAGE


class NotFound(CliError):
    """The job or host named on the command line does not exist."""

    exit_code = EXIT_NOT_FOUND


FAILURES = (
    CleanError,
    CliError,
    ConfigError,
    SubmitError,
    BootstrapError,
    ProvisionError,
    ProviderError,
    RemoteError,
    S3IndexError,
    SkillError,
    TransportError,
)
"""Every error a command reports as its own failure rather than a traceback."""


def exit_code_for(exc: BaseException) -> int | None:
    """The exit code a failure maps to, or None for one that is a real bug.

    One table for the CLI's top-level handler and the dashboard's, so a refused
    job is exit 4 on the command line and 404 on the wire, never two opinions.
    """
    if isinstance(exc, LocalStateUnreadable):
        return EXIT_LOCAL_STATE
    if isinstance(exc, HostNotFound):
        return EXIT_NOT_FOUND
    if isinstance(exc, FAILURES):
        return getattr(exc, "exit_code", EXIT_ERROR)
    if isinstance(exc, json.JSONDecodeError):
        return EXIT_ERROR
    return None


def failure_message(exc: BaseException) -> str:
    if isinstance(exc, json.JSONDecodeError):
        return f"a host returned malformed JSON: {exc}"
    return str(exc)


def warn(message: str) -> None:
    print(f"warning: {message}", file=sys.stderr)


def note(message: str) -> None:
    print(f"note: {message}", file=sys.stderr)


def read_registry_warned() -> RegistryRead:
    """The registry, with each entry it could not parse warned about on stderr."""
    read = read_registry()
    for error in read.errors:
        warn(error)
    return read


def named_registry() -> Registry:
    """The registry, for a command that was given a host or job name to find.

    A registry that could not be parsed is exit 3 (unknown), not exit 4 (does
    not exist): "no host named gpubox" would be a lie when the file holding
    gpubox is the thing that is broken. Listing commands do not use this --
    they can honestly show what parsed.
    """
    read = read_registry_warned()
    if read.unreadable:
        raise LocalStateUnreadable("\n".join(read.errors))
    return read.registry


def make_provider(settings: Settings) -> Provider:
    return RunPodProvider(prefix=settings.runpod_pod_prefix)


def provider_for_status(
    entries: Sequence[HostEntry], settings: Settings, report: Callable[[str], None] = note
) -> Provider | None:
    """Only build a provider when an ephemeral host is on screen, and never fail on it."""
    if not any(entry.kind == "runpod" for entry in entries):
        return None
    try:
        return make_provider(settings)
    except ProviderError as exc:
        report(f"pod status unavailable: {exc}")
        return None


MAX_PARALLEL_HOSTS = 8


def gather_all(
    entries: Sequence[HostEntry], settings: Settings, provider: Provider | None
) -> Iterator[status_mod.HostView]:
    """Every host's status, asked for at once and yielded in registry order.

    One ssh round trip per host, and a wedged host takes its whole timeout to
    say so; asked one after another that is a dashboard that takes a minute to
    draw when one box is down. Yielded rather than collected so the text
    `gpuc status` still prints each host as soon as it, and every host before
    it, has answered.
    """
    if not entries:
        return
    with ThreadPoolExecutor(max_workers=min(MAX_PARALLEL_HOSTS, len(entries))) as pool:
        yield from pool.map(
            lambda entry: status_mod.gather(entry, settings, provider=provider), entries
        )


def hosts_for(registry: Registry, only: str | None) -> list[HostEntry]:
    if only:
        return [registry.require(only)]
    return list(registry.hosts.values())


def status_document(
    read: RegistryRead,
    settings: Settings,
    *,
    host: str | None = None,
    recent: int = status_mod.RECENT_FINISHED,
    since_s: float | None = None,
    report: Callable[[str], None] = note,
) -> dict[str, Any]:
    """`gpuc status --json`: one object, whatever happened.

    The registry's own errors ride along as top-level `errors`, and an
    unreadable registry is an empty `hosts` *with* an error saying so -- the
    caller decides whether that is exit 3 or HTTP 503, but the document is the
    same either way.
    """
    entries = hosts_for(read.registry, host) if not read.unreadable else []
    provider = provider_for_status(entries, settings, report) if entries else None
    views = list(gather_all(entries, settings, provider))
    errors = list(read.errors)
    if read.unreadable:
        errors.append(
            f"{hosts_file()} could not be read, so `hosts` is empty because nothing is known"
        )
    return status_mod.document(views, errors=errors, recent=recent, since_s=since_s)


def shipped_note(entry: HostEntry) -> str | None:
    """The offline half of the version story: what the host last said it ran.

    `gpuc host list` and `gpuc version` never ask a host anything, so the
    cached commit is all they have. What the host is *running* now is `gpuc
    status`, which asks it.
    """
    return version_mod.shipped_commit_note(entry.name, entry.pkg_commit, version_mod.local_commit())


def host_document(entry: HostEntry) -> dict[str, Any]:
    """One registered host as `gpuc host list --json` reports it.

    The address, the host's config as this machine last read it, and what the
    text listing computes from them. This command never asks the host
    anything, so everything out of that config is as of `config_seen_at` --
    `gpuc status` is what asks -- and it is flattened beside the address
    because that is the shape every consumer of this document already reads.
    """
    document: dict[str, Any] = json.loads(entry.model_dump_json())
    config = entry.config.to_dict()
    # The host file's own version says nothing about this document's shape, and
    # beside the address it reads as if it did.
    config.pop("schema_version", None)
    # `--env` is free-form and is where somebody hand-sets an HF_TOKEN, so the
    # names are reported and the values are not -- in the flattened copy and in
    # the cache it came from. The text listing shows neither, and this document
    # ends up in transcripts and bug reports.
    names = dict.fromkeys(entry.env, "<set>")
    cache: dict[str, Any] = document.get("cache") or {}
    cache["config"] = {**(cache.get("config") or {}), "env": names}
    stale = shipped_note(entry)
    return {
        **document,
        **config,
        "env": names,
        "cache": cache,
        "cache_dir": entry.cache_dir,
        "config_seen_at": entry.seen_at,
        "remote_home": entry.remote_home,
        "ephemeral": entry.ephemeral,
        "warnings": [stale] if stale else [],
    }


def hosts_document(read: RegistryRead) -> dict[str, Any]:
    return {
        "hosts": [host_document(entry) for entry in read.registry.hosts.values()],
        "errors": list(read.errors),
    }


def connection_document(
    entry: HostEntry, connection: Connection, *, warnings: Sequence[str] = ()
) -> dict[str, Any]:
    """`gpuc host add --json` and `gpuc host set --json`: the host as it now is.

    The same shape as one entry of `host list --json`, since that is what the
    registry now holds, plus what this command did to get there: `adopted` is
    whether the host already had a config, `config_path` where that config
    lives on the host, and `changes` the per-field lines the text output prints
    for what was written through to it. `warnings` carries what the text output
    says beside the result -- a host that owns no card, a pod nothing has
    bootstrapped -- on top of the re-bootstrap note `host list` gives.
    """
    document = host_document(entry)
    return {
        **document,
        "adopted": connection.adopted,
        "config_path": f"{connection.home}/config.json",
        "changes": list(connection.changes),
        "warnings": [*document["warnings"], *warnings],
    }


def remove_host(name: str) -> dict[str, Any]:
    """Forget a host here. Nothing on the host, or at its provider, changes.

    A rental in particular is not terminated: after handoff it ends itself
    when idle, and nothing on a client watches it. The document says so for an
    ephemeral host, so a caller does not take "removed" for "stopped billing".
    """
    with registry_transaction() as registry:
        entry = registry.require(name)
        del registry.hosts[name]
    notes = []
    if entry.ephemeral:
        pod = f"pod {entry.pod_id}" if entry.pod_id else "pod"
        notes.append(
            f"its {pod} is not terminated by this: it ends itself once its queue has been "
            f"idle, and `gpuc pods` shows it until then"
        )
    return {"host": entry.name, "kind": entry.kind, "pod_id": entry.pod_id, "notes": notes}


def init_config(*, force: bool) -> dict[str, Any]:
    """`gpuc config init`: write the commented settings file, and say where."""
    path = config_file()
    existed = path.exists()
    write_config_template(force=force)
    return {"config_file": str(path), "existed": existed}


def config_document(settings: Settings) -> dict[str, Any]:
    """The effective settings and where they came from: `gpuc config show`."""
    path = config_file()
    notes = ["no s3_bucket, so nothing is mirrored to S3"] if settings.s3_bucket is None else []
    return {
        "config_file": str(path),
        "config_file_exists": path.exists(),
        "state_dir": str(state_dir()),
        "settings": settings.model_dump(mode="json"),
        "notes": notes,
    }


def version_document(read: RegistryRead) -> dict[str, Any]:
    """`gpuc version --json`: this build, and what each host was last seen on.

    `hosts[].pkg_commit` is the commit a host's own config named the last time
    anything here read it, `seen_at` says when that was, and `current` compares
    it with this build -- the same judgement the text output prints as
    `DIFFERS: re-bootstrap`. Neither asks the host now: what it is running this
    minute is `gpuc status --json`'s `pkg_commit`. Nothing recorded on either
    side is not evidence of a mismatch, so it reads as current -- `submit`
    checks the host itself before it enqueues anyway.
    """
    commit = version_mod.local_commit()
    # Every host anything here has read, not only the ones *this* machine
    # bootstrapped: adopting a host is the ordinary way to register one.
    hosts = [
        entry
        for entry in read.registry.hosts.values()
        if entry.pkg_commit or entry.bootstrapped_at or entry.seen_at
    ]
    return {
        "version": version_mod.__version__,
        "commit": commit,
        "source": "installed" if version_mod.installed_commit() else "source checkout",
        "dirty": version_mod.dirty(),
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "hosts": [
            {
                "name": entry.name,
                "pkg_commit": entry.pkg_commit,
                "seen_at": entry.seen_at,
                "current": version_mod.same_commit(commit, entry.pkg_commit),
            }
            for entry in hosts
        ],
        "errors": list(read.errors),
    }


def find_job_host(
    job_id: str, registry: Registry, explicit: str | None
) -> tuple[HostEntry, IndexEntry | None]:
    index = LocalIndex().get(job_id)
    if explicit:
        return registry.require(explicit), index
    if index is not None and index.host in registry.hosts:
        return registry.hosts[index.host], index
    for entry in registry.hosts.values():
        try:
            payload = open_session(entry).host_json(f"status {shlex.quote(job_id)}", timeout=60.0)
        except (RemoteError, TransportError):
            continue
        if payload.get("jobs"):
            return entry, index
    raise NotFound(
        f"no registered host knows job {job_id}.\n"
        f"Pass --host <name>, or check `gpuc host list` and `gpuc status --all`."
    )


def cancel_job(job_id: str, host: str | None, settings: Settings) -> dict[str, Any]:
    entry, _ = find_job_host(job_id, named_registry(), host)
    payload = open_session(entry, settings).host_json(f"cancel {shlex.quote(job_id)}")
    # The host's own word for what it did: `cancelled` for a queued job it
    # dequeued, `cancelling` for a running one whose runner has been marked.
    return {"job_id": job_id, "host": entry.name, "status": payload.get("status")}


def check_priority(priority: int) -> None:
    if not 0 <= priority <= 99:
        raise UsageError(f"priority must be 0-99 (lower dispatches first), got {priority}")


def host_answer(
    session: HostSession,
    entry: HostEntry,
    request: str,
    job_id: str,
    verb: str,
    *,
    legacy_key: str | None = None,
) -> dict[str, Any]:
    """Run a host subcommand that answers with its own verdict, refusal included.

    `check=False`: a refusal (a finished job, an id this host does not know) *is*
    the host's document, and raising on the exit code would throw away the
    reason it gave. An answer with no `status` -- a build too old to know the
    command, or something else entirely -- is an error too: reporting success
    for a job the host never touched is worse than any exception.
    """
    payload = session.host_json(request, check=False)
    document = payload if isinstance(payload, dict) else {}
    if document.get("error"):
        raise CliError(f"host {entry.name} did not {verb} {job_id}: {document['error']}")
    if not document.get("status"):
        if legacy_key and document.get(legacy_key) is True:
            # A build from before the command answered with a status: it did
            # the thing, and said so in the shape it knew.
            return document
        raise CliError(
            f"host {entry.name} did not say what it did with {job_id}: {json.dumps(payload)[:200]}"
        )
    return document


def reorder_job(job_id: str, priority: int, host: str | None, settings: Settings) -> dict[str, Any]:
    check_priority(priority)
    entry, _ = find_job_host(job_id, named_registry(), host)
    session = open_session(entry, settings)
    host_answer(
        session,
        entry,
        f"reorder {shlex.quote(job_id)} {priority}",
        job_id,
        "reorder",
        legacy_key="reordered",
    )
    # The mirror, for the same reason `estimate` updates it: `requeue` submits
    # what S3 holds, so a reorder left out of it would hand the re-run back at
    # the priority the job was first submitted with.
    warning = mirror_spec_field(job_id, "priority", priority, settings, what="priority")
    return {
        "job_id": job_id,
        "host": entry.name,
        "priority": priority,
        "warnings": [warning] if warning else [],
        **placement_after(entry, job_id, settings, session=session),
    }


def preempt_job(
    job_id: str, priority: int | None, host: str | None, settings: Settings
) -> dict[str, Any]:
    """Stop a running job and put it back in its host's queue.

    The job keeps its id and re-runs from the start as its next attempt, from
    the workdir that is already on the host -- nothing is re-synced from here,
    and the job never leaves the host it was submitted to. `gpuc requeue` is
    the other half of that pair: a fresh job id, from the mirrored spec, on
    whichever host you name.
    """
    if priority is not None:
        check_priority(priority)
    entry, _ = find_job_host(job_id, named_registry(), host)
    session = open_session(entry, settings)
    request = f"preempt {shlex.quote(job_id)}"
    if priority is not None:
        request += f" --priority {priority}"
    document = host_answer(session, entry, request, job_id, "preempt")
    warnings = []
    if priority is not None:
        # The same reason `reorder` re-mirrors: `requeue` submits what S3
        # holds, and would otherwise hand the job back at its old priority.
        warning = mirror_spec_field(job_id, "priority", priority, settings, what="priority")
        if warning:
            warnings.append(warning)
    return {
        "job_id": job_id,
        "host": entry.name,
        "status": document["status"],
        "priority": document.get("priority"),
        "warnings": warnings,
    }


def placement_after(
    entry: HostEntry, job_id: str, settings: Settings, *, session: HostSession | None = None
) -> dict[str, Any]:
    """Where the job now sits in the host's queue: what `submit` and `reorder`
    answer "so when does it run" with.

    Asked *after* the enqueue or the move, so it is best effort by
    construction: whatever goes wrong here costs a document of nulls, never the
    command's exit code -- the job is queued either way, and a submit that
    printed a traceback over a job it had already enqueued would be worse than
    one that said nothing about the queue.
    """
    try:
        view = status_mod.gather(entry, settings, session=session)
    except (ConfigError, ProviderError, RemoteError, TransportError, OSError):
        return status_mod.placement_unknown()
    return status_mod.queue_placement(view, job_id)


def check_estimate(minutes: float | None, *, clear: bool) -> float | None:
    """The estimate a request asks for, or a usage error before any host is asked."""
    if clear is (minutes is not None):
        raise UsageError("give --minutes N or --clear, not both")
    wanted: float | None = None if clear else minutes
    if wanted is not None and not wanted > 0.0:
        raise UsageError(f"--minutes must be a positive number of minutes, got {wanted:g}")
    if wanted is not None and jobs.utc_in(wanted * 60.0) is None:
        # An `inf`, or the `1e10` units typo: no date can hold it, so the host
        # would record it and then publish no eta at all.
        raise UsageError(f"--minutes {wanted:g} is too far away to be an end time")
    return wanted


def mirror_spec_field(
    job_id: str, field: str, value: Any, settings: Settings, *, what: str
) -> str | None:
    """Put a change made to a job's spec on the host in its mirrored spec too,
    or say why it could not be.

    `requeue` submits what the *mirror* holds, so leaving it behind would hand a
    re-run of the job back with the old value, silently. A mirror that cannot be
    updated is a note and never a failure: the change is already recorded where
    `status` reads it, which is what was asked for.
    """
    s3 = S3Index.from_settings(settings)
    if s3 is None:
        return None
    try:
        document = s3.get_spec(job_id)
        document[field] = value
        s3.put_spec_document(job_id, document)
    except (S3IndexError, S3ObjectMissing, ValueError) as exc:
        return (
            f"the host has the new {what}, but its mirrored spec still has the old one, "
            f"so `gpuc requeue {job_id}` would not carry it: {str(exc).splitlines()[0]}"
        )
    return None


def estimate_job(
    job_id: str, wanted: float | None, host: str | None, settings: Settings
) -> dict[str, Any]:
    """Add, change or clear a job's `estimated_runtime_min` after submitting it.

    `wanted` has been through `check_estimate`; None clears the estimate.
    """
    entry, _ = find_job_host(job_id, named_registry(), host)
    session = open_session(entry, settings)
    request = "--clear" if wanted is None else repr(wanted)
    document = host_answer(
        session, entry, f"estimate {shlex.quote(job_id)} {request}", job_id, "set the estimate on"
    )
    recorded = document.get("estimated_runtime_min")
    expected = recorded is None if wanted is None else isinstance(recorded, (int, float))
    if not expected:
        # Otherwise a host whose answer lacks the key reports a successful
        # *clear* of a job it never touched.
        raise CliError(
            f"host {entry.name} did not say what estimate it recorded for {job_id}: "
            f"{json.dumps(document)[:200]}"
        )
    warnings = [str(document["warning"])] if document.get("warning") else []
    mirror_note = mirror_spec_field(
        job_id, "estimated_runtime_min", wanted, settings, what="estimate"
    )
    if mirror_note:
        warnings.append(mirror_note)
    return {
        "job_id": job_id,
        "host": entry.name,
        "estimated_runtime_min": recorded,
        "status": document["status"],
        "warnings": warnings,
    }


@dataclass
class LogText:
    """A job's log and where it was read from, for both output forms."""

    source: str
    """`host` or `s3`."""
    location: str | None
    text: str = ""
    notes: list[str] = field(default_factory=list)

    def document(self, job_id: str, host: str) -> dict[str, Any]:
        return {
            "job_id": job_id,
            "host": host,
            "source": self.source,
            "location": self.location,
            "lines": self.text.splitlines(),
            "notes": self.notes,
        }


def job_log_path(entry: HostEntry, job_id: str, settings: Settings) -> tuple[HostSession, str]:
    session = open_session(entry, settings)
    return session, f"{session.job_dir(job_id)}/log.txt"


def read_log(
    job_id: str,
    host: str | None,
    lines: int,
    settings: Settings,
    *,
    report: Callable[[str], None] = note,
) -> tuple[HostEntry, LogText]:
    """The tail of a job's log from its host, else from the S3 mirror."""
    entry, index = find_job_host(job_id, named_registry(), host)
    remote = None
    purged = False
    try:
        session, remote = job_log_path(entry, job_id, settings)
        result = session.transport.tail(remote, lines=lines)
        if result.returncode == 0:
            return entry, LogText("host", remote, result.stdout)
        purged = job_dir_gone(session, job_id)
        why = (result.output.strip().splitlines() or ["no log file on the host"])[-1]
    except (RemoteError, TransportError) as exc:
        why = str(exc).splitlines()[0]
    # A job dir that is gone entirely is what `gpuc clean --purge` does on
    # purpose. Saying "purged" beats printing a `tail: No such file`.
    missing = (
        f"job {job_id} was purged from host {entry.name} "
        f"(gpuc clean --purge removes the whole job dir once it is mirrored)"
        if purged
        else f"could not read {remote or 'the host log'}: {why}"
    )
    report(missing)
    log = logs_from_s3(job_id, entry, index, settings, purged=purged, report=report)
    log.notes.insert(0, missing)
    return entry, log


def job_dir_gone(session: HostSession, job_id: str) -> bool:
    """Is the job dir itself missing, rather than just its log?"""
    try:
        result = session.run(f"test -d {shlex.quote(session.job_dir(job_id))}", timeout=30.0)
    except TransportError:
        return False
    return result.returncode != 0


def logs_from_s3(
    job_id: str,
    entry: HostEntry,
    index: IndexEntry | None,
    settings: Settings,
    *,
    purged: bool = False,
    report: Callable[[str], None] = note,
) -> LogText:
    s3 = S3Index.from_settings(settings)
    prefix = (index.s3_prefix if index else None) or entry.s3_prefix
    if s3 is None or not prefix:
        gone = (
            f"Its job dir was purged from host {entry.name}, so this log no longer exists "
            f"anywhere.\n"
            if purged
            else ""
        )
        raise CliError(
            f"no S3 mirror to fall back on for {job_id}.\n"
            f"{gone}"
            f"Set s3_bucket in ~/.config/gpu-coordinator/config.toml to keep logs after a "
            f"host goes away."
        )
    uri = job_log_uri(prefix, job_id)
    fallback = f"falling back to the S3 mirror at {uri}"
    report(fallback)
    return LogText("s3", uri, s3.get_uri(uri), [fallback])
