"""What `gpuc` and the web dashboard both do.

Each function here does one thing a command does and returns the document its
`--json` form prints. The CLI renders that document as text or JSON; the
dashboard serves it over HTTP. Neither adds judgement of its own, so a job the
CLI would refuse to cancel is one the dashboard refuses too, with the same
words.

An `Answer` is what a command hands back: the document, the text form, and
what failed -- `exit_code_of` is the one place that becomes a number, and
`exits.http_status` the one place that number becomes an HTTP status.
"""

from __future__ import annotations

import json
import math
import shlex
import sys
from collections.abc import Callable, Collection, Iterator, Sequence
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
    forget_host,
    hosts_file,
    load_settings,
    open_registry,
    state_dir,
    write_config_template,
)
from gpuc.control.connect import Connection
from gpuc.control.exits import (
    EXIT_ERROR,
    EXIT_INTERRUPTED,
    EXIT_LOCAL_STATE,
    EXIT_NOT_FOUND,
    EXIT_OK,
    EXIT_USAGE,
)
from gpuc.control.jsonout import note
from gpuc.control.providers.base import Provider, ProviderError
from gpuc.control.providers.runpod import RunPodProvider
from gpuc.control.provision import ProvisionError
from gpuc.control.remote import (
    Answered,
    Asked,
    Gone,
    HostSession,
    RemoteError,
    Unaskable,
    ask,
    reason_of,
)
from gpuc.control.remote import config_file as remote_config_file
from gpuc.control.s3index import (
    IndexEntry,
    JobIndex,
    LocalIndex,
    S3Index,
    S3IndexError,
    S3ObjectMissing,
    job_log_uri,
    job_uri,
)
from gpuc.control.skill import SkillError
from gpuc.control.submit import SubmitError
from gpuc.control.teardown import TerminateError
from gpuc.control.transport import TransportError
from gpuc.host import jobs
from gpuc.host.jobs import FINISHED_STATUSES

__all__ = [
    "EXIT_ERROR",
    "EXIT_INTERRUPTED",
    "EXIT_LOCAL_STATE",
    "EXIT_NOT_FOUND",
    "EXIT_OK",
    "EXIT_USAGE",
]


class CliError(RuntimeError):
    exit_code = EXIT_ERROR


class UsageError(CliError):
    """The invocation was wrong, not the world."""

    exit_code = EXIT_USAGE


class NotFound(CliError):
    """The job or host named on the command line does not exist."""

    exit_code = EXIT_NOT_FOUND


class Interrupted(CliError):
    """A Ctrl-C, with something command-specific to say about what was in flight.

    Raised rather than returned so that a blocking command cannot get the exit
    code or the `--json` document wrong: `main` turns an unhandled Ctrl-C into
    the same thing with a bare message, and this only adds detail -- the
    words, and under `--json` the `document` a partial run still has to show.
    """

    exit_code = EXIT_INTERRUPTED

    def __init__(self, message: str, document: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.document = document or {}


FAILURES = (
    CleanError,
    CliError,
    ConfigError,
    SubmitError,
    TerminateError,
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


@dataclass
class Answer:
    """What a command that ran to the end hands back.

    `document` is the `--json` form and `text` the other one (None when the
    command already streamed its text). `failures` is everything that did not
    work along the way -- a host that could not be read, a bootstrap that
    failed -- and `unknown` that local state could not be read at all.
    `outcome` is a code the command relays rather than owns: the job's, for
    `wait`, or the remote command's, for `ssh`.
    """

    document: dict[str, Any]
    text: str | None = None
    failures: list[str] = field(default_factory=list)
    unknown: bool = False
    outcome: int | None = None

    @property
    def exit_code(self) -> int:
        return exit_code_of(self)


def exit_code_of(answer: Answer) -> int:
    """The one rule: unknown beats everything, a relayed outcome is itself,
    else 1 for any failure and 0 for none."""
    if answer.unknown:
        return EXIT_LOCAL_STATE
    if answer.outcome is not None:
        return answer.outcome
    return EXIT_ERROR if answer.failures else EXIT_OK


PROVIDERS: dict[str, Callable[[Settings], Provider]] = {
    "runpod": lambda settings: RunPodProvider(prefix=settings.runpod_pod_prefix),
}
"""Every rental provider, by the name a host's `rental.provider` gives it."""


def make_provider(settings: Settings, provider: str = "runpod") -> Provider:
    try:
        factory = PROVIDERS[provider]
    except KeyError:
        raise ProviderError(
            f"no such provider: {provider!r} (known: {', '.join(PROVIDERS)})"
        ) from None
    return factory(settings)


def provider_for(
    entries: Sequence[HostEntry], settings: Settings, report: Callable[[str], None] = note
) -> Provider | None:
    """The provider to ask about these hosts' pods, if any of them is rented.

    Only built when a rental is on screen, and never fatal: a status of two
    ssh boxes must not need an API key, and one with a rental must not fail
    outright for want of one -- the pod line is missing and says so.
    """
    names = {entry.rental.provider for entry in entries if entry.rental is not None}
    if not names:
        return None
    try:
        return make_provider(settings, sorted(names)[0])
    except ProviderError as exc:
        report(f"pod status unavailable: {exc}")
        return None


MAX_PARALLEL_HOSTS = 8


def gather_all(
    entries: Sequence[HostEntry], settings: Settings, provider: Provider | None, request: str
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
            lambda entry: status_mod.gather(entry, settings, request=request, provider=provider),
            entries,
        )


@dataclass
class StatusResult:
    """Everything `gpuc status` found out, before any of it is printed.

    One object for the text form, `--json` and the dashboard, so the three
    cannot disagree about what was asked, what failed, or -- through
    `answer` -- what the exit code is.
    """

    read: RegistryRead
    views: list[status_mod.HostView] = field(default_factory=list)
    every_host: bool = True
    """Whether the registry's own problems count: with `--host X` the answer
    was never meant to cover an entry that is not X."""
    unhosted: list[IndexEntry] = field(default_factory=list)
    """`--all`: the jobs only the index knows, in id order."""
    mirrored: dict[str, dict[str, Any]] = field(default_factory=dict)
    """The mirrored `state.json` of the first of `unhosted`, by id."""
    index_error: str | None = None
    """Why `unhosted`, or a gone host's jobs, may be short: an S3 index that
    could not be read."""

    @property
    def errors(self) -> list[str]:
        """The problems that belong to no host: the registry's, the index's."""
        errors = list(self.read.errors)
        if self.read.unreadable:
            errors.append(
                f"{hosts_file()} could not be read, so `hosts` is empty because nothing is known"
            )
        if self.index_error:
            errors.append(f"{self.index_error}; {INDEX_SHORT}")
        return errors

    @property
    def failures(self) -> list[str]:
        """What makes this exit 1: a host that could not be read, a registry
        entry this build could not parse (only when every host was asked), an
        index that could not be read."""
        failures = [view.failure for view in self.views if view.failure]
        if self.every_host:
            failures += self.read.errors
        if self.index_error:
            failures.append(self.index_error)
        return failures

    @property
    def seen(self) -> set[str]:
        return {
            job.job_id for view in self.views for job in view.queue + view.running + view.finished
        }

    def lost(self, entry: IndexEntry) -> bool:
        """The mirror records this job as having lost its outputs."""
        return bool((self.mirrored.get(entry.job_id) or {}).get("outputs_lost"))

    def final_status(self, entry: IndexEntry) -> str | None:
        """The mirror's final status for a job whose host is gone, and only
        then: for any other host the mirror is not the answer."""
        if self.host_state(entry) is not status_mod.HostState.GONE:
            return None
        found = (self.mirrored.get(entry.job_id) or {}).get("status")
        return found if found in FINISHED_STATUSES else None

    def host_state(self, entry: IndexEntry) -> status_mod.HostState:
        """What this run found the job's host to be. A host with no registry
        entry here is gone (a rental that ended was forgotten); one whose
        entry this build could not read was not asked."""
        view = next((view for view in self.views if view.entry.name == entry.host), None)
        if view is not None:
            return view.state
        if entry.host in self.read.skipped:
            return status_mod.HostState.UNASKABLE
        return status_mod.HostState.GONE

    def document(
        self, *, recent: int = status_mod.RECENT_FINISHED, since_s: float | None = None
    ) -> dict[str, Any]:
        return status_mod.document(
            self.views,
            errors=self.errors,
            unhosted=[
                status_mod.unhosted_json(
                    entry,
                    lost=self.lost(entry),
                    state=self.host_state(entry),
                    final=self.final_status(entry),
                )
                for entry in self.unhosted
            ],
            recent=recent,
            since_s=since_s,
        )

    def unhosted_text(self, host: str | None) -> str | None:
        if not self.unhosted:
            return None
        scope = f" for host {host}" if host else ""
        lines = [f"jobs known only to the index{scope}:"]
        lines += [
            status_mod.unhosted_line(
                entry,
                lost=self.lost(entry),
                state=self.host_state(entry),
                final=self.final_status(entry),
            )
            for entry in self.unhosted
        ]
        first = next(
            (e for e in self.unhosted if status_mod.requeue_offered(self.host_state(e))), None
        )
        if first is not None:
            # The host is named only when it answered: it is there and empty,
            # which is the wiped-home case; a gone host is not a place to send
            # anything.
            answered = self.host_state(first) is status_mod.HostState.ANSWERED
            target = first.host if answered else "<name>"
            lines.append(f"  bring one back with: gpuc requeue {first.job_id} --host {target}")
        return "\n".join(lines)

    def answer(
        self,
        *,
        recent: int = status_mod.RECENT_FINISHED,
        since_s: float | None = None,
        text: str | None = None,
    ) -> Answer:
        return Answer(
            self.document(recent=recent, since_s=since_s),
            text,
            failures=self.failures,
            unknown=self.read.unreadable,
        )


INDEX_SHORT = "the jobs read from the index may be short"


class IndexRead:
    """The job index, read at most once per command and only if something
    needs it: listing the S3 index is a request per job."""

    def __init__(self, settings: Settings) -> None:
        self.index = JobIndex(settings)
        self._read: tuple[dict[str, IndexEntry], str | None] | None = None

    def all(self) -> tuple[dict[str, IndexEntry], str | None]:
        if self._read is None:
            self._read = self.index.all()
        return self._read

    @property
    def error(self) -> str | None:
        """Why what was read may be short; None if nothing was read."""
        return self._read[1] if self._read is not None else None

    def on(self, host: str) -> list[IndexEntry]:
        return [entry for entry in self.all()[0].values() if entry.host == host]


def status(
    read: RegistryRead,
    settings: Settings,
    *,
    host: str | None = None,
    all_jobs: bool = False,
    recent: int = status_mod.RECENT_FINISHED,
    since_s: float | None = None,
    on_view: Callable[[status_mod.HostView], None] | None = None,
    report: Callable[[str], None] = note,
) -> StatusResult:
    """`gpuc status`, however it is going to be shown.

    The registry's own errors ride along, and an unreadable registry is an
    empty `hosts` *with* an error saying so -- exit 3 or HTTP 503 come from
    the same `Answer`. `on_view` is called with each host as it answers, so
    the text form can print each block without waiting for the slowest.
    Nothing here writes the registry: forgetting a rental that has ended is
    `forget_gone_rentals`, which only a typed command calls.

    A gone host's jobs are the mirror's (`fill_from_mirror`). A gone host is
    shown when it was found gone this run or is named with `--host`: listing
    every rental ever used would bury the hosts that exist.

    Each host sends only the finished jobs inside `recent` and `since_s`,
    except under `all_jobs`: a job the index knows is listed as one no host
    has whenever its host did not send it, so that list needs every job.
    """
    result = StatusResult(read, every_host=host is None)
    index = IndexRead(settings)
    registry = read.registry if not read.unreadable else Registry()
    entries = [entry for name, entry in registry.hosts.items() if host in (None, name)]

    def shown(view: status_mod.HostView) -> None:
        result.views.append(view)
        if on_view is not None:
            on_view(view)

    if host is not None and host not in registry.hosts and not read.unreadable:
        if host in read.skipped:
            shown(
                status_mod.HostView(
                    HostEntry(name=host), error=unreadable_entry(host).reason, registered=False
                )
            )
        elif index.on(host):
            shown(gone_view(host, index.index, index.on(host)))
        else:
            # Neither here nor in the index: a typo, not a rental that ended.
            registry.require(host)
    provider = provider_for(entries, settings, report) if entries else None
    request = (
        status_mod.status_request()
        if all_jobs
        else status_mod.status_request(recent=recent, since_s=since_s)
    )
    for view in gather_all(entries, settings, provider, request):
        if view.gone:
            fill_from_mirror(view, index.index, index.on(view.entry.name))
        shown(view)
    result.index_error = index.error
    if all_jobs:
        result.unhosted, result.mirrored, short = unhosted_jobs(index, result.seen, host)
        result.index_error = result.index_error or short
    return result


def gone_view(name: str, index: JobIndex, entries: Sequence[IndexEntry]) -> status_mod.HostView:
    """A host this machine has no entry for, with what the mirror holds of it."""
    view = status_mod.HostView(
        HostEntry(name=name),
        state=status_mod.HostState.GONE,
        error=not_registered(name).reason,
        registered=False,
    )
    fill_from_mirror(view, index, entries)
    return view


def fill_from_mirror(
    view: status_mod.HostView, index: JobIndex, entries: Sequence[IndexEntry]
) -> None:
    """A gone host's finished jobs, as the mirror has them, and the ones it
    has no final state for: those went with the host, and say so.

    The newest `MIRROR_STATE_LOOKUPS` of them, one GET each, all at once.
    """
    cached = view.entry.config.s3_prefix if view.registered else None
    view.mirror_prefix = cached
    newest = sorted(entries, key=lambda entry: entry.job_id, reverse=True)[:MIRROR_STATE_LOOKUPS]
    if index.s3 is None:
        view.lost = [entry.job_id for entry in newest]
        view.lost_reason = (
            f"s3_bucket is unset in {config_file()}, so nothing of this host is mirrored "
            f"and its jobs went with it"
        )
        return
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_HOSTS) as pool:
        found = list(
            pool.map(
                lambda entry: read_mirror(index, entry.job_id, entry.s3_prefix or cached, entry),
                newest,
            )
        )
    for entry, mirrored in zip(newest, found, strict=True):
        if mirrored.view is not None:
            view.finished.append(mirrored.view)
        else:
            view.lost.append(entry.job_id)
    if view.lost:
        view.lost_reason = (
            "the mirror has no final state for these, so they went with the host; "
            "`gpuc status --all` lists them for `gpuc requeue`"
        )
    view.finished.sort(key=lambda job: job.ended_at or "", reverse=True)


def forget_gone_rentals(
    views: Sequence[status_mod.HostView], report: Callable[[str], None] = note
) -> None:
    """Drop the registry entry of every rental the provider says no longer exists.

    A rental ends itself when its queue goes idle, so this is the ordinary end
    of one rather than something gone wrong; leaving the entry would have the
    next command ssh to an address somebody else now owns. Only a command
    somebody typed does this: deleting a registry entry off the back of a
    dashboard's poll would mean a stray 404 costing a live host its record with
    nobody watching.
    """
    for view in views:
        if view.gone and view.registered:
            report(f"forgetting host {view.entry.name}: its pod is gone")
            forget_host(view.entry.name, view.entry.pod_id, report)


def shipped_note(entry: HostEntry) -> str | None:
    """The offline half of the version story: what the host last said it ran.

    `gpuc host list` and `gpuc version` never ask a host anything, so the
    cached commit is all they have. What the host is *running* now is `gpuc
    status`, which asks it.
    """
    return version_mod.shipped_commit_note(
        entry.name, entry.config.pkg_commit, version_mod.local_commit()
    )


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
    names = dict.fromkeys(entry.config.env, "<set>")
    cache: dict[str, Any] = document.get("cache") or {}
    cache["config"] = {**(cache.get("config") or {}), "env": names}
    stale = shipped_note(entry)
    return {
        **document,
        **config,
        # Derived from the address, so not in the dump, and the words a reader
        # scans the list by.
        "kind": entry.kind,
        "pod_id": entry.pod_id,
        "env": names,
        "cache": cache,
        "cache_dir": entry.config.env.get("UV_CACHE_DIR"),
        "config_seen_at": entry.seen_at,
        "remote_home": entry.remote_home,
        "ephemeral": entry.rental is not None,
        "warnings": [stale] if stale else [],
    }


def hosts_document(read: RegistryRead) -> dict[str, Any]:
    return {
        "hosts": [host_document(entry) for entry in read.registry.hosts.values()],
        "errors": list(read.errors),
    }


def registry_answer(read: RegistryRead, document: dict[str, Any], text: str | None) -> Answer:
    """A command reporting on the registry: 0 only if all of it parsed.

    A registry that could not be read at all is unknown (exit 3), while an
    entry this build could not parse is one host missing from an answer that
    is otherwise complete: a failure, with the rest of the answer printed.
    """
    return Answer(document, text, failures=list(read.errors), unknown=read.unreadable)


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
        "config_path": remote_config_file(connection.home),
        "changes": list(connection.changes),
        "warnings": [*document["warnings"], *warnings],
    }


def remove_host(name: str) -> dict[str, Any]:
    """Forget a host here. Nothing on the host, or at its provider, changes.

    A rental in particular is not terminated: after handoff it ends itself
    when idle, and nothing on a client watches it. The document says so for a
    rental, so a caller does not take "removed" for "stopped billing" -- and
    names `gpuc host terminate`, which is the command that does.
    """
    entry = open_registry().require(name)
    if not forget_host(name):
        raise CliError(f"host {name} could not be removed from {hosts_file()}")
    notes = []
    if entry.rental is not None:
        notes.append(
            f"its pod {entry.rental.pod_id} is not terminated by this: it ends itself once its "
            f"queue has been idle, and `gpuc pods` shows it until then"
        )
        # By pod id, not by name: this machine has just forgotten the name.
        notes.append(f"`gpuc host terminate {entry.rental.pod_id}` ends it now instead")
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
    minute is `gpuc status --json`'s `pkg_commit`.
    """
    commit = version_mod.local_commit()
    # Every host anything here has read, not only the ones *this* machine
    # bootstrapped: adopting a host is the ordinary way to register one.
    hosts = [
        entry for entry in read.registry.hosts.values() if entry.seen_at or entry.bootstrapped_at
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
                "pkg_commit": entry.config.pkg_commit,
                "seen_at": entry.seen_at,
                "current": not version_mod.is_other_build(entry.config.pkg_commit, commit),
            }
            for entry in hosts
        ],
        "errors": list(read.errors),
    }


Trouble = Unaskable | Gone
"""Why a job's host could not confirm it holds the job."""


def mirror_is_the_answer(trouble: Trouble) -> bool:
    """The one rule `logs` and `wait` read the mirror by.

    The spec reads the mirror only when the host is *gone*. For a gone host
    the mirror is the answer, now, and nothing failed. A host that could not
    be asked may still hold the job: `wait` keeps asking for
    `TROUBLE_GRACE_S` before it reads the mirror, and `logs` prints what the
    mirror has -- doing as much as it can -- but as a failure, exit 1 with
    the reason, never as the answer.
    """
    return isinstance(trouble, Gone)


@dataclass
class Location:
    """Where a job is: its host, its index entry, and what the host said when
    it was asked to find it (None when nothing had to be asked). `entry` is
    None for a host the index names that this machine cannot open: one with
    no registry entry (`Gone`) or one whose entry it could not read
    (`Unaskable`)."""

    host: str
    entry: HostEntry | None = None
    index: IndexEntry | None = None
    asked: Asked | None = None

    @property
    def session(self) -> HostSession | None:
        return self.asked.session if isinstance(self.asked, Answered) else None

    @property
    def trouble(self) -> Trouble | None:
        """The index says the job is on this host, and the host could not be
        asked to confirm it: the caller decides whether that is the mirror's
        moment or a failure."""
        return None if self.asked is None or isinstance(self.asked, Answered) else self.asked

    @property
    def trouble_reason(self) -> str:
        trouble = self.trouble
        return trouble.reason if trouble is not None else ""

    def require_entry(self) -> HostEntry:
        """The host, for a caller that needs to open it -- which a forgotten
        host cannot be: the job's remains are in the mirror, if anywhere."""
        if self.entry is None:
            raise CliError(
                f"job's host {self.host} cannot be asked: {self.trouble_reason}\n"
                f"`gpuc logs` reads its mirror; `gpuc requeue <id> --host <name>` re-runs it."
            )
        return self.entry


def not_registered(name: str) -> Gone:
    """A host name this machine has no entry for. The same state as a pod the
    provider reports terminated, because it becomes one: the first `gpuc
    status` that sees a rental's pod gone forgets its entry, and the next
    command about one of its jobs must not start answering differently."""
    return Gone(
        f"host {name} is not registered on this machine (a rental that ended is "
        f"forgotten), so it cannot be asked"
    )


def unreadable_entry(name: str) -> Unaskable:
    return Unaskable(
        f"host {name}'s registry entry could not be read (see the warning above), so it "
        f"was not asked; fix the entry, or register the host again with "
        f"`gpuc host add {name} --pod <id>`"
    )


@dataclass
class Unlocated:
    """Why a job could not be placed on any host. `missing` when every host
    that could have it answered without it: "no such job", exit 4."""

    reason: str
    missing: bool = False

    def exception(self) -> CliError:
        return NotFound(self.reason) if self.missing else CliError(self.reason)


Placed = Location | Unlocated


def locate(
    job_id: str,
    registry: Registry,
    explicit: str | None,
    settings: Settings | None = None,
    *,
    provider: Provider | None = None,
    skipped: Collection[str] = (),
) -> Location:
    """Where one job is, by `locate_many`'s rule, raising what it could not say."""
    placed = locate_many(
        [job_id], registry, explicit, settings, provider=provider, skipped=skipped
    )[job_id]
    if isinstance(placed, Unlocated):
        raise placed.exception()
    return placed


def locate_many(
    job_ids: Sequence[str],
    registry: Registry,
    explicit: str | None,
    settings: Settings | None = None,
    *,
    provider: Provider | None = None,
    skipped: Collection[str] = (),
) -> dict[str, Placed]:
    """The host each job is on: `--host` if given, else the index (local, then
    the mirror), else whichever registered host admits to it. Keyed by id, in
    the order asked, each repeat once.

    A `--host` this machine has no entry for is `Gone`, like any other name it
    does not have, when the index says the job ran there: naming the host
    must not make the answer worse than leaving it out.

    A name in the mirror's index is the *submitting* client's name for the
    host, which need not be this machine's: it is asked first, not believed.
    A host the index names that cannot be asked is still the answer -- the job
    is there as far as anything knows -- carried as `trouble` for the caller
    to judge, never hidden. A host the index names that this machine has no
    entry for is `Gone`: a rental that ended and was forgotten, the ordinary
    end of one, and `gpuc logs` and `gpuc wait` on its jobs must reach the
    mirror rather than "no such job". One whose registry entry this build
    could not read (warned about on the way in) is `Unaskable`: it is known,
    may well be running the job, and was not asked. An id no answering host
    knows is `missing` only when every host answered: with one unaskable, the
    job may well be on it, and "no such job" would be a lie told over a
    connection error.

    However many ids there are, each host is asked at most twice, all hosts
    at once each time: about the jobs the index says it has, then about every
    id still unplaced. A sweep of hundreds costs a round trip per host, not
    one per job.
    """
    settings = settings if settings is not None else load_settings()
    ids = list(dict.fromkeys(job_ids))
    local = LocalIndex()
    index = JobIndex(settings)
    if explicit:
        return {
            job_id: _on_named_host(job_id, explicit, registry, local.get(job_id), index, skipped)
            for job_id in ids
        }
    placed: dict[str, Placed] = {}
    unplaced: dict[str, IndexEntry | None] = {}
    for job_id in ids:
        indexed = local.get(job_id)
        if indexed is not None and indexed.host in registry.hosts:
            # Not asked: the verb or the poll that follows asks it anyway.
            placed[job_id] = Location(indexed.host, registry.hosts[indexed.host], indexed)
            continue
        indexed = indexed or index.get(job_id)
        if indexed is not None and indexed.host not in registry.hosts and indexed.host in skipped:
            placed[job_id] = Location(indexed.host, None, indexed, unreadable_entry(indexed.host))
            continue
        unplaced[job_id] = indexed
    if unplaced:
        provider = provider or provider_for(list(registry.hosts.values()), settings)
        placed.update(_ask_around(unplaced, registry, settings, provider))
    return {job_id: placed[job_id] for job_id in ids}


def _on_named_host(
    job_id: str,
    name: str,
    registry: Registry,
    local: IndexEntry | None,
    index: JobIndex,
    skipped: Collection[str],
) -> Placed:
    """A job on the host `--host` named. Not asked: the caller already knows
    the host, and the poll or the verb that follows asks it anyway."""
    entry = registry.hosts.get(name)
    if entry is not None:
        return Location(entry.name, entry, local)
    if name in skipped:
        return Location(name, None, local, unreadable_entry(name))
    # Gone only where the index agrees the job ran there. A typo'd host name
    # is not a rental that ended, and treating it as one would read the
    # mirror's copy of a job that is still running somewhere else.
    indexed = local or index.get(job_id)
    if indexed is None or indexed.host != name:
        where = f"\nThe index has job {job_id} on host {indexed.host}." if indexed else ""
        try:
            registry.require(name)
        except HostNotFound as exc:
            return Unlocated(f"{exc}{where}", missing=True)
    return Location(name, None, indexed, not_registered(name))


def _ask_around(
    unplaced: dict[str, IndexEntry | None],
    registry: Registry,
    settings: Settings,
    provider: Provider | None,
) -> dict[str, Placed]:
    """`locate_many` for the ids that need a host to say it has them."""
    named = {
        job_id: registry.hosts.get(indexed.host) if indexed is not None else None
        for job_id, indexed in unplaced.items()
    }
    latest: dict[str, Asked] = {}
    first: dict[str, Asked] = {}
    found: dict[str, tuple[str, Answered]] = {}

    def ask_each(questions: dict[str, list[str]]) -> None:
        def one(name: str) -> tuple[str, Asked]:
            before = latest.get(name)
            session = before.session if isinstance(before, Answered) else None
            request = status_mod.status_request(questions[name])
            return name, ask(
                registry.hosts[name], request, settings, provider=provider, session=session
            )

        if not questions:
            return
        with ThreadPoolExecutor(max_workers=min(MAX_PARALLEL_HOSTS, len(questions))) as pool:
            for name, asked in pool.map(one, questions):
                latest[name] = asked
                first.setdefault(name, asked)
                if not isinstance(asked, Answered):
                    continue
                for job in (asked.payload or {}).get("jobs") or []:
                    if isinstance(job, dict) and job.get("job_id") in questions[name]:
                        found.setdefault(job["job_id"], (name, asked))

    by_named: dict[str, list[str]] = {}
    for job_id, entry in named.items():
        if entry is not None:
            by_named.setdefault(entry.name, []).append(job_id)
    ask_each(by_named)
    rest = [job_id for job_id in unplaced if job_id not in found]
    everyone: dict[str, list[str]] = {}
    for name in registry.hosts:
        if name in latest and not isinstance(latest[name], Answered):
            continue
        wanted = [job_id for job_id in rest if _name(named[job_id]) != name]
        if wanted:
            everyone[name] = wanted
    ask_each(everyone)
    placed: dict[str, Placed] = {}
    for job_id, indexed in unplaced.items():
        entry = named[job_id]
        if job_id in found:
            name, asked = found[job_id]
            placed[job_id] = Location(name, registry.hosts[name], indexed, asked)
            continue
        trouble = first.get(entry.name) if entry is not None else None
        if entry is not None and trouble is not None and not isinstance(trouble, Answered):
            placed[job_id] = Location(entry.name, entry, indexed, trouble)
            continue
        unasked = [
            f"{name}: {asked.reason}"
            for name, asked in latest.items()
            if isinstance(asked, Unaskable) and (entry is None or name != entry.name)
        ]
        if indexed is not None and entry is None and not unasked:
            placed[job_id] = Location(indexed.host, None, indexed, not_registered(indexed.host))
        elif unasked:
            placed[job_id] = Unlocated(
                f"no host that answered knows job {job_id}, and these could not be asked:\n"
                + "\n".join(f"  {line}" for line in unasked)
                + "\nPass --host <name>, or check `gpuc host list` and `gpuc status --all`."
            )
        else:
            placed[job_id] = Unlocated(
                f"no registered host knows job {job_id}.\n"
                f"Pass --host <name>, or check `gpuc host list` and `gpuc status --all`.",
                missing=True,
            )
    return placed


def _name(entry: HostEntry | None) -> str | None:
    return entry.name if entry is not None else None


def find_job(job_id: str, host: str | None, settings: Settings) -> Location:
    read = open_registry()
    return locate(job_id, read.named(), host, settings, skipped=read.skipped)


@dataclass
class Done:
    """What a job verb did to one job, or why it did not.

    `fields` are the host's own words for what it did (`status`, `priority`,
    `estimated_runtime_min`); `source` is `mirror` for a job whose host is
    gone, answered from what the mirror says it ended as.
    """

    job_id: str
    host: str | None
    fields: dict[str, Any] = field(default_factory=dict)
    source: str = "host"
    error: str | None = None
    missing: bool = False
    warnings: list[str] = field(default_factory=list)

    def document(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "host": self.host,
            **self.fields,
            "source": self.source,
            "error": self.error,
            "warnings": list(self.warnings),
        }


def jobs_answer(done: Sequence[Done], text: str | None = None) -> Answer:
    """The answer of every command that acts on jobs by id: `{jobs, errors}`,
    one entry per id. Exit 4 when an id is unknown and 1 when anything else
    failed, as `wait` does -- after every other id has been carried out, since
    one typo in a list of twenty must not cost the nineteen."""
    errors = [job.error for job in done if job.error is not None]
    return Answer(
        {"jobs": [job.document() for job in done], "errors": errors},
        text,
        failures=errors,
        outcome=EXIT_NOT_FOUND if any(job.missing for job in done) else None,
    )


def job_verbs(
    verb: str,
    job_ids: Sequence[str],
    host: str | None,
    settings: Settings,
    *,
    args: str = "",
    mirror: tuple[str, Any] | None = None,
    ended_is_answer: bool = False,
) -> tuple[list[Done], dict[str, HostSession]]:
    """Run one on-host verb against jobs: the path every job command takes.

    Find each job's host (`locate_many`), ask each host once, over the session
    the lookup opened, about every job on it, insist on a verdict for each,
    and put any spec change in the mirror too. Returns what became of each
    id, in the order asked, and the session each host was asked over (for a
    follow-up question like the queue placement).

    `check=False` on the host call: a refusal (a finished job, an id this host
    does not know) *is* the host's document, and raising on the exit code
    would throw away the reason it gave. A refusal that says `missing` is the
    host answering "no such job", exit 4 like every other unknown name, and
    told apart from a refusal of a job that is there by that key rather than
    by the words. An answer with no `status` is an error too: reporting
    success for a job the host never touched is worse than any exception.

    `mirror` is `(spec field, value)`: `requeue` submits what S3 holds, so a
    priority or estimate changed on the host and not in the mirror would hand
    a re-run back with the old one, silently. A mirror that cannot be updated
    is a warning, never a failure: the change is already where `status` reads
    it, which is what was asked for.

    A job whose host is gone has ended, and the mirror says how: with
    `ended_is_answer` that is the answer, as a host answers a cancel of a
    finished job with its status; otherwise the verb is refused with it, as a
    host refuses the others on a finished job.
    """
    read = open_registry()
    registry = read.named()
    provider = provider_for(list(registry.hosts.values()), settings)
    placed = locate_many(job_ids, registry, host, settings, provider=provider, skipped=read.skipped)
    done: dict[str, Done] = {}
    on_host: dict[str, list[str]] = {}
    for job_id, where in placed.items():
        if isinstance(where, Unlocated):
            done[job_id] = Done(job_id, None, error=where.reason, missing=where.missing)
        elif where.trouble is not None and mirror_is_the_answer(where.trouble):
            done[job_id] = ended_on_gone_host(verb, job_id, where, settings, ended_is_answer)
        elif where.entry is None or where.trouble is not None:
            done[job_id] = Done(
                job_id,
                where.host,
                error=f"job {job_id} is on host {where.host}, which could not be asked: "
                f"{where.trouble_reason}",
            )
        else:
            on_host.setdefault(where.host, []).append(job_id)

    def one_host(ids: list[str]) -> tuple[list[Done], HostSession | None]:
        here = [where for job_id in ids if isinstance(where := placed[job_id], Location)]
        location = here[0]
        assert location.entry is not None
        session = next((where.session for where in here if where.session), None)
        asked = ask(
            location.entry,
            f"{verb} {' '.join(shlex.quote(job_id) for job_id in ids)}{args}",
            settings,
            provider=provider,
            session=session,
            check=False,
        )
        return _verdicts(verb, location.host, ids, asked), (
            asked.session if isinstance(asked, Answered) else None
        )

    sessions: dict[str, HostSession] = {}
    if on_host:
        with ThreadPoolExecutor(max_workers=min(MAX_PARALLEL_HOSTS, len(on_host))) as pool:
            for name, (verdicts, session) in zip(
                on_host, pool.map(one_host, on_host.values()), strict=True
            ):
                done.update((job.job_id, job) for job in verdicts)
                if session is not None:
                    sessions[name] = session
    if mirror is not None:
        changed = [job for job in done.values() if job.error is None and job.source == "host"]
        with ThreadPoolExecutor(max_workers=MAX_PARALLEL_HOSTS) as pool:
            warnings = pool.map(
                lambda job: mirror_spec_field(job.job_id, mirror[0], mirror[1], settings), changed
            )
            for job, warning in zip(changed, warnings, strict=True):
                if warning:
                    job.warnings.append(warning)
    return [done[job_id] for job_id in placed], sessions


def _verdicts(verb: str, host: str, job_ids: Sequence[str], asked: Asked) -> list[Done]:
    """What one host said it did with each of these jobs."""
    if not isinstance(asked, Answered):
        return [
            Done(
                job_id,
                host,
                error=f"job {job_id} is on host {host}, which could not be asked: {asked.reason}",
            )
            for job_id in job_ids
        ]
    payload = asked.payload or {}
    answers = {
        answer["job_id"]: answer
        for answer in payload.get("jobs") or []
        if isinstance(answer, dict) and isinstance(answer.get("job_id"), str)
    }
    done: list[Done] = []
    for job_id in job_ids:
        answer = dict(answers.get(job_id) or {})
        if answer.get("missing"):
            done.append(
                Done(
                    job_id,
                    host,
                    error=f"host {host} has no job {job_id}: {answer.get('error')}; "
                    f"check the id with `gpuc status --all`",
                    missing=True,
                )
            )
        elif answer.get("error"):
            done.append(
                Done(job_id, host, error=f"host {host} did not {verb} {job_id}: {answer['error']}")
            )
        elif not answer.get("status"):
            said = json.dumps(answer or payload)[:200]
            done.append(
                Done(
                    job_id, host, error=f"host {host} did not say what it did with {job_id}: {said}"
                )
            )
        else:
            answer.pop("job_id")
            warning = answer.pop("warning", None)
            done.append(Done(job_id, host, answer, warnings=[str(warning)] if warning else []))
    return done


def ended_on_gone_host(
    verb: str, job_id: str, location: Location, settings: Settings, ended_is_answer: bool
) -> Done:
    """A verb on a job whose host is gone: what the mirror says it ended as,
    as the answer or as the reason the verb is refused -- or, with no final
    state there, that the job went with its host."""
    index = JobIndex(settings)
    mirrored = read_mirror(index, job_id, index.mirror_prefix(job_id, location.entry))
    reason = location.trouble_reason
    if mirrored.view is None:
        return Done(
            job_id,
            location.host,
            error=f"job {job_id} was on host {location.host}, which is gone ({reason}), and "
            f"{mirrored.lost}",
        )
    status = mirrored.view.status
    if ended_is_answer:
        return Done(job_id, location.host, {"status": status}, source="mirror")
    return Done(
        job_id,
        location.host,
        source="mirror",
        error=f"cannot {verb} job {job_id}: it ended {status} on host {location.host}, "
        f"which is gone ({reason}); read from the S3 mirror at {mirrored.uri}",
    )


def cancel_jobs(job_ids: Sequence[str], host: str | None, settings: Settings) -> list[Done]:
    """The host's own word for what it did to each: `cancelled` for a queued
    job it dequeued, `cancelling` for a running one whose runner has been
    marked, a finished job's own status. A job whose host is gone has already
    ended, and the mirror says how -- the same answer its host would have
    given."""
    done, _ = job_verbs("cancel", job_ids, host, settings, ended_is_answer=True)
    return done


def check_priority(priority: int) -> None:
    if not 0 <= priority <= 99:
        raise UsageError(f"priority must be 0-99 (lower dispatches first), got {priority}")


def reorder_jobs(
    job_ids: Sequence[str], priority: int, host: str | None, settings: Settings
) -> list[Done]:
    """Move queued jobs, each with where it now sits in its host's queue."""
    check_priority(priority)
    done, sessions = job_verbs(
        "reorder",
        job_ids,
        host,
        settings,
        args=f" --priority {priority}",
        mirror=("priority", priority),
    )
    moved = [job for job in done if job.error is None and job.host in sessions]
    views = {
        name: status_mod.gather(session.entry, settings, session=session)
        for name, session in sessions.items()
        if any(job.host == name for job in moved)
    }
    for job in moved:
        assert job.host is not None
        job.fields.update(status_mod.queue_placement(views[job.host], job.job_id))
    return done


def preempt_jobs(
    job_ids: Sequence[str], priority: int | None, host: str | None, settings: Settings
) -> list[Done]:
    """Stop running jobs and put them back in their hosts' queues.

    Each keeps its id and re-runs from the start as its next attempt, from
    the workdir that is already on the host -- nothing is re-synced from here,
    and a job never leaves the host it was submitted to. `gpuc requeue` is
    the other half of that pair: a fresh job id, from the mirrored spec, on
    whichever host you name.
    """
    if priority is not None:
        check_priority(priority)
    done, _ = job_verbs(
        "preempt",
        job_ids,
        host,
        settings,
        args="" if priority is None else f" --priority {priority}",
        mirror=None if priority is None else ("priority", priority),
    )
    return done


def placement_after(session: HostSession, job_id: str, settings: Settings) -> dict[str, Any]:
    """Where the job now sits in the host's queue: what `submit` answers "so
    when does it run" with.

    Asked *after* the enqueue, over the same session, so it is best effort by
    construction: whatever goes wrong here costs a document of nulls, never
    the command's exit code -- the job is queued either way, and a submit that
    printed a traceback over a job it had already enqueued would be worse than
    one that said nothing about the queue.
    """
    view = status_mod.gather(session.entry, settings, session=session)
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


def check_max_runtime(minutes: float | None, *, clear: bool) -> float | None:
    """The wall-clock limit a request asks for, or a usage error before any
    host is asked."""
    if clear is (minutes is not None):
        raise UsageError("give one of --minutes N or --clear")
    if minutes is not None and not 0.0 < minutes < math.inf:
        raise UsageError(f"--minutes must be a positive number of minutes, got {minutes:g}")
    return minutes


def mirror_spec_field(job_id: str, field: str, value: Any, settings: Settings) -> str | None:
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
            f"the host has the new {field}, but its mirrored spec still has the old one, "
            f"so `gpuc requeue {job_id}` would not carry it: {str(exc).splitlines()[0]}"
        )
    return None


def estimate_jobs(
    job_ids: Sequence[str], wanted: float | None, host: str | None, settings: Settings
) -> list[Done]:
    """Add, change or clear jobs' `estimated_runtime_min` after submitting them.

    `wanted` has been through `check_estimate`; None clears the estimate.
    """
    done, _ = job_verbs(
        "estimate",
        job_ids,
        host,
        settings,
        args=" --clear" if wanted is None else f" --minutes {wanted!r}",
        mirror=("estimated_runtime_min", wanted),
    )
    for job in done:
        if job.error is not None:
            continue
        recorded = job.fields.get("estimated_runtime_min")
        if not (recorded is None if wanted is None else isinstance(recorded, (int, float))):
            # Otherwise a host whose answer lacks the key reports a successful
            # *clear* of a job it never touched.
            job.error = (
                f"host {job.host} did not say what estimate it recorded for {job.job_id}: "
                f"{json.dumps(job.fields)[:200]}"
            )
            job.fields = {}
    return done


def max_runtime_jobs(
    job_ids: Sequence[str], wanted: float | None, host: str | None, settings: Settings
) -> list[Done]:
    """Raise, lower or clear jobs' `max_runtime_min` after submitting them.

    `wanted` has been through `check_max_runtime`; None removes the limit.
    """
    done, _ = job_verbs(
        "max-runtime",
        job_ids,
        host,
        settings,
        args=" --clear" if wanted is None else f" --minutes {wanted!r}",
        mirror=("max_runtime_min", wanted),
    )
    for job in done:
        if job.error is not None:
            continue
        recorded = job.fields.get("max_runtime_min", "absent")
        if not (recorded is None if wanted is None else isinstance(recorded, (int, float))):
            # As for the estimate: a host whose answer lacks the key would
            # otherwise report a successful clear of a job it never touched.
            job.error = (
                f"host {job.host} did not say what limit it recorded for {job.job_id}: "
                f"{json.dumps(job.fields)[:200]}"
            )
            job.fields = {}
    return done


@dataclass
class LogText:
    """A job's log and where it was read from, for both output forms."""

    source: str
    """`host` or `s3`."""
    location: str | None
    text: str = ""
    notes: list[str] = field(default_factory=list)
    failure: str | None = None
    """Why this is not the answer: the host may still hold the log and could
    not be asked. The mirror was read anyway (doing as much as it can), but
    the command exits 1 with this. None when the host produced the log, or
    is gone and the mirror is the answer."""

    def document(self, job_id: str, host: str) -> dict[str, Any]:
        return {
            "job_id": job_id,
            "host": host,
            "source": self.source,
            "location": self.location,
            "lines": self.text.splitlines(),
            "notes": self.notes,
        }

    def answer(self, job_id: str, host: str) -> Answer:
        return Answer(
            self.document(job_id, host),
            self.text,
            failures=[self.failure] if self.failure else [],
        )


def read_log(
    job_id: str,
    host: str | None,
    lines: int,
    settings: Settings,
    *,
    report: Callable[[str], None] = note,
) -> tuple[str, LogText]:
    """The tail of a job's log from its host, else from the S3 mirror.

    Host first, always: the mirror is a copy of what the host uploaded last,
    and is read only when the host cannot produce the log. When that is
    because the host is gone (`mirror_is_the_answer`) or the job dir was
    purged, the mirror is the answer; when the host merely could not be
    asked, the mirror is printed and the command still fails, because the
    host may hold a newer log than the copy.
    """
    location = find_job(job_id, host, settings)
    reached = location.asked
    if reached is None:
        entry = location.require_entry()
        reached = ask(entry, None, settings, provider=provider_for([entry], settings))
    remote = None
    purged = False
    failed = False
    if isinstance(reached, Answered):
        session = reached.session
        remote = f"{session.job_dir(job_id)}/log.txt"
        try:
            result = session.transport.tail(remote, lines=lines)
            if result.returncode == 0:
                return location.host, LogText("host", remote, result.stdout)
            purged = job_dir_gone(session, job_id)
            # With --host nothing has checked the id exists. Judged by its shape,
            # not by the indexes: a mirror that cannot be read, or none set up
            # here, would make a job purged elsewhere look like one never made.
            if purged and not jobs.is_job_id(job_id):
                raise NotFound(
                    f"host {location.host} has no job {job_id}, and that is not a job id "
                    f"(they look like 20260917-184548-43bc15); a job's `name` is not one.\n"
                    f"`gpuc status --host {location.host}` lists the ids."
                )
            why = (result.output.strip().splitlines() or ["no log file on the host"])[-1]
        except (RemoteError, TransportError) as exc:
            why, failed = reason_of(exc), True
    else:
        why, failed = reached.reason, not mirror_is_the_answer(reached)
    # A job dir that is gone entirely is what `gpuc clean --purge` does on
    # purpose. Saying "purged" beats printing a `tail: No such file`.
    missing = (
        f"job {job_id} was purged from host {location.host} "
        f"(gpuc clean --purge removes the whole job dir once it is mirrored)"
        if purged
        else f"could not read {remote or 'the host log'}: {why}"
    )
    report(missing)
    log = logs_from_s3(job_id, location, settings, purged=purged, report=report)
    log.notes.insert(0, missing)
    if failed:
        log.failure = missing
    return location.host, log


def job_dir_gone(session: HostSession, job_id: str) -> bool:
    """Is the job dir itself missing, rather than just its log?"""
    try:
        result = session.run(f"test -d {shlex.quote(session.job_dir(job_id))}", timeout=30.0)
    except TransportError:
        return False
    return result.returncode != 0


def logs_from_s3(
    job_id: str,
    location: Location,
    settings: Settings,
    *,
    purged: bool = False,
    report: Callable[[str], None] = note,
) -> LogText:
    index = JobIndex(settings)
    prefix = index.mirror_prefix(job_id, location.entry)
    missing = unmirrored(index, job_id, prefix)
    if index.s3 is None or not prefix:
        purge = (
            f"; its job dir was purged from host {location.host}, so this log no longer exists "
            f"anywhere"
            if purged
            else ""
        )
        raise CliError(f"no S3 mirror to fall back on: {missing}{purge}")
    s3 = index.s3
    uri = job_log_uri(prefix, job_id)
    fallback = f"falling back to the S3 mirror at {uri}"
    report(fallback)
    return LogText("s3", uri, s3.get_uri(uri), [fallback])


@dataclass
class Mirrored:
    """What the mirror holds of a job whose host is gone: its final state and
    where that was read, or -- `view` None -- why there is none, which means
    the job went with its host."""

    view: status_mod.JobView | None = None
    uri: str | None = None
    lost: str = ""


def unmirrored(index: JobIndex, job_id: str, prefix: str | None) -> str | None:
    """Why this job has no mirror to read at all, or None when it has one."""
    if index.s3 is None:
        return f"s3_bucket is unset in {config_file()}, so nothing of job {job_id} is mirrored"
    if not prefix:
        return f"no mirror location is known for job {job_id}"
    return None


def read_mirror(
    index: JobIndex, job_id: str, prefix: str | None, indexed: IndexEntry | None = None
) -> Mirrored:
    """A job's final state from the S3 mirror: what every command answers
    with once the job's host is gone, per the spec's "the mirror is read only
    when the host is gone".

    The one reader of a mirrored `state.json`, through `job_views`, so another
    build's state.json is read as tolerantly here as a host's own answer is.
    The two fields the file cannot carry are supplied: `name` lives in the
    spec, and `outputs_pending` is the host's own check against the spec,
    which is why a mirrored `outputs_lost` has to stand on its own here --
    this is the dead-rental case, and it is the same case that loses outputs.
    A state that is not terminal is no answer: the host went before the job
    ended, or before the mirror caught up, and either way nothing will end it.
    """
    missing = unmirrored(index, job_id, prefix)
    if missing is not None or prefix is None:
        return Mirrored(lost=f"{missing}, so it went with its host")
    uri = job_uri(prefix, job_id, "state.json")
    document = index.mirrored_state(job_id, prefix)
    view = None
    if document is not None:
        indexed = indexed or index.get(job_id)
        _, _, finished = status_mod.job_views(
            {
                "jobs": [
                    {
                        **document,
                        "job_id": job_id,
                        "name": (indexed.name if indexed else "") or "",
                        "outputs_pending": bool(document.get("outputs_lost")),
                        # An earlier build's state has no limit and the spec's
                        # is not read here, so not saying beats saying "none".
                        "max_runtime_min": document.get("max_runtime_min")
                        if document.get("live_max_runtime") is True
                        else None,
                    }
                ]
            }
        )
        view = next((v for v in finished if v.status in FINISHED_STATUSES), None)
    if view is None:
        return Mirrored(
            uri=uri,
            lost=f"the mirror has no final state for job {job_id} at {uri}, so its host went "
            f"before it ended and it never will",
        )
    return Mirrored(view, uri)


def unhosted_jobs(
    index: IndexRead, seen: set[str], host: str | None = None
) -> tuple[list[IndexEntry], dict[str, dict[str, Any]], str | None]:
    """The index's view of jobs no host admitted to having, and whether that is
    all of it: an S3 index that could not be read leaves this list short.

    After a host loses its state -- a container whose $HOME was wiped, a pod
    that is gone -- this is the only list of what was on it, and `gpuc requeue
    <id> --host <name>` is how each one comes back, so `--host H --all` narrows
    it to the host being recovered. "No host admitted to having" includes a
    host that could not be asked, so each entry is labelled with what its
    host was found to be (`StatusResult.host_state`) before anyone acts on it.

    Returns the entries, the mirrored `state.json` of the first
    `MIRROR_STATE_LOOKUPS` of them by id, and why the list may be short (None
    when the index was read in full).
    """
    entries, short = index.all()
    elsewhere = [
        entry
        for job_id, entry in sorted(entries.items())
        if job_id not in seen and (host is None or entry.host == host)
    ]
    if not elsewhere:
        return [], {}, short
    return elsewhere, _mirrored_states(index.index, elsewhere[:MIRROR_STATE_LOOKUPS]), short


MIRROR_STATE_LOOKUPS = 25
"""How many jobs a `status` reads `state.json` for, per list it reads them
for. One GET each, and the answer matters most for the handful at the top."""


def _mirrored_states(index: JobIndex, entries: Sequence[IndexEntry]) -> dict[str, dict[str, Any]]:
    """Each job's mirrored `state.json`, by id. Best effort: a job whose
    state.json is missing or unreadable is left out, which is a note missing
    from a listing and nothing more."""
    found: dict[str, dict[str, Any]] = {}
    for entry in entries:
        document = index.mirrored_state(entry.job_id, entry.s3_prefix)
        if document is not None:
            found[entry.job_id] = document
    return found
