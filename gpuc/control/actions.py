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
    HostSession,
    PodDead,
    PodGone,
    RemoteError,
    Unreachable,
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
    lost: set[str] = field(default_factory=set)
    """Which of `unhosted` the mirror records as having lost their outputs."""
    index_error: str | None = None
    """Why `unhosted` may be short: an S3 index that could not be read."""

    @property
    def errors(self) -> list[str]:
        """The problems that belong to no host: the registry's, the index's."""
        errors = list(self.read.errors)
        if self.read.unreadable:
            errors.append(
                f"{hosts_file()} could not be read, so `hosts` is empty because nothing is known"
            )
        if self.index_error:
            errors.append(f"{self.index_error}; the list of index-only jobs may be short")
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

    def host_state(self, entry: IndexEntry) -> str:
        """What this run found the job's host to be: one of `HostState`, or
        `NOT_REGISTERED`. Only a host that answered and does not list the job,
        or is gone, makes the job the index's alone; one that could not be
        asked may still be running it."""
        view = next((view for view in self.views if view.entry.name == entry.host), None)
        return view.state.value if view is not None else status_mod.NOT_REGISTERED

    def document(
        self, *, recent: int = status_mod.RECENT_FINISHED, since_s: float | None = None
    ) -> dict[str, Any]:
        return status_mod.document(
            self.views,
            errors=self.errors,
            unhosted=[
                status_mod.unhosted_json(
                    entry, lost=entry.job_id in self.lost, host_state=self.host_state(entry)
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
                entry, lost=entry.job_id in self.lost, host_state=self.host_state(entry)
            )
            for entry in self.unhosted
        ]
        # The hint names a job whose host cannot still be running it: a
        # requeue offered over a connection error is a run done twice.
        first = next(
            (e for e in self.unhosted if self.host_state(e) in status_mod.REQUEUEABLE), None
        )
        if first is not None:
            # The host is named only when it answered: it is there and empty,
            # which is the wiped-home case; a gone or unregistered host is not
            # a place to send anything.
            answered = self.host_state(first) == status_mod.HostState.ANSWERED.value
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


def status(
    read: RegistryRead,
    settings: Settings,
    *,
    host: str | None = None,
    all_jobs: bool = False,
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
    """
    result = StatusResult(read, every_host=host is None)
    entries = read.registry.listing(host) if not read.unreadable else []
    provider = provider_for(entries, settings, report) if entries else None
    for view in gather_all(entries, settings, provider):
        result.views.append(view)
        if on_view is not None:
            on_view(view)
    if all_jobs:
        result.unhosted, result.lost, result.index_error = unhosted_jobs(
            settings, result.seen, host
        )
    return result


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
        if view.pod_gone:
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


@dataclass
class Forgotten:
    """The index names a host this machine has no record of: the ordinary end
    of a rental, whose entry `status` forgot once the pod was gone. Nothing
    can be asked, so the mirror is the answer, as for `PodGone`."""

    host: str

    @property
    def reason(self) -> str:
        return (
            f"host {self.host} is not registered on this machine (a rental that ended is "
            f"forgotten), so it cannot be asked"
        )


Trouble = Unreachable | PodDead | PodGone | Forgotten
"""Why a job's host could not confirm it holds the job."""


def mirror_is_the_answer(trouble: Trouble) -> bool:
    """The one rule `logs` and `wait` read the mirror by.

    The spec reads the mirror only when the host is *gone*: a rental whose
    pod the provider no longer has, or a host the index names that this
    machine has forgotten. For those the mirror is the answer, now, and
    nothing failed. A host that is merely unreachable, or a pod that is
    stopped but still there, may still hold the job: `wait` keeps asking
    for `TROUBLE_GRACE_S` before it reads the mirror, and `logs` prints what
    the mirror has -- doing as much as it can -- but as a failure, exit 1
    with the reason, never as the answer.
    """
    return isinstance(trouble, (PodGone, Forgotten))


@dataclass
class Location:
    """Where a job is: its host, its index entry, and what the host said when
    it was asked to find it (None when nothing had to be asked). `entry` is
    None only for a host this machine has forgotten (`Forgotten`)."""

    entry: HostEntry | None
    index: IndexEntry | None = None
    asked: Asked | Forgotten | None = None

    @property
    def host(self) -> str:
        if self.entry is not None:
            return self.entry.name
        assert isinstance(self.asked, Forgotten)
        return self.asked.host

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
                f"job's host {self.host} is gone: {self.trouble_reason}\n"
                f"`gpuc logs` reads its mirror; `gpuc requeue <id> --host <name>` re-runs it."
            )
        return self.entry


def locate(
    job_id: str,
    registry: Registry,
    explicit: str | None,
    settings: Settings | None = None,
    *,
    provider: Provider | None = None,
) -> Location:
    """The host a job is on: `--host` if given, else the index (local, then
    the mirror), else whichever registered host admits to it.

    A name in the mirror's index is the *submitting* client's name for the
    host, which need not be this machine's: it is asked first, not believed.
    A host the index names that cannot be asked is still the answer -- the job
    is there as far as anything knows -- carried as `trouble` for the caller
    to judge, never hidden. So is a host the index names that this machine
    has no entry for (`Forgotten`): that is a rental that ended and was
    forgotten, the ordinary end of one, and `gpuc logs` and `gpuc wait` on
    its jobs must reach the mirror rather than "no such job". An id no
    answering host knows is exit 4 only when every host answered: with one
    unreachable, the job may well be on it, and "no such job" would be a lie
    told over a connection error.
    """
    settings = settings if settings is not None else load_settings()
    if explicit:
        # The local index only: the caller already knows the host, and the
        # entry is a convenience for whoever wants the mirror prefix.
        return Location(registry.require(explicit), LocalIndex().get(job_id))
    local = LocalIndex().get(job_id)
    if local is not None and local.host in registry.hosts:
        return Location(registry.hosts[local.host], local)
    index = local or JobIndex(settings).get(job_id)
    named = registry.hosts.get(index.host) if index is not None else None
    rest = [entry for entry in registry.hosts.values() if entry is not named]
    provider = provider or provider_for(list(registry.hosts.values()), settings)
    named_trouble: Asked | None = None
    unasked: list[str] = []
    for entry in [named, *rest] if named is not None else rest:
        asked = ask(entry, f"status {shlex.quote(job_id)}", settings, provider=provider)
        if isinstance(asked, Answered):
            if (asked.payload or {}).get("jobs"):
                return Location(entry, index, asked)
            continue
        if entry is named:
            named_trouble = asked
        elif not isinstance(asked, PodGone):
            unasked.append(f"{entry.name}: {asked.reason}")
    if named is not None and named_trouble is not None:
        return Location(named, index, named_trouble)
    if index is not None and named is None and not unasked:
        return Location(None, index, Forgotten(index.host))
    if unasked:
        raise CliError(
            f"no host that answered knows job {job_id}, and these could not be asked:\n"
            + "\n".join(f"  {line}" for line in unasked)
            + "\nPass --host <name>, or check `gpuc host list` and `gpuc status --all`."
        )
    raise NotFound(
        f"no registered host knows job {job_id}.\n"
        f"Pass --host <name>, or check `gpuc host list` and `gpuc status --all`."
    )


def job_verb(
    verb: str,
    job_id: str,
    host: str | None,
    settings: Settings,
    *,
    args: str = "",
    mirror: tuple[str, Any] | None = None,
) -> tuple[Location, HostSession, dict[str, Any], list[str]]:
    """Run one on-host verb against one job: the path every job command takes.

    Find the host, ask it over one session, insist on a verdict, and put any
    spec change in the mirror too. Returns the location, the session (for a
    follow-up question like the queue placement), the host's document and
    the warnings so far.

    `check=False` on the host call: a refusal (a finished job, an id this host
    does not know) *is* the host's document, and raising on the exit code
    would throw away the reason it gave. A refusal that says `missing` is
    the host answering "no such job", exit 4 like every other unknown name,
    and told apart from a refusal of a job that is there by that key rather
    than by the words. An answer with no `status` is an error too: reporting
    success for a job the host never touched is worse than any exception.

    `mirror` is `(spec field, value)`: `requeue` submits what S3 holds, so a
    priority or estimate changed on the host and not in the mirror would hand
    a re-run back with the old one, silently. A mirror that cannot be updated
    is a warning, never a failure: the change is already where `status` reads
    it, which is what was asked for.
    """
    registry = open_registry().named()
    location = locate(job_id, registry, host, settings)
    trouble = location.trouble
    if location.entry is None or trouble is not None:
        gone = trouble is not None and mirror_is_the_answer(trouble)
        raise CliError(
            f"job {job_id} is on host {location.host}, which "
            f"{'is gone' if gone else 'could not be asked'}: {location.trouble_reason}"
        )
    entry = location.entry
    asked = ask(
        entry,
        f"{verb} {shlex.quote(job_id)}{args}",
        settings,
        provider=provider_for([entry], settings),
        session=location.session,
        check=False,
    )
    if not isinstance(asked, Answered):
        raise CliError(
            f"job {job_id} is on host {entry.name}, which could not be asked: {asked.reason}"
        )
    document = asked.payload or {}
    if document.get("missing"):
        raise NotFound(
            f"host {entry.name} has no job {job_id}: {document.get('error')}\n"
            f"Check the id with `gpuc status --all`."
        )
    if document.get("error"):
        raise CliError(f"host {entry.name} did not {verb} {job_id}: {document['error']}")
    if not document.get("status"):
        raise CliError(
            f"host {entry.name} did not say what it did with {job_id}: {json.dumps(document)[:200]}"
        )
    warnings: list[str] = []
    if mirror is not None:
        warning = mirror_spec_field(job_id, mirror[0], mirror[1], settings, what=mirror[0])
        if warning:
            warnings.append(warning)
    return location, asked.session, document, warnings


def cancel_job(job_id: str, host: str | None, settings: Settings) -> dict[str, Any]:
    location, _, document, _ = job_verb("cancel", job_id, host, settings)
    # The host's own word for what it did: `cancelled` for a queued job it
    # dequeued, `cancelling` for a running one whose runner has been marked.
    return {"job_id": job_id, "host": location.host, "status": document["status"]}


def check_priority(priority: int) -> None:
    if not 0 <= priority <= 99:
        raise UsageError(f"priority must be 0-99 (lower dispatches first), got {priority}")


def reorder_job(job_id: str, priority: int, host: str | None, settings: Settings) -> dict[str, Any]:
    check_priority(priority)
    location, session, _, warnings = job_verb(
        "reorder", job_id, host, settings, args=f" {priority}", mirror=("priority", priority)
    )
    return {
        "job_id": job_id,
        "host": location.host,
        "priority": priority,
        "warnings": warnings,
        **placement_after(session, job_id, settings),
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
    location, _, document, warnings = job_verb(
        "preempt",
        job_id,
        host,
        settings,
        args="" if priority is None else f" --priority {priority}",
        mirror=None if priority is None else ("priority", priority),
    )
    return {
        "job_id": job_id,
        "host": location.host,
        "status": document["status"],
        "priority": document.get("priority"),
        "warnings": warnings,
    }


def placement_after(session: HostSession, job_id: str, settings: Settings) -> dict[str, Any]:
    """Where the job now sits in the host's queue: what `submit` and `reorder`
    answer "so when does it run" with.

    Asked *after* the enqueue or the move, over the same session, so it is
    best effort by construction: whatever goes wrong here costs a document of
    nulls, never the command's exit code -- the job is queued either way, and
    a submit that printed a traceback over a job it had already enqueued would
    be worse than one that said nothing about the queue.
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
    location, _, document, warnings = job_verb(
        "estimate",
        job_id,
        host,
        settings,
        args=" --clear" if wanted is None else f" {wanted!r}",
        mirror=("estimated_runtime_min", wanted),
    )
    recorded = document.get("estimated_runtime_min")
    expected = recorded is None if wanted is None else isinstance(recorded, (int, float))
    if not expected:
        # Otherwise a host whose answer lacks the key reports a successful
        # *clear* of a job it never touched.
        raise CliError(
            f"host {location.host} did not say what estimate it recorded for {job_id}: "
            f"{json.dumps(document)[:200]}"
        )
    if document.get("warning"):
        warnings.insert(0, str(document["warning"]))
    return {
        "job_id": job_id,
        "host": location.host,
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
    location = locate(job_id, open_registry().named(), host, settings)
    entry = location.entry
    reached: Asked | Forgotten
    if entry is None:
        assert isinstance(location.asked, Forgotten)
        reached = location.asked
    else:
        reached = location.asked or ask(
            entry, None, settings, provider=provider_for([entry], settings)
        )
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
    s3 = S3Index.from_settings(settings)
    prefix = JobIndex(settings).mirror_prefix(job_id, location.entry)
    if s3 is None or not prefix:
        gone = (
            f"Its job dir was purged from host {location.host}, so this log no longer exists "
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


def mirrored_outcome(
    index: JobIndex, job_id: str, prefix: str | None
) -> tuple[status_mod.JobView, str] | None:
    """A job's terminal state from the S3 mirror, and where it was read, or
    None when the mirror has no terminal state for it.

    The one reader of a mirrored `state.json`: what `gpuc wait` and `gpuc logs
    -f` fall back to once a host is gone, per the spec's "the mirror is read
    only when the host is gone". Through `job_views`, so another build's
    state.json is read as tolerantly here as a host's own answer is. The two
    fields the file cannot carry are supplied: `name` lives in the spec, and
    `outputs_pending` is the host's own check against the spec, which is why a
    mirrored `outputs_lost` has to stand on its own here -- this is the
    dead-rental case, and it is the same case that loses outputs.
    """
    document = index.mirrored_state(job_id, prefix)
    if document is None or not prefix:
        return None
    indexed = index.get(job_id)
    _, _, finished = status_mod.job_views(
        {
            "jobs": [
                {
                    **document,
                    "job_id": job_id,
                    "name": (indexed.name if indexed else "") or "",
                    "outputs_pending": bool(document.get("outputs_lost")),
                }
            ]
        }
    )
    view = next((v for v in finished if v.status in FINISHED_STATUSES), None)
    if view is None:
        return None
    return view, job_uri(prefix, job_id, "state.json")


def unhosted_jobs(
    settings: Settings, seen: set[str], host: str | None = None
) -> tuple[list[IndexEntry], set[str], str | None]:
    """The index's view of jobs no host admitted to having, and whether that is
    all of it: an S3 index that could not be read leaves this list short.

    After a host loses its state -- a container whose $HOME was wiped, a pod
    that is gone -- this is the only list of what was on it, and `gpuc requeue
    <id> --host <name>` is how each one comes back, so `--host H --all` narrows
    it to the host being recovered. "No host admitted to having" includes a
    host that could not be asked, so each entry is labelled with what its
    host was found to be (`StatusResult.host_state`) before anyone acts on it.

    Returns the entries, the ids among them whose outputs the mirror records as
    lost, and why the list may be short (None when the index was read in full).
    """
    index = JobIndex(settings)
    entries, short = index.all()
    elsewhere = [
        entry
        for job_id, entry in sorted(entries.items())
        if job_id not in seen and (host is None or entry.host == host)
    ]
    if not elsewhere:
        return [], set(), short
    return elsewhere, _outputs_lost_ids(index, elsewhere[:MIRROR_STATE_LOOKUPS]), short


MIRROR_STATE_LOOKUPS = 25
"""How many index-only jobs `--all` reads `state.json` for. One GET each, and
the answer (did this job's outputs make it off the host?) matters most for the
handful at the top of a recovery list."""


def _outputs_lost_ids(index: JobIndex, entries: Sequence[IndexEntry]) -> set[str]:
    """Which of these jobs the mirror records as having lost their outputs.

    Best effort: a job whose state.json is missing or unreadable simply does not
    get the flag, because this is a note on a listing, not a decision.
    """
    return {
        entry.job_id
        for entry in entries
        if (document := index.mirrored_state(entry.job_id, entry.s3_prefix))
        and document.get("outputs_lost")
    }
