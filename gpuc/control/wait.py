"""Blocking on a job until its host says it has ended.

`gpuc wait` is this loop on its own; `gpuc logs -f` is this loop with a
`tail -f` running beside it. Both used to be somebody's shell script around
`gpuc status --json`, which is the reason it lives here once.

Nothing on a host knows a client is waiting. The host owns the job from the
moment it accepts it, so a wait that is killed, or a client that goes away,
changes nothing about the run -- and a wait is therefore free to be as
impatient or as forgiving as it likes about a host it cannot reach.
"""

from __future__ import annotations

import shlex
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from gpuc.control import status as status_mod
from gpuc.control.actions import (
    EXIT_ERROR,
    EXIT_NOT_FOUND,
    EXIT_OK,
    Answer,
    NotFound,
    locate,
    mirror_is_the_answer,
    mirrored_outcome,
    provider_for,
)
from gpuc.control.config import HostEntry, Reporter, Settings, load_settings, open_registry
from gpuc.control.jsonout import note
from gpuc.control.providers.base import Provider
from gpuc.control.remote import Answered, HostSession, ask, open_session
from gpuc.control.s3index import JobIndex
from gpuc.control.status import JobView
from gpuc.host.jobs import FINISHED_STATUSES

FIRST_INTERVAL_S = 2.0
MAX_INTERVAL_S = 30.0
BACKOFF = 1.5
"""How the poll paces itself when `--interval` does not pin it.

A job that dies in its preflight dies in the first minute, and that is the
whole point of waiting at all, so the first polls are close together. A job
that is still going after ten minutes is a job somebody is waiting hours for,
and one round trip every 30s is enough for it -- the alternative is thousands
of ssh invocations across an afternoon to learn nothing.
"""

TROUBLE_GRACE_S = 300.0
"""How long a host may stay unaskable before the jobs on it are given up on.

An ssh blip must not end a six-hour wait, and a pod that has gone away must not
hang one for ever. Five minutes is long enough that everything transient has
been ridden out and short enough that a dead host is reported the same
afternoon.
"""

FLUSH_GRACE_S = 2.0
"""What `logs -f` gives the stream to catch up before it stops it.

The runner logs the outcome line before its terminal write, but what it
says about the secrets file comes after it, so a poll that sees `succeeded`
can be a little ahead of the log's last lines. A runner slower than this
loses those lines from the stream, never from the log itself.
"""


@dataclass
class Watched:
    """One job being waited on, and the last thing its host said about it."""

    job_id: str
    host: str
    view: JobView | None = None
    error: str | None = None
    """Why this job stopped being waited for without reaching a terminal state.

    The view may still hold what the host last said, which is `running` for a
    job whose host went away mid-run: a consumer that needs "did this job end"
    reads this, not `status`."""
    seen: bool = False
    missing: bool = False
    """The host answered and does not have this job at all."""
    mirror_prefix: str | None = None
    """The host's `s3_prefix`, for the links in `document()` and for the last
    resort when the host itself is gone."""
    source: str = "host"
    """`host`, or `mirror` for an outcome read from S3 after the host went."""

    @property
    def status(self) -> str:
        return self.view.status if self.view else "unknown"

    @property
    def finished(self) -> bool:
        return self.status in FINISHED_STATUSES

    @property
    def settled(self) -> bool:
        """Nothing more will be learned by asking again."""
        return self.finished or self.error is not None

    @property
    def succeeded(self) -> bool:
        return self.status == "succeeded"

    def line(self) -> str:
        """The one line a wait prints per job: outcome, detail, how long it took."""
        job = self.view
        if self.error is not None or job is None:
            return f"job {self.job_id} on {self.host}: {self.error or 'unknown'}"
        # `cancelled (cancelled)` says nothing twice, as in `gpuc status`.
        detail = job.reason if job.reason and job.reason != job.status else ""
        if not detail and job.exit_code:
            detail = f"exit {job.exit_code}"
        took = (
            ""
            if job.minutes is None
            else f" after {status_mod.format_duration(job.minutes * 60.0)}"
        )
        flag = "  OUTPUTS LOST" if job.outputs_lost and job.outputs_pending else ""
        # Said, because it changes what the answer is worth: the mirror holds
        # what the host uploaded last, and a host that died mid-upload uploaded
        # nothing after that.
        whence = " (from the S3 mirror)" if self.source == "mirror" else ""
        return (
            f"{status_mod.job_label(job)} on {self.host}: {job.status}"
            f"{f' ({detail})' if detail else ''}{took}{flag}{whence}"
        )

    def document(self) -> dict[str, Any]:
        """`status --json`'s job shape, plus where this answer came from.

        `error` is null for a job that ended; when it is not, `status` is
        whatever the host last managed to say and is **not** a final state.
        """
        body: dict[str, Any] = (
            {"job_id": self.job_id, "status": None}
            if self.view is None
            else status_mod.job_json(self.view, self.mirror_prefix)
        )
        return {**body, "host": self.host, "source": self.source, "error": self.error}


class Watch:
    """The jobs a wait is blocking on, and one `status` round trip per host.

    One call per host per round rather than one per job: a sweep is usually
    twenty ids on one box, and the host's own `status` already answers for all
    of them at once.
    """

    def __init__(
        self,
        targets: Sequence[tuple[str, str, HostEntry | None]],
        settings: Settings | None = None,
        *,
        report: Reporter = note,
        provider: Provider | None = None,
        sessions: dict[str, HostSession] | None = None,
        gone: dict[str, str] | None = None,
    ) -> None:
        """`targets` is `(job_id, host name, entry)`, the entry None for a host
        this machine has forgotten; `gone` is the reason each host the locator
        already found gone is not going to be asked at all."""
        # Resolved once here rather than per use: the mirror fallback needs a
        # real `Settings` to find a bucket in, and a watch outlives many reads.
        self.settings = settings if settings is not None else load_settings()
        self.report = report
        self.provider = provider
        self.index = JobIndex(self.settings)
        self.entries = {host: entry for _, host, entry in targets if entry is not None}
        self.gone = dict(gone or {})
        self.jobs = {
            job_id: Watched(job_id, host, mirror_prefix=self.index.mirror_prefix(job_id, entry))
            for job_id, host, entry in targets
        }
        self.by_host: dict[str, list[str]] = {}
        for job_id, host, _ in targets:
            self.by_host.setdefault(host, []).append(job_id)
        self._sessions: dict[str, HostSession] = dict(sessions or {})
        self._trouble: dict[str, tuple[float, str]] = {}
        self._dispatcher_warned: set[str] = set()
        self._announced: set[str] = set()
        self.polled = False

    @property
    def pending(self) -> list[Watched]:
        return [watched for watched in self.jobs.values() if not watched.settled]

    def poll(self) -> list[Watched]:
        """Ask every host once. Returns the jobs that settled in this round."""
        for name in self.by_host:
            self._poll_host(name)
        self.polled = True
        settled = [
            watched
            for watched in self.jobs.values()
            if watched.settled and watched.job_id not in self._announced
        ]
        self._announced.update(watched.job_id for watched in settled)
        return settled

    def check_known(self) -> None:
        """Refuse an id its host has never heard of, after the first poll.

        `locate` believes the local index and an explicit `--host` without
        asking anybody, so this is where a typo'd id -- or one whose host lost
        its state -- becomes exit 4 rather than a wait that never ends. For
        `logs -f`, which follows one job; a `wait` on several carries on with
        the rest and reports the missing one in its answer.
        """
        missing = [watched.job_id for watched in self.jobs.values() if watched.missing]
        if not missing:
            return
        raise NotFound(
            f"no host has job {', '.join(missing)}.\nCheck the id with `gpuc status --all`."
        )

    def troubled(self, name: str) -> bool:
        """Whether the last poll of this host got no answer."""
        return name in self._trouble

    def session(self, name: str) -> HostSession:
        """The one session this watch holds for a host, opened on demand.

        Public because `logs -f` needs the same host to run its `tail` on, and
        opening a second one costs another round trip to resolve the same home.
        A poll's session never writes the registry (`record=False`).
        """
        session = self._sessions.get(name)
        if session is None:
            session = open_session(self.entries[name], self.settings, record=False)
            self._sessions[name] = session
        return session

    def _poll_host(self, name: str) -> None:
        pending = [
            self.jobs[job_id] for job_id in self.by_host[name] if not self.jobs[job_id].settled
        ]
        if not pending:
            return
        if name in self.gone:
            self._trouble_with(name, pending, self.gone[name], gone=True)
            return
        request = f"status {shlex.quote(pending[0].job_id)}" if len(pending) == 1 else "status"
        asked = ask(
            self.entries[name],
            request,
            self.settings,
            provider=self.provider,
            session=self._sessions.get(name),
        )
        if not isinstance(asked, Answered):
            # Dropped, not kept: a session holds a resolved home and a
            # ControlMaster that may be exactly what broke.
            self._sessions.pop(name, None)
            self._trouble_with(name, pending, asked.reason, gone=mirror_is_the_answer(asked))
            return
        self._sessions[name] = asked.session
        self._clear_trouble(name)
        view = status_mod.parse_status(self.entries[name], asked)
        views = {job.job_id: job for job in view.queue + view.running + view.finished}
        waiting_to_start = [
            w for w in pending if (job := views.get(w.job_id)) and job.status == "queued"
        ]
        self._check_dispatcher(view, bool(waiting_to_start))
        for watched in pending:
            job = views.get(watched.job_id)
            if job is None:
                self._vanished(name, watched)
                continue
            watched.view = job
            watched.seen = True

    def _check_dispatcher(self, view: status_mod.HostView, queued: bool) -> None:
        """Say so, once, when a reachable host has nobody serving its queue.

        Otherwise a queued job waits for a dispatcher that is never coming back
        and the wait looks identical to one behind a long job. Not an error: the
        job really is still queued, and one `gpuc host bootstrap` starts it.

        Only where something we are waiting on is actually queued: a *running*
        job is written to its end by its own runner, so the wait finishes
        whatever the dispatcher is doing and the warning would be false.
        """
        name = view.entry.name
        if not queued or view.dispatcher_alive or name in self._dispatcher_warned:
            return
        self._dispatcher_warned.add(name)
        self.report(
            f"host {name}: dispatcher DOWN, so nothing there will start a queued job. "
            f"Still waiting; `gpuc host bootstrap {name}` restarts it"
        )

    def _vanished(self, name: str, watched: Watched) -> None:
        if not watched.seen:
            watched.missing = True
            watched.error = f"host {name} has no job {watched.job_id}"
            return
        # It was there and is not now: a purge, or a host that lost its state.
        # Treated as trouble rather than an answer, because the job may well
        # have finished and `gpuc status --all` is the place to find out.
        self._trouble_with(name, [watched], f"no longer knows job {watched.job_id}")

    def _trouble_with(
        self, name: str, pending: Iterable[Watched], why: str, *, gone: bool = False
    ) -> None:
        now = time.monotonic()
        since, said = self._trouble.get(name, (now, ""))
        if said != why:
            self.report(f"host {name}: {why}; still waiting")
        self._trouble[name] = (since, why)
        # A host that is gone (`mirror_is_the_answer`) is not going to answer,
        # however long we wait: it is the one case the mirror exists for, so
        # it is read now rather than after the grace period.
        if not gone and now - since < TROUBLE_GRACE_S:
            return
        waited = status_mod.format_duration(now - since)
        for watched in pending:
            # The mirror before giving up, and only now: the spec's rule is
            # that monitoring asks the host and reads the mirror when the host
            # is gone. A rental that idled itself down after finishing the job
            # is exactly that, and reporting its success as "could not ask"
            # would be wrong about the one run the user was waiting for.
            if self._from_mirror(watched):
                continue
            watched.error = (
                f"host {name} is gone and the mirror has no final state for this job: {why}"
                if gone
                else f"host {name} could not be asked for {waited}: {why}"
            )

    def _from_mirror(self, watched: Watched) -> bool:
        """This job's outcome from S3, if the mirror has a terminal one."""
        found = mirrored_outcome(self.index, watched.job_id, watched.mirror_prefix)
        if found is None:
            return False
        view, uri = found
        self.report(f"read {watched.job_id} from the mirror at {uri}")
        watched.view = view
        watched.source = "mirror"
        return True

    def _clear_trouble(self, name: str) -> None:
        if self._trouble.pop(name, None) is not None:
            self.report(f"host {name} is answering again")


def start(
    job_ids: Sequence[str],
    host: str | None,
    settings: Settings | None = None,
    *,
    report: Reporter = note,
) -> Watch:
    """Resolve each id to a host. A host asked on the way is kept: its
    session is the one the poll goes on using."""
    registry = open_registry().named()
    settings = settings if settings is not None else load_settings()
    provider = provider_for(list(registry.hosts.values()), settings, report)
    targets: list[tuple[str, str, HostEntry | None]] = []
    sessions: dict[str, HostSession] = {}
    gone: dict[str, str] = {}
    # Deduplicated, so `xargs gpuc wait` on a list with a repeat in it
    # neither polls twice nor prints the outcome twice.
    for job_id in dict.fromkeys(job_ids):
        location = locate(job_id, registry, host, settings, provider=provider)
        targets.append((job_id, location.host, location.entry))
        if location.session is not None:
            sessions[location.host] = location.session
        trouble = location.trouble
        if trouble is not None and mirror_is_the_answer(trouble):
            gone[location.host] = trouble.reason
    return Watch(targets, settings, report=report, provider=provider, sessions=sessions, gone=gone)


def document(waited: Sequence[Watched]) -> dict[str, Any]:
    """`gpuc wait --json`: every job's final state, and what went wrong."""
    return {
        "jobs": [job.document() for job in waited],
        "errors": [job.error for job in waited if job.error is not None],
    }


def block(
    watch: Watch,
    *,
    interval: float | None = None,
    on_settled: Callable[[Watched], None] = lambda _: None,
    each_round: Callable[[], None] = lambda: None,
) -> list[Watched]:
    """Poll until every job has settled, announcing each one as it does.

    `interval` pins the poll; without it the pace backs off from
    `FIRST_INTERVAL_S` to `MAX_INTERVAL_S`. A watch that has already been
    polled -- `logs -f` asks once before it opens a stream -- sleeps before
    asking again rather than sending two round trips back to back.
    `each_round` is anything else the caller wants checked at the poll's pace,
    which for `logs -f` is whether its stream is still alive.
    """
    delay = FIRST_INTERVAL_S if interval is None else interval
    while True:
        if watch.polled:
            time.sleep(delay)
            if interval is None:
                delay = min(delay * BACKOFF, MAX_INTERVAL_S)
        settled = watch.poll()
        each_round()
        for watched in settled:
            on_settled(watched)
        if not watch.pending:
            return list(watch.jobs.values())


def answer(waited: Sequence[Watched], text: str | None = None) -> Answer:
    """A wait's answer carries the jobs' outcome, not its own: 0 only if every
    one of them succeeded.

    The one place `gpuc` uses exit 1 for something other than its own failure,
    and deliberately -- `gpuc ssh <host> -- cmd` does the same with the remote
    command's code, and it is what makes a wait usable in a script. An id no
    host has is exit 4, as everywhere else -- after the others have been
    waited for and reported, because one typo in a list of twenty must not
    throw away the nineteen outcomes.
    """
    if any(job.missing for job in waited):
        outcome = EXIT_NOT_FOUND
    elif all(job.succeeded for job in waited):
        outcome = EXIT_OK
    else:
        outcome = EXIT_ERROR
    return Answer(document(waited), text, outcome=outcome)
