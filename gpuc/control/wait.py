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

import json
import shlex
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from gpuc.control import status as status_mod
from gpuc.control.actions import (
    EXIT_ERROR,
    EXIT_OK,
    NotFound,
    find_job_host,
    named_registry,
    note,
)
from gpuc.control.config import ConfigError, HostEntry, Reporter, Settings, load_settings
from gpuc.control.remote import HostSession, RemoteError, open_session
from gpuc.control.s3index import LocalIndex, S3Index, S3IndexError, S3ObjectMissing, job_uri
from gpuc.control.status import JobView
from gpuc.control.transport import TransportError
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

The runner writes its terminal state *then* logs the outcome line and whatever
its workdir cleanup has to say, so a poll that sees `succeeded` is by
construction a little ahead of the log. A cleanup that takes longer than this
loses its last lines from the stream, never from the log itself.
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
        whence = " (from the S3 mirror; the host is gone)" if self.source == "mirror" else ""
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
        targets: Sequence[tuple[str, HostEntry]],
        settings: Settings | None = None,
        *,
        report: Reporter = note,
    ) -> None:
        # Resolved once here rather than per use: the mirror fallback needs a
        # real `Settings` to find a bucket in, and a watch outlives many reads.
        self.settings = settings if settings is not None else load_settings()
        self.report = report
        self.entries = {entry.name: entry for _, entry in targets}
        self.jobs = {
            job_id: Watched(job_id, entry.name, mirror_prefix=mirror_prefix(job_id, entry))
            for job_id, entry in targets
        }
        self.by_host: dict[str, list[str]] = {}
        for job_id, entry in targets:
            self.by_host.setdefault(entry.name, []).append(job_id)
        self._sessions: dict[str, HostSession] = {}
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
        """Refuse ids their host has never heard of, after the first poll.

        `find_job_host` believes the local index and an explicit `--host`
        without asking anybody, so this is where a typo'd id -- or one whose
        host lost its state -- becomes exit 4 rather than a wait that never
        ends.
        """
        missing = [watched.job_id for watched in self.jobs.values() if watched.missing]
        if not missing:
            return
        raise NotFound(
            f"no host has job {', '.join(missing)}.\nCheck the id with `gpuc status --all`."
        )

    def session(self, name: str) -> HostSession:
        """The one session this watch holds for a host, opened on demand.

        Public because `logs -f` needs the same host to run its `tail` on, and
        opening a second one costs another round trip to resolve the same home.
        """
        session = self._sessions.get(name)
        if session is None:
            session = open_session(self.entries[name], self.settings)
            self._sessions[name] = session
        return session

    def _poll_host(self, name: str) -> None:
        pending = [
            self.jobs[job_id] for job_id in self.by_host[name] if not self.jobs[job_id].settled
        ]
        if not pending:
            return
        request = f"status {shlex.quote(pending[0].job_id)}" if len(pending) == 1 else "status"
        try:
            payload = self.session(name).host_json(request, timeout=60.0)
        except (RemoteError, TransportError, ConfigError, OSError) as exc:
            # Dropped, not kept: a session holds a resolved home and a
            # ControlMaster that may be exactly what broke.
            self._sessions.pop(name, None)
            self._trouble_with(name, pending, str(exc).splitlines()[0])
            return
        if not isinstance(payload, dict):
            kind = type(payload).__name__
            self._trouble_with(name, pending, f"answered `status` with {kind}, not JSON")
            return
        self._clear_trouble(name)
        self._check_dispatcher(name, payload)
        views = {view.job_id: view for group in status_mod.job_views(payload) for view in group}
        for watched in pending:
            view = views.get(watched.job_id)
            if view is None:
                self._vanished(name, watched)
                continue
            watched.view = view
            watched.seen = True

    def _check_dispatcher(self, name: str, payload: dict[str, Any]) -> None:
        """Say so, once, when a reachable host has nobody serving its queue.

        Otherwise a queued job waits for a dispatcher that is never coming back
        and the wait looks identical to one behind a long job. Not an error: the
        job really is still queued, and one `gpuc host bootstrap` starts it.
        """
        # Another build's JSON: a string here must cost a missing warning, not
        # the whole wait.
        age = payload.get("dispatcher_heartbeat_age_s")
        alive = isinstance(age, (int, float)) and float(age) < status_mod.HEARTBEAT_STALE_S
        if alive or name in self._dispatcher_warned:
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

    def _trouble_with(self, name: str, pending: Iterable[Watched], why: str) -> None:
        now = time.monotonic()
        since, said = self._trouble.get(name, (now, ""))
        if said != why:
            self.report(f"host {name}: {why}; still waiting")
        self._trouble[name] = (since, why)
        if now - since < TROUBLE_GRACE_S:
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
            watched.error = f"host {name} could not be asked for {waited}: {why}"

    def _from_mirror(self, watched: Watched) -> bool:
        """This job's outcome from S3, if the mirror has a terminal one."""
        s3 = S3Index.from_settings(self.settings)
        if s3 is None or not watched.mirror_prefix:
            return False
        uri = job_uri(watched.mirror_prefix, watched.job_id, "state.json")
        try:
            document = json.loads(s3.get_uri(uri))
        except (S3IndexError, S3ObjectMissing, json.JSONDecodeError, OSError):
            return False
        if not isinstance(document, dict):
            return False
        # Through `job_views`, so another build's state.json is read as
        # tolerantly here as a host's own answer is.
        _, _, finished = status_mod.job_views({"jobs": [{**document, "job_id": watched.job_id}]})
        view = next((v for v in finished if v.status in FINISHED_STATUSES), None)
        if view is None:
            return False
        self.report(f"host {watched.host} is gone; read {watched.job_id} from {uri}")
        watched.view = view
        watched.source = "mirror"
        return True

    def _clear_trouble(self, name: str) -> None:
        if self._trouble.pop(name, None) is not None:
            self.report(f"host {name} is answering again")


def mirror_prefix(job_id: str, entry: HostEntry) -> str | None:
    """Where this job's own mirror is: the index's answer, else the host's.

    The job's is the one that counts -- a host whose `s3_prefix` changed after
    the job ran still has the old jobs under the old prefix.
    """
    indexed = LocalIndex().get(job_id)
    return (indexed.s3_prefix if indexed else None) or entry.s3_prefix


def start(
    job_ids: Sequence[str],
    host: str | None,
    settings: Settings | None = None,
    *,
    report: Reporter = note,
) -> Watch:
    """Resolve each id to a host, without asking a host anything yet."""
    registry = named_registry()
    targets = [
        (job_id, find_job_host(job_id, registry, host)[0])
        # Deduplicated, so `xargs gpuc wait` on a list with a repeat in it
        # neither polls twice nor prints the outcome twice.
        for job_id in dict.fromkeys(job_ids)
    ]
    return Watch(targets, settings, report=report)


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
        # Before anything is announced: an id no host has is exit 4, not a job
        # that ended badly.
        watch.check_known()
        each_round()
        for watched in settled:
            on_settled(watched)
        if not watch.pending:
            return list(watch.jobs.values())


def exit_code(watched: Iterable[Watched]) -> int:
    """A wait's exit code is the jobs': 0 only if every one of them succeeded.

    The one place `gpuc` uses exit 1 for something other than its own failure,
    and deliberately -- `gpuc ssh <host> -- cmd` does the same with the remote
    command's code, and it is what makes a wait usable in a script.
    """
    return EXIT_OK if all(job.succeeded for job in watched) else EXIT_ERROR
