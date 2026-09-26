"""The dispatch rule, as one pure function over one pass's inputs.

`plan` decides what a dispatch pass does with each queued job, in queue order:
assign it cards, hold cards for it, step over it, or fail it -- and, for the
first job the queue is stuck on, which `auto_preempt` jobs to stop for it. The
dispatcher acts on the decisions, and `project` runs the same walk forward over
the running jobs' end times to say when each queued job is expected to start.
Two consumers, one rule, so they cannot disagree.

The rule: the queue is taken strictly in order. A job that does not fit holds
the cards it could take, owned and borrowed alike, and nothing behind it may
have them -- any other rule makes priority advisory the moment the job at the
front is wider than the free pool, and it is what made automatic preemption
livelock. What is held is cards, not the queue: a job behind that needs none
of them goes ahead -- a job asking for no cards on its first pass, or a
borrower onto a shared card the holder may not use. The one exemption is a job that could
not fit even once every job of ours ends: it is short of a shared card
somebody else is using, which comes free when *their* job ends, and that is
not ours to wait on. It is stepped
over, not failed, since the configured host is big enough for it. A job that
asks for more than the host is configured with, counting shared cards only if
it may borrow, can never run and is failed.

The other exemption is a job that said it may be stopped (`auto_preempt`). It
may take free cards held for a job ahead of it: a card waiting for its
siblings does work in the meantime, and automatic preemption stops it once
that is what the job ahead is short of. Not a card held for a job that is
already *covered* -- one that starts once the stops in flight land, or the
first job with a gap when the `auto_preempt` jobs behind it cover it, since
those are the ones automatic preemption is about to stop -- or the stopped jobs would be
relaunched onto the very cards they gave up, one after another, for ever.
Estimates are informational, so unlike Slurm's backfill nothing proves a
filler ends in time; it is the promise to be stopped that makes it safe.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

Sample = Callable[[], tuple[list[str], int]]
"""Read the shared cards once: the UUIDs nobody is on, and how many somebody
else is on. Called at most once per `plan`, and only if a job needs to borrow.
The reading is the one thing standing between a borrowed card and somebody
else's training run, so it is taken fresh each pass and never inferred."""


@dataclass(frozen=True)
class Request:
    """One queued job, as the rule sees it."""

    job_id: str
    gpus: int
    borrows: bool
    """`HostConfig.may_borrow(spec)`: the job asked and the host has shared cards."""
    estimate_s: float | None = None
    """How long it expects to run, for `project` only."""
    priority: int = 50
    fills: bool = False
    """The spec's `auto_preempt`: it may take cards held for a job ahead of it."""

    @property
    def key(self) -> tuple[int, str]:
        return (self.priority, self.job_id)


@dataclass(frozen=True)
class Preemptable:
    """A running job whose spec said it may be stopped for something better."""

    job_id: str
    priority: int
    gpus: list[str]
    started: float
    """When it started, in any unit that orders starts."""
    owned: int
    borrowed: int
    """How many of `gpus` this host owns, and how many it is borrowing. Only a
    waiting job that asked to borrow can be started on a borrowed one -- and a
    card in neither list, one that has dropped off nvidia-smi under a running
    job, is counted by neither, because it is never handed out to anybody and
    stopping a job for it would start nothing."""

    @property
    def key(self) -> tuple[int, str]:
        return (self.priority, self.job_id)

    def frees(self, *, borrowing: bool) -> int:
        """Cards this would hand to a waiting job that may (or may not) borrow."""
        return self.owned + self.borrowed if borrowing else self.owned


def enough_to_start(
    candidates: Sequence[Preemptable], key: tuple[int, str], gap: int, *, borrowing: bool
) -> list[Preemptable]:
    """Which of these to stop so the job queued at `key` gets `gap` more cards.

    All of them or none: freeing one of the two cards a job needs would cost an
    attempt and start nothing. Least important first, and among equals the one
    that has been running the shortest time, because what a preempt throws away
    is the work the attempt has already done.

    Only a job the waiting one is ahead of in dispatch order, since a stopped
    job is queued again at its own `(priority, job_id)`: one ahead of the
    waiting job would win the next pass, take its own cards straight back, and
    be stopped again for ever. That is also all an equal-priority filler needs:
    it passed the waiting job, so it sorts behind it.

    `borrowing` is the waiting job's `use_shared`: a borrowed card handed back
    by a stopped job is no use to a job that may not be dispatched onto one, so
    it does not count towards the gap and cannot be the reason a job is stopped.
    """
    chosen: list[Preemptable] = []
    freed = 0
    for candidate in sorted(candidates, key=lambda c: (c.priority, c.started), reverse=True):
        if freed >= gap:
            break
        if candidate.key <= key or not candidate.frees(borrowing=borrowing):
            continue
        chosen.append(candidate)
        freed += candidate.frees(borrowing=borrowing)
    return chosen if freed >= gap else []


@dataclass
class Pool:
    """What one pass has to hand out.

    Counts of *configured* cards are what "could this job ever fit" is judged
    against: a card missing from nvidia-smi this minute makes a job wait, it
    does not fail one the host is set up to run.
    """

    owned_free: list[str]
    """Owned cards nothing of ours holds, as UUIDs, visible now."""
    owned_configured: int
    """`len(config.gpus)`, missing cards included."""
    shared_configured: int
    """Shared entries not also owned, for capacity."""
    shared_visible: int
    """Shared cards nvidia-smi reports, for whether a short job holds."""
    sample: Sample | None = None
    """How to read the shared cards, if a job needs them."""
    shared: frozenset[str] = frozenset()
    """Every visible shared card, to tell a borrowed card in `coming` from an
    owned one."""
    coming: list[str] = field(default_factory=list)
    """Cards a job on its way out is handing back. No reading is needed for a
    shared one: the job on it is ours."""
    preemptable: list[Preemptable] = field(default_factory=list)
    """Running `auto_preempt` jobs not already stopping."""
    shared_free: list[str] | None = None
    """Set once `sample` has been called."""
    theirs: int = 0
    """Shared cards somebody else is on, from that reading."""

    def borrowable(self) -> list[str]:
        if self.shared_free is None:
            free, self.theirs = self.sample() if self.sample else ([], 0)
            self.shared_free = list(free)
        return self.shared_free


@dataclass(frozen=True)
class Assigned:
    job_id: str
    gpus: list[str]
    held_for: tuple[str, ...] = ()
    """For an `auto_preempt` job, the jobs holding the cards it was given."""


@dataclass(frozen=True)
class Holds:
    """Does not fit yet, and keeps the cards it could take -- including any a
    job on its way out is handing back."""

    job_id: str
    taken_owned: int
    taken_shared: int
    gap: int
    """Cards it is still short of once those are back. Zero is a job that is
    not stuck, only waiting for a stop to land."""
    preempts: tuple[str, ...] = ()
    """The `auto_preempt` jobs whose stopping covers `gap`, if some set does."""


@dataclass(frozen=True)
class SteppedOver:
    """Short of a shared card somebody else is using; holds nothing."""

    job_id: str
    short: int
    """Shared cards it would need that somebody else is on."""


@dataclass(frozen=True)
class Fails:
    job_id: str
    reason: str


Decision = Assigned | Holds | SteppedOver | Fails


def capacity_failure(gpus: int, owned: int, shared: int, *, borrows: bool) -> str | None:
    """Why a job asking for `gpus` cards can *never* run on this host, or None.

    Everything read here is fixed for the life of a queued job -- the
    configured counts and the spec's `use_shared` -- because the answer
    deletes the job from the queue.
    """
    if gpus <= owned + (shared if borrows else 0):
        return None
    have = f"host owns {owned}"
    if borrows:
        have += f" and may borrow {shared} shared"
    elif shared:
        have += f" and shares {shared} this job did not ask for (`use_shared: true` would let it)"
    return f"needs {gpus} GPUs, {have}"


def plan(requests: Sequence[Request], pool: Pool) -> list[Decision]:
    """One decision per request, in order, consuming `pool` as it goes.

    Owned cards first, always: a job borrows only its shortfall, so a shared
    card is held for the shortest time that runs the job. The cards a job
    ahead is holding are not on offer to the ones behind it, except to fill.
    """
    decisions: list[Decision] = []
    held: dict[str, str] = {}
    """Card -> the job holding it."""
    covered: set[str] = set()
    stuck = False
    """Whether the first job with a gap has been passed: automatic preemption
    acts for that one only, so only that one is covered by it."""
    for request in requests:
        failure = capacity_failure(
            request.gpus, pool.owned_configured, pool.shared_configured, borrows=request.borrows
        )
        if failure is not None:
            decisions.append(Fails(request.job_id, failure))
            continue
        want = request.gpus
        free = [c for c in pool.owned_free if c not in held][:want]
        n_owned = len(free)
        if len(free) < want and request.borrows:
            free += [c for c in pool.borrowable() if c not in held][: want - len(free)]
        if len(free) == want:
            _take(pool, free)
            decisions.append(Assigned(request.job_id, free))
            continue
        coming = [
            c for c in pool.coming if c not in held and (request.borrows or c not in pool.shared)
        ]
        coming = sorted(coming, key=lambda c: c in pool.shared)[: want - len(free)]
        short = want - len(free) - len(coming)
        ours = pool.owned_configured
        if request.borrows:
            ours += pool.shared_visible - pool.theirs
        if request.gpus > ours:
            decisions.append(SteppedOver(request.job_id, request.gpus - ours))
            continue
        fill = _fill(request, pool, held, covered) if request.fills and short else None
        if fill is not None:
            _take(pool, fill.gpus)
            decisions.append(fill)
            continue
        preempts = []
        if short and not stuck:
            stuck = True
            preempts = enough_to_start(
                pool.preemptable, request.key, short, borrowing=request.borrows
            )
        taken = [*free, *coming]
        for card in taken:
            held[card] = request.job_id
        if not short or preempts:
            covered.add(request.job_id)
        n_owned += sum(1 for c in coming if c not in pool.shared)
        decisions.append(
            Holds(
                request.job_id,
                n_owned,
                len(taken) - n_owned,
                short,
                tuple(p.job_id for p in preempts),
            )
        )
    return decisions


def _take(pool: Pool, cards: Sequence[str]) -> None:
    gone = set(cards)
    pool.owned_free = [c for c in pool.owned_free if c not in gone]
    if pool.shared_free is not None:
        pool.shared_free = [c for c in pool.shared_free if c not in gone]


def _fill(request: Request, pool: Pool, held: dict[str, str], covered: set[str]) -> Assigned | None:
    """The free cards that would start this `auto_preempt` job now, counting
    those held for a job that is not covered. Unheld before held, owned
    before borrowed, so it sits on as few held cards as it can."""

    def open_to_it(card: str) -> bool:
        return held.get(card) not in covered

    owned = sorted(filter(open_to_it, pool.owned_free), key=lambda c: c in held)
    shared = (
        sorted(filter(open_to_it, pool.borrowable()), key=lambda c: c in held)
        if request.borrows
        else []
    )
    cards = [*owned, *shared][: request.gpus]
    if len(cards) < request.gpus:
        return None
    holders = dict.fromkeys(held[c] for c in cards if c in held)
    return Assigned(request.job_id, cards, tuple(holders))


@dataclass(frozen=True)
class Card:
    """A card `project` may schedule onto, and when it is free.

    `release_s` is now if nothing holds it, the holder's eta if one of our jobs
    does, and None when nothing can say: the holder gave no end time. A shared
    card somebody else is on is not a `Card` at all -- when they will stop is
    the one thing this host cannot know -- and counts in `theirs` instead.
    """

    uuid: str
    shared: bool
    release_s: float | None


@dataclass(frozen=True)
class Projection:
    starts_in_s: dict[str, float]
    """Seconds until each queued job is expected to start, for those that can be."""
    unknown: dict[str, str]
    """Why each of the rest has no start time."""
    yields_to: dict[str, str] = field(default_factory=dict)
    """For each running `auto_preempt` job, the first queued job it would be
    stopped for once that job can start."""


def yields_to(job: Preemptable, requests: Sequence[Request]) -> str | None:
    """The first queued job ahead of this running one that could use its
    cards, by `enough_to_start`'s own test. For `status`, to say why a job at
    priority 90 is running while one at 10 waits: it is not necessarily the
    job it will be stopped for, which only a dispatch pass decides."""
    for request in requests:
        if request.key < job.key and request.gpus and job.frees(borrowing=request.borrows):
            return request.job_id
    return None


def project(
    requests: Sequence[Request],
    cards: Sequence[Card],
    *,
    owned_configured: int,
    owned_missing: Sequence[str],
    shared_configured: int,
    theirs: int,
    running: Sequence[tuple[Preemptable, Request]] = (),
    draining: bool = False,
) -> Projection:
    """When each queued job is expected to start, by replaying the rule.

    Cards come free at the eta of whatever holds them, and at each release the
    queue is taken in order exactly as a dispatch pass takes it. A job is in
    the answer or it is not: one whose turn depends on a job that gave no
    estimate is absent, never guessed at, and says why. A draining host
    projects nothing, since nothing more will be dispatched on it.

    A job that starts is assumed to run for its own estimate; one with none
    holds its cards for ever as far as this projection can tell -- unless it
    is `auto_preempt`, when it holds them until the job it `yields_to` can
    start, as a dispatch pass decides. `running` are those jobs, each with the
    request it is queued again as. A job stopped for another is projected to
    start again as a queued one; a filler's start is its first. The stop
    itself is taken to be instant, which it is not: the job it makes room for
    starts a pass and the stop later than projected.
    """
    if draining:
        why = "the host is draining, so nothing more will be dispatched"
        return Projection({}, {r.job_id: why for r in requests})
    releases: dict[str, float | None] = {card.uuid: card.release_s for card in cards}
    is_shared = {card.uuid: card.shared for card in cards}
    shared_visible = sum(1 for card in cards if card.shared) + theirs
    already = {p.job_id for p, _ in running}
    stoppable = {p.job_id: (p, r) for p, r in running}
    ends = {p.job_id: releases.get(p.gpus[0]) if p.gpus else None for p, _ in running}
    pending = list(requests)
    gives_way = {p.job_id: to for p, _ in running if (to := yields_to(p, requests)) is not None}
    starts: dict[str, float] = {}
    clock = 0.0
    decisions: list[Decision] = []
    while pending:
        stoppable = {
            job_id: entry
            for job_id, entry in stoppable.items()
            if (end := ends[job_id]) is None or end > clock
        }
        free = [uuid for uuid, at in releases.items() if at is not None and at <= clock]
        pool = Pool(
            owned_free=[u for u in free if not is_shared[u]],
            owned_configured=owned_configured,
            shared_configured=shared_configured,
            shared_visible=shared_visible,
            shared=frozenset(u for u, shared in is_shared.items() if shared),
            preemptable=[p for p, _ in stoppable.values()],
            shared_free=[u for u in free if is_shared[u]],
            theirs=theirs,
        )
        decisions = plan(pending, pool)
        by_id = {r.job_id: r for r in pending}
        for decision in decisions:
            if not isinstance(decision, Assigned):
                continue
            request = by_id[decision.job_id]
            starts.setdefault(request.job_id, clock)
            done = None if request.estimate_s is None else clock + request.estimate_s
            for uuid in decision.gpus:
                releases[uuid] = done
            pending.remove(request)
            if request.fills and decision.gpus:
                n_shared = sum(1 for u in decision.gpus if is_shared[u])
                stoppable[request.job_id] = (
                    Preemptable(
                        request.job_id,
                        request.priority,
                        decision.gpus,
                        clock,
                        len(decision.gpus) - n_shared,
                        n_shared,
                    ),
                    request,
                )
                ends[request.job_id] = done
        # As the dispatcher does: launch what fits, then stop what covers the
        # first stuck job, and walk again at the same moment to start it.
        stuck = next((d for d in decisions if isinstance(d, Holds) and d.gap), None)
        if stuck is not None and stuck.preempts:
            for job_id in stuck.preempts:
                stopped, again = stoppable.pop(job_id)
                for uuid in stopped.gpus:
                    releases[uuid] = clock
                pending.append(again)
            pending.sort(key=lambda r: r.key)
            continue
        later = [at for at in releases.values() if at is not None and at > clock]
        if not later:
            break
        clock = min(later)
    queued = [r for r in pending if r.job_id not in already]
    return Projection(
        {job_id: at for job_id, at in starts.items() if job_id not in already},
        _unknown_reasons(decisions, queued, owned_configured, owned_missing),
        gives_way,
    )


def _unknown_reasons(
    decisions: Sequence[Decision],
    pending: Sequence[Request],
    owned_configured: int,
    owned_missing: Sequence[str],
) -> dict[str, str]:
    """From the last pass that could start nothing, why each job is still waiting.

    The reasons are not interchangeable: a job the host steps over is waiting
    on somebody else, a job behind a held one is waiting on the queue, and a
    job held for a card nvidia-smi cannot see is waiting on the host being
    fixed, which the submitter should hear.
    """
    by_id = {r.job_id: r for r in pending}
    reasons: dict[str, str] = {}
    first_holder: str | None = None
    owned_visible = owned_configured - len(owned_missing)
    for decision in decisions:
        if decision.job_id not in by_id:
            continue
        request = by_id[decision.job_id]
        if isinstance(decision, Fails):
            reasons[request.job_id] = f"{decision.reason}, so it will never be dispatched"
        elif isinstance(decision, SteppedOver):
            reasons[request.job_id] = (
                f"it needs {decision.short} shared card(s) somebody else is using, and when "
                f"they stop is not something this host can predict"
            )
        elif isinstance(decision, Holds):
            if first_holder is not None:
                reasons[request.job_id] = (
                    f"job {first_holder} is ahead of it and has no start time yet"
                )
            elif owned_visible < request.gpus <= owned_configured:
                reasons[request.job_id] = (
                    f"it needs {request.gpus} card(s) and only {owned_visible} of the "
                    f"{owned_configured} this host owns answer to nvidia-smi "
                    f"({', '.join(owned_missing)} missing), so it is held until they do"
                )
            else:
                reasons[request.job_id] = "the jobs holding the cards it needs gave no end time"
            first_holder = first_holder or request.job_id
    return reasons
