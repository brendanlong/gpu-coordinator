"""The dispatch rule, as one pure function over one pass's inputs.

`plan` decides what a dispatch pass does with each queued job, in queue order:
assign it cards, hold cards for it, step over it, fail it, or launch it as a
filler. The dispatcher acts on the decisions; automatic preemption runs the
same walk to find the one job the queue is stuck on; and `project` runs it
forward over the running jobs' end times to say when each queued job is
expected to start. Three questions, one rule, so they cannot disagree.

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

**Fillers.** A held card that is free does no work until its siblings arrive,
which on a two- or four-card host is a large share of everything the user has.
So a job that said it may be stopped (`auto_preempt`) and would otherwise hold
may be launched onto held cards (`Fills`). It occupies them; it does not
acquire them. A running filler stays in the walk at its own place in the
queue, and its cards are on offer to every job ahead of it exactly as free
ones are, except that a job which needs them to fit is not assigned but
`Reclaims` them: the filler is stopped, and the job starts once it is gone.
Being stopped queues the filler again at its own `(priority, job_id)`, which
is still behind the job it made room for, so it cannot take the card back --
the livelock of a stopped job relaunched onto the card it gave up cannot
happen. A filler that ends on its own hands the card back to the walk, where
the job ahead of it holds it again.

Slurm's conservative backfill can let any job onto a reserved node because it
can prove the job ends before the reservation starts. Estimates here are
informational, so instead of proving the filler finishes in time the host
keeps the right to stop it, and only a job that agreed to that is eligible.
The price is that the job filled for starts a little later than it would have:
the pass that notices, plus the stop's grace period.

Cards a stop in flight is handing back (`Pool.coming`) are on offer the same
way: a job that fits with them `Reclaims` with nobody to stop, and holds the
free cards it will start on. Without that, each card freed by one stop would
be taken by the next filler, and the job waiting would chase them for ever.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace

Sample = Callable[[], tuple[list[str], int]]
"""Read the shared cards once: the UUIDs nobody is on, and how many somebody
else is on. Called at most once per `plan`, and only if a job needs to borrow.
The reading is the one thing standing between a borrowed card and somebody
else's training run, so it is taken fresh each pass and never inferred."""


@dataclass(frozen=True)
class Request:
    """One queued job, or one running filler, as the rule sees it."""

    job_id: str
    gpus: int
    borrows: bool
    """`HostConfig.may_borrow(spec)`: the job asked and the host has shared cards."""
    estimate_s: float | None = None
    """How long it expects to run, for `project` only."""
    fills: bool = False
    """The spec's `auto_preempt`: it may be launched onto cards held for a job
    ahead of it."""
    occupying: tuple[str, ...] = ()
    """Set for a running filler: the held cards it is on. It stands at its
    place in the queue so that only the jobs ahead of it can reclaim them."""


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
    """Every visible shared card, to tell a borrowed card in `coming` or under
    a filler from an owned one."""
    coming: list[str] = field(default_factory=list)
    """Cards a stop in flight is handing back, owned and borrowed; no reading
    is needed for a shared one, since the job on it is ours."""
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


@dataclass(frozen=True)
class Holds:
    """Does not fit yet, and keeps the cards it could take."""

    job_id: str
    taken_owned: int
    taken_shared: int
    gap: int
    """Cards it is still short of."""


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


@dataclass(frozen=True)
class Reclaims:
    """Fits, counting cards under fillers behind it and cards a stop in flight
    is handing back. Holds all of `gpus`; `evicts` are the fillers to stop."""

    job_id: str
    gpus: list[str]
    evicts: tuple[str, ...]


@dataclass(frozen=True)
class Fills:
    """An `auto_preempt` job that would hold, launched onto held cards."""

    job_id: str
    gpus: list[str]
    held_for: tuple[str, ...]
    """The jobs holding the cards it is launched onto."""


@dataclass(frozen=True)
class Filling:
    """A running filler, reached in its place in the queue: nothing ahead of
    it needs its cards to start. `held_for` is who holds them now, if anybody."""

    job_id: str
    held_for: tuple[str, ...]


Decision = Assigned | Holds | SteppedOver | Fails | Reclaims | Fills | Filling


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
    card is held for the shortest time that runs the job. Free cards before
    ones in the way, so a job that can start now does, and a job that has to
    stop a filler stops as few as it can. The cards a job ahead is holding
    are not on offer to the ones behind it, except to fill.
    """
    decisions: list[Decision] = []
    held: dict[str, str] = {}
    """Card -> the job holding it, for the walk so far."""
    occupant = {card: r.job_id for r in requests for card in r.occupying}
    """Cards under a filler not yet reached, which is to say behind this job."""
    for request in requests:
        if request.occupying:
            for card in request.occupying:
                occupant.pop(card, None)
            holders = dict.fromkeys(held[c] for c in request.occupying if c in held)
            decisions.append(Filling(request.job_id, tuple(holders)))
            continue
        failure = capacity_failure(
            request.gpus, pool.owned_configured, pool.shared_configured, borrows=request.borrows
        )
        if failure is not None:
            decisions.append(Fails(request.job_id, failure))
            continue

        def usable(card: str, request: Request = request) -> bool:
            return card not in held and (request.borrows or card not in pool.shared)

        want = request.gpus
        owned_part = [c for c in pool.owned_free if c not in held][:want]
        shared_part: list[str] = []
        if len(owned_part) < want and request.borrows:
            shared_part = [c for c in pool.borrowable() if c not in held]
            shared_part = shared_part[: want - len(owned_part)]
        free = [*owned_part, *shared_part]
        in_the_way: list[str] = []
        if len(free) < want:
            candidates = [c for c in [*pool.coming, *occupant] if usable(c)]
            candidates.sort(key=lambda c: c in pool.shared)
            in_the_way = candidates[: want - len(free)]
        short = want - len(free) - len(in_the_way)
        if not short:
            _take(pool, free)
            if not in_the_way:
                decisions.append(Assigned(request.job_id, free))
                continue
            for card in in_the_way:
                held[card] = request.job_id
            evicts = dict.fromkeys(occupant[c] for c in in_the_way if c in occupant)
            decisions.append(Reclaims(request.job_id, [*free, *in_the_way], tuple(evicts)))
            continue
        ours = pool.owned_configured
        if request.borrows:
            ours += pool.shared_visible - pool.theirs
        if request.gpus > ours:
            decisions.append(SteppedOver(request.job_id, request.gpus - ours))
            continue
        fill = _fill(request, pool, held) if request.fills else None
        if fill is not None:
            _take(pool, fill.gpus)
            decisions.append(fill)
            continue
        taken = [*free, *in_the_way]
        for card in taken:
            held[card] = request.job_id
        n_shared = sum(1 for c in taken if c in pool.shared or c in shared_part)
        decisions.append(Holds(request.job_id, len(taken) - n_shared, n_shared, short))
    return decisions


def _take(pool: Pool, cards: Sequence[str]) -> None:
    gone = set(cards)
    pool.owned_free = [c for c in pool.owned_free if c not in gone]
    if pool.shared_free is not None:
        pool.shared_free = [c for c in pool.shared_free if c not in gone]


def _fill(request: Request, pool: Pool, held: dict[str, str]) -> Fills | None:
    """The free cards, held ones included, that would start `request` now.

    Unheld before held and owned before borrowed, so it sits on as few held
    cards as it can. Only free ones: a filler never displaces anything.
    """
    owned = sorted(pool.owned_free, key=lambda c: c in held)
    shared = sorted(pool.borrowable(), key=lambda c: c in held) if request.borrows else []
    cards = [*owned, *shared][: request.gpus]
    if len(cards) < request.gpus:
        return None
    holders = dict.fromkeys(held[c] for c in cards if c in held)
    return Fills(request.job_id, cards, tuple(holders))


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
    held_for: dict[str, list[str]] = field(default_factory=dict)
    """For each running filler, the jobs holding the cards it is on now."""


def project(
    requests: Sequence[Request],
    cards: Sequence[Card],
    *,
    owned_configured: int,
    owned_missing: Sequence[str],
    shared_configured: int,
    theirs: int,
    draining: bool = False,
) -> Projection:
    """When each queued job is expected to start, by replaying the rule.

    Cards come free at the eta of whatever holds them, and at each release the
    queue is taken in order exactly as a dispatch pass takes it. A job is in
    the answer or it is not: one whose turn depends on a job that gave no
    estimate is absent, never guessed at, and says why. A draining host
    projects nothing, since nothing more will be dispatched on it.

    A job that starts is assumed to run for its own estimate; one with none
    holds its cards for ever as far as this projection can tell -- except a
    filler, whose cards are reclaimed the moment the job ahead can start.
    That moment is when the reclaiming job is projected to start, not the pass
    and grace period later that it really does. A filler is projected to start
    when it is launched as one, and to run again from the start after it is
    stopped, like any queued job; its start is the first of those.

    `requests` includes the running fillers, each at its place in the queue
    and occupying cards whose `release_s` is its own eta.
    """
    if draining:
        why = "the host is draining, so nothing more will be dispatched"
        return Projection({}, {r.job_id: why for r in requests if not r.occupying})
    releases: dict[str, float | None] = {card.uuid: card.release_s for card in cards}
    is_shared = {card.uuid: card.shared for card in cards}
    shared_visible = sum(1 for card in cards if card.shared) + theirs
    pending = list(requests)
    until: dict[str, float | None] = {
        r.job_id: releases.get(r.occupying[0]) for r in requests if r.occupying
    }
    starts: dict[str, float] = {}
    held_for: dict[str, list[str]] | None = None
    clock = 0.0
    decisions: list[Decision] = []
    while pending:
        pending = [
            r for r in pending if not r.occupying or (at := until[r.job_id]) is None or at > clock
        ]
        occupied = {card for r in pending for card in r.occupying}
        free = [
            uuid
            for uuid, at in releases.items()
            if at is not None and at <= clock and uuid not in occupied
        ]
        pool = Pool(
            owned_free=[u for u in free if not is_shared[u]],
            owned_configured=owned_configured,
            shared_configured=shared_configured,
            shared_visible=shared_visible,
            shared=frozenset(u for u, shared in is_shared.items() if shared),
            shared_free=[u for u in free if is_shared[u]],
            theirs=theirs,
        )
        decisions = plan(pending, pool)
        if held_for is None:
            held_for = {d.job_id: list(d.held_for) for d in decisions if isinstance(d, Filling)}
        by_id = {r.job_id: r for r in pending}
        evicted = {e for d in decisions if isinstance(d, Reclaims) for e in d.evicts}
        for i, request in enumerate(pending):
            if request.job_id in evicted:
                # Queued again, and whatever the reclaiming job does not take
                # is free now rather than at the filler's own end.
                for uuid in request.occupying:
                    releases[uuid] = clock
                until.pop(request.job_id, None)
                pending[i] = replace(request, occupying=())
        for decision in decisions:
            if not isinstance(decision, (Assigned, Reclaims, Fills)):
                continue
            request = by_id[decision.job_id]
            starts.setdefault(request.job_id, clock)
            done = None if request.estimate_s is None else clock + request.estimate_s
            for uuid in decision.gpus:
                releases[uuid] = done
            if isinstance(decision, Fills):
                until[request.job_id] = done
                pending[pending.index(request)] = replace(request, occupying=tuple(decision.gpus))
            else:
                pending.remove(request)
        if evicted:
            continue  # a card no reclaim took is free now; take the queue again
        later = [at for at in releases.values() if at is not None and at > clock]
        if not later:
            break
        clock = min(later)
    running = {r.job_id for r in requests if r.occupying}
    queued = [r for r in pending if not r.occupying and r.job_id not in starts]
    return Projection(
        {job_id: at for job_id, at in starts.items() if job_id not in running},
        _unknown_reasons(decisions, queued, owned_configured, owned_missing),
        held_for or {},
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
