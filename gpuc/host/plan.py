"""The dispatch rule, as one pure function over one pass's inputs.

`plan` decides what a dispatch pass does with each queued job, in queue order:
assign it cards, hold cards for it, step over it, or fail it. The dispatcher
acts on the decisions; automatic preemption runs the same walk over the cards a
stop in flight will hand back, to find the one job the queue is stuck on; and
`project` runs it forward over the running jobs' end times to say when each
queued job is expected to start. Three questions, one rule, so they cannot
disagree.

The rule: the queue is taken strictly in order. A job that does not fit holds
the cards it could take, owned and borrowed alike, and nothing behind it may
have them -- any other rule makes priority advisory the moment the job at the
front is wider than the free pool, and it is what made automatic preemption
livelock. The one exemption is a job that could not fit even once every job
of ours ends: it is short of a shared card somebody else is using, which comes
free when *their* job ends, and that is not ours to wait on. It is stepped
over, not failed, since the configured host is big enough for it. A job that
asks for more than the host is configured with, counting shared cards only if
it may borrow, can never run and is failed.
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
    shared_extra: list[str] = field(default_factory=list)
    """Shared cards on offer without a reading: those a stopping job is about
    to hand back. Only `preempt_for_waiting` has any."""
    shared_free: list[str] | None = None
    """Set once `sample` has been called."""
    theirs: int = 0
    """Shared cards somebody else is on, from that reading."""

    def borrowable(self) -> list[str]:
        if self.shared_free is None:
            free, self.theirs = self.sample() if self.sample else ([], 0)
            self.shared_free = [*free, *self.shared_extra]
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


Decision = Assigned | Holds | SteppedOver | Fails


def capacity_failure(gpus: int, owned: int, shared: int, *, borrows: bool) -> str | None:
    """Why a job asking for `gpus` cards can *never* run on this host, or None.

    Everything read here is fixed for the life of a queued job -- the
    configured counts and the spec's `use_shared` -- because the answer
    deletes the job from the queue. `gpus` is at least one: `JobSpec` refuses
    anything less at the reader, so a spec asking for none is `bad-spec`.
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
    ahead is holding are not on offer to the ones behind it.
    """
    decisions: list[Decision] = []
    held = held_shared = 0
    for request in requests:
        failure = capacity_failure(
            request.gpus, pool.owned_configured, pool.shared_configured, borrows=request.borrows
        )
        if failure is not None:
            decisions.append(Fails(request.job_id, failure))
            continue
        owned_part = pool.owned_free[: min(request.gpus, max(0, len(pool.owned_free) - held))]
        shared_part: list[str] = []
        short = request.gpus - len(owned_part)
        if short and request.borrows:
            free_shared = pool.borrowable()
            shared_part = free_shared[: min(short, max(0, len(free_shared) - held_shared))]
            short -= len(shared_part)
        if short:
            ours = pool.owned_configured
            if request.borrows:
                ours += pool.shared_visible - pool.theirs
            if request.gpus <= ours:
                held += len(owned_part)
                held_shared += len(shared_part)
                decisions.append(Holds(request.job_id, len(owned_part), len(shared_part), short))
            else:
                decisions.append(SteppedOver(request.job_id, request.gpus - ours))
            continue
        pool.owned_free = pool.owned_free[len(owned_part) :]
        if shared_part:
            pool.shared_free = pool.borrowable()[len(shared_part) :]
        decisions.append(Assigned(request.job_id, [*owned_part, *shared_part]))
    return decisions


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
    holds its cards for ever as far as this projection can tell.
    """
    if draining:
        why = "the host is draining, so nothing more will be dispatched"
        return Projection({}, {r.job_id: why for r in requests})
    releases: dict[str, float | None] = {card.uuid: card.release_s for card in cards}
    is_shared = {card.uuid: card.shared for card in cards}
    shared_visible = sum(1 for card in cards if card.shared) + theirs
    pending = list(requests)
    starts: dict[str, float] = {}
    clock = 0.0
    decisions: list[Decision] = []
    while pending:
        free = [uuid for uuid, at in releases.items() if at is not None and at <= clock]
        pool = Pool(
            owned_free=[u for u in free if not is_shared[u]],
            owned_configured=owned_configured,
            shared_configured=shared_configured,
            shared_visible=shared_visible,
            shared_free=[u for u in free if is_shared[u]],
            theirs=theirs,
        )
        decisions = plan(pending, pool)
        by_id = {r.job_id: r for r in pending}
        for decision in decisions:
            if not isinstance(decision, Assigned):
                continue
            request = by_id[decision.job_id]
            starts[request.job_id] = clock
            done = None if request.estimate_s is None else clock + request.estimate_s
            for uuid in decision.gpus:
                releases[uuid] = done
            pending.remove(request)
        later = [at for at in releases.values() if at is not None and at > clock]
        if not later:
            break
        clock = min(later)
    return Projection(starts, _unknown_reasons(decisions, pending, owned_configured, owned_missing))


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
