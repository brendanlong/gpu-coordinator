"""The dispatch rule (`plan.plan`), its capacity check, and the projection of
start times that replays it (`plan.project`). Pure: no host, no filesystem."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from gpuc.host import plan
from gpuc.host.plan import Assigned, Card, Fails, Holds, Pool, Projection, Request, SteppedOver

MIN = 60.0


def req(
    job_id: str, gpus: int = 1, *, borrows: bool = False, est_min: float | None = None
) -> Request:
    return Request(job_id, gpus, borrows, None if est_min is None else est_min * MIN)


def owned(*releases_min: float | None) -> list[Card]:
    """Owned cards `o0`, `o1`, ...: free in that many minutes, or never (None)."""
    return [
        Card(f"o{i}", False, None if at is None else at * MIN) for i, at in enumerate(releases_min)
    ]


def shared(*releases_min: float | None) -> list[Card]:
    return [
        Card(f"s{i}", True, None if at is None else at * MIN) for i, at in enumerate(releases_min)
    ]


def project(
    requests: Sequence[Request],
    cards: Sequence[Card],
    *,
    owned_missing: Sequence[str] = (),
    theirs: int = 0,
    shared_missing: int = 0,
    draining: bool = False,
) -> Projection:
    """`plan.project` with the configured counts read off the cards: every
    visible card is configured, plus whatever is missing or somebody else's."""
    n_owned = sum(1 for card in cards if not card.shared)
    n_shared = sum(1 for card in cards if card.shared)
    return plan.project(
        requests,
        cards,
        owned_configured=n_owned + len(owned_missing),
        owned_missing=owned_missing,
        shared_configured=n_shared + theirs + shared_missing,
        theirs=theirs,
        draining=draining,
    )


def minutes(projection: Projection) -> dict[str, float]:
    return {job_id: round(s / MIN, 3) for job_id, s in projection.starts_in_s.items()}


# -- capacity_failure ---------------------------------------------------------


@pytest.mark.parametrize(
    ("gpus", "owned_n", "shared_n", "borrows", "expected"),
    [
        (1, 2, 0, False, None),
        (2, 2, 0, False, None),
        (3, 2, 1, True, None),
        (0, 2, 0, False, "needs at least 1 GPU, asked for 0"),
        (3, 2, 0, False, "needs 3 GPUs, host owns 2"),
        (4, 2, 1, True, "needs 4 GPUs, host owns 2 and may borrow 1 shared"),
        (
            3,
            2,
            1,
            False,
            "needs 3 GPUs, host owns 2 and shares 1 this job did not ask for "
            "(`use_shared: true` would let it)",
        ),
    ],
)
def test_capacity_failure(
    gpus: int, owned_n: int, shared_n: int, borrows: bool, expected: str | None
) -> None:
    assert plan.capacity_failure(gpus, owned_n, shared_n, borrows=borrows) == expected


# -- plan: one pass -----------------------------------------------------------


def pool(
    owned_free: list[str],
    *,
    owned_configured: int | None = None,
    shared_free: list[str] | None = None,
    theirs: int = 0,
    shared_configured: int | None = None,
    extra: Sequence[str] = (),
) -> tuple[Pool, list[int]]:
    """A pool whose shared cards are read through a counting `sample`."""
    calls: list[int] = []
    free = list(shared_free or [])

    def sample() -> tuple[list[str], int]:
        calls.append(1)
        return list(free), theirs

    visible = len(free) + theirs
    return (
        Pool(
            owned_free=list(owned_free),
            owned_configured=len(owned_free) if owned_configured is None else owned_configured,
            shared_configured=visible if shared_configured is None else shared_configured,
            shared_visible=visible,
            sample=sample,
            shared_extra=list(extra),
        ),
        calls,
    )


def test_a_job_that_fits_is_assigned_owned_cards_first() -> None:
    p, calls = pool(["a", "b"], shared_free=["s"])
    assert plan.plan([req("j", 2, borrows=True)], p) == [Assigned("j", ["a", "b"])]
    assert calls == [], "nothing needed to borrow, so nothing asked nvidia-smi"


def test_a_borrower_borrows_only_its_shortfall() -> None:
    p, calls = pool(["a"], owned_configured=2, shared_free=["s0", "s1"])
    assert plan.plan([req("j", 2, borrows=True)], p) == [Assigned("j", ["a", "s0"])]
    assert calls == [1]


def test_the_shared_cards_are_read_at_most_once_a_pass() -> None:
    p, calls = pool([], owned_configured=1, shared_free=["s0", "s1"])
    decisions = plan.plan([req("a", borrows=True), req("b", borrows=True)], p)
    assert decisions == [Assigned("a", ["s0"]), Assigned("b", ["s1"])]
    assert calls == [1]


def test_a_job_that_did_not_ask_never_triggers_a_reading() -> None:
    p, calls = pool([], owned_configured=1, shared_free=["s"])
    assert plan.plan([req("j")], p) == [Holds("j", 0, 0, 1)]
    assert calls == []


def test_a_job_that_does_not_fit_holds_and_nothing_behind_it_takes_those_cards() -> None:
    """Strict order: the card is idle, and the narrow job behind still waits."""
    p, _ = pool(["a"], owned_configured=2)
    assert plan.plan([req("wide", 2), req("narrow")], p) == [
        Holds("wide", 1, 0, 1),
        Holds("narrow", 0, 0, 1),
    ]


def test_a_held_shared_card_is_not_on_offer_behind_the_holder() -> None:
    p, _ = pool([], owned_configured=2, shared_free=["s"])
    assert plan.plan([req("wide", 2, borrows=True), req("small", borrows=True)], p) == [
        Holds("wide", 0, 1, 1),
        Holds("small", 0, 0, 1),
    ]


def test_a_borrower_short_of_somebody_elses_card_is_stepped_over() -> None:
    """It could not fit even once every job of ours ends, so the host does not
    wait on it -- and it holds nothing, so the job behind it starts."""
    p, _ = pool(["a", "b"], shared_free=[], theirs=1)
    assert plan.plan([req("wide", 3, borrows=True), req("narrow")], p) == [
        SteppedOver("wide", 1),
        Assigned("narrow", ["a"]),
    ]


def test_a_borrower_short_only_of_an_owned_card_holds_like_any_other() -> None:
    p, _ = pool(["a"], owned_configured=2, shared_free=["s"])
    assert plan.plan([req("wide", 3, borrows=True), req("narrow")], p) == [
        Holds("wide", 1, 1, 1),
        Holds("narrow", 0, 0, 1),
    ]


def test_a_job_short_of_a_missing_owned_card_holds() -> None:
    """Configured is what "could it ever fit" is judged against, so a card off
    nvidia-smi this minute makes the job wait, not fail."""
    p, _ = pool(["a"], owned_configured=2)
    assert plan.plan([req("wide", 2)], p) == [Holds("wide", 1, 0, 1)]


@pytest.mark.parametrize(
    ("request_", "reason"),
    [
        (req("j", 3), "needs 3 GPUs, host owns 2"),
        (req("j", 0), "needs at least 1 GPU, asked for 0"),
        (req("j", 4, borrows=True), "needs 4 GPUs, host owns 2 and may borrow 1 shared"),
    ],
)
def test_a_job_that_can_never_fit_fails_and_holds_nothing(request_: Request, reason: str) -> None:
    p, _ = pool(["a", "b"], shared_free=["s"])
    failed, after = plan.plan([request_, req("next")], p)
    assert isinstance(failed, Fails) and failed.reason.startswith(reason)
    assert after == Assigned("next", ["a"])


def test_a_shared_card_somebody_else_is_on_still_counts_against_never() -> None:
    p, _ = pool(["a", "b"], shared_free=[], theirs=1)
    assert not isinstance(plan.plan([req("j", 3, borrows=True)], p)[0], Fails)


def test_cards_a_stopping_job_hands_back_count_as_free() -> None:
    """What `preempt_for_waiting` walks: the stop in flight covers the gap."""
    p, _ = pool(["a"], owned_configured=2, shared_free=[], extra=["s"])
    p.shared_configured = p.shared_visible = 1
    assert plan.plan([req("j", 2, borrows=True)], p) == [Assigned("j", ["a", "s"])]


# -- project: when each queued job starts -------------------------------------

# (name, requests, cards, project kwargs, expected starts in minutes)
STARTS: list[tuple[str, list[Request], list[Card], dict[str, object], dict[str, float]]] = [
    ("a free card means now", [req("next")], owned(None, 0), {}, {"next": 0.0}),
    (
        "each job waits for the card handed back first",
        [req("next", est_min=60), req("after", est_min=60)],
        owned(20, 130),
        {},
        # `next` takes the 20m card for its own hour, so `after` has it at 1h20m,
        # sooner than the 2h10m card.
        {"next": 20.0, "after": 80.0},
    ),
    (
        "no estimate hides only the job that inherits its card",
        [req("silent"), req("behind")],
        owned(20, 130),
        {},
        {"silent": 20.0, "behind": 130.0},
    ),
    (
        "a wide job holds the card that comes back first",
        [req("wide", 2), req("narrow", est_min=10)],
        owned(45, 90),
        {},
        {"wide": 90.0},
    ),
    (
        "the job behind a wide one starts when the wide one is done",
        [req("wide", 2, est_min=30), req("narrow", est_min=10)],
        owned(45, 90),
        {},
        {"wide": 90.0, "narrow": 120.0},
    ),
    (
        "a missing owned card holds the whole queue",
        [req("wide", 2), req("narrow"), req("wide-too", 2)],
        owned(45),
        {"owned_missing": ["7"]},
        {},
    ),
    (
        "a job behind one whose card never frees has no start",
        [req("wide", 2), req("narrow")],
        owned(None, 90),
        {},
        {},
    ),
    ("nothing starts on a draining host", [req("next")], owned(None, 0), {"draining": True}, {}),
    (
        "a wide job at the front holds the idle card",
        [req("next", 2), req("later", est_min=10)],
        owned(20, 0),
        {},
        {"next": 20.0},
    ),
    (
        "two one-card jobs, one card free now",
        [req("first"), req("second")],
        owned(20, 0),
        {},
        {"first": 0.0, "second": 20.0},
    ),
    # Shared cards.
    (
        "a borrower is not told it waits for the owned cards",
        [req("q", borrows=True)],
        [*owned(360, 360), *shared(0)],
        {},
        {"q": 0.0},
    ),
    (
        "a job that did not ask waits for the owned cards",
        [req("q")],
        [*owned(360, 360), *shared(0)],
        {},
        {"q": 360.0},
    ),
    (
        "a shared card somebody else holds is not scheduled onto",
        [req("q", borrows=True)],
        owned(360, 360),
        {"theirs": 1},
        {"q": 360.0},
    ),
    (
        "a wide borrower short of an owned card holds like any other",
        [req("wide", 3, borrows=True, est_min=30), req("narrow")],
        [*owned(45, 0), *shared(0)],
        {},
        {"wide": 45.0, "narrow": 75.0},
    ),
    (
        "a wide borrower whose owned card never frees has no start",
        [req("wide", 3, borrows=True), req("narrow")],
        [*owned(None, 0), *shared(0)],
        {},
        {},
    ),
    (
        "a borrower short of somebody else's card is stepped over",
        [req("wide", 3, borrows=True), req("narrow")],
        owned(45, 0),
        {"theirs": 1},
        {"narrow": 0.0},
    ),
    (
        "there is only one shared card to go round",
        [req("first", borrows=True), req("second", borrows=True)],
        [*owned(360, 360), *shared(0)],
        {},
        {"first": 0.0, "second": 360.0},
    ),
    (
        "a borrowed card comes back at the borrower's estimate",
        [req("first", borrows=True, est_min=15), req("second", borrows=True)],
        [*owned(360, 360), *shared(0)],
        {},
        {"first": 0.0, "second": 15.0},
    ),
]


@pytest.mark.parametrize(
    ("requests", "cards", "kwargs", "expected"),
    [case[1:] for case in STARTS],
    ids=[case[0] for case in STARTS],
)
def test_project_starts(
    requests: list[Request],
    cards: list[Card],
    kwargs: dict[str, object],
    expected: dict[str, float],
) -> None:
    projection = project(requests, cards, **kwargs)  # type: ignore[arg-type]
    assert minutes(projection) == pytest.approx(expected)
    # A job is in the answer or it says why not, never both and never neither.
    ids = {r.job_id for r in requests}
    assert set(projection.starts_in_s) | set(projection.unknown) == ids
    assert not set(projection.starts_in_s) & set(projection.unknown)


# -- project: why a job has no start ------------------------------------------

NO_END = "the jobs holding the cards it needs gave no end time"

# (name, requests, cards, project kwargs, expected reason fragment per job)
REASONS: list[tuple[str, list[Request], list[Card], dict[str, object], dict[str, str]]] = [
    (
        "draining",
        [req("next"), req("other", 2)],
        owned(None, 0),
        {"draining": True},
        {
            "next": "the host is draining, so nothing more will be dispatched",
            "other": "the host is draining, so nothing more will be dispatched",
        },
    ),
    (
        "capacity",
        [req("huge", 4), req("next")],
        owned(None, 0),
        {},
        {"huge": "needs 4 GPUs, host owns 2, so it will never be dispatched"},
    ),
    (
        "capacity, shared included",
        [req("huge", 9, borrows=True)],
        [*owned(None, 0), *shared(0)],
        {},
        {"huge": "needs 9 GPUs, host owns 2 and may borrow 1 shared, so it will never"},
    ),
    (
        "capacity, did not ask for the shared card",
        [req("q", 3)],
        [*owned(0, 0), *shared(0)],
        {},
        {"q": "host owns 2 and shares 1 this job did not ask for"},
    ),
    (
        "stepped over, owned cards idle",
        [req("q", 3, borrows=True)],
        owned(0, 0),
        {"theirs": 1},
        {
            "q": "it needs 1 shared card(s) somebody else is using, and when they stop "
            "is not something this host can predict"
        },
    ),
    (
        # The host has one shared card, so "needs 2 shared card(s)" cannot be
        # right: the old control-side reason counted what the job is short of
        # once every job of ours ends (3 - 2 owned = 1).
        "stepped over, an owned card busy",
        [req("wide", 3, borrows=True), req("narrow")],
        owned(45, 0),
        {"theirs": 1},
        {"wide": "it needs 1 shared card(s) somebody else is using"},
    ),
    (
        "stepped over, the shared entry is missing from nvidia-smi",
        [req("q", 3, borrows=True)],
        owned(0, 0),
        {"shared_missing": 1},
        {"q": "somebody else is using"},
    ),
    (
        "the only job, every card busy with no end time",
        [req("next")],
        owned(None, None),
        {},
        {"next": NO_END},
    ),
    (
        "behind a wide job that starts but gave no estimate",
        [req("wide", 2), req("narrow", est_min=10)],
        owned(45, 90),
        {},
        {"narrow": NO_END},
    ),
    (
        "blocked by a job ahead",
        [req("wide", 2), req("narrow")],
        owned(None, 90),
        {},
        {"wide": NO_END, "narrow": "job wide is ahead of it and has no start time yet"},
    ),
    (
        "held on a missing owned card",
        [req("wide", 2), req("narrow"), req("wide-too", 2)],
        owned(45),
        {"owned_missing": ["7"]},
        {
            "wide": "it needs 2 card(s) and only 1 of the 2 this host owns answer to "
            "nvidia-smi (7 missing), so it is held until they do",
            # Behind the holder, each is waiting on the queue, not the card.
            "narrow": "job wide is ahead of it",
            "wide-too": "job wide is ahead of it",
        },
    ),
    (
        "a wide borrower waiting on our card is not told it waits on somebody else",
        [req("wide", 3, borrows=True), req("narrow")],
        [*owned(None, 0), *shared(0)],
        {},
        {"wide": NO_END, "narrow": "job wide is ahead of it"},
    ),
]


@pytest.mark.parametrize(
    ("requests", "cards", "kwargs", "expected"),
    [case[1:] for case in REASONS],
    ids=[case[0] for case in REASONS],
)
def test_project_says_why(
    requests: list[Request], cards: list[Card], kwargs: dict[str, object], expected: dict[str, str]
) -> None:
    unknown = project(requests, cards, **kwargs).unknown  # type: ignore[arg-type]
    for job_id, fragment in expected.items():
        assert fragment in unknown.get(job_id, ""), (job_id, unknown)


def test_a_job_behind_a_holder_is_not_told_about_the_missing_card() -> None:
    unknown = project([req("wide", 2), req("wide-too", 2)], owned(45), owned_missing=["7"]).unknown
    assert "missing" in unknown["wide"]
    assert "missing" not in unknown["wide-too"]


def test_an_empty_queue_projects_nothing() -> None:
    assert project([], owned(None, 0)) == Projection({}, {})
