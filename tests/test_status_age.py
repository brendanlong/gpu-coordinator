"""Relative ages on `status`, and the filters that decide what it shows."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from gpuc.control.status import (
    HostState,
    HostView,
    JobView,
    format_age,
    parse_duration,
    render,
    within,
)
from tests.conftest import host_entry


def ago(**delta: float) -> str:
    return (datetime.now(UTC) - timedelta(**delta)).isoformat()


def finished(job_id: str, status: str = "succeeded", reason: str | None = None, **delta: float):
    return JobView(
        job_id=job_id,
        name="t",
        status=status,
        reason=reason,
        started_at=ago(**delta),
        ended_at=ago(**delta),
    )


def view(*jobs: JobView) -> HostView:
    return HostView(
        entry=host_entry(name="h", gpus=["GPU-a"]),
        state=HostState.ANSWERED,
        heartbeat_age_s=1.0,
        owned=["GPU-a"],
        finished=list(jobs),
    )


@pytest.mark.parametrize(
    ("delta", "expected"),
    [
        ({"seconds": 5}, "5s ago"),
        ({"minutes": 3}, "3m ago"),
        ({"hours": 5}, "5h ago"),
        ({"days": 2}, "2d ago"),
        ({"days": 2, "hours": 23}, "2d ago"),
    ],
)
def test_age_reads_as_a_human_would_say_it(delta: dict[str, float], expected: str) -> None:
    assert format_age(ago(**delta)) == expected


def test_an_unusable_timestamp_says_so_rather_than_lying() -> None:
    assert format_age(None) == "age unknown"
    assert format_age("not-a-date") == "age unknown"


@pytest.mark.parametrize(
    ("text", "seconds"),
    [("45s", 45.0), ("90m", 5400.0), ("24h", 86400.0), ("7d", 604800.0), ("2", 7200.0)],
)
def test_durations_parse(text: str, seconds: float) -> None:
    assert parse_duration(text) == seconds


def test_a_bad_duration_says_what_it_wanted() -> None:
    with pytest.raises(ValueError, match="30m, 24h, 7d"):
        parse_duration("yesterday")


def test_done_lines_carry_the_age() -> None:
    text = render(view(finished("20260915-1", reason=None, minutes=3)))
    assert "done    t (20260915-1) succeeded 3m ago" in text


def test_a_failure_reason_and_age_appear_together() -> None:
    text = render(view(finished("20260915-1", status="failed", reason="timeout", days=2)))
    assert "failed (timeout) 2d ago" in text


def test_recent_limits_how_many_finished_jobs_are_shown() -> None:
    jobs = [finished(f"job-{i}", minutes=i + 1) for i in range(8)]
    assert render(view(*jobs), recent=2).count("  done ") == 2
    assert render(view(*jobs), recent=8).count("  done ") == 8


def test_since_drops_older_jobs_and_says_how_many_it_hid() -> None:
    recent_job = finished("fresh", hours=1)
    old_job = finished("stale", days=9)
    text = render(view(recent_job, old_job), since_s=parse_duration("24h"))
    assert "fresh" in text and "stale" not in text
    only_old = render(view(old_job), since_s=parse_duration("24h"))
    assert "none in the last" in only_old and "1 older" in only_old


def test_within_needs_a_usable_ended_at() -> None:
    assert within(finished("a", minutes=1), 3600.0)
    assert not within(JobView(job_id="a", status="failed"), 3600.0)
    assert within(JobView(job_id="a", status="failed"), None)
