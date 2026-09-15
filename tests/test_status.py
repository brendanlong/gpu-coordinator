from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from gpuc.control.config import HostEntry
from gpuc.control.status import HostView, JobView, job_views, render

GPU = "GPU-a"


def minutes_ago(minutes: float) -> str:
    return (datetime.now(UTC) - timedelta(minutes=minutes)).isoformat()


def payload(**overrides: Any) -> dict[str, Any]:
    document: dict[str, Any] = {
        "host": "spar",
        "gpus": [GPU, "GPU-b"],
        "ephemeral": False,
        "draining": False,
        "paused": False,
        "dispatcher_heartbeat_age_s": 2.0,
        "queue": [{"priority": 10, "job_id": "j-queued"}],
        "jobs": [
            {"job_id": "j-queued", "name": "next", "status": "queued", "attempt": 1},
            {
                "job_id": "j-running",
                "name": "train",
                "status": "running",
                "phase": "main",
                "gpus": [GPU],
                "started_at": minutes_ago(30),
                "util_recent": [90.0] * 20,
                "attempt": 2,
            },
            {
                "job_id": "j-old",
                "name": "prev",
                "status": "failed",
                "reason": "exit 17",
                "exit_code": 17,
                "ended_at": minutes_ago(60),
            },
        ],
    }
    document.update(overrides)
    return document


def view(**overrides: Any) -> HostView:
    entry = HostEntry(name="spar", kind="ssh", ssh="me@box", gpus=[GPU, "GPU-b"])
    host_view = HostView(entry=entry, reachable=True, owned=[GPU, "GPU-b"], heartbeat_age_s=2.0)
    host_view.queue, host_view.running, host_view.finished = job_views(payload(**overrides))
    return host_view


def test_jobs_are_split_by_status_and_carry_the_queue_priority() -> None:
    queued, running, finished = job_views(payload())
    assert [j.job_id for j in queued] == ["j-queued"]
    assert queued[0].priority == 10
    assert running[0].attempt == 2
    assert running[0].minutes is not None and 29 < running[0].minutes < 32
    assert finished[0].reason == "exit 17"


def test_free_gpus_exclude_the_ones_a_running_job_holds() -> None:
    assert view().free == ["GPU-b"]


def test_a_busy_job_is_not_a_suspect() -> None:
    assert view().suspects == []


def test_ten_minutes_of_floor_utilization_in_main_is_a_suspect() -> None:
    idle = view()
    idle.running[0].util_recent = [0.0] * 25
    assert [j.job_id for j in idle.suspects] == ["j-running"]
    assert "SUSPECT j-running" in render(idle, suspects_only=True)


def test_a_long_setup_phase_is_never_a_suspect() -> None:
    setup = view()
    setup.running[0].phase = "setup"
    setup.running[0].util_recent = [0.0] * 25
    assert setup.suspects == []
    assert "no suspects" in render(setup, suspects_only=True)


def test_a_job_with_too_few_samples_is_not_yet_a_suspect() -> None:
    fresh = view()
    fresh.running[0].util_recent = [0.0] * 5
    assert fresh.suspects == []


def test_a_cpu_only_job_is_never_a_suspect() -> None:
    cpu = JobView(job_id="j", status="running", phase="main", gpus=[], util_recent=[0.0] * 30)
    assert not cpu.suspect


def test_render_shows_the_host_line_queue_running_and_recent() -> None:
    text = render(view())
    assert "host spar [ssh] me@box  dispatcher 2s ago  gpus 1/2 free" in text
    assert "running j-running train phase=main" in text
    assert "util 90%" in text
    assert "queued  j-queued next prio=10" in text
    assert "done    j-old prev failed (exit 17)" in text


def test_an_unreachable_host_says_what_to_run_next() -> None:
    down = HostView(entry=HostEntry(name="spar", kind="ssh", ssh="me@box"), error="ssh timed out")
    text = render(down)
    assert "UNREACHABLE" in text
    assert "gpuc host probe spar" in text


def test_a_stale_heartbeat_reads_as_a_dead_dispatcher() -> None:
    stale = view()
    stale.heartbeat_age_s = 400.0
    assert not stale.dispatcher_alive
    assert "dispatcher DOWN" in render(stale)


def test_an_ephemeral_host_past_its_ttl_is_a_suspect() -> None:
    old = view()
    old.entry = HostEntry(
        name="pod", kind="runpod", ttl_hours=1.0, created_at=minutes_ago(180), gpus=[GPU]
    )
    assert old.past_ttl
    assert "older than 1.0h" in render(old, suspects_only=True)
