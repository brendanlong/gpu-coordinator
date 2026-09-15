from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any, cast

from gpuc.control.config import HostEntry
from gpuc.control.providers.base import Pod
from gpuc.control.status import HostView, JobView, gather, job_views, render

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


def test_null_utilization_samples_are_dropped() -> None:
    _, running, _ = job_views(
        {
            "jobs": [
                {
                    "job_id": "j",
                    "status": "running",
                    "phase": "main",
                    "gpus": [GPU],
                    "util_recent": [90.0, None, 80.0],
                }
            ]
        }
    )
    assert running[0].util_recent == [90.0, 80.0]
    assert not running[0].suspect


class _GoneProvider:
    """A provider that knows nothing about the pod the registry still lists."""

    def __init__(self, pod: Any = None) -> None:
        self._pod = pod
        self.asked: list[str] = []

    def get(self, pod_id: str) -> Any:
        self.asked.append(pod_id)
        return self._pod


def _runpod_entry() -> HostEntry:
    return HostEntry(
        name="gpuc-e2e-1", kind="runpod", ssh="root@1.2.3.4", port=22, pod_id="pod-1", gpus=[GPU]
    )


def test_a_host_whose_pod_is_gone_says_so_instead_of_trying_ssh() -> None:
    def explode(*_: Any, **__: Any) -> Any:
        raise AssertionError("status must not ssh to a pod that no longer exists")

    view = gather(_runpod_entry(), session=cast(Any, explode), provider=cast(Any, _GoneProvider()))
    assert view.pod_gone and not view.reachable
    assert "missing" in (view.error or "")
    text = render(view)
    assert "POD GONE" in text
    assert "gpuc reconcile --once" in text
    assert "host probe" not in text


def test_a_terminated_pod_reads_as_gone_too() -> None:
    terminated = Pod(
        id="pod-1",
        name="gpuc-e2e-1",
        status="TERMINATED",
        cost_usd_hr=0.0,
        gpu_name="A40",
    )
    view = gather(_runpod_entry(), provider=cast(Any, _GoneProvider(terminated)))
    assert view.pod_gone
    assert "TERMINATED" in render(view)
    assert "gpuc reconcile --once" in (view.error or "")
