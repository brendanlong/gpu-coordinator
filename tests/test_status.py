from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any, cast

from gpuc.control.config import HostEntry
from gpuc.control.providers.base import Pod
from gpuc.control.status import (
    HostView,
    JobView,
    gather,
    host_json,
    job_views,
    owned_gpus,
    render,
)

GPU = "GPU-a"


def minutes_ago(minutes: float) -> str:
    return (datetime.now(UTC) - timedelta(minutes=minutes)).isoformat()


def payload(**overrides: Any) -> dict[str, Any]:
    document: dict[str, Any] = {
        "host": "gpubox",
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
    entry = HostEntry(name="gpubox", kind="ssh", ssh="me@box", gpus=[GPU, "GPU-b"])
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


def test_a_host_on_another_build_cannot_break_the_whole_status() -> None:
    """Every field here is another build's JSON, so none of it is trusted.

    One host answering with a string heartbeat used to raise out of `render`
    and take every other host's status with it.
    """
    document = payload(
        dispatcher_heartbeat_age_s="2.0",
        queue=[{"job_id": "j-queued"}, {"priority": 3}, "nonsense"],
        jobs=[
            {"name": "no id at all", "status": "running"},
            {
                "job_id": "j-running",
                "status": "running",
                "phase": "main",
                "gpus": [GPU],
                "util_recent": [None, "90", 90.0],
            },
        ],
    )
    queued, running, finished = job_views(document)
    assert (queued, finished) == ([], [])
    assert [j.job_id for j in running] == ["j-running"]
    assert running[0].priority is None
    assert running[0].util_recent == [90.0]

    host_view = HostView(entry=HostEntry(name="gpubox", kind="ssh", ssh="me@box"), reachable=True)
    host_view.queue, host_view.running, host_view.finished = queued, running, finished
    assert "dispatcher DOWN" in render(host_view)


def test_free_gpus_exclude_the_ones_a_running_job_holds() -> None:
    assert view().free == ["GPU-b"]


def test_a_busy_job_is_not_a_suspect() -> None:
    assert view().suspects == []


def test_a_full_window_of_floor_utilization_in_main_is_a_suspect() -> None:
    idle = view()
    idle.running[0].util_recent = [0.0] * 40
    assert [j.job_id for j in idle.suspects] == ["j-running"]
    assert "SUSPECT j-running" in render(idle, suspects_only=True)


def test_the_suspect_rule_is_the_jobs_own_low_util_settings() -> None:
    """The host's watchdog is per job, so `--suspects` has to be too."""
    off = view()
    off.running[0].util_recent = [0.0] * 40
    off.running[0].low_util.enabled = False
    assert off.suspects == []

    raised = view()
    raised.running[0].util_recent = [30.0] * 40
    assert raised.suspects == []
    raised.running[0].low_util.floor_pct = 50.0
    assert [j.job_id for j in raised.suspects] == ["j-running"]

    short_window = view()
    short_window.running[0].util_recent = [0.0] * 6
    assert short_window.suspects == []
    short_window.running[0].low_util.window_min = 1.0
    short_window.running[0].low_util.grace_min = 1.0
    assert [j.job_id for j in short_window.suspects] == ["j-running"]


def test_low_util_settings_come_from_the_hosts_payload() -> None:
    running = job_views(
        payload(
            jobs=[
                {
                    "job_id": "j",
                    "status": "running",
                    "phase": "main",
                    "gpus": [GPU],
                    "util_recent": [0.0] * 40,
                    "low_util": {"enabled": False},
                }
            ]
        )
    )[1]
    assert running[0].low_util.enabled is False
    assert running[0].suspect is False


def test_the_idle_and_no_suspect_lines_survive_gpu_and_pod_lines() -> None:
    """The "nothing here" lines are about the jobs, not about the host block.

    A host with GPUs (every real host) printed neither, because the sentinel
    compared against the whole rendered block rather than the job body.
    """
    empty = view(jobs=[], queue=[])
    empty.pod = Pod(id="pod1", name="gpuc-x", status="RUNNING", cost_usd_hr=0.4)
    assert "  gpu     " in render(empty)
    assert "idle; nothing queued, running or finished" in render(empty)
    assert "no suspects" in render(empty, suspects_only=True)


def test_a_long_setup_phase_is_never_a_suspect() -> None:
    setup = view()
    setup.running[0].phase = "setup"
    setup.running[0].util_recent = [0.0] * 40
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
    assert "host gpubox [ssh] me@box  dispatcher 2s ago  gpus 1/2 free" in text
    assert "running j-running train phase=main" in text
    assert "util 90%" in text
    assert "queued  j-queued next prio=10" in text
    assert "done    j-old prev failed (exit 17)" in text


def test_an_unreachable_host_says_what_to_run_next() -> None:
    down = HostView(entry=HostEntry(name="gpubox", kind="ssh", ssh="me@box"), error="ssh timed out")
    text = render(down)
    assert "UNREACHABLE" in text
    assert "gpuc host probe gpubox" in text


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


def test_the_two_utilizations_say_where_they_came_from() -> None:
    """The provider's number and the host sampler's legitimately differ; an
    unlabelled pair of percentages reads as a bug."""
    pod_view = view()
    pod_view.entry = HostEntry(name="pod", kind="runpod", pod_id="p1", gpus=[GPU, "GPU-b"])
    pod_view.pod = Pod(
        id="p1",
        name="gpuc-pod",
        status="RUNNING",
        cost_usd_hr=0.4,
        gpu_name="NVIDIA A40",
        gpu_utils=[71],
    )
    text = render(pod_view)
    assert "provider util 71%" in text
    assert "util 90% (host)" in text
    assert host_json(pod_view)["provider_util"] == [71]
    assert host_json(pod_view)["running"][0]["util"] == 90.0
    assert host_json(view())["provider_util"] is None


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


def test_ttl_is_measured_from_the_pod_createdat_not_the_registry() -> None:
    """The registry's created_at is when we heard of the pod; the reaper uses the provider's."""
    fresh_registration = view()
    fresh_registration.entry = HostEntry(
        name="pod", kind="runpod", ttl_hours=1.0, created_at=minutes_ago(5), gpus=[GPU]
    )
    fresh_registration.pod = Pod(
        id="pod1",
        name="gpuc-pod",
        status="RUNNING",
        cost_usd_hr=0.49,
        created_at=datetime.now(UTC) - timedelta(hours=3),
    )
    assert fresh_registration.past_ttl
    assert "PAST TTL" in render(fresh_registration)


# -- leftover workdirs --------------------------------------------------------

GIB = 1 << 30


def with_workdir_bytes(*sizes: int) -> HostView:
    jobs = [
        {
            "job_id": f"j-done-{i}",
            "status": "succeeded",
            "ended_at": minutes_ago(60 + i),
            "workdir_bytes": size,
        }
        for i, size in enumerate(sizes)
    ]
    return view(jobs=[*payload()["jobs"], *jobs])


def test_finished_workdirs_over_a_gigabyte_are_called_out() -> None:
    rendered = render(with_workdir_bytes(4 * GIB, 3 * GIB))
    assert "7.0 GiB still in 2 finished job workdir(s)" in rendered
    assert "gpuc clean --host gpubox --all-finished" in rendered


def test_a_small_leftover_is_not_worth_a_line() -> None:
    rendered = render(with_workdir_bytes(20 * 1024 * 1024))
    assert "gpuc clean" not in rendered


def test_a_running_job_never_counts_towards_leftover_disk() -> None:
    host_view = view(
        jobs=[
            {
                "job_id": "j-big",
                "status": "running",
                "phase": "main",
                "gpus": [GPU],
                "started_at": minutes_ago(5),
                "workdir_bytes": 40 * GIB,
            }
        ]
    )
    assert host_view.leftover_bytes == 0
    assert "gpuc clean" not in render(host_view)


def test_workdir_bytes_survives_the_host_payload() -> None:
    _, _, finished = job_views(
        payload(
            jobs=[
                {
                    "job_id": "j",
                    "status": "succeeded",
                    "ended_at": minutes_ago(1),
                    "workdir_bytes": 123,
                }
            ]
        )
    )
    assert finished[0].workdir_bytes == 123


def test_a_host_that_never_reports_sizes_is_fine() -> None:
    assert view().leftover_bytes == 0


def test_finished_jobs_with_unconfirmed_outputs_are_flagged() -> None:
    document = payload()
    document["jobs"].append(
        {
            "job_id": "j-pending",
            "name": "bulky",
            "status": "failed",
            "reason": "sync",
            "ended_at": minutes_ago(5),
            "outputs_pending": True,
        }
    )
    view = HostView(entry=HostEntry(name="gpubox", kind="ssh", ssh="me@box"), reachable=True)
    view.queue, view.running, view.finished = job_views(document)
    out = render(view)
    assert "outputs not uploaded" in out
    assert "never reached S3/HF: j-pending" in out


def test_a_lost_output_says_so_louder() -> None:
    document = payload()
    document["jobs"].append(
        {
            "job_id": "j-lost",
            "name": "bulky",
            "status": "succeeded",
            "ended_at": minutes_ago(5),
            "outputs_pending": True,
            "outputs_lost": True,
        }
    )
    view = HostView(entry=HostEntry(name="pod", kind="runpod"), reachable=True)
    view.queue, view.running, view.finished = job_views(document)
    assert "OUTPUTS LOST" in render(view)


def test_the_host_resolved_gpu_table_is_what_status_shows() -> None:
    """`config.gpus` may name cards by index, and only the host knows today's
    numbering -- so free/busy, and the per-card lines, come from its answer."""
    entry = HostEntry(name="gpubox", kind="ssh", ssh="me@box", gpus=["0", "7"])
    host_view = HostView(entry=entry, reachable=True, heartbeat_age_s=2.0)
    host_view.owned, host_view.indices = owned_gpus(
        payload(
            gpus=["0", "7"],
            gpus_resolved=[{"index": 0, "uuid": GPU}],
            gpus_unavailable=["7"],
        ),
        entry,
    )
    host_view.unavailable = ["7"]
    host_view.queue, host_view.running, host_view.finished = job_views(payload())

    assert host_view.owned == [GPU]
    assert host_view.free == []
    text = render(host_view)
    assert "gpu     [0] ?" in text and GPU in text
    assert "gpu     [7] UNAVAILABLE" in text
    assert host_json(host_view)["gpus"][-1] == {"owned_as": "7", "available": False}


def test_a_host_from_before_the_resolved_table_still_reports_its_gpus() -> None:
    entry = HostEntry(name="gpubox", kind="ssh", ssh="me@box", gpus=[GPU, "GPU-b"])
    owned, indices = owned_gpus(payload(), entry)
    assert owned == [GPU, "GPU-b"]
    assert indices == {}


def in_minutes(minutes: float) -> str:
    return (datetime.now(UTC) + timedelta(minutes=minutes)).isoformat()


def running_job(**overrides: Any) -> JobView:
    document: dict[str, Any] = {
        "job_id": "j-running",
        "name": "train",
        "status": "running",
        "phase": "main",
        "gpus": [GPU, "GPU-b"],
        "started_at": minutes_ago(30),
    }
    document.update(overrides)
    return JobView(**document)


def busy(*jobs: JobView, queued: list[JobView] | None = None) -> HostView:
    entry = HostEntry(name="gpubox", kind="ssh", ssh="me@box", gpus=[GPU, "GPU-b"])
    host_view = HostView(entry=entry, reachable=True, owned=[GPU, "GPU-b"], heartbeat_age_s=2.0)
    host_view.running = list(jobs)
    host_view.queue = list(queued or [])
    return host_view


def test_a_measured_eta_is_labelled_with_the_percentage_it_came_from() -> None:
    text = render(busy(running_job(eta=in_minutes(130), progress_pct=42.0)))
    assert "eta 2h10m (42%)" in text


def test_an_eta_with_no_measurement_behind_it_says_so() -> None:
    text = render(busy(running_job(eta=in_minutes(45), estimated_runtime_min=90.0)))
    assert "eta 45m (est)" in text


def test_a_job_past_its_own_eta_is_overdue_not_negative() -> None:
    text = render(busy(running_job(eta=minutes_ago(20), progress_pct=80.0)))
    assert "eta overdue (80%)" in text


def test_a_job_with_no_estimate_at_all_gets_no_eta_column() -> None:
    running = [line for line in render(busy(running_job())).splitlines() if "running" in line]
    assert running and not any("eta" in line for line in running)


def test_zero_percent_is_labelled_as_the_guess_it_still_is() -> None:
    """The runner does not replace the eta at 0%, so the eta on show is the
    submitter's estimate; tagging it `(0%)` would claim evidence."""
    text = render(busy(running_job(eta=in_minutes(90), progress_pct=0.0)))
    assert "eta 1h30m (est)" in text


def test_a_cpu_only_job_is_never_named_as_the_next_card_to_free_up() -> None:
    """`gpus: 0` jobs run but hold nothing, so a five-minute preprocessing job
    must not be offered as the reason a card frees up in five minutes."""
    text = render(
        busy(
            running_job(job_id="j-train", gpus=[GPU, "GPU-b"]),
            running_job(job_id="j-cpu", gpus=[], eta=in_minutes(5)),
        )
    )
    assert "free    " not in text
    text = render(
        busy(
            running_job(job_id="j-train", gpus=[GPU, "GPU-b"], eta=in_minutes(200)),
            running_job(job_id="j-cpu", gpus=[], eta=in_minutes(5)),
        )
    )
    assert "free    next card in ~3h20m (j-train)" in text


def test_a_cpu_only_job_is_not_counted_among_the_ones_that_gave_no_estimate() -> None:
    """It cannot free a card, so it is not a reason the real answer is sooner."""
    text = render(
        busy(
            running_job(job_id="j-known", gpus=[GPU, "GPU-b"], eta=in_minutes(200)),
            running_job(job_id="j-cpu", gpus=[]),
        )
    )
    assert "free    next card in ~3h20m (j-known)" in text
    assert "gave no estimate" not in text


def test_a_host_running_only_cpu_jobs_has_no_next_card_line() -> None:
    view = busy(running_job(job_id="j-cpu", gpus=[], eta=in_minutes(5)))
    assert "free    " not in render(view)


def test_a_multi_day_estimate_is_shown_in_days() -> None:
    queued = JobView(job_id="j-queued", priority=10, estimated_runtime_min=3 * 24 * 60 + 120)
    assert "est 3d02h" in render(busy(running_job(), queued=[queued]))


def test_the_json_carries_a_broken_progress_command() -> None:
    """Otherwise a typo'd progress command is indistinguishable from a job that
    never had one: the log line scrolls away under hours of training output."""
    _, running, _ = job_views(
        payload(
            jobs=[
                {
                    "job_id": "j-running",
                    "status": "running",
                    "progress_error": "progress command `cat p.txt` exited 1: no such file",
                }
            ]
        )
    )
    view = busy(running[0])
    assert host_json(view)["running"][0]["progress_error"].endswith("no such file")


def test_a_running_job_with_an_estimate_but_no_eta_still_shows_it() -> None:
    """The host publishes `eta` from a copy of the spec, so an estimate added
    to a job already running is in `--json` before it is in any eta. The text
    may never show less than `--json` does."""
    job = running_job(estimated_runtime_min=150.0)
    text = render(busy(job))
    assert "est 2h30m total" in text
    assert host_json(busy(job))["running"][0]["estimated_runtime_min"] == 150.0


def test_a_queued_job_shows_the_submitters_estimate() -> None:
    queued = JobView(job_id="j-queued", name="next", priority=10, estimated_runtime_min=360.0)
    assert "prio=10 est 6h00m" in render(busy(running_job(), queued=[queued]))


def test_a_fully_busy_host_says_when_the_next_card_frees_up() -> None:
    text = render(
        busy(
            running_job(job_id="j-long", gpus=[GPU], eta=in_minutes(200), progress_pct=30.0),
            running_job(job_id="j-short", gpus=["GPU-b"], eta=in_minutes(20), progress_pct=90.0),
        )
    )
    assert "free    next card in ~20m (j-short)" in text


def test_the_next_free_line_owns_up_to_the_jobs_it_could_not_estimate() -> None:
    text = render(
        busy(
            running_job(job_id="j-known", gpus=[GPU], eta=in_minutes(200)),
            running_job(job_id="j-silent", gpus=["GPU-b"]),
        )
    )
    assert "next card in ~3h20m (j-known); 1 other running job(s) gave no estimate" in text


def test_a_busy_host_where_nothing_estimated_anything_stays_quiet() -> None:
    text = render(busy(running_job(gpus=[GPU, "GPU-b"])))
    assert "free    " not in text


def test_a_host_with_a_free_card_does_not_guess_about_the_next_one() -> None:
    text = render(busy(running_job(gpus=[GPU], eta=in_minutes(200))))
    assert "free    " not in text


def test_the_json_carries_the_estimate_and_what_it_was_based_on() -> None:
    host_view = busy(running_job(eta=in_minutes(60), progress_pct=50.0))
    job = host_json(host_view)["running"][0]
    assert job["progress_pct"] == 50.0
    assert 3400 < job["eta_s"] < 3700


def test_job_views_read_the_estimate_fields_off_the_hosts_payload() -> None:
    _, running, _ = job_views(
        payload(
            jobs=[
                {
                    "job_id": "j-running",
                    "status": "running",
                    "phase": "main",
                    "progress_pct": 12.5,
                    "eta": in_minutes(10),
                    "estimated_runtime_min": 60,
                }
            ]
        )
    )
    assert running[0].progress_pct == 12.5
    assert running[0].estimated_runtime_min == 60.0
    assert running[0].eta_seconds is not None and running[0].eta_seconds > 0


def test_an_eta_a_host_reports_as_nonsense_is_ignored() -> None:
    _, running, _ = job_views(
        payload(
            jobs=[
                {
                    "job_id": "j-running",
                    "status": "running",
                    "eta": 1750000000,
                    "progress_pct": "half",
                }
            ]
        )
    )
    assert (running[0].eta, running[0].progress_pct, running[0].eta_seconds) == (None, None, None)
