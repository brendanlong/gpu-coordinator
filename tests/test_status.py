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
    host_warnings,
    job_views,
    owned_gpus,
    queue_note,
    queue_placement,
    queue_start_estimates,
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
    assert "SUSPECT train (j-running)" in render(idle, suspects_only=True)


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
    assert "host gpubox [ssh]  gpus 1/2 free" in text
    assert "  dispatcher 2s ago" in text
    assert "running train (j-running) phase=main" in text
    assert "util 90%" in text
    assert "queued  next (j-queued) prio=10" in text
    assert "done    prev (j-old) failed (exit 17)" in text


def test_a_running_job_names_the_cards_it_holds_by_index() -> None:
    """The other direction from the gpu lines: they say busy, the job says which."""
    host_view = busy(running_job(gpus=[GPU, "GPU-b"]))
    host_view.indices = {GPU: 2, "GPU-b": 3}
    text = render(host_view)
    assert "gpu=2,3" in text
    assert "  gpu     [2] busy" in text and "  gpu     [3] busy" in text
    assert "j-running" not in "\n".join(
        line for line in text.splitlines() if line.startswith("  gpu ")
    )


def test_a_cpu_only_job_says_it_holds_no_card() -> None:
    assert "gpu=none" in render(busy(running_job(gpus=[])))


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
    # ...and on a host with no pod there is no second percentage to confuse it
    # with, so the tag is left off every running line.
    assert "(host)" not in render(view())
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
    assert "gpu     [0] busy ?" in text
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
    assert "gave no end time" not in text


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
    assert "next card in ~3h20m (j-known); 1 other running job(s) gave no end time" in text


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


def test_job_links_name_every_destination_and_quote_them() -> None:
    """The dashboard's anchors: S3 console, HF tree, W&B run, and the host's mirror.

    Derived from what the job declared and never checked, so the one thing
    they must get right is the URL itself: a repo name with a `?` in it
    would otherwise swallow the path into a query string.
    """
    from gpuc.control.status import job_links

    _, running, _ = job_views(
        payload(
            jobs=[
                {
                    "job_id": "j-running",
                    "status": "running",
                    "outputs": [
                        {"path": "results", "s3": "s3://bucket/lego/j-running"},
                        {"path": "ckpt", "hf": "me/repo?x", "hf_path": "runs/j-running"},
                        {"path": "extra", "hf": "me/plain"},
                        {"path": "nowhere"},
                    ],
                    "wandb": {"entity": "me", "project": "lego", "run_id": "r1"},
                }
            ]
        )
    )
    links = {
        (link["kind"], link["path"]): link["url"] for link in job_links(running[0], "s3://m/p/")
    }
    assert links[("s3", "results")] == (
        "https://s3.console.aws.amazon.com/s3/buckets/bucket?prefix=lego/j-running/"
    )
    assert links[("hf", "ckpt")] == "https://huggingface.co/me/repo%3Fx/tree/main/runs/j-running"
    assert links[("hf", "extra")] == "https://huggingface.co/me/plain"
    assert links[("wandb", None)] == "https://wandb.ai/me/lego/runs/r1"
    assert links[("mirror", None)] == (
        "https://s3.console.aws.amazon.com/s3/buckets/m?prefix=p/jobs/j-running/"
    )
    assert len(links) == 5


def test_job_links_need_a_whole_wandb_run_and_no_mirror_for_a_queued_job() -> None:
    from gpuc.control.status import job_links

    queued, _, _ = job_views(
        payload(jobs=[{"job_id": "j-queued", "status": "queued", "wandb": {"project": "lego"}}])
    )
    assert job_links(queued[0], "s3://m/p") == []


class _ScriptedSession:
    """A host that answers `status` with one payload and nothing else."""

    def __init__(self, document: dict[str, Any]) -> None:
        self.document = document

    def host_json(self, args: str, timeout: float = 60.0) -> Any:
        assert args == "status"
        return self.document


def test_gather_takes_the_build_and_the_config_from_the_hosts_own_answer() -> None:
    """Everything `gpuc status` says about what a host is is the host's, so a
    box configured from somebody else's laptop reads as what it now is."""
    entry = HostEntry(name="gpubox", kind="ssh", ssh="me@box", gpus=[GPU], pkg_commit="a" * 40)
    session = _ScriptedSession(payload(pkg_commit="c" * 40, gpus=["0", "1"]))
    got = gather(entry, session=cast(Any, session))
    assert got.pkg_commit == "c" * 40
    assert got.configured == {"host": "gpubox", "gpus": ["0", "1"]}
    assert host_json(got)["pkg_commit"] == "c" * 40


def test_a_reachable_host_that_never_reported_a_commit_is_not_read_as_current() -> None:
    """The `pkg_commit` key is newer than some hosts: one still running the
    build before it answers `status` without it, and that is the oldest code
    there is, not a reason to say nothing."""
    entry = HostEntry(name="gpubox", kind="ssh", ssh="me@box", gpus=[GPU], pkg_commit="a" * 40)
    document = payload()
    assert "pkg_commit" not in document, "this is what a host on the older build answers"
    got = gather(entry, session=cast(Any, _ScriptedSession(document)))
    assert got.reachable and got.pkg_commit is None
    assert host_json(got)["pkg_commit"] is None
    assert any("too old to say which" in warning for warning in host_warnings(got))


def waiting(job_id: str, **overrides: Any) -> JobView:
    document: dict[str, Any] = {"job_id": job_id, "priority": 50, "gpus_requested": 1}
    document.update(overrides)
    return JobView(**document)


def test_a_running_jobs_priority_comes_from_the_spec_when_its_marker_is_gone() -> None:
    """The one field that explains dispatch order was absent from every job the
    host had already started, because the queue marker is deleted at dispatch."""
    document = payload(
        queue=[{"priority": 10, "job_id": "j-queued"}],
        jobs=[
            {"job_id": "j-queued", "status": "queued", "priority": 50},
            {"job_id": "j-running", "status": "running", "priority": 88},
            {"job_id": "j-old", "status": "failed", "priority": 0},
        ],
    )
    queued, running, finished = job_views(document)
    # The marker is what the dispatcher orders by, so it wins over the spec's
    # copy while the job is still queued -- `gpuc reorder` moves the marker.
    assert queued[0].priority == 10
    assert running[0].priority == 88
    assert finished[0].priority == 0


def test_a_host_too_old_to_report_a_priority_says_nothing_rather_than_guessing() -> None:
    _, running, _ = job_views(payload(jobs=[{"job_id": "j-running", "status": "running"}]))
    assert running[0].priority is None


def test_a_queued_job_starts_now_when_a_card_is_already_free() -> None:
    view = busy(running_job(gpus=[GPU]), queued=[waiting("j-next")])
    assert queue_start_estimates(view) == {"j-next": 0.0}
    assert "starts now" in render(view)


def test_a_queued_job_waits_for_the_card_the_running_job_gives_back() -> None:
    view = busy(
        running_job(gpus=[GPU], eta=in_minutes(20)),
        running_job(job_id="j-other", gpus=["GPU-b"], eta=in_minutes(130)),
        queued=[
            waiting("j-next", estimated_runtime_min=60.0),
            waiting("j-after", estimated_runtime_min=60.0),
        ],
    )
    starts = queue_start_estimates(view)
    assert 19 * 60 < starts["j-next"] < 21 * 60
    # The first queued job takes that card at 20m and holds it for its own
    # hour, so the second one has it at 1h20m -- sooner than the 2h10m card.
    assert 79 * 60 < starts["j-after"] < 81 * 60
    assert "starts in ~20m" in render(view)


def test_a_job_that_gave_no_estimate_hides_only_the_jobs_behind_it() -> None:
    """A start time is evidence or it is absent. The job that inherits an
    unestimated job's card cannot be placed; one that needs no card from it can."""
    view = busy(
        running_job(gpus=[GPU], eta=in_minutes(20)),
        running_job(job_id="j-other", gpus=["GPU-b"], eta=in_minutes(130)),
        queued=[waiting("j-silent"), waiting("j-behind"), waiting("j-cpu", gpus_requested=0)],
    )
    starts = queue_start_estimates(view)
    assert 19 * 60 < starts["j-silent"] < 21 * 60
    assert 129 * 60 < starts["j-behind"] < 131 * 60
    assert starts["j-cpu"] == 0.0


def test_a_two_card_job_does_not_hold_up_the_one_card_job_behind_it() -> None:
    """The dispatcher walks the whole queue every pass rather than blocking on
    the head of it, so the estimate has to as well."""
    view = busy(
        running_job(gpus=[GPU], eta=in_minutes(45)),
        running_job(job_id="j-other", gpus=["GPU-b"], eta=in_minutes(90)),
        queued=[
            waiting("j-wide", gpus_requested=2),
            waiting("j-narrow", estimated_runtime_min=10.0),
        ],
    )
    starts = queue_start_estimates(view)
    # The one-card job takes the first card back at 45m and is done with it by
    # 55m; the two-card job still waits for the second card, at 90m.
    assert 44 * 60 < starts["j-narrow"] < 46 * 60
    assert 89 * 60 < starts["j-wide"] < 91 * 60
    assert "needs 2 gpus" in render(view)


def test_nothing_starts_on_a_paused_or_draining_host() -> None:
    view = busy(running_job(gpus=[GPU]), queued=[waiting("j-next")])
    view.paused = True
    assert queue_start_estimates(view) == {}
    view.paused, view.draining = False, True
    assert queue_start_estimates(view) == {}
    assert "starts" not in render(view)


def test_a_host_that_does_not_say_what_a_queued_job_asked_for_estimates_nothing() -> None:
    view = busy(
        running_job(gpus=[GPU]),
        queued=[waiting("j-unknown", gpus_requested=None), waiting("j-behind")],
    )
    assert queue_start_estimates(view) == {}


def test_the_json_carries_the_queue_order_and_when_each_job_starts() -> None:
    """`sort_by(.priority)` over `queued` has to reproduce dispatch order: it is
    the only thing that says whether your job runs next."""
    view = busy(
        running_job(gpus=[GPU], eta=in_minutes(20)),
        queued=[
            waiting("j-next", priority=10, gpus_requested=2),
            waiting("j-later", priority=90, estimated_runtime_min=10.0),
        ],
    )
    document = host_json(view)
    assert [(j["job_id"], j["priority"]) for j in document["queued"]] == [
        ("j-next", 10),
        ("j-later", 90),
    ]
    first = document["queued"][0]
    assert first["gpus_requested"] == 2
    assert 19 * 60 < first["starts_in_s"] < 21 * 60
    assert first["starts_at"] > datetime.now(UTC).isoformat()
    assert document["running"][0]["starts_in_s"] is None
    assert document["running"][0]["starts_at"] is None


def test_the_placement_of_a_job_in_its_hosts_queue() -> None:
    view = busy(
        running_job(gpus=[GPU], eta=in_minutes(20)),
        queued=[waiting("j-first"), waiting("j-second")],
    )
    placement = queue_placement(view, "j-second")
    assert (placement["queue_position"], placement["queue_length"]) == (2, 2)
    assert placement["dispatched"] is False
    note = queue_note(placement)
    assert note is not None and note.startswith("  queue: position 2 of 2; starts in ~")

    started = queue_placement(view, "j-running")
    assert (started["queue_position"], started["dispatched"]) == (None, True)
    assert queue_note(started) == "  queue: dispatched already; it is running now"


def test_an_unreachable_host_places_nothing_rather_than_reporting_an_empty_queue() -> None:
    """Null is not `not queued`: the job was enqueued before anything asked."""
    view = HostView(entry=HostEntry(name="gpubox", kind="ssh", ssh="me@box"), reachable=False)
    placement = queue_placement(view, "j-next")
    assert set(placement.values()) == {None}
    assert queue_note(placement) is None


def test_a_queued_job_whose_turn_cannot_be_dated_says_so_rather_than_nothing() -> None:
    view = busy(running_job(gpus=[GPU, "GPU-b"]), queued=[waiting("j-next")])
    note = queue_note(queue_placement(view, "j-next"))
    assert (
        note == "  queue: position 1 of 1; start time unknown (a job ahead of it gave no estimate)"
    )
