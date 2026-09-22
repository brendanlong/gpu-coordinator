from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any, cast

import pytest

from gpuc.control.config import HostEntry
from gpuc.control.providers.base import Pod, PodStatus
from gpuc.control.status import (
    HostView,
    JobView,
    gather,
    host_json,
    host_warnings,
    job_json,
    job_views,
    no_start_reason,
    owned_gpus,
    queue_note,
    queue_placement,
    queue_start_estimates,
    render,
)
from gpuc.control.transport import TransportError
from tests.conftest import host_entry

GPU = "GPU-a"


def minutes_ago(minutes: float) -> str:
    return (datetime.now(UTC) - timedelta(minutes=minutes)).isoformat()


def payload(**overrides: Any) -> dict[str, Any]:
    document: dict[str, Any] = {
        "host": "gpubox",
        "gpus": [GPU, "GPU-b"],
        "ephemeral": False,
        "draining": False,
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
    entry = host_entry(name="gpubox", kind="ssh", ssh="me@box", gpus=[GPU, "GPU-b"])
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

    host_view = HostView(entry=host_entry(name="gpubox", kind="ssh", ssh="me@box"), reachable=True)
    host_view.queue, host_view.running, host_view.finished = queued, running, finished
    assert "dispatcher DOWN" in render(host_view)


def test_free_gpus_exclude_the_ones_a_running_job_holds() -> None:
    assert view().free == ["GPU-b"]


def test_the_idle_line_survives_gpu_and_pod_lines() -> None:
    """The "nothing here" line is about the jobs, not about the host block.

    A host with GPUs (every real host) never printed it, because the sentinel
    compared against the whole rendered block rather than the job body.
    """
    empty = view(jobs=[], queue=[])
    empty.pod = Pod(id="pod1", name="gpuc-x", status="RUNNING", cost_usd_hr=0.4)
    assert "  gpu     " in render(empty)
    assert "idle; nothing queued, running or finished" in render(empty)


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


def test_a_running_job_the_host_names_no_cards_for_still_renders() -> None:
    """Every job holds a card now, but a state file from an older build may
    not say so, and a file another build wrote is never a reason to fail."""
    assert "gpu=none" in render(busy(running_job(gpus=[])))


def test_an_unreachable_host_says_what_to_run_next() -> None:
    down = HostView(
        entry=host_entry(name="gpubox", kind="ssh", ssh="me@box"), error="ssh timed out"
    )
    text = render(down)
    assert "UNREACHABLE" in text
    assert "gpuc host probe gpubox" in text


def test_a_stale_heartbeat_reads_as_a_dead_dispatcher() -> None:
    stale = view()
    stale.heartbeat_age_s = 400.0
    assert not stale.dispatcher_alive
    assert "dispatcher DOWN" in render(stale)


def test_the_two_utilizations_say_where_they_came_from() -> None:
    """The provider's number and the host sampler's legitimately differ; an
    unlabelled pair of percentages reads as a bug."""
    pod_view = view()
    pod_view.entry = host_entry(name="pod", kind="runpod", pod_id="p1", gpus=[GPU, "GPU-b"])
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


def _pod(status: PodStatus) -> Pod:
    return Pod(id="pod-1", name="gpuc-e2e-1", status=status, cost_usd_hr=0.0, gpu_name="A40")


class _GoneProvider:
    """A provider that knows nothing about the pod the registry still lists."""

    def __init__(self, pod: Any = None) -> None:
        self._pod = pod
        self.asked: list[str] = []

    def get(self, pod_id: str) -> Any:
        self.asked.append(pod_id)
        return self._pod


def _runpod_entry() -> HostEntry:
    return host_entry(
        name="gpuc-e2e-1", kind="runpod", ssh="root@1.2.3.4", port=22, pod_id="pod-1", gpus=[GPU]
    )


def test_a_host_whose_pod_is_gone_says_so_instead_of_trying_ssh() -> None:
    def explode(*_: Any, **__: Any) -> Any:
        raise AssertionError("status must not ssh to a pod that no longer exists")

    view = gather(_runpod_entry(), session=cast(Any, explode), provider=cast(Any, _GoneProvider()))
    assert view.pod_gone and view.pod_terminated and not view.reachable
    assert "missing" in (view.error or "")
    # The provider answered, so this is a rental that ended rather than a host
    # the command could not read.
    assert view.failure is None
    text = render(view)
    assert "POD GONE" in text
    assert "forgetting this host" in text
    assert "host probe" not in text


def test_a_terminated_pod_reads_as_gone_too() -> None:
    terminated = _pod("TERMINATED")
    view = gather(_runpod_entry(), provider=cast(Any, _GoneProvider(terminated)))
    assert view.pod_gone and view.pod_terminated
    assert "TERMINATED" in render(view)
    assert "forgetting this host" in (view.error or "")


def test_a_stopped_pod_is_gone_but_not_forgotten() -> None:
    """An EXITED pod is one the provider still has, so the entry is left alone."""
    view = gather(_runpod_entry(), provider=cast(Any, _GoneProvider(_pod("EXITED"))))
    assert view.pod_gone and not view.pod_terminated
    assert view.failure is None
    assert "gpuc host remove gpuc-e2e-1" in (view.error or "")


class _RefusingSession:
    """A host that is registered and does not answer."""

    def host_json(self, *_: Any, **__: Any) -> Any:
        raise TransportError(message="ssh: connect to 1.2.3.4 port 22: No route to host")


def test_a_host_that_could_not_be_read_is_a_failure() -> None:
    view = gather(_runpod_entry(), session=cast(Any, _RefusingSession()), provider=None)
    assert not view.reachable
    assert view.failure == "ssh: connect to 1.2.3.4 port 22: No route to host"


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


def test_a_finished_job_the_host_said_nothing_about_holds_no_disk() -> None:
    """The host answers for every finished job, so a null is not a hidden pile:
    it is a build too old to answer, which the build warning covers."""
    quiet = view(jobs=[{"job_id": "j-unsized", "status": "succeeded", "ended_at": minutes_ago(60)}])
    assert quiet.leftover_bytes == 0
    assert "gpuc clean" not in render(quiet)


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
    view = HostView(entry=host_entry(name="gpubox", kind="ssh", ssh="me@box"), reachable=True)
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
    view = HostView(entry=host_entry(name="pod", kind="runpod"), reachable=True)
    view.queue, view.running, view.finished = job_views(document)
    assert "OUTPUTS LOST" in render(view)


def test_the_host_resolved_gpu_table_is_what_status_shows() -> None:
    """`config.gpus` may name cards by index, and only the host knows today's
    numbering -- so free/busy, and the per-card lines, come from its answer."""
    entry = host_entry(name="gpubox", kind="ssh", ssh="me@box", gpus=["0", "7"])
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
    entry = host_entry(name="gpubox", kind="ssh", ssh="me@box", gpus=[GPU, "GPU-b"])
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
    entry = host_entry(name="gpubox", kind="ssh", ssh="me@box", gpus=[GPU, "GPU-b"])
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
    entry = host_entry(name="gpubox", kind="ssh", ssh="me@box", gpus=[GPU], pkg_commit="a" * 40)
    session = _ScriptedSession(payload(pkg_commit="c" * 40, gpus=["0", "1"]))
    got = gather(entry, session=cast(Any, session))
    assert got.pkg_commit == "c" * 40
    assert got.owned == ["0", "1"]
    assert host_json(got)["pkg_commit"] == "c" * 40


def test_a_reachable_host_that_never_reported_a_commit_is_not_read_as_current() -> None:
    """The `pkg_commit` key is newer than some hosts: one still running the
    build before it answers `status` without it, and that is the oldest code
    there is, not a reason to say nothing."""
    entry = host_entry(name="gpubox", kind="ssh", ssh="me@box", gpus=[GPU], pkg_commit="a" * 40)
    document = payload()
    assert "pkg_commit" not in document, "this is what a host on the older build answers"
    got = gather(entry, session=cast(Any, _ScriptedSession(document)))
    assert got.reachable and got.pkg_commit is None
    assert host_json(got)["pkg_commit"] is None
    assert any("too old to say which" in warning for warning in host_warnings(got))


def waiting(job_id: str, **overrides: Any) -> JobView:
    # A job the host reported in full, so `use_shared` is the spec's own `false`
    # and not the null it sends for a spec it could not read. The tests about
    # that unknown pass `use_shared=None` and say so.
    document: dict[str, Any] = {
        "job_id": job_id,
        "priority": 50,
        "gpus_requested": 1,
        "use_shared": False,
    }
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


def test_the_highest_priority_there_is_still_comes_from_the_marker() -> None:
    """`0` is a real priority, so the fallback to the spec cannot be an `or`:
    a job reordered to the front would report the priority it was submitted at."""
    document = payload(
        queue=[{"priority": 0, "job_id": "j-queued"}],
        jobs=[{"job_id": "j-queued", "status": "queued", "priority": 50}],
    )
    queued, _, _ = job_views(document)
    assert queued[0].priority == 0


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
        queued=[waiting("j-silent"), waiting("j-behind")],
    )
    starts = queue_start_estimates(view)
    assert 19 * 60 < starts["j-silent"] < 21 * 60
    assert 129 * 60 < starts["j-behind"] < 131 * 60


def test_a_two_card_job_holds_up_the_one_card_job_behind_it() -> None:
    """The dispatcher takes the queue in order and a job that does not fit
    holds the free cards it is waiting for, so the estimate has to as well: the
    card that comes back at 45m is not the narrow job's to take."""
    view = busy(
        running_job(gpus=[GPU], eta=in_minutes(45)),
        running_job(job_id="j-other", gpus=["GPU-b"], eta=in_minutes(90)),
        queued=[
            waiting("j-wide", gpus_requested=2),
            waiting("j-narrow", estimated_runtime_min=10.0),
        ],
    )
    starts = queue_start_estimates(view)
    assert 89 * 60 < starts["j-wide"] < 91 * 60
    # And the job behind it has no turn anyone can name: the wide job takes
    # both cards at 90m and never said when it would be done with them.
    assert "j-narrow" not in starts
    assert "gave no end time" in no_start_reason(view, view.queue[1])
    assert "needs 2 gpus" in render(view)


def test_a_job_behind_a_wide_one_starts_when_the_wide_one_is_done() -> None:
    view = busy(
        running_job(gpus=[GPU], eta=in_minutes(45)),
        running_job(job_id="j-other", gpus=["GPU-b"], eta=in_minutes(90)),
        queued=[
            waiting("j-wide", gpus_requested=2, estimated_runtime_min=30.0),
            waiting("j-narrow", estimated_runtime_min=10.0),
        ],
    )
    starts = queue_start_estimates(view)
    # Both cards at 90m, held for the wide job's own half hour.
    assert 89 * 60 < starts["j-wide"] < 91 * 60
    assert 119 * 60 < starts["j-narrow"] < 121 * 60


def test_a_job_waiting_for_a_missing_owned_card_holds_up_the_queue() -> None:
    """The host holds for a job whose owned card has dropped off nvidia-smi,
    so the projection may not hand the card that comes back at 45m to the job
    behind it -- and the reason has to name the missing card, since the host
    is the thing to fix."""
    view = busy(
        running_job(gpus=[GPU], eta=in_minutes(45)),
        queued=[
            waiting("j-wide", gpus_requested=2),
            waiting("j-narrow"),
            waiting("j-wide-too", gpus_requested=2),
        ],
    )
    view.owned = [GPU]
    view.unavailable = ["7"]
    assert queue_start_estimates(view) == {}
    reason = no_start_reason(view, view.queue[0])
    assert "only 1 of the 2 this host owns answer to nvidia-smi (7 missing)" in reason
    assert "j-wide is ahead of it" in no_start_reason(view, view.queue[1])
    # The job at the front is the one holding the queue for the missing card;
    # a second two-card job behind it is waiting on the queue, not the card.
    assert "j-wide is ahead of it" in no_start_reason(view, view.queue[2])
    assert "missing" not in no_start_reason(view, view.queue[2])
    assert "a job waiting for it holds the queue" in render(view)


def test_a_job_behind_one_that_can_never_be_placed_says_which_job() -> None:
    """Not "the cards it needs": this job is waiting on the queue, not on a
    card, and the job ahead of it is the whole reason."""
    view = busy(
        running_job(gpus=[GPU]),  # no eta, so its card is not schedulable
        running_job(job_id="j-other", gpus=["GPU-b"], eta=in_minutes(90)),
        queued=[waiting("j-wide", gpus_requested=2), waiting("j-narrow")],
    )
    assert queue_start_estimates(view) == {}
    assert "j-wide is ahead of it" in no_start_reason(view, view.queue[1])


def test_nothing_starts_on_a_draining_host() -> None:
    view = busy(running_job(gpus=[GPU]), queued=[waiting("j-next")])
    view.draining = True
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


def test_a_queued_job_whose_turn_cannot_be_dated_says_why() -> None:
    """Each reason is a different thing to do about it, and "a job ahead of it"
    is a lie to tell the only job in the queue of a host that is draining."""
    view = busy(running_job(gpus=[GPU, "GPU-b"]), queued=[waiting("j-next")])
    note = queue_note(queue_placement(view, "j-next"))
    assert note is not None and note.endswith(
        "start time unknown (the jobs holding the cards it needs gave no end time)"
    )

    view.draining = True
    assert "host gpubox is draining" in str(queue_note(queue_placement(view, "j-next")))

    view.draining = False
    view.queue = [waiting("j-next", gpus_requested=4)]
    assert "asks for 4 card(s) and the host has 2" in str(
        queue_note(queue_placement(view, "j-next"))
    )

    view.queue = [waiting("j-next", gpus_requested=None)]
    assert "does not report how many cards" in str(queue_note(queue_placement(view, "j-next")))


# -- shared GPUs --------------------------------------------------------------

SHARED = "GPU-shared"


def shared_payload(**overrides: Any) -> dict[str, Any]:
    document = payload(**overrides)
    document.setdefault(
        "shared_gpus_resolved",
        [
            {
                "index": 4,
                "uuid": SHARED,
                "memory_mib": 0.0,
                "utilization_pct": 0.0,
                "unused": True,
            }
        ],
    )
    document.setdefault("shared_gpus_unavailable", [])
    return document


def shared_view(**overrides: Any) -> HostView:
    entry = host_entry(
        name="gpubox", kind="ssh", ssh="me@box", gpus=[GPU, "GPU-b"], shared_gpus=[SHARED]
    )
    return gather(entry, session=cast(Any, _ScriptedSession(shared_payload(**overrides))))


def test_gather_reads_the_shared_cards_and_what_is_on_them() -> None:
    got = shared_view()
    assert [card.uuid for card in got.shared] == [SHARED]
    assert got.shared[0].unused and got.shared[0].index == 4
    assert [card.uuid for card in got.borrowable] == [SHARED]


def test_a_shared_card_somebody_else_is_on_is_rendered_with_their_numbers() -> None:
    """The question this block gets asked is "there is a card there, why is my
    job queued", and the answer is whose it is right now."""
    got = shared_view(
        shared_gpus_resolved=[
            {
                "index": 4,
                "uuid": SHARED,
                "memory_mib": 21504.0,
                "utilization_pct": 98.0,
                "unused": False,
            }
        ]
    )
    rendered = render(got)
    assert "shared  [4] IN USE" in rendered
    assert "(somebody else: 21504 MiB, 98% util)" in rendered
    assert got.borrowable == []


def test_a_shared_card_one_of_our_own_jobs_holds_reads_busy_not_in_use() -> None:
    got = shared_view(
        jobs=[
            {
                "job_id": "j-running",
                "name": "train",
                "status": "running",
                "phase": "main",
                "gpus": [SHARED],
                "started_at": minutes_ago(5),
            }
        ],
        shared_gpus_resolved=[
            {
                "index": 4,
                "uuid": SHARED,
                "memory_mib": 8192.0,
                "utilization_pct": 90.0,
                "unused": False,
            }
        ],
    )
    assert "shared  [4] busy" in render(got)
    assert got.borrowable == []


def test_a_shared_entry_the_host_cannot_see_says_so() -> None:
    got = shared_view(shared_gpus_resolved=[], shared_gpus_unavailable=["7"])
    assert "shared  [7] UNAVAILABLE" in render(got)
    assert got.shared_unavailable == ["7"]


def test_a_host_on_an_older_build_simply_has_no_shared_cards() -> None:
    entry = host_entry(name="gpubox", kind="ssh", ssh="me@box", gpus=[GPU])
    got = gather(entry, session=cast(Any, _ScriptedSession(payload())))
    assert got.shared == []
    assert "shared" not in render(got)


def test_the_json_carries_the_shared_cards_and_their_verdict() -> None:
    document = host_json(shared_view())
    assert document["shared_gpus"] == [
        {
            "index": 4,
            "uuid": SHARED,
            "name": "",
            "vram_mib": None,
            "busy_job": None,
            "memory_mib": 0.0,
            "utilization_pct": 0.0,
            "unused": True,
        }
    ]


def test_the_json_says_which_jobs_may_have_a_shared_card() -> None:
    """`shared_gpus` says which cards the host may borrow and this says which
    jobs may have them; without it the document shows an idle shared card
    beside two queued jobs and cannot say why only one of them starts.

    Three answers, not two: the host sends null for a job whose spec it could
    not read, and `false` there would say "this one did not ask" about a job
    nobody asked.
    """
    document = host_json(
        shared_view(
            jobs=[
                {"job_id": "j-borrower", "status": "queued", "use_shared": True},
                {"job_id": "j-purist", "status": "queued", "use_shared": False},
                {"job_id": "j-unreadable", "status": "queued", "use_shared": None},
                {"job_id": "j-running", "status": "running", "gpus": [GPU], "use_shared": True},
            ]
        )
    )
    assert {j["job_id"]: j["use_shared"] for j in document["queued"]} == {
        "j-borrower": True,
        "j-purist": False,
        "j-unreadable": None,
    }
    assert document["running"][0]["use_shared"] is True


def test_a_host_too_old_to_report_use_shared_says_null_not_false() -> None:
    """The field is absent from the whole payload, which is a build predating
    shared cards. It borrows nothing, and it must not claim these jobs asked
    to borrow nothing either."""
    queued, _, _ = job_views(payload(jobs=[{"job_id": "j-queued", "status": "queued"}]))
    assert queued[0].use_shared is None
    assert job_json(queued[0])["use_shared"] is None


def test_a_job_waiting_for_a_shared_card_is_told_that_and_not_called_impossible() -> None:
    """The host owns two cards and the job wants three, and the shared one is
    somebody else's for now: without it that reads "it will never be
    dispatched", which would be a lie."""
    got = shared_view(shared_gpus_resolved=[somebody_elses_shared_card()])
    job = waiting("j-queued", gpus_requested=3, use_shared=True)
    got.queue = [job]
    reason = no_start_reason(got, job)
    assert "needs 1 shared card(s) somebody else is using" in reason
    assert "not something this host can predict" in reason


def test_a_job_that_cannot_fit_even_with_shared_cards_is_still_called_impossible() -> None:
    got = shared_view()
    job = waiting("j-queued", gpus_requested=9, use_shared=True)
    got.queue = [job]
    assert "the host has 3 (shared included)" in no_start_reason(got, job)


def test_a_job_whose_permission_to_borrow_is_unknown_is_not_called_impossible() -> None:
    """`never be dispatched` turns on whether it may borrow, and the host did
    not say. Reading the unknown as no would tell a submitter their job is
    hopeless on the strength of a spec nobody could read."""
    got = shared_view()
    job = waiting("j-queued", gpus_requested=3, use_shared=None)
    got.queue = [job]
    reason = no_start_reason(got, job)
    assert "never be dispatched" not in reason
    assert "does not report whether a queued job may borrow" in reason


def test_a_job_that_did_not_ask_is_not_credited_with_the_shared_card() -> None:
    got = shared_view()
    job = waiting("j-queued", gpus_requested=3)
    got.queue = [job]
    assert "the host has 2, so it will never be dispatched" in no_start_reason(got, job)


def somebody_elses_shared_card() -> dict[str, Any]:
    """The shared card as the host reports it while its real owner is on it."""
    return {
        "index": 4,
        "uuid": SHARED,
        "memory_mib": 21504.0,
        "utilization_pct": 98.0,
        "unused": False,
    }


def busy_owned(eta_minutes: float | None, gpus: list[str] | None = None) -> list[dict[str, Any]]:
    """The owned cards (both by default) held by one job, with its eta if any."""
    return [
        {
            "job_id": "j-running",
            "name": "train",
            "status": "running",
            "phase": "main",
            "gpus": [GPU, "GPU-b"] if gpus is None else gpus,
            "started_at": minutes_ago(1),
            "eta": None
            if eta_minutes is None
            else (datetime.now(UTC) + timedelta(minutes=eta_minutes)).isoformat(),
        }
    ]


def test_a_borrowing_job_is_not_told_it_waits_for_the_owned_cards() -> None:
    """The bug this is for: a one-card borrower on a host whose own cards are
    six hours from free was told `starts in ~6h`, when the dispatcher's very
    next pass hands it the idle shared card."""
    got = shared_view(jobs=busy_owned(360.0))
    got.queue = [waiting("j-queued", gpus_requested=1, use_shared=True)]
    assert queue_start_estimates(got) == {"j-queued": 0.0}
    assert queue_note(queue_placement(got, "j-queued")) is not None


def test_a_job_that_did_not_ask_still_waits_for_the_owned_cards() -> None:
    got = shared_view(jobs=busy_owned(360.0))
    got.queue = [waiting("j-queued", gpus_requested=1)]
    assert queue_start_estimates(got)["j-queued"] == pytest.approx(360.0 * 60.0, abs=1.0)


def test_a_shared_card_somebody_else_holds_is_not_scheduled_onto_at_all() -> None:
    """When they will stop is the one thing this host cannot know, so the card
    is left out rather than given a release time."""
    got = shared_view(jobs=busy_owned(360.0), shared_gpus_resolved=[somebody_elses_shared_card()])
    got.queue = [waiting("j-queued", gpus_requested=1, use_shared=True)]
    assert queue_start_estimates(got)["j-queued"] == pytest.approx(360.0 * 60.0, abs=1.0)


def test_a_wide_borrower_short_of_an_owned_card_holds_like_any_other() -> None:
    """The dispatcher steps over a job only when it could not fit even once
    every job of ours ends. This one wants more than the host owns, but the
    shared card is idle and what it is short of is an owned card back at 45m:
    it holds, so the projection may not hand that card to the job behind it."""
    got = shared_view(jobs=busy_owned(45.0, gpus=[GPU]))
    got.queue = [
        waiting("j-wide", gpus_requested=3, use_shared=True, estimated_runtime_min=30.0),
        waiting("j-narrow"),
    ]
    starts = queue_start_estimates(got)
    assert starts["j-wide"] == pytest.approx(45.0 * 60.0, abs=1.0)
    assert starts["j-narrow"] == pytest.approx(75.0 * 60.0, abs=1.0)


def test_a_wide_borrower_that_holds_is_not_told_it_waits_on_somebody_else() -> None:
    """Same shape, but the job holding our card gave no eta: the wide job is
    waiting on that job, not on the idle shared card, and the reason says so."""
    got = shared_view(jobs=busy_owned(None, gpus=[GPU]))
    got.queue = [waiting("j-wide", gpus_requested=3, use_shared=True), waiting("j-narrow")]
    assert queue_start_estimates(got) == {}
    assert "gave no end time" in no_start_reason(got, got.queue[0])
    assert "j-wide is ahead of it" in no_start_reason(got, got.queue[1])


def test_a_borrower_short_of_somebody_elses_card_is_stepped_over() -> None:
    """The exemption itself, from the client's side: the shared card is in use
    and every card of ours would still leave this job short, so the host steps
    over it and the one-card job behind it gets the card that frees at 45m."""
    got = shared_view(
        jobs=busy_owned(45.0, gpus=[GPU]), shared_gpus_resolved=[somebody_elses_shared_card()]
    )
    got.queue = [waiting("j-wide", gpus_requested=3, use_shared=True), waiting("j-narrow")]
    starts = queue_start_estimates(got)
    assert "j-wide" not in starts
    assert starts["j-narrow"] == 0.0
    assert "somebody else is using" in no_start_reason(got, got.queue[0])


def test_there_is_only_one_shared_card_to_go_round() -> None:
    """A borrowed card is a card, not a loophole: the second borrower waits for
    the owned ones exactly as it would have without any sharing at all."""
    got = shared_view(jobs=busy_owned(360.0))
    got.queue = [
        waiting("j-first", gpus_requested=1, use_shared=True),
        waiting("j-second", gpus_requested=1, use_shared=True, priority=60),
    ]
    starts = queue_start_estimates(got)
    assert starts["j-first"] == 0.0
    assert starts["j-second"] == pytest.approx(360.0 * 60.0, abs=1.0)


def test_a_host_that_only_borrows_does_not_read_as_having_no_gpus() -> None:
    got = shared_view()
    got.owned = []
    assert "shared 1/1 free, none owned" in render(got)


def test_a_shared_card_the_host_cannot_see_still_counts_against_never() -> None:
    """The host makes a job wait for a card that is missing this minute; it
    does not fail it. `it will never be dispatched` may not disagree."""
    got = shared_view(shared_gpus_resolved=[], shared_gpus_unavailable=["7"])
    job = waiting("j-queued", gpus_requested=3, use_shared=True)
    got.queue = [job]
    assert "never be dispatched" not in no_start_reason(got, job)
