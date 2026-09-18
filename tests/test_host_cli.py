from __future__ import annotations

import json
from pathlib import Path

import pytest

from gpuc.host import __main__ as cli
from gpuc.host import dispatcher, jobs, paths, queue
from gpuc.host.jobs import HostConfig
from tests.conftest import FAKE_GPUS, make_spec


def run(capsys: pytest.CaptureFixture[str], *args: str) -> tuple[int, object]:
    code = cli.main(list(args))
    out = capsys.readouterr().out
    return code, json.loads(out) if out.strip() else None


def test_enqueue_from_a_file(
    gpuc_home: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    spec_path = tmp_path / "job.json"
    spec_path.write_text(json.dumps({"name": "demo", "command": "true", "gpus": 0}))
    code, payload = run(capsys, "enqueue", str(spec_path))
    assert code == 0
    assert isinstance(payload, dict)
    job_id = payload["job_id"]
    assert jobs.read_spec(job_id).name == "demo"
    assert [e.job_id for e in queue.list_queued()] == [job_id]


def test_enqueue_starts_a_dispatcher(
    gpuc_home: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started: list[int] = []
    monkeypatch.setattr(dispatcher, "spawn_detached_dispatcher", lambda: started.append(1) or 4242)
    spec_path = tmp_path / "job.json"
    spec_path.write_text(json.dumps({"command": "true", "gpus": 0}))
    _, payload = run(capsys, "enqueue", str(spec_path))
    assert isinstance(payload, dict)
    assert payload["dispatcher_pid"] == 4242
    assert started == [1]


def test_enqueue_from_stdin(
    gpuc_home: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "sys.stdin", __import__("io").StringIO(json.dumps({"command": "true", "gpus": 0}))
    )
    code, payload = run(capsys, "enqueue", "-")
    assert code == 0
    assert isinstance(payload, dict)
    assert jobs.read_state(payload["job_id"]).status == "queued"


def test_config_merge_replaces_the_keys_it_is_given_and_no_others(
    gpuc_home: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`gpuc host set` is this: the host applies the patch, atomically, with
    the same code the dispatcher reads the file with."""
    patch = tmp_path / "patch.json"
    patch.write_text(json.dumps({"gpus": ["0"], "retention_days": 7.0, "env": {"HF_HOME": "/big"}}))
    code, payload = run(capsys, "config", "--merge", str(patch))
    assert code == 0
    assert isinstance(payload, dict)
    assert (payload["gpus"], payload["retention_days"]) == (["0"], 7.0)
    # Untouched: the host's name, and every key the patch did not name.
    assert payload["host"] == "test-host"
    on_disk = jobs.read_config()
    assert (on_disk.gpus, on_disk.retention_days, on_disk.env) == (["0"], 7.0, {"HF_HOME": "/big"})


def test_config_merge_keeps_the_keys_this_build_does_not_know(
    gpuc_home: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A control machine on a newer build may have written fields this host has
    never heard of; a `host set` from an older one must not drop them."""
    document = json.loads(paths.config_file().read_text())
    document["power_cap_watts"] = 220
    paths.config_file().write_text(json.dumps(document))
    patch = tmp_path / "patch.json"
    patch.write_text(json.dumps({"idle_minutes": 5.0}))
    _, payload = run(capsys, "config", "--merge", str(patch))
    assert isinstance(payload, dict)
    assert payload["power_cap_watts"] == 220
    assert payload["idle_minutes"] == 5.0


def test_config_merge_can_clear_a_nullable_field(
    gpuc_home: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    jobs.write_config(HostConfig(host="test-host", retention_days=14.0, s3_prefix="s3://b/p"))
    patch = tmp_path / "patch.json"
    patch.write_text(json.dumps({"retention_days": None, "s3_prefix": None}))
    _, payload = run(capsys, "config", "--merge", str(patch))
    assert isinstance(payload, dict)
    assert payload["retention_days"] is None and payload["s3_prefix"] is None
    assert jobs.read_config().retention_days is None


def test_config_merge_on_a_host_with_no_config_writes_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("GPUC_HOME", str(tmp_path / "gpuc-home"))
    patch = tmp_path / "patch.json"
    patch.write_text(json.dumps({"host": "fresh", "gpus": ["GPU-a"]}))
    code, payload = run(capsys, "config", "--merge", str(patch))
    assert code == 0
    assert isinstance(payload, dict)
    assert (payload["host"], payload["gpus"]) == ("fresh", ["GPU-a"])
    assert payload["schema_version"] == 1
    assert jobs.read_config().host == "fresh"


def test_status(gpuc_home: Path, capsys: pytest.CaptureFixture[str]) -> None:
    job_id = queue.enqueue(make_spec(name="n", priority=12))
    _, status = run(capsys, "status")
    assert isinstance(status, dict)
    assert status["host"] == "test-host"
    assert status["ephemeral"] is False
    assert status["jobs"][0]["job_id"] == job_id
    assert status["queue"] == [{"priority": 12, "job_id": job_id}]


def test_status_of_one_job(gpuc_home: Path, capsys: pytest.CaptureFixture[str]) -> None:
    job_id = queue.enqueue(make_spec())
    queue.enqueue(make_spec())
    _, status = run(capsys, "status", job_id)
    assert isinstance(status, dict)
    assert [j["job_id"] for j in status["jobs"]] == [job_id]


def test_cancel_and_reorder(gpuc_home: Path, capsys: pytest.CaptureFixture[str]) -> None:
    first = queue.enqueue(make_spec(priority=50))
    second = queue.enqueue(make_spec(priority=50))
    code, payload = run(capsys, "reorder", second, "3")
    assert code == 0 and payload == {"job_id": second, "status": "queued", "priority": 3}
    assert [e.job_id for e in queue.list_queued()] == [second, first]

    code, payload = run(capsys, "cancel", first)
    assert code == 0 and isinstance(payload, dict) and payload["status"] == "cancelled"

    code, _ = run(capsys, "reorder", "no-such-job", "1")
    assert code == 1


def test_estimate_sets_a_queued_jobs_runtime(
    gpuc_home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    job_id = queue.enqueue(make_spec())
    code, payload = run(capsys, "estimate", job_id, "150")
    assert code == 0 and isinstance(payload, dict)
    assert payload["estimated_runtime_min"] == 150.0 and payload["status"] == "queued"
    assert jobs.read_spec(job_id).estimated_runtime_min == 150.0

    code, payload = run(capsys, "estimate", job_id, "--clear")
    assert code == 0 and isinstance(payload, dict) and payload["estimated_runtime_min"] is None
    assert jobs.read_spec(job_id).estimated_runtime_min is None


def test_estimate_keeps_the_keys_this_build_does_not_know(gpuc_home: Path) -> None:
    """A spec written by a newer build round-tripped through `JobSpec` would
    lose them, and an estimate is not a reason to rewrite somebody's job."""
    job_id = queue.enqueue(make_spec())
    document = json.loads(paths.spec_file(job_id).read_text())
    document["some_future_field"] = ["keep", "me"]
    paths.spec_file(job_id).write_text(json.dumps(document))
    jobs.update_spec(job_id, estimated_runtime_min=42.0)
    written = json.loads(paths.spec_file(job_id).read_text())
    assert written["some_future_field"] == ["keep", "me"]
    assert written["estimated_runtime_min"] == 42.0


def test_estimate_refuses_a_finished_job_and_an_unknown_one(
    gpuc_home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    job_id = queue.enqueue(make_spec())
    jobs.update_state(job_id, status="succeeded")
    code, payload = run(capsys, "estimate", job_id, "10")
    assert code == 1 and isinstance(payload, dict) and "already succeeded" in payload["error"]
    assert jobs.read_spec(job_id).estimated_runtime_min is None

    code, payload = run(capsys, "estimate", "no-such-job", "10")
    assert code == 1 and isinstance(payload, dict) and "no job with that id" in payload["error"]


@pytest.mark.parametrize("minutes", ["0", "-5", "nan", "inf", "1e10"])
def test_estimate_refuses_a_number_that_is_not_a_runtime(
    gpuc_home: Path, capsys: pytest.CaptureFixture[str], minutes: str
) -> None:
    """`inf`, and the `1e10` units typo, mean "no estimate" by the time they
    reach `utc_in` -- so recording one would report success for a job whose
    status then shows nothing at all."""
    job_id = queue.enqueue(make_spec())
    code, payload = run(capsys, "estimate", job_id, minutes)
    assert code == 1 and isinstance(payload, dict) and payload["error"]
    assert jobs.read_spec(job_id).estimated_runtime_min is None


def test_estimate_needs_a_number_or_clear_and_not_both(
    gpuc_home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    job_id = queue.enqueue(make_spec(estimated_runtime_min=30.0))
    for args in ((job_id,), (job_id, "60", "--clear")):
        code, payload = run(capsys, "estimate", *args)
        assert code == 1 and isinstance(payload, dict) and "not both" in payload["error"]
    assert jobs.read_spec(job_id).estimated_runtime_min == 30.0


def test_estimate_warns_when_the_job_will_be_killed_first(
    gpuc_home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    job_id = queue.enqueue(make_spec(max_runtime_min=60.0))
    _, payload = run(capsys, "estimate", job_id, "120")
    assert isinstance(payload, dict) and "max_runtime_min" in (payload["warning"] or "")


def test_dispatch_is_routed_to_the_dispatcher(
    gpuc_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ran: list[bool] = []
    monkeypatch.setattr(dispatcher.Dispatcher, "run", lambda self, lock: ran.append(True) or 0)
    assert cli.main(["dispatch"]) == 0
    assert ran


def test_health_is_routed_to_health(
    gpuc_home: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from gpuc.host import health

    monkeypatch.setattr(health, "http_download", lambda url, n, t: n)
    monkeypatch.setattr(
        health.gpus,
        "run_nvidia_smi",
        __import__("tests.conftest", fromlist=["fake_smi"]).fake_smi(),
    )
    code = cli.main(["health", "--min-free-gb", "0", "--download-url", "http://x"])
    assert code == 0
    assert json.loads(capsys.readouterr().out)["ok"]


def test_status_resolves_the_owned_gpus(
    gpuc_home: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The control side cannot: `gpus` may name cards by index, and only the
    host knows today's numbering."""
    monkeypatch.setattr(cli.gpus, "list_gpus", lambda *_: [cli.gpus.Gpu(3, FAKE_GPUS[0])])
    monkeypatch.setattr(cli.gpus, "resolve_owned", lambda owned, *_: ([FAKE_GPUS[0]], ["9"]))
    jobs.write_config(HostConfig(host="test-host", gpus=["3", "9"]))

    _, status = run(capsys, "status")
    assert isinstance(status, dict)
    assert status["gpus"] == ["3", "9"]
    assert status["gpus_resolved"] == [{"index": 3, "uuid": FAKE_GPUS[0]}]
    assert status["gpus_unavailable"] == ["9"]


def test_status_reports_a_queued_jobs_estimate_from_its_spec(
    gpuc_home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A queued job has no eta yet, and its estimate is exactly what somebody
    deciding whether to queue behind it needs. Only the spec has it."""
    queue.enqueue(make_spec(estimated_runtime_min=360.0))
    queue.enqueue(make_spec())

    _, status = run(capsys, "status")
    assert isinstance(status, dict)
    estimates = {entry["job_id"]: entry["estimated_runtime_min"] for entry in status["jobs"]}
    assert sorted(estimates.values(), key=lambda v: v is None) == [360.0, None]


def test_status_reports_the_commit_the_host_was_bootstrapped_with(
    gpuc_home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The only honest answer to "what build is this host running". Whichever
    control machine bootstrapped last wrote it, which is the one thing the
    registry on any one of those machines cannot know."""
    jobs.write_config(HostConfig(host="test-host", pkg_commit="c" * 40))
    _, status = run(capsys, "status")
    assert isinstance(status, dict)
    assert status["pkg_commit"] == "c" * 40

    jobs.write_config(HostConfig(host="test-host"))
    _, status = run(capsys, "status")
    assert isinstance(status, dict)
    assert status["pkg_commit"] is None


def test_status_reports_the_commit_the_running_dispatcher_was_started_on(
    gpuc_home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Not the same question as `pkg_commit`, and the difference is the bug it
    exists for: a dispatcher imports its code once, so a host re-bootstrapped
    under a live one has the new package on disk and the old one dispatching."""
    jobs.write_config(HostConfig(host="test-host", pkg_commit="c" * 40))
    lock = dispatcher.DispatcherLock()
    assert lock.acquire()
    try:
        jobs.write_config(HostConfig(host="test-host", pkg_commit="d" * 40))
        _, status = run(capsys, "status")
    finally:
        lock.release()
    assert isinstance(status, dict)
    assert (status["pkg_commit"], status["dispatcher_pkg_commit"]) == ("d" * 40, "c" * 40)


def test_status_reports_each_jobs_priority_and_card_count_from_its_spec(
    gpuc_home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The queue marker only carries a priority while the job is queued, and a
    queued job holds no cards, so the spec is the only place either survives."""
    job_id = queue.enqueue(make_spec(priority=12, gpus=2))
    jobs.update_state(job_id, status="running", gpus=[FAKE_GPUS[0], FAKE_GPUS[1]])
    queue.remove_marker(job_id)

    _, status = run(capsys, "status")
    assert isinstance(status, dict)
    assert status["queue"] == []
    assert (status["jobs"][0]["priority"], status["jobs"][0]["gpus_requested"]) == (12, 2)


def test_reorder_records_the_new_priority_in_the_spec(
    gpuc_home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Otherwise the priority a job was dispatched at is lost with its marker,
    and `gpuc status` could only ever report a running job's as its original."""
    job_id = queue.enqueue(make_spec(priority=50))
    run(capsys, "reorder", job_id, "7")
    assert jobs.read_spec(job_id).priority == 7

    _, status = run(capsys, "status")
    assert isinstance(status, dict)
    assert status["jobs"][0]["priority"] == 7


def test_preempt_marks_a_running_job_and_starts_a_dispatcher(
    gpuc_home: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The dispatcher is what puts the job back: on a host whose dispatcher had
    died, the kill would land and nothing would ever queue the job again."""
    monkeypatch.setattr(dispatcher, "spawn_detached_dispatcher", lambda: 4242)
    job_id = queue.enqueue(make_spec(priority=50))
    queue.remove_marker(job_id)
    jobs.update_state(job_id, status="running")
    queue.enqueue(make_spec(priority=10))  # the job that wants the GPUs

    code, payload = run(capsys, "preempt", job_id, "--priority", "70")
    assert code == 0 and isinstance(payload, dict)
    assert (payload["status"], payload["priority"]) == ("preempting", 70)
    assert payload["dispatcher_pid"] == 4242
    assert queue.kill_reason(job_id) == "preempted"


def test_preempt_of_a_job_that_is_not_running_is_a_refusal_not_a_traceback(
    gpuc_home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    job_id = queue.enqueue(make_spec())
    code, payload = run(capsys, "preempt", job_id)
    assert code == 1 and isinstance(payload, dict)
    assert "not running" in str(payload["error"])
    assert not queue.is_preempted(job_id)

    code, payload = run(capsys, "preempt", "no-such-job")
    assert code == 1 and isinstance(payload, dict) and "no such job" in str(payload["error"])


def test_preempt_that_would_free_the_host_for_nothing_is_refused(
    gpuc_home: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """And without starting a dispatcher for it: nothing was asked of one."""
    started: list[int] = []
    monkeypatch.setattr(dispatcher, "spawn_detached_dispatcher", lambda: started.append(1) or 1)
    job_id = queue.enqueue(make_spec())
    queue.remove_marker(job_id)
    jobs.update_state(job_id, status="running")

    code, payload = run(capsys, "preempt", job_id)
    assert code == 1 and isinstance(payload, dict)
    assert "nothing else is queued" in str(payload["error"])
    assert (started, queue.kill_reason(job_id)) == ([], None)
