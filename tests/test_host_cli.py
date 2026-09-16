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
    code, payload = run(capsys, "enqueue", str(spec_path), "--no-dispatch")
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
    code, payload = run(capsys, "enqueue", "-", "--no-dispatch")
    assert code == 0
    assert isinstance(payload, dict)
    assert jobs.read_state(payload["job_id"]).status == "queued"


def test_list_and_status(gpuc_home: Path, capsys: pytest.CaptureFixture[str]) -> None:
    job_id = queue.enqueue(make_spec(name="n", priority=12))
    _, listed = run(capsys, "list")
    assert listed == [{"priority": 12, "job_id": job_id, "name": "n"}]

    _, status = run(capsys, "status")
    assert isinstance(status, dict)
    assert status["host"] == "test-host"
    assert status["ephemeral"] is False
    assert status["paused"] is False
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
    assert code == 0 and isinstance(payload, dict) and payload["reordered"]
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


def test_resume_clears_the_pause(
    gpuc_home: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(dispatcher, "spawn_detached_dispatcher", lambda: 0)
    paths.paused_file().write_text("low-util\n")
    assert cli.main(["resume"]) == 0
    assert not paths.paused_file().exists()


def test_dispatch_once_is_routed_to_the_dispatcher(gpuc_home: Path) -> None:
    job_id = queue.enqueue(make_spec(gpus=0, command="true"))
    assert cli.main(["dispatch", "--once", "--interval", "0.1"]) == 0
    assert jobs.read_state(job_id).status in ("running", "succeeded")


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


def test_status_resolves_the_owned_gpus_and_reports_each_jobs_watchdog(
    gpuc_home: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The control side cannot do either: `gpus` may name cards by index, and it
    never sees the spec the low-util rule lives in."""
    monkeypatch.setattr(cli.gpus, "list_gpus", lambda *_: [cli.gpus.Gpu(3, FAKE_GPUS[0])])
    monkeypatch.setattr(cli.gpus, "resolve_owned", lambda owned, *_: ([FAKE_GPUS[0]], ["9"]))
    jobs.write_config(HostConfig(host="test-host", gpus=["3", "9"]))
    job_id = queue.enqueue(make_spec(low_util={"enabled": False, "floor_pct": 20.0}))

    _, status = run(capsys, "status")
    assert isinstance(status, dict)
    assert status["gpus"] == ["3", "9"]
    assert status["gpus_resolved"] == [{"index": 3, "uuid": FAKE_GPUS[0]}]
    assert status["gpus_unavailable"] == ["9"]
    low_util = status["jobs"][0]["low_util"]
    assert (job_id, low_util["enabled"], low_util["floor_pct"]) == (job_id, False, 20.0)


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
