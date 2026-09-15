from __future__ import annotations

import json
from pathlib import Path

import pytest

from gpuc.host import __main__ as cli
from gpuc.host import dispatcher, jobs, paths, queue
from tests.conftest import make_spec


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
