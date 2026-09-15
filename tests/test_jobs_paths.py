from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import pytest

from gpuc.host import jobs, paths
from gpuc.host.jobs import HostConfig, JobSpec, JobState


def test_gpuc_home_honours_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GPUC_HOME", str(tmp_path / "elsewhere"))
    assert paths.home() == tmp_path / "elsewhere"
    assert paths.state_file("j") == tmp_path / "elsewhere/jobs/j/state.json"
    monkeypatch.delenv("GPUC_HOME")
    assert paths.home() == Path.home() / ".gpuc"


def test_job_ids_are_unique_and_sortable() -> None:
    ids = {jobs.new_job_id() for _ in range(200)}
    assert len(ids) == 200
    assert all(len(i) == len("YYYYMMDD-HHMMSS-abcdef") for i in ids)


def test_atomic_write_leaves_no_temp_files(tmp_path: Path) -> None:
    target = tmp_path / "state.json"
    jobs.atomic_write_json(target, {"a": 1})
    assert json.loads(target.read_text()) == {"a": 1}
    assert [p.name for p in tmp_path.iterdir()] == ["state.json"]


def test_read_json_tolerates_a_concurrent_writer(tmp_path: Path) -> None:
    target = tmp_path / "state.json"
    jobs.atomic_write_json(target, {"n": 0})
    stop = threading.Event()

    def writer() -> None:
        n = 0
        while not stop.is_set():
            n += 1
            jobs.atomic_write_json(target, {"n": n})

    thread = threading.Thread(target=writer)
    thread.start()
    try:
        for _ in range(500):
            assert "n" in jobs.read_json(target)
    finally:
        stop.set()
        thread.join()


def test_spec_round_trip_applies_defaults(gpuc_home: Path) -> None:
    spec = JobSpec.from_dict(
        {
            "job_id": "j1",
            "command": "echo hi",
            "outputs": [{"path": "results", "s3": "s3://b/{job_id}/results"}],
        }
    )
    jobs.write_spec(spec)
    loaded = jobs.read_spec("j1")
    assert loaded.gpus == 1
    assert loaded.priority == 50
    assert loaded.low_util.window_min == 25.0
    assert loaded.outputs[0].s3 == "s3://b/{job_id}/results"
    assert loaded == spec


def test_update_state_rejects_unknown_fields(gpuc_home: Path) -> None:
    jobs.write_state("j1", JobState())
    jobs.update_state("j1", status="running")
    assert jobs.read_state("j1").status == "running"
    with pytest.raises(KeyError):
        jobs.update_state("j1", nonsense=1)


def test_state_file_survives_partial_unknown_keys(gpuc_home: Path) -> None:
    jobs.write_state("j1", JobState(status="running"))
    raw = json.loads(paths.state_file("j1").read_text())
    raw["future_field"] = 7
    paths.state_file("j1").write_text(json.dumps(raw))
    assert jobs.read_state("j1").status == "running"


def test_config_defaults_when_missing(gpuc_home: Path) -> None:
    os.remove(paths.config_file())
    config = jobs.read_config()
    assert config.gpus == []
    assert not config.ephemeral
    jobs.write_config(HostConfig(host="pod", provider={"kind": "runpod", "pod_id": "p"}))
    assert jobs.read_config().ephemeral


def test_the_layout_is_private_to_the_owner(gpuc_home: Path) -> None:
    import stat

    paths.ensure_layout()
    assert stat.S_IMODE(paths.home().stat().st_mode) == 0o700
    assert stat.S_IMODE(paths.secrets_dir().stat().st_mode) == 0o700


def test_state_round_trips_the_new_identity_and_sync_fields(gpuc_home: Path) -> None:
    state = jobs.JobState(
        status="running",
        runner_pid=7,
        runner_boot_id="boot",
        runner_starttime="123",
        sync_error="s3 said no",
        util_recent=[1.0, None],
    )
    jobs.write_state("j1", state)
    assert jobs.read_state("j1") == state
    assert jobs.PHASES == ("setup", "preflight", "main", "sync")
