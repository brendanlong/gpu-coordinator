from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import pytest

from gpuc.host import jobs, paths
from gpuc.host.jobs import HostConfig, JobSpec, JobState
from tests.conftest import accept_job


def test_gpuc_home_honours_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GPUC_HOME", str(tmp_path / "elsewhere"))
    assert paths.home() == tmp_path / "elsewhere"
    assert paths.state_file("j") == tmp_path / "elsewhere/jobs/j/state.json"
    assert paths.job_lock_file("j") == tmp_path / "elsewhere/jobs/j/.lock"
    assert paths.incoming_job_dir("j") == tmp_path / "elsewhere/incoming/j"
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
    accept_job(spec)
    loaded = jobs.read_spec("j1")
    assert loaded.gpus == 1
    assert loaded.priority == 50
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
    assert config.gpus is None
    assert not config.ephemeral
    jobs.write_config(HostConfig(host="pod", provider={"kind": "runpod", "pod_id": "p"}))
    assert jobs.read_config().ephemeral


def test_the_layout_is_private_to_the_owner(gpuc_home: Path) -> None:
    import stat

    paths.ensure_layout()
    assert stat.S_IMODE(paths.home().stat().st_mode) == 0o700
    assert stat.S_IMODE(paths.secrets_dir().stat().st_mode) == 0o700


def test_state_round_trips_the_new_identity_and_upload_fields(gpuc_home: Path) -> None:
    state = jobs.JobState(
        status="running",
        runner_pid=7,
        runner_boot_id="boot",
        runner_starttime="123",
        uploads=[
            jobs.Upload(to="s3://b/gpuc/h/jobs/j1", ok_at="2026-01-01T00:00:00+00:00"),
            jobs.Upload(to="s3://bucket/j1", output="results", error="s3 said no"),
        ],
        problems=["sync"],
        util_recent=[1.0, None],
    )
    jobs.write_state("j1", state)
    assert jobs.read_state("j1") == state


def test_transition_writes_nothing_when_the_status_is_not_what_was_expected(
    gpuc_home: Path,
) -> None:
    """The compare-and-set every change of a job's ownership goes through: the
    loser of a race learns it from the None, not from a job that is both
    running and cancelled."""
    jobs.write_state("j1", JobState())

    claimed = jobs.transition("j1", expect="queued", status="running", gpus=["GPU-x"])
    assert claimed is not None and claimed.status == "running"

    again = jobs.transition("j1", expect="queued", status="cancelled", gpus=["GPU-y"])
    assert again is None
    state = jobs.read_state("j1")
    assert (state.status, state.gpus) == ("running", ["GPU-x"])


def test_transition_takes_any_of_the_statuses_it_was_given(gpuc_home: Path) -> None:
    """`estimate` is the caller with two: it may reach a job that is still
    queued or one already running, and neither is a race it lost."""
    jobs.write_state("j1", JobState(status="running"))
    for status in ("queued", "running"):
        jobs.update_state("j1", status=status)
        assert jobs.transition("j1", expect=("queued", "running"), estimated_runtime_min=15.0)

    jobs.update_state("j1", status="succeeded")
    assert jobs.transition("j1", expect=("queued", "running"), estimated_runtime_min=1.0) is None
    assert jobs.read_state("j1").estimated_runtime_min == 15.0


def test_transition_rejects_unknown_fields(gpuc_home: Path) -> None:
    jobs.write_state("j1", JobState())
    with pytest.raises(KeyError):
        jobs.transition("j1", expect="queued", nonsense=1)


def test_state_carries_the_intent_and_the_live_priority(gpuc_home: Path) -> None:
    """Both are what somebody changed after the job was submitted, and the spec
    is never rewritten, so the state is the only copy of either."""
    state = JobState(status="running", intent=jobs.PREEMPT, priority=7, estimated_runtime_min=30.0)
    jobs.write_state("j1", state)
    assert jobs.read_state("j1") == state


def test_an_intent_this_build_does_not_understand_is_no_intent(gpuc_home: Path) -> None:
    """A word only a newer build acts on must not be read as one of ours: the
    intents are what a runner kills a job for."""
    jobs.write_state("j1", JobState(status="running"))
    raw = json.loads(paths.state_file("j1").read_text())
    raw["intent"] = "hibernate"
    paths.state_file("j1").write_text(json.dumps(raw))
    assert jobs.read_state("j1").intent is None
