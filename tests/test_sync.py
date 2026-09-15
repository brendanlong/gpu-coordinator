from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from gpuc.host import jobs, paths, sync
from gpuc.host.jobs import Output
from tests.conftest import make_spec


class RecordingRunner:
    def __init__(self, returncode: int = 0, output: str = "") -> None:
        self.calls: list[list[str]] = []
        self.returncode = returncode
        self.output = output

    def __call__(self, argv: list[str], timeout: float) -> sync.CommandResult:
        self.calls.append(argv)
        return sync.CommandResult(argv, self.returncode, self.output)


@pytest.fixture
def fake_aws(monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setattr(sync, "aws_binary", lambda: "/fake/aws")
    monkeypatch.setattr(sync, "hf_binary", lambda: "/fake/hf")
    return "/fake/aws"


def test_recently_modified_finds_only_fresh_files(tmp_path: Path) -> None:
    old = tmp_path / "old.bin"
    new = tmp_path / "nested" / "new.bin"
    new.parent.mkdir()
    old.write_text("x")
    new.write_text("y")
    stale = time.time() - 3600
    os.utime(old, (stale, stale))
    assert sync.recently_modified(tmp_path, min_age_s=10.0) == ["nested/new.bin"]
    assert sync.recently_modified(tmp_path, min_age_s=0.0) == []


def test_s3_sync_excludes_files_written_in_the_last_ten_seconds(
    tmp_path: Path, fake_aws: str
) -> None:
    (tmp_path / "ckpt.pt").write_text("half-written")
    runner = RecordingRunner()
    sync.sync_dir_to_s3(tmp_path, "s3://bucket/prefix/", runner=runner)
    argv = runner.calls[0]
    assert argv[:4] == ["/fake/aws", "s3", "sync", str(tmp_path)]
    assert argv[4] == "s3://bucket/prefix"
    assert argv[argv.index("--exclude") + 1] == "ckpt.pt"


def test_final_sync_includes_the_files_just_written(tmp_path: Path, fake_aws: str) -> None:
    (tmp_path / "ckpt.pt").write_text("done")
    runner = RecordingRunner()
    sync.sync_dir_to_s3(tmp_path, "s3://bucket/prefix", min_age_s=0.0, runner=runner)
    assert "--exclude" not in runner.calls[0]


def test_sync_error_carries_command_and_output(tmp_path: Path, fake_aws: str) -> None:
    runner = RecordingRunner(returncode=1, output="line1\nAccess Denied\n")
    with pytest.raises(sync.SyncError) as excinfo:
        sync.sync_dir_to_s3(tmp_path, "s3://bucket/p", runner=runner)
    message = str(excinfo.value)
    assert "/fake/aws s3 sync" in message
    assert "Access Denied" in message
    assert "exited 1" in message


def test_missing_aws_binary_is_a_clear_sync_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sync, "aws_binary", lambda: None)
    with pytest.raises(sync.SyncError) as excinfo:
        sync.sync_dir_to_s3(tmp_path, "s3://bucket/p", runner=RecordingRunner())
    assert "`aws` CLI not found" in str(excinfo.value)


def test_missing_hf_binary_is_a_clear_sync_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sync, "hf_binary", lambda: None)
    with pytest.raises(sync.SyncError) as excinfo:
        sync.upload_dir_to_hf(tmp_path, "org/repo", "p", runner=RecordingRunner())
    assert "`hf` CLI not found" in str(excinfo.value)


def test_job_id_is_expanded_in_destinations(tmp_path: Path, fake_aws: str) -> None:
    (tmp_path / "results").mkdir()
    runner = RecordingRunner()
    output = Output(path="results", s3="s3://b/exp/{job_id}/results")
    sync.sync_output(output, tmp_path, "20260101-000000-abcdef", runner=runner)
    assert runner.calls[0][4] == "s3://b/exp/20260101-000000-abcdef/results"


def test_hf_upload_argv(tmp_path: Path, fake_aws: str) -> None:
    (tmp_path / "ckpt").mkdir()
    runner = RecordingRunner()
    output = Output(path="ckpt", hf="org/repo", hf_path="{job_id}/ckpt")
    sync.sync_output(output, tmp_path, "jid", min_age_s=0.0, runner=runner)
    assert runner.calls[0] == ["/fake/hf", "upload", "org/repo", str(tmp_path / "ckpt"), "jid/ckpt"]


def test_sync_outputs_reports_every_failure(tmp_path: Path, fake_aws: str) -> None:
    (tmp_path / "a").mkdir()
    runner = RecordingRunner(returncode=2, output="boom")
    outputs = [Output(path="a", s3="s3://b/a"), Output(path="missing", s3="s3://b/m")]
    with pytest.raises(sync.SyncError) as excinfo:
        sync.sync_outputs(outputs, tmp_path, "jid", runner=runner)
    assert "boom" in str(excinfo.value)
    assert "does not exist" in str(excinfo.value)


def test_sync_job_meta_uploads_log_and_state(gpuc_home: Path, fake_aws: str) -> None:
    from gpuc.host import queue as q

    job_id = q.enqueue(make_spec())
    paths.log_file(job_id).write_text("hello\n")
    runner = RecordingRunner()
    sync.sync_job_meta(job_id, "s3://bucket/gpuc/host/", runner=runner)
    destinations = [call[-2] for call in runner.calls]
    assert destinations == [
        f"s3://bucket/gpuc/host/jobs/{job_id}/log.txt",
        f"s3://bucket/gpuc/host/jobs/{job_id}/state.json",
    ]
    sync.sync_job_meta(job_id, None, runner=runner)
    assert len(runner.calls) == 2


def test_sync_loop_final_runs_one_pass_and_stops(gpuc_home: Path, fake_aws: str) -> None:
    job_id = jobs.new_job_id()
    paths.ensure_job_layout(job_id)
    spec = make_spec(
        job_id=job_id, sync_interval_s=1, outputs=[{"path": "outputs", "s3": "s3://b/o"}]
    )
    jobs.write_spec(spec)
    jobs.write_state(job_id, jobs.JobState())
    (paths.workdir(job_id) / "outputs").mkdir()
    runner = RecordingRunner()
    loop = sync.SyncLoop(spec, paths.workdir(job_id), None, runner=runner)
    loop.start()
    loop.final()
    assert len(runner.calls) >= 1
    assert loop._thread is None
