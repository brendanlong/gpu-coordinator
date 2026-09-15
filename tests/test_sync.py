from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

import pytest

from gpuc.host import jobs, paths, sync
from gpuc.host.jobs import Output
from tests.conftest import make_spec


class RecordingRunner:
    def __init__(self, returncode: int = 0, output: str = "") -> None:
        self.calls: list[list[str]] = []
        self.timeouts: list[float | None] = []
        self.envs: list[sync.Env] = []
        self.returncode = returncode
        self.output = output

    def __call__(
        self, argv: list[str], timeout: float | None = None, env: sync.Env = None
    ) -> sync.CommandResult:
        self.calls.append(argv)
        self.timeouts.append(timeout)
        self.envs.append(env)
        return sync.CommandResult(argv, self.returncode, self.output)


@pytest.fixture
def fake_aws(monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setattr(sync, "aws_binary", lambda env=None: "/fake/aws")
    monkeypatch.setattr(sync, "hf_binary", lambda env=None: "/fake/hf")
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
    monkeypatch.setattr(sync, "aws_binary", lambda env=None: None)
    with pytest.raises(sync.SyncError) as excinfo:
        sync.sync_dir_to_s3(tmp_path, "s3://bucket/p", runner=RecordingRunner())
    assert "`aws` CLI not found" in str(excinfo.value)


def test_missing_hf_binary_is_a_clear_sync_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sync, "hf_binary", lambda env=None: None)
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


def loop_for(job_id: str, runner: sync.CommandRunner, **overrides: object) -> sync.SyncLoop:
    document: dict[str, object] = {
        "job_id": job_id,
        "sync_interval_s": 1,
        "outputs": [{"path": "outputs", "s3": "s3://b/o"}],
    }
    document.update(overrides)
    spec = make_spec(**document)
    paths.ensure_job_layout(job_id)
    jobs.write_spec(spec)
    jobs.write_state(job_id, jobs.JobState())
    paths.log_file(job_id).touch()
    return sync.SyncLoop(spec, paths.workdir(job_id), None, runner=runner)


def test_run_command_turns_a_missing_binary_into_a_sync_error() -> None:
    with pytest.raises(sync.SyncError, match="not found"):
        sync.run_command(["definitely-not-a-binary-xyz"])


def test_run_command_turns_a_timeout_into_a_sync_error() -> None:
    with pytest.raises(sync.SyncError, match="timed out"):
        sync.run_command(["sleep", "30"], timeout=0.2)


def test_the_exclude_list_is_capped_instead_of_building_a_huge_argv(
    tmp_path: Path, fake_aws: str
) -> None:
    for index in range(sync.MAX_EXCLUDES + 1):
        (tmp_path / f"shard-{index:04d}.bin").write_text("x")
    runner = RecordingRunner()
    with pytest.raises(sync.TooManyRecentFiles, match="skipping this sync tick"):
        sync.sync_dir_to_s3(tmp_path, "s3://b/p", runner=runner)
    assert runner.calls == []


def test_a_capped_tick_is_skipped_with_a_log_line_and_is_not_an_error(
    gpuc_home: Path, fake_aws: str
) -> None:
    job_id = jobs.new_job_id()
    runner = RecordingRunner()
    loop = loop_for(job_id, runner)
    outputs = paths.workdir(job_id) / "outputs"
    outputs.mkdir(parents=True)
    for index in range(sync.MAX_EXCLUDES + 1):
        (outputs / f"shard-{index:04d}.bin").write_text("x")
    loop.start()
    _wait_for(lambda: "skipping this sync tick" in paths.log_file(job_id).read_text())
    loop.stop()
    assert loop.last_error is None
    assert jobs.read_state(job_id).sync_error is None


def test_a_missing_output_dir_is_warned_about_on_a_periodic_tick(
    gpuc_home: Path, fake_aws: str
) -> None:
    job_id = jobs.new_job_id()
    loop = loop_for(job_id, RecordingRunner())
    loop.start()
    _wait_for(lambda: "does not exist" in paths.log_file(job_id).read_text())
    loop.stop()
    assert "WARNING" in paths.log_file(job_id).read_text()
    assert loop.last_error is None


def test_a_missing_output_dir_on_the_final_sync_is_its_own_error(
    gpuc_home: Path, fake_aws: str
) -> None:
    job_id = jobs.new_job_id()
    loop = loop_for(job_id, RecordingRunner())
    with pytest.raises(sync.MissingOutput):
        loop.final()


def test_the_periodic_thread_survives_any_exception_and_records_it(
    gpuc_home: Path, fake_aws: str
) -> None:
    def exploding(
        argv: list[str], timeout: float | None = None, env: sync.Env = None
    ) -> sync.CommandResult:
        raise KeyboardInterrupt("something unspeakable")

    job_id = jobs.new_job_id()
    loop = loop_for(job_id, exploding)
    (paths.workdir(job_id) / "outputs").mkdir(parents=True)
    loop.start()
    _wait_for(lambda: jobs.read_state(job_id).sync_error is not None)
    assert loop._thread is not None and loop._thread.is_alive()
    loop.stop()
    assert "something unspeakable" in (jobs.read_state(job_id).sync_error or "")


def test_final_never_overlaps_a_periodic_tick(gpuc_home: Path, fake_aws: str) -> None:
    overlaps: list[str] = []
    inside = threading.Lock()

    def slow(
        argv: list[str], timeout: float | None = None, env: sync.Env = None
    ) -> sync.CommandResult:
        if not inside.acquire(blocking=False):
            overlaps.append(" ".join(argv))
        else:
            time.sleep(0.2)
            inside.release()
        return sync.CommandResult(argv, 0, "")

    job_id = jobs.new_job_id()
    loop = loop_for(job_id, slow)
    (paths.workdir(job_id) / "outputs").mkdir(parents=True)
    loop.start()
    time.sleep(1.1)
    loop.final()
    assert overlaps == []


def test_the_final_sync_has_no_wall_clock_timeout(gpuc_home: Path, fake_aws: str) -> None:
    job_id = jobs.new_job_id()
    runner = RecordingRunner()
    loop = loop_for(job_id, runner)
    (paths.workdir(job_id) / "outputs").mkdir(parents=True)
    loop.final()
    assert runner.timeouts == [None]


def _wait_for(predicate, timeout: float = 20.0) -> None:  # type: ignore[no-untyped-def]
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(0.05)
    raise AssertionError("condition never became true")


# -- recording the backup -----------------------------------------------------


class FlakyRunner(RecordingRunner):
    """Fails the Nth command onwards, so a partial mirror can be tested."""

    def __init__(self, fail_from: int) -> None:
        super().__init__()
        self.fail_from = fail_from

    def __call__(
        self, argv: list[str], timeout: float | None = None, env: sync.Env = None
    ) -> sync.CommandResult:
        result = super().__call__(argv, timeout, env)
        if len(self.calls) >= self.fail_from:
            return sync.CommandResult(argv, 1, "AccessDenied")
        return result


def enqueued_job() -> str:
    from gpuc.host import queue as q

    job_id = q.enqueue(make_spec())
    paths.log_file(job_id).write_text("hello\n")
    return job_id


def test_final_meta_sync_records_when_and_where(gpuc_home: Path, fake_aws: str) -> None:
    job_id = enqueued_job()
    runner = RecordingRunner()
    assert sync.final_meta_sync(job_id, "s3://bucket/gpuc/host/", runner=runner) is None
    state = jobs.read_state(job_id)
    assert state.meta_synced_at and state.meta_synced_to == "s3://bucket/gpuc/host"
    # log, state, then state again now that it records the backup.
    assert [call[-2] for call in runner.calls] == [
        f"s3://bucket/gpuc/host/jobs/{job_id}/log.txt",
        f"s3://bucket/gpuc/host/jobs/{job_id}/state.json",
        f"s3://bucket/gpuc/host/jobs/{job_id}/state.json",
    ]
    mirrored = json.loads(paths.state_file(job_id).read_text())
    assert mirrored["meta_synced_at"] == state.meta_synced_at


def test_a_failed_meta_sync_records_nothing(gpuc_home: Path, fake_aws: str) -> None:
    job_id = enqueued_job()
    with pytest.raises(sync.SyncError):
        sync.final_meta_sync(job_id, "s3://bucket/x", runner=RecordingRunner(returncode=1))
    state = jobs.read_state(job_id)
    assert (state.meta_synced_at, state.meta_synced_to) == (None, None)


def test_no_prefix_records_nothing_and_uploads_nothing(gpuc_home: Path, fake_aws: str) -> None:
    job_id = enqueued_job()
    runner = RecordingRunner()
    assert sync.final_meta_sync(job_id, None, runner=runner) is None
    assert runner.calls == []
    assert jobs.read_state(job_id).meta_synced_at is None


def test_only_the_trailing_state_put_failing_keeps_the_record(
    gpuc_home: Path, fake_aws: str
) -> None:
    job_id = enqueued_job()
    warning = sync.final_meta_sync(job_id, "s3://bucket/x", runner=FlakyRunner(fail_from=3))
    assert warning and "one revision behind" in warning
    # The local state is the authority: log and state did reach S3.
    assert jobs.read_state(job_id).meta_synced_at is not None


def test_final_clears_outputs_synced_at_before_trying(gpuc_home: Path, fake_aws: str) -> None:
    job_id = jobs.new_job_id()
    paths.ensure_job_layout(job_id)
    spec = make_spec(job_id=job_id, outputs=[{"path": "outputs", "s3": "s3://b/o"}])
    loop = sync.SyncLoop(spec, paths.workdir(job_id), None, runner=RecordingRunner())
    with pytest.raises(sync.MissingOutput):
        loop.final()  # the output path was never written
    assert loop.outputs_synced_at is None
    (paths.workdir(job_id) / "outputs").mkdir(parents=True)
    loop.final()
    assert loop.outputs_synced_at is not None
    no_outputs = sync.SyncLoop(make_spec(), paths.workdir(job_id), None)
    no_outputs.final()
    assert no_outputs.outputs_synced_at is None  # nothing was declared
