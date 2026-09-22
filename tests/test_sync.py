from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

import pytest

from gpuc.host import destinations, jobs, paths, sync
from gpuc.host.destinations import S3, HuggingFace
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
    monkeypatch.setattr(destinations, "find_binary", lambda name, env=None: f"/fake/{name}")
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
    S3("s3://bucket/prefix/").upload_dir(
        tmp_path, exclude=sync.recently_modified(tmp_path), runner=runner
    )
    argv = runner.calls[0]
    assert argv[:4] == ["/fake/aws", "s3", "sync", str(tmp_path)]
    assert argv[4] == "s3://bucket/prefix"
    assert argv[argv.index("--exclude") + 1] == "ckpt.pt"


def test_final_sync_includes_the_files_just_written(gpuc_home: Path, fake_aws: str) -> None:
    job_id = enqueued_job()
    workdir = paths.workdir(job_id)
    (workdir / "out").mkdir(parents=True)
    (workdir / "out" / "ckpt.pt").write_text("done")
    runner = RecordingRunner()
    sync.sync_output(
        Output(path="out", s3="s3://bucket/prefix"), workdir, job_id, min_age_s=0.0, runner=runner
    )
    assert "--exclude" not in runner.calls[0]


def test_a_periodic_sync_output_excludes_files_still_being_written(
    gpuc_home: Path, fake_aws: str
) -> None:
    job_id = enqueued_job()
    workdir = paths.workdir(job_id)
    (workdir / "out").mkdir(parents=True)
    (workdir / "out" / "ckpt.pt").write_text("half-written")
    runner = RecordingRunner()
    sync.sync_output(Output(path="out", s3="s3://b/o"), workdir, job_id, runner=runner)
    argv = runner.calls[0]
    assert argv[argv.index("--exclude") + 1] == "ckpt.pt"


def test_sync_error_carries_command_and_output(tmp_path: Path, fake_aws: str) -> None:
    runner = RecordingRunner(returncode=1, output="line1\nAccess Denied\n")
    with pytest.raises(sync.SyncError) as excinfo:
        S3("s3://bucket/p").upload_dir(tmp_path, runner=runner)
    message = str(excinfo.value)
    assert "/fake/aws s3 sync" in message
    assert "Access Denied" in message
    assert "exited 1" in message


def test_missing_aws_binary_is_a_clear_sync_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(destinations, "find_binary", lambda name, env=None: None)
    with pytest.raises(sync.SyncError) as excinfo:
        S3("s3://bucket/p").upload_dir(tmp_path, runner=RecordingRunner())
    assert "`aws` CLI not found" in str(excinfo.value)


def test_missing_hf_binary_is_a_clear_sync_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(destinations, "find_binary", lambda name, env=None: None)
    with pytest.raises(sync.SyncError) as excinfo:
        HuggingFace("org/repo", "p").upload_dir(tmp_path, runner=RecordingRunner())
    assert "`hf` CLI not found" in str(excinfo.value)


def test_job_id_is_expanded_in_destinations(gpuc_home: Path, fake_aws: str) -> None:
    job_id = enqueued_job()
    workdir = paths.workdir(job_id)
    (workdir / "results").mkdir(parents=True)
    runner = RecordingRunner()
    output = Output(path="results", s3="s3://b/exp/{job_id}/results")
    sync.sync_output(output, workdir, job_id, runner=runner)
    assert runner.calls[0][4] == f"s3://b/exp/{job_id}/results"


def test_hf_upload_argv(gpuc_home: Path, fake_aws: str) -> None:
    job_id = enqueued_job()
    workdir = paths.workdir(job_id)
    (workdir / "ckpt").mkdir(parents=True)
    runner = RecordingRunner()
    output = Output(path="ckpt", hf="org/repo", hf_path="{job_id}/ckpt")
    sync.sync_output(output, workdir, job_id, min_age_s=0.0, runner=runner)
    assert runner.calls[0] == [
        "/fake/hf",
        "upload",
        "org/repo",
        str(workdir / "ckpt"),
        f"{job_id}/ckpt",
    ]


def test_sync_outputs_reports_every_failure(gpuc_home: Path, fake_aws: str) -> None:
    job_id = enqueued_job()
    workdir = paths.workdir(job_id)
    (workdir / "a").mkdir(parents=True)
    runner = RecordingRunner(returncode=2, output="boom")
    outputs = [Output(path="a", s3="s3://b/a"), Output(path="missing", s3="s3://b/m")]
    with pytest.raises(sync.SyncError) as excinfo:
        sync.sync_outputs(outputs, workdir, job_id, runner=runner)
    assert "boom" in str(excinfo.value)
    assert "does not exist" in str(excinfo.value)


# -- destinations -------------------------------------------------------------


def test_destinations_of_expands_job_id_in_every_field() -> None:
    output = Output(
        path="ckpt",
        s3="s3://b/exp/{job_id}/ckpt/",
        hf="org/{job_id}",
        hf_path="runs/{job_id}",
        hf_create=True,
    )
    found = destinations.of(output, "jid")
    assert found == [
        S3("s3://b/exp/jid/ckpt"),
        HuggingFace("org/jid", "runs/jid", create=True),
    ]
    assert [d.uri for d in found] == ["s3://b/exp/jid/ckpt", "hf://org/jid/runs/jid"]


def test_the_hf_path_defaults_to_the_job_id() -> None:
    (hf,) = destinations.of(Output(path="ckpt", hf="org/repo"), "jid")
    assert isinstance(hf, HuggingFace)
    assert (hf.repo, hf.path, hf.uri) == ("org/repo", "jid", "hf://org/repo/jid")


def test_an_output_with_no_destination_expands_to_nothing() -> None:
    assert destinations.of(Output(path="ckpt"), "jid") == []


def test_s3_put_file_puts_one_object_under_the_uri(tmp_path: Path, fake_aws: str) -> None:
    (tmp_path / "log.txt").write_text("x")
    runner = RecordingRunner()
    local = tmp_path / "log.txt"
    S3("s3://b/p/").put_file(local, "log.txt", runner=runner)
    assert runner.calls == [
        ["/fake/aws", "s3", "cp", str(local), "s3://b/p/log.txt", "--only-show-errors"]
    ]


def test_hf_put_file_lands_under_the_path(tmp_path: Path, fake_aws: str) -> None:
    (tmp_path / "f").write_text("x")
    runner = RecordingRunner()
    HuggingFace("org/repo", "/runs/jid/").put_file(tmp_path / "f", "f", runner=runner)
    assert runner.calls == [["/fake/hf", "upload", "org/repo", str(tmp_path / "f"), "runs/jid/f"]]


def test_an_upload_of_a_missing_path_is_missing_output_and_runs_nothing(
    tmp_path: Path, fake_aws: str
) -> None:
    runner = RecordingRunner()
    with pytest.raises(sync.MissingOutput):
        S3("s3://b/p").upload_dir(tmp_path / "nope", runner=runner)
    assert runner.calls == []


# -- upload records -----------------------------------------------------------


CKPT_BOTH = {"path": "ckpt", "s3": "s3://b/{job_id}", "hf": "org/repo"}


class FailingAt(RecordingRunner):
    """Fails every command that mentions `target`, succeeds elsewhere."""

    def __init__(self, target: str) -> None:
        super().__init__()
        self.target = target

    def __call__(
        self, argv: list[str], timeout: float | None = None, env: sync.Env = None
    ) -> sync.CommandResult:
        super().__call__(argv, timeout, env)
        if any(self.target in word for word in argv):
            return sync.CommandResult(argv, 1, "AccessDenied")
        return sync.CommandResult(argv, 0, "")


def test_each_destination_of_an_output_gets_its_own_record(gpuc_home: Path, fake_aws: str) -> None:
    job_id = enqueued_job()
    workdir = paths.workdir(job_id)
    (workdir / "ckpt").mkdir(parents=True)
    output = Output(path="ckpt", s3="s3://b/{job_id}", hf="org/repo")
    sync.sync_output(output, workdir, job_id, runner=RecordingRunner())
    records = jobs.read_state(job_id).output_uploads()
    assert sorted((u.output, u.to) for u in records) == [
        ("ckpt", f"hf://org/repo/{job_id}"),
        ("ckpt", f"s3://b/{job_id}"),
    ]
    assert all(u.ok_at and u.error is None for u in records)
    spec = make_spec(job_id=job_id, outputs=[CKPT_BOTH])
    assert jobs.read_state(job_id).outputs_uploaded(spec)


def test_a_failure_at_one_destination_is_recorded_there_and_success_elsewhere(
    gpuc_home: Path, fake_aws: str
) -> None:
    job_id = enqueued_job()
    workdir = paths.workdir(job_id)
    (workdir / "ckpt").mkdir(parents=True)
    output = Output(path="ckpt", s3="s3://b/{job_id}", hf="org/repo")
    with pytest.raises(sync.SyncError, match="AccessDenied"):
        sync.sync_output(output, workdir, job_id, runner=FailingAt("org/repo"))
    state = jobs.read_state(job_id)
    by_uri = {u.to: u for u in state.output_uploads()}
    s3, hf = by_uri[f"s3://b/{job_id}"], by_uri[f"hf://org/repo/{job_id}"]
    assert s3.ok_at and s3.error is None
    assert hf.ok_at is None and hf.error and "AccessDenied" in hf.error
    assert state.upload_errors() == [hf.error]
    assert not state.outputs_uploaded(make_spec(job_id=job_id, outputs=[CKPT_BOTH]))


def test_a_later_success_clears_the_error_and_a_later_failure_keeps_ok_at(
    gpuc_home: Path, fake_aws: str
) -> None:
    job_id = enqueued_job()
    workdir = paths.workdir(job_id)
    (workdir / "ckpt").mkdir(parents=True)
    output = Output(path="ckpt", s3="s3://b/o")
    sync.sync_output(output, workdir, job_id, runner=RecordingRunner())
    (first,) = jobs.read_state(job_id).output_uploads()
    with pytest.raises(sync.SyncError):
        sync.sync_output(output, workdir, job_id, runner=RecordingRunner(returncode=1))
    (failed,) = jobs.read_state(job_id).output_uploads()
    assert failed.ok_at == first.ok_at and failed.error
    sync.sync_output(output, workdir, job_id, runner=RecordingRunner())
    (healed,) = jobs.read_state(job_id).output_uploads()
    assert healed.ok_at and healed.error is None


def test_sync_job_meta_uploads_log_and_state(gpuc_home: Path, fake_aws: str) -> None:
    from gpuc.host import queue as q

    job_id = q.enqueue(make_spec())
    paths.log_file(job_id).write_text("hello\n")
    runner = RecordingRunner()
    sync.sync_job_meta(job_id, "s3://bucket/gpuc/host/", runner=runner)
    uploaded_to = [call[-2] for call in runner.calls]
    assert uploaded_to == [
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
    gpuc_home: Path, fake_aws: str
) -> None:
    job_id = enqueued_job()
    outputs = paths.workdir(job_id) / "outputs"
    outputs.mkdir(parents=True)
    for index in range(sync.MAX_EXCLUDES + 1):
        (outputs / f"shard-{index:04d}.bin").write_text("x")
    runner = RecordingRunner()
    output = Output(path="outputs", s3="s3://b/p")
    with pytest.raises(sync.TooManyRecentFiles, match="skipping this sync tick"):
        sync.sync_output(output, paths.workdir(job_id), job_id, runner=runner)
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
    assert jobs.read_state(job_id).upload_errors() == []


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


def test_a_periodic_missing_output_is_the_destinations_error_and_the_loop_goes_on(
    gpuc_home: Path, fake_aws: str
) -> None:
    job_id = jobs.new_job_id()
    runner = RecordingRunner()
    loop = loop_for(job_id, runner)
    loop.start()
    try:
        _wait_for(lambda: jobs.read_state(job_id).upload_errors() != [])
        (record,) = jobs.read_state(job_id).output_uploads()
        assert (record.to, record.output) == ("s3://b/o", "outputs")
        assert record.error and "does not exist" in record.error
        assert loop._thread is not None and loop._thread.is_alive()
        # The next tick, with the path written, uploads it and clears the error.
        (paths.workdir(job_id) / "outputs").mkdir(parents=True)
        _wait_for(lambda: jobs.read_state(job_id).upload_errors() == [])
    finally:
        loop.stop()
    (record,) = jobs.read_state(job_id).output_uploads()
    assert record.ok_at is not None


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
    _wait_for(lambda: loop.last_error is not None)
    assert loop._thread is not None and loop._thread.is_alive()
    loop.stop()
    assert "something unspeakable" in (loop.last_error or "")
    assert "something unspeakable" in paths.log_file(job_id).read_text()


def test_final_never_overlaps_a_periodic_tick(gpuc_home: Path, fake_aws: str) -> None:
    """`final()` has to wait for a tick already in flight, not race it.

    Driven off the tick's own event rather than a sleep: `final()` is called
    while a periodic tick is provably still inside the runner, which is the
    only moment an overlap could happen, and no wall-clock guess decides it.
    """
    overlaps: list[str] = []
    inside = threading.Lock()
    tick_running = threading.Event()

    def slow(
        argv: list[str], timeout: float | None = None, env: sync.Env = None
    ) -> sync.CommandResult:
        if not inside.acquire(blocking=False):
            overlaps.append(" ".join(argv))
        else:
            tick_running.set()
            time.sleep(0.2)
            inside.release()
        return sync.CommandResult(argv, 0, "")

    job_id = jobs.new_job_id()
    loop = loop_for(job_id, slow)
    (paths.workdir(job_id) / "outputs").mkdir(parents=True)
    loop.start()
    try:
        assert tick_running.wait(30.0), "the periodic tick never ran"
        loop.final()
    finally:
        loop.stop()
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
    assert state.mirrored
    assert state.mirror is not None
    assert state.mirror.to == f"s3://bucket/gpuc/host/jobs/{job_id}"
    assert state.output_uploads() == []
    # log, state, then state again now that it records the backup.
    assert [call[-2] for call in runner.calls] == [
        f"s3://bucket/gpuc/host/jobs/{job_id}/log.txt",
        f"s3://bucket/gpuc/host/jobs/{job_id}/state.json",
        f"s3://bucket/gpuc/host/jobs/{job_id}/state.json",
    ]
    mirrored = json.loads(paths.state_file(job_id).read_text())
    assert mirrored["uploads"] == [
        {"to": state.mirror.to, "output": None, "ok_at": state.mirror.ok_at, "error": None}
    ]


def test_a_failed_meta_sync_records_the_error_and_not_a_success(
    gpuc_home: Path, fake_aws: str
) -> None:
    job_id = enqueued_job()
    with pytest.raises(sync.SyncError):
        sync.final_meta_sync(job_id, "s3://bucket/x", runner=RecordingRunner(returncode=1))
    state = jobs.read_state(job_id)
    assert not state.mirrored
    assert state.mirror is not None
    assert state.mirror.ok_at is None and state.mirror.error


def test_a_mirror_that_worked_before_and_then_fails_is_not_mirrored(
    gpuc_home: Path, fake_aws: str
) -> None:
    job_id = enqueued_job()
    sync.final_meta_sync(job_id, "s3://bucket/x", runner=RecordingRunner())
    with pytest.raises(sync.SyncError):
        sync.final_meta_sync(job_id, "s3://bucket/x", runner=RecordingRunner(returncode=1))
    state = jobs.read_state(job_id)
    assert state.mirror is not None and state.mirror.ok_at and state.mirror.error
    assert not state.mirrored


def test_no_prefix_records_nothing_and_uploads_nothing(gpuc_home: Path, fake_aws: str) -> None:
    job_id = enqueued_job()
    runner = RecordingRunner()
    assert sync.final_meta_sync(job_id, None, runner=runner) is None
    assert runner.calls == []
    assert jobs.read_state(job_id).mirror is None


def test_only_the_trailing_state_put_failing_keeps_the_record(
    gpuc_home: Path, fake_aws: str
) -> None:
    job_id = enqueued_job()
    warning = sync.final_meta_sync(job_id, "s3://bucket/x", runner=FlakyRunner(fail_from=3))
    assert warning and "one revision behind" in warning
    # The local state is the authority: log and state did reach S3.
    assert jobs.read_state(job_id).mirrored


def test_final_records_every_output_as_uploaded(gpuc_home: Path, fake_aws: str) -> None:
    job_id = jobs.new_job_id()
    loop = loop_for(job_id, RecordingRunner())
    with pytest.raises(sync.MissingOutput):
        loop.final()  # the output path was never written
    spec = jobs.read_spec(job_id)
    assert not jobs.read_state(job_id).outputs_uploaded(spec)
    (paths.workdir(job_id) / "outputs").mkdir(parents=True)
    loop.final()
    assert jobs.read_state(job_id).outputs_uploaded(spec)


def test_final_clears_ok_at_so_a_failed_last_upload_leaves_none(
    gpuc_home: Path, fake_aws: str
) -> None:
    """A tick that worked an hour ago says nothing about the files the job
    wrote in its last minute: only the final upload can confirm them."""
    job_id = jobs.new_job_id()
    runner = RecordingRunner()
    loop = loop_for(job_id, runner)
    (paths.workdir(job_id) / "outputs").mkdir(parents=True)
    jobs.record_upload(job_id, "s3://b/o", "outputs", ok_at="2026-01-01T00:00:00Z")
    runner.returncode = 1
    runner.output = "AccessDenied"
    with pytest.raises(sync.SyncError):
        loop.final()
    (record,) = jobs.read_state(job_id).output_uploads()
    assert record.ok_at is None
    assert record.error and "AccessDenied" in record.error
    assert not jobs.read_state(job_id).outputs_uploaded(jobs.read_spec(job_id))


def test_final_leaves_the_mirror_record_alone(gpuc_home: Path, fake_aws: str) -> None:
    job_id = jobs.new_job_id()
    loop = loop_for(job_id, RecordingRunner())
    (paths.workdir(job_id) / "outputs").mkdir(parents=True)
    jobs.record_upload(job_id, f"s3://p/jobs/{job_id}", None, ok_at="2026-01-01T00:00:00Z")
    loop.final()
    state = jobs.read_state(job_id)
    assert state.mirror is not None and state.mirror.ok_at == "2026-01-01T00:00:00Z"


def test_final_with_nothing_declared_records_no_uploads(gpuc_home: Path, fake_aws: str) -> None:
    job_id = jobs.new_job_id()
    loop = loop_for(job_id, RecordingRunner(), outputs=[])
    loop.final()
    state = jobs.read_state(job_id)
    assert state.uploads == []
    assert state.outputs_uploaded(jobs.read_spec(job_id))  # nothing was declared
