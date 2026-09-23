"""Files that came with the checkout are not the job's outputs."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

import pytest

from gpuc.host import baseline, destinations, jobs, paths, queue, runner
from gpuc.host.jobs import HostConfig
from gpuc.host.runner import RunnerDeps
from gpuc.host.sync import CommandResult
from tests.conftest import FAKE_GPUS, fake_smi, make_spec


class Recorder:
    def __init__(self) -> None:
        self.commands: list[list[str]] = []

    def __call__(
        self, argv: list[str], timeout: float | None = None, env: Mapping[str, str] | None = None
    ) -> CommandResult:
        self.commands.append(argv)
        return CommandResult(argv, 0, "")

    def excluded(self) -> list[str]:
        out: list[str] = []
        for argv in self.commands:
            out += [argv[i + 1] for i, word in enumerate(argv) if word == "--exclude"]
        return out

    def uploads(self) -> list[list[str]]:
        return [argv for argv in self.commands if argv[1:3] == ["s3", "sync"]]


def prepare(gpuc_home: Path, command: str, **overrides: object) -> tuple[str, Path]:
    spec = make_spec(
        command=command,
        outputs=[{"path": "results", "s3": "s3://bucket/{job_id}"}],
        **overrides,
    )
    job_id = queue.enqueue(spec)
    results = paths.workdir(job_id) / "results"
    results.mkdir(parents=True, exist_ok=True)
    (results / "report-elephant.md").write_text("committed in the repo\n")
    return job_id, results


@pytest.fixture
def aws(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(destinations, "find_binary", lambda name, env=None: f"/fake/{name}")


def test_a_pre_existing_file_is_excluded_and_a_new_one_is_not(gpuc_home: Path, aws: None) -> None:
    job_id, _ = prepare(gpuc_home, "echo produced > results/new.txt")
    recorder = Recorder()
    code = runner.run_job(
        job_id,
        [FAKE_GPUS[0]],
        1,
        RunnerDeps(smi=fake_smi(), command_runner=recorder, preflight=False, poll_interval_s=0.02),
    )
    assert (code, jobs.read_state(job_id).status) == (0, "succeeded")
    assert "report-elephant.md" in recorder.excluded()
    assert "new.txt" not in recorder.excluded()
    assert jobs.read_state(job_id).outputs_uploaded(jobs.read_spec(job_id))


def test_a_modified_pre_existing_file_is_uploaded(gpuc_home: Path, aws: None) -> None:
    job_id, _ = prepare(gpuc_home, "echo rewritten > results/report-elephant.md")
    recorder = Recorder()
    assert (
        runner.run_job(
            job_id,
            [FAKE_GPUS[0]],
            1,
            RunnerDeps(
                smi=fake_smi(), command_runner=recorder, preflight=False, poll_interval_s=0.02
            ),
        )
        == 0
    )
    assert "report-elephant.md" not in recorder.excluded()


def test_a_job_that_produced_nothing_new_fails_as_no_outputs(gpuc_home: Path, aws: None) -> None:
    job_id, _ = prepare(gpuc_home, "true")
    recorder = Recorder()
    code = runner.run_job(
        job_id,
        [FAKE_GPUS[0]],
        1,
        RunnerDeps(smi=fake_smi(), command_runner=recorder, preflight=False, poll_interval_s=0.02),
    )
    state = jobs.read_state(job_id)
    assert (code, state.status, state.reason) == (1, "failed", "no-outputs")
    assert recorder.uploads() == []
    (record,) = state.output_uploads()
    assert record.ok_at is None and record.error and "nothing new" in record.error


def test_the_baseline_is_taken_before_setup_so_setup_output_counts(
    gpuc_home: Path, aws: None
) -> None:
    job_id, _ = prepare(gpuc_home, "true", setup="echo from-setup > results/setup.txt")
    recorder = Recorder()
    assert (
        runner.run_job(
            job_id,
            [FAKE_GPUS[0]],
            1,
            RunnerDeps(
                smi=fake_smi(), command_runner=recorder, preflight=False, poll_interval_s=0.02
            ),
        )
        == 0
    )
    assert "setup.txt" not in recorder.excluded()
    assert "report-elephant.md" in recorder.excluded()


def test_the_baseline_file_records_size_and_mtime(gpuc_home: Path) -> None:
    job_id, results = prepare(gpuc_home, "true")
    spec = jobs.read_spec(job_id)
    captured = baseline.capture(spec, paths.workdir(job_id), job_id)
    recorded = captured["results"]["report-elephant.md"]
    info = (results / "report-elephant.md").stat()
    assert recorded == [info.st_size, info.st_mtime_ns]
    assert baseline.read(job_id) == captured


def test_a_file_restored_with_the_same_mtime_is_still_baseline(gpuc_home: Path) -> None:
    job_id, results = prepare(gpuc_home, "true")
    spec = jobs.read_spec(job_id)
    baseline.capture(spec, paths.workdir(job_id), job_id)
    target = results / "report-elephant.md"
    info = target.stat()
    target.write_text("committed in the repo\n")
    os.utime(target, ns=(info.st_atime_ns, info.st_mtime_ns))
    entries = baseline.read(job_id)["results"]
    assert baseline.unchanged(results, entries) == ["report-elephant.md"]
    assert not baseline.has_new_content(results, entries)


def test_final_sync_still_excludes_the_baseline(gpuc_home: Path, aws: None) -> None:
    jobs.write_config(HostConfig(host="h", s3_prefix=None))
    job_id, _ = prepare(gpuc_home, "sleep 0.1; echo late > results/late.txt")
    recorder = Recorder()
    runner.run_job(
        job_id,
        [FAKE_GPUS[0]],
        1,
        RunnerDeps(smi=fake_smi(), command_runner=recorder, preflight=False, poll_interval_s=0.02),
    )
    # The final pass drops the min-age exclusions but keeps the baseline ones.
    final = recorder.uploads()[-1]
    assert "--exclude" in final and "report-elephant.md" in final
    assert "late.txt" not in final


def test_a_preempted_re_run_keeps_the_baseline_the_first_attempt_took(
    gpuc_home: Path, aws: None
) -> None:
    """Re-scanned at the start of attempt 2, the files attempt 1 produced would
    be recorded as files the *checkout* arrived with -- so this attempt would
    never upload them, and one that died before rewriting them would report
    `no-outputs` with its results sitting right there."""
    job_id, results = prepare(gpuc_home, "true")
    recorder = Recorder()
    deps = RunnerDeps(
        smi=fake_smi(), command_runner=recorder, preflight=False, poll_interval_s=0.02
    )
    runner.run_job(job_id, [FAKE_GPUS[0]], 1, deps)
    first = baseline.read(job_id)
    assert "report-elephant.md" in first["results"]

    # ...and then a preempted attempt leaves a checkpoint of its own behind.
    (results / "ckpt-100.bin").write_text("weights\n")
    jobs.update_state(job_id, status="queued", attempt=2, ended_at=None)
    runner.run_job(job_id, [FAKE_GPUS[0]], 2, deps)

    assert baseline.read(job_id) == first
    assert "ckpt-100.bin" not in baseline.read(job_id)["results"]
    assert "ckpt-100.bin" not in recorder.excluded()
    log = paths.log_file(job_id).read_text()
    assert "keeping the one taken before the first attempt" in log
