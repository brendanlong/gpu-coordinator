"""Workdir cleanup: the `cleanup:` policy matrix, and the `clean` sweep.

The whole point of this feature is deleting things, so most of these tests are
about what it must *not* delete.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from gpuc.host import __main__ as host_cli
from gpuc.host import cleanup, jobs, paths, queue, runner
from gpuc.host.jobs import JobSpec
from tests.conftest import make_spec
from tests.test_runner import deps, log_of, prepare

# -- the policy matrix --------------------------------------------------------

POLICY_MATRIX = [
    ("on_success", "succeeded", True),
    ("on_success", "failed", False),
    ("on_success", "cancelled", False),
    ("always", "succeeded", True),
    ("always", "failed", True),
    ("always", "cancelled", True),
    ("never", "succeeded", False),
    ("never", "failed", False),
    ("never", "cancelled", False),
]


@pytest.mark.parametrize(("policy", "status", "expected"), POLICY_MATRIX)
def test_policy_matrix(policy: str, status: str, expected: bool) -> None:
    assert cleanup.should_remove(policy, status) is expected


@pytest.mark.parametrize("policy", ["on_success", "always", "never"])
@pytest.mark.parametrize("status", ["running", "queued"])
def test_no_policy_ever_removes_an_unfinished_job(policy: str, status: str) -> None:
    assert cleanup.should_remove(policy, status) is False


def test_the_default_policy_is_on_success() -> None:
    assert make_spec().cleanup == "on_success"


def test_an_unknown_policy_is_rejected_at_the_spec() -> None:
    with pytest.raises(ValueError, match="on_success, always, never"):
        JobSpec.from_dict({"command": "true", "cleanup": "on-success"})


def test_cleanup_round_trips_through_the_spec_file(gpuc_home: Path) -> None:
    job_id = queue.enqueue(make_spec(cleanup="always"))
    assert jobs.read_spec(job_id).cleanup == "always"


# -- the runner applies it ----------------------------------------------------


def run_with_policy(policy: str, command: str) -> str:
    job_id = prepare(command=command, cleanup=policy)
    paths.workdir(job_id).mkdir(parents=True, exist_ok=True)
    (paths.workdir(job_id) / "big.bin").write_bytes(b"x" * 4096)
    runner.run_job(job_id, deps())
    return job_id


@pytest.mark.parametrize(("policy", "status", "expected"), POLICY_MATRIX)
def test_the_runner_applies_the_policy(
    gpuc_home: Path, policy: str, status: str, expected: bool
) -> None:
    command = {"succeeded": "true", "failed": "exit 7", "cancelled": "sleep 30"}[status]
    job_id = prepare(command=command, cleanup=policy)
    paths.workdir(job_id).mkdir(parents=True, exist_ok=True)
    (paths.workdir(job_id) / "big.bin").write_bytes(b"x" * 4096)
    if status == "cancelled":
        queue.cancel(job_id)
    runner.run_job(job_id, deps())

    state = jobs.read_state(job_id)
    assert state.status == status
    assert paths.workdir(job_id).exists() is not expected
    assert state.workdir_removed is expected
    # Deleting the workdir must not cost the one fact a failed run is kept for.
    if status == "failed":
        assert (state.exit_code, state.reason) == (7, "exit 7")


def test_cleanup_keeps_spec_state_and_log(gpuc_home: Path) -> None:
    job_id = run_with_policy("always", "echo keep-me")
    job_dir = paths.job_dir(job_id)
    assert not (job_dir / "workdir").exists()
    for name in ("spec.json", "state.json", "log.txt"):
        assert (job_dir / name).exists(), name
    assert "keep-me" in log_of(job_id)


def test_the_log_says_the_workdir_went_and_why(gpuc_home: Path) -> None:
    job_id = run_with_policy("always", "true")
    assert "removed workdir (cleanup=always)" in log_of(job_id)
    assert "spec.json, state.json and log.txt are kept" in log_of(job_id)


def test_a_failed_job_keeps_its_workdir_for_inspection(gpuc_home: Path) -> None:
    job_id = run_with_policy("on_success", "echo oops >&2; exit 5")
    assert (paths.workdir(job_id) / "big.bin").exists()
    assert jobs.read_state(job_id).workdir_removed is False


def test_outputs_are_synced_before_the_workdir_goes(gpuc_home: Path) -> None:
    """The final sync reads from inside the workdir, so order is load-bearing."""
    uploaded: list[str] = []

    def record(argv: list[str], _timeout: float | None, _env: object) -> object:
        from gpuc.host.sync import CommandResult

        source = next((a for a in argv if "workdir" in a), "")
        uploaded.append(f"{source}:{Path(source).exists()}")
        return CommandResult(argv, 0, "")

    job_id = prepare(
        command="mkdir -p results && echo done > results/r.txt",
        cleanup="always",
        outputs=[{"path": "results", "s3": "s3://bucket/{job_id}"}],
    )
    paths.workdir(job_id).mkdir(parents=True, exist_ok=True)
    runner.run_job(job_id, deps(command_runner=record))

    assert uploaded, "the final sync never ran"
    assert all(entry.endswith(":True") for entry in uploaded), uploaded
    assert not paths.workdir(job_id).exists()


# -- the clean sweep ----------------------------------------------------------


def finished_job(job_id_status: str, *, ended: datetime | None = None, size: int = 8192) -> str:
    job_id = queue.enqueue(make_spec())
    queue.remove_marker(job_id)
    paths.workdir(job_id).mkdir(parents=True, exist_ok=True)
    (paths.workdir(job_id) / "blob.bin").write_bytes(b"y" * size)
    jobs.update_state(
        job_id,
        status=job_id_status,
        ended_at=(ended or datetime.now(UTC)).isoformat(timespec="microseconds"),
    )
    return job_id


def test_clean_removes_every_finished_workdir(gpuc_home: Path) -> None:
    ids = [finished_job(s) for s in ("succeeded", "failed", "cancelled")]
    result = cleanup.clean(all_finished=True)
    assert sorted(c.job_id for c in result.removed) == sorted(ids)
    assert result.freed_bytes > 0
    for job_id in ids:
        assert not paths.workdir(job_id).exists()
        assert jobs.read_state(job_id).workdir_removed is True


def test_dry_run_deletes_nothing_but_reports_the_same_jobs(gpuc_home: Path) -> None:
    job_id = finished_job("succeeded")
    result = cleanup.clean(all_finished=True, dry_run=True)
    assert [c.job_id for c in result.removed] == [job_id]
    assert result.freed_bytes > 0
    assert (paths.workdir(job_id) / "blob.bin").exists()
    assert jobs.read_state(job_id).workdir_removed is False


def test_clean_never_touches_a_running_job(gpuc_home: Path) -> None:
    running = finished_job("running")
    finished = finished_job("succeeded")
    result = cleanup.clean(all_finished=True)
    assert [c.job_id for c in result.removed] == [finished]
    assert (paths.workdir(running) / "blob.bin").exists()
    assert any(s.job_id == running and s.why == "status running" for s in result.skipped)


def test_clean_never_touches_a_queued_job(gpuc_home: Path) -> None:
    queued = queue.enqueue(make_spec())
    paths.workdir(queued).mkdir(parents=True, exist_ok=True)
    (paths.workdir(queued) / "blob.bin").write_bytes(b"q")
    result = cleanup.clean(all_finished=True)
    assert not result.removed
    assert (paths.workdir(queued) / "blob.bin").exists()


def test_clean_never_touches_a_job_with_no_state(gpuc_home: Path) -> None:
    orphan = "20260101-000000-abcdef"
    paths.ensure_job_layout(orphan)
    (paths.workdir(orphan) / "blob.bin").write_bytes(b"z")
    result = cleanup.clean(all_finished=True)
    assert not result.removed
    assert (paths.workdir(orphan) / "blob.bin").exists()
    assert any(s.job_id == orphan and "no readable state" in s.why for s in result.skipped)


def test_clean_leaves_a_job_whose_state_is_corrupt(gpuc_home: Path) -> None:
    job_id = finished_job("succeeded")
    paths.state_file(job_id).write_text("{not json")
    result = cleanup.clean(all_finished=True)
    assert not result.removed
    assert (paths.workdir(job_id) / "blob.bin").exists()


def test_clean_keeps_spec_state_and_log(gpuc_home: Path) -> None:
    job_id = finished_job("succeeded")
    cleanup.clean(all_finished=True)
    for name in ("spec.json", "state.json", "log.txt"):
        assert (paths.job_dir(job_id) / name).exists(), name


def test_older_than_only_takes_jobs_past_the_cutoff(gpuc_home: Path) -> None:
    now = datetime.now(UTC)
    old = finished_job("succeeded", ended=now - timedelta(days=9))
    young = finished_job("succeeded", ended=now - timedelta(hours=2))
    result = cleanup.clean(older_than_days=7.0)
    assert [c.job_id for c in result.removed] == [old]
    assert (paths.workdir(young) / "blob.bin").exists()


def test_older_than_skips_a_job_with_no_end_time(gpuc_home: Path) -> None:
    job_id = finished_job("succeeded")
    jobs.update_state(job_id, ended_at=None)
    result = cleanup.clean(older_than_days=1.0)
    assert not result.removed
    assert any("ended_at" in s.why for s in result.skipped)


def test_neither_flag_removes_nothing(gpuc_home: Path) -> None:
    job_id = finished_job("succeeded")
    result = cleanup.clean()
    assert not result.removed
    assert (paths.workdir(job_id) / "blob.bin").exists()


def test_a_workdir_already_gone_is_not_an_error(gpuc_home: Path) -> None:
    job_id = finished_job("succeeded")
    cleanup.clean(all_finished=True)
    again = cleanup.clean(all_finished=True)
    assert not again.removed and not again.errors
    assert jobs.read_state(job_id).status == "succeeded"


# -- leftover staged specs ----------------------------------------------------


def staged(job_id: str, age_s: float = 0.0) -> Path:
    directory = paths.home() / "incoming"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{job_id}.json"
    path.write_text("{}\n")
    if age_s:
        stamp = path.stat().st_mtime - age_s
        import os

        os.utime(path, (stamp, stamp))
    return path


def test_clean_removes_a_staged_spec_whose_job_finished(gpuc_home: Path) -> None:
    job_id = finished_job("succeeded")
    path = staged(job_id)
    result = cleanup.clean(all_finished=True)
    assert result.incoming_removed == [path.name]
    assert not path.exists()


def test_clean_removes_an_orphaned_staged_spec_once_it_is_old(gpuc_home: Path) -> None:
    fresh = staged("20260101-000000-aaaaaa")
    old = staged("20260101-000000-bbbbbb", age_s=cleanup.INCOMING_STALE_S + 60)
    result = cleanup.clean(all_finished=True)
    assert result.incoming_removed == [old.name]
    assert fresh.exists(), "a spec a concurrent submit may still be enqueueing"


def test_clean_leaves_a_staged_spec_for_a_running_job(gpuc_home: Path) -> None:
    job_id = finished_job("running")
    path = staged(job_id)
    cleanup.clean(all_finished=True)
    assert path.exists()


def test_dry_run_does_not_remove_staged_specs(gpuc_home: Path) -> None:
    job_id = finished_job("succeeded")
    path = staged(job_id)
    result = cleanup.clean(all_finished=True, dry_run=True)
    assert result.incoming_removed == [path.name]
    assert path.exists()


# -- the host CLI -------------------------------------------------------------


def test_host_cli_clean_prints_json(gpuc_home: Path, capsys: pytest.CaptureFixture[str]) -> None:
    job_id = finished_job("succeeded")
    assert host_cli.main(["clean", "--all-finished"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["dry_run"] is False
    assert [c["job_id"] for c in payload["removed"]] == [job_id]
    assert payload["freed_bytes"] > 0


def test_host_cli_clean_requires_a_selection(gpuc_home: Path) -> None:
    with pytest.raises(SystemExit):
        host_cli.main(["clean"])


def test_host_cli_clean_only_takes_the_named_jobs_however_recent(
    gpuc_home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Naming a job id is the selection: its age is not a reason to keep it."""
    wanted = finished_job("succeeded")
    other = finished_job("succeeded")
    assert host_cli.main(["clean", "--only", wanted]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert [c["job_id"] for c in payload["removed"]] == [wanted]
    assert not paths.workdir(wanted).exists()
    assert (paths.workdir(other) / "blob.bin").exists()


def test_host_cli_clean_only_empty_means_none(
    gpuc_home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    job_id = finished_job("succeeded")
    assert host_cli.main(["clean", "--only", ""]) == 0
    assert json.loads(capsys.readouterr().out)["removed"] == []
    assert (paths.workdir(job_id) / "blob.bin").exists()


def test_host_cli_clean_reports_a_job_id_it_has_never_heard_of(
    gpuc_home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    job_id = finished_job("succeeded")
    assert host_cli.main(["clean", "--only", f"{job_id},20260101-000000-typo11"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["errors"] == ["20260101-000000-typo11: no job with that id on this host"]
    # The typo does not stop the job that does exist from being cleaned.
    assert [c["job_id"] for c in payload["removed"]] == [job_id]


def test_host_cli_clean_only_says_when_a_named_workdir_is_already_gone(
    gpuc_home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    job_id = finished_job("succeeded")
    cleanup.clean(all_finished=True)
    assert host_cli.main(["clean", "--only", job_id]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["removed"] == []
    assert [s["why"] for s in payload["skipped"]] == ["workdir already gone"]


def test_host_cli_status_reports_workdir_bytes(
    gpuc_home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    done = finished_job("succeeded")
    running = finished_job("running")
    assert host_cli.main(["status"]) == 0
    payload = json.loads(capsys.readouterr().out)
    sizes = {job["job_id"]: job["workdir_bytes"] for job in payload["jobs"]}
    assert sizes[done] and sizes[done] > 0
    # A live job's workdir is still being written to; its size means nothing.
    assert sizes[running] is None


def test_dir_size_counts_a_hardlink_once(gpuc_home: Path, tmp_path: Path) -> None:
    root = tmp_path / "tree"
    (root / "sub").mkdir(parents=True)
    original = root / "a.bin"
    original.write_bytes(b"x" * 100_000)
    (root / "sub" / "b.bin").hardlink_to(original)
    once = cleanup.dir_size(root)
    (root / "sub" / "c.bin").write_bytes(b"x" * 100_000)
    assert cleanup.dir_size(root) > once


def test_human_bytes_reads_like_du() -> None:
    assert cleanup.human_bytes(0) == "0 B"
    assert cleanup.human_bytes(2048) == "2.0 KiB"
    assert cleanup.human_bytes(7 * (1 << 30)) == "7.0 GiB"
