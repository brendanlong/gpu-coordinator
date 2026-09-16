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
    # The runner is the primary writer of the figure `status` reads, and it
    # must agree with what it just did either way.
    if expected:
        assert state.workdir_bytes == 0
    else:
        assert state.workdir_bytes is not None and state.workdir_bytes > 0
    # Deleting the workdir must not cost the one fact a failed run is kept for.
    if status == "failed":
        assert (state.exit_code, state.reason) == (7, "exit 7")


def test_the_runner_records_the_size_before_it_mirrors_state(
    gpuc_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Written after the final meta sync, the mirror would never carry it."""
    mirrored: list[int | None] = []

    def capture(job_id: str, prefix: str, **kwargs: object) -> str | None:
        mirrored.append(jobs.read_state(job_id).workdir_bytes)
        return None

    monkeypatch.setattr(runner.sync, "final_meta_sync", capture)
    job_id = prepare(command="true", cleanup="never")
    paths.workdir(job_id).mkdir(parents=True, exist_ok=True)
    (paths.workdir(job_id) / "big.bin").write_bytes(b"x" * 4096)
    runner.run_job(job_id, deps())
    assert mirrored and mirrored[0] is not None and mirrored[0] > 0


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


# -- what only the automatic sweep refuses ------------------------------------


def job_wanting_its_workdir_kept(policy: str) -> str:
    job_id = queue.enqueue(make_spec(cleanup=policy))
    queue.remove_marker(job_id)
    paths.workdir(job_id).mkdir(parents=True, exist_ok=True)
    (paths.workdir(job_id) / "blob.bin").write_bytes(b"y" * 8192)
    jobs.update_state(job_id, status="failed", ended_at=jobs.utc_now())
    return job_id


def job_with_outputs_still_only_here() -> str:
    spec = make_spec(outputs=[{"path": "results", "s3": "s3://bucket/{job_id}"}])
    job_id = queue.enqueue(spec)
    queue.remove_marker(job_id)
    results = paths.workdir(job_id) / "results"
    results.mkdir(parents=True, exist_ok=True)
    (results / "checkpoint.pt").write_bytes(b"w" * 8192)
    jobs.update_state(job_id, status="failed", ended_at=jobs.utc_now())
    return job_id


def test_the_automatic_sweep_leaves_a_job_that_asked_to_keep_its_workdir(
    gpuc_home: Path,
) -> None:
    never = job_wanting_its_workdir_kept("never")
    on_success = job_wanting_its_workdir_kept("on_success")
    result = cleanup.clean(all_finished=True, automatic=True)
    assert [c.job_id for c in result.removed] == [on_success]
    assert (paths.workdir(never) / "blob.bin").exists()
    assert any(s.job_id == never and s.why == "cleanup: never" for s in result.skipped)


def test_the_automatic_sweep_leaves_outputs_that_are_still_only_here(gpuc_home: Path) -> None:
    """The sweep must not bin what `purge` refuses to, and `status` warns about."""
    job_id = job_with_outputs_still_only_here()
    result = cleanup.clean(all_finished=True, automatic=True)
    assert not result.removed
    assert (paths.workdir(job_id) / "results" / "checkpoint.pt").exists()
    assert any(s.job_id == job_id and "outputs" in s.why for s in result.skipped)

    # ...and once they are somewhere else, it is free to go.
    jobs.update_state(job_id, outputs_synced_at=jobs.utc_now())
    assert [c.job_id for c in cleanup.clean(all_finished=True, automatic=True).removed] == [job_id]


def test_a_person_naming_the_job_still_takes_it(gpuc_home: Path) -> None:
    """Both guards are about a sweep nobody asked for, not about `gpuc clean`."""
    never = job_wanting_its_workdir_kept("never")
    unconfirmed = job_with_outputs_still_only_here()
    result = cleanup.clean(all_finished=True, only=[never, unconfirmed])
    assert sorted(c.job_id for c in result.removed) == sorted([never, unconfirmed])


def test_the_automatic_sweep_leaves_a_job_whose_spec_is_unreadable(gpuc_home: Path) -> None:
    job_id = finished_job("succeeded")
    paths.job_dir(job_id).joinpath("spec.json").write_text("{not json")
    result = cleanup.clean(all_finished=True, automatic=True)
    assert not result.removed
    assert (paths.workdir(job_id) / "blob.bin").exists()


def test_the_purge_implied_sweep_is_guarded_too_when_it_is_automatic(gpuc_home: Path) -> None:
    """`retention_days` must not be a way around what `workdir_days` respects."""
    job_id = job_with_outputs_still_only_here()
    result = cleanup.purge(older_than_days=0.0, automatic=True)
    assert not result.purged
    assert not result.removed
    assert (paths.workdir(job_id) / "results" / "checkpoint.pt").exists()


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


def test_host_cli_clean_refuses_a_selection_holding_a_job_id_it_does_not_know(
    gpuc_home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    job_id = finished_job("succeeded")
    assert host_cli.main(["clean", "--only", f"{job_id},20260101-000000-typo11"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["errors"] == [
        "20260101-000000-typo11: no job with that id on this host",
        "refused the whole selection: nothing was removed",
    ]
    assert payload["removed"] == []
    assert (paths.workdir(job_id) / "blob.bin").exists()


def test_host_cli_clean_only_ignores_a_job_with_no_usable_ended_at(gpuc_home: Path) -> None:
    job_id = finished_job("failed")
    jobs.update_state(job_id, ended_at=None)
    assert host_cli.main(["clean", "--only", job_id]) == 0
    assert not paths.workdir(job_id).exists()


def test_host_cli_clean_only_says_when_a_named_workdir_is_already_gone(
    gpuc_home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    job_id = finished_job("succeeded")
    cleanup.clean(all_finished=True)
    assert host_cli.main(["clean", "--only", job_id]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["removed"] == []
    assert [s["why"] for s in payload["skipped"]] == ["workdir already gone"]


def sizes_from_status(capsys: pytest.CaptureFixture[str]) -> dict[str, int | None]:
    assert host_cli.main(["status"]) == 0
    payload = json.loads(capsys.readouterr().out)
    return {job["job_id"]: job["workdir_bytes"] for job in payload["jobs"]}


def test_host_cli_status_reports_the_recorded_workdir_bytes(
    gpuc_home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    done = finished_job("succeeded")
    jobs.update_state(done, workdir_bytes=4096)
    assert sizes_from_status(capsys)[done] == 4096


def test_host_cli_status_does_not_re_measure_what_it_already_knows(
    gpuc_home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Walking every finished venv per call cost 4 s on a host holding sixty."""
    done = finished_job("succeeded")
    jobs.update_state(done, workdir_bytes=4096)

    def refuse(root: Path) -> int:
        raise AssertionError("status walked a workdir it had a figure for")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(cleanup, "reclaimable_bytes", refuse)
        assert sizes_from_status(capsys)[done] == 4096


def test_host_cli_status_sizes_a_gone_workdir_without_measuring_anything(
    gpuc_home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """No workdir frees nothing, and nobody has to have written that down: it
    is what left every pre-upgrade job reading as `not sized yet` forever."""
    done = finished_job("succeeded")
    cleanup.remove_workdir(done)
    jobs.update_state(done, workdir_bytes=None)

    def refuse(root: Path) -> int:
        raise AssertionError("status walked a workdir that is gone")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(cleanup, "reclaimable_bytes", refuse)
        assert sizes_from_status(capsys)[done] == 0


def test_host_cli_status_measures_the_workdir_nobody_measured(
    gpuc_home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A runner that died before writing the figure, or a job older than it."""
    done = finished_job("succeeded")
    assert jobs.read_state(done).workdir_bytes is None
    measured = sizes_from_status(capsys)[done]
    assert measured is not None and measured > 0
    assert jobs.read_state(done).workdir_bytes == measured, "the next call must not walk again"


def test_host_cli_status_believes_a_measured_zero(
    gpuc_home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Zero is a real answer for a workdir that is there: a tree whose extents
    are all shared measures nothing, and so does an empty dir on tmpfs. Reading
    it as "unknown" would walk that tree again on every single call."""
    done = finished_job("succeeded")
    jobs.update_state(done, workdir_bytes=0)

    def refuse(root: Path) -> int:
        raise AssertionError("re-walked a workdir that had been measured at zero")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(cleanup, "reclaimable_bytes", refuse)
        assert sizes_from_status(capsys)[done] == 0


class RunsOutOfTime:
    """A monotonic clock that is past any deadline after its first answer."""

    def __init__(self) -> None:
        self.answers = 0

    def monotonic(self) -> float:
        self.answers += 1
        return 0.0 if self.answers == 1 else 1e12


def test_host_cli_status_stops_measuring_when_its_budget_is_spent(
    gpuc_home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Blowing the control side's 60 s deadline would cost the whole host line
    -- its queue, its running jobs and its cards -- not just these figures."""
    ids = [finished_job("succeeded"), finished_job("failed")]
    walked: list[Path] = []

    def watch(root: Path) -> int:
        walked.append(root)
        return 4096

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(cleanup, "time", RunsOutOfTime())
        patch.setattr(cleanup, "reclaimable_bytes", watch)
        sizes = sizes_from_status(capsys)
    assert len(walked) == 1, "the second job is past the budget"
    measured = [job for job in ids if sizes[job] == 4096]
    assert len(measured) == 1
    assert [sizes[job] for job in ids if job not in measured] == [None]
    # Written down, so the next call carries on from here rather than starting
    # the same queue of walks over again.
    assert jobs.read_state(measured[0]).workdir_bytes == 4096


def test_host_cli_status_never_sizes_a_running_job(
    gpuc_home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Its workdir is still being written to, so any figure would be a lie."""
    running = finished_job("running")

    def refuse(root: Path) -> int:
        raise AssertionError("status walked a live job's workdir")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(cleanup, "reclaimable_bytes", refuse)
        assert sizes_from_status(capsys)[running] is None


def test_record_workdir_size_writes_what_a_later_status_reads(
    gpuc_home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    done = finished_job("succeeded")
    measured = cleanup.record_workdir_size(done)
    assert measured and measured > 0
    assert jobs.read_state(done).workdir_bytes == measured
    assert sizes_from_status(capsys)[done] == measured


def test_record_workdir_size_says_zero_once_the_workdir_is_gone(gpuc_home: Path) -> None:
    done = finished_job("succeeded")
    cleanup.remove_workdir(done)
    assert cleanup.record_workdir_size(done) == 0
    assert jobs.read_state(done).workdir_bytes == 0


def test_the_sweep_records_that_it_freed_everything(gpuc_home: Path) -> None:
    done = finished_job("succeeded")
    jobs.update_state(done, workdir_bytes=999999)
    cleanup.clean(all_finished=True)
    state = jobs.read_state(done)
    assert state.workdir_removed is True
    assert state.workdir_bytes == 0, "status would still be quoting a workdir that is gone"


def test_reclaimable_bytes_counts_a_hardlink_once(gpuc_home: Path, tmp_path: Path) -> None:
    root = tmp_path / "tree"
    (root / "sub").mkdir(parents=True)
    original = root / "a.bin"
    original.write_bytes(b"x" * 100_000)
    (root / "sub" / "b.bin").hardlink_to(original)
    once = cleanup.reclaimable_bytes(root)
    (root / "sub" / "c.bin").write_bytes(b"x" * 100_000)
    assert cleanup.reclaimable_bytes(root) > once


def test_dir_size_counts_an_in_tree_hardlink_once(gpuc_home: Path, tmp_path: Path) -> None:
    """du semantics still means one inode, one count."""
    root = tmp_path / "tree"
    (root / "sub").mkdir(parents=True)
    original = root / "a.bin"
    original.write_bytes(b"x" * 100_000)
    (root / "sub" / "b.bin").hardlink_to(original)
    both_names = cleanup.dir_size(root)
    original.unlink()
    assert cleanup.dir_size(root) == both_names


def test_dir_size_counts_what_another_tree_links_to(gpuc_home: Path, tmp_path: Path) -> None:
    """The uv cache's own size is a `du` question: it holds those bytes."""
    cache = tmp_path / "cache"
    cache.mkdir()
    cached = cache / "wheel.bin"
    cached.write_bytes(b"x" * 100_000)
    venv = tmp_path / "venv"
    venv.mkdir()
    (venv / "wheel.bin").hardlink_to(cached)
    assert cleanup.dir_size(cache) >= 100_000
    assert cleanup.reclaimable_bytes(cache) < 100_000


def test_reclaimable_bytes_skips_what_a_link_outside_the_tree_still_holds(
    gpuc_home: Path, tmp_path: Path
) -> None:
    """The uv cache case: deleting the workdir frees none of those bytes."""
    cache = tmp_path / "cache"
    cache.mkdir()
    cached = cache / "wheel.bin"
    cached.write_bytes(b"x" * 100_000)

    root = tmp_path / "tree"
    (root / "sub").mkdir(parents=True)
    (root / "sub" / "linked.bin").hardlink_to(cached)
    linked_only = cleanup.reclaimable_bytes(root)

    (root / "own.bin").write_bytes(b"y" * 100_000)
    assert cleanup.reclaimable_bytes(root) - linked_only >= 100_000
    assert linked_only < 100_000, "counted bytes the cache still holds"


def reflink(source: Path, target: Path) -> bool:
    """`cp --reflink=always`, or False where the filesystem cannot."""
    import subprocess

    return (
        subprocess.run(
            ["cp", "--reflink=always", str(source), str(target)],
            capture_output=True,
        ).returncode
        == 0
    )


def test_reclaimable_bytes_skips_extents_a_file_outside_the_tree_shares(
    gpuc_home: Path, tmp_path: Path
) -> None:
    """uv's `clone` link mode: shared extents, `st_nlink` of 1, nothing to see
    without FIEMAP."""
    cache = tmp_path / "cache"
    cache.mkdir()
    cached = cache / "wheel.bin"
    cached.write_bytes(b"x" * (256 * 1024))

    root = tmp_path / "tree"
    root.mkdir()
    if not reflink(cached, root / "cloned.bin"):
        pytest.skip("this filesystem has no reflinks, so there is nothing to detect")
    assert (root / "cloned.bin").stat().st_nlink == 1, "a reflink is not a hardlink"

    assert cleanup.dir_size(root) >= 256 * 1024
    assert cleanup.reclaimable_bytes(root) < 64 * 1024


def test_shared_extent_bytes_answers_zero_rather_than_raising(
    gpuc_home: Path, tmp_path: Path
) -> None:
    """Every failure means "assume it is all yours": over-report, never raise."""
    assert cleanup.shared_extent_bytes(str(tmp_path / "does-not-exist")) == 0
    ordinary = tmp_path / "plain.bin"
    ordinary.write_bytes(b"z" * (128 * 1024))
    # A file nothing else references: zero either way, whatever the fs answers.
    assert cleanup.shared_extent_bytes(str(ordinary)) == 0


def test_every_file_with_blocks_is_asked_about(gpuc_home: Path, tmp_path: Path) -> None:
    """No size floor: the walk happens once per job, so it may as well be exact."""
    asked: list[str] = []
    root = tmp_path / "tree"
    root.mkdir()
    (root / "big.bin").write_bytes(b"x" * (128 * 1024))
    (root / "small.bin").write_bytes(b"y" * 32)
    (root / "empty.bin").touch()

    real = cleanup.shared_extent_bytes

    def spy(path: str) -> int:
        asked.append(Path(path).name)
        return real(path)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(cleanup, "shared_extent_bytes", spy)
        cleanup.reclaimable_bytes(root)
    # An empty file has no blocks to share, so it is not worth the syscall.
    assert sorted(asked) == ["big.bin", "small.bin"]


def test_shared_extents_come_off_the_total(gpuc_home: Path, tmp_path: Path) -> None:
    """What the reflink test proves where reflinks exist, without needing them."""
    root = tmp_path / "tree"
    root.mkdir()
    (root / "a.bin").write_bytes(b"x" * (128 * 1024))
    whole = cleanup.reclaimable_bytes(root)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(cleanup, "shared_extent_bytes", lambda path: 64 * 1024)
        assert cleanup.reclaimable_bytes(root) == whole - 64 * 1024


def test_a_share_larger_than_the_file_cannot_drive_the_total_negative(
    gpuc_home: Path, tmp_path: Path
) -> None:
    """btrfs compression makes `fe_length` logical and `st_blocks` compressed,
    so the shared figure really can exceed what the file occupies."""
    root = tmp_path / "tree"
    root.mkdir()
    (root / "a.bin").write_bytes(b"x" * (128 * 1024))

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(cleanup, "shared_extent_bytes", lambda path: 1 << 40)
        assert cleanup.reclaimable_bytes(root) >= 0


def test_a_symlink_is_never_opened_to_ask_about_it(gpuc_home: Path, tmp_path: Path) -> None:
    """A long target makes a symlink's own st_blocks non-zero on ext4, and
    following it would open whatever a job pointed at."""
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"x" * (128 * 1024))
    root = tmp_path / "tree"
    root.mkdir()
    (root / "link").symlink_to(outside)

    asked: list[str] = []
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(cleanup, "shared_extent_bytes", lambda path: asked.append(path) or 0)
        cleanup.reclaimable_bytes(root)
    assert asked == []
    # And the backstop holds even if something calls it directly.
    assert cleanup.shared_extent_bytes(str(root / "link")) == 0


def test_a_clean_measures_each_workdir_once(gpuc_home: Path) -> None:
    """An ioctl per file is too much to pay twice for one delete."""
    finished_job("succeeded")
    walks = 0
    real = cleanup.reclaimable_bytes

    def counted(root: Path) -> int:
        nonlocal walks
        walks += 1
        return real(root)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(cleanup, "reclaimable_bytes", counted)
        result = cleanup.clean(all_finished=True)
    assert result.freed_bytes > 0
    assert walks == 1


def test_measuring_after_a_clean_took_the_workdir_records_zero(gpuc_home: Path) -> None:
    """The race: `gpuc clean` is another process, and the walk takes seconds."""
    job_id = finished_job("succeeded")
    real = cleanup.workdir_size

    def measure_then_someone_cleans(wanted: str) -> int | None:
        size = real(wanted)
        cleanup.remove_workdir(wanted)  # the other process, mid-walk
        return size

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(cleanup, "workdir_size", measure_then_someone_cleans)
        assert cleanup.record_workdir_size(job_id) == 0
    assert jobs.read_state(job_id).workdir_bytes == 0, (
        "status would advertise disk that clean cannot free"
    )


def test_du_sizing_never_asks_the_filesystem_about_sharing(gpuc_home: Path, tmp_path: Path) -> None:
    root = tmp_path / "tree"
    root.mkdir()
    (root / "big.bin").write_bytes(b"x" * (128 * 1024))

    def refuse(path: str) -> int:
        raise AssertionError("du sizing asked about shared extents")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(cleanup, "shared_extent_bytes", refuse)
        assert cleanup.dir_size(root) >= 128 * 1024


def test_reclaimable_bytes_counts_a_file_once_every_link_to_it_is_in_the_tree(
    gpuc_home: Path, tmp_path: Path
) -> None:
    root = tmp_path / "tree"
    (root / "sub").mkdir(parents=True)
    original = root / "a.bin"
    original.write_bytes(b"x" * 100_000)
    (root / "sub" / "b.bin").hardlink_to(original)
    # Both names are inside the tree, so the bytes really are reclaimable.
    assert cleanup.reclaimable_bytes(root) >= 100_000


def test_reclaimable_bytes_counts_a_directory_whose_nlink_counts_subdirectories(
    gpuc_home: Path, tmp_path: Path
) -> None:
    """A dir's st_nlink is 2 + its subdirs, which must not read as "shared"."""
    root = tmp_path / "tree"
    for name in ("a", "b", "c"):
        (root / name).mkdir(parents=True)
    assert root.stat().st_nlink > 1, "the case this is about"
    every_dir = [root, *(root / name for name in ("a", "b", "c"))]
    blocks = sum(d.stat().st_blocks for d in every_dir)
    if blocks == 0:
        pytest.skip("directories occupy no blocks here (tmpfs), so there is nothing to lose")
    assert cleanup.reclaimable_bytes(root) == blocks * 512


def test_human_bytes_reads_like_du() -> None:
    assert cleanup.human_bytes(0) == "0 B"
    assert cleanup.human_bytes(2048) == "2.0 KiB"
    assert cleanup.human_bytes(7 * (1 << 30)) == "7.0 GiB"
