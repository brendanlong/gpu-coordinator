"""`purge`: deleting a whole job dir, and every reason not to.

`clean` can always be undone by re-running the job; `purge` deletes the record
that it ever ran, so almost all of these tests are about what it refuses.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from gpuc.host import __main__ as host_cli
from gpuc.host import baseline, cleanup, jobs, paths, queue
from gpuc.host.jobs import HostConfig
from tests.conftest import make_spec

PREFIX = "s3://bucket/gpuc/test-host"
OUTPUT = {"path": "results", "s3": "s3://bucket/exp/{job_id}"}


def make_job(
    *,
    status: str = "succeeded",
    days_old: float = 30.0,
    meta_synced: bool = True,
    outputs: bool = False,
    outputs_synced: bool = False,
    produced: bool = True,
    workdir: bool = True,
) -> str:
    spec = make_spec(outputs=[OUTPUT] if outputs else [])
    job_id = queue.enqueue(spec)
    queue.remove_marker(job_id)
    ended = datetime.now(UTC) - timedelta(days=days_old)
    fields: dict[str, Any] = {"status": status}
    if status in jobs.FINISHED_STATUSES:
        fields["ended_at"] = ended.isoformat()
    if meta_synced:
        fields["meta_synced_at"] = ended.isoformat()
        fields["meta_synced_to"] = PREFIX
    if outputs_synced:
        fields["outputs_synced_at"] = ended.isoformat()
    jobs.update_state(job_id, **fields)
    if workdir:
        paths.workdir(job_id).mkdir(parents=True, exist_ok=True)
        (paths.workdir(job_id) / "venv.bin").write_bytes(b"x" * 4096)
        if outputs and produced:
            results = paths.workdir(job_id) / "results"
            results.mkdir(exist_ok=True)
            (results / "checkpoint.pt").write_bytes(b"y" * 2048)
    else:
        for path in sorted(paths.workdir(job_id).rglob("*"), reverse=True):
            path.unlink()
        paths.workdir(job_id).rmdir()
    return job_id


def with_prefix() -> None:
    jobs.write_config(HostConfig(host="test-host", s3_prefix=PREFIX))


def why(result: cleanup.CleanResult, job_id: str) -> str:
    return next(s.why for s in result.purge_skipped if s.job_id == job_id)


# -- the matrix ---------------------------------------------------------------


@pytest.mark.parametrize("status", ["succeeded", "failed", "cancelled"])
def test_every_finished_status_is_purgeable_once_old_and_mirrored(
    gpuc_home: Path, status: str
) -> None:
    job_id = make_job(status=status)
    result = cleanup.purge(older_than_days=7.0)
    assert [c.job_id for c in result.purged] == [job_id]
    assert not paths.job_dir(job_id).exists()
    assert result.freed_bytes > 0


@pytest.mark.parametrize("status", ["running", "queued"])
@pytest.mark.parametrize("force", [False, True])
def test_a_live_job_is_never_purged(gpuc_home: Path, status: str, force: bool) -> None:
    job_id = make_job(status=status, meta_synced=True)
    result = cleanup.purge(older_than_days=0.0, force=force)
    assert result.purged == []
    assert why(result, job_id) == f"status {status}"
    assert paths.spec_file(job_id).exists()


def test_a_job_younger_than_the_horizon_is_kept(gpuc_home: Path) -> None:
    job_id = make_job(days_old=2.0)
    result = cleanup.purge(older_than_days=7.0)
    assert result.purged == []
    assert "2.0 days old" in why(result, job_id)
    assert paths.job_dir(job_id).is_dir()


def test_a_job_with_no_mirror_says_the_host_has_no_prefix(gpuc_home: Path) -> None:
    job_id = make_job(meta_synced=False)
    result = cleanup.purge(older_than_days=7.0)
    assert why(result, job_id) == "not backed up: no s3_prefix on this host"
    assert paths.job_dir(job_id).is_dir()


def test_a_job_with_a_prefix_but_no_record_says_the_upload_failed(gpuc_home: Path) -> None:
    with_prefix()
    job_id = make_job(meta_synced=False)
    result = cleanup.purge(older_than_days=7.0)
    assert why(result, job_id) == "not backed up: final upload failed"


def test_force_purges_an_unmirrored_job_and_says_so(gpuc_home: Path) -> None:
    job_id = make_job(meta_synced=False)
    result = cleanup.purge(older_than_days=7.0, force=True)
    assert [(c.job_id, c.forced) for c in result.purged] == [(job_id, True)]
    assert not paths.job_dir(job_id).exists()


def test_a_job_with_no_usable_ended_at_is_kept(gpuc_home: Path) -> None:
    job_id = make_job()
    jobs.update_state(job_id, ended_at="not a timestamp")
    result = cleanup.purge(older_than_days=0.0)
    assert why(result, job_id) == "finished but records no usable ended_at"


def test_unreadable_state_fails_closed(gpuc_home: Path) -> None:
    job_id = make_job()
    paths.state_file(job_id).write_text("{ this is not json")
    result = cleanup.purge(older_than_days=0.0, force=True)
    assert why(result, job_id) == "no readable state.json"
    assert paths.job_dir(job_id).is_dir()


# -- outputs ------------------------------------------------------------------


def test_unconfirmed_outputs_keep_a_mirrored_job(gpuc_home: Path) -> None:
    job_id = make_job(outputs=True, outputs_synced=False)
    result = cleanup.purge(older_than_days=7.0)
    assert result.purged == []
    assert why(result, job_id) == "outputs not confirmed uploaded"
    assert paths.job_dir(job_id).is_dir()


def test_a_job_that_never_wrote_its_outputs_has_nothing_to_lose(gpuc_home: Path) -> None:
    """Declaring `outputs:` is not producing one. A job that died in its GPU
    preflight or its setup never wrote the path, so calling it unconfirmed both
    misreports it and keeps its dir forever."""
    job_id = make_job(outputs=True, outputs_synced=False, produced=False)
    assert cleanup.outputs_confirmed(job_id, jobs.read_state(job_id)) == (True, None)
    result = cleanup.purge(older_than_days=7.0)
    assert [c.job_id for c in result.purged] == [job_id]
    assert not any(c.forced for c in result.purged)


def test_an_output_dir_holding_only_the_checkout_is_not_a_lost_result(gpuc_home: Path) -> None:
    job_id = make_job(outputs=True, outputs_synced=False)
    baseline.capture(jobs.read_spec(job_id), paths.workdir(job_id), job_id)
    assert cleanup.outputs_confirmed(job_id, jobs.read_state(job_id)) == (True, None)

    (paths.workdir(job_id) / "results" / "new.pt").write_bytes(b"z")
    confirmed, why_not = cleanup.outputs_confirmed(job_id, jobs.read_state(job_id))
    assert not confirmed and why_not == "outputs not confirmed uploaded"


def test_confirmed_outputs_allow_the_purge(gpuc_home: Path) -> None:
    job_id = make_job(outputs=True, outputs_synced=True)
    result = cleanup.purge(older_than_days=7.0)
    assert [c.job_id for c in result.purged] == [job_id]


def test_outputs_lost_is_named_in_the_reason(gpuc_home: Path) -> None:
    job_id = make_job(outputs=True)
    jobs.update_state(job_id, outputs_lost=True)
    assert "outputs_lost" in why(cleanup.purge(older_than_days=7.0), job_id)


def test_a_job_whose_workdir_is_already_gone_has_no_outputs_left_to_lose(
    gpuc_home: Path,
) -> None:
    job_id = make_job(outputs=True, outputs_synced=False, workdir=False)
    assert [c.job_id for c in cleanup.purge(older_than_days=7.0).purged] == [job_id]


def test_a_spec_that_declares_no_outputs_needs_no_confirmation(gpuc_home: Path) -> None:
    job_id = make_job(outputs=False)
    assert [c.job_id for c in cleanup.purge(older_than_days=7.0).purged] == [job_id]


def test_an_unreadable_spec_fails_closed_on_outputs(gpuc_home: Path) -> None:
    job_id = make_job(outputs=True)
    paths.spec_file(job_id).unlink()
    assert "spec.json is unreadable" in why(cleanup.purge(older_than_days=7.0), job_id)
    assert paths.job_dir(job_id).is_dir()


# -- what a purge takes with it -----------------------------------------------


def test_a_dry_run_deletes_nothing(gpuc_home: Path) -> None:
    job_id = make_job()
    result = cleanup.purge(older_than_days=7.0, dry_run=True)
    assert [c.job_id for c in result.purged] == [job_id]
    assert result.freed_bytes > 0
    assert paths.job_dir(job_id).is_dir()


def test_a_dry_run_does_not_count_a_purged_workdir_twice(gpuc_home: Path) -> None:
    job_id = make_job()
    result = cleanup.purge(older_than_days=7.0, dry_run=True)
    assert [c.job_id for c in result.removed] == []
    assert result.freed_bytes == result.purged[0].bytes
    assert job_id


def test_purge_implies_the_workdir_clean_for_jobs_it_keeps(gpuc_home: Path) -> None:
    kept = make_job(meta_synced=False)
    result = cleanup.purge(older_than_days=7.0)
    assert [c.job_id for c in result.removed] == [kept]
    assert not paths.workdir(kept).exists()
    assert paths.state_file(kept).exists()
    assert jobs.read_state(kept).workdir_removed is True


def test_a_stray_queue_marker_and_secrets_file_go_with_the_job(gpuc_home: Path) -> None:
    job_id = make_job()
    (paths.queue_dir() / queue.marker_name(50, job_id)).touch()
    paths.job_env_file(job_id).write_text("SECRET=1\n")
    cleanup.purge(older_than_days=7.0)
    assert queue.find_marker(job_id) is None
    assert not paths.job_env_file(job_id).exists()


def test_only_restricts_what_may_be_purged_but_not_the_sweep(gpuc_home: Path) -> None:
    keep = make_job()
    go = make_job()
    result = cleanup.purge(older_than_days=7.0, only=[go])
    assert [c.job_id for c in result.purged] == [go]
    # The other job dir survives, but the sweep still reclaimed its workdir.
    assert paths.state_file(keep).exists()
    assert not paths.workdir(keep).exists()


def test_only_nothing_purges_nothing(gpuc_home: Path) -> None:
    job_id = make_job()
    result = cleanup.purge(older_than_days=7.0, only=[])
    assert result.purged == []
    assert paths.state_file(job_id).exists()


# -- the host CLI -------------------------------------------------------------


def test_host_cli_purge_prints_json(gpuc_home: Path, capsys: pytest.CaptureFixture[str]) -> None:
    job_id = make_job()
    assert host_cli.main(["purge", "--older-than", "7", "--dry-run"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["dry_run"] is True
    assert [job["job_id"] for job in payload["purged"]] == [job_id]
    assert payload["s3_prefix"] is None


def test_host_cli_purge_defaults_to_a_week(
    gpuc_home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    make_job(days_old=3.0)
    assert host_cli.main(["purge", "--dry-run"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["purged"] == []


def test_host_cli_purge_only_empty_means_none(
    gpuc_home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    make_job()
    assert host_cli.main(["purge", "--older-than", "0", "--only", ""]) == 0
    assert json.loads(capsys.readouterr().out)["purged"] == []
