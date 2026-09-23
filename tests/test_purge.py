"""`purge`: deleting a whole job dir, and every reason not to.

`clean` can always be undone by re-running the job; `purge` deletes the record
that it ever ran, so almost all of these tests are about what it refuses.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from gpuc.host import __main__ as host_cli
from gpuc.host import baseline, cleanup, destinations, jobs, paths, queue
from gpuc.host.jobs import HostConfig
from tests.conftest import make_spec

PREFIX = "s3://bucket/gpuc/test-host"
OUTPUT = {"path": "results", "s3": "s3://bucket/exp/{job_id}"}


def make_job(
    *,
    status: str = "succeeded",
    days_old: float = 30.0,
    mirrored: bool = True,
    outputs: bool = False,
    output: dict[str, Any] = OUTPUT,
    outputs_uploaded: bool = False,
    produced: bool = True,
    workdir: bool = True,
) -> str:
    spec = make_spec(outputs=[output] if outputs else [])
    job_id = queue.enqueue(spec)
    ended = datetime.now(UTC) - timedelta(days=days_old)
    fields: dict[str, Any] = {"status": status}
    if status in jobs.FINISHED_STATUSES:
        fields["ended_at"] = ended.isoformat()
    uploads: list[jobs.Upload] = []
    if mirrored:
        uploads.append(jobs.Upload(to=f"{PREFIX}/jobs/{job_id}", ok_at=ended.isoformat()))
    if outputs_uploaded:
        uploads += [
            jobs.Upload(to=destination.uri, output=output.path, ok_at=ended.isoformat())
            for output in spec.outputs
            for destination in destinations.of(output, job_id)
        ]
    jobs.update_state(job_id, uploads=uploads, **fields)
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


def pending(job_id: str) -> str | None:
    return cleanup.outputs_pending(job_id, jobs.read_spec(job_id), jobs.read_state(job_id))


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
    job_id = make_job(status=status, mirrored=True)
    result = cleanup.purge(older_than_days=0.0, evidence=cleanup.Evidence(force=force))
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
    job_id = make_job(mirrored=False)
    result = cleanup.purge(older_than_days=7.0)
    assert why(result, job_id) == "not backed up: no s3_prefix on this host"
    assert paths.job_dir(job_id).is_dir()


def test_a_job_with_a_prefix_but_no_record_says_the_upload_failed(gpuc_home: Path) -> None:
    with_prefix()
    job_id = make_job(mirrored=False)
    result = cleanup.purge(older_than_days=7.0)
    assert why(result, job_id) == "not backed up: final upload failed"


def test_force_purges_an_unmirrored_job_and_says_so(gpuc_home: Path) -> None:
    job_id = make_job(mirrored=False)
    result = cleanup.purge(older_than_days=7.0, evidence=cleanup.Evidence(force=True))
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
    result = cleanup.purge(older_than_days=0.0, evidence=cleanup.Evidence(force=True))
    assert why(result, job_id) == "no readable state.json"
    assert paths.job_dir(job_id).is_dir()


# -- outputs ------------------------------------------------------------------


def test_unconfirmed_outputs_keep_a_mirrored_job(gpuc_home: Path) -> None:
    job_id = make_job(outputs=True, outputs_uploaded=False)
    result = cleanup.purge(older_than_days=7.0)
    assert result.purged == []
    assert why(result, job_id) == "outputs not confirmed uploaded"
    assert paths.job_dir(job_id).is_dir()


def test_a_job_that_never_wrote_its_outputs_has_nothing_to_lose(gpuc_home: Path) -> None:
    """Declaring `outputs:` is not producing one. A job that died in its GPU
    preflight or its setup never wrote the path, so calling it unconfirmed both
    misreports it and keeps its dir forever."""
    job_id = make_job(outputs=True, outputs_uploaded=False, produced=False)
    assert pending(job_id) is None
    result = cleanup.purge(older_than_days=7.0)
    assert [c.job_id for c in result.purged] == [job_id]
    assert not any(c.forced for c in result.purged)


def test_an_output_dir_holding_only_the_checkout_is_not_a_lost_result(gpuc_home: Path) -> None:
    job_id = make_job(outputs=True, outputs_uploaded=False)
    baseline.capture(jobs.read_spec(job_id), paths.workdir(job_id), job_id)
    assert pending(job_id) is None

    (paths.workdir(job_id) / "results" / "new.pt").write_bytes(b"z")
    assert pending(job_id) == "outputs not confirmed uploaded"


def test_a_captured_baseline_with_nothing_under_it_is_not_a_lost_result(
    gpuc_home: Path,
) -> None:
    """The died-in-setup shape: the baseline was taken, and the job then wrote
    nothing. Distinct from never having taken one at all."""
    job_id = make_job(outputs=True, outputs_uploaded=False, produced=False)
    baseline.capture(jobs.read_spec(job_id), paths.workdir(job_id), job_id)
    assert pending(job_id) is None


def test_an_output_path_that_cannot_be_resolved_is_never_purged(gpuc_home: Path) -> None:
    """Nothing validates `outputs.path` at submit, so `{step}` reaches here and
    `.format(job_id=...)` raises. Answering "produced nothing" would delete the
    job dir; answering at all with a traceback would take `gpuc status` for the
    whole host down with it."""
    job_id = make_job(outputs=True, outputs_uploaded=False, produced=False)
    spec = json.loads(paths.spec_file(job_id).read_text())
    spec["outputs"] = [{"path": "results/{step}", "s3": "s3://bucket/x"}]
    paths.spec_file(job_id).write_text(json.dumps(spec))

    assert pending(job_id)
    result = cleanup.purge(older_than_days=7.0)
    assert result.purged == []
    assert why(result, job_id) == "outputs not confirmed uploaded"


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads a 000 directory anyway")
def test_an_unreadable_output_dir_is_never_purged(gpuc_home: Path) -> None:
    job_id = make_job(outputs=True, outputs_uploaded=False)
    results = paths.workdir(job_id) / "results"
    os.chmod(results, 0o000)
    try:
        assert pending(job_id)
        result = cleanup.purge(older_than_days=7.0)
        assert result.purged == []
    finally:
        os.chmod(results, 0o755)


def test_outputs_reached_through_a_symlink_are_never_purged(gpuc_home: Path) -> None:
    """`rglob` does not descend a directory symlink and `aws s3 sync` follows
    one, so `latest -> checkpoint-9/` would read as an empty output dir while
    holding the whole run."""
    job_id = make_job(outputs=True, outputs_uploaded=False, produced=False)
    workdir = paths.workdir(job_id)
    (workdir / "checkpoint-9").mkdir()
    (workdir / "checkpoint-9" / "model.pt").write_bytes(b"w" * 4096)
    (workdir / "results").mkdir(exist_ok=True)
    (workdir / "results" / "latest").symlink_to("../checkpoint-9")

    assert pending(job_id)
    result = cleanup.purge(older_than_days=7.0)
    assert result.purged == []


def test_confirmed_outputs_allow_the_purge(gpuc_home: Path) -> None:
    job_id = make_job(outputs=True, outputs_uploaded=True)
    result = cleanup.purge(older_than_days=7.0)
    assert [c.job_id for c in result.purged] == [job_id]


def test_one_failing_destination_of_two_leaves_the_outputs_unconfirmed(gpuc_home: Path) -> None:
    both = {**OUTPUT, "hf": "someone/exp", "hf_path": "{job_id}"}
    job_id = make_job(outputs=True, output=both, outputs_uploaded=True)
    _, hf = destinations.of(jobs.read_spec(job_id).outputs[0], job_id)
    assert pending(job_id) is None

    jobs.record_upload(job_id, hf.uri, "results", error="403 Forbidden")
    assert pending(job_id) == "outputs not confirmed uploaded"
    assert [c.job_id for c in cleanup.purge(older_than_days=7.0).purged] == []


def test_outputs_lost_is_named_in_the_reason(gpuc_home: Path) -> None:
    job_id = make_job(outputs=True)
    jobs.update_state(job_id, outputs_lost=True)
    assert "outputs_lost" in why(cleanup.purge(older_than_days=7.0), job_id)


def test_a_job_whose_workdir_is_already_gone_has_no_outputs_left_to_lose(
    gpuc_home: Path,
) -> None:
    job_id = make_job(outputs=True, outputs_uploaded=False, workdir=False)
    assert [c.job_id for c in cleanup.purge(older_than_days=7.0).purged] == [job_id]


def test_a_spec_that_declares_no_outputs_needs_no_confirmation(gpuc_home: Path) -> None:
    job_id = make_job(outputs=False)
    assert [c.job_id for c in cleanup.purge(older_than_days=7.0).purged] == [job_id]


def test_an_unreadable_spec_fails_closed_on_outputs(gpuc_home: Path) -> None:
    job_id = make_job(outputs=True)
    paths.spec_file(job_id).unlink()
    assert why(cleanup.purge(older_than_days=7.0), job_id) == "no readable spec.json"
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
    kept = make_job(mirrored=False)
    result = cleanup.purge(older_than_days=7.0)
    assert [c.job_id for c in result.removed] == [kept]
    assert not paths.workdir(kept).exists()
    assert paths.state_file(kept).exists()
    assert jobs.read_state(kept).workdir_bytes == 0


def test_the_secrets_file_goes_with_the_purged_job(gpuc_home: Path) -> None:
    """It lives outside the job dir, in `secrets/`, so removing the dir alone
    would leave a job's credentials on the host after its record had gone."""
    job_id = make_job()
    paths.job_env_file(job_id).write_text("SECRET=1\n")
    cleanup.purge(older_than_days=7.0)
    assert not paths.job_dir(job_id).exists()
    assert not paths.job_env_file(job_id).exists()


def test_only_scopes_both_the_purge_and_the_implied_sweep(gpuc_home: Path) -> None:
    """What `gpuc clean --purge --only` sends: purge one job, sweep no others."""
    keep = make_job()
    go = make_job()
    result = cleanup.purge(older_than_days=7.0, only=[go])
    assert [c.job_id for c in result.purged] == [go]
    assert result.removed == []
    assert paths.state_file(keep).exists()
    assert paths.workdir(keep).is_dir()


def test_only_nothing_purges_nothing(gpuc_home: Path) -> None:
    job_id = make_job()
    result = cleanup.purge(older_than_days=7.0, only=[])
    assert result.purged == []
    assert paths.state_file(job_id).exists()


def test_a_named_job_the_caller_could_not_verify_is_swept_but_not_purged(
    gpuc_home: Path,
) -> None:
    """The `--verify` shape: purge what the mirror confirmed, sweep what was asked."""
    unverified = make_job()
    result = cleanup.purge(
        older_than_days=7.0, only=[unverified], evidence=cleanup.Evidence(verified=frozenset())
    )
    assert result.purged == []
    assert why(result, unverified) == "not backed up: the mirror has no log for it"
    assert [c.job_id for c in result.removed] == [unverified]
    assert paths.state_file(unverified).exists()
    assert not paths.workdir(unverified).exists()


# -- the host CLI -------------------------------------------------------------


def test_host_cli_purge_prints_json(gpuc_home: Path, capsys: pytest.CaptureFixture[str]) -> None:
    job_id = make_job()
    assert host_cli.main(["purge", "--older-than", "7", "--dry-run"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["dry_run"] is True
    assert [job["job_id"] for job in payload["purged"]] == [job_id]
    assert payload["purged"][0]["mirror"] == f"{PREFIX}/jobs/{job_id}"
    assert payload["purged"][0]["mirrored_at"]
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


def test_host_cli_purge_only_scopes_the_sweep(
    gpuc_home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    keep = make_job()
    go = make_job()
    assert host_cli.main(["purge", "--older-than", "0", "--only", go]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert [c["job_id"] for c in payload["purged"]] == [go]
    assert payload["removed"] == []
    assert paths.workdir(keep).is_dir()


def test_host_cli_purge_verified_empty_purges_nothing_and_says_the_mirror_has_no_log(
    gpuc_home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The host's own mirror record says yes; the caller checked and found
    nothing, and the caller's answer is the one that counts."""
    job_id = make_job(mirrored=True)
    assert host_cli.main(["purge", "--older-than", "0", "--verified", ""]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["purged"] == []
    assert [(s["job_id"], s["why"]) for s in payload["purge_skipped"]] == [
        (job_id, "not backed up: the mirror has no log for it")
    ]
    assert paths.state_file(job_id).exists()


def test_host_cli_purge_verified_narrows_what_the_records_already_allow(
    gpuc_home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A job needs both: the caller's listing and the host's own record of a
    successful final upload. The listing cannot vouch for a job the host
    never recorded, since every periodic tick mirrors the log too."""
    both = make_job(mirrored=True)
    recorded_only = make_job(mirrored=True)
    listed_only = make_job(mirrored=False)
    argv = ["purge", "--older-than", "0", "--verified", f"{both},{listed_only}"]
    assert host_cli.main(argv) == 0
    payload = json.loads(capsys.readouterr().out)
    assert [c["job_id"] for c in payload["purged"]] == [both]
    assert {s["job_id"]: s["why"] for s in payload["purge_skipped"]} == {
        recorded_only: "not backed up: the mirror has no log for it",
        listed_only: "not backed up: no s3_prefix on this host",
    }


def test_host_cli_purge_reads_a_long_verified_list_from_a_file(
    gpuc_home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    job_id = make_job(mirrored=True)
    listed = gpuc_home / "incoming" / ".verified-1"
    listed.parent.mkdir(parents=True, exist_ok=True)
    listed.write_text(f"{job_id}\n")
    argv = ["purge", "--older-than", "0", "--verified-file", str(listed)]
    assert host_cli.main(argv) == 0
    payload = json.loads(capsys.readouterr().out)
    assert [c["job_id"] for c in payload["purged"]] == [job_id]
    assert not listed.exists()


def test_host_cli_purge_refuses_a_selection_holding_a_job_id_it_does_not_know(
    gpuc_home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """One typo in a pair is a mistyped id far more often than a deliberate
    pair, and the half this would delete does not come back."""
    job_id = make_job()
    selection = f"{job_id},20260101-000000-typo11"
    assert host_cli.main(["purge", "--older-than", "0", "--only", selection]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["errors"] == [
        "20260101-000000-typo11: no job with that id on this host",
        "refused the whole selection: nothing was removed",
    ]
    assert payload["purged"] == []
    assert paths.state_file(job_id).exists()
    assert paths.workdir(job_id).is_dir()


def test_naming_a_running_job_neither_purges_it_nor_takes_its_workdir(gpuc_home: Path) -> None:
    job_id = make_job(status="running")
    result = cleanup.purge(
        older_than_days=0.0, only=[job_id], evidence=cleanup.Evidence(force=True)
    )
    assert result.purged == [] and result.removed == []
    assert [s.why for s in result.purge_skipped] == ["status running"]
    assert paths.workdir(job_id).is_dir()


def test_purging_a_named_job_does_not_need_a_usable_ended_at(gpuc_home: Path) -> None:
    """The stuck job somebody names is often one whose state write was cut
    short; a bare `clean --only` would still reclaim its workdir."""
    job_id = make_job()
    jobs.update_state(job_id, ended_at=None)
    result = cleanup.purge(older_than_days=7.0, only=[job_id])
    assert [c.job_id for c in result.purged] == [job_id]
    assert not paths.job_dir(job_id).exists()
