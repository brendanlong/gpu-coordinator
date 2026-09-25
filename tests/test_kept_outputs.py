"""Kept outputs: an `outputs:` entry with no `s3` or `hf` stays on the host,
where the job wrote it, while every sweep takes the checkout around it.

Only a purge removes a kept output, and only when a person forces it: the
host holds the one copy, and nothing is ever deleted automatically that is
not confirmed somewhere else."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from gpuc.control.submit import SubmitError, check_kept_allowed, check_kept_path
from gpuc.host import baseline, cleanup, fetch, jobs, paths, queue
from gpuc.host.cleanup import Evidence
from tests.conftest import make_spec


def finished_with(
    outputs: list[dict[str, Any]],
    checkout: dict[str, str],
    written: dict[str, str],
    *,
    status: str = "succeeded",
    ran: bool = True,
    cleanup_policy: str = "on_success",
) -> str:
    """A finished job whose workdir holds `checkout` from before `setup` and
    `written` from its run."""
    spec = make_spec(outputs=outputs, cleanup=cleanup_policy)
    job_id = queue.enqueue(spec)
    workdir = paths.workdir(job_id)
    for rel, text in checkout.items():
        (workdir / rel).parent.mkdir(parents=True, exist_ok=True)
        (workdir / rel).write_text(text)
    baseline.capture(jobs.read_spec(job_id), workdir, job_id)
    for rel, text in written.items():
        (workdir / rel).parent.mkdir(parents=True, exist_ok=True)
        (workdir / rel).write_text(text)
    ended = (datetime.now(UTC) - timedelta(days=30)).isoformat()
    jobs.update_state(job_id, status=status, ended_at=ended, ran=ran)
    return job_id


def tree(job_id: str) -> list[str]:
    workdir = paths.workdir(job_id)
    return sorted(str(p.relative_to(workdir)) for p in workdir.rglob("*") if not p.is_dir())


def test_the_sweep_takes_the_checkout_and_leaves_what_the_job_kept(gpuc_home: Path) -> None:
    job_id = finished_with(
        [{"path": "results"}],
        {"train.py": "code", "results/old.md": "checked in", ".venv/lib": "torch"},
        {"results/model.pt": "weights"},
    )
    result = cleanup.clean(all_finished=True, evidence=Evidence(automatic=True))
    assert [c.job_id for c in result.removed] == [job_id]
    # The whole kept path stays as it is; fetch still leaves out the old file.
    assert tree(job_id) == ["results/model.pt", "results/old.md"]
    state = jobs.read_state(job_id)
    assert state.checkout_removed_at is not None and state.workdir_bytes == 0
    assert not cleanup.has_checkout(job_id)
    assert [f["path"] for f in fetch.listing(job_id, jobs.read_spec(job_id), [])["files"]] == [
        "results/model.pt"
    ]
    # Nothing is left for another sweep, and asking by id says why.
    again = cleanup.clean(only=[job_id])
    assert again.removed == []
    assert "only kept outputs remain" in again.skipped[0].why


def test_a_kept_file_and_a_nested_path(gpuc_home: Path) -> None:
    job_id = finished_with(
        [{"path": "model.pt"}, {"path": "runs/a/results"}],
        {"runs/a/config.yaml": "x", "runs/b/config.yaml": "y"},
        {"model.pt": "w", "runs/a/results/eval.json": "{}", "runs/a/scratch": "tmp"},
    )
    cleanup.clean(all_finished=True)
    assert tree(job_id) == ["model.pt", "runs/a/results/eval.json"]


def test_a_kept_path_holding_only_the_checkout_keeps_nothing(gpuc_home: Path) -> None:
    job_id = finished_with([{"path": "results"}], {"results/old.md": "x"}, {})
    cleanup.clean(all_finished=True)
    assert not paths.workdir(job_id).exists()


def test_a_job_whose_main_never_started_keeps_nothing(gpuc_home: Path) -> None:
    job_id = finished_with(
        [{"path": "results"}], {}, {"results/partial": "x"}, status="cancelled", ran=False
    )
    cleanup.clean(all_finished=True)
    assert not paths.workdir(job_id).exists()


def test_the_runners_cleanup_policy_keeps_them_too(gpuc_home: Path) -> None:
    job_id = finished_with([{"path": "results"}], {"a.py": "x"}, {"results/r": "1"})
    state = jobs.read_state(job_id)
    assert cleanup.may_delete(job_id, state, cleanup.WORKDIR, Evidence(policy=True)) is None
    cleanup.remove_workdir(job_id)
    assert tree(job_id) == ["results/r"]


def test_a_purge_refuses_kept_outputs_unless_forced(gpuc_home: Path) -> None:
    job_id = finished_with([{"path": "results"}], {}, {"results/r": "1"})
    jobs.update_state(job_id, uploads=[jobs.Upload(to="s3://b/m", ok_at="2026-01-01T00:00:00")])
    refused = cleanup.purge(only=[job_id])
    assert refused.purged == []
    assert "keeps outputs on this host: results" in refused.purge_skipped[0].why
    forced = cleanup.purge(only=[job_id], evidence=Evidence(force=True))
    assert [c.job_id for c in forced.purged] == [job_id]
    assert forced.purged[0].forced
    assert not paths.job_dir(job_id).exists()


def test_status_counts_a_swept_job_as_holding_no_workdir(gpuc_home: Path) -> None:
    job_id = finished_with([{"path": "results"}], {"a.py": "x"}, {"results/r": "1"})
    cleanup.clean(all_finished=True)
    state = jobs.read_state(job_id)
    assert cleanup.reported_workdir_bytes(job_id, state) == 0
    assert cleanup.kept_outputs(job_id) == ["results"]


@pytest.mark.parametrize("path", ["results", "a/b", "model.pt", "{job_id}/out"])
def test_a_kept_output_inside_the_workdir_is_accepted(path: str) -> None:
    check_kept_path(path)


@pytest.mark.parametrize("path", [".", "./", "", "/abs", "../x", "a/../../x"])
def test_a_kept_output_must_name_something_inside_the_workdir(path: str) -> None:
    with pytest.raises(SubmitError, match="inside the workdir"):
        check_kept_path(path)


def test_a_rental_refuses_kept_outputs_and_takes_the_rest() -> None:
    spec = make_spec(outputs=[{"path": "results"}, {"path": "ckpt", "s3": "s3://b/{job_id}"}])
    with pytest.raises(SubmitError, match="lost with it: results"):
        check_kept_allowed(spec, "host pod", ephemeral=True)
    check_kept_allowed(spec, "host box", ephemeral=False)
    uploaded = make_spec(outputs=[{"path": "ckpt", "s3": "s3://b/{job_id}"}])
    check_kept_allowed(uploaded, "host pod", ephemeral=True)
