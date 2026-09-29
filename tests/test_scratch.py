"""`GPUC_SCRATCH_DIR`: workdirs off gpuc home, and the archived checkout that
lets a job survive scratch being wiped under it."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from gpuc.control.submit import SubmitError, check_kept_allowed
from gpuc.host import checkout, cleanup, health, jobs, paths, queue
from tests.conftest import fake_smi, make_spec
from tests.test_health import fast_downloader
from tests.test_runner import log_of, run


@pytest.fixture
def scratch(gpuc_home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "scratch"
    monkeypatch.setenv("GPUC_SCRATCH_DIR", str(root))
    return root


def submitted(**overrides: object) -> str:
    """A job as `gpuc submit` leaves it: a checkout rsynced under `incoming/`,
    then the enqueue."""
    spec = make_spec(**overrides)
    staged = paths.incoming_job_dir(spec.job_id) / "workdir"
    (staged / "pkg").mkdir(parents=True)
    (staged / "pkg" / "train.py").write_text("print('trained')\n")
    (staged / "run.sh").write_text("#!/bin/sh\necho ran\n")
    (staged / "run.sh").chmod(0o755)
    os.symlink("pkg/train.py", staged / "link.py")
    return queue.enqueue(spec)


def test_without_scratch_the_workdir_is_in_the_job_dir_and_nothing_is_archived(
    gpuc_home: Path,
) -> None:
    job_id = submitted()
    assert paths.workdir(job_id) == paths.job_dir(job_id) / "workdir"
    assert (paths.workdir(job_id) / "pkg" / "train.py").is_file()
    assert not paths.checkout_archive(job_id).exists()


def test_with_scratch_the_checkout_is_archived_and_the_workdir_is_on_scratch(
    scratch: Path,
) -> None:
    job_id = submitted()
    assert paths.checkout_archive(job_id).is_file()
    assert not (paths.job_dir(job_id) / "workdir").exists()
    assert paths.workdir(job_id) == scratch / job_id
    assert not paths.workdir(job_id).exists()


def test_a_workdir_already_in_the_job_dir_is_still_found_once_scratch_is_set(
    gpuc_home: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    job_id = submitted()
    monkeypatch.setenv("GPUC_SCRATCH_DIR", str(tmp_path / "scratch"))
    assert paths.workdir(job_id) == paths.job_dir(job_id) / "workdir"


def test_the_runner_unpacks_the_checkout_as_submitted(scratch: Path) -> None:
    job_id = submitted(command="./run.sh && cat link.py", cleanup="never")
    assert run(job_id) == 0
    workdir = scratch / job_id
    assert (workdir / "link.py").is_symlink()
    assert os.access(workdir / "run.sh", os.X_OK)
    assert "ran" in log_of(job_id)
    assert "print('trained')" in log_of(job_id)


def test_a_wiped_scratch_is_unpacked_again_for_the_next_attempt(scratch: Path) -> None:
    job_id = submitted(
        command="test ! -e left-by-attempt-1 && touch left-by-attempt-1", cleanup="never"
    )
    assert checkout.restore(job_id) is None
    shutil.rmtree(scratch)
    assert run(job_id) == 0
    assert (scratch / job_id / "left-by-attempt-1").exists()


def test_a_preempted_job_keeps_the_workdir_its_last_attempt_left(scratch: Path) -> None:
    job_id = submitted()
    assert checkout.restore(job_id) is None
    (paths.workdir(job_id) / "checkpoint").write_text("step 10\n")
    assert checkout.restore(job_id) is None
    assert (paths.workdir(job_id) / "checkpoint").read_text() == "step 10\n"


def test_no_workdir_and_no_archive_on_scratch_fails_the_job_checkout_lost(
    scratch: Path,
) -> None:
    job_id = submitted()
    paths.checkout_archive(job_id).unlink()
    assert run(job_id) != 0
    state = jobs.read_state(job_id)
    assert (state.status, state.reason, state.ran) == ("failed", "checkout-lost", False)
    assert "checkout lost" in log_of(job_id)


def test_the_archive_is_the_checkout_until_a_sweep_takes_it(scratch: Path) -> None:
    job_id = submitted(cleanup="never")
    assert run(job_id) == 0
    shutil.rmtree(scratch)
    assert cleanup.has_checkout(job_id)
    assert cleanup.workdir_size(job_id)
    assert cleanup.remove_workdir(job_id) > 0
    assert not paths.checkout_archive(job_id).exists()
    assert not cleanup.has_checkout(job_id)


def test_a_policy_cleanup_takes_the_archive_with_the_workdir(scratch: Path) -> None:
    job_id = submitted(cleanup="always")
    assert run(job_id) == 0
    assert not (scratch / job_id).exists()
    assert not paths.checkout_archive(job_id).exists()


def test_purging_a_job_removes_its_workdir_on_scratch(scratch: Path) -> None:
    job_id = submitted(cleanup="never")
    assert run(job_id) == 0
    assert cleanup.job_dir_bytes(job_id) > cleanup.reclaimable_bytes(paths.job_dir(job_id))
    cleanup.remove_job_dir(job_id)
    assert not (scratch / job_id).exists()
    assert not paths.job_dir(job_id).exists()


def test_health_checks_scratchs_disk_and_puts_the_uv_cache_beside_it(scratch: Path) -> None:
    """Read from the host's config, as a health check over ssh gets it."""
    config = jobs.read_config()
    config.env["GPUC_SCRATCH_DIR"] = str(scratch)
    jobs.write_config(config)
    report = health.run_checks(
        smi=fake_smi(), downloader=fast_downloader, min_free_gb=0.0, url="http://x"
    )
    checks = {check["name"]: check for check in report["checks"]}
    assert str(scratch) in checks["scratch_disk"]["detail"]
    assert health.uv_cache_placement(config)["gpuc_home"] == str(scratch)


def test_kept_outputs_are_refused_on_a_host_with_scratch() -> None:
    spec = make_spec(outputs=[{"path": "results"}])
    with pytest.raises(SubmitError, match="scratch"):
        check_kept_allowed(spec, "host h", ephemeral=False, scratch=True)
    check_kept_allowed(spec, "host h", ephemeral=False, scratch=False)
