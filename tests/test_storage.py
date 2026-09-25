"""The host's shared storage: the data directory jobs keep things in, and the
Hugging Face cache, and the only ways either is ever emptied."""

from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path

import pytest

from gpuc.control.bootstrap import storage_line
from gpuc.host import health, jobs, paths, runner, storage
from gpuc.host.jobs import HostConfig
from tests.conftest import FAKE_GPUS, make_spec


def test_the_data_dir_is_in_gpuc_home_unless_the_host_names_one(gpuc_home: Path) -> None:
    assert paths.data_dir({}) == gpuc_home / "data"
    assert paths.data_dir({"GPUC_DATA_DIR": "/big/data"}) == Path("/big/data")


def test_a_job_is_told_where_the_data_dir_is(
    gpuc_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = runner.build_env(make_spec(job_id="j1"), [FAKE_GPUS[0]])
    assert env["GPUC_DATA_DIR"] == str(gpuc_home / "data")
    # The host's `env` reaches the runner through the dispatcher's environment.
    monkeypatch.setenv("GPUC_DATA_DIR", "/big/data")
    env = runner.build_env(make_spec(job_id="j1"), [FAKE_GPUS[0]])
    assert env["GPUC_DATA_DIR"] == "/big/data"


def test_a_hosts_data_dir_survives_an_env_that_does_not_name_it() -> None:
    kept = jobs.sticky_env({"GPUC_DATA_DIR": "/big/data", "HF_TOKEN": "x"}, {"A": "b"})
    assert kept == {"A": "b", "GPUC_DATA_DIR": "/big/data"}


def test_the_hf_cache_follows_huggingface_hubs_own_precedence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", "/xdg")
    assert health.hf_hub_cache_dir() == Path("/xdg/huggingface/hub")
    assert health.hf_hub_cache_dir(HostConfig(env={"HF_HOME": "/hf"})) == Path("/hf/hub")
    both = HostConfig(env={"HF_HOME": "/hf", "HF_HUB_CACHE": "/hub"})
    assert health.hf_hub_cache_dir(both) == Path("/hub")


def test_health_reports_where_the_shared_bytes_are(gpuc_home: Path) -> None:
    (gpuc_home / "data" / "set").mkdir(parents=True)
    (gpuc_home / "data" / "set" / "a.bin").write_bytes(b"x" * 4096)
    check = health.check_size("data_dir", paths.data_dir())
    assert check.ok and not check.warn
    assert check.path == str(gpuc_home / "data")
    assert isinstance(check.value, int) and check.value >= 4096


def test_bootstrap_prints_each_shared_directory_and_its_size() -> None:
    checks = [
        {"name": "disk", "ok": True, "detail": "plenty", "value": 90.0},
        {"name": "uv_cache", "ok": True, "detail": "...", "value": 2**30, "path": "/c/uv"},
        {"name": "data_dir", "ok": True, "detail": "...", "value": 0, "path": "/g/data"},
    ]
    assert (
        storage_line({"checks": checks}) == "storage: uv cache /c/uv (1.0 GiB); data /g/data (0 B)"
    )
    # A host on an earlier build names no paths, and gets no line.
    assert storage_line({"checks": checks[:1]}) is None


def test_data_remove_deletes_inside_the_data_dir_and_nothing_else(tmp_path: Path) -> None:
    root = tmp_path / "data"
    (root / "set").mkdir(parents=True)
    (root / "set" / "a").write_text("a")
    (root / "one.txt").write_text("1")
    outside = tmp_path / "outside.txt"
    outside.write_text("keep")
    os.symlink(outside, root / "link")

    report = storage.remove_data(
        ["set", "one.txt", "missing", "../outside.txt", "/etc", ".", "link/../.."], root
    )
    assert [r["path"] for r in report["removed"]] == ["set", "one.txt"]
    assert len(report["errors"]) == 5
    assert not (root / "set").exists() and not (root / "one.txt").exists()
    assert outside.read_text() == "keep" and root.is_dir()


def test_data_remove_takes_a_symlink_itself_not_what_it_points_at(tmp_path: Path) -> None:
    root = tmp_path / "data"
    root.mkdir()
    target = root / "real"
    target.mkdir()
    (root / "alias").symlink_to(target)
    report = storage.remove_data(["alias"], root)
    assert report["errors"] == []
    assert not (root / "alias").exists() and target.is_dir()


def test_hf_cache_prune_runs_hf_on_the_jobs_cache(
    gpuc_home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hub = tmp_path / "hf" / "hub"
    hub.mkdir(parents=True)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    called = tmp_path / "called"
    (bin_dir / "hf").write_text(f'#!/bin/sh\necho "$@" > {called}\n')
    (bin_dir / "hf").chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    jobs.write_config(HostConfig(host="h", gpus=list(FAKE_GPUS), env={"HF_HOME": str(hub.parent)}))

    report = storage.prune_hf_cache()
    assert report["errors"] == []
    assert report["cache_dir"] == str(hub)
    # `prune`, never `rm`: every revision a job may ask for again stays.
    assert called.read_text().split() == ["cache", "prune", "--yes", "--cache-dir", str(hub)]


def test_hf_cache_prune_on_a_host_with_no_cache_does_nothing(
    gpuc_home: Path, tmp_path: Path
) -> None:
    jobs.write_config(HostConfig(host="h", gpus=list(FAKE_GPUS), env={"HF_HOME": str(tmp_path)}))
    report = storage.prune_hf_cache()
    assert report == {
        "cache_dir": str(tmp_path / "hub"),
        "before_bytes": 0,
        "after_bytes": 0,
        "errors": [],
    }


def test_a_directory_too_big_to_measure_in_time_is_said_so_not_failed(tmp_path: Path) -> None:
    (tmp_path / "set").mkdir()
    (tmp_path / "set" / "a").write_text("a")
    check = health.check_size("data_dir", tmp_path, budget_s=-1.0)
    assert check.ok and check.value is None
    assert "too large to measure" in check.detail
    assert storage_line({"checks": [asdict(check)]}) == f"storage: data {tmp_path} (?)"
