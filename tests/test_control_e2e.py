"""The whole control side against this machine as a `local` host, no GPU needed.

Everything is redirected: GPUC_CONFIG_DIR, GPUC_STATE_DIR and the host's
GPUC_HOME, so the real ~/.gpuc and the real registry are never touched.
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from gpuc.control.cli import main
from gpuc.control.config import load_registry

HEALTH_ARGS = "--min-mbps 0.05 --min-free-gb 1"


def wait_until(predicate: Callable[[], bool], timeout: float, what: str) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(0.2)
    raise AssertionError(f"timed out after {timeout}s waiting for {what}")


def state_of(home: Path, job_id: str) -> dict[str, object]:
    path = home / "jobs" / job_id / "state.json"
    if not path.exists():
        return {}
    try:
        document: dict[str, object] = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}
    return document


def log_tail(home: Path, job_id: str, lines: int = 25) -> str:
    path = home / "jobs" / job_id / "log.txt"
    return "\n".join(path.read_text().splitlines()[-lines:]) if path.exists() else "(no log)"


def wait_for_main_phase(home: Path, job_id: str, timeout: float = 600.0) -> None:
    wait_until(
        lambda: state_of(home, job_id).get("phase") == "main" or finished(home, job_id),
        timeout,
        f"job {job_id} to reach phase=main",
    )
    assert state_of(home, job_id)["status"] == "running", log_tail(home, job_id)


def finished(home: Path, job_id: str) -> bool:
    return state_of(home, job_id).get("status") in ("succeeded", "failed", "cancelled")


@pytest.fixture
def workdir(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    root.mkdir()
    (root / "hello.txt").write_text("hello\n")
    for argv in (
        ["git", "init", "-q"],
        ["git", "config", "user.email", "t@example.com"],
        ["git", "config", "user.name", "t"],
        ["git", "add", "-A"],
        ["git", "commit", "-qm", "init"],
    ):
        subprocess.run(argv, cwd=root, check=True, capture_output=True)
    return root


@pytest.fixture
def bootstrapped_home(control_env: Path, tmp_path: Path) -> Iterator[Path]:
    home = tmp_path / "gpuc-home"
    assert main(["host", "add", "local", "--gpuc-home", str(home)]) == 0
    assert main(["host", "bootstrap", "local", "--health-args", HEALTH_ARGS]) == 0
    yield home
    _stop_dispatcher(home)


def _stop_dispatcher(home: Path) -> None:
    """Leave nothing running on this shared machine: jobs first, then the dispatcher."""
    for state_file in sorted(home.glob("jobs/*/state.json")):
        state = state_of(home, state_file.parent.name)
        pgid = state.get("pgid") if state.get("status") == "running" else None
        runner = state.get("runner_pid") if state.get("status") == "running" else None
        for group in (pgid, runner):
            if isinstance(group, int) and group > 1:
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    os.killpg(group, 9)
    lock = home / "dispatcher.lock"
    if not lock.exists():
        return
    body = lock.read_text().strip()
    if body.isdigit():
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(int(body), 15)


def submit(workdir: Path, document: str, name: str = "job.yaml") -> str:
    """Submit from `workdir` as the user would, and return the new job id."""
    job = workdir / name
    job.write_text(document)
    before = _indexed_job_ids()
    cwd = Path.cwd()
    os.chdir(workdir)
    try:
        assert main(["submit", str(job), "--host", "local"]) == 0
    finally:
        os.chdir(cwd)
    new = _indexed_job_ids() - before
    assert len(new) == 1, f"submit recorded {len(new)} index entries"
    return new.pop()


def _indexed_job_ids() -> set[str]:
    from gpuc.control.s3index import LocalIndex

    return {entry.job_id for entry in LocalIndex().list()}


def test_bootstrap_installs_the_package_and_records_the_interpreter(
    bootstrapped_home: Path,
) -> None:
    home = bootstrapped_home
    assert (home / "pkg/gpuc/host/dispatcher.py").exists()
    config = json.loads((home / "config.json").read_text())
    assert config["host"] == "local"
    assert (home / "secrets").stat().st_mode & 0o777 == 0o700
    entry = load_registry().require("local")
    assert entry.python and Path(entry.python).exists()
    assert entry.bootstrapped_at


def test_bootstrap_is_idempotent(bootstrapped_home: Path) -> None:
    assert main(["host", "bootstrap", "local", "--health-args", HEALTH_ARGS]) == 0


def test_probe_reports_this_machine(
    bootstrapped_home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["host", "probe", "local"]) == 0
    out = capsys.readouterr().out
    assert "host local" in out
    assert "systemd_scope:" in out
    assert "MB/s" in out or "B/s" in out


def test_submit_runs_a_job_and_logs_and_status_find_it(
    bootstrapped_home: Path, workdir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    home = bootstrapped_home
    job_id = submit(workdir, "name: hi\ncommand: cat hello.txt\ngpus: 0\n")
    capsys.readouterr()

    wait_until(lambda: finished(home, job_id), 120, f"job {job_id} to finish")
    state = state_of(home, job_id)
    assert (state["status"], state["exit_code"], state["attempt"]) == ("succeeded", 0, 1)
    assert (home / "jobs" / job_id / "workdir" / "hello.txt").exists()

    assert main(["logs", job_id]) == 0
    assert "hello" in capsys.readouterr().out

    assert main(["status", "--host", "local"]) == 0
    status = capsys.readouterr().out
    assert job_id in status
    assert "succeeded" in status

    assert main(["status", "--suspects"]) == 0
    assert "no suspects" in capsys.readouterr().out


def test_a_running_job_can_be_cancelled(
    bootstrapped_home: Path, workdir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    home = bootstrapped_home
    job_id = submit(workdir, "name: sleepy\ncommand: sleep 300\ngpus: 0\npriority: 50\n")
    capsys.readouterr()

    wait_until(
        lambda: state_of(home, job_id).get("status") == "running", 60, "the job to start running"
    )
    assert main(["cancel", job_id]) == 0
    wait_until(lambda: finished(home, job_id), 120, "the job to be cancelled")
    assert state_of(home, job_id)["status"] == "cancelled"


def test_a_failing_job_keeps_its_exit_code(bootstrapped_home: Path, workdir: Path) -> None:
    home = bootstrapped_home
    job_id = submit(workdir, "name: nope\ncommand: exit 23\ngpus: 0\n")
    wait_until(lambda: finished(home, job_id), 120, "the job to fail")
    state = state_of(home, job_id)
    assert (state["status"], state["exit_code"]) == ("failed", 23)


def test_requeue_resubmits_from_the_s3_spec_with_the_next_attempt(
    bootstrapped_home: Path,
    workdir: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from gpuc.control import config
    from tests.fakes3 import FakeS3Client

    fake = FakeS3Client()
    monkeypatch.setattr("boto3.client", lambda service, **_: fake)
    config.config_file().write_text('s3_bucket = "bkt"\n')

    home = bootstrapped_home
    first = submit(workdir, "name: hi\ncommand: cat hello.txt\ngpus: 0\n")
    wait_until(lambda: finished(home, first), 120, f"job {first} to finish")
    assert f"bkt/gpuc/specs/{first}.json" in fake.objects

    before = _indexed_job_ids()
    cwd = Path.cwd()
    os.chdir(workdir)
    try:
        assert main(["requeue", first, "--host", "local"]) == 0
    finally:
        os.chdir(cwd)
    assert "attempt 2" in capsys.readouterr().out
    second = (_indexed_job_ids() - before).pop()

    wait_until(lambda: finished(home, second), 120, f"job {second} to finish")
    state = state_of(home, second)
    assert (state["status"], state["attempt"]) == ("succeeded", 2)
    assert "hello" in log_tail(home, second)
