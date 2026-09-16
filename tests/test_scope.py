"""cgroup scopes: how a phase is launched, and what a kill actually reaches.

The escape test is deliberately real. A daemonised grandchild surviving a kill
is exactly the failure a mocked test would not notice, so both modes run a job
that double-forks and both are checked against the process table afterwards.
"""

from __future__ import annotations

import base64
import os
import threading
import time
from pathlib import Path

import pytest

from gpuc.control.status import HostView, host_json, job_views
from gpuc.host import jobs, paths, queue, runner, scope
from gpuc.host.runner import RunnerDeps
from tests.conftest import host_entry, make_spec

pytestmark = pytest.mark.usefixtures("gpuc_home")


@pytest.fixture(scope="session")
def systemd_scopes() -> bool:
    """Can this machine make a `systemd-run --user --scope`?

    A fixture, not a module-level `skipif`: `scope.probe()` execs systemd, and
    collecting this file must not run anything on the machine.
    """
    return scope.probe(use_cache=False)


@pytest.fixture
def needs_scopes(systemd_scopes: bool) -> None:
    if not systemd_scopes:
        pytest.skip(
            "no user systemd: `systemd-run --user --scope` is unusable here, so the cgroup "
            "kill path -- the only one that reaps a double-forked grandchild -- was NOT tested"
        )


def test_a_phase_without_a_scope_is_a_plain_pipefail_shell() -> None:
    assert scope.phase_argv("echo hi", None) == ["bash", "-eo", "pipefail", "-c", "echo hi"]


def test_a_scoped_phase_hides_the_script_from_systemds_expansion() -> None:
    argv = scope.phase_argv("echo $$ && echo ${HOME}", "gpuc-j-main.scope")
    assert argv[:9] == [
        "systemd-run",
        "--user",
        "--scope",
        "--collect",
        "--quiet",
        "-p",
        "TimeoutStopSec=15",
        "--unit=gpuc-j-main.scope",
        "--",
    ]
    assert argv[9:12] == ["bash", "-c", 'base64 -d <<<"$1" | bash']
    assert argv[12] == "_"
    # systemd rewrites `$$` and `$VAR` in an ExecStart, so nothing it can see
    # may contain a `$` at all.
    assert "$" not in argv[13]
    assert base64.b64decode(argv[13]).decode() == "set -eo pipefail\necho $$ && echo ${HOME}\n"


def test_the_unit_name_carries_the_job_and_phase() -> None:
    assert scope.unit_name("20260915-120000-abc", "main") == "gpuc-20260915-120000-abc-main.scope"


def test_isolation_honours_what_the_dispatcher_probed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(scope.ISOLATION_ENV, scope.CGROUP)
    assert scope.isolation() == scope.CGROUP
    monkeypatch.setenv(scope.ISOLATION_ENV, scope.PGID)
    assert scope.isolation() == scope.PGID


def test_a_pgid_host_records_its_isolation_in_state() -> None:
    job_id = prepare("true")
    runner.run_job(job_id, RunnerDeps(preflight=False, poll_interval_s=0.02))
    state = jobs.read_state(job_id)
    assert (state.isolation, state.cgroup_unit) == ("pgid", None)


@pytest.mark.usefixtures("needs_scopes")
def test_a_cgroup_host_records_its_unit_while_a_phase_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(scope.ISOLATION_ENV, scope.CGROUP)
    job_id = prepare("sleep 5")
    units: list[str | None] = []
    thread = _run_in_background(job_id)
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline and not units:
        state = jobs.read_state(job_id)
        if state.cgroup_unit:
            units.append(state.cgroup_unit)
        time.sleep(0.1)
    queue.cancel(job_id)
    thread.join(timeout=60)
    state = jobs.read_state(job_id)
    assert units == [scope.unit_name(job_id, "main")]
    assert state.isolation == "cgroup"
    # Cleared once the phase is over, so a stale unit is never stopped later.
    assert state.cgroup_unit is None


# -- the escape case, for real -------------------------------------------------


def prepare(command: str) -> str:
    spec = make_spec(command=command, gpus=0)
    job_id = queue.enqueue(spec)
    queue.remove_marker(job_id)
    jobs.update_state(job_id, status="running")
    return job_id


def _run_in_background(job_id: str) -> threading.Thread:
    thread = threading.Thread(
        target=runner.run_job,
        args=(job_id, RunnerDeps(preflight=False, poll_interval_s=0.05, kill_grace_s=5.0)),
        daemon=True,
    )
    thread.start()
    return thread


def daemonising_job(pid_file: Path) -> str:
    """A grandchild that leaves both the process group and the session."""
    return f"setsid bash -c 'echo $$ > {pid_file}; exec sleep 300' & sleep 120"


def wait_for_pid(pid_file: Path, timeout: float = 30.0) -> int:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        text = pid_file.read_text().strip() if pid_file.exists() else ""
        if text.isdigit():
            return int(text)
        time.sleep(0.1)
    raise AssertionError(f"the job never wrote a pid to {pid_file}")


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def gone_within(pid: int, seconds: float) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if not alive(pid):
            return True
        time.sleep(0.2)
    return not alive(pid)


def reap(pid: int) -> None:
    """Only ever this pid: other people share this machine."""
    if alive(pid):
        os.kill(pid, 9)


@pytest.mark.usefixtures("needs_scopes")
def test_cancel_under_a_cgroup_reaps_a_double_forked_grandchild(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(scope.ISOLATION_ENV, scope.CGROUP)
    pid_file = tmp_path / "daemon.pid"
    job_id = prepare(daemonising_job(pid_file))
    thread = _run_in_background(job_id)
    pid = wait_for_pid(pid_file)
    try:
        queue.cancel(job_id)
        thread.join(timeout=90)
        assert not thread.is_alive()
        assert gone_within(pid, 30.0), f"pid {pid} escaped the cgroup kill"
    finally:
        reap(pid)
    state = jobs.read_state(job_id)
    assert state.status == "cancelled"
    assert f"stopping scope {scope.unit_name(job_id, 'main')}" in paths.log_file(job_id).read_text()


def test_cancel_under_a_process_group_cannot_reach_a_double_forked_grandchild(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fallback's known hole, pinned so the difference stays visible."""
    monkeypatch.setenv(scope.ISOLATION_ENV, scope.PGID)
    pid_file = tmp_path / "daemon.pid"
    job_id = prepare(daemonising_job(pid_file))
    thread = _run_in_background(job_id)
    pid = wait_for_pid(pid_file)
    try:
        queue.cancel(job_id)
        thread.join(timeout=90)
        assert not thread.is_alive()
        assert alive(pid), "the pgid kill unexpectedly reached a setsid grandchild"
    finally:
        reap(pid)
    assert jobs.read_state(job_id).status == "cancelled"


def test_status_json_says_which_isolation_a_running_job_has() -> None:
    """Text dropped it -- what a kill reaps is a debugging question, and the
    answer costs a column on every running line -- but automation still gets it."""
    queued, running, finished = job_views(
        {"jobs": [{"job_id": "j1", "status": "running", "phase": "main", "isolation": "cgroup"}]}
    )
    assert (queued, finished) == ([], [])
    document = host_json(
        HostView(entry=host_entry(name="h"), reachable=True, heartbeat_age_s=1.0, running=running)
    )
    assert document["running"][0]["iso"] == "cgroup"
