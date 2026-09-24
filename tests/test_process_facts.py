from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path

import pytest

from gpuc.host import procs as procinfo
from gpuc.host.jobs import JobState
from gpuc.host.procs import JobProcesses

# comm is parenthesised and unsanitised, so a process can name itself anything.
NASTY_STAT = "4242 (evil ) (name) S " + " ".join(str(n) for n in range(1, 19)) + " 987654 20 21"


def test_starttime_survives_a_comm_with_spaces_and_parens() -> None:
    assert procinfo.parse_starttime(NASTY_STAT) == "987654"
    assert procinfo.parse_starttime("nonsense") is None
    assert procinfo.parse_starttime("1 (sh) S 1 2 3") is None


def test_our_own_process_is_recorded_alive() -> None:
    pid = os.getpid()
    assert procinfo.pid_alive(pid)
    assert procinfo.recorded_process_alive(pid, procinfo.boot_id(), procinfo.starttime(pid))


def test_a_different_boot_id_or_starttime_means_dead() -> None:
    pid = os.getpid()
    assert not procinfo.recorded_process_alive(pid, "not-this-boot", procinfo.starttime(pid))
    assert not procinfo.recorded_process_alive(pid, procinfo.boot_id(), "1")
    assert not procinfo.recorded_process_alive(None)


def test_cmdline_identifies_a_gpuc_process() -> None:
    proc = subprocess.Popen(["sleep", "5"])
    try:
        deadline = time.time() + 10
        while not procinfo.cmdline(proc.pid) and time.time() < deadline:
            time.sleep(0.02)
        assert "sleep" in procinfo.cmdline(proc.pid)
        assert not procinfo.is_gpuc_process(proc.pid)
    finally:
        proc.kill()
        proc.wait(timeout=10)
    assert procinfo.cmdline(2**30) == ""
    assert not procinfo.pid_alive(2**30)


# -- JobProcesses ---------------------------------------------------------------


def test_job_processes_are_exactly_what_the_state_recorded() -> None:
    """The job's group is the one the runner published for the phase, and the
    runner is the one that claimed the job; neither is ever inferred."""
    processes = JobProcesses.of(JobState(runner_pid=500, pgid=600, cgroup_unit="u.scope"))
    assert processes == JobProcesses("u.scope", 600, 500)
    assert JobProcesses.of(JobState(runner_pid=500)) == JobProcesses(None, None, 500)
    assert JobProcesses.of(JobState(pgid=0)).job_pgid is None


def test_a_state_from_another_boot_names_no_processes() -> None:
    """Its pids were reissued from 1 and its scopes did not survive: a kill
    at any of them would land on whatever this boot put at those numbers."""
    other = JobState(runner_pid=500, pgid=600, cgroup_unit="u.scope", runner_boot_id="not-this")
    assert JobProcesses.of(other) == JobProcesses()
    same = JobState(
        runner_pid=500, pgid=600, cgroup_unit="u.scope", runner_boot_id=procinfo.boot_id()
    )
    assert JobProcesses.of(same) == JobProcesses("u.scope", 600, 500)


JOB_GROUP = 4242
RUNNER = 4343


@pytest.fixture
def signals(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, int, int]]:
    """Every real signal `escalate` sends, with every target reading alive."""
    sent: list[tuple[str, int, int]] = []

    def fake(kind: str):
        def send(target: int, sig: int) -> None:
            if sig != 0:
                sent.append((kind, target, sig))

        return send

    monkeypatch.setattr(os, "kill", fake("pid"))
    monkeypatch.setattr(os, "killpg", fake("group"))
    monkeypatch.setattr(os, "getpgid", lambda _pid: 1)
    return sent


GRACE = 10.0
ESCALATION = [
    (5.0, []),
    (GRACE, []),
    (1.5 * GRACE, [("group", JOB_GROUP, signal.SIGKILL)]),
    (
        2.5 * GRACE,
        [("group", JOB_GROUP, signal.SIGKILL), ("pid", RUNNER, signal.SIGTERM)],
    ),
    (
        3.5 * GRACE,
        [
            ("group", JOB_GROUP, signal.SIGKILL),
            ("pid", RUNNER, signal.SIGTERM),
            ("group", RUNNER, signal.SIGKILL),
        ],
    ),
]


@pytest.mark.parametrize(("elapsed", "expected"), ESCALATION)
def test_escalate_climbs_one_rung_per_grace_period(
    signals: list[tuple[str, int, int]], elapsed: float, expected: list[tuple[str, int, int]]
) -> None:
    """Nothing inside the grace period (the runner acts within a poll); then
    the job, then a SIGTERM to the runner, then the runner's whole group."""
    JobProcesses(job_pgid=JOB_GROUP, runner_pid=RUNNER).escalate(elapsed, GRACE)
    assert signals == expected


def test_escalate_stops_the_scope_before_the_group(
    signals: list[tuple[str, int, int]], monkeypatch: pytest.MonkeyPatch
) -> None:
    stopped: list[str] = []
    monkeypatch.setattr(procinfo.scope, "stop_unit", lambda unit: stopped.append(unit) or True)
    lines: list[str] = []
    JobProcesses("job.scope", JOB_GROUP, RUNNER).escalate(1.5 * GRACE, GRACE, lines.append)
    assert stopped == ["job.scope"]
    assert signals == [("group", JOB_GROUP, signal.SIGKILL)]
    assert lines == ["stopping leftover scope job.scope", f"SIGKILLing process group {JOB_GROUP}"]


def test_escalate_with_no_job_group_still_reaches_the_runner(
    signals: list[tuple[str, int, int]],
) -> None:
    JobProcesses(runner_pid=RUNNER).escalate(3.5 * GRACE, GRACE)
    assert signals == [("pid", RUNNER, signal.SIGTERM), ("group", RUNNER, signal.SIGKILL)]


def _wait_until(predicate, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("timed out")


def test_stop_escalates_to_sigkill_after_the_grace_period(tmp_path: Path) -> None:
    ready = tmp_path / "ready"
    proc = subprocess.Popen(
        ["bash", "-c", f"trap '' TERM; touch {ready}; sleep 60"], start_new_session=True
    )
    try:
        _wait_until(ready.exists)
        start = time.monotonic()
        JobProcesses(job_pgid=proc.pid).stop("test", grace_s=1.0, reap=proc.poll)
        assert proc.wait(timeout=10) == -9
        assert 1.0 <= time.monotonic() - start < 10
    finally:
        if proc.poll() is None:
            proc.kill()


def test_stop_ignores_a_group_that_is_not_there() -> None:
    JobProcesses().stop("test")
    JobProcesses(job_pgid=0).stop("test")
    JobProcesses(job_pgid=2**30).stop("test")


def test_members_are_the_group_and_the_scope_whatever_their_comm(tmp_path: Path) -> None:
    def fake(pid: int, pgrp: int, cgroup: str) -> None:
        (tmp_path / str(pid)).mkdir()
        (tmp_path / str(pid) / "stat").write_text(f"{pid} (a ) (b) S 1 {pgrp} 1 0")
        (tmp_path / str(pid) / "cgroup").write_text(f"0::{cgroup}\n")

    fake(10, 10, "/user.slice/app.slice/job.scope")
    fake(11, 10, "/user.slice/app.slice/other.scope")
    fake(12, 12, "/user.slice/app.slice/job.scope")
    fake(13, 13, "/user.slice/app.slice/not-job.scope")
    fake(14, 13, "/user.slice/app.slice/job.scope/child")
    (tmp_path / "self").mkdir()
    assert JobProcesses("job.scope", 10).members(tmp_path) == [10, 11, 12]
    assert JobProcesses(None, 10).members(tmp_path) == [10, 11]
    assert JobProcesses("job.scope", None).members(tmp_path) == [10, 12]
    assert JobProcesses().members(tmp_path) == []
