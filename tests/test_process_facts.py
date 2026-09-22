from __future__ import annotations

import os
import signal
import subprocess
import sys
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


def _wait_for_cmdline(pid: int) -> str:
    deadline = time.time() + 10
    while not procinfo.cmdline(pid) and time.time() < deadline:
        time.sleep(0.02)
    return procinfo.cmdline(pid)


def test_a_live_runner_is_found_by_the_job_it_was_started_for() -> None:
    """How a job whose state names no runner is told from one that has none."""
    job_id = "20250101-000000-abcdef"
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time;time.sleep(30)", "gpuc.host", "run", job_id]
    )
    try:
        _wait_for_cmdline(proc.pid)
        assert procinfo.live_runner_pids().get(job_id) == proc.pid
    finally:
        proc.kill()
        proc.wait(timeout=10)
    assert job_id not in procinfo.live_runner_pids()


def test_a_runner_is_named_from_argv_not_from_a_string_of_words() -> None:
    """A job's own command reaches /proc as one argument however many words it
    holds, and `bash -c <script>` is how every phase is run."""
    assert procinfo.runner_job_id([sys.executable, "-m", "gpuc.host", "run", "J"]) == "J"
    assert procinfo.runner_job_id([sys.executable, "-m", "gpuc.host", "dispatch"]) is None
    assert procinfo.runner_job_id(["bash", "-c", "python train.py gpuc.host run J"]) is None


def test_cmdline_argv_keeps_an_argument_that_contains_spaces() -> None:
    proc = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(30)", "a b c"])
    try:
        _wait_for_cmdline(proc.pid)
        assert procinfo.cmdline_argv(proc.pid)[-1] == "a b c"
    finally:
        proc.kill()
        proc.wait(timeout=10)
    assert procinfo.cmdline_argv(2**30) == []


# -- JobProcesses ---------------------------------------------------------------


def test_the_job_group_is_never_the_runners_own() -> None:
    """In the launch window the runner is alone in its group: killing that as
    "the job" would kill the one process that can finish the job cleanly."""
    assert JobProcesses.of(JobState(runner_pid=500, pgid=500)).job_pgid is None
    assert JobProcesses.of(JobState(pgid=500), runner_pid=500).job_pgid is None
    assert JobProcesses.of(JobState(runner_pid=400, pgid=500), runner_pid=500).job_pgid is None
    processes = JobProcesses.of(JobState(runner_pid=500, pgid=600, cgroup_unit="u.scope"))
    assert processes == JobProcesses("u.scope", 600, 500)


def test_the_runner_the_caller_holds_wins_over_the_recorded_one() -> None:
    assert JobProcesses.of(JobState(runner_pid=400, pgid=600), runner_pid=500).runner_pid == 500
    assert JobProcesses.of(JobState(runner_pid=400, pgid=600)).runner_pid == 400


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
