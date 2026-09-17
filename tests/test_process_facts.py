from __future__ import annotations

import os
import subprocess
import sys
import time

from gpuc.host import runner as procinfo

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


def test_a_runner_is_found_by_the_job_it_was_started_for() -> None:
    """How a job whose state names no runner is told from one that has none."""
    job_id = "20250101-000000-abcdef"
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time;time.sleep(30)", "gpuc.host", "run", job_id]
    )
    try:
        _wait_for_cmdline(proc.pid)
        assert procinfo.find_runner_pid(job_id) == proc.pid
        # Another job's runner is not this job's, and neither is a dispatcher.
        assert procinfo.find_runner_pid("20250101-000000-fedcba") is None
    finally:
        proc.kill()
        proc.wait(timeout=10)
    assert procinfo.find_runner_pid(job_id) is None
