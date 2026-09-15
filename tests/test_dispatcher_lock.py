from __future__ import annotations

import contextlib
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from gpuc.host import paths
from gpuc.host.dispatcher import DispatcherLock

# A holder that takes the lock, writes its pgid and a heartbeat of our choosing,
# and then wedges forever with a child in the same process group -- exactly the
# shape of the failure this design exists to survive.
WEDGED_HOLDER = r"""
import fcntl, os, sys, time
lock_path, heartbeat_path, age = sys.argv[1], sys.argv[2], float(sys.argv[3])
fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
os.ftruncate(fd, 0)
os.write(fd, f"{os.getpgid(0)}\n".encode())
os.fsync(fd)
open(heartbeat_path, "w").close()
stamp = time.time() - age
os.utime(heartbeat_path, (stamp, stamp))
os.spawnvp(os.P_NOWAIT, "sleep", ["sleep", "600"])
print("holding", flush=True)
while True:
    time.sleep(3600)
"""


def start_wedged_holder(heartbeat_age_s: float) -> subprocess.Popen[str]:
    proc = subprocess.Popen(
        [
            sys.executable,
            "-c",
            WEDGED_HOLDER,
            str(paths.lock_file()),
            str(paths.heartbeat_file()),
            str(heartbeat_age_s),
        ],
        stdout=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    assert proc.stdout is not None
    assert proc.stdout.readline().strip() == "holding"
    return proc


def kill_tree(proc: subprocess.Popen[str]) -> None:
    with contextlib.suppress(ProcessLookupError):
        os.killpg(proc.pid, 9)
    proc.wait(timeout=10)


def test_acquire_writes_pgid_and_heartbeat(gpuc_home: Path) -> None:
    lock = DispatcherLock()
    assert lock.acquire()
    try:
        assert paths.lock_file().read_text().strip() == str(os.getpgid(0))
        age = lock.heartbeat_age()
        assert age is not None and age < 5
    finally:
        lock.release()


def test_heartbeat_is_rate_limited(gpuc_home: Path) -> None:
    clock = [1000.0]
    lock = DispatcherLock(now=lambda: clock[0], heartbeat_interval_s=5.0)
    assert lock.acquire()
    try:
        first = paths.heartbeat_file().stat().st_mtime
        clock[0] += 1.0
        lock.beat()
        assert paths.heartbeat_file().stat().st_mtime == first
        clock[0] += 5.0
        lock.beat()
        assert paths.heartbeat_file().stat().st_mtime > first
    finally:
        lock.release()


def test_second_dispatcher_exits_silently_when_the_heartbeat_is_fresh(
    gpuc_home: Path,
) -> None:
    holder = start_wedged_holder(heartbeat_age_s=1.0)
    try:
        assert not DispatcherLock().acquire(takeover_wait_s=1.0)
        assert holder.poll() is None
    finally:
        kill_tree(holder)


def test_takeover_kills_a_wedged_holder_with_a_stale_heartbeat(gpuc_home: Path) -> None:
    holder = start_wedged_holder(heartbeat_age_s=120.0)
    holder_pgid = os.getpgid(holder.pid)
    try:
        lock = DispatcherLock()
        assert lock.acquire(takeover_wait_s=20.0)
        assert lock.takeover_pgid == holder_pgid
        assert holder.wait(timeout=10) != 0
        deadline = time.time() + 10
        while time.time() < deadline:
            try:
                os.killpg(holder_pgid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.1)
        else:
            pytest.fail("the wedged holder's process group survived takeover")
        assert paths.lock_file().read_text().strip() == str(os.getpgid(0))
        age = lock.heartbeat_age()
        assert age is not None and age < 5
        lock.release()
    finally:
        kill_tree(holder)


def test_a_missing_heartbeat_counts_as_stale(gpuc_home: Path) -> None:
    holder = start_wedged_holder(heartbeat_age_s=120.0)
    try:
        paths.heartbeat_file().unlink()
        lock = DispatcherLock()
        assert lock.heartbeat_age() is None
        assert not lock.holder_is_fresh()
        assert lock.acquire(takeover_wait_s=20.0)
        lock.release()
    finally:
        kill_tree(holder)


def test_release_lets_the_next_dispatcher_in(gpuc_home: Path) -> None:
    first = DispatcherLock()
    assert first.acquire()
    first.release()
    second = DispatcherLock()
    assert second.acquire()
    second.release()
