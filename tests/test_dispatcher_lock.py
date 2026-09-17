from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from gpuc.host import dispatcher, jobs, paths
from gpuc.host import runner as procinfo
from gpuc.host.dispatcher import DispatcherLock, LockBody
from gpuc.host.jobs import HostConfig

# A holder that takes the lock, writes a dispatcher-shaped lock body and a
# heartbeat of our choosing, and then wedges forever with a child in the same
# process group -- exactly the shape of the failure this design exists to
# survive. The trailing argv marker is what makes it look like `gpuc.host` in
# /proc, which is the only kind of process a takeover is allowed to kill.
WEDGED_HOLDER = r"""
import fcntl, json, os, signal, sys, time
lock_path, heartbeat_path, age = sys.argv[1], sys.argv[2], float(sys.argv[3])
pkg_commit, on_term = sys.argv[4] or None, sys.argv[5]
if on_term == "ignore":
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
else:
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
stat = open(f"/proc/{os.getpid()}/stat").read()
starttime = stat.rpartition(")")[2].split()[19]
boot_id = open("/proc/sys/kernel/random/boot_id").read().strip()
body = {"pid": os.getpid(), "pgid": os.getpgid(0), "starttime": starttime, "boot_id": boot_id}
if pkg_commit:
    body["pkg_commit"] = pkg_commit
os.ftruncate(fd, 0)
os.write(fd, (json.dumps(body, sort_keys=True) + "\n").encode())
os.fsync(fd)
open(heartbeat_path, "w").close()
stamp = time.time() - age
os.utime(heartbeat_path, (stamp, stamp))
os.spawnvp(os.P_NOWAIT, "sleep", ["sleep", "600"])
print("holding", flush=True)
while True:
    time.sleep(3600)
"""


def start_wedged_holder(
    heartbeat_age_s: float, pkg_commit: str = "", on_term: str = "exit"
) -> subprocess.Popen[str]:
    proc = subprocess.Popen(
        [
            sys.executable,
            "-c",
            WEDGED_HOLDER,
            str(paths.lock_file()),
            str(paths.heartbeat_file()),
            str(heartbeat_age_s),
            pkg_commit,
            on_term,
            "gpuc.host",
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


def test_acquire_records_pid_pgid_starttime_and_boot_id(gpuc_home: Path) -> None:
    lock = DispatcherLock()
    assert lock.acquire()
    try:
        body = LockBody.parse(paths.lock_file().read_text())
        assert body.pid == os.getpid()
        assert body.starttime == procinfo.starttime(os.getpid())
        assert body.boot_id == procinfo.boot_id()
        assert body.pgid in (os.getpid(), None)
        age = lock.heartbeat_age()
        assert age is not None and age < 5
    finally:
        lock.release()


def test_adopt_refuses_to_record_a_pgid_that_is_not_its_own_pid(
    gpuc_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(os, "getpgid", lambda _pid: os.getpid() + 1)
    lock = DispatcherLock()
    assert lock.acquire()
    lock.release()
    body = LockBody.parse(paths.lock_file().read_text())
    assert body.pid == os.getpid()
    assert body.pgid is None
    assert "detached" in paths.dispatcher_log().read_text()


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


def test_the_heartbeat_thread_beats_while_the_main_loop_is_busy(gpuc_home: Path) -> None:
    lock = DispatcherLock(heartbeat_interval_s=0.05)
    assert lock.acquire()
    try:
        lock.start_heartbeat()
        stale = time.time() - 120
        os.utime(paths.heartbeat_file(), (stale, stale))
        # Generous: this asserts the thread beats *at all* while the main loop
        # holds the GIL, not that it manages it inside any particular second.
        deadline = time.time() + 30
        while time.time() < deadline:
            age = lock.heartbeat_age()
            if age is not None and age < 1.0:
                break
            time.sleep(0.05)
        else:
            pytest.fail("the heartbeat thread never beat")
    finally:
        lock.release()
    assert lock._beat_thread is None


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
        assert LockBody.parse(paths.lock_file().read_text()).pid == os.getpid()
        age = lock.heartbeat_age()
        assert age is not None and age < 5
        lock.release()
    finally:
        kill_tree(holder)


def write_lock_body(**body: object) -> None:
    paths.lock_file().write_text(json.dumps(body))


def test_a_stale_beat_never_kills_a_process_that_is_not_a_gpuc_dispatcher(
    gpuc_home: Path,
) -> None:
    innocent = subprocess.Popen(["sleep", "30"], start_new_session=True)
    try:
        write_lock_body(
            pid=innocent.pid,
            pgid=os.getpgid(innocent.pid),
            starttime=procinfo.starttime(innocent.pid),
            boot_id=procinfo.boot_id(),
        )
        lock = DispatcherLock()
        lock._evict_stale_holder()
        assert lock.takeover_pgid is None
        assert innocent.poll() is None
        assert "is not a gpuc dispatcher" in paths.dispatcher_log().read_text()
    finally:
        innocent.kill()
        innocent.wait(timeout=10)


def test_a_stale_beat_never_kills_a_group_that_is_not_the_holders_own_pid(
    gpuc_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(dispatcher, "is_gpuc_process", lambda _pid: True)
    write_lock_body(pid=os.getpid(), pgid=os.getpid() + 1, boot_id=procinfo.boot_id())
    lock = DispatcherLock()
    lock._evict_stale_holder()
    assert lock.takeover_pgid is None
    assert "only a process group led by the dispatcher itself" in paths.dispatcher_log().read_text()


def test_a_holder_recorded_under_a_different_boot_id_is_simply_gone(gpuc_home: Path) -> None:
    write_lock_body(pid=os.getpid(), pgid=os.getpid(), boot_id="0000-from-a-previous-boot")
    lock = DispatcherLock()
    lock._evict_stale_holder()
    assert lock.takeover_pgid is None
    assert "is gone" in paths.dispatcher_log().read_text()


def test_a_legacy_lock_body_is_taken_over_without_killing_anything(gpuc_home: Path) -> None:
    paths.lock_file().write_text(f"{os.getpgid(0)}\n")
    lock = DispatcherLock()
    lock._evict_stale_holder()
    assert lock.takeover_pgid is None
    assert "records no pid" in paths.dispatcher_log().read_text()


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


def test_the_heartbeat_is_fresh_before_the_lock_body_is_written(
    gpuc_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A second dispatcher must never see our pid next to a stale heartbeat.

    It reads the body to decide who to evict, so in the window between writing
    the body and the first beat it would have found a fresh pid with an ancient
    (or missing) heartbeat and SIGKILLed the dispatcher that had just won.
    """
    ages: list[float | None] = []
    render = LockBody.render
    monkeypatch.setattr(
        LockBody,
        "render",
        lambda self: (ages.append(dispatcher.heartbeat_age()), render(self))[1],
    )
    lock = DispatcherLock()
    assert lock.acquire()
    lock.release()
    assert ages and ages[0] is not None and ages[0] < 5


# -- a dispatcher older than the package it is dispatching ---------------------
#
# The failure these cover: a dispatcher imports its code once and then lives for
# days, so re-shipping the package under a live one changes nothing at all about
# what it dispatches with. `gpuc host bootstrap` looked like it fixed that, and
# did not -- the dispatcher it started found a fresh heartbeat and exited, and a
# job asking for a feature shipped that morning waited on a host that had never
# heard of it.

SHIPPED = "b" * 40
RUNNING = "a" * 40


def on_this_host(pkg_commit: str | None) -> None:
    jobs.write_config(HostConfig(host="test-host", pkg_commit=pkg_commit))


def test_a_holder_on_the_build_this_host_has_replaced_is_asked_to_stand_down(
    gpuc_home: Path,
) -> None:
    on_this_host(SHIPPED)
    holder = start_wedged_holder(heartbeat_age_s=1.0, pkg_commit=RUNNING)
    try:
        lock = DispatcherLock()
        assert lock.acquire(takeover_wait_s=10.0, handoff_wait_s=20.0)
        assert holder.wait(timeout=10) == 0  # it stopped on its own, not killed
        assert LockBody.parse(paths.lock_file().read_text()).pkg_commit == SHIPPED
        lock.release()
        log = paths.dispatcher_log().read_text()
        assert "SIGTERMing" in log and RUNNING[:12] in log and SHIPPED[:12] in log
    finally:
        kill_tree(holder)


def test_a_holder_too_old_to_record_a_commit_counts_as_replaced(gpuc_home: Path) -> None:
    """The case that actually happened: every dispatcher started before the lock
    carried a commit. "It did not say" is the oldest build of all, not a maybe."""
    on_this_host(SHIPPED)
    holder = start_wedged_holder(heartbeat_age_s=1.0)
    try:
        lock = DispatcherLock()
        assert lock.acquire(takeover_wait_s=10.0, handoff_wait_s=20.0)
        assert holder.wait(timeout=10) == 0
        lock.release()
    finally:
        kill_tree(holder)


def test_a_holder_that_will_not_stand_down_is_killed(gpuc_home: Path) -> None:
    """A dispatcher running code this host no longer has is worse than none at
    all: nothing else here can work around it, so it does not get a veto."""
    on_this_host(SHIPPED)
    holder = start_wedged_holder(heartbeat_age_s=1.0, pkg_commit=RUNNING, on_term="ignore")
    try:
        lock = DispatcherLock()
        assert lock.acquire(takeover_wait_s=10.0, handoff_wait_s=2.0)
        assert holder.wait(timeout=10) == -9
        lock.release()
    finally:
        kill_tree(holder)


def test_a_holder_on_the_same_build_is_left_alone(gpuc_home: Path) -> None:
    on_this_host(SHIPPED)
    holder = start_wedged_holder(heartbeat_age_s=1.0, pkg_commit=SHIPPED)
    try:
        assert not DispatcherLock().acquire(takeover_wait_s=1.0, handoff_wait_s=1.0)
        assert holder.poll() is None
    finally:
        kill_tree(holder)


def test_a_host_that_records_no_commit_never_evicts_anybody(gpuc_home: Path) -> None:
    """Nothing to compare against is not evidence of anything, and a host where
    no commit is ever known must not restart its dispatcher on every enqueue."""
    on_this_host(None)
    holder = start_wedged_holder(heartbeat_age_s=1.0)
    try:
        assert not DispatcherLock().acquire(takeover_wait_s=1.0, handoff_wait_s=1.0)
        assert holder.poll() is None
    finally:
        kill_tree(holder)


def test_acquire_records_the_commit_this_dispatcher_is_running(gpuc_home: Path) -> None:
    on_this_host(SHIPPED)
    lock = DispatcherLock()
    assert lock.acquire()
    lock.release()
    assert LockBody.parse(paths.lock_file().read_text()).pkg_commit == SHIPPED
    assert dispatcher.holder_pkg_commit() == SHIPPED


def test_sigterm_asks_the_loop_to_finish_its_pass_and_exit(gpuc_home: Path) -> None:
    loop = dispatcher.Dispatcher()
    dispatcher._stop_on_sigterm(loop)
    try:
        assert not loop.should_exit
        os.kill(os.getpid(), signal.SIGTERM)
        assert loop.should_exit
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
