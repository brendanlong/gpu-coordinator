"""Process facts and the one way a job's processes are stopped.

A recorded pid alone proves nothing: pids are reused within a boot and reused
from 1 again after a reboot, so "is the process that wrote this file still
running?" needs the boot id and the process start time as well. And a job is
three things at once -- a systemd scope where the host has one, the process
group of the phase, and the runner supervising both -- so every place that
stops one (the runner on a cancel, the dispatcher backing it up, adoption
clearing a dead runner's leftovers) goes through `JobProcesses` rather than
re-deriving which of the three to signal.
"""

from __future__ import annotations

import contextlib
import os
import signal
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from gpuc.host import scope
from gpuc.host.jobs import JobState

KILL_GRACE_S = 15.0
"""SIGTERM, then SIGKILL after this: a job gets one grace period to checkpoint."""

BOOT_ID_PATH = Path("/proc/sys/kernel/random/boot_id")


def boot_id() -> str | None:
    try:
        return BOOT_ID_PATH.read_text().strip() or None
    except OSError:
        return None


def parse_starttime(stat: str) -> str | None:
    """Field 22 of ``/proc/<pid>/stat``, as text.

    Split after the last ``)``: the comm field is parenthesised and may itself
    contain spaces and parentheses, which breaks a naive ``split()``.
    """
    _, sep, rest = stat.rpartition(")")
    if not sep:
        return None
    fields = rest.split()
    if len(fields) < 20:
        return None
    return fields[19]


def starttime(pid: int) -> str | None:
    try:
        return parse_starttime(Path(f"/proc/{pid}/stat").read_text())
    except OSError:
        return None


def cmdline(pid: int) -> str:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return ""
    return raw.decode("utf-8", "replace").replace("\0", " ").strip()


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def is_gpuc_process(pid: int) -> bool:
    return "gpuc.host" in cmdline(pid)


def cmdline_argv(pid: int) -> list[str]:
    """`/proc/<pid>/cmdline` as the argv it actually is.

    Not `cmdline().split()`: that joins the arguments with spaces, and splitting
    them again tears any argument that contains one into several. The runner
    starts a job as `bash -c <the whole script>`, so an argument full of words
    is the ordinary case on this host, not a contrived one.
    """
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return []
    if not raw:
        return []
    return [arg.decode("utf-8", "replace") for arg in raw.rstrip(b"\0").split(b"\0")]


def runner_job_id(argv: Sequence[str]) -> str | None:
    """The job a `python -m gpuc.host run <job_id>` argv belongs to, or None.

    The other half of `dispatcher._spawn_host_process`, which builds that
    command: one fact in two modules, so a test pins them together. A runner
    this stopped recognising would be adopted by nobody.
    """
    if len(argv) >= 2 and argv[-2] == "run" and "gpuc.host" in argv:
        return argv[-1]
    return None


def live_runner_pids() -> dict[str, int]:
    """Every live `gpuc.host run <job_id>` on this host, by the job it runs.

    The question a recorded pid cannot answer. `launch_ready` writes a job's
    state `running` before there is a process to name, and the runner pid only
    after the spawn, so a dispatcher that died between the two left a state
    naming no runner at all -- while the runner it did start is still going.
    /proc is the only remaining record of it.

    One walk, because the caller has every job to ask about. A runner that has
    already exited is not in it even before it is reaped: a defunct process has
    an empty cmdline.
    """
    found: dict[str, int] = {}
    try:
        pids = sorted(int(entry.name) for entry in Path("/proc").iterdir() if entry.name.isdigit())
    except OSError:
        return found
    for pid in pids:
        job_id = runner_job_id(cmdline_argv(pid))
        if job_id is not None:
            found.setdefault(job_id, pid)
    return found


def recorded_process_alive(
    pid: int | None, recorded_boot_id: str | None = None, recorded_starttime: str | None = None
) -> bool:
    """Is the *same* process we recorded still running?"""
    if not pid or not pid_alive(pid):
        return False
    current_boot = boot_id()
    if recorded_boot_id and current_boot and recorded_boot_id != current_boot:
        return False
    current_start = starttime(pid)
    return not (recorded_starttime and current_start and recorded_starttime != current_start)


def process_group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


Log = Callable[[str], None]


@dataclass(frozen=True)
class JobProcesses:
    """What a job is running as, from its state: the scope, the job's own
    process group, and the runner supervising it. Any may be unknown."""

    cgroup_unit: str | None = None
    job_pgid: int | None = None
    runner_pid: int | None = None

    @staticmethod
    def of(state: JobState, runner_pid: int | None = None) -> JobProcesses:
        """From a state file. The job's group is the one the runner published,
        never the runner's own: during the launch window the runner is the only
        member of its group, and killing that kills the one process that can
        finish the job cleanly."""
        runner = runner_pid or state.runner_pid
        job_pgid = state.pgid if state.pgid and state.pgid != runner else None
        return JobProcesses(state.cgroup_unit, job_pgid, runner)

    def stop(
        self,
        reason: str,
        *,
        grace_s: float = KILL_GRACE_S,
        sleep: Callable[[float], None] = time.sleep,
        now: Callable[[], float] = time.monotonic,
        reap: Callable[[], object] | None = None,
        log: Log = lambda _: None,
    ) -> None:
        """Ask the job to stop and make sure it does.

        The scope first, where there is one: a cgroup stop reaps the whole
        tree, a grandchild that double-forked included. Then SIGTERM the
        group and SIGKILL it after `grace_s` -- always, because the group kill
        is the fallback and against a cgroup that is already empty it costs one
        ProcessLookupError. `reap` must wait() a direct child if the caller has
        one: an unreaped zombie is still a member of its group, so without it
        every kill would burn the whole grace period before the group looked
        empty.
        """
        if self.cgroup_unit is not None:
            log(f"stopping scope {self.cgroup_unit}: {reason}")
            if not scope.stop_unit(self.cgroup_unit):
                log(f"systemctl --user stop {self.cgroup_unit} failed; falling back to the group")
        elif self.job_pgid:
            log(f"killing process group {self.job_pgid}: {reason}")
        pgid = self.job_pgid
        if not pgid or pgid <= 1:
            return
        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            return
        deadline = now() + grace_s
        while now() < deadline:
            if reap is not None:
                reap()
            if not process_group_alive(pgid):
                return
            sleep(0.25)
        with contextlib.suppress(ProcessLookupError):
            os.killpg(pgid, signal.SIGKILL)

    def kill(self, log: Log = lambda _: None) -> None:
        """Take the job's processes down now, with no grace: the scope, then
        SIGKILL the group. For leftovers nothing is supervising any more."""
        if self.cgroup_unit:
            log(f"stopping leftover scope {self.cgroup_unit}")
            scope.stop_unit(self.cgroup_unit)
        pgid = self.job_pgid
        if not pgid or pgid <= 1 or pgid == os.getpgid(0) or not process_group_alive(pgid):
            return
        log(f"SIGKILLing process group {pgid}")
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(pgid, signal.SIGKILL)

    def escalate(self, elapsed_s: float, grace_s: float, log: Log | None = None) -> None:
        """The backstop for a runner that has not honoured a stop request.

        One rung per grace period: the job's processes, then a SIGTERM to the
        runner (which handles it with a final sync), then the runner's own
        group. Nothing inside the first grace period -- the runner acts within
        a poll -- and nothing at all if the caller decided the runner is busy
        finishing (see `Dispatcher.escalate_stops`).
        """
        if elapsed_s <= grace_s:
            return
        log = log or (lambda _: None)
        self.kill(log)
        runner = self.runner_pid
        if not runner or runner <= 1:
            return
        if elapsed_s > 2 * grace_s and pid_alive(runner):
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.kill(runner, signal.SIGTERM)
        if elapsed_s > 3 * grace_s and process_group_alive(runner):
            log(f"runner pid {runner} is wedged; SIGKILLing its group")
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(runner, signal.SIGKILL)
