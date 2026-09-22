"""The per-host dispatcher: one lock, one loop, one runner per job.

Started idempotently by every enqueue. Holds an flock on an open fd (released
by the kernel if it dies) *and* touches a heartbeat file, so a second
dispatcher can tell "already running" from "wedged" and take over.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import signal
import subprocess
import sys
import threading
import time
import traceback
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from gpuc._version import is_other_build
from gpuc.host import baseline, cleanup, gpus, jobs, paths, plan, queue, scope, sync, terminate
from gpuc.host.gpus import SmiRunner
from gpuc.host.procs import (
    KILL_GRACE_S,
    JobProcesses,
    boot_id,
    cmdline,
    is_gpuc_process,
    live_runner_pids,
    pid_alive,
    recorded_process_alive,
    starttime,
)
from gpuc.host.terminate import TerminateCall

HEARTBEAT_INTERVAL_S = 5.0
HEARTBEAT_STALE_S = 30.0
LOOP_INTERVAL_S = 2.0
TERMINATE_RETRY_S = 600.0
MAX_CONSECUTIVE_FAILURES = 20
RETENTION_INTERVAL_S = 3600.0
"""How often a dispatcher with `retention_days` or `workdir_days` set reclaims.

Once at startup and then hourly: deleting day-old venvs is not urgent, and
on a non-ephemeral host the dispatcher only lives while there is work, so the
startup pass is the one that usually fires.
"""
OUTPUT_RETRY_ATTEMPTS = 3
OUTPUT_RETRY_INTERVAL_S = 60.0
OUTPUT_RETRY_BUDGET_S = 300.0


def log_line(message: str, now: datetime | None = None) -> None:
    """Append one stamped line to the dispatcher log.

    Never raises: a log we cannot write is not a reason to lose a job.
    """
    stamp = (now or datetime.now(UTC)).isoformat(timespec="seconds")
    with contextlib.suppress(OSError), paths.dispatcher_log().open("a") as handle:
        handle.write(f"{stamp} {message}\n")


@dataclass
class LockBody:
    """Who holds the lock, in enough detail to tell them apart from a pid that
    was reused by an unrelated process (or by a process from an earlier boot)."""

    pid: int | None = None
    pgid: int | None = None
    starttime: str | None = None
    boot_id: str | None = None
    pkg_commit: str | None = None
    """`config.pkg_commit` as it read when this dispatcher took the lock.

    A dispatcher runs the code it imported at exec and nothing re-imports it,
    so a long-lived one goes on dispatching last week's package however many
    times the host is re-bootstrapped underneath it. This is what lets the next
    one tell that it is the newer build and take over (`_holder_is_superseded`).
    """

    def render(self) -> str:
        return json.dumps(asdict(self), sort_keys=True) + "\n"

    @staticmethod
    def parse(text: str) -> LockBody:
        try:
            document = json.loads(text)
        except json.JSONDecodeError:
            return LockBody()
        if not isinstance(document, dict):
            return LockBody()
        # Tolerant on purpose: this file is written by whichever build of gpuc
        # last took the lock, and a pid we cannot read means "no known holder"
        # -- which the caller already handles -- not a crash on the way to
        # taking over.
        return LockBody(
            pid=jobs.as_opt_int(document, "pid"),
            pgid=jobs.as_opt_int(document, "pgid"),
            starttime=jobs.as_opt_str(document, "starttime") or None,
            boot_id=jobs.as_opt_str(document, "boot_id") or None,
            pkg_commit=jobs.as_opt_str(document, "pkg_commit") or None,
        )


def running_pkg_commit() -> str | None:
    """The commit of the package a dispatcher starting now would be running.

    `config.pkg_commit` is written by whoever shipped the package, *before* the
    dispatcher that serves it is started (`bootstrap.resync_package`), so it
    names the code on disk. Read once, when the lock object is built -- which
    is the first thing a dispatcher does -- and then recorded rather than
    re-read: the point is to remember which build this process is, and a later
    ship must not silently rewrite that answer.

    What it cannot see is a *second* control machine shipping in the gap
    between the config write and this read, an ssh round trip wide: this
    dispatcher would then record that machine's commit while running the one it
    was started from, and nothing would evict it. The alternative is threading
    the shipped commit through the spawn, a second source of truth for the same
    fact, and the trade is deliberate.
    """
    with contextlib.suppress(RuntimeError, OSError, ValueError):
        return jobs.read_config().pkg_commit
    return None


def holder_pkg_commit() -> str | None:
    """The commit recorded by whoever last took the lock, if they said.

    For `gpuc status`, which pairs it with the heartbeat: the lock file outlives
    the process that wrote it, so this answers "what was the last dispatcher
    here built from", and only a fresh heartbeat makes that a fact about now.
    """
    with contextlib.suppress(OSError, ValueError):
        return LockBody.parse(paths.lock_file().read_text()).pkg_commit
    return None


def heartbeat_age(now: Callable[[], float] = time.time) -> float | None:
    """Seconds since a dispatcher last beat, or None if none ever has.

    A plain function: the age of a file is not something a caller should have
    to build (and half-initialise) a lock object to ask about.
    """
    try:
        return now() - paths.heartbeat_file().stat().st_mtime
    except FileNotFoundError:
        return None


class DispatcherLock:
    def __init__(
        self,
        *,
        stale_after_s: float = HEARTBEAT_STALE_S,
        heartbeat_interval_s: float = HEARTBEAT_INTERVAL_S,
        now: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.stale_after_s = stale_after_s
        self.heartbeat_interval_s = heartbeat_interval_s
        self._now = now
        self._sleep = sleep
        self._fd: int | None = None
        self._last_beat = 0.0
        self._beat_stop = threading.Event()
        self._beat_thread: threading.Thread | None = None
        self.takeover_pgid: int | None = None
        self.pkg_commit = running_pkg_commit()

    @property
    def lock_path(self) -> Path:
        return paths.lock_file()

    @property
    def heartbeat_path(self) -> Path:
        return paths.heartbeat_file()

    def heartbeat_age(self) -> float | None:
        return heartbeat_age(self._now)

    def holder_is_fresh(self) -> bool:
        age = self.heartbeat_age()
        return age is not None and age < self.stale_after_s

    def holder(self) -> LockBody:
        with contextlib.suppress(OSError, ValueError):
            return LockBody.parse(self.lock_path.read_text())
        return LockBody()

    def acquire(self, takeover_wait_s: float = 10.0, handoff_wait_s: float = 30.0) -> bool:
        paths.ensure_layout()
        fd = os.open(self.lock_path, os.O_RDWR | os.O_CREAT, 0o644)
        if self._try_flock(fd):
            self._adopt(fd)
            return True
        incumbent = self.holder()
        if self.holder_is_fresh():
            if not self._is_another_build(incumbent):
                os.close(fd)
                return False
            if not self._ask_to_stand_down(incumbent):
                # Nothing we are allowed to signal, so nothing is going to
                # stand down: waiting out a handoff nobody was asked for would
                # cost every enqueue 40s for as long as that holder lives.
                os.close(fd)
                return False
            if self._take_over(fd, handoff_wait_s):
                return True
            if not self._still_held_by(incumbent):
                # Somebody else won the lock while we waited. Whatever it is,
                # it is not the process we asked to stand down, and killing the
                # dispatcher that just took over -- possibly the one running
                # exactly the code we wanted -- is the worst thing we could do
                # here. Two dispatchers start within seconds of each other on
                # every `gpuc submit` (resync starts one, the enqueue another),
                # so this is the ordinary case, not the exotic one.
                os.close(fd)
                return False
            # It would not go. A dispatcher running a build this host does not
            # have is worse than none at all -- it is the one thing nothing
            # else here can work around -- so it gets the same SIGKILL a wedged
            # one does, by the same rules about what may be signalled.
            self._signal_holder(signal.SIGKILL, incumbent, "it did not stand down when asked")
        else:
            self._evict_stale_holder()
        if self._take_over(fd, takeover_wait_s):
            return True
        os.close(fd)
        return False

    def _still_held_by(self, incumbent: LockBody) -> bool:
        """Is the lock still held by the process we were negotiating with?

        pid *and* start time: a pid alone is reused, and the whole question is
        whether the thing on the other end of our SIGTERM is the thing about to
        get our SIGKILL.
        """
        now = self.holder()
        return now.pid == incumbent.pid and now.starttime == incumbent.starttime

    def _take_over(self, fd: int, wait_s: float) -> bool:
        deadline = self._now() + wait_s
        while self._now() < deadline:
            if self._try_flock(fd):
                self._adopt(fd)
                return True
            self._sleep(0.25)
        return False

    def _is_another_build(self, incumbent: LockBody) -> bool:
        """Is the dispatcher holding the lock running a build other than ours?

        The heartbeat says a dispatcher is *alive*, which was the whole test
        until the package underneath one could change while it ran. It can: a
        host is re-bootstrapped whenever `gpuc submit` finds it behind, and the
        incumbent re-imports nothing, so it serves the queue with whatever was
        on disk the day it started. That is not a stale lock and it is not a
        wedged process; it is the wrong code, and only a process starting from
        the package now on disk can tell.

        Other, not older -- see `is_other_build`. The code on disk is the code
        that should be running, whichever direction it moved.
        """
        return is_other_build(incumbent.pkg_commit, self.pkg_commit)

    def _ask_to_stand_down(self, incumbent: LockBody) -> bool:
        """SIGTERM the incumbent. False when there was nothing safe to signal."""
        was = incumbent.pkg_commit
        running = f"gpuc {was[:12]}" if was else "a build too old to say which"
        return self._signal_holder(
            signal.SIGTERM,
            incumbent,
            f"it is running {running} and this host now has "
            f"{(self.pkg_commit or 'unknown')[:12]} on disk",
        )

    def _signal_holder(self, sig: int, body: LockBody, why: str) -> bool:
        """Signal `body`'s process group if that is provably safe, and say so.

        The body is passed in rather than re-read: every caller has already
        decided something about a *particular* holder, and re-reading here
        would let the signal land on whichever process happened to hold the
        lock by the time it was sent.
        """
        pgid = self._signalable_pgid(body)
        if pgid is None:
            return False
        self.takeover_pgid = pgid
        log_line(f"{signal.Signals(sig).name}ing dispatcher process group {pgid}: {why}")
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(pgid, sig)
        return True

    def _signalable_pgid(self, body: LockBody) -> int | None:
        """The incumbent's process group, if signalling it is provably safe.

        The pid in the lock file may belong to something else entirely by now,
        and signalling a process group we do not own would take out an innocent
        bystander's shell and its jobs. A runner is started in its own session,
        so the dispatcher's group holds the dispatcher and nothing that is
        running a job.
        """
        if body.pid is None:
            log_line("lock is held by a dispatcher that records no pid; signalling nothing")
            return None
        if not recorded_process_alive(body.pid, body.boot_id, body.starttime):
            log_line(f"lock holder pid {body.pid} is gone; taking over")
            return None
        if not is_gpuc_process(body.pid):
            log_line(
                f"pid {body.pid} holds the lock but is not a gpuc dispatcher "
                f"({cmdline(body.pid)!r}); signalling nothing"
            )
            return None
        pgid = body.pgid
        if pgid != body.pid or pgid is None:
            log_line(
                f"lock holder pid {body.pid} records pgid {pgid}; only a process group "
                f"led by the dispatcher itself is ever signalled"
            )
            return None
        if pgid in (os.getpgid(0), os.getpid()):
            return None
        return pgid

    def _evict_stale_holder(self) -> None:
        """Kill the incumbent only when it is provably a wedged gpuc dispatcher.

        A stale heartbeat on its own is not enough, which is what
        `_signalable_pgid` is for: nothing is signalled until the pid in the
        lock file has been shown to still be the dispatcher that wrote it.
        """
        self._signal_holder(
            signal.SIGKILL, self.holder(), "its heartbeat is stale, so it is wedged"
        )

    def _try_flock(self, fd: int) -> bool:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return False
        return True

    def _adopt(self, fd: int) -> None:
        self._fd = fd
        # Beat *before* the body: between writing our pid and our first beat, a
        # second dispatcher racing for this lock would read a fresh pid next to
        # a stale heartbeat, conclude we were wedged, and SIGKILL the process
        # that just won.
        self.beat(force=True)
        pid = os.getpid()
        pgid = os.getpgid(0)
        if pgid != pid:
            log_line(
                f"dispatcher pid {pid} is not its own process group leader (pgid {pgid}); "
                f"recording no pgid. Start it detached (`setsid nohup ... &`) so a takeover "
                f"can clean it up."
            )
        body = LockBody(
            pid=pid,
            pgid=pgid if pgid == pid else None,
            starttime=starttime(pid),
            boot_id=boot_id(),
            pkg_commit=self.pkg_commit,
        )
        os.ftruncate(fd, 0)
        os.lseek(fd, 0, os.SEEK_SET)
        os.write(fd, body.render().encode())
        os.fsync(fd)

    def beat(self, force: bool = False) -> None:
        now = self._now()
        if not force and now - self._last_beat < self.heartbeat_interval_s:
            return
        self._last_beat = now
        self.heartbeat_path.parent.mkdir(parents=True, exist_ok=True)
        self.heartbeat_path.touch()
        os.utime(self.heartbeat_path, (now, now))

    def start_heartbeat(self) -> None:
        """Beat from a thread, so a long drain or final sync in the main loop
        cannot make a healthy dispatcher look wedged to the next one."""
        if self._beat_thread is not None:
            return
        self._beat_stop.clear()
        self._beat_thread = threading.Thread(
            target=self._beat_loop, name="gpuc-heartbeat", daemon=True
        )
        self._beat_thread.start()

    def _beat_loop(self) -> None:
        while not self._beat_stop.wait(self.heartbeat_interval_s):
            with contextlib.suppress(OSError):
                self.beat(force=True)

    def stop_heartbeat(self) -> None:
        self._beat_stop.set()
        thread, self._beat_thread = self._beat_thread, None
        if thread is not None:
            thread.join(timeout=self.heartbeat_interval_s * 2)

    def release(self) -> None:
        self.stop_heartbeat()
        if self._fd is None:
            return
        try:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
        finally:
            os.close(self._fd)
            self._fd = None


def _child_env(package_root: Path) -> dict[str, str]:
    """The environment every dispatcher-spawned process gets.

    PYTHONPATH so the rsynced package is importable, PATH so `uv` is findable
    (a pod's sshd PATH has no ~/.local/bin), and the host config's own `env`,
    which is how a host points a cache or a tool dir somewhere non-default.
    The dispatcher's environment is what the runner -- and through it every
    job -- inherits.
    """
    env = dict(os.environ)
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = f"{package_root}{os.pathsep}{existing}" if existing else str(package_root)
    # Probed once and handed down: creating a throwaway scope per phase to ask
    # the same question again would be one exec per phase for one bit.
    env[scope.ISOLATION_ENV] = scope.isolation()
    config = jobs.HostConfig()
    with contextlib.suppress(RuntimeError, OSError, ValueError):
        config = jobs.read_config()
    return config.apply_env(env)


def _spawn_host_process(*args: str) -> subprocess.Popen[bytes]:
    """`python -m gpuc.host <args>` in its own session -- and, where the host
    has user systemd, its own transient scope.

    A new session detaches from the terminal but not from the cgroup: a
    dispatcher started by `gpuc submit` from inside a systemd scope (an agent
    session, a `systemd-run --scope` wrapper, a CI job) stays in that scope,
    every runner it starts inherits it, and stopping the scope SIGTERMs them
    all -- which the runner reads as a stop and ends the job `terminated`,
    forty minutes into a run nobody asked to stop. `systemd-run --scope`
    execs the command in place, so the pid returned is still the process.
    """
    package_root = Path(__file__).resolve().parents[2]
    paths.ensure_layout()
    argv = [sys.executable, "-m", "gpuc.host", *args]
    if scope.probe():
        unit = f"gpuc-{'-'.join(args)}-{os.getpid()}-{int(time.time())}.scope"
        argv = [
            "systemd-run",
            "--user",
            "--scope",
            "--collect",
            "--quiet",
            f"--unit={unit}",
            "--",
            *argv,
        ]
    with paths.dispatcher_log().open("ab", buffering=0) as log:
        return subprocess.Popen(
            argv,
            cwd=str(package_root),
            env=_child_env(package_root),
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )


def default_spawn_runner(job_id: str) -> subprocess.Popen[bytes]:
    return _spawn_host_process("run", job_id)


def spawn_detached_dispatcher() -> int:
    """Start a dispatcher that outlives this process (and any SSH session).

    Always safe to call, so enqueue can fire it unconditionally: a second
    dispatcher exits silently when the incumbent is alive and was started from
    the same package. When it was not, this is the call that replaces it --
    see `DispatcherLock.acquire` -- which is why it is worth firing even on a
    host whose dispatcher is demonstrably healthy.
    """
    return _spawn_host_process("dispatch").pid


@dataclass
class DispatcherDeps:
    smi: SmiRunner = gpus.run_nvidia_smi
    command_runner: sync.CommandRunner = sync.run_command
    terminate_call: TerminateCall | None = None
    spawn_runner: Callable[[str], subprocess.Popen[bytes]] = default_spawn_runner
    monotonic: Callable[[], float] = time.monotonic
    utcnow: Callable[[], datetime] = lambda: datetime.now(UTC)
    sleep: Callable[[float], None] = time.sleep
    interval_s: float = LOOP_INTERVAL_S
    kill_grace_s: float = KILL_GRACE_S


@dataclass
class _Running:
    job_id: str
    pid: int
    gpus: list[str]
    popen: subprocess.Popen[bytes] | None = None

    def poll(self) -> int | None:
        if self.popen is not None:
            return self.popen.poll()
        return None if pid_alive(self.pid) else -1


@dataclass
class Preemptable:
    """A running job whose spec said it may be stopped for something better."""

    job_id: str
    priority: int
    gpus: list[str]
    started_at: str
    owned: int
    borrowed: int
    """How many of `gpus` this host owns, and how many it is borrowing. Only a
    waiting job that asked to borrow can be started on a borrowed one -- and a
    card in neither list, one that has dropped off nvidia-smi under a running
    job, is counted by neither, because it is never handed out to anybody and
    stopping a job for it would start nothing."""

    def frees(self, *, borrowing: bool) -> int:
        """Cards this would hand to a waiting job that may (or may not) borrow."""
        return self.owned + self.borrowed if borrowing else self.owned


def enough_to_start(
    candidates: list[Preemptable], priority: int, gap: int, *, borrowing: bool
) -> list[Preemptable]:
    """Which of these to stop so a job at `priority` gets `gap` more cards.

    All of them or none: freeing one of the two cards a job needs would cost an
    attempt and start nothing. Least important first, and among equals the one
    that has been running the shortest time, because what a preempt throws away
    is the work the attempt has already done.

    `borrowing` is the waiting job's `use_shared`: a borrowed card handed back
    by a stopped job is no use to a job that may not be dispatched onto one, so
    it does not count towards the gap and cannot be the reason a job is stopped.
    """
    chosen: list[Preemptable] = []
    freed = 0
    for candidate in sorted(candidates, key=lambda c: (c.priority, c.started_at), reverse=True):
        if freed >= gap:
            break
        if candidate.priority <= priority or not candidate.frees(borrowing=borrowing):
            continue
        chosen.append(candidate)
        freed += candidate.frees(borrowing=borrowing)
    return chosen if freed >= gap else []


@dataclass
class Dispatcher:
    deps: DispatcherDeps = field(default_factory=DispatcherDeps)
    running: dict[str, _Running] = field(default_factory=dict)
    _queue_empty_since: float | None = None
    _terminate_retry_at: float | None = None
    _last_reclaim_at: float | None = None
    _last_incoming_sweep_at: float | None = None
    _stop_sent: dict[str, float] = field(default_factory=dict)
    """When this dispatcher first saw each running job's stop intent, so the
    ladder in `escalate_stops` has a clock even for a request another process
    wrote."""
    _stop_escalated: set[str] = field(default_factory=set)
    _config: jobs.HostConfig | None = None
    _owned: list[str] | None = None
    _unavailable: tuple[str, ...] = ()
    _shared: list[str] | None = None
    _shared_unavailable: tuple[str, ...] = ()
    _borrowable: tuple[list[str], int] | None = None
    """This pass's one reading of the shared cards: the ones nobody else was
    on, and how many they were on. Reset every pass; see `borrowable_gpus`."""
    _queued: list[queue.QueueEntry] | None = None
    """This pass's one listing of the queue. Every state file is parsed to
    list it, and a pass asks three times."""
    _shared_in_use: tuple[str, ...] | None = None
    """The shared cards somebody else was on, last time this was asked.

    None rather than `()` until the first reading, so that the first one is
    logged even when it is "all of them are free": that line is the record of
    a job being allowed onto somebody else's card, and it is the first thing
    anybody looks for when one was not.
    """
    consecutive_failures: int = 0
    should_exit: bool = False

    @property
    def config(self) -> jobs.HostConfig:
        """The host config, read once per loop pass.

        One pass asks for it a dozen times (free GPUs, the idle timer,
        the drain); re-reading and re-parsing the file each time bought nothing
        but syscalls, and a mid-pass change is not something any of those
        decisions should straddle.
        """
        if self._config is None:
            self._config = jobs.read_config()
        return self._config

    def log(self, message: str) -> None:
        log_line(message, self.deps.utcnow())

    def queued(self) -> list[queue.QueueEntry]:
        if self._queued is None:
            self._queued = queue.list_queued()
        return self._queued

    def _claim(self, job_id: str, **state: object) -> bool:
        """`queue.claim`, and the cached listing forgets the job with it."""
        taken = queue.claim(job_id, **state)
        if taken and self._queued is not None:
            self._queued = [entry for entry in self._queued if entry.job_id != job_id]
        return taken

    # -- startup ---------------------------------------------------------
    def adopt_orphans(self) -> None:
        """Reconcile jobs left `running` by a dispatcher that died."""
        # Walked at most once, and only for a job whose state names no live
        # runner: `launch_ready` writes `running` before it has a process to
        # name and the pid only after the spawn, so a dispatcher killed between
        # the two -- every first takeover by a newer build, and both SIGKILL
        # paths -- leaves a live runner nothing points at. Failing that job
        # would lose it *and* put its cards back in the free pool underneath a
        # process still training on them, with no pgid recorded to kill it by:
        # the runner does not publish the job's own group until later.
        runners: dict[str, int] | None = None

        def live_runner(job_id: str) -> int | None:
            nonlocal runners
            if runners is None:
                runners = live_runner_pids()
            return runners.get(job_id)

        for job_id in jobs.list_job_ids():
            try:
                state = jobs.read_state(job_id)
            except RuntimeError as exc:
                self.log(f"job {job_id} has an unreadable state.json ({exc}); skipping")
                continue
            # A recorded pid means nothing across a reboot, and little after a
            # pid rollover: the boot id and start time recorded at launch are
            # what make "still running" a real answer.
            alive = bool(state.runner_pid) and recorded_process_alive(
                state.runner_pid, state.runner_boot_id, state.runner_starttime
            )
            if state.finished:
                if state.intent == jobs.PREEMPT:
                    # Finished, preempted, and its runner is *still there*: it
                    # is in its final sync, writing to the workdir and to
                    # state.json. Queueing the job now would launch the next
                    # attempt into that same workdir. Adopt the runner instead
                    # and let `reap` do it when the runner is really gone.
                    syncing = state.runner_pid if alive else live_runner(job_id)
                    if syncing is not None:
                        self.running[job_id] = _Running(
                            job_id, syncing, self._held_gpus(job_id, state)
                        )
                        self.log(f"job {job_id} was preempted and is still syncing; waiting for it")
                        continue
                # Otherwise its runner is gone and nothing was left to put it
                # back: that is this dispatcher's job now.
                self.requeue_if_preempted(job_id)
                continue
            if state.status != "running" or job_id in self.running:
                continue
            runner_pid = state.runner_pid if alive else live_runner(job_id)
            if runner_pid is not None:
                self.running[job_id] = _Running(job_id, runner_pid, self._held_gpus(job_id, state))
                # Not written back to the state: the runner records its own
                # identity a moment later (`_run_phases`), and a read-modify-
                # write from here would race the one it makes in between --
                # `_resolve_assigned`, whose resolved UUIDs would be the loss.
                unrecorded = "" if alive else ", which its dispatcher died before recording"
                self.log(f"adopted running job {job_id} (runner pid {runner_pid}{unrecorded})")
            else:
                self._mark_runner_died(job_id)
                # A job stopped by `gpuc preempt` whose runner then died still
                # asked to come back, and `reap` will never see this one: it
                # belongs to a dispatcher that is gone.
                self.requeue_if_preempted(job_id)

    def _held_gpus(self, job_id: str, state: jobs.JobState) -> list[str]:
        """The cards an adopted job is holding, as UUIDs.

        Through the resolver: a job launched before assignments were resolved
        host-side has indices in its state, and busy/free accounting is in
        UUIDs. An index adopted as-is would match nothing owned, so the card
        would read free and be handed out a second time while the job is still
        training on it.
        """
        try:
            held, _ = gpus.resolve_owned(state.gpus, self.deps.smi)
        except gpus.GpuError as exc:
            # Adoption runs once, at startup: a job left unadopted here is
            # never picked up, so an unreadable nvidia-smi must cost the
            # resolution, not the adoption.
            self.log(f"could not resolve the GPUs of {job_id} ({exc}); adopting as given")
            held = list(state.gpus)
        return held

    def _mark_runner_died(self, job_id: str) -> None:
        """Fail the job, after making sure nothing of it is left on the GPUs.

        The GPUs go back in the free pool the moment this returns, so a job
        process that outlived its runner has to die first; otherwise it keeps
        computing on a card the next job is about to be handed.
        """
        try:
            state: jobs.JobState | None = jobs.read_state(job_id)
        except RuntimeError as exc:
            self.log(f"job {job_id} has an unreadable state.json ({exc}); treating as runner-died")
            state = None
        if state is not None:
            JobProcesses.of(state).kill(
                lambda m: self.log(f"job {job_id}: {m} before freeing its GPUs")
            )
        with jobs.locked(job_id):
            # Re-read under the lock: a cancel or preempt that landed since
            # the read above must not be written over.
            try:
                final = jobs.read_state(job_id)
            except RuntimeError:
                final = jobs.JobState()
            final.status = "failed"
            final.reason = "runner-died"
            final.exit_code = final.exit_code or 1
            final.ended_at = jobs.utc_now()
            final.phase = None
            # The runner clears this itself on every path it survives; here it
            # did not survive, and a finished job carrying an eta reads to
            # anything keying on it as a job that is still going.
            final.eta = None
            # A periodic tick's success says nothing about what the job wrote
            # after it, and the final upload that would have never ran: these
            # outputs are not confirmed anywhere, and the sweep must not take
            # them.
            for record in final.output_uploads():
                record.ok_at = None
            jobs.write_state(job_id, final)
        self.log(f"job {job_id} failed: runner died without writing final state")

    # -- loop pieces -----------------------------------------------------
    def reap(self) -> None:
        for job_id, entry in list(self.running.items()):
            if entry.poll() is None:
                continue
            del self.running[job_id]
            self._stop_sent.pop(job_id, None)
            self._stop_escalated.discard(job_id)
            try:
                state: jobs.JobState | None = jobs.read_state(job_id)
            except RuntimeError:
                state = None
            if state is None or not state.finished:
                self._mark_runner_died(job_id)
            else:
                self.log(
                    f"job {job_id} {state.status}"
                    f"{f' ({state.reason})' if state.reason else ''} "
                    f"exit={state.exit_code}"
                )
            self.requeue_if_preempted(job_id)

    def requeue_if_preempted(self, job_id: str) -> None:
        """Put a job `gpuc preempt` stopped back in the queue.

        Here rather than in the runner: the queue is the dispatcher's, and a
        job whose runner died on the way out must still come back.
        """
        if not queue.is_preempted(job_id):
            return
        going = self._going_away()
        if going is not None:
            # The job would be queued onto a host that is about to stop
            # existing, where nothing would run it and the drain would not even
            # count its outputs as unconfirmed (that list is finished jobs).
            # Left finished, it keeps its record, its `preempted` reason and
            # its place in the drain's last upload attempt.
            with contextlib.suppress(RuntimeError, OSError):
                jobs.update_state(job_id, intent=None)
            self.log(
                f"job {job_id} was preempted, but this host is {going}, so it is not going "
                f"back in the queue: it stays {self._state_or_empty(job_id).status} and "
                f"`gpuc requeue` is what re-runs it"
            )
            return
        try:
            attempt = queue.requeue_preempted(job_id)
        except (OSError, RuntimeError, ValueError) as exc:
            self.log(
                f"job {job_id} was preempted but could not be queued again ({exc}); "
                f"it stays finished"
            )
            return
        if attempt is None:
            self.log(
                f"job {job_id} was preempted but is not going back in the queue: it "
                f"{self._state_or_empty(job_id).status} before the kill reached it, it was "
                f"cancelled while it stopped, or its workdir is gone"
            )
            return
        self.log(f"job {job_id} was preempted; queued again as attempt {attempt}")

    def _going_away(self) -> str | None:
        """Why this host will not be running anything else, or None."""
        if paths.draining_file().exists():
            return "draining"
        return None

    def escalate_stops(self) -> None:
        """Make a stop request stick when the runner never acts on it.

        The runner owns the kill: it polls its state, stops the job's
        processes, syncs and writes the final state. That is right when the
        runner is healthy and nothing at all when it is wedged, so a request
        it has not honoured within the grace period gets `JobProcesses.escalate`.
        A runner in its final sync is honouring it: the upload has no
        wall-clock cap by design, and a ladder that reached the runner there
        would kill the one upload that matters most, so nothing is escalated
        while the phase is `sync`.
        """
        now = self.deps.monotonic()
        grace = self.deps.kill_grace_s
        for job_id, entry in list(self.running.items()):
            state = self._state_or_empty(job_id)
            if state.intent is None or state.phase == "sync":
                continue
            # A request another process wrote -- `gpuc cancel`, or a
            # dispatcher we took over from -- needs a clock of its own, or a
            # runner that never acts on it is never escalated either.
            elapsed = now - self._stop_sent.setdefault(job_id, now)
            if elapsed <= grace:
                continue
            if job_id not in self._stop_escalated:
                self._stop_escalated.add(job_id)
                self.log(
                    f"job {job_id}: its runner has not stopped it {elapsed:.0f}s after the "
                    f"{state.intent} request; escalating"
                )
            JobProcesses.of(state, entry.pid).escalate(
                elapsed, grace, lambda m, job_id=job_id: self.log(f"job {job_id}: {m}")
            )

    @staticmethod
    def _state_or_empty(job_id: str) -> jobs.JobState:
        try:
            return jobs.read_state(job_id)
        except RuntimeError:
            return jobs.JobState()

    def owned_gpus(self) -> list[str]:
        """The UUIDs of the cards this host owns *and* can see, this pass.

        `config.gpus` may name cards by nvidia-smi index, which only means
        anything against the host's current numbering, so the list is resolved
        here rather than trusted. Everything downstream -- assignment, free/busy
        accounting, `CUDA_VISIBLE_DEVICES` -- is UUIDs.
        """
        if self._owned is None:
            self._owned, missing = gpus.resolve_owned(self.config.gpus, self.deps.smi)
            if tuple(missing) != self._unavailable:
                self._unavailable = tuple(missing)
                if missing:
                    self.log(
                        f"config.gpus lists {', '.join(missing)}, which nvidia-smi does not "
                        f"report on this host ({gpus.describe_table(self.deps.smi)}); those "
                        f"cards are not being handed out"
                    )
        return self._owned

    def shared_gpus(self) -> list[str]:
        """The UUIDs of the cards this host may *borrow*, this pass.

        Resolved exactly as `owned_gpus` is -- a shared card is named by
        nvidia-smi index or UUID like any other -- and then minus anything this
        host owns outright. A card listed in both is a configuration mistake
        somebody will make, and owning it is the stronger claim: it would
        otherwise be handed out freely as owned and then have its usage
        second-guessed as shared.
        """
        if self._shared is None:
            owned = set(self.owned_gpus())
            resolved, missing = gpus.resolve_owned(self.config.shared_gpus, self.deps.smi)
            if tuple(missing) != self._shared_unavailable:
                self._shared_unavailable = tuple(missing)
                if missing:
                    self.log(
                        f"config.shared_gpus lists {', '.join(missing)}, which nvidia-smi does "
                        f"not report on this host ({gpus.describe_table(self.deps.smi)}); those "
                        f"cards are not being borrowed"
                    )
            self._shared = [uuid for uuid in resolved if uuid not in owned]
        return self._shared

    def _busy_gpus(self) -> set[str]:
        return {uuid for entry in self.running.values() for uuid in entry.gpus}

    def free_gpus(self) -> list[str]:
        busy = self._busy_gpus()
        return [uuid for uuid in self.owned_gpus() if uuid not in busy]

    def borrowable_gpus(self) -> tuple[list[str], int]:
        """Shared cards nothing of ours holds *and* nobody else is using either,
        and how many shared cards somebody else *is* on.

        The nvidia-smi read is the whole of the preflight, and it is the one
        thing standing between a borrowed card and somebody else's training
        run, so it is taken fresh each pass rather than inferred from anything
        longer-lived. It is taken *once* a pass, and only when a job actually
        needs to borrow: every job that pass is then judged against one
        reading, which is also what stops two of them being handed the same
        card, and what stops `preempt_for_waiting` disagreeing with the
        `launch_ready` it is reasoning about. The count comes from the same
        reading because it decides which short job is stepped over.

        The cards the reading found free are re-filtered on the way out: a job
        launched later in the same pass has taken its own, and those are not on
        offer twice.
        """
        busy = self._busy_gpus()
        if self._borrowable is None:
            unused, in_use = gpus.unused_gpus(
                [uuid for uuid in self.shared_gpus() if uuid not in busy], self.deps.smi
            )
            self._borrowable = (unused, len(in_use))
            # Logged when the *set* changes, not when the numbers do: this is
            # sampled every pass a job is waiting, and somebody else's job moves
            # a utilization figure twice a second.
            if tuple(sorted(in_use)) != self._shared_in_use:
                self._shared_in_use = tuple(sorted(in_use))
                for uuid, why in sorted(in_use.items()):
                    self.log(f"shared GPU {uuid} is in use ({why}), so it is not being borrowed")
                if unused:
                    self.log(f"shared GPU(s) free to borrow: {', '.join(unused)}")
        unused, theirs = self._borrowable
        return [uuid for uuid in unused if uuid not in busy], theirs

    def _requests(self) -> list[tuple[queue.QueueEntry, plan.Request]]:
        """The queue as `plan` sees it. A job whose spec cannot be read is
        failed here: it is the one thing about a queued job that only the
        dispatcher can decide."""
        requests: list[tuple[queue.QueueEntry, plan.Request]] = []
        for entry in self.queued():
            try:
                spec = jobs.read_spec(entry.job_id)
            except (RuntimeError, ValueError) as exc:
                self.log(f"job {entry.job_id} has an unreadable spec ({exc}); dropping from queue")
                self._claim(
                    entry.job_id, status="failed", reason="bad-spec", ended_at=jobs.utc_now()
                )
                continue
            requests.append(
                (entry, plan.Request(entry.job_id, spec.gpus, self.config.may_borrow(spec)))
            )
        return requests

    def _pool(
        self, *, extra_owned: Sequence[str] = (), extra_shared: Sequence[str] = ()
    ) -> plan.Pool:
        """This pass's cards. `extra_*` are the ones a stop in flight will hand
        back, which `preempt_for_waiting` counts as free."""
        return plan.Pool(
            owned_free=[*self.free_gpus(), *extra_owned],
            owned_configured=len(self.config.gpus),
            shared_configured=len(self.config.shared_entries()),
            shared_visible=len(self.shared_gpus()),
            sample=self.borrowable_gpus,
            shared_extra=list(extra_shared),
        )

    def launch_ready(self) -> None:
        """Act on `plan`: start what fits, fail what never will, hold the rest.

        A job waiting for an owned card that has dropped off nvidia-smi holds
        like any other: `config.gpus` says the host has that card, so the host
        is misconfigured or broken, and idling the queue behind the job is how
        that gets noticed rather than quietly worked around.
        """
        if paths.draining_file().exists():
            return
        requests = self._requests()
        decisions = plan.plan([request for _, request in requests], self._pool())
        for decision in decisions:
            job_id = decision.job_id
            if isinstance(decision, plan.Fails):
                self._claim(
                    job_id,
                    status="failed",
                    reason=decision.reason,
                    exit_code=1,
                    ended_at=jobs.utc_now(),
                )
                continue
            if not isinstance(decision, plan.Assigned):
                continue
            self._launch(job_id, decision.gpus)

    def _launch(self, job_id: str, assigned: list[str]) -> None:
        if not self._claim(
            job_id, status="running", gpus=assigned, phase="setup", started_at=jobs.utc_now()
        ):
            # Cancelled between listing the queue and here; its cards stay free.
            return
        try:
            proc = self.deps.spawn_runner(job_id)
        except OSError as exc:
            # The state said `running` a line ago; leaving it there would leave
            # a job nothing is running, with no runner pid to notice the
            # absence of, holding its GPUs against every later pass.
            self.log(f"job {job_id}: could not spawn a runner ({exc})")
            jobs.update_state(
                job_id,
                status="failed",
                reason="spawn-failed",
                exit_code=1,
                ended_at=jobs.utc_now(),
                phase=None,
            )
            return
        # pgid stays unset until the runner publishes the *job's* group: it
        # is what a stop signals, and the runner's own group is not it.
        jobs.update_state(
            job_id,
            pid=proc.pid,
            pgid=None,
            runner_pid=proc.pid,
            runner_boot_id=boot_id(),
            runner_starttime=starttime(proc.pid),
        )
        self.running[job_id] = _Running(job_id, proc.pid, assigned, proc)
        shared = set(self.shared_gpus())
        borrowed = [uuid for uuid in assigned if uuid in shared]
        note = f", borrowing {','.join(borrowed)}" if borrowed else ""
        self.log(f"launched {job_id} (pid {proc.pid}) on {','.join(assigned) or 'cpu'}{note}")

    # -- automatic preemption --------------------------------------------
    def preempt_for_waiting(self) -> None:
        """Stop `auto_preempt` jobs when that starts a more important one now.

        After `launch_ready`, so everything still queued is something the free
        cards could not take, and the only question left is whether stopping a
        job that said it may be stopped would let one of them run.

        **Exactly one queued job is asked that question**: the first one the
        queue is actually stuck on. Anything behind it is not a reason to stop
        anything, and the priorities are not why -- the queue is in priority
        order, so a candidate less important than a job back there is less
        important than this one too. It is the cards: a card handed back is
        dispatched in queue order like any other, so this job takes it first,
        and it is short of more than every candidate could free, or it would
        have been the job stopped for. Stopping something for the job behind it
        discards an attempt and starts neither.

        What the walk does pass is a job that is not stuck: one already holding
        every card it needs, because a stop in flight is bringing them, and the
        one job the strict order steps over, which is short of a shared card
        somebody else is using and so holds nothing (`plan.SteppedOver`).

        Three things then have to hold for the job it settles on, and they are
        what keep this from being a way to lose work for nothing:

        * it has to be enough. A preempt that frees one of the two cards the
          waiting job needs costs an attempt and starts nothing.
        * the waiting job has to be *strictly* more important. At equal
          priority the stopped job's id is the older one, so it would win the
          tie, take its own cards straight back, and be preempted again on the
          next pass for ever.
        * the cards have to be ones the waiting job could be dispatched onto.
          A borrowed card handed back is no use to a job that did not ask to
          borrow, so stopping a job for it would spend an attempt on nothing.

        One job a pass is the whole of the rule, and the cost is a pass: where
        two waiting jobs each deserve a stop, the second gets its own next time
        round, once the first one's cards count as on their way. What it gives
        up outright is narrower -- a borrower behind a job that is stuck for
        good never has a *shared* card freed for it, though the job in front
        could not be dispatched onto that card anyway -- and winning that back
        costs a second model of the dispatch order, which is the thing this is
        deliberately not.

        What makes the second attempt of a preempted job wait its turn rather
        than take its own cards straight back is `launch_ready`: the queue is
        in priority order and a job that does not fit holds the cards it is
        waiting for, so nothing behind it -- the job just stopped very much
        included -- can be launched onto them.

        There is no limit on how often one job gives way: `auto_preempt` says
        it would rather start over than hold a card something better wants, and
        a host with a steady supply of better work may never run it at all.
        """
        if self._going_away() is not None:
            return
        candidates = self.auto_preemptable()
        if not candidates:
            return
        # Cards held by a job that is already stopping count as free: a gap the
        # cards of a preempt already in flight will cover needs no second job
        # stopped for it. A card that has dropped off nvidia-smi is in neither
        # list, since it is never handed out at all.
        owned = set(self.owned_gpus())
        shared = set(self.shared_gpus())
        stopping = [
            uuid
            for job_id, entry in self.running.items()
            if self._stopping(job_id)
            for uuid in entry.gpus
        ]
        requests = self._requests()
        by_id = {request.job_id: (entry, request) for entry, request in requests}
        pool = self._pool(
            extra_owned=[u for u in stopping if u in owned],
            extra_shared=[u for u in stopping if u in shared],
        )
        for decision in plan.plan([request for _, request in requests], pool):
            if not isinstance(decision, plan.Holds):
                continue
            waiting, request = by_id[decision.job_id]
            for candidate in enough_to_start(
                candidates, waiting.priority, decision.gap, borrowing=request.borrows
            ):
                if not self._preempt_for(candidate, waiting):
                    # The gap is not covered any more, so the rest of the set
                    # would be attempts spent on a job that still cannot start.
                    break
            return

    def auto_preemptable(self) -> list[Preemptable]:
        """The running jobs whose spec said they may be stopped for better work.

        Never one that is already stopping: its cards are counted as coming
        free instead, and a second kill request would say nothing new.
        """
        found: list[Preemptable] = []
        owned = set(self.owned_gpus())
        shared = set(self.shared_gpus())
        for job_id, entry in self.running.items():
            if self._stopping(job_id):
                continue
            try:
                spec = jobs.read_spec(job_id)
            except (RuntimeError, ValueError):
                continue
            state = self._state_or_empty(job_id)
            if not spec.auto_preempt or state.status != "running":
                continue
            found.append(
                Preemptable(
                    job_id,
                    state.priority,
                    list(entry.gpus),
                    state.started_at or "",
                    owned=sum(1 for uuid in entry.gpus if uuid in owned),
                    borrowed=sum(1 for uuid in entry.gpus if uuid in shared),
                )
            )
        return found

    def _stopping(self, job_id: str) -> bool:
        """Already asked to stop, so its cards are on their way back anyway."""
        return self._state_or_empty(job_id).intent is not None

    def _preempt_for(self, candidate: Preemptable, waiting: queue.QueueEntry) -> bool:
        """Stop one auto-preemptable job, saying in both logs who took its place.

        The job's own log because that is where somebody looks at output that
        stops mid-run; a failure is only the dispatcher's, since the job itself
        is untouched and still running.
        """
        try:
            queue.preempt(candidate.job_id)
        except (ValueError, RuntimeError, OSError) as exc:
            self.log(
                f"job {candidate.job_id} is auto_preempt but was not stopped for "
                f"{waiting.job_id} ({exc}); it keeps its cards"
            )
            return False
        self.log(
            f"job {candidate.job_id} (auto_preempt, priority {candidate.priority}) is being "
            f"stopped so job {waiting.job_id} (priority {waiting.priority}) can have its "
            f"{len(candidate.gpus)} card(s)"
        )
        queue.note(
            candidate.job_id,
            f"auto_preempt: stopping so job {waiting.job_id} (priority {waiting.priority}) "
            f"can have these GPUs",
        )
        return True

    # -- terminate ------------------------------------------------------
    def maybe_terminate(self) -> None:
        config = self.config
        if not config.ephemeral:
            return
        now = self.deps.monotonic()
        if self._terminate_retry_at is not None and now < self._terminate_retry_at:
            return
        if self.running:
            self._queue_empty_since = None
            return
        if self.queued():
            self._queue_empty_since = None
            return
        if self._queue_empty_since is None:
            self._queue_empty_since = now
        idle_s = now - self._queue_empty_since
        if idle_s >= config.idle_minutes * 60.0:
            self.drain_and_terminate(f"idle for {idle_s / 60.0:.1f} min")

    def drain_and_terminate(self, why: str) -> bool:
        config = self.config
        self.log(f"draining: {why}")
        jobs.atomic_write_text(paths.draining_file(), f"{why}\n")
        self._guard(lambda: self.retry_unconfirmed_outputs(config), "drain: output retry")
        # Mirroring state is best-effort, and terminating is not: a bucket we
        # cannot reach is not a reason to keep a paid pod alive. Treated as a
        # terminate failure it kept the host up, retrying every ten minutes
        # with a fresh heartbeat -- for as long as the credentials stayed
        # broken, which is forever.
        self._guard(lambda: self._final_sync_all(config), "drain: final state mirror")
        try:
            terminate.self_terminate(config, terminate_call=self.deps.terminate_call)
        except terminate.TerminateError as exc:
            paths.draining_file().unlink(missing_ok=True)
            self._terminate_retry_at = self.deps.monotonic() + TERMINATE_RETRY_S
            self.log(
                f"SELF-TERMINATE FAILED ({exc}); staying up and dispatching, "
                f"retrying in {TERMINATE_RETRY_S / 60:.0f} min"
            )
            return False
        self.log("terminate requested; waiting for the provider to stop this host")
        self.should_exit = True
        return True

    def _final_sync_all(self, config: jobs.HostConfig) -> None:
        if not config.s3_prefix:
            return
        errors: list[str] = []
        for job_id in jobs.list_job_ids():
            try:
                warning = sync.final_meta_sync(
                    job_id, config.s3_prefix, runner=self.deps.command_runner
                )
            except sync.SyncError as exc:
                errors.append(str(exc))
                continue
            if warning:
                self.log(f"job {job_id}: {warning}")
        if errors:
            raise sync.SyncError("; ".join(errors))

    # -- outputs the pod would otherwise take with it --------------------
    def unconfirmed_output_jobs(self) -> list[str]:
        """Jobs whose `outputs:` are still only on this host."""
        pending: list[str] = []
        for job_id in jobs.list_job_ids():
            try:
                state = jobs.read_state(job_id)
            except RuntimeError:
                continue
            if state.status == "running":
                # Still being written to; the runner owns those files. Every
                # other status is fair game -- including `queued`, which is
                # what a preempted job is while still holding the outputs the
                # stopped attempt produced.
                continue
            # A job that failed its sync preflight proved these uploads cannot
            # work *before* it ran, and produced nothing. Retrying it three
            # times here only burns the budget the jobs with real outputs need.
            if state.reason == "sync-preflight":
                continue
            if not cleanup.outputs_confirmed(job_id, state)[0]:
                pending.append(job_id)
        return pending

    def retry_unconfirmed_outputs(self, config: jobs.HostConfig) -> None:
        """One last attempt to upload what a terminating host is still holding.

        Bounded on purpose: three tries a minute apart, five minutes in total.
        A pod that cannot reach S3 now is billing while it tries -- so a job
        whose outputs still will not go up is marked `outputs_lost` and the
        host terminates anyway.
        """
        pending = self.unconfirmed_output_jobs()
        if not pending:
            return
        self.log(f"drain: {len(pending)} job(s) have unconfirmed outputs; retrying")
        deadline = self.deps.monotonic() + OUTPUT_RETRY_BUDGET_S
        last_error: dict[str, str] = {}
        for attempt in range(1, OUTPUT_RETRY_ATTEMPTS + 1):
            for job_id in list(pending):
                try:
                    # Bounded by what is left of the budget: an upload with no
                    # timeout at all can hang on a half-open socket for hours,
                    # and every one of them is billed.
                    self._upload_outputs(job_id, config, max(1.0, deadline - self.deps.monotonic()))
                except (sync.SyncError, RuntimeError, OSError) as exc:
                    last_error[job_id] = str(exc)
                    continue
                pending.remove(job_id)
                with contextlib.suppress(RuntimeError, OSError, KeyError):
                    jobs.update_state(job_id, outputs_lost=False)
                self.log(f"drain: job {job_id} outputs uploaded on attempt {attempt}")
            if not pending or attempt == OUTPUT_RETRY_ATTEMPTS:
                break
            remaining = deadline - self.deps.monotonic()
            if remaining <= 0:
                break
            self.deps.sleep(min(OUTPUT_RETRY_INTERVAL_S, remaining))
        for job_id in pending:
            error = last_error.get(job_id, "outputs were never confirmed uploaded")
            self.log(f"drain: job {job_id} OUTPUTS LOST: {error}")
            with contextlib.suppress(RuntimeError, OSError, KeyError):
                jobs.update_state(job_id, outputs_lost=True)

    def _upload_outputs(self, job_id: str, config: jobs.HostConfig, timeout: float) -> None:
        spec = jobs.read_spec(job_id)
        # The job's own secrets file if the runner left it (it does exactly for
        # this case on an ephemeral host); otherwise our own environment, which
        # already carries the host env.
        env: dict[str, str] | None = None
        env_file = paths.job_env_file(job_id)
        if env_file.exists():
            env = config.apply_env(dict(os.environ))
            env.update(jobs.parse_env_file(env_file))
        sync.sync_outputs(
            spec.outputs,
            paths.workdir(job_id),
            job_id,
            min_age_s=0.0,
            runner=self.deps.command_runner,
            timeout=timeout,
            env=env,
            baseline_map=baseline.read(job_id),
        )

    # -- retention -------------------------------------------------------
    def sweep_stale_incoming(self) -> None:
        """Remove job dirs a `gpuc submit` started building and never accepted.

        The client owns a job until `enqueue` renames its dir into `jobs/`, so
        a dir still under `incoming/` an hour later is a submit that died, and
        the host was never asked to run it. Hourly, like the other reclaims.
        """
        now = self.deps.monotonic()
        if (
            self._last_incoming_sweep_at is not None
            and now - self._last_incoming_sweep_at < RETENTION_INTERVAL_S
        ):
            return
        self._last_incoming_sweep_at = now
        for name in cleanup.remove_stale_incoming():
            self.log(f"removed incoming/{name}: a submit that never finished enqueueing it")

    def maybe_reclaim(self) -> None:
        """Run the two retention horizons at startup, then at most once an hour.

        They are separate because they cost different things. `workdir_days`
        reclaims only what `gpuc requeue` can rebuild from git, so it is short
        by default and needs no mirror; `retention_days` deletes the record of
        a run, so it is long, opt-in, and never forced -- an automatic sweep
        that could bin the only copy of a job's log is not something anyone
        should have to opt out of.

        Both passes go through `automatic=True`, which is what keeps a sweep
        nobody asked for off a job that said `cleanup: never` or whose
        `outputs:` have not reached the mirror yet.

        Purge first: it takes whole job dirs, and the workdir sweep afterwards
        should not spend its report on dirs that are already gone.
        """
        now = self.deps.monotonic()
        if self._last_reclaim_at is not None and now - self._last_reclaim_at < RETENTION_INTERVAL_S:
            return
        self._last_reclaim_at = now
        purge_days = self.config.retention_days
        workdir_days = self.config.workdir_days
        if purge_days is not None:
            self._purge(purge_days)
        if workdir_days is not None:
            self._sweep_workdirs(workdir_days)

    def _purge(self, days: float) -> None:
        result = cleanup.purge(
            older_than_days=days, now=self.deps.utcnow(), evidence=cleanup.Evidence(automatic=True)
        )
        if result.purged or result.removed:
            purged = ", ".join(c.job_id for c in result.purged) or "none"
            self.log(
                f"retention ({days:g} days): purged {len(result.purged)} job dir(s) and "
                f"{len(result.removed)} workdir(s), freeing "
                f"{cleanup.human_bytes(result.freed_bytes)}; purged: {purged}"
            )
        for error in result.errors:
            self.log(f"retention: {error}")

    def _sweep_workdirs(self, days: float) -> None:
        result = cleanup.clean(
            older_than_days=days, now=self.deps.utcnow(), evidence=cleanup.Evidence(automatic=True)
        )
        if result.removed:
            self.log(
                f"workdirs ({days:g} days): removed {len(result.removed)} workdir(s), freeing "
                f"{cleanup.human_bytes(result.freed_bytes)}; "
                f"{', '.join(c.job_id for c in result.removed)}"
            )
        for error in result.errors:
            self.log(f"workdirs: {error}")

    # -- main ------------------------------------------------------------
    def run_once(self) -> None:
        self._config = jobs.read_config()
        self._owned = None
        self._shared = None
        self._borrowable = None
        self._queued = None
        self.reap()
        self.escalate_stops()
        self.launch_ready()
        self.preempt_for_waiting()
        self.maybe_reclaim()
        self.sweep_stale_incoming()
        self.maybe_terminate()

    def idle_and_not_ephemeral(self) -> bool:
        return not self.running and not self.queued() and not self.config.ephemeral

    def run(self, lock: DispatcherLock) -> int:
        lock.start_heartbeat()
        self.log(f"dispatcher started (pid {os.getpid()}, pgid {os.getpgid(0)})")
        code = 0
        try:
            self._guard(self.adopt_orphans)
            while not self.should_exit:
                lock.beat()
                healthy = self._guard(self.run_once)
                if self.consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    code = 1
                    break
                # "Nothing to do" is only trustworthy from an iteration that
                # actually ran: a failing one knows nothing about the queue.
                if healthy and self.idle_and_not_ephemeral():
                    break
                self.deps.sleep(self.deps.interval_s)
        finally:
            lock.release()
        self.log("dispatcher exiting")
        return code

    def _guard(self, step: Callable[[], None], label: str | None = None) -> bool:
        """Run one loop step, returning whether it succeeded. A bug in one
        iteration must not take the queue down, but an error that repeats
        forever is not worth spinning on."""
        try:
            step()
        except Exception:
            self.consecutive_failures += 1
            self.log(
                f"{label or getattr(step, '__name__', step)} failed "
                f"({self.consecutive_failures}/{MAX_CONSECUTIVE_FAILURES}): "
                f"{traceback.format_exc()}"
            )
            if self.consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                self.log(
                    f"GIVING UP: {MAX_CONSECUTIVE_FAILURES} consecutive dispatcher failures. "
                    f"No further jobs will be launched until a dispatcher is restarted; "
                    f"running jobs are untouched. See the traceback above."
                )
            return False
        self.consecutive_failures = 0
        return True


def _stop_on_sigterm(dispatcher: Dispatcher) -> None:
    """Finish the pass, then exit -- the polite half of a handoff.

    A dispatcher asked to stand down for another build is SIGTERMed before it
    is killed (`DispatcherLock.acquire`), and this is what makes the asking
    worth anything: the loop finishes what it is doing, releases the lock, and
    the newcomer takes it without a signal landing between two writes. The
    window that matters is in `launch_ready`, between the state that says a job
    is running and the runner pid that proves it: a dispatcher killed there
    leaves a job the next one reads as `runner-died`.

    Installed before the lock is taken, not after: the several seconds
    `acquire` can spend negotiating are exactly when the SIGTERM of a *third*
    dispatcher may arrive, and dying there with the lock held helps nobody.
    Setting a flag no loop is reading yet is harmless.

    It cannot help on the first takeover of any host, whose incumbent is by
    definition a build without this handler and dies where it stands.
    """

    def stop(_signum: int, _frame: object) -> None:
        dispatcher.should_exit = True

    with contextlib.suppress(ValueError, OSError):
        signal.signal(signal.SIGTERM, stop)


def main(_: object = None) -> int:
    dispatcher = Dispatcher()
    _stop_on_sigterm(dispatcher)
    lock = DispatcherLock()
    if not lock.acquire():
        return 0
    return dispatcher.run(lock)
