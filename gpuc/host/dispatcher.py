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

from gpuc.host import baseline, cleanup, gpus, jobs, paths, queue, scope, sync, terminate
from gpuc.host.gpus import SmiRunner
from gpuc.host.runner import (
    boot_id,
    cmdline,
    is_gpuc_process,
    pid_alive,
    process_group_alive,
    recorded_process_alive,
    starttime,
)
from gpuc.host.terminate import TerminateCall

HEARTBEAT_INTERVAL_S = 5.0
HEARTBEAT_STALE_S = 30.0
LOOP_INTERVAL_S = 2.0
KILL_GRACE_S = 15.0
TERMINATE_RETRY_S = 600.0
MAX_CONSECUTIVE_FAILURES = 20
RETENTION_INTERVAL_S = 3600.0
"""How often a dispatcher with `retention_days` set runs the purge.

Once at startup and then hourly: deleting week-old job dirs is not urgent, and
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

    def render(self) -> str:
        return json.dumps(asdict(self), sort_keys=True) + "\n"

    @staticmethod
    def parse(text: str) -> LockBody:
        text = text.strip()
        if not text:
            return LockBody()
        try:
            document = json.loads(text)
        except json.JSONDecodeError:
            return LockBody()
        if not isinstance(document, dict):
            return LockBody()
        return LockBody(
            pid=_maybe_int(document.get("pid")),
            pgid=_maybe_int(document.get("pgid")),
            starttime=_maybe_str(document.get("starttime")),
            boot_id=_maybe_str(document.get("boot_id")),
        )


def _maybe_int(value: object) -> int | None:
    """A pid from whatever the lock file holds, or None.

    Tolerant on purpose: this file is written by whichever build of gpuc last
    took the lock, and a pid we cannot read means "no known holder" -- which
    the caller already handles -- not a crash on the way to taking over.
    """
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float | str):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None
    return None


def _maybe_str(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


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
        try:
            return LockBody.parse(self.lock_path.read_text())
        except OSError:
            return LockBody()

    def acquire(self, takeover_wait_s: float = 10.0) -> bool:
        paths.ensure_layout()
        fd = os.open(self.lock_path, os.O_RDWR | os.O_CREAT, 0o644)
        if self._try_flock(fd):
            self._adopt(fd)
            return True
        if self.holder_is_fresh():
            os.close(fd)
            return False
        self._evict_stale_holder()
        deadline = self._now() + takeover_wait_s
        while self._now() < deadline:
            if self._try_flock(fd):
                self._adopt(fd)
                return True
            self._sleep(0.25)
        os.close(fd)
        return False

    def _evict_stale_holder(self) -> None:
        """Kill the incumbent only when it is provably a wedged gpuc dispatcher.

        A stale heartbeat on its own is not enough: the pid in the lock file may
        belong to something else entirely by now, and killing a process group we
        do not own would take out an innocent bystander's shell and its jobs.
        """
        body = self.holder()
        if body.pid is None:
            log_line("lock is held with a stale heartbeat but records no pid; killing nothing")
            return
        if not recorded_process_alive(body.pid, body.boot_id, body.starttime):
            log_line(f"lock holder pid {body.pid} is gone; taking over")
            return
        if not is_gpuc_process(body.pid):
            log_line(
                f"pid {body.pid} holds the lock with a stale heartbeat but is not a gpuc "
                f"dispatcher ({cmdline(body.pid)!r}); killing nothing"
            )
            return
        pgid = body.pgid
        if pgid != body.pid or pgid is None:
            log_line(
                f"lock holder pid {body.pid} records pgid {pgid}; only a process group "
                f"led by the dispatcher itself is ever killed"
            )
            return
        if pgid in (os.getpgid(0), os.getpid()):
            return
        self.takeover_pgid = pgid
        log_line(f"heartbeat is stale; SIGKILLing wedged dispatcher process group {pgid}")
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(pgid, signal.SIGKILL)

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


def default_spawn_runner(job_id: str) -> subprocess.Popen[bytes]:
    package_root = Path(__file__).resolve().parents[2]
    env = _child_env(package_root)
    log = paths.dispatcher_log().open("ab", buffering=0)
    try:
        return subprocess.Popen(
            [sys.executable, "-m", "gpuc.host", "run", job_id],
            cwd=str(package_root),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    finally:
        log.close()


def spawn_detached_dispatcher() -> int:
    """Start a dispatcher that outlives this process (and any SSH session).

    Always safe to call: a second dispatcher exits silently when the incumbent
    heartbeat is fresh, so enqueue can fire this unconditionally.
    """
    package_root = Path(__file__).resolve().parents[2]
    env = _child_env(package_root)
    paths.ensure_layout()
    with paths.dispatcher_log().open("ab", buffering=0) as log:
        proc = subprocess.Popen(
            [sys.executable, "-m", "gpuc.host", "dispatch"],
            cwd=str(package_root),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    return proc.pid


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
class Dispatcher:
    deps: DispatcherDeps = field(default_factory=DispatcherDeps)
    running: dict[str, _Running] = field(default_factory=dict)
    _queue_empty_since: float | None = None
    _cancel_sent: dict[str, float] = field(default_factory=dict)
    _terminate_retry_at: float | None = None
    _last_purge_at: float | None = None
    _kill_sent: dict[str, float] = field(default_factory=dict)
    _kill_escalated: set[str] = field(default_factory=set)
    _pause_drain_pending: bool = False
    _config: jobs.HostConfig | None = None
    _owned: list[str] | None = None
    _unavailable: tuple[str, ...] = ()
    consecutive_failures: int = 0
    should_exit: bool = False

    @property
    def config(self) -> jobs.HostConfig:
        """The host config, read once per loop pass.

        One pass asks for it a dozen times (free GPUs, the TTL, the idle timer,
        the drain); re-reading and re-parsing the file each time bought nothing
        but syscalls, and a mid-pass change is not something any of those
        decisions should straddle.
        """
        if self._config is None:
            self._config = jobs.read_config()
        return self._config

    def log(self, message: str) -> None:
        log_line(message, self.deps.utcnow())

    # -- startup ---------------------------------------------------------
    def adopt_orphans(self) -> None:
        """Reconcile jobs left `running` by a dispatcher that died."""
        for job_id in jobs.list_job_ids():
            try:
                state = jobs.read_state(job_id)
            except RuntimeError as exc:
                self.log(f"job {job_id} has an unreadable state.json ({exc}); skipping")
                continue
            if state.status != "running" or job_id in self.running:
                continue
            # A recorded pid means nothing across a reboot, and little after a
            # pid rollover: the boot id and start time recorded at launch are
            # what make "still running" a real answer.
            if state.runner_pid and recorded_process_alive(
                state.runner_pid, state.runner_boot_id, state.runner_starttime
            ):
                # Through the resolver: a job launched before assignments were
                # resolved host-side has indices in its state, and busy/free
                # accounting is in UUIDs. An index adopted as-is would match
                # nothing owned, so the card would read free and be handed out
                # a second time while the job is still training on it.
                try:
                    held, _ = gpus.resolve_owned(state.gpus, self.deps.smi)
                except gpus.GpuError as exc:
                    # Adoption runs once, at startup: a job left unadopted here
                    # is never picked up, so an unreadable nvidia-smi must cost
                    # the resolution, not the adoption.
                    self.log(f"could not resolve the GPUs of {job_id} ({exc}); adopting as given")
                    held = list(state.gpus)
                self.running[job_id] = _Running(job_id, state.runner_pid, held)
                self.log(f"adopted running job {job_id} (runner pid {state.runner_pid})")
            else:
                self._mark_runner_died(job_id)

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
        self._kill_orphaned_group(job_id, state.pgid if state else None, state)
        final = state or jobs.JobState()
        final.status = "failed"
        final.reason = "runner-died"
        final.exit_code = final.exit_code or 1
        final.ended_at = jobs.utc_now()
        final.phase = None
        jobs.write_state(job_id, final)
        self.log(f"job {job_id} failed: runner died without writing final state")

    def _kill_orphaned_group(
        self, job_id: str, pgid: int | None, state: jobs.JobState | None = None
    ) -> None:
        unit = state.cgroup_unit if state else None
        if unit:
            self.log(f"job {job_id}: stopping leftover scope {unit} before freeing its GPUs")
            scope.stop_unit(unit)
        if not pgid or pgid <= 1 or pgid == os.getpgid(0):
            return
        if not process_group_alive(pgid):
            return
        self.log(f"job {job_id}: SIGKILLing orphaned process group {pgid} before freeing its GPUs")
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(pgid, signal.SIGKILL)

    # -- loop pieces -----------------------------------------------------
    def reap(self) -> None:
        for job_id, entry in list(self.running.items()):
            if entry.poll() is None:
                continue
            del self.running[job_id]
            self._cancel_sent.pop(job_id, None)
            self._kill_sent.pop(job_id, None)
            self._kill_escalated.discard(job_id)
            try:
                state = jobs.read_state(job_id)
            except RuntimeError:
                self._mark_runner_died(job_id)
                continue
            if not state.finished:
                self._mark_runner_died(job_id)
            else:
                self.log(
                    f"job {job_id} {state.status}"
                    f"{f' ({state.reason})' if state.reason else ''} "
                    f"exit={state.exit_code}"
                )

    def handle_cancels(self) -> None:
        """Escalate a cancel, without ever killing the runner's own group during
        the launch window.

        Between spawn and the first phase the runner *is* the only member of its
        group, so signalling it there would kill the one process that can finish
        the job cleanly. Until the job's own pgid appears in state.json the
        cancel marker is the whole mechanism; the runner checks it before every
        phase and ends as `cancelled`.
        """
        now = self.deps.monotonic()
        grace = self.deps.kill_grace_s
        for job_id, entry in list(self.running.items()):
            if not queue.is_cancelled(job_id):
                continue
            sent = self._cancel_sent.get(job_id)
            if sent is None:
                self._cancel_sent[job_id] = now
                state = self._state_or_empty(job_id)
                job_pgid = self._job_pgid(state, entry)
                if state.cgroup_unit:
                    # A cgroup stop reaps the whole tree, daemonised
                    # grandchildren included; the group kill below cannot.
                    self.log(f"cancelling job {job_id} (scope {state.cgroup_unit})")
                    scope.stop_unit(state.cgroup_unit)
                elif job_pgid:
                    self.log(f"cancelling job {job_id} (pgid {job_pgid})")
                    self._signal_group(job_pgid, signal.SIGTERM)
                else:
                    self.log(
                        f"cancelling job {job_id}: the runner has not published a job process "
                        f"group yet, so the cancel marker alone stops it"
                    )
                continue
            self._escalate(job_id, entry, now - sent, grace)

    def escalate_kills(self) -> None:
        """Make a kill *request* stick when the runner never acts on it.

        A TTL (or a low-util pause) asks the runner to stop its job and sync,
        which is right when the runner is healthy and is nothing at all when it
        is wedged: the marker sits there, the job keeps running, and an
        ephemeral host that should have died hours ago keeps billing with a
        fresh heartbeat. So the ask gets the same ladder a cancel gets.
        """
        now = self.deps.monotonic()
        grace = self.deps.kill_grace_s
        for job_id, entry in list(self.running.items()):
            sent = self._kill_sent.get(job_id)
            # A cancelled job already has an escalation, and one owner is enough.
            if sent is None or queue.is_cancelled(job_id):
                continue
            elapsed = now - sent
            if elapsed <= grace:
                continue
            if job_id not in self._kill_escalated:
                self._kill_escalated.add(job_id)
                self.log(
                    f"job {job_id}: its runner has not stopped it {elapsed:.0f}s after the "
                    f"{queue.kill_reason(job_id) or 'kill'} request; escalating"
                )
            self._escalate(job_id, entry, elapsed, grace)

    def _escalate(self, job_id: str, entry: _Running, elapsed: float, grace: float) -> None:
        """The kill ladder: the job's scope and group first, the runner last.

        The runner handles SIGTERM itself (final sync, final state), so it gets
        a signal of its own -- and the time to use it -- before its group goes.
        """
        state = self._state_or_empty(job_id)
        job_pgid = self._job_pgid(state, entry)
        if state.cgroup_unit and elapsed > grace:
            scope.stop_unit(state.cgroup_unit)
        if job_pgid and elapsed > grace:
            self._signal_group(job_pgid, signal.SIGKILL)
        if elapsed > 2 * grace:
            self._signal_pid(entry.pid, signal.SIGTERM)
        if elapsed > 3 * grace and process_group_alive(entry.pid):
            self.log(f"job {job_id}: runner pid {entry.pid} is wedged; SIGKILLing its group")
            self._signal_group(entry.pid, signal.SIGKILL)

    @staticmethod
    def _state_or_empty(job_id: str) -> jobs.JobState:
        try:
            return jobs.read_state(job_id)
        except RuntimeError:
            return jobs.JobState()

    @staticmethod
    def _job_pgid(state: jobs.JobState, entry: _Running) -> int | None:
        """The *job's* process group, never the runner's own."""
        return state.pgid if state.pgid and state.pgid != entry.pid else None

    def _signal_pid(self, pid: int, sig: int) -> None:
        if pid <= 1 or not pid_alive(pid):
            return
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.kill(pid, sig)

    def _signal_group(self, pgid: int | None, sig: int) -> None:
        if not pgid or pgid <= 1 or not process_group_alive(pgid):
            return
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(pgid, sig)

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

    def free_gpus(self) -> list[str]:
        busy = {uuid for entry in self.running.values() for uuid in entry.gpus}
        return [uuid for uuid in self.owned_gpus() if uuid not in busy]

    def launch_ready(self) -> None:
        if self.paused() or paths.draining_file().exists():
            return
        # The *configured* count, not the resolved one: a card that is missing
        # this minute makes a job wait, but it must not permanently fail a job
        # the host is perfectly well configured to run.
        owned = self.config.gpus
        free = self.free_gpus()
        for entry in queue.list_queued():
            job_id = entry.job_id
            if queue.is_cancelled(job_id):
                entry.marker.unlink(missing_ok=True)
                jobs.update_state(
                    job_id, status="cancelled", reason="cancelled", ended_at=jobs.utc_now()
                )
                continue
            try:
                spec = jobs.read_spec(job_id)
            except RuntimeError as exc:
                self.log(f"job {job_id} has an unreadable spec ({exc}); dropping from queue")
                entry.marker.unlink(missing_ok=True)
                jobs.update_state(
                    job_id, status="failed", reason="bad-spec", ended_at=jobs.utc_now()
                )
                continue
            if spec.gpus > len(owned):
                entry.marker.unlink(missing_ok=True)
                jobs.update_state(
                    job_id,
                    status="failed",
                    reason=f"needs {spec.gpus} GPUs, host owns {len(owned)}",
                    exit_code=1,
                    ended_at=jobs.utc_now(),
                )
                continue
            if spec.gpus > len(free):
                continue
            assigned, free = free[: spec.gpus], free[spec.gpus :]
            entry.marker.unlink(missing_ok=True)
            jobs.update_state(
                job_id,
                status="running",
                gpus=assigned,
                phase="setup",
                started_at=jobs.utc_now(),
            )
            try:
                proc = self.deps.spawn_runner(job_id)
            except OSError as exc:
                # The state said `running` a line ago; leaving it there would
                # leave a job nothing is running, with no runner pid to notice
                # the absence of, holding its GPUs against every later pass.
                self.log(f"job {job_id}: could not spawn a runner ({exc})")
                jobs.update_state(
                    job_id,
                    status="failed",
                    reason="spawn-failed",
                    exit_code=1,
                    ended_at=jobs.utc_now(),
                    phase=None,
                )
                free = [*assigned, *free]
                continue
            # pgid stays unset until the runner publishes the *job's* group: it
            # is what `cancel` signals, and the runner's own group is not it.
            jobs.update_state(
                job_id,
                pid=proc.pid,
                pgid=None,
                runner_pid=proc.pid,
                runner_boot_id=boot_id(),
                runner_starttime=starttime(proc.pid),
            )
            self.running[job_id] = _Running(job_id, proc.pid, assigned, proc)
            self.log(
                f"launched {job_id} (pid {proc.pid}) on {','.join(assigned) if assigned else 'cpu'}"
            )

    # -- pause / terminate ----------------------------------------------
    def paused(self) -> bool:
        return paths.paused_file().exists()

    def recent_low_util_failures(self, count: int = 2) -> bool:
        finished: list[tuple[str, jobs.JobState]] = []
        for job_id in jobs.list_job_ids():
            try:
                state = jobs.read_state(job_id)
            except RuntimeError:
                continue
            if state.finished and state.ended_at:
                finished.append((state.ended_at, state))
        finished.sort(key=lambda pair: pair[0], reverse=True)
        latest = [state for _, state in finished[:count]]
        return len(latest) == count and all(
            s.status == "failed" and s.reason == "low-util" for s in latest
        )

    def check_pause(self) -> None:
        """Pause on two consecutive low-util failures, and on an ephemeral host
        go away afterwards -- but never out from under a job that is still
        running. Draining there terminated the pod with the other jobs' runners
        still working: no kill marker, no final sync, outputs gone with the pod.
        Like the TTL, we ask; the drain happens on a later pass with nothing
        left running.
        """
        if not self.paused():
            if not self.recent_low_util_failures():
                return
            jobs.atomic_write_text(
                paths.paused_file(),
                "two consecutive jobs failed with reason low-util; queue paused\n",
            )
            self.log("PAUSED: two consecutive low-util failures; not dispatching further jobs")
            self._pause_drain_pending = self.config.ephemeral
        if not self._pause_drain_pending:
            return
        if self.running:
            self._request_kills("low-util-pause", "the queue paused on low utilization")
            return
        self._pause_drain_pending = False
        self.drain_and_terminate("two consecutive low-util failures")

    def maybe_terminate(self) -> None:
        config = self.config
        if not config.ephemeral:
            return
        now = self.deps.monotonic()
        if self._terminate_retry_at is not None and now < self._terminate_retry_at:
            return
        # The TTL is a hard cap, so it is checked before anything that returns
        # early on a busy host: the reaper used to be the only thing that
        # enforced it, and it terminates a pod out from under a running job
        # without a final sync.
        if self._ttl_expired(config):
            if self.running:
                self._request_kills("ttl", f"the ttl of {config.ttl_hours:g} h elapsed")
                return
            self.drain_and_terminate(f"ttl of {config.ttl_hours:g} h elapsed")
            return
        if self.running:
            self._queue_empty_since = None
            return
        if queue.list_queued():
            self._queue_empty_since = None
            return
        if self._queue_empty_since is None:
            self._queue_empty_since = now
        idle_s = now - self._queue_empty_since
        if idle_s >= config.idle_minutes * 60.0:
            self.drain_and_terminate(f"idle for {idle_s / 60.0:.1f} min")

    def _request_kills(self, reason: str, why: str) -> None:
        """Stop the running jobs so their runners can sync before we terminate.

        The runner owns the kill and the final sync, so this asks rather than
        signals: each job ends `failed: <reason>` with its outputs uploaded, and
        the next pass -- with nothing running -- drains and terminates. When the
        ask goes unanswered `escalate_kills` stops being polite.
        """
        now = self.deps.monotonic()
        for job_id in sorted(self.running):
            if queue.kill_reason(job_id):
                # A marker from before this dispatcher took over still needs a
                # clock, or nothing would ever escalate it.
                self._kill_sent.setdefault(job_id, now)
                continue
            queue.request_kill(job_id, reason)
            self._kill_sent[job_id] = now
            self.log(
                f"{why} with job {job_id} running: asked its runner to stop it "
                f"(reason {reason}) and sync before this host terminates"
            )

    def _ttl_expired(self, config: jobs.HostConfig) -> bool:
        """Null `ttl_hours` -- the default -- never expires.

        An overall TTL is opt-in precisely because the failure it causes (a
        training run killed at hour 24) is worse than the one it prevents.
        """
        if config.ttl_hours is None or not config.created_at:
            return False
        try:
            created = datetime.fromisoformat(config.created_at)
        except ValueError:
            return False
        if created.tzinfo is None:
            created = created.replace(tzinfo=UTC)
        age_h = (self.deps.utcnow() - created).total_seconds() / 3600.0
        return age_h >= config.ttl_hours

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
        """Finished jobs whose `outputs:` are still only on this host."""
        pending: list[str] = []
        for job_id in jobs.list_job_ids():
            try:
                state = jobs.read_state(job_id)
            except (RuntimeError, OSError):
                continue
            if not state.finished:
                continue
            # A job that failed its sync preflight proved these uploads cannot
            # work *before* it ran, and produced nothing. Retrying it three
            # times here only burns the budget the jobs with real outputs need.
            if (state.reason or "").startswith("sync-preflight"):
                continue
            if not cleanup.outputs_confirmed(job_id, state)[0]:
                pending.append(job_id)
        return pending

    def retry_unconfirmed_outputs(self, config: jobs.HostConfig) -> None:
        """One last attempt to upload what a terminating host is still holding.

        Bounded on purpose: three tries a minute apart, five minutes in total.
        A pod that cannot reach S3 now is billing while it tries, and the TTL
        that sent us here does not pause -- so a job whose outputs still will
        not go up is marked `outputs_lost` and the host terminates anyway.
        """
        pending = self.unconfirmed_output_jobs()
        if not pending:
            return
        self.log(f"drain: {len(pending)} finished job(s) have unconfirmed outputs; retrying")
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
                    jobs.update_state(job_id, outputs_synced_at=jobs.utc_now(), outputs_lost=False)
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
                jobs.update_state(job_id, outputs_lost=True, sync_error=error)

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
    def maybe_purge(self) -> None:
        """Run the retention purge at startup, then at most once an hour.

        Never forced: an automatic sweep that could delete the only copy of a
        job's log is not something anyone should have to opt out of.
        """
        days = self.config.retention_days
        if days is None:
            return
        now = self.deps.monotonic()
        if self._last_purge_at is not None and now - self._last_purge_at < RETENTION_INTERVAL_S:
            return
        self._last_purge_at = now
        result = cleanup.purge(older_than_days=days, now=self.deps.utcnow())
        if result.purged or result.removed:
            purged = ", ".join(c.job_id for c in result.purged) or "none"
            self.log(
                f"retention ({days:g} days): purged {len(result.purged)} job dir(s) and "
                f"{len(result.removed)} workdir(s), freeing "
                f"{cleanup.human_bytes(result.freed_bytes)}; purged: {purged}"
            )
        for error in result.errors:
            self.log(f"retention: {error}")

    # -- main ------------------------------------------------------------
    def run_once(self) -> None:
        self._config = jobs.read_config()
        self._owned = None
        self.reap()
        self.handle_cancels()
        self.escalate_kills()
        self.check_pause()
        self.launch_ready()
        self.maybe_purge()
        self.maybe_terminate()

    def idle_and_not_ephemeral(self) -> bool:
        return not self.running and not queue.list_queued() and not self.config.ephemeral

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


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="gpuc.host dispatch")
    parser.add_argument("--once", action="store_true", help="run a single loop iteration")
    parser.add_argument("--interval", type=float, default=LOOP_INTERVAL_S)
    args = parser.parse_args(list(argv) if argv is not None else None)

    paths.ensure_layout()
    lock = DispatcherLock()
    if not lock.acquire():
        return 0
    dispatcher = Dispatcher(deps=DispatcherDeps(interval_s=args.interval))
    if args.once:
        lock.start_heartbeat()
        try:
            dispatcher.adopt_orphans()
            dispatcher.run_once()
        finally:
            lock.release()
        return 0
    return dispatcher.run(lock)
