"""The per-host dispatcher: one lock, one loop, one runner per job.

Started idempotently by every enqueue. Holds an flock on an open fd (released
by the kernel if it dies) *and* touches a heartbeat file, so a second
dispatcher can tell "already running" from "wedged" and take over.
"""

from __future__ import annotations

import contextlib
import fcntl
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from gpuc.host import gpus, jobs, paths, queue, sync, terminate
from gpuc.host.gpus import SmiRunner
from gpuc.host.runner import kill_process_group, process_group_alive
from gpuc.host.terminate import TerminateCall

HEARTBEAT_INTERVAL_S = 5.0
HEARTBEAT_STALE_S = 30.0
LOOP_INTERVAL_S = 2.0
KILL_GRACE_S = 15.0
TERMINATE_RETRY_S = 600.0


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


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
        self.takeover_pgid: int | None = None

    @property
    def lock_path(self) -> Path:
        return paths.lock_file()

    @property
    def heartbeat_path(self) -> Path:
        return paths.heartbeat_file()

    def heartbeat_age(self) -> float | None:
        try:
            return self._now() - self.heartbeat_path.stat().st_mtime
        except FileNotFoundError:
            return None

    def holder_is_fresh(self) -> bool:
        age = self.heartbeat_age()
        return age is not None and age < self.stale_after_s

    def _holder_pgid(self) -> int | None:
        try:
            body = self.lock_path.read_text().strip()
        except OSError:
            return None
        return int(body) if body.isdigit() else None

    def acquire(self, takeover_wait_s: float = 10.0) -> bool:
        paths.ensure_layout()
        fd = os.open(self.lock_path, os.O_RDWR | os.O_CREAT, 0o644)
        if self._try_flock(fd):
            self._adopt(fd)
            return True
        if self.holder_is_fresh():
            os.close(fd)
            return False
        pgid = self._holder_pgid()
        if pgid and pgid != os.getpgid(0):
            self.takeover_pgid = pgid
            kill_process_group(pgid, grace_s=5.0, sleep=self._sleep)
        deadline = self._now() + takeover_wait_s
        while self._now() < deadline:
            if self._try_flock(fd):
                self._adopt(fd)
                return True
            self._sleep(0.25)
        os.close(fd)
        return False

    def _try_flock(self, fd: int) -> bool:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return False
        return True

    def _adopt(self, fd: int) -> None:
        self._fd = fd
        os.ftruncate(fd, 0)
        os.lseek(fd, 0, os.SEEK_SET)
        os.write(fd, f"{os.getpgid(0)}\n".encode())
        os.fsync(fd)
        self.beat(force=True)

    def beat(self, force: bool = False) -> None:
        now = self._now()
        if not force and now - self._last_beat < self.heartbeat_interval_s:
            return
        self._last_beat = now
        self.heartbeat_path.parent.mkdir(parents=True, exist_ok=True)
        self.heartbeat_path.touch()
        os.utime(self.heartbeat_path, (now, now))

    def release(self) -> None:
        if self._fd is None:
            return
        try:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
        finally:
            os.close(self._fd)
            self._fd = None


def default_spawn_runner(job_id: str) -> subprocess.Popen[bytes]:
    package_root = Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = f"{package_root}{os.pathsep}{existing}" if existing else str(package_root)
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
    env = dict(os.environ)
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = f"{package_root}{os.pathsep}{existing}" if existing else str(package_root)
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
        return None if _pid_alive(self.pid) else -1


@dataclass
class Dispatcher:
    deps: DispatcherDeps = field(default_factory=DispatcherDeps)
    running: dict[str, _Running] = field(default_factory=dict)
    _queue_empty_since: float | None = None
    _cancel_sent: dict[str, float] = field(default_factory=dict)
    _terminate_retry_at: float | None = None
    should_exit: bool = False

    @property
    def config(self) -> jobs.HostConfig:
        return jobs.read_config()

    def log(self, message: str) -> None:
        stamp = self.deps.utcnow().isoformat(timespec="seconds")
        with paths.dispatcher_log().open("a") as handle:
            handle.write(f"{stamp} {message}\n")

    # -- startup ---------------------------------------------------------
    def adopt_orphans(self) -> None:
        """Reconcile jobs left `running` by a dispatcher that died."""
        for job_id in jobs.list_job_ids():
            try:
                state = jobs.read_state(job_id)
            except RuntimeError:
                continue
            if state.status != "running" or job_id in self.running:
                continue
            if state.runner_pid and _pid_alive(state.runner_pid):
                self.running[job_id] = _Running(job_id, state.runner_pid, list(state.gpus))
                self.log(f"adopted running job {job_id} (runner pid {state.runner_pid})")
            else:
                self._mark_runner_died(job_id)

    def _mark_runner_died(self, job_id: str) -> None:
        jobs.update_state(
            job_id,
            status="failed",
            reason="runner-died",
            exit_code=jobs.read_state(job_id).exit_code or 1,
            ended_at=jobs.utc_now(),
            phase=None,
        )
        self.log(f"job {job_id} failed: runner died without writing final state")

    # -- loop pieces -----------------------------------------------------
    def reap(self) -> None:
        for job_id, entry in list(self.running.items()):
            if entry.poll() is None:
                continue
            del self.running[job_id]
            self._cancel_sent.pop(job_id, None)
            state = jobs.read_state(job_id)
            if not state.finished:
                self._mark_runner_died(job_id)
            else:
                self.log(
                    f"job {job_id} {state.status}"
                    f"{f' ({state.reason})' if state.reason else ''} "
                    f"exit={state.exit_code}"
                )

    def handle_cancels(self) -> None:
        now = self.deps.monotonic()
        for job_id, entry in list(self.running.items()):
            if not queue.is_cancelled(job_id):
                continue
            state = jobs.read_state(job_id)
            sent = self._cancel_sent.get(job_id)
            if sent is None:
                self._cancel_sent[job_id] = now
                self.log(f"cancelling job {job_id} (pgid {state.pgid})")
                self._signal_group(state.pgid, signal.SIGTERM)
                continue
            elapsed = now - sent
            if elapsed > self.deps.kill_grace_s:
                self._signal_group(state.pgid, signal.SIGKILL)
            # The runner does its own SIGTERM/SIGKILL and then finalises state;
            # only if it is itself wedged do we take out its group too.
            if elapsed > 2 * self.deps.kill_grace_s:
                self._signal_group(entry.pid, signal.SIGKILL)

    def _signal_group(self, pgid: int | None, sig: int) -> None:
        if not pgid or pgid <= 1 or not process_group_alive(pgid):
            return
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(pgid, sig)

    def free_gpus(self) -> list[str]:
        busy = {uuid for entry in self.running.values() for uuid in entry.gpus}
        return [uuid for uuid in self.config.gpus if uuid not in busy]

    def launch_ready(self) -> None:
        if self.paused() or paths.draining_file().exists():
            return
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
            proc = self.deps.spawn_runner(job_id)
            jobs.update_state(job_id, pid=proc.pid, pgid=proc.pid, runner_pid=proc.pid)
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
        if self.paused() or not self.recent_low_util_failures():
            return
        jobs.atomic_write_text(
            paths.paused_file(),
            "two consecutive jobs failed with reason low-util; queue paused\n",
        )
        self.log("PAUSED: two consecutive low-util failures; not dispatching further jobs")
        if self.config.ephemeral:
            self.drain_and_terminate("two consecutive low-util failures")

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
        if queue.list_queued():
            self._queue_empty_since = None
            return
        if self._queue_empty_since is None:
            self._queue_empty_since = now
        idle_s = now - self._queue_empty_since
        if idle_s >= config.idle_minutes * 60.0:
            self.drain_and_terminate(f"idle for {idle_s / 60.0:.1f} min")
            return
        if self._ttl_expired(config):
            self.drain_and_terminate(f"ttl of {config.ttl_hours} h elapsed")

    def _ttl_expired(self, config: jobs.HostConfig) -> bool:
        if not config.created_at:
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
        try:
            self._final_sync_all(config)
            terminate.self_terminate(config, terminate_call=self.deps.terminate_call)
        except (terminate.TerminateError, sync.SyncError) as exc:
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
                sync.sync_job_meta(job_id, config.s3_prefix, runner=self.deps.command_runner)
            except sync.SyncError as exc:
                errors.append(str(exc))
        if errors:
            raise sync.SyncError("; ".join(errors))

    # -- main ------------------------------------------------------------
    def run_once(self) -> None:
        self.reap()
        self.handle_cancels()
        self.check_pause()
        self.launch_ready()
        self.maybe_terminate()

    def idle_and_not_ephemeral(self) -> bool:
        return not self.running and not queue.list_queued() and not self.config.ephemeral

    def run(self, lock: DispatcherLock) -> int:
        self.adopt_orphans()
        self.log(f"dispatcher started (pid {os.getpid()}, pgid {os.getpgid(0)})")
        try:
            while not self.should_exit:
                lock.beat()
                self.run_once()
                if self.idle_and_not_ephemeral():
                    break
                self.deps.sleep(self.deps.interval_s)
        finally:
            lock.release()
        self.log("dispatcher exiting")
        return 0


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
        try:
            dispatcher.adopt_orphans()
            dispatcher.run_once()
        finally:
            lock.release()
        return 0
    return dispatcher.run(lock)
