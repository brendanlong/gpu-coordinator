"""The reaper: desired state versus what the provider is actually billing for.

Fails closed in every direction. If `desired/` cannot be read we do nothing at
all, because "no state" must never be read as "terminate everything with our
prefix". Pods without our prefix are never even considered; other sessions'
freshly created pods are given the provisioning ceiling before they count as
strays, so a race between `create` and the desired-state write cannot cost
someone else their pod.
"""

from __future__ import annotations

import contextlib
import shutil
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from gpuc.control.bootstrap import package_root
from gpuc.control.config import (
    ConfigError,
    DesiredHost,
    Settings,
    config_dir,
    forget_host,
    load_desired,
    load_registry,
    read_desired,
    state_dir,
    state_lock,
    write_desired,
)
from gpuc.control.providers.base import Pod, Provider, ProviderError
from gpuc.control.provision import CEILING_MINUTES, host_status
from gpuc.control.s3index import IndexEntry, LocalIndex, S3Index, S3IndexError

DEFAULT_INTERVAL_S = 60.0
STRAY_GRACE_MINUTES = CEILING_MINUTES
LIVENESS_TIMEOUT_S = 20.0
HEARTBEAT_FRESH_S = 120.0
"""A heartbeat this old still counts as alive. The dispatcher beats every 5 s;
the slack is for a host that was busy syncing, not for one that is gone."""

HostLiveness = Callable[[DesiredHost, Settings], "Liveness"]

Reporter = Callable[[str], None]

SERVICE_NAME = "gpuc-reconcile.service"
TIMER_NAME = "gpuc-reconcile.timer"


@dataclass
class Liveness:
    """What one look at a host says: reachable, beating, busy."""

    reachable: bool
    heartbeat_age_s: float | None = None
    running_jobs: int = 0

    @property
    def alive(self) -> bool:
        if self.running_jobs:
            return True
        age = self.heartbeat_age_s
        return age is not None and age < HEARTBEAT_FRESH_S

    def describe(self) -> str:
        if not self.reachable:
            return "ssh did not answer"
        if self.heartbeat_age_s is None:
            return "reachable, but the dispatcher has never beaten"
        return f"heartbeat {self.heartbeat_age_s:.0f}s old, {self.running_jobs} job(s) running"


def probe_liveness(host: DesiredHost, settings: Settings) -> Liveness:
    entry = load_registry().hosts.get(host.name)
    if entry is None:
        return Liveness(reachable=False)
    # Short even though the lock is not held here: a pass that stalls on one
    # wedged pod is a pass that does not reach the next one, which is billing.
    payload = host_status(entry, settings, timeout=LIVENESS_TIMEOUT_S)
    if payload is None:
        return Liveness(reachable=False)
    age = payload.get("dispatcher_heartbeat_age_s")
    running = sum(1 for job in payload.get("jobs", []) if job.get("status") == "running")
    return Liveness(
        reachable=True,
        heartbeat_age_s=float(age) if isinstance(age, (int, float)) else None,
        running_jobs=running,
    )


@dataclass
class ReconcileResult:
    terminated: list[str] = field(default_factory=list)
    forgotten: list[str] = field(default_factory=list)
    kept: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def render(self) -> str:
        parts = [
            f"{len(self.kept)} healthy",
            f"{len(self.terminated)} terminated",
            f"{len(self.forgotten)} forgotten",
        ]
        if self.errors:
            parts.append(f"{len(self.errors)} error(s)")
        return "reconcile: " + ", ".join(parts)


def _parse(stamp: str | None) -> datetime | None:
    if not stamp:
        return None
    try:
        parsed = datetime.fromisoformat(stamp)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _age_hours(pod: Pod, desired: DesiredHost | None) -> float | None:
    created = pod.created_at or _parse(desired.created_at if desired else None)
    if created is None:
        return None
    return (datetime.now(UTC) - created).total_seconds() / 3600.0


def jobs_on_host(host: str, settings: Settings) -> list[IndexEntry]:
    """Best effort: the local index first, then the S3 mirror if it is configured."""
    entries = {e.job_id: e for e in LocalIndex().list() if e.host == host}
    s3 = S3Index.from_settings(settings)
    if s3 is not None:
        with contextlib.suppress(S3IndexError):
            entries.update({e.job_id: e for e in s3.list_index() if e.host == host})
    return [entries[job_id] for job_id in sorted(entries)]


def _describe_lost_jobs(host: str, settings: Settings) -> str:
    entries = jobs_on_host(host, settings)
    if not entries:
        return "host gone; no jobs are recorded against it"
    listed = ", ".join(f"{e.job_id} ({e.name or 'unnamed'})" for e in entries[:10])
    return (
        f"host gone; {len(entries)} job(s) were submitted there: {listed}\n"
        f"  resubmit any that had not finished: gpuc requeue <job-id> --runpod --gpu <name>"
    )


def _forget(name: str, report: Reporter) -> None:
    """Drop every local trace of a host, taking the state lock for just that.

    The lock is per mutation, never held across the provider and ssh calls that
    decide *whether* to mutate: a terminate polls for up to five minutes, and a
    concurrent `gpuc submit --runpod` gives up on the lock after two.
    """
    try:
        with state_lock():
            forget_host(name)
    except ConfigError as exc:
        report(f"WARNING: could not remove host {name} from the registry: {exc}")


def _terminate(
    provider: Provider, pod: Pod, why: str, report: Reporter, result: ReconcileResult
) -> bool:
    """The only terminate on the reaper's side, and the only prefix check it needs.

    Desired records are local files: a hand-edited or stale one could name a
    pod that is not ours, and "never touch someone else's pod" rests on code
    here rather than on permissions, so it is checked at the call itself.
    """
    pod_id = pod.id
    if not pod.name.startswith(provider.caps.prefix):
        message = (
            f"refusing to terminate {pod.name} ({pod_id}): it does not start with "
            f"{provider.caps.prefix!r}, so it is not ours. Fix the desired/ record that names it."
        )
        report(f"ERROR: {message}")
        result.errors.append(message)
        return False
    report(f"terminating {pod_id}: {why}")
    try:
        provider.terminate(pod_id)
    except ProviderError as exc:
        message = (
            f"terminate of {pod_id} failed: {exc}\n"
            f"  It is still billing. Retry with `gpuc reconcile --once`, or terminate it in "
            f"the RunPod console."
        )
        report(f"ERROR: {message}")
        # An error, not just a log line: `gpuc reconcile --once` must exit
        # non-zero while a pod we wanted gone is still charging.
        result.errors.append(message)
        return False
    report(f"terminated {pod_id} and confirmed it is gone")
    return True


def reconcile_once(
    settings: Settings,
    provider: Provider,
    report: Reporter = print,
    *,
    liveness: HostLiveness = probe_liveness,
) -> ReconcileResult:
    result = ReconcileResult()
    try:
        # The lock covers only the read: `desired/` is a directory of files, and
        # a half-listed one would look like "these pods are nobody's".
        with state_lock():
            desired = load_desired()
    except ConfigError as exc:  # DesiredUnreadable, or the lock itself timing out
        report(f"ERROR: {exc}")
        result.errors.append(str(exc))
        return result
    try:
        pods = provider.list_ours()
    except ProviderError as exc:
        report(f"ERROR: could not list pods: {exc}")
        result.errors.append(str(exc))
        return result

    # From here on nothing holds the lock: every ssh probe and every terminate
    # (which polls for up to five minutes) happens outside it, and each local
    # mutation takes it for itself. Holding it across all of that starved the
    # `gpuc submit --runpod` that was waiting to record a pod it had created --
    # and a create that cannot write desired/ is a leaked, billing pod.
    by_id = {pod.id: pod for pod in pods}
    _reconcile_desired(desired, by_id, settings, provider, report, result, liveness)
    _reap_strays(desired, pods, provider, report, result)
    return result


def _reconcile_desired(
    desired: list[DesiredHost],
    by_id: dict[str, Pod],
    settings: Settings,
    provider: Provider,
    report: Reporter,
    result: ReconcileResult,
    liveness: HostLiveness,
) -> None:
    for host in desired:
        pod = by_id.get(host.pod_id)
        if pod is None:
            try:
                pod = provider.get(host.pod_id)
            except ProviderError as exc:
                report(f"ERROR: could not read pod {host.pod_id} for host {host.name}: {exc}")
                result.errors.append(str(exc))
                continue
        if pod is None or pod.status == "TERMINATED":
            report(f"{host.name} ({host.pod_id}): {_describe_lost_jobs(host.name, settings)}")
            _forget(host.name, report)
            result.forgotten.append(host.name)
            continue

        age_h = _age_hours(pod, host)
        if host.ttl_hours is not None and age_h is not None and age_h > host.ttl_hours:
            # Only forget a host whose pod is confirmed gone: while a terminate
            # is failing, the record is what keeps retrying it (and what still
            # tells `gpuc logs` where that host's jobs ran).
            if _terminate(
                provider,
                pod,
                f"host {host.name} is {age_h:.1f} h old, past its {host.ttl_hours:g} h TTL",
                report,
                result,
            ):
                result.terminated.append(host.name)
                _forget(host.name, report)
                result.forgotten.append(host.name)
            continue

        ceiling = _parse(host.ceiling_at)
        if not host.bootstrapped and ceiling is not None and datetime.now(UTC) > ceiling:
            if _terminate(
                provider,
                pod,
                f"host {host.name} never bootstrapped by its ceiling at {host.ceiling_at}",
                report,
                result,
            ):
                result.terminated.append(host.name)
                _forget(host.name, report)
                result.forgotten.append(host.name)
            continue

        if host.bootstrapped and _reap_if_silent(
            host, pod, settings, provider, report, result, liveness
        ):
            continue

        report(
            f"{host.name} ({pod.id}): {pod.status}, ${pod.cost_usd_hr:.3f}/h, "
            f"{'' if host.bootstrapped else 'not yet bootstrapped, '}"
            f"{'age unknown' if age_h is None else f'age {age_h:.1f} h'}, "
            f"{'no TTL' if host.ttl_hours is None else f'{host.ttl_hours:g} h TTL'}"
        )
        result.kept.append(host.name)


def _reap_if_silent(
    host: DesiredHost,
    pod: Pod,
    settings: Settings,
    provider: Provider,
    report: Reporter,
    result: ReconcileResult,
    liveness: HostLiveness,
) -> bool:
    """Terminate a bootstrapped host that has stopped answering for too long.

    This is what replaces the overall TTL. A pod whose dispatcher has died, or
    which has stopped answering ssh entirely, cannot self-terminate on idle and
    cannot be seen to be doing anything -- and it bills all the same. A job
    running per the host's own state resets the clock, so a long training run is
    never touched.
    """
    state = liveness(host, settings)
    if state.alive:
        _remember_seen(host, report)
        return False
    silent_for = _minutes_since(host.silent_since())
    limit = settings.dead_dispatcher_minutes
    if silent_for is None or silent_for < limit:
        report(
            f"{host.name} ({pod.id}): {state.describe()}; silent for "
            f"{'unknown' if silent_for is None else f'{silent_for:.0f}'} min of the "
            f"{limit:.0f} min limit"
        )
        return False
    why = (
        f"host {host.name} has been silent for {silent_for:.0f} min "
        f"(limit {limit:.0f}); {state.describe()}, nothing is running, and the pod is still "
        f"billing ${pod.cost_usd_hr:.3f}/h"
    )
    report(f"DEAD DISPATCHER: {why}")
    if _terminate(provider, pod, why, report, result):
        result.terminated.append(host.name)
        _forget(host.name, report)
        result.forgotten.append(host.name)
        return True
    return False


def _remember_seen(host: DesiredHost, report: Reporter) -> None:
    """Stamp `last_seen_at`, re-reading the record under the lock first.

    The copy this pass started from was read minutes and several ssh calls ago;
    a `gpuc submit --runpod` may have written `bootstrapped_at` in between, and
    writing the stale copy back would undo it.
    """
    try:
        with state_lock():
            current = read_desired(host.name)
            if current is None:
                # Another session forgot this host while we were probing it.
                # Writing the record back would resurrect a pod nothing owns.
                return
            write_desired(current.model_copy(update={"last_seen_at": _now_text()}))
    except (ConfigError, OSError) as exc:
        report(f"WARNING: could not record that {host.name} is alive: {exc}")


def _now_text() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _minutes_since(stamp: str | None) -> float | None:
    parsed = _parse(stamp)
    if parsed is None:
        return None
    return (datetime.now(UTC) - parsed).total_seconds() / 60.0


def _reap_strays(
    desired: list[DesiredHost],
    pods: list[Pod],
    provider: Provider,
    report: Reporter,
    result: ReconcileResult,
) -> None:
    known = {host.pod_id for host in desired}
    for pod in pods:
        if pod.id in known or pod.status == "TERMINATED":
            continue
        age = pod.age
        if age is None:
            # Cannot prove it is past the provisioning ceiling, so cannot prove
            # it is not another session's pod mid-create. Fail closed and say so.
            report(
                f"{pod.name} ({pod.id}) has our prefix and no desired/ record, but the provider "
                f"reports no creation time, so its age cannot be checked against the "
                f"{STRAY_GRACE_MINUTES:.0f} min ceiling; leaving it alone. Check `gpuc pods`."
            )
            result.kept.append(pod.name)
            continue
        if age.total_seconds() < STRAY_GRACE_MINUTES * 60.0:
            report(
                f"{pod.name} ({pod.id}) has our prefix but no desired/ record and is only "
                f"{age.total_seconds() / 60.0:.0f} min old; leaving it for now in case another "
                f"session is still provisioning it"
            )
            result.kept.append(pod.name)
            continue
        if _terminate(
            provider,
            pod,
            f"{pod.name} has our prefix but no desired/ record (${pod.cost_usd_hr:.3f}/h)",
            report,
            result,
        ):
            result.terminated.append(pod.name)


def run_loop(
    settings: Settings,
    provider: Provider,
    *,
    interval_s: float = DEFAULT_INTERVAL_S,
    report: Reporter = print,
    iterations: int | None = None,
    sleep: Callable[[float], None] = time.sleep,
    liveness: HostLiveness = probe_liveness,
) -> ReconcileResult:
    last = ReconcileResult()
    count = 0
    while iterations is None or count < iterations:
        count += 1
        last = reconcile_once(settings, provider, report, liveness=liveness)
        report(last.render())
        if iterations is not None and count >= iterations:
            break
        sleep(interval_s)
    return last


def gpuc_argv() -> str:
    """An absolute command line for `gpuc reconcile --once`, for systemd."""
    beside = Path(sys.executable).resolve().parent / "gpuc"
    if beside.exists():
        return f"{beside} reconcile --once"
    installed = shutil.which("gpuc")
    if installed:
        return f"{Path(installed).resolve()} reconcile --once"
    uv = shutil.which("uv") or str(Path.home() / ".local/bin/uv")
    return f"{uv} run --project {package_root()} gpuc reconcile --once"


def systemd_dir() -> Path:
    return Path.home() / ".config/systemd/user"


def unit_files(interval_s: float = DEFAULT_INTERVAL_S) -> dict[str, str]:
    service = f"""[Unit]
Description=gpuc reconcile: terminate leaked or expired GPU pods
After=network-online.target

[Service]
Type=oneshot
Environment=GPUC_CONFIG_DIR={config_dir()}
Environment=GPUC_STATE_DIR={state_dir()}
EnvironmentFile=-{config_dir()}/env
ExecStart={gpuc_argv()}
"""
    timer = f"""[Unit]
Description=Run gpuc reconcile every {interval_s:.0f}s

[Timer]
OnBootSec=2min
OnUnitActiveSec={interval_s:.0f}s
AccuracySec=15s
Unit={SERVICE_NAME}

[Install]
WantedBy=timers.target
"""
    return {SERVICE_NAME: service, TIMER_NAME: timer}


def install(interval_s: float = DEFAULT_INTERVAL_S, report: Reporter = print) -> list[Path]:
    """Write the unit files only. Enabling is the user's call, not ours."""
    directory = systemd_dir()
    directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, body in unit_files(interval_s).items():
        path = directory / name
        path.write_text(body)
        written.append(path)
        report(f"wrote {path}")
    report(
        f"not enabled. The service reads RUNPOD_API_KEY from {config_dir()}/env, which it does "
        f"not create:\n"
        f"  install -m 600 /dev/null {config_dir()}/env && "
        f"echo RUNPOD_API_KEY=... >> {config_dir()}/env\n"
        "Then:\n"
        "  systemctl --user daemon-reload\n"
        f"  systemctl --user enable --now {TIMER_NAME}\n"
        f"  systemctl --user list-timers {TIMER_NAME}\n"
        f"  journalctl --user -u {SERVICE_NAME} -f"
    )
    return written
