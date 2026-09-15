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
    DesiredUnreadable,
    Settings,
    config_dir,
    load_desired,
    load_registry,
    pod_known_hosts_file,
    remove_desired,
    save_registry,
    state_dir,
    state_lock,
)
from gpuc.control.providers.base import Pod, Provider, ProviderError
from gpuc.control.provision import CEILING_MINUTES
from gpuc.control.s3index import IndexEntry, LocalIndex, S3Index, S3IndexError

DEFAULT_INTERVAL_S = 60.0
STRAY_GRACE_MINUTES = CEILING_MINUTES

Reporter = Callable[[str], None]

SERVICE_NAME = "gpuc-reconcile.service"
TIMER_NAME = "gpuc-reconcile.timer"


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
    """Drop every local trace of a host. The caller already holds the state lock."""
    remove_desired(name)
    pod_known_hosts_file(name).unlink(missing_ok=True)
    try:
        registry = load_registry()
        if registry.hosts.pop(name, None) is not None:
            save_registry(registry)
    except ConfigError as exc:
        report(f"WARNING: could not remove host {name} from the registry: {exc}")


def _terminate(provider: Provider, pod_id: str, why: str, report: Reporter) -> bool:
    report(f"terminating {pod_id}: {why}")
    try:
        provider.terminate(pod_id)
    except ProviderError as exc:
        report(
            f"ERROR: terminate of {pod_id} failed: {exc}\n"
            f"  It is still billing. Retry with `gpuc reconcile --once`, or terminate it in "
            f"the RunPod console."
        )
        return False
    report(f"terminated {pod_id} and confirmed it is gone")
    return True


def reconcile_once(
    settings: Settings, provider: Provider, report: Reporter = print
) -> ReconcileResult:
    result = ReconcileResult()
    with state_lock():
        try:
            desired = load_desired()
        except DesiredUnreadable as exc:
            report(f"ERROR: {exc}")
            result.errors.append(str(exc))
            return result
        try:
            pods = provider.list_ours()
        except ProviderError as exc:
            report(f"ERROR: could not list pods: {exc}")
            result.errors.append(str(exc))
            return result

        by_id = {pod.id: pod for pod in pods}
        _reconcile_desired(desired, by_id, settings, provider, report, result)
        _reap_strays(desired, pods, provider, report, result)
    return result


def _reconcile_desired(
    desired: list[DesiredHost],
    by_id: dict[str, Pod],
    settings: Settings,
    provider: Provider,
    report: Reporter,
    result: ReconcileResult,
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
        if age_h is not None and age_h > host.ttl_hours:
            if _terminate(
                provider,
                pod.id,
                f"host {host.name} is {age_h:.1f} h old, past its {host.ttl_hours:g} h TTL",
                report,
            ):
                result.terminated.append(host.name)
            _forget(host.name, report)
            result.forgotten.append(host.name)
            continue

        ceiling = _parse(host.ceiling_at)
        if not host.bootstrapped and ceiling is not None and datetime.now(UTC) > ceiling:
            if _terminate(
                provider,
                pod.id,
                f"host {host.name} never bootstrapped by its ceiling at {host.ceiling_at}",
                report,
            ):
                result.terminated.append(host.name)
            _forget(host.name, report)
            result.forgotten.append(host.name)
            continue

        report(
            f"{host.name} ({pod.id}): {pod.status}, ${pod.cost_usd_hr:.3f}/h, "
            f"{'' if host.bootstrapped else 'not yet bootstrapped, '}"
            f"{'age unknown' if age_h is None else f'age {age_h:.1f} h'} of "
            f"{host.ttl_hours:g} h TTL"
        )
        result.kept.append(host.name)


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
        if age is not None and age.total_seconds() < STRAY_GRACE_MINUTES * 60.0:
            report(
                f"{pod.name} ({pod.id}) has our prefix but no desired/ record and is only "
                f"{age.total_seconds() / 60.0:.0f} min old; leaving it for now in case another "
                f"session is still provisioning it"
            )
            result.kept.append(pod.name)
            continue
        if _terminate(
            provider,
            pod.id,
            f"{pod.name} has our prefix but no desired/ record (${pod.cost_usd_hr:.3f}/h)",
            report,
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
) -> ReconcileResult:
    last = ReconcileResult()
    count = 0
    while iterations is None or count < iterations:
        count += 1
        last = reconcile_once(settings, provider, report)
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
