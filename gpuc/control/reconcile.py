"""The reaper: desired state versus what the provider is actually billing for.

Every pod with our prefix is asked what it is before anything is done about it,
because a `desired/<host>.json` record only exists on the machine that created
the pod (see `rented`). A pod that holds a gpuc config is ours whoever bought
it, and is judged by the same rules as a host this machine provisioned itself;
the local `desired/` directory is a cache of those answers, and what keeps a
pod that has stopped answering under watch.

Fails closed in every direction. If `desired/` cannot be read we do nothing at
all, because "no state" must never be read as "terminate everything with our
prefix". Pods without our prefix are never even considered; a pod this machine
could not reach is reported and left alone, because "not my ssh key" and "dead"
look identical from here; and a pod young enough to still be provisioning is
given the ceiling, on whichever machine is creating it.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gpuc.control.config import (
    ConfigError,
    DesiredHost,
    HostEntry,
    Settings,
    config_dir,
    forget_host,
    load_desired,
    load_registry,
    read_desired,
    state_dir,
    state_lock,
    transport_for,
    write_desired,
)
from gpuc.control.providers.base import Pod, Provider, ProviderError
from gpuc.control.provision import CEILING_MINUTES
from gpuc.control.rented import (
    Liveness,
    PodAnswer,
    address_for,
    ask_pod,
    pulse,
    remember,
)
from gpuc.control.s3index import IndexEntry, LocalIndex, S3Index, S3IndexError
from gpuc.control.systemd import gpuc_command, systemd_dir, write_units

DEFAULT_INTERVAL_S = 60.0
STRAY_GRACE_MINUTES = CEILING_MINUTES

HostLiveness = Callable[[DesiredHost, "HostEntry | None", Settings], Liveness]
PodQuestion = Callable[[Pod, Settings], PodAnswer]

Reporter = Callable[[str], None]

SERVICE_NAME = "gpuc-reconcile.service"
TIMER_NAME = "gpuc-reconcile.timer"


@dataclass
class Rented:
    """One pod we own: the record that judges it, and how to reach it."""

    desired: DesiredHost
    entry: HostEntry | None


def probe_liveness(host: DesiredHost, entry: HostEntry | None, settings: Settings) -> Liveness:
    """Whether this host is beating, asked of the host and nothing else.

    `entry` is the address the pass resolved -- the registry's, or the one the
    provider gives for the pod -- because the machine running the reaper may
    never have registered this pod at all.
    """
    if entry is None:
        return Liveness(reachable=False)
    try:
        transport = transport_for(entry, settings)
    except ConfigError:
        return Liveness(reachable=False)
    # `remote_home` unexpanded: the pulse script is run by the host's own
    # shell, which resolves `$HOME` without costing a round trip to ask.
    return pulse(transport, entry.remote_home)


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

    def document(self) -> dict[str, Any]:
        """`gpuc reconcile --once --json`. Host names, in the order they were judged.

        A terminate that failed leaves its record in place and lands in
        `errors`, which is the exit-1 case: the next pass retries it.
        """
        return {
            "terminated": list(self.terminated),
            "forgotten": list(self.forgotten),
            "kept": list(self.kept),
            "errors": list(self.errors),
        }


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
    ask: PodQuestion = ask_pod,
) -> ReconcileResult:
    result = ReconcileResult()
    try:
        # The lock covers only the read: `desired/` is a directory of files, and
        # a half-listed one would look like "these pods are nobody's".
        with state_lock():
            cached = load_desired()
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
    registry = load_registry().hosts
    ours = [
        Rented(host, _how_to_reach(host.name, by_id.get(host.pod_id), registry)) for host in cached
    ]
    known = {host.pod_id for host in cached}
    adopted, unclaimed = _ask_the_rest(pods, known, settings, ask, report)
    _reconcile_desired([*ours, *adopted], by_id, settings, provider, report, result, liveness)
    _reap_strays(unclaimed, provider, report, result)
    return result


def _how_to_reach(
    name: str, pod: Pod | None, registry: Mapping[str, HostEntry]
) -> HostEntry | None:
    """This machine's address for a host, or the one the provider gives for its pod."""
    entry = registry.get(name)
    if entry is not None:
        return entry
    return address_for(name, pod) if pod is not None else None


def _ask_the_rest(
    pods: list[Pod],
    known: set[str],
    settings: Settings,
    ask: PodQuestion,
    report: Reporter,
) -> tuple[list[Rented], list[PodAnswer]]:
    """Ask every prefixed pod this machine has no record of what it is.

    A pod that holds a gpuc config is ours -- the machine that created it is
    not special, and may be switched off -- so it joins the desired hosts and
    is judged by the same rules. Its answer is cached in `desired/`, which is
    what keeps it under watch on a later pass that cannot reach it at all.
    """
    adopted: list[Rented] = []
    unclaimed: list[PodAnswer] = []
    for pod in pods:
        if pod.id in known or pod.status == "TERMINATED":
            continue
        answer = ask(pod, settings)
        if answer.desired is None:
            unclaimed.append(answer)
            continue
        report(
            f"{pod.name} ({pod.id}) has no desired/ record here, but {answer.detail}, so it is ours"
        )
        try:
            if not remember(answer.desired):
                report(
                    f"WARNING: desired/{answer.desired.name}.json already names another pod, so "
                    f"what {pod.name} said was not cached. Two pods answering to one host name "
                    f"is a `host` somebody set by hand; check `gpuc pods`."
                )
        except (ConfigError, OSError) as exc:
            report(f"WARNING: could not cache what {pod.name} said in desired/: {exc}")
        adopted.append(Rented(answer.desired, answer.entry))
    return adopted, unclaimed


def _reconcile_desired(
    desired: list[Rented],
    by_id: dict[str, Pod],
    settings: Settings,
    provider: Provider,
    report: Reporter,
    result: ReconcileResult,
    liveness: HostLiveness,
) -> None:
    for rented in desired:
        host = rented.desired
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
            rented, pod, settings, provider, report, result, liveness
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
    rented: Rented,
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
    host = rented.desired
    state = liveness(host, rented.entry, settings)
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
    unclaimed: list[PodAnswer],
    provider: Provider,
    report: Reporter,
    result: ReconcileResult,
) -> None:
    """Terminate the pods with our prefix that nothing claims -- and only those.

    A stray has to be *shown* to be one. The pod itself is the record (see
    `rented`), so a pod that answered and has no gpuc config on it is a create
    that leaked, and a pod the provider never gave an ssh endpoint is one that
    never came up: both bill for nothing. A pod that has an endpoint and did not
    answer this machine is neither, because "wedged" and "this machine holds no
    key for it" are the same silence -- and terminating on that is what took
    someone's running job.

    The ceiling still covers every case: another machine's `create` is a pod
    with no config on it yet, for as long as its bootstrap takes.
    """
    for answer in unclaimed:
        pod = answer.pod
        age = pod.age
        if age is None:
            # Cannot prove it is past the provisioning ceiling, so cannot prove
            # it is not another session's pod mid-create. Fail closed and say so.
            report(
                f"{pod.name} ({pod.id}) is unclaimed ({answer.detail}), but the provider "
                f"reports no creation time, so its age cannot be checked against the "
                f"{STRAY_GRACE_MINUTES:.0f} min ceiling; leaving it alone. Check `gpuc pods`."
            )
            result.kept.append(pod.name)
            continue
        minutes = age.total_seconds() / 60.0
        if minutes < STRAY_GRACE_MINUTES:
            report(
                f"{pod.name} ({pod.id}) is unclaimed ({answer.detail}) and is only "
                f"{minutes:.0f} min old; leaving it for now in case another session is still "
                f"provisioning it"
            )
            result.kept.append(pod.name)
            continue
        if answer.verdict == "silent" and pod.ssh_direct is not None:
            report(
                f"{pod.name} ({pod.id}) has our prefix, is {minutes:.0f} min old and is billing "
                f"${pod.cost_usd_hr:.3f}/h, but this machine could not ask it what it is "
                f"({answer.detail}), and a pod this machine has no key for looks exactly like a "
                f"dead one. Nothing was terminated.\n"
                f"  Adopt it here with `gpuc host add <name> --pod {pod.id}`, or terminate it "
                f"from the machine that created it (or the RunPod console)."
            )
            result.kept.append(pod.name)
            continue
        if _terminate(
            provider,
            pod,
            f"{pod.name} is {minutes:.0f} min old and nothing claims it: {answer.detail} "
            f"(${pod.cost_usd_hr:.3f}/h)",
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
    ask: PodQuestion = ask_pod,
) -> ReconcileResult:
    last = ReconcileResult()
    count = 0
    while iterations is None or count < iterations:
        count += 1
        last = reconcile_once(settings, provider, report, liveness=liveness, ask=ask)
        report(last.render())
        if iterations is not None and count >= iterations:
            break
        sleep(interval_s)
    return last


def gpuc_argv() -> str:
    """An absolute command line for `gpuc reconcile --once`, for systemd."""
    return gpuc_command(["reconcile", "--once"])


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
    written = write_units(systemd_dir(), unit_files(interval_s), report)
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
