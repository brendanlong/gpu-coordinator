"""The reaper: desired state versus what the provider is actually billing for.

Every pod with our prefix is asked what it is before anything is done about it,
because a `desired/<host>.json` record only exists on the machine that created
the pod (see `rented`). A pod that holds a gpuc config is ours whoever bought
it, and is judged by the same rules as a host this machine provisioned itself;
the local `desired/` directory is a cache of those answers, and what keeps a
pod that has stopped answering under watch.

Nothing is ever terminated for the *absence* of a record. A pod with our prefix
that this machine has no record of and cannot get an answer out of is reported
every pass and left running: it may be wedged, it may hold no key of ours, or it
may be another machine's create still bootstrapping, and those look identical
from here. What the reaper terminates is a pod that never bootstrapped by its
ceiling, and one that has stopped beating with nothing running -- the two
states a host cannot get itself out of, since a healthy pod drains and
terminates itself when its queue goes quiet.

Fails closed in every other direction too. If `desired/` cannot be read we do
nothing at all, because "no state" must never be read as "terminate everything
with our prefix", and pods without our prefix are never even considered.
"""

from __future__ import annotations

import contextlib
import json
import os
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
    Reporter,
    Settings,
    config_dir,
    forget_host_locked,
    load_desired,
    load_registry,
    parse_timestamp,
    read_desired,
    state_dir,
    state_lock,
    transport_for,
    utc_now,
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
WATCH_GAP_MINUTES = 5.0
"""A pass this long after the previous one means this machine was not watching
in between: suspended, rebooted, or the timer disabled. Comfortably longer than
the 60 s timer interval, and far shorter than the silence a host is allowed."""

HostLiveness = Callable[[DesiredHost, "HostEntry | None", Settings], Liveness]
PodQuestion = Callable[[Pod, Settings], PodAnswer]


SERVICE_NAME = "gpuc-reconcile.service"
TIMER_NAME = "gpuc-reconcile.timer"


@dataclass
class Rented:
    """One pod we own: the record that judges it, and how to reach it."""

    desired: DesiredHost
    entry: HostEntry | None


def watch_file() -> Path:
    return state_dir() / "watch.json"


def begin_watch(report: Reporter) -> datetime:
    """When this machine's current unbroken stretch of watching began.

    The silence rule has to count silence this machine *observed*, not wall
    clock it was absent for. A desktop asleep for three days watched nothing in
    between, and the pass where it wakes is the one where its own ssh is most
    likely to fail -- the timer fires two minutes after boot, before a VPN or a
    key agent is necessarily there. Reading `last_seen_at` at face value on that
    pass terminates a pod that is answering everybody else perfectly well.

    So a gap resets the clock: this pass may find a host silent, but it cannot
    claim the host was silent while nothing was listening. Written without the
    state lock -- it is one machine's note to itself, an atomic replace, and a
    lost update costs at most one pass of patience.
    """
    # Truncated, because that is what the file can hold: a `since` that comes
    # back from disk differing from the one just returned by microseconds is a
    # difference nothing here wants to reason about.
    now = datetime.now(UTC).replace(microsecond=0)
    previous, since = _last_watch()
    away = None if previous is None else (now - previous).total_seconds() / 60.0
    if away is not None and away > WATCH_GAP_MINUTES:
        report(
            f"this machine last reconciled {away:.0f} min ago, so it was not watching in "
            f"between; every host gets the full silence allowance again from now"
        )
    if since is None or away is None or away > WATCH_GAP_MINUTES:
        since = now
    try:
        _write_watch(now, since)
    except OSError as exc:
        # Nothing accumulates, so nothing is reaped for silence: the safe way
        # to fail, and loud, because it is also the way a dead pod survives.
        report(f"WARNING: could not record this pass in {watch_file()}: {exc}")
        return now
    return since


def _last_watch() -> tuple[datetime | None, datetime | None]:
    try:
        document = json.loads(watch_file().read_text())
    except (OSError, json.JSONDecodeError):
        return None, None
    if not isinstance(document, dict):
        return None, None
    return parse_timestamp(document.get("pass_at")), parse_timestamp(document.get("since"))


def _write_watch(now: datetime, since: datetime) -> None:
    path = watch_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(
        {"pass_at": now.isoformat(timespec="seconds"), "since": since.isoformat(timespec="seconds")}
    )
    tmp = path.parent / f".{path.name}.{os.getpid()}.tmp"
    tmp.write_text(body + "\n")
    os.replace(tmp, path)


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
    unclaimed: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def render(self) -> str:
        parts = [
            f"{len(self.kept)} healthy",
            f"{len(self.terminated)} terminated",
            f"{len(self.forgotten)} forgotten",
        ]
        # Counted apart from `kept`, because a pod nothing here can place is not
        # a healthy host -- it is money nobody has accounted for.
        if self.unclaimed:
            parts.append(f"{len(self.unclaimed)} unclaimed")
        if self.errors:
            parts.append(f"{len(self.errors)} error(s)")
        return "reconcile: " + ", ".join(parts)

    def document(self) -> dict[str, Any]:
        """`gpuc reconcile --once --json`. Host names, in the order they were judged.

        A terminate that failed leaves its record in place and lands in
        `errors`, which is the exit-1 case: the next pass retries it.
        `unclaimed` is pod names, not host names: the pods with our prefix that
        this pass did not judge, because this machine has no record of them and
        could not get one out of them (or because the name one answers to is
        already another pod's record). Nothing here will terminate them.
        """
        return {
            "terminated": list(self.terminated),
            "forgotten": list(self.forgotten),
            "kept": list(self.kept),
            "unclaimed": list(self.unclaimed),
            "errors": list(self.errors),
        }


def _age_hours(pod: Pod, desired: DesiredHost | None) -> float | None:
    created = pod.created_at or parse_timestamp(desired.created_at if desired else None)
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


def _terminate_and_forget(
    provider: Provider,
    pod: Pod,
    host: DesiredHost,
    why: str,
    report: Reporter,
    result: ReconcileResult,
) -> bool:
    """Terminate, and forget the host only once the pod is confirmed gone.

    While a terminate is failing, the record is what keeps retrying it (and
    what still tells `gpuc logs` where that host's jobs ran).
    """
    if not _terminate(provider, pod, why, report, result):
        return False
    result.terminated.append(host.name)
    forget_host_locked(host.name, host.pod_id, report)
    result.forgotten.append(host.name)
    return True


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
    watching_since = begin_watch(report)
    ours = [
        Rented(host, _how_to_reach(host.name, by_id.get(host.pod_id), registry)) for host in cached
    ]
    known = {host.pod_id for host in cached}
    adopted, unclaimed = _ask_the_rest(pods, known, settings, ask, report, result)
    _reconcile_desired(
        [*ours, *adopted], by_id, settings, provider, report, result, liveness, watching_since
    )
    _report_unclaimed(unclaimed, report, result)
    return result


def _how_to_reach(
    name: str, pod: Pod | None, registry: Mapping[str, HostEntry]
) -> HostEntry | None:
    """Where this host is now: the provider's endpoint, this machine's home.

    Each side is asked what only it knows. The provider is the authority on
    where a rented pod is reachable *now* -- a registry entry pinned to an
    endpoint the pod no longer has fails every probe, which reads as a dead
    dispatcher -- while the entry is the only thing that knows a gpuc home
    somebody moved off `$HOME/.gpuc`.
    """
    entry = registry.get(name)
    address = address_for(name, pod) if pod is not None else None
    if entry is None or address is None:
        return entry or address
    if entry.kind != "runpod" or (entry.pod_id and pod is not None and entry.pod_id != pod.id):
        # The name collides with some other host of this machine's. Believe
        # the pod, and touch nothing of the entry's.
        return address
    return entry.model_copy(update={"ssh": address.ssh, "port": address.port})


def _ask_the_rest(
    pods: list[Pod],
    known: set[str],
    settings: Settings,
    ask: PodQuestion,
    report: Reporter,
    result: ReconcileResult,
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
        if not _cache(answer.desired, pod, report):
            # Not judged this pass rather than judged against a record that is
            # not its own: every terminate here forgets the host it names, and
            # that record belongs to a different pod.
            result.unclaimed.append(pod.name)
            continue
        adopted.append(Rented(answer.desired, answer.entry))
    return adopted, unclaimed


def _cache(record: DesiredHost, pod: Pod, report: Reporter) -> bool:
    """Keep what a pod said, and say whether this pass may act on it.

    A record this machine could not write is only a lost cache: the pod's own
    answer is still what judges it. A record it *refused* to write is a name
    that belongs to another pod, and every terminate here forgets the host it
    names -- so that pod is left for a pass whose record is its own.
    """
    try:
        if remember(record):
            return True
    except (ConfigError, OSError) as exc:
        report(
            f"WARNING: could not cache what {pod.name} said in desired/ ({exc}); judging it on "
            f"what it just said instead"
        )
        return True
    report(
        f"WARNING: desired/{record.name}.json is already here and names another pod, so "
        f"{pod.name} ({pod.id}) was left alone this pass. Two pods answering to one host name "
        f"is a `host` somebody set by hand; check `gpuc pods`."
    )
    return False


def _reconcile_desired(
    desired: list[Rented],
    by_id: dict[str, Pod],
    settings: Settings,
    provider: Provider,
    report: Reporter,
    result: ReconcileResult,
    liveness: HostLiveness,
    watching_since: datetime,
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
            forget_host_locked(host.name, host.pod_id, report)
            result.forgotten.append(host.name)
            continue

        age_h = _age_hours(pod, host)
        ceiling = parse_timestamp(host.ceiling_at)
        if not host.bootstrapped and ceiling is not None and datetime.now(UTC) > ceiling:
            _terminate_and_forget(
                provider,
                pod,
                host,
                f"host {host.name} never bootstrapped by its ceiling at {host.ceiling_at}",
                report,
                result,
            )
            continue

        if host.bootstrapped and _reap_if_silent(
            rented, pod, settings, provider, report, result, liveness, watching_since
        ):
            continue

        report(
            f"{host.name} ({pod.id}): {pod.status}, ${pod.cost_usd_hr:.3f}/h, "
            f"{'' if host.bootstrapped else 'not yet bootstrapped, '}"
            f"{'age unknown' if age_h is None else f'age {age_h:.1f} h'}"
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
    watching_since: datetime,
) -> bool:
    """Terminate a bootstrapped host that has stopped answering for too long.

    A pod whose dispatcher has died, or which has stopped answering ssh
    entirely, cannot self-terminate on idle and cannot be seen to be doing
    anything -- and it bills all the same. A job running per the host's own
    state resets the clock, so a long training run is never touched.
    """
    host = rented.desired
    state = liveness(host, rented.entry, settings)
    if state.alive:
        _remember_seen(host, report)
        return False
    # Never longer than this machine has been watching: see `begin_watch`.
    watched_for = _minutes_since(watching_since.isoformat())
    unseen_for = _minutes_since(host.silent_since())
    silent_for = unseen_for
    capped = False
    if unseen_for is not None and watched_for is not None and watched_for < unseen_for:
        silent_for, capped = watched_for, True
    limit = settings.dead_dispatcher_minutes
    if silent_for is None or silent_for < limit:
        # Said out loud when the cap is what is holding the terminate back:
        # "silent for 0 min" about a pod nothing has heard from in three days
        # reads like a bug otherwise.
        because = (
            f" (this machine has only been watching for {watched_for:.0f} min; the host has "
            f"not been heard from for {unseen_for:.0f})"
            if capped and watched_for is not None and unseen_for is not None
            else ""
        )
        report(
            f"{host.name} ({pod.id}): {state.describe()}; silent for "
            f"{'unknown' if silent_for is None else f'{silent_for:.0f}'} min of the "
            f"{limit:.0f} min limit{because}"
        )
        return False
    why = (
        f"host {host.name} has been silent for {silent_for:.0f} min "
        f"(limit {limit:.0f}); {state.describe()}, nothing is running, and the pod is still "
        f"billing ${pod.cost_usd_hr:.3f}/h"
    )
    report(f"DEAD DISPATCHER: {why}")
    return _terminate_and_forget(provider, pod, host, why, report, result)


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
            write_desired(current.model_copy(update={"last_seen_at": utc_now()}))
    except (ConfigError, OSError) as exc:
        report(f"WARNING: could not record that {host.name} is alive: {exc}")


def _minutes_since(stamp: str | None) -> float | None:
    parsed = parse_timestamp(stamp)
    if parsed is None:
        return None
    return (datetime.now(UTC) - parsed).total_seconds() / 60.0


def _report_unclaimed(
    unclaimed: list[PodAnswer],
    report: Reporter,
    result: ReconcileResult,
) -> None:
    """Say what is billing that nothing here claims, and touch none of it.

    Nothing is terminated on the *absence* of a record. A pod this machine has
    no record of and could not get an answer out of may be wedged, may hold no
    key of ours, or may be another machine's create still bootstrapping -- and
    those look identical from here. The one case that is genuinely a leak (a
    pod that never got a config, whose creating machine is never coming back)
    is worth a line every pass and a person's judgement, not a guess that can
    cost a running job.

    The machine that *does* hold a pod's record still reaps it on a dead
    dispatcher, and `gpuc host add <name> --pod <id>` moves that duty here.
    """
    for answer in unclaimed:
        pod = answer.pod
        result.unclaimed.append(pod.name)
        age = pod.age
        minutes = None if age is None else age.total_seconds() / 60.0
        if minutes is not None and minutes < CEILING_MINUTES:
            report(
                f"{pod.name} ({pod.id}) is {minutes:.0f} min old and nothing here wants it yet "
                f"({answer.detail}); another session may still be provisioning it"
            )
            continue
        report(
            f"{pod.name} ({pod.id}) is billing ${pod.cost_usd_hr:.3f}/h"
            f"{'' if minutes is None else f', is {minutes:.0f} min old'} and nothing here "
            f"claims it: {answer.detail}. Nothing was terminated.\n"
            f"  Take it over here with `gpuc host add <name> --pod {pod.id}`, or end it from "
            f"the machine that created it (or the RunPod console)."
        )


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


def unit_files(interval_s: float = DEFAULT_INTERVAL_S) -> dict[str, str]:
    service = f"""[Unit]
Description=gpuc reconcile: terminate leaked or expired GPU pods
# Both, not just After=: without the Wants= nothing pulls the target in, and a
# pass whose ssh fails because the network is two minutes old is a pass that
# judges every host unreachable.
Wants=network-online.target
After=network-online.target

[Service]
Type=oneshot
Environment=GPUC_CONFIG_DIR={config_dir()}
Environment=GPUC_STATE_DIR={state_dir()}
EnvironmentFile=-{config_dir()}/env
ExecStart={gpuc_command(["reconcile", "--once"])}
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
