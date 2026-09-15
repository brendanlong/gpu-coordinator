"""The `gpuc` command line. Thin: parse, call a module, print, map errors to 1."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from gpuc.control import pods as pods_mod
from gpuc.control import reconcile as reconcile_mod
from gpuc.control import status as status_mod
from gpuc.control.bootstrap import BootstrapError, bootstrap_host
from gpuc.control.clean import CleanError, clean_host, prune_uv_cache
from gpuc.control.config import (
    ConfigError,
    HostEntry,
    Registry,
    Settings,
    config_file,
    load_registry,
    load_settings,
    registry_transaction,
    state_dir,
    utc_now,
    write_config_template,
)
from gpuc.control.probe import probe_host
from gpuc.control.providers.base import Cloud, Constraints, Provider, ProviderError
from gpuc.control.providers.runpod import RunPodProvider
from gpuc.control.provision import ProvisionError, runpod_host
from gpuc.control.remote import RemoteError, open_session
from gpuc.control.s3index import IndexEntry, LocalIndex, S3Index, S3IndexError, job_log_uri
from gpuc.control.submit import (
    JobSpecModel,
    SubmitError,
    expand_job_id,
    load_document,
    precheck_local,
    submit_file,
    submit_spec,
    validate,
)
from gpuc.control.transport import SshTransport, Transport, TransportError
from gpuc.host import jobs


class CliError(RuntimeError):
    pass


def _gpu_list(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [part.strip() for part in raw.replace(" ", ",").split(",") if part.strip()]


def _env_dict(pairs: Sequence[str] | None) -> dict[str, str]:
    env: dict[str, str] = {}
    for pair in pairs or []:
        key, sep, value = pair.partition("=")
        if not sep or not key.strip():
            raise CliError(f"--env wants KEY=VALUE, got {pair!r}")
        env[key.strip()] = value
    return env


def cmd_host_add(args: argparse.Namespace) -> int:
    entry = HostEntry(
        name=args.name,
        kind="ssh" if args.ssh else "local",
        ssh=args.ssh,
        port=args.port,
        gpus=_gpu_list(args.gpus),
        gpuc_home=args.gpuc_home,
        persistent_root=args.persistent_root,
        env=_env_dict(args.env),
        cache_dir=args.cache_dir,
        s3_prefix=args.s3_prefix,
        idle_minutes=args.idle_min,
        ttl_hours=args.ttl_hours,
        created_at=utc_now(),
    )
    with registry_transaction() as registry:
        registry.put(entry)
    print(
        f"added host {entry.name} [{entry.kind}] "
        f"{entry.ssh or 'this machine'} with {len(entry.gpus)} GPU(s)\n"
        f"{_home_line(entry)}"
        f"next: gpuc host bootstrap {entry.name}"
    )
    return 0


def _home_line(entry: HostEntry) -> str:
    if entry.root is None:
        return ""
    return (
        f"persistent root {entry.root}, so gpuc home (queue, specs, state, logs, "
        f"workdirs) is {entry.remote_home}\n"
    )


# None means "not given, leave it alone"; an empty string means "clear it".
_SET_FIELDS = (
    "gpus",
    "persistent_root",
    "gpuc_home",
    "env",
    "cache_dir",
    "s3_prefix",
    "idle_min",
    "ttl_hours",
)


def cmd_host_set(args: argparse.Namespace) -> int:
    """Edit one registered host in place, without remove/add losing the rest."""
    changes: dict[str, object] = {}
    if args.gpus is not None:
        changes["gpus"] = _gpu_list(args.gpus)
    for flag, field in (
        ("persistent_root", "persistent_root"),
        ("gpuc_home", "gpuc_home"),
        ("cache_dir", "cache_dir"),
        ("s3_prefix", "s3_prefix"),
    ):
        value = getattr(args, flag)
        if value is not None:
            changes[field] = value or None
    if args.env is not None:
        # The whole dict, not a merge: "set it to exactly this" is the only
        # rule that can also express "set it to nothing" (`--env ''`).
        changes["env"] = _env_dict([pair for pair in args.env if pair])
    if args.idle_min is not None:
        changes["idle_minutes"] = args.idle_min
    if args.ttl_hours is not None:
        changes["ttl_hours"] = args.ttl_hours
    if not changes:
        raise CliError(
            "host set changes nothing: pass at least one of "
            + ", ".join(f"--{f.replace('_', '-')}" for f in _SET_FIELDS)
        )
    with registry_transaction() as registry:
        entry = registry.require(args.name)
        updated = entry.model_copy(update=changes)
        registry.put(updated)
    print(
        f"host {updated.name}: "
        + ", ".join(f"{key}={value!r}" for key, value in sorted(changes.items()))
        + f"\n{_home_line(updated)}"
        + f"the host itself is unchanged until: gpuc host bootstrap {updated.name}"
    )
    return 0


def cmd_host_remove(args: argparse.Namespace) -> int:
    with registry_transaction() as registry:
        registry.require(args.name)
        del registry.hosts[args.name]
    print(f"removed host {args.name} from {state_dir()}/hosts.json (nothing on the host changed)")
    return 0


def cmd_host_list(_: argparse.Namespace) -> int:
    registry = load_registry()
    if not registry.hosts:
        print("no hosts registered. Add one: gpuc host add local --gpus GPU-uuid")
        return 0
    for entry in registry.hosts.values():
        bootstrapped = entry.bootstrapped_at or "never bootstrapped"
        print(
            f"{entry.name:<16} {entry.kind:<7} {entry.ssh or 'this machine':<28} "
            f"gpus={len(entry.gpus)} python={entry.python or '-'} bootstrapped={bootstrapped}"
        )
        if entry.root:
            print(f"  persistent root {entry.root} (gpuc home {entry.remote_home})")
        for uuid in entry.gpus:
            print(f"  {uuid}")
    return 0


def cmd_host_bootstrap(args: argparse.Namespace) -> int:
    settings = load_settings()
    entry = load_registry().require(args.name)
    updated, result = bootstrap_host(entry, settings, health_args=args.health_args)
    with registry_transaction() as registry:
        registry.put(updated)
    print(
        f"host {result.host} ready: {result.files} package files at {result.home}/pkg, "
        f"dispatcher pid {result.dispatcher_pid}"
    )
    for warning in result.warnings:
        print(f"WARNING: {warning}")
    return 0


def cmd_clean(args: argparse.Namespace) -> int:
    settings = load_settings()
    entry = load_registry().require(args.host)
    report = clean_host(
        entry,
        settings,
        all_finished=args.all_finished,
        older_than_days=args.older_than,
        dry_run=args.dry_run,
    )
    print(report.render())
    return 1 if report.errors else 0


def cmd_host_clean(args: argparse.Namespace) -> int:
    if not args.uv_cache:
        raise CliError("host clean needs --uv-cache (job workdirs are `gpuc clean --host H`)")
    entry = load_registry().require(args.name)
    print(prune_uv_cache(entry, load_settings()))
    return 0


def cmd_host_probe(args: argparse.Namespace) -> int:
    settings = load_settings()
    entry = load_registry().require(args.name)
    print(probe_host(entry, settings).render())
    return 0


CLOUDS: dict[str, list[Cloud]] = {
    "secure": ["SECURE"],
    "community": ["COMMUNITY"],
    "any": ["SECURE", "COMMUNITY"],
}


def make_provider(settings: Settings) -> Provider:
    return RunPodProvider(caps=settings.caps())


def constraints_from(args: argparse.Namespace) -> Constraints:
    names = _gpu_list(args.gpu)
    if not names:
        raise CliError(
            "--runpod needs --gpu <name>[,<name>] (for example --gpu A40,RTX4090).\n"
            "Names are matched against the RunPod catalog, short or full."
        )
    return Constraints(
        gpu_names=names,
        min_vram_gb=args.min_vram,
        max_price_usd_hr=args.max_price,
        clouds=CLOUDS[args.cloud],
        cuda_min=args.cuda_min,
        gpu_count=args.gpu_count,
    )


def runpod_target(args: argparse.Namespace, settings: Settings) -> HostEntry:
    return runpod_host(
        constraints_from(args),
        settings,
        provider=make_provider(settings),
        reuse=not args.no_reuse,
        name_hint=args.name_hint,
        idle_minutes=args.idle_min,
        ttl_hours=args.ttl_hours,
        disk_gb=args.disk if args.disk is not None else settings.disk_gb,
        image=args.image or settings.image,
        health_args=args.health_args,
    )


def cmd_config_init(args: argparse.Namespace) -> int:
    path = write_config_template(force=args.force)
    print(f"wrote {path}\nEvery key is commented with its default; edit what you need.")
    return 0


def cmd_config_show(_: argparse.Namespace) -> int:
    path = config_file()
    settings = load_settings()
    print(f"config file: {path}{'' if path.exists() else ' (does not exist; using defaults)'}")
    print(f"state dir:   {state_dir()}")
    for name, value in settings.model_dump().items():
        print(f"  {name} = {value!r}")
    if settings.s3_bucket is None:
        print("  note: no s3_bucket, so nothing is mirrored to S3")
    return 0


def mirror_spec_first(model: JobSpecModel, job_id: str, settings: Settings) -> list[str]:
    """Put the spec in S3 before spending any money, so a lost pod is still requeueable."""
    s3 = S3Index.from_settings(settings)
    if s3 is None:
        return [
            "s3_bucket is unset, so the spec was not mirrored before provisioning; "
            "`gpuc requeue` will need the job file again"
        ]
    try:
        s3.put_spec(expand_job_id(model.to_spec(job_id)))
    except S3IndexError as exc:
        return [f"could not mirror the spec to S3 before provisioning: {exc}"]
    return []


def cmd_submit(args: argparse.Namespace) -> int:
    settings = load_settings()
    if args.runpod:
        document = load_document(args.job_file)
        model = validate(document, str(args.job_file))
        precheck_local(model, Path.cwd(), gpu_count=args.gpu_count)
        job_id = jobs.new_job_id()
        notes = mirror_spec_first(model, job_id, settings)
        entry = runpod_target(args, settings)
        result = submit_spec(entry, model, settings, workdir=Path.cwd(), job_id=job_id)
        result.notes.extend(notes)
        print(result.render())
        return 0
    if not args.host:
        raise CliError("submit needs --host <name> (see `gpuc host list`)")
    entry = load_registry().require(args.host)
    result = submit_file(entry, args.job_file, settings, workdir=Path.cwd())
    print(result.render())
    return 0


def _hosts(registry: Registry, only: str | None) -> list[HostEntry]:
    if only:
        return [registry.require(only)]
    return list(registry.hosts.values())


def cmd_status(args: argparse.Namespace) -> int:
    settings = load_settings()
    registry = load_registry()
    entries = _hosts(registry, args.host)
    if not entries:
        print("no hosts registered. Add one: gpuc host add local --gpus GPU-uuid")
        # ...but `--all` still has something to say: the index remembers jobs
        # whose host has since been removed.
        if args.all and not args.suspects:
            _print_unhosted(settings, set(), args.host)
        return 0
    provider = _provider_for_status(entries, settings)
    seen: set[str] = set()
    for entry in entries:
        view = status_mod.gather(entry, settings, provider=provider)
        seen.update(job.job_id for job in view.queue + view.running + view.finished)
        print(status_mod.render(view, suspects_only=args.suspects))
    if args.all and not args.suspects:
        _print_unhosted(settings, seen, args.host)
    return 0


def _provider_for_status(entries: list[HostEntry], settings: Settings) -> Provider | None:
    """Only build a provider when an ephemeral host is on screen, and never fail on it."""
    if not any(entry.kind == "runpod" for entry in entries):
        return None
    try:
        return make_provider(settings)
    except ProviderError as exc:
        print(f"note: pod status unavailable: {exc}", file=sys.stderr)
        return None


def _print_unhosted(settings: Settings, seen: set[str], host: str | None = None) -> None:
    """The index's view of jobs no host admitted to having.

    After a host loses its state -- a container whose $HOME was wiped, a pod
    that is gone -- this is the only list of what was on it, and `gpuc requeue
    <id> --host <name>` is how each one comes back, so `--host H --all` narrows
    it to the host being recovered.
    """
    entries = {entry.job_id: entry for entry in LocalIndex().list()}
    s3 = S3Index.from_settings(settings)
    if s3 is not None:
        try:
            entries.update({e.job_id: e for e in s3.list_index()})
        except S3IndexError as exc:
            print(f"note: could not read the S3 index: {exc}")
    elsewhere = [
        entry
        for job_id, entry in sorted(entries.items())
        if job_id not in seen and (host is None or entry.host == host)
    ]
    if not elsewhere:
        return
    scope = f" for host {host}" if host else ""
    print(f"jobs known only to the index{scope} (their host is gone, or lost its state):")
    for entry in elsewhere:
        print(f"  {entry.job_id} {entry.name or '-'} host={entry.host} attempt={entry.attempt}")
    print(f"  bring one back with: gpuc requeue {elsewhere[0].job_id} --host {elsewhere[0].host}")


def find_job_host(
    job_id: str, registry: Registry, explicit: str | None
) -> tuple[HostEntry, IndexEntry | None]:
    index = LocalIndex().get(job_id)
    if explicit:
        return registry.require(explicit), index
    if index is not None and index.host in registry.hosts:
        return registry.hosts[index.host], index
    for entry in registry.hosts.values():
        try:
            payload = open_session(entry).host_json(f"status {shlex.quote(job_id)}", timeout=60.0)
        except (RemoteError, TransportError):
            continue
        if payload.get("jobs"):
            return entry, index
    raise CliError(
        f"no registered host knows job {job_id}.\n"
        f"Pass --host <name>, or check `gpuc host list` and `gpuc status --all`."
    )


def cmd_cancel(args: argparse.Namespace) -> int:
    entry, _ = find_job_host(args.job_id, load_registry(), args.host)
    payload = open_session(entry, load_settings()).host_json(f"cancel {shlex.quote(args.job_id)}")
    print(f"job {args.job_id} on host {entry.name}: {payload.get('status')}")
    return 0


def cmd_reorder(args: argparse.Namespace) -> int:
    entry, _ = find_job_host(args.job_id, load_registry(), args.host)
    session = open_session(entry, load_settings())
    result = session.host_cli(f"reorder {shlex.quote(args.job_id)} {args.priority}", check=False)
    if result.returncode != 0:
        raise CliError(
            f"job {args.job_id} is not in host {entry.name}'s queue, so its priority cannot "
            f"change (a running or finished job cannot be reordered)."
        )
    print(f"job {args.job_id} on host {entry.name} moved to priority {args.priority}")
    return 0


def _follow_argv(transport: Transport, remote_path: str, lines: int) -> list[str]:
    command = f"tail -n {lines} -f {shlex.quote(remote_path)}"
    if isinstance(transport, SshTransport):
        return transport.ssh_argv(command)
    return ["bash", "-lc", command]


def cmd_logs(args: argparse.Namespace) -> int:
    settings = load_settings()
    entry, index = find_job_host(args.job_id, load_registry(), args.host)
    remote = None
    try:
        session = open_session(entry, settings)
        remote = f"{session.job_dir(args.job_id)}/log.txt"
        if args.follow:
            return _follow(session.transport, remote, args.lines)
        result = session.transport.tail(remote, lines=args.lines)
        if result.returncode == 0:
            sys.stdout.write(result.stdout)
            return 0
        note = result.output.strip().splitlines()[-1:] or ["no log file on the host"]
    except (RemoteError, TransportError) as exc:
        note = [str(exc).splitlines()[0]]
    print(f"note: could not read {remote or 'the host log'}: {note[0]}", file=sys.stderr)
    return _logs_from_s3(args.job_id, entry, index, settings)


def _follow(transport: Transport, remote: str, lines: int) -> int:
    argv = _follow_argv(transport, remote, lines)
    try:
        return subprocess.call(argv)
    except KeyboardInterrupt:
        return 0


def _logs_from_s3(
    job_id: str, entry: HostEntry, index: IndexEntry | None, settings: Settings
) -> int:
    s3 = S3Index.from_settings(settings)
    prefix = (index.s3_prefix if index else None) or entry.s3_prefix
    if s3 is None or not prefix:
        raise CliError(
            f"no S3 mirror to fall back on for {job_id}.\n"
            f"Set s3_bucket in ~/.config/gpu-coordinator/config.toml to keep logs after a "
            f"host goes away."
        )
    uri = job_log_uri(prefix, job_id)
    print(f"note: falling back to the S3 mirror at {uri}", file=sys.stderr)
    sys.stdout.write(s3.get_uri(uri))
    return 0


def cmd_requeue(args: argparse.Namespace) -> int:
    settings = load_settings()
    registry = load_registry()
    index = LocalIndex().get(args.job_id)
    target = args.host or (None if args.runpod else (index.host if index else None))
    if not target and not args.runpod:
        raise CliError(f"requeue needs --host <name>: nothing local knows where {args.job_id} ran")
    s3 = S3Index.from_settings(settings)
    if s3 is None:
        raise CliError(
            "requeue reads the spec from S3, but s3_bucket is unset in "
            "~/.config/gpu-coordinator/config.toml. Re-submit the job file instead."
        )
    document = s3.get_spec(args.job_id)
    for key in ("job_id", "attempt"):
        document.pop(key, None)
    attempt = (index.attempt if index else 1) + 1
    model = validate(document, f"spec for {args.job_id}")
    if target is None:
        precheck_local(model, Path.cwd(), gpu_count=args.gpu_count)
    entry = runpod_target(args, settings) if target is None else registry.require(target)
    result = submit_spec(
        entry,
        model,
        settings,
        workdir=Path.cwd(),
        attempt=attempt,
    )
    print(result.render())
    print(f"  requeued from {args.job_id} (attempt {attempt}); workdir re-synced from {Path.cwd()}")
    return 0


def cmd_reconcile(args: argparse.Namespace) -> int:
    if args.install:
        reconcile_mod.install(args.interval)
        return 0
    settings = load_settings()
    provider = make_provider(settings)
    if args.once:
        result = reconcile_mod.reconcile_once(settings, provider)
        print(result.render())
        return 1 if result.errors else 0
    print(f"reconciling every {args.interval:.0f}s; Ctrl-C to stop")
    try:
        reconcile_mod.run_loop(settings, provider, interval_s=args.interval)
    except KeyboardInterrupt:
        print("stopped")
    return 0


def cmd_pods(args: argparse.Namespace) -> int:
    settings = load_settings()
    view = pods_mod.gather(settings, make_provider(settings), heartbeats=not args.no_heartbeat)
    print(pods_mod.render(view))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gpuc", description="GPU job coordinator")
    sub = parser.add_subparsers(dest="command", required=True)

    host = sub.add_parser("host", help="manage hosts").add_subparsers(
        dest="host_command", required=True
    )
    add = host.add_parser("add", help="register a host")
    add.add_argument("name")
    add.add_argument("--ssh", help="user@host; omit for this machine")
    add.add_argument("--port", type=int, default=22)
    add.add_argument("--gpus", help="comma-separated GPU UUIDs this host may use")
    add.add_argument("--gpuc-home", help="override ~/.gpuc on the host")
    add.add_argument(
        "--persistent-root",
        help="a directory on a volume that survives restarts; gpuc home "
        "(queue, specs, state, logs, workdirs) moves to PATH/gpuc",
    )
    add.add_argument(
        "--env",
        action="append",
        metavar="KEY=VALUE",
        help="extra environment for every job on this host; repeatable",
    )
    add.add_argument(
        "--cache-dir",
        help="uv cache for this host (UV_CACHE_DIR); bootstrap picks one on gpuc home's "
        "filesystem when they differ",
    )
    add.add_argument("--s3-prefix", help="s3://bucket/prefix for log and state mirroring")
    add.add_argument("--idle-min", type=float, default=15.0)
    add.add_argument("--ttl-hours", type=float, default=24.0)
    add.set_defaults(func=cmd_host_add)

    edit = host.add_parser("set", help="change a registered host without remove/add")
    edit.add_argument("name")
    edit.add_argument("--gpus", help="replace the GPU UUIDs; pass '' for none")
    edit.add_argument("--persistent-root", help="pass '' to go back to $HOME")
    edit.add_argument("--gpuc-home", help="pass '' for the default under the root or $HOME")
    edit.add_argument(
        "--env",
        action="append",
        metavar="KEY=VALUE",
        help="replace this host's job environment; repeatable, '' for none",
    )
    edit.add_argument(
        "--cache-dir", help="pin UV_CACHE_DIR for this host; pass '' to let bootstrap decide"
    )
    edit.add_argument("--s3-prefix", help="pass '' to stop mirroring")
    edit.add_argument("--idle-min", type=float)
    edit.add_argument("--ttl-hours", type=float)
    edit.set_defaults(func=cmd_host_set)

    bootstrap = host.add_parser("bootstrap", help="install uv, the package and the dispatcher")
    bootstrap.add_argument("name")
    bootstrap.add_argument(
        "--health-args", default="", help="extra flags for `gpuc.host health`, e.g. --min-mbps 0.1"
    )
    bootstrap.set_defaults(func=cmd_host_bootstrap)

    probe = host.add_parser("probe", help="report what a host has, before bootstrap")
    probe.add_argument("name")
    probe.set_defaults(func=cmd_host_probe)

    host_clean = host.add_parser("clean", help="prune the host's uv cache")
    host_clean.add_argument("name")
    host_clean.add_argument(
        "--uv-cache", action="store_true", help="run `uv cache prune` on the host"
    )
    host_clean.set_defaults(func=cmd_host_clean)

    host.add_parser("list", help="list registered hosts").set_defaults(func=cmd_host_list)
    remove = host.add_parser("remove", help="forget a host")
    remove.add_argument("name")
    remove.set_defaults(func=cmd_host_remove)

    submit = sub.add_parser("submit", help="submit a job file to a host")
    submit.add_argument("job_file")
    submit.add_argument("--host")
    add_runpod_flags(submit)
    submit.set_defaults(func=cmd_submit)

    status = sub.add_parser("status", help="per-host queue, running and recent jobs")
    status.add_argument("--host")
    status.add_argument("--all", action="store_true", help="also list jobs only the index knows")
    status.add_argument("--suspects", action="store_true", help="billing but idle; never kills")
    status.set_defaults(func=cmd_status)

    clean = sub.add_parser("clean", help="remove finished jobs' workdirs on a host")
    clean.add_argument("--host", required=True)
    selection = clean.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--all-finished", action="store_true", help="every succeeded, failed or cancelled job"
    )
    selection.add_argument(
        "--older-than", type=float, metavar="DAYS", help="only jobs that ended over DAYS ago"
    )
    clean.add_argument("--dry-run", action="store_true", help="list what would go, delete nothing")
    clean.set_defaults(func=cmd_clean)

    logs = sub.add_parser("logs", help="tail a job log from its host")
    logs.add_argument("job_id")
    logs.add_argument("-f", "--follow", action="store_true")
    logs.add_argument("-n", "--lines", type=int, default=200)
    logs.add_argument("--host")
    logs.set_defaults(func=cmd_logs)

    cancel = sub.add_parser("cancel", help="cancel a queued or running job")
    cancel.add_argument("job_id")
    cancel.add_argument("--host")
    cancel.set_defaults(func=cmd_cancel)

    reorder = sub.add_parser("reorder", help="change a queued job's priority")
    reorder.add_argument("job_id")
    reorder.add_argument("--priority", type=int, required=True)
    reorder.add_argument("--host")
    reorder.set_defaults(func=cmd_reorder)

    requeue = sub.add_parser("requeue", help="resubmit a job from its S3 spec")
    requeue.add_argument("job_id")
    requeue.add_argument("--host")
    add_runpod_flags(requeue)
    requeue.set_defaults(func=cmd_requeue)

    reconcile = sub.add_parser(
        "reconcile", help="terminate leaked or expired pods; --install for a systemd timer"
    )
    reconcile.add_argument("--once", action="store_true", help="one pass, then exit")
    reconcile.add_argument("--interval", type=float, default=reconcile_mod.DEFAULT_INTERVAL_S)
    reconcile.add_argument(
        "--install", action="store_true", help="write (but do not enable) systemd --user units"
    )
    reconcile.set_defaults(func=cmd_reconcile)

    pods = sub.add_parser("pods", help="every pod with our prefix, cost, util, age, desired?")
    pods.add_argument(
        "--no-heartbeat", action="store_true", help="skip the per-pod dispatcher ssh check"
    )
    pods.set_defaults(func=cmd_pods)

    config = sub.add_parser("config", help="show or create the settings file").add_subparsers(
        dest="config_command", required=True
    )
    config_init = config.add_parser("init", help="write a commented config.toml")
    config_init.add_argument("--force", action="store_true", help="overwrite an existing file")
    config_init.set_defaults(func=cmd_config_init)
    config.add_parser("show", help="print the effective settings").set_defaults(
        func=cmd_config_show
    )
    return parser


def add_runpod_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--runpod", action="store_true", help="reuse or provision a RunPod pod")
    parser.add_argument("--gpu", help="comma-separated GPU names, cheapest match wins")
    parser.add_argument("--gpu-count", type=int, default=1)
    parser.add_argument("--min-vram", type=int)
    parser.add_argument("--max-price", type=float, help="USD per hour, for the whole pod")
    parser.add_argument("--cloud", choices=sorted(CLOUDS), default="secure")
    parser.add_argument("--cuda-min", default=None, help="host CUDA floor, default 12.8")
    parser.add_argument("--idle-min", type=float, default=15.0)
    parser.add_argument("--ttl-hours", type=float, default=24.0)
    parser.add_argument("--disk", type=int, help="container disk in GB; default from config")
    parser.add_argument("--image", help="pod image; default from config")
    parser.add_argument("--no-reuse", action="store_true", help="always create a new pod")
    parser.add_argument("--name-hint", default="job", help="goes into the pod name")
    parser.add_argument("--health-args", default="", help="extra flags for `gpuc.host health`")


def wants_runpod(args: argparse.Namespace) -> bool:
    if getattr(args, "install", False):
        return False  # `reconcile --install` only writes unit files
    return bool(getattr(args, "runpod", False)) or args.command in ("pods", "reconcile")


def first_run_note() -> None:
    """One line, once, on stderr: defaults are fine, but say where to change them."""
    if not config_file().exists():
        print(
            f"note: no config file at {config_file()} (using defaults, no S3 mirror); "
            f"run `gpuc config init` to create one",
            file=sys.stderr,
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    if wants_runpod(args) and not os.environ.get("RUNPOD_API_KEY"):
        # Before anything else: provisioning spends money, and finding out after
        # the spec has been mirrored and a host picked helps nobody.
        print(
            "error: RUNPOD_API_KEY is not set; export it before using --runpod, "
            "`gpuc pods` or `gpuc reconcile`",
            file=sys.stderr,
        )
        return 1
    if args.command != "config":
        first_run_note()
    try:
        return int(args.func(args))
    except (
        CleanError,
        CliError,
        ConfigError,
        SubmitError,
        BootstrapError,
        ProvisionError,
        ProviderError,
        RemoteError,
        S3IndexError,
        TransportError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except json.JSONDecodeError as exc:
        print(f"error: a host returned malformed JSON: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
