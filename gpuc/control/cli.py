"""The `gpuc` command line. Thin: parse, call a module, print, map errors to 1."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from gpuc.control import jsonout
from gpuc.control import pods as pods_mod
from gpuc.control import reconcile as reconcile_mod
from gpuc.control import ssh as ssh_mod
from gpuc.control import status as status_mod
from gpuc.control import version as version_mod
from gpuc.control.bootstrap import BootstrapError, bootstrap_host, resync_package
from gpuc.control.clean import CleanError, clean_host, parse_only, prune_uv_cache
from gpuc.control.clean import check_flags as check_clean_flags
from gpuc.control.config import (
    ConfigError,
    HostEntry,
    HostNotFound,
    LocalStateUnreadable,
    Registry,
    RegistryRead,
    Settings,
    config_file,
    hosts_file,
    load_settings,
    read_registry,
    registry_transaction,
    state_dir,
    transport_for,
    utc_now,
    write_config_template,
)
from gpuc.control.gpuinfo import rows as gpu_rows
from gpuc.control.gpuinfo import summarize
from gpuc.control.probe import probe_host
from gpuc.control.providers.base import Cloud, Constraints, Provider, ProviderError
from gpuc.control.providers.runpod import RunPodProvider
from gpuc.control.provision import ProvisionError, runpod_host
from gpuc.control.remote import HostSession, RemoteError, open_session
from gpuc.control.s3index import (
    IndexEntry,
    LocalIndex,
    S3Index,
    S3IndexError,
    S3ObjectMissing,
    job_log_uri,
)
from gpuc.control.skill import SkillError, install_skill, read_skill
from gpuc.control.submit import (
    JobSpecModel,
    Reporter,
    SubmitError,
    SubmitResult,
    expand_job_id,
    load_document,
    precheck_local,
    submit_file,
    submit_spec,
    validate,
)
from gpuc.control.transport import (
    NO_GIT_EXCLUDES,
    SshTransport,
    Transport,
    TransportError,
)
from gpuc.host import jobs
from gpuc.host.cleanup import DEFAULT_RETENTION_DAYS

EXIT_OK = 0
"""Everything the command was asked to do happened, including reporting that a
host is unreachable: that is data about a host, not a failure of the command."""
EXIT_ERROR = 1
"""The command failed: a transport error, a provider error, a refused submit."""
EXIT_USAGE = 2
"""The command line itself was wrong (argparse uses this too)."""
EXIT_LOCAL_STATE = 3
"""Local state -- the registry or the config file -- could not be read, so the
answer is unknown. Automation must not read this as `nothing is running`."""
EXIT_NOT_FOUND = 4
"""The named job or host does not exist."""


class CliError(RuntimeError):
    exit_code = EXIT_ERROR


class UsageError(CliError):
    """The invocation was wrong, not the world."""

    exit_code = EXIT_USAGE


class NotFound(CliError):
    """The job or host named on the command line does not exist."""

    exit_code = EXIT_NOT_FOUND


GPUS_HELP = (
    "GPU UUIDs or nvidia-smi indices this host may use, comma-separated "
    "(`--gpus 2,3` or `--gpus GPU-8064...,3`); indices are resolved to UUIDs on "
    "the host at every dispatch pass, so jobs are always pinned by UUID"
)


def _comma_list(raw: str | None) -> list[str]:
    """A comma- or space-separated flag value, as a list."""
    if not raw:
        return []
    return [part.strip() for part in raw.replace(" ", ",").split(",") if part.strip()]


def _gpu_list(raw: str | None) -> list[str]:
    """`--gpus`: UUIDs, nvidia-smi indices, or a mix, stored exactly as given.

    Ownership of part of a shared box is an agreement in nvidia-smi numbering
    ("you have 2 and 3"), so an index has to be sayable and has to stay what
    was said -- resolving it here would freeze this morning's numbering into
    the registry. The host redoes the mapping each pass; all this has to do is
    refuse the third thing, which is always a typo.
    """
    owned = _comma_list(raw)
    bad = [item for item in owned if not item.isdigit() and not item.startswith("GPU-")]
    if bad:
        raise UsageError(
            f"--gpus wants nvidia-smi indices or GPU UUIDs, got {', '.join(repr(b) for b in bad)}."
            f"\nRun `gpuc host probe <name>` to see this host's index and UUID for each card."
        )
    return owned


def _env_dict(pairs: Sequence[str] | None) -> dict[str, str]:
    env: dict[str, str] = {}
    for pair in pairs or []:
        key, sep, value = pair.partition("=")
        if not sep or not key.strip():
            raise UsageError(f"--env wants KEY=VALUE, got {pair!r}")
        env[key.strip()] = value
    return env


def named_registry() -> Registry:
    """The registry, for a command that was given a host or job name to find.

    A registry that could not be parsed is exit 3 (unknown), not exit 4 (does
    not exist): "no host named gpubox" would be a lie when the file holding
    gpubox is the thing that is broken. Listing commands do not use this --
    they can honestly show what parsed.
    """
    read = read_registry()
    for error in read.errors:
        print(f"warning: {error}", file=sys.stderr)
    if read.unreadable:
        raise LocalStateUnreadable("\n".join(read.errors))
    return read.registry


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
        retention_days=_retention(args.retention_days),
        idle_minutes=args.idle_min,
        ttl_hours=_ttl_hours(args.ttl_hours),
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


def _ttl_hours(raw: float | None) -> float | None:
    """`--ttl-hours`: hours, or a negative sentinel meaning "no TTL at all".

    A stored -1 would be a host that is *already* past its TTL, so the next
    reaper pass terminates it -- the opposite of what anyone types it for, and
    the same rule `gpuc host set` has always used for clearing one.
    """
    if raw is None or raw < 0:
        return None
    if raw == 0:
        raise UsageError(
            "--ttl-hours 0 would expire the host the moment it exists; "
            "pass -1 (or omit it) for no TTL"
        )
    return raw


def _retention(raw: str | None) -> float | None:
    """`--retention-days`: a number, or '' to go back to keeping everything."""
    if raw is None or raw == "":
        return None
    try:
        days = float(raw)
    except ValueError as exc:
        raise UsageError(f"--retention-days wants a number of days or '', got {raw!r}") from exc
    if days < 0:
        raise UsageError("--retention-days cannot be negative")
    return days


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
    "retention_days",
    "idle_min",
    "ttl_hours",
)


def cmd_host_set(args: argparse.Namespace) -> int:
    """Edit one registered host in place, without remove/add losing the rest."""
    changes: dict[str, object] = {}
    if args.gpus is not None:
        changes["gpus"] = _gpu_list(args.gpus)
    for flag, attribute in (
        ("persistent_root", "persistent_root"),
        ("gpuc_home", "gpuc_home"),
        ("cache_dir", "cache_dir"),
        ("s3_prefix", "s3_prefix"),
    ):
        value = getattr(args, flag)
        if value is not None:
            changes[attribute] = value or None
    if args.env is not None:
        # The whole dict, not a merge: "set it to exactly this" is the only
        # rule that can also express "set it to nothing" (`--env ''`).
        changes["env"] = _env_dict([pair for pair in args.env if pair])
    if args.retention_days is not None:
        changes["retention_days"] = _retention(args.retention_days)
    if args.idle_min is not None:
        changes["idle_minutes"] = args.idle_min
    if args.ttl_hours is not None:
        # argparse cannot express "given but empty" for a float flag, and a TTL
        # that can be set but never unset is a trap.
        changes["ttl_hours"] = _ttl_hours(args.ttl_hours)
    if not changes:
        raise UsageError(
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


def cmd_host_resume(args: argparse.Namespace) -> int:
    entry = named_registry().require(args.name)
    open_session(entry, load_settings()).host_cli("resume")
    print(f"host {args.name}: low-util pause cleared, dispatcher restarted")
    return 0


def host_document(entry: HostEntry) -> dict[str, Any]:
    """One registered host as `gpuc host list --json` reports it.

    The registry entry itself, plus what the text listing computes from it:
    where gpuc home resolves to on the host, and the re-bootstrap warning.
    """
    document: dict[str, Any] = json.loads(entry.model_dump_json())
    stale = status_mod.stale_warning(entry)
    return {
        **document,
        # `--env` is free-form and is where somebody hand-sets an HF_TOKEN, so
        # the names are reported and the values are not: the text listing shows
        # neither, and this document ends up in transcripts and bug reports.
        "env": dict.fromkeys(entry.env, "<set>"),
        "remote_home": entry.remote_home,
        "ephemeral": entry.ephemeral,
        "warnings": [stale] if stale else [],
    }


def cmd_host_list(args: argparse.Namespace) -> int:
    read = read_registry()
    for error in read.errors:
        print(f"warning: {error}", file=sys.stderr)
    registry = read.registry
    if args.json:
        jsonout.emit(
            {
                "hosts": [host_document(entry) for entry in registry.hosts.values()],
                "errors": list(read.errors),
            }
        )
        return EXIT_LOCAL_STATE if read.unreadable else EXIT_OK
    if not registry.hosts:
        if read.unreadable:
            return EXIT_LOCAL_STATE
        print("no hosts registered. Add one: gpuc host add local --gpus GPU-uuid")
        return EXIT_OK
    for entry in registry.hosts.values():
        bootstrapped = entry.bootstrapped_at or "never bootstrapped"
        summary = summarize(entry.gpus, entry.gpu_info) if entry.gpus else "no GPUs"
        driver = f", driver {entry.driver_version}" if entry.driver_version else ""
        print(
            f"{entry.name:<16} {entry.kind:<7} {entry.ssh or 'this machine':<28} "
            f"gpus={len(entry.gpus)} ({summary}{driver}) python={entry.python or '-'} "
            f"pkg={version_mod.short(entry.pkg_commit)} bootstrapped={bootstrapped}"
        )
        stale = status_mod.stale_warning(entry)
        if stale:
            print(f"  WARNING {stale}")
        if entry.root:
            print(f"  persistent root {entry.root} (gpuc home {entry.remote_home})")
        for index, name, vram, uuid in gpu_rows(entry.gpus, entry.gpu_info):
            print(f"  [{index}] {name:<28} {vram:<7} {uuid}")
    return 0


def bootstrap_and_record(entry: HostEntry, settings: Settings, health_args: str) -> None:
    """Bootstrap one host, persist what it told us about itself, and say so."""
    updated, result = bootstrap_host(entry, settings, health_args=health_args)
    with registry_transaction() as registry:
        registry.put(updated)
    print(
        f"host {result.host} ready: {result.files} package files at {result.home}/pkg "
        f"({version_mod.short(result.pkg_commit)}), dispatcher pid {result.dispatcher_pid}"
    )
    if result.warnings:
        print(f"{len(result.warnings)} warning(s) above")


def bootstrap_tally(total: int, done: int, failed: Sequence[HostEntry], skipped: int) -> str:
    """The last word of a `--all` run: what worked, what did not, what was never read.

    Counted rather than claimed, because the run this ends can be long enough
    that nobody reads the middle of it: a host this build could not parse out
    of the registry was never bootstrapped either, and saying "all of them"
    over the top of that warning is how one gets missed for a month.
    """
    lines = [f"{done}/{total} host(s) bootstrapped"]
    if failed:
        lines.append(f"failed: {', '.join(entry.name for entry in failed)}")
        if any(entry.ephemeral for entry in failed):
            lines.append(
                "an ephemeral host whose pod is already gone is forgotten by "
                "`gpuc reconcile --once`"
            )
    if skipped:
        lines.append(f"{skipped} host(s) in the registry could not be read (warnings above)")
    return "\n".join(lines)


def bootstrap_every_host(settings: Settings, health_args: str) -> int:
    """`gpuc host bootstrap --all`: the upgrade loop, one command.

    A host that fails does not stop the others: an ephemeral host whose pod is
    already gone is the ordinary case, and the hosts that are still there are
    the reason the flag exists. Each failure is named again in the tally and
    the command exits 1, so nobody reads a wall of output as "all upgraded".
    """
    read = read_registry()
    for error in read.errors:
        print(f"warning: {error}", file=sys.stderr)
    if read.unreadable:
        raise LocalStateUnreadable("\n".join(read.errors))
    hosts = list(read.registry.hosts.values())
    if not hosts:
        print("no hosts registered. Add one: gpuc host add local --gpus GPU-uuid")
        return EXIT_OK
    done = 0
    failed: list[HostEntry] = []
    for index, entry in enumerate(hosts, start=1):
        if index > 1:
            print()
        print(f"== {entry.name} ({index}/{len(hosts)}) ==")
        try:
            bootstrap_and_record(entry, settings, health_args)
            done += 1
        except KeyboardInterrupt:
            # Health alone allows five minutes a host, so this is a command
            # somebody does give up on; what it got through is still true.
            print(f"\ninterrupted during {entry.name}")
            print(bootstrap_tally(len(hosts), done, failed, len(read.skipped)))
            return EXIT_ERROR
        except LocalStateUnreadable:
            # The registry stopped being readable mid-run, so the next host's
            # write would be a guess: say how far this got, and exit 3.
            print(f"\n{bootstrap_tally(len(hosts), done, failed, len(read.skipped))}")
            raise
        except (BootstrapError, ConfigError, RemoteError, TransportError) as exc:
            print(f"error: host {entry.name}: {exc}", file=sys.stderr)
            failed.append(entry)
    print()
    print(bootstrap_tally(len(hosts), done, failed, len(read.skipped)))
    return EXIT_ERROR if failed else EXIT_OK


def cmd_host_bootstrap(args: argparse.Namespace) -> int:
    settings = load_settings()
    if args.all:
        if args.name:
            raise UsageError(
                f"host bootstrap takes a host name or --all, not both (got {args.name!r})"
            )
        return bootstrap_every_host(settings, args.health_args)
    if not args.name:
        raise UsageError("host bootstrap wants a host name, or --all for every registered host")
    bootstrap_and_record(named_registry().require(args.name), settings, args.health_args)
    return EXIT_OK


def cmd_clean(args: argparse.Namespace) -> int:
    settings = load_settings()
    # Checked before the host lookup so a wrong command line answers in the
    # same way whether or not the host exists. `clean` owns the rule; keeping a
    # second copy here is what let the two disagree about the exit code.
    only = parse_only(args.only)
    check_clean_flags(
        all_finished=args.all_finished,
        older_than_days=args.older_than,
        dry_run=args.dry_run,
        purge=args.purge,
        force=args.force,
        verify=args.verify,
        yes=args.yes,
        only=only,
    )
    entry = named_registry().require(args.host)
    report = clean_host(
        entry,
        settings,
        all_finished=args.all_finished,
        older_than_days=args.older_than,
        dry_run=args.dry_run,
        purge=args.purge,
        force=args.force,
        verify=args.verify,
        yes=args.yes,
        only=only,
    )
    if args.json:
        jsonout.emit(report.document())
    else:
        print(report.render())
    return EXIT_ERROR if report.errors else EXIT_OK


def cmd_host_clean(args: argparse.Namespace) -> int:
    if not args.uv_cache:
        raise UsageError("host clean needs --uv-cache (job workdirs are `gpuc clean --host H`)")
    entry = named_registry().require(args.name)
    print(prune_uv_cache(entry, load_settings()))
    return 0


def cmd_host_probe(args: argparse.Namespace) -> int:
    settings = load_settings()
    entry = named_registry().require(args.name)
    report = probe_host(entry, settings)
    if not args.json:
        print(report.render(all_gpus=args.all_gpus))
    # A probe is the one command that runs before bootstrap, so it is also the
    # first chance to learn what the cards are. Every card, not just the assigned
    # ones: this is what makes `gpuc host set <name> --gpus 5` nameable later.
    if report.gpu_info:
        with registry_transaction() as registry:
            current = registry.hosts.get(args.name)
            if current is not None:
                registry.put(
                    current.model_copy(
                        update={
                            "gpu_info": {**current.gpu_info, **report.gpu_info},
                            "driver_version": report.driver_version or current.driver_version,
                        }
                    )
                )
    # After the registry write, not before: that write can fail (a held lock, a
    # registry that changed under us) and print an error document of its own,
    # and stdout may hold only one.
    if args.json:
        jsonout.emit(report.document())
    return EXIT_OK


CLOUDS: dict[str, list[Cloud]] = {
    "secure": ["SECURE"],
    "community": ["COMMUNITY"],
    "any": ["SECURE", "COMMUNITY"],
}


def make_provider(settings: Settings) -> Provider:
    return RunPodProvider(caps=settings.caps())


def constraints_from(args: argparse.Namespace) -> Constraints:
    names = _comma_list(args.gpu)
    if not names:
        raise UsageError(
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
        report=reporter(args),
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


def mirror_spec_first(
    model: JobSpecModel, job_id: str, settings: Settings
) -> tuple[str | None, list[str]]:
    """Put the spec in S3 before spending any money, so a lost pod is still requeueable.

    Returns the uri it landed at, so the submit that follows does not PUT the
    same object a second time.
    """
    s3 = S3Index.from_settings(settings)
    if s3 is None:
        return None, [
            "s3_bucket is unset, so the spec was not mirrored before provisioning; "
            "`gpuc requeue` will need the job file again"
        ]
    try:
        return s3.put_spec(expand_job_id(model.to_spec(job_id))), []
    except S3IndexError as exc:
        return None, [f"could not mirror the spec to S3 before provisioning: {exc}"]


def check_runpod_args(args: argparse.Namespace) -> None:
    """Judge the provisioning flags once, before anything is bought or written."""
    if getattr(args, "runpod", False) and args.host:
        raise UsageError(
            f"--runpod creates a pod and --host {args.host} names a host that already "
            f"exists, so they cannot be combined. Drop one."
        )
    args.ttl_hours = _ttl_hours(args.ttl_hours)


def reporter(args: argparse.Namespace) -> Reporter:
    """Where a step's progress goes: stdout, or stderr when stdout is a document."""
    return jsonout.note if getattr(args, "json", False) else print


def ensure_package_current(
    entry: HostEntry, settings: Settings, *, bootstrap: bool = True, report: Reporter = print
) -> HostEntry:
    """Re-ship the package when the host is not running this build.

    A host on an older commit dispatches the job with code that does not match
    the spec this machine just wrote, and that mismatch is invisible until a
    job fails strangely. An *unrecorded* commit counts as older, because the
    hosts with nothing recorded were bootstrapped by the oldest builds of all.
    Only the package and the dispatcher: uv, the interpreter and health cannot
    have gone stale, and the job is waiting.
    """
    if not bootstrap or not entry.python:
        return entry
    local = version_mod.local_commit()
    if not version_mod.needs_package_sync(local, entry.pkg_commit):
        return entry
    report(
        f"host {entry.name} has gpuc {version_mod.short(entry.pkg_commit)} and this machine "
        f"has {version_mod.short(local)}: re-syncing the package and restarting the "
        f"dispatcher before enqueueing"
    )
    updated = resync_package(entry, settings, report=_quiet)
    with registry_transaction() as registry:
        registry.put(updated)
    return updated


def _quiet(_: str) -> None:
    """Swallow a step's progress: the caller has already said what it is doing."""


def cmd_submit(args: argparse.Namespace) -> int:
    settings = load_settings()
    check_runpod_args(args)
    use_git = not args.no_git
    report = reporter(args)
    if args.runpod:
        document = load_document(args.job_file)
        model = validate(document, str(args.job_file))
        precheck_local(
            model,
            Path.cwd(),
            gpu_count=args.gpu_count,
            use_git=use_git,
            ttl_hours=args.ttl_hours,
            report=report,
        )
        job_id = jobs.new_job_id()
        spec_uri, notes = mirror_spec_first(model, job_id, settings)
        entry = runpod_target(args, settings)
        entry = ensure_package_current(
            entry, settings, bootstrap=not args.no_bootstrap, report=report
        )
        result = submit_spec(
            entry,
            model,
            settings,
            workdir=Path.cwd(),
            job_id=job_id,
            spec_uri=spec_uri,
            use_git=use_git,
            report=report,
        )
        result.notes.extend(notes)
        return _queued(result, args)
    if not args.host:
        raise UsageError("submit needs --host <name> (see `gpuc host list`)")
    entry = named_registry().require(args.host)
    entry = ensure_package_current(entry, settings, bootstrap=not args.no_bootstrap, report=report)
    result = submit_file(
        entry, args.job_file, settings, workdir=Path.cwd(), use_git=use_git, report=report
    )
    return _queued(result, args)


def _queued(
    result: SubmitResult, args: argparse.Namespace, *, requeued_from: str | None = None
) -> int:
    """The last word of `submit` and `requeue`, in whichever form was asked for."""
    if args.json:
        jsonout.emit(result.document(requeued_from=requeued_from))
        return EXIT_OK
    print(result.render())
    if requeued_from is not None:
        print(
            f"  requeued from {requeued_from} (attempt {result.attempt}); "
            f"workdir re-synced from {Path.cwd()}"
        )
    return EXIT_OK


def _hosts(registry: Registry, only: str | None) -> list[HostEntry]:
    if only:
        return [registry.require(only)]
    return list(registry.hosts.values())


def cmd_status(args: argparse.Namespace) -> int:
    """Report on every host, and only fail for reasons that are not a host.

    An unreachable host, a dead dispatcher and a pod that is gone are all
    *answers*, printed per host with exit 0. The one non-zero case is exit 3:
    the local registry could not be read, so "no jobs running" would be a
    guess. Automation must treat that as unknown and never as idle.
    """
    settings = load_settings()
    read = read_registry()
    try:
        since_s = status_mod.parse_duration(args.since) if args.since else None
    except ValueError as exc:
        raise UsageError(f"--since: {exc}") from exc
    entries = _hosts(read.registry, args.host) if not read.unreadable else []
    if args.json:
        return _status_json(args, settings, read, entries, since_s)
    for error in read.errors:
        print(f"warning: {error}", file=sys.stderr)
    if not entries:
        if read.unreadable:
            print(
                f"cannot read {hosts_file()}, so no host status is known "
                f"(this is not `no jobs running`)",
                file=sys.stderr,
            )
            return EXIT_LOCAL_STATE
        print("no hosts registered. Add one: gpuc host add local --gpus GPU-uuid")
        # ...but `--all` still has something to say: the index remembers jobs
        # whose host has since been removed.
        if args.all and not args.suspects:
            _print_unhosted(settings, set(), args.host)
        return EXIT_OK
    provider = _provider_for_status(entries, settings)
    seen: set[str] = set()
    for entry in entries:
        view = status_mod.gather(entry, settings, provider=provider)
        seen.update(job.job_id for job in view.queue + view.running + view.finished)
        print(
            status_mod.render(
                view, recent=args.recent, suspects_only=args.suspects, since_s=since_s
            )
        )
    if args.all and not args.suspects:
        _print_unhosted(settings, seen, args.host)
    return EXIT_OK


def _status_json(
    args: argparse.Namespace,
    settings: Settings,
    read: RegistryRead,
    entries: list[HostEntry],
    since_s: float | None,
) -> int:
    """One JSON document on stdout, whatever happened. Notes stay on stderr."""
    provider = _provider_for_status(entries, settings) if entries else None
    views = [status_mod.gather(entry, settings, provider=provider) for entry in entries]
    errors = list(read.errors)
    if read.unreadable:
        errors.append(
            f"{hosts_file()} could not be read, so `hosts` is empty because nothing is known"
        )
    print(
        json.dumps(
            status_mod.document(views, errors=errors, recent=args.recent, since_s=since_s),
            indent=2,
        )
    )
    return EXIT_LOCAL_STATE if read.unreadable else EXIT_OK


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
    lost = _outputs_lost_ids(s3, elsewhere[:MIRROR_STATE_LOOKUPS])
    for entry in elsewhere:
        note = (
            " OUTPUTS LOST (the host went away before they uploaded)"
            if entry.job_id in lost
            else ""
        )
        print(
            f"  {entry.job_id} {entry.name or '-'} host={entry.host} attempt={entry.attempt} "
            f"submitted {status_mod.format_age(entry.submitted_at)}{note}"
        )
    print(f"  bring one back with: gpuc requeue {elsewhere[0].job_id} --host {elsewhere[0].host}")


MIRROR_STATE_LOOKUPS = 25
"""How many index-only jobs `--all` reads `state.json` for. One GET each, and
the answer (did this job's outputs make it off the host?) matters most for the
handful at the top of a recovery list."""


def _outputs_lost_ids(s3: S3Index | None, entries: Sequence[IndexEntry]) -> set[str]:
    """Which of these jobs the mirror records as having lost their outputs.

    Best effort: a job whose state.json is missing or unreadable simply does not
    get the flag, because this is a note on a listing, not a decision.
    """
    if s3 is None:
        return set()
    lost: set[str] = set()
    for entry in entries:
        if not entry.s3_prefix:
            continue
        uri = f"{entry.s3_prefix.rstrip('/')}/jobs/{entry.job_id}/state.json"
        try:
            document = json.loads(s3.get_uri(uri))
        except (S3IndexError, json.JSONDecodeError):
            continue
        if isinstance(document, dict) and document.get("outputs_lost"):
            lost.add(entry.job_id)
    return lost


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
    raise NotFound(
        f"no registered host knows job {job_id}.\n"
        f"Pass --host <name>, or check `gpuc host list` and `gpuc status --all`."
    )


def ssh_target(args: argparse.Namespace) -> tuple[HostEntry, str, str | None]:
    """`(host, directory, fallback)` for a host name or a job id.

    A registered host name wins over a job id: host names are ours and job ids
    are timestamps, so they cannot collide, and looking the name up locally
    keeps `gpuc ssh <host>` from asking every host whether it knows a job.
    """
    registry = named_registry()
    entry = registry.hosts.get(args.target)
    if entry is not None:
        return entry, entry.remote_home, None
    entry, _ = find_job_host(args.target, registry, args.host)
    job_dir = f"{entry.remote_home}/jobs/{args.target}"
    return entry, f"{job_dir}/workdir", job_dir


def cmd_ssh(args: argparse.Namespace) -> int:
    """A shell on a host (or in a job's workdir), with gpuc's own ssh options."""
    command = list(args.command or [])
    # argparse.REMAINDER swallows flags that follow the target, and typing
    # `gpuc ssh myhost --print` is the obvious thing to do.
    if command and command[0] == "--print":
        args.print_only, command = True, command[1:]
    if command and command[0] == "--":
        command = command[1:]
    entry, directory, fallback = ssh_target(args)
    transport = transport_for(entry, load_settings())
    local = entry.kind == "local"
    if command:
        # Joined with spaces and handed to a shell, which is what `ssh host CMD`
        # has always done and what anyone typing `-- 'ls | wc -l'` expects.
        # shlex.join would quote the pipe back into a filename.
        joined = " ".join(command)
        if args.print_only:
            print(ssh_mod.print_line(ssh_mod.command_argv(transport, directory, joined, fallback)))
            return EXIT_OK
        result = ssh_mod.run_command(transport, directory, joined, fallback)
        sys.stdout.write(result.stdout)
        sys.stderr.write(result.stderr)
        return result.returncode
    if args.print_only:
        print(ssh_mod.print_line(ssh_mod.interactive_argv(transport, directory, fallback)))
        return EXIT_OK
    print(f"# {entry.name}:{directory}", file=sys.stderr)
    if local:
        # No ssh to this machine: chdir and hand over the terminal directly.
        os.chdir(ssh_mod.local_directory(directory, fallback))
        shell = os.environ.get("SHELL", ssh_mod.DEFAULT_SHELL)
        os.execvp(shell, [shell, "-l"])
    argv = ssh_mod.interactive_argv(transport, directory, fallback)
    os.execvp(argv[0], argv)


def cmd_cancel(args: argparse.Namespace) -> int:
    entry, _ = find_job_host(args.job_id, named_registry(), args.host)
    payload = open_session(entry, load_settings()).host_json(f"cancel {shlex.quote(args.job_id)}")
    # The host's own word for what it did: `cancelled` for a queued job it
    # dequeued, `cancelling` for a running one whose runner has been marked.
    status = payload.get("status")
    if args.json:
        jsonout.emit({"job_id": args.job_id, "host": entry.name, "status": status})
    else:
        print(f"job {args.job_id} on host {entry.name}: {status}")
    return EXIT_OK


def cmd_reorder(args: argparse.Namespace) -> int:
    entry, _ = find_job_host(args.job_id, named_registry(), args.host)
    session = open_session(entry, load_settings())
    result = session.host_cli(f"reorder {shlex.quote(args.job_id)} {args.priority}", check=False)
    if result.returncode != 0:
        raise CliError(
            f"job {args.job_id} is not in host {entry.name}'s queue, so its priority cannot "
            f"change (a running or finished job cannot be reordered)."
        )
    if args.json:
        jsonout.emit({"job_id": args.job_id, "host": entry.name, "priority": args.priority})
    else:
        print(f"job {args.job_id} on host {entry.name} moved to priority {args.priority}")
    return EXIT_OK


def mirror_estimate(job_id: str, minutes: float | None, settings: Settings) -> str | None:
    """Put the new estimate in the job's mirrored spec too, or say why not.

    `requeue` submits what the *mirror* holds, so leaving it behind would hand
    a re-run of an estimated job back with no estimate, silently. A mirror that
    cannot be updated is a note and never a failure: the estimate is already
    recorded where `status` reads it, which is what was asked for.
    """
    s3 = S3Index.from_settings(settings)
    if s3 is None:
        return None
    try:
        document = s3.get_spec(job_id)
        document["estimated_runtime_min"] = minutes
        s3.put_spec_document(job_id, document)
    except (S3IndexError, S3ObjectMissing, ValueError) as exc:
        return (
            f"the host has the new estimate, but its mirrored spec still has the old one, "
            f"so `gpuc requeue {job_id}` would not carry it: {str(exc).splitlines()[0]}"
        )
    return None


def cmd_estimate(args: argparse.Namespace) -> int:
    """Add, change or clear a job's `estimated_runtime_min` after submitting it.

    It is the one spec field somebody else needs and only the submitter knows,
    and the job that most needs one is the long job already running when the
    next person arrives -- which is too late to edit a file before `submit`.
    """
    if args.clear is (args.minutes is not None):
        raise UsageError("give --minutes N or --clear, not both")
    wanted: float | None = None if args.clear else args.minutes
    if wanted is not None and not wanted > 0.0:
        raise UsageError(f"--minutes must be a positive number of minutes, got {wanted:g}")
    if wanted is not None and jobs.utc_in(wanted * 60.0) is None:
        # An `inf`, or the `1e10` units typo: no date can hold it, so the host
        # would record it and then publish no eta at all.
        raise UsageError(f"--minutes {wanted:g} is too far away to be an end time")
    settings = load_settings()
    entry, _ = find_job_host(args.job_id, named_registry(), args.host)
    session = open_session(entry, settings)
    request = "--clear" if wanted is None else repr(wanted)
    # `check=False`: a refusal (a finished job, an id this host does not know)
    # *is* the host's document, and raising on the exit code would throw away
    # the reason it gave for one that only says it exited 1.
    payload = session.host_json(f"estimate {shlex.quote(args.job_id)} {request}", check=False)
    document = payload if isinstance(payload, dict) else {}
    error = document.get("error")
    if error:
        raise CliError(f"host {entry.name} did not set the estimate: {error}")
    recorded = document.get("estimated_runtime_min")
    if wanted is not None and not isinstance(recorded, (int, float)):
        # Otherwise a host that answered with something else -- a build that
        # does not know this command, a document with the key missing --
        # reports a successful *clear* of a job it never touched.
        raise CliError(
            f"host {entry.name} did not say what estimate it recorded for {args.job_id}: "
            f"{json.dumps(payload)[:200]}"
        )
    warnings = [str(document["warning"])] if document.get("warning") else []
    note = mirror_estimate(args.job_id, wanted, settings)
    if note:
        warnings.append(note)
    for text in warnings:
        print(f"WARNING: {text}", file=sys.stderr)
    if args.json:
        jsonout.emit(
            {
                "job_id": args.job_id,
                "host": entry.name,
                "estimated_runtime_min": recorded,
                "status": document.get("status"),
                "warnings": warnings,
            }
        )
    elif recorded is None:
        print(f"job {args.job_id} on host {entry.name} no longer estimates a runtime")
    else:
        shown = f"{recorded:g}" if isinstance(recorded, (int, float)) else recorded
        print(f"job {args.job_id} on host {entry.name} now estimates {shown} min")
    return EXIT_OK


def _follow_argv(transport: Transport, remote_path: str, lines: int) -> list[str]:
    command = f"tail -n {lines} -f {shlex.quote(remote_path)}"
    if isinstance(transport, SshTransport):
        return transport.ssh_argv(command)
    return ["bash", "-lc", command]


@dataclass
class LogText:
    """A job's log and where it was read from, for both output forms."""

    source: str
    """`host` or `s3`."""
    location: str | None
    text: str = ""
    notes: list[str] = field(default_factory=list)


def cmd_logs(args: argparse.Namespace) -> int:
    if args.json and args.follow:
        raise UsageError(
            "logs --json cannot follow: -f streams a log that has no end, and a JSON "
            "document has to be complete. Drop one of them."
        )
    settings = load_settings()
    entry, index = find_job_host(args.job_id, named_registry(), args.host)
    remote = None
    purged = False
    try:
        session = open_session(entry, settings)
        remote = f"{session.job_dir(args.job_id)}/log.txt"
        if args.follow:
            return _follow(session.transport, remote, args.lines)
        result = session.transport.tail(remote, lines=args.lines)
        if result.returncode == 0:
            return _write_log(args, entry, LogText("host", remote, result.stdout))
        purged = _job_dir_gone(session, args.job_id)
        why = (result.output.strip().splitlines() or ["no log file on the host"])[-1]
    except (RemoteError, TransportError) as exc:
        why = str(exc).splitlines()[0]
    # A job dir that is gone entirely is what `gpuc clean --purge` does on
    # purpose. Saying "purged" beats printing a `tail: No such file`.
    note = (
        f"job {args.job_id} was purged from host {entry.name} "
        f"(gpuc clean --purge removes the whole job dir once it is mirrored)"
        if purged
        else f"could not read {remote or 'the host log'}: {why}"
    )
    print(f"note: {note}", file=sys.stderr)
    log = _logs_from_s3(args.job_id, entry, index, settings, purged=purged)
    log.notes.insert(0, note)
    return _write_log(args, entry, log)


def _write_log(args: argparse.Namespace, entry: HostEntry, log: LogText) -> int:
    """Bytes for a human; lines plus where they came from for a script."""
    if args.json:
        jsonout.emit(
            {
                "job_id": args.job_id,
                "host": entry.name,
                "source": log.source,
                "location": log.location,
                "lines": log.text.splitlines(),
                "notes": log.notes,
            }
        )
    else:
        sys.stdout.write(log.text)
    return EXIT_OK


def _job_dir_gone(session: HostSession, job_id: str) -> bool:
    """Is the job dir itself missing, rather than just its log?"""
    try:
        result = session.run(f"test -d {shlex.quote(session.job_dir(job_id))}", timeout=30.0)
    except TransportError:
        return False
    return result.returncode != 0


def _follow(transport: Transport, remote: str, lines: int) -> int:
    argv = _follow_argv(transport, remote, lines)
    try:
        return subprocess.call(argv)
    except KeyboardInterrupt:
        return 0


def _logs_from_s3(
    job_id: str,
    entry: HostEntry,
    index: IndexEntry | None,
    settings: Settings,
    *,
    purged: bool = False,
) -> LogText:
    s3 = S3Index.from_settings(settings)
    prefix = (index.s3_prefix if index else None) or entry.s3_prefix
    if s3 is None or not prefix:
        gone = (
            f"Its job dir was purged from host {entry.name}, so this log no longer exists "
            f"anywhere.\n"
            if purged
            else ""
        )
        raise CliError(
            f"no S3 mirror to fall back on for {job_id}.\n"
            f"{gone}"
            f"Set s3_bucket in ~/.config/gpu-coordinator/config.toml to keep logs after a "
            f"host goes away."
        )
    uri = job_log_uri(prefix, job_id)
    fallback = f"falling back to the S3 mirror at {uri}"
    print(f"note: {fallback}", file=sys.stderr)
    return LogText("s3", uri, s3.get_uri(uri), [fallback])


def cmd_requeue(args: argparse.Namespace) -> int:
    settings = load_settings()
    check_runpod_args(args)
    registry = named_registry()
    index = LocalIndex().get(args.job_id)
    target = args.host or (None if args.runpod else (index.host if index else None))
    if not target and not args.runpod:
        raise UsageError(
            f"requeue needs --host <name>: nothing local knows where {args.job_id} ran"
        )
    s3 = S3Index.from_settings(settings)
    if s3 is None:
        raise CliError(
            "requeue reads the spec from S3, but s3_bucket is unset in "
            "~/.config/gpu-coordinator/config.toml. Re-submit the job file instead."
        )
    try:
        document = s3.get_spec(args.job_id)
    except S3ObjectMissing as exc:
        # A job id nobody ever mirrored a spec for does not exist as far as
        # requeue is concerned: exit 4, like every other unknown name.
        raise NotFound(
            f"no mirrored spec for job {args.job_id}.\n"
            f"Check the id with `gpuc status --all`; only jobs submitted with s3_bucket "
            f"set can be requeued."
        ) from exc
    for key in ("job_id", "attempt"):
        document.pop(key, None)
    attempt = (index.attempt if index else 1) + 1
    model = validate(document, f"spec for {args.job_id}")
    use_git = not args.no_git
    report = reporter(args)
    if target is None:
        precheck_local(
            model,
            Path.cwd(),
            gpu_count=args.gpu_count,
            use_git=use_git,
            ttl_hours=args.ttl_hours,
            report=report,
        )
    entry = runpod_target(args, settings) if target is None else registry.require(target)
    entry = ensure_package_current(entry, settings, bootstrap=not args.no_bootstrap, report=report)
    result = submit_spec(
        entry,
        model,
        settings,
        workdir=Path.cwd(),
        attempt=attempt,
        use_git=use_git,
        report=report,
    )
    return _queued(result, args, requeued_from=args.job_id)


def cmd_reconcile(args: argparse.Namespace) -> int:
    if args.json and (args.install or not args.once):
        raise UsageError(
            "reconcile --json needs --once and nothing else: the loop and --install have "
            "no document to print, only a running commentary."
        )
    if args.install:
        reconcile_mod.install(args.interval)
        return EXIT_OK
    settings = load_settings()
    provider = make_provider(settings)
    if args.once:
        # Every pod it judges is a line of commentary, and under --json stdout
        # belongs to the document.
        result = reconcile_mod.reconcile_once(settings, provider, report=reporter(args))
        if args.json:
            jsonout.emit(result.document())
        else:
            print(result.render())
        return EXIT_ERROR if result.errors else EXIT_OK
    print(f"reconciling every {args.interval:.0f}s; Ctrl-C to stop")
    try:
        reconcile_mod.run_loop(settings, provider, interval_s=args.interval)
    except KeyboardInterrupt:
        print("stopped")
    return EXIT_OK


def cmd_pods(args: argparse.Namespace) -> int:
    settings = load_settings()
    view = pods_mod.gather(settings, make_provider(settings), heartbeats=not args.no_heartbeat)
    if args.json:
        jsonout.emit(view.document())
    else:
        print(pods_mod.render(view))
    return EXIT_OK


def cmd_version(args: argparse.Namespace) -> int:
    """What is installed here, and what each host was last given.

    The host commits are read from the registry, which bootstrap wrote -- no
    ssh, so this stays a command you can run before anything else.
    """
    commit = version_mod.local_commit()
    source = "installed" if version_mod.installed_commit() else "source checkout"
    dirty = " (+uncommitted changes)" if version_mod.dirty() else ""
    if args.json:
        return _version_json(commit, source=source, dirty=bool(dirty))
    print(f"gpuc {version_mod.__version__}")
    print(f"commit {version_mod.short(commit)} [{source}]{dirty}")
    print(f"python {sys.version.split()[0]} at {sys.executable}")
    read = read_registry()
    for error in read.errors:
        print(f"warning: {error}", file=sys.stderr)
    hosts = [e for e in read.registry.hosts.values() if e.bootstrapped_at]
    if not hosts:
        print("hosts: none bootstrapped")
        return EXIT_LOCAL_STATE if read.unreadable else EXIT_OK
    print("hosts:")
    for entry in hosts:
        note = "" if version_mod.same_commit(commit, entry.pkg_commit) else "  OLDER: re-bootstrap"
        print(f"  {entry.name:<16} pkg {version_mod.short(entry.pkg_commit)}{note}")
    if any(not version_mod.same_commit(commit, e.pkg_commit) for e in hosts):
        print(
            "upgrade a host with: gpuc host bootstrap <host> (or --all for every host; "
            "running jobs are not disturbed)"
        )
    return EXIT_LOCAL_STATE if read.unreadable else EXIT_OK


def _version_json(commit: str | None, *, source: str, dirty: bool) -> int:
    """`gpuc version --json`: this build, and each bootstrapped host's package.

    `hosts[].current` is the same judgement the text output prints as `OLDER:
    re-bootstrap`: a commit that does not match this build's. Nothing recorded
    on either side is not evidence of a mismatch, so it reads as current --
    `submit` re-ships the package to such a host anyway.
    """
    read = read_registry()
    for error in read.errors:
        print(f"warning: {error}", file=sys.stderr)
    hosts = [entry for entry in read.registry.hosts.values() if entry.bootstrapped_at]
    jsonout.emit(
        {
            "version": version_mod.__version__,
            "commit": commit,
            "source": source,
            "dirty": dirty,
            "python": sys.version.split()[0],
            "executable": sys.executable,
            "hosts": [
                {
                    "name": entry.name,
                    "pkg_commit": entry.pkg_commit,
                    "current": version_mod.same_commit(commit, entry.pkg_commit),
                }
                for entry in hosts
            ],
            "errors": list(read.errors),
        }
    )
    return EXIT_LOCAL_STATE if read.unreadable else EXIT_OK


def cmd_skill(args: argparse.Namespace) -> int:
    """Print the agent guide, or drop a copy into a project.

    Printing is the point: an agent can pipe `gpuc skill` into its own context
    without being told where the file lives, or which checkout it is in.
    """
    if args.install is None:
        sys.stdout.write(read_skill())
        return EXIT_OK
    print(f"wrote {install_skill(Path(args.install), force=args.force)}")
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gpuc", description="GPU job coordinator")
    sub = parser.add_subparsers(dest="command", required=True)

    host = sub.add_parser("host", help="manage hosts").add_subparsers(
        dest="host_command", required=True
    )
    add = host.add_parser("add", help="register a host")
    add.add_argument("name")
    add.add_argument("--ssh", help="user@host; omit for this machine")
    add.add_argument("--port", type=int, default=22, help="ssh port (default 22)")
    add.add_argument("--gpus", help=GPUS_HELP)
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
    add.add_argument(
        "--retention-days",
        help="auto-purge job dirs older than this, once the host has confirmed their log "
        "and state are mirrored; omit to keep everything forever",
    )
    add.add_argument(
        "--idle-min",
        type=float,
        default=15.0,
        metavar="MINUTES",
        help="how long an ephemeral host may sit with an empty queue before it terminates "
        "itself (default 15); ignored for hosts that are not ephemeral",
    )
    add.add_argument(
        "--ttl-hours",
        type=float,
        default=None,
        help="hard cap on the host's life; omit or pass -1 for none (the default). When "
        "set, the dispatcher kills the running job with reason ttl, syncs, and terminates",
    )
    add.set_defaults(func=cmd_host_add)

    edit = host.add_parser("set", help="change a registered host without remove/add")
    edit.add_argument("name")
    edit.add_argument("--gpus", help=f"replace what this host owns; pass '' for none. {GPUS_HELP}")
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
    edit.add_argument(
        "--retention-days", help="auto-purge horizon in days; pass '' to keep everything"
    )
    edit.add_argument(
        "--idle-min",
        type=float,
        metavar="MINUTES",
        help="idle minutes before an ephemeral host terminates itself",
    )
    edit.add_argument(
        "--ttl-hours", type=float, help="hard cap in hours; -1 clears it (no TTL, the default)"
    )
    edit.set_defaults(func=cmd_host_set)

    bootstrap = host.add_parser("bootstrap", help="install uv, the package and the dispatcher")
    bootstrap.add_argument("name", nargs="?", help="the host to bootstrap; omit it with --all")
    bootstrap.add_argument(
        "--all",
        action="store_true",
        help="bootstrap every registered host instead of one, in the order `host list` shows "
        "them; a host that fails does not stop the rest, and the command exits 1 if any did",
    )
    bootstrap.add_argument(
        "--health-args", default="", help="extra flags for `gpuc.host health`, e.g. --min-mbps 0.1"
    )
    bootstrap.set_defaults(func=cmd_host_bootstrap)

    probe = host.add_parser("probe", help="report what a host has, before bootstrap")
    probe.add_argument("name")
    probe.add_argument(
        "--all-gpus",
        action="store_true",
        help="list every GPU in the box, not just the ones assigned to this host "
        "(text output; --json always carries them all)",
    )
    add_json_flag(probe)
    probe.set_defaults(func=cmd_host_probe)

    host_clean = host.add_parser("clean", help="prune the host's uv cache")
    host_clean.add_argument("name")
    host_clean.add_argument(
        "--uv-cache", action="store_true", help="run `uv cache prune` on the host"
    )
    host_clean.set_defaults(func=cmd_host_clean)

    host_list = host.add_parser("list", help="list registered hosts")
    add_json_flag(host_list)
    host_list.set_defaults(func=cmd_host_list)
    resume = host.add_parser(
        "resume", help="clear a low-util pause on a host and restart its dispatcher"
    )
    resume.add_argument("name")
    resume.set_defaults(func=cmd_host_resume)
    remove = host.add_parser("remove", help="forget a host")
    remove.add_argument("name")
    remove.set_defaults(func=cmd_host_remove)

    submit = sub.add_parser("submit", help="submit a job file to a host")
    submit.add_argument("job_file")
    submit.add_argument(
        "--host", metavar="NAME", help="a registered host to submit to (see `gpuc host list`)"
    )
    add_no_git_flag(submit)
    add_bootstrap_flag(submit)
    add_runpod_flags(submit)
    add_json_flag(submit)
    submit.set_defaults(func=cmd_submit)

    status = sub.add_parser("status", help="per-host queue, running and recent jobs")
    status.add_argument(
        "--host", metavar="NAME", help="only this host; omit for every registered host"
    )
    status.add_argument("--all", action="store_true", help="also list jobs only the index knows")
    status.add_argument("--suspects", action="store_true", help="billing but idle; never kills")
    status.add_argument(
        "--recent",
        type=int,
        default=status_mod.RECENT_FINISHED,
        metavar="N",
        help=f"how many finished jobs to show per host (default {status_mod.RECENT_FINISHED})",
    )
    status.add_argument(
        "--since",
        metavar="DURATION",
        help="only finished jobs that ended within this long ago, e.g. 24h, 7d, 90m",
    )
    status.add_argument(
        "--json",
        action="store_true",
        help="one JSON document on stdout: key on hosts[].running, and treat exit 3 "
        "(local state unreadable) as unknown, never as nothing running",
    )
    status.set_defaults(func=cmd_status)

    skill = sub.add_parser("skill", help="print the agent guide, or install it into a project")
    skill.add_argument(
        "--install",
        nargs="?",
        const=".",
        metavar="DIR",
        help="write it to DIR/.claude/skills/gpuc/SKILL.md instead of printing (DIR defaults to .)",
    )
    skill.add_argument("--force", action="store_true", help="overwrite an existing installed copy")
    skill.set_defaults(func=cmd_skill)

    version = sub.add_parser(
        "version", help="version, installed commit, and each host's package commit"
    )
    add_json_flag(version)
    version.set_defaults(func=cmd_version)

    clean = sub.add_parser("clean", help="remove finished jobs' workdirs on a host")
    clean.add_argument("--host", required=True, metavar="NAME", help="the host to reclaim disk on")
    selection = clean.add_mutually_exclusive_group()
    selection.add_argument(
        "--all-finished",
        action="store_true",
        help="every succeeded, failed or cancelled job, however recently it ended; with "
        "--purge this is an age horizon of 0 and needs --yes (or --dry-run)",
    )
    selection.add_argument(
        "--older-than", type=float, metavar="DAYS", help="only jobs that ended over DAYS ago"
    )
    selection.add_argument(
        "--only",
        metavar="ID[,ID...]",
        help="exactly these job ids, however recently they ended; with --purge the "
        "implied workdir sweep is scoped to them too, and no --yes is needed",
    )
    clean.add_argument(
        "--purge",
        action="store_true",
        help=f"remove the whole jobs/<id>/ of finished jobs whose log and state are "
        f"confirmed mirrored and whose outputs are confirmed uploaded "
        f"(default --older-than {DEFAULT_RETENTION_DAYS:g}); implies the workdir clean",
    )
    clean.add_argument(
        "--force",
        action="store_true",
        help="with --purge: delete job dirs that have no confirmed backup anyway",
    )
    clean.add_argument(
        "--verify",
        action="store_true",
        help="with --purge: HEAD each job's mirrored log.txt in S3 before deleting it",
    )
    clean.add_argument(
        "--yes",
        action="store_true",
        help="confirm `--purge --all-finished`, which deletes the whole job dir of every "
        "finished job with no age horizon at all",
    )
    clean.add_argument("--dry-run", action="store_true", help="list what would go, delete nothing")
    add_json_flag(clean)
    clean.set_defaults(func=cmd_clean)

    logs = sub.add_parser("logs", help="tail a job log from its host")
    logs.add_argument("job_id")
    logs.add_argument("-f", "--follow", action="store_true", help="stream the log as it is written")
    logs.add_argument(
        "-n", "--lines", type=int, default=200, metavar="N", help="lines of history (default 200)"
    )
    logs.add_argument(
        "--host", metavar="NAME", help="which host the job is on, if it cannot be found"
    )
    add_json_flag(
        logs,
        "the log as a list of lines, with the host or S3 uri it came from; "
        "not with -f, which has no end",
    )
    logs.set_defaults(func=cmd_logs)

    ssh = sub.add_parser(
        "ssh",
        help="a shell on a host, or in a job's workdir; or one command there",
        description="A host name lands in that host's gpuc home. A job id lands in that "
        "job's workdir/, falling back to the job dir itself when the workdir has been "
        "cleaned away (the log and state are still there).",
    )
    ssh.add_argument("target", help="a registered host name, or a job id")
    ssh.add_argument("--host", help="which host a job id is on, if it is ambiguous")
    ssh.add_argument(
        "--print",
        action="store_true",
        dest="print_only",
        help="print the equivalent command line instead of running it",
    )
    ssh.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="a command line to run there (after `--`), joined with spaces and interpreted "
        "by a login bash on the host, so pipes and redirections work; gpuc exits with the "
        "remote command's own exit code. Omit for an interactive shell",
    )
    ssh.set_defaults(func=cmd_ssh)

    cancel = sub.add_parser("cancel", help="cancel a queued or running job")
    cancel.add_argument("job_id")
    cancel.add_argument(
        "--host", metavar="NAME", help="which host the job is on, if it cannot be found"
    )
    add_json_flag(cancel)
    cancel.set_defaults(func=cmd_cancel)

    reorder = sub.add_parser("reorder", help="change a queued job's priority")
    reorder.add_argument("job_id")
    reorder.add_argument(
        "--priority",
        type=int,
        required=True,
        help="0-99; lower dispatches first (submit defaults to 50)",
    )
    reorder.add_argument(
        "--host", metavar="NAME", help="which host the job is on, if it cannot be found"
    )
    add_json_flag(reorder)
    reorder.set_defaults(func=cmd_reorder)

    estimate = sub.add_parser(
        "estimate",
        help="set (or clear) a queued or running job's estimated_runtime_min",
        description="Records how long the job expects to take, from the runner's start. "
        "Nothing kills a job for running past it: it is what `gpuc status` shows the next "
        "person deciding whether to queue behind this job. A running job's runner picks the "
        "new estimate up within a minute; a finished job is refused.",
    )
    estimate.add_argument("job_id")
    estimate.add_argument("--minutes", type=float, metavar="N", help="how long the job will take")
    estimate.add_argument("--clear", action="store_true", help="remove the estimate instead")
    estimate.add_argument(
        "--host", metavar="NAME", help="which host the job is on, if it cannot be found"
    )
    add_json_flag(estimate)
    estimate.set_defaults(func=cmd_estimate)

    requeue = sub.add_parser("requeue", help="resubmit a job from its S3 spec")
    requeue.add_argument("job_id")
    requeue.add_argument(
        "--host",
        metavar="NAME",
        help="submit to this host instead of the one the job ran on; not with --runpod",
    )
    add_no_git_flag(requeue)
    add_bootstrap_flag(requeue)
    add_runpod_flags(requeue)
    add_json_flag(requeue)
    requeue.set_defaults(func=cmd_requeue)

    reconcile = sub.add_parser(
        "reconcile", help="terminate leaked or expired pods; --install for a systemd timer"
    )
    reconcile.add_argument("--once", action="store_true", help="one pass, then exit")
    reconcile.add_argument(
        "--interval",
        type=float,
        default=reconcile_mod.DEFAULT_INTERVAL_S,
        metavar="SECONDS",
        help=f"seconds between passes of the loop, and of the installed timer "
        f"(default {reconcile_mod.DEFAULT_INTERVAL_S:.0f})",
    )
    reconcile.add_argument(
        "--install", action="store_true", help="write (but do not enable) systemd --user units"
    )
    add_json_flag(reconcile, f"{JSON_HELP}; needs --once")
    reconcile.set_defaults(func=cmd_reconcile)

    pods = sub.add_parser("pods", help="every pod with our prefix, cost, util, age, desired?")
    pods.add_argument(
        "--no-heartbeat", action="store_true", help="skip the per-pod dispatcher ssh check"
    )
    add_json_flag(pods)
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


JSON_HELP = (
    "one JSON document on stdout and nothing else; progress and warnings go to "
    "stderr. `error` is set (and nothing else but schema_version) when the command "
    "failed, and the exit code is the same as without --json"
)


def add_json_flag(parser: argparse.ArgumentParser, help_text: str = JSON_HELP) -> None:
    parser.add_argument("--json", action="store_true", help=help_text)


def add_bootstrap_flag(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--no-bootstrap",
        action="store_true",
        help="enqueue even if the host's package is older than this machine's; without it, "
        "a host on another commit (or none) gets the package re-synced and its dispatcher "
        "restarted first",
    )


def add_no_git_flag(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--no-git",
        action="store_true",
        help="the workdir is not a git repository: rsync all of it except "
        f"{', '.join(NO_GIT_EXCLUDES)}",
    )


def add_runpod_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--runpod", action="store_true", help="reuse or provision a RunPod pod")
    parser.add_argument("--gpu", help="comma-separated GPU names, cheapest match wins")
    parser.add_argument(
        "--gpu-count",
        type=int,
        default=1,
        metavar="N",
        help="GPUs on the pod (default 1); the spec's `gpus:` must fit in it",
    )
    parser.add_argument(
        "--min-vram", type=int, metavar="GB", help="skip offers with less VRAM per GPU"
    )
    parser.add_argument("--max-price", type=float, help="USD per hour, for the whole pod")
    parser.add_argument(
        "--cloud",
        choices=sorted(CLOUDS),
        default="secure",
        help="which RunPod tier to buy from; community is cheaper and less reliable "
        "(default secure)",
    )
    parser.add_argument("--cuda-min", default=None, help="host CUDA floor, default 12.8")
    parser.add_argument(
        "--idle-min",
        type=float,
        default=15.0,
        metavar="MINUTES",
        help="terminate the pod once its queue has been empty this long (default 15)",
    )
    parser.add_argument(
        "--ttl-hours",
        type=float,
        default=None,
        help="hard cap on the pod's life; omit or pass -1 for none (the default), leaving "
        "--idle-min and `gpuc reconcile` to stop it",
    )
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


def failed(args: argparse.Namespace, message: str, exit_code: int) -> int:
    """One exit for every failure: the message on stderr, and under `--json` a
    document on stdout saying the same thing, so a caller parsing stdout is
    never handed half an answer or nothing at all."""
    print(f"error: {message}", file=sys.stderr)
    if getattr(args, "json", False):
        jsonout.emit_error(message, exit_code)
    return exit_code


def main(argv: Sequence[str] | None = None) -> int:
    raw = list(argv) if argv is not None else sys.argv[1:]
    try:
        args = build_parser().parse_args(raw)
    except SystemExit as exc:
        # argparse writes its own message and exits before `failed()` can be
        # reached. The reason stays on stderr, where every other note goes, but
        # stdout still gets a document: "exit 2 and nothing at all" is the one
        # answer --json promises never to give. `--help` exits 0 and is not one.
        code = exc.code if isinstance(exc.code, int) else EXIT_USAGE
        if code and "--json" in raw:
            jsonout.emit_error("the command line was rejected; the reason is on stderr", code)
        raise
    if wants_runpod(args) and not os.environ.get("RUNPOD_API_KEY"):
        # Before anything else: provisioning spends money, and finding out after
        # the spec has been mirrored and a host picked helps nobody.
        return failed(
            args,
            "RUNPOD_API_KEY is not set; export it before using --runpod, "
            "`gpuc pods` or `gpuc reconcile`",
            EXIT_ERROR,
        )
    if args.command not in ("config", "skill"):
        first_run_note()
    try:
        return int(args.func(args))
    except LocalStateUnreadable as exc:
        return failed(args, str(exc), EXIT_LOCAL_STATE)
    except HostNotFound as exc:
        return failed(args, str(exc), EXIT_NOT_FOUND)
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
        SkillError,
        TransportError,
    ) as exc:
        return failed(args, str(exc), getattr(exc, "exit_code", EXIT_ERROR))
    except json.JSONDecodeError as exc:
        return failed(args, f"a host returned malformed JSON: {exc}", EXIT_ERROR)


if __name__ == "__main__":
    raise SystemExit(main())
