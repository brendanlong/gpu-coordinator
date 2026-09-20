"""The `gpuc` command line. Thin: parse, call a module, print, map errors to 1."""

from __future__ import annotations

import argparse
import getpass
import json
import math
import os
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from gpuc.control import jsonout, rented
from gpuc.control import pods as pods_mod
from gpuc.control import ssh as ssh_mod
from gpuc.control import status as status_mod
from gpuc.control import version as version_mod
from gpuc.control import wait as wait_mod
from gpuc.control import web as web_mod
from gpuc.control.actions import (
    EXIT_ERROR,
    EXIT_INTERRUPTED,
    EXIT_LOCAL_STATE,
    EXIT_NOT_FOUND,
    EXIT_OK,
    EXIT_USAGE,
    CliError,
    NotFound,
    UsageError,
    cancel_job,
    check_estimate,
    config_document,
    connection_document,
    estimate_job,
    exit_code_for,
    failure_message,
    find_job_host,
    gather_all,
    hosts_document,
    hosts_for,
    init_config,
    job_log_path,
    make_provider,
    named_registry,
    note,
    placement_after,
    preempt_job,
    provider_for_status,
    read_log,
    read_registry_warned,
    remove_host,
    reorder_job,
    shipped_note,
    status_document,
    version_document,
    warn,
)
from gpuc.control.bootstrap import (
    BootstrapError,
    BootstrapResult,
    bootstrap_host,
    resync_package,
)
from gpuc.control.clean import check_flags as check_clean_flags
from gpuc.control.clean import clean_host, parse_only, prune_uv_cache
from gpuc.control.config import (
    ConfigError,
    HostEntry,
    LocalStateUnreadable,
    Reporter,
    Settings,
    config_file,
    hosts_file,
    load_registry,
    load_settings,
    registry_transaction,
    state_dir,
    transport_for,
)
from gpuc.control.connect import Connection, connect_host, push_config
from gpuc.control.gpuinfo import rows as gpu_rows
from gpuc.control.gpuinfo import summarize
from gpuc.control.probe import ProbeReport, probe_host
from gpuc.control.providers.base import Cloud, Constraints
from gpuc.control.provision import runpod_host
from gpuc.control.remote import (
    HostSession,
    RemoteError,
    open_session,
    read_remote_config,
    resolve_home,
)
from gpuc.control.s3index import (
    IndexEntry,
    LocalIndex,
    S3Index,
    S3IndexError,
    S3ObjectMissing,
    job_uri,
)
from gpuc.control.skill import install_skill, read_skill
from gpuc.control.submit import (
    JobSpecModel,
    SubmitResult,
    from_mirror,
    load_document,
    precheck_local,
    submit_file,
    submit_spec,
    validate,
    with_overrides,
)
from gpuc.control.transport import (
    NO_GIT_EXCLUDES,
    SshTransport,
    Transport,
    TransportError,
    tail_command,
)
from gpuc.host import jobs
from gpuc.host.cleanup import DEFAULT_RETENTION_DAYS

__all__ = [
    "EXIT_ERROR",
    "EXIT_INTERRUPTED",
    "EXIT_LOCAL_STATE",
    "EXIT_NOT_FOUND",
    "EXIT_OK",
    "EXIT_USAGE",
    "CliError",
    "NotFound",
    "UsageError",
    "build_parser",
    "main",
]

NO_HOSTS = "no hosts registered. Add one: gpuc host add local"

GPUS_HELP = (
    "GPU UUIDs or nvidia-smi indices this host may use, comma-separated "
    "(`--gpus 2,3` or `--gpus GPU-8064...,3`); indices are resolved to UUIDs on "
    "the host at every dispatch pass, so jobs are always pinned by UUID"
)

SHARED_GPUS_HELP = (
    "cards on this box that gpuc may borrow but does not own, spelled like "
    "--gpus and never overlapping it. Only a job with `use_shared: true` is "
    "dispatched to one, only once the owned cards are full, and only while "
    "nvidia-smi says the card holds no memory and is doing no work"
)


def _comma_list(raw: str | None) -> list[str]:
    """A comma- or space-separated flag value, as a list."""
    if not raw:
        return []
    return [part.strip() for part in raw.replace(" ", ",").split(",") if part.strip()]


def _gpu_list(raw: str | None, flag: str = "--gpus") -> list[str]:
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
            f"{flag} wants nvidia-smi indices or GPU UUIDs, got {', '.join(repr(b) for b in bad)}."
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


def _config_fields(args: argparse.Namespace) -> dict[str, Any]:
    """The host-config keys these flags name, and only the ones given.

    A flag that was not typed is not an opinion: what is here is exactly what
    this command is about to change about a host's own `config.json`, which is
    what it then reports field by field.
    """
    fields: dict[str, Any] = {}
    if args.gpus is not None:
        fields["gpus"] = _gpu_list(args.gpus)
    if args.shared_gpus is not None:
        fields["shared_gpus"] = _gpu_list(args.shared_gpus, "--shared-gpus")
    if args.env is not None:
        # The whole dict, not a merge: "set it to exactly this" is the only
        # rule that can also express "set it to nothing" (`--env ''`).
        fields["env"] = _env_dict([pair for pair in args.env if pair])
    if args.s3_prefix is not None:
        fields["s3_prefix"] = args.s3_prefix or None
    if args.retention_days is not None:
        fields["retention_days"] = _days(args.retention_days, "--retention-days")
    if args.workdir_days is not None:
        fields["workdir_days"] = _days(args.workdir_days, "--workdir-days")
    if args.idle_min is not None:
        fields["idle_minutes"] = args.idle_min
    return fields


def _env_updates(args: argparse.Namespace) -> dict[str, str | None]:
    """`--cache-dir`: one variable of the host's env, resolved against the host."""
    if args.cache_dir is None:
        return {}
    return {"UV_CACHE_DIR": args.cache_dir or None}


def _address(args: argparse.Namespace) -> HostEntry:
    return HostEntry(
        name=args.name,
        kind="ssh" if args.ssh else "local",
        ssh=args.ssh,
        port=args.port,
        gpuc_home=args.gpuc_home,
        persistent_root=args.persistent_root,
    )


def _pod_address(address: HostEntry, pod_id: str, settings: Settings) -> HostEntry:
    """Where a rented pod is now, asked of the provider that is billing for it.

    Adopting a pod another machine created is the ordinary path, not a special
    one: the pod owns its config and carries its own record of what it was
    bought as (`rented`), so all this has to find is the door.
    """
    pod = make_provider(settings).get(pod_id)
    if pod is None or pod.status == "TERMINATED":
        raise CliError(
            f"pod {pod_id} is {'gone' if pod is None else 'TERMINATED'} on this account, so "
            f"there is nothing to add. `gpuc pods` lists the pods it can see."
        )
    reached = rented.address_for(address.name, pod)
    if reached is None:
        raise CliError(
            f"pod {pod_id} ({pod.name}) is {pod.status} and has no direct SSH endpoint yet, so "
            f"it cannot be asked what it is. Try again once `gpuc pods` shows it RUNNING."
        )
    return address.model_copy(
        update={"kind": "runpod", "ssh": reached.ssh, "port": reached.port, "pod_id": pod.id}
    )


def cmd_host_add(args: argparse.Namespace) -> int:
    """Register a host by asking it what it is.

    The registry holds the address; the host holds its config. So this probes,
    and a host that already has a `config.json` is adopted as it stands --
    which is what makes a second machine driving a box somebody else set up the
    ordinary path. Flags are explicit overrides of it, and say so.
    """
    settings = load_settings()
    if args.pod and args.ssh:
        raise UsageError(
            f"--pod {args.pod} finds the host's ssh endpoint from the provider, so it cannot be "
            f"given --ssh {args.ssh} as well."
        )
    address = _address(args)
    if args.pod:
        address = _pod_address(address, args.pod, settings)
    # The flags are judged before the host is touched: a typo in `--gpus` is
    # the caller's mistake and should not cost a probe to find out.
    fields, env_updates = _config_fields(args), _env_updates(args)
    report = probe_host(address, settings)
    address = address.with_cache(
        gpu_info=report.gpu_info,
        driver_version=report.driver_version,
        python=report.host_python,
    )
    connection = connect_host(
        address,
        settings,
        fields=fields,
        env_updates=env_updates,
        force=args.force,
        before_write=lambda adopted: _refuse_a_taken_name(
            load_registry().hosts.get(adopted.name), adopted, args.name
        ),
    )
    entry = connection.entry
    with registry_transaction() as registry:
        current = registry.hosts.get(entry.name)
        _refuse_a_taken_name(current, entry, args.name)
        if current is not None:
            # Re-registering a host this machine knows: its own bootstrap and
            # the interpreter that bootstrap chose are still true, and worth
            # more than what a probe can see.
            entry = entry.with_cache(python=current.python).model_copy(
                update={"bootstrapped_at": current.bootstrapped_at}
            )
        registry.put(entry)
    warnings: list[str] = []
    if not connection.adopted and not entry.gpus:
        warnings.append(_owns_nothing_warning(entry, args, report))
    if args.pod and not connection.adopted:
        # A pod nobody has set up has no dispatcher, so nothing will ever idle
        # it out: it bills until bootstrap gives it one or a person ends it.
        warnings.append(
            f"nothing has bootstrapped this pod, so nothing on it will ever terminate it: "
            f"`gpuc host bootstrap {entry.name}` gives it a dispatcher that does"
        )
    if args.json:
        jsonout.emit(connection_document(entry, connection, warnings=warnings))
        return EXIT_OK
    lines = [_added_line(entry, connection, args.name)]
    lines += [f"  {warning}" for warning in warnings]
    print("\n".join(lines))
    return EXIT_OK


def _owns_nothing_warning(entry: HostEntry, args: argparse.Namespace, report: ProbeReport) -> str:
    """A first config that owns no card is legal and useless; say which it was."""
    if args.gpus is not None:
        why = "--gpus '' asked for none"
    elif not report.has_nvidia_smi:
        why = "it has no nvidia-smi"
    elif entry.config.shared_gpus:
        why = "every card it has is shared"
    else:
        why = "nvidia-smi found no cards on it"
    return (
        f"it owns no GPUs ({why}), so nothing can be submitted to it: "
        f"`gpuc host set {entry.name} --gpus <list>` assigns some"
    )


def _refuse_a_taken_name(current: HostEntry | None, entry: HostEntry, asked_for: str) -> None:
    """Never let an adopted name replace a different host registered under it.

    A host's `config.json` says what it calls itself, and taking that name is
    what keeps two machines agreeing about one box. But a box somebody set up
    as `local` on their own machine is called `local` here too, and registering
    it would otherwise overwrite *this* machine's `local` -- silently, since
    the address is the only thing that differs.

    Judged before the host is written to as well as before the registry is, so
    a refusal does not leave the flags applied to somebody's host.
    """
    if entry.name == asked_for or current is None:
        return
    if (current.ssh, current.port, current.remote_home) == (
        entry.ssh,
        entry.port,
        entry.remote_home,
    ):
        return
    raise CliError(
        f"host {asked_for} calls itself {entry.name!r}, and a different host is already "
        f"registered here under that name ({current.ssh or 'this machine'}).\n"
        f"Registering it would replace that one. If they are the same box, remove the entry "
        f"here first (`gpuc host remove {entry.name}`); if they are not, give one of them a "
        f"name of its own by editing `host` in its ~/.gpuc/config.json."
    )


def _added_line(entry: HostEntry, connection: Connection, asked_for: str) -> str:
    lines = [
        f"added host {entry.name} [{entry.kind}] "
        f"{entry.ssh or 'this machine'} with {len(entry.gpus)} GPU(s)"
    ]
    if connection.adopted:
        lines.append(f"adopted the config on the host ({connection.home}/config.json)")
        if entry.name != asked_for:
            lines.append(
                f"the host calls itself {entry.name!r}, not {asked_for!r}, so that is the name "
                f"it is registered under here"
            )
        # What the flags changed about somebody's host. On a host that had no
        # config every field "changed", and the line above already said so.
        lines += [f"  host <- {change}" for change in connection.changes]
    else:
        lines.append(f"wrote its first config to {connection.home}/config.json")
    home = _home_line(entry)
    if home:
        lines.append(home)
    lines.append(f"next: gpuc host bootstrap {entry.name}")
    return "\n".join(lines)


def _days(raw: str | None, flag: str) -> float | None:
    """A horizon flag: a number of days, or '' to go back to keeping everything.

    A string, not `type=float`, because argparse cannot express "given but
    empty" for a float -- and a horizon that can be set but never unset is a
    trap. Zero is a real answer: "reclaim it as soon as it finishes" is what
    `cleanup: always` says per job.
    """
    if raw is None or raw == "":
        return None
    try:
        days = float(raw)
    except ValueError as exc:
        raise UsageError(f"{flag} wants a number of days or '', got {raw!r}") from exc
    # `float` takes "nan" and "inf". A NaN horizon compares false against every
    # age, so it would sweep everything that has finished at all, and it is not
    # even JSON either half could write.
    if not math.isfinite(days):
        raise UsageError(f"{flag} wants a number of days or '', got {raw!r}")
    if days < 0:
        raise UsageError(f"{flag} cannot be negative")
    return days


def _home_line(entry: HostEntry) -> str:
    if entry.root is None:
        return ""
    return (
        f"persistent root {entry.root}, so gpuc home (queue, specs, state, logs, "
        f"workdirs) is {entry.remote_home}"
    )


_ADDRESS_FIELDS = ("persistent_root", "gpuc_home")
"""What `gpuc host set` changes here rather than on the host: how to reach it."""

_SET_FIELDS = (
    "gpus",
    "shared_gpus",
    "persistent_root",
    "gpuc_home",
    "env",
    "cache_dir",
    "s3_prefix",
    "retention_days",
    "workdir_days",
    "idle_min",
)


def cmd_host_set(args: argparse.Namespace) -> int:
    """Change one host: its address here, its config on the host itself.

    The config half writes through to the host's `config.json`, because that is
    the only copy of it. It therefore needs the host to answer -- there is
    nothing to set offline -- and what it changed is reported field by field.
    """
    address: dict[str, object] = {}
    for flag in _ADDRESS_FIELDS:
        value = getattr(args, flag)
        if value is not None:
            address[flag] = value or None
    fields = _config_fields(args)
    env_updates = _env_updates(args)
    if not address and not fields and not env_updates:
        raise UsageError(
            "host set changes nothing: pass at least one of "
            + ", ".join(f"--{f.replace('_', '-')}" for f in _SET_FIELDS)
        )
    # The lookup goes through a transaction so that a registry this build
    # cannot read refuses the whole command, in the words that say nothing was
    # written, rather than failing halfway through with the host already changed.
    with registry_transaction() as registry:
        entry = registry.require(args.name)
    lines = [f"host {entry.name}:"]
    # The config first, and through the address the host still has: a
    # `--persistent-root` in the same command moves gpuc home, and writing the
    # config to where the host is not would leave the real one behind.
    config: dict[str, Any] | None = None
    if fields or env_updates:
        connection = push_config(entry, load_settings(), fields=fields, env_updates=env_updates)
        entry, config = connection.entry, connection.entry.cache.config
        lines += [f"  host <- {change}" for change in connection.changes] or [
            "  host already holds that config; nothing changed"
        ]
    else:
        # The address alone changed, so the host was not asked: the document
        # still says which config it holds, from the cache, dated as such.
        connection = Connection(entry=entry, home=entry.remote_home, adopted=True)
    entry = entry.model_copy(update=address)
    lines += [f"  here <- {key}={value!r}" for key, value in sorted(address.items())]
    warnings: list[str] = []
    # Re-read under the lock: the entry above was read before an ssh round
    # trip, and writing it back whole would undo whatever a concurrent `gpuc
    # host probe` or submit learned about the same host in between.
    with registry_transaction() as registry:
        current = registry.hosts.get(entry.name)
        if current is None:
            warnings.append(f"host {entry.name} was removed while this ran; nothing was registered")
        else:
            updated = current.model_copy(update=address)
            registry.put(updated if config is None else updated.with_config(config))
    for warning in warnings:
        warn(warning)
    home = _home_line(entry)
    if home:
        lines.append(home)
        lines.append(f"the host moves there on: gpuc host bootstrap {entry.name}")
    if args.json:
        document = connection_document(entry, connection, warnings=warnings)
        jsonout.emit({**document, "address": address})
        return EXIT_OK
    print("\n".join(lines))
    return EXIT_OK


def cmd_host_remove(args: argparse.Namespace) -> int:
    document = remove_host(args.name)
    if args.json:
        jsonout.emit(document)
        return EXIT_OK
    print(f"removed host {args.name}")
    for text in document["notes"]:
        print(f"  {text}")
    return EXIT_OK


def cmd_host_list(args: argparse.Namespace) -> int:
    read = read_registry_warned()
    registry = read.registry
    if args.json:
        jsonout.emit(hosts_document(read))
        return EXIT_LOCAL_STATE if read.unreadable else EXIT_OK
    if not registry.hosts:
        if read.unreadable:
            return EXIT_LOCAL_STATE
        print(NO_HOSTS)
        return EXIT_OK
    for entry in registry.hosts.values():
        summary = summarize(entry.gpus, entry.gpu_info) if entry.gpus else "no GPUs"
        driver = f", driver {entry.driver_version}" if entry.driver_version else ""
        # One block per host, shaped like `gpuc status`: what the host is, then
        # its cards, then the bootstrap facts. The interpreter path used to sit
        # in the header and was longer than everything else on the line put
        # together; `gpuc host list --json` and `gpuc host probe` still have it.
        print(
            f"host {entry.name} [{entry.kind}] {entry.ssh or 'this machine'}  "
            f"gpus {len(entry.gpus)} ({summary}{driver})"
        )
        stale = shipped_note(entry)
        if stale:
            print(f"  NOTE {stale}")
        for index, name, vram, uuid in gpu_rows(entry.gpus, entry.gpu_info):
            print(f"  gpu     [{index}] {name:<28} {vram:<7} {uuid}")
        for index, name, vram, uuid in gpu_rows(entry.config.shared_gpus, entry.gpu_info):
            print(f"  shared  [{index}] {name:<28} {vram:<7} {uuid}")
        # Everything above and here is the cache: what the host said the last
        # time anything on this machine asked it. The host owns all of it, so
        # it is labelled with its age rather than printed as current.
        seen = (
            f"as of {status_mod.format_age(entry.seen_at)}"
            if entry.seen_at
            else f"never read; run gpuc host probe {entry.name}"
        )
        print(f"  pkg     {version_mod.short(entry.pkg_commit)} on the host, {seen}")
        if entry.bootstrapped_at:
            print(
                f"  boot    bootstrapped from here {status_mod.format_age(entry.bootstrapped_at)}"
            )
        if entry.root:
            print(f"  root    {entry.root} (gpuc home {entry.remote_home})")
    return 0


def bootstrap_and_record(
    entry: HostEntry, settings: Settings, health_args: str, report: Reporter = print
) -> BootstrapResult:
    """Bootstrap one host, persist what it told us about itself, and say so."""
    updated, result = bootstrap_host(entry, settings, health_args=health_args, report=report)
    with registry_transaction() as registry:
        registry.put(updated)
    report(result.render())
    return result


@dataclass
class BootstrapTally:
    """The last word of a `--all` run: what worked, what did not, what was never read.

    Counted rather than claimed, because the run this ends can be long enough
    that nobody reads the middle of it: a host this build could not parse out
    of the registry was never bootstrapped either, and saying "all of them"
    over the top of that warning is how one gets missed for a month. One
    entry per registered host, in the order they were taken, so the `--json`
    form is the tally as data rather than as a sentence.
    """

    hosts: list[HostEntry]
    unreadable: list[str]
    """Registry entries this build could not read, by name: never attempted."""
    errors: list[str]
    outcomes: list[dict[str, Any]] = field(default_factory=list)
    interrupted: bool = False

    def record(
        self,
        entry: HostEntry,
        outcome: str,
        result: BootstrapResult | None = None,
        *,
        error: str | None = None,
    ) -> None:
        detail = result.document() if result else BootstrapResult.no_document()
        detail.pop("host", None)
        self.outcomes.append(
            {
                "name": entry.name,
                "outcome": outcome,
                "error": error,
                "ephemeral": entry.ephemeral,
                **detail,
            }
        )

    @property
    def done(self) -> list[str]:
        return [o["name"] for o in self.outcomes if o["outcome"] == "bootstrapped"]

    @property
    def failed(self) -> list[dict[str, Any]]:
        return [o for o in self.outcomes if o["outcome"] == "failed"]

    def render(self) -> str:
        lines = [f"{len(self.done)}/{len(self.hosts)} host(s) bootstrapped"]
        if self.failed:
            lines.append(f"failed: {', '.join(o['name'] for o in self.failed)}")
            if any(o["ephemeral"] for o in self.failed):
                lines.append(
                    "an ephemeral host whose pod is already gone is forgotten by "
                    "`gpuc host remove <name>`"
                )
        if self.unreadable:
            lines.append(
                f"{len(self.unreadable)} host(s) in the registry could not be read (warnings above)"
            )
        return "\n".join(lines)

    def document(self) -> dict[str, Any]:
        """`gpuc host bootstrap --all --json`: every registered host and what became of it."""
        attempted = {o["name"] for o in self.outcomes}
        for entry in self.hosts:
            if entry.name not in attempted:
                self.record(entry, "not_attempted")
        return {
            "hosts": self.outcomes,
            "total": len(self.hosts),
            "bootstrapped": self.done,
            "failed": [o["name"] for o in self.failed],
            "unreadable": list(self.unreadable),
            "interrupted": self.interrupted,
            "errors": list(self.errors),
        }


def bootstrap_every_host(
    settings: Settings, health_args: str, *, as_json: bool, report: Reporter
) -> int:
    """`gpuc host bootstrap --all`: the upgrade loop, one command.

    A host that fails does not stop the others: an ephemeral host whose pod is
    already gone is the ordinary case, and the hosts that are still there are
    the reason the flag exists. Each failure is named again in the tally and
    the command exits 1, so nobody reads a wall of output as "all upgraded".
    """
    read = read_registry_warned()
    if read.unreadable:
        raise LocalStateUnreadable("\n".join(read.errors))
    hosts = list(read.registry.hosts.values())
    tally = BootstrapTally(hosts, sorted(read.skipped), list(read.errors))
    if not hosts:
        if as_json:
            jsonout.emit(tally.document())
        else:
            print(NO_HOSTS)
        return EXIT_OK
    code = EXIT_OK
    for index, entry in enumerate(hosts, start=1):
        if index > 1:
            report("")
        report(f"== {entry.name} ({index}/{len(hosts)}) ==")
        try:
            tally.record(
                entry, "bootstrapped", bootstrap_and_record(entry, settings, health_args, report)
            )
        except KeyboardInterrupt:
            # Health alone allows five minutes a host, so this is a command
            # somebody does give up on; what it got through is still true.
            report(f"\ninterrupted during {entry.name}")
            tally.record(entry, "interrupted")
            tally.interrupted = True
            code = EXIT_ERROR
            break
        except LocalStateUnreadable:
            # The registry stopped being readable mid-run, so the next host's
            # write would be a guess: say how far this got, and exit 3. Under
            # --json the error document is the one stdout gets.
            report(f"\n{tally.render()}")
            raise
        except (BootstrapError, ConfigError, RemoteError, TransportError) as exc:
            print(f"error: host {entry.name}: {exc}", file=sys.stderr)
            tally.record(entry, "failed", error=str(exc))
            code = EXIT_ERROR
    if as_json:
        jsonout.emit(tally.document())
        return code
    print()
    print(tally.render())
    return code


def cmd_host_bootstrap(args: argparse.Namespace) -> int:
    settings = load_settings()
    if args.all:
        if args.name:
            raise UsageError(
                f"host bootstrap takes a host name or --all, not both (got {args.name!r})"
            )
        return bootstrap_every_host(
            settings, args.health_args, as_json=args.json, report=reporter(args)
        )
    if not args.name:
        raise UsageError("host bootstrap wants a host name, or --all for every registered host")
    entry = named_registry().require(args.name)
    result = bootstrap_and_record(entry, settings, args.health_args, reporter(args))
    if args.json:
        jsonout.emit(result.document())
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
    report = prune_uv_cache(entry, load_settings())
    if args.json:
        jsonout.emit(report.document())
    else:
        print(report.render())
    return EXIT_OK


def cmd_host_probe(args: argparse.Namespace) -> int:
    """Refresh what this machine knows about a host, and print it. Nothing else.

    A probe is the one command that runs before bootstrap, so it is also the
    first chance to learn what the cards are -- every card, not just the
    assigned ones, which is what makes `gpuc host set <name> --gpus 5` nameable
    later. It reads the host's config too, so the offline listings stop being
    stale, but it never writes one: a probe changes nothing about the host.
    """
    settings = load_settings()
    entry = named_registry().require(args.name)
    report = probe_host(entry, settings)
    if not args.json:
        print(report.render(all_gpus=args.all_gpus))
    config = _probe_config(entry, settings)
    # Written even when the host had nothing new to say: *when* it was last
    # read is half of what the offline listings report.
    with registry_transaction() as registry:
        current = registry.hosts.get(args.name)
        if current is not None:
            current = current.with_cache(
                gpu_info=report.gpu_info or None,
                driver_version=report.driver_version,
                # Only until a bootstrap of our own records the interpreter uv
                # picked: a host somebody else set up is worth being able to
                # read before then.
                python=current.python or report.host_python,
            )
            registry.put(current if config is None else current.with_config(config))
    # After the registry write, not before: that write can fail (a held lock, a
    # registry that changed under us) and print an error document of its own,
    # and stdout may hold only one.
    if args.json:
        jsonout.emit(report.document())
    return EXIT_OK


def _probe_config(entry: HostEntry, settings: Settings) -> dict[str, Any] | None:
    """The host's `config.json`, or None if it has none or could not be read."""
    try:
        transport = transport_for(entry, settings)
        return read_remote_config(transport, resolve_home(transport, entry)) or None
    except (ConfigError, RemoteError, TransportError):
        return None


CLOUDS: dict[str, list[Cloud]] = {
    "secure": ["SECURE"],
    "community": ["COMMUNITY"],
    "any": ["SECURE", "COMMUNITY"],
}


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
        disk_gb=args.disk if args.disk is not None else settings.disk_gb,
        image=args.image or settings.image,
        health_args=args.health_args,
    )


def cmd_config_init(args: argparse.Namespace) -> int:
    document = init_config(force=args.force)
    if args.json:
        jsonout.emit(document)
        return EXIT_OK
    print(
        f"wrote {document['config_file']}\n"
        f"Every key is commented with its default; edit what you need."
    )
    return EXIT_OK


def cmd_config_show(args: argparse.Namespace) -> int:
    settings = load_settings()
    document = config_document(settings)
    if args.json:
        jsonout.emit(document)
        return EXIT_OK
    path = config_file()
    print(f"config file: {path}{'' if path.exists() else ' (does not exist; using defaults)'}")
    print(f"state dir:   {state_dir()}")
    for name, value in settings.model_dump().items():
        print(f"  {name} = {value!r}")
    for text in document["notes"]:
        print(f"  note: {text}")
    return EXIT_OK


def mirror_spec_first(
    model: JobSpecModel, job_id: str, settings: Settings
) -> tuple[str | None, list[str]]:
    """Put the spec in S3 before spending any money, so a lost pod is still requeueable.

    Returns the uri it landed at, so the submit that follows does not PUT the
    same object a second time. `{job_id}` goes up unexpanded: the mirror is
    what `requeue` submits, and that run must land in its own namespace.
    """
    s3 = S3Index.from_settings(settings)
    if s3 is None:
        return None, [
            "s3_bucket is unset, so the spec was not mirrored before provisioning; "
            "`gpuc requeue` will need the job file again"
        ]
    try:
        return s3.put_spec(model.to_spec(job_id)), []
    except S3IndexError as exc:
        return None, [f"could not mirror the spec to S3 before provisioning: {exc}"]


def check_runpod_args(args: argparse.Namespace) -> None:
    """Judge the provisioning flags once, before anything is bought or written."""
    if args.runpod and args.host:
        raise UsageError(
            f"--runpod creates a pod and --host {args.host} names a host that already "
            f"exists, so they cannot be combined. Drop one."
        )


def reporter(args: argparse.Namespace) -> Reporter:
    """Where a step's progress goes: stdout, or stderr when stdout is a document."""
    return jsonout.note if args.json else print


def ensure_package_current(
    entry: HostEntry, settings: Settings, *, bootstrap: bool = True, report: Reporter = print
) -> HostEntry:
    """Read the host's config, and re-ship the package if it is not this build.

    The host's `config.json` is the only copy of what the host is, so this read
    is also what makes the rest of the submit true: the `gpus` the spec is
    judged against and the `s3_prefix` the job's outputs are recorded under are
    the host's own answer, from a moment ago, not whatever this machine last
    wrote down.

    The commit comes from the same file rather than from this registry, which
    only ever recorded what this machine shipped: two control machines against
    one box -- a laptop and a desktop -- each leave that record describing a
    host the other has since re-bootstrapped, and every submit would then skip
    the check it exists for. An *unrecorded* commit counts as older, because
    the hosts with nothing recorded were bootstrapped by the oldest builds of all.

    Only the package and the dispatcher are re-shipped: uv, the interpreter and
    health cannot have gone stale, and the job is waiting.
    """
    if not bootstrap:
        return entry
    if not (entry.bootstrapped_at or entry.pkg_commit):
        # Nothing has ever installed gpuc on this host -- not this machine, and
        # not whoever else would have left a commit in its config. Re-shipping
        # the package alone would start a dispatcher on a host with no uv and
        # no interpreter of its own, and the first anyone would hear of it is
        # the job failing there.
        raise CliError(
            f"host {entry.name} has no gpuc on it yet: nothing recorded here or in its own "
            f"config says it was ever bootstrapped.\nRun: gpuc host bootstrap {entry.name}"
        )
    if not entry.python:
        return entry
    local = version_mod.local_commit()
    session = _try_session(entry, settings)
    config = session.read_config() if session else None
    # A host that could not be asked keeps the cache it had; the submit right
    # behind this produces the transport error in full.
    if config is None:
        return entry
    if config:
        entry = _record_config(entry, config)
        host_commit = entry.pkg_commit
    else:
        # The host answered and has no config at all: its gpuc home was wiped,
        # taking the package with it. The cache here is the only copy of what
        # that host was, so it is kept rather than overwritten with nothing --
        # and an unrecorded commit re-ships below, which restores both.
        host_commit = None
    if not version_mod.needs_package_sync(local, host_commit):
        return entry
    report(
        f"host {entry.name} is running gpuc {version_mod.short(host_commit)} and this machine "
        f"has {version_mod.short(local)}: re-syncing the package and restarting the "
        f"dispatcher before enqueueing"
    )
    transport = session.transport if session else None
    updated = resync_package(entry, settings, transport=transport, report=_quiet)
    with registry_transaction() as registry:
        registry.put(updated)
    return updated


def _try_session(entry: HostEntry, settings: Settings) -> HostSession | None:
    try:
        return open_session(entry, settings)
    except (ConfigError, RemoteError, TransportError):
        return None


def _record_config(entry: HostEntry, config: dict[str, Any]) -> HostEntry:
    """Cache what the host just said about itself, for the offline commands.

    `gpuc host list` and `gpuc version` never ask a host anything, so this is
    what keeps them from repeating a bootstrap somebody else replaced. Written
    even when the config has not changed, because *when* it was read is half of
    what those commands report. Re-read under the lock and written as one
    field, because this runs on every submit and writing back the whole entry
    read at startup would undo whatever a concurrent `gpuc host probe` learned
    about the same host.
    """
    updated = entry.with_config(config)
    with registry_transaction() as registry:
        current = registry.hosts.get(entry.name)
        registry.put(current.with_config(config) if current else updated)
    return updated


def _quiet(_: str) -> None:
    """Swallow a step's progress: the caller has already said what it is doing."""


def cmd_submit(args: argparse.Namespace) -> int:
    settings = load_settings()
    check_runpod_args(args)
    use_git = not args.no_git
    report = reporter(args)
    # None, not False, for a flag that was not passed: a spec that says
    # `use_shared: true` keeps saying it when nobody typed --use-shared.
    overrides = {"use_shared": True if args.use_shared else None}
    if args.runpod:
        document = with_overrides(load_document(args.job_file), **overrides)
        model = validate(document, str(args.job_file))
        precheck_local(
            model,
            Path.cwd(),
            gpu_count=args.gpu_count,
            use_git=use_git,
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
        return _queued(result, args, entry, settings)
    if not args.host:
        raise UsageError("submit needs --host <name> (see `gpuc host list`)")
    entry = named_registry().require(args.host)
    entry = ensure_package_current(entry, settings, bootstrap=not args.no_bootstrap, report=report)
    result = submit_file(
        entry,
        args.job_file,
        settings,
        overrides,
        workdir=Path.cwd(),
        use_git=use_git,
        report=report,
    )
    return _queued(result, args, entry, settings)


def _queued(
    result: SubmitResult,
    args: argparse.Namespace,
    entry: HostEntry,
    settings: Settings,
    *,
    requeued_from: str | None = None,
) -> int:
    """The last word of `submit` and `requeue`, in whichever form was asked for.

    The queue is looked up again here rather than inferred from the enqueue:
    the dispatcher the enqueue started may well have taken the job already, and
    "position 3 of 5, starts in ~2h" is the thing the submitter actually wants
    to know and cannot work out from a job id.
    """
    result.placement = placement_after(entry, result.job_id, settings, session=result.session)
    if args.json:
        jsonout.emit(result.document(requeued_from=requeued_from))
        return EXIT_OK
    print(result.render(status_mod.queue_note(result.placement)))
    if requeued_from is not None:
        print(
            f"  requeued from {requeued_from} (attempt {result.attempt}); "
            f"workdir re-synced from {Path.cwd()}"
        )
    return EXIT_OK


def cmd_status(args: argparse.Namespace) -> int:
    """Report on every host, and only fail for reasons that are not a host.

    An unreachable host, a dead dispatcher and a pod that is gone are all
    *answers*, printed per host with exit 0. The one non-zero case is exit 3:
    the local registry could not be read, so "no jobs running" would be a
    guess. Automation must treat that as unknown and never as idle.
    """
    settings = load_settings()
    read = read_registry_warned()
    try:
        since_s = status_mod.parse_duration(args.since) if args.since else None
    except ValueError as exc:
        raise UsageError(f"--since: {exc}") from exc
    if args.json:
        jsonout.emit(
            status_document(read, settings, host=args.host, recent=args.recent, since_s=since_s)
        )
        return EXIT_LOCAL_STATE if read.unreadable else EXIT_OK
    entries = hosts_for(read.registry, args.host) if not read.unreadable else []
    if not entries:
        if read.unreadable:
            print(
                f"cannot read {hosts_file()}, so no host status is known "
                f"(this is not `no jobs running`)",
                file=sys.stderr,
            )
            return EXIT_LOCAL_STATE
        print(NO_HOSTS)
        # ...but `--all` still has something to say: the index remembers jobs
        # whose host has since been removed.
        if args.all:
            _print_unhosted(settings, set(), args.host)
        return EXIT_OK
    provider = provider_for_status(entries, settings)
    seen: set[str] = set()
    for view in gather_all(entries, settings, provider):
        seen.update(job.job_id for job in view.queue + view.running + view.finished)
        print(status_mod.render(view, recent=args.recent, since_s=since_s))
    if args.all:
        _print_unhosted(settings, seen, args.host)
    return EXIT_OK


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
            note(f"could not read the S3 index: {exc}")
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
        flag = (
            " OUTPUTS LOST (the host went away before they uploaded)"
            if entry.job_id in lost
            else ""
        )
        label = f"{entry.name} ({entry.job_id})" if entry.name else entry.job_id
        print(
            f"  {label} host={entry.host} attempt={entry.attempt} "
            f"submitted {status_mod.format_age(entry.submitted_at)}{flag}"
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
        uri = job_uri(entry.s3_prefix, entry.job_id, "state.json")
        try:
            document = json.loads(s3.get_uri(uri))
        except (S3IndexError, json.JSONDecodeError):
            continue
        if isinstance(document, dict) and document.get("outputs_lost"):
            lost.add(entry.job_id)
    return lost


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


def _answer(args: argparse.Namespace, document: dict[str, Any], *text: str | None) -> int:
    """A job command's last word: its warnings on stderr, then the document or the text."""
    for warning in document.get("warnings", []):
        warn(warning)
    if args.json:
        jsonout.emit(document)
    else:
        print("\n".join(line for line in text if line))
    return EXIT_OK


def cmd_cancel(args: argparse.Namespace) -> int:
    document = cancel_job(args.job_id, args.host, load_settings())
    return _answer(
        args, document, f"job {args.job_id} on host {document['host']}: {document['status']}"
    )


def cmd_reorder(args: argparse.Namespace) -> int:
    document = reorder_job(args.job_id, args.priority, args.host, load_settings())
    return _answer(
        args,
        document,
        f"job {args.job_id} on host {document['host']} moved to priority {args.priority}",
        status_mod.queue_note(document),
    )


def cmd_preempt(args: argparse.Namespace) -> int:
    """Stop a running job and put it back in its host's queue.

    The text output says which priority it comes back at, because that is what
    decides which job runs next: dispatch order is `<priority>-<job id>`, so a
    job waiting at a lower number takes the cards, and one waiting at the
    *same* priority does not -- the preempted job was submitted first, so its
    id sorts ahead and it takes its own cards straight back. The host refuses
    outright when nothing at all would go first, rather than throw away what
    the job has done to re-run the same job.
    """
    document = preempt_job(args.job_id, args.priority, args.host, load_settings())
    priority = document["priority"]
    at = f"; it will be queued again at priority {priority}" if priority is not None else ""
    return _answer(
        args, document, f"job {args.job_id} on host {document['host']}: {document['status']}{at}"
    )


def cmd_estimate(args: argparse.Namespace) -> int:
    """Add, change or clear a job's `estimated_runtime_min` after submitting it.

    It is the one spec field somebody else needs and only the submitter knows,
    and the job that most needs one is the long job already running when the
    next person arrives -- which is too late to edit a file before `submit`.
    """
    wanted = check_estimate(args.minutes, clear=args.clear)
    document = estimate_job(args.job_id, wanted, args.host, load_settings())
    recorded = document["estimated_runtime_min"]
    job = f"job {args.job_id} on host {document['host']}"
    return _answer(
        args,
        document,
        f"{job} no longer estimates a runtime"
        if recorded is None
        else f"{job} now estimates {recorded:g} min",
    )


def _follow_argv(transport: Transport, remote_path: str, lines: int) -> list[str]:
    # `-F`, not `-f`: `gpuc submit && gpuc logs -f` is the obvious pair to type,
    # and a job that has not been dispatched yet has no log.txt to open.
    command = tail_command(remote_path, lines, follow=True, retry=True)
    if isinstance(transport, SshTransport):
        return transport.ssh_argv(command)
    return ["bash", "-c", command]


def check_interval(args: argparse.Namespace, *, polls: bool) -> None:
    """`--interval` paces a poll, so it is a usage error where nothing polls."""
    if args.interval is None:
        return
    if not polls:
        raise UsageError(
            "--interval paces the check for whether the job has ended, which only "
            "`gpuc logs -f` and `gpuc wait` do. Drop it, or use -f."
        )
    if not args.interval > 0.0:
        raise UsageError(f"--interval must be a positive number of seconds, got {args.interval:g}")


def cmd_logs(args: argparse.Namespace) -> int:
    if args.follow and args.follow_forever:
        raise UsageError(
            "-f and --follow-forever are the two different things you can mean by "
            "following: -f stops when the job does, --follow-forever never stops."
        )
    if args.json and (args.follow or args.follow_forever):
        raise UsageError(
            "logs --json cannot follow: the document is printed once, and a follow is a "
            "stream. `gpuc wait <job-id> --json` is the JSON form of waiting for a job."
        )
    check_interval(args, polls=args.follow)
    settings = load_settings()
    if args.follow_forever:
        entry, _ = find_job_host(args.job_id, named_registry(), args.host)
        session, remote = job_log_path(entry, args.job_id, settings)
        return _follow_forever(session.transport, remote, args.lines)
    if args.follow:
        return _follow_until_done(args, settings)
    entry, log = read_log(args.job_id, args.host, args.lines, settings)
    # Bytes for a human; lines plus where they came from for a script.
    if args.json:
        jsonout.emit(log.document(args.job_id, entry.name))
    else:
        sys.stdout.write(log.text)
    return EXIT_OK


def _follow_forever(transport: Transport, remote: str, lines: int) -> int:
    """`--follow-forever`: the stream with no end, and no claim about the job.

    Ctrl-C is the only way out and is exit 0 here, unlike `-f`: this form never
    promised its exit code meant anything about the run.
    """
    argv = _follow_argv(transport, remote, lines)
    try:
        return subprocess.call(argv)
    except KeyboardInterrupt:
        return EXIT_OK


def _follow_until_done(args: argparse.Namespace, settings: Settings) -> int:
    """`logs -f`: the log while the job runs, its outcome, and exit with it.

    The stream and the polling are two things at once because there is nothing
    in a log that says a job has ended -- the host's state file is the only
    thing that does. So `tail` runs as a child writing straight to our stdout
    while this thread asks the host, and the job's own outcome becomes the exit
    code, which is what makes `gpuc logs -f "$id"` a foreground wait on its own.
    """
    watch = wait_mod.start([args.job_id], args.host, settings)
    watched = watch.jobs[args.job_id]
    watch.poll()
    watch.check_known()
    if watched.settled:
        # Nothing more is coming, and `tail -f` on it would simply hang. The
        # ordinary read, so a purged job still falls back to the S3 mirror.
        _, log = read_log(args.job_id, watched.host, args.lines, settings)
        sys.stdout.write(log.text)
        print(watched.line())
        return wait_mod.exit_code([watched])
    if watched.status == "queued":
        # Said before tail is started, because tail then says `cannot open ...
        # No such file or directory` about a log the host has not opened yet,
        # and on its own that reads like a failure rather than a queue.
        note(f"job {args.job_id} is queued; following the log from when it starts")
    # The watch's own session, rather than a second one: opening another costs
    # a round trip to resolve the same home on the same host.
    session = watch.session(watched.host)
    tail = subprocess.Popen(
        _follow_argv(session.transport, f"{session.job_dir(args.job_id)}/log.txt", args.lines)
    )
    reported_stream_end = False

    def check_tail() -> None:
        """Say so once if the stream died, rather than freeze the log in silence.

        An ssh whose keepalives ran out takes the tail with it. The wait itself
        is fine -- it polls over its own connection -- so this is a note and
        not an ending.
        """
        nonlocal reported_stream_end
        if reported_stream_end or tail.poll() is None:
            return
        reported_stream_end = True
        note("the log stream ended before the job did; still waiting for the job")

    ended = False
    try:
        try:
            wait_mod.block(watch, interval=args.interval, each_round=check_tail)
            ended = True
        finally:
            # Only a job that ended has last lines worth waiting for; an
            # interrupt wants the stream gone now.
            _end_tail(tail, flush=ended)
    except KeyboardInterrupt:
        # Ctrl-C reached the tail too: it shares this process group. The job
        # does not care either way -- its host owns it, not us. A second one,
        # landing in the flush above, arrives here with the job already ended,
        # and then its outcome is still the answer.
        if not ended:
            # Never the job's exit code, because we never learned it: `logs -f`
            # promises 0 means *succeeded*, and a script must not read "the
            # user got bored" as one.
            note(f"interrupted; job {args.job_id} is still {watched.status} on {watched.host}")
            return EXIT_INTERRUPTED
    print(watched.line())
    return wait_mod.exit_code([watched])


def _end_tail(tail: subprocess.Popen[bytes], *, flush: bool) -> None:
    """Stop the stream and reap it, whatever happens in the grace.

    The `finally` matters: a second Ctrl-C lands in the sleep below, and a tail
    left behind keeps writing into a terminal gpuc has already left.
    """
    try:
        if flush and tail.poll() is None:
            time.sleep(wait_mod.FLUSH_GRACE_S)
    finally:
        tail.terminate()
        try:
            tail.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            tail.kill()
            tail.wait()


def cmd_wait(args: argparse.Namespace) -> int:
    """Block until every named job has ended, then exit with their outcome.

    No log: this is the half of `logs -f` a sweep wants, where twenty jobs'
    output interleaved would be unreadable and only the verdicts matter.
    """
    check_interval(args, polls=True)
    watch = wait_mod.start(args.job_ids, args.host, load_settings())
    # Under --json stdout belongs to the document, so the outcomes go to stderr
    # as they happen and are in the document at the end.
    announce = jsonout.note if args.json else print
    try:
        waited = wait_mod.block(
            watch, interval=args.interval, on_settled=lambda watched: announce(watched.line())
        )
    except KeyboardInterrupt:
        # This command exists to be abandoned, so a Ctrl-C is an ordinary way
        # for it to end -- but never a silent one, and never exit 0: under
        # --json an empty stdout is the one thing the flag promises never to
        # give, and the jobs are all still running.
        waiting = ", ".join(job.job_id for job in watch.pending)
        return failed(
            args,
            f"interrupted; still running or queued on their hosts: {waiting}",
            EXIT_INTERRUPTED,
        )
    if args.json:
        jsonout.emit(wait_mod.document(waited))
    return wait_mod.exit_code(waited)


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
    document = from_mirror(document)
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
    return _queued(result, args, entry, settings, requeued_from=args.job_id)


def cmd_pods(args: argparse.Namespace) -> int:
    settings = load_settings()
    view = pods_mod.gather(settings, make_provider(settings), heartbeats=not args.no_heartbeat)
    if args.json:
        jsonout.emit(view.document())
    else:
        print(pods_mod.render(view))
    return EXIT_OK


def cmd_version(args: argparse.Namespace) -> int:
    """What is installed here, and what each host was running when last read.

    The host commits come from the registry's cache -- no ssh, so this stays a
    command you can run before anything else. That also means it cannot see a
    host somebody else has bootstrapped since it was read: `gpuc status` asks
    each host what it is running.
    """
    read = read_registry_warned()
    document = version_document(read)
    if args.json:
        jsonout.emit(document)
        return EXIT_LOCAL_STATE if read.unreadable else EXIT_OK
    dirty = " (+uncommitted changes)" if document["dirty"] else ""
    print(f"gpuc {document['version']}")
    print(f"commit {version_mod.short(document['commit'])} [{document['source']}]{dirty}")
    print(f"python {document['python']} at {document['executable']}")
    hosts = document["hosts"]
    if not hosts:
        print("hosts: none read yet")
        return EXIT_LOCAL_STATE if read.unreadable else EXIT_OK
    print("hosts (as last read from here):")
    for host in hosts:
        note = "" if host["current"] else "  DIFFERS: re-bootstrap"
        seen = f"  {status_mod.format_age(host['seen_at'])}" if host["seen_at"] else ""
        print(f"  {host['name']:<16} pkg {version_mod.short(host['pkg_commit'])}{seen}{note}")
    if not all(host["current"] for host in hosts):
        print(
            "upgrade a host with: gpuc host bootstrap <host> (or --all for every host; "
            "running jobs are not disturbed)"
        )
    return EXIT_LOCAL_STATE if read.unreadable else EXIT_OK


def cmd_web_set_password(args: argparse.Namespace) -> int:
    if args.stdin:
        password = sys.stdin.readline().rstrip("\r\n")
    else:
        password = getpass.getpass("dashboard password: ")
        if password != getpass.getpass("again: "):
            raise UsageError("the two passwords differ; nothing was written")
    path = web_mod.write_password(password)
    print(f"wrote {path} (0600)\nserve the dashboard with: gpuc web serve")
    return EXIT_OK


def cmd_web_serve(args: argparse.Namespace) -> int:
    if args.install:
        web_mod.install_service(args.bind, args.port)
        return EXIT_OK
    server = web_mod.make_server(args.bind, args.port)
    print(
        f"gpuc dashboard on http://{args.bind}:{server.server_port}/ (Ctrl-C to stop)",
        file=sys.stderr,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("stopped", file=sys.stderr)
    finally:
        server.server_close()
    return EXIT_OK


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
    add = host.add_parser(
        "add",
        help="register a host: read the config it already has, or give it its first",
    )
    add.add_argument("name")
    add.add_argument("--ssh", help="user@host; omit for this machine")
    add.add_argument("--port", type=int, default=22, help="ssh port (default 22)")
    add.add_argument(
        "--pod",
        metavar="POD_ID",
        help="adopt a RunPod pod this account is already renting, whichever machine created "
        "it: its address comes from the provider and its config from the pod",
    )
    add.add_argument(
        "--gpus",
        help=f"a host with no config of its own owns every card nvidia-smi reports unless "
        f"this narrows it ('' for none); on a host that has one this reassigns its cards, "
        f"and a list that overlaps the host's is refused. {GPUS_HELP}",
    )
    add.add_argument("--shared-gpus", help=SHARED_GPUS_HELP)
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
        "--workdir-days",
        metavar="DAYS",
        help="reclaim a finished job's workdir (checkout and venv, never its log or "
        "state) once it ended this long ago; pass '' to keep workdirs until you run "
        "`gpuc clean`. A host being configured for the first time gets 1",
    )
    add.add_argument(
        "--idle-min",
        type=float,
        default=None,
        metavar="MINUTES",
        help="how long an ephemeral host may sit with an empty queue before it terminates "
        "itself (default 15); ignored for hosts that are not ephemeral",
    )
    add.add_argument(
        "--force",
        action="store_true",
        help="allow a --gpus that claims some but not all of the cards the host is already "
        "configured with",
    )
    add_json_flag(add, "the host as `host list --json` reports it, plus what this wrote to it")
    add.set_defaults(func=cmd_host_add)

    edit = host.add_parser(
        "set", help="change a host's address here, or its own config on the host"
    )
    edit.add_argument("name")
    edit.add_argument(
        "--gpus",
        help=f"replace what this host owns, on the host itself; pass '' for none. {GPUS_HELP}",
    )
    edit.add_argument(
        "--shared-gpus",
        help=f"replace what this host may borrow; pass '' for none. {SHARED_GPUS_HELP}",
    )
    edit.add_argument("--persistent-root", help="pass '' to go back to $HOME")
    edit.add_argument("--gpuc-home", help="pass '' for the default under the root or $HOME")
    edit.add_argument(
        "--env",
        action="append",
        metavar="KEY=VALUE",
        help="replace this host's job environment; repeatable, '' for none",
    )
    edit.add_argument(
        "--cache-dir", help="pin UV_CACHE_DIR in the host's env; pass '' to let bootstrap decide"
    )
    edit.add_argument("--s3-prefix", help="pass '' to stop mirroring")
    edit.add_argument(
        "--retention-days", help="auto-purge horizon in days; pass '' to keep everything"
    )
    edit.add_argument(
        "--workdir-days",
        metavar="DAYS",
        help="workdir sweep horizon in days; pass '' to keep workdirs until `gpuc clean`",
    )
    edit.add_argument(
        "--idle-min",
        type=float,
        metavar="MINUTES",
        help="idle minutes before an ephemeral host terminates itself",
    )
    add_json_flag(edit, "the host as `host list --json` reports it, plus what this changed")
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
    add_json_flag(
        bootstrap,
        "what the bootstrap left on the host; with --all, one entry per registered host "
        "saying whether it was bootstrapped, failed (and why) or was never reached. "
        "Progress goes to stderr, and the exit code is the same as without it",
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
    add_json_flag(host_clean, "the cache directory and its size before and after the prune")
    host_clean.set_defaults(func=cmd_host_clean)

    host_list = host.add_parser("list", help="list registered hosts")
    add_json_flag(host_list)
    host_list.set_defaults(func=cmd_host_list)
    remove = host.add_parser("remove", help="forget a host")
    remove.add_argument("name")
    add_json_flag(remove, "what was forgotten: the entry's name, kind and pod id")
    remove.set_defaults(func=cmd_host_remove)

    submit = sub.add_parser("submit", help="submit a job file to a host")
    submit.add_argument("job_file")
    submit.add_argument(
        "--host", metavar="NAME", help="a registered host to submit to (see `gpuc host list`)"
    )
    submit.add_argument(
        "--use-shared",
        action="store_true",
        help="let this job run on the host's shared GPUs (`gpuc host set <name> "
        "--shared-gpus`) as well as the ones it owns: cards gpuc does not own and takes "
        "only while nvidia-smi says nobody else is on them. Same as `use_shared: true` "
        "in the spec",
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
    add_json_flag(
        status,
        "one JSON document on stdout: key on hosts[].running, and treat exit 3 "
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

    logs = sub.add_parser(
        "logs",
        help="tail a job log from its host, or follow it until the job ends",
        description="With -f this is also a wait: the log streams until the host says the "
        "job has ended, the outcome is the last line, and gpuc exits 0 only if the job "
        "succeeded.",
    )
    logs.add_argument("job_id")
    logs.add_argument(
        "-f",
        "--follow",
        action="store_true",
        help="stream the log until the job ends, then print its outcome and exit with it "
        "(0 succeeded, 1 anything else)",
    )
    logs.add_argument(
        "--follow-forever",
        action="store_true",
        help="stream the log and never stop, as -f used to: for watching a host's own "
        "writing past the end of a run. Ctrl-C is the only way out",
    )
    logs.add_argument(
        "-n", "--lines", type=int, default=200, metavar="N", help="lines of history (default 200)"
    )
    add_interval_flag(logs)
    logs.add_argument(
        "--host", metavar="NAME", help="which host the job is on, if it cannot be found"
    )
    add_json_flag(
        logs,
        "the log as a list of lines, with the host or S3 uri it came from; "
        "not with -f, which has no end",
    )
    logs.set_defaults(func=cmd_logs)

    wait = sub.add_parser(
        "wait",
        help="block until jobs end; exit non-zero if any of them did not succeed",
        description="Polls each job's host and prints one line per job as it ends. No log "
        "output: `gpuc logs -f <job-id>` is the one-job version that streams. Exit 0 only "
        "if every job succeeded, so `gpuc submit --json | jq -r .job_id | xargs gpuc wait` "
        "is a script.",
    )
    wait.add_argument("job_ids", nargs="+", metavar="job_id")
    add_interval_flag(wait)
    wait.add_argument(
        "--host", metavar="NAME", help="which host the jobs are on, if they cannot be found"
    )
    add_json_flag(wait, "each job's final state, as `status --json` reports one")
    wait.set_defaults(func=cmd_wait)

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

    preempt = sub.add_parser(
        "preempt",
        help="stop a running job and queue it again, to run from the start",
        description="Frees a running job's GPUs for something more important without "
        "losing the job: its runner stops it and syncs whatever it produced, and the host "
        "queues it again under the same job id as its next attempt. It re-runs from the "
        "start, in the workdir the stopped attempt left behind -- nothing is re-synced from "
        "here -- so preempt a job that tolerates being re-run over its own leftovers. "
        "Queue the job you want to run FIRST: this is refused unless something already "
        "waiting would be dispatched ahead of the preempted job, since otherwise it would "
        "only stop it and start it again. Use `gpuc requeue` to re-run a finished job, or "
        "to run one on another host.",
    )
    preempt.add_argument("job_id")
    preempt.add_argument(
        "--priority",
        type=int,
        metavar="N",
        help="0-99; queue it again at this priority instead of its own (lower dispatches "
        "first, so a higher number keeps it out of the way)",
    )
    preempt.add_argument(
        "--host", metavar="NAME", help="which host the job is on, if it cannot be found"
    )
    add_json_flag(preempt)
    preempt.set_defaults(func=cmd_preempt)

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

    pods = sub.add_parser(
        "pods", help="every pod with our prefix: cost, util, age, which host it is here"
    )
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
    add_json_flag(config_init, "the path written, and whether a file was already there")
    config_init.set_defaults(func=cmd_config_init)
    config_show = config.add_parser("show", help="print the effective settings")
    add_json_flag(config_show)
    config_show.set_defaults(func=cmd_config_show)

    web = sub.add_parser(
        "web", help="the web dashboard: serve it, or set the password it asks for"
    ).add_subparsers(dest="web_command", required=True)
    serve = web.add_parser(
        "serve",
        help="serve the dashboard over HTTP until Ctrl-C",
        description="Every page and API call is behind the password `gpuc web set-password` "
        "records. The dashboard is a thin view over the same code the CLI runs: what it "
        "shows is `gpuc status --json`, `gpuc host list --json` and `gpuc config show "
        "--json`, and what it can do is `gpuc cancel`, `gpuc preempt`, `gpuc reorder` and "
        "`gpuc estimate`.",
    )
    serve.add_argument(
        "--bind",
        default=web_mod.DEFAULT_BIND,
        metavar="ADDRESS",
        help=f"address to listen on (default {web_mod.DEFAULT_BIND}; 0.0.0.0 for every "
        f"interface, which is only sensible behind a VPN or a TLS proxy)",
    )
    serve.add_argument(
        "--port",
        type=int,
        default=web_mod.DEFAULT_PORT,
        help=f"port to listen on (default {web_mod.DEFAULT_PORT})",
    )
    serve.add_argument(
        "--install",
        action="store_true",
        help="write (but do not enable) a systemd --user service that serves with these "
        "flags at login, then print the systemctl lines to turn it on",
    )
    serve.set_defaults(func=cmd_web_serve)
    set_password = web.add_parser(
        "set-password", help="record the dashboard password (bcrypt-hashed, never stored)"
    )
    set_password.add_argument(
        "--stdin",
        action="store_true",
        help="read the password from the first line of stdin instead of prompting",
    )
    set_password.set_defaults(func=cmd_web_set_password)
    return parser


JSON_HELP = (
    "one JSON document on stdout and nothing else; progress and warnings go to "
    "stderr. `error` is set (and nothing else but schema_version) when the command "
    "failed, and the exit code is the same as without --json"
)


def add_json_flag(parser: argparse.ArgumentParser, help_text: str = JSON_HELP) -> None:
    parser.add_argument("--json", action="store_true", help=help_text)


def add_interval_flag(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--interval",
        type=float,
        default=None,
        metavar="SECONDS",
        help=f"how often to ask the host whether the job has ended; the default backs off "
        f"from {wait_mod.FIRST_INTERVAL_S:g}s to {wait_mod.MAX_INTERVAL_S:g}s as the wait "
        f"gets longer",
    )


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
    parser.add_argument("--disk", type=int, help="container disk in GB; default from config")
    parser.add_argument("--image", help="pod image; default from config")
    parser.add_argument("--no-reuse", action="store_true", help="always create a new pod")
    parser.add_argument("--name-hint", default="job", help="goes into the pod name")
    parser.add_argument("--health-args", default="", help="extra flags for `gpuc.host health`")


def wants_runpod(args: argparse.Namespace) -> bool:
    if getattr(args, "pod", None):
        return True  # `gpuc host add --pod` asks the provider where that pod is
    return bool(getattr(args, "runpod", False)) or args.command == "pods"


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
            "`gpuc host add --pod` or `gpuc pods`",
            EXIT_ERROR,
        )
    if args.command not in ("config", "skill"):
        first_run_note()
    try:
        return int(args.func(args))
    except Exception as exc:
        code = exit_code_for(exc)
        if code is None:
            raise
        return failed(args, failure_message(exc), code)


if __name__ == "__main__":
    raise SystemExit(main())
