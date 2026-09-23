"""The `gpuc` command line. Thin: parse, call a module, render, and let `main`
emit the answer once -- as text, or as the `--json` document -- and exit with
what the answer says."""

from __future__ import annotations

import argparse
import getpass
import math
import os
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from gpuc.control import jsonout, teardown
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
    Answer,
    CliError,
    Interrupted,
    NotFound,
    UsageError,
    cancel_job,
    check_estimate,
    config_document,
    estimate_job,
    exit_code_for,
    failure_message,
    forget_gone_rentals,
    hosts_document,
    init_config,
    locate,
    make_provider,
    preempt_job,
    read_log,
    registry_answer,
    remove_host,
    reorder_job,
    shipped_note,
    status,
    version_document,
)
from gpuc.control.bootstrap import HealthOptions
from gpuc.control.clean import check_flags as check_clean_flags
from gpuc.control.clean import clean_host, parse_only, prune_uv_cache
from gpuc.control.config import (
    ConfigError,
    HostEntry,
    Reporter,
    Settings,
    config_file,
    hosts_file,
    load_settings,
    open_registry,
    state_dir,
    transport_for,
    update_cache,
)
from gpuc.control.gpuinfo import rows as gpu_rows
from gpuc.control.gpuinfo import summarize
from gpuc.control.hosts import add_host, bootstrap_and_record, bootstrap_every_host, set_host
from gpuc.control.jsonout import note, warn
from gpuc.control.probe import probe_host
from gpuc.control.providers.base import DEFAULT_CUDA_MIN, Cloud
from gpuc.control.remote import HostSession, RemoteError, open_session, read_config, resolve_home
from gpuc.control.skill import install_skill, read_skill
from gpuc.control.submit import SubmitResult
from gpuc.control.submitting import (
    DEFAULT_IDLE_MINUTES,
    RentalOptions,
    requeue_job,
    submit_job,
)
from gpuc.control.transport import (
    NO_GIT_EXCLUDES,
    Transport,
    TransportError,
    tail_command,
)
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


def cmd_host_add(args: argparse.Namespace) -> Answer:
    # The flags are judged before the host is touched: a typo in `--gpus` is
    # the caller's mistake and should not cost a probe to find out.
    fields, env_updates = _config_fields(args), _env_updates(args)
    change = add_host(
        args.name,
        ssh=args.ssh,
        port=args.port,
        gpuc_home=args.gpuc_home,
        persistent_root=args.persistent_root,
        pod_id=args.pod,
        fields=fields,
        env_updates=env_updates,
        force=args.force,
    )
    return Answer(change.document, change.render())


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


def cmd_host_set(args: argparse.Namespace) -> Answer:
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
    change = set_host(args.name, address=address, fields=fields, env_updates=env_updates)
    for warning in change.warnings:
        warn(warning)
    return Answer(change.document, change.render())


def cmd_host_remove(args: argparse.Namespace) -> Answer:
    document = remove_host(args.name)
    lines = [f"removed host {args.name}", *[f"  {text}" for text in document["notes"]]]
    return Answer(document, "\n".join(lines))


def cmd_host_terminate(args: argparse.Namespace) -> Answer:
    """End a rental now: the provider call, then forget the host here.

    Progress goes to stderr so that `--json` keeps its single document on
    stdout, and the result line is printed whatever happened -- a terminate
    that could not be confirmed raises, and the pod is still billing, which is
    the one outcome nobody may miss.
    """
    settings = load_settings()
    result = teardown.terminate(
        args.name,
        settings,
        registry=open_registry().named(),
        provider=make_provider(settings),
        force=args.force,
        report=note,
    )
    pod = result.target.pod
    if result.terminated:
        cost = f" (was ${pod.cost_usd_hr:.3f}/h)" if pod and pod.cost_usd_hr else ""
        lines = [f"terminated {result.target.label}{cost}"]
    else:
        lines = [f"nothing to terminate: {result.target.label}"]
    if result.forgotten:
        lines.append("  forgotten here; the provider lists it as ended for a while yet")
    lines += [f"  {text}" for text in result.notes]
    return Answer(result.document(), "\n".join(lines))


def cmd_host_list(args: argparse.Namespace) -> Answer:
    read = open_registry()
    registry = read.registry
    lines: list[str] = []
    if not registry.hosts and not read.unreadable:
        lines.append(NO_HOSTS)
    for entry in registry.hosts.values():
        config = entry.config
        summary = summarize(config.gpus, entry.gpu_info) if config.gpus else "no GPUs"
        driver = f", driver {entry.driver_version}" if entry.driver_version else ""
        # One block per host, shaped like `gpuc status`: what the host is, then
        # its cards, then the bootstrap facts. The interpreter path is in
        # `gpuc host list --json` and `gpuc host probe`, not here: it is longer
        # than everything else on the line put together.
        lines.append(
            f"host {entry.name} [{entry.kind}] {entry.ssh or 'this machine'}  "
            f"gpus {len(config.gpus)} ({summary}{driver})"
        )
        stale = shipped_note(entry)
        if stale:
            lines.append(f"  NOTE {stale}")
        for index, name, vram, uuid in gpu_rows(config.gpus, entry.gpu_info):
            lines.append(f"  gpu     [{index}] {name:<28} {vram:<7} {uuid}")
        for index, name, vram, uuid in gpu_rows(config.shared_gpus, entry.gpu_info):
            lines.append(f"  shared  [{index}] {name:<28} {vram:<7} {uuid}")
        # Everything above and here is the cache: what the host said the last
        # time anything on this machine asked it. The host owns all of it, so
        # it is labelled with its age rather than printed as current.
        seen = (
            f"as of {status_mod.format_age(entry.seen_at)}"
            if entry.seen_at
            else f"never read; run gpuc host probe {entry.name}"
        )
        lines.append(f"  pkg     {version_mod.short(config.pkg_commit)} on the host, {seen}")
        if entry.bootstrapped_at:
            lines.append(
                f"  boot    bootstrapped from here {status_mod.format_age(entry.bootstrapped_at)}"
            )
        if entry.root:
            lines.append(f"  root    {entry.root} (gpuc home {entry.remote_home})")
    return registry_answer(read, hosts_document(read), "\n".join(lines) or None)


def cmd_host_bootstrap(args: argparse.Namespace) -> Answer:
    settings = load_settings()
    if args.all:
        if args.name:
            raise UsageError(
                f"host bootstrap takes a host name or --all, not both (got {args.name!r})"
            )
        tally = bootstrap_every_host(settings, health_options(args), report=progress(args))
        return tally.answer(NO_HOSTS if not tally.hosts else f"\n{tally.render()}")
    if not args.name:
        raise UsageError("host bootstrap wants a host name, or --all for every registered host")
    entry = open_registry().require(args.name)
    result = bootstrap_and_record(entry, settings, health_options(args), progress(args))
    return Answer(result.document())


def cmd_clean(args: argparse.Namespace) -> Answer:
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
    entry = open_registry().require(args.host)
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
    return Answer(report.document(), report.render(), failures=list(report.errors))


def cmd_host_clean(args: argparse.Namespace) -> Answer:
    if not args.uv_cache:
        raise UsageError("host clean needs --uv-cache (job workdirs are `gpuc clean --host H`)")
    entry = open_registry().require(args.name)
    report = prune_uv_cache(entry, load_settings())
    return Answer(report.document(), report.render())


def cmd_host_probe(args: argparse.Namespace) -> Answer:
    """Refresh what this machine knows about a host, and print it. Nothing else.

    A probe is the one command that runs before bootstrap, so it is also the
    first chance to learn what the cards are -- every card, not just the
    assigned ones, which is what makes `gpuc host set <name> --gpus 5` nameable
    later. It reads the host's config too, so the offline listings stop being
    stale, but it never writes one: a probe changes nothing about the host.
    """
    settings = load_settings()
    entry = open_registry().require(args.name)
    report = probe_host(entry, settings)
    # Written even when the host had nothing new to say: *when* it was last
    # read is half of what the offline listings report.
    update_cache(
        args.name,
        gpu_info=report.gpu_info or None,
        driver_version=report.driver_version,
        # Only until a bootstrap of our own records the interpreter uv picked:
        # a host somebody else set up is worth being able to read before then.
        python=None if entry.python else report.host_python,
        config=_probe_config(entry, settings),
    )
    return Answer(report.document(), report.render(all_gpus=args.all_gpus))


def _probe_config(entry: HostEntry, settings: Settings) -> dict[str, Any] | None:
    """The host's `config.json`, or None if it has none or could not be read."""
    try:
        transport = transport_for(entry, settings)
        return read_config(transport, resolve_home(transport, entry)).document
    except (ConfigError, RemoteError, TransportError):
        return None


CLOUDS: dict[str, list[Cloud]] = {
    "secure": ["SECURE"],
    "community": ["COMMUNITY"],
    "any": ["SECURE", "COMMUNITY"],
}


def rental_options(args: argparse.Namespace) -> RentalOptions | None:
    """`--runpod` and its flags, or None when the job names a host instead."""
    if not args.runpod:
        return None
    options = RentalOptions(
        gpu_names=_comma_list(args.gpu),
        min_vram_gb=args.min_vram,
        max_price_usd_hr=args.max_price,
        clouds=CLOUDS[args.cloud],
        cuda_min=args.cuda_min,
        gpu_count=args.gpu_count,
        reuse=not args.no_reuse,
        name_hint=args.name_hint,
        disk_gb=args.disk,
        image=args.image,
        health=health_options(args),
    )
    if args.idle_min is not None:
        options.idle_minutes = args.idle_min
    return options


def cmd_config_init(args: argparse.Namespace) -> Answer:
    document = init_config(force=args.force)
    return Answer(
        document,
        f"wrote {document['config_file']}\nEvery key is commented with its default; edit what "
        f"you need.",
    )


def cmd_config_show(args: argparse.Namespace) -> Answer:
    settings = load_settings()
    document = config_document(settings)
    path = config_file()
    lines = [
        f"config file: {path}{'' if path.exists() else ' (does not exist; using defaults)'}",
        f"state dir:   {state_dir()}",
    ]
    lines += [f"  {name} = {value!r}" for name, value in settings.model_dump().items()]
    lines += [f"  note: {text}" for text in document["notes"]]
    return Answer(document, "\n".join(lines))


def health_options(args: argparse.Namespace) -> HealthOptions:
    """`--health-args`, judged here so a bad flag is a usage error and not a
    health check that fails on the host."""
    try:
        return HealthOptions.parse(args.health_args)
    except ValueError as exc:
        raise UsageError(str(exc)) from exc


def progress(args: argparse.Namespace) -> Reporter:
    """Where a step's progress goes: stdout, or stderr when stdout is a document."""
    return jsonout.note if getattr(args, "json", False) else print


def cmd_submit(args: argparse.Namespace) -> Answer:
    settings = load_settings()
    # None, not False, for a flag that was not passed: a spec that says
    # `use_shared: true` keeps saying it when nobody typed --use-shared.
    overrides = {"use_shared": True if args.use_shared else None}
    result = submit_job(
        args.job_file,
        settings,
        host=args.host,
        rental=rental_options(args),
        overrides=overrides,
        workdir=Path.cwd(),
        use_git=not args.no_git,
        bootstrap=not args.no_bootstrap,
        report=progress(args),
    )
    return _queued(result)


def _queued(result: SubmitResult) -> Answer:
    """The last word of `submit` and `requeue`, in whichever form was asked for."""
    return Answer(
        result.document(),
        result.render(status_mod.queue_note(result.placement), workdir=Path.cwd()),
    )


def cmd_status(args: argparse.Namespace) -> Answer:
    """Report on every host, and exit non-zero if any of them could not be read.

    Every host that answered is printed either way: a box that is down is the
    moment the others matter most. Exit 3 is the one that means *unknown* --
    the local registry could not be read, so "no jobs running" would be a
    guess, and automation must never take it for idle.
    """
    settings = load_settings()
    read = open_registry()
    try:
        since_s = status_mod.parse_duration(args.since) if args.since else None
    except ValueError as exc:
        raise UsageError(f"--since: {exc}") from exc
    # Each host as it answers, so a slow one does not hold the others' blocks.
    show = (
        None
        if args.json
        else lambda v: print(status_mod.render(v, recent=args.recent, since_s=since_s))
    )
    result = status(read, settings, host=args.host, all_jobs=args.all, on_view=show)
    forget_gone_rentals(result.views)
    lines: list[str] = []
    if read.unreadable:
        note(
            f"cannot read {hosts_file()}, so no host status is known "
            f"(this is not `no jobs running`)"
        )
    elif not result.views:
        lines.append(NO_HOSTS)
    if result.index_error:
        note(f"{result.index_error}; the list of index-only jobs may be short")
    unhosted = result.unhosted_text(args.host)
    if unhosted:
        lines.append(unhosted)
    return result.answer(recent=args.recent, since_s=since_s, text="\n".join(lines) or None)


def ssh_target(args: argparse.Namespace) -> tuple[HostEntry, str, str | None]:
    """`(host, directory, fallback)` for a host name or a job id.

    A registered host name wins over a job id: host names are ours and job ids
    are timestamps, so they cannot collide, and looking the name up locally
    keeps `gpuc ssh <host>` from asking every host whether it knows a job.
    """
    registry = open_registry().named()
    entry = registry.hosts.get(args.target)
    if entry is not None:
        return entry, entry.remote_home, None
    entry = locate(args.target, registry, args.host).require_entry()
    job_dir = f"{entry.remote_home}/jobs/{args.target}"
    return entry, f"{job_dir}/workdir", job_dir


def cmd_ssh(args: argparse.Namespace) -> Answer:
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
    if command:
        # Joined with spaces and handed to a shell, which is what `ssh host CMD`
        # has always done and what anyone typing `-- 'ls | wc -l'` expects.
        # shlex.join would quote the pipe back into a filename.
        joined = " ".join(command)
        if args.print_only:
            argv = ssh_mod.command_argv(transport, directory, joined, fallback)
            return Answer({}, ssh_mod.print_line(argv))
        result = ssh_mod.run_command(transport, directory, joined, fallback)
        sys.stdout.write(result.stdout)
        sys.stderr.write(result.stderr)
        return Answer({}, outcome=result.returncode)
    if args.print_only:
        return Answer(
            {}, ssh_mod.print_line(ssh_mod.interactive_argv(transport, directory, fallback))
        )
    print(f"# {entry.name}:{directory}", file=sys.stderr)
    argv = ssh_mod.interactive_argv(transport, directory, fallback)
    os.execvp(argv[0], argv)


def _job_answer(document: dict[str, Any], *text: str | None) -> Answer:
    """A job command's last word: its warnings on stderr, then the document or the text."""
    for warning in document.get("warnings", []):
        warn(warning)
    return Answer(document, "\n".join(line for line in text if line))


def cmd_cancel(args: argparse.Namespace) -> Answer:
    document = cancel_job(args.job_id, args.host, load_settings())
    return _job_answer(
        document, f"job {args.job_id} on host {document['host']}: {document['status']}"
    )


def cmd_reorder(args: argparse.Namespace) -> Answer:
    document = reorder_job(args.job_id, args.priority, args.host, load_settings())
    return _job_answer(
        document,
        f"job {args.job_id} on host {document['host']} moved to priority {args.priority}",
        status_mod.queue_note(document),
    )


def cmd_preempt(args: argparse.Namespace) -> Answer:
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
    return _job_answer(
        document, f"job {args.job_id} on host {document['host']}: {document['status']}{at}"
    )


def cmd_estimate(args: argparse.Namespace) -> Answer:
    """Add, change or clear a job's `estimated_runtime_min` after submitting it.

    It is the one spec field somebody else needs and only the submitter knows,
    and the job that most needs one is the long job already running when the
    next person arrives -- which is too late to edit a file before `submit`.
    """
    wanted = check_estimate(args.minutes, clear=args.clear)
    document = estimate_job(args.job_id, wanted, args.host, load_settings())
    recorded = document["estimated_runtime_min"]
    job = f"job {args.job_id} on host {document['host']}"
    return _job_answer(
        document,
        f"{job} no longer estimates a runtime"
        if recorded is None
        else f"{job} now estimates {recorded:g} min",
    )


def _follow_argv(transport: Transport, remote_path: str, lines: int) -> list[str]:
    # `-F`, not `-f`: `gpuc submit && gpuc logs -f` is the obvious pair to type,
    # and a job that has not been dispatched yet has no log.txt to open.
    return transport.argv(tail_command(remote_path, lines, follow=True, retry=True))


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


def cmd_logs(args: argparse.Namespace) -> Answer:
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
        location = locate(args.job_id, open_registry().named(), args.host, settings)
        session = location.session or open_session(location.require_entry(), settings)
        return _follow_forever(session, args.job_id, args.lines)
    if args.follow:
        return _follow_until_done(args, settings)
    host, log = read_log(args.job_id, args.host, args.lines, settings)
    # Bytes for a human; lines plus where they came from for a script.
    return log.answer(args.job_id, host)


def _follow_forever(session: HostSession, job_id: str, lines: int) -> Answer:
    """`--follow-forever`: the stream with no end, and no claim about the job."""
    remote = f"{session.job_dir(job_id)}/log.txt"
    return Answer({}, outcome=subprocess.call(_follow_argv(session.transport, remote, lines)))


def _follow_until_done(args: argparse.Namespace, settings: Settings) -> Answer:
    """`logs -f`: the log while the job runs, its outcome, and exit with it.

    The stream and the polling are two things at once because there is nothing
    in a log that says a job has ended -- the host's state file is the only
    thing that does. So `tail` runs as a child writing straight to our stdout
    while this thread asks the host, and the job's own outcome becomes the exit
    code, which is what makes `gpuc logs -f "$id"` a foreground wait on its own.

    The stream is started on the first poll the host answers, not before: a
    host in trouble is the wait's business (`Watch` retries it and reads the
    mirror once it is gone), and opening a session to it here would turn an
    ssh blip into exit 1 before the wait had its say.
    """
    watched: wait_mod.Watched | None = None
    ended = False
    tail: subprocess.Popen[bytes] | None = None
    reported_stream_end = False
    # Everything is inside, not just the polling: finding the job's host can
    # ask every registered host in turn, 60s each, which is exactly where
    # somebody who mistyped an id reaches for Ctrl-C.
    try:
        watch = wait_mod.start([args.job_id], args.host, settings)
        watched = watch.jobs[args.job_id]
        watch.poll()
        watch.check_known()
        if watched.settled:
            # Nothing more is coming, and `tail -f` on it would simply hang.
            # The ordinary read, so a purged job falls back to the S3 mirror.
            _, log = read_log(args.job_id, watched.host, args.lines, settings)
            sys.stdout.write(log.text)
            return wait_mod.answer([watched], watched.line())

        def each_round() -> None:
            """Start the stream once the host answers; after that, say so once
            if it died, rather than freeze the log in silence.

            An ssh whose keepalives ran out takes the tail with it. The wait
            itself is fine -- it polls over its own connection -- so this is a
            note and not an ending.
            """
            nonlocal tail, reported_stream_end
            assert watched is not None
            if tail is None:
                if watched.settled or watch.troubled(watched.host):
                    return
                if watched.status == "queued":
                    # Said before tail is started, because tail then says
                    # `cannot open ... No such file or directory` about a log
                    # the host has not opened yet, and alone that reads like a
                    # failure not a queue.
                    note(f"job {args.job_id} is queued; following the log from when it starts")
                # The watch's own session, rather than a second one: opening
                # another costs a round trip to resolve the same home.
                session = watch.session(watched.host)
                remote = f"{session.job_dir(args.job_id)}/log.txt"
                tail = subprocess.Popen(_follow_argv(session.transport, remote, args.lines))
                return
            if reported_stream_end or tail.poll() is None:
                return
            reported_stream_end = True
            note("the log stream ended before the job did; still waiting for the job")

        try:
            each_round()
            wait_mod.block(watch, interval=args.interval, each_round=each_round)
            ended = True
        finally:
            # Only a job that ended has last lines worth waiting for; an
            # interrupt wants the stream gone now.
            if tail is not None:
                _end_tail(tail, flush=ended)
        if tail is None:
            # The host never answered and the job settled from the mirror:
            # its log is there too, or nowhere.
            _, log = read_log(args.job_id, watched.host, args.lines, settings)
            sys.stdout.write(log.text)
    except KeyboardInterrupt:
        # Ctrl-C reached the tail too: it shares this process group. The job
        # does not care either way -- its host owns it, not us. A second one,
        # landing in the flush above, arrives here with the job already ended,
        # and then its outcome is still the answer.
        if not ended:
            where = f"is still {watched.status} on {watched.host}" if watched else "was not reached"
            raise Interrupted(f"interrupted; job {args.job_id} {where}") from None
    if watched is None:
        raise CliError(f"job {args.job_id} was never looked up")
    return wait_mod.answer([watched], watched.line())


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


def cmd_wait(args: argparse.Namespace) -> Answer:
    """Block until every named job has ended, then exit with their outcome.

    No log: this is the half of `logs -f` a sweep wants, where twenty jobs'
    output interleaved would be unreadable and only the verdicts matter.
    """
    check_interval(args, polls=True)
    watch: wait_mod.Watch | None = None
    # Under --json stdout belongs to the document, so the outcomes go to stderr
    # as they happen and are in the document at the end.
    announce = progress(args)
    try:
        # `start` is inside too: with no --host and nothing in the index it asks
        # every registered host in turn, 60s each, and that is the wait a user
        # interrupts most.
        watch = wait_mod.start(args.job_ids, args.host, load_settings())
        waited = wait_mod.block(
            watch, interval=args.interval, on_settled=lambda watched: announce(watched.line())
        )
    except KeyboardInterrupt:
        # This command exists to be abandoned, so a Ctrl-C is an ordinary way
        # for it to end -- and naming what is still out there is the whole
        # value of saying anything at all.
        pending = [job.job_id for job in watch.pending] if watch else list(args.job_ids)
        raise Interrupted(f"interrupted; still on their hosts: {', '.join(pending)}") from None
    return wait_mod.answer(waited)


def cmd_requeue(args: argparse.Namespace) -> Answer:
    result = requeue_job(
        args.job_id,
        load_settings(),
        host=args.host,
        rental=rental_options(args),
        workdir=Path.cwd(),
        use_git=not args.no_git,
        bootstrap=not args.no_bootstrap,
        report=progress(args),
    )
    return _queued(result)


def cmd_pods(args: argparse.Namespace) -> Answer:
    settings = load_settings()
    view = pods_mod.gather(settings, make_provider(settings), heartbeats=not args.no_heartbeat)
    return Answer(view.document(), pods_mod.render(view))


def cmd_version(args: argparse.Namespace) -> Answer:
    """What is installed here, and what each host was running when last read.

    The host commits come from the registry's cache -- no ssh, so this stays a
    command you can run before anything else. That also means it cannot see a
    host somebody else has bootstrapped since it was read: `gpuc status` asks
    each host what it is running.
    """
    read = open_registry()
    document = version_document(read)
    dirty = " (+uncommitted changes)" if document["dirty"] else ""
    lines = [
        f"gpuc {document['version']}",
        f"commit {version_mod.short(document['commit'])} [{document['source']}]{dirty}",
        f"python {document['python']} at {document['executable']}",
    ]
    hosts = document["hosts"]
    if not hosts:
        lines.append("hosts: none read yet")
    else:
        lines.append("hosts (as last read from here):")
        for host in hosts:
            differs = "" if host["current"] else "  DIFFERS: re-bootstrap"
            seen = f"  {status_mod.format_age(host['seen_at'])}" if host["seen_at"] else ""
            lines.append(
                f"  {host['name']:<16} pkg {version_mod.short(host['pkg_commit'])}{seen}{differs}"
            )
        if not all(host["current"] for host in hosts):
            lines.append(
                "upgrade a host with: gpuc host bootstrap <host> (or --all for every host; "
                "running jobs are not disturbed)"
            )
    return registry_answer(read, document, "\n".join(lines))


def cmd_web_set_password(args: argparse.Namespace) -> Answer:
    if args.stdin:
        password = sys.stdin.readline().rstrip("\r\n")
    else:
        password = getpass.getpass("dashboard password: ")
        if password != getpass.getpass("again: "):
            raise UsageError("the two passwords differ; nothing was written")
    path = web_mod.write_password(password)
    return Answer({}, f"wrote {path} (0600)\nserve the dashboard with: gpuc web serve")


def cmd_web_serve(args: argparse.Namespace) -> Answer:
    if args.install:
        web_mod.install_service(args.bind, args.port)
        return Answer({})
    server = web_mod.make_server(args.bind, args.port)
    note(f"gpuc dashboard on http://{args.bind}:{server.server_port}/ (Ctrl-C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        note("stopped")
    finally:
        server.server_close()
    return Answer({})


def cmd_skill(args: argparse.Namespace) -> Answer:
    """Print the agent guide, or drop a copy into a project.

    Printing is the point: an agent can pipe `gpuc skill` into its own context
    without being told where the file lives, or which checkout it is in.
    """
    if args.install is None:
        return Answer({}, read_skill())
    return Answer({}, f"wrote {install_skill(Path(args.install), force=args.force)}")


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
        help="how long a rental may sit with an empty queue before it terminates itself "
        "(the host's own default is 15); ignored for hosts that are not rented",
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
        help="idle minutes before a rental terminates itself",
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

    terminate = host.add_parser(
        "terminate",
        help="end a rented pod now and forget it here (the pod stops billing)",
    )
    terminate.add_argument(
        "name",
        help="a registered rental, or a pod id or pod name from `gpuc pods`",
    )
    terminate.add_argument(
        "--force",
        action="store_true",
        help="terminate without asking the host whether it is idle: kills whatever is "
        "running and loses anything not yet uploaded",
    )
    add_json_flag(
        terminate, "what was ended, what it was running, and whether the pod is confirmed gone"
    )
    terminate.set_defaults(func=cmd_host_terminate)

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
        help="with --purge: list the mirrored logs in S3 yourself and purge only jobs "
        "that have one, on top of the host's own record",
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
        help="stream the log and never stop: for watching a host's own "
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
        "--json`, and what it can do is `gpuc cancel`, `gpuc preempt`, `gpuc reorder`, "
        "`gpuc estimate` and `gpuc host remove`.",
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
    parser.add_argument(
        "--cuda-min",
        default=DEFAULT_CUDA_MIN,
        help=f"the CUDA floor the catalog is asked for and the pod is created with "
        f"(default {DEFAULT_CUDA_MIN})",
    )
    parser.add_argument(
        "--idle-min",
        type=float,
        default=None,
        metavar="MINUTES",
        help=f"terminate the pod once its queue has been empty this long "
        f"(default {DEFAULT_IDLE_MINUTES:g})",
    )
    parser.add_argument("--disk", type=int, help="container disk in GB; default from config")
    parser.add_argument("--image", help="pod image; default from config")
    parser.add_argument("--no-reuse", action="store_true", help="always create a new pod")
    parser.add_argument("--name-hint", default="job", help="goes into the pod name")
    parser.add_argument("--health-args", default="", help="extra flags for `gpuc.host health`")


def wants_runpod(args: argparse.Namespace) -> bool:
    if getattr(args, "pod", None):
        return True  # `gpuc host add --pod` asks the provider where that pod is
    if args.command == "host" and args.host_command == "terminate":
        return True
    return bool(getattr(args, "runpod", False)) or args.command == "pods"


def first_run_note() -> None:
    """One line, once, on stderr: defaults are fine, but say where to change them."""
    if not config_file().exists():
        print(
            f"note: no config file at {config_file()} (using defaults, no S3 mirror); "
            f"run `gpuc config init` to create one",
            file=sys.stderr,
        )


def failed(
    args: argparse.Namespace, message: str, exit_code: int, document: dict[str, Any] | None = None
) -> int:
    """One exit for every failure: the message on stderr, and under `--json` a
    document on stdout saying the same thing, so a caller parsing stdout is
    never handed half an answer or nothing at all. `document` is what a
    command interrupted midway still has to show, beside the error."""
    print(f"error: {message}", file=sys.stderr)
    if getattr(args, "json", False):
        jsonout.emit_error(message, exit_code, **(document or {}))
    return exit_code


def emit(args: argparse.Namespace, answer: Answer) -> int:
    """The one place a command's answer reaches stdout: the document under
    `--json`, the text otherwise, and the exit code the answer says."""
    if getattr(args, "json", False):
        jsonout.emit(answer.document)
    elif answer.text:
        sys.stdout.write(answer.text if answer.text.endswith("\n") else answer.text + "\n")
    return answer.exit_code


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
            "`gpuc host add --pod`, `gpuc host terminate` or `gpuc pods`",
            EXIT_ERROR,
        )
    if args.command not in ("config", "skill"):
        first_run_note()
    try:
        answer = args.func(args)
    except KeyboardInterrupt:
        # Every Ctrl-C out of a blocking command lands here, so none of them can
        # return 0 by accident or leave `--json` with an empty stdout -- the two
        # ways `gpuc wait` and `gpuc logs -f` each got this wrong when they
        # handled it themselves. A command with something specific to say about
        # what was in flight raises `Interrupted` instead, which is the line
        # below.
        return failed(args, "interrupted", EXIT_INTERRUPTED)
    except Interrupted as exc:
        return failed(args, str(exc), EXIT_INTERRUPTED, exc.document)
    except Exception as exc:
        code = exit_code_for(exc)
        if code is None:
            raise
        return failed(args, failure_message(exc), code)
    return emit(args, answer)


if __name__ == "__main__":
    raise SystemExit(main())
