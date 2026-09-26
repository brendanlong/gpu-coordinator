"""`python -m gpuc.host <cmd>` — the on-host CLI the control side drives.

Output is JSON on stdout so the control side never has to parse prose.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gpuc.host import (
    cleanup,
    dispatcher,
    fetch,
    gpus,
    health,
    jobs,
    paths,
    plan,
    queue,
    runner,
    storage,
)
from gpuc.host import settable as settable_mod
from gpuc.host.jobs import JobSpec, JobState


def _read_json_object(source: str, what: str) -> dict[str, Any]:
    text = sys.stdin.read() if source == "-" else Path(source).read_text()
    document = json.loads(text)
    if not isinstance(document, dict):
        raise SystemExit(f"{what} must be a JSON object")
    return document


def cmd_enqueue(args: argparse.Namespace) -> int:
    document = _read_json_object(args.spec, "spec")
    spec = JobSpec.from_dict(document)
    queue.enqueue(spec)
    pid = dispatcher.spawn_detached_dispatcher()
    print(json.dumps({"job_id": spec.job_id, "dispatcher_pid": pid}))
    return 0


def _state_or_none(job_id: str) -> JobState | None:
    try:
        return jobs.read_state(job_id)
    except RuntimeError:
        return None


def _spec(job_id: str) -> JobSpec | None:
    try:
        return jobs.read_spec(job_id)
    except (RuntimeError, ValueError):
        return None


def _gpu_table(config: jobs.HostConfig) -> dict[str, Any]:
    """What `config.gpus` and `config.shared_gpus` resolve to on this host now.

    The control side cannot work this out: both may name cards by index, and
    only the host knows what its driver is calling them today. The shared cards
    carry their current memory and utilization too, because whether one is
    borrowable is a fact about this second that only nvidia-smi here can answer
    -- and when it is not, the numbers are how somebody sees why. The same
    `gpus.resolve` the dispatcher decides with, over one reading; a driver
    that will not answer reads as every entry unavailable, as it does there.
    """
    try:
        table = gpus.list_gpus()
    except gpus.GpuError:
        table = []
    indices = {gpu.uuid: gpu.index for gpu in table}
    cards = gpus.resolve(config.gpus, table, config.shared_gpus)
    # Through the same reader the dispatcher borrows on, so an absent entry
    # means here exactly what it means there: not a card we would take.
    usage, _failure = gpus.usage_or_nothing(cards.shared)
    return {
        "gpus": config.gpus,
        "gpus_resolved": [{"index": indices.get(uuid), "uuid": uuid} for uuid in cards.owned],
        "gpus_unavailable": cards.missing,
        "shared_gpus": config.shared_gpus,
        "shared_gpus_resolved": [
            {
                "index": indices.get(uuid),
                "uuid": uuid,
                "memory_mib": usage[uuid].memory_mib if uuid in usage else None,
                "utilization_pct": usage[uuid].utilization_pct if uuid in usage else None,
                "unused": uuid in usage and usage[uuid].unused,
            }
            for uuid in cards.shared
        ],
        "shared_gpus_unavailable": cards.shared_missing,
        "shared_configured": len(cards.shared) + len(cards.shared_missing),
    }


WANDB_HINTS = {"WANDB_ENTITY": "entity", "WANDB_PROJECT": "project", "WANDB_RUN_ID": "run_id"}


def wandb_hints(env: dict[str, str]) -> dict[str, str]:
    return {name: env[key] for key, name in WANDB_HINTS.items() if env.get(key)}


def _seconds_until(stamp: str | None) -> float | None:
    if not stamp:
        return None
    try:
        when = datetime.fromisoformat(stamp)
    except ValueError:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return max(0.0, (when - datetime.now(UTC)).total_seconds())


def projected_starts(
    config: jobs.HostConfig, table: dict[str, Any], states: dict[str, JobState]
) -> plan.Projection:
    """When each queued job is expected to start, from what this host knows:
    its cards, who holds them and until when, and the queue in order.

    Here rather than on the control side because every input is the host's:
    the resolved cards, the one nvidia-smi reading of the shared ones, the
    running jobs' etas and the queue. A client only renders the answer.
    """
    running = {job_id: s for job_id, s in states.items() if s.status == "running"}
    holder = {uuid: s for s in running.values() for uuid in s.gpus}

    def release(uuid: str) -> float | None:
        state = holder.get(uuid)
        return 0.0 if state is None else _seconds_until(state.eta)

    cards = [plan.Card(row["uuid"], False, release(row["uuid"])) for row in table["gpus_resolved"]]
    theirs = 0
    for row in table["shared_gpus_resolved"]:
        if row["uuid"] in holder or row["unused"]:
            cards.append(plan.Card(row["uuid"], True, release(row["uuid"])))
        else:
            theirs += 1
    # The queue from the same snapshot as everything else in this document:
    # listed separately, a job the dispatcher claimed in between would be
    # reported queued with no start time and no reason.
    queued = sorted(
        queue.QueueEntry(state.priority, job_id)
        for job_id, state in states.items()
        if state.status == "queued"
    )
    requests: list[plan.Request] = []
    for entry in queued:
        spec = _spec(entry.job_id)
        state = states[entry.job_id]
        if spec is None:
            # A spec the dispatcher is about to fail: the next call will say.
            continue
        estimate = state.estimated_runtime_min
        requests.append(
            plan.Request(
                entry.job_id,
                spec.gpus,
                config.may_borrow(spec),
                None if estimate is None else estimate * 60.0,
            )
        )
    return plan.project(
        requests,
        cards,
        owned_configured=len(config.gpus),
        owned_missing=table["gpus_unavailable"],
        shared_configured=table["shared_configured"],
        theirs=theirs,
        draining=paths.draining_file().exists(),
    )


def _ended_within(ended_at: str | None, since_s: float, now: datetime) -> bool:
    if not ended_at:
        return False
    try:
        ended = datetime.fromisoformat(ended_at)
    except ValueError:
        return False
    if ended.tzinfo is None:
        ended = ended.replace(tzinfo=UTC)
    return (now - ended).total_seconds() <= since_s


def in_window(states: dict[str, JobState], recent: int | None, since_s: float | None) -> list[str]:
    """The jobs a `status` that names none reports: every job that is not
    finished, the newest `recent` finished ones that ended within `since_s`,
    and every finished job still holding its workdir.

    Trimmed here, not by the client, so that what a status costs grows with
    what is on the host now rather than with everything it has ever run: a
    finished job's spec, its outputs check and its workdir size are the part
    that grows, and a long-lived box nobody purges would otherwise get slower
    with every job. A finished job with its workdir is sent whatever its age
    because the host-level lines are about exactly those -- the disk they
    hold and the outputs that are only here, which `host terminate` refuses
    over -- and without a workdir neither can apply. No window sends every job.
    """
    now = datetime.now(UTC)
    finished = sorted(
        ((state.ended_at or "", job_id) for job_id, state in states.items() if state.finished),
        reverse=True,
    )
    shown = [
        job_id
        for ended_at, job_id in finished
        if since_s is None or _ended_within(ended_at, since_s, now)
    ]
    kept = set(shown if recent is None else shown[: max(recent, 0)])
    return [
        job_id
        for job_id, state in states.items()
        if not state.finished or job_id in kept or paths.workdir(job_id).is_dir()
    ]


def _job_entry(
    job_id: str, state: JobState, projection: plan.Projection, measuring_until: float
) -> dict[str, Any]:
    spec = _spec(job_id)
    entry = {"job_id": job_id, "name": spec.name if spec else "", **state.to_dict()}
    entry["max_runtime_min"] = state.max_runtime(spec)
    entry.pop("live_max_runtime")
    # When a queued job's turn is expected, and why not if it cannot be
    # said. Null on anything that is not queued.
    entry["starts_in_s"] = projection.starts_in_s.get(job_id)
    entry["starts_unknown"] = projection.unknown.get(job_id)
    # Whether this job gives its cards up to anything more important. It
    # changes what "running" promises, and only the spec knows.
    entry["auto_preempt"] = spec.auto_preempt if spec else None
    # How many cards this job asked for. A queued job holds none, so its
    # `gpus` is empty and nothing else says whether it is waiting for one
    # card or for eight.
    entry["gpus_requested"] = spec.gpus if spec else None
    # Whether this job may be dispatched to a shared card, which is half of
    # why a queued job asking for more cards than the host owns is waiting
    # rather than already failed.
    entry["use_shared"] = spec.use_shared if spec else None
    # Where the results went, for anything that wants to link to them. The
    # W&B keys are the three that name a run; the job's env is otherwise
    # its own business and never leaves the host.
    entry["outputs"] = [asdict(o) for o in spec.outputs] if spec else []
    entry["wandb"] = wandb_hints(spec.env) if spec else {}
    # Which job this one was resubmitted from; the host only carries it.
    entry["requeued_from"] = spec.requeued_from if spec else None
    # Null for a job that is not over -- a running job's workdir is being
    # written to, so any size for it would be a lie -- and for a finished
    # one only when the call's measuring budget is spent. Otherwise free
    # for the workdirs that are already gone, and read from `state.json`
    # for the rest.
    entry["workdir_bytes"] = (
        cleanup.reported_workdir_bytes(job_id, state, deadline=measuring_until)
        if state.finished
        else None
    )
    # "this job produced something that is still only here": the control
    # side cannot work it out, since it never sees the spec's `outputs:`.
    # A running job's files are its runner's; a queued one may be holding
    # what a preempted attempt produced.
    entry["outputs_pending"] = bool(
        spec is not None
        and state.status != "running"
        and cleanup.outputs_pending(job_id, spec, state)
    )
    # The same question for outputs with no destination, whose answer is not
    # a warning: they are where the spec said to keep them.
    entry["kept_outputs"] = (
        cleanup.kept_outputs(job_id, spec, state) if spec is not None and state.finished else []
    )
    entry["kept_bytes"] = state.kept_bytes
    return entry


def cmd_status(args: argparse.Namespace) -> int:
    """The host, and the jobs asked about: exactly the ids named, those this
    host has, or with none named the ones `in_window` picks."""
    config = jobs.read_config()
    measuring_until = time.monotonic() + cleanup.MEASURING_BUDGET_S
    states: dict[str, JobState] = {}
    for job_id in jobs.list_job_ids():
        try:
            states[job_id] = jobs.read_state(job_id)
        except (RuntimeError, FileNotFoundError):
            continue
    table = _gpu_table(config)
    projection = projected_starts(config, table, states)
    selected = (
        [job_id for job_id in dict.fromkeys(args.job_ids) if job_id in states]
        if args.job_ids
        else in_window(states, args.recent, args.since)
    )
    heartbeat = dispatcher.heartbeat_age()
    print(
        json.dumps(
            {
                "host": config.host,
                # The commit whoever bootstrapped this host last shipped. The
                # control side cannot infer it: its own registry only records
                # what *this* machine shipped, and a second control machine --
                # a laptop against the same box -- leaves that record
                # describing a host it no longer matches.
                "pkg_commit": config.pkg_commit,
                # And the commit the dispatcher *now running* was started on,
                # which is not the same question: a dispatcher imports its code
                # once and keeps serving the queue from it however many times
                # the package underneath is replaced.
                "dispatcher_pkg_commit": dispatcher.holder_pkg_commit(),
                **table,
                "ephemeral": config.ephemeral,
                "draining": paths.draining_file().exists(),
                "dispatcher_heartbeat_age_s": None if heartbeat is None else round(heartbeat, 1),
                "queue": [
                    {"priority": state.priority, "job_id": job_id}
                    for job_id, state in sorted(
                        states.items(), key=lambda item: (item[1].priority, item[0])
                    )
                    if state.status == "queued"
                ],
                # However many `jobs` leaves out, so "none in the last hour"
                # can still say how many older ones there are.
                "finished_count": sum(1 for state in states.values() if state.finished),
                "jobs": [
                    _job_entry(job_id, states[job_id], projection, measuring_until)
                    for job_id in selected
                ],
            },
            indent=2,
        )
    )
    return 0


def _no_such_job(job_id: str, why: str) -> dict[str, Any]:
    """The answer every job verb gives for an id this host does not have.

    `missing` is what tells the control side apart "no such job" (its exit 4)
    from a refusal of a job that is here (exit 1): both are an `error`
    document, and the words alone are not something to parse.
    """
    return {"job_id": job_id, "error": why, "missing": True}


def _answer(answers: list[dict[str, Any]], **extra: Any) -> int:
    """Every job verb's output: `{"jobs": [...]}`, one document per id in the
    order asked, each what was done or `{job_id, error}`; exit 1 if any job
    was refused. One refusal never stops the others being done."""
    print(json.dumps({"jobs": answers, **extra}))
    return 1 if any("error" in answer for answer in answers) else 0


def _cancel(job_id: str) -> dict[str, Any]:
    try:
        status = queue.cancel(job_id)
    except FileNotFoundError as exc:
        return _no_such_job(job_id, str(exc))
    except (OSError, RuntimeError) as exc:
        return {"job_id": job_id, "error": str(exc)}
    return {"job_id": job_id, "status": status}


def cmd_cancel(args: argparse.Namespace) -> int:
    return _answer([_cancel(job_id) for job_id in dict.fromkeys(args.job_ids)])


def _preempt(job_id: str, priority: int | None) -> dict[str, Any]:
    """The error document rather than a traceback, like `set`: which job
    this host will not preempt, and why, is the whole answer the control side
    needs."""
    try:
        status = queue.preempt(job_id, priority)
    except FileNotFoundError as exc:
        return _no_such_job(job_id, str(exc))
    except (OSError, ValueError, RuntimeError) as exc:
        return {"job_id": job_id, "error": str(exc)}
    state = _state_or_none(job_id)
    # What it will be queued at once its runner stops, whether or not this
    # call changed it.
    return {"job_id": job_id, "status": status, "priority": state.priority if state else None}


def cmd_preempt(args: argparse.Namespace) -> int:
    """Stop running jobs and queue them again, to run from the start."""
    answers = [_preempt(job_id, args.priority) for job_id in dict.fromkeys(args.job_ids)]
    # The runner queues the job again; the dispatcher is what launches the
    # next attempt, so make sure there is one.
    started = any("error" not in answer for answer in answers)
    return _answer(
        answers, dispatcher_pid=dispatcher.spawn_detached_dispatcher() if started else None
    )


def _fetch_list(job_id: str, wanted: list[str]) -> dict[str, Any]:
    if not paths.job_dir(job_id).is_dir():
        return _no_such_job(job_id, f"no such job: {job_id}")
    try:
        state = jobs.read_state(job_id)
        spec = jobs.read_spec(job_id)
    except (OSError, RuntimeError) as exc:
        return {"job_id": job_id, "error": str(exc)}
    try:
        found = fetch.listing(job_id, spec, wanted)
    except fetch.NotFetchable as exc:
        return {"job_id": job_id, "error": str(exc)}
    return {"job_id": job_id, "status": state.status, **found}


def cmd_fetch_list(args: argparse.Namespace) -> int:
    """What `gpuc fetch` copies from each job; changes nothing."""
    return _answer([_fetch_list(job_id, args.path) for job_id in dict.fromkeys(args.job_ids)])


def _patch(fields: Sequence[str]) -> dict[str, Any]:
    """`--field name=<json>` pairs as a patch; a JSON `null` clears a field."""
    patch: dict[str, Any] = {}
    for item in fields:
        name, sep, raw = item.partition("=")
        if not sep:
            raise SystemExit(f"--field wants name=<json value>, got {item!r}")
        try:
            value = json.loads(raw)
        except ValueError as exc:
            raise SystemExit(f"--field {name}: {raw!r} is not JSON: {exc}") from exc
        settable = settable_mod.BY_FIELD.get(name)
        if settable is not None and settable.kind is int and isinstance(value, float):
            value = int(value) if value == int(value) else value
        patch[name] = value
    return patch


def cmd_set(args: argparse.Namespace) -> int:
    """Change queued or running jobs' settings: see `settable.apply`.

    A patch no job could take is one refusal per job rather than an argparse
    error, so the control side reads the reason like any other."""
    patch = _patch(args.field)
    error = settable_mod.patch_error(patch)
    ids = dict.fromkeys(args.job_ids)
    if error is not None:
        return _answer([{"job_id": job_id, "error": error} for job_id in ids])
    return _answer([settable_mod.apply(job_id, patch) for job_id in ids])


def cmd_run(args: argparse.Namespace) -> int:
    return runner.run_job(
        args.job_id, [uuid for uuid in args.gpus.split(",") if uuid], args.attempt
    )


def _selection(value: str | None) -> list[str] | None:
    """A `--only`-style comma-separated list of job ids.

    `--only ''` means "none of them", which is not the same as not passing it
    at all: the control side's `--verify` sends exactly that when no
    candidate's mirror could be confirmed.
    """
    return None if value is None else [j for j in (p.strip() for p in value.split(",")) if j]


def _refusal(dry_run: bool, *selections: list[str] | None) -> cleanup.CleanResult | None:
    """A report saying nothing was deleted, if any named job id is a typo.

    Checked before anything goes, and it refuses the whole selection rather
    than honouring the half it recognises: `--only a,b` with one typo is far
    more likely to be a mistyped id than a deliberate pair, and the half this
    would delete is not recoverable. Nothing else here half-honours a delete
    either.
    """
    named = {job_id for selection in selections if selection for job_id in selection}
    unknown = sorted(named - set(jobs.list_job_ids()))
    if not unknown:
        return None
    return cleanup.CleanResult(
        dry_run=dry_run,
        s3_prefix=cleanup.host_s3_prefix(),
        errors=[f"{job_id}: no job with that id on this host" for job_id in unknown]
        + ["refused the whole selection: nothing was removed"],
    )


def _emit(result: cleanup.CleanResult) -> int:
    print(json.dumps(result.to_dict(), indent=2))
    return 1 if result.errors else 0


def cmd_clean(args: argparse.Namespace) -> int:
    only = _selection(args.only)
    refused = _refusal(args.dry_run, only)
    if refused:
        return _emit(refused)
    return _emit(
        cleanup.clean(
            # Naming a job id is the selection: a named job's age is not a
            # reason to keep its workdir.
            all_finished=args.all_finished or only is not None,
            older_than_days=args.older_than,
            dry_run=args.dry_run,
            only=only,
        )
    )


def cmd_purge(args: argparse.Namespace) -> int:
    only = _selection(args.only)
    refused = _refusal(args.dry_run, only)
    if refused:
        return _emit(refused)
    verified = _selection(args.verified)
    if args.verified_file:
        listed = Path(args.verified_file)
        verified = _selection(listed.read_text().strip())
        listed.unlink(missing_ok=True)
    return _emit(
        cleanup.purge(
            older_than_days=args.older_than,
            dry_run=args.dry_run,
            only=only,
            evidence=cleanup.Evidence(
                force=args.force, verified=None if verified is None else frozenset(verified)
            ),
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m gpuc.host")
    sub = parser.add_subparsers(dest="command", required=True)

    enqueue = sub.add_parser("enqueue", help="enqueue a JSON JobSpec and start the dispatcher")
    enqueue.add_argument("spec", help="path to a JSON spec, or - for stdin")
    enqueue.set_defaults(func=cmd_enqueue)

    status = sub.add_parser("status", help="host and job status as JSON")
    status.add_argument("job_ids", nargs="*", metavar="job_id", help="exactly these jobs")
    status.add_argument(
        "--recent", type=int, metavar="N", help="with no ids: only the newest N finished jobs"
    )
    status.add_argument(
        "--since",
        type=float,
        metavar="SECONDS",
        help="with no ids: only finished jobs that ended this recently",
    )
    status.set_defaults(func=cmd_status)

    cancel = sub.add_parser("cancel", help="cancel queued or running jobs")
    cancel.add_argument("job_ids", nargs="+", metavar="job_id")
    cancel.set_defaults(func=cmd_cancel)

    preempt = sub.add_parser("preempt", help="stop running jobs and queue them again")
    preempt.add_argument("job_ids", nargs="+", metavar="job_id")
    preempt.add_argument(
        "--priority", type=int, help="queue them again at this priority instead of their own"
    )
    preempt.set_defaults(func=cmd_preempt)

    set_ = sub.add_parser("set", help="change queued or running jobs' settings")
    set_.add_argument("job_ids", nargs="+", metavar="job_id")
    set_.add_argument(
        "--field",
        action="append",
        required=True,
        metavar="NAME=JSON",
        help="a field and its new value as JSON; null clears it",
    )
    set_.set_defaults(func=cmd_set)

    fetch_list = sub.add_parser("fetch", help="list the files `gpuc fetch` copies; changes nothing")
    fetch_list.add_argument("job_ids", nargs="+", metavar="job_id")
    fetch_list.add_argument("--path", action="append", default=[])
    fetch_list.set_defaults(func=cmd_fetch_list)

    run = sub.add_parser("run", help="run one job in the foreground (used by the dispatcher)")
    run.add_argument("job_id")
    run.add_argument(
        "--gpus",
        required=True,
        metavar="UUIDS",
        help="the cards the dispatcher assigned, comma-separated; the runner claims the job "
        "with them",
    )
    run.add_argument(
        "--attempt",
        required=True,
        type=int,
        help="the attempt the dispatcher launched; the claim is for that attempt only",
    )
    run.set_defaults(func=cmd_run)

    clean = sub.add_parser("clean", help="remove finished jobs' workdirs")
    selection = clean.add_mutually_exclusive_group(required=True)
    selection.add_argument("--all-finished", action="store_true")
    selection.add_argument("--older-than", type=float, metavar="DAYS")
    selection.add_argument(
        "--only",
        help="comma-separated job ids whose workdirs may go, and no others, whatever "
        "their age. Empty means clean nothing.",
    )
    clean.add_argument("--dry-run", action="store_true")
    clean.set_defaults(func=cmd_clean)

    purge = sub.add_parser(
        "purge", help="remove whole job dirs of old, mirrored, finished jobs (implies clean)"
    )
    purge.add_argument(
        "--older-than",
        type=float,
        default=cleanup.DEFAULT_RETENTION_DAYS,
        metavar="DAYS",
        help=f"measured from ended_at; default {cleanup.DEFAULT_RETENTION_DAYS:g}",
    )
    purge.add_argument("--dry-run", action="store_true")
    purge.add_argument(
        "--force",
        action="store_true",
        help="purge even without a confirmed mirror or confirmed outputs",
    )
    purge.add_argument(
        "--only",
        help="comma-separated job ids that may be purged (and swept), and no others, "
        "whatever their age. Empty means purge nothing.",
    )
    purge.add_argument(
        "--verified",
        help="comma-separated job ids whose mirrored log the caller listed itself; given, "
        "a job must be in it as well as recorded here to count as backed up. Empty "
        "means none of them are.",
    )
    purge.add_argument(
        "--verified-file",
        metavar="PATH",
        help="the same list read from PATH (and PATH removed), for a list too long for argv",
    )
    purge.set_defaults(func=cmd_purge)

    sub.add_parser("dispatch", help="run the dispatcher loop").set_defaults(func=dispatcher.main)

    health_cmd = sub.add_parser("health", help="run host health checks")
    health.add_arguments(health_cmd)
    health_cmd.set_defaults(func=health.main)
    storage.add_parsers(sub)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parsed = build_parser().parse_args(list(argv) if argv is not None else sys.argv[1:])
    return int(parsed.func(parsed))


if __name__ == "__main__":
    raise SystemExit(main())
