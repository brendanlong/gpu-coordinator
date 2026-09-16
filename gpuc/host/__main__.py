"""`python -m gpuc.host <cmd>` — the on-host CLI the control side drives.

Output is JSON on stdout so the control side never has to parse prose.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from gpuc.host import cleanup, dispatcher, gpus, health, jobs, paths, queue, runner
from gpuc.host.jobs import JobSpec


def _read_spec_document(source: str) -> dict[str, Any]:
    text = sys.stdin.read() if source == "-" else Path(source).read_text()
    document = json.loads(text)
    if not isinstance(document, dict):
        raise SystemExit("spec must be a JSON object")
    return document


def cmd_enqueue(args: argparse.Namespace) -> int:
    document = _read_spec_document(args.spec)
    spec = JobSpec.from_dict(document)
    queue.enqueue(spec)
    pid = 0 if args.no_dispatch else dispatcher.spawn_detached_dispatcher()
    print(json.dumps({"job_id": spec.job_id, "dispatcher_pid": pid}))
    return 0


def cmd_list(_: argparse.Namespace) -> int:
    print(
        json.dumps(
            [
                {"priority": e.priority, "job_id": e.job_id, "name": _name(e.job_id)}
                for e in queue.list_queued()
            ],
            indent=2,
        )
    )
    return 0


def _spec(job_id: str) -> JobSpec | None:
    try:
        return jobs.read_spec(job_id)
    except (RuntimeError, FileNotFoundError, ValueError):
        return None


def _name(job_id: str) -> str:
    spec = _spec(job_id)
    return spec.name if spec else ""


def _gpu_table(config: jobs.HostConfig) -> dict[str, Any]:
    """What `config.gpus` resolves to on this host right now.

    The control side cannot work this out: `gpus` may name cards by index, and
    only the host knows what its driver is calling them today.
    """
    try:
        indices = {gpu.uuid: gpu.index for gpu in gpus.list_gpus()}
    except gpus.GpuError:
        indices = {}
    try:
        resolved, unavailable = gpus.resolve_owned(config.gpus)
    except gpus.GpuError:
        resolved, unavailable = [], list(config.gpus)
    return {
        "gpus": config.gpus,
        "gpus_resolved": [{"index": indices.get(uuid), "uuid": uuid} for uuid in resolved],
        "gpus_unavailable": unavailable,
    }


WANDB_HINTS = {"WANDB_ENTITY": "entity", "WANDB_PROJECT": "project", "WANDB_RUN_ID": "run_id"}


def wandb_hints(env: dict[str, str]) -> dict[str, str]:
    return {name: env[key] for key, name in WANDB_HINTS.items() if env.get(key)}


def cmd_status(args: argparse.Namespace) -> int:
    config = jobs.read_config()
    job_ids = [args.job_id] if args.job_id else jobs.list_job_ids()
    entries: list[dict[str, Any]] = []
    for job_id in job_ids:
        try:
            state = jobs.read_state(job_id)
        except (RuntimeError, FileNotFoundError):
            continue
        spec = _spec(job_id)
        entry = {"job_id": job_id, "name": spec.name if spec else "", **state.to_dict()}
        # The watchdog rule this job is actually being judged by, so
        # `gpuc status --suspects` names the jobs the host is about to kill
        # rather than applying a constant of its own.
        entry["low_util"] = asdict(spec.low_util) if spec else None
        # From the spec, not the state: a *queued* job has no eta yet, and its
        # estimate is exactly what somebody deciding whether to queue behind it
        # needs. The control side never sees the spec.
        entry["estimated_runtime_min"] = spec.estimated_runtime_min if spec else None
        # Where the results went, for anything that wants to link to them. The
        # W&B keys are the three that name a run; the job's env is otherwise
        # its own business and never leaves the host.
        entry["outputs"] = [asdict(o) for o in spec.outputs] if spec else []
        entry["wandb"] = wandb_hints(spec.env) if spec else {}
        # Only for finished jobs: a running job's workdir is being written to,
        # its size is meaningless, and walking a live venv on every `gpuc
        # status` would be pure cost.
        entry["workdir_bytes"] = cleanup.workdir_size(job_id) if state.finished else None
        # "this job produced something that is still only here": the control
        # side cannot work it out, since it never sees the spec's `outputs:`.
        entry["outputs_pending"] = (
            not cleanup.outputs_confirmed(job_id, state)[0] if state.finished else False
        )
        entries.append(entry)
    heartbeat = dispatcher.heartbeat_age()
    print(
        json.dumps(
            {
                "host": config.host,
                **_gpu_table(config),
                "ephemeral": config.ephemeral,
                "draining": paths.draining_file().exists(),
                "paused": paths.paused_file().exists(),
                "dispatcher_heartbeat_age_s": None if heartbeat is None else round(heartbeat, 1),
                "queue": [
                    {"priority": e.priority, "job_id": e.job_id} for e in queue.list_queued()
                ],
                "jobs": entries,
            },
            indent=2,
        )
    )
    return 0


def cmd_cancel(args: argparse.Namespace) -> int:
    status = queue.cancel(args.job_id)
    print(json.dumps({"job_id": args.job_id, "status": status}))
    return 0


def cmd_reorder(args: argparse.Namespace) -> int:
    moved = queue.reorder(args.job_id, args.priority)
    print(json.dumps({"job_id": args.job_id, "reordered": moved}))
    return 0 if moved else 1


def _estimate_error(job_id: str, minutes: float | None) -> str | None:
    """Why this host will not record this estimate, or None."""
    if not paths.job_dir(job_id).is_dir():
        return f"no job with that id on this host: {job_id}"
    if minutes is not None and not minutes > 0.0:
        return f"an estimate must be a positive number of minutes, got {minutes!r}"
    if minutes is not None and jobs.utc_in(minutes * 60.0) is None:
        # `1e10` -- the units typo `utc_in` already defends the runner against
        # -- is not an end time any date can hold, so the runner would publish
        # no eta and this command would have reported success for nothing.
        return f"an estimate of {minutes:g} minutes is too far away to be an end time"
    try:
        state = jobs.read_state(job_id)
    except (RuntimeError, FileNotFoundError) as exc:
        return f"could not read the state of {job_id}: {exc}"
    if state.finished:
        # Nothing would ever show it: `eta` is a live job's business, and the
        # status line of a finished job says how long it actually took.
        return f"job {job_id} has already {state.status}, so an estimate cannot change anything"
    return None


def cmd_estimate(args: argparse.Namespace) -> int:
    """Set (or clear) `estimated_runtime_min` on a job that is already here.

    A running job's runner re-reads `spec.json` on a timer, so this reaches it
    without any message passing: see `runner.SPEC_REFRESH_S`.
    """
    job_id = args.job_id
    if args.clear is (args.minutes is not None):
        print(json.dumps({"job_id": job_id, "error": "give MINUTES, or --clear, not both"}))
        return 1
    minutes = None if args.clear else args.minutes
    error = _estimate_error(job_id, minutes)
    if error is not None:
        print(json.dumps({"job_id": job_id, "error": error}))
        return 1
    spec = jobs.update_spec(job_id, estimated_runtime_min=minutes)
    warning = None
    if (
        spec.estimated_runtime_min is not None
        and spec.max_runtime_min is not None
        and spec.estimated_runtime_min > spec.max_runtime_min
    ):
        # The same contradiction `submit` warns about, and the only place
        # anyone will see it before the job dies as `timeout`.
        warning = (
            f"estimated_runtime_min ({spec.estimated_runtime_min:g}) is longer than this job's "
            f"max_runtime_min ({spec.max_runtime_min:g}), so it expects to be killed as "
            f"`timeout` before it finishes"
        )
    print(
        json.dumps(
            {
                "job_id": job_id,
                "estimated_runtime_min": spec.estimated_runtime_min,
                "status": jobs.read_state(job_id).status,
                "warning": warning,
            }
        )
    )
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    return runner.run_job(args.job_id)


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
    sweep_only = _selection(args.sweep_only)
    refused = _refusal(args.dry_run, only, sweep_only)
    if refused:
        return _emit(refused)
    return _emit(
        cleanup.purge(
            older_than_days=args.older_than,
            dry_run=args.dry_run,
            force=args.force,
            only=only,
            sweep_only=sweep_only,
        )
    )


def cmd_resume(_: argparse.Namespace) -> int:
    paths.paused_file().unlink(missing_ok=True)
    dispatcher.spawn_detached_dispatcher()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m gpuc.host")
    sub = parser.add_subparsers(dest="command", required=True)

    enqueue = sub.add_parser("enqueue", help="enqueue a JSON JobSpec and start the dispatcher")
    enqueue.add_argument("spec", help="path to a JSON spec, or - for stdin")
    enqueue.add_argument("--no-dispatch", action="store_true")
    enqueue.set_defaults(func=cmd_enqueue)

    sub.add_parser("list", help="list queued jobs").set_defaults(func=cmd_list)

    status = sub.add_parser("status", help="host and job status as JSON")
    status.add_argument("job_id", nargs="?")
    status.set_defaults(func=cmd_status)

    cancel = sub.add_parser("cancel", help="cancel a queued or running job")
    cancel.add_argument("job_id")
    cancel.set_defaults(func=cmd_cancel)

    reorder = sub.add_parser("reorder", help="change a queued job's priority")
    reorder.add_argument("job_id")
    reorder.add_argument("priority", type=int)
    reorder.set_defaults(func=cmd_reorder)

    estimate = sub.add_parser("estimate", help="set a queued or running job's runtime estimate")
    estimate.add_argument("job_id")
    estimate.add_argument("minutes", nargs="?", type=float)
    estimate.add_argument("--clear", action="store_true", help="remove the estimate instead")
    estimate.set_defaults(func=cmd_estimate)

    run = sub.add_parser("run", help="run one job in the foreground (used by the dispatcher)")
    run.add_argument("job_id")
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
        help="comma-separated job ids that may be purged, and no others, whatever their "
        "age; the implied workdir sweep is unaffected. Empty means purge nothing.",
    )
    purge.add_argument(
        "--sweep-only",
        help="comma-separated job ids the implied workdir sweep may touch, and no "
        "others, whatever their age; by default it covers every finished job past "
        "the horizon",
    )
    purge.set_defaults(func=cmd_purge)

    sub.add_parser("resume", help="clear a low-util pause and restart dispatching").set_defaults(
        func=cmd_resume
    )

    # Listed for `--help` only; main() hands these their own argv before parsing.
    sub.add_parser("dispatch", help="run the dispatcher loop", add_help=False)
    sub.add_parser("health", help="run host health checks", add_help=False)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = list(argv) if argv is not None else sys.argv[1:]
    # These two own their own flags, so hand them the remaining argv verbatim.
    if args and args[0] == "dispatch":
        return dispatcher.main(args[1:])
    if args and args[0] == "health":
        return health.main(args[1:])
    parsed = build_parser().parse_args(args)
    return int(parsed.func(parsed))


if __name__ == "__main__":
    raise SystemExit(main())
