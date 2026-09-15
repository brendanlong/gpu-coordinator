"""`python -m gpuc.host <cmd>` — the on-host CLI the control side drives.

Output is JSON on stdout so the control side never has to parse prose.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from gpuc.host import cleanup, dispatcher, health, jobs, paths, queue, runner
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


def _name(job_id: str) -> str:
    try:
        return jobs.read_spec(job_id).name
    except (RuntimeError, FileNotFoundError):
        return ""


def cmd_status(args: argparse.Namespace) -> int:
    config = jobs.read_config()
    job_ids = [args.job_id] if args.job_id else jobs.list_job_ids()
    entries: list[dict[str, Any]] = []
    for job_id in job_ids:
        try:
            state = jobs.read_state(job_id)
        except (RuntimeError, FileNotFoundError):
            continue
        entry = {"job_id": job_id, "name": _name(job_id), **state.to_dict()}
        # Only for finished jobs: a running job's workdir is being written to,
        # its size is meaningless, and walking a live venv on every `gpuc
        # status` would be pure cost.
        entry["workdir_bytes"] = cleanup.workdir_size(job_id) if state.finished else None
        entries.append(entry)
    heartbeat = dispatcher.DispatcherLock().heartbeat_age()
    print(
        json.dumps(
            {
                "host": config.host,
                "gpus": config.gpus,
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


def cmd_run(args: argparse.Namespace) -> int:
    return runner.run_job(args.job_id)


def cmd_clean(args: argparse.Namespace) -> int:
    result = cleanup.clean(
        all_finished=args.all_finished,
        older_than_days=args.older_than,
        dry_run=args.dry_run,
    )
    print(json.dumps(result.to_dict(), indent=2))
    return 1 if result.errors else 0


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

    run = sub.add_parser("run", help="run one job in the foreground (used by the dispatcher)")
    run.add_argument("job_id")
    run.set_defaults(func=cmd_run)

    clean = sub.add_parser("clean", help="remove finished jobs' workdirs")
    selection = clean.add_mutually_exclusive_group(required=True)
    selection.add_argument("--all-finished", action="store_true")
    selection.add_argument("--older-than", type=float, metavar="DAYS")
    clean.add_argument("--dry-run", action="store_true")
    clean.set_defaults(func=cmd_clean)

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
