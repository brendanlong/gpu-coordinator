"""`gpuc fetch`: copy what jobs produced from their workdirs to this machine.

Every job's files come from one place, `jobs/<id>/workdir/<path>` on its host,
whether the job is running, failed or finished. The host says which files (it
alone can tell a result from a file that came with the checkout); this side
copies them into `<to>/<job_id>/`, so two jobs never land on each other.
"""

from __future__ import annotations

import shlex
from collections.abc import Sequence
from pathlib import Path

from gpuc.control.actions import Done, job_verbs
from gpuc.control.config import Settings
from gpuc.control.transport import TransportError
from gpuc.host.cleanup import human_bytes


def fetch_jobs(
    job_ids: Sequence[str],
    host: str | None,
    settings: Settings,
    *,
    wanted: Sequence[str] = (),
    to: Path = Path(),
    list_only: bool = False,
) -> list[Done]:
    """Ask each host once for its jobs' files, then copy each job's.

    A copy that fails is that job's error and never stops the next one."""
    args = "".join(f" --path {shlex.quote(path)}" for path in wanted)
    done, sessions = job_verbs("fetch", job_ids, host, settings, args=args)
    for job in done:
        if job.error is not None:
            continue
        job.fields["to"] = None
        files = [str(f["path"]) for f in job.fields.get("files") or []]
        for rel in job.fields.get("missing") or []:
            job.warnings.append(f"{rel} is not in its workdir")
        if list_only or not files:
            continue
        session = sessions.get(job.host or "")
        if session is None:
            job.error = f"job {job.job_id}: host {job.host} answered but cannot be copied from"
            continue
        dest = to / job.job_id
        try:
            dest.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            job.error = f"job {job.job_id}: cannot create {dest}: {exc}"
            continue
        try:
            session.transport.pull(str(job.fields["workdir"]), dest, files)
        except TransportError as exc:
            job.error = f"copying job {job.job_id} from host {job.host} failed: {exc}"
            continue
        job.fields["to"] = str(dest)
    return done


def fetch_line(job: Done) -> str:
    files = job.fields.get("files") or []
    size = human_bytes(int(job.fields.get("bytes") or 0))
    head = f"job {job.job_id} on host {job.host} ({job.fields.get('status')}): "
    if not files:
        return head + "nothing to fetch"
    if job.fields.get("to") is None:
        rows = [f"  {human_bytes(int(f['bytes'])):>10}  {f['path']}" for f in files]
        return "\n".join([head + f"{len(files)} file(s), {size}", *rows])
    return head + f"fetched {len(files)} file(s), {size}, to {job.fields['to']}"
