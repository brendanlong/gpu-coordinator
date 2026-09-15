"""`gpuc clean` and `gpuc host clean --uv-cache`: reclaim disk on a host.

Both are thin: the host package decides what is safe to delete (it is the only
thing that can read a job's `state.json` without a race), and this module asks
it and formats the answer. `--purge` adds the one thing the control side can
do and the host cannot: `--verify` HEADs the mirrored log in S3 with our own
credentials before the job dir that holds the original goes away.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass, field
from typing import Any

from gpuc.control.config import HostEntry, Settings, load_settings, transport_for
from gpuc.control.remote import HostSession, open_session
from gpuc.control.s3index import S3Index, job_log_uri, make_s3_client, split_uri
from gpuc.control.transport import Transport
from gpuc.host.cleanup import DEFAULT_RETENTION_DAYS, human_bytes


class CleanError(RuntimeError):
    pass


@dataclass
class CleanReport:
    host: str
    dry_run: bool = False
    removed: list[dict[str, Any]] = field(default_factory=list)
    skipped: list[dict[str, Any]] = field(default_factory=list)
    purged: list[dict[str, Any]] = field(default_factory=list)
    purge_skipped: list[dict[str, Any]] = field(default_factory=list)
    incoming_removed: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    freed_bytes: int = 0
    purge: bool = False
    verified: list[str] = field(default_factory=list)
    """Job ids whose mirrored `log.txt` we HEADed ourselves under `--verify`."""
    notes: list[str] = field(default_factory=list)

    def render(self) -> str:
        verb = "would free" if self.dry_run else "freed"
        what = f"{len(self.removed)} workdir(s)"
        if self.purge:
            what = f"{len(self.purged)} job dir(s) and {what}"
        head = f"host {self.host}: {verb} {human_bytes(self.freed_bytes)} from {what}"
        if self.dry_run:
            head += "  (dry run, nothing was deleted)"
        lines = [head]
        for job in self.purged:
            lines.append(_purged_line(job, dry_run=self.dry_run))
            if job.get("forced"):
                lines.append(
                    "          ^ FORCED: this job had no confirmed backup and its record "
                    "is now gone for good"
                )
        for job in self.purge_skipped:
            if job.get("why") == "no selection given":
                continue
            lines.append(f"  SKIPPED {job['job_id']}  {job.get('why', '')}")
        for note in self.notes:
            lines.append(f"  note: {note}")
        for job in self.removed:
            age = job.get("age_days")
            when = f"{age:.1f}d old" if isinstance(age, (int, float)) else "age unknown"
            lines.append(
                f"  {job['job_id']}  {job.get('status', '?'):<9} "
                f"{human_bytes(int(job.get('bytes') or 0)):>9}  {when}"
            )
        for job in self.skipped:
            # "no selection given" is every job the flags simply did not ask
            # for; saying so once per job would bury the ones that matter.
            if job.get("why") != "no selection given":
                lines.append(f"  kept    {job['job_id']}  {job.get('why', '')}")
        if self.incoming_removed:
            staged = ", ".join(self.incoming_removed)
            lines.append(
                f"  removed {len(self.incoming_removed)} leftover staged spec(s): {staged}"
            )
        for error in self.errors:
            lines.append(f"  ERROR {error}")
        if not self.removed and not self.purged and not self.incoming_removed:
            lines.append("  nothing to remove")
        return "\n".join(lines)


def _purged_line(job: dict[str, Any], *, dry_run: bool) -> str:
    age = job.get("age_days")
    when = f"{age:.1f}d old" if isinstance(age, (int, float)) else "age unknown"
    verb = "WOULD PURGE" if dry_run else "PURGED "
    return (
        f"  {verb} {job['job_id']}  {job.get('status', '?'):<9} "
        f"{human_bytes(int(job.get('bytes') or 0)):>9}  {when}"
    )


def clean_host(
    entry: HostEntry,
    settings: Settings | None = None,
    *,
    session: HostSession | None = None,
    all_finished: bool = False,
    older_than_days: float | None = None,
    dry_run: bool = False,
    purge: bool = False,
    force: bool = False,
    verify: bool = False,
    s3_client: Any | None = None,
) -> CleanReport:
    if not purge and not all_finished and older_than_days is None:
        raise CleanError("clean needs --all-finished or --older-than DAYS")
    if verify and not purge:
        raise CleanError("--verify only means anything with --purge")
    session = session or open_session(entry, settings)
    if purge:
        return purge_host(
            entry,
            settings,
            session=session,
            all_finished=all_finished,
            older_than_days=older_than_days,
            dry_run=dry_run,
            force=force,
            verify=verify,
            s3_client=s3_client,
        )
    args = ["clean"]
    if all_finished:
        args.append("--all-finished")
    if older_than_days is not None:
        args += ["--older-than", str(older_than_days)]
    if dry_run:
        args.append("--dry-run")
    # A workdir walk over many jobs is minutes of stat() on a slow volume, and
    # the host CLI is doing the deleting too.
    payload = session.host_json(" ".join(args), timeout=900.0)
    return _report(entry.name, payload)


def _report(host: str, payload: Any, **extra: Any) -> CleanReport:
    if not isinstance(payload, dict):
        raise CleanError(f"unexpected clean response from host {host}: {payload!r}")
    return CleanReport(
        host=host,
        dry_run=bool(payload.get("dry_run")),
        removed=list(payload.get("removed") or []),
        skipped=list(payload.get("skipped") or []),
        purged=list(payload.get("purged") or []),
        purge_skipped=list(payload.get("purge_skipped") or []),
        incoming_removed=list(payload.get("incoming_removed") or []),
        errors=list(payload.get("errors") or []),
        freed_bytes=int(payload.get("freed_bytes") or 0),
        **extra,
    )


def _purge_args(
    older_than_days: float, *, dry_run: bool, force: bool, only: list[str] | None
) -> str:
    args = ["purge", "--older-than", str(older_than_days)]
    if dry_run:
        args.append("--dry-run")
    if force:
        args.append("--force")
    if only is not None:
        args += ["--only", shlex.quote(",".join(only))]
    return " ".join(args)


def purge_host(
    entry: HostEntry,
    settings: Settings | None = None,
    *,
    session: HostSession | None = None,
    all_finished: bool = False,
    older_than_days: float | None = None,
    dry_run: bool = False,
    force: bool = False,
    verify: bool = False,
    s3_client: Any | None = None,
) -> CleanReport:
    """Remove whole job dirs on a host, optionally checking the mirror first.

    Without `--verify` this trusts `meta_synced_at`, which the host wrote only
    after its own upload returned 0 -- the cheap answer, and the only one
    available to a host with no credentials of ours. With it we HEAD the
    mirrored `log.txt` ourselves: a dry run then *labels* each candidate, and a
    real purge only deletes the ones that answered.
    """
    session = session or open_session(entry, settings)
    days = (
        0.0
        if all_finished
        else (DEFAULT_RETENTION_DAYS if older_than_days is None else older_than_days)
    )
    if not verify:
        payload = session.host_json(
            _purge_args(days, dry_run=dry_run, force=force, only=None), timeout=900.0
        )
        return _report(entry.name, payload, purge=True)

    preview = session.host_json(
        _purge_args(days, dry_run=True, force=force, only=None), timeout=900.0
    )
    report = _report(entry.name, preview, purge=True)
    verified, unverified = verify_mirror(entry, report, settings, client=s3_client)
    report.verified = verified
    for job_id, why in unverified:
        if force:
            report.notes.append(f"{job_id}: {why} -- purged anyway because --force was given")
        else:
            report.purge_skipped.append({"job_id": job_id, "why": why})
    keep = set(verified) | ({job_id for job_id, _ in unverified} if force else set())
    report.purged = [job for job in report.purged if job["job_id"] in keep]
    report.freed_bytes = sum(int(job.get("bytes") or 0) for job in report.purged) + sum(
        int(job.get("bytes") or 0) for job in report.removed
    )
    if dry_run:
        return report
    # `--only` with the verified ids -- possibly none of them, which the host
    # reads as "purge nothing", while the workdir sweep `--purge` implies still
    # runs.
    payload = session.host_json(
        _purge_args(days, dry_run=False, force=force, only=sorted(keep)), timeout=900.0
    )
    final = _report(entry.name, payload, purge=True)
    final.verified = verified
    final.notes = report.notes
    final.purge_skipped += report.purge_skipped
    return final


def _s3_client(settings: Settings) -> Any:
    """A client for the HEADs. The bucket comes from each job's own recorded
    prefix, so this works even for a host mirroring outside `s3_bucket`."""
    index = S3Index.from_settings(settings)
    if index is not None:
        return index.client
    return make_s3_client()


def verify_mirror(
    entry: HostEntry,
    report: CleanReport,
    settings: Settings | None = None,
    *,
    client: Any | None = None,
) -> tuple[list[str], list[tuple[str, str]]]:
    """HEAD each candidate's mirrored `log.txt`. Returns (verified, (id, why))."""
    s3 = client if client is not None else _s3_client(settings or load_settings())
    verified: list[str] = []
    unverified: list[tuple[str, str]] = []
    for job in report.purged:
        job_id = str(job["job_id"])
        prefix = job.get("meta_synced_to") or entry.s3_prefix
        if not prefix:
            unverified.append((job_id, "no mirror prefix to verify against"))
            continue
        uri = job_log_uri(str(prefix), job_id)
        bucket, key = split_uri(uri)
        try:
            s3.head_object(Bucket=bucket, Key=key)
        except Exception as exc:  # botocore raises its own per-operation classes
            unverified.append((job_id, f"no mirrored log at {uri} ({type(exc).__name__})"))
            continue
        verified.append(job_id)
    return verified, unverified


UV_CACHE_PRUNE = """\
cache=$({env}{uv} cache dir 2>/dev/null || echo "$HOME/.cache/uv")
echo "before=$(du -sh "$cache" 2>/dev/null | cut -f1)"
{env}{uv} cache prune
echo "after=$(du -sh "$cache" 2>/dev/null | cut -f1)"
echo "dir=$cache"
"""


def prune_uv_cache(
    entry: HostEntry, settings: Settings | None = None, *, transport: Transport | None = None
) -> str:
    """`uv cache prune` on the host: drop cache entries no venv can link to.

    Deliberately `prune` and not `clean`: pruning removes unused and
    unreachable entries, while `uv cache clean` would throw away exactly the
    wheels the next job wants to link out of.
    """
    from gpuc.control.bootstrap import env_prefix

    transport = transport or transport_for(entry, settings)
    uv = entry.uv or "uv"
    result = transport.run(
        UV_CACHE_PRUNE.format(env=env_prefix(entry), uv=shlex.quote(uv)),
        timeout=900.0,
        check=False,
    )
    if result.returncode != 0:
        raise CleanError(
            f"`uv cache prune` on host {entry.name} exited {result.returncode}:\n"
            f"{result.output.strip()[-800:]}"
        )
    values = dict(line.split("=", 1) for line in result.stdout.splitlines() if line.count("=") == 1)
    return (
        f"host {entry.name}: uv cache {values.get('dir', '?')} "
        f"pruned {values.get('before', '?')} -> {values.get('after', '?')}"
    )
