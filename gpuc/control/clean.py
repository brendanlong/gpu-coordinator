"""`gpuc clean` and `gpuc host clean --uv-cache`: reclaim disk on a host.

Both are thin: the host package decides what is safe to delete (it is the only
thing that can read a job's `state.json` without a race), and this module asks
it and formats the answer. `--purge` adds the one thing the control side can
do and the host cannot: `--verify` lists the mirrored logs in S3 with our own
credentials before a job dir that holds the original goes away.
"""

from __future__ import annotations

import os
import shlex
from dataclasses import dataclass, field
from typing import Any

from gpuc.control.config import HostEntry, Settings, load_settings
from gpuc.control.exits import EXIT_USAGE
from gpuc.control.remote import HostSession, env_prefix, open_session
from gpuc.control.s3index import S3Index, job_uri, make_s3_client, split_uri
from gpuc.host.cleanup import DEFAULT_RETENTION_DAYS, human_bytes


class CleanError(RuntimeError):
    pass


class CleanUsageError(CleanError):
    """The flags cannot be honoured: the command line is wrong, not the host.

    Exit 2, the same as argparse's, so a script can tell "I asked for something
    impossible" from "the clean ran and something failed".
    """

    exit_code = EXIT_USAGE


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
                f"  removed {len(self.incoming_removed)} leftover staged job dir(s): {staged}"
            )
        for error in self.errors:
            lines.append(f"  ERROR {error}")
        if not self.removed and not self.purged and not self.incoming_removed:
            lines.append("  nothing to remove")
        return "\n".join(lines)

    def document(self) -> dict[str, Any]:
        """`gpuc clean --json`. A non-empty `errors` is the exit-1 case.

        The job objects are the host's own, passed through: each carries at
        least `job_id`, and `status`, `bytes` and `age_days` where the host
        could read them. `skipped`/`purge_skipped` add `why`.
        """
        return {
            "host": self.host,
            "dry_run": self.dry_run,
            "purge": self.purge,
            "freed_bytes": self.freed_bytes,
            "removed": list(self.removed),
            "skipped": list(self.skipped),
            "purged": list(self.purged),
            "purge_skipped": list(self.purge_skipped),
            "incoming_removed": list(self.incoming_removed),
            "verified": list(self.verified),
            "notes": list(self.notes),
            "errors": list(self.errors),
        }


def _purged_line(job: dict[str, Any], *, dry_run: bool) -> str:
    age = job.get("age_days")
    when = f"{age:.1f}d old" if isinstance(age, (int, float)) else "age unknown"
    verb = "WOULD PURGE" if dry_run else "PURGED "
    return (
        f"  {verb} {job['job_id']}  {job.get('status', '?'):<9} "
        f"{human_bytes(int(job.get('bytes') or 0)):>9}  {when}"
    )


PURGE_EVERYTHING_NOTE = "purging every finished job (horizon 0), however recently it ended"


def only_note(only: list[str]) -> str:
    return (
        f"--only {','.join(only)}: age horizon 0 for exactly these jobs, and the workdir "
        f"sweep --purge implies is scoped to them too"
    )


def parse_only(value: str | None) -> list[str] | None:
    """`--only a,b` as a list of job ids, or None if the flag was not given.

    Empty is a usage error rather than the host's "none of them": a `--only`
    that expanded from a shell variable nobody set must not quietly become a
    no-op (or, worse, everything).
    """
    if value is None:
        return None
    wanted = [job_id.strip() for job_id in value.split(",")]
    wanted = [job_id for job_id in wanted if job_id]
    if not wanted:
        raise CleanUsageError(
            "--only needs at least one job id, e.g. --only 20260101-000000-abcdef"
        )
    return list(dict.fromkeys(wanted))


def check_flags(
    *,
    all_finished: bool = False,
    older_than_days: float | None = None,
    dry_run: bool = False,
    purge: bool = False,
    force: bool = False,
    verify: bool = False,
    yes: bool = False,
    only: list[str] | None = None,
) -> None:
    """The one place `gpuc clean`'s flag combinations are judged.

    The CLI delegates here rather than repeating it, because the two copies had
    drifted to different exit codes for the same mistake.
    """
    if (force or verify) and not purge:
        raise CleanUsageError("--force and --verify only mean something with --purge")
    if only is not None and (all_finished or older_than_days is not None):
        raise CleanUsageError(
            "--only names the jobs itself, so it cannot be combined with --all-finished "
            "or --older-than"
        )
    if only is not None and not only:
        raise CleanUsageError("--only needs at least one job id")
    if not purge and only is None and not all_finished and older_than_days is None:
        raise CleanUsageError(
            f"clean needs --all-finished, --older-than DAYS, --only ID[,ID...], or --purge "
            f"(which defaults to --older-than {DEFAULT_RETENTION_DAYS:g})"
        )
    if purge and all_finished and not (yes or dry_run):
        # --all-finished means an age horizon of 0, so this deletes the job dir
        # of something that ended a minute ago -- log, state and all. Worth
        # typing one more word for.
        raise CleanUsageError(
            "--purge --all-finished deletes the whole job dir of every finished job, "
            "however recently it ended (age horizon 0).\n"
            "Add --yes to confirm, --dry-run to see the list first, or pick a horizon "
            "with --older-than DAYS."
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
    yes: bool = False,
    only: list[str] | None = None,
    s3_client: Any | None = None,
) -> CleanReport:
    check_flags(
        all_finished=all_finished,
        older_than_days=older_than_days,
        dry_run=dry_run,
        purge=purge,
        force=force,
        verify=verify,
        yes=yes,
        only=only,
    )
    session = session or open_session(entry, settings)
    if purge:
        report = purge_host(
            entry,
            settings,
            session=session,
            all_finished=all_finished,
            older_than_days=older_than_days,
            dry_run=dry_run,
            force=force,
            verify=verify,
            only=only,
            s3_client=s3_client,
        )
        if all_finished:
            report.notes.insert(0, PURGE_EVERYTHING_NOTE)
        if only:
            report.notes.insert(0, only_note(only))
        return report
    args = ["clean"]
    if all_finished:
        args.append("--all-finished")
    if only is not None:
        args += ["--only", shlex.quote(",".join(only))]
    if older_than_days is not None:
        args += ["--older-than", str(older_than_days)]
    if dry_run:
        args.append("--dry-run")
    # A workdir walk over many jobs is minutes of stat() on a slow volume, and
    # the host CLI is doing the deleting too.
    payload = session.host_json(" ".join(args), timeout=900.0, check=False)
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


VERIFIED_INLINE_MAX = 32 * 1024
"""Longer than this and the id list travels as a file: every id ever mirrored
under a busy host's prefix, in one argument, would pass the kernel's
per-argument limit long before the host stopped being worth purging."""


def _verified_arg(session: HostSession, verified: list[str]) -> str:
    joined = ",".join(verified)
    if len(joined) <= VERIFIED_INLINE_MAX:
        return f"--verified {shlex.quote(joined)}"
    path = f"{session.home}/incoming/.verified-{os.getpid()}"
    session.transport.put_file(joined + "\n", path, 0o600)
    return f"--verified-file {shlex.quote(path)}"


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
    only: list[str] | None = None,
    s3_client: Any | None = None,
) -> CleanReport:
    """Remove whole job dirs on a host, optionally checking the mirror first.

    Without `--verify` the host trusts its own mirror record, which it wrote
    only after its upload returned 0 -- the cheap answer, and the only one
    available to a host with no credentials of ours. With it we list the
    mirrored logs ourselves and hand the host the ids that have one: the host
    then counts only those as backed up, whatever its records say. One round
    trip either way.

    `only` is the user naming job ids: an age horizon of 0 for exactly those
    jobs, with the implied workdir sweep scoped to them as well, so purging one
    job does not reclaim the rest of the host's venvs as a side effect.
    """
    session = session or open_session(entry, settings)
    days = (
        0.0
        if all_finished or only is not None
        else (DEFAULT_RETENTION_DAYS if older_than_days is None else older_than_days)
    )
    verified = (
        verified_mirrors(session.config.s3_prefix, settings, client=s3_client) if verify else None
    )
    args = _purge_args(days, dry_run=dry_run, force=force, only=only)
    if verified is not None:
        args += f" {_verified_arg(session, verified)}"
    payload = session.host_json(args, timeout=900.0, check=False)
    report = _report(entry.name, payload, purge=True)
    if verified is not None:
        report.verified = [job["job_id"] for job in report.purged if job["job_id"] in verified]
    return report


def _s3_client(settings: Settings) -> Any:
    """A client for the listing, from `s3_bucket` when there is one."""
    index = S3Index.from_settings(settings)
    if index is not None:
        return index.client
    return make_s3_client()


def verified_mirrors(
    s3_prefix: str | None, settings: Settings | None = None, *, client: Any | None = None
) -> list[str]:
    """The job ids whose mirrored `log.txt` exists under the host's prefix.

    One listing rather than one HEAD per candidate, and before the host is
    asked anything, so the purge is a single round trip. `s3_prefix` is the
    host's own, from the session's read of its config; a host with none has
    nothing mirrored, and a listing that fails is an error: "could not check"
    must not read as "nothing is backed up".
    """
    if not s3_prefix:
        return []
    s3 = client if client is not None else _s3_client(settings or load_settings())
    bucket, key = split_uri(job_uri(s3_prefix, ""))
    prefix = key.rstrip("/") + "/"
    ids: list[str] = []
    token: str | None = None
    while True:
        request: dict[str, Any] = {"Bucket": bucket, "Prefix": prefix}
        if token:
            request["ContinuationToken"] = token
        try:
            response = s3.list_objects_v2(**request)
        except Exception as exc:  # botocore raises its own per-operation classes
            raise CleanError(f"could not list the mirror at s3://{bucket}/{prefix}: {exc}") from exc
        for item in response.get("Contents", []):
            rest = str(item.get("Key", ""))[len(prefix) :]
            job_id, _, name = rest.partition("/")
            if name == "log.txt":
                ids.append(job_id)
        token = response.get("NextContinuationToken")
        if not response.get("IsTruncated") or not token:
            return sorted(ids)


UV_CACHE_PRUNE = """\
cache=$({env}{uv} cache dir 2>/dev/null || echo "$HOME/.cache/uv")
echo "before_kib=$(du -sk "$cache" 2>/dev/null | cut -f1)"
{env}{uv} cache prune
echo "after_kib=$(du -sk "$cache" 2>/dev/null | cut -f1)"
echo "dir=$cache"
"""


@dataclass
class PruneReport:
    """What `uv cache prune` on a host did: the cache, and its size either side."""

    host: str
    cache_dir: str | None
    before_bytes: int | None
    """`du -sk` of the cache before the prune, in bytes; null if `du` failed."""
    after_bytes: int | None

    @property
    def before(self) -> str | None:
        return None if self.before_bytes is None else human_bytes(self.before_bytes)

    @property
    def after(self) -> str | None:
        return None if self.after_bytes is None else human_bytes(self.after_bytes)

    @property
    def freed_bytes(self) -> int | None:
        if self.before_bytes is None or self.after_bytes is None:
            return None
        return max(self.before_bytes - self.after_bytes, 0)

    def render(self) -> str:
        return (
            f"host {self.host}: uv cache {self.cache_dir or '?'} "
            f"pruned {self.before or '?'} -> {self.after or '?'}"
        )

    def document(self) -> dict[str, Any]:
        return {
            "host": self.host,
            "cache_dir": self.cache_dir,
            "before": self.before,
            "after": self.after,
            "before_bytes": self.before_bytes,
            "after_bytes": self.after_bytes,
            "freed_bytes": self.freed_bytes,
        }


def _kib_to_bytes(raw: str | None) -> int | None:
    return int(raw) * 1024 if raw is not None and raw.isdigit() else None


def prune_uv_cache(
    entry: HostEntry, settings: Settings | None = None, *, session: HostSession | None = None
) -> PruneReport:
    """`uv cache prune` on the host: drop cache entries no venv can link to.

    Deliberately `prune` and not `clean`: pruning removes unused and
    unreachable entries, while `uv cache clean` would throw away exactly the
    wheels the next job wants to link out of. Run with the host's own env, so
    it prunes the cache the host's jobs actually use.
    """
    session = session or open_session(entry, settings)
    uv = entry.uv or "uv"
    result = session.transport.run(
        UV_CACHE_PRUNE.format(env=env_prefix(session.env), uv=shlex.quote(uv)),
        timeout=900.0,
        check=False,
    )
    if result.returncode != 0:
        raise CleanError(
            f"`uv cache prune` on host {entry.name} exited {result.returncode}:\n"
            f"{result.output.strip()[-800:]}"
        )
    values = dict(line.split("=", 1) for line in result.stdout.splitlines() if line.count("=") == 1)
    return PruneReport(
        host=entry.name,
        cache_dir=values.get("dir") or None,
        before_bytes=_kib_to_bytes(values.get("before_kib")),
        after_bytes=_kib_to_bytes(values.get("after_kib")),
    )
