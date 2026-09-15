"""`gpuc clean` and `gpuc host clean --uv-cache`: reclaim disk on a host.

Both are thin: the host package decides what is safe to delete (it is the only
thing that can read a job's `state.json` without a race), and this module asks
it and formats the answer.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass, field
from typing import Any

from gpuc.control.config import HostEntry, Settings, transport_for
from gpuc.control.remote import HostSession, open_session
from gpuc.control.transport import Transport
from gpuc.host.cleanup import human_bytes


class CleanError(RuntimeError):
    pass


@dataclass
class CleanReport:
    host: str
    dry_run: bool = False
    removed: list[dict[str, Any]] = field(default_factory=list)
    skipped: list[dict[str, Any]] = field(default_factory=list)
    incoming_removed: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    freed_bytes: int = 0

    def render(self) -> str:
        verb = "would free" if self.dry_run else "freed"
        head = (
            f"host {self.host}: {verb} {human_bytes(self.freed_bytes)} "
            f"from {len(self.removed)} workdir(s)"
        )
        if self.dry_run:
            head += "  (dry run, nothing was deleted)"
        lines = [head]
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
        if not self.removed and not self.incoming_removed:
            lines.append("  nothing to remove")
        return "\n".join(lines)


def clean_host(
    entry: HostEntry,
    settings: Settings | None = None,
    *,
    session: HostSession | None = None,
    all_finished: bool = False,
    older_than_days: float | None = None,
    dry_run: bool = False,
) -> CleanReport:
    if not all_finished and older_than_days is None:
        raise CleanError("clean needs --all-finished or --older-than DAYS")
    args = ["clean"]
    if all_finished:
        args.append("--all-finished")
    if older_than_days is not None:
        args += ["--older-than", str(older_than_days)]
    if dry_run:
        args.append("--dry-run")
    session = session or open_session(entry, settings)
    # A workdir walk over many jobs is minutes of stat() on a slow volume, and
    # the host CLI is doing the deleting too.
    payload = session.host_json(" ".join(args), timeout=900.0)
    if not isinstance(payload, dict):
        raise CleanError(f"unexpected clean response from host {entry.name}: {payload!r}")
    return CleanReport(
        host=entry.name,
        dry_run=bool(payload.get("dry_run")),
        removed=list(payload.get("removed") or []),
        skipped=list(payload.get("skipped") or []),
        incoming_removed=list(payload.get("incoming_removed") or []),
        errors=list(payload.get("errors") or []),
        freed_bytes=int(payload.get("freed_bytes") or 0),
    )


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
