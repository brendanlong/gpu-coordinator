"""Workdir cleanup: the `cleanup:` policy the runner applies, and the
`python -m gpuc.host clean` sweep for what earlier runs left behind.

A finished job's `workdir` is nearly always the largest thing gpuc owns -- a
torch venv measures ~6.5 GB -- and it is the only part of a job dir that can be
recreated: `gpuc requeue` re-syncs it from git. `spec.json`, `state.json` and
`log.txt` are the record of what happened and are never touched here.
"""

from __future__ import annotations

import contextlib
import os
import shutil
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from gpuc.host import jobs, paths
from gpuc.host.jobs import ALWAYS, FINISHED_STATUSES, ON_SUCCESS

INCOMING_STALE_S = 3600.0
"""How old an orphaned staged spec must be before `clean` removes it.

`submit` writes `incoming/<id>.json`, runs `enqueue`, then deletes it, so the
window in which the file is load-bearing is one SSH round trip. An hour is far
past that and still cannot race a submit that is merely slow.
"""


def should_remove(policy: str, status: str) -> bool:
    """Does `policy` remove the workdir of a job that finished as `status`?

    Never for an unfinished job: `always` means "whatever the outcome", not
    "while the job is still writing to it". `on_success` keeps a failed or
    cancelled workdir precisely so it can be inspected.
    """
    if status not in FINISHED_STATUSES:
        return False
    if policy == ALWAYS:
        return True
    if policy == ON_SUCCESS:
        return status == "succeeded"
    return False


def human_bytes(count: int) -> str:
    size = float(count)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if abs(size) < 1024.0:
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TiB"


def dir_size(root: Path) -> int:
    """Disk usage of `root` in bytes, counting allocated blocks the way `du` does.

    `st_blocks`, not `st_size`: a venv is mostly files uv linked out of its
    cache, and an inode reached twice inside this tree must only be counted
    once. Blocks shared with something *outside* the tree (uv's reflinks) still
    count, exactly as `du` counts them, so a reported figure is an upper bound
    on what the filesystem actually gets back.
    """
    total = 0
    seen: set[tuple[int, int]] = set()
    stack = [root]
    while stack:
        current = stack.pop()
        try:
            entries = list(os.scandir(current))
        except OSError:
            continue
        for entry in entries:
            try:
                info = entry.stat(follow_symlinks=False)
            except OSError:
                continue
            key = (info.st_dev, info.st_ino)
            if key in seen:
                continue
            seen.add(key)
            total += info.st_blocks * 512
            if entry.is_dir(follow_symlinks=False):
                stack.append(Path(entry.path))
    with contextlib.suppress(OSError):
        total += root.stat().st_blocks * 512
    return total


def workdir_size(job_id: str) -> int | None:
    """Bytes held by a job's workdir, or None if it has none left."""
    workdir = paths.workdir(job_id)
    if not workdir.is_dir():
        return None
    return dir_size(workdir)


def remove_workdir(job_id: str) -> int:
    """Delete `jobs/<id>/workdir` and nothing else. Returns the bytes freed."""
    workdir = paths.workdir(job_id)
    if not workdir.is_dir():
        return 0
    size = dir_size(workdir)
    shutil.rmtree(workdir)
    return size


def _parse(stamp: str | None) -> datetime | None:
    if not stamp:
        return None
    try:
        parsed = datetime.fromisoformat(stamp)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


@dataclass
class Candidate:
    job_id: str
    status: str
    bytes: int
    ended_at: str | None = None
    age_days: float | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "bytes": self.bytes,
            "ended_at": self.ended_at,
            "age_days": None if self.age_days is None else round(self.age_days, 2),
        }


@dataclass
class Skipped:
    job_id: str
    why: str

    def to_dict(self) -> dict[str, str]:
        return {"job_id": self.job_id, "why": self.why}


@dataclass
class CleanResult:
    dry_run: bool = False
    removed: list[Candidate] = field(default_factory=list)
    skipped: list[Skipped] = field(default_factory=list)
    incoming_removed: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def freed_bytes(self) -> int:
        return sum(c.bytes for c in self.removed)

    def to_dict(self) -> dict[str, object]:
        return {
            "dry_run": self.dry_run,
            "freed_bytes": self.freed_bytes,
            "removed": [c.to_dict() for c in self.removed],
            "skipped": [s.to_dict() for s in self.skipped],
            "incoming_removed": self.incoming_removed,
            "errors": self.errors,
        }


def candidates(
    *,
    all_finished: bool = False,
    older_than_days: float | None = None,
    now: datetime | None = None,
) -> tuple[list[Candidate], list[Skipped]]:
    """Which finished jobs' workdirs may be removed, and why the rest may not.

    Fails closed at every step: a job with no readable state, a job that is not
    finished, and (under `--older-than`) a job whose end time cannot be read are
    all skipped. A workdir is only ever removed because its own `state.json`
    says the job is over.
    """
    moment = now or datetime.now(UTC)
    picked: list[Candidate] = []
    skipped: list[Skipped] = []
    for job_id in jobs.list_job_ids():
        workdir = paths.workdir(job_id)
        if not workdir.is_dir():
            continue
        try:
            state = jobs.read_state(job_id)
        except (RuntimeError, FileNotFoundError, OSError):
            # No state means we cannot know this is not a live job.
            skipped.append(Skipped(job_id, "no readable state.json"))
            continue
        if not state.finished:
            skipped.append(Skipped(job_id, f"status {state.status}"))
            continue
        ended = _parse(state.ended_at)
        age_days = None if ended is None else (moment - ended).total_seconds() / 86400.0
        if older_than_days is not None:
            if age_days is None:
                skipped.append(Skipped(job_id, "finished but records no usable ended_at"))
                continue
            if age_days < older_than_days:
                skipped.append(Skipped(job_id, f"only {age_days:.1f} days old"))
                continue
        elif not all_finished:
            skipped.append(Skipped(job_id, "no selection given"))
            continue
        picked.append(
            Candidate(
                job_id=job_id,
                status=state.status,
                bytes=dir_size(workdir),
                ended_at=state.ended_at,
                age_days=age_days,
            )
        )
    return picked, skipped


def stale_incoming(now: float | None = None) -> list[Path]:
    """Staged spec files in `incoming/` that no submit can still be using.

    A file is left behind either by a submit that died between staging and
    deleting, or by one whose delete lost its connection. Two safe signatures:
    the job it names has finished (so the spec was consumed), or no job dir was
    ever created for it and the file is older than `INCOMING_STALE_S`.
    """
    directory = paths.home() / "incoming"
    if not directory.is_dir():
        return []
    moment = now if now is not None else datetime.now(UTC).timestamp()
    stale: list[Path] = []
    try:
        entries = sorted(directory.glob("*.json"))
    except OSError:
        return []
    for path in entries:
        job_id = path.stem
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        if paths.job_dir(job_id).is_dir():
            try:
                if jobs.read_state(job_id).finished:
                    stale.append(path)
            except (RuntimeError, FileNotFoundError, OSError):
                continue
            continue
        if moment - mtime > INCOMING_STALE_S:
            stale.append(path)
    return stale


def clean(
    *,
    all_finished: bool = False,
    older_than_days: float | None = None,
    dry_run: bool = False,
    now: datetime | None = None,
) -> CleanResult:
    picked, skipped = candidates(
        all_finished=all_finished, older_than_days=older_than_days, now=now
    )
    result = CleanResult(dry_run=dry_run, skipped=skipped)
    for candidate in picked:
        if dry_run:
            result.removed.append(candidate)
            continue
        try:
            remove_workdir(candidate.job_id)
        except OSError as exc:
            result.errors.append(f"{candidate.job_id}: could not remove workdir: {exc}")
            continue
        result.removed.append(candidate)
        try:
            jobs.update_state(candidate.job_id, workdir_removed=True)
        except (RuntimeError, OSError, KeyError) as exc:
            result.errors.append(
                f"{candidate.job_id}: workdir removed but state not updated: {exc}"
            )
    for path in stale_incoming():
        if not dry_run:
            try:
                path.unlink()
            except OSError as exc:
                result.errors.append(f"could not remove {path}: {exc}")
                continue
        result.incoming_removed.append(path.name)
    return result
