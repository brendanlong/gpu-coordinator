"""Workdir cleanup: the `cleanup:` policy the runner applies, and the
`python -m gpuc.host clean` sweep for what earlier runs left behind.

A finished job's `workdir` is nearly always the largest thing gpuc owns -- a
torch venv measures ~6.5 GB -- and it is the only part of a job dir that can be
recreated: `gpuc requeue` re-syncs it from git. `spec.json`, `state.json` and
`log.txt` are the record of what happened, and `clean` never touches them.

`purge` is the one thing here that does: it deletes a whole `jobs/<id>/` once
the job is old enough *and* its own `state.json` says both the record
(`meta_synced_at`) and whatever the job produced (`outputs_synced_at`) are
somewhere else. Without those, a purge is the only copy of a run's log and its
checkpoints going in the bin, which is why it is refused unless the caller
passes `--force` -- and then told, loudly, what it just did.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import stat
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from gpuc.host import baseline, jobs, paths, queue
from gpuc.host.jobs import ALWAYS, FINISHED_STATUSES, ON_SUCCESS

DEFAULT_RETENTION_DAYS = 7.0
"""How old a finished job must be before `purge` will consider it.

A week is long enough that "I'll look at that failure tomorrow" survives a
weekend, and the mirror precondition means nothing is actually lost either way.
"""

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
    meta_synced_at: str | None = None
    meta_synced_to: str | None = None
    forced: bool = False
    """Purged without a confirmed mirror or confirmed outputs, because the
    caller passed `--force`."""

    def to_dict(self) -> dict[str, object]:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "bytes": self.bytes,
            "ended_at": self.ended_at,
            "age_days": None if self.age_days is None else round(self.age_days, 2),
            "meta_synced_at": self.meta_synced_at,
            "meta_synced_to": self.meta_synced_to,
            "forced": self.forced,
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
    """Jobs whose `workdir/` went; the rest of the job dir stayed."""
    skipped: list[Skipped] = field(default_factory=list)
    purged: list[Candidate] = field(default_factory=list)
    """Jobs whose whole `jobs/<id>/` went, mirror confirmed (or forced)."""
    purge_skipped: list[Skipped] = field(default_factory=list)
    incoming_removed: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    s3_prefix: str | None = None
    """This host's mirror target, so the control side can say why nothing is
    backed up without a second round trip."""

    @property
    def freed_bytes(self) -> int:
        return sum(c.bytes for c in self.removed) + sum(c.bytes for c in self.purged)

    def to_dict(self) -> dict[str, object]:
        return {
            "dry_run": self.dry_run,
            "freed_bytes": self.freed_bytes,
            "removed": [c.to_dict() for c in self.removed],
            "skipped": [s.to_dict() for s in self.skipped],
            "purged": [c.to_dict() for c in self.purged],
            "purge_skipped": [s.to_dict() for s in self.purge_skipped],
            "incoming_removed": self.incoming_removed,
            "errors": self.errors,
            "s3_prefix": self.s3_prefix,
        }


def candidates(
    *,
    all_finished: bool = False,
    older_than_days: float | None = None,
    now: datetime | None = None,
    only: Iterable[str] | None = None,
) -> tuple[list[Candidate], list[Skipped]]:
    """Which finished jobs' workdirs may be removed, and why the rest may not.

    Fails closed at every step: a job with no readable state, a job that is not
    finished, and (under `--older-than`) a job whose end time cannot be read are
    all skipped. A workdir is only ever removed because its own `state.json`
    says the job is over.
    """
    moment = now or datetime.now(UTC)
    wanted = None if only is None else set(only)
    picked: list[Candidate] = []
    skipped: list[Skipped] = []
    for job_id in jobs.list_job_ids():
        if wanted is not None and job_id not in wanted:
            continue
        workdir = paths.workdir(job_id)
        if not workdir.is_dir():
            # Worth a line only when the caller named this job: otherwise it is
            # every job a previous clean already dealt with.
            if wanted is not None:
                skipped.append(Skipped(job_id, "workdir already gone"))
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
                meta_synced_at=state.meta_synced_at,
                meta_synced_to=state.meta_synced_to,
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
    only: Iterable[str] | None = None,
) -> CleanResult:
    picked, skipped = candidates(
        all_finished=all_finished, older_than_days=older_than_days, now=now, only=only
    )
    result = CleanResult(dry_run=dry_run, skipped=skipped, s3_prefix=host_s3_prefix())
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


# -- purge: the whole job dir, once its record lives somewhere else ------------


def host_s3_prefix() -> str | None:
    """This host's mirror target, or None if it has no `s3_prefix` at all."""
    try:
        return jobs.read_config().s3_prefix or None
    except (RuntimeError, OSError, ValueError):
        return None


def _holds_content(root: Path, entries: baseline.Entries) -> bool:
    """Is there anything under `root` that the job did not find already there?

    `baseline.has_new_content` asks the same question for `sync`, where a wrong
    "nothing here" costs an upload that can be retried. Here it decides whether
    a job dir may be deleted, so every way of not knowing has to count as
    content: a path that cannot be read, a symlink (which `aws s3 sync` follows
    and `rglob` does not), a walk that errors part way down.
    """
    try:
        info = root.lstat()
    except FileNotFoundError:
        return False
    except OSError:
        return True
    if stat.S_ISLNK(info.st_mode):
        return True
    if not stat.S_ISDIR(info.st_mode):
        return baseline.has_new_content(root, entries)
    unreadable = False

    def note(_: OSError) -> None:
        nonlocal unreadable
        unreadable = True

    files = 0
    for parent, dirs, names in os.walk(root, onerror=note):
        for name in (*dirs, *names):
            if Path(parent, name).is_symlink():
                return True
        files += len(names)
    return unreadable or files > len(baseline.unchanged(root, entries))


def produced_outputs(job_id: str, spec: jobs.JobSpec) -> bool:
    """Did any declared `outputs:` path actually gain content?

    Declaring an output is not producing one: a job that died before it wrote
    the path produced nothing, and neither did one whose output dir holds only
    files that came with the checkout. Fails closed in every direction --
    including an `outputs.path` this cannot even resolve, which nothing
    validates at submit -- because the point of asking is to avoid throwing
    away the only copy of a result.
    """
    workdir = paths.workdir(job_id)
    entries = baseline.read(job_id)
    for output in spec.outputs:
        try:
            key = baseline.output_key(output, job_id)
        except (KeyError, IndexError, ValueError):
            return True
        if _holds_content(workdir / key, entries.get(key, {})):
            return True
    return False


def outputs_confirmed(job_id: str, state: jobs.JobState) -> tuple[bool, str | None]:
    """Is everything this job produced known to be somewhere other than here?

    `outputs:` paths resolve *inside* `workdir/`, and a failed or cancelled job
    keeps its workdir by default, so a job that ended `failed: sync` can be
    holding the only copy of a checkpoint. Four ways to be satisfied: the final
    upload was confirmed, the spec declared no outputs, the workdir is already
    gone (whatever it held went with `cleanup:`, not with us), or the job never
    wrote its outputs in the first place. An unreadable spec cannot answer the
    question, so it fails closed.
    """
    if state.outputs_synced_at:
        return True, None
    try:
        spec = jobs.read_spec(job_id)
    except (RuntimeError, FileNotFoundError, OSError):
        return False, "spec.json is unreadable, so its outputs cannot be checked"
    if not spec.outputs:
        return True, None
    if not paths.workdir(job_id).is_dir():
        return True, None
    if not produced_outputs(job_id, spec):
        return True, None
    detail = " (the drain gave up: outputs_lost)" if state.outputs_lost else ""
    return False, f"outputs not confirmed uploaded{detail}"


def _not_backed_up(prefix: str | None) -> str:
    if prefix is None:
        return "not backed up: no s3_prefix on this host"
    return "not backed up: final upload failed"


def purge_candidates(
    *,
    older_than_days: float = DEFAULT_RETENTION_DAYS,
    now: datetime | None = None,
    force: bool = False,
    only: Iterable[str] | None = None,
) -> tuple[list[Candidate], list[Skipped]]:
    """Which finished jobs' whole directories may go, and why the rest may not.

    Fails closed exactly as `clean` does -- unreadable state, not finished, no
    usable `ended_at`, too young -- and then adds the two preconditions that
    make deleting the record itself safe: `meta_synced_at` (log and state are
    mirrored) and confirmed outputs. `force` overrides only those last two, and
    the candidate is marked `forced` so the report can say so.
    """
    moment = now or datetime.now(UTC)
    wanted = None if only is None else set(only)
    prefix = host_s3_prefix()
    picked: list[Candidate] = []
    skipped: list[Skipped] = []
    for job_id in jobs.list_job_ids():
        if wanted is not None and job_id not in wanted:
            continue
        try:
            state = jobs.read_state(job_id)
        except (RuntimeError, FileNotFoundError, OSError):
            skipped.append(Skipped(job_id, "no readable state.json"))
            continue
        if not state.finished:
            skipped.append(Skipped(job_id, f"status {state.status}"))
            continue
        ended = _parse(state.ended_at)
        age_days = None if ended is None else (moment - ended).total_seconds() / 86400.0
        # The age gate is how an unnamed job is chosen, so naming ids replaces
        # it rather than adding to it -- including for a job whose `ended_at`
        # never got written, which is exactly the stuck kind somebody names.
        # The preconditions below are not waived by naming anything.
        if wanted is None:
            if age_days is None:
                skipped.append(Skipped(job_id, "finished but records no usable ended_at"))
                continue
            if age_days < older_than_days:
                skipped.append(Skipped(job_id, f"only {age_days:.1f} days old"))
                continue
        reasons: list[str] = []
        if not state.meta_synced_at:
            reasons.append(_not_backed_up(prefix))
        confirmed, why = outputs_confirmed(job_id, state)
        if not confirmed and why:
            reasons.append(why)
        if reasons and not force:
            skipped.append(Skipped(job_id, "; ".join(reasons)))
            continue
        picked.append(
            Candidate(
                job_id=job_id,
                status=state.status,
                bytes=dir_size(paths.job_dir(job_id)),
                ended_at=state.ended_at,
                age_days=age_days,
                meta_synced_at=state.meta_synced_at,
                meta_synced_to=state.meta_synced_to,
                forced=bool(reasons),
            )
        )
    return picked, skipped


def remove_job_dir(job_id: str) -> int:
    """Delete `jobs/<id>/` and every stray trace of the job. Bytes freed."""
    directory = paths.job_dir(job_id)
    size = dir_size(directory) if directory.is_dir() else 0
    if directory.is_dir():
        shutil.rmtree(directory)
    # A finished job has no business in the queue, but a marker left by a
    # crash would make the next dispatcher launch a job with no spec.
    queue.remove_marker(job_id)
    with contextlib.suppress(OSError):
        paths.job_env_file(job_id).unlink(missing_ok=True)
    with contextlib.suppress(OSError):
        (paths.home() / "incoming" / f"{job_id}.json").unlink(missing_ok=True)
    return size


def purge(
    *,
    older_than_days: float = DEFAULT_RETENTION_DAYS,
    dry_run: bool = False,
    force: bool = False,
    now: datetime | None = None,
    only: Iterable[str] | None = None,
    sweep_only: Iterable[str] | None = None,
) -> CleanResult:
    """Remove whole job dirs, then run the ordinary workdir sweep over the rest.

    `--purge` implying `clean` is what makes one command enough: the jobs a
    purge refuses (no mirror, unconfirmed outputs) are exactly the ones whose
    workdirs are still worth reclaiming. `only` therefore narrows what may be
    *purged* and nothing else -- it exists for the control side's `--verify`,
    which cannot let an unverified job dir go but has no reason to keep its
    venv either.

    `sweep_only` narrows that implied sweep, and is what a user naming job ids
    wants: purging two jobs should not also reclaim every other finished job's
    venv. The two are separate because `--verify` needs both at once -- purge
    the ids whose mirror answered, sweep the ids the user asked about.
    """
    picked, skipped = purge_candidates(
        older_than_days=older_than_days, now=now, force=force, only=only
    )
    result = CleanResult(dry_run=dry_run, purge_skipped=skipped, s3_prefix=host_s3_prefix())
    for candidate in picked:
        if dry_run:
            result.purged.append(candidate)
            continue
        try:
            remove_job_dir(candidate.job_id)
        except OSError as exc:
            result.errors.append(f"{candidate.job_id}: could not remove the job dir: {exc}")
            continue
        result.purged.append(candidate)
    purged_ids = {candidate.job_id for candidate in result.purged}
    # Named ids replace the age gate here too, or `--purge --only X` would
    # reclaim less than a bare `clean --only X` does for a job whose state
    # never recorded when it ended.
    named = sweep_only is not None
    sweep = clean(
        all_finished=named,
        older_than_days=None if named else older_than_days,
        dry_run=dry_run,
        now=now,
        only=sweep_only,
    )
    # In a dry run the purged dirs are still there, so the sweep sees their
    # workdirs too; counting both would report the same bytes twice.
    result.removed = [c for c in sweep.removed if c.job_id not in purged_ids]
    result.skipped = [s for s in sweep.skipped if s.job_id not in purged_ids]
    result.incoming_removed = sweep.incoming_removed
    result.errors += sweep.errors
    return result
