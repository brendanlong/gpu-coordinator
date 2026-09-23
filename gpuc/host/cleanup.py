"""Workdir cleanup: the `cleanup:` policy the runner applies, and the
`python -m gpuc.host clean` sweep for what earlier runs left behind.

A finished job's `workdir` is nearly always the largest thing gpuc owns -- a
torch venv measures ~6.5 GB -- and it is the only part of a job dir that can be
recreated: `gpuc requeue` re-syncs it from git. `spec.json`, `state.json` and
`log.txt` are the record of what happened, and `clean` never touches them.

`purge` is the one thing here that does: it deletes a whole `jobs/<id>/` once
the job is old enough *and* its own `state.json` upload records say both the
record (the mirror) and whatever the job produced (every output destination)
are somewhere else. Without those, a purge is the only copy of a run's log and its
checkpoints going in the bin, which is why it is refused unless the caller
passes `--force` -- and then told, loudly, what it just did.
"""

from __future__ import annotations

import array
import contextlib
import fcntl
import os
import shutil
import stat
import struct
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gpuc.host import baseline, jobs, paths
from gpuc.host.jobs import ALWAYS, FINISHED_STATUSES, ON_SUCCESS, JobSpec, JobState

DEFAULT_WORKDIR_DAYS = 1.0
"""What `connect` writes for `workdir_days` into a host's very first config.

Here rather than on `HostConfig`, whose default stays null: a host being
configured for the first time should reclaim its venvs, and nothing that
merely reads a config may turn a sweep on. See `HostConfig.workdir_days`.
"""

DEFAULT_RETENTION_DAYS = 7.0
"""How old a finished job must be before `purge` will consider it.

A week is long enough that "I'll look at that failure tomorrow" survives a
weekend, and the mirror precondition means nothing is actually lost either way.
"""

INCOMING_STALE_S = 3600.0
"""How old a job dir under `incoming/` must be before it is removed.

`submit` rsyncs the workdir there and then runs `enqueue`, which renames the
dir into `jobs/`; a dir still there is a submit that died. Measured from the
dir's last change, so a large rsync still in progress is never mistaken for
one that stopped. An hour is far past any ssh round trip.
"""


def remove_secrets(job_id: str) -> None:
    """Delete the job's secrets file, if any. The one place that does.

    Called by the runner as its last act -- after the final upload and the
    mirror, which authenticate with what the file holds -- and by the sweeps
    for a job whose dir is going or whose submit never finished.
    """
    with contextlib.suppress(OSError):
        paths.job_env_file(job_id).unlink(missing_ok=True)


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


FS_IOC_FIEMAP = 0xC020660B
FIEMAP_EXTENT_LAST = 0x0001
FIEMAP_EXTENT_SHARED = 0x2000
_FIEMAP_HEADER = 32
"""struct fiemap: u64 start, u64 length, u32 flags, u32 mapped, u32 count, u32 pad."""
_FIEMAP_EXTENT = 56
"""struct fiemap_extent: u64 logical, physical, length, 2x reserved64, u32 flags, 3x pad."""
_FIEMAP_BATCH = 128


def shared_extent_bytes(path: str) -> int:
    """Bytes of `path` whose extents another file also references.

    A reflink -- uv's `clone` link mode, and what it uses on a filesystem that
    supports it -- shares extents without sharing an inode, so `st_nlink` is 1
    and nothing about the file says its bytes are also the cache's. FIEMAP is
    the only thing that does, and `FIEMAP_EXTENT_SHARED` is the flag for it.

    Zero for a filesystem that cannot answer (ext4 has no reflinks to find,
    tmpfs has no FIEMAP at all), for a file we may not open, and for anything
    that goes wrong: every failure here means "assume it is all yours", which
    over-reports what a delete frees rather than under-reporting it.

    One open/ioctl/close per file, which is why this is asked once per finished
    job and recorded in `state.json` rather than on every `status`: it triples
    the walk (1.60 s against 0.54 s over a 15.00 GiB venv of 67520 files), and
    a walk that happens once can afford to be exact. See `JobState.workdir_bytes`.
    """
    try:
        # O_NOFOLLOW because the walk never follows one: a symlink's own
        # st_blocks can be non-zero (ext4 stores a long target out of line), so
        # without this the sweep opens whatever path a job happened to point at
        # -- a device node, or a file on a hung mount that O_NONBLOCK will not
        # save us from. The caller checks S_ISREG too; this is the backstop.
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
    except OSError:
        return 0
    try:
        shared = 0
        start = 0
        buf = array.array("b", bytes(_FIEMAP_HEADER + _FIEMAP_EXTENT * _FIEMAP_BATCH))
        # Bounded rather than `while True`: a filesystem that keeps answering
        # without ever setting LAST or advancing must not hang the sweep. A file
        # past the cap is under-counted as shared, never over-counted.
        for _ in range(64):
            # Only the header needs resetting; `fm_mapped_extents` says how much
            # of the rest the kernel wrote.
            struct.pack_into("=QQIIII", buf, 0, start, 1 << 62, 0, 0, _FIEMAP_BATCH, 0)
            try:
                fcntl.ioctl(fd, FS_IOC_FIEMAP, buf, True)
            except OSError:
                return 0
            mapped: int = struct.unpack_from("=I", buf, 20)[0]
            if not mapped:
                return shared
            done = False
            for index in range(mapped):
                at = _FIEMAP_HEADER + index * _FIEMAP_EXTENT
                logical, _physical, length = struct.unpack_from("=QQQ", buf, at)
                flags: int = struct.unpack_from("=I", buf, at + 40)[0]
                if flags & FIEMAP_EXTENT_SHARED:
                    shared += length
                if flags & FIEMAP_EXTENT_LAST:
                    done = True
                next_start = logical + length
                if next_start <= start:
                    done = True
                start = next_start
            if done:
                return shared
        return shared
    finally:
        os.close(fd)


def _walk_size(root: Path, *, reclaimable_only: bool) -> int:
    """Sum `st_blocks` under `root`, counting each inode once.

    `st_blocks`, not `st_size`, so a sparse or compressed file is counted as it
    actually sits on disk -- the same thing `du` counts.

    `reclaimable_only` is what separates the two callers. See the wrappers.
    """
    total = 0
    seen: set[tuple[int, int]] = set()
    # inode -> (blocks, links found in this tree, links the filesystem has)
    shared: dict[tuple[int, int], tuple[int, int, int]] = {}
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
            # A directory's st_nlink counts its subdirectories, not other names
            # for it -- nothing can hardlink one -- so it is never "shared".
            if entry.is_dir(follow_symlinks=False):
                stack.append(Path(entry.path))
                total += info.st_blocks * 512
                continue
            if info.st_nlink <= 1:
                # One link is one path: this cannot be the same file twice.
                blocks = info.st_blocks * 512
                # Regular files only: a symlink has no extents worth asking
                # about, and asking means opening what it points at.
                if reclaimable_only and blocks and stat.S_ISREG(info.st_mode):
                    blocks -= min(blocks, shared_extent_bytes(entry.path))
                total += blocks
                continue
            key = (info.st_dev, info.st_ino)
            if not reclaimable_only:
                if key not in seen:
                    seen.add(key)
                    total += info.st_blocks * 512
                continue
            blocks, found, _ = shared.get(key, (info.st_blocks, 0, info.st_nlink))
            shared[key] = (blocks, found + 1, info.st_nlink)
    total += sum(blocks * 512 for blocks, found, nlink in shared.values() if found >= nlink)
    with contextlib.suppress(OSError):
        total += root.stat().st_blocks * 512
    return total


def dir_size(root: Path) -> int:
    """Disk usage of `root` in bytes, the way `du` counts it.

    How much space this tree occupies, whoever else has a name for it. That is
    the question to ask about the uv cache, whose whole job is to hold bytes
    other trees link to.
    """
    return _walk_size(root, reclaimable_only=False)


def reclaimable_bytes(root: Path) -> int:
    """Bytes deleting `root` would actually give back to the filesystem.

    Where this parts company with `du` is the bytes something outside the tree
    also has. uv materialises a venv out of its wheel cache, so most of a
    6.5 GB torch venv is bytes the cache still holds when the workdir goes.
    Measured: 8.36 GiB by `du`, 0.13 GiB actually returned on one host; 15.00
    GiB by `du`, 0.34 GiB returned on another. Reporting the `du` figure makes
    every `status` disk line and every `clean` report an overstatement nobody
    can act on.

    uv shares two ways and this has to see both. **Hardlinks** (its `hardlink`
    link mode) are exact and free: `st_nlink` comes with the `stat` the walk
    already does, so a file counts only once every one of its links has been
    found inside this tree, and only multiply-linked inodes are held in memory
    until the end -- a single-link file cannot be reached twice, so it is
    counted on sight and never remembered. **Reflinks** (its `clone` mode,
    where the filesystem supports it) share extents *without* sharing an inode,
    so `st_nlink` is 1 and nothing about the file says its bytes are also the
    cache's; only FIEMAP can, at one ioctl per file.

    That ioctl is the expensive part, and the reason this is measured once per
    job rather than on demand: nothing cheaper is exact. The kernel will tell
    you an extent is shared (FIEMAP) or who else references it (btrfs
    `LOGICAL_INO`, a backref walk that costs *more*), and a filesystem-wide
    scan (btrfs `TREE_SEARCH_V2`, XFS `GETFSMAP`) is O(extents on the device).
    Only btrfs qgroups answer in O(1), and only per subvolume, with quotas on.
    So: pay it once, exactly, and write the number down.

    Which mode a host gets is the host's business, not ours -- the same uv
    against the same cache hardlinks on one box and reflinks on the next -- so
    a figure that saw only one of them would be right on one host and off by
    fifteen gigabytes on another.

    The error is one-sided per *file* -- anything unanswerable counts as yours
    -- but the figure as a whole can still come out low, because
    `FIEMAP_EXTENT_SHARED` means "shared with something", not "shared with
    something outside this tree". Learning who the other referrer is costs a
    backref walk per extent, which is more than the whole measurement. So:

    - Sharing *within* the tree reads as sharing out of it. A workdir that
      reflinks a dataset into a second copy of itself is told it frees neither.
    - On a snapshotted filesystem (btrfs with snapper or timeshift), every
      extent in every workdir is shared with the snapshot, and this reports
      little more than the directories. That is arguably honest -- the delete
      really does free nothing until the snapshot expires -- but it is not what
      anyone reading a `clean` report expects.
    - Two sibling workdirs holding the last two links to one file are each told
      they free nothing, though deleting both would. `du` splits that pair the
      same way when asked about them separately.
    - A file both hardlinked wholly within this tree and reflinked out of it
      counts in full: the hardlink branch does not go on to ask FIEMAP.
    """
    return _walk_size(root, reclaimable_only=True)


def workdir_size(job_id: str) -> int | None:
    """Bytes held by a job's workdir, or None if it has none left."""
    workdir = paths.workdir(job_id)
    if not workdir.is_dir():
        return None
    return reclaimable_bytes(workdir)


def record_workdir_size(job_id: str) -> int:
    """Measure a finished job's workdir once and write the figure to its state.

    The walk is exact and therefore not cheap, so the figure is written down
    where the next reader finds it instead of being walked again. A failure to
    record is not worth failing anything over: the figure is a disk report, and
    the caller still gets the number it asked for.

    The workdir is re-checked *after* the walk because the walk takes seconds
    and `gpuc clean` is a different process. Without this, a clean landing in
    that window is overwritten with the figure for a workdir that no longer
    exists -- and `status` then advertises disk that `clean` cannot free,
    because a job with no workdir is not a candidate for anything.
    """
    size = workdir_size(job_id)
    if size is not None and not paths.workdir(job_id).is_dir():
        size = None
    recorded = 0 if size is None else size
    with contextlib.suppress(RuntimeError, OSError, KeyError):
        jobs.update_state(job_id, workdir_bytes=recorded)
    return recorded


MEASURING_BUDGET_S = 10.0
"""How long one `status` call will spend walking workdirs nobody measured.

The control side gives `status` 60 seconds (`control/status.py`), and a host
that blew it would not render as "some figures are missing" -- it would render
as an error line with no queue, no running jobs and no cards, and
`provision`'s reusable-host check would read the same timeout as unreachable.
So the walking is bounded rather than paced: every walk that finishes inside
the budget is written down permanently, so successive calls converge on a fully
measured host instead of re-doing the same work.
"""


def reported_workdir_bytes(
    job_id: str, state: jobs.JobState, *, deadline: float | None = None
) -> int | None:
    """What deleting a finished job's workdir would free.

    A job with no workdir left frees nothing, and answering that costs one
    `is_dir()`: it is the common case -- most jobs carry a `cleanup:` policy
    that takes the workdir the moment they end -- and it needs no recorded
    figure, which is what kept jobs that ended before the figure existed
    reading as "not sized yet" forever.

    A workdir still on disk is the only case worth walking, and normally
    nobody here walks it either: the runner recorded the figure as the job
    ended. The walk happens here when nothing did (the runner's own death, or a
    job older than the field), and what it finds goes to `state.json` for the
    next reader, so a host with no dispatcher running still gives a straight
    answer -- and gives it once, unless writing it down failed too, in which
    case the walk is repeated rather than the answer withheld.

    A recorded zero is believed. It is the usual figure for a workdir that has
    gone, but it is also a real measurement of one that has not: a tree whose
    extents are all shared with something outside it reclaims nothing, and its
    directories add nothing on a filesystem that reports no blocks for them.
    Reading that as "unmeasured" would walk the same tree on every call.

    Null only for the straggler that arrives after `deadline`, which is a
    figure this call declined to go and get rather than one that does not
    exist. `MEASURING_BUDGET_S` says why there is a deadline at all.
    """
    if not paths.workdir(job_id).is_dir():
        return 0
    recorded = state.workdir_bytes
    if recorded is not None:
        return recorded
    if deadline is not None and time.monotonic() > deadline:
        return None
    return record_workdir_size(job_id)


def remove_workdir(job_id: str, *, measured: int | None = None) -> int:
    """Delete `jobs/<id>/workdir` and nothing else. Returns the bytes freed.

    `measured` is a figure the caller already walked for, which `clean` always
    has: measuring is now an ioctl per file, and doing it twice to delete once
    is most of the cost of `gpuc clean --all-finished`.
    """
    workdir = paths.workdir(job_id)
    if not workdir.is_dir():
        return 0
    size = reclaimable_bytes(workdir) if measured is None else measured
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
    mirrored_at: str | None = None
    """When the job's log and state last reached its mirror, and where."""
    mirror: str | None = None
    forced: bool = False
    """Purged without a confirmed mirror or confirmed outputs, because the
    caller passed `--force`."""

    @staticmethod
    def of(state: JobState, **fields: Any) -> Candidate:
        mirror = state.mirror if state.mirrored else None
        return Candidate(
            status=state.status,
            ended_at=state.ended_at,
            mirrored_at=mirror.ok_at if mirror else None,
            mirror=mirror.to if mirror else None,
            **fields,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "bytes": self.bytes,
            "ended_at": self.ended_at,
            "age_days": None if self.age_days is None else round(self.age_days, 2),
            "mirrored_at": self.mirrored_at,
            "mirror": self.mirror,
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


@dataclass(frozen=True)
class Evidence:
    """What the caller can vouch for, beyond what the job's own records say.

    Every delete goes through `may_delete` with one of these; the flags a
    person types and the fact that nobody typed anything are both evidence,
    not separate policies.
    """

    automatic: bool = False
    """Nobody typed this: the dispatcher sweeping on its own horizon."""
    policy: bool = False
    """The job's own `cleanup:` policy, applied by its runner as it ends. The
    only evidence under which `on_success` and `always` are read at all: a
    sweep honours `never` and nothing else about the policy, since the horizon
    is the operator's decision and the policy was the submitter's."""
    force: bool = False
    """A person waives the backup preconditions, and is told so per job."""
    verified: frozenset[str] | None = None
    """Job ids whose mirror the caller listed itself. When given, a job must be
    in it *and* have the host's own record of a successful final upload: the
    listing proves the log is there, the record proves it is the final one --
    every periodic tick mirrors the log too, so a listing alone would vouch
    for a job whose last upload failed."""


ASKED = Evidence()
"""A person typed the command and waived nothing."""

WORKDIR = "workdir"
JOBDIR = "jobdir"


def may_delete(job_id: str, state: JobState, what: str, evidence: Evidence) -> str | None:
    """Why this job's `what` may not be deleted, or None if it may.

    Fails closed at every step. `workdir/` is recreatable and only ever
    deleted because the job's state says it is over -- the runner asks with
    the status it is about to write, since the workdir goes before that write
    and the outputs it holds go with it. A delete nobody typed adds two
    refusals: the spec's `cleanup:` (in full for the runner applying it, and
    `never` alone for a sweep), and outputs still pending, since those paths
    live *inside* the workdir. The whole `jobs/<id>/` is the record of a run,
    so it needs that record mirrored (or vouched for) and the outputs
    confirmed, unless a person forces it. A person naming a job with `gpuc
    clean --only <id>` gets its workdir with no further questions: that is a
    delete typed with the id in front of them. An unreadable spec cannot say
    what it wanted kept, so it keeps everything.
    """
    if not state.finished:
        return f"status {state.status}"
    if what == WORKDIR and not (evidence.automatic or evidence.policy):
        return None
    try:
        spec = jobs.read_spec(job_id)
    except (RuntimeError, ValueError):
        return "no readable spec.json"
    if what == WORKDIR:
        if evidence.policy and not should_remove(spec.cleanup, state.status):
            return f"cleanup: {spec.cleanup}"
        if evidence.automatic and spec.cleanup == jobs.NEVER:
            return "cleanup: never"
        return outputs_pending(job_id, spec, state)
    reasons: list[str] = []
    if not state.mirrored:
        reasons.append(_not_backed_up(host_s3_prefix()))
    elif evidence.verified is not None and job_id not in evidence.verified:
        reasons.append("not backed up: the mirror has no log for it")
    pending = outputs_pending(job_id, spec, state)
    if pending:
        reasons.append(pending)
    if reasons and not evidence.force:
        return "; ".join(reasons)
    return None


def candidates(
    *,
    all_finished: bool = False,
    older_than_days: float | None = None,
    now: datetime | None = None,
    only: Iterable[str] | None = None,
    evidence: Evidence = ASKED,
) -> tuple[list[Candidate], list[Skipped]]:
    """Which finished jobs' workdirs may be removed, and why the rest may not.

    Selection first -- the ids named, else the age horizon, else every
    finished job -- then `may_delete` on each.
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
        finished = _finished_age(job_id, moment)
        if isinstance(finished, Skipped):
            skipped.append(finished)
            continue
        state, age_days = finished
        if older_than_days is not None:
            too_young = _too_young(job_id, age_days, older_than_days)
            if too_young:
                skipped.append(too_young)
                continue
        elif not all_finished:
            skipped.append(Skipped(job_id, "no selection given"))
            continue
        why = may_delete(job_id, state, WORKDIR, evidence)
        if why:
            skipped.append(Skipped(job_id, why))
            continue
        picked.append(
            Candidate.of(state, job_id=job_id, bytes=reclaimable_bytes(workdir), age_days=age_days)
        )
    return picked, skipped


def _finished_age(job_id: str, moment: datetime) -> tuple[JobState, float | None] | Skipped:
    """A finished job's state and its age in days, or why it is not a candidate.

    No readable state means we cannot know this is not a live job.
    """
    try:
        state = jobs.read_state(job_id)
    except RuntimeError:
        return Skipped(job_id, "no readable state.json")
    if not state.finished:
        return Skipped(job_id, f"status {state.status}")
    ended = _parse(state.ended_at)
    return state, None if ended is None else (moment - ended).total_seconds() / 86400.0


def _too_young(job_id: str, age_days: float | None, older_than_days: float) -> Skipped | None:
    if age_days is None:
        return Skipped(job_id, "finished but records no usable ended_at")
    if age_days < older_than_days:
        return Skipped(job_id, f"only {age_days:.1f} days old")
    return None


def stale_incoming(now: float | None = None) -> list[Path]:
    """Job dirs under `incoming/` that no submit can still be building.

    Left behind by a submit that died between the rsync and the enqueue. Old
    by the dir's own mtime, which every file added to it refreshes.
    """
    directory = paths.incoming_dir()
    if not directory.is_dir():
        return []
    moment = now if now is not None else datetime.now(UTC).timestamp()
    stale: list[Path] = []
    try:
        entries = sorted(p for p in directory.iterdir() if p.is_dir())
    except OSError:
        return []
    for path in entries:
        try:
            changed = max(p.stat().st_mtime for p in [path, *path.rglob("*")])
        except OSError:
            continue
        if moment - changed > INCOMING_STALE_S:
            stale.append(path)
    return stale


def remove_stale_incoming(now: float | None = None) -> tuple[list[str], list[str]]:
    """Delete what `stale_incoming` found: the names of what went, and what
    could not be removed. The secrets file goes with the dir: it was delivered
    before the enqueue that never came, and nothing else would unlink it."""
    removed: list[str] = []
    errors: list[str] = []
    for path in stale_incoming(now):
        try:
            shutil.rmtree(path)
        except OSError as exc:
            errors.append(f"could not remove {path}: {exc}")
            continue
        remove_secrets(path.name)
        removed.append(path.name)
    return removed, errors


def clean(
    *,
    all_finished: bool = False,
    older_than_days: float | None = None,
    dry_run: bool = False,
    now: datetime | None = None,
    only: Iterable[str] | None = None,
    evidence: Evidence = ASKED,
) -> CleanResult:
    picked, skipped = candidates(
        all_finished=all_finished,
        older_than_days=older_than_days,
        now=now,
        only=only,
        evidence=evidence,
    )
    result = CleanResult(dry_run=dry_run, skipped=skipped, s3_prefix=host_s3_prefix())
    for candidate in picked:
        if dry_run:
            result.removed.append(candidate)
            continue
        try:
            remove_workdir(candidate.job_id, measured=candidate.bytes)
        except OSError as exc:
            result.errors.append(f"{candidate.job_id}: could not remove workdir: {exc}")
            continue
        result.removed.append(candidate)
        try:
            jobs.update_state(candidate.job_id, workdir_bytes=0)
        except (RuntimeError, OSError, KeyError) as exc:
            result.errors.append(
                f"{candidate.job_id}: workdir removed but state not updated: {exc}"
            )
    if dry_run:
        result.incoming_removed = [path.name for path in stale_incoming()]
    else:
        result.incoming_removed, errors = remove_stale_incoming()
        result.errors += errors
    return result


# -- purge: the whole job dir, once its record lives somewhere else ------------


def host_s3_prefix() -> str | None:
    """This host's mirror target, or None if it has no `s3_prefix` at all."""
    try:
        return jobs.read_config().s3_prefix or None
    except (RuntimeError, OSError, ValueError):
        return None


def outputs_pending(job_id: str, spec: JobSpec, state: JobState) -> str | None:
    """Why this job's outputs are only on this host, or None if nothing here
    needs saving.

    The one question behind a rental's drain, the runner's decision to keep a
    job's secrets for that drain, `purge`, the automatic workdir sweep and the
    flag `status` shows. `outputs:` paths resolve *inside* `workdir/`, and a
    failed or cancelled job keeps its workdir by default, so a job that ended
    `failed: sync` can be holding the only copy of a checkpoint. Nothing is
    pending when the spec declares no outputs, the upload records say every
    destination has the last upload, the workdir is already gone (whatever it
    held went with `cleanup:`, not with us), or nothing was ever written under
    the declared paths -- declaring an output is not producing one, and a job
    that died in its preflight, or whose output dir holds only files that came
    with the checkout, produced nothing. Every way of not knowing counts as
    content, an `outputs.path` that cannot even be resolved included, because
    the point of asking is to avoid throwing away the only copy of a result.
    """
    if not spec.outputs or state.outputs_uploaded(spec):
        return None
    if not paths.workdir(job_id).is_dir():
        return None
    if not _produced_outputs(job_id, spec):
        return None
    detail = " (the drain gave up: outputs_lost)" if state.outputs_lost else ""
    return f"outputs not confirmed uploaded{detail}"


def _produced_outputs(job_id: str, spec: JobSpec) -> bool:
    workdir = paths.workdir(job_id)
    entries = baseline.read(job_id)
    for output in spec.outputs:
        try:
            key = baseline.output_key(output, job_id)
        except (KeyError, IndexError, ValueError):
            return True
        if baseline.has_new_content(workdir / key, entries.get(key, {})):
            return True
    return False


def _not_backed_up(prefix: str | None) -> str:
    if prefix is None:
        return "not backed up: no s3_prefix on this host"
    return "not backed up: final upload failed"


def purge_candidates(
    *,
    older_than_days: float = DEFAULT_RETENTION_DAYS,
    now: datetime | None = None,
    only: Iterable[str] | None = None,
    evidence: Evidence = ASKED,
) -> tuple[list[Candidate], list[Skipped]]:
    """Which finished jobs' whole directories may go, and why the rest may not.

    The same selection as `candidates` -- named ids replace the age gate,
    including for a job whose `ended_at` never got written, which is exactly
    the stuck kind somebody names -- then `may_delete` on the job dir. A
    forced candidate is marked so the report can say so.
    """
    moment = now or datetime.now(UTC)
    wanted = None if only is None else set(only)
    picked: list[Candidate] = []
    skipped: list[Skipped] = []
    for job_id in jobs.list_job_ids():
        if wanted is not None and job_id not in wanted:
            continue
        finished = _finished_age(job_id, moment)
        if isinstance(finished, Skipped):
            skipped.append(finished)
            continue
        state, age_days = finished
        if wanted is None:
            too_young = _too_young(job_id, age_days, older_than_days)
            if too_young:
                skipped.append(too_young)
                continue
        why = may_delete(job_id, state, JOBDIR, evidence)
        if why:
            skipped.append(Skipped(job_id, why))
            continue
        # Forced means: without the waiver it would have been refused -- judged
        # against the same mirror evidence, so `--verify --force` still marks a
        # job the mirror has no log for.
        unforced = Evidence(verified=evidence.verified)
        forced = evidence.force and may_delete(job_id, state, JOBDIR, unforced) is not None
        picked.append(
            Candidate.of(
                state,
                job_id=job_id,
                bytes=reclaimable_bytes(paths.job_dir(job_id)),
                age_days=age_days,
                forced=forced,
            )
        )
    return picked, skipped


def remove_job_dir(job_id: str) -> int:
    """Delete `jobs/<id>/` and every stray trace of the job. Bytes freed."""
    directory = paths.job_dir(job_id)
    size = reclaimable_bytes(directory) if directory.is_dir() else 0
    if directory.is_dir():
        shutil.rmtree(directory)
    remove_secrets(job_id)
    return size


def purge_job_dirs(
    *,
    older_than_days: float = DEFAULT_RETENTION_DAYS,
    dry_run: bool = False,
    now: datetime | None = None,
    only: Iterable[str] | None = None,
    evidence: Evidence = ASKED,
) -> CleanResult:
    """Remove whole job dirs, and nothing else. `purge` is this plus the sweep;
    the dispatcher, which sweeps on its own horizon anyway, calls this."""
    picked, skipped = purge_candidates(
        older_than_days=older_than_days, now=now, only=only, evidence=evidence
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
    return result


def purge(
    *,
    older_than_days: float = DEFAULT_RETENTION_DAYS,
    dry_run: bool = False,
    now: datetime | None = None,
    only: Iterable[str] | None = None,
    evidence: Evidence = ASKED,
) -> CleanResult:
    """Remove whole job dirs, then run the ordinary workdir sweep over the rest.

    `--purge` implying `clean` is what makes one command enough: the jobs a
    purge refuses (no mirror, unconfirmed outputs) are exactly the ones whose
    workdirs are still worth reclaiming. `only` scopes both: purging two jobs
    should not also reclaim every other finished job's venv.
    """
    result = purge_job_dirs(
        older_than_days=older_than_days, dry_run=dry_run, now=now, only=only, evidence=evidence
    )
    purged_ids = {candidate.job_id for candidate in result.purged}
    # Named ids replace the age gate here too, or `--purge --only X` would
    # reclaim less than a bare `clean --only X` does for a job whose state
    # never recorded when it ended.
    named = only is not None
    sweep = clean(
        all_finished=named,
        older_than_days=None if named else older_than_days,
        dry_run=dry_run,
        now=now,
        only=only,
        evidence=evidence,
    )
    # In a dry run the purged dirs are still there, so the sweep sees their
    # workdirs too; counting both would report the same bytes twice.
    result.removed = [c for c in sweep.removed if c.job_id not in purged_ids]
    result.skipped = [s for s in sweep.skipped if s.job_id not in purged_ids]
    result.incoming_removed = sweep.incoming_removed
    result.errors += sweep.errors
    return result
