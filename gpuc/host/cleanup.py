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

import array
import contextlib
import fcntl
import os
import shutil
import stat
import struct
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from gpuc.host import baseline, jobs, paths, queue
from gpuc.host.jobs import ALWAYS, FINISHED_STATUSES, ON_SUCCESS

DEFAULT_WORKDIR_DAYS = 1.0
"""What a host with no config of its own is given for `workdir_days`.

Here rather than on `HostConfig`, whose default stays null, because those are
different questions: a host being configured for the first time should reclaim
its venvs, and a host whose `config.json` predates the key should not start
deleting because somebody shipped it a newer package. `connect_host` applies
this one; nothing applies the other.
"""

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


def reported_workdir_bytes(job_id: str, state: jobs.JobState) -> int:
    """What deleting a finished job's workdir would free. Always a number.

    A job with no workdir left frees nothing, and answering that costs one
    `is_dir()`: it is the common case -- most jobs carry a `cleanup:` policy
    that takes the workdir the moment they end -- and it needs no recorded
    figure, which is what kept jobs that ended before the figure existed
    reading as "not sized yet" forever.

    A workdir still on disk is the only case worth walking, and only the first
    caller walks it: the runner normally recorded the figure as the job ended,
    and if it did not (its own death, or a job older than the field) the walk
    happens here and is written to `state.json` for the next reader. Nothing
    else needs to schedule it -- a host with no dispatcher running still gives
    a straight answer.

    A recorded zero against a workdir that is still there cannot be right, so
    it is re-measured rather than believed.
    """
    if not paths.workdir(job_id).is_dir():
        return 0
    return state.workdir_bytes or record_workdir_size(job_id)


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
    automatic: bool = False,
) -> tuple[list[Candidate], list[Skipped]]:
    """Which finished jobs' workdirs may be removed, and why the rest may not.

    Fails closed at every step: a job with no readable state, a job that is not
    finished, and (under `--older-than`) a job whose end time cannot be read are
    all skipped. A workdir is only ever removed because its own `state.json`
    says the job is over.

    `automatic` is the dispatcher sweeping on its own horizon rather than a
    person typing a delete, and it adds the two guards that only make sense
    when nobody is watching:

    - a job whose spec says `cleanup: never`, which is the one way to ask for
      a workdir to be kept and would otherwise mean "kept for a day";
    - a job whose `outputs:` are not confirmed to be anywhere else. Those paths
      live *inside* the workdir, so this is the difference between reclaiming a
      venv and binning the only copy of a checkpoint -- the same precondition
      `purge` fails closed on, and the one the `outputs not uploaded` warning in
      `gpuc status` is pointing at.

    A person can still take both with `gpuc clean --only <id>`, which is a
    delete somebody typed with the id in front of them.
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
        if automatic:
            try:
                policy = jobs.read_spec(job_id).cleanup
            except (RuntimeError, FileNotFoundError, OSError, ValueError):
                # An unreadable spec cannot say it wanted this kept, but it
                # cannot say it did not either.
                skipped.append(Skipped(job_id, "no readable spec.json"))
                continue
            if policy == jobs.NEVER:
                skipped.append(Skipped(job_id, "cleanup: never"))
                continue
            confirmed, why = outputs_confirmed(job_id, state)
            if not confirmed and why:
                skipped.append(Skipped(job_id, why))
                continue
        picked.append(
            Candidate(
                job_id=job_id,
                status=state.status,
                bytes=reclaimable_bytes(workdir),
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
    automatic: bool = False,
) -> CleanResult:
    picked, skipped = candidates(
        all_finished=all_finished,
        older_than_days=older_than_days,
        now=now,
        only=only,
        automatic=automatic,
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
            jobs.update_state(candidate.job_id, workdir_removed=True, workdir_bytes=0)
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
                bytes=reclaimable_bytes(paths.job_dir(job_id)),
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
    size = reclaimable_bytes(directory) if directory.is_dir() else 0
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
    automatic: bool = False,
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
        automatic=automatic,
    )
    # In a dry run the purged dirs are still there, so the sweep sees their
    # workdirs too; counting both would report the same bytes twice.
    result.removed = [c for c in sweep.removed if c.job_id not in purged_ids]
    result.skipped = [s for s in sweep.skipped if s.job_id not in purged_ids]
    result.incoming_removed = sweep.incoming_removed
    result.errors += sweep.errors
    return result
