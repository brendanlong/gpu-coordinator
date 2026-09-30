"""The archived checkout of a host whose workdirs live on scratch.

A host with `GPUC_SCRATCH_DIR` keeps its queue on gpuc home and its workdirs
somewhere faster that a restart may wipe. What makes that safe is a copy of
each job's checkout, as submitted, beside its spec: the runner unpacks it
whenever the workdir is not there, so a job queued before the restart, or
running during it, still has its code afterwards.
"""

from __future__ import annotations

import os
import shutil
import tarfile
from pathlib import Path

from gpuc.host import paths

COMPRESSLEVEL = 1
"""Fastest gzip. A checkout is mostly source that level 1 already shrinks
most of the way, and the archive is written while `gpuc submit` waits."""

ARCHIVE_NAME = "checkout.tar.gz"


def archive_staged(job_id: str) -> Path:
    """Archive a staged job's workdir beside it and remove the workdir, for
    `gpuc submit` to ask for before the enqueue.

    Its own step with its own timeout, not part of the enqueue: a large
    checkout on a slow volume can take minutes to read, and an enqueue the
    client gave up on while the host went on to accept the job would be a
    job the client reports as failed and the host runs anyway. Done again,
    it finds the archive and does nothing.
    """
    staged = paths.incoming_job_dir(job_id)
    dest = staged / ARCHIVE_NAME
    workdir = staged / "workdir"
    if not dest.is_file():
        archive(workdir, dest)
    shutil.rmtree(workdir, ignore_errors=True)
    return dest


def archive(workdir: Path, dest: Path) -> None:
    """Write `workdir`'s tree to `dest`, symlinks as symlinks."""
    partial = dest.with_name(f".{dest.name}.partial")
    with tarfile.open(partial, "w:gz", compresslevel=COMPRESSLEVEL) as tar:
        tar.add(workdir, arcname=".")
    os.replace(partial, dest)


def ensure_scratch(scratch: Path) -> None:
    """Create scratch 0700, or make it so if it is ours; refuse one that is
    not.

    Scratch holds every checkout, and on a shared box a directory somebody
    else owns, or can write to, is one where they could plant the workdir a
    job is about to run. gpuc home is kept 0700 for the same reason
    (`paths.ensure_layout`).
    """
    scratch.mkdir(mode=0o700, parents=True, exist_ok=True)
    if scratch.stat().st_uid != os.getuid():
        raise PermissionError(f"scratch {scratch} is not owned by this user")
    scratch.chmod(0o700)


def restore(job_id: str) -> str | None:
    """Make sure the job has a workdir, unpacking its archive if it has none.

    None when there is one: already there (a preempted job reuses the tree its
    last attempt left, as on any host), or unpacked just now. Otherwise why
    there is none -- the job cannot run without its code.

    Unpacked beside the workdir and renamed into place, so a restore cut short
    leaves no half a checkout that the next attempt would take for a whole one.
    """
    scratch = paths.scratch_dir()
    workdir = paths.workdir(job_id)
    try:
        if scratch is not None:
            ensure_scratch(scratch)
        if workdir.is_dir():
            return None
        source = paths.checkout_archive(job_id)
        if not source.is_file():
            if scratch is None:
                return None
            return f"no workdir at {workdir} and no archived checkout to restore it from"
        partial = paths.partial_workdir(job_id)
        shutil.rmtree(partial, ignore_errors=True)
        with tarfile.open(source, "r:gz") as tar:
            # Our own archive of the submitter's tree: extract it exactly as
            # rsync delivered it, links included. 3.11 before 3.11.4 has no
            # filters and does the same.
            if hasattr(tarfile, "fully_trusted_filter"):
                tar.extractall(partial, filter="fully_trusted")
            else:
                tar.extractall(partial)
        os.rename(partial, workdir)
    except (OSError, tarfile.TarError) as exc:
        if scratch is not None:
            shutil.rmtree(paths.partial_workdir(job_id), ignore_errors=True)
        return f"could not restore the checkout to {workdir}: {exc}"
    return None
