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
"""Fastest gzip. The archive is written while `gpuc submit` waits, and a
checkout is mostly source that level 1 already shrinks most of the way."""


def archive(workdir: Path, dest: Path) -> None:
    """Write `workdir`'s tree to `dest`, symlinks as symlinks."""
    partial = dest.with_name(f".{dest.name}.partial")
    with tarfile.open(partial, "w:gz", compresslevel=COMPRESSLEVEL) as tar:
        tar.add(workdir, arcname=".")
    os.replace(partial, dest)


def restore(job_id: str) -> str | None:
    """Make sure the job has a workdir, unpacking its archive if it has none.

    None when there is one: already there (a preempted job reuses the tree its
    last attempt left, as on any host), or unpacked just now. Otherwise why
    there is none -- the job cannot run without its code.

    Unpacked beside the workdir and renamed into place, so a restore cut short
    leaves no half a checkout that the next attempt would take for a whole one.
    """
    workdir = paths.workdir(job_id)
    if workdir.is_dir():
        return None
    source = paths.checkout_archive(job_id)
    if not source.is_file():
        if paths.scratch_dir() is None:
            return None
        return f"no workdir at {workdir} and no archived checkout to restore it from"
    workdir.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    partial = workdir.with_name(f".{workdir.name}.partial-{os.getpid()}")
    shutil.rmtree(partial, ignore_errors=True)
    try:
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
        shutil.rmtree(partial, ignore_errors=True)
        return f"could not restore the checkout from {source}: {exc}"
    return None
