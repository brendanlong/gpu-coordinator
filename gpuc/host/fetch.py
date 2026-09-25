"""What `gpuc fetch` would copy out of a job's workdir.

The host decides and the client copies: only the host can read the outputs
baseline, which is what tells a job's results apart from files that came with
the checkout. The list is the same one uploads use, so a fetch and an S3
upload of the same job hold the same files.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from gpuc.host import baseline, jobs, paths


class NotFetchable(Exception):
    pass


def inside(workdir: Path, rel: str) -> Path:
    """`rel` inside the workdir, or NotFetchable. `.` is the whole workdir."""
    if os.path.isabs(rel):
        raise NotFetchable(f"{rel!r}: give a path relative to the job's workdir")
    target = workdir / rel
    if not target.resolve().is_relative_to(workdir.resolve()):
        raise NotFetchable(f"{rel!r} is not inside the job's workdir")
    return target


def walk(root: Path) -> Iterator[Path]:
    """Every file and symlink under `root`, or `root` itself if it is a file.

    Symlinks below `root` are listed, never followed: rsync copies them as
    links. `root` itself is followed when it is a link to a directory, as an
    upload follows it; `inside` has already checked where it leads."""
    if not root.is_dir():
        yield root
        return
    for parent, dirs, names in os.walk(root):
        for name in sorted(names):
            yield Path(parent, name)
        for name in sorted(dirs):
            if Path(parent, name).is_symlink():
                yield Path(parent, name)


def listing(job_id: str, spec: jobs.JobSpec, wanted: Sequence[str]) -> dict[str, Any]:
    """The files to copy, relative to the workdir, with their sizes.

    With `wanted` empty, each of the spec's `outputs:` less what the baseline
    says came with the checkout. With paths named, everything under them: a
    person naming a path wants what is there."""
    workdir = paths.workdir(job_id)
    if not workdir.is_dir():
        raise NotFetchable(
            "its workdir is gone: the sweep or `gpuc clean` removed it once the job ended"
        )
    if wanted:
        roots = [(rel, inside(workdir, rel), {}) for rel in wanted]
    elif spec.outputs:
        found = baseline.read(job_id)
        roots = []
        for output in spec.outputs:
            try:
                key = baseline.output_key(output, job_id)
            except (KeyError, IndexError, ValueError) as exc:
                raise NotFetchable(
                    f"output {output.path!r} cannot be resolved ({exc!r}); name what to "
                    f"fetch with --path"
                ) from exc
            roots.append((key, inside(workdir, key), found.get(key, {})))
    else:
        raise NotFetchable("its spec declares no `outputs:`; name what to fetch with --path")

    files: list[dict[str, Any]] = []
    missing: list[str] = []
    for rel, root, entries in roots:
        if not root.exists() and not root.is_symlink():
            missing.append(rel)
            continue
        base = root if root.is_dir() else root.parent
        preexisting = set(baseline.unchanged(base, entries))
        for path in walk(root):
            if str(path.relative_to(base)) in preexisting:
                continue
            try:
                size = path.lstat().st_size
            except OSError:
                continue
            files.append({"path": str(path.relative_to(workdir)), "bytes": size})
    unique = list({f["path"]: f for f in files}.values())
    return {
        "workdir": str(workdir),
        "files": unique,
        "bytes": sum(f["bytes"] for f in unique),
        "missing": missing,
    }
