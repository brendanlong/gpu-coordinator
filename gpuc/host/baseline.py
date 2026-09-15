"""What was already under an `outputs:` path before the job ran.

A checkout routinely has committed files where the outputs go -- a
`results/report-elephant.md` from the last run, a figure in `figures/`. Those
arrive in the workdir with the code, and without this the first sync uploads
them as if this job had produced them: the run's output namespace fills up with
someone else's results, and a job that produced nothing at all looks successful.

The baseline is (relative path, size, mtime) per declared output, taken before
`setup`. A file that still matches all three is not uploaded and does not count
as an output.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from gpuc.host import paths
from gpuc.host.jobs import JobSpec, Output, atomic_write_json, read_json

MAX_TRACKED = 500
"""Above this, the exclude list would be longer than the upload command can
carry. The job is told, and that output falls back to uploading everything --
the honest degradation, since the alternative is a sync that cannot run."""

Entries = dict[str, list[int]]
Baseline = dict[str, Entries]


def output_key(output: Output, job_id: str) -> str:
    return output.path.format(job_id=job_id)


def scan(root: Path) -> Entries:
    entries: Entries = {}
    if not root.is_dir():
        return {} if not root.is_file() else {root.name: _stat(root)}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        try:
            entries[str(path.relative_to(root))] = _stat(path)
        except OSError:
            continue
        if len(entries) > MAX_TRACKED:
            return entries
    return entries


def _stat(path: Path) -> list[int]:
    info = path.stat()
    return [info.st_size, info.st_mtime_ns]


def capture(spec: JobSpec, workdir: Path, job_id: str) -> Baseline:
    """Snapshot every declared output path. Written even when empty, so a
    reader can tell "nothing was there" from "never taken"."""
    found: Baseline = {}
    for output in spec.outputs:
        key = output_key(output, job_id)
        found[key] = scan(workdir / key)
    atomic_write_json(paths.outputs_baseline_file(job_id), found)
    return found


def read(job_id: str) -> Baseline:
    path = paths.outputs_baseline_file(job_id)
    if not path.exists():
        return {}
    try:
        document = read_json(path)
    except RuntimeError:
        return {}
    if not isinstance(document, dict):
        return {}
    return {
        str(key): {str(name): list(value) for name, value in (entries or {}).items()}
        for key, entries in document.items()
        if isinstance(entries, dict)
    }


def unchanged(root: Path, entries: Entries) -> list[str]:
    """Relative paths that are byte-for-byte where the job found them.

    Size and mtime, not a hash: a job that rewrites a file with identical
    contents has still produced it, and hashing a 200 GB checkpoint directory
    on every sync tick would cost more than the upload.
    """
    if not entries:
        return []
    still: list[str] = []
    for name, recorded in entries.items():
        path = root / name
        try:
            info = path.stat()
        except OSError:
            continue
        if [info.st_size, info.st_mtime_ns] == list(recorded):
            still.append(name)
    return sorted(still)


def has_new_content(root: Path, entries: Entries) -> bool:
    """Did this output path gain or change anything at all?"""
    if not root.exists():
        return False
    if root.is_file():
        return not unchanged(root.parent, {root.name: entries.get(root.name, [])})
    files = [p for p in root.rglob("*") if p.is_file()]
    return len(files) > len(unchanged(root, entries))


def too_many(entries: Entries) -> bool:
    return len(entries) > MAX_TRACKED


def describe(baseline: Baseline) -> Sequence[str]:
    return [
        f"{path}: {len(entries)} pre-existing file(s) will not be uploaded"
        for path, entries in sorted(baseline.items())
        if entries
    ]
