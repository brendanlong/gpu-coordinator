"""What jobs on a host share outside their own job dirs: the data directory
and the Hugging Face cache. `gpuc host clean --data` and `--hf-cache`.

Nothing here runs on its own. Only a person clears either one: the data
directory holds what jobs chose to keep for later jobs, and no job's ending
says whether a later one still wants it.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from gpuc.host import cleanup, health, jobs, paths

HF_PRUNE_TIMEOUT_S = 900.0


def data_path(rel: str, root: Path) -> Path:
    """`rel` inside the data directory, or ValueError. Never the directory
    itself, and never anything a `..` or a symlink would take outside it."""
    if not rel or os.path.isabs(rel):
        raise ValueError(f"{rel!r}: give a path relative to the data directory {root}")
    target = root / rel
    resolved_root = root.resolve()
    if target.resolve() == resolved_root or not target.resolve().is_relative_to(resolved_root):
        raise ValueError(f"{rel!r} is not inside the data directory {root}")
    return target


def remove_data(rels: Sequence[str], root: Path | None = None) -> dict[str, Any]:
    """Delete each path under the data directory, reporting every one.

    A path that is not there is an error for that path only, the way `gpuc
    clean --only` treats a job it cannot find: a typo is worth hearing about."""
    root = root or paths.data_dir()
    removed: list[dict[str, Any]] = []
    errors: list[str] = []
    for rel in rels:
        try:
            target = data_path(rel, root)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        if not target.exists() and not target.is_symlink():
            errors.append(f"{rel}: not in {root}")
            continue
        try:
            if target.is_dir() and not target.is_symlink():
                freed = cleanup.reclaimable_bytes(target)
                shutil.rmtree(target)
            else:
                freed = target.lstat().st_size
                target.unlink()
        except OSError as exc:
            errors.append(f"{rel}: {exc}")
            continue
        removed.append({"path": rel, "freed_bytes": freed})
    return {"data_dir": str(root), "removed": removed, "errors": errors}


def prune_hf_cache() -> dict[str, Any]:
    """`hf cache prune`: detached revisions and unfinished downloads only.

    The analogue of `uv cache prune`, and for the same reason never a wipe:
    every revision a job may ask for again stays."""
    config = jobs.read_config()
    cache = health.hf_hub_cache_dir(config)
    document: dict[str, Any] = {"cache_dir": str(cache), "before_bytes": 0, "after_bytes": 0}
    if not cache.is_dir():
        return {**document, "errors": []}
    document["before_bytes"] = cleanup.dir_size(cache)
    env = config.apply_env(dict(os.environ))
    try:
        result = subprocess.run(
            ["hf", "cache", "prune", "--yes", "--cache-dir", str(cache)],
            env=env,
            capture_output=True,
            text=True,
            timeout=HF_PRUNE_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {**document, "after_bytes": document["before_bytes"], "errors": [f"hf: {exc}"]}
    document["after_bytes"] = cleanup.dir_size(cache)
    errors = []
    if result.returncode != 0:
        output = (result.stdout + result.stderr).strip()[-800:]
        errors.append(f"`hf cache prune` exited {result.returncode}: {output}")
    return {**document, "errors": errors}


def cmd_data_remove(args: argparse.Namespace) -> int:
    report = remove_data(args.paths)
    print(json.dumps(report, indent=2))
    return 1 if report["errors"] else 0


def cmd_hf_cache_prune(_args: argparse.Namespace) -> int:
    report = prune_hf_cache()
    print(json.dumps(report, indent=2))
    return 1 if report["errors"] else 0


def add_parsers(sub: Any) -> None:
    data = sub.add_parser("data-remove", help="delete paths under the host's data directory")
    data.add_argument("paths", nargs="+", metavar="PATH")
    data.set_defaults(func=cmd_data_remove)
    prune = sub.add_parser("hf-cache-prune", help="`hf cache prune` on the jobs' HF cache")
    prune.set_defaults(func=cmd_hf_cache_prune)
