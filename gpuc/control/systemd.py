"""What `reconcile --install` and `web serve --install` share: where user units
go, how to name this `gpuc` absolutely, and writing without enabling."""

from __future__ import annotations

import shutil
import sys
from collections.abc import Callable, Mapping
from pathlib import Path

from gpuc.control.bootstrap import package_root

Reporter = Callable[[str], None]


def systemd_dir() -> Path:
    return Path.home() / ".config/systemd/user"


def gpuc_command(args: str) -> str:
    """An absolute command line for `gpuc <args>`, for a unit's ExecStart.

    The `gpuc` beside this interpreter first (a `uv tool install`), then the
    one on PATH, and failing both a `uv run` from the checkout this package
    lives in; systemd has no PATH of ours to search.
    """
    beside = Path(sys.executable).resolve().parent / "gpuc"
    if beside.exists():
        return f"{beside} {args}"
    installed = shutil.which("gpuc")
    if installed:
        return f"{Path(installed).resolve()} {args}"
    uv = shutil.which("uv") or str(Path.home() / ".local/bin/uv")
    return f"{uv} run --project {package_root()} gpuc {args}"


def write_units(directory: Path, files: Mapping[str, str], report: Reporter) -> list[Path]:
    """Write the unit files only. Enabling is the user's call, not ours."""
    directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, body in files.items():
        path = directory / name
        path.write_text(body)
        written.append(path)
        report(f"wrote {path}")
    return written
