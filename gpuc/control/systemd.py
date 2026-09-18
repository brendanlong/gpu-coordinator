"""What `web serve --install` needs of systemd: where user units go, how to
name this `gpuc` absolutely, and writing without enabling."""

from __future__ import annotations

import re
import shutil
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from gpuc.control.config import Reporter
from gpuc.control.version import package_root


def systemd_dir() -> Path:
    return Path.home() / ".config/systemd/user"


SAFE_WORD = re.compile(r"^[A-Za-z0-9_./:=@+,\[\]-]+$")


def quote(word: str) -> str:
    """One ExecStart word, as systemd will read it back.

    Left bare when it needs nothing, so a unit for an ordinary path reads
    like one somebody typed. Otherwise double-quoted with `\\` and `"`
    escaped, and `%` doubled either way: it is a specifier to systemd, and a
    home directory named `100%` would otherwise expand to nothing.
    """
    if SAFE_WORD.match(word):
        return word
    escaped = word.replace("\\", "\\\\").replace('"', '\\"').replace("%", "%%")
    return f'"{escaped}"'


def gpuc_command(args: Sequence[str]) -> str:
    """An absolute command line for `gpuc <args>`, for a unit's ExecStart.

    The `gpuc` beside this interpreter first (a `uv tool install`), then the
    one on PATH, and failing both a `uv run` from the checkout this package
    lives in; systemd has no PATH of ours to search.
    """
    beside = Path(sys.executable).resolve().parent / "gpuc"
    if beside.exists():
        argv = [str(beside), *args]
    elif installed := shutil.which("gpuc"):
        argv = [str(Path(installed).resolve()), *args]
    else:
        uv = shutil.which("uv") or str(Path.home() / ".local/bin/uv")
        argv = [uv, "run", "--project", str(package_root()), "gpuc", *args]
    return " ".join(quote(word) for word in argv)


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
