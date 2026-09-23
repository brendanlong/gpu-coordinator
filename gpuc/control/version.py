"""Which build of gpuc is this, and is a host running it.

`__version__` has not moved in the life of the project, so it cannot answer
"is the thing in my PATH the thing the docs describe?". The commit can, and
two sessions of the same user running different commits against one shared
registry is exactly the failure this exists to make visible.

One comparison, `is_other_build`, and it is strict: a host that names no
commit is re-shipped, and a dirty checkout is not the commit it sits on.
"""

from __future__ import annotations

import json
import subprocess
from functools import lru_cache
from importlib.metadata import Distribution, PackageNotFoundError
from pathlib import Path

import gpuc
from gpuc._version import DIRTY
from gpuc._version import __version__ as __version__
from gpuc._version import is_other_build as is_other_build

DIST_NAME = "gpu-coordinator"
SHORT = 12


def short(commit: str | None) -> str:
    if not commit:
        return "unknown"
    base, dirty = (commit[: -len(DIRTY)], DIRTY) if commit.endswith(DIRTY) else (commit, "")
    return base[:SHORT] + dirty


def package_root() -> Path:
    return Path(gpuc.__file__).resolve().parents[1]


def installed_commit() -> str | None:
    """The commit `uv tool install git+...` recorded for the installed dist.

    pip and uv write `direct_url.json` beside the dist-info for anything
    installed from a URL, and its `vcs_info.commit_id` is the resolved commit.
    A dist installed from a local path (`uv tool install .`) has no commit
    there, which is why the source checkout is the fallback.
    """
    try:
        text = Distribution.from_name(DIST_NAME).read_text("direct_url.json")
    except (PackageNotFoundError, OSError, ValueError):
        return None
    if not text:
        return None
    try:
        document = json.loads(text)
    except json.JSONDecodeError:
        return None
    info = document.get("vcs_info") if isinstance(document, dict) else None
    commit = info.get("commit_id") if isinstance(info, dict) else None
    return str(commit) if commit else None


def _git(*args: str) -> str | None:
    """`git` in the package's checkout, or None: git may not be there at all."""
    try:
        result = subprocess.run(
            ["git", "-C", str(package_root()), *args],
            capture_output=True,
            text=True,
            check=False,
            timeout=10.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout if result.returncode == 0 else None


def source_commit() -> str | None:
    """`git rev-parse HEAD` where the package lives, for a checkout or an
    editable install, with `-dirty` appended when the tree has changes."""
    commit = (_git("rev-parse", "HEAD") or "").strip()
    if not commit:
        return None
    return f"{commit}{DIRTY}" if dirty() else commit


@lru_cache(maxsize=1)
def local_commit() -> str | None:
    """The build this `gpuc` is running, installed build first.

    A dirty checkout is `<commit>-dirty`: it ships code HEAD does not have, so
    a host bootstrapped from it must not read as running HEAD, and the next
    submit from a clean checkout of the same commit must re-ship. Cached:
    `status` asks once per host, and it cannot change under a process.
    """
    return installed_commit() or source_commit()


def dirty() -> bool:
    """Whether the source checkout has uncommitted changes."""
    return bool((_git("status", "--porcelain") or "").strip())


def host_build_warning(name: str, host_commit: str | None, local: str | None) -> str | None:
    """`host_commit` is the host's own answer, from its `config.json`.

    Judged by exactly the rule `submit` re-ships on. A host that *answered*
    and named no commit is warned about rather than passed as current; a host
    nobody could ask is the caller's to skip -- that one really is "we do not
    know".
    """
    if not is_other_build(host_commit, local):
        return None
    running = f"gpuc {short(host_commit)}" if host_commit else "a build that named no commit"
    return (
        f"host {name} is running {running} and this machine has "
        f"{short(local)}; run gpuc host bootstrap {name}"
    )


def shipped_commit_note(name: str, recorded: str | None, local: str | None) -> str | None:
    """What an *offline* command can honestly say about a host's build.

    `gpuc host list` and `gpuc version` never touch the host, so all they have
    is the `pkg_commit` cached from the last time something here did ask. Any
    machine may have re-bootstrapped the host since, so it is reported as what
    it is -- last seen -- and `gpuc status` is where the live answer lives. A
    host that never named a commit gets no note: the listing already prints
    `pkg unknown`, and `gpuc host add` already says to bootstrap next.
    """
    if recorded is None or not is_other_build(recorded, local):
        return None
    return (
        f"host {name} was last seen running gpuc {short(recorded)} and this machine has "
        f"{short(local)}; run gpuc host bootstrap {name} (`gpuc status` asks the host itself)"
    )


def dispatcher_build_warning(name: str, running: str | None, shipped: str | None) -> str | None:
    """The dispatcher on this host is serving the queue with other code.

    Both sides are the host's own answers: the commit recorded by the
    dispatcher holding the lock, and the commit of the package now on disk.
    They come apart because a dispatcher imports its code once and then lives
    for days -- so a host re-bootstrapped underneath one goes on dispatching
    with whatever was there when it started, and everything shipped since is
    simply not running. A dispatcher started from the package on disk takes
    over from one that was not; this is for the host where that did not happen.
    """
    if not is_other_build(running, shipped):
        return None
    was = f"gpuc {short(running)}" if running else "a build that named no commit"
    return (
        f"host {name} has gpuc {short(shipped)} on disk but its running dispatcher was "
        f"started on {was}; nothing shipped since is in effect. Restart it with "
        f"gpuc host bootstrap {name}"
    )
