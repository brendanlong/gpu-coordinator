"""Which build of gpuc is this, and which one is on each host.

`__version__` has not moved in the life of the project, so it cannot answer
"is the thing in my PATH the thing the docs describe?". The commit can, and
two sessions of the same user running different commits against one shared
registry is exactly the failure this exists to make visible.
"""

from __future__ import annotations

import json
import subprocess
from functools import lru_cache
from importlib.metadata import Distribution, PackageNotFoundError
from pathlib import Path

import gpuc
from gpuc._version import __version__ as __version__
from gpuc._version import same_commit as same_commit
from gpuc._version import superseded

DIST_NAME = "gpu-coordinator"
SHORT = 12


def short(commit: str | None) -> str:
    return commit[:SHORT] if commit else "unknown"


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


def source_commit() -> str | None:
    """`git rev-parse HEAD` where the package lives, for a checkout or an
    editable install. Never raises: git may not be there at all."""
    try:
        result = subprocess.run(
            ["git", "-C", str(package_root()), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = result.stdout.strip()
    return commit if result.returncode == 0 and commit else None


@lru_cache(maxsize=1)
def local_commit() -> str | None:
    """The commit this `gpuc` is running, installed build first.

    Cached: `status` asks once per host, and it cannot change under a process.
    """
    return installed_commit() or source_commit()


def dirty() -> bool:
    """Whether the source checkout has uncommitted changes, so `version` can
    say that the commit it printed is not the whole truth."""
    try:
        result = subprocess.run(
            ["git", "-C", str(package_root()), "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10.0,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0 and bool(result.stdout.strip())


def needs_package_sync(local: str | None, host: str | None) -> bool:
    """Should this host be shipped the package again before it runs anything?

    `superseded`, which is stricter than `same_commit` in the one place that
    matters: a host with no recorded commit was bootstrapped by a build too old
    to record one, so it is running the oldest code of all.
    """
    return superseded(host, local)


def host_build_warning(name: str, host_commit: str | None, local: str | None) -> str | None:
    """`host_commit` is the host's own answer, from its `config.json`.

    Judged by exactly the rule `submit` re-ships on, which is stricter than
    `same_commit` in the case that matters here: a host that *answered* and
    named no commit is a host on a build old enough not to report one, so
    saying nothing about it would leave `status` quiet about the very hosts
    `submit` re-ships on every single run. A host nobody could ask is the
    caller's to skip -- that one really is "we do not know".

    Not "older": whichever machine bootstrapped the host last is the one it
    runs, and that can as easily be a laptop on a newer build as this machine
    on an older one. Both directions are the same problem -- the host is not
    running the code that wrote the spec -- and the same fix.
    """
    if not needs_package_sync(local, host_commit):
        return None
    running = f"gpuc {short(host_commit)}" if host_commit else "a build too old to say which"
    return (
        f"host {name} is running {running} and this machine has "
        f"{short(local)}; run gpuc host bootstrap {name}"
    )


def shipped_commit_note(name: str, recorded: str | None, local: str | None) -> str | None:
    """What an *offline* command can honestly say about a host's build.

    `gpuc host list` and `gpuc version` never touch the host, so all they have
    is the `pkg_commit` cached from the last time something here did ask. Any
    machine may have re-bootstrapped the host since, so it is reported as what
    it is -- last seen -- and `gpuc status` is where the live answer lives.
    """
    if same_commit(local, recorded):
        return None
    return (
        f"host {name} was last seen running gpuc {short(recorded)} and this machine has "
        f"{short(local)}; run gpuc host bootstrap {name} (`gpuc status` asks the host itself)"
    )


def dispatcher_build_warning(name: str, running: str | None, shipped: str | None) -> str | None:
    """The dispatcher on this host is serving the queue with older code.

    Both sides are the host's own answers: the commit recorded by the
    dispatcher holding the lock, and the commit of the package now on disk.
    They come apart because a dispatcher imports its code once and then lives
    for days -- so a host re-bootstrapped underneath one goes on dispatching
    with whatever was there when it started, and every feature shipped since
    is simply not running. A newer dispatcher takes over from an older one by
    itself; this is for the host where that did not happen.
    """
    if not superseded(running, shipped):
        return None
    was = f"gpuc {short(running)}" if running else "a build too old to say which"
    return (
        f"host {name} has gpuc {short(shipped)} on disk but its running dispatcher was "
        f"started on {was}; nothing shipped since is in effect. Restart it with "
        f"gpuc host bootstrap {name}"
    )
