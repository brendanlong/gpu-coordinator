"""Version and User-Agent, in one stdlib-only place.

`gpuc.host` may not import anything outside the stdlib, and the control side
must send the same string, so this module is imported by both and depends on
nothing.
"""

from __future__ import annotations

__version__ = "0.1.0"

HOMEPAGE = "https://github.com/brendanlong/gpu-coordinator"
CONTACT = "self@brendanlong.com"


def user_agent() -> str:
    """Who we are to every service we call: the tool, its version, how to reach us.

    Sent by the RunPod API calls, the host's self-terminate, the health check's
    download, the uv installer fetch, boto3 and `hf`. The `aws` CLI is the one
    exception: its User-Agent is not overridable.
    """
    return f"gpuc/{__version__} (+{HOMEPAGE}; {CONTACT})"


DIRTY = "-dirty"
"""The suffix `control.version.local_commit` adds for a checkout with
uncommitted changes, followed by a short hash of those changes
(`<commit>-dirty-1a2b3c4d`): the tree behind it is not the commit it names,
and two different dirty trees on one commit are two builds."""


def is_other_build(recorded: str | None, current: str | None) -> bool:
    """Is `recorded` a different build from `current`?

    Different, deliberately, and not "older": a commit id carries no ordering,
    so nothing here can tell which of two came first. Every caller wants the
    same thing anyway -- the host should be running the build this machine
    has, and a host running one from *ahead* of it is the same problem with
    the same fix. Here rather than in `control.version` because the host side
    asks it too: the dispatcher lock records which build its holder started
    on, and `gpuc.host` may not import anything outside the stdlib.

    Unknowns are asymmetric. A host that names no commit was never shipped by
    a build that records one, so it cannot be running this one; reading that
    as "probably fine" is how last week's code goes on running. An unknown
    `current` is the other way round -- nothing to compare against, so nothing
    is claimed. Commits may be recorded at different lengths, so a prefix
    matches, but `-dirty` and what follows it is part of the identity and
    compared exactly: a bare `-dirty` from an earlier build, or another
    hash, is another tree on the same commit and is re-shipped.
    """
    if not current:
        return False
    if not recorded:
        return True
    recorded_base, _, recorded_tag = recorded.partition(DIRTY)
    current_base, _, current_tag = current.partition(DIRTY)
    if (DIRTY in recorded) != (DIRTY in current) or recorded_tag != current_tag:
        return True
    return not (recorded_base.startswith(current_base) or current_base.startswith(recorded_base))
