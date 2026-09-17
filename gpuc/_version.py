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


def same_commit(one: str | None, other: str | None) -> bool:
    """Compare two commits that may be recorded at different lengths.

    Here rather than in `control.version` because the host side asks it too:
    the dispatcher compares the commit it is running against the one the host
    was last shipped, and `gpuc.host` may not import anything outside the
    stdlib.
    """
    if not one or not other:
        return True  # nothing recorded is not evidence of a mismatch
    return one.startswith(other) or other.startswith(one)


def superseded(recorded: str | None, current: str | None) -> bool:
    """Is the build `recorded` one that `current` has since replaced?

    The judgement `same_commit` will not make on its own: a record of *no*
    commit is a build old enough never to have written one, which is the
    oldest of all, and reading that as "probably fine" is how last week's code
    goes on running. An unknown `current` is the other way round -- there is
    nothing to compare against, so nothing is claimed.
    """
    if not current:
        return False
    return not recorded or not same_commit(current, recorded)
