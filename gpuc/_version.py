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
