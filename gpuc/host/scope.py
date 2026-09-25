"""Run each job phase in a transient `systemd --user` scope where one exists
and outlives logout.

A process cannot leave its cgroup without privilege, so stopping the scope
reaps the entire tree -- including a grandchild that double-forked out of the
job's process group (`setsid`, `nohup`, a daemonising server), which the
process-group kill cannot reach. Hosts with no user systemd (most containers,
including every RunPod pod and the shared box) keep the process-group kill.
"""

from __future__ import annotations

import base64
import os
import subprocess

CGROUP = "cgroup"
PGID = "pgid"

ISOLATION_ENV = "GPUC_ISOLATION"
"""How the dispatcher tells its runners what it already decided, so one answer
serves every job of that dispatcher's life rather than one exec per phase --
and how a test pins the mode on a machine whose systemd it must not depend on.
`isolation()` is the one reader; nothing else asks the probe."""

STOP_TIMEOUT_S = 15.0
"""`TimeoutStopSec`: SIGTERM, then SIGKILL after this. systemd's own default is
90 s, which is 90 s of a held GPU."""

PROBE_ARGV = ["systemd-run", "--user", "--scope", "--collect", "--quiet", "--", "true"]
LINGER_ARGV = ["loginctl", "show-user", str(os.getuid()), "-p", "Linger", "--value"]

_probed: bool | None = None


def _run(argv: list[str], timeout: float) -> int:
    return _output(argv, timeout)[0]


def _output(argv: list[str], timeout: float) -> tuple[int, str]:
    try:
        done = subprocess.run(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=timeout,
            check=False,
            text=True,
        )
    except (OSError, subprocess.SubprocessError):
        return 1, ""
    return done.returncode, done.stdout.strip()


def lingering(*, timeout: float = 20.0) -> bool:
    """Does this user's systemd instance outlive their last login?

    Without linger it stops a few seconds after the last session ends and
    takes every scope under it along: the dispatcher, its runners and their
    jobs would all die when the SSH session that submitted them closed. A
    process group left in the login session's scope survives logout (unless
    logind has `KillUserProcesses=yes`), so without linger that is the safer
    home.
    """
    code, out = _output(LINGER_ARGV, timeout)
    return code == 0 and out == "yes"


def probe(*, timeout: float = 20.0, use_cache: bool = True) -> bool:
    """Can this user create transient scopes that outlive their login?
    Cached for the process's life.

    Needs linger, a user D-Bus and cgroup delegation, not just the binary, so
    the only reliable answer is to check linger and then create one.
    """
    global _probed
    if use_cache and _probed is not None:
        return _probed
    result = lingering(timeout=timeout) and _run(PROBE_ARGV, timeout) == 0
    if use_cache:
        _probed = result
    return result


def isolation(*, timeout: float = 20.0) -> str:
    """`cgroup` or `pgid`: what was announced, else what one probe finds.

    Decided once per process. The dispatcher announces its answer to every
    child, so the whole tree -- dispatcher, runners, phases -- agrees on what
    a kill reaches.
    """
    announced = os.environ.get(ISOLATION_ENV)
    if announced in (CGROUP, PGID):
        return announced
    return CGROUP if probe(timeout=timeout) else PGID


def unit_name(job_id: str, phase: str) -> str:
    return f"gpuc-{job_id}-{phase}.scope"


def phase_argv(command: str, unit: str | None) -> list[str]:
    """The argv for one phase, in a scope when `unit` is given.

    The script is passed base64-encoded and decoded inside the scope: systemd
    treats the words after `--` as an ExecStart and does its own `$`
    substitution there ($$ -> $, $VAR -> environment), which silently corrupts
    any shell script written inline. The base64 alphabet has no `$`, and `$1`
    is digit-led, which systemd leaves alone.
    """
    if unit is None:
        return ["bash", "-eo", "pipefail", "-c", command]
    payload = base64.b64encode(f"set -eo pipefail\n{command}\n".encode()).decode()
    return [
        "systemd-run",
        "--user",
        "--scope",
        "--collect",
        "--quiet",
        "-p",
        f"TimeoutStopSec={STOP_TIMEOUT_S:.0f}",
        f"--unit={unit}",
        "--",
        "bash",
        "-c",
        'base64 -d <<<"$1" | bash',
        "_",
        payload,
    ]


def stop_unit(unit: str, *, timeout: float = STOP_TIMEOUT_S + 15.0) -> bool:
    """cgroup-kill the whole tree. False means systemctl could not do it."""
    return _run(["systemctl", "--user", "stop", unit], timeout) == 0
