"""The optional `progress_command`: the job's own answer to "how far along?".

Only the job knows what its work is made of -- epochs, sweep points, shards --
so gpuc does not guess. It runs a command the spec supplies, reads a percentage
off its last line of stdout, and turns that into an end time. A command that is
missing, broken, slow or nonsense costs a recorded `progress_error` and
nothing else: an estimate is never allowed to decide a job's outcome.
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
from collections.abc import Mapping
from pathlib import Path

TIMEOUT_S = 10.0
"""Short on purpose. The runner polls this from the same loop that watches for
a cancel, a TTL and `max_runtime_min`, so a wedged progress command delays a
kill by at most this -- well inside the 15 s the runner gets before the
dispatcher escalates."""

DEFAULT_INTERVAL_S = 60.0


class ProgressError(RuntimeError):
    """The command failed, or said something that is not a percentage."""


def _number(text: str, whole: str) -> float:
    try:
        return float(text.strip())
    except ValueError:
        raise ProgressError(
            f"printed {whole!r}, which is not a percentage; print `42`, `42%` or `300/5000`"
        ) from None


def parse(stdout: str) -> float:
    """The last non-empty line of stdout, as a percentage of 0-100.

    `42`, `42.5` and `42%` are the percentage itself; `300/5000` is how much of
    how many, which is how a training loop usually knows. Deliberately *not* a
    fraction of 1: `0.42` would otherwise have to mean either 0.42% or 42%, and
    guessing wrong is a hundredfold error in an end time somebody is planning
    around. Trailing lines are what a script appends last, so reading the last
    one lets a progress command be a shell pipeline that also logs.
    """
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if not lines:
        raise ProgressError("printed nothing on stdout")
    last = lines[-1]
    if "/" in last:
        done, _, total = last.partition("/")
        divisor = _number(total, last)
        if divisor <= 0:
            raise ProgressError(f"printed {last!r}; the total after `/` must be above zero")
        percent = _number(done, last) / divisor * 100.0
    else:
        percent = _number(last.removesuffix("%"), last)
    if not 0.0 <= percent <= 100.0:
        raise ProgressError(f"printed {last!r}, which is not between 0% and 100%")
    return round(percent, 1)


def _tail(text: str, lines: int = 3) -> str:
    return " / ".join(text.strip().splitlines()[-lines:]) or "(no output)"


def _kill(proc: subprocess.Popen[str]) -> None:
    """Take down the whole session, not just the shell we started.

    `bash -c 'sleep 999'` leaves the sleep behind if only the shell is killed,
    and this runs again every interval for the rest of the job.
    """
    with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
        os.killpg(proc.pid, signal.SIGKILL)
    with contextlib.suppress(OSError):
        proc.kill()


def poll(
    command: str,
    cwd: Path,
    env: Mapping[str, str] | None = None,
    timeout_s: float = TIMEOUT_S,
) -> float:
    """Run the progress command once and return the percentage it reported."""
    try:
        proc = subprocess.Popen(
            ["bash", "-c", command],
            cwd=str(cwd),
            env=None if env is None else dict(env),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            text=True,
            start_new_session=True,
        )
    except OSError as exc:
        raise ProgressError(f"could not run `{command}`: {exc}") from exc
    try:
        stdout, stderr = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        _kill(proc)
        proc.communicate()
        raise ProgressError(f"`{command}` took longer than {timeout_s:g}s and was killed") from None
    if proc.returncode != 0:
        raise ProgressError(
            f"`{command}` exited {proc.returncode}: {_tail(stderr or stdout)}"
        ) from None
    return parse(stdout)
