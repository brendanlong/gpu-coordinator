"""The optional `progress_command`: the job's own answer to "how far along?".

Only the job knows what its work is made of -- epochs, sweep points, shards --
so gpuc does not guess. It runs a command the spec supplies, reads how far
along it is off the last line of stdout, and turns that into an end time. A
command that is missing, broken, slow or nonsense costs a recorded
`progress_error` and nothing else: an estimate is never allowed to decide a
job's outcome.
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import IO

TIMEOUT_S = 10.0
"""Short on purpose. The runner polls this from the same loop that watches for
a cancel, a TTL and `max_runtime_min`, so a wedged progress command delays a
kill by at most this. The runner's own grace before the dispatcher escalates a
kill is 15 s, so this is most of that budget rather than a rounding error on
it -- which is why it is not configurable and why the reap below is
unconditional."""

REAP_GRACE_S = 1.0

DEFAULT_INTERVAL_S = 60.0

MAX_OUTPUT_BYTES = 64 * 1024
"""How much of the command's output we are willing to look at.

Output goes to a temp file and only the tail is read back, so
`progress_command: "cat train.log"` -- an easy slip next to the documented
`tail -1` -- costs a bounded read instead of the whole log in the runner's
memory. On a cgroup host that memory is the *job's*, so an OOM there would
kill the job an estimate is not allowed to touch."""


class ProgressError(RuntimeError):
    """The command failed, or said something that is not a progress reading."""


def _number(text: str, whole: str) -> float:
    try:
        return float(text.strip())
    except ValueError:
        raise ProgressError(
            f"printed {whole!r}, which is not a progress reading; print a fraction of 1 "
            f"like `0.42` or a percentage like `42%`"
        ) from None


def parse(stdout: str) -> float:
    """The last non-empty line of stdout, as a percentage of 0-100.

    Two forms, and which one you meant is always written down: a fraction of
    one carries a decimal point (`0.42`), and a percentage carries a `%`
    (`42%`). A bare integer is *refused*, not guessed at -- `42` could be
    either, `1` could be 1% or a finished job, and reading either the wrong way
    is a hundredfold error in an end time somebody is planning around. No rule
    applied afterwards can tell the two apart, so the unit has to be in the
    input.

    The decimal point costs the intended audience nothing: `step / total` in
    any language prints `0.42`, `0.0` and `1.0`, never a bare integer. What it
    catches is the shell one-liner echoing a raw counter, which is exactly the
    case that would otherwise read `1` as "finished".

    Trailing lines are what a script appends last, so reading the last one lets
    a progress command be a shell pipeline that also logs.
    """
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if not lines:
        raise ProgressError("printed nothing on stdout")
    last = lines[-1]
    if last.endswith("%"):
        # Rounded before the range check, not after: a job computing its own
        # percentage in floating point prints `100.0000001` at the exact moment
        # it finishes, and erroring there would be absurd.
        return _in_range(round(_number(last.removesuffix("%"), last), 1), last)
    # Parsed before the shape is judged, so `almost done` is reported as not
    # being a reading at all rather than as an ambiguous one.
    fraction = _number(last, last)
    if "." not in last:
        percent = f" -- write `{last}%` for a percentage" if 0.0 <= fraction <= 100.0 else ""
        raise ProgressError(
            f"printed {last!r}: a bare number could be a percentage or a fraction, so gpuc "
            f"refuses to guess{percent}; a fraction of 1 needs a decimal point, like `0.42`"
        )
    if not 0.0 <= fraction <= 1.0:
        raise ProgressError(
            f"printed {last!r}: a number with a decimal point is a fraction of 1 (`0.42` is "
            f"42%), so it must be between 0.0 and 1.0 -- write `{last}%` for a percentage"
        )
    return round(fraction * 100.0, 1)


def _in_range(percent: float, last: str) -> float:
    if not 0.0 <= percent <= 100.0:
        raise ProgressError(f"printed {last!r}, which is not between 0% and 100%")
    return percent


def _describe(text: str) -> str:
    return " / ".join(text.strip().splitlines()[-3:]) or "(no output)"


def _tail(handle: IO[bytes]) -> str:
    size = handle.seek(0, os.SEEK_END)
    handle.seek(max(0, size - MAX_OUTPUT_BYTES))
    return handle.read().decode("utf-8", "replace")


def _reap(proc: subprocess.Popen[bytes]) -> None:
    """Take down the whole session, not just the shell we started.

    `bash -c 'sleep 999'` leaves the sleep behind if only the shell is killed,
    and this runs again every interval for the rest of the job.
    """
    with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
        os.killpg(proc.pid, signal.SIGKILL)
    with contextlib.suppress(OSError):
        proc.kill()
    # A grandchild that called setsid survives the killpg, so even this can
    # time out. Output went to a file rather than a pipe precisely so that a
    # survivor cannot hold one open and block us here for ever.
    with contextlib.suppress(subprocess.TimeoutExpired):
        proc.wait(timeout=REAP_GRACE_S)


def poll(
    command: str,
    cwd: Path,
    env: Mapping[str, str] | None = None,
    timeout_s: float = TIMEOUT_S,
) -> float:
    """Run the progress command once and return the percentage it reported."""
    with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
        try:
            proc = subprocess.Popen(
                ["bash", "-c", command],
                cwd=str(cwd),
                env=None if env is None else dict(env),
                stdout=out,
                stderr=err,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
        except OSError as exc:
            raise ProgressError(f"could not run `{command}`: {exc}") from exc
        try:
            proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            _reap(proc)
            raise ProgressError(
                f"`{command}` took longer than {timeout_s:g}s and was killed"
            ) from None
        except BaseException:
            # The runner was signalled mid-poll. Leaving the command running
            # would strand it on a host whose job is about to go away.
            _reap(proc)
            raise
        if proc.returncode != 0:
            raise ProgressError(
                f"`{command}` exited {proc.returncode}: {_describe(_tail(err) or _tail(out))}"
            )
        return parse(_tail(out))
