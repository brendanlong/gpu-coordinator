"""Which machine is serving this queue.

A queue on a volume that outlives its machine can be picked up by another:
the replacement of a pod, a Kubernetes rollout that briefly runs the old pod
beside the new, a second host set up by mistake over the same root. The one
serving it says so in `owner.json` and renews it with its heartbeat. A
newcomer waits for that to go stale -- the only sign it can have that the
machine before it is gone -- and takes over; one that finds it fresh leaves
the queue to its owner. Everything of a machine that finds the record naming
another stops and writes nothing more: its dispatcher, and each runner for
its own job.

Detection, not prevention: a stale machine can still write between the
takeover and its next look, and one whose heartbeat stalled for longer than
`STALE_S` without dying is taken for dead. Where the volume's `flock` works
across machines the dispatcher lock keeps a second dispatcher out anyway.
"""

from __future__ import annotations

import os
import socket
import time
from collections.abc import Callable
from contextlib import suppress

from gpuc.host import jobs, paths
from gpuc.host.procs import boot_id, init_start

STALE_S = 30.0
"""How long a record may go unrenewed before its machine is taken for gone:
the dispatcher heartbeat's own staleness, since the heartbeat renews it."""

LONG_GONE_S = 10 * STALE_S
"""A record older than this by the reader's clock is stale without waiting
to watch it: no plausible clock skew makes a live owner look that old."""

CLAIM_WAIT_S = STALE_S + 5.0
"""How long a newcomer watches a record for renewals before deciding. Long
enough to outlast the heartbeat of a machine that has just died, so a
replacement started at once still takes the queue."""

CHECK_S = 5.0
"""How often a runner looks at the record while its job runs."""


class Busy(RuntimeError):
    """Another machine is serving this queue and renewing its claim."""


def instance() -> str:
    """This machine in this boot: the kernel's boot id and pid 1's start
    time (see `procs.init_start`), which a hostname change does not move and
    a container restart does. The hostname stands in for pid 1 only where
    /proc hides it, to tell apart containers sharing a kernel."""
    boot = boot_id() or "unknown-boot"
    init = init_start()
    return f"{boot}/{init}" if init else f"{boot}@{socket.gethostname()}"


def current() -> tuple[str, float] | None:
    """The instance the record names and its modification time, or None
    with no readable record: a queue nobody claimed is nobody's."""
    path = paths.owner_file()
    try:
        renewed = path.stat().st_mtime
        owner = jobs.as_opt_str(jobs.fields_of(jobs.read_json(path, attempts=2)), "instance")
    except (OSError, RuntimeError):
        return None
    return (owner, renewed) if owner else None


def claim(
    me: str,
    *,
    wait_s: float = CLAIM_WAIT_S,
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], float] = time.monotonic,
) -> str | None:
    """Record `me` as serving the queue once whoever else is recorded has
    gone quiet; the instance it took over from, if any. Raises `Busy` when
    the other is still renewing after `wait_s`.

    Quiet is judged by watching the record's modification time not move for
    `STALE_S` on this machine's own clock: the time itself was set by
    another machine, or by a file server, and comparing it with ours would
    turn clock skew into a takeover or into a queue nobody can ever claim.
    """
    record = current()
    if record is not None and record[0] != me and time.time() - record[1] < LONG_GONE_S:
        deadline = now() + wait_s
        quiet_since = now()
        while True:
            if now() - quiet_since >= STALE_S:
                break
            if now() >= deadline:
                raise Busy(f"{record[0]} is serving this queue and still renewing its claim")
            sleep(1.0)
            seen = current()
            if seen is None or seen[0] == me:
                record = seen
                break
            if seen != record:
                record, quiet_since = seen, now()
    jobs.atomic_write_json(
        paths.owner_file(),
        {"instance": me, "host": socket.gethostname(), "since": jobs.utc_now()},
    )
    return record[0] if record is not None and record[0] != me else None


def renew(me: str) -> None:
    """Keep a claim fresh, if it is still `me`'s. Never writes over another's."""
    record = current()
    if record is not None and record[0] == me:
        with suppress(OSError):
            os.utime(paths.owner_file())


def replaced_by(me: str) -> str | None:
    """The instance that took this queue over from `me`, or None."""
    record = current()
    return record[0] if record is not None and record[0] != me else None
