"""Which machine is serving this queue.

A queue on a volume that outlives its machine can be picked up by another:
the replacement of a pod, a Kubernetes rollout that briefly runs the old pod
beside the new, a second host set up by mistake over the same root. The
newcomer takes the queue at once rather than waiting to be sure the old
machine is dead -- nothing here can be sure of that -- and says so in
`owner.json`. A dispatcher that finds the record naming another machine has
been replaced, and stands down: it kills what it was running and writes
nothing more.

Detection, not prevention: a stale dispatcher can still write between the
takeover and its next look. Where the volume's `flock` works across machines
the dispatcher lock already keeps a second one out; this is for everywhere
else.
"""

from __future__ import annotations

import socket
from contextlib import suppress
from typing import Any

from gpuc.host import jobs, paths
from gpuc.host.procs import boot_id


def instance() -> str:
    """This machine in this boot. A restart is a new instance too, but the
    one it replaces is dead, so nothing is left to stand down."""
    return f"{socket.gethostname()}/{boot_id() or 'unknown-boot'}"


def current() -> str | None:
    """The instance the record names, or None with no readable record: a
    queue nobody has claimed yet is nobody else's."""
    path = paths.owner_file()
    if not path.exists():
        return None
    with suppress(RuntimeError):
        return jobs.as_opt_str(jobs.fields_of(jobs.read_json(path, attempts=2)), "instance")
    return None


def claim() -> str | None:
    """Record this instance as the one serving the queue; the one it
    replaces, if another."""
    previous = current()
    record: dict[str, Any] = {"instance": instance(), "since": jobs.utc_now()}
    jobs.atomic_write_json(paths.owner_file(), record)
    return previous if previous != record["instance"] else None


def replaced_by() -> str | None:
    """The instance that took this queue over, or None while it is ours."""
    owner = current()
    return owner if owner is not None and owner != instance() else None
