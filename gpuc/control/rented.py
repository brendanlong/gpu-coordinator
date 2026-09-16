"""What a rented pod says about itself, to whichever machine is asking.

`desired/<host>.json` only ever exists on the machine that ran `gpuc submit
--runpod`. A reaper that trusts that file alone reads "another machine's pod"
as "a leak" and terminates it mid-job, which is exactly what the laptop/desktop
split is supposed to survive.

So a pod carries the same record itself. `config.json` -- the file a host owns,
and the only copy of what it is -- holds the offer it was bought on, when it
was created and when it was bootstrapped, under the `provider` block that
already named its `kind` and `pod_id`. Any machine holding the API key can ask
a pod what it is and get the same answer, and `desired/` becomes a cache of
that rather than the only record of it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import ValidationError

from gpuc.control.config import (
    ConfigError,
    DesiredHost,
    HostEntry,
    Settings,
    read_desired,
    state_lock,
    transport_for,
    write_desired,
)
from gpuc.control.providers.base import Offer, Pod
from gpuc.control.remote import RemoteError, read_remote_config, resolve_home
from gpuc.control.transport import Transport, TransportError
from gpuc.host.jobs import HostConfig

HEARTBEAT_FRESH_S = 120.0
"""A heartbeat this old still counts as alive. The dispatcher beats every 5 s;
the slack is for a host that was busy syncing, not for one that is gone."""

ASK_TIMEOUT_S = 20.0
"""Short even though no lock is held: a pass that stalls on one wedged pod is a
pass that does not reach the next one, which is billing."""


@dataclass
class Liveness:
    """What one look at a host says: reachable, beating, busy."""

    reachable: bool
    heartbeat_age_s: float | None = None
    running_jobs: int = 0

    @property
    def alive(self) -> bool:
        if self.running_jobs:
            return True
        age = self.heartbeat_age_s
        return age is not None and age < HEARTBEAT_FRESH_S

    def describe(self) -> str:
        if not self.reachable:
            return "ssh did not answer"
        if self.heartbeat_age_s is None:
            return "reachable, but the dispatcher has never beaten"
        return f"heartbeat {self.heartbeat_age_s:.0f}s old, {self.running_jobs} job(s) running"


def pod_record(address: HostEntry, offer: Offer, created_at: str) -> dict[str, Any]:
    """The `provider` block a pod is given: its own copy of the desired record.

    `kind` and `pod_id` were always here. What follows is what a machine that
    did not create this pod needs in order to judge it: the offer
    `pick_reusable_host` compares against, and when the pod was bought.
    `bootstrapped_at` is added by bootstrap, once there is one.
    """
    return {
        **(address.provider() or {}),
        "offer": offer.model_dump(mode="json"),
        "created_at": created_at,
    }


def address_for(name: str, pod: Pod) -> HostEntry | None:
    """How to reach this pod, and nothing about what it is. None: no door yet."""
    if pod.ssh_direct is None:
        return None
    return HostEntry(
        name=name,
        kind="runpod",
        ssh=f"{pod.ssh_direct.username}@{pod.ssh_direct.host}",
        port=pod.ssh_direct.port,
        pod_id=pod.id,
    )


def desired_from(
    pod_id: str, document: Mapping[str, Any], *, name: str = "", created_at: str = ""
) -> DesiredHost:
    """The desired record a pod's own config implies. `name` and `created_at`
    are what to call it and when it began if the config itself does not say.

    `bootstrapped_at` falls back to the moment the pod was created, because a
    pod that has a gpuc config at all is past the create-to-bootstrap window
    the ceiling covers -- whether or not the build that set it up recorded the
    stamp. A record without one would be read as "never bootstrapped" and
    terminated at the ceiling; what judges this pod from here is the
    dead-dispatcher rule, like any other host.
    """
    config = HostConfig.from_dict(document)
    provider = config.provider or {}
    created = _text(provider.get("created_at")) or config.created_at or created_at
    return DesiredHost(
        name=config.host or name,
        pod_id=pod_id,
        offer=_offer(provider.get("offer")),
        created_at=created,
        idle_minutes=config.idle_minutes,
        ttl_hours=config.ttl_hours,
        bootstrapped_at=_text(provider.get("bootstrapped_at")) or created,
    )


def desired_from_entry(entry: HostEntry) -> DesiredHost:
    """The same record, off a registered pod's entry and the config it cached."""
    return desired_from(entry.pod_id or "", entry.cache.config, name=entry.name)


def _text(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _pod_created(pod: Pod) -> str:
    return pod.created_at.isoformat(timespec="seconds") if pod.created_at else ""


def _offer(value: Any) -> Offer:
    if not isinstance(value, dict):
        return Offer()
    try:
        return Offer.model_validate(value)
    except ValidationError:
        # An offer we cannot read costs a reuse, never a pod: every field of
        # `Offer` has a default and nothing here terminates on one.
        return Offer()


Verdict = Literal["ours", "empty", "silent"]


@dataclass
class PodAnswer:
    """What one pod said when a machine with no record of it asked what it is.

    - `ours`: it holds a gpuc config, whoever created it.
    - `empty`: it answered, and there is no gpuc home there at all.
    - `silent`: it could not be asked, so this machine knows nothing about it.
    """

    pod: Pod
    verdict: Verdict
    detail: str
    entry: HostEntry | None = None
    desired: DesiredHost | None = None


def ask_pod(pod: Pod, settings: Settings, *, timeout: float = ASK_TIMEOUT_S) -> PodAnswer:
    """Ask a pod what it is, over ssh, using nothing this machine remembers."""
    address = address_for(pod.name, pod)
    if address is None:
        return PodAnswer(pod, "silent", "the provider gives it no ssh endpoint")
    try:
        transport = transport_for(address, settings)
        home = resolve_home(transport, address, timeout=timeout)
    except (ConfigError, RemoteError, TransportError) as exc:
        return PodAnswer(pod, "silent", _why(exc), entry=address)
    document = read_remote_config(transport, home, timeout=timeout)
    if document is None:
        return PodAnswer(
            pod, "silent", f"{home}/config.json is there but could not be read", entry=address
        )
    if not document:
        return PodAnswer(
            pod, "empty", f"it answers ssh and has no {home}/config.json", entry=address
        )
    record = desired_from(pod.id, document, name=pod.name, created_at=_pod_created(pod))
    created = record.created_at or "an unknown time"
    return PodAnswer(
        pod,
        "ours",
        f"{home}/config.json calls it {record.name}, created {created}",
        entry=address,
        desired=record,
    )


PULSE = """\
home="{home}"
now=$(date +%s)
beat=$(stat -c %Y "$home/dispatcher.heartbeat" 2>/dev/null || true)
running=$(grep -l '"status": "running"' "$home"/jobs/*/state.json 2>/dev/null | wc -l)
printf 'now=%s beat=%s running=%s\\n' "$now" "$beat" "$running"
"""
"""The two facts the dead-dispatcher rule judges, read straight off the host's
own files (`gpuc.host.paths`): the dispatcher's heartbeat is a file's mtime,
and a job the host believes is running says so in its `state.json`.

Read with `stat` and `grep` rather than by running the host's own package,
because the machine reconciling a pod may never have bootstrapped it and so
knows no interpreter there to run anything with. Both clocks are the host's, so
no skew between the two machines can make a live pod look silent.
"""


def pulse(transport: Transport, home: str, *, timeout: float = ASK_TIMEOUT_S) -> Liveness:
    """Whether the host behind `transport` is beating, or running anything."""
    try:
        result = transport.run(PULSE.format(home=home), timeout=timeout, check=False)
    except TransportError:
        return Liveness(reachable=False)
    if result.returncode != 0:
        return Liveness(reachable=False)
    values = dict(pair.split("=", 1) for pair in result.stdout.split() if pair.count("=") == 1)
    if "now" not in values:
        return Liveness(reachable=False)
    return Liveness(
        reachable=True,
        heartbeat_age_s=_age(values.get("now"), values.get("beat")),
        running_jobs=_int(values.get("running")),
    )


def _age(now: str | None, beat: str | None) -> float | None:
    try:
        return float(now or "") - float(beat or "")
    except ValueError:
        return None


def _int(value: str | None) -> int:
    try:
        return int(value or "")
    except ValueError:
        return 0


def remember(record: DesiredHost) -> bool:
    """Cache what a pod said in `desired/`. False: something else got there first.

    The cache is what lets this machine keep watching a pod it can no longer
    ask -- a wedged pod nobody would otherwise reap from here -- and what makes
    `gpuc pods` show it as wanted. A record already under that name is left
    alone: it belongs to whatever wrote it, and this one can be asked for again.
    """
    with state_lock():
        if read_desired(record.name) is not None:
            return False
        write_desired(record)
    return True


def _why(exc: BaseException) -> str:
    """The first two lines of a failure, on one line.

    Two rather than one because the line that matters -- `Permission denied
    (publickey)`, which is how "this machine holds no key for that pod" reads
    -- is the one under the "could not reach host" the wrapper adds.
    """
    lines = [line.strip() for line in str(exc).strip().splitlines() if line.strip()]
    return "; ".join(lines[:2]) if lines else type(exc).__name__
