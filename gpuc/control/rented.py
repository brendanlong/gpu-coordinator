"""What a rented pod says about itself, to whichever machine is asking.

A pod's `config.json` -- the file a host owns, and the only copy of what it is
-- holds the offer it was bought on and when it was created, under the
`provider` block that already named its `kind` and `pod_id`. That is what lets
`gpuc host add --pod` on a second machine register the pod from the pod
itself, and what `submit --runpod` compares a request against when it reuses
one, with nothing the creating machine remembers involved.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import ValidationError

from gpuc.control.config import HostEntry
from gpuc.control.providers.base import Offer, Pod


def pod_record(address: HostEntry, offer: Offer, created_at: str) -> dict[str, Any]:
    """The `provider` block a pod is given: its own record of what it was bought as.

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


def offer_of(provider: Mapping[str, Any] | None) -> Offer:
    """The offer a pod's own `provider` block says it was bought on.

    An offer that is missing or unreadable is an empty one: it costs a reuse,
    since `offer_satisfies` will not match it, and nothing more.
    """
    value = (provider or {}).get("offer")
    if not isinstance(value, dict):
        return Offer()
    try:
        return Offer.model_validate(value)
    except ValidationError:
        return Offer()
