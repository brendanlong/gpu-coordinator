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

from gpuc.control.config import HostEntry, Rental
from gpuc.control.providers.base import Offer, Pod


def pod_record(address: HostEntry, offer: Offer, created_at: str) -> dict[str, Any]:
    """The `provider` block a pod is given: its own record of what it was bought as.

    `kind` and `pod_id` name the rental. What follows is what a machine that
    did not create this pod needs in order to judge it: the offer
    `pick_reusable_host` compares against, and when the pod was bought.
    """
    return {
        **(address.provider_block() or {}),
        "offer": offer.model_dump(mode="json"),
        "created_at": created_at,
    }


def address_for(name: str, pod: Pod, provider: str) -> HostEntry | None:
    """How to reach this pod, and nothing about what it is. None: no door yet.
    `provider` is the name of the one the pod came from (`Provider.name`),
    which is what `actions.PROVIDERS` finds it by again."""
    if pod.ssh_direct is None:
        return None
    return HostEntry(
        name=name,
        ssh=f"{pod.ssh_direct.username}@{pod.ssh_direct.host}",
        port=pod.ssh_direct.port,
        rental=Rental(provider=provider, pod_id=pod.id),
    )


def offer_of(provider: Mapping[str, Any] | None) -> Offer | None:
    """The offer a pod's own `provider` block says it was bought on, or None
    when the block has none this build can read: that costs a reuse, since
    nothing is known to compare a request with, and nothing more.
    """
    value = (provider or {}).get("offer")
    if not isinstance(value, dict):
        return None
    try:
        return Offer.model_validate(value)
    except ValidationError:
        return None
