"""What a rented pod says about itself, and what a second machine makes of it."""

from __future__ import annotations

from gpuc.control.config import HostEntry, Rental
from gpuc.control.rented import address_for, offer_of, pod_record
from tests.fakeprovider import make_offer, running_pod


def test_the_provider_block_a_pod_is_given_holds_its_own_record() -> None:
    address = HostEntry(name="gpuc-a-111", rental=Rental(pod_id="pod1"))
    record = pod_record(address, make_offer(), "2026-09-15T12:00:00+00:00")
    assert record["kind"] == "runpod" and record["pod_id"] == "pod1"
    assert record["offer"]["name"] == "A40"
    assert record["created_at"] == "2026-09-15T12:00:00+00:00"


def test_the_offer_is_read_back_off_the_pods_own_record() -> None:
    address = HostEntry(name="gpuc-a-111", rental=Rental(pod_id="pod1"))
    offer = offer_of(pod_record(address, make_offer(), "2026-09-15T12:00:00+00:00"))
    assert offer is not None
    assert offer.name == "A40" and offer.price_usd_hr == 0.49


def test_a_record_with_no_offer_or_an_unreadable_one_costs_a_reuse_not_a_crash() -> None:
    """A config written before the record existed, or by a build whose `Offer`
    this one cannot read, is a pod that will not be reused -- never an error
    on the path to buying another, and never an empty offer that a request
    with no constraints would match."""
    assert offer_of(None) is None
    assert offer_of({"kind": "runpod", "pod_id": "pod1"}) is None
    assert offer_of({"offer": {"vram_gb": "lots"}}) is None


def test_an_address_is_only_what_the_provider_says() -> None:
    pod = running_pod("gpuc-a-111", "pod1")
    address = address_for("gpuc-a-111", pod, "runpod")
    assert address is not None
    assert (address.kind, address.ssh, address.port, address.pod_id) == (
        "rental",
        "root@1.2.3.4",
        22000,
        "pod1",
    )
    assert address.config.gpus == []


def test_a_pod_with_no_ssh_endpoint_has_no_address_yet() -> None:
    doorless = running_pod("gpuc-a-111", "pod1").model_copy(update={"ssh_direct": None})
    assert address_for("gpuc-a-111", doorless, "runpod") is None
