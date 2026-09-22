from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from gpuc.control.actions import make_provider
from gpuc.control.config import Settings
from gpuc.control.providers.base import (
    Constraints,
    Offer,
    Pod,
    PodStatus,
    ProviderError,
    owned_pods,
)
from gpuc.control.providers.runpod import RunPodProvider

FIXTURES = Path(__file__).parent / "fixtures"


class RecordedRunPod(RunPodProvider):
    def __init__(self, pods: list[dict[str, Any]] | None = None) -> None:
        super().__init__(api_key="test-key")
        self.pods = pods or []
        self.requests: list[tuple[str, str, dict[str, Any] | None]] = []

    def _json(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        body: Any = None,
    ) -> Any:
        self.requests.append((method, path, params))
        if path == "/catalog/gpus":
            assert params is not None
            tier = str(params["cloud"]).lower()
            return json.loads((FIXTURES / f"catalog_gpus_{tier}.json").read_text())
        if path == "/pods":
            return {"pods": self.pods}
        raise AssertionError(f"unexpected request {method} {path}")


def pod(name: str, cost: float, status: PodStatus = "RUNNING") -> Pod:
    return Pod(id=name, name=name, status=status, cost_usd_hr=cost)


def test_name_list_matches_short_name_and_catalog_id() -> None:
    provider = RecordedRunPod()
    by_short = provider.offers(Constraints(gpu_names=["A40"]))
    by_id = provider.offers(Constraints(gpu_names=["nvidia a40"]))
    assert [o.gpu_id for o in by_short] == ["NVIDIA A40"]
    assert [o.gpu_id for o in by_id] == ["NVIDIA A40"]
    assert by_short[0].price_usd_hr == 0.49
    assert by_short[0].vram_gb == 48


def test_unavailable_tier_is_dropped_and_tiers_are_queried_separately() -> None:
    provider = RecordedRunPod()
    offers = provider.offers(Constraints(gpu_names=["A40"], clouds=["SECURE", "COMMUNITY"]))
    assert [(o.cloud, o.price_usd_hr) for o in offers] == [("SECURE", 0.49)]
    assert [params and params["cloud"] for _, _, params in provider.requests] == [
        "SECURE",
        "COMMUNITY",
    ]


def test_offers_sorted_by_price_across_tiers() -> None:
    offers = RecordedRunPod().offers(Constraints(clouds=["SECURE", "COMMUNITY"]))
    prices = [o.price_usd_hr for o in offers]
    assert prices == sorted(prices)
    assert (offers[0].gpu_id, offers[0].cloud) == ("NVIDIA GeForce RTX 3090", "COMMUNITY")


def test_min_vram_and_max_price_filters() -> None:
    provider = RecordedRunPod()
    assert all(o.vram_gb >= 48 for o in provider.offers(Constraints(min_vram_gb=48)))
    assert [o.gpu_id for o in provider.offers(Constraints(min_vram_gb=80))] == [
        "NVIDIA A100 80GB PCIe"
    ]
    assert all(
        o.price_usd_hr <= 0.60
        for o in provider.offers(Constraints(max_price_usd_hr=0.60, clouds=["SECURE"]))
    )
    assert provider.offers(Constraints(gpu_names=["A40"], max_price_usd_hr=0.40)) == []


def test_cuda_floor_needs_an_available_version() -> None:
    provider = RecordedRunPod()
    assert [o.gpu_id for o in provider.offers(Constraints(gpu_names=["RTX 3090"]))] == [
        "NVIDIA GeForce RTX 3090"
    ]
    # 3090 offers 13.2 and 13.3 but only 13.0 has capacity.
    assert provider.offers(Constraints(gpu_names=["RTX 3090"], cuda_min="13.2")) == []
    assert provider.offers(Constraints(gpu_names=["RTX 4090"], cuda_min="13.2"))[
        0
    ].cuda_versions == [
        "12.8",
        "13.0",
        "13.2",
    ]


def test_cuda_floor_compares_numerically_not_lexically() -> None:
    offers = RecordedRunPod().offers(Constraints(gpu_names=["A40"], cuda_min="12.11"))
    assert [o.gpu_id for o in offers] == ["NVIDIA A40"]


def test_gpu_count_scales_price_and_respects_max_count() -> None:
    offers = RecordedRunPod().offers(Constraints(gpu_names=["A40"], gpu_count=2))
    assert offers[0].price_usd_hr == pytest.approx(0.98)
    assert RecordedRunPod().offers(Constraints(gpu_names=["A40"], gpu_count=20)) == []


def test_cuda_min_is_forwarded_to_the_catalog_query() -> None:
    provider = RecordedRunPod()
    provider.offers(Constraints(gpu_names=["A40"], cuda_min="12.8"))
    assert provider.requests[0][2] == {
        "include": "AVAILABILITY",
        "product": "POD",
        "cloud": "SECURE",
        "count": 1,
        "minCudaVersion": "12.8",
    }


def test_owned_pods_ignore_foreign_pods() -> None:
    pods = [pod("other-p-head", 0.49), pod("other-d3-head", 0.49), pod("gpuc-a", 0.49)]
    assert [p.name for p in owned_pods(pods)] == ["gpuc-a"]


def test_create_requires_our_prefix() -> None:
    provider = RecordedRunPod()
    offer = provider.offers(Constraints(gpu_names=["A40"]))[0]
    with pytest.raises(ProviderError, match="must start with"):
        provider.create(offer, "scratch-pod")
    assert not any(path == "/pods" for _, path, _ in provider.requests)


def test_an_offer_with_a_null_field_parses_with_the_default() -> None:
    """A pod's config records the offer it was rented on, written by whichever
    build rented it; a null there may not make the pod unreadable."""
    offer = Offer.model_validate(
        {"gpu_id": "NVIDIA A40", "vram_gb": None, "price_usd_hr": None, "future": 1}
    )
    assert (offer.gpu_id, offer.vram_gb, offer.price_usd_hr) == ("NVIDIA A40", 0, 0.0)


def test_an_unknown_provider_kind_is_a_provider_error() -> None:
    with pytest.raises(ProviderError, match="nosuch"):
        make_provider(Settings(), "nosuch")


def test_runpod_says_which_statuses_are_dead_and_which_are_gone() -> None:
    provider = RecordedRunPod()
    assert provider.is_dead(None) and provider.is_gone(None)
    assert not provider.is_dead(pod("gpuc-a", 0.49))
    exited = pod("gpuc-a", 0.49, status="EXITED")
    assert provider.is_dead(exited) and not provider.is_gone(exited)
    terminated = pod("gpuc-a", 0.49, status="TERMINATED")
    assert provider.is_dead(terminated) and provider.is_gone(terminated)
    assert provider.broken_host is not None
    assert provider.broken_host.search("failed to create shim task: OCI runtime create failed")


@pytest.mark.runpod
@pytest.mark.skipif(not os.environ.get("RUNPOD_API_KEY"), reason="needs RUNPOD_API_KEY")
def test_live_a40_secure_offers() -> None:
    offers = RunPodProvider().offers(
        Constraints(gpu_names=["A40"], clouds=["SECURE"], cuda_min="12.8", max_price_usd_hr=0.60)
    )
    assert len(offers) == 1
    offer = offers[0]
    assert offer.gpu_id == "NVIDIA A40"
    assert offer.vram_gb == 48
    assert offer.cloud == "SECURE"
    assert 0.1 < offer.price_usd_hr <= 0.60
    assert offer.availability != "NONE"
    assert offer.matches_cuda_floor("12.8")


def test_max_price_caps_the_whole_pod_not_one_gpu() -> None:
    """--max-price is documented as the pod's price; at --gpu-count 2 it is 2x."""
    cheap = RecordedRunPod().offers(
        Constraints(gpu_names=["A40"], gpu_count=2, max_price_usd_hr=1.20)
    )
    assert [o.price_usd_hr for o in cheap] == [pytest.approx(0.98)]
    assert (
        RecordedRunPod().offers(Constraints(gpu_names=["A40"], gpu_count=2, max_price_usd_hr=0.60))
        == []
    )
