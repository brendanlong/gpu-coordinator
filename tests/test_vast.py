from __future__ import annotations

import io
import json
import os
import urllib.error
import urllib.request
from email.message import Message
from pathlib import Path
from typing import Any

import pytest

from gpuc.control.actions import UsageError
from gpuc.control.config import Settings
from gpuc.control.hosts import RentalOptions
from gpuc.control.providers.base import Constraints, CreateRefused, ProviderError, Unauthorized
from gpuc.control.providers.runpod import RunPodProvider
from gpuc.control.providers.vast import VastError, VastProvider, pod_from_api, search_query
from gpuc.host import terminate
from gpuc.host.jobs import HostConfig

FIXTURES = Path(__file__).parent / "fixtures"
"""`vast_bundles.json` is three offers from a real search, trimmed to the
fields gpuc reads, one renamed and slowed down so the filters have something
to drop."""

INSTANCE = {
    "id": 4242,
    "label": "gpuc-e2e-abc123",
    "actual_status": "running",
    "cur_state": "running",
    "dph_total": 0.374,
    "gpu_name": "RTX 4090",
    "num_gpus": 1,
    "cuda_max_good": 13.2,
    "gpu_util": 37.0,
    "start_date": 1794680054.6,
    "public_ipaddr": "203.0.113.7\n",
    "ports": {"22/tcp": [{"HostIp": "0.0.0.0", "HostPort": "40022"}]},
}


class RecordedVast(VastProvider):
    """The real provider, answered from fixtures instead of the network."""

    def __init__(self, answers: dict[tuple[str, str], Any] | None = None) -> None:
        super().__init__(api_key="test-key", sleep=lambda _: None)
        self.answers = answers or {}
        self.requests: list[tuple[str, str, Any]] = []

    def _json(self, method: str, path: str, body: Any = None) -> Any:
        self.requests.append((method, path, body))
        if (method, path) == ("POST", "/bundles/"):
            return json.loads((FIXTURES / "vast_bundles.json").read_text())
        answer = self.answers[(method, path)]
        if isinstance(answer, Exception):
            raise answer
        return answer


def test_names_match_loosely_and_the_cheapest_comes_first() -> None:
    offers = RecordedVast().offers(Constraints(gpu_names=["rtx4090"]))
    assert [o.offer_id for o in offers] == ["59244968", "59360049"]
    offer = offers[0]
    assert (offer.name, offer.gpu_ram_gb, offer.cuda_versions) == ("RTX 4090", 24.6, ["13.2"])
    assert offer.cloud is None and offer.label == "RTX 4090 $0.374/h"


def test_the_search_carries_every_floor_and_never_the_names() -> None:
    constraints = Constraints(
        gpu_names=["RTX 4090"],
        max_price_usd_hr=0.5,
        gpu_count=2,
        disk_gb=80,
        minimums={"gpu_ram_gb": 24, "inet_down_mbps": 500, "reliability": 0.98},
    )
    query = search_query(constraints)
    assert "gpu_name" not in query
    assert query["num_gpus"] == {"eq": 2}
    assert query["disk_space"] == {"gte": 80}
    assert query["dph_total"] == {"lte": 0.5}
    assert query["gpu_ram"] == {"gte": 24 * 1000}
    assert query["allocated_storage"] == 80
    assert query["inet_down"] == {"gte": 500}
    assert query["reliability"] == {"gte": 0.98}
    assert query["direct_port_count"] == {"gte": 1}


def test_floors_drop_offers_the_search_returned_anyway() -> None:
    """The same rule a registered rental is matched by: a slow machine is not
    one `--min-inet-down 500` rents, whatever the search said."""
    offers = RecordedVast().offers(Constraints(minimums={"inet_down_mbps": 500}))
    assert "58777468" not in [o.offer_id for o in offers]
    assert len(offers) == 2


def test_a_name_no_offer_has_says_which_names_there_are() -> None:
    with pytest.raises(ProviderError) as caught:
        RecordedVast().offers(Constraints(gpu_names=["H100 SXM"]))
    assert "the GPUs that do: RTX 3090, RTX 4090" in str(caught.value)


def test_an_offer_someone_else_took_is_a_refusal_not_a_maybe() -> None:
    gone = VastError("PUT", "/asks/1/", 410, '{"error": "no_such_ask"}')
    provider = RecordedVast({("PUT", "/asks/59244968/"): gone})
    offer = provider.offers(Constraints(gpu_names=["RTX 4090"]))[0]
    with pytest.raises(CreateRefused):
        provider.create(offer, "gpuc-e2e-abc123", image="img", disk_gb=20)


def test_create_rents_the_offer_by_its_id_with_direct_ssh() -> None:
    provider = RecordedVast({("PUT", "/asks/59244968/"): {"success": True, "new_contract": 4242}})
    offer = provider.offers(Constraints(gpu_names=["RTX 4090"]))[0]
    pod = provider.create(offer, "gpuc-e2e-abc123", image="img", disk_gb=20)
    assert pod.id == "4242"
    _, _, body = provider.requests[-1]
    assert body["runtype"] == "ssh_direct" and body["disk"] == 20 and body["image"] == "img"
    assert body["label"] == "gpuc-e2e-abc123"


def test_create_refuses_a_name_without_our_prefix() -> None:
    provider = RecordedVast()
    offer = provider.offers(Constraints(gpu_names=["RTX 4090"]))[0]
    with pytest.raises(ProviderError, match="must start with"):
        provider.create(offer, "scratch", image="img", disk_gb=20)
    assert not any(method == "PUT" for method, _, _ in provider.requests)


def test_an_instance_reads_as_a_pod_with_its_direct_ssh_door() -> None:
    pod = pod_from_api(INSTANCE)
    assert (pod.id, pod.name, pod.status) == ("4242", "gpuc-e2e-abc123", "running")
    assert pod.ssh_direct is not None
    assert (pod.ssh_direct.host, pod.ssh_direct.port, pod.ssh_direct.username) == (
        "203.0.113.7",
        40022,
        "root",
    )
    assert pod.gpu_utils == [37] and pod.cuda_version == "13.2"
    assert VastProvider(api_key="k").is_running(pod)


def test_an_instance_without_a_mapped_port_has_no_door_yet() -> None:
    loading = {**INSTANCE, "actual_status": "loading", "ports": {}}
    assert pod_from_api(loading).ssh_direct is None
    assert pod_from_api({**INSTANCE, "actual_status": None}).status == "created"


def test_a_released_contract_is_gone_and_an_exited_one_dead() -> None:
    provider = VastProvider(api_key="k")
    destroyed = pod_from_api({**INSTANCE, "cur_state": "unloaded"})
    assert provider.is_gone(destroyed)
    exited = pod_from_api({**INSTANCE, "actual_status": "exited"})
    assert provider.is_dead(exited) and not provider.is_gone(exited)


def test_get_is_none_only_when_vast_says_it_has_no_such_instance() -> None:
    provider = RecordedVast(
        {
            ("GET", "/instances/1/"): {"instances": None},
            ("GET", "/instances/2/"): VastError("GET", "/instances/2/", 404, "{}"),
        }
    )
    assert provider.get("1") is None
    with pytest.raises(ProviderError):
        provider.get("2")


def test_a_key_vast_does_not_know_is_never_an_instance_that_is_gone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Vast answers an unknown key with a 404, the status a missing
    instance would have, and a rental read as gone is forgotten while it
    bills."""

    def refuse(request: urllib.request.Request, timeout: float) -> Any:
        body = io.BytesIO(b'{"success":false,"error":"auth_error","msg":"Invalid user key"}')
        raise urllib.error.HTTPError(request.full_url, 404, "Not Found", Message(), body)

    monkeypatch.setattr("urllib.request.urlopen", refuse)
    provider = VastProvider(api_key="revoked", sleep=lambda _: None)
    with pytest.raises(Unauthorized):
        provider.get("4242")
    with pytest.raises(Unauthorized):
        provider.terminate("4242")


def test_terminate_treats_an_instance_already_gone_as_done() -> None:
    missing = VastError("DELETE", "/instances/1/", 404, "")
    RecordedVast({("DELETE", "/instances/1/"): missing}).terminate("1")


def test_ports_in_another_shape_are_no_door_rather_than_a_crash() -> None:
    assert pod_from_api({**INSTANCE, "ports": [8080, 8081]}).ssh_direct is None
    assert pod_from_api({**INSTANCE, "ports": {"22/tcp": []}}).ssh_direct is None


def test_an_offline_machine_is_not_dead() -> None:
    """Vast lost touch with it; it may still be running a job over ssh."""
    offline = pod_from_api({**INSTANCE, "actual_status": "offline"})
    assert not VastProvider(api_key="k").is_dead(offline)


def test_a_card_meets_the_floor_its_name_says() -> None:
    """Vast reports a 24 GB card as 24564 MB."""
    offers = RecordedVast().offers(Constraints(minimums={"gpu_ram_gb": 24}))
    assert len(offers) == 3
    assert search_query(Constraints(minimums={"gpu_ram_gb": 24}))["gpu_ram"] == {"gte": 24000}


def test_the_ssh_key_is_added_once() -> None:
    key = "ssh-ed25519 AAAAblob me@laptop"
    known = RecordedVast({("GET", "/ssh/"): [{"public_key": "ssh-ed25519 AAAAblob other"}]})
    assert known.ensure_ssh_key(key) is False
    fresh = RecordedVast({("GET", "/ssh/"): [], ("POST", "/ssh/"): {"success": True}})
    assert fresh.ensure_ssh_key(key) is True
    assert fresh.requests[-1] == ("POST", "/ssh/", {"ssh_key": key})


def test_a_floor_the_provider_cannot_search_by_is_a_usage_error() -> None:
    options = RentalOptions(
        provider="runpod", gpu_names=["A40"], minimums={"inet_down_mbps": 500.0}
    )
    with pytest.raises(UsageError, match="--min-inet-down"):
        options.constraints(RunPodProvider(api_key="k"), Settings())


def test_cloud_is_a_runpod_choice_only() -> None:
    options = RentalOptions(provider="vast", gpu_names=["RTX 4090"], clouds=["COMMUNITY"])
    with pytest.raises(UsageError, match="--cloud"):
        options.constraints(VastProvider(api_key="k"), Settings())
    plain = RentalOptions(provider="vast", gpu_names=["RTX 4090"])
    assert plain.constraints(VastProvider(api_key="k"), Settings()).clouds == []
    runpod = RentalOptions(provider="runpod", gpu_names=["A40"])
    assert runpod.constraints(RunPodProvider(api_key="k"), Settings()).clouds == ["SECURE"]


# -- the instance ending itself ---------------------------------------------------


def test_an_instance_finds_its_own_id_and_key_in_pid_1s_environment(
    gpuc_home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    environ = tmp_path / "environ"
    environ.write_bytes(b"PATH=/usr/bin\0CONTAINER_ID=4242\0CONTAINER_API_KEY=inst-key\0")
    monkeypatch.setenv("GPUC_CONTAINER_ENVIRONMENT", str(environ))
    monkeypatch.delenv("CONTAINER_ID", raising=False)
    monkeypatch.delenv("CONTAINER_API_KEY", raising=False)
    calls: list[tuple[str, str]] = []
    config = HostConfig(host="gpuc-e2e-abc123", provider={"kind": "vast"})
    terminate.self_terminate(config, terminate_call=lambda pod, key: calls.append((pod, key)) or "")
    assert calls == [("4242", "inst-key")]


@pytest.mark.rental
@pytest.mark.skipif(not os.environ.get("VAST_API_KEY"), reason="needs VAST_API_KEY")
def test_live_search_finds_a_cheap_rtx_4090_with_every_floor_known() -> None:
    """Read-only: the search and its parsing against the real marketplace."""
    constraints = Constraints(
        gpu_names=["rtx4090"], max_price_usd_hr=1.0, minimums={"inet_down_mbps": 100}
    )
    offers = VastProvider().offers(constraints)
    assert offers, "no RTX 4090 under $1/h on Vast right now"
    offer = offers[0]
    assert offer.name == "RTX 4090" and offer.offer_id
    assert offer.gpu_ram_gb and offer.gpu_ram_gb > 20
    assert offer.inet_down_mbps is not None and offer.inet_down_mbps >= 100
    assert [o.price_usd_hr for o in offers] == sorted(o.price_usd_hr for o in offers)


def test_a_destroy_vast_refused_with_a_200_is_no_destroy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(terminate, "_call", lambda *a, **k: '{"success": false, "msg": "nope"}')
    with pytest.raises(terminate.TerminateError):
        terminate.vast_terminate("4242", "inst-key")
    monkeypatch.setattr(terminate, "_call", lambda *a, **k: '{"success": true}')
    terminate.vast_terminate("4242", "inst-key")
