"""`gpuc pods` and the ephemeral extras in `gpuc status`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gpuc.control import pods as pods_mod
from gpuc.control.cli import EXIT_OK, main
from gpuc.control.config import Settings, load_registry, registry_transaction
from gpuc.control.status import HostState, HostView, render
from tests.conftest import host_entry
from tests.fakeprovider import FakeProvider, PodScript, running_pod

FOREIGN = "other-someone-else"


def _register(pod_name: str, pod_id: str) -> None:
    with registry_transaction() as registry:
        registry.put(
            host_entry(
                name=pod_name,
                kind="runpod",
                pod_id=pod_id,
                ssh="root@1.2.3.4",
                python="/root/python",
            )
        )


@pytest.fixture
def provider() -> FakeProvider:
    fake = FakeProvider(existing=[running_pod(FOREIGN, "podF", cost=0.79, age_minutes=800)])
    fake.adopt(running_pod("gpuc-e2e-aaa", "pod1", age_minutes=12), PodScript(ssh_after_polls=0))
    fake.adopt(running_pod("gpuc-leak-bbb", "podL", age_minutes=300), PodScript(ssh_after_polls=0))
    return fake


def test_table_names_the_host_each_pod_is_here_and_counts_the_rest(
    control_env: Path, provider: FakeProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    _register("gpuc-e2e-aaa", "pod1")
    monkeypatch.setattr("gpuc.control.pods.dispatcher_heartbeat_age", lambda *a: 4.0)

    text = pods_mod.render(pods_mod.gather(Settings(), provider))

    lines = text.splitlines()
    assert lines[0].split() == list(pods_mod.COLUMNS)
    assert "gpuc-e2e-aaa" in text and "NVIDIA A40" in text and "12m" in text
    assert lines[1].split()[-2:] == ["gpuc-e2e-aaa", "4s"]
    assert lines[2].split()[-2:] == ["-", "-"]
    assert "not registered here: gpuc-leak-bbb (podL)" in text
    assert "2 pod(s) with our prefix, $0.98/h total" in text
    assert f"1 other pod(s) in the account, never touched: {FOREIGN} (RUNNING)" in text
    # Someone else's pod contributes no cost and no row of its own.
    assert FOREIGN not in "\n".join(lines[:3])


def test_a_young_unregistered_pod_is_flagged_as_possibly_still_provisioning(
    control_env: Path, provider: FakeProvider
) -> None:
    """The registry entry is written only after connect, minutes into a
    `submit --runpod`; a pod inside that window is not a leak to end."""
    text = pods_mod.render(pods_mod.gather(Settings(), provider, heartbeats=False))
    note = next(line for line in text.splitlines() if "provisioning ceiling" in line)
    assert note.startswith("gpuc-e2e-aaa:")
    assert "gpuc-leak-bbb" not in note


def test_two_pods_that_compare_equal_are_still_told_apart(control_env: Path) -> None:
    """Ours and theirs are separated by pod id, not by object equality: two
    pods with the same fields would otherwise hide each other from the table."""
    fake = FakeProvider(existing=[running_pod(FOREIGN, "podF"), running_pod(FOREIGN, "podG")])
    fake.adopt(running_pod("gpuc-e2e-aaa", "pod1"), PodScript(ssh_after_polls=0))
    view = pods_mod.gather(Settings(), fake, heartbeats=False)
    assert [row.pod.id for row in view.rows] == ["pod1"]
    assert [pod.id for pod in view.others] == ["podF", "podG"]


def test_a_pod_registered_nowhere_here_says_who_will_deal_with_it(
    control_env: Path, provider: FakeProvider
) -> None:
    text = pods_mod.render(pods_mod.gather(Settings(), provider, heartbeats=False))
    assert "gpuc host add <name> --pod <id>" in text
    # It bills until somebody ends it, and this is where the command to do that is.
    assert "gpuc host terminate <id> --force" in text
    view = pods_mod.gather(Settings(), provider, heartbeats=False)
    assert [row.host for row in view.rows] == [None, None]


def test_empty_account_renders_a_hint(control_env: Path) -> None:
    text = pods_mod.render(pods_mod.gather(Settings(), FakeProvider()))
    assert "(no pods with our prefix)" in text


def test_status_shows_the_pod_for_an_ephemeral_host() -> None:
    entry = host_entry(name="gpuc-e2e-aaa", kind="runpod", ssh="root@1.2.3.4")
    view = HostView(
        entry=entry,
        state=HostState.ANSWERED,
        heartbeat_age_s=3.0,
        owned=["GPU-1"],
        pod=running_pod("gpuc-e2e-aaa", "pod1", age_minutes=20),
    )
    text = render(view)
    assert "pod     pod1 RUNNING NVIDIA A40 $0.490/h cuda 12.8 age 20m" in text


def test_status_shows_the_pod_even_when_the_host_is_unreachable() -> None:
    entry = host_entry(name="gpuc-e2e-aaa", kind="runpod", ssh="root@1.2.3.4")
    view = HostView(
        entry=entry, state=HostState.UNREACHABLE, error="ssh timed out", pod=running_pod("n", "p")
    )
    text = render(view)
    assert "UNREACHABLE" in text and "pod     p RUNNING" in text


def test_status_forgets_a_rental_the_provider_no_longer_has(
    control_env: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rental that ended itself is a state, not a host nobody can reach."""
    _register("gpuc-e2e-aaa", "pod1")
    monkeypatch.setattr("gpuc.control.actions.make_provider", lambda settings: FakeProvider())
    capsys.readouterr()

    assert main(["status"]) == EXIT_OK
    captured = capsys.readouterr()
    assert "POD GONE" in captured.out
    assert "this rental has ended" in captured.out
    assert "forgetting host gpuc-e2e-aaa" in captured.err
    assert load_registry().hosts == {}


def test_status_json_forgets_the_rental_it_just_reported(
    control_env: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The document still lists the host; the entry is gone once it is printed."""
    _register("gpuc-e2e-aaa", "pod1")
    monkeypatch.setattr("gpuc.control.actions.make_provider", lambda settings: FakeProvider())
    capsys.readouterr()

    assert main(["status", "--json"]) == EXIT_OK
    document = json.loads(capsys.readouterr().out)
    (host,) = document["hosts"]
    assert host["pod_gone"] is True and host["pod_terminated"] is True
    assert load_registry().hosts == {}
