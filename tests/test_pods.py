"""`gpuc pods` and the ephemeral extras in `gpuc status`."""

from __future__ import annotations

from pathlib import Path

import pytest

from gpuc.control import pods as pods_mod
from gpuc.control.config import (
    DesiredHost,
    HostEntry,
    Settings,
    desired_dir,
    registry_transaction,
    utc_now,
    write_desired,
)
from gpuc.control.status import HostView, render
from tests.fakeprovider import FakeProvider, PodScript, make_offer, running_pod

FOREIGN = "subrep-someone-else"


def _register(pod_name: str, pod_id: str, *, desired: bool = True) -> None:
    with registry_transaction() as registry:
        registry.put(
            HostEntry(
                name=pod_name,
                kind="runpod",
                pod_id=pod_id,
                ssh="root@1.2.3.4",
                python="/root/python",
            )
        )
    if desired:
        write_desired(
            DesiredHost(
                name=pod_name,
                pod_id=pod_id,
                offer=make_offer(),
                created_at=utc_now(),
                ceiling_at=utc_now(),
                bootstrapped_at=utc_now(),
            )
        )


@pytest.fixture
def provider() -> FakeProvider:
    fake = FakeProvider(existing=[running_pod(FOREIGN, "podF", cost=0.79, age_minutes=800)])
    fake.adopt(running_pod("gpuc-e2e-aaa", "pod1", age_minutes=12), PodScript(ssh_after_polls=0))
    fake.adopt(running_pod("gpuc-leak-bbb", "podL", age_minutes=300), PodScript(ssh_after_polls=0))
    return fake


def test_table_marks_undesired_pods_and_counts_the_rest(
    control_env: Path, provider: FakeProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    desired_dir().mkdir(parents=True, exist_ok=True)
    _register("gpuc-e2e-aaa", "pod1")
    monkeypatch.setattr("gpuc.control.pods.dispatcher_heartbeat_age", lambda *a: 4.0)

    text = pods_mod.render(pods_mod.gather(Settings(), provider))

    lines = text.splitlines()
    assert lines[0].split() == list(pods_mod.COLUMNS)
    assert "gpuc-e2e-aaa" in text and "NVIDIA A40" in text and "12m" in text
    assert "yes" in lines[1] and "4s" in lines[1]
    assert "DESIRED=NO on gpuc-leak-bbb" in text
    assert "2 pod(s) with our prefix, $0.98/h total" in text
    assert f"1 other pod(s) in the account, never touched: {FOREIGN} (RUNNING)" in text
    # Someone else's pod contributes no cost and no row of its own.
    assert FOREIGN not in "\n".join(lines[:3])


def test_unreadable_desired_state_is_a_note_not_a_crash(
    control_env: Path, provider: FakeProvider
) -> None:
    view = pods_mod.gather(Settings(), provider, heartbeats=False)
    assert view.notes and "unreadable" in view.notes[0]
    assert all(not row.desired for row in view.rows)


def test_empty_account_renders_a_hint(control_env: Path) -> None:
    desired_dir().mkdir(parents=True, exist_ok=True)
    text = pods_mod.render(pods_mod.gather(Settings(), FakeProvider()))
    assert "(no pods with our prefix)" in text


def test_status_shows_the_pod_for_an_ephemeral_host() -> None:
    entry = HostEntry(name="gpuc-e2e-aaa", kind="runpod", ssh="root@1.2.3.4", ttl_hours=1.0)
    view = HostView(
        entry=entry,
        reachable=True,
        heartbeat_age_s=3.0,
        owned=["GPU-1"],
        pod=running_pod("gpuc-e2e-aaa", "pod1", age_minutes=20),
    )
    text = render(view)
    assert "pod     pod1 RUNNING NVIDIA A40 $0.490/h cuda 12.8 age 20m" in text


def test_status_shows_the_pod_even_when_the_host_is_unreachable() -> None:
    entry = HostEntry(name="gpuc-e2e-aaa", kind="runpod", ssh="root@1.2.3.4")
    view = HostView(entry=entry, reachable=False, error="ssh timed out", pod=running_pod("n", "p"))
    text = render(view)
    assert "UNREACHABLE" in text and "pod     p RUNNING" in text
