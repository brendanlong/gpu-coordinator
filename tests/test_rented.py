"""What a rented pod says about itself, and what a second machine makes of it."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from gpuc.control.config import HostEntry, Settings, read_desired
from gpuc.control.rented import (
    address_for,
    ask_pod,
    desired_from,
    pod_record,
    pulse,
    remember,
)
from gpuc.control.transport import CommandResult, LocalTransport, TransportError
from gpuc.host import jobs
from gpuc.host.jobs import JobState
from tests.fakehost import HOME, FakeHost
from tests.fakeprovider import make_offer, running_pod


def config_document(**overrides: Any) -> dict[str, Any]:
    document: dict[str, Any] = {
        "host": "gpuc-a-111",
        "gpus": ["GPU-1111"],
        "ttl_hours": 4.0,
        "idle_minutes": 5.0,
        "provider": {
            "kind": "runpod",
            "pod_id": "pod1",
            "offer": make_offer().model_dump(mode="json"),
            "created_at": "2026-09-15T12:00:00+00:00",
            "bootstrapped_at": "2026-09-15T12:09:00+00:00",
        },
    }
    document.update(overrides)
    return document


def test_the_provider_block_a_pod_is_given_holds_its_own_desired_record() -> None:
    address = HostEntry(name="gpuc-a-111", kind="runpod", pod_id="pod1")
    record = pod_record(address, make_offer(), "2026-09-15T12:00:00+00:00")
    assert record["kind"] == "runpod" and record["pod_id"] == "pod1"
    assert record["offer"]["name"] == "A40"
    assert record["created_at"] == "2026-09-15T12:00:00+00:00"


def test_the_desired_record_is_read_back_off_the_pods_config() -> None:
    record = desired_from("pod1", config_document())
    assert (record.name, record.pod_id) == ("gpuc-a-111", "pod1")
    assert record.offer.name == "A40" and record.offer.price_usd_hr == 0.49
    assert record.ttl_hours == 4.0 and record.idle_minutes == 5.0
    assert record.created_at == "2026-09-15T12:00:00+00:00"
    assert record.bootstrapped_at == "2026-09-15T12:09:00+00:00"


def test_a_config_that_names_no_host_does_not_adopt_as_local() -> None:
    """`HostConfig.from_dict` defaults `host` to `local`, which is this
    machine's own name: a record under it would have the reaper probing this
    box and then forgetting the user's own host."""
    nameless = config_document()
    nameless.pop("host")
    assert desired_from("pod1", nameless, name="gpuc-a-111").name == "gpuc-a-111"


def test_a_config_written_before_the_record_existed_is_still_bootstrapped() -> None:
    """Reading "no stamp" as "never bootstrapped" would reap it at the ceiling."""
    older = config_document(provider={"kind": "runpod", "pod_id": "pod1"}, created_at="")
    record = desired_from("pod1", older, name="gpuc-a-111", created_at="2026-09-15T12:00:00+00:00")
    assert record.bootstrapped and record.created_at == "2026-09-15T12:00:00+00:00"
    # An offer nobody recorded costs a reuse, never a pod.
    assert record.offer.name == ""


def test_an_unreadable_offer_does_not_take_the_record_with_it() -> None:
    record = desired_from("pod1", config_document(provider={"offer": {"vram_gb": "lots"}}))
    assert record.offer.vram_gb == 0 and record.pod_id == "pod1"


@pytest.fixture
def pod_host(monkeypatch: pytest.MonkeyPatch) -> FakeHost:
    host = FakeHost()
    monkeypatch.setattr(
        "gpuc.control.rented.transport_for", lambda entry, settings=None: host, raising=True
    )
    return host


def test_a_pod_holding_a_gpuc_config_is_ours_whoever_created_it(pod_host: FakeHost) -> None:
    pod_host.files[f"{HOME}/.gpuc/config.json"] = json.dumps(config_document())
    answer = ask_pod(running_pod("gpuc-a-111", "pod1"), Settings())
    assert answer.desired is not None and answer.desired.ttl_hours == 4.0
    assert answer.entry is not None and answer.entry.ssh == "root@1.2.3.4"


def test_a_pod_answering_just_now_is_recorded_as_seen_just_now(pod_host: FakeHost) -> None:
    """Without the stamp, the silence clock for a pod this machine has only now
    met starts at whenever it was bootstrapped -- and the first pulse that
    misses terminates it."""
    pod_host.files[f"{HOME}/.gpuc/config.json"] = json.dumps(config_document())
    answer = ask_pod(running_pod("gpuc-a-111", "pod1"), Settings())
    assert answer.desired is not None
    assert answer.desired.silent_since() == answer.desired.last_seen_at


def test_a_pod_with_no_gpuc_config_on_it_claims_nothing(pod_host: FakeHost) -> None:
    answer = ask_pod(running_pod("gpuc-a-111", "pod1"), Settings())
    assert answer.desired is None
    assert "has no /home/u/.gpuc/config.json" in answer.detail


def test_a_pod_with_no_ssh_endpoint_cannot_be_asked() -> None:
    doorless = running_pod("gpuc-a-111", "pod1").model_copy(update={"ssh_direct": None})
    answer = ask_pod(doorless, Settings())
    assert (answer.desired, answer.entry) == (None, None)
    assert "no ssh endpoint" in answer.detail


def test_a_pod_that_does_not_answer_ssh_claims_nothing_either(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Silence is not a verdict: this is the pod the reaper must not terminate."""
    refused = CommandResult("fake", ["ssh"], 255, "", "Permission denied (publickey).")

    class Refuses(FakeHost):
        def run(self, command: str, *, timeout: float = 120.0, check: bool = True) -> CommandResult:
            raise TransportError(refused)

    monkeypatch.setattr("gpuc.control.rented.transport_for", lambda entry, settings=None: Refuses())
    answer = ask_pod(running_pod("gpuc-a-111", "pod1"), Settings())
    assert answer.desired is None
    assert "Permission denied" in answer.detail


def gpuc_home(root: Path, *, beat_age_s: float | None = None, running: int = 0) -> str:
    """A gpuc home on this machine, so the pulse script runs for real."""
    home = root / ".gpuc"
    (home / "jobs").mkdir(parents=True)
    if beat_age_s is not None:
        beat = home / "dispatcher.heartbeat"
        beat.touch()
        stamp = beat.stat().st_mtime - beat_age_s
        os.utime(beat, (stamp, stamp))
    for index in range(running):
        job = home / "jobs" / f"job{index}"
        job.mkdir()
        # Through the host's own writer: the pulse greps this file, and the
        # spelling it greps for is whatever the host package writes.
        jobs.atomic_write_json(job / "state.json", JobState(status="running").to_dict())
    finished = home / "jobs" / "done"
    finished.mkdir()
    jobs.atomic_write_json(finished / "state.json", JobState(status="succeeded").to_dict())
    return str(home)


def test_the_pulse_reads_the_heartbeat_off_the_hosts_own_files(tmp_path: Path) -> None:
    home = gpuc_home(tmp_path, beat_age_s=8.0, running=2)
    state = pulse(LocalTransport(), home)
    assert state.reachable and state.running_jobs == 2
    assert state.heartbeat_age_s is not None and 7.0 <= state.heartbeat_age_s <= 30.0
    assert state.alive


def test_a_host_that_has_never_beaten_is_not_alive(tmp_path: Path) -> None:
    state = pulse(LocalTransport(), gpuc_home(tmp_path))
    assert state.reachable and not state.alive
    assert state.heartbeat_age_s is None and state.running_jobs == 0
    assert "never beaten" in state.describe()


def test_a_running_job_keeps_a_host_alive_with_no_heartbeat(tmp_path: Path) -> None:
    state = pulse(LocalTransport(), gpuc_home(tmp_path, running=1))
    assert state.alive and state.running_jobs == 1


def test_a_gpuc_home_that_is_not_there_answers_silence_not_a_crash(tmp_path: Path) -> None:
    state = pulse(LocalTransport(), str(tmp_path / "nowhere"))
    assert state.reachable and not state.alive


def test_what_a_pod_said_is_cached_but_never_overwrites_a_record(control_env: Path) -> None:
    record = desired_from("pod1", config_document())
    assert remember(record)
    assert read_desired("gpuc-a-111") is not None
    # A record already under that name belongs to whatever wrote it.
    assert not remember(record.model_copy(update={"pod_id": "pod2"}))
    cached = read_desired("gpuc-a-111")
    assert cached is not None and cached.pod_id == "pod1"


def test_an_address_is_only_what_the_provider_says(control_env: Path) -> None:
    pod = running_pod("gpuc-a-111", "pod1")
    address = address_for("gpuc-a-111", pod)
    assert address is not None
    assert (address.kind, address.ssh, address.port, address.pod_id) == (
        "runpod",
        "root@1.2.3.4",
        22000,
        "pod1",
    )
    assert address.gpus == []


def test_the_pulse_expands_a_tilde_the_way_the_host_would(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--gpuc-home ~/.gpuc` is not expanded by the quoting that keeps the path
    safe, and a home read literally finds no heartbeat -- which on this path
    means "terminate it"."""
    monkeypatch.setenv("HOME", str(tmp_path))
    gpuc_home(tmp_path, beat_age_s=5.0)
    assert pulse(LocalTransport(), "~/.gpuc").alive


def test_the_pulse_survives_a_home_with_a_space_in_it(tmp_path: Path) -> None:
    root = tmp_path / "my pods"
    root.mkdir()
    assert pulse(LocalTransport(), gpuc_home(root, beat_age_s=5.0)).alive
