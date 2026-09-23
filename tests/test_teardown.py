"""`gpuc host terminate`: the one command that ends a pod on purpose."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from gpuc.control import teardown
from gpuc.control.config import HostNotFound, Settings, registry_transaction
from gpuc.control.providers.base import ProviderError
from gpuc.control.remote import RemoteError
from gpuc.host.jobs import HostConfig
from tests.conftest import host_entry, load_registry
from tests.fakeprovider import FakeProvider, PodScript, running_pod

FOREIGN = "other-someone-else"


@pytest.fixture
def provider() -> FakeProvider:
    fake = FakeProvider(existing=[running_pod(FOREIGN, "podF")])
    fake.adopt(running_pod("gpuc-e2e-aaa", "pod1"), PodScript(ssh_after_polls=0))
    fake.adopt(running_pod("gpuc-leak-bbb", "podL"), PodScript(ssh_after_polls=0))
    return fake


def register(name: str = "gpuc-e2e-aaa", pod_id: str = "pod1") -> None:
    with registry_transaction() as registry:
        registry.put(
            host_entry(
                name=name,
                kind="rental",
                pod_id=pod_id,
                ssh="root@1.2.3.4",
                python="/root/python",
            )
        )


def host_payload(**overrides: Any) -> dict[str, Any]:
    document: dict[str, Any] = {
        "host": "gpuc-e2e-aaa",
        "gpus": ["GPU-1111"],
        "ephemeral": True,
        "draining": False,
        "dispatcher_heartbeat_age_s": 2.0,
        "queue": [],
        "jobs": [],
    }
    document.update(overrides)
    return document


def answering(monkeypatch: pytest.MonkeyPatch, payload: dict[str, Any] | None) -> None:
    """Point `status.gather`'s ssh half at a payload, or at a host that is gone."""

    class Session:
        config = HostConfig()

        def host_json(self, args: str, *, timeout: float = 0.0, check: bool = True) -> object:
            if payload is None:
                raise TimeoutError("ssh timed out")
            return payload

    def open_session(*args: object, **kwargs: object) -> Any:
        if payload is None:
            raise RemoteError("gpuc-e2e-aaa", "status", "ssh: connect: no route to host")
        return Session()

    monkeypatch.setattr("gpuc.control.remote.open_session", open_session)


def terminate(provider: FakeProvider, target: str, **kwargs: Any) -> teardown.Termination:
    return teardown.terminate(
        target,
        Settings(),
        registry=load_registry(),
        provider=provider,
        report=lambda _: None,
        sleep=lambda _: None,
        **kwargs,
    )


def test_an_idle_host_is_terminated_and_forgotten(
    control_env: Path, provider: FakeProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    register()
    answering(monkeypatch, host_payload())

    result = terminate(provider, "gpuc-e2e-aaa")

    assert provider.terminated == ["pod1"]
    assert (result.terminated, result.forgotten, result.checked) == (True, True, True)
    assert load_registry().hosts == {}
    document = result.document()
    assert document["host"] == "gpuc-e2e-aaa" and document["pod_id"] == "pod1"
    assert document["pod_status"] == "RUNNING" and document["cost_usd_hr"] == 0.49


def test_a_running_job_refuses_the_terminate_and_says_what_would_be_lost(
    control_env: Path, provider: FakeProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    register()
    answering(
        monkeypatch,
        host_payload(
            queue=[{"priority": 10, "job_id": "j-queued"}],
            jobs=[
                {"job_id": "j-queued", "name": "next", "status": "queued"},
                {
                    "job_id": "j-running",
                    "name": "train",
                    "status": "running",
                    "phase": "main",
                    "gpus": ["GPU-1111"],
                },
            ],
        ),
    )

    with pytest.raises(teardown.TerminateRefused) as error:
        terminate(provider, "gpuc-e2e-aaa")

    assert provider.terminated == []
    assert load_registry().hosts  # nothing forgotten either
    message = str(error.value)
    assert "train (j-running)" in message and "next (j-queued)" in message
    assert "--idle-min 0" in message and "--force" in message


def test_force_terminates_a_busy_host_without_asking_it_anything(
    control_env: Path, provider: FakeProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ssh round trip is skipped, not just its answer ignored: the pod this
    command exists for is one that cannot answer at all."""
    register()

    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("--force must not ask the host anything")

    monkeypatch.setattr("gpuc.control.remote.open_session", refuse)

    result = terminate(provider, "gpuc-e2e-aaa", force=True)

    assert provider.terminated == ["pod1"]
    assert (result.checked, result.terminated, result.forgotten) == (False, True, True)
    assert load_registry().hosts == {}


def test_a_host_that_cannot_be_asked_is_refused_until_force(
    control_env: Path, provider: FakeProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One ssh blip must not cost somebody six hours of training. An empty
    queue and an unanswered question are not the same fact, and only one of
    them is safe to act on -- so the unanswered one costs a `--force`."""
    register()
    answering(monkeypatch, None)

    with pytest.raises(teardown.TerminateRefused) as error:
        terminate(provider, "gpuc-e2e-aaa")

    assert provider.terminated == []
    assert "gpuc-e2e-aaa" in load_registry().hosts
    assert "could not ask gpuc-e2e-aaa" in str(error.value)
    assert "--force" in str(error.value)

    result = terminate(provider, "gpuc-e2e-aaa", force=True)

    assert provider.terminated == ["pod1"]
    assert (result.checked, result.terminated) == (False, True)


def test_a_finished_job_whose_outputs_are_not_confirmed_holds_the_terminate(
    control_env: Path, provider: FakeProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    register()
    answering(
        monkeypatch,
        host_payload(
            jobs=[
                {
                    "job_id": "j-done",
                    "name": "sweep",
                    "status": "succeeded",
                    "outputs_pending": True,
                }
            ]
        ),
    )

    with pytest.raises(teardown.TerminateRefused) as error:
        terminate(provider, "gpuc-e2e-aaa")

    assert "sweep (j-done)" in str(error.value)
    assert provider.terminated == []


def test_a_pod_this_machine_never_registered_is_refused_until_force(
    control_env: Path, provider: FakeProvider
) -> None:
    """Nothing here can ask it anything -- and another machine may well be
    running a job on it. The refusal says both ways on: adopt it, or insist."""
    with pytest.raises(teardown.TerminateRefused) as error:
        terminate(provider, "podL")

    assert provider.terminated == []
    assert "not registered on this machine" in str(error.value)
    assert "gpuc host add <name> --pod podL" in str(error.value)

    result = terminate(provider, "podL", force=True)

    assert provider.terminated == ["podL"]
    # No registry entry to forget, and `--force` asked nothing, so the document
    # claims neither: `host` is null and `checked` false.
    assert (result.document()["host"], result.checked, result.forgotten) == (None, False, False)


def test_a_pod_can_be_named_by_its_pod_name_too(control_env: Path, provider: FakeProvider) -> None:
    assert terminate(provider, "gpuc-leak-bbb", force=True).terminated
    assert provider.terminated == ["podL"]


def test_somebody_elses_pod_is_not_ours_to_end(control_env: Path, provider: FakeProvider) -> None:
    with pytest.raises(HostNotFound) as error:
        terminate(provider, "podF")
    assert provider.terminated == []
    assert "not ours" in str(error.value)


def test_a_non_rental_host_has_nothing_to_terminate(control_env: Path) -> None:
    with registry_transaction() as registry:
        registry.put(host_entry(name="gpubox", kind="ssh", ssh="me@gpubox"))

    with pytest.raises(teardown.TerminateRefused) as error:
        terminate(FakeProvider(), "gpubox")
    assert "gpuc host remove gpubox" in str(error.value)


def test_a_pod_that_is_already_gone_still_clears_the_registry_entry(
    control_env: Path, provider: FakeProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The stale-entry case `gpuc status` calls GONE. There is nothing to
    bill for and nothing that could be running, so no `--force` is demanded for
    a pod the provider itself says is dead -- the entry is all that is left."""
    register()
    provider.terminate("pod1")
    provider.terminated.clear()
    answering(monkeypatch, None)

    result = terminate(provider, "gpuc-e2e-aaa")

    assert provider.terminated == []
    assert (result.terminated, result.forgotten) == (False, True)
    assert load_registry().hosts == {}


def test_a_pod_the_provider_has_never_heard_of_is_not_a_terminate(
    control_env: Path, provider: FakeProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The registry entry outlived its pod entirely. Nothing to end, and the
    entry is the only thing left to clear."""
    register(pod_id="pod-long-gone")
    answering(monkeypatch, None)

    result = terminate(provider, "gpuc-e2e-aaa")

    assert (result.terminated, result.forgotten) == (False, True)
    assert any("does not exist at the provider" in note for note in result.notes)
    assert load_registry().hosts == {}


def test_an_exited_pod_is_ended_without_force(
    control_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """EXITED is `UNASKABLE` in `gpuc status` but a live rental at the provider.
    Nothing can be running in a container that is not running, so refusing here
    would hold the one command that stops the bill behind a flag for nothing."""
    fake = FakeProvider()
    fake.adopt(running_pod("gpuc-e2e-aaa", "pod1"), PodScript(status_after_polls={1: "EXITED"}))
    register()
    answering(monkeypatch, None)

    result = terminate(fake, "gpuc-e2e-aaa")

    assert fake.terminated == ["pod1"]
    assert (result.terminated, result.checked) == (True, False)
    assert any("EXITED" in note for note in result.notes)


def test_a_provider_that_will_not_answer_does_not_read_as_gone(
    control_env: Path, provider: FakeProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 5xx on the read must not forget a host whose pod is still billing."""
    register()

    def refuse(pod_id: str) -> None:
        raise ProviderError("HTTP 503 from runpod")

    monkeypatch.setattr(provider, "get", refuse)

    with pytest.raises(ProviderError):
        terminate(provider, "gpuc-e2e-aaa", force=True)
    assert "gpuc-e2e-aaa" in load_registry().hosts


def test_a_provider_read_that_fails_does_not_read_as_a_dead_pod(
    control_env: Path, provider: FakeProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The host says it is busy and the provider's first answer is a 503:
    that is a live pod nobody could describe, not a dead one, and without
    `--force` it is refused like any other busy host."""
    register()
    answering(
        monkeypatch,
        host_payload(
            jobs=[{"job_id": "j-running", "name": "train", "status": "running", "phase": "main"}]
        ),
    )
    real = provider.get
    reads: list[str] = []

    def flaky(pod_id: str) -> Any:
        reads.append(pod_id)
        if len(reads) == 1:
            raise ProviderError("HTTP 503 from runpod")
        return real(pod_id)

    monkeypatch.setattr(provider, "get", flaky)

    with pytest.raises(teardown.TerminateRefused) as error:
        terminate(provider, "gpuc-e2e-aaa")

    assert "train (j-running)" in str(error.value)
    assert provider.terminated == []
    assert "gpuc-e2e-aaa" in load_registry().hosts


def test_a_terminate_the_provider_will_not_confirm_keeps_the_host_visible(
    control_env: Path, provider: FakeProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pod that may still be billing must stay in `gpuc status`, and the
    error has to say so: this is the one failure that costs money."""
    register()

    def refuse(pod_id: str) -> None:
        raise ProviderError("HTTP 500 from runpod")

    monkeypatch.setattr(provider, "terminate", refuse)

    with pytest.raises(teardown.TerminateFailed) as error:
        terminate(provider, "gpuc-e2e-aaa", force=True)

    assert "still billing" in str(error.value)
    assert "gpuc-e2e-aaa" in load_registry().hosts


def test_a_terminate_that_fails_once_is_retried(
    control_env: Path, provider: FakeProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    register()
    real = provider.terminate
    attempts: list[str] = []

    def flaky(pod_id: str) -> None:
        attempts.append(pod_id)
        if len(attempts) == 1:
            raise ProviderError("rate limited")
        real(pod_id)

    monkeypatch.setattr(provider, "terminate", flaky)

    assert terminate(provider, "gpuc-e2e-aaa", force=True).terminated
    assert attempts == ["pod1", "pod1"]


def test_a_registry_entry_for_another_pod_is_not_this_ones_to_forget(
    control_env: Path, provider: FakeProvider
) -> None:
    """A host registered here under the pod's own name, pointing at a different
    pod. Ending podL must not drop it: the entry is not about podL."""
    register(name="gpuc-leak-bbb", pod_id="pod-somebody-elses")

    result = terminate(provider, "podL", force=True)

    assert (result.terminated, result.forgotten) == (True, False)
    assert "gpuc-leak-bbb" in load_registry().hosts


def test_forgotten_is_what_happened_not_what_was_asked(
    control_env: Path, provider: FakeProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A registry another session has locked leaves the entry in place. Saying
    `forgotten` anyway would have `gpuc status` contradict this command's own
    JSON, and a caller keying on it would never re-run the removal."""
    register()
    monkeypatch.setattr("gpuc.control.teardown.forget_host", lambda name, pod_id, report: False)

    result = terminate(provider, "gpuc-e2e-aaa", force=True)

    assert (result.terminated, result.forgotten) == (True, False)
    assert result.document()["forgotten"] is False


def test_an_unregistered_pod_the_provider_reports_stopped_is_ended_without_force(
    control_env: Path,
) -> None:
    """Nothing here can ask it, but the provider's own word that its container
    is not running is the same answer it is for a registered one: nothing
    can be running there, and the bill is the only thing left to stop."""
    fake = FakeProvider()
    stopped = running_pod("gpuc-leak-bbb", "podL").model_copy(update={"status": "EXITED"})
    fake.adopt(stopped, PodScript(ssh_after_polls=0))

    result = terminate(fake, "podL")

    assert fake.terminated == ["podL"]
    assert (result.terminated, result.checked) == (True, False)
