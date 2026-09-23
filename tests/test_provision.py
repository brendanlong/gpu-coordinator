"""Provisioning against a scripted provider: every path that can cost money.

The host behind each pod is a real one in a temporary home (`FakeHost`), so
the probe, the connect, the bootstrap and the host's own `status` all run
for real; only the provider and `nvidia-smi` are stood in for.
"""

from __future__ import annotations

import itertools
import json
import re
import urllib.error
from collections.abc import Callable
from pathlib import Path
from typing import ClassVar

import pytest

from gpuc.control.bootstrap import HealthOptions
from gpuc.control.config import (
    HostEntry,
    Settings,
    registry_transaction,
    utc_now,
)
from gpuc.control.providers.base import Constraints, Offer, Pod, ProviderError
from gpuc.control.provision import (
    ProvisionDeps,
    ProvisionError,
    offer_satisfies,
    pick_reusable_host,
    pod_name,
    provision,
    runpod_host,
)
from gpuc.host.jobs import HostConfig
from tests.conftest import host_entry, load_registry
from tests.fakehost import FakeHost
from tests.fakeprovider import (
    BROKEN_LOG,
    CAPACITY_ERROR,
    FakeProvider,
    PodScript,
    make_offer,
    running_pod,
)

CONSTRAINTS = Constraints(gpu_names=["A40"], max_price_usd_hr=0.60, cuda_min="12.8")
POD_GPUS = ("GPU-1111", "GPU-2222")


@pytest.fixture
def ssh_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "home"
    (home / ".ssh").mkdir(parents=True)
    key = home / ".ssh/id_ed25519.pub"
    key.write_text("ssh-ed25519 AAAAfake test@example\n")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    return key


@pytest.fixture
def host(fake_host: FakeHost) -> FakeHost:
    """The pod's host: every ssh in the flow lands on the temporary home."""
    fake_host.set_gpus(POD_GPUS)
    return fake_host


@pytest.fixture
def offline_health(tmp_path: Path) -> HealthOptions:
    """A health check that needs no network: the throughput test reads a local
    file, and the disk floor is one a tmpfs meets."""
    blob = tmp_path / "blob"
    blob.write_bytes(b"\0" * (1 << 20))
    return HealthOptions(download_url=blob.as_uri(), min_free_gb=0.0)


def deps(host: FakeHost, **overrides: object) -> ProvisionDeps:
    return ProvisionDeps(
        sleep=lambda _: None,
        transport_factory=lambda entry, settings: host,
        poll_interval_s=0.0,
        log_check_interval_s=0.0,
        **overrides,  # type: ignore[arg-type]
    )


def ticking(step: float = 100.0) -> Callable[[], float]:
    """A clock that jumps `step` seconds per reading, so ceilings arrive fast."""
    counter = itertools.count()
    return lambda: next(counter) * step


def run(
    provider: FakeProvider,
    host: FakeHost,
    health: HealthOptions,
    *,
    now: Callable[[], float] | None = None,
    reports: list[str] | None = None,
    **kwargs: object,
) -> HostEntry:
    return provision(
        CONSTRAINTS,
        Settings(),
        provider=provider,
        name_hint="e2e",
        idle_minutes=2.0,
        report=(reports.append if reports is not None else lambda _: None),
        health_options=health,
        deps=deps(host, **({"now": now} if now is not None else {})),
        **kwargs,  # type: ignore[arg-type]
    )


def test_happy_path_registers_a_bootstrapped_host(
    control_env: Path, ssh_key: Path, host: FakeHost, offline_health: HealthOptions
) -> None:
    provider = FakeProvider([make_offer()], scripts=[PodScript(ssh_after_polls=3)])
    host.refuse = 2
    reports: list[str] = []
    entry = run(provider, host, offline_health, reports=reports)

    assert entry.kind == "rental"
    assert entry.name.startswith("gpuc-e2e-")
    assert entry.ssh == "root@1.2.3.4"
    assert entry.port == 22000
    assert entry.config.gpus == list(POD_GPUS)
    assert entry.python and entry.bootstrapped_at
    assert entry.config.idle_minutes == 2.0
    # The bootstrap ran for real: the package is on the host and a dispatcher was started.
    assert (host.path(host.home) / "pkg/gpuc/host/dispatcher.py").exists()
    assert any("dispatcher pid" in line for line in reports)

    assert load_registry().hosts[entry.name].pod_id == entry.pod_id
    assert provider.terminated == []
    assert provider.registered_keys == [ssh_key.read_text()]
    assert all(line.startswith("[") and "+" in line for line in reports)
    assert any("ssh.direct" in line for line in reports)
    assert any("ceiling" in line and "whole attempt" in line for line in reports)


def test_a_fresh_host_runs_none_of_its_own_code_before_the_package_lands(
    control_env: Path, ssh_key: Path, host: FakeHost, offline_health: HealthOptions
) -> None:
    """The pod's interpreter has nothing installed, so the first `from
    gpuc.host import ...` before the rsync is a ModuleNotFoundError -- which
    provisioning read as "next offer" and bought a pod per offer to reproduce."""
    run(FakeProvider([make_offer()]), host, offline_health)
    first_host_code = next(
        i for i, command in enumerate(host.commands) if "from gpuc.host" in command
    )
    shipped = next(
        i
        for i, command in enumerate(host.commands)
        if 'mkdir -p "' in command and "/pkg" in command
    )
    assert shipped < first_host_code


def test_create_uses_the_spec_defaults(
    control_env: Path, ssh_key: Path, host: FakeHost, offline_health: HealthOptions
) -> None:
    provider = FakeProvider([make_offer()])
    run(provider, host, offline_health, image="runpod/pytorch:test", disk_gb=20)
    created = provider.created[0]
    assert created["image"] == "runpod/pytorch:test"
    assert created["disk_gb"] == 20
    assert created["cuda_min"] == "12.8"
    assert created["env"] is None  # the provider fills its own default
    assert created["name"].startswith("gpuc-")


def test_the_one_cuda_floor_reaches_the_catalog_query_and_the_create(
    control_env: Path, ssh_key: Path, host: FakeHost, offline_health: HealthOptions
) -> None:
    asked: list[str] = []

    class Recording(FakeProvider):
        def offers(self, constraints: Constraints) -> list[Offer]:
            asked.append(constraints.cuda_min)
            return super().offers(constraints)

    provider = Recording([make_offer()])
    provision(
        Constraints(gpu_names=["A40"]),
        Settings(),
        provider=provider,
        report=lambda _: None,
        health_options=offline_health,
        deps=deps(host),
    )
    assert asked == ["12.8"]
    assert provider.created[0]["cuda_min"] == "12.8"


def test_capacity_error_advances_to_the_next_offer(
    control_env: Path, ssh_key: Path, host: FakeHost, offline_health: HealthOptions
) -> None:
    provider = FakeProvider(
        [make_offer(price=0.20, gpu_id="cheap"), make_offer(price=0.40, gpu_id="dearer")],
        scripts=[PodScript(create_error=CAPACITY_ERROR), PodScript()],
    )
    entry = run(provider, host, offline_health)
    assert [c["offer"].gpu_id for c in provider.created] == ["cheap", "dearer"]
    assert provider.terminated == []  # nothing was ever created for the first offer
    assert load_registry().hosts[entry.name].pod_id == "pod1"


def test_broken_host_log_terminates_and_replaces(
    control_env: Path, ssh_key: Path, host: FakeHost, offline_health: HealthOptions
) -> None:
    provider = FakeProvider(
        [make_offer(price=0.20, gpu_id="first"), make_offer(price=0.40, gpu_id="second")],
        scripts=[PodScript(ssh_after_polls=-1, log_text=BROKEN_LOG), PodScript()],
    )
    reports: list[str] = []
    entry = run(provider, host, offline_health, reports=reports)
    assert provider.terminated == ["pod1"]
    assert entry.pod_id == "pod2"
    assert [provider.is_gone(p) for p in provider.list()] == [True, False]
    assert any("broken host" in line for line in reports)
    assert sorted(load_registry().hosts) == [entry.name]


def test_the_broken_host_signature_is_the_providers(
    control_env: Path, ssh_key: Path, host: FakeHost, offline_health: HealthOptions
) -> None:
    """Provisioning reads the pattern off the provider it was given, not a
    copy of its own: a provider with another vocabulary changes nothing here."""

    class Other(FakeProvider):
        broken_host = re.compile(r"kaboom")

    provider = Other(
        [make_offer(price=0.20, gpu_id="first"), make_offer(price=0.40, gpu_id="second")],
        scripts=[PodScript(ssh_after_polls=-1, log_text="system: kaboom\n"), PodScript()],
    )
    entry = run(provider, host, offline_health)
    assert provider.terminated == ["pod1"] and entry.pod_id == "pod2"

    plain = FakeProvider(
        [make_offer()], scripts=[PodScript(ssh_after_polls=-1, log_text="system: kaboom\n")]
    )
    with pytest.raises(ProvisionError, match="ceiling"):
        run(plain, host, offline_health, now=ticking())


def test_dead_pod_status_is_a_placement_failure(
    control_env: Path, ssh_key: Path, host: FakeHost, offline_health: HealthOptions
) -> None:
    provider = FakeProvider(
        [make_offer(price=0.20, gpu_id="first"), make_offer(price=0.40, gpu_id="second")],
        scripts=[PodScript(ssh_after_polls=-1, status_after_polls={1: "EXITED"}), PodScript()],
    )
    entry = run(provider, host, offline_health)
    assert provider.terminated == ["pod1"]
    assert entry.pod_id == "pod2"


def test_one_ceiling_bounds_the_whole_attempt(
    control_env: Path, ssh_key: Path, host: FakeHost, offline_health: HealthOptions
) -> None:
    """Two offers, neither pod ever gets an endpoint. The ceiling is the
    attempt's, so once the first pod has used it up the second is never
    bought: one create, one terminate, and the report says which offer was
    left untried."""
    provider = FakeProvider(
        [make_offer(price=0.20, gpu_id="first"), make_offer(price=0.40, gpu_id="second")],
        scripts=[PodScript(ssh_after_polls=-1), PodScript(ssh_after_polls=-1)],
    )
    with pytest.raises(ProvisionError) as error:
        run(provider, host, offline_health, now=ticking())
    assert "ceiling passed" in str(error.value)
    assert len(provider.created) == 1
    assert provider.terminated == ["pod1"]
    assert load_registry().hosts == {}


def test_an_offer_that_fails_late_leaves_the_next_untried_past_the_ceiling(
    control_env: Path, ssh_key: Path, host: FakeHost, offline_health: HealthOptions
) -> None:
    """The first pod dies for a reason of its own, but only once the ceiling
    has passed: the next offer is reported as untried, not bought."""
    clock = [0.0]

    class LateDeath(FakeProvider):
        def get(self, pod_id: str) -> Pod | None:
            pod = super().get(pod_id)
            if pod is not None and self.is_dead(pod):
                clock[0] += 3600.0
            return pod

    provider = LateDeath(
        [make_offer(price=0.20, gpu_id="first"), make_offer(price=0.40, gpu_id="second")],
        scripts=[PodScript(ssh_after_polls=-1, status_after_polls={2: "EXITED"}), PodScript()],
    )
    with pytest.raises(ProvisionError) as error:
        run(provider, host, offline_health, now=lambda: clock[0])
    assert "not tried, the ceiling had passed" in str(error.value)
    assert len(provider.created) == 1


def test_ssh_never_answers_terminates(
    control_env: Path, ssh_key: Path, host: FakeHost, offline_health: HealthOptions
) -> None:
    provider = FakeProvider([make_offer()])
    host.refuse = 99
    with pytest.raises(ProvisionError) as error:
        run(provider, host, offline_health, now=ticking())
    assert "waiting for ssh to" in str(error.value)
    assert provider.terminated == ["pod1"]
    assert load_registry().hosts == {}


def test_a_local_ssh_misconfiguration_ends_the_attempt_at_the_first_pod(
    control_env: Path, ssh_key: Path, host: FakeHost, offline_health: HealthOptions
) -> None:
    """No offer fixes a broken local ssh config: every pod would be bought,
    waited on and terminated identically. So the first one is terminated and
    the attempt stops there -- never a second create, and never the ceiling."""
    provider = FakeProvider(
        [make_offer(price=0.20, gpu_id="first"), make_offer(price=0.40, gpu_id="second")]
    )
    host.refuse = 99
    host.refusal = "/home/me/.ssh/config: line 3: Bad configuration option: prxycommand"
    with pytest.raises(ProvisionError) as error:
        run(provider, host, offline_health)
    message = str(error.value)
    assert "cannot work as configured" in message
    assert "no other offer could fix this" in message
    assert len(provider.created) == 1
    assert provider.terminated == ["pod1"]
    assert load_registry().hosts == {}


def test_a_tool_missing_on_this_machine_ends_the_attempt_at_the_first_pod(
    control_env: Path, ssh_key: Path, host: FakeHost, offline_health: HealthOptions
) -> None:
    """A missing local `rsync` is rc 127 from the transport, which read as
    "next offer" bought and terminated one pod per offer; a missing `ssh`
    read as "keep waiting" and billed the first pod to the ceiling. Neither
    is the offer's, so the first pod is terminated and the attempt stops."""
    from gpuc.control.transport import CommandResult, LocalToolMissing

    provider = FakeProvider(
        [make_offer(price=0.20, gpu_id="first"), make_offer(price=0.40, gpu_id="second")]
    )

    def no_rsync(*args: object, **kwargs: object) -> CommandResult:
        raise LocalToolMissing(
            CommandResult("fake", ["rsync"], 127, "", "rsync not found on this machine")
        )

    host.rsync = no_rsync  # type: ignore[method-assign]
    with pytest.raises(ProvisionError) as error:
        run(provider, host, offline_health, now=ticking())
    message = str(error.value)
    assert "no other offer could fix this" in message
    assert "rsync not found on this machine" in message
    assert len(provider.created) == 1
    assert provider.terminated == ["pod1"]
    assert load_registry().hosts == {}


def test_an_unreadable_identity_file_ends_the_attempt_too(
    control_env: Path, ssh_key: Path, host: FakeHost, offline_health: HealthOptions
) -> None:
    provider = FakeProvider([make_offer(price=0.20, gpu_id="first"), make_offer(gpu_id="second")])
    host.refuse = 99
    host.refusal = "Warning: Identity file /home/me/.ssh/id_ed25519 not accessible: No such file"
    with pytest.raises(ProvisionError) as error:
        run(provider, host, offline_health, now=ticking())
    assert "no other offer could fix this" in str(error.value)
    assert len(provider.created) == 1 and provider.terminated == ["pod1"]


def test_health_failure_terminates_and_forgets(
    control_env: Path, ssh_key: Path, host: FakeHost, offline_health: HealthOptions
) -> None:
    """A real health failure: the disk floor is one no host meets."""
    provider = FakeProvider([make_offer()])
    impossible = HealthOptions(download_url=offline_health.download_url, min_free_gb=1e12)
    with pytest.raises(ProvisionError) as error:
        run(provider, host, impossible)
    assert "host health failed" in str(error.value)
    assert provider.terminated == ["pod1"]
    assert load_registry().hosts == {}


def test_a_failed_terminate_keeps_the_pod_visible(
    control_env: Path, ssh_key: Path, host: FakeHost, offline_health: HealthOptions
) -> None:
    """A terminate is retried, and one that still fails is not retried by
    anything after this process moves on: the pod bills until a person ends
    it, so it must stay in the registry for `gpuc status` to show, the report
    must say where to look, and the final error must not claim it was ended."""
    provider = FakeProvider([make_offer()])
    attempts: list[str] = []

    def refuse(pod_id: str) -> None:
        attempts.append(pod_id)
        raise ProviderError("502 Bad Gateway")

    provider.terminate = refuse  # type: ignore[method-assign]
    reports: list[str] = []
    impossible = HealthOptions(download_url=offline_health.download_url, min_free_gb=1e12)
    with pytest.raises(ProvisionError) as error:
        run(provider, host, impossible, reports=reports)
    (name,) = load_registry().hosts  # still known, so `gpuc status` shows its pod
    assert name.startswith("gpuc-")
    assert attempts == ["pod1"] * 3
    assert any("still billing" in line and "gpuc pods" in line for line in reports)
    assert "pod1 could NOT be terminated and are still billing" in str(error.value)
    assert "All pods created here were terminated" not in str(error.value)


def test_a_terminate_that_fails_once_is_retried_and_confirmed(
    control_env: Path, ssh_key: Path, host: FakeHost, offline_health: HealthOptions
) -> None:
    provider = FakeProvider([make_offer()])
    real_terminate = provider.terminate
    attempts: list[str] = []

    def flaky(pod_id: str) -> None:
        attempts.append(pod_id)
        if len(attempts) == 1:
            raise ProviderError("502 Bad Gateway")
        real_terminate(pod_id)

    provider.terminate = flaky  # type: ignore[method-assign]
    impossible = HealthOptions(download_url=offline_health.download_url, min_free_gb=1e12)
    with pytest.raises(ProvisionError) as error:
        run(provider, host, impossible)
    assert attempts == ["pod1", "pod1"]
    assert provider.terminated == ["pod1"]
    assert load_registry().hosts == {}
    assert "All pods created here were terminated" in str(error.value)


def test_a_terminate_the_provider_will_not_confirm_is_a_failure(
    control_env: Path, ssh_key: Path, host: FakeHost, offline_health: HealthOptions
) -> None:
    """The POST is accepted and the pod never leaves RUNNING: that is a pod
    still billing, however politely the API answered."""

    class Stuck(FakeProvider):
        def terminate(self, pod_id: str) -> None:
            self.terminated.append(pod_id)

    provider = Stuck([make_offer()])
    provider.confirm_polls = 2
    impossible = HealthOptions(download_url=offline_health.download_url, min_free_gb=1e12)
    reports: list[str] = []
    with pytest.raises(ProvisionError) as error:
        run(provider, host, impossible, reports=reports)
    assert "still RUNNING" in "\n".join(reports)
    assert "could NOT be terminated" in str(error.value)


def test_a_read_that_fails_inside_the_confirm_loop_is_asked_again() -> None:
    """The POST went through; one 5xx on the poll after it must not report
    the pod as still billing when the next poll would have confirmed it."""
    provider = FakeProvider()
    provider.adopt(running_pod("gpuc-e2e-aaa", "pod1"))
    real = provider.get
    reads = [0]

    def flaky(pod_id: str) -> Pod | None:
        reads[0] += 1
        if reads[0] == 1:
            raise ProviderError("HTTP 502 from runpod")
        return real(pod_id)

    provider.get = flaky  # type: ignore[method-assign]
    reports: list[str] = []
    provider.terminate_confirmed("pod1", report=reports.append, sleep=lambda _: None)
    assert provider.terminated == ["pod1"]
    assert any("asking again" in line for line in reports)


@pytest.mark.parametrize(
    "error",
    [
        urllib.error.URLError("name resolution failed"),
        TimeoutError("timed out"),
        OSError(104, "reset"),
    ],
)
def test_a_network_failure_at_the_provider_is_a_provider_error(
    monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    """Everything on the terminate path catches `ProviderError` to say "still
    billing"; a `URLError` or a reset escaping as `OSError` said nothing."""
    from gpuc.control.providers.runpod import RunPodProvider

    def down(*_: object, **__: object) -> object:
        raise error

    monkeypatch.setattr("urllib.request.urlopen", down)
    with pytest.raises(ProviderError) as caught:
        RunPodProvider(api_key="k").get("pod1")
    assert "GET" in str(caught.value) and "/pods/pod1" in str(caught.value)


def test_a_failure_while_the_body_is_read_is_a_provider_error_too(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The connection opened and then reset mid-body: the same `OSError`, one
    call later, and the same callers would have let it through."""
    from gpuc.control.providers.runpod import RunPodProvider

    class Resetting:
        headers: ClassVar[dict[str, str]] = {}

        def __enter__(self) -> Resetting:
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def read(self) -> bytes:
            raise OSError(104, "reset")

    monkeypatch.setattr("urllib.request.urlopen", lambda *_, **__: Resetting())
    with pytest.raises(ProviderError) as caught:
        RunPodProvider(api_key="k").get("pod1")
    assert "reset" in str(caught.value)


def test_a_public_key_path_is_the_private_one_plus_pub(control_env: Path, tmp_path: Path) -> None:
    """`my.key` has a public half called `my.key.pub`, not `my.pub`."""
    from gpuc.control.provision import public_key_path

    key = tmp_path / "my.key"
    key.write_text("private\n")
    (tmp_path / "my.key.pub").write_text("ssh-ed25519 AAAA test\n")
    assert public_key_path(Settings(ssh_key=str(key))) == tmp_path / "my.key.pub"

    (tmp_path / "my.key.pub").unlink()
    with pytest.raises(ProvisionError, match="public half"):
        public_key_path(Settings(ssh_key=str(key)))


def test_no_offers_says_what_to_relax(
    control_env: Path, ssh_key: Path, host: FakeHost, offline_health: HealthOptions
) -> None:
    with pytest.raises(ProvisionError) as error:
        run(FakeProvider([]), host, offline_health)
    assert "--max-price" in str(error.value)


def test_every_offer_failing_lists_them(
    control_env: Path, ssh_key: Path, host: FakeHost, offline_health: HealthOptions
) -> None:
    provider = FakeProvider(
        [make_offer(price=0.2, gpu_id="a"), make_offer(price=0.3, gpu_id="b")],
        scripts=[PodScript(create_error=CAPACITY_ERROR)] * 2,
    )
    with pytest.raises(ProvisionError) as error:
        run(provider, host, offline_health)
    message = str(error.value)
    # One line per offer tried, cheapest first, each carrying its own reason.
    assert "- A40/secure $0.200/h" in message
    assert "- A40/secure $0.300/h" in message
    assert message.count(CAPACITY_ERROR) == 2
    assert "terminated" in message


def test_pod_name_is_prefixed_and_unique() -> None:
    names = {pod_name("gpuc-", "My Job!") for _ in range(5)}
    assert len(names) == 5
    assert all(name.startswith("gpuc-my-job-") for name in names)


def test_offer_satisfies_checks_every_constraint() -> None:
    offer = make_offer()
    assert offer_satisfies(offer, CONSTRAINTS)
    assert not offer_satisfies(offer, Constraints(gpu_names=["RTX4090"]))
    assert not offer_satisfies(offer, Constraints(min_vram_gb=80))
    assert not offer_satisfies(offer, Constraints(max_price_usd_hr=0.20))
    assert not offer_satisfies(offer, Constraints(clouds=["COMMUNITY"]))
    assert not offer_satisfies(offer, Constraints(cuda_min="13.0"))


# -- reuse: the host, not the cache, says what it is -----------------------------


def _register_reusable(
    pod: Pod, host: FakeHost, *, price: float = 0.49, gpus: list[str] | None = None
) -> HostEntry:
    """A pod registered here, whose own `config.json` on the host says what it
    was bought as and which cards it owns. The registry's cache deliberately
    disagrees (four cards), so a decision taken off the cache shows."""
    on_host = HostConfig(
        host=pod.name,
        gpus=gpus if gpus is not None else ["GPU-1111"],
        created_at=utc_now(),
        provider={
            "kind": "runpod",
            "pod_id": pod.id,
            "offer": make_offer(price=price).model_dump(mode="json"),
            "created_at": utc_now(),
        },
    )
    host.put_file(json.dumps(on_host.to_dict()), f"{host.home}/config.json", 0o644)
    host.ship_package()
    entry = host_entry(
        name=pod.name,
        kind="rental",
        ssh="root@1.2.3.4",
        port=22000,
        pod_id=pod.id,
        gpus=["GPU-1", "GPU-2", "GPU-3", "GPU-4"],
        created_at=utc_now(),
        provider={"kind": "runpod", "pod_id": pod.id},
    )
    with registry_transaction() as registry:
        registry.put(entry)
    return entry


def _beating(host: FakeHost) -> None:
    (host.path(host.home) / "dispatcher.heartbeat").touch()


def test_reuse_picks_a_live_matching_host_by_asking_it(
    control_env: Path, ssh_key: Path, host: FakeHost
) -> None:
    pod = running_pod("gpuc-e2e-aaa", "pod9")
    provider = FakeProvider([make_offer()])
    provider.adopt(pod)
    entry = _register_reusable(pod, host)
    _beating(host)

    reports: list[str] = []
    chosen = runpod_host(
        CONSTRAINTS, Settings(), provider=provider, report=reports.append, deps=deps(host)
    )
    assert chosen.name == entry.name
    assert provider.created == []
    assert any("-m gpuc.host status" in command for command in host.commands)
    assert any(line.startswith("reusing host gpuc-e2e-aaa") for line in reports)


def test_reuse_judges_the_card_count_by_the_hosts_answer_not_the_cache(
    control_env: Path, ssh_key: Path, host: FakeHost
) -> None:
    """The cache says four cards; the host says one. The host wins."""
    pod = running_pod("gpuc-e2e-aaa", "pod9")
    provider = FakeProvider([make_offer()])
    provider.adopt(pod)
    _register_reusable(pod, host)
    _beating(host)
    wanted = CONSTRAINTS.model_copy(update={"gpu_count": 2})
    reports: list[str] = []
    assert pick_reusable_host(wanted, Settings(), provider=provider, report=reports.append) is None
    assert any("owns 1 GPU(s)" in line for line in reports)


def test_reuse_skips_a_stale_dispatcher(control_env: Path, ssh_key: Path, host: FakeHost) -> None:
    pod = running_pod("gpuc-e2e-aaa", "pod9")
    provider = FakeProvider([make_offer()])
    provider.adopt(pod)
    _register_reusable(pod, host)
    reports: list[str] = []
    assert (
        pick_reusable_host(CONSTRAINTS, Settings(), provider=provider, report=reports.append)
        is None
    )
    assert any("dispatcher heartbeat is unknown" in line for line in reports)


def test_reuse_skips_a_host_that_cannot_be_asked(
    control_env: Path, ssh_key: Path, host: FakeHost
) -> None:
    pod = running_pod("gpuc-e2e-aaa", "pod9")
    provider = FakeProvider([make_offer()])
    provider.adopt(pod)
    _register_reusable(pod, host)
    host.refuse = 99
    reports: list[str] = []
    assert (
        pick_reusable_host(CONSTRAINTS, Settings(), provider=provider, report=reports.append)
        is None
    )
    assert any("could not be asked" in line for line in reports)


def test_reuse_skips_a_pod_that_is_too_expensive(
    control_env: Path, ssh_key: Path, host: FakeHost
) -> None:
    pod = running_pod("gpuc-e2e-aaa", "pod9")
    provider = FakeProvider([make_offer()])
    provider.adopt(pod)
    _register_reusable(pod, host, price=2.50)
    _beating(host)
    assert (
        pick_reusable_host(CONSTRAINTS, Settings(), provider=provider, report=lambda _: None)
        is None
    )


def test_reuse_skips_a_pod_that_is_not_running(
    control_env: Path, ssh_key: Path, host: FakeHost
) -> None:
    pod = running_pod("gpuc-e2e-aaa", "pod9").model_copy(update={"status": "EXITED"})
    provider = FakeProvider([make_offer()])
    provider.adopt(pod, PodScript(ssh_after_polls=0, status_after_polls={1: "EXITED"}))
    _register_reusable(pod, host)
    _beating(host)
    reports: list[str] = []
    assert (
        pick_reusable_host(CONSTRAINTS, Settings(), provider=provider, report=reports.append)
        is None
    )
    assert any("pod pod9 is EXITED" in line for line in reports)
    assert host.commands == []  # a dead pod is never dialled


def test_no_reuse_always_provisions(
    control_env: Path, ssh_key: Path, host: FakeHost, offline_health: HealthOptions
) -> None:
    pod = running_pod("gpuc-e2e-aaa", "pod9")
    provider = FakeProvider([make_offer()])
    provider.adopt(pod)
    _register_reusable(pod, host)
    _beating(host)
    host.wipe()  # the new pod's home is empty, as a fresh pod's is
    chosen = runpod_host(
        CONSTRAINTS,
        Settings(),
        provider=provider,
        reuse=False,
        report=lambda _: None,
        health_options=offline_health,
        deps=deps(host),
    )
    assert chosen.name != pod.name
    assert len(provider.created) == 1


def test_ctrl_c_during_bootstrap_terminates_the_pod(
    control_env: Path, ssh_key: Path, host: FakeHost, offline_health: HealthOptions
) -> None:
    """A KeyboardInterrupt is not a narrow provisioning error, and still owns a pod."""

    def interrupted(*_: object, **__: object) -> tuple[HostEntry, object]:
        raise KeyboardInterrupt

    provider = FakeProvider([make_offer()])
    with pytest.raises(KeyboardInterrupt):
        provision(
            CONSTRAINTS,
            Settings(),
            provider=provider,
            report=lambda _: None,
            deps=deps(host, bootstrap=interrupted),
        )
    assert provider.terminated == ["pod1"]
    assert provider.live_names() == []
    assert load_registry().hosts == {}


def test_a_transient_provider_error_while_polling_does_not_burn_the_pod(
    control_env: Path, ssh_key: Path, host: FakeHost, offline_health: HealthOptions
) -> None:
    class Flaky(FakeProvider):
        gets = 0

        def get(self, pod_id: str) -> Pod | None:
            self.gets += 1
            if self.gets == 1:
                raise ProviderError("GET /pods/pod1 -> HTTP 502: bad gateway")
            return super().get(pod_id)

    provider = Flaky([make_offer()], scripts=[PodScript(ssh_after_polls=2)])
    reports: list[str] = []
    entry = run(provider, host, offline_health, reports=reports)
    assert entry.pod_id == "pod1"
    assert provider.terminated == []
    assert any("provider read failed, retrying" in line for line in reports)


def test_a_provider_that_never_answers_still_stops_at_the_ceiling(
    control_env: Path, ssh_key: Path, host: FakeHost, offline_health: HealthOptions
) -> None:
    class Down(FakeProvider):
        def get(self, pod_id: str) -> Pod | None:
            raise ProviderError("GET /pods -> HTTP 500")

    provider = Down([make_offer()])
    with pytest.raises(ProvisionError) as error:
        run(provider, host, offline_health, now=ticking())
    assert "ceiling" in str(error.value)
    assert provider.terminated == ["pod1"]


def test_reuse_skips_a_draining_host(control_env: Path, ssh_key: Path, host: FakeHost) -> None:
    """A draining pod is terminating itself; a job enqueued there dies with it."""
    pod = running_pod("gpuc-e2e-aaa", "pod9")
    provider = FakeProvider([make_offer()])
    provider.adopt(pod)
    _register_reusable(pod, host)
    _beating(host)
    (host.path(host.home) / "draining").touch()
    reports: list[str] = []
    assert (
        pick_reusable_host(CONSTRAINTS, Settings(), provider=provider, report=reports.append)
        is None
    )
    assert any("draining" in line for line in reports)


def test_reuse_forgets_a_host_whose_pod_is_gone(
    control_env: Path, ssh_key: Path, host: FakeHost
) -> None:
    """A registry entry for a dead pod must be cleaned, not left to break submit."""
    pod = running_pod("gpuc-e2e-aaa", "podGONE")
    provider = FakeProvider([make_offer()])  # never adopted: the provider has no such pod
    _register_reusable(pod, host)
    reports: list[str] = []

    assert (
        pick_reusable_host(CONSTRAINTS, Settings(), provider=provider, report=reports.append)
        is None
    )
    assert host.commands == []  # no ssh to an address the pod no longer owns
    assert load_registry().hosts == {}
    assert any("forgetting" in line for line in reports)


def test_reuse_falls_through_to_a_fresh_pod_when_the_old_one_is_gone(
    control_env: Path, ssh_key: Path, host: FakeHost, offline_health: HealthOptions
) -> None:
    pod = running_pod("gpuc-e2e-aaa", "podGONE")
    provider = FakeProvider([make_offer()])
    _register_reusable(pod, host)
    host.wipe()
    entry = runpod_host(
        CONSTRAINTS,
        Settings(),
        provider=provider,
        report=lambda _: None,
        health_options=offline_health,
        deps=deps(host),
    )
    assert entry.pod_id == "pod1"
    assert sorted(load_registry().hosts) == [entry.name]


def test_the_pod_is_given_its_own_record_of_what_it_was_bought_as(
    control_env: Path, ssh_key: Path, host: FakeHost, offline_health: HealthOptions
) -> None:
    """Nothing about the pod lives only on this machine: a second machine
    reads what it was rented as off the pod itself (`rented`), and so does the
    next `submit` here when it decides whether to reuse it.
    """
    provider = FakeProvider([make_offer()])
    entry = run(provider, host, offline_health)

    document = host.config
    assert document is not None
    provider_block = document["provider"]
    assert provider_block["kind"] == "runpod"
    assert provider_block["pod_id"] == entry.pod_id
    assert provider_block["offer"]["name"] == "A40"
    assert provider_block["created_at"] == entry.config.created_at
    assert entry.config.provider == provider_block
