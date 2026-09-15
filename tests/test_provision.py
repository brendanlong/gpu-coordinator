"""Provisioning against a scripted provider: every path that can cost money."""

from __future__ import annotations

import itertools
from collections.abc import Callable
from pathlib import Path

import pytest

from gpuc.control.config import (
    DesiredHost,
    HostEntry,
    Settings,
    desired_dir,
    load_registry,
    read_desired,
    registry_transaction,
    utc_now,
    write_desired,
)
from gpuc.control.providers.base import Caps, Constraints, Pod
from gpuc.control.provision import (
    ProvisionDeps,
    ProvisionError,
    deliver_s3_credentials,
    offer_satisfies,
    pick_reusable_host,
    pod_name,
    provision,
    runpod_host,
)
from tests.fakeprovider import (
    BROKEN_LOG,
    CAPACITY_ERROR,
    FakeProvider,
    FakeTransport,
    PodScript,
    fake_bootstrap,
    make_offer,
    running_pod,
)

CONSTRAINTS = Constraints(gpu_names=["A40"], max_price_usd_hr=0.60, cuda_min="12.8")


@pytest.fixture
def ssh_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "home"
    (home / ".ssh").mkdir(parents=True)
    key = home / ".ssh/id_ed25519.pub"
    key.write_text("ssh-ed25519 AAAAfake test@example\n")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    return key


def deps(transport: FakeTransport | None = None, **overrides: object) -> ProvisionDeps:
    transport = transport or FakeTransport()
    return ProvisionDeps(
        sleep=lambda _: None,
        bootstrap=fake_bootstrap,
        transport_factory=lambda entry, settings: transport,
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
    transport: FakeTransport | None = None,
    *,
    settings: Settings | None = None,
    reports: list[str] | None = None,
    **kwargs: object,
) -> HostEntry:
    return provision(
        CONSTRAINTS,
        settings or Settings(),
        provider=provider,
        name_hint="e2e",
        idle_minutes=2.0,
        ttl_hours=1.0,
        report=(reports.append if reports is not None else lambda _: None),
        deps=deps(transport),
        **kwargs,  # type: ignore[arg-type]
    )


def test_happy_path_registers_a_bootstrapped_host(control_env: Path, ssh_key: Path) -> None:
    provider = FakeProvider([make_offer()], scripts=[PodScript(ssh_after_polls=3)])
    transport = FakeTransport(ssh_failures=2)
    reports: list[str] = []
    entry = run(provider, transport, reports=reports)

    assert entry.kind == "runpod"
    assert entry.name.startswith("gpuc-e2e-")
    assert entry.ssh == "root@1.2.3.4"
    assert entry.port == 22000
    assert entry.gpus == ["GPU-1111", "GPU-2222"]
    assert entry.python and entry.bootstrapped_at
    assert entry.idle_minutes == 2.0 and entry.ttl_hours == 1.0

    assert load_registry().hosts[entry.name].pod_id == entry.pod_id
    desired = read_desired(entry.name)
    assert desired is not None and desired.bootstrapped_at is not None
    assert provider.terminated == []
    assert provider.registered_keys == [ssh_key.read_text()]
    assert all(line.startswith("[") and "+" in line for line in reports)
    assert any("ssh.direct" in line for line in reports)


def test_create_uses_the_spec_defaults(control_env: Path, ssh_key: Path) -> None:
    provider = FakeProvider([make_offer()])
    run(provider, image="runpod/pytorch:test", disk_gb=20)
    created = provider.created[0]
    assert created["image"] == "runpod/pytorch:test"
    assert created["disk_gb"] == 20
    assert created["cuda_min"] == "12.8"
    assert created["env"] == {"HF_HUB_ENABLE_HF_TRANSFER": "0"}
    assert created["name"].startswith("gpuc-")


def test_cuda_min_defaults_to_12_8_when_unconstrained(control_env: Path, ssh_key: Path) -> None:
    provider = FakeProvider([make_offer()])
    provision(
        Constraints(gpu_names=["A40"]),
        Settings(),
        provider=provider,
        report=lambda _: None,
        deps=deps(),
    )
    assert provider.created[0]["cuda_min"] == "12.8"


def test_desired_state_is_written_before_the_pod_is_ready(control_env: Path, ssh_key: Path) -> None:
    """The record has to exist even if we die mid-wait, or the pod is a leak."""
    provider = FakeProvider([make_offer()], scripts=[PodScript(ssh_after_polls=-1)])
    seen: list[list[str]] = []

    def watch(_: float) -> None:
        seen.append(sorted(p.name for p in desired_dir().glob("*.json")))

    with pytest.raises(ProvisionError):
        provision(
            CONSTRAINTS,
            Settings(),
            provider=provider,
            report=lambda _: None,
            deps=ProvisionDeps(
                sleep=watch,
                now=ticking(),
                bootstrap=fake_bootstrap,
                transport_factory=lambda entry, settings: FakeTransport(),
                poll_interval_s=0.0,
                log_check_interval_s=0.0,
            ),
        )
    assert seen and seen[0] and seen[0][0].startswith("gpuc-")
    assert list(desired_dir().glob("*.json")) == []


def test_capacity_error_advances_to_the_next_offer(control_env: Path, ssh_key: Path) -> None:
    provider = FakeProvider(
        [make_offer("cheap", 0.20), make_offer("dearer", 0.40)],
        scripts=[PodScript(create_error=CAPACITY_ERROR), PodScript()],
    )
    entry = run(provider)
    assert [c["offer"].gpu_id for c in provider.created] == ["cheap", "dearer"]
    assert provider.terminated == []  # nothing was ever created for the first offer
    assert load_registry().hosts[entry.name].pod_id == "pod1"


def test_broken_host_log_terminates_and_replaces(control_env: Path, ssh_key: Path) -> None:
    provider = FakeProvider(
        [make_offer("first", 0.20), make_offer("second", 0.40)],
        scripts=[PodScript(ssh_after_polls=-1, log_text=BROKEN_LOG), PodScript()],
    )
    reports: list[str] = []
    entry = run(provider, reports=reports)
    assert provider.terminated == ["pod1"]
    assert entry.pod_id == "pod2"
    assert [p.status for p in provider.list()] == ["TERMINATED", "RUNNING"]
    assert any("broken host" in line for line in reports)
    assert sorted(p.name for p in desired_dir().glob("*.json")) == [f"{entry.name}.json"]


def test_dead_pod_status_is_a_placement_failure(control_env: Path, ssh_key: Path) -> None:
    provider = FakeProvider(
        [make_offer("first", 0.20), make_offer("second", 0.40)],
        scripts=[PodScript(ssh_after_polls=-1, status_after_polls={1: "EXITED"}), PodScript()],
    )
    entry = run(provider)
    assert provider.terminated == ["pod1"]
    assert entry.pod_id == "pod2"


def test_ceiling_terminates_and_reports_every_failure(control_env: Path, ssh_key: Path) -> None:
    provider = FakeProvider([make_offer()], scripts=[PodScript(ssh_after_polls=-1)])
    with pytest.raises(ProvisionError) as error:
        provision(
            CONSTRAINTS,
            Settings(),
            provider=provider,
            report=lambda _: None,
            deps=ProvisionDeps(
                sleep=lambda _: None,
                now=ticking(),
                bootstrap=fake_bootstrap,
                transport_factory=lambda entry, settings: FakeTransport(),
                poll_interval_s=0.0,
                log_check_interval_s=0.0,
            ),
        )
    assert "no direct SSH endpoint" in str(error.value)
    assert provider.terminated == ["pod1"]
    assert load_registry().hosts == {}
    assert list(desired_dir().glob("*.json")) == []


def test_ssh_never_answers_terminates(control_env: Path, ssh_key: Path) -> None:
    provider = FakeProvider([make_offer()])
    transport = FakeTransport(ssh_failures=99)
    with pytest.raises(ProvisionError) as error:
        provision(
            CONSTRAINTS,
            Settings(),
            provider=provider,
            report=lambda _: None,
            deps=ProvisionDeps(
                sleep=lambda _: None,
                now=ticking(),
                bootstrap=fake_bootstrap,
                transport_factory=lambda entry, settings: transport,
                poll_interval_s=0.0,
                log_check_interval_s=0.0,
            ),
        )
    assert "ssh to" in str(error.value)
    assert provider.terminated == ["pod1"]
    assert load_registry().hosts == {}


def test_health_failure_terminates_and_forgets(control_env: Path, ssh_key: Path) -> None:
    from gpuc.control.bootstrap import BootstrapError

    def failing_bootstrap(entry: HostEntry, *args: object, **kwargs: object):
        raise BootstrapError(f"host health failed on {entry.name}: driver missing")

    provider = FakeProvider([make_offer()])
    with pytest.raises(ProvisionError) as error:
        provision(
            CONSTRAINTS,
            Settings(),
            provider=provider,
            report=lambda _: None,
            deps=ProvisionDeps(
                sleep=lambda _: None,
                bootstrap=failing_bootstrap,  # type: ignore[arg-type]
                transport_factory=lambda entry, settings: FakeTransport(),
                poll_interval_s=0.0,
                log_check_interval_s=0.0,
            ),
        )
    assert "host health failed" in str(error.value)
    assert provider.terminated == ["pod1"]
    assert load_registry().hosts == {}
    assert list(desired_dir().glob("*.json")) == []


def test_caps_refusal_never_creates(control_env: Path, ssh_key: Path) -> None:
    provider = FakeProvider(
        [make_offer()],
        caps=Caps(max_pods=1),
        existing=[running_pod("gpuc-other-abc", "podX")],
    )
    with pytest.raises(ProvisionError) as error:
        run(provider)
    assert "max_pods=1" in str(error.value)
    assert provider.created == []


def test_caps_count_ignores_foreign_pods(control_env: Path, ssh_key: Path) -> None:
    provider = FakeProvider(
        [make_offer()],
        caps=Caps(max_pods=1),
        existing=[running_pod("subrep-someone-else", "podY")],
    )
    entry = run(provider)
    assert entry.pod_id == "pod1"


def test_no_offers_says_what_to_relax(control_env: Path, ssh_key: Path) -> None:
    with pytest.raises(ProvisionError) as error:
        run(FakeProvider([]))
    assert "--max-price" in str(error.value)


def test_every_offer_failing_lists_them(control_env: Path, ssh_key: Path) -> None:
    provider = FakeProvider(
        [make_offer("a", 0.2), make_offer("b", 0.3)],
        scripts=[PodScript(create_error=CAPACITY_ERROR)] * 2,
    )
    with pytest.raises(ProvisionError) as error:
        run(provider)
    assert "- a/secure" in str(error.value) and "- b/secure" in str(error.value)
    assert "terminated" in str(error.value)


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


def _register_reusable(pod: Pod, price: float = 0.49) -> HostEntry:
    entry = HostEntry(
        name=pod.name,
        kind="runpod",
        ssh="root@1.2.3.4",
        port=22000,
        pod_id=pod.id,
        gpus=["GPU-1111"],
        python="/root/python",
        created_at=utc_now(),
    )
    with registry_transaction() as registry:
        registry.put(entry)
    write_desired(
        DesiredHost(
            name=entry.name,
            pod_id=pod.id,
            offer=make_offer(price=price),
            created_at=utc_now(),
            ceiling_at=utc_now(),
            bootstrapped_at=utc_now(),
        )
    )
    return entry


def test_reuse_picks_a_live_matching_host(
    control_env: Path, ssh_key: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pod = running_pod("gpuc-e2e-aaa", "pod9")
    provider = FakeProvider([make_offer()])
    provider.adopt(pod)
    entry = _register_reusable(pod)
    monkeypatch.setattr("gpuc.control.provision.dispatcher_heartbeat_age", lambda *a: 3.0)

    chosen = runpod_host(
        CONSTRAINTS, Settings(), provider=provider, report=lambda _: None, deps=deps()
    )
    assert chosen.name == entry.name
    assert provider.created == []


def test_reuse_skips_a_stale_dispatcher(
    control_env: Path, ssh_key: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pod = running_pod("gpuc-e2e-aaa", "pod9")
    provider = FakeProvider([make_offer()])
    provider.adopt(pod)
    _register_reusable(pod)
    monkeypatch.setattr("gpuc.control.provision.dispatcher_heartbeat_age", lambda *a: None)
    reports: list[str] = []
    assert (
        pick_reusable_host(CONSTRAINTS, Settings(), provider=provider, report=reports.append)
        is None
    )
    assert any("heartbeat is unreachable" in line for line in reports)


def test_reuse_skips_a_pod_that_is_too_expensive(
    control_env: Path, ssh_key: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pod = running_pod("gpuc-e2e-aaa", "pod9")
    provider = FakeProvider([make_offer()])
    provider.adopt(pod)
    _register_reusable(pod, price=2.50)
    monkeypatch.setattr("gpuc.control.provision.dispatcher_heartbeat_age", lambda *a: 1.0)
    assert (
        pick_reusable_host(CONSTRAINTS, Settings(), provider=provider, report=lambda _: None)
        is None
    )


def test_reuse_skips_a_pod_that_is_not_running(
    control_env: Path, ssh_key: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pod = running_pod("gpuc-e2e-aaa", "pod9").model_copy(update={"status": "EXITED"})
    provider = FakeProvider([make_offer()])
    provider.adopt(pod, PodScript(ssh_after_polls=0, status_after_polls={1: "EXITED"}))
    _register_reusable(pod)
    monkeypatch.setattr("gpuc.control.provision.dispatcher_heartbeat_age", lambda *a: 1.0)
    assert (
        pick_reusable_host(CONSTRAINTS, Settings(), provider=provider, report=lambda _: None)
        is None
    )


def test_no_reuse_always_provisions(
    control_env: Path, ssh_key: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pod = running_pod("gpuc-e2e-aaa", "pod9")
    provider = FakeProvider([make_offer()])
    provider.adopt(pod)
    _register_reusable(pod)
    monkeypatch.setattr("gpuc.control.provision.dispatcher_heartbeat_age", lambda *a: 1.0)
    chosen = runpod_host(
        CONSTRAINTS,
        Settings(),
        provider=provider,
        reuse=False,
        report=lambda _: None,
        deps=deps(),
    )
    assert chosen.name != pod.name
    assert len(provider.created) == 1


def test_s3_credentials_are_delivered_0600_when_configured(control_env: Path) -> None:
    transport = FakeTransport()
    entry = HostEntry(name="gpuc-x", kind="runpod", s3_prefix="s3://bucket/gpuc/gpuc-x")
    progress: list[str] = []

    class _P:
        def __call__(self, message: str) -> None:
            progress.append(message)

    assert deliver_s3_credentials(
        transport,
        entry,
        _P(),  # type: ignore[arg-type]
        {"AWS_ACCESS_KEY_ID": "AKIA", "AWS_SECRET_ACCESS_KEY": "shhh", "AWS_REGION": "us-east-1"},
    )
    body = transport.files["/root/.aws/credentials"]
    assert "aws_access_key_id = AKIA" in body and "region = us-east-1" in body
    assert not any("shhh" in line for line in progress)


def test_s3_credentials_are_skipped_without_a_prefix(control_env: Path) -> None:
    transport = FakeTransport()
    entry = HostEntry(name="gpuc-x", kind="runpod")
    assert not deliver_s3_credentials(transport, entry, lambda m: None, {})  # type: ignore[arg-type]
    assert transport.files == {}
