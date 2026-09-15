"""The reaper. Every branch, including the ones whose bug costs money."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from gpuc.control.config import (
    DesiredHost,
    HostEntry,
    Settings,
    desired_dir,
    desired_file,
    load_registry,
    registry_transaction,
    write_desired,
)
from gpuc.control.providers.base import Caps, ProviderError
from gpuc.control.reconcile import (
    SERVICE_NAME,
    TIMER_NAME,
    install,
    reconcile_once,
    run_loop,
    unit_files,
)
from gpuc.control.s3index import IndexEntry, LocalIndex
from tests.fakeprovider import FakeProvider, PodScript, make_offer, running_pod

FOREIGN = "subrep-someone-else"


def stamp(**delta: float) -> str:
    return (datetime.now(UTC) + timedelta(**delta)).isoformat(timespec="seconds")


def desire(
    name: str,
    pod_id: str,
    *,
    ttl_hours: float = 24.0,
    created_hours_ago: float = 0.5,
    ceiling_minutes: float = 15.0,
    bootstrapped: bool = True,
) -> DesiredHost:
    host = DesiredHost(
        name=name,
        pod_id=pod_id,
        offer=make_offer(),
        created_at=stamp(hours=-created_hours_ago),
        ceiling_at=stamp(hours=-created_hours_ago, minutes=ceiling_minutes),
        ttl_hours=ttl_hours,
        bootstrapped_at=stamp(hours=-created_hours_ago) if bootstrapped else None,
    )
    write_desired(host)
    with registry_transaction() as registry:
        registry.put(HostEntry(name=name, kind="runpod", pod_id=pod_id, ssh="root@1.2.3.4"))
    return host


def provider_with(*pods: object, **kwargs: object) -> FakeProvider:
    provider = FakeProvider(**kwargs)  # type: ignore[arg-type]
    for pod in pods:
        provider.adopt(pod, PodScript(ssh_after_polls=0))  # type: ignore[arg-type]
    return provider


def test_healthy_host_is_left_alone(control_env: Path) -> None:
    pod = running_pod("gpuc-a-111", "pod1")
    provider = provider_with(pod)
    desire("gpuc-a-111", "pod1")
    result = reconcile_once(Settings(), provider, lambda _: None)
    assert result.kept == ["gpuc-a-111"]
    assert provider.terminated == []
    assert desired_file("gpuc-a-111").exists()


def test_terminated_pod_is_forgotten_with_its_jobs(control_env: Path) -> None:
    pod = running_pod("gpuc-a-111", "pod1").model_copy(update={"status": "TERMINATED"})
    provider = provider_with(pod)
    desire("gpuc-a-111", "pod1")
    LocalIndex().record(IndexEntry(job_id="20260915-1", host="gpuc-a-111", name="train"))
    reports: list[str] = []

    result = reconcile_once(Settings(), provider, reports.append)

    assert result.forgotten == ["gpuc-a-111"]
    assert provider.terminated == []  # already gone; terminating again would 404
    assert not desired_file("gpuc-a-111").exists()
    assert load_registry().hosts == {}
    assert any("20260915-1 (train)" in line for line in reports)
    assert any("gpuc requeue" in line for line in reports)


def test_missing_pod_says_host_gone(control_env: Path) -> None:
    provider = FakeProvider()
    desire("gpuc-a-111", "podGONE")
    reports: list[str] = []
    result = reconcile_once(Settings(), provider, reports.append)
    assert result.forgotten == ["gpuc-a-111"]
    assert any("no jobs are recorded" in line for line in reports)


def test_host_past_its_ttl_is_terminated(control_env: Path) -> None:
    pod = running_pod("gpuc-a-111", "pod1", age_minutes=130)
    provider = provider_with(pod)
    desire("gpuc-a-111", "pod1", ttl_hours=1.0, created_hours_ago=2.2)
    reports: list[str] = []

    result = reconcile_once(Settings(), provider, reports.append)

    assert provider.terminated == ["pod1"]
    assert result.terminated == ["gpuc-a-111"] and result.forgotten == ["gpuc-a-111"]
    assert not desired_file("gpuc-a-111").exists()
    assert any("past its 1 h TTL" in line for line in reports)


def test_host_past_its_ceiling_without_bootstrap_is_terminated(control_env: Path) -> None:
    pod = running_pod("gpuc-a-111", "pod1", age_minutes=30)
    provider = provider_with(pod)
    desire("gpuc-a-111", "pod1", created_hours_ago=0.5, bootstrapped=False)

    result = reconcile_once(Settings(), provider, lambda _: None)

    assert provider.terminated == ["pod1"]
    assert result.forgotten == ["gpuc-a-111"]


def test_bootstrapped_host_past_its_ceiling_is_kept(control_env: Path) -> None:
    pod = running_pod("gpuc-a-111", "pod1", age_minutes=30)
    provider = provider_with(pod)
    desire("gpuc-a-111", "pod1", created_hours_ago=0.5, bootstrapped=True)
    assert reconcile_once(Settings(), provider, lambda _: None).kept == ["gpuc-a-111"]
    assert provider.terminated == []


def test_stray_pod_with_our_prefix_is_terminated(control_env: Path) -> None:
    provider = provider_with(running_pod("gpuc-leaked-999", "podX", age_minutes=120))
    desired_dir().mkdir(parents=True, exist_ok=True)
    reports: list[str] = []

    result = reconcile_once(Settings(), provider, reports.append)

    assert provider.terminated == ["podX"]
    assert result.terminated == ["gpuc-leaked-999"]
    assert any("no desired/ record" in line for line in reports)


def test_a_young_stray_is_left_for_the_session_that_may_be_creating_it(
    control_env: Path,
) -> None:
    provider = provider_with(running_pod("gpuc-new-999", "podX", age_minutes=2))
    desired_dir().mkdir(parents=True, exist_ok=True)
    result = reconcile_once(Settings(), provider, lambda _: None)
    assert provider.terminated == []
    assert result.kept == ["gpuc-new-999"]


def test_foreign_pods_are_never_touched(control_env: Path) -> None:
    provider = FakeProvider(existing=[running_pod(FOREIGN, "podF", age_minutes=600)])
    desired_dir().mkdir(parents=True, exist_ok=True)
    reports: list[str] = []
    result = reconcile_once(Settings(), provider, reports.append)
    assert provider.terminated == []
    assert result.terminated == []
    assert not any(FOREIGN in line for line in reports)


def test_missing_desired_directory_does_nothing(control_env: Path) -> None:
    provider = provider_with(running_pod("gpuc-leaked-999", "podX", age_minutes=600))
    reports: list[str] = []

    result = reconcile_once(Settings(), provider, reports.append)

    assert provider.terminated == []
    assert result.errors and "does not exist" in result.errors[0]
    assert any("not the same as `no pods`" in line for line in reports)


def test_unreadable_desired_entry_does_nothing(control_env: Path) -> None:
    provider = provider_with(running_pod("gpuc-leaked-999", "podX", age_minutes=600))
    desire("gpuc-a-111", "pod1")
    desired_file("gpuc-a-111").write_text("{not json")

    result = reconcile_once(Settings(), provider, lambda _: None)

    assert provider.terminated == []
    assert result.errors and "Nothing was terminated" in result.errors[0]


def test_provider_failure_is_reported_and_nothing_is_touched(control_env: Path) -> None:
    class Broken(FakeProvider):
        def list(self) -> list[object]:  # type: ignore[override]
            raise ProviderError("HTTP 500")

    provider = Broken(caps=Caps())
    desired_dir().mkdir(parents=True, exist_ok=True)
    result = reconcile_once(Settings(), provider, lambda _: None)
    assert result.errors == ["HTTP 500"]
    assert provider.terminated == []


def test_terminate_failure_is_loud_and_keeps_the_record(control_env: Path) -> None:
    class Stubborn(FakeProvider):
        def terminate(self, pod_id: str) -> None:
            raise ProviderError("HTTP 403 forbidden")

    provider = Stubborn()
    provider.adopt(running_pod("gpuc-a-111", "pod1", age_minutes=600))
    desire("gpuc-a-111", "pod1", ttl_hours=1.0, created_hours_ago=10)
    reports: list[str] = []

    result = reconcile_once(Settings(), provider, reports.append)

    assert result.terminated == []
    assert any("still billing" in line for line in reports)
    # A pod we could not terminate is still ours: keep the record so the next
    # pass retries it, and make `gpuc reconcile --once` exit non-zero.
    assert result.errors and "pod1" in result.errors[0]
    assert result.forgotten == []
    assert desired_file("gpuc-a-111").exists()
    assert "gpuc-a-111" in load_registry().hosts


def test_run_loop_stops_after_the_requested_iterations(control_env: Path) -> None:
    provider = provider_with(running_pod("gpuc-a-111", "pod1"))
    desire("gpuc-a-111", "pod1")
    slept: list[float] = []
    result = run_loop(
        Settings(),
        provider,
        interval_s=60.0,
        report=lambda _: None,
        iterations=2,
        sleep=slept.append,
    )
    assert slept == [60.0]
    assert result.kept == ["gpuc-a-111"]


def test_unit_files_are_absolute_and_not_enabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    reports: list[str] = []

    written = install(30.0, reports.append)

    assert [p.name for p in written] == [SERVICE_NAME, TIMER_NAME]
    service = (home / ".config/systemd/user" / SERVICE_NAME).read_text()
    exec_start = next(
        line for line in service.splitlines() if line.startswith("ExecStart=")
    ).removeprefix("ExecStart=")
    assert exec_start.startswith("/") and exec_start.endswith("gpuc reconcile --once")
    assert "OnUnitActiveSec=30s" in unit_files(30.0)[TIMER_NAME]
    assert "EnvironmentFile=-" in service and "/env" in service
    assert any("RUNPOD_API_KEY" in line for line in reports)
    assert any("systemctl --user enable --now" in line for line in reports)
    # Nothing else in the user's config may be touched.
    assert sorted(p.name for p in (home / ".config/systemd/user").iterdir()) == [
        SERVICE_NAME,
        TIMER_NAME,
    ]


def test_a_stray_with_no_creation_time_is_left_alone(control_env: Path) -> None:
    """Unknown age cannot be proved past the ceiling, so it cannot be proved a stray."""
    stray = running_pod("gpuc-other-999", "podX").model_copy(update={"created_at": None})
    provider = provider_with(stray)
    desired_dir().mkdir(parents=True, exist_ok=True)
    reports: list[str] = []

    result = reconcile_once(Settings(), provider, reports.append)

    assert provider.terminated == []
    assert result.kept == ["gpuc-other-999"]
    assert any("no creation time" in line for line in reports)


def test_a_failed_stray_terminate_is_an_error(control_env: Path) -> None:
    class Stubborn(FakeProvider):
        def terminate(self, pod_id: str) -> None:
            raise ProviderError("HTTP 500")

    provider = Stubborn()
    provider.adopt(running_pod("gpuc-other-999", "podX", age_minutes=600))
    desired_dir().mkdir(parents=True, exist_ok=True)

    result = reconcile_once(Settings(), provider, lambda _: None)

    assert result.terminated == [] and result.errors


def test_a_desired_record_naming_a_foreign_pod_is_refused_not_terminated(
    control_env: Path,
) -> None:
    """The never-touch-others rule rests on code here, so it is checked at the call."""
    foreign = running_pod(FOREIGN, "podF", age_minutes=600)
    provider = provider_with(foreign)
    desire("gpuc-a-111", "podF", ttl_hours=1.0, created_hours_ago=10)
    reports: list[str] = []

    result = reconcile_once(Settings(), provider, reports.append)

    assert provider.terminated == []
    assert result.errors and "not ours" in result.errors[0]
    assert any("refusing to terminate" in line for line in reports)
    assert desired_file("gpuc-a-111").exists()
