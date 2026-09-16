"""The reaper. Every branch, including the ones whose bug costs money."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from gpuc.control.config import (
    ConfigError,
    DesiredHost,
    HostEntry,
    Settings,
    desired_dir,
    desired_file,
    load_registry,
    read_desired,
    registry_transaction,
    state_lock,
    write_desired,
)
from gpuc.control.providers.base import Caps, Pod, ProviderError
from gpuc.control.reconcile import (
    SERVICE_NAME,
    TIMER_NAME,
    HostLiveness,
    Liveness,
    PodQuestion,
    install,
    probe_liveness,
    reconcile_once,
    run_loop,
    unit_files,
)
from gpuc.control.rented import PodAnswer, address_for, desired_from
from gpuc.control.s3index import IndexEntry, LocalIndex
from tests.conftest import host_entry, register_host
from tests.fakeprovider import FakeProvider, PodScript, make_offer, running_pod

FOREIGN = "other-someone-else"


def stamp(**delta: float) -> str:
    return (datetime.now(UTC) + timedelta(**delta)).isoformat(timespec="seconds")


def desire(
    name: str,
    pod_id: str,
    *,
    ttl_hours: float | None = None,
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
        registry.put(host_entry(name=name, kind="runpod", pod_id=pod_id, ssh="root@1.2.3.4"))
    return host


def alive(**overrides: object) -> HostLiveness:
    """A host that answers: the reaper only reaps hosts that have gone quiet."""
    state = Liveness(reachable=True, heartbeat_age_s=5.0, running_jobs=0)
    for key, value in overrides.items():
        setattr(state, key, value)
    return lambda host, entry, settings: state


def asked(
    detail: str = "it has no config.json", config: dict[str, Any] | None = None
) -> PodQuestion:
    """What every pod with no local record says when this pass asks it.

    With no `config` it is a pod that claims nothing -- because gpuc was never
    on it, or because it would not answer at all, which are the same answer
    here. `config` is what a pod another machine set up says.
    """

    def ask(pod: Pod, settings: Settings) -> PodAnswer:
        if config is None:
            return PodAnswer(pod, detail, entry=address_for(pod.name, pod))
        return PodAnswer(
            pod,
            "its config.json says so",
            entry=address_for(pod.name, pod),
            desired=desired_from(pod.id, config, name=pod.name, seen_at=stamp()),
        )

    return ask


def provider_with(*pods: object, **kwargs: object) -> FakeProvider:
    provider = FakeProvider(**kwargs)  # type: ignore[arg-type]
    for pod in pods:
        provider.adopt(pod, PodScript(ssh_after_polls=0))  # type: ignore[arg-type]
    return provider


def test_healthy_host_is_left_alone(control_env: Path) -> None:
    pod = running_pod("gpuc-a-111", "pod1")
    provider = provider_with(pod)
    desire("gpuc-a-111", "pod1")
    result = reconcile_once(Settings(), provider, lambda _: None, liveness=alive())
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


def test_the_state_lock_is_not_held_across_probes_and_terminates(control_env: Path) -> None:
    """A pass must not starve a concurrent `gpuc submit --runpod`.

    The liveness probe waits up to 20 s per host and a terminate polls for up
    to five minutes; submit gives up on the lock after 120 s, and a create that
    cannot write `desired/` is a leaked, billing pod.
    """
    provider = provider_with(running_pod("gpuc-a-111", "pod1"))
    desire("gpuc-a-111", "pod1", created_hours_ago=2.0)
    free: list[str] = []

    def note_if_free(label: str) -> None:
        try:
            with state_lock(timeout_s=0.05):
                free.append(label)
        except ConfigError:
            pass

    terminate = provider.terminate

    def watched(pod_id: str) -> None:
        note_if_free("terminate")
        terminate(pod_id)

    provider.terminate = watched  # type: ignore[method-assign]

    def probe(host: DesiredHost, entry: HostEntry | None, settings: Settings) -> Liveness:
        note_if_free("liveness")
        return Liveness(reachable=False)

    result = reconcile_once(Settings(), provider, lambda _: None, liveness=probe)
    assert result.terminated == ["gpuc-a-111"]
    assert free == ["liveness", "terminate"]


def test_a_record_written_while_the_pass_was_probing_is_not_clobbered(control_env: Path) -> None:
    """The copy this pass read is minutes and several ssh calls old."""
    provider = provider_with(running_pod("gpuc-a-111", "pod1"))
    desire("gpuc-a-111", "pod1")

    def probe(host: DesiredHost, entry: HostEntry | None, settings: Settings) -> Liveness:
        current = read_desired("gpuc-a-111")
        assert current is not None
        write_desired(current.model_copy(update={"image": "written:by-another-session"}))
        return Liveness(reachable=True, heartbeat_age_s=5.0)

    reconcile_once(Settings(), provider, lambda _: None, liveness=probe)
    current = read_desired("gpuc-a-111")
    assert current is not None
    assert current.image == "written:by-another-session"
    assert current.last_seen_at


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
    kept = reconcile_once(Settings(), provider, lambda _: None, liveness=alive()).kept
    assert kept == ["gpuc-a-111"]
    assert provider.terminated == []


@pytest.mark.parametrize(
    "detail",
    [
        "it answers ssh and has no /root/.gpuc/config.json",
        "could not reach host; Permission denied (publickey)",
        "the provider gives it no ssh endpoint",
    ],
)
def test_no_pod_is_ever_terminated_for_having_no_record_here(
    control_env: Path, detail: str
) -> None:
    """Whatever it says, or does not say. A pod with no config may be a leak --
    or another machine's create, or a container whose $HOME was wiped under a
    running job -- and one that will not answer may be wedged or may simply
    hold no key of ours. None of that is worth a terminate; it is worth a line."""
    provider = provider_with(running_pod("gpuc-leaked-999", "podX", age_minutes=600))
    desired_dir().mkdir(parents=True, exist_ok=True)
    reports: list[str] = []

    result = reconcile_once(Settings(), provider, reports.append, ask=asked(detail))

    assert provider.terminated == []
    assert (result.terminated, result.kept) == ([], [])
    assert result.unclaimed == ["gpuc-leaked-999"]
    assert any("Nothing was terminated" in line and detail in line for line in reports)
    assert any("gpuc host add <name> --pod podX" in line for line in reports)


def test_a_young_unclaimed_pod_says_it_may_still_be_provisioning(control_env: Path) -> None:
    provider = provider_with(running_pod("gpuc-new-999", "podX", age_minutes=2))
    desired_dir().mkdir(parents=True, exist_ok=True)
    reports: list[str] = []
    result = reconcile_once(Settings(), provider, reports.append, ask=asked())
    assert provider.terminated == []
    assert result.unclaimed == ["gpuc-new-999"]
    assert any("may still be provisioning it" in line for line in reports)


def test_foreign_pods_are_never_touched(control_env: Path) -> None:
    provider = FakeProvider(existing=[running_pod(FOREIGN, "podF", age_minutes=600)])
    desired_dir().mkdir(parents=True, exist_ok=True)
    reports: list[str] = []
    result = reconcile_once(Settings(), provider, reports.append, ask=asked())
    assert provider.terminated == []
    assert result.terminated == []
    assert not any(FOREIGN in line for line in reports)


def test_missing_desired_directory_does_nothing(control_env: Path) -> None:
    provider = provider_with(running_pod("gpuc-leaked-999", "podX", age_minutes=600))
    reports: list[str] = []

    result = reconcile_once(Settings(), provider, reports.append, ask=asked())

    assert provider.terminated == []
    assert result.errors and "does not exist" in result.errors[0]
    assert any("not the same as `no pods`" in line for line in reports)


def test_unreadable_desired_entry_does_nothing(control_env: Path) -> None:
    provider = provider_with(running_pod("gpuc-leaked-999", "podX", age_minutes=600))
    desire("gpuc-a-111", "pod1")
    desired_file("gpuc-a-111").write_text("{not json")

    result = reconcile_once(Settings(), provider, lambda _: None, ask=asked())

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
        liveness=alive(),
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


# -- the dead-dispatcher safety net that replaced the overall TTL --------------


def dead(**overrides: object) -> HostLiveness:
    state = Liveness(reachable=False)
    for key, value in overrides.items():
        setattr(state, key, value)
    return lambda host, entry, settings: state


def test_a_host_with_no_ttl_is_never_terminated_for_age(control_env: Path) -> None:
    provider = provider_with(running_pod("gpuc-a-111", "pod1", age_minutes=60 * 200))
    desire("gpuc-a-111", "pod1", ttl_hours=None, created_hours_ago=200)

    result = reconcile_once(Settings(), provider, lambda _: None, liveness=alive())

    assert (result.kept, provider.terminated) == (["gpuc-a-111"], [])


def test_a_long_running_job_keeps_an_old_host_alive(control_env: Path) -> None:
    provider = provider_with(running_pod("gpuc-a-111", "pod1", age_minutes=60 * 50))
    desire("gpuc-a-111", "pod1", ttl_hours=None, created_hours_ago=50)
    # The dispatcher is silent, but the host's own state says a job is running.
    liveness = alive(heartbeat_age_s=None, running_jobs=1)

    result = reconcile_once(Settings(), provider, lambda _: None, liveness=liveness)

    assert (result.kept, provider.terminated) == (["gpuc-a-111"], [])


def test_a_dead_dispatcher_past_the_limit_is_terminated_loudly(control_env: Path) -> None:
    provider = provider_with(running_pod("gpuc-a-111", "pod1", age_minutes=120))
    desire("gpuc-a-111", "pod1", ttl_hours=None, created_hours_ago=2)
    reports: list[str] = []

    result = reconcile_once(
        Settings(dead_dispatcher_minutes=30.0),
        provider,
        reports.append,
        liveness=alive(heartbeat_age_s=4000.0),
    )

    assert result.terminated == ["gpuc-a-111"]
    assert provider.terminated == ["pod1"]
    assert any("DEAD DISPATCHER" in line for line in reports)
    assert not desired_file("gpuc-a-111").exists()


def test_an_unreachable_host_is_terminated_once_it_has_been_silent_long_enough(
    control_env: Path,
) -> None:
    provider = provider_with(running_pod("gpuc-a-111", "pod1", age_minutes=120))
    desire("gpuc-a-111", "pod1", ttl_hours=None, created_hours_ago=2)
    reports: list[str] = []

    result = reconcile_once(
        Settings(dead_dispatcher_minutes=30.0), provider, reports.append, liveness=dead()
    )

    assert result.terminated == ["gpuc-a-111"]
    assert any("ssh did not answer" in line for line in reports)


def test_a_host_that_has_only_just_gone_quiet_is_left_alone(control_env: Path) -> None:
    provider = provider_with(running_pod("gpuc-a-111", "pod1", age_minutes=10))
    desire("gpuc-a-111", "pod1", ttl_hours=None, created_hours_ago=0.1)

    result = reconcile_once(
        Settings(dead_dispatcher_minutes=30.0), provider, lambda _: None, liveness=dead()
    )

    assert (result.kept, provider.terminated) == (["gpuc-a-111"], [])


def test_a_healthy_pass_records_that_the_host_was_seen(control_env: Path) -> None:
    provider = provider_with(running_pod("gpuc-a-111", "pod1", age_minutes=10))
    desire("gpuc-a-111", "pod1", ttl_hours=None, created_hours_ago=1)

    reconcile_once(Settings(), provider, lambda _: None, liveness=alive())

    seen = DesiredHost.model_validate_json(desired_file("gpuc-a-111").read_text()).last_seen_at
    assert seen is not None


def test_a_never_bootstrapped_host_still_gets_the_ceiling_not_the_silence_rule(
    control_env: Path,
) -> None:
    provider = provider_with(running_pod("gpuc-a-111", "pod1", age_minutes=60))
    desire("gpuc-a-111", "pod1", ttl_hours=None, created_hours_ago=1, bootstrapped=False)
    reports: list[str] = []

    result = reconcile_once(Settings(), provider, reports.append, liveness=dead())

    assert result.terminated == ["gpuc-a-111"]
    assert any("never bootstrapped by its ceiling" in line for line in reports)


# -- a pod created on another machine, reconciled from this one ----------------
#
# `desired/` is only ever written by the machine that ran `gpuc submit
# --runpod`. Reconciling from a second machine used to read that as "a leak"
# and terminate a healthy pod mid-job at the 15 min ceiling (issue #36).


def pod_config(
    name: str,
    pod_id: str,
    *,
    ttl_hours: float | None = None,
    created_hours_ago: float = 2.0,
    bootstrapped: bool = True,
) -> dict[str, Any]:
    """The `config.json` a pod another machine provisioned is carrying."""
    created = stamp(hours=-created_hours_ago)
    provider: dict[str, Any] = {
        "kind": "runpod",
        "pod_id": pod_id,
        "offer": make_offer().model_dump(mode="json"),
        "created_at": created,
    }
    if bootstrapped:
        provider["bootstrapped_at"] = created
    return {
        "host": name,
        "gpus": ["GPU-1111"],
        "ttl_hours": ttl_hours,
        "created_at": created,
        "provider": provider,
    }


def test_a_pod_another_machine_created_is_adopted_not_terminated(control_env: Path) -> None:
    """The bug: past the ceiling, with no desired/ record here, and mid-job."""
    provider = provider_with(running_pod("gpuc-a-111", "pod1", age_minutes=120))
    desired_dir().mkdir(parents=True, exist_ok=True)
    reports: list[str] = []

    result = reconcile_once(
        Settings(),
        provider,
        reports.append,
        liveness=alive(),
        ask=asked(config=pod_config("gpuc-a-111", "pod1")),
    )

    assert provider.terminated == []
    assert result.kept == ["gpuc-a-111"]
    assert any("so it is ours" in line for line in reports)
    # Cached, so the next pass judges it even if the pod stops answering.
    cached = read_desired("gpuc-a-111")
    assert cached is not None
    assert (cached.pod_id, cached.offer.name) == ("pod1", "A40")


def test_an_adopted_pod_is_judged_by_the_ttl_it_carries(control_env: Path) -> None:
    provider = provider_with(running_pod("gpuc-a-111", "pod1", age_minutes=130))
    desired_dir().mkdir(parents=True, exist_ok=True)
    reports: list[str] = []

    result = reconcile_once(
        Settings(),
        provider,
        reports.append,
        liveness=alive(),
        ask=asked(config=pod_config("gpuc-a-111", "pod1", ttl_hours=1.0, created_hours_ago=2.2)),
    )

    assert provider.terminated == ["pod1"]
    assert result.terminated == ["gpuc-a-111"]
    assert any("past its 1 h TTL" in line for line in reports)


def test_an_adopted_pod_that_has_gone_silent_is_reaped_from_here(control_env: Path) -> None:
    """The watchdog role: any machine running the timer reaps a dead pod.

    Not on the pass that met it, though. The pod answered `cat config.json`
    seconds earlier, so adoption stamps `last_seen_at` and the pod gets the
    same `dead_dispatcher_minutes` allowance as one this machine provisioned --
    which is what stops a single misread by a pulse probe that has never run
    against this pod before from ending someone's job.
    """
    provider = provider_with(running_pod("gpuc-a-111", "pod1", age_minutes=120))
    desired_dir().mkdir(parents=True, exist_ok=True)
    settings = Settings(dead_dispatcher_minutes=30.0)
    ask = asked(config=pod_config("gpuc-a-111", "pod1"))
    reports: list[str] = []

    first = reconcile_once(settings, provider, reports.append, liveness=dead(), ask=ask)

    assert (first.kept, provider.terminated) == (["gpuc-a-111"], [])
    assert any("silent for 0 min of the 30 min limit" in line for line in reports)

    # ...and once it has been silent that long, from here, it goes.
    cached = read_desired("gpuc-a-111")
    assert cached is not None
    write_desired(cached.model_copy(update={"last_seen_at": stamp(minutes=-31)}))
    result = reconcile_once(settings, provider, reports.append, liveness=dead(), ask=ask)

    assert result.terminated == ["gpuc-a-111"]
    assert any("DEAD DISPATCHER" in line for line in reports)


def test_a_pod_with_a_config_but_no_bootstrap_stamp_is_still_ours(control_env: Path) -> None:
    """A pod set up by a build that did not record the stamp is not a stray.

    Its record has no `bootstrapped_at` of its own, and reading that as "never
    bootstrapped" would terminate it at the ceiling; the dead-dispatcher rule
    is what judges it instead.
    """
    provider = provider_with(running_pod("gpuc-a-111", "pod1", age_minutes=120))
    desired_dir().mkdir(parents=True, exist_ok=True)

    result = reconcile_once(
        Settings(),
        provider,
        lambda _: None,
        liveness=alive(),
        ask=asked(config=pod_config("gpuc-a-111", "pod1", bootstrapped=False)),
    )

    assert (result.kept, provider.terminated) == (["gpuc-a-111"], [])


def test_a_pod_already_in_desired_is_never_asked(control_env: Path) -> None:
    """One `cat config.json` per pod, and only for the pods nothing here wants."""
    provider = provider_with(running_pod("gpuc-a-111", "pod1"))
    desire("gpuc-a-111", "pod1")
    asked_about: list[str] = []

    def ask(pod: Pod, settings: Settings) -> PodAnswer:
        asked_about.append(pod.id)
        return PodAnswer(pod, "should not have been asked")

    reconcile_once(Settings(), provider, lambda _: None, liveness=alive(), ask=ask)

    assert asked_about == []


def test_a_pod_whose_config_names_no_host_is_not_called_local(control_env: Path) -> None:
    """`HostConfig`'s default `host` is `local`, which is this machine's own.

    A record under that name would be matched against this machine's `local`
    entry on the next pass: the reaper would probe the wrong box and then
    forget somebody's own host.
    """
    provider = provider_with(running_pod("gpuc-a-111", "pod1", age_minutes=120))
    desired_dir().mkdir(parents=True, exist_ok=True)
    register_host(name="local", gpus="0")
    config = pod_config("gpuc-a-111", "pod1")
    config.pop("host")

    result = reconcile_once(
        Settings(), provider, lambda _: None, liveness=alive(), ask=asked(config=config)
    )

    assert result.kept == ["gpuc-a-111"]
    assert read_desired("gpuc-a-111") is not None
    assert read_desired("local") is None
    assert "local" in load_registry().hosts


def test_a_pod_answering_to_a_name_already_taken_is_left_alone(control_env: Path) -> None:
    """Judging it would mean forgetting the host whose record that name is."""
    provider = provider_with(
        running_pod("gpuc-a-111", "pod1", age_minutes=120),
        running_pod("gpuc-a-222", "pod2", age_minutes=120),
    )
    desire("gpuc-a-111", "pod1", ttl_hours=None)
    reports: list[str] = []

    # The second pod says it is called `gpuc-a-111` too (a hand-set `host`).
    result = reconcile_once(
        Settings(),
        provider,
        reports.append,
        liveness=alive(),
        ask=asked(config=pod_config("gpuc-a-111", "pod2")),
    )

    assert provider.terminated == []
    assert (result.kept, result.unclaimed) == (["gpuc-a-111"], ["gpuc-a-222"])
    assert any("already here and names another pod" in line for line in reports)
    cached = read_desired("gpuc-a-111")
    assert cached is not None and cached.pod_id == "pod1"


def test_the_provider_says_where_a_pod_is_now_and_the_registry_where_its_home_is(
    control_env: Path,
) -> None:
    """A registry entry pinned to an endpoint the pod has moved off fails every
    probe, which reads as a dead dispatcher; its `gpuc_home` is still the only
    copy of where gpuc lives on that pod."""
    provider = provider_with(running_pod("gpuc-a-111", "pod1"))
    desire("gpuc-a-111", "pod1")
    with registry_transaction() as registry:
        entry = registry.require("gpuc-a-111")
        registry.put(entry.model_copy(update={"ssh": "root@5.6.7.8", "gpuc_home": "/vol/gpuc"}))
    seen: list[HostEntry | None] = []

    def probe(host: DesiredHost, entry: HostEntry | None, settings: Settings) -> Liveness:
        seen.append(entry)
        return Liveness(reachable=True, heartbeat_age_s=5.0)

    reconcile_once(Settings(), provider, lambda _: None, liveness=probe)

    assert len(seen) == 1 and seen[0] is not None
    assert (seen[0].ssh, seen[0].port) == ("root@1.2.3.4", 22000)
    assert seen[0].remote_home == "/vol/gpuc"


def test_the_liveness_probe_reads_the_hosts_own_files(control_env: Path, tmp_path: Path) -> None:
    """The one seam every other test here stubs: `probe_liveness` -> `pulse`.

    A `local` entry means the real transport is this machine's shell, so this
    exercises the glue -- the gpuc home the entry carries, and the script run
    against it -- rather than a fake in the shape of an answer.
    """
    home = tmp_path / "gpuc"
    (home / "jobs").mkdir(parents=True)
    (home / "dispatcher.heartbeat").touch()
    entry = host_entry(name="gpuc-a-111", kind="local", gpuc_home=str(home))

    state = probe_liveness(DesiredHost(name="gpuc-a-111", pod_id="pod1"), entry, Settings())

    assert state.reachable and state.alive
    assert state.heartbeat_age_s is not None and state.heartbeat_age_s < 60.0
    # And a host this machine has no address for at all is not "alive by default".
    assert not probe_liveness(DesiredHost(name="x"), None, Settings()).reachable


def test_an_unclaimed_pod_is_named_in_the_json_document(control_env: Path) -> None:
    provider = provider_with(running_pod("gpuc-leaked-999", "podX", age_minutes=600))
    desired_dir().mkdir(parents=True, exist_ok=True)

    document = reconcile_once(Settings(), provider, lambda _: None, ask=asked()).document()

    assert document["unclaimed"] == ["gpuc-leaked-999"]
    assert document["kept"] == [] and document["terminated"] == []


def test_an_unclaimed_pod_of_unknown_age_is_still_only_reported(control_env: Path) -> None:
    ageless = running_pod("gpuc-other-999", "podX").model_copy(update={"created_at": None})
    provider = provider_with(ageless)
    desired_dir().mkdir(parents=True, exist_ok=True)
    reports: list[str] = []

    result = reconcile_once(Settings(), provider, reports.append, ask=asked())

    assert (provider.terminated, result.unclaimed) == ([], ["gpuc-other-999"])
    assert any("nothing here claims it" in line for line in reports)


def test_a_terminated_pod_is_not_asked_anything(control_env: Path) -> None:
    gone = running_pod("gpuc-a-111", "pod1").model_copy(update={"status": "TERMINATED"})
    provider = provider_with(gone)
    desired_dir().mkdir(parents=True, exist_ok=True)
    asked_about: list[str] = []

    def ask(pod: Pod, settings: Settings) -> PodAnswer:
        asked_about.append(pod.id)
        return PodAnswer(pod, "should not have been asked")

    result = reconcile_once(Settings(), provider, lambda _: None, ask=ask)

    assert asked_about == []
    assert (result.unclaimed, result.kept) == ([], [])
