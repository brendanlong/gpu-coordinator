"""The regression test for the shared-state incident.

Two sessions of one user share `~/.local/share/gpu-coordinator/hosts.json`, and
a host's `~/.gpuc/config.json` outlives the build that wrote it. When
`ttl_hours` (a field since removed altogether) became `float | None` and a
`null` reached both files, the other session's older build failed validation
on *every* subcommand and the host's dispatcher died 20 times in `float(None)`.

So: every shape of those files -- today's, an older build's, a newer build's --
must parse under today's models, nulls for optional fields must survive
unchanged, nulls for non-optional ones must mean the default, and a key this
build no longer has -- `ttl_hours` is in every fixture -- is ignored.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gpuc.control.config import (
    HostEntry,
    Registry,
    Settings,
    read_registry,
)
from gpuc.control.s3index import IndexEntry
from gpuc.host.dispatcher import LockBody
from gpuc.host.jobs import SCHEMA_VERSION, HostConfig, JobSpec, JobState
from tests.conftest import host_entry

FIXTURES = Path(__file__).parent / "fixtures" / "schema"
HOSTS_SHAPES = (
    "hosts.current.json",
    "hosts.presplit.json",
    "hosts.older.json",
    "hosts.newer.json",
)
CONFIG_SHAPES = ("config.current.json", "config.older.json", "config.newer.json")


def load(name: str) -> dict[str, object]:
    return json.loads((FIXTURES / name).read_text())


@pytest.mark.parametrize("name", HOSTS_SHAPES)
def test_every_committed_registry_shape_parses(name: str, control_env: Path) -> None:
    (control_env / "state" / "hosts.json").write_text((FIXTURES / name).read_text())
    read = read_registry()
    assert not read.unreadable
    assert not read.errors
    assert read.registry.hosts
    for entry in read.registry.hosts.values():
        assert entry.name
        assert entry.config  # the shape the host holds, as this registry last saw it
        assert entry.initial_config()  # the shape written to a host that has none


@pytest.mark.parametrize("name", CONFIG_SHAPES)
def test_every_committed_host_config_shape_parses(name: str) -> None:
    config = HostConfig.from_dict(load(name))
    assert config.host == "gpubox"
    assert config.gpus
    assert isinstance(config.idle_minutes, float)


def test_the_older_registry_drops_its_ttl_and_defaults_what_it_never_had(
    control_env: Path,
) -> None:
    (control_env / "state" / "hosts.json").write_text((FIXTURES / "hosts.older.json").read_text())
    entry = read_registry().registry.hosts["gpubox"]
    assert not hasattr(entry, "ttl_hours")
    assert entry.idle_minutes == 15.0
    assert entry.retention_days is None
    assert entry.gpu_info == {}
    assert entry.pkg_commit is None


def test_a_pre_split_registry_reads_as_a_cache_of_each_hosts_config(control_env: Path) -> None:
    """The shape this build wrote until the host became the owner of its config:
    everything it says about a host is now the last thing seen, not the truth."""
    (control_env / "state" / "hosts.json").write_text(
        (FIXTURES / "hosts.presplit.json").read_text()
    )
    entry = read_registry().registry.hosts["gpubox"]
    assert entry.ssh == "gpubox-ssh"
    assert entry.gpus[0] == "GPU-80646905-50a9-afc1-4375-43ca475b15e4"
    assert entry.s3_prefix == "s3://brendanlong-experiments/gpuc/gpubox"
    assert entry.retention_days == 14.0
    assert entry.python is not None and entry.python.endswith("python3.12")
    assert entry.config.host == "gpubox"
    # Nothing has confirmed any of it with the host yet, and it says so.
    assert entry.seen_at is None


def test_the_newer_registry_ignores_what_it_does_not_know(control_env: Path) -> None:
    (control_env / "state" / "hosts.json").write_text((FIXTURES / "hosts.newer.json").read_text())
    read = read_registry()
    assert set(read.registry.hosts) == {"local", "pod-a40"}
    local = read.registry.hosts["local"]
    assert local.idle_minutes == 15.0  # a null for a non-optional field: the default
    assert local.port == 22
    assert local.env == {}
    assert not hasattr(local, "power_cap_watts")
    assert local.gpu_info["GPU-2a4bad3b-9fe3-7031-914d-384254e92908"].vram_mib == 8192
    # A key only the newer build knows survives the round trip through here,
    # because the cached config is kept verbatim and written back as it came.
    assert local.cache.config["power_cap_watts"] == 220
    assert local.cache.config["ttl_hours"] is None


def test_the_older_host_config_survives_the_null_that_crashed_the_dispatcher() -> None:
    older = HostConfig.from_dict(load("config.older.json"))
    assert "ttl_hours" not in older.to_dict()
    assert older.retention_days is None
    # A host whose config predates the key sweeps nothing until something
    # rewrites that file: shipping a package may not start deleting on its own.
    assert older.workdir_days is None
    assert older.schema_version == SCHEMA_VERSION  # missing means 1

    newer = HostConfig.from_dict(load("config.newer.json"))
    assert newer.retention_days is None
    assert newer.idle_minutes == 15.0
    assert newer.env == {}
    assert newer.schema_version == 2


def test_todays_files_carry_a_schema_version(control_env: Path) -> None:
    assert json.loads((FIXTURES / "config.current.json").read_text())["schema_version"] == 1
    assert Registry().model_dump()["schema_version"] == SCHEMA_VERSION


# -- null means default, except where null is the value -----------------------


def test_a_ttl_from_a_build_that_still_had_one_is_ignored_on_both_sides() -> None:
    """`ttl_hours` was a config key until the cap was removed; a file another
    build wrote still carries it, as a number or as an explicit null."""
    for value in (24.0, None, "24"):
        entry = HostEntry.model_validate({"name": "gpubox", "ttl_hours": value})
        assert entry.name == "gpubox" and not hasattr(entry, "ttl_hours")
        config = HostConfig.from_dict({"host": "gpubox", "ttl_hours": value})
        assert config.host == "gpubox" and "ttl_hours" not in config.to_dict()


def test_a_null_non_optional_field_falls_back_to_its_default() -> None:
    entry = HostEntry.model_validate(
        {"name": "gpubox", "idle_minutes": None, "port": None, "gpus": None, "env": None}
    )
    assert (entry.idle_minutes, entry.port, entry.gpus, entry.env) == (15.0, 22, [], {})


OPTIONAL_REGISTRY_FIELDS = ["ssh", "pod_id", "gpuc_home", "persistent_root", "bootstrapped_at"]
OPTIONAL_CACHE_FIELDS = ["read_at", "python", "uv", "driver_version"]


def test_an_explicit_null_optional_field_survives_a_populated_registry_entry() -> None:
    """The rule that broke: `null` must mean null, not "apply the default".

    A default-constructed entry proves nothing -- every optional field is
    already None -- so this round-trips a fully populated entry in which each
    optional field has been explicitly nulled, one at a time and all at once.
    """
    populated = host_entry(
        name="gpubox",
        kind="ssh",
        ssh="me@box",
        port=2222,
        gpuc_home="/home/u/.gpuc",
        persistent_root="/workspace/me",
        pod_id="pod1",
        bootstrapped_at="2026-09-15T20:00:00+00:00",
        python="/home/u/.local/python3.12",
        uv="/home/u/.local/bin/uv",
        driver_version="580.173.02",
        gpus=["GPU-a"],
    )
    document = json.loads(populated.model_dump_json())
    assert [key for key in OPTIONAL_REGISTRY_FIELDS if document[key] is None] == []
    assert [key for key in OPTIONAL_CACHE_FIELDS if document["cache"][key] is None] == []

    for field in OPTIONAL_REGISTRY_FIELDS:
        entry = HostEntry.model_validate({**document, field: None})
        assert getattr(entry, field) is None, field
        # Nulling one field may not quietly reset any of the others.
        untouched = [k for k in OPTIONAL_REGISTRY_FIELDS if k != field]
        assert [getattr(entry, k) for k in untouched] == [
            getattr(populated, k) for k in untouched
        ], field

    for field in OPTIONAL_CACHE_FIELDS:
        entry = HostEntry.model_validate({**document, "cache": {**document["cache"], field: None}})
        assert getattr(entry.cache, field) is None, field
        assert entry.gpus == ["GPU-a"], field

    all_null = HostEntry.model_validate(
        {
            **document,
            **dict.fromkeys(OPTIONAL_REGISTRY_FIELDS, None),
            "cache": {**document["cache"], **dict.fromkeys(OPTIONAL_CACHE_FIELDS, None)},
        }
    )
    assert [getattr(all_null, key) for key in OPTIONAL_REGISTRY_FIELDS] == [None] * len(
        OPTIONAL_REGISTRY_FIELDS
    )
    # And a re-serialised entry still carries the nulls, rather than dropping
    # the keys and letting the next reader default them.
    again = json.loads(all_null.model_dump_json())
    assert [again[key] for key in OPTIONAL_REGISTRY_FIELDS] == [None] * len(
        OPTIONAL_REGISTRY_FIELDS
    )


OPTIONAL_CONFIG_FIELDS = [
    "retention_days",
    "workdir_days",
    "s3_prefix",
    "provider",
    "pkg_commit",
]


def test_an_explicit_null_optional_field_survives_a_populated_host_config() -> None:
    """The host-side twin: `float(None)` in the dispatcher is what started this."""
    populated = HostConfig(
        host="gpubox",
        gpus=["GPU-a"],
        idle_minutes=30.0,
        s3_prefix="s3://bucket/gpuc/gpubox",
        retention_days=14.0,
        workdir_days=1.0,
        provider={"kind": "runpod", "pod_id": "p"},
        pkg_commit="b" * 40,
    )
    document = json.loads(json.dumps(populated.to_dict()))
    assert [key for key in OPTIONAL_CONFIG_FIELDS if document[key] is None] == []

    for field in OPTIONAL_CONFIG_FIELDS:
        config = HostConfig.from_dict({**document, field: None})
        assert getattr(config, field) is None, field
        assert config.host == "gpubox" and config.idle_minutes == 30.0

    all_null = HostConfig.from_dict({**document, **dict.fromkeys(OPTIONAL_CONFIG_FIELDS, None)})
    assert [getattr(all_null, key) for key in OPTIONAL_CONFIG_FIELDS] == [None] * len(
        OPTIONAL_CONFIG_FIELDS
    )
    assert not all_null.ephemeral  # a null provider is not an ephemeral host
    again = HostConfig.from_dict(json.loads(json.dumps(all_null.to_dict())))
    assert again == all_null


def test_an_unknown_key_never_reaches_a_model() -> None:
    entry = HostEntry.model_validate({"name": "gpubox", "favourite_colour": "blue"})
    assert entry.name == "gpubox"
    settings = Settings.model_validate({"s3_bucket": None, "disk_gb": None, "future": 1})
    assert settings.s3_bucket is None and settings.disk_gb == 50
    index = IndexEntry.model_validate({"job_id": "j", "host": None, "attempt": None, "x": 1})
    assert (index.host, index.attempt) == ("", 1)


def test_settings_an_older_build_wrote_still_load() -> None:
    """`dead_dispatcher_minutes` was a key until the client-side reaper went,
    and `max_pods` / `max_total_usd_per_hour` until the account caps did; a
    config.toml that still has them is not an error."""
    settings = Settings.model_validate(
        {
            "dead_dispatcher_minutes": 30.0,
            "max_pods": 2,
            "max_total_usd_per_hour": 1.5,
            "disk_gb": 20,
        }
    )
    assert settings.disk_gb == 20


# -- the host side, which has no pydantic to lean on --------------------------


def test_host_config_from_a_null_or_junk_document_never_raises() -> None:
    assert HostConfig.from_dict(None).host == "local"
    assert HostConfig.from_dict([1, 2, 3]).gpus == []
    junk = HostConfig.from_dict(
        {
            "host": None,
            "gpus": None,
            "idle_minutes": "not a number",
            "ttl_hours": None,
            "retention_days": "soon",
            "provider": "runpod",
            "env": ["HF_HOME=/x"],
            "unknown": {"deeply": "nested"},
        }
    )
    assert junk.host == "local"
    assert junk.idle_minutes == 15.0
    assert junk.retention_days is None
    assert junk.provider is None
    assert junk.env == {}


def test_a_string_number_is_still_a_number() -> None:
    config = HostConfig.from_dict({"host": "gpubox", "idle_minutes": "30", "retention_days": "6"})
    assert (config.idle_minutes, config.retention_days) == (30.0, 6.0)


def test_job_spec_tolerates_nulls_and_unknown_keys() -> None:
    spec = JobSpec.from_dict(
        {
            "command": "true",
            "gpus": None,
            "priority": None,
            "sync_interval_s": None,
            "env": None,
            "secrets": None,
            "outputs": None,
            "max_runtime_min": None,
            "cleanup": None,
            "telemetry": {"unknown": True},
            # What every build before the watchdog was removed wrote.
            "low_util": {"enabled": True, "window_min": 25, "floor_pct": 5, "grace_min": 10},
        }
    )
    assert (spec.gpus, spec.priority, spec.sync_interval_s) == (1, 50, 180)
    assert spec.max_runtime_min is None  # optional: no cap
    assert spec.cleanup == "on_success"


def test_a_spec_with_no_command_is_still_an_error() -> None:
    with pytest.raises(ValueError, match="command"):
        JobSpec.from_dict({"name": "nothing to run"})


def test_job_state_ignores_nulls_and_unknown_keys() -> None:
    state = JobState.from_dict(
        {"status": "running", "attempt": None, "phase": None, "future_field": 7}
    )
    assert state.status == "running"
    assert state.attempt == 1
    assert state.phase is None
    assert not hasattr(state, "future_field")
    assert JobState.from_dict(None).status == "queued"


def test_the_lock_body_reads_a_pid_however_it_was_written() -> None:
    assert LockBody.parse('{"pid": 42, "pgid": 42}').pid == 42
    assert LockBody.parse('{"pid": "42", "pgid": 42.0, "unknown": 1}').pid == 42
    assert LockBody.parse('{"pid": null, "starttime": null}').pid is None
    assert LockBody.parse("[]").pid is None
    assert LockBody.parse("not json").pid is None
    # A lock written before the commit was recorded reads as "not said", which
    # is what makes the next dispatcher treat it as the older build.
    assert LockBody.parse('{"pid": 42, "pgid": 42}').pkg_commit is None
    assert LockBody.parse('{"pkg_commit": ""}').pkg_commit is None


def test_job_state_coerces_the_types_it_acts_on() -> None:
    """A pgid as a string used to reach `os.killpg` and crash every pass."""
    state = JobState.from_dict(
        {
            "status": 7,
            "pid": "1234",
            "pgid": 1234.0,
            "runner_pid": "99",
            "exit_code": "0",
            "attempt": "2",
            "gpus": ["GPU-1", 2],
            "util_recent": [1, "2.5", None, "junk"],
            "workdir_removed": 1,
            "reason": 5,
        }
    )
    assert (state.pid, state.pgid, state.runner_pid) == (1234, 1234, 99)
    assert state.exit_code == 0
    assert state.attempt == 2
    assert state.status == "7" and state.reason == "5"
    assert state.gpus == ["GPU-1", "2"]
    assert state.util_recent == [1.0, 2.5, None, None]
    assert state.workdir_removed is True
    assert JobState.from_dict({"pid": "not a pid", "pgid": []}).pid is None


def test_a_config_from_before_shared_gpus_borrows_nothing() -> None:
    """The key is additive, and its absence is not "share the whole box": a
    host whose config predates it must never start taking somebody else's card.
    A null means the same, from a build that made it optional again."""
    older = HostConfig.from_dict(load("config.older.json"))
    assert older.shared_gpus == []

    explicit_null = HostConfig.from_dict({"gpus": ["0"], "shared_gpus": None})
    assert explicit_null.shared_gpus == []


def test_a_spec_from_before_use_shared_does_not_borrow_either() -> None:
    spec = JobSpec.from_dict({"command": "true", "gpus": 1})
    assert spec.use_shared is False
    assert JobSpec.from_dict({"command": "true", "use_shared": None}).use_shared is False
