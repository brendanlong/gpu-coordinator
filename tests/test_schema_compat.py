"""The regression test for the shared-state incident.

Two sessions of one user share `~/.local/share/gpu-coordinator/hosts.json`, and
a host's `~/.gpuc/config.json` outlives the build that wrote it. When
`ttl_hours` became `float | None` and a `null` reached both files, the other
session's older build failed validation on *every* subcommand and the host's
dispatcher died 20 times in `float(None)`.

So: every shape of those files -- today's, an older build's, a newer build's --
must parse under today's models, nulls for optional fields must survive
unchanged, and nulls for non-optional ones must mean the default.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gpuc.control.config import (
    DesiredHost,
    HostEntry,
    Registry,
    Settings,
    read_registry,
)
from gpuc.control.s3index import IndexEntry
from gpuc.host.dispatcher import LockBody
from gpuc.host.jobs import SCHEMA_VERSION, HostConfig, JobSpec, JobState

FIXTURES = Path(__file__).parent / "fixtures" / "schema"
HOSTS_SHAPES = ("hosts.current.json", "hosts.older.json", "hosts.newer.json")
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
        entry.host_config()  # the shape bootstrap writes to the host


@pytest.mark.parametrize("name", CONFIG_SHAPES)
def test_every_committed_host_config_shape_parses(name: str) -> None:
    config = HostConfig.from_dict(load(name))
    assert config.host == "spar"
    assert config.gpus
    assert isinstance(config.idle_minutes, float)


def test_the_older_registry_keeps_its_ttl_and_defaults_what_it_never_had(
    control_env: Path,
) -> None:
    (control_env / "state" / "hosts.json").write_text((FIXTURES / "hosts.older.json").read_text())
    entry = read_registry().registry.hosts["spar"]
    assert entry.ttl_hours == 24.0
    assert entry.retention_days is None
    assert entry.gpu_info == {}
    assert entry.pkg_commit is None


def test_the_newer_registry_ignores_what_it_does_not_know(control_env: Path) -> None:
    (control_env / "state" / "hosts.json").write_text((FIXTURES / "hosts.newer.json").read_text())
    read = read_registry()
    assert set(read.registry.hosts) == {"local", "pod-a40"}
    local = read.registry.hosts["local"]
    assert local.ttl_hours is None  # a real value: never expires
    assert local.idle_minutes == 15.0  # a null for a non-optional field: the default
    assert local.port == 22
    assert local.env == {}
    assert not hasattr(local, "power_cap_watts")
    assert local.gpu_info["GPU-2a4bad3b-9fe3-7031-914d-384254e92908"].vram_mib == 8192


def test_the_older_host_config_survives_the_null_that_crashed_the_dispatcher() -> None:
    older = HostConfig.from_dict(load("config.older.json"))
    assert older.ttl_hours == 24.0
    assert older.retention_days is None
    assert older.schema_version == SCHEMA_VERSION  # missing means 1

    newer = HostConfig.from_dict(load("config.newer.json"))
    assert newer.ttl_hours is None
    assert newer.retention_days is None
    assert newer.idle_minutes == 15.0
    assert newer.env == {}
    assert newer.schema_version == 2


def test_todays_files_carry_a_schema_version(control_env: Path) -> None:
    assert json.loads((FIXTURES / "config.current.json").read_text())["schema_version"] == 1
    assert Registry().model_dump()["schema_version"] == SCHEMA_VERSION


# -- null means default, except where null is the value -----------------------


def test_a_null_ttl_in_the_registry_is_no_ttl_not_the_old_default() -> None:
    entry = HostEntry.model_validate({"name": "spar", "ttl_hours": None})
    assert entry.ttl_hours is None


def test_a_null_non_optional_field_falls_back_to_its_default() -> None:
    entry = HostEntry.model_validate(
        {"name": "spar", "idle_minutes": None, "port": None, "gpus": None, "env": None}
    )
    assert (entry.idle_minutes, entry.port, entry.gpus, entry.env) == (15.0, 22, [], {})


def test_every_optional_registry_field_round_trips_as_none() -> None:
    entry = HostEntry(name="spar")
    optional = [
        "ssh",
        "driver_version",
        "pod_id",
        "python",
        "uv",
        "gpuc_home",
        "persistent_root",
        "cache_dir",
        "ttl_hours",
        "s3_prefix",
        "retention_days",
        "created_at",
        "bootstrapped_at",
        "pkg_commit",
    ]
    document = json.loads(entry.model_dump_json())
    assert [key for key in optional if document[key] is not None] == []
    again = HostEntry.model_validate(document)
    assert again == entry
    assert again.ttl_hours is None and again.retention_days is None and again.s3_prefix is None


def test_every_optional_host_config_field_round_trips_as_none() -> None:
    config = HostConfig(host="spar")
    again = HostConfig.from_dict(json.loads(json.dumps(config.to_dict())))
    assert again == config
    assert again.ttl_hours is None
    assert again.retention_days is None
    assert again.s3_prefix is None
    assert again.provider is None
    assert again.pkg_commit is None


def test_an_unknown_key_never_reaches_a_model() -> None:
    entry = HostEntry.model_validate({"name": "spar", "favourite_colour": "blue"})
    assert entry.name == "spar"
    settings = Settings.model_validate({"s3_bucket": None, "max_pods": None, "future": 1})
    assert settings.s3_bucket is None and settings.max_pods == 3
    index = IndexEntry.model_validate({"job_id": "j", "host": None, "attempt": None, "x": 1})
    assert (index.host, index.attempt) == ("", 1)
    desired = DesiredHost.model_validate({"name": "pod", "offer": None, "idle_minutes": None})
    assert desired.idle_minutes == 15.0 and desired.offer.name == ""


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
    assert junk.ttl_hours is None
    assert junk.retention_days is None
    assert junk.provider is None
    assert junk.env == {}


def test_a_string_number_is_still_a_number() -> None:
    config = HostConfig.from_dict({"host": "spar", "idle_minutes": "30", "ttl_hours": "6"})
    assert (config.idle_minutes, config.ttl_hours) == (30.0, 6.0)


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
            "low_util": None,
            "cleanup": None,
            "telemetry": {"unknown": True},
        }
    )
    assert (spec.gpus, spec.priority, spec.sync_interval_s) == (1, 50, 180)
    assert spec.max_runtime_min is None  # optional: no cap
    assert spec.low_util.window_min == 25.0
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
