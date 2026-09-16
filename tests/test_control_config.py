from __future__ import annotations

from pathlib import Path

import pytest

from gpuc.control import config
from gpuc.control.config import (
    JOB_CONFIG_KEYS,
    ConfigError,
    HostEntry,
    Registry,
    config_drift,
    load_registry,
    load_settings,
    registry_transaction,
    save_registry,
    transport_for,
)
from gpuc.control.transport import LocalTransport, SshTransport


def test_settings_default_when_no_file(control_env: Path) -> None:
    settings = load_settings()
    assert settings.s3_bucket is None
    assert settings.runpod_pod_prefix == "gpuc-"
    assert settings.max_pods == 3


def test_settings_read_the_xdg_overridden_config(control_env: Path) -> None:
    config.config_file().write_text(
        's3_bucket = "my-bucket"\nmax_pods = 1\nssh_key = "~/.ssh/id_ed25519"\n'
    )
    settings = load_settings()
    assert settings.s3_bucket == "my-bucket"
    assert settings.max_pods == 1
    assert settings.ssh_key_path is not None and settings.ssh_key_path.startswith("/")


def test_bad_config_says_which_file_to_fix(control_env: Path) -> None:
    config.config_file().write_text("max_pods = 'three'\n")
    with pytest.raises(ConfigError) as exc:
        load_settings()
    assert str(config.config_file()) in str(exc.value)


def test_registry_round_trips(control_env: Path) -> None:
    entry = HostEntry(name="gpubox", kind="ssh", ssh="me@box", port=2222, gpus=["GPU-a", "GPU-b"])
    registry = Registry()
    registry.put(entry)
    save_registry(registry)
    assert load_registry().require("gpubox") == entry


def test_unknown_host_error_lists_the_known_ones(control_env: Path) -> None:
    with registry_transaction() as registry:
        registry.put(HostEntry(name="local"))
    with pytest.raises(ConfigError) as exc:
        load_registry().require("gpubox")
    assert "Known hosts: local" in str(exc.value)
    assert "gpuc host add" in str(exc.value)


def test_transport_kind_follows_the_entry(control_env: Path) -> None:
    assert isinstance(transport_for(HostEntry(name="local")), LocalTransport)
    ssh = transport_for(HostEntry(name="gpubox", kind="ssh", ssh="me@box", port=2222))
    assert isinstance(ssh, SshTransport)
    assert ssh.port == 2222
    assert ssh.known_hosts == config.state_dir() / "known_hosts"


def test_host_config_for_the_host_side(control_env: Path) -> None:
    entry = HostEntry(name="pod1", kind="runpod", pod_id="abc", gpus=["GPU-a"], ttl_hours=1.0)
    host_config = entry.host_config()
    assert host_config.host == "pod1"
    assert host_config.provider == {"kind": "runpod", "pod_id": "abc"}
    assert host_config.ephemeral
    assert HostEntry(name="local").host_config().provider is None


def test_remote_home_defaults_to_dot_gpuc(control_env: Path) -> None:
    assert HostEntry(name="local").remote_home == "$HOME/.gpuc"
    assert HostEntry(name="local", gpuc_home="/tmp/x").remote_home == "/tmp/x"


def test_config_drift_compares_only_what_the_host_reported() -> None:
    """The whole config.json off a host, or the subset `status` answers with."""
    entry = HostEntry(name="gpubox", kind="ssh", ssh="me@box", gpus=["2", "3"])
    assert config_drift({"host": "gpubox", "gpus": ["2", "3"]}, entry.host_config()) == []
    assert config_drift({"gpus": ["0", "1"]}, entry.host_config()) == ["gpus 0,1 -> 2,3"]
    # A key the host did not report is not a difference.
    assert config_drift({}, entry.host_config()) == []
    assert config_drift(None, entry.host_config()) == []


def test_config_drift_ignores_the_commit_and_names_env_without_its_values() -> None:
    """`--env` is where somebody hand-sets an HF_TOKEN, and this text is printed."""
    entry = HostEntry(name="gpubox", env={"HF_TOKEN": "ours", "HF_HOME": "/big"})
    existing = {
        "pkg_commit": "c" * 40,
        "created_at": "2020-01-01T00:00:00+00:00",
        "schema_version": 999,
        "env": {"HF_TOKEN": "theirs", "HF_HOME": "/big"},
    }
    assert config_drift(existing, entry.host_config()) == ["env differs in HF_TOKEN"]


def test_config_drift_reports_the_settings_that_change_what_a_host_does() -> None:
    entry = HostEntry(name="gpubox", s3_prefix="s3://mine/gpuc/gpubox", retention_days=7.0)
    drift = config_drift(
        {"host": "laptop-box", "s3_prefix": None, "retention_days": 30.0, "ttl_hours": None},
        entry.host_config(),
    )
    assert drift == [
        "host laptop-box -> gpubox",
        "s3_prefix none -> s3://mine/gpuc/gpubox",
        "retention_days 30.0 -> 7.0",
    ]


def test_config_drift_never_prints_an_env_value_whatever_the_host_has_there() -> None:
    """A config.json from another build may have anything at all under `env`,
    including a null, and the fallback formatting used to print our side of the
    comparison -- which is the side holding the token."""
    entry = HostEntry(name="gpubox", env={"HF_TOKEN": "hf_secret"})
    for existing in ({"env": None}, {"env": "HF_TOKEN=hf_theirs"}, {"env": []}):
        drift = config_drift(existing, entry.host_config())
        assert drift == ["env differs in HF_TOKEN"]
        assert "hf_secret" not in drift[0]
    assert (
        "hf_theirs" not in config_drift({"env": {"HF_TOKEN": "hf_theirs"}}, entry.host_config())[0]
    )
    # An env nobody set, however it is spelled, is not a difference.
    plain = HostEntry(name="gpubox")
    assert config_drift({"env": None}, plain.host_config()) == []


def test_config_drift_can_be_narrowed_to_the_keys_a_job_is_affected_by() -> None:
    """`gpuc host set` changes the registry and says the host is unchanged until
    the next bootstrap, so a submit repeating the host's own lifecycle settings
    back at the user would be noise it cannot even clear."""
    entry = HostEntry(name="gpubox", gpus=["2"], idle_minutes=30.0, retention_days=7.0)
    existing = {"gpus": ["0"], "idle_minutes": 15.0, "retention_days": None}
    assert config_drift(existing, entry.host_config(), JOB_CONFIG_KEYS) == ["gpus 0 -> 2"]
    assert len(config_drift(existing, entry.host_config())) == 3


def test_config_drift_is_quiet_about_a_runpod_host_bootstrap_just_wrote() -> None:
    entry = HostEntry(name="gpuc-1", kind="runpod", pod_id="pod-1", gpus=["GPU-a"])
    assert config_drift(entry.host_config().to_dict(), entry.host_config()) == []
