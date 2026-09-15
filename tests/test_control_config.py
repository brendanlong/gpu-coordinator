from __future__ import annotations

from pathlib import Path

import pytest

from gpuc.control import config
from gpuc.control.config import (
    ConfigError,
    HostEntry,
    Registry,
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
