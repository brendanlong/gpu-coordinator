from __future__ import annotations

import json
from pathlib import Path

import pytest

from gpuc.control import config
from gpuc.control.config import (
    ConfigError,
    HostEntry,
    Registry,
    Rental,
    Settings,
    config_changes,
    config_drift,
    first_config,
    forget_host,
    load_settings,
    pod_known_hosts_file,
    registry_transaction,
    save_registry,
    transport_for,
)
from gpuc.control.gpuinfo import GpuInfo
from gpuc.control.transport import LocalTransport, SshTransport
from gpuc.host.cleanup import DEFAULT_WORKDIR_DAYS
from gpuc.host.jobs import HostConfig
from tests.conftest import SEEN_AT, host_entry, load_registry


def test_settings_default_when_no_file(control_env: Path) -> None:
    settings = load_settings()
    assert settings.s3_bucket is None
    assert settings.runpod_pod_prefix == "gpuc-"
    assert settings.disk_gb == 50


def test_settings_read_the_xdg_overridden_config(control_env: Path) -> None:
    config.config_file().write_text(
        's3_bucket = "my-bucket"\ndisk_gb = 20\nssh_key = "~/.ssh/id_ed25519"\n'
    )
    settings = load_settings()
    assert settings.s3_bucket == "my-bucket"
    assert settings.disk_gb == 20
    assert settings.ssh_key_path is not None and settings.ssh_key_path.startswith("/")


def test_bad_config_says_which_file_to_fix(control_env: Path) -> None:
    config.config_file().write_text("disk_gb = 'twenty'\n")
    with pytest.raises(ConfigError) as exc:
        load_settings()
    assert str(config.config_file()) in str(exc.value)


def test_registry_round_trips(control_env: Path) -> None:
    entry = host_entry(name="gpubox", kind="ssh", ssh="me@box", port=2222, gpus=["GPU-a", "GPU-b"])
    registry = Registry()
    registry.put(entry)
    save_registry(registry)
    assert load_registry().require("gpubox") == entry


def test_unknown_host_error_lists_the_known_ones(control_env: Path) -> None:
    with registry_transaction() as registry:
        registry.put(host_entry(name="local"))
    with pytest.raises(ConfigError) as exc:
        load_registry().require("gpubox")
    assert "Known hosts: local" in str(exc.value)
    assert "gpuc host add" in str(exc.value)


def test_transport_kind_follows_the_entry(control_env: Path) -> None:
    assert isinstance(transport_for(host_entry(name="local")), LocalTransport)
    ssh = transport_for(host_entry(name="gpubox", kind="ssh", ssh="me@box", port=2222))
    assert isinstance(ssh, SshTransport)
    assert ssh.port == 2222
    assert ssh.known_hosts == config.state_dir() / "known_hosts"


def test_the_entry_reads_the_hosts_own_config_out_of_its_cache() -> None:
    """Every field about what a host *is* comes from the copy of `config.json`
    the last connect left behind, not from anything the registry decides."""
    entry = host_entry(
        name="pod1",
        kind="rental",
        pod_id="abc",
        gpus=["GPU-a"],
        retention_days=1.0,
        s3_prefix="s3://bucket/gpuc/pod1",
        env={"HF_HOME": "/big"},
        cache_dir="/vol/uv",
    )
    assert (entry.config.gpus, entry.config.retention_days, entry.config.s3_prefix) == (
        ["GPU-a"],
        1.0,
        "s3://bucket/gpuc/pod1",
    )
    assert entry.config.env == {"HF_HOME": "/big", "UV_CACHE_DIR": "/vol/uv"}
    assert entry.config.ephemeral
    assert entry.seen_at == SEEN_AT
    # Nothing is known about a host nobody has read yet, and it says so.
    blank = HostEntry(name="gpubox", ssh="me@box")
    assert (blank.config.gpus, blank.config.s3_prefix, blank.seen_at) == (None, None, None)


def test_the_provider_block_comes_from_the_rental_for_a_config_we_initialise() -> None:
    address = HostEntry(name="pod1", rental=Rental(pod_id="abc"))
    assert address.provider_block() == {"kind": "runpod", "pod_id": "abc"}
    assert HostEntry(name="local").provider_block() is None
    initial = first_config(address, Settings(s3_bucket="b"))
    assert (initial["host"], initial["provider"]) == ("pod1", {"kind": "runpod", "pod_id": "abc"})
    assert initial["created_at"]
    assert initial["s3_prefix"] == "s3://b/gpuc/pod1"


def test_first_config_owns_every_card_less_the_shared_ones() -> None:
    """The one constructor for a host that has none: the same defaults for a
    box, a pod and a wiped home, and overrides on top. It owns every card, not
    the ones the probe saw, so a card added later is used too."""
    address = HostEntry(name="box", ssh="me@box").with_cache(
        gpu_info={"GPU-a": GpuInfo(index=0), "GPU-b": GpuInfo(index=1)}
    )
    document = first_config(address, Settings(), {"shared_gpus": ["1"], "idle_minutes": 5.0})
    assert "gpus" in document and document["gpus"] is None
    assert document["shared_gpus"] == ["1"]
    assert (document["idle_minutes"], document["workdir_days"], document["s3_prefix"]) == (
        5.0,
        DEFAULT_WORKDIR_DAYS,
        None,
    )
    assert first_config(address, Settings(), {"gpus": ["GPU-b"]})["gpus"] == ["GPU-b"]


def test_a_rental_is_an_address_with_a_pod_behind_it() -> None:
    """Only `rental` makes an entry a rental; `ssh` says nothing about it."""
    assert HostEntry(name="pod1", rental=Rental(pod_id="abc")).kind == "rental"
    assert HostEntry(name="box", ssh="me@box", rental=Rental(pod_id="abc")).pod_id == "abc"
    podless = HostEntry(name="pod1", ssh="root@1.2.3.4")
    assert podless.rental is None and podless.pod_id is None
    assert podless.provider_block() is None
    assert HostEntry(name="local").rental is None


@pytest.mark.parametrize(
    ("address", "kind"),
    [
        ({}, "local"),
        ({"ssh": "me@box"}, "ssh"),
        ({"ssh": "root@1.2.3.4", "rental": {"provider": "runpod", "pod_id": "abc"}}, "rental"),
        ({"rental": {"pod_id": "abc"}}, "rental"),
        # A stored `kind` is a key this build does not read: the address decides.
        ({"kind": "ssh"}, "local"),
        ({"kind": "rental", "ssh": "me@box"}, "ssh"),
    ],
)
def test_kind_is_what_the_address_says(address: dict[str, str], kind: str) -> None:
    entry = HostEntry.model_validate({"name": "h", **address})
    assert entry.kind == kind


def test_kind_is_derived_and_never_written() -> None:
    entry = HostEntry(name="box", ssh="me@box")
    assert "kind" not in json.loads(entry.model_dump_json())


def test_a_pre_split_registry_entry_parses_as_an_address_with_an_empty_cache() -> None:
    """A registry written when this machine believed it owned a host's config
    still parses; the keys it no longer knows are ignored, not folded into the
    cache, so nothing about the host is believed until the host is read."""
    entry = HostEntry.model_validate(
        {
            "name": "gpubox",
            "kind": "ssh",
            "ssh": "me@box",
            "gpus": ["2", "3"],
            "python": "/usr/bin/python3.12",
            "cache_dir": "/mnt/ssd/uv",
            "env": {"HF_HOME": "/big"},
            "retention_days": 24.0,
            "pkg_commit": "a" * 40,
        }
    )
    assert (entry.name, entry.kind, entry.ssh) == ("gpubox", "ssh", "me@box")
    assert entry.config.gpus is None
    assert entry.python is None
    assert entry.config.env == {}
    assert entry.config.pkg_commit is None
    assert entry.seen_at is None


def test_with_config_and_with_cache_stamp_when_the_host_was_read() -> None:
    entry = HostEntry(name="gpubox", ssh="me@box")
    read = entry.with_config({"host": "gpubox", "gpus": ["0"]}, read_at="2026-01-01T00:00:00+00:00")
    assert read.config.gpus == ["0"]
    assert read.seen_at == "2026-01-01T00:00:00+00:00"
    # A later probe keeps the cards it already had names for.
    probed = read.with_cache(python="/usr/bin/python3", read_at="2026-01-02T00:00:00+00:00")
    assert probed.config.gpus == ["0"]
    assert probed.python == "/usr/bin/python3"
    assert probed.seen_at == "2026-01-02T00:00:00+00:00"


def test_remote_home_defaults_to_dot_gpuc(control_env: Path) -> None:
    assert host_entry(name="local").remote_home == "$HOME/.gpuc"
    assert host_entry(name="local", gpuc_home="/tmp/x").remote_home == "/tmp/x"


def config_of(**fields: object) -> HostConfig:
    return HostConfig.from_dict({"host": "gpubox", **fields})


def test_config_drift_compares_only_what_the_host_reported() -> None:
    """The whole config.json off a host, or a subset of one."""
    ours = config_of(gpus=["2", "3"])
    assert config_drift({"host": "gpubox", "gpus": ["2", "3"]}, ours) == []
    assert config_drift({"gpus": ["0", "1"]}, ours) == ["gpus 0,1 -> 2,3"]
    # A key the host did not report is not a difference.
    assert config_drift({}, ours) == []
    assert config_drift(None, ours) == []


def test_config_drift_ignores_the_commit_and_names_env_without_its_values() -> None:
    """`--env` is where somebody hand-sets an HF_TOKEN, and this text is printed."""
    ours = config_of(env={"HF_TOKEN": "ours", "HF_HOME": "/big"})
    existing = {
        "pkg_commit": "c" * 40,
        "created_at": "2020-01-01T00:00:00+00:00",
        "schema_version": 999,
        "env": {"HF_TOKEN": "theirs", "HF_HOME": "/big"},
    }
    assert config_drift(existing, ours) == ["env differs in HF_TOKEN"]


def test_config_drift_reports_the_settings_that_change_what_a_host_does() -> None:
    drift = config_drift(
        {"host": "laptop-box", "s3_prefix": None, "retention_days": 30.0},
        config_of(s3_prefix="s3://mine/gpuc/gpubox", retention_days=7.0),
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
    ours = config_of(env={"HF_TOKEN": "hf_secret"})
    for existing in (
        {"env": None},
        {"env": "HF_TOKEN=hf_theirs"},
        {"env": []},
        {"env": {"HF_TOKEN": "hf_theirs"}},
    ):
        assert config_drift(existing, ours) == ["env differs in HF_TOKEN"]
    # An env nobody set, however it is spelled, is not a difference.
    assert config_drift({"env": None}, config_of()) == []


def test_config_changes_names_every_key_a_flag_would_change_on_the_host() -> None:
    """What `gpuc host add` and `gpuc host set` print: the host's config is the
    only copy of it, so a flag that touches it is an edit of somebody's host."""
    existing = {"host": "gpubox", "gpus": ["0"], "idle_minutes": 15.0}
    assert config_changes(existing, {"gpus": ["2", "3"], "idle_minutes": 15.0}) == ["gpus 0 -> 2,3"]
    # A key the host does not have yet is still a change, and reads as one.
    assert config_changes(existing, {"retention_days": 7.0}) == ["retention_days none -> 7.0"]
    assert config_changes({}, {}) == []
    # But a key it has never written, set to the default it already behaves by,
    # is not: `--gpus all` on a host with no gpus key changes nothing...
    assert config_changes({"host": "gpubox"}, {"gpus": None, "idle_minutes": 15.0}) == []
    # ...while `--gpus ''` there takes every card away, and says so.
    assert config_changes({"host": "gpubox"}, {"gpus": []}) == ["gpus all -> none"]
    assert config_changes({"gpus": None}, {"gpus": ["0", "1"]}) == ["gpus all -> 0,1"]
    assert config_changes(existing, {"gpus": None}) == ["gpus 0 -> all"]


def test_config_drift_is_quiet_about_a_config_just_written() -> None:
    entry = host_entry(name="gpuc-1", kind="rental", pod_id="pod-1", gpus=["GPU-a"])
    assert config_drift(entry.config.to_dict(), entry.config) == []
    # A difference in the provider block reads as a sentence, not as punctuation.
    fresh = HostConfig.from_dict(
        first_config(HostEntry(name="gpuc-1", rental=Rental(pod_id="pod-2")), Settings())
    )
    assert config_drift({"provider": {"kind": "runpod", "pod_id": "pod-1"}}, fresh) == [
        "provider kind=runpod pod_id=pod-1 -> kind=runpod pod_id=pod-2"
    ]


def test_forgetting_a_pod_leaves_a_different_host_of_the_same_name_alone(
    control_env: Path,
) -> None:
    """A pod and the registry can disagree about what a name means.

    Reuse forgets a host when its *pod* is gone or terminated; dropping
    somebody's registered box because a pod answered to the same name is not
    something that should be possible.
    """
    mine = host_entry(name="shared", kind="ssh", ssh="me@box")
    with registry_transaction() as registry:
        registry.put(mine)
    pinned = pod_known_hosts_file("shared")
    pinned.parent.mkdir(parents=True, exist_ok=True)
    pinned.write_text("[1.2.3.4]:22 ssh-ed25519 AAAA\n")

    assert not forget_host("shared", "podX")
    assert load_registry().hosts["shared"].ssh == "me@box"  # not this pod's entry
    assert pinned.exists()  # and its pinned host key was not this pod's to drop

    assert forget_host("shared", None)  # no pod named: forget the host too
    assert load_registry().hosts == {}
    assert not pinned.exists()
