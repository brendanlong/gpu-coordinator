from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from gpuc.control.bootstrap import BootstrapError, bootstrap_host, package_files
from gpuc.control.config import HostEntry
from gpuc.control.remote import NO_CONFIG
from gpuc.control.transport import CommandResult
from gpuc.host import jobs
from gpuc.host.jobs import HostConfig
from tests.conftest import host_entry

HEALTH_OK = {
    "host": "h",
    "gpus": ["GPU-a"],
    "ok": True,
    "checks": [
        {"name": "driver", "ok": True, "detail": "nvidia driver 580", "value": "580.173.02"},
        {"name": "disk", "ok": True, "detail": "900 GB free"},
    ],
}
CONFIG_ON_HOST = HostConfig(host="h", gpus=["GPU-a"]).to_dict()
"""What a host that has been set up already says about itself."""

HEALTH_BAD = {
    "host": "h",
    "gpus": ["GPU-a"],
    "ok": False,
    "checks": [{"name": "disk", "ok": False, "detail": "3.0 GB free, floor 20 GB"}],
}


@dataclass
class ScriptedHost:
    """A host that answers the probes bootstrap makes, and remembers installs."""

    host: str = "fake"
    uv_present: bool = True
    python_present: bool = True
    aws_present: bool = True
    hf_present: bool = True
    health: dict[str, object] = field(default_factory=lambda: dict(HEALTH_OK))
    hf_install_fails: bool = False
    uv_cache: str = "/home/u/.cache/uv"
    config: dict[str, object] | None = None
    """What `config.json` on this host already says, if anything."""
    cache_dev: str = "66"
    home_dev: str = "66"
    events: list[str] = field(default_factory=list)
    puts: dict[str, tuple[str, int]] = field(default_factory=dict)
    rsyncs: list[tuple[Path, str, list[str] | None]] = field(default_factory=list)

    def _answer(self, command: str) -> tuple[int, str]:
        if command.startswith("if [ -f") and "config.json" in command:
            return 0, NO_CONFIG if self.config is None else json.dumps(self.config)
        if "-m gpuc.host config --merge" in command:
            # The host's own merge, done by the host's own code: whichever
            # control machine sends a patch, this is what applies it.
            patch = json.loads(self.puts[command.rsplit(" ", 1)[1]][0])
            self.config = jobs.merged_config(self.config or {}, patch)
            return 0, json.dumps(self.config)
        if command.startswith("mv -f") and "config.json" in command:
            source, target = command.split()[2].strip('"'), command.split()[3].strip('"')
            self.config = json.loads(self.puts[source][0])
            self.puts[target] = self.puts.pop(source)
            return 0, ""
        if "astral.sh/uv" in command:
            self.uv_present = True
            return 0, ""
        if ".local/bin/uv" in command and "-x" in command:
            return 0, "/home/u/.local/bin/uv\n" if self.uv_present else ""
        if "python find" in command:
            return (0, "/home/u/.local/python3.12\n") if self.python_present else (1, "")
        if "python install" in command:
            self.python_present = True
            return 0, "installed"
        if "aws-cli/v2/current/bin/aws" in command:
            return 0, "/usr/bin/aws\n" if self.aws_present else ""
        if "awscli-exe-linux" in command:
            self.aws_present = True
            return 0, ""
        if ".local/bin/hf" in command and "-x" in command:
            return 0, "/home/u/.local/bin/hf\n" if self.hf_present else ""
        if "cache dir" in command:
            return 0, (
                f"cache={self.uv_cache}\ncache_dev={self.cache_dev}\nhome_dev={self.home_dev}\n"
            )
        if "tool install huggingface_hub" in command:
            if self.hf_install_fails:
                return 1, "no network"
            self.hf_present = True
            return 0, ""
        if "-m gpuc.host health" in command:
            return (0 if self.health.get("ok") else 1), json.dumps(self.health)
        if "spawn_detached_dispatcher" in command:
            return 0, "4242\n"
        if command.startswith("printf %s"):
            return 0, "/home/u/.gpuc"
        if "nvidia-smi --query-gpu=index,uuid" in command:
            return 0, "0, GPU-a, NVIDIA A40, 46068\n"
        return 0, ""

    def run(self, command: str, *, timeout: float = 120.0, check: bool = True) -> CommandResult:
        self.events.append(command)
        code, out = self._answer(command)
        result = CommandResult(self.host, ["sh", "-c", command], code, out, "")
        return result.check() if check else result

    def put_file(self, content: str | bytes, remote_path: str, mode: int = 0o600) -> None:
        self.events.append(f"put_file {remote_path}")
        text = content.decode() if isinstance(content, bytes) else content
        self.puts[remote_path] = (text, mode)

    def rsync(
        self,
        local_root: Path,
        remote_path: str,
        files: Sequence[str] | None = None,
        excludes: Sequence[str] = (),
    ) -> CommandResult:
        self.events.append(f"rsync {remote_path}")
        self.rsyncs.append((local_root, remote_path, list(files) if files else None))
        return CommandResult(self.host, ["rsync"], 0, "", "")

    def tail(self, remote_path: str, lines: int = 200, follow: bool = False) -> CommandResult:
        return CommandResult(self.host, ["tail"], 0, "", "")

    def argv(self, command: str) -> list[str]:
        return ["sh", "-c", command]

    def interactive_argv(self, command: str) -> list[str]:
        return ["sh", "-c", command]

    def index_of(self, needle: str) -> int:
        return next(i for i, event in enumerate(self.events) if needle in event)


def entry(**overrides: object) -> HostEntry:
    """A registered host whose cache says the host has one GPU.

    The cache, not an instruction: every test here also gives `ScriptedHost` a
    `config`, or leaves it unset to mean "this host has never been set up".
    """
    return host_entry(name="h", **{"gpus": ["GPU-a"], **overrides})


def test_package_files_are_the_git_tracked_host_modules() -> None:
    files = package_files()
    assert "gpuc/host/jobs.py" in files
    assert "gpuc/host/dispatcher.py" in files
    assert all(f.endswith(".py") for f in files)
    assert not any("__pycache__" in f for f in files)


def test_a_fully_provisioned_host_installs_nothing(control_env: Path) -> None:
    host = ScriptedHost()
    updated, result = bootstrap_host(entry(), transport=host, report=lambda _: None)
    assert result.dispatcher_pid == 4242
    assert result.warnings == []
    assert updated.python == "/home/u/.local/python3.12"
    assert updated.uv == "/home/u/.local/bin/uv"
    assert updated.bootstrapped_at
    assert not any("astral.sh/uv" in e for e in host.events)
    assert not any("awscli-exe" in e for e in host.events)
    assert not any("tool install" in e for e in host.events)


def test_a_bare_host_installs_uv_python_aws_and_hf(control_env: Path) -> None:
    host = ScriptedHost(uv_present=False, python_present=False, aws_present=False, hf_present=False)
    _, result = bootstrap_host(entry(), transport=host, report=lambda _: None)
    assert any("astral.sh/uv" in e for e in host.events)
    assert any("python install 3.12" in e for e in host.events)
    assert any("awscli-exe-linux" in e for e in host.events)
    assert any("tool install huggingface_hub" in e for e in host.events)
    assert result.warnings == []


def test_the_shipped_package_replaces_the_one_on_the_host(control_env: Path) -> None:
    """A module deleted upstream must not survive on the host.

    rsync of a file list only ever adds, so without this the host keeps
    importing a module this build no longer has.
    """
    host = ScriptedHost()
    bootstrap_host(entry(), transport=host, report=lambda _: None)
    assert host.index_of('rm -rf "/home/u/.gpuc/pkg/gpuc"') < host.index_of(
        "rsync /home/u/.gpuc/pkg"
    )


def test_resync_package_ships_the_code_without_the_health_check(control_env: Path) -> None:
    """What `gpuc submit` runs for a host on an older commit: the package and
    the dispatcher, not the ten-minute half of bootstrap."""
    from gpuc.control.bootstrap import resync_package

    host = ScriptedHost(config=dict(CONFIG_ON_HOST))
    updated = resync_package(
        entry(python="/home/u/.local/python3.12"), transport=host, report=lambda _: None
    )
    assert any("rsync /home/u/.gpuc/pkg" in e for e in host.events)
    assert any("spawn_detached_dispatcher" in e for e in host.events)
    assert not any("gpuc.host health" in e for e in host.events)
    assert not any("astral.sh/uv" in e for e in host.events)
    # The commit, and nothing else of the host's config: a re-ship before a
    # submit is not the moment to re-decide what the host is.
    assert host.config == {**CONFIG_ON_HOST, "pkg_commit": updated.pkg_commit}
    assert updated.pkg_commit


def test_resync_restores_a_config_the_host_has_lost(control_env: Path) -> None:
    """A pod that restarted with a wiped $HOME: re-shipping the package to it
    without its config would dispatch the job to a host that owns no cards."""
    from gpuc.control.bootstrap import resync_package

    host = ScriptedHost()
    resync_package(
        entry(python="/home/u/.local/python3.12", s3_prefix="s3://mine/gpuc/h"),
        transport=host,
        report=lambda _: None,
    )
    assert host.config is not None
    assert host.config["gpus"] == ["GPU-a"]
    assert host.config["s3_prefix"] == "s3://mine/gpuc/h"


def test_the_package_and_config_land_before_health_runs(control_env: Path) -> None:
    host = ScriptedHost(config=dict(CONFIG_ON_HOST))
    bootstrap_host(entry(), transport=host, report=lambda _: None)
    rsync_root, rsync_dest, files = host.rsyncs[0]
    assert rsync_root.name == "gpu-coordinator"
    assert rsync_dest == "/home/u/.gpuc/pkg"
    assert files is not None and "gpuc/host/dispatcher.py" in files
    assert host.config is not None and host.config["gpus"] == ["GPU-a"]
    assert host.index_of("config --merge") < host.index_of("gpuc.host health")
    assert host.index_of("gpuc.host health") < host.index_of("spawn_detached_dispatcher")


def test_bootstrap_leaves_the_config_the_host_already_has_alone(control_env: Path) -> None:
    """The point of the split: bootstrapping a host installs things on it, it
    does not re-decide what the host is. A second control machine, registered
    with other cards and another mirror, used to replace both in silence."""
    theirs = {
        "host": "h",
        "gpus": ["GPU-b"],
        "s3_prefix": "s3://theirs/gpuc/h",
        "retention_days": 30.0,
        "env": {"HF_HOME": "/big"},
    }
    host = ScriptedHost(config=dict(theirs))
    updated, result = bootstrap_host(
        entry(gpus=["GPU-a"], s3_prefix="s3://mine/gpuc/h"), transport=host, report=lambda _: None
    )
    assert result.warnings == []
    assert host.config is not None
    assert {key: host.config[key] for key in theirs} == theirs
    # And the registry now holds what the host says, not what it was told.
    assert updated.gpus == ["GPU-b"]
    assert updated.s3_prefix == "s3://theirs/gpuc/h"


def test_bootstrap_restores_a_config_on_a_host_that_has_none(control_env: Path) -> None:
    """A host whose $HOME was wiped, or one registered before the split: there
    is nothing on the host to preserve, so the last config seen is written."""
    host = ScriptedHost()
    updated, _ = bootstrap_host(
        entry(s3_prefix="s3://mine/gpuc/h"), transport=host, report=lambda _: None
    )
    assert host.config is not None
    assert host.config["gpus"] == ["GPU-a"]
    assert host.config["s3_prefix"] == "s3://mine/gpuc/h"
    assert host.config["host"] == "h"
    assert updated.gpus == ["GPU-a"]


def test_bootstrap_refuses_to_replace_a_config_it_cannot_read(control_env: Path) -> None:
    """A half-written `config.json` is a file the host is running on. "There is
    none" is the host saying so, not a parse that failed."""

    class Corrupt(ScriptedHost):
        def _answer(self, command: str) -> tuple[int, str]:
            if "config.json" in command and command.startswith("if [ -f"):
                return 0, '{"host": "h", "gpus": ['
            return super()._answer(command)

    with pytest.raises(BootstrapError) as caught:
        bootstrap_host(entry(), transport=Corrupt(), report=lambda _: None)
    assert "could not be read" in str(caught.value)
    assert "delete it" in str(caught.value)


def test_bootstrap_says_so_when_it_has_no_config_to_give_a_bare_host(
    control_env: Path,
) -> None:
    host = ScriptedHost()
    _, result = bootstrap_host(host_entry(name="h"), transport=host, report=lambda _: None)
    (warning,) = result.warnings
    assert "no config of its own" in warning
    assert "gpuc host set h --gpus" in warning


def test_every_host_command_pins_pythonpath_and_gpuc_home(control_env: Path) -> None:
    host = ScriptedHost()
    bootstrap_host(entry(), transport=host, report=lambda _: None)
    for event in host.events:
        if "gpuc.host" in event or "from gpuc.host import" in event:
            assert 'PYTHONPATH="/home/u/.gpuc/pkg"' in event
            assert 'GPUC_HOME="/home/u/.gpuc"' in event
            assert '"/home/u/.local/python3.12"' in event


def test_a_failed_health_check_fails_bootstrap_with_its_json(control_env: Path) -> None:
    host = ScriptedHost(health=dict(HEALTH_BAD))
    with pytest.raises(BootstrapError) as exc:
        bootstrap_host(entry(), transport=host, report=lambda _: None)
    assert "3.0 GB free" in str(exc.value)
    assert "gpuc host bootstrap h" in str(exc.value)
    assert not any("spawn_detached_dispatcher" in e for e in host.events)


def test_an_optional_tool_failure_is_a_warning_not_an_error(control_env: Path) -> None:
    host = ScriptedHost(hf_present=False, hf_install_fails=True)
    _, result = bootstrap_host(entry(), transport=host, report=lambda _: None)
    assert result.dispatcher_pid == 4242
    assert len(result.warnings) == 1
    assert "huggingface_hub" in result.warnings[0]


def test_bootstrap_is_idempotent(control_env: Path) -> None:
    host = ScriptedHost(uv_present=False, hf_present=False, aws_present=False)
    bootstrap_host(entry(), transport=host, report=lambda _: None)
    first = list(host.events)
    host.events.clear()
    bootstrap_host(entry(), transport=host, report=lambda _: None)
    assert any("astral.sh/uv" in e for e in first)
    assert not any("astral.sh/uv" in e for e in host.events)
    assert not any("awscli-exe" in e for e in host.events)


def test_the_dispatcher_is_started_with_the_home_tool_dirs_on_path(control_env: Path) -> None:
    host = ScriptedHost()
    bootstrap_host(entry(), transport=host, report=lambda _: None)
    command = next(e for e in host.events if "spawn_detached_dispatcher" in e)
    assert command.startswith('PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"')


def test_a_mirroring_host_fails_bootstrap_when_the_aws_cli_will_not_install(
    control_env: Path,
) -> None:
    class NoAws(ScriptedHost):
        def _answer(self, command: str) -> tuple[int, str]:
            if "awscli-exe-linux" in command:
                return 1, "curl: (6) could not resolve host"
            return super()._answer(command)

    host = NoAws(aws_present=False)
    with pytest.raises(BootstrapError) as caught:
        bootstrap_host(entry(s3_prefix="s3://bucket/gpuc/h"), transport=host, report=lambda _: None)
    assert "aws CLI could not be installed" in str(caught.value)
    assert "--s3-prefix ''" in str(caught.value)


def test_a_host_without_a_mirror_only_warns_about_a_missing_aws_cli(control_env: Path) -> None:
    class NoAws(ScriptedHost):
        def _answer(self, command: str) -> tuple[int, str]:
            if "awscli-exe-linux" in command:
                return 1, "curl: (6) could not resolve host"
            return super()._answer(command)

    _, result = bootstrap_host(entry(), transport=NoAws(aws_present=False), report=lambda _: None)
    assert any("aws CLI install failed" in warning for warning in result.warnings)


def test_bootstrap_records_what_the_cards_are(control_env: Path) -> None:
    updated, _ = bootstrap_host(entry(), transport=ScriptedHost(), report=lambda _: None)
    assert updated.gpu_info["GPU-a"].name == "NVIDIA A40"
    assert updated.gpu_info["GPU-a"].vram_mib == 46068
    assert updated.driver_version == "580.173.02"


def test_bootstrap_records_the_commit_on_the_host_and_in_the_registry(control_env: Path) -> None:
    """`gpuc version` warns by comparing these two, so a bootstrap that wrote
    neither would report every host as up to date forever."""
    from gpuc.control import version as version_mod

    host = ScriptedHost(
        config={**CONFIG_ON_HOST, "pkg_commit": "a" * 40, "created_at": "2020-01-01T00:00:00+00:00"}
    )
    updated, result = bootstrap_host(entry(), transport=host, report=lambda _: None)
    assert result.warnings == []
    assert updated.pkg_commit == version_mod.local_commit()
    assert host.config is not None
    assert host.config["pkg_commit"] == updated.pkg_commit
    assert host.config["schema_version"] == 1
    # Whoever registered the host first still owns when that was.
    assert host.config["created_at"] == "2020-01-01T00:00:00+00:00"


def test_a_rented_pod_is_stamped_as_bootstrapped_in_its_own_config(control_env: Path) -> None:
    """The pod's own record (`rented`) says when it was set up, and by
    whichever machine did it, so a second machine adopting it can tell."""
    bought = {"kind": "runpod", "pod_id": "pod1", "created_at": "2026-09-15T12:00:00+00:00"}
    host = ScriptedHost(config={**CONFIG_ON_HOST, "provider": dict(bought)})
    bootstrap_host(
        entry(kind="runpod", pod_id="pod1", provider=dict(bought)),
        transport=host,
        report=lambda _: None,
    )
    assert host.config is not None
    provider = host.config["provider"]
    assert isinstance(provider, dict)
    # Stamped, and nothing else about the pod's own record disturbed.
    assert provider["bootstrapped_at"]
    assert (provider["pod_id"], provider["created_at"]) == ("pod1", bought["created_at"])


def test_a_host_nobody_rents_gets_no_provider_block(control_env: Path) -> None:
    host = ScriptedHost(config=dict(CONFIG_ON_HOST))
    bootstrap_host(entry(), transport=host, report=lambda _: None)
    assert host.config is not None and host.config.get("provider") is None
