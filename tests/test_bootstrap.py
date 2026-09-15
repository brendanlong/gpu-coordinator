from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from gpuc.control.bootstrap import BootstrapError, bootstrap_host, package_files
from gpuc.control.config import HostEntry
from gpuc.control.transport import CommandResult

HEALTH_OK = {
    "host": "h",
    "gpus": ["GPU-a"],
    "ok": True,
    "checks": [
        {"name": "driver", "ok": True, "detail": "nvidia driver 580", "value": "580.173.02"},
        {"name": "disk", "ok": True, "detail": "900 GB free"},
    ],
}
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
    cache_dev: str = "66"
    home_dev: str = "66"
    events: list[str] = field(default_factory=list)
    puts: dict[str, tuple[str, int]] = field(default_factory=dict)
    rsyncs: list[tuple[Path, str, list[str] | None]] = field(default_factory=list)

    def _answer(self, command: str) -> tuple[int, str]:
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

    def index_of(self, needle: str) -> int:
        return next(i for i, event in enumerate(self.events) if needle in event)


def entry(**overrides: object) -> HostEntry:
    return HostEntry.model_validate({"name": "h", "gpus": ["GPU-a"], **overrides})


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

    host = ScriptedHost()
    updated = resync_package(
        entry(python="/home/u/.local/python3.12"), transport=host, report=lambda _: None
    )
    assert any("rsync /home/u/.gpuc/pkg" in e for e in host.events)
    assert any("spawn_detached_dispatcher" in e for e in host.events)
    assert not any("gpuc.host health" in e for e in host.events)
    assert not any("astral.sh/uv" in e for e in host.events)
    assert "/home/u/.gpuc/config.json" in host.puts
    assert updated.pkg_commit


def test_the_package_and_config_land_before_health_runs(control_env: Path) -> None:
    host = ScriptedHost()
    bootstrap_host(entry(), transport=host, report=lambda _: None)
    rsync_root, rsync_dest, files = host.rsyncs[0]
    assert rsync_root.name == "gpu-coordinator"
    assert rsync_dest == "/home/u/.gpuc/pkg"
    assert files is not None and "gpuc/host/dispatcher.py" in files
    config = json.loads(host.puts["/home/u/.gpuc/config.json"][0])
    assert config["host"] == "h"
    assert config["gpus"] == ["GPU-a"]
    assert host.puts["/home/u/.gpuc/config.json"][1] == 0o644
    assert host.index_of("put_file /home/u/.gpuc/config.json") < host.index_of("gpuc.host health")
    assert host.index_of("gpuc.host health") < host.index_of("spawn_detached_dispatcher")


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

    host = ScriptedHost()
    updated, _ = bootstrap_host(entry(), transport=host, report=lambda _: None)
    assert updated.pkg_commit == version_mod.local_commit()
    config = json.loads(host.puts["/home/u/.gpuc/config.json"][0])
    assert config["pkg_commit"] == updated.pkg_commit
    assert config["schema_version"] == 1
