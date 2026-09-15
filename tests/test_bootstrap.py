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
        {"name": "driver", "ok": True, "detail": "nvidia driver 580"},
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
    events: list[str] = field(default_factory=list)
    puts: dict[str, tuple[str, int]] = field(default_factory=dict)
    rsyncs: list[tuple[Path, str, list[str] | None]] = field(default_factory=list)

    def _answer(self, command: str) -> tuple[int, str]:
        if "curl -LsSf https://astral.sh/uv" in command:
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
        self, local_root: Path, remote_path: str, files: Sequence[str] | None = None
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
