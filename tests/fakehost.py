"""An in-memory host, for the commands that now have to talk to one.

`gpuc host add` reads the host's `config.json` (and writes its first one), and
`gpuc host set` writes through to it, so a CLI test that registers a host needs
something on the other end of the transport. This is that: a dict of files,
plus the handful of commands those two paths send.
"""

from __future__ import annotations

import json
import shlex
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from gpuc.control.remote import NO_CONFIG
from gpuc.control.transport import CommandResult
from gpuc.host import jobs

HOME = "/home/u"
GPU_ROWS = ["0, GPU-a, NVIDIA A40, 46068 MiB", "1, GPU-b, NVIDIA A40, 46068 MiB"]

PROBE_SECTIONS = {
    "system": "Linux 6.8.0 x86_64\nuser=u home=/home/u shell=/bin/bash",
    "driver": "580.173.02",
    "gpus": "\n".join(GPU_ROWS),
    "disk": "/dev/sda1 900G 100G 800G 12% /home",
    "home_fs": "/dev/sda1 ext4 900G 100G 800G 12% /home",
    "killuserprocesses": "KillUserProcesses=no",
    "systemd_scope": "yes",
    "uv": "uv 0.9.2",
    "uv_cache": "dir=/home/u/.cache/uv\nsize=1.2G\ncache_dev=66\nhome_dev=66",
    "python3": "/usr/bin/python3 3.12.3",
    "download": "20 MB in 1.0s = 20.0 MB/s",
}


class FakeHost:
    """A transport whose host is a dict. Every command it does not know is a no-op."""

    host = "fake"

    def __init__(
        self, config: dict[str, Any] | None = None, *, home: str = f"{HOME}/.gpuc"
    ) -> None:
        self.files: dict[str, str] = {}
        self.commands: list[str] = []
        self.home = home
        if config is not None:
            self.files[f"{home}/config.json"] = json.dumps(config)

    @property
    def config(self) -> dict[str, Any] | None:
        """What this host's `config.json` holds, if it has one."""
        body = self.files.get(f"{self.home}/config.json")
        document: dict[str, Any] | None = json.loads(body) if body else None
        return document

    def run(self, command: str, *, timeout: float = 120.0, check: bool = True) -> CommandResult:
        self.commands.append(command)
        code, out = self._answer(command)
        result = CommandResult(self.host, ["sh", "-c", command], code, out, "")
        return result.check() if check else result

    def _answer(self, command: str) -> tuple[int, str]:
        if command.startswith("printf %s"):
            return 0, shlex.split(command)[2].replace("$HOME", HOME)
        if command.startswith("if [ -f") and "config.json" in command:
            body = self.files.get(f"{self.home}/config.json")
            return 0, body if body is not None else NO_CONFIG
        if "say()" in command:
            return 0, "".join(f"==={name}===\n{body}\n" for name, body in PROBE_SECTIONS.items())
        if "-m gpuc.host config --merge" in command:
            patch = json.loads(self.files[command.rsplit(" ", 1)[1]])
            merged = jobs.merged_config(self.config or {}, patch)
            self.files[f"{self.home}/config.json"] = json.dumps(merged)
            return 0, json.dumps(merged)
        if command.startswith("mv -f "):
            source, target = shlex.split(command)[2:4]
            self.files[target] = self.files.pop(source, "")
            return 0, ""
        if command.startswith("rm -f "):
            self.files.pop(shlex.split(command)[2], None)
            return 0, ""
        return 0, ""

    def put_file(self, content: str | bytes, remote_path: str, mode: int = 0o600) -> None:
        self.files[remote_path] = content.decode() if isinstance(content, bytes) else content

    def rsync(
        self,
        local_root: Path,
        remote_path: str,
        files: Sequence[str] | None = None,
        excludes: Sequence[str] = (),
    ) -> CommandResult:
        return CommandResult(self.host, ["rsync"], 0, "", "")

    def tail(self, remote_path: str, lines: int = 200, follow: bool = False) -> CommandResult:
        return CommandResult(self.host, ["tail"], 0, "", "")


@pytest.fixture
def fake_host(monkeypatch: pytest.MonkeyPatch) -> FakeHost:
    """Point `gpuc host add|set` at an in-memory host instead of ssh."""
    host = FakeHost()

    def factory(entry: Any, settings: Any = None) -> Any:
        host.home = entry.remote_home.replace("$HOME", HOME)
        return host

    for module in ("connect", "probe", "cli"):
        monkeypatch.setattr(f"gpuc.control.{module}.transport_for", factory, raising=False)
    return host
