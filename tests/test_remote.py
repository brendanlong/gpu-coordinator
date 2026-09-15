from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from gpuc.control.config import HostEntry
from gpuc.control.remote import HostSession, RemoteError, host_command, parse_last_json
from gpuc.control.transport import CommandResult, Transport

MOTD = """Welcome to Ubuntu 24.04!
 * Support: https://ubuntu.com/pro
"""


class ScriptedTransport:
    host = "spar"

    def __init__(self, stdout: str, returncode: int = 0) -> None:
        self.stdout = stdout
        self.returncode = returncode

    def run(self, command: str, *, timeout: float = 120.0, check: bool = True) -> CommandResult:
        return CommandResult(self.host, ["ssh", command], self.returncode, self.stdout, "")

    def put_file(self, content: str | bytes, remote_path: str, mode: int = 0o600) -> None:
        raise AssertionError("not used")

    def rsync(self, local_root: Path, remote_path: str, files: object = None) -> CommandResult:
        raise AssertionError("not used")

    def tail(self, remote_path: str, lines: int = 200, follow: bool = False) -> CommandResult:
        raise AssertionError("not used")


def session(stdout: str, returncode: int = 0) -> HostSession:
    entry = HostEntry(name="spar", kind="ssh", ssh="u@h")
    transport: Transport = cast("Transport", ScriptedTransport(stdout, returncode))
    return HostSession(entry, transport, "/home/u/.gpuc", "python3")


def test_the_last_json_line_wins_over_shell_noise() -> None:
    assert parse_last_json(MOTD + '{"host": "spar"}\n') == {"host": "spar"}


def test_a_pretty_printed_document_still_parses() -> None:
    assert parse_last_json(MOTD + '{\n  "ok": true\n}\n') == {"ok": True}


def test_noise_after_the_json_does_not_hide_an_earlier_document() -> None:
    assert parse_last_json('{"a": 1}\nbye from .bashrc\n') == {"a": 1}


def test_host_json_reads_the_last_document(tmp_path: Path) -> None:
    payload = session(MOTD + '{"job_id": "j1", "dispatcher_pid": 42}').host_json("enqueue -")
    assert payload == {"job_id": "j1", "dispatcher_pid": 42}


def test_host_json_without_any_json_says_what_it_got() -> None:
    with pytest.raises(RemoteError) as excinfo:
        session("command not found: python3\n").host_json("status")
    assert "expected JSON on stdout" in str(excinfo.value)
    assert "command not found" in str(excinfo.value)


def test_host_command_pins_gpuc_home_and_pythonpath() -> None:
    command = host_command("/py", "/home/u/.gpuc", "status")
    assert 'GPUC_HOME="/home/u/.gpuc"' in command
    assert 'PYTHONPATH="/home/u/.gpuc/pkg"' in command


def test_a_pretty_printed_report_is_not_mistaken_for_its_last_nested_object() -> None:
    report = {"host": "spar", "ok": True, "checks": [{"name": "driver"}, {"name": "disk"}]}
    document = parse_last_json(MOTD + json.dumps(report, indent=2) + "\n")
    assert document == report


def test_the_last_of_two_documents_wins() -> None:
    assert parse_last_json('{"first": 1}\n{"second": 2}\n') == {"second": 2}
