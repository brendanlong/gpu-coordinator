from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from gpuc.control.config import HostEntry
from gpuc.control.remote import (
    HostSession,
    RemoteError,
    host_command,
    parse_last_json,
    read_remote_config,
)
from gpuc.control.transport import CommandResult, Transport

MOTD = """Welcome to Ubuntu 24.04!
 * Support: https://ubuntu.com/pro
"""


class ScriptedTransport:
    host = "gpubox"

    def __init__(self, stdout: str, returncode: int = 0) -> None:
        self.stdout = stdout
        self.returncode = returncode

    def run(self, command: str, *, timeout: float = 120.0, check: bool = True) -> CommandResult:
        return CommandResult(self.host, ["ssh", command], self.returncode, self.stdout, "")

    def put_file(self, content: str | bytes, remote_path: str, mode: int = 0o600) -> None:
        raise AssertionError("not used")

    def rsync(
        self,
        local_root: Path,
        remote_path: str,
        files: object = None,
        excludes: object = (),
    ) -> CommandResult:
        raise AssertionError("not used")

    def tail(self, remote_path: str, lines: int = 200, follow: bool = False) -> CommandResult:
        raise AssertionError("not used")


def session(stdout: str, returncode: int = 0) -> HostSession:
    entry = HostEntry(name="gpubox", kind="ssh", ssh="u@h")
    transport: Transport = cast("Transport", ScriptedTransport(stdout, returncode))
    return HostSession(entry, transport, "/home/u/.gpuc", "python3")


def test_the_last_json_line_wins_over_shell_noise() -> None:
    assert parse_last_json(MOTD + '{"host": "gpubox"}\n') == {"host": "gpubox"}


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


def test_host_json_can_keep_a_report_from_a_host_that_exited_one() -> None:
    """`clean` and `purge` exit 1 *with* the account of what they deleted."""
    report = '{"freed_bytes": 0, "errors": ["a: no job with that id on this host"]}'
    assert session(report, returncode=1).host_json("purge --only a", check=False) == json.loads(
        report
    )


def test_host_json_that_kept_going_still_fails_when_there_is_no_document() -> None:
    with pytest.raises(RemoteError) as excinfo:
        session("Traceback (most recent call last):\n", returncode=1).host_json(
            "purge", check=False
        )
    assert "exited 1 and expected JSON on stdout" in str(excinfo.value)


def test_host_command_pins_gpuc_home_and_pythonpath() -> None:
    command = host_command("/py", "/home/u/.gpuc", "status")
    assert 'GPUC_HOME="/home/u/.gpuc"' in command
    assert 'PYTHONPATH="/home/u/.gpuc/pkg"' in command


def test_a_pretty_printed_report_is_not_mistaken_for_its_last_nested_object() -> None:
    report = {"host": "gpubox", "ok": True, "checks": [{"name": "driver"}, {"name": "disk"}]}
    document = parse_last_json(MOTD + json.dumps(report, indent=2) + "\n")
    assert document == report


def test_the_last_of_two_documents_wins() -> None:
    assert parse_last_json('{"first": 1}\n{"second": 2}\n') == {"second": 2}


def test_the_remote_config_is_read_from_the_host_not_the_registry() -> None:
    document = {"host": "gpubox", "gpus": ["0"], "pkg_commit": "c" * 40}
    assert read_remote_config(session(MOTD + json.dumps(document))) == document


def test_a_host_with_no_config_is_told_apart_from_one_that_could_not_be_asked() -> None:
    """`cat` swallows its own failure, so a non-zero exit is the transport's.

    The difference decides whether `submit` re-ships the package (a host with
    no config was never bootstrapped) or leaves this machine's record alone.
    """
    assert read_remote_config(session("")) == {}
    assert read_remote_config(session("ssh: could not resolve hostname", returncode=255)) is None
