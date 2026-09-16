from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from gpuc.control.remote import (
    HostSession,
    RemoteError,
    host_command,
    parse_last_json,
    read_remote_config,
    write_remote_config,
)
from gpuc.control.transport import CommandResult, Transport, TransportError
from tests.conftest import host_entry

MOTD = """Welcome to Ubuntu 24.04!
 * Support: https://ubuntu.com/pro
"""


class ScriptedTransport:
    host = "gpubox"

    def __init__(self, stdout: str, returncode: int = 0, raises: Exception | None = None) -> None:
        self.stdout = stdout
        self.returncode = returncode
        self.raises = raises
        self.commands: list[str] = []

    def run(self, command: str, *, timeout: float = 120.0, check: bool = True) -> CommandResult:
        self.commands.append(command)
        if self.raises is not None:
            raise self.raises
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


class Recorder(ScriptedTransport):
    """A transport that remembers what was written to it, not just what was run."""

    def __init__(self, stdout: str, returncode: int = 0) -> None:
        super().__init__(stdout, returncode)
        self.puts: dict[str, tuple[str, int]] = {}

    def put_file(self, content: str | bytes, remote_path: str, mode: int = 0o600) -> None:
        text = content.decode() if isinstance(content, bytes) else content
        self.puts[remote_path] = (text, mode)


def session(stdout: str, returncode: int = 0) -> HostSession:
    entry = host_entry(name="gpubox", kind="ssh", ssh="u@h")
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
    host = ScriptedTransport(MOTD + json.dumps(document))
    assert read_remote_config(cast("Transport", host), "/mnt/ssd/gpuc") == document
    # Read out of *this host's* gpuc home. A host with a persistent root or
    # a `--gpuc-home` keeps its config there, and reading the default path
    # would report "no config" and re-ship the package on every submit.
    assert host.commands == ['cat "/mnt/ssd/gpuc/config.json" 2>/dev/null || true']


def test_a_host_with_no_config_is_told_apart_from_one_that_could_not_be_asked() -> None:
    """`cat` swallows its own failure, so a non-zero exit is the transport's.

    The difference decides whether a host is adopted as it stands or is given
    its first config, and whether `submit` trusts what it read at all.
    """
    assert read_remote_config(cast("Transport", ScriptedTransport("")), "/h/.gpuc") == {}
    refused = ScriptedTransport("ssh: could not resolve hostname", returncode=255)
    assert read_remote_config(cast("Transport", refused), "/h/.gpuc") is None


def test_a_host_that_cannot_be_reached_at_all_is_not_an_exception_to_handle() -> None:
    """`submit` asks every host this before it enqueues, including the ones
    that are down. The error belongs to the submit behind it, in full, not to
    a traceback out of the version check."""
    down = ScriptedTransport("", raises=TransportError(CommandResult("h", ["ssh"], 124, "", "")))
    assert read_remote_config(cast("Transport", down), "/h/.gpuc") is None


def test_writing_the_config_goes_through_the_hosts_own_cli() -> None:
    """The host applies the patch: one atomic write, by the code that reads it.

    And the patch travels as a file, because `env` may hold a token and argv is
    readable by every other user of a shared box.
    """
    merged = {"host": "gpubox", "gpus": ["0"], "env": {"HF_TOKEN": "hf_secret"}}
    host = Recorder(json.dumps(merged))
    document = write_remote_config(
        cast("Transport", host), "/home/u/.gpuc", {"env": {"HF_TOKEN": "hf_secret"}}, python="py"
    )
    assert document == merged
    ((path, (body, mode)),) = host.puts.items()
    assert json.loads(body) == {"env": {"HF_TOKEN": "hf_secret"}}
    assert mode == 0o600
    assert any(f"-m gpuc.host config --merge {path}" in c for c in host.commands)
    assert any(c.startswith(f"rm -f {path}") for c in host.commands)
    assert not any("hf_secret" in c for c in host.commands)


def test_a_host_with_no_package_yet_has_its_config_written_by_rename() -> None:
    """`gpuc host add` writes the first config before anything is installed, so
    there is no host CLI to do the merge -- and a truncated `config.json` is
    something the dispatcher could read."""
    host = Recorder("")
    document = write_remote_config(
        cast("Transport", host), "/home/u/.gpuc", {"host": "gpubox", "gpus": ["0"]}
    )
    assert document["gpus"] == ["0"]
    tmp = next(path for path in host.puts if path.startswith("/home/u/.gpuc/.config.json."))
    assert json.loads(host.puts[tmp][0])["host"] == "gpubox"
    assert any(f'mv -f "{tmp}" "/home/u/.gpuc/config.json"' == c for c in host.commands)
    assert any('mkdir -p "/home/u/.gpuc" && chmod 700' in c for c in host.commands)
