from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from gpuc.control.remote import (
    NO_CONFIG,
    HostConfigRead,
    HostSession,
    RemoteError,
    host_command,
    host_python,
    parse_last_json,
    read_config,
    reason_of,
    usable_python,
    write_config,
)
from gpuc.control.transport import CommandResult, Transport, TransportError
from tests.conftest import host_entry

MOTD = """Welcome to Ubuntu 24.04!
 * Support: https://ubuntu.com/pro
"""


class ScriptedTransport:
    host = "gpubox"

    def __init__(
        self,
        stdout: str,
        returncode: int = 0,
        raises: Exception | None = None,
        stderr: str = "",
    ) -> None:
        self.stdout = stdout
        self.returncode = returncode
        self.raises = raises
        self.stderr = stderr
        self.commands: list[str] = []

    def run(self, command: str, *, timeout: float = 120.0, check: bool = True) -> CommandResult:
        self.commands.append(command)
        if self.raises is not None:
            raise self.raises
        return CommandResult(self.host, ["ssh", command], self.returncode, self.stdout, self.stderr)

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


def session(stdout: str, returncode: int = 0, stderr: str = "") -> HostSession:
    entry = host_entry(name="gpubox", kind="ssh", ssh="u@h")
    transport: Transport = cast("Transport", ScriptedTransport(stdout, returncode, stderr=stderr))
    return HostSession(entry, transport, "/home/u/.gpuc", "python3", HostConfigRead({}))


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


def test_every_host_invocation_starts_with_the_one_python_prefix() -> None:
    """`-m gpuc.host` and bootstrap's `-c` snippets share the prefix, so the
    host env and the pinned home cannot drift apart between them."""
    env = {"UV_CACHE_DIR": "/vol/uv", "HF_HOME": "/big"}
    prefix = host_python("/py", "/home/u/.gpuc", env)
    assert prefix == (
        'HF_HOME="/big" UV_CACHE_DIR="/vol/uv" '
        'GPUC_HOME="/home/u/.gpuc" PYTHONPATH="/home/u/.gpuc/pkg" "/py"'
    )
    assert host_command("/py", "/home/u/.gpuc", "status", env) == f"{prefix} -m gpuc.host status"


def test_a_pretty_printed_report_is_not_mistaken_for_its_last_nested_object() -> None:
    report = {"host": "gpubox", "ok": True, "checks": [{"name": "driver"}, {"name": "disk"}]}
    document = parse_last_json(MOTD + json.dumps(report, indent=2) + "\n")
    assert document == report


def test_the_last_of_two_documents_wins() -> None:
    assert parse_last_json('{"first": 1}\n{"second": 2}\n') == {"second": 2}


SSH_STDERR = """\
Warning: Permanently added '[1.2.3.4]:22000' (ED25519) to the list of known hosts.
root@1.2.3.4: Permission denied (publickey).
"""


def test_the_reason_a_host_could_not_be_asked_is_what_ssh_said_last() -> None:
    """A `TransportError`'s first line is the argv dump; what a person needs
    -- refused, denied, key changed -- is the last thing ssh wrote."""
    refused = TransportError(
        CommandResult("gpubox", ["ssh", "-p", "22000", "u@h"], 255, "", SSH_STDERR)
    )
    assert reason_of(refused) == "root@1.2.3.4: Permission denied (publickey)."
    # Wrapped once on the way out of `resolve_home`: the cause still carries it.
    wrapped = RemoteError("gpubox", "printf", f"could not reach host gpubox: {refused}")
    wrapped.__cause__ = refused
    assert reason_of(wrapped) == "root@1.2.3.4: Permission denied (publickey)."
    # No stderr to quote: the first line of the message, never an empty string.
    assert reason_of(RemoteError("gpubox", "status", "expected JSON on stdout, got:\nbanner")) == (
        "expected JSON on stdout, got:"
    )
    assert reason_of(TransportError(CommandResult("h", ["ssh"], 124, "", "  \n"))).startswith(
        "`ssh` on host h exited 124"
    )
    assert reason_of(OSError()) == "OSError"


def test_ask_reports_the_reason_ssh_gave_not_the_argv(monkeypatch: pytest.MonkeyPatch) -> None:
    from gpuc.control.remote import Unreachable, ask

    def down(*_: object, **__: object) -> object:
        raise TransportError(CommandResult("gpubox", ["ssh", "u@h", "printf"], 255, "", SSH_STDERR))

    monkeypatch.setattr("gpuc.control.remote.open_session", down)
    asked = ask(host_entry(name="gpubox", kind="ssh", ssh="u@h"), "status")
    assert isinstance(asked, Unreachable)
    assert asked.reason == "root@1.2.3.4: Permission denied (publickey)."


def test_the_remote_config_is_read_from_the_host_not_the_registry() -> None:
    document = {"host": "gpubox", "gpus": ["0"], "pkg_commit": "c" * 40}
    host = ScriptedTransport(MOTD + json.dumps(document))
    assert read_config(cast("Transport", host), "/mnt/ssd/gpuc").document == document
    # Read out of *this host's* gpuc home. A host with a persistent root or
    # a `--gpuc-home` keeps its config there, and reading the default path
    # would report "no config" and re-ship the package on every submit.
    assert host.commands == [
        'if [ -f "/mnt/ssd/gpuc/config.json" ]; then cat "/mnt/ssd/gpuc/config.json"; '
        f"else echo {NO_CONFIG}; fi"
    ]


def test_a_host_with_no_config_is_told_apart_from_one_that_could_not_be_asked() -> None:
    """The difference decides whether a host is adopted as it stands or is
    given its first config, so "there is none" is a marker the host prints and
    never the absence of parseable output: a half-written `config.json` is a
    file somebody's host is running on, and replacing it is exactly the drift
    this model exists to stop.
    """
    no_config = ScriptedTransport(MOTD + NO_CONFIG + "\n")
    assert read_config(cast("Transport", no_config), "/h/.gpuc").missing
    for unreadable in (ScriptedTransport(MOTD), ScriptedTransport('{"host": "gpub')):
        read = read_config(cast("Transport", unreadable), "/h/.gpuc")
        assert read.unreadable and not read.missing and read.document is None
    refused = ScriptedTransport("ssh: could not resolve hostname", returncode=255)
    assert read_config(cast("Transport", refused), "/h/.gpuc").unreadable


def test_a_config_that_could_not_be_read_is_never_written_over() -> None:
    """The fallback merge has to know what it is merging into. A host that did
    not answer, or answered with something unparseable, is not a host with no
    config -- and writing the patch alone would drop everything else."""
    host = Recorder('{"host": "gpub')
    with pytest.raises(RemoteError) as caught:
        write_config(cast("Transport", host), "/home/u/.gpuc", {"gpus": ["0"]}, host="gpubox")
    assert "left alone" in str(caught.value)
    assert host.puts == {}


def test_a_host_that_cannot_be_reached_at_all_is_not_an_exception_to_handle() -> None:
    """`submit` asks every host this before it enqueues, including the ones
    that are down. The error belongs to the submit behind it, in full, not to
    a traceback out of the version check."""
    down = ScriptedTransport("", raises=TransportError(CommandResult("h", ["ssh"], 124, "", "")))
    assert read_config(cast("Transport", down), "/h/.gpuc").unreadable


def test_writing_the_config_merges_here_and_never_puts_a_secret_in_argv() -> None:
    """The client is the one writer: read, merge by the host's own rule, replace
    by rename. The patch travels as a file, because `env` may hold a token and
    argv is readable by every other user of a shared box.
    """
    existing = {"host": "gpubox", "gpus": ["0"], "idle_minutes": 30.0}
    host = Recorder(json.dumps(existing))
    document = write_config(
        cast("Transport", host), "/home/u/.gpuc", {"env": {"HF_TOKEN": "hf_secret"}}, host="gpubox"
    )
    assert document["env"] == {"HF_TOKEN": "hf_secret"}
    assert document["idle_minutes"] == 30.0 and document["gpus"] == ["0"]
    ((path, (body, _mode)),) = host.puts.items()
    assert path.startswith("/home/u/.gpuc/.config.json.")
    assert json.loads(body)["env"] == {"HF_TOKEN": "hf_secret"}
    assert not any("hf_secret" in c for c in host.commands)


def test_a_host_with_no_config_has_its_first_one_written_by_rename() -> None:
    """`gpuc host add` writes the first config before anything is installed --
    and a truncated `config.json` is something the dispatcher could read."""
    host = Recorder(NO_CONFIG + "\n")
    document = write_config(
        cast("Transport", host), "/home/u/.gpuc", {"host": "gpubox", "gpus": ["0"]}, host="gpubox"
    )
    assert document["gpus"] == ["0"]
    tmp = next(path for path in host.puts if path.startswith("/home/u/.gpuc/.config.json."))
    assert json.loads(host.puts[tmp][0])["host"] == "gpubox"
    assert any(f"mv -f {tmp} /home/u/.gpuc/config.json" == c for c in host.commands)
    assert any('mkdir -p "/home/u/.gpuc"; chmod 700' in c for c in host.commands)


def test_usable_python_takes_uvs_answer_or_a_new_enough_python3() -> None:
    """The probe prints uv's interpreter as a bare path (uv applied the floor)
    and `python3` as `path version`; an old python3 alone is no answer."""
    assert usable_python(
        "/root/.local/share/uv/python/bin/python3.12 3.12.4\n/usr/bin/python3 3.8.10\n"
    ) == ("/root/.local/share/uv/python/bin/python3.12")
    assert usable_python("/usr/bin/python3 3.11.4\n") == "/usr/bin/python3"
    assert usable_python("/usr/bin/python3 3.8.10\n") is None
    assert usable_python("") is None


def test_a_reused_session_that_fails_still_says_what_ssh_said() -> None:
    """`wait` polls and the job verbs go on using the session the locator
    opened, so its failures are the ones most people see; a reason made of
    the argv and an exit code names nothing."""
    with pytest.raises(RemoteError) as caught:
        session("", returncode=255, stderr=SSH_STDERR).host_cli("status")
    assert reason_of(caught.value) == "root@1.2.3.4: Permission denied (publickey)."
