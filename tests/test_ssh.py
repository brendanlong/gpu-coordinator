"""`gpuc ssh`: the same connection options as everything else, by hand."""

from __future__ import annotations

import os
from pathlib import Path
from typing import cast

import pytest

from gpuc.control import ssh as ssh_mod
from gpuc.control.cli import EXIT_NOT_FOUND, EXIT_OK, main
from gpuc.control.transport import CommandResult, SshTransport, Transport

GPU = "GPU-2a4bad3b-9fe3-7031-914d-384254e92908"
JOB = "20260915-120000-abc123"


class RecordingTransport:
    """A transport that runs nothing and remembers what it was asked to run."""

    host = "spar"

    def __init__(self, returncode: int = 0) -> None:
        self.commands: list[str] = []
        self.returncode = returncode

    def run(self, command: str, *, timeout: float = 120.0, check: bool = True) -> CommandResult:
        self.commands.append(command)
        return CommandResult(self.host, ["ssh", command], self.returncode, "out\n", "err\n")

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


def ssh_transport(tmp_path: Path) -> SshTransport:
    # A short control dir: the socket path has to fit in sun_path, and pytest's
    # tmp_path does not (which control_path() refuses, loudly, by design).
    control = Path("/tmp") / f"gpuc-test-{os.getpid()}"
    return SshTransport(
        host="spar",
        target="me@box",
        port=2222,
        key=str(tmp_path / "id_ed25519"),
        control_dir=control,
        known_hosts=tmp_path / "known_hosts",
    )


def test_the_interactive_argv_carries_the_transports_own_options(tmp_path: Path) -> None:
    argv = ssh_mod.interactive_argv(ssh_transport(tmp_path), "$HOME/.gpuc")
    assert argv[0] == "ssh"
    assert "-t" in argv
    assert argv[-2] == "me@box"
    assert "2222" in argv
    assert str(tmp_path / "known_hosts") in " ".join(argv)
    assert "ControlMaster=auto" in argv
    # An interactive session may need to ask for a passphrase.
    assert not any(option.startswith("BatchMode") for option in argv)
    assert argv[-1] == 'cd "$HOME/.gpuc"; exec "${SHELL:-/bin/bash}" -l'


def test_a_job_target_lands_in_the_workdir_and_falls_back_to_the_job_dir() -> None:
    command = ssh_mod.login_command(f"$HOME/.gpuc/jobs/{JOB}/workdir", f"$HOME/.gpuc/jobs/{JOB}")
    assert command.startswith(f'cd "$HOME/.gpuc/jobs/{JOB}/workdir" 2>/dev/null || ')
    assert f'cd "$HOME/.gpuc/jobs/{JOB}"' in command
    assert command.endswith('exec "${SHELL:-/bin/bash}" -l')


def test_one_command_runs_in_that_directory_and_propagates_the_exit_code() -> None:
    transport = cast("Transport", RecordingTransport(returncode=7))
    result = ssh_mod.run_command(transport, "$HOME/.gpuc", "ls -la")
    assert result.returncode == 7
    assert cast("RecordingTransport", transport).commands == ['cd "$HOME/.gpuc" && ls -la']


def test_a_local_host_gets_a_shell_not_an_ssh(tmp_path: Path) -> None:
    from gpuc.control.transport import LocalTransport

    argv = ssh_mod.interactive_argv(LocalTransport(), str(tmp_path))
    assert argv[0] == "bash"
    assert str(tmp_path) in argv[-1]
    assert ssh_mod.local_directory(str(tmp_path), None) == str(tmp_path)
    assert ssh_mod.local_directory(str(tmp_path / "gone"), str(tmp_path)) == str(tmp_path)


def test_print_shows_a_copyable_command_line(
    control_env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    main(["host", "add", "spar", "--ssh", "me@box", "--port", "2222", "--gpus", GPU])
    capsys.readouterr()
    assert main(["ssh", "spar", "--print"]) == EXIT_OK
    line = capsys.readouterr().out.strip()
    assert line.startswith("ssh ")
    assert "me@box" in line
    assert "-p 2222" in line

    assert main(["ssh", "spar", "--print", "--", "nvidia-smi"]) == EXIT_OK
    with_command = capsys.readouterr().out.strip()
    assert "nvidia-smi" in with_command


def test_a_command_on_the_local_host_really_runs(
    control_env: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The live check, on `local` only: a real LocalTransport, a real shell."""
    home = tmp_path / "gpuc-home"
    (home / "jobs").mkdir(parents=True)
    main(["host", "add", "local", "--gpuc-home", str(home), "--gpus", GPU])
    capsys.readouterr()
    assert main(["ssh", "local", "--", "pwd"]) == EXIT_OK
    assert capsys.readouterr().out.strip() == str(home)
    assert main(["ssh", "local", "--", "exit", "3"]) == 3


def test_an_unknown_target_is_exit_four(
    control_env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["ssh", "nope"]) == EXIT_NOT_FOUND
    assert "no registered host knows job" in capsys.readouterr().err
