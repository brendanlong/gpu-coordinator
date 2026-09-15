from __future__ import annotations

import shutil
import stat
import subprocess
from pathlib import Path

import pytest

from gpuc.control import transport
from gpuc.control.transport import LocalTransport, SshTransport, TransportError


def ssh_localhost_works() -> bool:
    if shutil.which("ssh") is None:
        return False
    proc = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", "localhost", "true"],
        capture_output=True,
        timeout=30,
        check=False,
    )
    return proc.returncode == 0


needs_ssh_localhost = pytest.mark.skipif(
    not ssh_localhost_works(), reason="`ssh localhost` needs passwordless key auth"
)


def make_ssh(tmp_path: Path) -> SshTransport:
    return SshTransport(
        host="spar",
        target="user@box",
        port=2222,
        key="/keys/id_ed25519",
        control_dir=tmp_path / "control",
        known_hosts=tmp_path / "known_hosts",
    )


def test_local_run_captures_output() -> None:
    result = LocalTransport().run("echo out; echo err >&2")
    assert result.returncode == 0
    assert "out" in result.stdout
    assert "err" in result.stderr


def test_local_run_raises_with_command_host_and_output() -> None:
    with pytest.raises(TransportError) as excinfo:
        LocalTransport(host="desktop").run("echo boom >&2; exit 5")
    message = str(excinfo.value)
    assert "on host desktop" in message
    assert "exited 5" in message
    assert "boom" in message


def test_local_run_without_check_returns_the_failure() -> None:
    result = LocalTransport().run("exit 3", check=False)
    assert result.returncode == 3


def test_local_run_timeout_is_a_transport_error() -> None:
    with pytest.raises(TransportError) as excinfo:
        LocalTransport().run("sleep 30", timeout=0.5)
    assert "timed out" in str(excinfo.value)


def test_local_put_file_is_0600(tmp_path: Path) -> None:
    target = tmp_path / "secrets" / "job.env"
    LocalTransport().put_file("AWS_SECRET_ACCESS_KEY=shh\n", str(target))
    assert target.read_text() == "AWS_SECRET_ACCESS_KEY=shh\n"
    assert stat.S_IMODE(target.stat().st_mode) == 0o600


def test_ssh_options_pin_batch_mode_timeout_and_known_hosts(tmp_path: Path) -> None:
    options = make_ssh(tmp_path).ssh_options()
    joined = " ".join(options)
    assert "BatchMode=yes" in joined
    assert "ConnectTimeout=15" in joined
    assert f"UserKnownHostsFile={tmp_path / 'known_hosts'}" in joined
    assert "StrictHostKeyChecking=accept-new" in joined
    assert "ControlMaster=auto" in joined
    assert f"ControlPath={tmp_path / 'control' / 'cm-spar'}" in joined
    assert "ControlPersist=60" in joined
    assert options[-2:] == ["-p", "2222"]
    assert "-i" in options and "/keys/id_ed25519" in options


def test_ssh_argv_puts_the_command_last(tmp_path: Path) -> None:
    argv = make_ssh(tmp_path).ssh_argv("uname -a")
    assert argv[0] == "ssh"
    assert argv[-2:] == ["user@box", "uname -a"]


def test_put_file_never_puts_the_secret_in_argv(tmp_path: Path) -> None:
    argv = make_ssh(tmp_path).put_file_argv("~/.gpuc/secrets/job.env")
    command = argv[-1]
    assert "chmod 600" in command
    assert "cat >" in command
    assert "mkdir -p" in command
    assert not any("secret" in part.lower() for part in argv[:-1])


def test_rsync_argv_for_a_file_list(tmp_path: Path) -> None:
    ssh = make_ssh(tmp_path)
    argv = transport.rsync_argv(
        tmp_path,
        f"{ssh.target}:~/.gpuc/jobs/j1/workdir",
        ["a.py", "b/c.py"],
        ssh.rsync_ssh_command(),
    )
    assert argv[:2] == ["rsync", "-a"]
    assert "--files-from=-" in argv
    assert "--delete-after" not in argv
    assert argv[argv.index("-e") + 1].startswith("ssh ")
    assert argv[-1] == "user@box:~/.gpuc/jobs/j1/workdir"


def test_rsync_argv_for_a_whole_directory(tmp_path: Path) -> None:
    argv = transport.rsync_argv(tmp_path, "/dest", None, None)
    assert "--delete-after" in argv
    assert "--files-from=-" not in argv
    assert "-e" not in argv


def test_local_rsync_copies_only_the_listed_files(tmp_path: Path) -> None:
    src = tmp_path / "src"
    (src / "sub").mkdir(parents=True)
    (src / "keep.py").write_text("keep")
    (src / "sub" / "also.py").write_text("also")
    (src / "ignored.bin").write_text("ignored")
    dest = tmp_path / "dest"
    LocalTransport().rsync(src, str(dest), ["keep.py", "sub/also.py"])
    assert (dest / "keep.py").read_text() == "keep"
    assert (dest / "sub" / "also.py").read_text() == "also"
    assert not (dest / "ignored.bin").exists()


def test_local_tail_returns_the_last_lines(tmp_path: Path) -> None:
    log = tmp_path / "log.txt"
    log.write_text("\n".join(str(i) for i in range(100)) + "\n")
    result = LocalTransport().tail(str(log), lines=3)
    assert result.stdout.split() == ["97", "98", "99"]


def test_tail_of_a_missing_file_does_not_raise(tmp_path: Path) -> None:
    result = LocalTransport().tail(str(tmp_path / "nope"), lines=5)
    assert result.returncode != 0


def test_git_tracked_files_lists_the_repo() -> None:
    files = transport.git_tracked_files(Path(__file__).resolve().parents[1])
    assert "pyproject.toml" in files
    assert "docs/ARCHITECTURE.md" in files


def test_make_transport_picks_the_right_kind(tmp_path: Path) -> None:
    assert isinstance(transport.make_transport("local"), LocalTransport)
    remote = transport.make_transport("spar", ssh="u@h", port=2200, state_dir=tmp_path)
    assert isinstance(remote, SshTransport)
    assert remote.control_dir == tmp_path / "control"
    assert remote.known_hosts == tmp_path / "known_hosts"


def test_a_missing_binary_is_a_transport_error(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(TransportError) as excinfo:
        transport._execute("h", ["definitely-not-a-binary"], timeout=5, check=True)
    assert "not found" in str(excinfo.value)


@needs_ssh_localhost
def test_ssh_transport_against_localhost(tmp_path: Path) -> None:
    ssh = SshTransport(
        host="localhost",
        target="localhost",
        control_dir=tmp_path / "control",
        known_hosts=tmp_path / "known_hosts",
    )
    assert "hello" in ssh.run("echo hello").stdout
    target = tmp_path / "secret.env"
    ssh.put_file("TOKEN=abc\n", str(target))
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    with pytest.raises(TransportError):
        ssh.run("exit 9")
