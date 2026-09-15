from __future__ import annotations

import os
import shlex
import shutil
import stat
import subprocess
import tempfile
from pathlib import Path

import pytest

from gpuc.control import transport
from gpuc.control.transport import LocalTransport, SshTransport, SshUnusable, TransportError

# A real ControlPath has to fit in sun_path, so the fixtures use a short one;
# pytest's own tmp_path is deliberately too long (see the control-path tests).
# The pid keeps two concurrent runs (or two users) off one socket directory
# while staying far inside the 108-byte limit.
SHORT_CONTROL_DIR = Path(tempfile.gettempdir()) / f"gpuc-test-cm-{os.getpid()}"


@pytest.fixture(scope="session")
def ssh_localhost() -> None:
    """Skip unless `ssh localhost` works without a password.

    A fixture rather than a module-level `skipif`: the old form ran a real ssh
    (and waited on its 5s connect timeout) during *collection*, on every run of
    the suite, whether or not anything in this file was selected.
    """
    if shutil.which("ssh") is None:
        pytest.skip("no ssh binary")
    proc = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", "localhost", "true"],
        capture_output=True,
        timeout=30,
        check=False,
    )
    if proc.returncode != 0:
        pytest.skip("`ssh localhost` needs passwordless key auth")


def make_ssh(tmp_path: Path) -> SshTransport:
    return SshTransport(
        host="spar",
        target="user@box",
        port=2222,
        key="/keys/id_ed25519",
        control_dir=SHORT_CONTROL_DIR,
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
    assert f"ControlPath={SHORT_CONTROL_DIR / 'cm-%C'}" in joined
    assert "ControlPersist=60" in joined
    assert options[-2:] == ["-p", "2222"]
    assert "-i" in options and "/keys/id_ed25519" in options


def test_ssh_argv_runs_a_non_login_bash_with_the_command_last(tmp_path: Path) -> None:
    argv = make_ssh(tmp_path).ssh_argv("uname -a")
    assert argv[0] == "ssh"
    assert argv[-2] == "user@box"
    assert argv[-1] == "bash -c 'uname -a'"


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
    assert "--from0" in argv
    assert "--ignore-missing-args" in argv
    assert "--delete-after" not in argv
    assert argv[argv.index("-e") + 1].startswith("ssh ")
    assert argv[-1] == "user@box:~/.gpuc/jobs/j1/workdir"


def test_the_file_list_is_nul_separated(tmp_path: Path) -> None:
    assert transport._files_stdin(["a.py", "weird\nname.py"]) == b"a.py\0weird\nname.py\0"
    assert transport._files_stdin(None) is None


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
    assert remote.control_dir == transport.control_socket_dir()
    assert remote.known_hosts == tmp_path / "known_hosts"


def test_a_missing_binary_is_a_transport_error(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(TransportError) as excinfo:
        transport._execute("h", ["definitely-not-a-binary"], timeout=5, check=True)
    assert "not found" in str(excinfo.value)


def test_ssh_transport_against_localhost(ssh_localhost: None, tmp_path: Path) -> None:
    ssh = SshTransport(
        host="localhost",
        target="localhost",
        control_dir=SHORT_CONTROL_DIR,
        known_hosts=tmp_path / "known_hosts",
    )
    assert "hello" in ssh.run("echo hello").stdout
    target = tmp_path / "secret.env"
    ssh.put_file("TOKEN=abc\n", str(target))
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    with pytest.raises(TransportError):
        ssh.run("exit 9")


def test_put_file_creates_the_file_with_umask_077_and_quotes_the_dirname(tmp_path: Path) -> None:
    command = make_ssh(tmp_path).put_file_argv("/home/u/.gpuc/secrets/j1.env")[-1]
    assert "umask 077" in command
    assert '"$(dirname' in command
    assert command.index("umask 077") < command.index("chmod 600")


def test_local_run_uses_a_non_login_shell() -> None:
    result = LocalTransport().run("shopt -q login_shell && echo login || echo non-login")
    assert "non-login" in result.stdout


def test_rsync_ssh_command_quotes_a_key_path_with_a_space(tmp_path: Path) -> None:
    key = tmp_path / "my keys" / "id_ed25519"
    key.parent.mkdir()
    key.touch()
    ssh = SshTransport(host="spar", target="user@box", key=str(key))
    command = ssh.rsync_ssh_command()
    assert f"'{key}'" in command
    assert shlex.split(command) == ["ssh", *ssh.ssh_options()]


def test_rsync_actually_runs_a_remote_shell_whose_path_contains_a_space(tmp_path: Path) -> None:
    """rsync splits -e itself, so the quoting has to survive *its* parser too."""
    wrapper = tmp_path / "a dir with spaces" / "fake-ssh"
    wrapper.parent.mkdir()
    wrapper.write_text('#!/bin/bash\nshift\nexec "$@"\n')
    wrapper.chmod(0o755)
    src = tmp_path / "src"
    src.mkdir()
    (src / "f.txt").write_text("payload")
    dest = tmp_path / "dest"
    dest.mkdir()
    argv = transport.rsync_argv(src, f"fakehost:{dest}", ["f.txt"], shlex.join([str(wrapper)]))
    transport._execute(
        "spar", argv, timeout=60, check=True, stdin=transport._files_stdin(["f.txt"])
    )
    assert (dest / "f.txt").read_text() == "payload"


def test_git_tracked_files_asks_for_nul_separated_unquoted_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen: list[list[str]] = []
    (tmp_path / "a.py").write_text("")
    (tmp_path / "dir").mkdir()
    (tmp_path / "dir" / "b with space.py").write_text("")

    class Result:
        returncode = 0
        stdout = b"a.py\0dir/b with space.py\0"
        stderr = b""

    def fake_run(argv: list[str], **_: object) -> Result:
        seen.append(argv)
        return Result()

    monkeypatch.setattr(transport.subprocess, "run", fake_run)
    assert transport.git_tracked_files(tmp_path) == ["a.py", "dir/b with space.py"]
    assert "core.quotePath=false" in seen[0]
    assert seen[0][-4:] == ["-z", "--cached", "--others", "--exclude-standard"]


def repo_with(root: Path, files: dict[str, str], commit: list[str]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for name, body in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body)
    for argv in (
        ["git", "init", "-q"],
        ["git", "config", "user.email", "t@example.com"],
        ["git", "config", "user.name", "t"],
        ["git", "add", *commit],
        ["git", "commit", "-qm", "init"],
    ):
        subprocess.run(argv, cwd=root, check=True, capture_output=True)
    return root


def test_untracked_files_are_synced_and_ignored_ones_are_not(tmp_path: Path) -> None:
    root = repo_with(
        tmp_path / "r",
        {"tracked.py": "1\n", "fresh.py": "2\n", ".gitignore": "junk\n", "junk": "3\n"},
        ["tracked.py", ".gitignore"],
    )
    assert sorted(transport.git_tracked_files(root)) == [".gitignore", "fresh.py", "tracked.py"]


def test_a_deleted_but_cached_file_is_dropped(tmp_path: Path) -> None:
    root = repo_with(tmp_path / "r", {"a.py": "1\n", "b.py": "2\n"}, ["a.py", "b.py"])
    (root / "b.py").unlink()
    assert transport.git_tracked_files(root) == ["a.py"]


def test_git_summary_counts_modified_and_untracked(tmp_path: Path) -> None:
    root = repo_with(tmp_path / "r", {"a.py": "1\n", "b.py": "2\n"}, ["a.py", "b.py"])
    (root / "a.py").write_text("changed\n")
    (root / "new.py").write_text("new\n")
    summary = transport.git_summary(root)
    assert (len(summary.files), summary.modified, summary.untracked) == (3, 1, 1)
    assert summary.render() == "syncing 3 files (1 modified, 1 untracked, ignoring .gitignore'd)"


def test_the_patch_carries_untracked_files_without_touching_the_real_index(
    tmp_path: Path,
) -> None:
    root = repo_with(tmp_path / "r", {"a.py": "old\n"}, ["a.py"])
    (root / "a.py").write_text("new\n")
    (root / "extra.py").write_text("brand new\n")
    patch = transport.uncommitted_patch(root)
    assert "brand new" in patch and "new\n" in patch
    staged = subprocess.run(
        ["git", "diff", "--cached", "--name-only"], cwd=root, capture_output=True, text=True
    )
    assert staged.stdout.strip() == ""


def test_the_patch_is_only_the_directory_being_submitted(tmp_path: Path) -> None:
    """A submit from a subdirectory syncs that subdirectory, so the patch that
    goes with it must not carry the rest of the repository."""
    root = repo_with(
        tmp_path / "r", {"sub/a.py": "old\n", "elsewhere.py": "old\n"}, ["sub/a.py", "elsewhere.py"]
    )
    (root / "sub/a.py").write_text("changed here\n")
    (root / "elsewhere.py").write_text("changed there\n")
    patch = transport.uncommitted_patch(root / "sub")
    assert "changed here" in patch
    assert "changed there" not in patch


def test_the_patch_includes_staged_changes(tmp_path: Path) -> None:
    root = repo_with(tmp_path / "r", {"a.py": "old\n"}, ["a.py"])
    (root / "a.py").write_text("staged\n")
    subprocess.run(["git", "add", "a.py"], cwd=root, check=True, capture_output=True)
    assert "staged" in transport.uncommitted_patch(root)


def test_both_transports_build_the_same_tail(tmp_path: Path) -> None:
    assert transport.tail_command("/a b/log.txt", 20) == "tail -n 20 '/a b/log.txt'"
    assert transport.tail_command("/log.txt", 5, follow=True) == "tail -f -n 5 /log.txt"


def test_rsync_excludes_reach_the_command_line(tmp_path: Path) -> None:
    argv = transport.rsync_argv(tmp_path, "host:/dest", None, None, excludes=(".venv", "*.pyc"))
    assert argv[:7] == [
        "rsync",
        "-a",
        "--exclude",
        ".venv",
        "--exclude",
        "*.pyc",
        "--delete-after",
    ]


def test_make_transport_takes_a_per_pod_known_hosts_file(tmp_path: Path) -> None:
    per_pod = tmp_path / "pods" / "pod-1.known_hosts"
    remote = transport.make_transport(
        "pod-1", ssh="root@1.2.3.4", state_dir=tmp_path, known_hosts=per_pod
    )
    assert isinstance(remote, SshTransport)
    assert remote.known_hosts == per_pod
    assert remote.control_dir == transport.control_socket_dir()


def test_control_socket_dir_prefers_xdg_runtime_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_RUNTIME_DIR", "/run/user/4242")
    assert transport.control_socket_dir() == Path("/run/user/4242/gpuc")


def test_control_socket_dir_falls_back_to_a_per_uid_tmp_dir(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
    import os

    assert transport.control_socket_dir() == Path(f"/tmp/gpuc-{os.getuid()}")


def test_the_control_socket_dir_is_0700(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    ssh = SshTransport(host="h", target="u@h", control_dir=transport.control_socket_dir())
    ssh._prepare()
    assert stat.S_IMODE((tmp_path / "gpuc").stat().st_mode) == 0o700


def test_a_control_path_that_would_overrun_sun_path_fails_before_ssh_runs() -> None:
    long_dir = Path("/tmp") / ("x" * 90)
    with pytest.raises(SshUnusable) as excinfo:
        transport.control_path(long_dir)
    message = str(excinfo.value)
    assert "ControlMaster socket path" in message
    assert "XDG_RUNTIME_DIR" in message


def test_the_real_control_path_fits_in_a_unix_socket() -> None:
    path = transport.control_path(transport.control_socket_dir())
    assert len(path.encode()) - len("%C") + transport.CONTROL_HASH_LEN < 100


def test_an_ssh_socket_failure_raises_immediately_even_without_check(
    tmp_path: Path,
) -> None:
    """The 15-minute silent retry loop: every poll treats non-zero as `not up
    yet`, so this class of failure has to be an exception, not a return code."""
    script = tmp_path / "fake-ssh"
    script.write_text(
        "#!/bin/bash\n"
        'echo "unix_listener: path "/x" too long for Unix domain socket" >&2\nexit 255\n'
    )
    script.chmod(0o755)
    with pytest.raises(SshUnusable) as excinfo:
        transport._execute("pod-1", [str(script)], timeout=10, check=False)
    assert "no retry can succeed" in str(excinfo.value)


def test_an_ordinary_ssh_failure_still_honours_check_false(tmp_path: Path) -> None:
    result = LocalTransport().run("echo connection refused >&2; exit 255", check=False)
    assert result.returncode == 255
