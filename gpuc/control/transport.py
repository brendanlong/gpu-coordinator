"""How the control side reaches a host: locally, or over ssh/rsync.

Both transports present the same surface so the rest of the control side never
branches on host kind. Errors carry the command, the host, and the tail of the
output, because "it failed" on a remote host is otherwise undebuggable.
"""

from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

DEFAULT_TIMEOUT_S = 120.0
CONNECT_TIMEOUT_S = 15

# sun_path is 108 bytes; ssh expands %C to 40 hex characters. 100 leaves room
# for the ".<pid>" suffix ssh appends while the master is being set up.
CONTROL_PATH_MAX = 100
CONTROL_HASH_LEN = 40

SSH_SOCKET_FATAL = re.compile(r"ControlPath too long|unix_listener", re.IGNORECASE)
"""ssh could not bind its ControlMaster socket. No amount of waiting fixes a
path that does not fit in sun_path, and a caller that retries anyway spends its
whole budget in silence -- which is exactly how one provisioning run burnt a
15-minute ceiling on nothing."""


@dataclass
class CommandResult:
    host: str
    argv: list[str]
    returncode: int
    stdout: str
    stderr: str

    @property
    def output(self) -> str:
        return self.stdout + self.stderr

    def check(self) -> CommandResult:
        if self.returncode != 0:
            raise TransportError(self)
        return self


class TransportError(RuntimeError):
    def __init__(self, result: CommandResult | None = None, message: str | None = None) -> None:
        if message is None:
            if result is None:
                raise ValueError("TransportError needs a result or a message")
            tail = "\n".join(result.output.strip().splitlines()[-10:])
            message = (
                f"`{shlex.join(result.argv)}` on host {result.host} "
                f"exited {result.returncode}\n{tail}"
            )
        super().__init__(message)
        self.result = result


class SshUnusable(TransportError):
    """A local ssh misconfiguration that retrying cannot fix.

    Raised even when the caller passed ``check=False``: every polling loop in
    this codebase treats a non-zero ssh as "not up yet", and this class of
    failure is never that.
    """


def control_socket_dir() -> Path:
    """Where ControlMaster sockets live: a short, per-user, private directory.

    Deliberately *not* the state dir. A unix socket path must fit in sun_path,
    and ``$XDG_DATA_HOME/gpu-coordinator/control/cm-<40 hex>`` under a long
    ``$HOME`` (or a pytest tmp dir) does not.
    """
    runtime = os.environ.get("XDG_RUNTIME_DIR")
    if runtime:
        return Path(runtime) / "gpuc"
    return Path(f"/tmp/gpuc-{os.getuid()}")


def control_path(directory: Path) -> str:
    """The ``ControlPath`` template, checked against sun_path before ssh runs."""
    template = f"{directory}/cm-%C"
    expanded = len(template.encode()) - len("%C") + CONTROL_HASH_LEN
    if expanded >= CONTROL_PATH_MAX:
        raise SshUnusable(
            message=(
                f"the ControlMaster socket path would be {expanded} bytes "
                f"({template}), over the {CONTROL_PATH_MAX}-byte limit a unix socket can "
                f"hold.\nSet XDG_RUNTIME_DIR to a short directory (or unset it so gpuc uses "
                f"/tmp/gpuc-{os.getuid()}) and try again."
            )
        )
    return template


class Transport(Protocol):
    host: str

    def run(self, command: str, *, timeout: float = ..., check: bool = ...) -> CommandResult: ...

    def put_file(self, content: str | bytes, remote_path: str, mode: int = ...) -> None: ...

    def rsync(
        self,
        local_root: Path,
        remote_path: str,
        files: Sequence[str] | None = ...,
        excludes: Sequence[str] = ...,
    ) -> CommandResult: ...

    def tail(self, remote_path: str, lines: int = ..., follow: bool = ...) -> CommandResult: ...

    def argv(self, command: str) -> list[str]:
        """The argv that runs `command` on the host, for a caller that streams
        or prints it rather than waiting on `run`."""
        ...

    def interactive_argv(self, command: str) -> list[str]:
        """The argv for a session a person types into: a tty, and no BatchMode."""
        ...


def _execute(
    host: str,
    argv: list[str],
    *,
    timeout: float,
    check: bool,
    stdin: bytes | None = None,
) -> CommandResult:
    try:
        proc = subprocess.run(
            argv,
            input=stdin,
            stdin=None if stdin is not None else subprocess.DEVNULL,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as exc:
        result = CommandResult(host, argv, 127, "", f"{argv[0]} not found: {exc}")
        raise TransportError(result) from exc
    except subprocess.TimeoutExpired as exc:
        result = CommandResult(host, argv, 124, "", f"timed out after {timeout}s: {exc}")
        raise TransportError(result) from exc
    result = CommandResult(
        host,
        argv,
        proc.returncode,
        proc.stdout.decode("utf-8", "replace"),
        proc.stderr.decode("utf-8", "replace"),
    )
    if result.returncode != 0 and SSH_SOCKET_FATAL.search(result.stderr):
        tail = "\n".join(result.stderr.strip().splitlines()[-5:])
        raise SshUnusable(
            result,
            f"ssh to host {host} cannot create its ControlMaster socket, so no retry can "
            f"succeed:\n{tail}\nThe socket lives under {control_socket_dir()}; check that it "
            f"exists, is writable, and that its path is short.",
        )
    return result.check() if check else result


@dataclass
class LocalTransport:
    host: str = "local"

    def argv(self, command: str) -> list[str]:
        # Not a login shell: a profile that prints a banner (or edits PATH)
        # would end up in the output we parse as JSON.
        return ["bash", "-c", command]

    def interactive_argv(self, command: str) -> list[str]:
        return ["bash", "-lc", command]

    def run(
        self, command: str, *, timeout: float = DEFAULT_TIMEOUT_S, check: bool = True
    ) -> CommandResult:
        return _execute(self.host, self.argv(command), timeout=timeout, check=check)

    def put_file(self, content: str | bytes, remote_path: str, mode: int = 0o600) -> None:
        path = Path(remote_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        data = content.encode() if isinstance(content, str) else content
        tmp = path.parent / f".{path.name}.tmp"
        # Created with the final mode, never briefly world-readable: these are
        # secrets files on boxes with other users.
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode)
        try:
            os.write(fd, data)
        finally:
            os.close(fd)
        tmp.chmod(mode)
        os.replace(tmp, path)

    def rsync(
        self,
        local_root: Path,
        remote_path: str,
        files: Sequence[str] | None = None,
        excludes: Sequence[str] = (),
    ) -> CommandResult:
        argv = rsync_argv(local_root, remote_path, files, ssh_command=None, excludes=excludes)
        return _execute(self.host, argv, timeout=3600.0, check=True, stdin=_files_stdin(files))

    def tail(self, remote_path: str, lines: int = 200, follow: bool = False) -> CommandResult:
        return self.run(tail_command(remote_path, lines, follow), check=False)


@dataclass
class SshTransport:
    host: str
    target: str
    port: int = 22
    key: str | None = None
    control_dir: Path | None = None
    known_hosts: Path | None = None

    def ssh_options(self) -> list[str]:
        options = [
            "-o",
            "BatchMode=yes",
            "-o",
            f"ConnectTimeout={CONNECT_TIMEOUT_S}",
            "-o",
            "ServerAliveInterval=30",
        ]
        if self.known_hosts is not None:
            # accept-new pins the key on first contact and then behaves like
            # StrictHostKeyChecking=yes, which is what we want for an ephemeral
            # pod that we will never see again.
            options += [
                "-o",
                f"UserKnownHostsFile={self.known_hosts}",
                "-o",
                "StrictHostKeyChecking=accept-new",
            ]
        if self.control_dir is not None:
            # %C is a hash of (host, port, user, jump): one socket per real
            # connection, and a fixed 40 characters whatever the host is called.
            options += [
                "-o",
                "ControlMaster=auto",
                "-o",
                f"ControlPath={control_path(self.control_dir)}",
                "-o",
                "ControlPersist=60",
            ]
        if self.key:
            options += ["-i", str(Path(self.key).expanduser()), "-o", "IdentitiesOnly=yes"]
        options += ["-p", str(self.port)]
        return options

    def ssh_argv(self, command: str) -> list[str]:
        # The remote login shell may be anything; bash -c makes the command we
        # send mean the same thing everywhere, without sourcing a profile.
        return ["ssh", *self.ssh_options(), self.target, f"bash -c {shlex.quote(command)}"]

    def argv(self, command: str) -> list[str]:
        return self.ssh_argv(command)

    def interactive_argv(self, command: str) -> list[str]:
        """`-t` forces a tty: without it the remote shell has no job control,
        no prompt and no `clear`. BatchMode is dropped, because the user may
        well need to type a key passphrase; it is right for every automated
        call, where a prompt would hang a polling loop for ever."""
        kept: list[str] = []
        for opt in self.ssh_options():
            if opt.startswith("BatchMode") and kept and kept[-1] == "-o":
                kept.pop()
                continue
            kept.append(opt)
        return ["ssh", *kept, "-t", self.target, command]

    def _prepare(self) -> None:
        if self.control_dir is not None:
            self.control_dir.mkdir(parents=True, exist_ok=True)
            # 0700: the socket is a live authenticated channel to the host.
            self.control_dir.chmod(0o700)
        if self.known_hosts is not None:
            self.known_hosts.parent.mkdir(parents=True, exist_ok=True)
            self.known_hosts.touch(exist_ok=True)

    def run(
        self, command: str, *, timeout: float = DEFAULT_TIMEOUT_S, check: bool = True
    ) -> CommandResult:
        self._prepare()
        return _execute(self.host, self.ssh_argv(command), timeout=timeout, check=check)

    def put_file_argv(self, remote_path: str, mode: int = 0o600) -> list[str]:
        quoted = shlex.quote(remote_path)
        # umask first: `cat >` alone creates the file 0644, so a secret is
        # world-readable for as long as it takes the chmod to land.
        return self.ssh_argv(
            f'mkdir -p "$(dirname {quoted})" && (umask 077 && cat > {quoted}) '
            f"&& chmod {mode:o} {quoted}"
        )

    def put_file(self, content: str | bytes, remote_path: str, mode: int = 0o600) -> None:
        self._prepare()
        data = content.encode() if isinstance(content, str) else content
        # Content goes over stdin: secrets must never appear in argv, where any
        # other user on the box can read them out of /proc.
        _execute(
            self.host,
            self.put_file_argv(remote_path, mode),
            timeout=DEFAULT_TIMEOUT_S,
            check=True,
            stdin=data,
        )

    def rsync_ssh_command(self) -> str:
        return shlex.join(["ssh", *self.ssh_options()])

    def rsync(
        self,
        local_root: Path,
        remote_path: str,
        files: Sequence[str] | None = None,
        excludes: Sequence[str] = (),
    ) -> CommandResult:
        self._prepare()
        argv = rsync_argv(
            local_root,
            f"{self.target}:{remote_path}",
            files,
            self.rsync_ssh_command(),
            excludes=excludes,
        )
        return _execute(self.host, argv, timeout=3600.0, check=True, stdin=_files_stdin(files))

    def tail(self, remote_path: str, lines: int = 200, follow: bool = False) -> CommandResult:
        return self.run(
            tail_command(remote_path, lines, follow), timeout=DEFAULT_TIMEOUT_S, check=False
        )


def tail_command(
    remote_path: str, lines: int = 200, follow: bool = False, retry: bool = False
) -> str:
    """The one `tail` line both transports (and `gpuc logs -f`) send.

    `retry` waits for a log that is not there yet instead of failing, which is
    what a follow of a job still in the queue needs and what a plain read must
    not do: a missing log is exactly the non-zero exit that sends `gpuc logs`
    to the S3 mirror.
    """
    flags = "-F " if follow and retry else "-f " if follow else ""
    return f"tail {flags}-n {lines} {shlex.quote(remote_path)}"


def rsync_argv(
    local_root: Path,
    destination: str,
    files: Sequence[str] | None,
    ssh_command: str | None,
    excludes: Sequence[str] = (),
) -> list[str]:
    argv = ["rsync", "-a"]
    if ssh_command:
        argv += ["-e", ssh_command]
    for pattern in excludes:
        argv += ["--exclude", pattern]
    if files is None:
        argv += ["--delete-after"]
    else:
        # --files-from keeps `git ls-files` output off the command line, which
        # otherwise blows the argv limit on a real repository. NUL-separated so
        # that a filename with a newline in it cannot split into two entries,
        # and --ignore-missing-args so a file deleted between `git ls-files` and
        # the transfer is skipped instead of failing the whole sync.
        argv += ["--from0", "--files-from=-", "--ignore-missing-args"]
    argv += [f"{str(local_root).rstrip('/')}/", destination]
    return argv


def _files_stdin(files: Sequence[str] | None) -> bytes | None:
    if files is None:
        return None
    return "".join(f"{name}\0" for name in files).encode()


NO_GIT_EXCLUDES = (".venv", "__pycache__", ".git", "*.pyc", "node_modules", ".uv-cache")
"""What `--no-git` leaves behind when there is no `.gitignore` to obey.

Not a policy, just the handful of things that are always regenerable and always
enormous; anything else in a non-repo directory is assumed to be wanted."""


def _git(root: Path, args: list[str], env: dict[str, str] | None = None) -> tuple[int, bytes, str]:
    argv = ["git", "-C", str(root), "-c", "core.quotePath=false", *args]
    proc = subprocess.run(argv, capture_output=True, check=False, env=env)
    return proc.returncode, proc.stdout, proc.stderr.decode("utf-8", "replace")


def _git_or_raise(root: Path, args: list[str]) -> list[str]:
    code, stdout, stderr = _git(root, args)
    if code != 0:
        argv = ["git", "-C", str(root), *args]
        raise TransportError(
            CommandResult("local", argv, code, stdout.decode("utf-8", "replace"), stderr)
        )
    return [name for name in stdout.decode("utf-8", "replace").split("\0") if name]


def git_tracked_files(root: Path) -> list[str]:
    """Everything worth syncing: tracked files *and* untracked ones git would keep.

    Tracked-only was the old rule, and it silently dropped the file someone had
    just written and not yet `git add`ed -- which on a fresh experiment is the
    whole experiment. `--exclude-standard` keeps .gitignore honoured, so venvs
    and caches still stay home. quotePath=false and -z: without both, a path
    with a space or a non-ASCII byte comes back C-quoted and rsync then looks
    for a file whose name contains literal backslashes.

    A file that is in the index but deleted on disk is dropped rather than
    named: rsync would otherwise be asked for a file that is not there.
    """
    names = _git_or_raise(root, ["ls-files", "-z", "--cached", "--others", "--exclude-standard"])
    # lexists, not exists: a dangling symlink is a real entry rsync can copy.
    return [name for name in dict.fromkeys(names) if os.path.lexists(root / name)]


@dataclass
class GitSummary:
    """What `submit` prints, so the sync is never a surprise."""

    files: list[str]
    modified: int
    untracked: int

    def render(self) -> str:
        return (
            f"syncing {len(self.files)} files ({self.modified} modified, "
            f"{self.untracked} untracked, ignoring .gitignore'd)"
        )


def git_summary(root: Path) -> GitSummary:
    files = git_tracked_files(root)
    untracked = set(_git_or_raise(root, ["ls-files", "-z", "--others", "--exclude-standard"]))
    code, stdout, _ = _git(root, ["diff", "-z", "--name-only", "HEAD"])
    changed = (
        {n for n in stdout.decode("utf-8", "replace").split("\0") if n} if code == 0 else set()
    )
    known = set(files)
    return GitSummary(
        files=files,
        modified=len(changed & known),
        untracked=len(untracked & known),
    )


def uncommitted_patch(root: Path) -> str:
    """`git diff HEAD` including untracked files, via a throwaway index.

    A patch that omits the files the run's results depend on is worse than no
    patch: `git add -N` in a copy of the index makes new files show up as
    additions without touching the real index or the user's staging.
    """
    code, stdout, _ = _git(root, ["rev-parse", "--absolute-git-dir"])
    if code != 0:
        return ""
    real_index = Path(stdout.decode().strip()) / "index"
    with tempfile.TemporaryDirectory(prefix="gpuc-index-") as tmp:
        index = Path(tmp) / "index"
        if real_index.exists():
            shutil.copy2(real_index, index)
        env = {**os.environ, "GIT_INDEX_FILE": str(index)}
        _git(root, ["add", "-N", "--", "."], env)
        # `-- .` because `root` may be a subdirectory of the repository: without
        # it the patch carries every change in the repo, including files this
        # job never syncs, while `git add -N` above only ever saw `.`.
        code, stdout, _ = _git(root, ["diff", "HEAD", "--", "."], env)
    return stdout.decode("utf-8", "replace") if code == 0 else ""


def make_transport(
    host: str,
    *,
    ssh: str | None = None,
    port: int = 22,
    key: str | None = None,
    known_hosts: Path | None = None,
) -> Transport:
    """``known_hosts`` is where host keys are pinned: the shared file for most
    hosts, a per-pod one for provisioning, so a recycled RunPod address cannot
    collide with a pinned key. The ControlMaster socket always goes in the
    short runtime directory (see ``control_socket_dir``).
    """
    if ssh is None:
        return LocalTransport(host=host)
    return SshTransport(
        host=host,
        target=ssh,
        port=port,
        key=key,
        control_dir=control_socket_dir(),
        known_hosts=known_hosts,
    )
