"""How the control side reaches a host: locally, or over ssh/rsync.

Both transports present the same surface so the rest of the control side never
branches on host kind. Errors carry the command, the host, and the tail of the
output, because "it failed" on a remote host is otherwise undebuggable.
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
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
        self, local_root: Path, remote_path: str, files: Sequence[str] | None = ...
    ) -> CommandResult: ...

    def tail(self, remote_path: str, lines: int = ..., follow: bool = ...) -> CommandResult: ...


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

    def run(
        self, command: str, *, timeout: float = DEFAULT_TIMEOUT_S, check: bool = True
    ) -> CommandResult:
        # Not a login shell: a profile that prints a banner (or edits PATH)
        # would end up in the output we parse as JSON.
        return _execute(self.host, ["bash", "-c", command], timeout=timeout, check=check)

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
        self, local_root: Path, remote_path: str, files: Sequence[str] | None = None
    ) -> CommandResult:
        argv = rsync_argv(local_root, remote_path, files, ssh_command=None)
        return _execute(self.host, argv, timeout=3600.0, check=True, stdin=_files_stdin(files))

    def tail(self, remote_path: str, lines: int = 200, follow: bool = False) -> CommandResult:
        flag = "-f" if follow else ""
        return self.run(f"tail {flag} -n {lines} {shlex.quote(remote_path)}", check=False)


@dataclass
class SshTransport:
    host: str
    target: str
    port: int = 22
    key: str | None = None
    control_dir: Path | None = None
    known_hosts: Path | None = None
    extra_options: list[str] = field(default_factory=list)

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
        return options + self.extra_options

    def ssh_argv(self, command: str) -> list[str]:
        # The remote login shell may be anything; bash -c makes the command we
        # send mean the same thing everywhere, without sourcing a profile.
        return ["ssh", *self.ssh_options(), self.target, f"bash -c {shlex.quote(command)}"]

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
        self, local_root: Path, remote_path: str, files: Sequence[str] | None = None
    ) -> CommandResult:
        self._prepare()
        argv = rsync_argv(
            local_root, f"{self.target}:{remote_path}", files, self.rsync_ssh_command()
        )
        return _execute(self.host, argv, timeout=3600.0, check=True, stdin=_files_stdin(files))

    def tail(self, remote_path: str, lines: int = 200, follow: bool = False) -> CommandResult:
        flag = "-f " if follow else ""
        return self.run(
            f"tail {flag}-n {lines} {shlex.quote(remote_path)}",
            timeout=DEFAULT_TIMEOUT_S,
            check=False,
        )


def rsync_argv(
    local_root: Path,
    destination: str,
    files: Sequence[str] | None,
    ssh_command: str | None,
) -> list[str]:
    argv = ["rsync", "-a"]
    if ssh_command:
        argv += ["-e", ssh_command]
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


def git_tracked_files(root: Path) -> list[str]:
    # quotePath=false and -z: without both, a path with a space, a quote or a
    # non-ASCII byte comes back C-quoted and rsync then looks for a file whose
    # name contains literal backslashes.
    argv = ["git", "-C", str(root), "-c", "core.quotePath=false", "ls-files", "-z"]
    proc = subprocess.run(argv, capture_output=True, check=False)
    stdout = proc.stdout.decode("utf-8", "replace")
    if proc.returncode != 0:
        raise TransportError(
            CommandResult(
                "local", argv, proc.returncode, stdout, proc.stderr.decode("utf-8", "replace")
            )
        )
    return [name for name in stdout.split("\0") if name]


def uncommitted_patch(root: Path) -> str:
    proc = subprocess.run(
        ["git", "-C", str(root), "diff", "HEAD"], capture_output=True, text=True, check=False
    )
    return proc.stdout if proc.returncode == 0 else ""


def make_transport(
    host: str,
    *,
    ssh: str | None = None,
    port: int = 22,
    key: str | None = None,
    state_dir: Path | None = None,
    known_hosts: Path | None = None,
    extra_options: Iterable[str] = (),
) -> Transport:
    """``known_hosts`` overrides the shared file: provisioning gives each pod
    its own, so a recycled RunPod address cannot collide with a pinned key.

    ``state_dir`` is only the known_hosts location; the ControlMaster socket
    always goes in the short runtime directory (see ``control_socket_dir``).
    """
    if ssh is None:
        return LocalTransport(host=host)
    if known_hosts is None and state_dir is not None:
        known_hosts = state_dir / "known_hosts"
    return SshTransport(
        host=host,
        target=ssh,
        port=port,
        key=key,
        control_dir=control_socket_dir(),
        known_hosts=known_hosts,
        extra_options=list(extra_options),
    )
