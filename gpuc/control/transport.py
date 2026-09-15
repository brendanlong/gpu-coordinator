"""How the control side reaches a host: locally, or over ssh/rsync.

Both transports present the same surface so the rest of the control side never
branches on host kind. Errors carry the command, the host, and the tail of the
output, because "it failed" on a remote host is otherwise undebuggable.
"""

from __future__ import annotations

import os
import shlex
import subprocess
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

DEFAULT_TIMEOUT_S = 120.0
CONNECT_TIMEOUT_S = 15


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
    def __init__(self, result: CommandResult) -> None:
        tail = "\n".join(result.output.strip().splitlines()[-10:])
        super().__init__(
            f"`{shlex.join(result.argv)}` on host {result.host} exited {result.returncode}\n{tail}"
        )
        self.result = result


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
    return result.check() if check else result


@dataclass
class LocalTransport:
    host: str = "local"

    def run(
        self, command: str, *, timeout: float = DEFAULT_TIMEOUT_S, check: bool = True
    ) -> CommandResult:
        return _execute(self.host, ["bash", "-lc", command], timeout=timeout, check=check)

    def put_file(self, content: str | bytes, remote_path: str, mode: int = 0o600) -> None:
        path = Path(remote_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        data = content.encode() if isinstance(content, str) else content
        tmp = path.parent / f".{path.name}.tmp"
        tmp.write_bytes(data)
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
            socket = self.control_dir / f"cm-{self.host}"
            options += [
                "-o",
                "ControlMaster=auto",
                "-o",
                f"ControlPath={socket}",
                "-o",
                "ControlPersist=60",
            ]
        if self.key:
            options += ["-i", str(Path(self.key).expanduser()), "-o", "IdentitiesOnly=yes"]
        options += ["-p", str(self.port)]
        return options + self.extra_options

    def ssh_argv(self, command: str) -> list[str]:
        return ["ssh", *self.ssh_options(), self.target, command]

    def _prepare(self) -> None:
        if self.control_dir is not None:
            self.control_dir.mkdir(parents=True, exist_ok=True)
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
        return self.ssh_argv(
            f"mkdir -p $(dirname {quoted}) && cat > {quoted} && chmod {mode:o} {quoted}"
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
        # otherwise blows the argv limit on a real repository.
        argv += ["--files-from=-"]
    argv += [f"{str(local_root).rstrip('/')}/", destination]
    return argv


def _files_stdin(files: Sequence[str] | None) -> bytes | None:
    if files is None:
        return None
    return ("\n".join(files) + "\n").encode()


def git_tracked_files(root: Path) -> list[str]:
    argv = ["git", "-C", str(root), "ls-files"]
    proc = subprocess.run(argv, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise TransportError(
            CommandResult("local", argv, proc.returncode, proc.stdout, proc.stderr)
        )
    return [line for line in proc.stdout.splitlines() if line]


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
    extra_options: Iterable[str] = (),
) -> Transport:
    if ssh is None:
        return LocalTransport(host=host)
    return SshTransport(
        host=host,
        target=ssh,
        port=port,
        key=key,
        control_dir=None if state_dir is None else state_dir / "control",
        known_hosts=None if state_dir is None else state_dir / "known_hosts",
        extra_options=list(extra_options),
    )
