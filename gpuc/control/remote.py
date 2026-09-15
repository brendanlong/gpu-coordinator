"""One way to invoke the on-host package, shared by bootstrap, submit and status.

Every remote invocation runs the interpreter discovered at bootstrap with
``PYTHONPATH`` pointing at the rsynced package and ``GPUC_HOME`` pinned, so a
host whose login shell has a different `python` on PATH still runs our code.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from gpuc.control.config import HostEntry, Settings, transport_for
from gpuc.control.transport import CommandResult, Transport, TransportError

DEFAULT_TIMEOUT_S = 120.0


class RemoteError(RuntimeError):
    def __init__(self, host: str, command: str, detail: str) -> None:
        super().__init__(f"{detail}\n  host: {host}\n  command: {command}")
        self.host = host
        self.command = command


def host_command(python: str, home: str, args: str) -> str:
    return f'GPUC_HOME="{home}" PYTHONPATH="{home}/pkg" "{python}" -m gpuc.host {args}'


@dataclass
class HostSession:
    entry: HostEntry
    transport: Transport
    home: str
    python: str

    @property
    def pkg_dir(self) -> str:
        return f"{self.home}/pkg"

    def job_dir(self, job_id: str) -> str:
        return f"{self.home}/jobs/{job_id}"

    def run(self, command: str, *, timeout: float = DEFAULT_TIMEOUT_S) -> CommandResult:
        return self.transport.run(command, timeout=timeout, check=False)

    def host_cli(
        self, args: str, *, timeout: float = DEFAULT_TIMEOUT_S, check: bool = True
    ) -> CommandResult:
        command = host_command(self.python, self.home, args)
        result = self.transport.run(command, timeout=timeout, check=False)
        if check and result.returncode != 0:
            raise RemoteError(
                self.entry.name,
                command,
                f"`python -m gpuc.host {args}` exited {result.returncode}\n"
                f"{_tail(result.output)}\n"
                f"If the package is missing, run: gpuc host bootstrap {self.entry.name}",
            )
        return result

    def host_json(self, args: str, *, timeout: float = DEFAULT_TIMEOUT_S) -> Any:
        result = self.host_cli(args, timeout=timeout)
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise RemoteError(
                self.entry.name,
                host_command(self.python, self.home, args),
                f"expected JSON on stdout, got:\n{_tail(result.output)}\n({exc})",
            ) from exc


def _tail(text: str, lines: int = 10) -> str:
    return "\n".join(text.strip().splitlines()[-lines:])


def resolve_home(transport: Transport, entry: HostEntry) -> str:
    """Expand ``$HOME/.gpuc`` on the host: rsync and tail need a real path."""
    template = entry.remote_home
    if "$" not in template and "~" not in template:
        return template.rstrip("/")
    try:
        result = transport.run(f'printf %s "{template}"', timeout=DEFAULT_TIMEOUT_S, check=True)
    except TransportError as exc:
        raise RemoteError(
            entry.name,
            f'printf %s "{template}"',
            f"could not reach host {entry.name}: {exc}\n"
            f"Check `gpuc host list`, then: gpuc host probe {entry.name}",
        ) from exc
    home = result.stdout.strip()
    if not home:
        raise RemoteError(entry.name, template, "host returned an empty $GPUC_HOME")
    return home.rstrip("/")


def open_session(
    entry: HostEntry, settings: Settings | None = None, transport: Transport | None = None
) -> HostSession:
    transport = transport or transport_for(entry, settings)
    if not entry.python:
        raise RemoteError(
            entry.name,
            "open_session",
            f"host {entry.name!r} has no bootstrapped interpreter recorded.\n"
            f"Run: gpuc host bootstrap {entry.name}",
        )
    return HostSession(entry, transport, resolve_home(transport, entry), entry.python)
