"""One way to invoke the on-host package, shared by bootstrap, submit and status.

Every remote invocation runs the interpreter discovered at bootstrap with
``PYTHONPATH`` pointing at the rsynced package and ``GPUC_HOME`` pinned, so a
host whose login shell has a different `python` on PATH still runs our code.
"""

from __future__ import annotations

import json
import os
import shlex
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from gpuc.control.config import HostEntry, Settings, transport_for
from gpuc.control.transport import CommandResult, Transport, TransportError
from gpuc.host import jobs

DEFAULT_TIMEOUT_S = 120.0


class RemoteError(RuntimeError):
    def __init__(self, host: str, command: str, detail: str) -> None:
        super().__init__(f"{detail}\n  host: {host}\n  command: {command}")
        self.host = host
        self.command = command


def host_command(python: str, home: str, args: str, env: Mapping[str, str] | None = None) -> str:
    """One invocation of the on-host package, with its environment pinned.

    ``env`` is the host's own `HostEntry.env`. Every host-package invocation
    gets it, because `enqueue` spawns the dispatcher and the dispatcher's
    environment is what every job on the host inherits.
    """
    assignments = "".join(f'{key}="{value}" ' for key, value in sorted((env or {}).items()))
    return f'{assignments}GPUC_HOME="{home}" PYTHONPATH="{home}/pkg" "{python}" -m gpuc.host {args}'


@dataclass
class HostSession:
    entry: HostEntry
    transport: Transport
    home: str
    python: str

    @property
    def env(self) -> dict[str, str]:
        return self.entry.env

    def read_config(self) -> dict[str, Any] | None:
        return read_remote_config(self.transport, self.home)

    def job_dir(self, job_id: str) -> str:
        return f"{self.home}/jobs/{job_id}"

    def run(self, command: str, *, timeout: float = DEFAULT_TIMEOUT_S) -> CommandResult:
        return self.transport.run(command, timeout=timeout, check=False)

    def host_cli(
        self, args: str, *, timeout: float = DEFAULT_TIMEOUT_S, check: bool = True
    ) -> CommandResult:
        command = host_command(self.python, self.home, args, self.env)
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

    def host_json(
        self, args: str, *, timeout: float = DEFAULT_TIMEOUT_S, check: bool = True
    ) -> Any:
        """The JSON document a host subcommand printed.

        `check=False` for the subcommands whose document *is* the report of a
        failure -- `clean` and `purge` exit 1 with a populated `errors` and a
        full account of what they did and did not delete, and raising on the
        exit code would throw that account away. A host that printed no
        document at all is still an error either way.
        """
        result = self.host_cli(args, timeout=timeout, check=check)
        document = parse_last_json(result.stdout)
        if document is _NO_JSON:
            exited = f"exited {result.returncode} and " if result.returncode else ""
            raise RemoteError(
                self.entry.name,
                host_command(self.python, self.home, args, self.env),
                f"{exited}expected JSON on stdout, got:\n{_tail(result.output)}",
            )
        return document


_NO_JSON = object()


def parse_last_json(text: str) -> Any:
    """The last JSON document on stdout, or ``_NO_JSON``.

    Hosts print things we do not control around our output -- a MOTD, an
    activation notice, a warning from a shell rc file -- so the document we
    want is the last one in the stream, not the whole stream. Every line that
    could begin a document is tried with raw_decode, and the winner is the one
    that *ends* last: inside a pretty-printed report every nested object also
    starts a line, and the outermost one is the answer.
    """
    decoder = json.JSONDecoder()
    best: tuple[int, int] | None = None
    document: Any = _NO_JSON
    offset = 0
    for line in text.splitlines(keepends=True):
        stripped = line.lstrip()
        if stripped[:1] in ("{", "["):
            start = offset + len(line) - len(stripped)
            try:
                parsed, consumed = decoder.raw_decode(text[start:])
            except json.JSONDecodeError:
                parsed, consumed = None, None
            if consumed is not None:
                candidate = (start + consumed, -start)
                if best is None or candidate > best:
                    best, document = candidate, parsed
        offset += len(line)
    return document


def _tail(text: str, lines: int = 10) -> str:
    return "\n".join(text.strip().splitlines()[-lines:])


def config_file(home: str) -> str:
    return f"{home}/config.json"


NO_CONFIG = "__gpuc_no_config__"
"""What the host says when it has no `config.json`, so that "there is none" is
never confused with "it is there and did not parse" -- the second is a file
somebody's host is running on, and replacing it would be the drift this whole
model exists to stop. A marker rather than an exit code, because a host prints
things around our output (a MOTD, a shell rc warning) that no parse can
distinguish from a config that is simply broken."""


def read_remote_config(transport: Transport, home: str) -> dict[str, Any] | None:
    """The host's own `config.json`; ``{}`` if it has none, ``None`` if we
    could not read it.

    This file is the only copy of what the host *is* -- its cards, its mirror,
    its env, its timers -- so everything that acts on a host reads it here
    rather than trusting the registry's cache of it, and nothing writes over a
    `None`: a host that could not be asked, or one whose config is there but
    unreadable, is not a host with no config.
    """
    path = config_file(home)
    try:
        result = transport.run(
            f'if [ -f "{path}" ]; then cat "{path}"; else echo {NO_CONFIG}; fi',
            timeout=DEFAULT_TIMEOUT_S,
            check=False,
        )
    except TransportError:
        return None
    if result.returncode != 0:
        return None
    if NO_CONFIG in result.stdout:
        return {}
    document = parse_last_json(result.stdout)
    return document if isinstance(document, dict) else None


def write_remote_config(
    transport: Transport,
    home: str,
    patch: Mapping[str, Any],
    *,
    python: str | None = None,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Apply `patch` to the host's `config.json` and return what it now holds.

    Through the host's own CLI wherever the package is there: the merge then
    happens on the host, in one atomic write, by the same code the dispatcher
    reads the file with. A host that has not been bootstrapped yet has no
    package to run, so the same merge is done here and the file replaced by
    rename -- never truncated in place, because a dispatcher may be reading it.

    The patch travels as a file rather than as an argument: `env` may hold a
    token, and argv is readable by every other user of a shared box.
    """
    body = json.dumps(dict(patch), indent=2, sort_keys=True) + "\n"
    if python:
        try:
            return _merge_on_host(transport, home, body, python, env)
        except (RemoteError, TransportError):
            # The package is not where the registry says it is (a wiped $HOME,
            # a gpuc home that moved), or it is a build old enough not to have
            # this subcommand. The host still owns its config either way, and
            # the same merge below is what its own CLI would have done.
            pass
    existing = read_remote_config(transport, home)
    if existing is None:
        raise RemoteError(
            transport.host,
            f'cat "{config_file(home)}"',
            f"the host's own config could not be read, so it was left alone.\n"
            f"Check that {config_file(home)} is readable and holds JSON; delete it to start "
            f"that host again.",
        )
    document = jobs.merged_config(existing, patch)
    put_remote_config(transport, home, document)
    return document


def _merge_on_host(
    transport: Transport,
    home: str,
    body: str,
    python: str,
    env: Mapping[str, str] | None,
) -> dict[str, Any]:
    remote = f"{home}/.config-patch.{os.getpid()}.json"
    transport.put_file(body, remote, 0o600)
    command = host_command(python, home, f"config --merge {shlex.quote(remote)}", env)
    try:
        result = transport.run(command, timeout=DEFAULT_TIMEOUT_S, check=False)
    finally:
        transport.run(f"rm -f {shlex.quote(remote)}", timeout=DEFAULT_TIMEOUT_S, check=False)
    document = parse_last_json(result.stdout)
    if result.returncode != 0 or not isinstance(document, dict):
        raise RemoteError(
            transport.host,
            command,
            f"`python -m gpuc.host config --merge` exited {result.returncode} and printed no "
            f"config:\n{_tail(result.output)}",
        )
    return document


def put_remote_config(transport: Transport, home: str, document: Mapping[str, Any]) -> None:
    """Replace `config.json` wholesale on a host with no package to run.

    Creates gpuc home 0700 *if it is not there*: this runs before
    `paths.ensure_layout` on a host being registered for the first time, and a
    default-umask mkdir would leave the queue and every job dir readable by
    every other user of a shared box. An existing directory keeps its mode, as
    `ensure_persistent_root` does -- re-chmodding one is not ours to do.
    """
    tmp = f"{home}/.config.json.{os.getpid()}.tmp"
    quoted = shlex.quote(tmp)
    transport.run(
        f'if [ ! -d "{home}" ]; then mkdir -p "{home}"; chmod 700 "{home}"; fi',
        timeout=DEFAULT_TIMEOUT_S,
        check=True,
    )
    transport.put_file(json.dumps(dict(document), indent=2, sort_keys=True) + "\n", tmp, 0o644)
    try:
        transport.run(
            f"mv -f {quoted} {shlex.quote(config_file(home))}",
            timeout=DEFAULT_TIMEOUT_S,
            check=True,
        )
    except TransportError:
        transport.run(f"rm -f {quoted}", timeout=DEFAULT_TIMEOUT_S, check=False)
        raise


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
