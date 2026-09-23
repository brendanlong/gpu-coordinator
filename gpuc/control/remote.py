"""One way to reach a host and one way to ask it something.

A `HostSession` is a transport plus the three things every on-host command
needs: the resolved gpuc home, an interpreter to run the package with, and
the host's own `config.json`, read fresh when the session opens. Everything
that decides something about a host reads `session.config`; the registry's
cache is for listings.

`ask` is the one answer to "can this host be asked, and what did it say":
`Answered`, `Unreachable` with the reason, or -- for a rental -- what the
provider said about its pod instead. Every command that talks to a host goes
through it, so "could not ask" is spelled once.
"""

from __future__ import annotations

import json
import os
import shlex
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from gpuc.control.config import (
    ConfigError,
    HostEntry,
    Settings,
    transport_for,
    update_cache,
)
from gpuc.control.providers.base import Pod, Provider, ProviderError
from gpuc.control.transport import DEFAULT_TIMEOUT_S, CommandResult, Transport, TransportError
from gpuc.host import jobs
from gpuc.host.jobs import HostConfig

PYTHON_FLOOR = (3, 11)
"""What the on-host package needs; bootstrap spells the same tuple for `uv`."""


class RemoteError(RuntimeError):
    def __init__(self, host: str, command: str, detail: str) -> None:
        super().__init__(f"{detail}\n  host: {host}\n  command: {command}")


def reason_of(exc: BaseException) -> str:
    """One line saying why a host could not be asked.

    The last non-empty line of the stderr the failed command carried, when
    the error carries a `CommandResult` -- or is a `RemoteError` wrapping one,
    whose own first line repeats the argv: that is where ssh puts `Connection
    refused`, `Permission denied (publickey)` and `Host key verification
    failed`, while the first line of a `TransportError` is the argv, which
    names nothing. Otherwise the first line of the message: an error with
    words of its own keeps them.
    """
    result = getattr(exc, "result", None)
    if result is None and isinstance(exc, RemoteError):
        result = getattr(exc.__cause__, "result", None)
    if isinstance(result, CommandResult):
        lines = [line.strip() for line in result.stderr.splitlines() if line.strip()]
        if lines:
            return lines[-1]
    lines = [line.strip() for line in str(exc).splitlines() if line.strip()]
    return lines[0] if lines else type(exc).__name__


def env_prefix(env: Mapping[str, str] | None) -> str:
    """``K="v" `` assignments for a remote command, or ``""`` for most hosts.

    ``env`` is the host's own `HostConfig.env`. Everything started on the host
    gets it -- the package (whose dispatcher's environment every job inherits)
    and bootstrap's installs, so `uv tool install` populates the cache this
    host uses.
    """
    return "".join(f'{key}="{value}" ' for key, value in sorted((env or {}).items()))


def host_python(python: str, home: str, env: Mapping[str, str] | None = None) -> str:
    """The interpreter, with the on-host package importable and gpuc home
    pinned: the prefix of every command that runs the host's own code."""
    return f'{env_prefix(env)}GPUC_HOME="{home}" PYTHONPATH="{home}/pkg" "{python}"'


def host_command(python: str, home: str, args: str, env: Mapping[str, str] | None = None) -> str:
    """One invocation of the on-host package, with its environment pinned."""
    return f"{host_python(python, home, env)} -m gpuc.host {args}"


def config_file(home: str) -> str:
    return f"{home}/config.json"


NO_CONFIG = "__gpuc_no_config__"
"""What the host says when it has no `config.json`, so that "there is none" is
never confused with "it is there and did not parse" -- the second is a file
somebody's host is running on, and replacing it would be the drift this whole
model exists to stop. A marker rather than an exit code, because a host prints
things around our output (a MOTD, a shell rc warning) that no parse can
distinguish from a config that is simply broken."""


@dataclass
class HostConfigRead:
    """One read of a host's `config.json`: the document, no file, or a reason.

    Three answers because they lead three places. A document is what the host
    is. No file is a host nobody has set up (or whose home was wiped), and the
    one case anything may write a first config for. Unreadable -- there but
    not JSON, or a transport that failed mid-read -- is a file the host may be
    running on, and nothing here writes over it.
    """

    document: dict[str, Any] | None = None
    unreadable: str | None = None

    @property
    def missing(self) -> bool:
        return self.document is None and self.unreadable is None

    @property
    def config(self) -> HostConfig:
        return HostConfig.from_dict(self.document or {})


def read_config(
    transport: Transport, home: str, *, timeout: float = DEFAULT_TIMEOUT_S
) -> HostConfigRead:
    """The host's own `config.json`, the one read every command works from."""
    path = config_file(home)
    command = f'if [ -f "{path}" ]; then cat "{path}"; else echo {NO_CONFIG}; fi'
    try:
        result = transport.run(command, timeout=timeout, check=False)
    except TransportError as exc:
        return HostConfigRead(unreadable=reason_of(exc))
    if result.returncode != 0:
        return HostConfigRead(
            unreadable=f"`cat {path}` exited {result.returncode}: {_tail(result.output, 3)}"
        )
    # Parsed first: a config whose own values happen to hold the marker is
    # still a config, and it is the one thing here that must not be mistaken
    # for a host that has none.
    document = parse_last_json(result.stdout)
    if isinstance(document, dict):
        return HostConfigRead(document)
    if NO_CONFIG in result.stdout:
        return HostConfigRead()
    return HostConfigRead(unreadable=f"{path} is there but holds no JSON object")


def write_config(
    transport: Transport, home: str, patch: Mapping[str, Any], *, host: str
) -> dict[str, Any]:
    """Apply `patch` to the host's `config.json` and return what it now holds.

    The client is the one writer of this file: read what is there, merge with
    the same rule the host reads it by (`jobs.merged_config`), and replace by
    rename -- never truncated in place, because a dispatcher may be reading
    it. A config that could not be read is left alone, however small the
    patch: a host is that file, and a write over one we cannot see is the
    drift the whole model exists to stop.
    """
    read = read_config(transport, home)
    if read.unreadable:
        raise RemoteError(
            host,
            f'cat "{config_file(home)}"',
            f"the host's own config could not be read, so it was left alone: {read.unreadable}\n"
            f"Check that {config_file(home)} is readable and holds JSON; delete it to start "
            f"that host again.",
        )
    document = jobs.merged_config(read.document or {}, patch)
    put_config(transport, home, document)
    return document


def put_config(transport: Transport, home: str, document: Mapping[str, Any]) -> None:
    """Replace `config.json` wholesale, by rename.

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


@dataclass
class HostSession:
    entry: HostEntry
    transport: Transport
    home: str
    python: str
    config_read: HostConfigRead
    """The host's `config.json` as read when this session opened, or why it
    could not be. Refreshed by `write_config`, so what a session decides on
    is always the host's own answer."""
    record: bool = True
    """Whether what this session reads and writes goes into the registry's
    cache on the way past. Off for a poll: a dashboard never writes the
    registry."""

    @property
    def config(self) -> HostConfig:
        return self.config_read.config

    @property
    def env(self) -> dict[str, str]:
        """The host's own job environment, prefixed to every command run here."""
        return self.config.env

    def write_config(self, patch: Mapping[str, Any]) -> dict[str, Any]:
        document = write_config(self.transport, self.home, patch, host=self.entry.name)
        self.config_read = HostConfigRead(document)
        if self.record:
            update_cache(self.entry.name, config=document)
        return document

    def job_dir(self, job_id: str) -> str:
        return f"{self.home}/jobs/{job_id}"

    def staging_dir(self, job_id: str) -> str:
        """Where `submit` builds a job before `enqueue` accepts it by renaming
        the dir into `jobs/`; see `gpuc.host.queue.enqueue`."""
        return f"{self.home}/incoming/{job_id}"

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


def resolve_home(
    transport: Transport, entry: HostEntry, *, timeout: float = DEFAULT_TIMEOUT_S
) -> str:
    """Expand ``$HOME/.gpuc`` on the host: rsync and tail need a real path."""
    template = entry.remote_home
    if "$" not in template and "~" not in template:
        return template.rstrip("/")
    try:
        result = transport.run(f'printf %s "{template}"', timeout=timeout, check=True)
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


_SAY_VERSION = "-c 'import sys; print(sys.executable, sys.version.split()[0])'"
PYTHON_PROBE = (
    'if [ -x "$HOME/.local/bin/uv" ]; then p=$(cd "$HOME" && '
    'env -u VIRTUAL_ENV -u UV_PROJECT_ENVIRONMENT "$HOME/.local/bin/uv" python find '
    f"--no-project '>={PYTHON_FLOOR[0]}.{PYTHON_FLOOR[1]}' 2>/dev/null); "
    f'[ -n "$p" ] && "$p" {_SAY_VERSION}; fi; '
    f"if command -v python3 >/dev/null 2>&1; then python3 {_SAY_VERSION}; fi"
)
"""An interpreter that can run the on-host package: the one uv manages for
this user if there is one, else the system `python3`, each as `path version`
for `usable_python` to judge. The same question `gpuc host probe` asks,
without the rest of the probe."""


def usable_python(text: str) -> str | None:
    """The first `path version` line in `text` whose version can run the package.

    The version is judged here rather than trusted: a `python3` that is 3.8 is
    the commonest thing on an old box, and the package would import and then
    fail on the first `match`.
    """
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 2 or not parts[0].startswith("/"):
            continue
        try:
            version = tuple(int(piece) for piece in parts[1].split(".")[:2])
        except ValueError:
            continue
        if version >= PYTHON_FLOOR:
            return parts[0]
    return None


def probe_python(transport: Transport, *, timeout: float = DEFAULT_TIMEOUT_S) -> str | None:
    result = transport.run(PYTHON_PROBE, timeout=timeout, check=False)
    return usable_python(result.stdout)


def open_session(
    entry: HostEntry,
    settings: Settings | None = None,
    transport: Transport | None = None,
    *,
    record: bool = True,
) -> HostSession:
    """Open a host: resolve its home, read its config, find an interpreter.

    The interpreter is the cached one where a bootstrap or probe left one,
    else the host is asked (`PYTHON_PROBE`): a host somebody else bootstrapped
    answers `status` and `host set` from here the moment it is registered.
    What was read is recorded in the registry's cache on the way past
    (`record`), so the offline listings say what the host said a moment ago;
    a poll that runs every few seconds passes `record=False`, because a
    dashboard must never write the registry.
    """
    transport = transport or transport_for(entry, settings)
    home = resolve_home(transport, entry)
    config_read = read_config(transport, home)
    python = entry.python or probe_python(transport)
    if python is None:
        raise RemoteError(
            entry.name,
            "open_session",
            f"host {entry.name!r} has no Python >= {PYTHON_FLOOR[0]}.{PYTHON_FLOOR[1]} to run "
            f"the package with.\nRun: gpuc host bootstrap {entry.name}",
        )
    if record and (config_read.document is not None or python != entry.python):
        update_cache(
            entry.name,
            config=config_read.document,
            python=None if python == entry.python else python,
        )
    return HostSession(entry, transport, home, python, config_read, record)


@dataclass
class Answered:
    """The host was reached; `payload` is what it printed for `verb`, or None
    when no verb was asked. `pod` is the provider's view of a rental's pod,
    and `pod_error` why the provider could not be asked -- an answer with one
    of those is still an answer, and still a failure to report."""

    session: HostSession
    payload: dict[str, Any] | None = None
    pod: Pod | None = None
    pod_error: str | None = None


@dataclass
class Unreachable:
    reason: str
    pod: Pod | None = None
    pod_error: str | None = None


@dataclass
class PodDead:
    """The provider still has the pod and nothing can run on it: a failure,
    and a registry entry kept for `gpuc host terminate` or `gpuc host remove`."""

    pod: Pod

    @property
    def reason(self) -> str:
        return f"pod {self.pod.id} is {self.pod.status}"


@dataclass
class PodGone:
    """The provider says the rental has ended: the state every rental reaches,
    not a failure. The caller that commands share forgets the entry."""

    pod_id: str
    pod: Pod | None = None

    @property
    def reason(self) -> str:
        status = "no longer exists" if self.pod is None else f"is {self.pod.status}"
        return f"pod {self.pod_id} {status}; this rental has ended"


Asked = Answered | Unreachable | PodDead | PodGone
"""What asking a host produces. Exactly one of four, decided once."""


def rental_state(
    entry: HostEntry, provider: Provider | None
) -> PodDead | PodGone | tuple[Pod | None, str | None]:
    """The provider's word on a rental's pod, before any ssh is tried.

    A pod the provider reports gone or dead is never dialled: the ssh would
    hang and then print a stack about a refused connection, which tells nobody
    anything. Otherwise the pod (or None for a host that is not rented) and
    why the provider could not be asked, which rides along as a failure
    rather than stopping the host being asked.
    """
    if provider is None or entry.rental is None:
        return None, None
    try:
        pod = provider.get(entry.rental.pod_id)
    except ProviderError as exc:
        return None, f"could not read pod {entry.rental.pod_id}: {exc}"
    if pod is None or provider.is_gone(pod):
        return PodGone(entry.rental.pod_id, pod)
    if provider.is_dead(pod):
        return PodDead(pod)
    return pod, None


def ask(
    entry: HostEntry,
    verb: str | None,
    settings: Settings | None = None,
    *,
    provider: Provider | None = None,
    session: HostSession | None = None,
    timeout: float = 60.0,
    check: bool = True,
    record: bool = False,
) -> Asked:
    """Ask a host `verb` (an on-host subcommand, or None for the session alone).

    A rental is looked up at its provider first, when one is given
    (`rental_state`). `check=False` for a verb whose refusal *is* the document
    -- a cancel of a finished job -- so the host's reason survives the exit
    code.
    """
    state = rental_state(entry, provider)
    if isinstance(state, (PodDead, PodGone)):
        return state
    pod, pod_error = state
    try:
        session = session or open_session(entry, settings, record=record)
        payload = session.host_json(verb, timeout=timeout, check=check) if verb else None
    except (RemoteError, TransportError, ConfigError, OSError) as exc:
        return Unreachable(reason_of(exc), pod, pod_error)
    if verb is not None and not isinstance(payload, dict):
        kind = type(payload).__name__
        return Unreachable(
            f"host {entry.name} answered `{verb}` with {kind}, not a JSON object", pod, pod_error
        )
    return Answered(session, payload, pod, pod_error)
