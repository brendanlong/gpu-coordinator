"""`gpuc ssh`: a shell on a host, or in a job's workdir, with our own options.

Everything else here talks to hosts through the transport, which is exactly
what makes a host hard to poke at by hand: the key, port, known_hosts file and
ControlMaster socket are all gpuc's, and none of them are in `~/.ssh/config`.
This turns those same options into an interactive session -- or one command, or
a line you can copy.
"""

from __future__ import annotations

import os
import shlex

from gpuc.control.transport import CommandResult, SshTransport, Transport

DEFAULT_SHELL = "/bin/bash"


def quoted_cd(directory: str, fallback: str | None = None) -> str:
    """`cd` to a directory whose path may still contain `$HOME`.

    Double quotes, not `shlex.quote`: `HostEntry.remote_home` is deliberately
    unexpanded (`$HOME/.gpuc`) so the *host* resolves it, and single-quoting
    would land us in a directory with a literal dollar in its name. The
    fallback is for a job whose `workdir/` was cleaned away: the job dir itself
    still has the log and the state.
    """
    first = f'cd "{directory}"'
    if fallback is None:
        return first
    return f'{first} 2>/dev/null || cd "{fallback}"'


def login_command(directory: str, fallback: str | None = None) -> str:
    """Land in `directory` and hand over to the user's own login shell."""
    return f'{quoted_cd(directory, fallback)}; exec "${{SHELL:-{DEFAULT_SHELL}}}" -l'


def interactive_options(transport: SshTransport) -> list[str]:
    """The transport's ssh options, minus `BatchMode`.

    BatchMode is right for every automated call -- a prompt there would hang a
    polling loop forever -- and wrong here, where the user may well need to
    type a key passphrase.
    """
    options = transport.ssh_options()
    kept: list[str] = []
    index = 0
    while index < len(options):
        if (
            options[index] == "-o"
            and index + 1 < len(options)
            and options[index + 1].startswith("BatchMode")
        ):
            index += 2
            continue
        kept.append(options[index])
        index += 1
    return kept


def interactive_argv(
    transport: Transport, directory: str, fallback: str | None = None
) -> list[str]:
    """The argv to `exec` for an interactive session.

    `-t` forces a tty: without it the remote shell has no job control, no
    prompt and no `clear`, which is not a shell anyone wants.
    """
    if isinstance(transport, SshTransport):
        return [
            "ssh",
            *interactive_options(transport),
            "-t",
            transport.target,
            login_command(directory, fallback),
        ]
    return ["bash", "-lc", login_command(directory, fallback)]


def command_argv(transport: Transport, directory: str, command: str) -> list[str]:
    """The argv for one non-interactive command, for `--print` to show."""
    remote = f"{quoted_cd(directory)} && {command}"
    if isinstance(transport, SshTransport):
        return transport.ssh_argv(remote)
    return ["bash", "-c", remote]


def run_command(
    transport: Transport, directory: str, command: str, *, timeout: float = 3600.0
) -> CommandResult:
    """Run one command in `directory` on the host, whatever it exits."""
    return transport.run(f"{quoted_cd(directory)} && {command}", timeout=timeout, check=False)


def local_directory(directory: str, fallback: str | None = None) -> str:
    """The first of these paths that exists here, `$HOME` expanded."""
    for candidate in (directory, fallback):
        if candidate is None:
            continue
        path = os.path.expanduser(os.path.expandvars(candidate))
        if os.path.isdir(path):
            return path
    return os.path.expanduser("~")


def print_line(argv: list[str]) -> str:
    return shlex.join(argv)
