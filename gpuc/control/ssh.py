"""`gpuc ssh`: a shell on a host, or in a job's workdir, with our own options.

Everything else here talks to hosts through the transport, which is exactly
what makes a host hard to poke at by hand: the key, port, known_hosts file and
ControlMaster socket are all gpuc's, and none of them are in `~/.ssh/config`.
This turns those same options into an interactive session -- or one command, or
a line you can copy.
"""

from __future__ import annotations

import shlex

from gpuc.control.transport import CommandResult, Transport

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


def interactive_argv(
    transport: Transport, directory: str, fallback: str | None = None
) -> list[str]:
    """The argv to `exec` for an interactive session in `directory`."""
    return transport.interactive_argv(login_command(directory, fallback))


def shell_command(directory: str, command: str, fallback: str | None = None) -> str:
    """One command line, in `directory`, through a login shell.

    A command *line*, not an argv: `gpuc ssh <job> -- 'ls | wc -l'` has to mean
    the pipeline, and exec'ing the words directly would look for a program
    called `|`. Login, because this is the hand version of what the job itself
    gets -- a pod's sshd hands out a PATH with neither uv nor the aws CLI on
    it. The same workdir and the same fallback as the interactive session, so
    the two never land in different places.
    """
    return f"{quoted_cd(directory, fallback)} && exec {DEFAULT_SHELL} -lc {shlex.quote(command)}"


def command_argv(
    transport: Transport, directory: str, command: str, fallback: str | None = None
) -> list[str]:
    """The argv for one non-interactive command, for `--print` to show."""
    return transport.argv(shell_command(directory, command, fallback))


def run_command(
    transport: Transport,
    directory: str,
    command: str,
    fallback: str | None = None,
    *,
    timeout: float = 3600.0,
) -> CommandResult:
    """Run one command in `directory` on the host, whatever it exits."""
    return transport.run(shell_command(directory, command, fallback), timeout=timeout, check=False)


def print_line(argv: list[str]) -> str:
    return shlex.join(argv)
