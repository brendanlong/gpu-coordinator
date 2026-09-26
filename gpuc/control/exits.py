"""The exit codes, and the one table that turns them into HTTP statuses.

Its own module because both ends of the tool read it: `actions` maps a
command's outcome to one of these, `cli.main` exits with it, and the web
dashboard answers with the status beside it. Nothing else decides a code.
"""

from __future__ import annotations

from http import HTTPStatus

EXIT_OK = 0
"""Everything the command was asked to do happened."""
EXIT_ERROR = 1
"""Some part of the command failed: a transport error, a provider error, a
refused submit, a host that could not be read. Whatever did work is reported
anyway -- one unreachable host never costs the others their status."""
EXIT_USAGE = 2
"""The command line itself was wrong (argparse uses this too)."""
EXIT_LOCAL_STATE = 3
"""Local state -- the registry or the config file -- could not be read, so the
answer is unknown. Automation must not read this as `nothing is running`."""
EXIT_NOT_FOUND = 4
"""The named job or host does not exist."""
EXIT_INTERRUPTED = 130
"""A Ctrl-C out of a command that blocks. The shell's own convention for SIGINT.

Distinct from 0 because `gpuc wait` and `gpuc logs -f` exit with the *job's*
outcome, where 0 means it succeeded: a script must never read "the user got
bored" as "the job worked"."""

HTTP_FOR_EXIT: dict[int, HTTPStatus] = {
    EXIT_OK: HTTPStatus.OK,
    EXIT_ERROR: HTTPStatus.INTERNAL_SERVER_ERROR,
    EXIT_USAGE: HTTPStatus.BAD_REQUEST,
    EXIT_LOCAL_STATE: HTTPStatus.SERVICE_UNAVAILABLE,
    EXIT_NOT_FOUND: HTTPStatus.NOT_FOUND,
}
"""The CLI's exit codes, on the wire. A command that exits 1 -- an unreachable
host in a status, a refused set -- is a 500 carrying the same document the
CLI printed, so a consumer of either reads one rule."""


def http_status(exit_code: int) -> HTTPStatus:
    return HTTP_FOR_EXIT.get(exit_code, HTTPStatus.INTERNAL_SERVER_ERROR)
