"""`--json` on a command: one document on stdout, whatever happened.

The rules every `--json` command obeys, so a caller can read them all the same
way:

- stdout holds exactly one JSON object and nothing else. Progress, warnings and
  the notes the text output prints inline go to stderr instead.
- the object always carries `schema_version`, and unknown keys are added over
  time -- ignore the ones you do not know.
- a command that failed prints `{"schema_version", "error", "exit_code"}` and
  exits with that code, so `error` is the one key a consumer has to check.
  `errors` (plural) is different: it is per-host or per-job trouble the command
  survived, and it never implies a non-zero exit on its own.
"""

from __future__ import annotations

import json
import sys
from typing import Any

from gpuc.host.jobs import SCHEMA_VERSION


def emit(payload: dict[str, Any]) -> None:
    print(json.dumps({"schema_version": SCHEMA_VERSION, **payload}, indent=2))


def emit_error(message: str, exit_code: int) -> None:
    emit({"error": message, "exit_code": exit_code})


def note(message: str) -> None:
    """Progress from a step that would otherwise print onto the document."""
    print(message, file=sys.stderr)
