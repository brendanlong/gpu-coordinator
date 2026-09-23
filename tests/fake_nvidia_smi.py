"""The one fake `nvidia-smi`: a table of cards, answering `--query-gpu`.

`render` is the whole behaviour. `conftest.fake_smi` calls it in-process for
the code that takes an injectable `smi`, and `conftest.install_fake_nvidia_smi`
puts a wrapper on `PATH` that runs this file as a script for everything that
execs the real binary -- the dispatcher and runner as subprocesses, the probe
script, the health check. One set of rows either way.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping, Sequence

NAME = "Fake A40"
MEMORY_MIB = "46068"
DRIVER = "580.173.02"
UNITS = {"memory.total": " MiB", "memory.used": " MiB", "utilization.gpu": " %"}
UUIDS_ENV = "FAKE_SMI_UUIDS"


def render(
    argv: Sequence[str],
    uuids: Sequence[str],
    utilization: Mapping[str, float] | None = None,
    memory_used: Mapping[str, float] | None = None,
) -> str:
    """What `nvidia-smi argv` prints for a box holding `uuids`.

    `--format=` is honoured rather than assumed: real nvidia-smi prints a
    header row unless `noheader` is asked for, and a caller that forgets it
    gets a field-name line where it expected data. A fake that never emits
    one would hide exactly that bug. Only `--query-gpu` is answered; anything
    else is the error the real binary would not give, loudly.
    """
    query = next((a for a in argv if a.startswith("--query-gpu=")), None)
    if query is None:
        raise SystemExit("fake nvidia-smi: only --query-gpu is supported")
    fields = query.split("=", 1)[1].split(",")
    fmt = next((a for a in argv if a.startswith("--format=")), "--format=csv")
    options = fmt.split("=", 1)[1].split(",")
    nounits = "nounits" in options
    selected = list(uuids)
    if "-i" in argv:
        wanted = argv[list(argv).index("-i") + 1].split(",")
        selected = [u for u in uuids if u in wanted]
    rows: list[str] = []
    for index, uuid in enumerate(uuids):
        if uuid not in selected:
            continue
        cells: list[str] = []
        for field in fields:
            if field == "index":
                value = str(index)
            elif field == "uuid":
                value = uuid
            elif field == "name":
                value = NAME
            elif field == "driver_version":
                value = DRIVER
            elif field == "memory.total":
                value = MEMORY_MIB
            elif field == "utilization.gpu":
                value = f"{(utilization or {}).get(uuid, 0.0):g}"
            elif field == "memory.used":
                value = f"{(memory_used or {}).get(uuid, 0.0):g}"
            else:
                value = ""
            cells.append(value + ("" if nounits or not value else UNITS.get(field, "")))
        rows.append(", ".join(cells))
    if "noheader" not in options:
        rows.insert(0, ", ".join(fields))
    return "\n".join(rows) + "\n"


if __name__ == "__main__":
    uuids = [u for u in os.environ.get(UUIDS_ENV, "").split(",") if u]
    sys.stdout.write(render(sys.argv[1:], uuids))
