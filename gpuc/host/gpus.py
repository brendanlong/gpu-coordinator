"""nvidia-smi parsing: UUID/index mapping, utilization sampling, and who is
using a card we do not own.

Every entry point takes an injectable ``smi`` callable so tests run on hosts
with no GPU, and so the dispatcher can share one fake in integration tests.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass

SmiRunner = Callable[[list[str]], str]

DEFAULT_TIMEOUT_S = 20.0


class GpuError(RuntimeError):
    pass


def run_nvidia_smi(args: list[str], timeout: float = DEFAULT_TIMEOUT_S) -> str:
    command = ["nvidia-smi", *args]
    try:
        proc = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
    except FileNotFoundError as exc:
        raise GpuError(f"nvidia-smi not found on PATH (command: {' '.join(command)})") from exc
    except subprocess.TimeoutExpired as exc:
        raise GpuError(f"nvidia-smi timed out after {timeout}s: {' '.join(command)}") from exc
    if proc.returncode != 0:
        tail = "\n".join((proc.stderr or proc.stdout).strip().splitlines()[-5:])
        raise GpuError(f"{' '.join(command)} exited {proc.returncode}: {tail}")
    return proc.stdout


@dataclass(frozen=True)
class Gpu:
    """One row of the table: what the driver calls a card this boot (`index`),
    what it is for ever (`uuid`), and what it is (`name`, `memory_mib`).
    `index` is None only for a row rebuilt from a registry's cache that never
    recorded one."""

    index: int | None
    uuid: str
    name: str = ""
    memory_mib: int | None = None


TABLE_FIELDS = ("index", "uuid", "name", "memory.total")
"""The one `--query-gpu` both halves read cards with, so there is one parser."""


def parse_table(text: str) -> list[Gpu]:
    """`index, uuid, name, memory.total` rows, with or without a unit suffix.

    The one nvidia-smi parser: the host queries it live (`list_gpus`), the
    probe reads it off its own shell output (`--format=csv,noheader`, units
    on), and the registry's cache is rebuilt into the same rows. Anything that
    is not a row -- a MOTD, `nvidia-smi: command not found` -- is skipped, so
    a driver that is missing reads as no cards rather than a parse error.
    """
    rows: list[Gpu] = []
    for line in text.splitlines():
        cells = [cell.strip() for cell in line.split(",")]
        if len(cells) < 2 or not cells[1].startswith("GPU-"):
            continue
        if not cells[0].isdigit():
            continue
        memory = cells[3].split()[0] if len(cells) > 3 and cells[3] else ""
        rows.append(
            Gpu(
                int(cells[0]),
                cells[1],
                cells[2] if len(cells) > 2 else "",
                int(float(memory)) if memory.replace(".", "", 1).isdigit() else None,
            )
        )
    return rows


def _query(fields: list[str], smi: SmiRunner, extra: list[str] | None = None) -> list[list[str]]:
    args = [f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits", *(extra or [])]
    rows: list[list[str]] = []
    for line in smi(args).splitlines():
        line = line.strip()
        if not line:
            continue
        cells = [cell.strip() for cell in line.split(",")]
        if len(cells) != len(fields):
            raise GpuError(
                f"nvidia-smi --query-gpu={','.join(fields)} returned {len(cells)} field(s) "
                f"in {line!r}, expected {len(fields)}"
            )
        rows.append(cells)
    return rows


def _as_float(cell: str, field: str) -> float:
    try:
        return float(cell)
    except ValueError as exc:
        raise GpuError(f"nvidia-smi reported {field}={cell!r}, which is not a number") from exc


def _maybe_float(cell: str) -> float | None:
    """`[N/A]` and `[Not Supported]` as None, for a reading that may be absent."""
    try:
        return float(cell)
    except ValueError:
        return None


def list_gpus(smi: SmiRunner = run_nvidia_smi) -> list[Gpu]:
    """Every card the driver reports right now, as `TABLE_FIELDS` rows."""
    text = smi([f"--query-gpu={','.join(TABLE_FIELDS)}", "--format=csv,noheader,nounits"])
    table = parse_table(text)
    if not table and text.strip():
        raise GpuError(
            f"nvidia-smi --query-gpu={','.join(TABLE_FIELDS)} returned no card rows: "
            f"{text.strip()[-200:]!r}"
        )
    return table


def driver_version(smi: SmiRunner = run_nvidia_smi) -> str:
    rows = _query(["driver_version"], smi)
    if not rows:
        raise GpuError("nvidia-smi reported no GPUs, so no driver version")
    return rows[0][0]


@dataclass(frozen=True)
class Resolution:
    """What a host's owned and shared entries name against one table.

    `owned` and `shared` are UUIDs, each card once and in entry order, with a
    shared entry that names an owned card left out of `shared`: owning is
    the stronger claim, and the dispatcher hands the card out as its own.
    `missing` and `shared_missing` are the entries no card answers to, as
    spelled. `duplicates` are the entries that named a card already named --
    an index and its own UUID, or a card in both lists -- also as spelled,
    which is how a health check refuses them and a listing explains them.
    """

    owned: list[str]
    shared: list[str]
    missing: list[str]
    shared_missing: list[str]
    duplicates: list[str]


def resolve(owned: Sequence[str], table: Sequence[Gpu], shared: Sequence[str] = ()) -> Resolution:
    """Entries -- nvidia-smi indices, UUIDs, or a mix -- against `table`.

    Pure, and the one rule: the dispatcher and the runner apply it to what
    nvidia-smi says now, the health check to the same, and the control side
    to the probe's rows or the registry's cache of them. An absent UUID is
    missing here exactly as an absent index is; nothing passes an entry
    through unresolved, so a card the driver stopped reporting is never
    handed out to fail inside a job.
    """
    by_index = {str(gpu.index): gpu.uuid for gpu in table if gpu.index is not None}
    present = {gpu.uuid for gpu in table}

    def lookup(entry: str) -> str | None:
        if entry.isdigit():
            return by_index.get(entry)
        return entry if entry in present else None

    owned_uuids: list[str] = []
    shared_uuids: list[str] = []
    missing: list[str] = []
    shared_missing: list[str] = []
    duplicates: list[str] = []
    for entry in owned:
        uuid = lookup(entry)
        if uuid is None:
            missing.append(entry)
        elif uuid in owned_uuids:
            duplicates.append(entry)
        else:
            owned_uuids.append(uuid)
    for entry in shared:
        uuid = lookup(entry)
        if uuid is None:
            shared_missing.append(entry)
        elif uuid in owned_uuids or uuid in shared_uuids:
            duplicates.append(entry)
        else:
            shared_uuids.append(uuid)
    return Resolution(owned_uuids, shared_uuids, missing, shared_missing, duplicates)


def describe_table(table: Sequence[Gpu]) -> str:
    """`0=GPU-..., 1=GPU-...`, for an error that has to say what is here."""
    return ", ".join(f"{gpu.index}={gpu.uuid}" for gpu in table) or "(no GPUs)"


def sample_utilization(uuids: Sequence[str], smi: SmiRunner = run_nvidia_smi) -> dict[str, float]:
    if not uuids:
        return {}
    rows = _query(["uuid", "utilization.gpu"], smi, extra=["-i", ",".join(uuids)])
    out: dict[str, float] = {}
    for uuid, util in rows:
        if uuid in uuids:
            out[uuid] = _as_float(util, f"utilization.gpu for {uuid}")
    return out


def mean_utilization(uuids: Sequence[str], smi: SmiRunner = run_nvidia_smi) -> float:
    """Mean `utilization.gpu` over `uuids`. No GPUs asked about means 0%.

    Asking about GPUs and getting nothing back is a *failed sample*, not an idle
    one: reporting 0% there would show a perfectly busy job as idle. The runner
    catches this and records the sample as unknown.
    """
    if not uuids:
        return 0.0
    samples = sample_utilization(uuids, smi)
    if not samples:
        raise GpuError(
            f"nvidia-smi reported no utilization for any of {', '.join(uuids)}; "
            f"treating this as a failed sample rather than 0% util"
        )
    return sum(samples.values()) / len(samples)


@dataclass(frozen=True)
class Usage:
    """What somebody -- anybody -- is doing with one card right now.

    Both readings are optional because nvidia-smi answers `[N/A]` or
    `[Not Supported]` on some cards and in some containers, and an absent
    reading is the one thing that must never be read as an idle one.
    """

    uuid: str
    memory_mib: float | None
    utilization_pct: float | None

    @property
    def unused(self) -> bool:
        """Nothing at all is on this card: no memory held, no work running.

        Memory is the stronger half. A CUDA context costs hundreds of MiB the
        moment it is created and holds them between steps, so a card at 0%
        util *and* 0 MiB has no process on it -- while util alone dips to zero
        between epochs of somebody else's training run.

        A None is not zero: see the dataclass docstring.
        """
        return self.memory_mib == 0.0 and self.utilization_pct == 0.0

    def describe(self) -> str:
        memory = "?" if self.memory_mib is None else f"{self.memory_mib:.0f}"
        util = "?" if self.utilization_pct is None else f"{self.utilization_pct:.0f}"
        return f"{memory} MiB, {util}% util"


def sample_usage(uuids: Sequence[str], smi: SmiRunner = run_nvidia_smi) -> dict[str, Usage]:
    """Memory held and utilization, per card. Cards nvidia-smi skipped are absent."""
    if not uuids:
        return {}
    wanted = set(uuids)
    rows = _query(["uuid", "memory.used", "utilization.gpu"], smi, extra=["-i", ",".join(uuids)])
    return {
        uuid: Usage(uuid, _maybe_float(memory), _maybe_float(util))
        for uuid, memory, util in rows
        if uuid in wanted
    }


USAGE_FIELDS = ("memory.used", "utilization.gpu")


def snapshot(smi: SmiRunner = run_nvidia_smi) -> tuple[list[Gpu], dict[str, Usage]]:
    """Every card and what is on each, from one nvidia-smi call.

    For `status`, which wants both at the same instant: a card that breaks
    nvidia-smi breaks the listing anyway, so a second call for the readings
    only adds a window for the two to disagree. Per row, not per call: a row
    that does not parse costs that card its reading and nothing else. (No
    JSON to lean on instead -- nvidia-smi's only structured output is the
    full `-q -x` XML dump, far slower than a `--query-gpu`.)
    """
    fields = [*TABLE_FIELDS, *USAGE_FIELDS]
    text = smi([f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits"])
    table = parse_table(text)
    if not table and text.strip():
        raise GpuError(
            f"nvidia-smi --query-gpu={','.join(fields)} returned no card rows: "
            f"{text.strip()[-200:]!r}"
        )
    usage: dict[str, Usage] = {}
    for line in text.splitlines():
        cells = [cell.strip() for cell in line.split(",")]
        if len(cells) == len(fields) and cells[1].startswith("GPU-"):
            usage[cells[1]] = Usage(cells[1], _maybe_float(cells[4]), _maybe_float(cells[5]))
    return table, usage


def usage_or_nothing(
    uuids: Sequence[str], smi: SmiRunner = run_nvidia_smi
) -> tuple[dict[str, Usage], str | None]:
    """Every reading nvidia-smi gives for `uuids`, and why there are none.

    The one place "nvidia-smi would not answer" turns into "we know nothing
    about these cards", for the dispatcher deciding whether to borrow one.
    `status` reads the same `Usage` from `snapshot`, so the verdict on a
    reading -- `Usage.unused` -- is still written once.
    """
    if not uuids:
        return {}, None
    try:
        return sample_usage(uuids, smi), None
    except GpuError as exc:
        return {}, str(exc)


def unused_gpus(
    uuids: Sequence[str], smi: SmiRunner = run_nvidia_smi
) -> tuple[list[str], dict[str, str]]:
    """Split `uuids` into the cards nobody is using and the rest, with a reason.

    For shared cards, which gpuc may only take while their real owner is not on
    them. Every way of not knowing -- nvidia-smi failing, a card it did not
    answer about, a reading it would not give -- lands in the second half: this
    decides whether to run a job on somebody else's GPU, so the absence of
    evidence has to count against.
    """
    samples, failure = usage_or_nothing(uuids, smi)
    unused: list[str] = []
    in_use: dict[str, str] = {}
    for uuid in uuids:
        usage = samples.get(uuid)
        if usage is None:
            in_use[uuid] = (
                f"could not be read ({failure})"
                if failure
                else "nvidia-smi reported nothing about it"
            )
        elif usage.unused:
            unused.append(uuid)
        else:
            in_use[uuid] = usage.describe()
    return unused, in_use
