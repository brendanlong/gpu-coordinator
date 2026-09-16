"""nvidia-smi parsing: UUID/index mapping and utilization sampling.

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
    index: int
    uuid: str


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


def _as_int(cell: str, field: str) -> int:
    """nvidia-smi prints `[N/A]` or `[Not Supported]` instead of failing."""
    try:
        return int(cell)
    except ValueError as exc:
        raise GpuError(f"nvidia-smi reported {field}={cell!r}, which is not a number") from exc


def _as_float(cell: str, field: str) -> float:
    try:
        return float(cell)
    except ValueError as exc:
        raise GpuError(f"nvidia-smi reported {field}={cell!r}, which is not a number") from exc


def list_gpus(smi: SmiRunner = run_nvidia_smi) -> list[Gpu]:
    return [Gpu(_as_int(index, "index"), uuid) for index, uuid in _query(["index", "uuid"], smi)]


def driver_version(smi: SmiRunner = run_nvidia_smi) -> str:
    rows = _query(["driver_version"], smi)
    if not rows:
        raise GpuError("nvidia-smi reported no GPUs, so no driver version")
    return rows[0][0]


def index_uuids(smi: SmiRunner = run_nvidia_smi) -> dict[str, str]:
    """`{index: uuid}`, as nvidia-smi numbers this host's cards right now."""
    return {str(gpu.index): gpu.uuid for gpu in list_gpus(smi)}


def is_index(entry: str) -> bool:
    """Is this owned entry an nvidia-smi index rather than a UUID?"""
    return entry.isdigit()


def _against_table(entries: Sequence[str], table: dict[str, str]) -> tuple[list[str], list[str]]:
    present = set(table.values())
    resolved: list[str] = []
    missing: list[str] = []
    for entry in entries:
        uuid = table.get(entry) if is_index(entry) else (entry if entry in present else None)
        if uuid is None:
            missing.append(entry)
        elif uuid not in resolved:
            resolved.append(uuid)
    return resolved, missing


def resolve_owned(
    owned: Sequence[str], smi: SmiRunner = run_nvidia_smi
) -> tuple[list[str], list[str]]:
    """Owned entries -> (the UUIDs that are on this host, the entries that are not).

    Ownership on a shared box is an agreement written in nvidia-smi numbering
    -- "you have 2 and 3" -- so an index has to be sayable, and it is stored
    exactly as it was given. Everything downstream is UUIDs: an index names
    whichever card the driver is calling 2 *this boot*, and a job pinned with
    `CUDA_VISIBLE_DEVICES=2` after a renumber would quietly train on somebody
    else's GPU. So the mapping is redone from the host each dispatch pass, and
    an entry that does not resolve is simply not handed out.

    Owning UUIDs only needs no lookup at all, and does not get one: that is
    every pod and most boxes, and one nvidia-smi exec per pass for a mapping
    that is the identity would be pure cost.
    """
    entries = list(owned)
    if not entries or not any(is_index(entry) for entry in entries):
        return entries, []
    return _against_table(entries, index_uuids(smi))


def describe_table(smi: SmiRunner = run_nvidia_smi) -> str:
    """`0=GPU-..., 1=GPU-...`, for an error that has to say what is here."""
    try:
        table = index_uuids(smi)
    except GpuError as exc:
        return f"(nvidia-smi could not be read: {exc})"
    return ", ".join(f"{index}={uuid}" for index, uuid in sorted(table.items())) or "(no GPUs)"


def resolve_present(
    entries: Sequence[str], what: str = "assigned GPUs", smi: SmiRunner = run_nvidia_smi
) -> list[str]:
    """`resolve_owned`, for the places where a card was already promised.

    A job's assignment and a health check of `config.gpus` both describe cards
    that are supposed to be here, so an entry that names nothing is a failure
    rather than one to quietly skip. Entries are indices or UUIDs, the same as
    everywhere else, so the message names the *entry* that could not be found.

    Unlike `resolve_owned` this always asks the host, UUID entries included: a
    UUID that the driver no longer reports is exactly what has to fail here.
    """
    entries = list(entries)
    if not entries:
        return []
    resolved, missing = _against_table(entries, index_uuids(smi))
    if missing:
        raise GpuError(
            f"{what} not present on this host: {', '.join(missing)}; "
            f"nvidia-smi reports: {describe_table(smi)}"
        )
    return resolved


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
    one: reporting 0% there feeds the low-util watchdog a floor-breaking value
    every tick and kills a perfectly busy job. The runner catches this and
    records the sample as unknown.
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
