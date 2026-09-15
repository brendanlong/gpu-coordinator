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


def assert_uuids_present(uuids: Sequence[str], smi: SmiRunner = run_nvidia_smi) -> None:
    if not uuids:
        return
    present = {gpu.uuid for gpu in list_gpus(smi)}
    missing = [u for u in uuids if u not in present]
    if missing:
        raise GpuError(
            f"assigned GPU UUIDs not present on this host: {', '.join(missing)}; "
            f"nvidia-smi reports: {', '.join(sorted(present)) or '(none)'}"
        )


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
    samples = sample_utilization(uuids, smi)
    if not samples:
        return 0.0
    return sum(samples.values()) / len(samples)
