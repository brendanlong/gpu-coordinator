"""What each of a host's GPUs actually is, recorded once so listings can say.

The registry stores UUIDs, which are exactly right for assignment and useless
to read: `gpus=2` says nothing about whether those are A40s or 3060 Tis. This
is filled in by bootstrap and `gpuc host probe`, tolerated absent everywhere
(an old registry, a host with no nvidia-smi, a pod that has not booted yet).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from pydantic import BaseModel

from gpuc.control.transport import Transport

SMI_QUERY = "nvidia-smi --query-gpu=uuid,name,memory.total --format=csv,noheader,nounits"
MIB_PER_GB = 1024.0


class GpuInfo(BaseModel):
    name: str = ""
    vram_mib: int | None = None

    def label(self) -> str:
        return f"{self.name or '?'} {vram_text(self.vram_mib)}".strip()


def vram_text(vram_mib: int | None) -> str:
    """Rounded to whole GB: a card marketed as 48 GB reports 46068 MiB."""
    if not vram_mib:
        return ""
    return f"{round(vram_mib / MIB_PER_GB):g} GB"


def parse_smi(text: str) -> dict[str, GpuInfo]:
    """`uuid, name, memory` rows, with or without a `MiB` unit suffix."""
    found: dict[str, GpuInfo] = {}
    for line in text.splitlines():
        cells = [cell.strip() for cell in line.split(",")]
        if len(cells) < 2 or not cells[0].startswith("GPU-"):
            continue
        memory = cells[2].split()[0] if len(cells) > 2 and cells[2] else ""
        found[cells[0]] = GpuInfo(
            name=cells[1],
            vram_mib=int(float(memory)) if memory.replace(".", "", 1).isdigit() else None,
        )
    return found


def discover(transport: Transport) -> dict[str, GpuInfo]:
    """Never raises: a host with no driver simply has nothing to say about it."""
    try:
        result = transport.run(f"{SMI_QUERY} 2>/dev/null", check=False)
    except Exception:
        return {}
    return parse_smi(result.stdout) if result.returncode == 0 else {}


def summarize(uuids: Sequence[str], info: Mapping[str, GpuInfo]) -> str:
    """`2x NVIDIA A40 48 GB`, or several groups when the cards differ."""
    groups: dict[str, int] = {}
    for uuid in uuids:
        entry = info.get(uuid)
        label = entry.label() if entry else "unknown GPU"
        groups[label] = groups.get(label, 0) + 1
    return ", ".join(f"{count}x {label}" for label, count in groups.items())


def rows(uuids: Sequence[str], info: Mapping[str, GpuInfo]) -> list[tuple[str, str, str]]:
    """`(name, vram, uuid)` per GPU, in the host's own order."""
    out: list[tuple[str, str, str]] = []
    for uuid in uuids:
        entry = info.get(uuid) or GpuInfo()
        out.append((entry.name or "?", vram_text(entry.vram_mib), uuid))
    return out
