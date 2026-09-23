"""What each of a host's GPUs actually is, recorded once so listings can say.

The registry stores what the user typed -- a UUID, which is exactly right for
assignment and useless to read, or an nvidia-smi index, which is readable and
only means anything against a particular boot's numbering. Neither says whether
those are A40s or 3060 Tis, so this is filled in by bootstrap and `gpuc host
probe` and tolerated absent everywhere (an old registry, a host with no
nvidia-smi, a pod that has not booted yet).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from pydantic import BaseModel

from gpuc.control.transport import Transport

SMI_QUERY = "nvidia-smi --query-gpu=index,uuid,name,memory.total --format=csv,noheader,nounits"
MIB_PER_GB = 1024.0


class GpuInfo(BaseModel):
    name: str = ""
    vram_mib: int | None = None
    index: int | None = None
    """What nvidia-smi called this card when it was last looked at.

    Recorded because a host may own cards *by* index (`--gpus 2,3`), and a
    listing has to be able to show which UUID that was without asking the host.
    Never used to assign anything: the index is a label this boot, the UUID is
    the card."""

    def label(self) -> str:
        return f"{self.name or '?'} {vram_text(self.vram_mib)}".strip()


def vram_text(vram_mib: int | None) -> str:
    """Rounded to whole GB: a card marketed as 48 GB reports 46068 MiB."""
    if not vram_mib:
        return ""
    return f"{round(vram_mib / MIB_PER_GB):g} GB"


def parse_smi(text: str) -> dict[str, GpuInfo]:
    """`index, uuid, name, memory` rows, with or without a `MiB` unit suffix.

    The leading index is optional only because `gpuc host probe` hands rows
    here from its own query, which asks for it, and `discover` from one that
    does too; a row without one is simply a card with no index recorded.
    """
    found: dict[str, GpuInfo] = {}
    for line in text.splitlines():
        cells = [cell.strip() for cell in line.split(",")]
        index: int | None = None
        if cells and cells[0].isdigit():
            index, cells = int(cells[0]), cells[1:]
        if len(cells) < 2 or not cells[0].startswith("GPU-"):
            continue
        memory = cells[2].split()[0] if len(cells) > 2 and cells[2] else ""
        found[cells[0]] = GpuInfo(
            name=cells[1],
            vram_mib=int(float(memory)) if memory.replace(".", "", 1).isdigit() else None,
            index=index,
        )
    return found


def discover(transport: Transport) -> dict[str, GpuInfo]:
    """Never raises: a host with no driver simply has nothing to say about it."""
    try:
        result = transport.run(f"{SMI_QUERY} 2>/dev/null", check=False)
    except Exception:
        return {}
    return parse_smi(result.stdout) if result.returncode == 0 else {}


def uuid_of(owned: str, info: Mapping[str, GpuInfo]) -> str | None:
    """The UUID an owned entry names, as far as the recorded info can tell.

    A UUID is itself; an index is whichever recorded card carried that index
    when the host was last probed. Offline and best-effort on purpose -- the
    authority on today's numbering is the host, and it answers `status`.
    """
    if not owned.isdigit():
        return owned
    index = int(owned)
    for uuid, entry in info.items():
        if entry.index == index:
            return uuid
    return None


def summarize(owned: Sequence[str], info: Mapping[str, GpuInfo]) -> str:
    """`2x NVIDIA A40 48 GB`, or several groups when the cards differ."""
    groups: dict[str, int] = {}
    for item in owned:
        uuid = uuid_of(item, info)
        entry = info.get(uuid) if uuid else None
        label = entry.label() if entry else "unknown GPU"
        groups[label] = groups.get(label, 0) + 1
    return ", ".join(f"{count}x {label}" for label, count in groups.items())


def rows(
    owned: Sequence[str], info: Mapping[str, GpuInfo], indices: Mapping[str, int] | None = None
) -> list[tuple[str, str, str, str]]:
    """`(index, name, vram, uuid)` per owned card, in the host's own order.

    `indices` is the host's *current* numbering when a caller has it (`gpuc
    status` asks the host); without it the index recorded at bootstrap is shown,
    and `?` when even that is unknown.
    """
    out: list[tuple[str, str, str, str]] = []
    for item in owned:
        uuid = uuid_of(item, info)
        entry = (info.get(uuid) if uuid else None) or GpuInfo()
        index = (indices or {}).get(uuid or "", entry.index)
        if index is None and item.isdigit():
            index = int(item)
        out.append(
            (
                "?" if index is None else str(index),
                entry.name or "?",
                vram_text(entry.vram_mib),
                uuid or "?",
            )
        )
    return out
