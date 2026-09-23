"""What each of a host's GPUs actually is, recorded once so listings can say.

The registry stores what the user typed -- a UUID, which is exactly right for
assignment and useless to read, or an nvidia-smi index, which is readable and
only means anything against a particular boot's numbering. Neither says whether
those are A40s or 3060 Tis, so this is filled in by `gpuc host add` and `gpuc
host probe` and tolerated absent everywhere (an old registry, a host with no
nvidia-smi, a pod that has not booted yet).

The rows come from the host's own parser (`gpus.parse_table`) and are resolved
offline by the host's own rule (`gpus.resolve`) over a table rebuilt from the
cache: the same two functions the dispatcher uses live.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from pydantic import BaseModel

from gpuc.host import gpus

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


def from_table(table: Sequence[gpus.Gpu]) -> dict[str, GpuInfo]:
    """The rows as the registry stores them, keyed by UUID."""
    return {
        gpu.uuid: GpuInfo(name=gpu.name, vram_mib=gpu.memory_mib, index=gpu.index) for gpu in table
    }


def table_of(info: Mapping[str, GpuInfo]) -> list[gpus.Gpu]:
    """The cache as the table `gpus.resolve` reads, for an offline answer."""
    return [gpus.Gpu(entry.index, uuid, entry.name, entry.vram_mib) for uuid, entry in info.items()]


def uuid_of(owned: str, info: Mapping[str, GpuInfo]) -> str | None:
    """The UUID an owned entry names, as far as the recorded info can tell.

    Offline and best-effort on purpose: the cache can say what a card *is*,
    never whether it is there, and presence is the host's answer (`status`,
    the health check, the dispatcher's pass -- all `gpus.resolve` over a live
    table). So a UUID names itself here whether or not the cache knows it,
    and only an index is looked up, through the same rule.
    """
    if not owned.isdigit():
        return owned
    resolved = gpus.resolve([owned], table_of(info)).owned
    return resolved[0] if resolved else None


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
    status` asks the host); without it the index recorded at the last probe is
    shown, and `?` when even that is unknown.
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
