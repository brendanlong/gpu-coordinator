"""`gpuc host probe`: what a host is, before we have installed anything on it.

Pure `sh` plus `python3`-if-present, because the whole point is to run against
a host where nothing has been bootstrapped yet. What needs the tool's own
rules -- the throughput floor, whether uv's cache shares gpuc home's
filesystem -- is the health check's, asked once the package is there.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from gpuc.control.config import HostEntry, Settings, transport_for
from gpuc.control.gpuinfo import GpuInfo, from_table
from gpuc.control.remote import usable_python
from gpuc.control.transport import Transport
from gpuc.host import gpus

SECTION_ORDER = [
    "system",
    "driver",
    "gpus",
    "disk",
    "home_fs",
    "killuserprocesses",
    "systemd_scope",
    "uv",
    "python3",
]


def probe_script() -> str:
    """The whole probe as one `sh` script."""
    return f"""
say() {{ echo "===$1==="; }}
say system
uname -srm; echo "user=$(id -un) home=$HOME shell=$SHELL"
say driver
nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>&1 | head -1 \\
  || echo "nvidia-smi not found"
say gpus
nvidia-smi --query-gpu={",".join(gpus.TABLE_FIELDS)} --format=csv,noheader 2>&1 \\
  || echo "nvidia-smi not found"
say disk
df -Ph "$HOME" | tail -1
say home_fs
df -T "$HOME" 2>/dev/null | tail -1 || stat -f -c '%n %T' "$HOME" 2>/dev/null \\
  || echo "unknown unknown"
say killuserprocesses
kup=$(loginctl show-user "$(id -un)" -p KillUserProcesses 2>&1 | head -1)
echo "${{kup:-unknown: no logind session for this user}}"
say systemd_scope
if systemd-run --user --scope -- true >/dev/null 2>&1; then echo yes; else echo no; fi
say uv
if [ -x "$HOME/.local/bin/uv" ]; then "$HOME/.local/bin/uv" --version; \\
elif command -v uv >/dev/null 2>&1; then uv --version; else echo "not installed"; fi
say python3
if command -v python3 >/dev/null 2>&1; then python3 -c 'import sys; print(sys.executable, \\
sys.version.split()[0])'; else echo "not installed"; fi
"""


OVERLAY_FS_TYPES = frozenset({"overlay", "overlayfs", "aufs"})
"""Filesystem types that mean "this is a container's throwaway upper layer"."""


@dataclass
class ProbeReport:
    host: str
    sections: dict[str, str]
    persistent_root: str | None = None
    """The root this host is *already* registered with, if any."""
    owned: list[str] = field(default_factory=list)
    """This host's `--gpus`, exactly as the registry stores them: nvidia-smi
    indices, UUIDs, or a mix.

    nvidia-smi lists every card in the box, and on a shared box most of them
    are somebody else's. A probe that does not say which is which invites
    reading the whole list as yours."""
    shared: list[str] = field(default_factory=list)
    """This host's `--shared-gpus`, the same way: cards it may borrow while
    nobody else is on them, which are neither ours nor none of our business."""

    @property
    def has_nvidia_smi(self) -> bool:
        return "not found" not in self.sections.get("driver", "not found")

    @property
    def table(self) -> list[gpus.Gpu]:
        """Every card in the box, through the host's own parser."""
        if not self.has_nvidia_smi:
            return []
        return gpus.parse_table(self.sections.get("gpus", ""))

    @property
    def cards(self) -> gpus.Resolution:
        """The assignment against the box, by the host's own rule -- so what
        the probe calls missing or doubled is what the health check will."""
        return gpus.resolve(self.owned, self.table, self.shared)

    def owns(self, gpu: gpus.Gpu) -> bool:
        return gpu.uuid in self.cards.owned

    @property
    def owned_rows(self) -> list[gpus.Gpu]:
        return [gpu for gpu in self.table if self.owns(gpu)]

    def shares(self, gpu: gpus.Gpu) -> bool:
        return gpu.uuid in self.cards.shared

    @property
    def shared_rows(self) -> list[gpus.Gpu]:
        return [gpu for gpu in self.table if self.shares(gpu)]

    @property
    def owned_missing(self) -> list[str]:
        """`--gpus` entries no card on this host answers to: a typo, a card
        this container was not given, or a renumbered driver. Nothing is
        missing on a host with no driver: there is no table to be missing from."""
        return self.cards.missing if self.has_nvidia_smi else []

    @property
    def shared_missing(self) -> list[str]:
        """`--shared-gpus` entries no card on this host answers to."""
        return self.cards.shared_missing if self.has_nvidia_smi else []

    @property
    def gpu_info(self) -> dict[str, GpuInfo]:
        """The `gpus` section as the registry stores it, keyed by UUID."""
        return from_table(self.table)

    @property
    def host_python(self) -> str | None:
        """An interpreter on this host that could run the on-host package now.

        The `python3` the probe found, when it is new enough. Enough to read a
        host's config, ask it for its status and change it -- so a host
        somebody else bootstrapped is usable from here the moment it is
        registered, rather than after a full bootstrap of our own. Bootstrap
        replaces it with the interpreter uv picks, which is the one the
        dispatcher runs under.
        """
        return usable_python(self.sections.get("python3", ""))

    @property
    def driver_version(self) -> str | None:
        line = self.sections.get("driver", "").strip().splitlines()
        return line[0].strip() if line and self.has_nvidia_smi else None

    @property
    def home_fs_type(self) -> str | None:
        """The filesystem type of ``$HOME``: field 2 of `df -T` (and of the
        `stat -f` fallback, which prints ``<path> <type>``)."""
        fields = self.sections.get("home_fs", "").split()
        return fields[1] if len(fields) > 1 else None

    @property
    def home_is_overlay(self) -> bool:
        return (self.home_fs_type or "").lower() in OVERLAY_FS_TYPES

    def render(self, *, all_gpus: bool = False) -> str:
        lines = [f"host {self.host}"]
        for key in SECTION_ORDER:
            value = self.sections.get(key, "(no output)")
            if key == "gpus":
                lines += self._gpu_lines(all_gpus)
                continue
            lines.append(f"  {key}: {value.strip() or '(no output)'}")
        lines += [f"  note: {note}" for note in self.notes]
        return "\n".join(lines)

    def _gpu_lines(self, all_gpus: bool) -> list[str]:
        """The `gpus` section: ours by default, the whole box with `--all-gpus`."""
        rows, owned = self.table, self.owned_rows
        if not rows:
            return ["  gpus:", f"    {self.sections.get('gpus', '').strip() or '(no output)'}"]
        # Nothing of ours to show is not a reason to show nothing: a host whose
        # assignment matches no card needs the whole list more than anybody.
        ours = owned + self.shared_rows
        everything = all_gpus or not ours
        partly = 0 < len(ours) < len(rows)
        hidden = " (--all-gpus lists the rest)" if partly and not everything else ""
        borrowed = f", {len(ours) - len(owned)} shared" if len(ours) > len(owned) else ""
        header = f"  gpus: {len(owned)} of {len(rows)} assigned to {self.host}{borrowed}{hidden}"
        lines = [header if self.owned or self.shared else "  gpus:"]
        for gpu in rows if everything else ours:
            memory = f"  {gpu.memory_mib} MiB" if gpu.memory_mib is not None else ""
            if self.owns(gpu):
                mine = "  (assigned)" if everything and partly else ""
            else:
                mine = "  (shared)" if self.shares(gpu) else ""
            lines.append(f"    [{gpu.index}] {gpu.name}{memory}  {gpu.uuid}{mine}")
        return lines

    @property
    def notes(self) -> list[str]:
        """What this host will do to a job unless somebody acts, in words."""
        notes: list[str] = []
        if not self.has_nvidia_smi:
            notes.append("no nvidia-smi, so this host cannot run jobs")
        notes += self._gpu_notes()
        if self.sections.get("killuserprocesses", "").endswith("=yes"):
            notes.append(
                "logind kills user processes at logout; the dispatcher will not "
                "survive your SSH session ending"
            )
        if self.sections.get("uv") == "not installed":
            notes.append(f"uv is missing; `gpuc host bootstrap {self.host}` installs it")
        return notes

    def _gpu_notes(self) -> list[str]:
        """Which cards in this box are ours, and what to do about the answer."""
        rows, owned = self.table, self.owned_rows
        notes: list[str] = []
        if rows and not self.owned:
            notes.append(
                f"no GPUs are assigned to {self.host}, so nothing can be submitted to it;\n"
                f"        assign some with `gpuc host set {self.host} --gpus <list>`, "
                f"from the indices or UUIDs above"
            )
        elif owned and len(owned) + len(self.shared_rows) < len(rows):
            spare = len(rows) - len(owned) - len(self.shared_rows)
            notes.append(
                f"{spare} of this host's {len(rows)} GPUs are neither assigned to "
                f"{self.host} nor shared\n        with it, so gpuc will never use them; "
                f"`gpuc host set {self.host} --gpus <list>` changes the assignment, "
                f"and\n        `--shared-gpus <list>` lets gpuc borrow one while nobody "
                f"else is on it"
            )
        if self.shared_missing:
            notes.append(
                f"shared but not present on this host: {', '.join(self.shared_missing)}.\n"
                f"        `gpuc host bootstrap {self.host}` fails its gpu_uuids check on this "
                f"too: `gpuc host set {self.host} --shared-gpus <list>`"
            )
        if self.owned_missing:
            notes.append(
                f"assigned but not present on this host: {', '.join(self.owned_missing)}.\n"
                f"        `gpuc host bootstrap {self.host}` fails its gpu_uuids check on this, so "
                f"fix the\n        list first: `gpuc host set {self.host} --gpus <list>`"
            )
        doubled = self.cards.duplicates if self.has_nvidia_smi else []
        if doubled:
            notes.append(
                f"{', '.join(doubled)} name a card already named -- an index and its own UUID\n"
                f"        are one card, and a card is owned or shared, not both. "
                f"`gpuc host bootstrap {self.host}` fails rather than promise a card twice"
            )
        return notes

    def document(self) -> dict[str, Any]:
        """`gpuc host probe --json`.

        `sections` is the probe script's raw output, section by section, so
        anything this build does not interpret is still there. Everything
        beside it is the interpretation `render()` prints. Every card the host
        has is listed whatever `--all-gpus` said, each flagged `assigned` or not.
        """
        cards = self.cards
        return {
            "host": self.host,
            "sections": dict(self.sections),
            "driver_version": self.driver_version,
            "has_nvidia_smi": self.has_nvidia_smi,
            "gpus": [
                {
                    "uuid": uuid,
                    **info.model_dump(mode="json"),
                    "assigned": uuid in cards.owned,
                    "shared": uuid in cards.shared,
                }
                for uuid, info in self.gpu_info.items()
            ],
            "assigned_gpus": list(self.owned),
            "assigned_missing": self.owned_missing,
            "shared_gpus": list(self.shared),
            "shared_missing": self.shared_missing,
            "home_fs_type": self.home_fs_type,
            "home_is_overlay": self.home_is_overlay,
            "persistent_root": self.persistent_root,
            "notes": self.notes,
        }


def parse_probe(
    host: str,
    output: str,
    persistent_root: str | None = None,
    owned: Sequence[str] | None = None,
    shared: Sequence[str] | None = None,
) -> ProbeReport:
    sections: dict[str, str] = {}
    current = "preamble"
    buffer: list[str] = []
    for line in output.splitlines():
        stripped = line.strip()
        if stripped.startswith("===") and stripped.endswith("===") and len(stripped) > 6:
            sections[current] = "\n".join(buffer).strip()
            current = stripped.strip("=")
            buffer = []
            continue
        buffer.append(line)
    sections[current] = "\n".join(buffer).strip()
    return ProbeReport(
        host=host,
        sections=sections,
        persistent_root=persistent_root,
        owned=list(owned or []),
        shared=list(shared or []),
    )


def probe_host(
    entry: HostEntry, settings: Settings | None = None, *, transport: Transport | None = None
) -> ProbeReport:
    transport = transport or transport_for(entry, settings)
    result = transport.run(probe_script(), timeout=240.0, check=False)
    config = entry.config
    return parse_probe(entry.name, result.output, entry.root, config.gpus, config.shared_gpus)
