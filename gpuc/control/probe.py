"""`gpuc host probe`: what a host is, before we have installed anything on it.

Pure `sh` plus `python3`-if-present, because the whole point is to run against
a host where nothing has been bootstrapped yet.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from gpuc.control.config import HostEntry, Settings, transport_for
from gpuc.control.gpuinfo import GpuInfo, parse_smi
from gpuc.control.transport import Transport

SECTION_ORDER = [
    "system",
    "driver",
    "gpus",
    "disk",
    "home_fs",
    "killuserprocesses",
    "systemd_scope",
    "uv",
    "uv_cache",
    "python3",
    "download",
]

DOWNLOAD_URL = "https://download.pytorch.org/whl/cpu/torch-2.5.1%2Bcpu-cp311-cp311-linux_x86_64.whl"
DOWNLOAD_BYTES = 20 * 1024 * 1024


def probe_script(gpuc_home: str) -> str:
    """The whole probe as one `sh` script. ``gpuc_home`` may be unexpanded
    (``$HOME/.gpuc``): it is quoted for the host's own shell to resolve."""
    return f"""
say() {{ echo "===$1==="; }}
# The nearest existing ancestor of $1, so a directory that does not exist yet
# can still be attributed to a filesystem.
upto() {{ d=$1; while [ ! -e "$d" ] && [ "$d" != "/" ] && [ -n "$d" ]; do d=$(dirname "$d"); done; \
echo "${{d:-/}}"; }}
dev() {{ stat -c %d "$(upto "$1")" 2>/dev/null || echo unknown; }}
say system
uname -srm; echo "user=$(id -un) home=$HOME shell=$SHELL"
say driver
nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>&1 | head -1 \
  || echo "nvidia-smi not found"
say gpus
nvidia-smi --query-gpu=index,uuid,name,memory.total --format=csv,noheader 2>&1 \
  || echo "nvidia-smi not found"
say disk
df -Ph "$HOME" | tail -1
say home_fs
df -T "$HOME" 2>/dev/null | tail -1 || stat -f -c '%n %T' "$HOME" 2>/dev/null \
  || echo "unknown unknown"
say killuserprocesses
kup=$(loginctl show-user "$(id -un)" -p KillUserProcesses 2>&1 | head -1)
echo "${{kup:-unknown: no logind session for this user}}"
say systemd_scope
if systemd-run --user --scope -- true >/dev/null 2>&1; then echo yes; else echo no; fi
say uv
if [ -x "$HOME/.local/bin/uv" ]; then "$HOME/.local/bin/uv" --version; \
elif command -v uv >/dev/null 2>&1; then uv --version; else echo "not installed"; fi
say uv_cache
cache="${{UV_CACHE_DIR:-$HOME/.cache/uv}}"
echo "dir=$cache"
if [ -d "$cache" ]; then echo "size=$(du -sh "$cache" 2>/dev/null | cut -f1)"; \
else echo "size=absent"; fi
echo "gpuc_home={gpuc_home}"
echo "cache_dev=$(dev "$cache")"
echo "home_dev=$(dev "{gpuc_home}")"
say python3
if command -v python3 >/dev/null 2>&1; then python3 -c 'import sys; print(sys.executable, \
sys.version.split()[0])'; else echo "not installed"; fi
say download
if command -v python3 >/dev/null 2>&1; then
python3 - <<'PYEOF'
import time, urllib.request
url = "{DOWNLOAD_URL}"
want = {DOWNLOAD_BYTES}
start = time.time()
try:
    req = urllib.request.Request(url, headers={{"Range": f"bytes=0-{{want - 1}}"}})
    with urllib.request.urlopen(req, timeout=60) as response:
        read = 0
        while read < want:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            read += len(chunk)
    elapsed = max(time.time() - start, 1e-6)
    print(f"{{read / 1e6:.0f}} MB in {{elapsed:.1f}}s = {{read / 1e6 / elapsed:.1f}} MB/s")
except Exception as exc:
    print(f"FAILED: {{exc}}")
PYEOF
elif command -v curl >/dev/null 2>&1; then
  curl -s -o /dev/null -r 0-{DOWNLOAD_BYTES - 1} \
    -w '%{{size_download}} bytes at %{{speed_download}} B/s\\n' "{DOWNLOAD_URL}"
else
  echo "no python3 and no curl: cannot time a download"
fi
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

    @property
    def has_nvidia_smi(self) -> bool:
        return "not found" not in self.sections.get("driver", "not found")

    @property
    def gpu_rows(self) -> list[list[str]]:
        if not self.has_nvidia_smi:
            return []
        rows: list[list[str]] = []
        for line in self.sections.get("gpus", "").splitlines():
            cells = [c.strip() for c in line.split(",")]
            if len(cells) >= 3 and cells[0].isdigit():
                rows.append(cells)
        return rows

    def owns(self, cells: Sequence[str]) -> bool:
        """Is this `gpu_rows` row one of ours?

        By UUID, or by the index nvidia-smi *just* gave the card -- which is
        the numbering `--gpus 2,3` was agreed in, and fresher than anything the
        registry recorded."""
        owned = set(self.owned)
        return cells[1] in owned or cells[0] in owned

    @property
    def owned_rows(self) -> list[list[str]]:
        return [cells for cells in self.gpu_rows if self.owns(cells)]

    @property
    def owned_missing(self) -> list[str]:
        """`--gpus` entries no card on this host answers to: a typo, a card
        this container was not given, or a renumbered driver."""
        if not self.has_nvidia_smi:
            return []
        seen = {cell for cells in self.gpu_rows for cell in cells[:2]}
        return [item for item in self.owned if item not in seen]

    @property
    def gpu_info(self) -> dict[str, GpuInfo]:
        """The `gpus` section as the registry stores it, keyed by UUID."""
        return parse_smi("\n".join(",".join(cells) for cells in self.gpu_rows if len(cells) > 2))

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

    @property
    def uv_cache(self) -> dict[str, str]:
        """The `key=value` lines of the `uv_cache` section."""
        return dict(
            line.split("=", 1)
            for line in self.sections.get("uv_cache", "").splitlines()
            if "=" in line
        )

    @property
    def cache_shares_gpuc_home_fs(self) -> bool | None:
        """Can uv link a wheel out of its cache into a job's venv? None if unknown."""
        values = self.uv_cache
        cache, home = values.get("cache_dev"), values.get("home_dev")
        if not cache or not home or "unknown" in (cache, home):
            return None
        return cache == home

    def render(self, *, all_gpus: bool = False) -> str:
        lines = [f"host {self.host}"]
        for key in SECTION_ORDER:
            value = self.sections.get(key, "(no output)")
            if key == "gpus":
                lines += self._gpu_lines(all_gpus)
                continue
            if key == "uv_cache":
                values = self.uv_cache
                shared = {True: "yes", False: "NO", None: "unknown"}[self.cache_shares_gpuc_home_fs]
                lines.append(
                    f"  uv_cache: {values.get('dir', '?')} size {values.get('size', '?')} "
                    f"(same filesystem as gpuc home {values.get('gpuc_home', '?')}: {shared})"
                )
                continue
            lines.append(f"  {key}: {value.strip() or '(no output)'}")
        lines += [f"  note: {note}" for note in self.notes]
        return "\n".join(lines)

    def _gpu_lines(self, all_gpus: bool) -> list[str]:
        """The `gpus` section: ours by default, the whole box with `--all-gpus`."""
        rows, owned = self.gpu_rows, self.owned_rows
        if not rows:
            return ["  gpus:", f"    {self.sections.get('gpus', '').strip() or '(no output)'}"]
        # Nothing of ours to show is not a reason to show nothing: a host whose
        # assignment matches no card needs the whole list more than anybody.
        everything = all_gpus or not owned
        partly = 0 < len(owned) < len(rows)
        hidden = " (--all-gpus lists the rest)" if partly and not everything else ""
        header = f"  gpus: {len(owned)} of {len(rows)} assigned to {self.host}{hidden}"
        lines = [header if self.owned else "  gpus:"]
        for cells in rows if everything else owned:
            index, uuid, name = cells[0], cells[1], cells[2]
            memory = f"  {cells[3]}" if len(cells) > 3 else ""
            mine = "  (assigned)" if everything and partly and self.owns(cells) else ""
            lines.append(f"    [{index}] {uuid}  {name}{memory}{mine}")
        return lines

    @property
    def notes(self) -> list[str]:
        """What this host will do to a job unless somebody acts, in words."""
        notes: list[str] = []
        if not self.has_nvidia_smi:
            notes.append("no nvidia-smi, so this host can only run gpus: 0 jobs")
        notes += self._gpu_notes()
        if self.sections.get("killuserprocesses", "").endswith("=yes"):
            notes.append(
                "logind kills user processes at logout; the dispatcher will not "
                "survive your SSH session ending"
            )
        if self.home_is_overlay:
            notes.append(self._overlay_note())
        if self.sections.get("uv") == "not installed":
            notes.append(f"uv is missing; `gpuc host bootstrap {self.host}` installs it")
        if self.cache_shares_gpuc_home_fs is False:
            notes.append(
                f"uv's cache and gpuc home are on different filesystems, so uv cannot\n"
                f"        hardlink or reflink wheels into a job's venv and copies each one "
                f"instead\n        (~6.5 GB per torch venv). "
                f"`gpuc host bootstrap {self.host}` fixes this by pointing\n"
                f"        UV_CACHE_DIR at gpuc home's own volume."
            )
        return notes

    def _gpu_notes(self) -> list[str]:
        """Which cards in this box are ours, and what to do about the answer."""
        rows, owned = self.gpu_rows, self.owned_rows
        notes: list[str] = []
        if rows and not self.owned:
            notes.append(
                f"no GPUs are assigned to {self.host}, so it can only run gpus: 0 jobs;\n"
                f"        assign some with `gpuc host set {self.host} --gpus <list>`, "
                f"from the indices or UUIDs above"
            )
        elif owned and len(owned) < len(rows):
            notes.append(
                f"{len(rows) - len(owned)} of this host's {len(rows)} GPUs are not assigned to "
                f"{self.host}, so gpuc will never\n        use them; "
                f"`gpuc host set {self.host} --gpus <list>` changes the assignment"
            )
        if self.owned_missing:
            notes.append(
                f"assigned but not present on this host: {', '.join(self.owned_missing)}.\n"
                f"        `gpuc host bootstrap {self.host}` fails its gpu_uuids check on this, so "
                f"fix the\n        list first: `gpuc host set {self.host} --gpus <list>`"
            )
        doubled = len(self.owned) - len(self.owned_missing) - len(owned)
        if doubled > 0:
            notes.append(
                f"{len(self.owned) - len(self.owned_missing)} of the assigned entries name only "
                f"{len(owned)} card(s) -- an index and its own UUID\n        are one card. "
                f"`gpuc host bootstrap {self.host}` fails rather than promise a card twice"
            )
        return notes

    def _overlay_note(self) -> str:
        overlay = (
            f"$HOME is on an {self.home_fs_type} filesystem, so it is a container's "
            f"throwaway upper layer and is wiped on every restart.\n"
        )
        if self.persistent_root:
            return overlay + (
                f"        This host is registered with --persistent-root "
                f"{self.persistent_root}, so the queue and every job dir are already "
                f"off it.\n        After a restart, recover with: "
                f"gpuc host bootstrap {self.host}"
            )
        return overlay + (
            f"        Point this host at a volume that survives:\n"
            f"        gpuc host set {self.host} --persistent-root /mnt/<volume>/$USER\n"
            f"        (then `gpuc host bootstrap {self.host}`: the queue and every job dir\n"
            f"        move there, and uv's cache follows only to stay on gpuc home's\n"
            f"        filesystem; uv itself stays in $HOME and bootstrap reinstalls it)"
        )

    def document(self) -> dict[str, Any]:
        """`gpuc host probe --json`.

        `sections` is the probe script's raw output, section by section, so
        anything this build does not interpret is still there. Everything
        beside it is the interpretation `render()` prints. Every card the host
        has is listed whatever `--all-gpus` said, each flagged `assigned` or not.
        """
        assigned = {cells[1] for cells in self.owned_rows}
        return {
            "host": self.host,
            "sections": dict(self.sections),
            "driver_version": self.driver_version,
            "has_nvidia_smi": self.has_nvidia_smi,
            "gpus": [
                {
                    "uuid": uuid,
                    **info.model_dump(mode="json"),
                    "assigned": uuid in assigned,
                }
                for uuid, info in self.gpu_info.items()
            ],
            "assigned_gpus": list(self.owned),
            "assigned_missing": self.owned_missing,
            "home_fs_type": self.home_fs_type,
            "home_is_overlay": self.home_is_overlay,
            "persistent_root": self.persistent_root,
            "uv_cache": {
                **self.uv_cache,
                "shares_gpuc_home_fs": self.cache_shares_gpuc_home_fs,
            },
            "notes": self.notes,
        }


def parse_probe(
    host: str,
    output: str,
    persistent_root: str | None = None,
    owned: Sequence[str] | None = None,
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
        host=host, sections=sections, persistent_root=persistent_root, owned=list(owned or [])
    )


def probe_host(
    entry: HostEntry, settings: Settings | None = None, *, transport: Transport | None = None
) -> ProbeReport:
    transport = transport or transport_for(entry, settings)
    result = transport.run(probe_script(entry.remote_home), timeout=240.0, check=False)
    return parse_probe(entry.name, result.output, entry.root, entry.gpus)
