"""`gpuc host probe`: what a host is, before we have installed anything on it.

Pure `sh` plus `python3`-if-present, because the whole point is to run against
a host where nothing has been bootstrapped yet.
"""

from __future__ import annotations

from dataclasses import dataclass

from gpuc.control.config import HostEntry, Settings, transport_for
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
    "python3",
    "download",
]

DOWNLOAD_URL = "https://download.pytorch.org/whl/cpu/torch-2.5.1%2Bcpu-cp311-cp311-linux_x86_64.whl"
DOWNLOAD_BYTES = 20 * 1024 * 1024

PROBE_SCRIPT = f"""
say() {{ echo "===$1==="; }}
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

    @property
    def home_fs_type(self) -> str | None:
        """The filesystem type of ``$HOME``: field 2 of `df -T` (and of the
        `stat -f` fallback, which prints ``<path> <type>``)."""
        fields = self.sections.get("home_fs", "").split()
        return fields[1] if len(fields) > 1 else None

    @property
    def home_is_overlay(self) -> bool:
        return (self.home_fs_type or "").lower() in OVERLAY_FS_TYPES

    def render(self) -> str:
        lines = [f"host {self.host}"]
        for key in SECTION_ORDER:
            value = self.sections.get(key, "(no output)")
            if key == "gpus":
                lines.append("  gpus:")
                rows = self.gpu_rows
                if not rows:
                    lines.append(f"    {value.strip() or '(none)'}")
                for cells in rows:
                    index, uuid, name = cells[0], cells[1], cells[2]
                    memory = f"  {cells[3]}" if len(cells) > 3 else ""
                    lines.append(f"    [{index}] {uuid}  {name}{memory}")
                continue
            lines.append(f"  {key}: {value.strip() or '(no output)'}")
        if not self.has_nvidia_smi:
            lines.append("  note: no nvidia-smi, so this host can only run gpus: 0 jobs")
        if self.sections.get("killuserprocesses", "").endswith("=yes"):
            lines.append(
                "  note: logind kills user processes at logout; the dispatcher will not "
                "survive your SSH session ending"
            )
        if self.home_is_overlay:
            overlay = (
                f"  note: $HOME is on an {self.home_fs_type} filesystem, so it is a container's "
                f"throwaway upper layer and is wiped on every restart.\n"
            )
            if self.persistent_root:
                overlay += (
                    f"        This host is registered with --persistent-root "
                    f"{self.persistent_root}, so uv, the queue and every job dir are already "
                    f"off it.\n        After a restart, recover with: "
                    f"gpuc host bootstrap {self.host}"
                )
            else:
                overlay += (
                    f"        Point this host at a volume that survives:\n"
                    f"        gpuc host set {self.host} --persistent-root /mnt/<volume>/$USER\n"
                    f"        (then `gpuc host bootstrap {self.host}`; uv, the queue and every "
                    f"job dir move there)"
                )
            lines.append(overlay)
        if self.sections.get("uv") == "not installed":
            lines.append(f"  note: uv is missing; `gpuc host bootstrap {self.host}` installs it")
        return "\n".join(lines)


def parse_probe(host: str, output: str, persistent_root: str | None = None) -> ProbeReport:
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
    return ProbeReport(host=host, sections=sections, persistent_root=persistent_root)


def probe_host(
    entry: HostEntry, settings: Settings | None = None, *, transport: Transport | None = None
) -> ProbeReport:
    transport = transport or transport_for(entry, settings)
    result = transport.run(PROBE_SCRIPT, timeout=240.0, check=False)
    return parse_probe(entry.name, result.output, entry.root)
