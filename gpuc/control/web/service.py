"""`gpuc web serve --install`: a `systemd --user` service for the dashboard.

Written, never enabled: turning on something that listens on a port is the
user's call, and the printed lines are the whole of it.
"""

from __future__ import annotations

import re
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path

from gpuc.control.actions import UsageError
from gpuc.control.config import Reporter, config_dir, state_dir
from gpuc.control.version import package_root
from gpuc.control.web.auth import password_file

SERVICE_NAME = "gpuc-web.service"
BIND = re.compile(r"^[A-Za-z0-9.:\[\]_-]+$")
"""An address or a host name. Anything else is not one, and would land in the
unit file verbatim: a newline in it is a second `ExecStart=`."""
START_LIMIT_INTERVAL_S = 300
START_LIMIT_BURST = 5
"""`Restart=on-failure` under a start limit: a service with no password set
starts, logs the `gpuc web set-password` line and exits, and without the
limit systemd would loop it for ever. Five tries over five minutes leaves it
`failed`, where `systemctl --user status` shows the line."""


def systemd_dir() -> Path:
    return Path.home() / ".config/systemd/user"


SAFE_WORD = re.compile(r"^[A-Za-z0-9_./:=@+,\[\]-]+$")


def quote(word: str) -> str:
    """One ExecStart word, as systemd will read it back.

    Left bare when it needs nothing, so a unit for an ordinary path reads
    like one somebody typed. Otherwise double-quoted with `\\` and `"`
    escaped, and `%` doubled either way: it is a specifier to systemd, and a
    home directory named `100%` would otherwise expand to nothing.
    """
    if SAFE_WORD.match(word):
        return word
    escaped = word.replace("\\", "\\\\").replace('"', '\\"').replace("%", "%%")
    return f'"{escaped}"'


def gpuc_command(args: Sequence[str]) -> str:
    """An absolute command line for `gpuc <args>`, for a unit's ExecStart.

    The `gpuc` beside this interpreter first (a `uv tool install`), then the
    one on PATH, and failing both a `uv run` from the checkout this package
    lives in; systemd has no PATH of ours to search.
    """
    beside = Path(sys.executable).resolve().parent / "gpuc"
    if beside.exists():
        argv = [str(beside), *args]
    elif installed := shutil.which("gpuc"):
        argv = [str(Path(installed).resolve()), *args]
    else:
        uv = shutil.which("uv") or str(Path.home() / ".local/bin/uv")
        argv = [uv, "run", "--project", str(package_root()), "gpuc", *args]
    return " ".join(quote(word) for word in argv)


def unit_file(bind: str, port: int) -> str:
    if not BIND.match(bind):
        raise UsageError(f"--bind {bind!r} is not an address or host name")
    if not 0 < port < 65536:
        raise UsageError(f"--port {port} is not a port")
    return f"""[Unit]
Description=gpuc web dashboard on {bind}:{port}
After=network-online.target
StartLimitIntervalSec={START_LIMIT_INTERVAL_S}
StartLimitBurst={START_LIMIT_BURST}

[Service]
Environment=GPUC_CONFIG_DIR={config_dir()}
Environment=GPUC_STATE_DIR={state_dir()}
EnvironmentFile=-{config_dir()}/env
ExecStart={gpuc_command(["web", "serve", "--bind", bind, "--port", str(port)])}
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
"""


def install_service(bind: str, port: int, report: Reporter = print) -> Path:
    """Write the unit file only. Enabling is the user's call, not ours."""
    unit = unit_file(bind, port)
    directory = systemd_dir()
    directory.mkdir(parents=True, exist_ok=True)
    written = directory / SERVICE_NAME
    written.write_text(unit)
    report(f"wrote {written}")
    if not password_file().exists():
        report(
            f"no dashboard password at {password_file()}: until you run `gpuc web "
            f"set-password` the service exits at once, retries {START_LIMIT_BURST} times, and "
            f"then shows as failed"
        )
    report(
        "not enabled. Then:\n"
        "  systemctl --user daemon-reload\n"
        f"  systemctl --user enable --now {SERVICE_NAME}\n"
        f"  journalctl --user -u {SERVICE_NAME} -f\n"
        f"It reads provider keys (RUNPOD_API_KEY, VAST_API_KEY) from {config_dir()}/env if "
        f"that file exists, and needs "
        f"`loginctl enable-linger` to outlive your login session."
    )
    return written
