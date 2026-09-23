"""`gpuc web serve --install`: a `systemd --user` service for the dashboard.

Written, never enabled: turning on something that listens on a port is the
user's call, and the printed lines are the whole of it.
"""

from __future__ import annotations

import re
from pathlib import Path

from gpuc.control.actions import UsageError
from gpuc.control.config import config_dir, state_dir
from gpuc.control.systemd import Reporter, gpuc_command, systemd_dir, write_units
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
"""`Restart=on-failure` alone retries every `RestartSec` for ever, and a
server with no password exits 1 at once: without a limit that is a loop only
the journal can see. Five tries in five minutes, then `failed`."""


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


def install_service(bind: str, port: int, report: Reporter = print) -> list[Path]:
    written = write_units(systemd_dir(), {SERVICE_NAME: unit_file(bind, port)}, report)
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
        f"It reads RUNPOD_API_KEY from {config_dir()}/env if that file exists, and needs "
        f"`loginctl enable-linger` to outlive your login session."
    )
    return written
