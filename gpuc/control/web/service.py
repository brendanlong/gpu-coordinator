"""`gpuc web serve --install`: a `systemd --user` service for the dashboard.

Written, never enabled, like `reconcile --install`: turning on something that
listens on a port is the user's call, and the printed lines are the whole of
it.
"""

from __future__ import annotations

from pathlib import Path

from gpuc.control.config import config_dir, state_dir
from gpuc.control.systemd import Reporter, gpuc_command, systemd_dir, write_units
from gpuc.control.web.auth import password_file

SERVICE_NAME = "gpuc-web.service"


def unit_file(bind: str, port: int) -> str:
    return f"""[Unit]
Description=gpuc web dashboard on {bind}:{port}
After=network-online.target

[Service]
Environment=GPUC_CONFIG_DIR={config_dir()}
Environment=GPUC_STATE_DIR={state_dir()}
EnvironmentFile=-{config_dir()}/env
ExecStart={gpuc_command(f"web serve --bind {bind} --port {port}")}
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
"""


def install_service(bind: str, port: int, report: Reporter = print) -> list[Path]:
    written = write_units(systemd_dir(), {SERVICE_NAME: unit_file(bind, port)}, report)
    if not password_file().exists():
        report(
            f"no dashboard password at {password_file()}: the service will refuse to start "
            f"until you run `gpuc web set-password`"
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
