"""`gpuc web`: the dashboard, a thin HTTP view over `gpuc.control.actions`."""

from gpuc.control.web.app import DEFAULT_BIND, DEFAULT_PORT, Dashboard, make_server
from gpuc.control.web.auth import NoPassword, Sessions, password_file, write_password
from gpuc.control.web.service import SERVICE_NAME, install_service, unit_file

__all__ = [
    "DEFAULT_BIND",
    "DEFAULT_PORT",
    "SERVICE_NAME",
    "Dashboard",
    "NoPassword",
    "Sessions",
    "install_service",
    "make_server",
    "password_file",
    "unit_file",
    "write_password",
]
