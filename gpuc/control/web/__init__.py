"""`gpuc web`: the dashboard, a thin HTTP view over `gpuc.control.actions`."""

from gpuc.control.web.app import DEFAULT_BIND, DEFAULT_PORT, Dashboard, make_server
from gpuc.control.web.auth import NoPassword, Sessions, password_file, write_password

__all__ = [
    "DEFAULT_BIND",
    "DEFAULT_PORT",
    "Dashboard",
    "NoPassword",
    "Sessions",
    "make_server",
    "password_file",
    "write_password",
]
