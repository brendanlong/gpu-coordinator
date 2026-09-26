"""`gpuc web`: the dashboard, a thin HTTP view over `gpuc.control.actions`."""

from gpuc.control.web.app import DEFAULT_BIND, DEFAULT_PORT, make_server
from gpuc.control.web.auth import write_password
from gpuc.control.web.service import install_service

__all__ = ["DEFAULT_BIND", "DEFAULT_PORT", "install_service", "make_server", "write_password"]
