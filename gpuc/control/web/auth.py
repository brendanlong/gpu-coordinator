"""One password for the whole dashboard: bcrypt-hashed on disk, sessions in memory.

The hash lives beside the config file, written only by `gpuc web set-password`
and read once when the server starts. Sessions are random tokens the server
keeps in memory, so a restart logs everyone out, which is the simplest thing
that is also correct.
"""

from __future__ import annotations

import os
import secrets
import threading
import time
from collections.abc import Callable
from pathlib import Path

import bcrypt

from gpuc.control.actions import CliError, UsageError
from gpuc.control.config import config_dir

PASSWORD_FILE = "web-password"
SESSION_COOKIE = "gpuc_session"
SESSION_TTL_S = 7 * 86400.0
MIN_PASSWORD_CHARS = 8
BCRYPT_MAX_BYTES = 72
"""bcrypt reads at most 72 bytes of a password and, since bcrypt 5, refuses a
longer one rather than silently truncating it."""
FAILURE_DELAY_S = 0.5
"""Added per consecutive failed login, capped below. With one password for the
whole dashboard a guesser and the owner are throttled alike, and bcrypt's own
cost already makes each guess a quarter of a second."""
FAILURE_DELAY_MAX_S = 5.0


class NoPassword(CliError):
    """The dashboard cannot serve until a password has been recorded."""


def password_file() -> Path:
    return config_dir() / PASSWORD_FILE


def check_new_password(password: str) -> None:
    if len(password) < MIN_PASSWORD_CHARS:
        raise UsageError(f"the dashboard password must be at least {MIN_PASSWORD_CHARS} characters")
    if len(password.encode()) > BCRYPT_MAX_BYTES:
        raise UsageError(f"the dashboard password must be at most {BCRYPT_MAX_BYTES} bytes")


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def write_password(password: str) -> Path:
    """Record the hash 0600, atomically; the password itself is never written."""
    check_new_password(password)
    path = password_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.{os.getpid()}.tmp"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as handle:
        handle.write(hash_password(password) + "\n")
    os.replace(tmp, path)
    return path


def read_password_hash() -> str:
    path = password_file()
    try:
        text = path.read_text().strip()
    except FileNotFoundError:
        raise NoPassword(
            f"no dashboard password at {path}.\nSet one with: gpuc web set-password"
        ) from None
    if not text.startswith("$2"):
        raise NoPassword(f"{path} does not hold a bcrypt hash.\nRe-run: gpuc web set-password")
    return text


def verify(password: str, password_hash: str) -> bool:
    encoded = password.encode()
    if len(encoded) > BCRYPT_MAX_BYTES:
        # Longer than any password that could have been recorded, so it is
        # wrong; bcrypt 5 would raise rather than say so.
        return False
    try:
        return bcrypt.checkpw(encoded, password_hash.encode())
    except ValueError:
        return False


class Sessions:
    """Logins checked one at a time, and tokens that expire.

    The lock is the throttle: every guess waits for the one before it, and a
    run of failures adds a growing pause on top of bcrypt's own cost.
    """

    def __init__(
        self,
        password_hash: str,
        *,
        ttl_s: float = SESSION_TTL_S,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._hash = password_hash
        self._ttl_s = ttl_s
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        self._tokens: dict[str, float] = {}
        self.failures = 0

    def login(self, password: str) -> str | None:
        """A session token for the right password, None for a wrong one."""
        with self._lock:
            if self.failures:
                self._sleep(min(FAILURE_DELAY_S * self.failures, FAILURE_DELAY_MAX_S))
            if not verify(password, self._hash):
                self.failures += 1
                return None
            self.failures = 0
            self._expire()
            token = secrets.token_urlsafe(32)
            self._tokens[token] = self._clock() + self._ttl_s
            return token

    def check(self, token: str | None) -> bool:
        if not token:
            return False
        with self._lock:
            self._expire()
            return token in self._tokens

    def logout(self, token: str | None) -> None:
        if token:
            with self._lock:
                self._tokens.pop(token, None)

    def _expire(self) -> None:
        now = self._clock()
        for token in [t for t, until in self._tokens.items() if until <= now]:
            del self._tokens[token]
