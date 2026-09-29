"""Job secrets encrypted to the host, and the host's key to open them.

`gpuc submit` encrypts a job's secrets with this host's public key before
they leave the submitter's machine, so what sits in `secrets/<id>.env.age`
is useless to anyone without `secrets/host.age`. On a host whose disk is its
own that changes little; it is for the volumes a queue is kept on so it can
outlive its machine, which are shared with other machines, other people, or
a cloud provider's backups.

The standard library has no public-key cryptography, so the key is made and
used by pyrage under `uv run --with`: uv fetches it once into its cache, and
the host package itself still needs nothing but an interpreter. Plaintext
travels only through pipes.
"""

from __future__ import annotations

import shutil
import subprocess
from contextlib import suppress

from gpuc.host import jobs, paths

PYRAGE = "pyrage==1.4.0"
TIMEOUT_S = 120.0
"""Long enough for uv to fetch pyrage into an empty cache over a slow link;
every later call is a cached environment and takes a fraction of a second."""

_KEYGEN = """
import os, sys
from pyrage import x25519
identity = x25519.Identity.generate()
fd = os.open(sys.argv[1], os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(fd, "w") as out:
    out.write(str(identity) + "\\n")
print(identity.to_public())
"""

_PUBLIC = """
import sys
from pyrage import x25519
with open(sys.argv[1]) as key:
    print(x25519.Identity.from_str(key.read().strip()).to_public())
"""

_DECRYPT = """
import sys
import pyrage
from pyrage import x25519
with open(sys.argv[1]) as key:
    identity = x25519.Identity.from_str(key.read().strip())
sys.stdout.buffer.write(pyrage.decrypt(sys.stdin.buffer.read(), [identity]))
"""


class SealedError(RuntimeError):
    pass


def _uv() -> str:
    """uv as the dispatcher's children find it; a command over ssh may have
    a PATH without `~/.local/bin`."""
    config = jobs.HostConfig()
    with suppress(RuntimeError, OSError, ValueError):
        config = jobs.read_config()
    found = shutil.which("uv", path=paths.path_with_user_bins(extra=config.bin_dirs()))
    if found is None:
        raise SealedError("uv is not installed on this host; run `gpuc host bootstrap`")
    return found


def _pyrage(script: str, *args: str, stdin: bytes = b"") -> bytes:
    argv = [_uv(), "run", "--no-project", "--quiet", "--with", PYRAGE, "python", "-I", "-c"]
    try:
        done = subprocess.run(
            [*argv, script, *args], input=stdin, capture_output=True, timeout=TIMEOUT_S
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise SealedError(f"could not run pyrage: {exc}") from exc
    if done.returncode != 0:
        tail = done.stderr.decode(errors="replace").strip().splitlines()[-1:] or ["no output"]
        raise SealedError(f"pyrage failed (exit {done.returncode}): {tail[0]}")
    return done.stdout


def recipient() -> str:
    """This host's public key, made along with its private key on first use.

    Kept in a file of its own beside the private key so asking for it needs
    no pyrage. Two first asks at once are settled by the exclusive create:
    the loser, like a host whose keygen died before writing the public half,
    derives it from the private key the winner made.
    """
    public = paths.host_recipient_file()
    if public.is_file():
        return public.read_text().strip()
    paths.ensure_layout()
    identity = str(paths.host_identity_file())
    try:
        made = _pyrage(_KEYGEN, identity)
    except SealedError:
        if not paths.host_identity_file().exists():
            raise
        made = _pyrage(_PUBLIC, identity)
    key = made.decode().strip()
    jobs.atomic_write_text(public, key + "\n")
    return key


def job_secrets(job_id: str) -> dict[str, str]:
    """The job's secrets as `KEY: value`: opened from `<id>.env.age`, or read
    from a plain `<id>.env` a client from before encryption delivered. None
    delivered is no secrets. Raises `SealedError` when they are there and
    cannot be opened."""
    sealed = paths.sealed_env_file(job_id)
    if sealed.is_file():
        plain = _pyrage(_DECRYPT, str(paths.host_identity_file()), stdin=sealed.read_bytes())
        return jobs.parse_env_text(plain.decode())
    return jobs.parse_env_file(paths.job_env_file(job_id))
