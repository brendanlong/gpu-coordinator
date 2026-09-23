"""A host in a temporary home, driven exactly as an ssh host is.

Every command runs for real, through `bash -c` with `$HOME` pointed at a
directory of its own and `$HOME/.local/bin` first on `PATH`, so the on-host
package, the probe script, `read_config`/`write_config` and a bootstrap all
run as they do on a box. What is faked is only what a laptop does not have:

- `nvidia-smi`: `conftest.install_fake_nvidia_smi`, answering for `gpus`;
- `uv`, `aws` and `hf`: shims in the places bootstrap looks first, answering
  the questions it asks (`uv python find` is this test's own interpreter,
  `uv cache dir` is `$HOME/.cache/uv`) and installing nothing;
- the provider API (`fakeprovider.FakeProvider`) and S3 (`fakes3`).

A bootstrap here starts a real dispatcher in the temporary home; `close`
stops it, and the `fake_host` fixture always does.
"""

from __future__ import annotations

import contextlib
import json
import os
import shlex
import subprocess
import sys
import time
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest

from gpuc.control.transport import CommandResult, TransportError, rsync_argv, tail_command
from gpuc.host import scope
from tests.conftest import install_fake_nvidia_smi

DEFAULT_GPUS = ("GPU-a", "GPU-b")

UV_SHIM = """\
#!/bin/sh
# uv, as far as bootstrap and the interpreter probe ask it: nothing is installed.
case "$1 $2" in
  "python find") echo {python} ;;
  "cache dir") echo "$HOME/.cache/uv" ;;
  "--version "*) echo "uv 0.0-fake" ;;
esac
exit 0
"""

NO_NVIDIA_SMI = '#!/bin/sh\necho "sh: 1: nvidia-smi: not found"\nexit 127\n'
"""What a box with no driver says, whatever this machine has on its PATH."""


class FakeHost:
    host = "fake"

    def __init__(
        self,
        root: Path,
        *,
        gpus: Sequence[str] | None = DEFAULT_GPUS,
        refuse: int = 0,
        refusal: str = "ssh: connect to host 1.2.3.4 port 22000: Connection refused",
    ) -> None:
        self.home_dir = root / "home"
        self.refuse = refuse
        """Connections to refuse before the first command that gets through,
        as a pod whose sshd is still coming up does; `refusal` is what ssh
        says each time."""
        self.refusal = refusal
        self.commands: list[str] = []
        self.gpuc_home = "$HOME/.gpuc"
        """The host's `GPUC_HOME` as an entry spells it; `$HOME` expands here."""
        bin_dir = self.home_dir / ".local" / "bin"
        bin_dir.mkdir(parents=True, exist_ok=True)
        self.set_gpus(gpus)
        _shim(bin_dir / "uv", UV_SHIM.format(python=shlex.quote(sys.executable)))
        _shim(bin_dir / "hf", "#!/bin/sh\nexit 0\n")
        _shim(self.home_dir / ".local/aws-cli/v2/current/bin/aws", "#!/bin/sh\nexit 0\n")

    # -- what a test reads off the host --------------------------------------

    @property
    def home(self) -> str:
        return self.gpuc_home.replace("$HOME", str(self.home_dir))

    @property
    def config_path(self) -> str:
        return f"{self.home}/config.json"

    @property
    def config(self) -> dict[str, Any] | None:
        """What this host's `config.json` holds, if it has one."""
        path = Path(self.config_path)
        if not path.exists():
            return None
        document: dict[str, Any] = json.loads(path.read_text())
        return document

    def path(self, remote_path: str) -> Path:
        return Path(remote_path.replace("$HOME", str(self.home_dir)))

    def root(self, name: str) -> str:
        """An absolute path a `--persistent-root` can point at on this host."""
        return str(self.home_dir / name)

    def set_gpus(self, uuids: Sequence[str] | None) -> None:
        """The cards the box has from now on; None is a box with no driver."""
        bin_dir = self.home_dir / ".local" / "bin"
        if uuids is None:
            _shim(bin_dir / "nvidia-smi", NO_NVIDIA_SMI)
        else:
            install_fake_nvidia_smi(bin_dir, list(uuids))

    def wipe(self) -> None:
        """Take the host's gpuc home away, as a re-imaged pod would."""
        self.close()
        subprocess.run(["rm", "-rf", self.home], check=True)

    def close(self) -> None:
        """Stop the dispatcher a bootstrap started, if one is running.

        A bootstrap leaves a package behind; the dispatcher it spawned takes
        its lock a moment later, so a host with a package is given that
        moment before the lock is read."""
        home = Path(self.home)
        lock = home / "dispatcher.lock"
        deadline = time.monotonic() + 3.0
        while (home / "pkg").exists() and not lock.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        if not lock.exists():
            return
        body = lock.read_text().strip().splitlines()
        pgid = body[0].strip() if body else ""
        if pgid.isdigit():
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(int(pgid), 9)

    # -- Transport ------------------------------------------------------------

    def env(self) -> dict[str, str]:
        return {
            **os.environ,
            "HOME": str(self.home_dir),
            "PATH": f"{self.home_dir / '.local' / 'bin'}{os.pathsep}{os.environ.get('PATH', '')}",
            "XDG_CONFIG_HOME": str(self.home_dir / ".config"),
            "XDG_DATA_HOME": str(self.home_dir / ".local" / "share"),
            "XDG_CACHE_HOME": str(self.home_dir / ".cache"),
            scope.ISOLATION_ENV: scope.PGID,
        }

    def argv(self, command: str) -> list[str]:
        return ["bash", "-c", command]

    def interactive_argv(self, command: str) -> list[str]:
        return ["bash", "-lc", command]

    def run(self, command: str, *, timeout: float = 120.0, check: bool = True) -> CommandResult:
        self.commands.append(command)
        if self.refuse > 0:
            self.refuse -= 1
            result = CommandResult(self.host, ["ssh", command], 255, "", self.refusal)
        else:
            proc = subprocess.run(
                self.argv(command),
                capture_output=True,
                timeout=timeout,
                check=False,
                env=self.env(),
                stdin=subprocess.DEVNULL,
            )
            result = CommandResult(
                self.host,
                self.argv(command),
                proc.returncode,
                proc.stdout.decode("utf-8", "replace"),
                proc.stderr.decode("utf-8", "replace"),
            )
        return result.check() if check else result

    def put_file(self, content: str | bytes, remote_path: str, mode: int = 0o600) -> None:
        path = self.path(remote_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = content.encode() if isinstance(content, str) else content
        tmp = path.parent / f".{path.name}.tmp"
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode)
        try:
            os.write(fd, data)
        finally:
            os.close(fd)
        tmp.chmod(mode)
        os.replace(tmp, path)

    def rsync(
        self,
        local_root: Path,
        remote_path: str,
        files: Sequence[str] | None = None,
        excludes: Sequence[str] = (),
    ) -> CommandResult:
        argv = rsync_argv(
            local_root, str(self.path(remote_path)), files, ssh_command=None, excludes=excludes
        )
        stdin = "".join(f"{name}\0" for name in files).encode() if files is not None else None
        proc = subprocess.run(argv, input=stdin, capture_output=True, check=False, timeout=600.0)
        result = CommandResult(
            self.host, argv, proc.returncode, proc.stdout.decode(), proc.stderr.decode()
        )
        if result.returncode != 0:
            raise TransportError(result)
        return result

    def tail(self, remote_path: str, lines: int = 200, follow: bool = False) -> CommandResult:
        return self.run(tail_command(remote_path, lines, follow), check=False)


def _shim(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(0o755)


@pytest.fixture
def fake_host(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeHost]:
    """Point every command that opens a transport at a host in a temporary home."""
    host = FakeHost(tmp_path / "host")

    def factory(entry: Any, settings: Any = None) -> Any:
        host.gpuc_home = entry.remote_home
        return host

    for module in ("connect", "probe", "cli", "remote", "bootstrap"):
        monkeypatch.setattr(f"gpuc.control.{module}.transport_for", factory, raising=False)
    try:
        yield host
    finally:
        host.close()
