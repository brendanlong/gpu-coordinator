"""Output upload: shells out to the `aws` CLI and the `hf` CLI.

These binaries are installed into $HOME by the bootstrap. A missing binary or a
failed upload raises SyncError, which fails the *job's* sync step; the queue
itself never depends on either tool being present.
"""

from __future__ import annotations

import shutil
import subprocess
import threading
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from gpuc.host import paths
from gpuc.host.jobs import JobSpec, Output

MIN_AGE_S = 10.0
DEFAULT_TIMEOUT_S = 1800.0


class SyncError(RuntimeError):
    pass


@dataclass
class CommandResult:
    argv: list[str]
    returncode: int
    output: str


CommandRunner = Callable[[list[str], float], CommandResult]


def run_command(argv: list[str], timeout: float = DEFAULT_TIMEOUT_S) -> CommandResult:
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False)
    return CommandResult(argv, proc.returncode, (proc.stdout or "") + (proc.stderr or ""))


def _host_label() -> str:
    from gpuc.host.jobs import read_config

    try:
        return read_config().host
    except Exception:
        return "unknown-host"


def _fail(result: CommandResult) -> None:
    tail = "\n".join(result.output.strip().splitlines()[-10:])
    raise SyncError(
        f"`{' '.join(result.argv)}` on host {_host_label()} exited {result.returncode}\n{tail}"
    )


def aws_binary() -> str | None:
    bundled = Path.home() / ".local/aws-cli/v2/current/bin/aws"
    if bundled.exists():
        return str(bundled)
    return shutil.which("aws")


def hf_binary() -> str | None:
    bundled = Path.home() / ".local/bin/hf"
    if bundled.exists():
        return str(bundled)
    return shutil.which("hf")


def recently_modified(root: Path, min_age_s: float = MIN_AGE_S) -> list[str]:
    """Relative paths under ``root`` written within the last ``min_age_s``.

    We exclude these by name rather than copying the tree to a staging dir:
    checkpoints are routinely tens of gigabytes, so a staging copy would cost
    more disk and IO than the upload itself, and the next tick picks the file
    up anyway once it has stopped changing.
    """
    if not root.is_dir():
        return []
    cutoff = time.time() - min_age_s
    out: list[str] = []
    for path in root.rglob("*"):
        if path.is_file():
            try:
                if path.stat().st_mtime > cutoff:
                    out.append(str(path.relative_to(root)))
            except OSError:
                continue
    return sorted(out)


def _exclude_args(flag: str, names: Iterable[str]) -> list[str]:
    args: list[str] = []
    for name in names:
        args += [flag, name]
    return args


def sync_dir_to_s3(
    local: Path,
    dest: str,
    *,
    min_age_s: float = MIN_AGE_S,
    runner: CommandRunner = run_command,
    timeout: float = DEFAULT_TIMEOUT_S,
) -> None:
    aws = aws_binary()
    if aws is None:
        raise SyncError(
            "`aws` CLI not found (looked in ~/.local/aws-cli/v2/current/bin/aws and PATH) "
            f"on host {_host_label()}; cannot upload {local} to {dest}"
        )
    if not local.exists():
        raise SyncError(f"output path does not exist: {local}")
    argv = [aws, "s3", "sync", str(local), dest.rstrip("/"), "--only-show-errors"]
    argv += _exclude_args("--exclude", recently_modified(local, min_age_s))
    result = runner(argv, timeout)
    if result.returncode != 0:
        _fail(result)


def copy_file_to_s3(
    local: Path,
    dest: str,
    *,
    runner: CommandRunner = run_command,
    timeout: float = 300.0,
) -> None:
    aws = aws_binary()
    if aws is None:
        raise SyncError(
            f"`aws` CLI not found on host {_host_label()}; cannot upload {local} to {dest}"
        )
    result = runner([aws, "s3", "cp", str(local), dest, "--only-show-errors"], timeout)
    if result.returncode != 0:
        _fail(result)


def upload_dir_to_hf(
    local: Path,
    repo: str,
    path_in_repo: str,
    *,
    min_age_s: float = MIN_AGE_S,
    runner: CommandRunner = run_command,
    timeout: float = DEFAULT_TIMEOUT_S,
) -> None:
    hf = hf_binary()
    if hf is None:
        raise SyncError(
            "`hf` CLI not found (looked in ~/.local/bin/hf and PATH) on host "
            f"{_host_label()}; cannot upload {local} to {repo}:{path_in_repo}"
        )
    if not local.exists():
        raise SyncError(f"output path does not exist: {local}")
    argv = [hf, "upload", repo, str(local), path_in_repo]
    argv += _exclude_args("--exclude", recently_modified(local, min_age_s))
    result = runner(argv, timeout)
    if result.returncode != 0:
        _fail(result)


def resolve_local(output: Output, workdir: Path, job_id: str) -> Path:
    return workdir / output.path.format(job_id=job_id)


def sync_output(
    output: Output,
    workdir: Path,
    job_id: str,
    *,
    min_age_s: float = MIN_AGE_S,
    runner: CommandRunner = run_command,
) -> None:
    local = resolve_local(output, workdir, job_id)
    if output.s3:
        sync_dir_to_s3(local, output.s3.format(job_id=job_id), min_age_s=min_age_s, runner=runner)
    if output.hf:
        upload_dir_to_hf(
            local,
            output.hf.format(job_id=job_id),
            (output.hf_path or job_id).format(job_id=job_id),
            min_age_s=min_age_s,
            runner=runner,
        )


def sync_outputs(
    outputs: Sequence[Output],
    workdir: Path,
    job_id: str,
    *,
    min_age_s: float = MIN_AGE_S,
    runner: CommandRunner = run_command,
) -> None:
    errors: list[str] = []
    for output in outputs:
        try:
            sync_output(output, workdir, job_id, min_age_s=min_age_s, runner=runner)
        except SyncError as exc:
            errors.append(str(exc))
    if errors:
        raise SyncError("; ".join(errors))


def sync_job_meta(
    job_id: str, s3_prefix: str | None, *, runner: CommandRunner = run_command
) -> None:
    if not s3_prefix:
        return
    base = f"{s3_prefix.rstrip('/')}/jobs/{job_id}"
    for path in (paths.log_file(job_id), paths.state_file(job_id)):
        if path.exists():
            copy_file_to_s3(path, f"{base}/{path.name}", runner=runner)


class SyncLoop:
    """Background periodic sync; `final()` runs one last synchronous pass."""

    def __init__(
        self,
        spec: JobSpec,
        workdir: Path,
        s3_prefix: str | None,
        *,
        runner: CommandRunner = run_command,
        min_age_s: float = MIN_AGE_S,
    ) -> None:
        self._spec = spec
        self._workdir = workdir
        self._s3_prefix = s3_prefix
        self._runner = runner
        self._min_age_s = min_age_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.last_error: str | None = None

    def _tick(self, min_age_s: float) -> None:
        sync_outputs(
            self._spec.outputs,
            self._workdir,
            self._spec.job_id,
            min_age_s=min_age_s,
            runner=self._runner,
        )
        sync_job_meta(self._spec.job_id, self._s3_prefix, runner=self._runner)

    def _loop(self) -> None:
        interval = max(1, self._spec.sync_interval_s)
        while not self._stop.wait(interval):
            try:
                self._tick(self._min_age_s)
            except SyncError as exc:
                self.last_error = str(exc)

    def start(self) -> None:
        if not self._spec.outputs and not self._s3_prefix:
            return
        self._thread = threading.Thread(target=self._loop, name="gpuc-sync", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=30)
            self._thread = None

    def final(self) -> None:
        """Stop the loop and do one complete sync, including files just written."""
        self.stop()
        self._tick(min_age_s=0.0)
