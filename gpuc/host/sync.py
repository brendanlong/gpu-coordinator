"""Output upload: shells out to the `aws` CLI and the `hf` CLI.

These binaries are installed into $HOME by the bootstrap. A missing binary or a
failed upload raises SyncError, which fails the *job's* sync step; the queue
itself never depends on either tool being present.

Every upload takes an explicit ``env``. The runner passes the job's own
environment -- which includes its ``secrets:`` file -- so a job that declares
``secrets: [AWS_ACCESS_KEY_ID, ...]`` can upload without any host-level
credential file. ``env=None`` means "inherit this process's environment",
which is what the dispatcher's drain path uses.
"""

from __future__ import annotations

import contextlib
import shutil
import subprocess
import threading
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from gpuc.host import baseline, jobs, paths, queue
from gpuc.host.jobs import JobSpec, Output

MIN_AGE_S = 10.0
DEFAULT_TIMEOUT_S = 1800.0
MAX_EXCLUDES = 200


class SyncError(RuntimeError):
    pass


class MissingOutput(SyncError):
    """The output path the spec names is not there. Not the same failure as a
    broken upload: nothing was produced, so `sync` would be a misleading reason."""


class TooManyRecentFiles(SyncError):
    """More files are in flight than we are willing to name on the command line."""


@dataclass
class CommandResult:
    argv: list[str]
    returncode: int
    output: str


Env = Mapping[str, str] | None
CommandRunner = Callable[[list[str], "float | None", Env], CommandResult]


def run_command(
    argv: list[str], timeout: float | None = DEFAULT_TIMEOUT_S, env: Env = None
) -> CommandResult:
    """Never raises anything but SyncError: an upload tool that is missing,
    wedged or killed must fail the job's sync step, not the runner."""
    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            env=None if env is None else dict(env),
        )
    except subprocess.TimeoutExpired as exc:
        raise SyncError(
            f"`{' '.join(argv)}` on host {_host_label()} timed out after {timeout}s"
        ) from exc
    except FileNotFoundError as exc:
        raise SyncError(f"`{argv[0]}` not found on host {_host_label()}: {exc}") from exc
    except OSError as exc:
        raise SyncError(f"`{' '.join(argv)}` on host {_host_label()} could not run: {exc}") from exc
    return CommandResult(argv, proc.returncode, (proc.stdout or "") + (proc.stderr or ""))


def _host_label() -> str:
    """The host name for an error message, never a second failure."""
    try:
        return jobs.read_config().host
    except (RuntimeError, OSError, ValueError):
        return "unknown-host"


def _fail(result: CommandResult) -> None:
    tail = "\n".join(result.output.strip().splitlines()[-10:])
    raise SyncError(
        f"`{' '.join(result.argv)}` on host {_host_label()} exited {result.returncode}\n{tail}"
    )


BUNDLED = {"aws": ".local/aws-cli/v2/current/bin/aws", "hf": ".local/bin/hf"}
"""Where bootstrap installs each upload tool, tried before PATH."""


def _find(name: str, env: Env) -> str | None:
    bundled = Path.home() / BUNDLED[name]
    if bundled.exists():
        return str(bundled)
    return shutil.which(name, path=None if env is None else env.get("PATH"))


def aws_binary(env: Env = None) -> str | None:
    return _find("aws", env)


def hf_binary(env: Env = None) -> str | None:
    return _find("hf", env)


def _binary(name: str, env: Env, purpose: str) -> str:
    found = (aws_binary if name == "aws" else hf_binary)(env)
    if found is None:
        raise SyncError(
            f"`{name}` CLI not found (looked in ~/{BUNDLED[name]} and PATH) on host "
            f"{_host_label()}; cannot upload {purpose}"
        )
    return found


def _upload(
    argv: list[str], local: Path, runner: CommandRunner, timeout: float | None, env: Env
) -> None:
    if not local.exists():
        raise MissingOutput(f"output path does not exist: {local}")
    result = runner(argv, timeout, env)
    if result.returncode != 0:
        _fail(result)


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


def exclude_args(flag: str, root: Path, min_age_s: float) -> list[str]:
    """`--exclude NAME` pairs for files still being written.

    Capped: a job that writes thousands of small files in a tick would
    otherwise build an argv megabytes long (and blow E2BIG). Skipping the tick
    costs nothing -- the next one picks the files up.
    """
    names = recently_modified(root, min_age_s)
    if len(names) > MAX_EXCLUDES:
        raise TooManyRecentFiles(
            f"{len(names)} files under {root} were modified in the last {min_age_s:g}s "
            f"(cap {MAX_EXCLUDES}); skipping this sync tick"
        )
    return _named_excludes(flag, names)


def sync_dir_to_s3(
    local: Path,
    dest: str,
    *,
    min_age_s: float = MIN_AGE_S,
    runner: CommandRunner = run_command,
    timeout: float | None = DEFAULT_TIMEOUT_S,
    env: Env = None,
    exclude: Sequence[str] = (),
) -> None:
    aws = _binary("aws", env, f"{local} to {dest}")
    argv = [aws, "s3", "sync", str(local), dest.rstrip("/"), "--only-show-errors"]
    argv += _named_excludes("--exclude", exclude)
    argv += exclude_args("--exclude", local, min_age_s)
    _upload(argv, local, runner, timeout, env)


def copy_file_to_s3(
    local: Path,
    dest: str,
    *,
    runner: CommandRunner = run_command,
    timeout: float | None = 300.0,
    env: Env = None,
) -> None:
    aws = _binary("aws", env, f"{local} to {dest}")
    _upload([aws, "s3", "cp", str(local), dest, "--only-show-errors"], local, runner, timeout, env)


def upload_dir_to_hf(
    local: Path,
    repo: str,
    path_in_repo: str,
    *,
    min_age_s: float = MIN_AGE_S,
    runner: CommandRunner = run_command,
    timeout: float | None = DEFAULT_TIMEOUT_S,
    env: Env = None,
    exclude: Sequence[str] = (),
) -> None:
    hf = _binary("hf", env, f"{local} to {repo}:{path_in_repo}")
    argv = [hf, "upload", repo, str(local), path_in_repo]
    argv += _named_excludes("--exclude", exclude)
    argv += exclude_args("--exclude", local, min_age_s)
    _upload(argv, local, runner, timeout, env)


def _named_excludes(flag: str, names: Sequence[str]) -> list[str]:
    args: list[str] = []
    for name in names:
        args += [flag, name]
    return args


def sync_output(
    output: Output,
    workdir: Path,
    job_id: str,
    *,
    min_age_s: float = MIN_AGE_S,
    runner: CommandRunner = run_command,
    timeout: float | None = DEFAULT_TIMEOUT_S,
    env: Env = None,
    baseline_entries: baseline.Entries | None = None,
) -> None:
    local = workdir / output.path.format(job_id=job_id)
    entries = baseline_entries or {}
    # Files that were in the checkout and have not been touched are not this
    # job's output; uploading them would publish the last run's results under
    # this run's name.
    exclude = () if baseline.too_many(entries) else tuple(baseline.unchanged(local, entries))
    if entries and local.exists() and not baseline.has_new_content(local, entries):
        raise MissingOutput(
            f"nothing new under {local}: every file there was already in the checkout"
        )
    if output.s3:
        sync_dir_to_s3(
            local,
            output.s3.format(job_id=job_id),
            min_age_s=min_age_s,
            runner=runner,
            timeout=timeout,
            env=env,
            exclude=exclude,
        )
    if output.hf:
        upload_dir_to_hf(
            local,
            output.hf.format(job_id=job_id),
            (output.hf_path or job_id).format(job_id=job_id),
            min_age_s=min_age_s,
            runner=runner,
            timeout=timeout,
            env=env,
            exclude=exclude,
        )


def sync_outputs(
    outputs: Sequence[Output],
    workdir: Path,
    job_id: str,
    *,
    min_age_s: float = MIN_AGE_S,
    runner: CommandRunner = run_command,
    timeout: float | None = DEFAULT_TIMEOUT_S,
    env: Env = None,
    baseline_map: baseline.Baseline | None = None,
) -> None:
    errors: list[SyncError] = []
    for output in outputs:
        try:
            sync_output(
                output,
                workdir,
                job_id,
                min_age_s=min_age_s,
                runner=runner,
                timeout=timeout,
                env=env,
                baseline_entries=(baseline_map or {}).get(baseline.output_key(output, job_id)),
            )
        except SyncError as exc:
            errors.append(exc)
    if not errors:
        return
    # Keep the specific kind when every output agrees, so the runner can still
    # tell "produced nothing" from "upload broke".
    kind = type(errors[0]) if len({type(e) for e in errors}) == 1 else SyncError
    raise kind("; ".join(str(e) for e in errors))


def sync_job_meta(
    job_id: str,
    s3_prefix: str | None,
    *,
    runner: CommandRunner = run_command,
    timeout: float | None = 300.0,
    env: Env = None,
) -> None:
    if not s3_prefix:
        return
    base = f"{s3_prefix.rstrip('/')}/jobs/{job_id}"
    for path in (paths.log_file(job_id), paths.state_file(job_id)):
        if path.exists():
            copy_file_to_s3(path, f"{base}/{path.name}", runner=runner, timeout=timeout, env=env)


def final_meta_sync(
    job_id: str,
    s3_prefix: str | None,
    *,
    runner: CommandRunner = run_command,
    timeout: float | None = 300.0,
    env: Env = None,
) -> str | None:
    """Mirror a finished job's log and state, then record that it happened.

    `meta_synced_at` is the precondition `purge` checks before deleting a job
    dir, so it is written only once the upload has actually succeeded -- a
    raised SyncError, or no `s3_prefix` at all, leaves it null and the job dir
    unpurgeable. Writing it changes `state.json`, so state goes up once more
    afterwards (a few hundred bytes) and the mirror matches the local file.

    Returns a warning when only that trailing PUT failed: the local state is
    the authority on "was this backed up", and log + state are already in S3,
    so a stale mirrored copy of `state.json` is not worth undoing the record.
    """
    if not s3_prefix:
        return None
    sync_job_meta(job_id, s3_prefix, runner=runner, timeout=timeout, env=env)
    prefix = s3_prefix.rstrip("/")
    jobs.update_state(job_id, meta_synced_at=jobs.utc_now(), meta_synced_to=prefix)
    state = paths.state_file(job_id)
    try:
        if state.exists():
            copy_file_to_s3(
                state,
                f"{prefix}/jobs/{job_id}/state.json",
                runner=runner,
                timeout=timeout,
                env=env,
            )
    except SyncError as exc:
        return f"the mirrored state.json is one revision behind (re-upload failed): {exc}"
    return None


class SyncLoop:
    """Background periodic sync; `final()` runs one last synchronous pass.

    The final pass has no wall-clock timeout: it is the last chance to save a
    run's outputs, and a 30-minute cap on a 200 GB checkpoint upload would
    throw away exactly the work that is most expensive to recompute.
    """

    def __init__(
        self,
        spec: JobSpec,
        workdir: Path,
        s3_prefix: str | None,
        *,
        runner: CommandRunner = run_command,
        min_age_s: float = MIN_AGE_S,
        env: Env = None,
    ) -> None:
        self._spec = spec
        self._workdir = workdir
        self._s3_prefix = s3_prefix
        self._runner = runner
        self._min_age_s = min_age_s
        # The job's environment, secrets included: uploads authenticate as the
        # job, not as whatever the dispatcher happened to inherit.
        self._env = env
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._tick_lock = threading.Lock()
        self.last_error: str | None = None
        self.outputs_synced_at: str | None = None
        """Set by the tick that uploaded `outputs:` without raising. `final()`
        clears it first, so after `final()` it is non-null only when the *last*
        upload -- the one that includes everything written since the last tick
        -- actually succeeded."""

    def _note(self, message: str) -> None:
        queue.note(self._spec.job_id, f"sync: {message}")

    def _record_error(self, message: str) -> None:
        self.last_error = message
        self._note(message)
        with contextlib.suppress(RuntimeError, OSError, KeyError):
            jobs.update_state(self._spec.job_id, sync_error=message)

    def _tick(self, min_age_s: float, timeout: float | None = DEFAULT_TIMEOUT_S) -> None:
        with self._tick_lock:
            sync_outputs(
                self._spec.outputs,
                self._workdir,
                self._spec.job_id,
                min_age_s=min_age_s,
                runner=self._runner,
                timeout=timeout,
                env=self._env,
                baseline_map=baseline.read(self._spec.job_id),
            )
            # Only when there was something to upload: a spec with no
            # `outputs:` has nothing to confirm, and a timestamp there would
            # read as a promise nobody made.
            if self._spec.outputs:
                self.outputs_synced_at = jobs.utc_now()
            sync_job_meta(
                self._spec.job_id,
                self._s3_prefix,
                runner=self._runner,
                timeout=timeout,
                env=self._env,
            )

    def _loop(self) -> None:
        interval = max(1, self._spec.sync_interval_s)
        while not self._stop.wait(interval):
            try:
                self._tick(self._min_age_s)
            except (TooManyRecentFiles, MissingOutput) as exc:
                self._note(f"WARNING: {exc}")
            except SyncError as exc:
                self._record_error(str(exc))
            except BaseException as exc:
                self._record_error(f"periodic sync raised {exc!r}\n{traceback.format_exc()}")

    def start(self) -> None:
        if not self._spec.outputs and not self._s3_prefix:
            return
        self._thread = threading.Thread(target=self._loop, name="gpuc-sync", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the loop and wait for any tick already in flight."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def final(self) -> None:
        """Stop the loop and do one complete sync, including files just written.

        `outputs_synced_at` is cleared first: an earlier tick having worked says
        nothing about the files the job wrote in its last minute, and `purge`
        reads that field as "everything this job produced is somewhere else".
        """
        self.stop()
        self.outputs_synced_at = None
        self._tick(min_age_s=0.0, timeout=None)
