"""Where a job's files go: one interface, one class per kind of store.

Everything that uploads -- the sync loop, the sync preflight, the final meta
mirror, the drain's last retry -- talks to a `Destination` and nothing else,
so adding a store is one class here and no change anywhere that uploads.
Both kinds shell out to a CLI the bootstrap installs into `$HOME` (`aws` v2,
`hf`); a missing binary fails the *job's* sync, never the queue.

Every upload takes an explicit ``env``. The runner passes the job's own
environment -- which includes its ``secrets:`` file -- so a job that declares
``secrets: [AWS_ACCESS_KEY_ID, ...]`` can upload without any host-level
credential file. ``env=None`` means "inherit this process's environment".
"""

from __future__ import annotations

import shutil
import subprocess
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from gpuc.host import jobs
from gpuc.host.jobs import Output

DEFAULT_TIMEOUT_S = 1800.0
PREFLIGHT_TIMEOUT_S = 120.0
HF_API = "https://huggingface.co/api"

Env = Mapping[str, str] | None


class SyncError(RuntimeError):
    pass


class MissingOutput(SyncError):
    """The output path the spec names is not there. Not the same failure as a
    broken upload: nothing was produced, so `sync` would be a misleading
    reason. Raised by `sync.sync_output`, the one place that looks."""


class PreflightFailed(RuntimeError):
    """Carries the command that failed and what it said, for the job log."""

    def __init__(self, command: str, detail: str) -> None:
        super().__init__(f"{command}\n{detail.strip()}")
        self.command = command
        self.detail = detail


@dataclass
class CommandResult:
    argv: list[str]
    returncode: int
    output: str


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
            f"`{' '.join(argv)}` on host {host_label()} timed out after {timeout}s"
        ) from exc
    except FileNotFoundError as exc:
        raise SyncError(f"`{argv[0]}` not found on host {host_label()}: {exc}") from exc
    except OSError as exc:
        raise SyncError(f"`{' '.join(argv)}` on host {host_label()} could not run: {exc}") from exc
    return CommandResult(argv, proc.returncode, (proc.stdout or "") + (proc.stderr or ""))


def host_label() -> str:
    """The host name for an error message, never a second failure."""
    try:
        return jobs.read_config().host
    except (RuntimeError, OSError, ValueError):
        return "unknown-host"


def _fail(result: CommandResult) -> None:
    tail = "\n".join(result.output.strip().splitlines()[-10:])
    raise SyncError(
        f"`{' '.join(result.argv)}` on host {host_label()} exited {result.returncode}\n{tail}"
    )


BUNDLED = {"aws": ".local/aws-cli/v2/current/bin/aws", "hf": ".local/bin/hf"}
"""Where bootstrap installs each upload tool, tried before PATH."""


def find_binary(name: str, env: Env = None) -> str | None:
    bundled = Path.home() / BUNDLED[name]
    if bundled.exists():
        return str(bundled)
    return shutil.which(name, path=None if env is None else env.get("PATH"))


def _binary(name: str, env: Env, purpose: str) -> str:
    found = find_binary(name, env)
    if found is None:
        raise SyncError(
            f"`{name}` CLI not found (looked in ~/{BUNDLED[name]} and PATH) on host "
            f"{host_label()}; cannot upload {purpose}"
        )
    return found


def _missing_tool(name: str, drop: str) -> PreflightFailed:
    return PreflightFailed(
        name,
        f"`{name}` CLI not found (looked in ~/{BUNDLED[name]} and PATH). Re-run "
        f"`gpuc host bootstrap` for this host, or drop the job's {drop} outputs.",
    )


def _excludes(names: Sequence[str]) -> list[str]:
    args: list[str] = []
    for name in names:
        args += ["--exclude", name]
    return args


def _run(argv: list[str], runner: CommandRunner, timeout: float | None, env: Env) -> None:
    result = runner(argv, timeout, env)
    if result.returncode != 0:
        _fail(result)


def of(output: Output, job_id: str) -> list[Destination]:
    """The destinations one spec `outputs:` entry names, `{job_id}` expanded."""
    found: list[Destination] = []
    if output.s3:
        found.append(S3(output.s3.format(job_id=job_id)))
    if output.hf:
        found.append(
            HuggingFace(
                output.hf.format(job_id=job_id),
                (output.hf_path or job_id).format(job_id=job_id),
                create=output.hf_create,
            )
        )
    return found


class Destination:
    """One place files can be put. `uri` names it, and is the key every
    record of an upload to it is stored under."""

    uri: str

    def upload_dir(
        self,
        local: Path,
        *,
        exclude: Sequence[str] = (),
        runner: CommandRunner = run_command,
        timeout: float | None = DEFAULT_TIMEOUT_S,
        env: Env = None,
    ) -> None:
        raise NotImplementedError

    def put_file(
        self,
        local: Path,
        name: str,
        *,
        runner: CommandRunner = run_command,
        timeout: float | None = 300.0,
        env: Env = None,
    ) -> None:
        """Upload one file as `name` directly under this destination."""
        raise NotImplementedError

    def preflight(
        self, probe: Path, *, runner: CommandRunner = run_command, env: Env = None
    ) -> None:
        """Prove a write here works with `env`, or raise PreflightFailed."""
        raise NotImplementedError


@dataclass
class S3(Destination):
    uri: str

    def __post_init__(self) -> None:
        self.uri = self.uri.rstrip("/")

    def upload_dir(
        self,
        local: Path,
        *,
        exclude: Sequence[str] = (),
        runner: CommandRunner = run_command,
        timeout: float | None = DEFAULT_TIMEOUT_S,
        env: Env = None,
    ) -> None:
        aws = _binary("aws", env, f"{local} to {self.uri}")
        argv = [aws, "s3", "sync", str(local), self.uri, "--only-show-errors", *_excludes(exclude)]
        _run(argv, runner, timeout, env)

    def put_file(
        self,
        local: Path,
        name: str,
        *,
        runner: CommandRunner = run_command,
        timeout: float | None = 300.0,
        env: Env = None,
    ) -> None:
        aws = _binary("aws", env, f"{local} to {self.uri}")
        _run(
            [aws, "s3", "cp", str(local), f"{self.uri}/{name}", "--only-show-errors"],
            runner,
            timeout,
            env,
        )

    def preflight(
        self, probe: Path, *, runner: CommandRunner = run_command, env: Env = None
    ) -> None:
        aws = find_binary("aws", env)
        if aws is None:
            raise _missing_tool("aws", "s3")
        argv = [aws, "s3", "cp", str(probe), f"{self.uri}/{probe.name}", "--only-show-errors"]
        result = runner(argv, PREFLIGHT_TIMEOUT_S, env)
        if result.returncode != 0:
            raise PreflightFailed(
                " ".join(argv),
                f"exited {result.returncode}\n{result.output.strip()[-800:]}\n"
                f"The job's environment must be able to write {self.uri}: check the spec's "
                f"`secrets:` (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY) and the bucket name.",
            )


RepoExists = Callable[[str, Env], "bool | None"]


def hf_repo_exists(repo: str, env: Env = None) -> bool | None:
    """Ask the Hub directly; None when the question could not be answered.

    The CLI has no "does this exist" that does not also create or download, and
    the difference between "missing" and "not ours to write" is the whole point
    of the message this feeds.
    """
    token = (env or {}).get("HF_TOKEN") or (env or {}).get("HUGGING_FACE_HUB_TOKEN")
    request = urllib.request.Request(f"{HF_API}/models/{repo}")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=30.0) as response:
            return 200 <= response.status < 300
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403, 404):
            return False
        return None
    except (urllib.error.URLError, OSError, ValueError):
        return None


@dataclass
class HuggingFace(Destination):
    repo: str
    path: str
    create: bool = False
    """Create the repo if the preflight finds it missing. Off by default: a
    typo in a repo name should fail the job in seconds, not quietly create
    `org/lego-s4-typo` and upload a run into it."""
    repo_exists: RepoExists = hf_repo_exists

    def __post_init__(self) -> None:
        self.path = self.path.strip("/")
        self.uri = f"hf://{self.repo}/{self.path}" if self.path else f"hf://{self.repo}"

    def _target(self, name: str) -> str:
        return f"{self.path}/{name}" if self.path else name

    def upload_dir(
        self,
        local: Path,
        *,
        exclude: Sequence[str] = (),
        runner: CommandRunner = run_command,
        timeout: float | None = DEFAULT_TIMEOUT_S,
        env: Env = None,
    ) -> None:
        hf = _binary("hf", env, f"{local} to {self.uri}")
        argv = [hf, "upload", self.repo, str(local), self.path, *_excludes(exclude)]
        _run(argv, runner, timeout, env)

    def put_file(
        self,
        local: Path,
        name: str,
        *,
        runner: CommandRunner = run_command,
        timeout: float | None = 300.0,
        env: Env = None,
    ) -> None:
        hf = _binary("hf", env, f"{local} to {self.uri}")
        _run([hf, "upload", self.repo, str(local), self._target(name)], runner, timeout, env)

    def preflight(
        self, probe: Path, *, runner: CommandRunner = run_command, env: Env = None
    ) -> None:
        hf = find_binary("hf", env)
        if hf is None:
            raise _missing_tool("hf", "hf")
        whoami = runner([hf, "auth", "whoami"], PREFLIGHT_TIMEOUT_S, env)
        if whoami.returncode != 0:
            raise PreflightFailed(
                f"{hf} auth whoami",
                f"exited {whoami.returncode}\n{whoami.output.strip()[-500:]}\n"
                f"The job's environment has no usable Hugging Face token: add HF_TOKEN to the "
                f"spec's `secrets:`.",
            )
        if self.create:
            create = runner(
                [hf, "repos", "create", self.repo, "--type", "model", "--exist-ok"],
                PREFLIGHT_TIMEOUT_S,
                env,
            )
            if create.returncode != 0:
                raise PreflightFailed(
                    f"{hf} repos create {self.repo} --type model --exist-ok",
                    f"exited {create.returncode}\n{create.output.strip()[-500:]}",
                )
        elif self.repo_exists(self.repo, env) is False:
            raise PreflightFailed(
                f"{hf} upload {self.repo}",
                f"the repo {self.repo} does not exist, or this token cannot see it. Create it "
                f"on the Hub, or set `hf_create: true` on that output to have gpuc create it.",
            )
        argv = [hf, "upload", self.repo, str(probe), self._target(probe.name)]
        result = runner(argv, PREFLIGHT_TIMEOUT_S, env)
        if result.returncode != 0:
            raise PreflightFailed(
                " ".join(argv),
                f"exited {result.returncode}\n{result.output.strip()[-800:]}\n"
                f"The job's token must be able to write {self.repo}.",
            )
