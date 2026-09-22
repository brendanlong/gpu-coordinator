"""Idempotent host bootstrap: uv, a Python >= 3.11, the package, config, health.

Re-running is always safe and is the supported fix for "the host looks wrong".
Anything optional (the `aws` and `hf` upload helpers) degrades to a warning:
a missing uploader must fail a job's sync step, never the queue.
"""

from __future__ import annotations

import json
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from gpuc._version import user_agent
from gpuc.control import probe as probe_mod
from gpuc.control.config import HostEntry, Reporter, Settings, transport_for, utc_now
from gpuc.control.gpuinfo import discover, summarize
from gpuc.control.remote import (
    HostSession,
    env_prefix,
    host_python,
    parse_last_json,
    read_remote_config,
    resolve_home,
    write_remote_config,
)
from gpuc.control.transport import Transport, TransportError, git_tracked_files
from gpuc.control.version import local_commit, package_root, short
from gpuc.host import jobs, paths
from gpuc.host.jobs import cache_beside

UV_INSTALLER = "https://astral.sh/uv/install.sh"
AWS_CLI_ZIP = "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip"
PYTHON_FLOOR = ".".join(str(part) for part in probe_mod.PYTHON_FLOOR)
PYTHON_INSTALL = "3.12"
INSTALL_TIMEOUT_S = 900.0
HEALTH_TIMEOUT_S = 300.0


class BootstrapError(RuntimeError):
    pass


@dataclass
class BootstrapResult:
    host: str
    home: str
    files: int
    dispatcher_pid: int
    pkg_commit: str | None = None
    """The gpuc commit this bootstrap shipped, as `gpuc version` reports it."""
    warnings: list[str] = field(default_factory=list)

    def render(self) -> str:
        line = (
            f"host {self.host} ready: {self.files} package files at {self.home}/pkg "
            f"({short(self.pkg_commit)}), dispatcher pid {self.dispatcher_pid}"
        )
        if self.warnings:
            line += f"\n{len(self.warnings)} warning(s) above"
        return line

    def document(self) -> dict[str, Any]:
        """`gpuc host bootstrap --json`, and one entry of `--all`'s `hosts[]`."""
        return {
            "host": self.host,
            "home": self.home,
            "files": self.files,
            "pkg_commit": self.pkg_commit,
            "dispatcher_pid": self.dispatcher_pid,
            "warnings": list(self.warnings),
        }

    @classmethod
    def no_document(cls) -> dict[str, Any]:
        """The same keys as `document()`, for a host that was never bootstrapped."""
        return {
            "host": None,
            "home": None,
            "files": None,
            "pkg_commit": None,
            "dispatcher_pid": None,
            "warnings": [],
        }


def package_files(root: Path | None = None) -> list[str]:
    """Paths under ``gpuc/`` to ship, relative to the repo root.

    git is the source of truth so that editor droppings, ``__pycache__`` and
    local experiments never reach a host; a non-git install falls back to the
    installed tree.
    """
    root = root or package_root()
    try:
        tracked = git_tracked_files(root)
    except TransportError:
        tracked = []
    files = sorted(f for f in tracked if f.startswith("gpuc/") and f.endswith(".py"))
    if files:
        return files
    return sorted(
        str(p.relative_to(root))
        for p in (root / "gpuc").rglob("*.py")
        if "__pycache__" not in p.parts
    )


def _first_line(text: str) -> str:
    return text.strip().splitlines()[0].strip() if text.strip() else ""


def remote_path(entry: HostEntry) -> str:
    """The ``PATH=`` assignment for anything we start on the host.

    A pod's sshd hands out a PATH with none of these, and the dispatcher's
    environment is what every runner and every job command inherits. A host
    whose `env` names its own tool directories gets those in front.
    """
    head = "".join(f"{d}:" for d in entry.config.bin_dirs())
    user = "".join(f"$HOME/{d}:" for d in paths.USER_BIN_DIRS)
    return f'PATH="{head}{user}$PATH"'


def ensure_persistent_root(transport: Transport, entry: HostEntry, report: Reporter) -> None:
    """Create the persistent root 0700, leaving an existing root's mode alone.

    Only the root itself: everything inside it is gpuc home, which
    `paths.ensure_layout` already creates 0700. The root is usually a per-user
    directory on a world-writable shared volume, so a default-umask mkdir would
    leave the first level readable by every other user of the cluster; but an
    existing root may hold the user's own unrelated work, and re-chmodding
    someone's directory from under them is not ours to do.
    """
    root = entry.root
    if root is None:
        return
    result = transport.run(
        f"set -e; root={shlex.quote(root)}; "
        'if [ ! -d "$root" ]; then mkdir -p "$root"; chmod 700 "$root"; fi; '
        'ls -ld "$root"',
        check=False,
    )
    if result.returncode != 0:
        raise BootstrapError(
            f"could not prepare the persistent root {root} on host {entry.name} "
            f"(exit {result.returncode}):\n{result.output.strip()[-800:]}\n"
            f"Check that {root} (or its parent) is writable by this user, or point the host "
            f"somewhere else with `gpuc host set {entry.name} --persistent-root PATH`."
        )
    report(f"persistent root: {_first_line(result.stdout)}")


def uv_installer_command() -> str:
    return f"curl -LsSf -A {shlex.quote(user_agent())} {UV_INSTALLER} | sh"


def find_uv(transport: Transport) -> str | None:
    result = transport.run(
        'if [ -x "$HOME/.local/bin/uv" ]; then echo "$HOME/.local/bin/uv"; '
        "else command -v uv 2>/dev/null; fi",
        check=False,
    )
    return _first_line(result.stdout) or None


def install_uv(transport: Transport) -> str:
    transport.run(uv_installer_command(), timeout=INSTALL_TIMEOUT_S, check=False)
    uv = find_uv(transport)
    if uv is None:
        raise BootstrapError(
            f"uv is still missing after `{uv_installer_command()}` on host "
            f"{transport.host}.\nCheck outbound HTTPS on the host, or install uv by hand "
            f"into ~/.local/bin and re-run bootstrap."
        )
    return uv


def find_python(transport: Transport, uv: str, entry: HostEntry) -> str | None:
    # From $HOME, without the project and without an inherited VIRTUAL_ENV:
    # otherwise `uv python find` returns the venv of whatever directory the
    # control side was invoked from (`uv run gpuc ...` exports VIRTUAL_ENV),
    # and the host would be pinned to an interpreter that can disappear.
    result = transport.run(
        f'cd "$HOME" && {env_prefix(entry.env)}env -u VIRTUAL_ENV -u UV_PROJECT_ENVIRONMENT '
        f"{shlex.quote(uv)} python find --no-project '>={PYTHON_FLOOR}' 2>/dev/null",
        check=False,
    )
    path = _first_line(result.stdout)
    return path or None


def ensure_python(transport: Transport, uv: str, entry: HostEntry, report: Reporter) -> str:
    python = find_python(transport, uv, entry)
    if python:
        return python
    report(f"installing Python {PYTHON_INSTALL} with uv (no interpreter >= {PYTHON_FLOOR} found)")
    result = transport.run(
        f"{env_prefix(entry.env)}{shlex.quote(uv)} python install {PYTHON_INSTALL}",
        timeout=INSTALL_TIMEOUT_S,
        check=False,
    )
    python = find_python(transport, uv, entry)
    if python is None:
        raise BootstrapError(
            f"`uv python install {PYTHON_INSTALL}` on host {transport.host} left no usable "
            f"interpreter (exit {result.returncode}):\n{result.output.strip()[-800:]}"
        )
    return python


def sync_package(transport: Transport, home: str, report: Reporter) -> int:
    files = package_files()
    if not files:
        raise BootstrapError(
            f"found no gpuc/*.py files to ship from {package_root()}; "
            f"run bootstrap from a checkout of the repository."
        )
    # Removed first, not merged into: rsync of a file list only ever adds, so a
    # module deleted upstream would stay on the host and keep being imported --
    # this build's code calling last week's module is the hardest kind of wrong
    # to see from here.
    transport.run(f'rm -rf "{home}/pkg/gpuc"', check=True)
    transport.run(f'mkdir -p "{home}/pkg"', check=True)
    report(f"rsyncing {len(files)} package files to {home}/pkg")
    transport.rsync(package_root(), f"{home}/pkg", files)
    transport.run(
        f'find "{home}/pkg" -name __pycache__ -type d -prune -exec rm -rf {{}} +', check=False
    )
    return len(files)


def find_aws_cli(transport: Transport) -> str | None:
    install_dir = "$HOME/.local/aws-cli"
    present = transport.run(
        f'if [ -x "{install_dir}/v2/current/bin/aws" ]; then '
        f'echo "{install_dir}/v2/current/bin/aws"; '
        "else command -v aws 2>/dev/null; fi",
        check=False,
    )
    return _first_line(present.stdout) or None


def ensure_aws_cli(transport: Transport, report: Reporter) -> str | None:
    install_dir = "$HOME/.local/aws-cli"
    bin_dir = "$HOME/.local/bin"
    if find_aws_cli(transport):
        return None
    report("installing the aws CLI v2 bundle into ~/.local/aws-cli")
    result = transport.run(
        "set -e; tmp=$(mktemp -d); trap 'rm -rf \"$tmp\"' EXIT; "
        # No -A here for the bundle itself: this is a plain S3 object fetch, and
        # the aws CLI's own User-Agent is not overridable anyway.
        f'curl -LsSf {AWS_CLI_ZIP} -o "$tmp/awscliv2.zip"; '
        # No unzip on a slim container image, and no sudo to install one; the
        # stdlib module is always there, but it drops the exec bits.
        'if command -v unzip >/dev/null 2>&1; then unzip -q "$tmp/awscliv2.zip" -d "$tmp"; '
        'else python3 -m zipfile -e "$tmp/awscliv2.zip" "$tmp" && chmod -R u+x "$tmp/aws"; fi; '
        f'"$tmp/aws/install" -i "{install_dir}" -b "{bin_dir}"',
        timeout=INSTALL_TIMEOUT_S,
        check=False,
    )
    if result.returncode != 0 or not find_aws_cli(transport):
        return (
            f"aws CLI install failed (exit {result.returncode}); jobs with s3 outputs will "
            f"fail their sync step until it is installed:\n{result.output.strip()[-500:]}"
        )
    return None


def ensure_hf_cli(transport: Transport, uv: str, entry: HostEntry, report: Reporter) -> str | None:
    bin_dir = "$HOME/.local/bin"
    present = transport.run(
        f'if [ -x "{bin_dir}/hf" ]; then echo "{bin_dir}/hf"; else command -v hf 2>/dev/null; fi',
        check=False,
    )
    if _first_line(present.stdout):
        return None
    report("installing huggingface_hub as a uv tool")
    result = transport.run(
        f"{env_prefix(entry.env)}{shlex.quote(uv)} tool install huggingface_hub",
        timeout=INSTALL_TIMEOUT_S,
        check=False,
    )
    if result.returncode != 0:
        return (
            f"`uv tool install huggingface_hub` failed (exit {result.returncode}); jobs with hf "
            f"outputs will fail their sync step:\n{result.output.strip()[-500:]}"
        )
    return None


UV_CACHE_PROBE = """\
set -e
home={home}
mkdir -p "$home" 2>/dev/null || true
cache=$({env}{uv} cache dir 2>/dev/null || true)
[ -n "$cache" ] || cache="$HOME/.cache/uv"
mkdir -p "$cache" 2>/dev/null || true
echo "cache=$cache"
echo "cache_dev=$(stat -c %d "$cache" 2>/dev/null || echo unknown)"
echo "home_dev=$(stat -c %d "$home" 2>/dev/null || echo unknown)"
"""


def resolve_cache_dir(
    transport: Transport, entry: HostEntry, uv: str, home: str, report: Reporter
) -> str | None:
    """Decide this host's `UV_CACHE_DIR`. `None` means uv's default is right.

    uv builds a venv by reflinking or hardlinking wheels out of `~/.cache/uv`,
    and both only work inside a single filesystem; across one it silently falls
    back to copying. On a host whose gpuc home is on a network volume and whose
    `$HOME` is a container's overlay (a RunPod pod with `--persistent-root
    /workspace/...`) that costs the full size of every venv -- ~6.5 GB for
    torch -- written to the slowest disk the host has, on every single job.

    The rule is one comparison and nothing cleverer: if gpuc home and uv's
    cache are on different filesystems, move the cache next to gpuc home. A
    cache the host's config already names is never overridden, however it got
    there (`--cache-dir`, `--env UV_CACHE_DIR=...`, or an earlier bootstrap):
    this is the one key bootstrap fills in itself, and only when it is empty.
    """
    pinned = entry.env.get("UV_CACHE_DIR")
    if pinned:
        report(f"uv cache: {pinned} (set for this host; left alone)")
        return None
    script = UV_CACHE_PROBE.format(
        home=shlex.quote(home), env=env_prefix(entry.env), uv=shlex.quote(uv)
    )
    result = transport.run(script, check=False)
    if result.returncode != 0:
        report(f"WARNING: could not read the uv cache location on {entry.name}; leaving it default")
        return None
    values = dict(line.split("=", 1) for line in result.stdout.splitlines() if line.count("=") >= 1)
    cache, cache_dev, home_dev = (
        values.get("cache", ""),
        values.get("cache_dev", "unknown"),
        values.get("home_dev", "unknown"),
    )
    if "unknown" in (cache_dev, home_dev) or not cache:
        report(f"uv cache: {cache or 'unknown'} (could not compare filesystems; left alone)")
        return None
    if cache_dev == home_dev:
        report(f"uv cache: {cache}, same filesystem as {home}; uv will link wheels into venvs")
        return None
    target = cache_beside(home, "uv")
    report(
        f"uv cache: {cache} is on a different filesystem from gpuc home {home}, so uv would "
        f"copy every wheel into every venv. Setting UV_CACHE_DIR={target} for this host."
    )
    return target


def derive_env(
    transport: Transport, entry: HostEntry, uv: str, home: str, report: Reporter
) -> dict[str, str]:
    """The managed env keys this host names nothing for, filled in.

    `UV_CACHE_DIR` is the one with a real decision behind it
    (`resolve_cache_dir`); every other `beside_home` key in `jobs.MANAGED_ENV`
    lands beside gpuc home on a host with a persistent root, so the root keeps
    the Hugging Face cache the way it keeps uv's. A key the host already has is
    never touched.
    """
    derived: dict[str, str] = {}
    cache_dir = resolve_cache_dir(transport, entry, uv, home, report)
    if cache_dir:
        derived["UV_CACHE_DIR"] = cache_dir
    if entry.root is None:
        # Only a persistent root moves a cache: on an ordinary host the
        # default location is the user's own, holding their models and their
        # `hf auth login` token, and pointing jobs elsewhere would lose both.
        return derived
    for key, managed in jobs.MANAGED_ENV.items():
        if key == "UV_CACHE_DIR" or not managed.beside_home or entry.env.get(key):
            continue
        derived[key] = cache_beside(home, managed.beside_home)
        report(f"{key}: {derived[key]} (beside gpuc home, on the persistent root)")
    return derived


def ensure_layout(transport: Transport, entry: HostEntry, home: str, python: str) -> None:
    """Create gpuc home and its subdirectories 0700, using the host's own code."""
    transport.run(
        f'{host_python(python, home, entry.env)} -c "from gpuc.host import paths; '
        f'paths.ensure_layout()"',
        check=True,
    )


def run_health(session: HostSession, health_args: str = "") -> dict[str, Any]:
    args = f"health {health_args}".strip()
    result = session.host_cli(args, timeout=HEALTH_TIMEOUT_S, check=False)
    report = parse_last_json(result.stdout)
    if not isinstance(report, dict):
        raise BootstrapError(
            f"host health on {session.entry.name} produced no JSON (exit {result.returncode}):\n"
            f"{result.output.strip()[-1500:]}"
        )
    if not report.get("ok"):
        failed = [c for c in report.get("checks", []) if not c.get("ok")]
        summary = "\n".join(f"  - {c['name']}: {c['detail']}" for c in failed)
        raise BootstrapError(
            f"host health failed on {session.entry.name}:\n{summary}\n\n"
            f"{json.dumps(report, indent=2)}\n"
            f"Fix the failing check (disk, driver, or network) and re-run "
            f"`gpuc host bootstrap {session.entry.name}`."
        )
    return report


def driver_version(health: dict[str, Any]) -> str | None:
    """The `driver` check's value, so `gpuc host list` can name it for free."""
    for check in health.get("checks", []):
        if check.get("name") == "driver" and isinstance(check.get("value"), str):
            return str(check["value"])
    return None


STALE_UNITS = ("gpuc-reconcile.timer", "gpuc-reconcile.service")
"""Units an earlier build installed and no build serves any more; left in
place they fail every minute for ever. Removed best-effort on every bootstrap."""


def remove_stale_units(transport: Transport) -> None:
    units = " ".join(STALE_UNITS)
    files = " ".join(f"$HOME/.config/systemd/user/{unit}" for unit in STALE_UNITS)
    transport.run(
        f"systemctl --user disable --now {units} >/dev/null 2>&1; rm -f {files}; true",
        check=False,
    )


def start_dispatcher(session: HostSession) -> int:
    # The dispatcher re-derives PATH and the host env from config.json itself;
    # setting them here means the very first process in the chain already has
    # them, before it has read anything.
    command = (
        f"{remote_path(session.entry)} "
        f"{host_python(session.python, session.home, session.entry.env)} "
        f'-c "from gpuc.host import dispatcher; print(dispatcher.spawn_detached_dispatcher())"'
    )
    result = session.transport.run(command, check=False)
    pid = _first_line(result.stdout)
    if result.returncode != 0 or not pid.isdigit():
        raise BootstrapError(
            f"could not start the dispatcher on {session.entry.name} (exit {result.returncode}):\n"
            f"{result.output.strip()[-800:]}\n  command: {command}"
        )
    return int(pid)


def bootstrapped_provider(entry: HostEntry) -> dict[str, Any]:
    """`{"provider": ...}` with this moment stamped on it, for a rented host.

    A pod carries its own record of what it was bought as (`rented`), and this
    is the stamp that says it was set up, written by whichever machine did so.
    Empty for a host nobody is renting: inventing a provider block for one
    would make it read as a pod.

    The question is whether this host is *rented*, not whether its config
    already says so: a pod set up before the block existed has none, and it is
    the address that knows it is a pod.
    """
    provider = entry.config.provider or entry.provider()
    if provider is None:
        return {}
    return {"provider": {**provider, "bootstrapped_at": utc_now()}}


def read_host_config(transport: Transport, entry: HostEntry, home: str) -> dict[str, Any]:
    """The host's own config, or a refusal: `{}` means it has none, never that
    we could not tell. Bootstrap replaces a config that is not there; a config
    that is there and unreadable is a file the host is running on."""
    document = read_remote_config(transport, home)
    if document is None:
        raise BootstrapError(
            f"{home}/config.json on host {entry.name} could not be read, and bootstrap will not "
            f"replace a config it cannot see.\nFix the file (it should be JSON), or delete it "
            f"and run this again to write a fresh one."
        )
    return document


def resync_package(
    entry: HostEntry,
    settings: Settings | None = None,
    *,
    transport: Transport | None = None,
    report: Reporter = print,
) -> HostEntry:
    """Ship this build's package to a host and start the dispatcher again.

    The half of bootstrap that goes stale. uv, the interpreter and the upload
    CLIs cannot have changed since the host was bootstrapped, and health takes
    minutes, so `gpuc submit` re-runs only this before it enqueues: a host
    still running last week's package would otherwise dispatch the job with
    code that no longer matches the spec this machine just wrote.

    The only thing it writes to the host's config is the commit it just
    shipped. Returns the entry with the host's answer cached; the caller
    persists it.
    """
    transport = transport or transport_for(entry, settings)
    home = resolve_home(transport, entry)
    sync_package(transport, home, report)
    patch: dict[str, Any] = {"pkg_commit": local_commit()}
    if not read_host_config(transport, entry, home):
        # The host has lost its config (a wiped $HOME, most often a pod that
        # restarted). Restoring the last one seen is better than dispatching
        # this job to a host that now believes it owns no cards at all.
        patch = {**entry.initial_config().to_dict(), **patch}
        report(f"{home}/config.json was missing; restoring the last one seen")
    document = write_remote_config(transport, home, patch, python=entry.python, env=entry.env)
    updated = entry.with_config(document)
    start_dispatcher(HostSession(updated, transport, home, updated.python or ""))
    return updated


def bootstrap_host(
    entry: HostEntry,
    settings: Settings | None = None,
    *,
    transport: Transport | None = None,
    report: Reporter = print,
    health_args: str = "",
) -> tuple[HostEntry, BootstrapResult]:
    """Bring a host to a state where `python -m gpuc.host` runs and dispatches.

    Installs, ships and starts things; it does **not** configure the host.
    What the host is -- its cards, its mirror, its env, its timers -- is
    `config.json`'s and stays the host's, so the only keys bootstrap writes are
    the commit it just shipped and, on a host that has never had one,
    `UV_CACHE_DIR`. The exception is a host with no config at all (one
    registered before this split, or one whose gpuc home was wiped): there is
    nothing to preserve, so the last config this machine saw is restored.

    Returns the entry with what the host said cached; the caller persists it.
    """
    transport = transport or transport_for(entry, settings)
    warnings: list[str] = []

    ensure_persistent_root(transport, entry, report)
    home = resolve_home(transport, entry)

    # The host's own config decides every environment below -- which uv cache
    # the installs populate, which tool directories go on PATH -- so it is read
    # before anything else runs.
    existing = read_host_config(transport, entry, home)
    entry = entry.with_config(existing) if existing else entry
    patch: dict[str, Any] = {}
    if not existing:
        patch.update(entry.initial_config().to_dict())
        entry = entry.with_config(patch)
        report(
            f"{home}/config.json does not exist on {entry.name}: initialising it with "
            f"{len(entry.gpus)} GPU(s)"
        )
        if not entry.gpus:
            warnings.append(
                f"host {entry.name} has no config of its own and this machine has none cached "
                f"for it, so it will run nothing until "
                f"`gpuc host set {entry.name} --gpus <list>`"
            )
            report(f"WARNING: {warnings[-1]}")

    uv = find_uv(transport)
    if uv is None:
        report("installing uv into ~/.local/bin")
        uv = install_uv(transport)
    report(f"uv: {uv}")

    python = ensure_python(transport, uv, entry, report)
    report(f"python: {python}")

    files = sync_package(transport, home, report)

    # Before `uv tool install`, so that call already populates the cache this
    # host will actually use.
    derived = derive_env(transport, entry, uv, home, report)
    if derived:
        patch["env"] = {**entry.env, **derived}
        entry = entry.with_config({**entry.cache.config, **patch})

    aws_warning = ensure_aws_cli(transport, report)
    if aws_warning and entry.s3_prefix:
        # This host is registered to mirror every job's log and state to S3. With
        # no `aws` there, each job's final sync fails, every job ends
        # `failed: sync`, and nothing is ever purgeable -- a broken host that
        # looks bootstrapped is worse than a bootstrap that says no.
        raise BootstrapError(
            f"host {entry.name} has s3_prefix {entry.s3_prefix} but the aws CLI could not be "
            f"installed:\n{aws_warning}\n"
            f"Install it by hand into ~/.local/aws-cli, or drop the mirror with "
            f"`gpuc host set {entry.name} --s3-prefix ''`."
        )
    for warning in (aws_warning, ensure_hf_cli(transport, uv, entry, report)):
        if warning:
            warnings.append(warning)
            report(f"WARNING: {warning}")

    # Written before health runs, so the checks judge the config the host is
    # about to dispatch with, and so its `pkg_commit` says which commit this
    # package came from while `gpuc status` is still watching.
    ensure_layout(transport, entry, home, python)
    commit = local_commit()
    patch["pkg_commit"] = commit
    patch.update(bootstrapped_provider(entry))
    entry = entry.with_config(
        write_remote_config(transport, home, patch, python=python, env=entry.env)
    )
    report(f"{home}/config.json: {', '.join(sorted(patch))} (the rest is the host's)")

    session = HostSession(entry, transport, home, python)
    health = run_health(session, health_args)
    report("health: " + "; ".join(f"{c['name']} ok" for c in health.get("checks", [])))
    for warning in health.get("warnings", []):
        warnings.append(warning)
        report(f"WARNING: {warning}")

    remove_stale_units(transport)
    pid = start_dispatcher(session)
    report(f"dispatcher running (pid {pid})")

    gpu_info = discover(transport)
    if entry.gpus:
        report(f"gpus: {summarize(entry.gpus, gpu_info or entry.gpu_info)}")
    updated = entry.with_cache(
        uv=uv,
        python=python,
        gpu_info=gpu_info or None,
        driver_version=driver_version(health),
    ).model_copy(update={"bootstrapped_at": utc_now()})
    return updated, BootstrapResult(
        host=entry.name,
        home=home,
        files=files,
        pkg_commit=commit,
        dispatcher_pid=pid,
        warnings=warnings,
    )
