"""Idempotent host bootstrap: uv, a Python >= 3.11, the package, config, health.

Re-running is always safe and is the supported fix for "the host looks wrong".
Anything optional (the `aws` and `hf` upload helpers) degrades to a warning:
a missing uploader must fail a job's sync step, never the queue.
"""

from __future__ import annotations

import json
import shlex
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import gpuc
from gpuc.control.config import HostEntry, Settings, transport_for, utc_now
from gpuc.control.remote import HostSession, resolve_home
from gpuc.control.transport import Transport, TransportError, git_tracked_files

UV_INSTALLER = "https://astral.sh/uv/install.sh"
AWS_CLI_ZIP = "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip"
PYTHON_FLOOR = "3.11"
PYTHON_INSTALL = "3.12"
INSTALL_TIMEOUT_S = 900.0
HEALTH_TIMEOUT_S = 300.0

Reporter = Callable[[str], None]


class BootstrapError(RuntimeError):
    pass


@dataclass
class BootstrapResult:
    host: str
    uv: str
    python: str
    home: str
    files: int
    health: dict[str, Any]
    dispatcher_pid: int
    warnings: list[str] = field(default_factory=list)


def package_root() -> Path:
    return Path(gpuc.__file__).resolve().parents[1]


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


def find_uv(transport: Transport) -> str | None:
    result = transport.run(
        'if [ -x "$HOME/.local/bin/uv" ]; then echo "$HOME/.local/bin/uv"; '
        "else command -v uv 2>/dev/null; fi",
        check=False,
    )
    return _first_line(result.stdout) or None


def install_uv(transport: Transport) -> str:
    transport.run(f"curl -LsSf {UV_INSTALLER} | sh", timeout=INSTALL_TIMEOUT_S, check=False)
    uv = find_uv(transport)
    if uv is None:
        raise BootstrapError(
            f"uv is still missing after `curl -LsSf {UV_INSTALLER} | sh` on host "
            f"{transport.host}.\nCheck outbound HTTPS on the host, or install uv by hand "
            f"into ~/.local/bin and re-run bootstrap."
        )
    return uv


def find_python(transport: Transport, uv: str) -> str | None:
    # From $HOME, without the project and without an inherited VIRTUAL_ENV:
    # otherwise `uv python find` returns the venv of whatever directory the
    # control side was invoked from (`uv run gpuc ...` exports VIRTUAL_ENV),
    # and the host would be pinned to an interpreter that can disappear.
    result = transport.run(
        f'cd "$HOME" && env -u VIRTUAL_ENV -u UV_PROJECT_ENVIRONMENT {shlex.quote(uv)} '
        f"python find --no-project '>={PYTHON_FLOOR}' 2>/dev/null",
        check=False,
    )
    path = _first_line(result.stdout)
    return path or None


def ensure_python(transport: Transport, uv: str, report: Reporter) -> str:
    python = find_python(transport, uv)
    if python:
        return python
    report(f"installing Python {PYTHON_INSTALL} with uv (no interpreter >= {PYTHON_FLOOR} found)")
    result = transport.run(
        f"{shlex.quote(uv)} python install {PYTHON_INSTALL}",
        timeout=INSTALL_TIMEOUT_S,
        check=False,
    )
    python = find_python(transport, uv)
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
    transport.run(f'mkdir -p "{home}/pkg"', check=True)
    report(f"rsyncing {len(files)} package files to {home}/pkg")
    transport.rsync(package_root(), f"{home}/pkg", files)
    transport.run(
        f'find "{home}/pkg" -name __pycache__ -type d -prune -exec rm -rf {{}} +', check=False
    )
    return len(files)


def ensure_aws_cli(transport: Transport, report: Reporter) -> str | None:
    present = transport.run(
        'if [ -x "$HOME/.local/aws-cli/v2/current/bin/aws" ]; then '
        'echo "$HOME/.local/aws-cli/v2/current/bin/aws"; '
        "else command -v aws 2>/dev/null; fi",
        check=False,
    )
    if _first_line(present.stdout):
        return None
    report("installing the aws CLI v2 bundle into ~/.local/aws-cli")
    result = transport.run(
        "set -e; tmp=$(mktemp -d); trap 'rm -rf \"$tmp\"' EXIT; "
        f'curl -LsSf {AWS_CLI_ZIP} -o "$tmp/awscliv2.zip"; '
        'unzip -q "$tmp/awscliv2.zip" -d "$tmp"; '
        '"$tmp/aws/install" -i "$HOME/.local/aws-cli" -b "$HOME/.local/bin"',
        timeout=INSTALL_TIMEOUT_S,
        check=False,
    )
    if result.returncode != 0:
        return (
            f"aws CLI install failed (exit {result.returncode}); jobs with s3 outputs will "
            f"fail their sync step until it is installed:\n{result.output.strip()[-500:]}"
        )
    return None


def ensure_hf_cli(transport: Transport, uv: str, report: Reporter) -> str | None:
    present = transport.run(
        'if [ -x "$HOME/.local/bin/hf" ]; then echo "$HOME/.local/bin/hf"; '
        "else command -v hf 2>/dev/null; fi",
        check=False,
    )
    if _first_line(present.stdout):
        return None
    report("installing huggingface_hub as a uv tool")
    result = transport.run(
        f"{shlex.quote(uv)} tool install huggingface_hub", timeout=INSTALL_TIMEOUT_S, check=False
    )
    if result.returncode != 0:
        return (
            f"`uv tool install huggingface_hub` failed (exit {result.returncode}); jobs with hf "
            f"outputs will fail their sync step:\n{result.output.strip()[-500:]}"
        )
    return None


def write_host_config(session: HostSession, entry: HostEntry) -> None:
    session.transport.run(
        f'GPUC_HOME="{session.home}" PYTHONPATH="{session.home}/pkg" "{session.python}" '
        f'-c "from gpuc.host import paths; paths.ensure_layout()"',
        check=True,
    )
    session.transport.put_file(
        json.dumps(entry.host_config().to_dict(), indent=2, sort_keys=True) + "\n",
        f"{session.home}/config.json",
        0o644,
    )


def run_health(session: HostSession, health_args: str = "") -> dict[str, Any]:
    args = f"health {health_args}".strip()
    result = session.host_cli(args, timeout=HEALTH_TIMEOUT_S, check=False)
    try:
        report: dict[str, Any] = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise BootstrapError(
            f"host health on {session.entry.name} produced no JSON (exit {result.returncode}):\n"
            f"{result.output.strip()[-1500:]}\n({exc})"
        ) from exc
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


def start_dispatcher(session: HostSession) -> int:
    command = (
        f'GPUC_HOME="{session.home}" PYTHONPATH="{session.home}/pkg" "{session.python}" '
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


def bootstrap_host(
    entry: HostEntry,
    settings: Settings | None = None,
    *,
    transport: Transport | None = None,
    report: Reporter = print,
    health_args: str = "",
) -> tuple[HostEntry, BootstrapResult]:
    """Bring a host to a state where `python -m gpuc.host` runs and dispatches.

    Returns the registry entry updated with the discovered tool paths; the
    caller persists it.
    """
    transport = transport or transport_for(entry, settings)
    warnings: list[str] = []

    uv = find_uv(transport)
    if uv is None:
        report("installing uv into ~/.local/bin")
        uv = install_uv(transport)
    report(f"uv: {uv}")

    python = ensure_python(transport, uv, report)
    report(f"python: {python}")

    home = resolve_home(transport, entry)
    files = sync_package(transport, home, report)

    for warning in (ensure_aws_cli(transport, report), ensure_hf_cli(transport, uv, report)):
        if warning:
            warnings.append(warning)
            report(f"WARNING: {warning}")

    session = HostSession(entry, transport, home, python)
    write_host_config(session, entry)
    report(f"wrote {home}/config.json for host {entry.name} with {len(entry.gpus)} GPU(s)")

    health = run_health(session, health_args)
    report("health: " + "; ".join(f"{c['name']} ok" for c in health.get("checks", [])))

    pid = start_dispatcher(session)
    report(f"dispatcher running (pid {pid})")

    updated = entry.model_copy(update={"uv": uv, "python": python, "bootstrapped_at": utc_now()})
    return updated, BootstrapResult(
        host=entry.name,
        uv=uv,
        python=python,
        home=home,
        files=files,
        health=health,
        dispatcher_pid=pid,
        warnings=warnings,
    )
