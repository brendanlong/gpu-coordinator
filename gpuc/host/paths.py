"""Layout of ``~/.gpuc`` (or ``$GPUC_HOME``).

Every path is a function rather than a module constant so that tests (and a
bootstrap that sets GPUC_HOME) can redirect the whole tree at any time.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path

USER_BIN_DIRS = (".local/bin", ".cargo/bin")
"""Where `uv` (and anything `uv tool install` puts down) lands in $HOME.

A pod's sshd hands out a PATH that has neither, so `uv run` -- which the
runner's own GPU preflight uses before the job's first command -- is not found
and every job fails identically. Both the dispatcher and the runner put these
in front of whatever PATH they inherited.
"""


def home() -> Path:
    override = os.environ.get("GPUC_HOME")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".gpuc"


def config_file() -> Path:
    return home() / "config.json"


def path_with_user_bins(environ: Mapping[str, str] | None = None, extra: Sequence[str] = ()) -> str:
    """``PATH`` with ``extra`` then the $HOME tool dirs in front, no duplicates.

    ``extra`` is for a host whose tools live off ``$HOME`` (see
    ``jobs.MANAGED_ENV``); those come first, because a host that names a
    tool directory explicitly means it.

    Unlike the $HOME entries, an ``extra`` directory is prepended even when it
    does not exist yet: whatever is going to create it may not have run, and
    the dispatcher must not need a second bootstrap to see it.
    """
    environ = os.environ if environ is None else environ
    current = environ.get("PATH", os.defpath)
    entries = current.split(os.pathsep)
    user_home = Path(environ.get("HOME") or Path.home())
    prefix = [d for d in extra if d and d not in entries]
    prefix += [
        str(user_home / name)
        for name in USER_BIN_DIRS
        if (user_home / name).is_dir() and str(user_home / name) not in entries
    ]
    return os.pathsep.join([*prefix, *entries]) if prefix else current


def secrets_dir() -> Path:
    return home() / "secrets"


def secrets_file(name: str) -> Path:
    return secrets_dir() / name


def job_env_file(job_id: str) -> Path:
    return secrets_dir() / f"{job_id}.env"


def incoming_dir() -> Path:
    """Job dirs a `gpuc submit` is still building.

    A job is accepted by renaming its dir from here into `jobs/`, so a job dir
    under `jobs/` is one the host was asked to run, by construction, and a dir
    left here is a submit that died. See `queue.enqueue`.
    """
    return home() / "incoming"


def incoming_job_dir(job_id: str) -> Path:
    return incoming_dir() / job_id


def jobs_dir() -> Path:
    return home() / "jobs"


def job_dir(job_id: str) -> Path:
    return jobs_dir() / job_id


def job_lock_file(job_id: str) -> Path:
    """The flock every read-modify-write of `state.json` takes.

    The dispatcher, the job's runner and a `python -m gpuc.host cancel` from
    over ssh all update the one file, each with an atomic replace; without the
    lock the last writer silently discards the others' fields."""
    return job_dir(job_id) / ".lock"


def spec_file(job_id: str) -> Path:
    return job_dir(job_id) / "spec.json"


def state_file(job_id: str) -> Path:
    return job_dir(job_id) / "state.json"


def log_file(job_id: str) -> Path:
    return job_dir(job_id) / "log.txt"


def workdir(job_id: str) -> Path:
    return job_dir(job_id) / "workdir"


def outputs_dir(job_id: str) -> Path:
    return job_dir(job_id) / "outputs"


def outputs_baseline_file(job_id: str) -> Path:
    """What was already under the job's `outputs:` paths when it started."""
    return job_dir(job_id) / "outputs_baseline.json"


def lock_file() -> Path:
    return home() / "dispatcher.lock"


def heartbeat_file() -> Path:
    return home() / "dispatcher.heartbeat"


def dispatcher_log() -> Path:
    return home() / "dispatcher.log"


def draining_file() -> Path:
    return home() / "draining"


def ensure_layout() -> None:
    home().mkdir(parents=True, exist_ok=True)
    incoming_dir().mkdir(parents=True, exist_ok=True)
    jobs_dir().mkdir(parents=True, exist_ok=True)
    secrets_dir().mkdir(parents=True, exist_ok=True)
    # The whole tree, not just secrets/: job dirs hold a workdir and logs that
    # other users of a shared box have no business reading.
    home().chmod(0o700)
    secrets_dir().chmod(0o700)


def ensure_job_layout(job_id: str) -> None:
    job_dir(job_id).mkdir(parents=True, exist_ok=True)
    workdir(job_id).mkdir(parents=True, exist_ok=True)
    outputs_dir(job_id).mkdir(parents=True, exist_ok=True)
