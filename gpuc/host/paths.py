"""Layout of ``~/.gpuc`` (or ``$GPUC_HOME``).

Every path is a function rather than a module constant so that tests (and a
bootstrap that sets GPUC_HOME) can redirect the whole tree at any time.
"""

from __future__ import annotations

import os
from pathlib import Path


def home() -> Path:
    override = os.environ.get("GPUC_HOME")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".gpuc"


def config_file() -> Path:
    return home() / "config.json"


def secrets_dir() -> Path:
    return home() / "secrets"


def secrets_file(name: str) -> Path:
    return secrets_dir() / name


def job_env_file(job_id: str) -> Path:
    return secrets_dir() / f"{job_id}.env"


def queue_dir() -> Path:
    return home() / "queue"


def jobs_dir() -> Path:
    return home() / "jobs"


def job_dir(job_id: str) -> Path:
    return jobs_dir() / job_id


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


def cancel_file(job_id: str) -> Path:
    return job_dir(job_id) / "cancel"


def lock_file() -> Path:
    return home() / "dispatcher.lock"


def heartbeat_file() -> Path:
    return home() / "dispatcher.heartbeat"


def dispatcher_log() -> Path:
    return home() / "dispatcher.log"


def draining_file() -> Path:
    return home() / "draining"


def paused_file() -> Path:
    return home() / "paused"


def ensure_layout() -> None:
    home().mkdir(parents=True, exist_ok=True)
    queue_dir().mkdir(parents=True, exist_ok=True)
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
