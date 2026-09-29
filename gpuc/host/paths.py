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


def data_dir(environ: Mapping[str, str] | None = None) -> Path:
    """The host's data directory: `GPUC_DATA_DIR` if the host's `env` names
    one, else `data/` in gpuc home.

    Inside gpuc home rather than beside it like the caches: it holds what
    jobs chose to keep, which is this host's state as much as `jobs/` is, so
    it moves with a persistent root and goes with an `rm -rf` of gpuc home.
    """
    environ = os.environ if environ is None else environ
    override = environ.get("GPUC_DATA_DIR")
    return Path(override).expanduser() if override else home() / "data"


def scratch_dir(environ: Mapping[str, str] | None = None) -> Path | None:
    """Where workdirs live, if not in each job dir: `GPUC_SCRATCH_DIR` from
    the host's `env`.

    For a host whose gpuc home is on a volume that survives restarts but is
    too slow, or too small, for checkouts and their venvs. The queue stays on
    gpuc home; a restart that wipes scratch costs each job its workdir, which
    `checkout.restore` unpacks again from the job's archive.
    """
    environ = os.environ if environ is None else environ
    override = environ.get("GPUC_SCRATCH_DIR")
    return Path(override).expanduser() if override else None


def follow_scratch(host_env: Mapping[str, str]) -> None:
    """Make this process's scratch the host config's.

    Every process gpuc starts on a host gets the config's `env`, and reads
    scratch from its own environment. The dispatcher alone lives across a
    `gpuc host set --scratch-dir`: a rental's never exits while it is idle,
    and it sweeps workdirs and drains outputs, so it re-reads the config every
    pass and follows it here.
    """
    value = host_env.get("GPUC_SCRATCH_DIR")
    if value:
        os.environ["GPUC_SCRATCH_DIR"] = value
    else:
        os.environ.pop("GPUC_SCRATCH_DIR", None)


def workdirs_root(environ: Mapping[str, str] | None = None) -> Path:
    """Where jobs' workdirs, and so their venvs, are written."""
    return scratch_dir(environ) or home()


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
    """A job's secrets in plain text, as a client from before encryption
    delivers them; see `sealed`."""
    return secrets_dir() / f"{job_id}.env"


def sealed_env_file(job_id: str) -> Path:
    """A job's secrets encrypted to this host's key; see `sealed`."""
    return secrets_dir() / f"{job_id}.env.age"


def host_identity_file() -> Path:
    """This host's private key, which opens every `sealed_env_file`."""
    return secrets_dir() / "host.age"


def host_recipient_file() -> Path:
    """The public half of `host_identity_file`, which clients encrypt to."""
    return secrets_dir() / "host.age.pub"


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
    """`jobs/<id>/workdir`, or `<scratch>/<id>` on a host with scratch.

    A workdir already in the job dir stays the answer after scratch is
    configured, so the jobs from before it are still found where they are.
    """
    in_job_dir = job_dir(job_id) / "workdir"
    scratch = scratch_dir()
    if scratch is None or in_job_dir.exists():
        return in_job_dir
    return scratch / job_id


def checkout_archive(job_id: str) -> Path:
    """The checkout as submitted, kept only on a host with scratch."""
    return job_dir(job_id) / "checkout.tar.gz"


def partial_workdir(job_id: str) -> Path:
    """Where a checkout is unpacked before it is renamed into `workdir`."""
    return workdir(job_id).with_name(f".{job_id}.partial")


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


def owner_file() -> Path:
    return home() / "owner.json"


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
