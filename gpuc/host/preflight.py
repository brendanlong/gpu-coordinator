"""Prove the job's uploads can work *before* it spends an hour computing them.

Every failure this catches used to surface at the final sync: a job that ran
for hours ends `failed: sync` holding the only copy of its outputs, because a
secret was missing, the bucket was a typo, or the Hugging Face repo did not
exist. Here the same failure costs seconds and the job never starts.

Stdlib only, like everything under `gpuc.host`; each destination proves itself
with the same binary the sync loop uploads with.
"""

from __future__ import annotations

from pathlib import Path

from gpuc.host import destinations, jobs, paths, sync
from gpuc.host.destinations import CommandRunner, Env, PreflightFailed, RepoExists, run_command
from gpuc.host.jobs import HostConfig, JobSpec

__all__ = ["PREFLIGHT_NAME", "PreflightFailed", "describe", "run", "targets"]

PREFLIGHT_NAME = ".preflight"


def _probe_file(job_id: str) -> Path:
    path = paths.job_dir(job_id) / PREFLIGHT_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"gpuc sync preflight for job {job_id} at {jobs.utc_now()}\n")
    return path


def targets(spec: JobSpec, config: HostConfig) -> list[destinations.Destination]:
    """Every destination this job will upload to, the host's own mirror included.

    The mirror matters even for a job with no `outputs:`: the dispatcher's
    retention purge refuses to delete anything the host never managed to
    mirror, so a broken `s3_prefix` is silent until disk runs out.
    """
    found = [d for output in spec.outputs for d in destinations.of(output, spec.job_id)]
    mirror = sync.mirror_for(spec.job_id, config.s3_prefix)
    if mirror is not None:
        found.append(mirror)
    return found


def run(
    spec: JobSpec,
    config: HostConfig,
    *,
    runner: CommandRunner = run_command,
    env: Env = None,
    repo_exists: RepoExists | None = None,
) -> list[str]:
    """Every destination this job proved it can write. Raises PreflightFailed.

    A job with no outputs on a host with no mirror checks nothing at all: there
    is no upload in its future to be wrong about.
    """
    checked: list[str] = []
    probe: Path | None = None
    for destination in targets(spec, config):
        if probe is None:
            probe = _probe_file(spec.job_id)
        if repo_exists is not None and isinstance(destination, destinations.HuggingFace):
            destination.repo_exists = repo_exists
        destination.preflight(probe, runner=runner, env=env)
        checked.append(destination.uri)
    return checked


def describe(checked: list[str]) -> str:
    return (
        "sync preflight ok: wrote " + f"{PREFLIGHT_NAME} to " + ", ".join(checked)
        if checked
        else "sync preflight: nothing to check (no outputs and no s3_prefix)"
    )
