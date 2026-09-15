"""Prove the job's uploads can work *before* it spends an hour computing them.

Every failure this catches used to surface at the final sync: a job that ran
for hours ends `failed: sync` holding the only copy of its outputs, because a
secret was missing, the bucket was a typo, or the Hugging Face repo did not
exist. Here the same failure costs seconds and the job never starts.

Stdlib only, like everything under `gpuc.host`; the uploads themselves are the
same `aws` and `hf` binaries the sync loop uses.
"""

from __future__ import annotations

import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from pathlib import Path

from gpuc.host import jobs, paths, sync
from gpuc.host.jobs import HostConfig, JobSpec

HF_API = "https://huggingface.co/api"
PREFLIGHT_NAME = ".preflight"
COMMAND_TIMEOUT_S = 120.0

Env = Mapping[str, str] | None
RepoExists = Callable[[str, Env], "bool | None"]


class PreflightFailed(RuntimeError):
    """Carries the command that failed and what it said, for the job log."""

    def __init__(self, command: str, detail: str) -> None:
        super().__init__(f"{command}\n{detail.strip()}")
        self.command = command
        self.detail = detail


def _probe_body(job_id: str) -> str:
    return f"gpuc sync preflight for job {job_id} at {jobs.utc_now()}\n"


def _probe_file(job_id: str) -> Path:
    path = paths.job_dir(job_id) / PREFLIGHT_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_probe_body(job_id))
    return path


def s3_destinations(spec: JobSpec, config: HostConfig) -> list[str]:
    """Every prefix this job will upload to, the host's own mirror included.

    The mirror matters even for a job with no `outputs:`: the dispatcher's
    retention purge refuses to delete anything the host never managed to mirror,
    so a broken `s3_prefix` is silent until disk runs out.
    """
    dests = [
        output.s3.format(job_id=spec.job_id).rstrip("/") for output in spec.outputs if output.s3
    ]
    if config.s3_prefix:
        dests.append(f"{config.s3_prefix.rstrip('/')}/jobs/{spec.job_id}")
    return dests


def check_s3(
    spec: JobSpec,
    config: HostConfig,
    *,
    runner: sync.CommandRunner = sync.run_command,
    env: Env = None,
) -> list[str]:
    dests = s3_destinations(spec, config)
    if not dests:
        return []
    aws = sync.aws_binary(env)
    if aws is None:
        raise PreflightFailed(
            "aws s3 cp",
            "`aws` CLI not found (looked in ~/.local/aws-cli/v2/current/bin/aws and PATH). "
            "Re-run `gpuc host bootstrap` for this host, or drop the job's s3 outputs.",
        )
    probe = _probe_file(spec.job_id)
    for dest in dests:
        argv = [aws, "s3", "cp", str(probe), f"{dest}/{PREFLIGHT_NAME}", "--only-show-errors"]
        result = runner(argv, COMMAND_TIMEOUT_S, env)
        if result.returncode != 0:
            raise PreflightFailed(
                " ".join(argv),
                f"exited {result.returncode}\n{result.output.strip()[-800:]}\n"
                f"The job's environment must be able to write {dest}: check the spec's "
                f"`secrets:` (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY) and the bucket name.",
            )
    return dests


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


def check_hf(
    spec: JobSpec,
    *,
    runner: sync.CommandRunner = sync.run_command,
    env: Env = None,
    repo_exists: RepoExists = hf_repo_exists,
) -> list[str]:
    outputs = [output for output in spec.outputs if output.hf]
    if not outputs:
        return []
    hf = sync.hf_binary(env)
    if hf is None:
        raise PreflightFailed(
            "hf auth whoami",
            "`hf` CLI not found (looked in ~/.local/bin/hf and PATH). Re-run "
            "`gpuc host bootstrap` for this host, or drop the job's hf outputs.",
        )
    whoami = runner([hf, "auth", "whoami"], COMMAND_TIMEOUT_S, env)
    if whoami.returncode != 0:
        raise PreflightFailed(
            f"{hf} auth whoami",
            f"exited {whoami.returncode}\n{whoami.output.strip()[-500:]}\n"
            f"The job's environment has no usable Hugging Face token: add HF_TOKEN to the "
            f"spec's `secrets:`.",
        )
    probe = _probe_file(spec.job_id)
    checked: list[str] = []
    for output in outputs:
        assert output.hf is not None
        repo = output.hf.format(job_id=spec.job_id)
        path_in_repo = (output.hf_path or spec.job_id).format(job_id=spec.job_id).strip("/")
        if output.hf_create:
            create = runner(
                [hf, "repos", "create", repo, "--type", "model", "--exist-ok"],
                COMMAND_TIMEOUT_S,
                env,
            )
            if create.returncode != 0:
                raise PreflightFailed(
                    f"{hf} repos create {repo} --type model --exist-ok",
                    f"exited {create.returncode}\n{create.output.strip()[-500:]}",
                )
        elif repo_exists(repo, env) is False:
            raise PreflightFailed(
                f"{hf} upload {repo}",
                f"the repo {repo} does not exist, or this token cannot see it. Create it on "
                f"the Hub, or set `hf_create: true` on that output to have gpuc create it.",
            )
        target = f"{path_in_repo}/{PREFLIGHT_NAME}" if path_in_repo else PREFLIGHT_NAME
        argv = [hf, "upload", repo, str(probe), target]
        result = runner(argv, COMMAND_TIMEOUT_S, env)
        if result.returncode != 0:
            raise PreflightFailed(
                " ".join(argv),
                f"exited {result.returncode}\n{result.output.strip()[-800:]}\n"
                f"The job's token must be able to write {repo}.",
            )
        checked.append(repo)
    return checked


def run(
    spec: JobSpec,
    config: HostConfig,
    *,
    runner: sync.CommandRunner = sync.run_command,
    env: Env = None,
) -> list[str]:
    """Every destination this job proved it can write. Raises PreflightFailed.

    A job with no outputs on a host with no mirror checks nothing at all: there
    is no upload in its future to be wrong about.
    """
    destinations = check_s3(spec, config, runner=runner, env=env)
    destinations += check_hf(spec, runner=runner, env=env)
    return destinations


def describe(destinations: list[str]) -> str:
    return (
        "sync preflight ok: wrote " + f"{PREFLIGHT_NAME} to " + ", ".join(destinations)
        if destinations
        else "sync preflight: nothing to check (no outputs and no s3_prefix)"
    )
