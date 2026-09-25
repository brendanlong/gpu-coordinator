"""`snakemake --executor gpuc` against this machine as a `local` host.

The host is test_control_e2e's: a fake nvidia-smi, a stand-in torch in the
checkout, and every gpuc directory redirected. Snakemake runs as a separate
process, as it would for a user, and finds the plugin by its package name.
"""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import signal
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from snakemake_executor_plugin_gpuc import job_name, truthy
from tests.conftest import install_fake_torch
from tests.test_control_e2e import bootstrapped_home, state_of, wait_until

__all__ = ["bootstrapped_home"]

SNAKEFILE = """\
R = config["results"]

rule all:
    input: R + "/summary.txt"

rule a:
    output: R + "/a-{n}.txt"
    resources: gpu=1
    shell: 'test "$SMK_TOKEN" = s3cret && echo alpha {wildcards.n} > {output}'

rule b:
    input: R + "/a-1.txt"
    output: R + "/b.txt"
    resources: gpu=1, priority=10
    shell: "cat {input} > {output}; echo beta >> {output}"

rule summary:
    input: R + "/b.txt"
    output: R + "/summary.txt"
    shell: "wc -l < {input} > {output}"

rule broken:
    output: R + "/never.txt"
    resources: gpu=1
    shell: "exit 3"
"""


@pytest.fixture
def project(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    root.mkdir()
    (root / "Snakefile").write_text(SNAKEFILE)
    (root / ".gitignore").write_text(".snakemake/\n")
    install_fake_torch(root)
    for argv in (
        ["git", "init", "-q"],
        ["git", "config", "user.email", "t@example.com"],
        ["git", "config", "user.name", "t"],
        ["git", "add", "-A"],
        ["git", "commit", "-qm", "init"],
    ):
        subprocess.run(argv, cwd=root, check=True, capture_output=True)
    return root


def snakemake_argv(results: Path, *targets: str) -> list[str]:
    gpuc = shlex.join([sys.executable, "-m", "gpuc.control.cli"])
    return [
        sys.executable,
        "-m",
        "snakemake",
        "--executor",
        "gpuc",
        *targets,
        "--gpuc-host",
        "local",
        "--gpuc-setup",
        "",
        "--gpuc-python",
        sys.executable,
        "--gpuc-gpuc",
        gpuc,
        "--jobs",
        "2",
        "--seconds-between-status-checks",
        "1",
        "--envvars",
        "SMK_TOKEN",
        "--config",
        f"results={results}",
    ]


def snakemake_env() -> dict[str, str]:
    return {**os.environ, "SMK_TOKEN": "s3cret"}


def snakemake(project: Path, results: Path, *targets: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        snakemake_argv(results, *targets),
        cwd=project,
        env=snakemake_env(),
        capture_output=True,
        text=True,
        timeout=300,
    )


def test_gpu_rules_become_gpuc_jobs_and_the_rest_run_on_the_controller(
    bootstrapped_home: Path, project: Path, tmp_path: Path
) -> None:
    results = tmp_path / "results"
    results.mkdir()
    done = snakemake(project, results)
    assert done.returncode == 0, done.stderr[-4000:]
    assert (results / "b.txt").read_text() == "alpha 1\nbeta\n"
    assert (results / "summary.txt").read_text().strip() == "2"

    submitted = [
        line.split("gpuc job ")[1].split()[0]
        for line in done.stderr.splitlines()
        if "as gpuc job" in line
    ]
    assert len(submitted) == 2, done.stderr[-4000:]
    states = {job_id: state_of(bootstrapped_home, job_id) for job_id in submitted}
    assert {s["status"] for s in states.values()} == {"succeeded"}
    assert sorted(int(str(s["priority"])) for s in states.values()) == [10, 50]
    specs = [
        json.loads((bootstrapped_home / "jobs" / job_id / "spec.json").read_text())
        for job_id in submitted
    ]
    assert sorted(spec["name"] for spec in specs) == ["a[n=1]", "b"]
    for spec in specs:
        assert spec["secrets"] == ["SMK_TOKEN"]
        assert "s3cret" not in json.dumps(spec)

    again = snakemake(project, results)
    assert again.returncode == 0, again.stderr[-4000:]
    assert "as gpuc job" not in again.stderr


def test_a_failed_gpuc_job_fails_its_snakemake_job(
    bootstrapped_home: Path, project: Path, tmp_path: Path
) -> None:
    results = tmp_path / "results"
    results.mkdir()
    done = snakemake(project, results, str(results / "never.txt"))
    assert done.returncode != 0
    assert "failed: exit" in done.stderr, done.stderr[-4000:]


def test_a_directory_other_than_the_one_gpuc_would_copy_is_refused(
    project: Path, tmp_path: Path
) -> None:
    done = snakemake(project, tmp_path, "--directory", str(tmp_path))
    assert done.returncode != 0
    assert "--directory is not supported" in done.stderr, done.stderr[-4000:]


def test_a_rule_placed_on_a_host_without_a_gpu_is_refused(project: Path, tmp_path: Path) -> None:
    with (project / "Snakefile").open("a") as f:
        f.write('\nrule placed:\n    output: "x"\n    resources: host="spar"\n    shell: "true"\n')
    done = snakemake(project, tmp_path, "x")
    assert done.returncode != 0
    assert "rule placed sets `host` but no `gpu`" in done.stderr, done.stderr[-4000:]


def test_a_gpu_function_that_comes_to_zero_fails_the_job(
    bootstrapped_home: Path, project: Path, tmp_path: Path
) -> None:
    with (project / "Snakefile").open("a") as f:
        f.write(
            '\nrule maybe:\n    output: R + "/maybe.txt"\n'
            "    resources: gpu=lambda wildcards: 0\n"
            '    shell: "touch {output}"\n'
        )
    subprocess.run(["git", "commit", "-qam", "maybe"], cwd=project, check=True)
    done = snakemake(project, tmp_path, str(tmp_path / "maybe.txt"))
    assert done.returncode != 0
    assert "`gpu` came to 0 for this job" in done.stderr, done.stderr[-4000:]
    assert not (tmp_path / "maybe.txt").exists()


def test_resources_given_as_strings_still_read_as_flags() -> None:
    assert truthy("true") and truthy(1) and not truthy("0") and not truthy(False)


def test_a_job_is_named_by_its_rule_and_wildcards() -> None:
    class Job:
        name = "train"
        wildcards_dict = {"seed": "1", "k": "4"}  # noqa: RUF012

    assert job_name(Job()) == "train[seed=1,k=4]"  # type: ignore[arg-type]


def test_ctrl_c_cancels_the_jobs_in_flight(
    bootstrapped_home: Path, project: Path, tmp_path: Path
) -> None:
    results = tmp_path / "results"
    results.mkdir()
    with (project / "Snakefile").open("a") as f:
        f.write(
            '\nrule slow:\n    output: R + "/slow.txt"\n    resources: gpu=1\n'
            '    shell: "sleep 120; touch {output}"\n'
        )
    subprocess.run(["git", "commit", "-qam", "slow"], cwd=project, check=True)
    controller = subprocess.Popen(
        snakemake_argv(results, str(results / "slow.txt")),
        cwd=project,
        env=snakemake_env(),
        stderr=subprocess.PIPE,
        text=True,
    )
    assert controller.stderr is not None
    job_id = ""
    for line in controller.stderr:
        if "as gpuc job" in line:
            job_id = line.split("gpuc job ")[1].split()[0]
            break
    wait_until(lambda: state_of(bootstrapped_home, job_id).get("phase") == "main", 60, "phase main")
    controller.send_signal(signal.SIGINT)
    controller.communicate(timeout=120)
    wait_until(
        lambda: state_of(bootstrapped_home, job_id).get("status") == "cancelled",
        60,
        f"job {job_id} to be cancelled",
    )


def test_a_poll_fails_only_what_asking_again_would_not_change() -> None:
    from snakemake_interface_executor_plugins.executors.base import SubmittedJobInfo

    from snakemake_executor_plugin_gpuc import Executor

    answers = [
        {"job_id": "ok", "status": "succeeded", "host_state": "answered", "error": None},
        {"job_id": "bad", "status": "failed", "reason": "exit 3", "error": None},
        {"job_id": "run", "status": "running", "host_state": "answered", "error": None},
        {"job_id": "far", "status": None, "host_state": "unaskable", "error": "ssh: timed out"},
        {"job_id": "lost", "status": None, "host_state": "answered", "error": "has no job"},
    ]
    executor = Executor.__new__(Executor)
    executor.unaskable_shown = {}
    executor.logger = SimpleNamespace(info=lambda _msg: None)  # type: ignore[assignment]
    executor.gpuc = lambda *_a, **_k: {"jobs": answers}  # type: ignore[method-assign]
    outcomes: dict[str, str] = {}
    executor.report_job_success = lambda info: outcomes.update({info.external_jobid: "ok"})  # type: ignore[method-assign]
    executor.report_job_error = lambda info, **_k: outcomes.update({info.external_jobid: "failed"})  # type: ignore[method-assign]

    class Limiter:
        async def __aenter__(self) -> None: ...
        async def __aexit__(self, *_: object) -> None: ...

    executor.status_rate_limiter = Limiter()  # type: ignore[assignment]
    infos = [SubmittedJobInfo(None, external_jobid=a["job_id"]) for a in answers]  # type: ignore[arg-type]

    async def poll() -> list[str]:
        return [str(i.external_jobid) async for i in executor.check_active_jobs(infos)]

    assert asyncio.run(poll()) == ["run", "far"]
    assert outcomes == {"ok": "ok", "bad": "failed", "lost": "failed"}
