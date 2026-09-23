"""`snakemake --executor gpuc` against this machine as a `local` host.

The host is test_control_e2e's: a fake nvidia-smi, a stand-in torch in the
checkout, and every gpuc directory redirected. Snakemake runs as a separate
process, as it would for a user, and finds the plugin by its package name.
"""

from __future__ import annotations

import json
import os
import shlex
import signal
import subprocess
import sys
from pathlib import Path

import pytest

from snakemake_executor_plugin_gpuc import index_status, job_name, truthy
from tests.conftest import install_fake_torch
from tests.test_control_e2e import bootstrapped_home, state_of, wait_until

__all__ = ["bootstrapped_home"]

SNAKEFILE = """\
R = config["results"]

rule all:
    input: R + "/b.txt"

rule a:
    output: R + "/a-{n}.txt"
    shell: 'test "$SMK_TOKEN" = s3cret && echo alpha {wildcards.n} > {output}'

rule b:
    input: R + "/a-1.txt"
    output: R + "/b.txt"
    resources: priority=10
    shell: "cat {input} > {output}; echo beta >> {output}"

rule broken:
    output: R + "/never.txt"
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


def test_a_workflow_runs_as_gpuc_jobs_and_its_outputs_land_where_the_snakefile_said(
    bootstrapped_home: Path, project: Path, tmp_path: Path
) -> None:
    results = tmp_path / "results"
    results.mkdir()
    done = snakemake(project, results)
    assert done.returncode == 0, done.stderr[-4000:]
    assert (results / "b.txt").read_text() == "alpha 1\nbeta\n"

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


def test_status_is_indexed_across_hosts_and_lists() -> None:
    jobs, states = index_status(
        {
            "hosts": [
                {
                    "name": "a",
                    "state": "answered",
                    "queued": [{"job_id": "q", "status": "queued"}],
                    "running": [{"job_id": "r", "status": "running"}],
                    "finished": [{"job_id": "f", "status": "failed", "reason": "exit 3"}],
                },
                {"name": "b", "state": "unaskable", "queued": [], "running": [], "finished": []},
            ]
        }
    )
    assert jobs == {"q": ("queued", None), "r": ("running", None), "f": ("failed", "exit 3")}
    assert states == {"a": "answered", "b": "unaskable"}


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
            '\nrule slow:\n    output: R + "/slow.txt"\n    shell: "sleep 120; touch {output}"\n'
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
