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
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from snakemake_executor_plugin_gpuc import GpucError, job_name, truthy
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


def submitted_ids(stderr: str) -> list[str]:
    return [
        line.split("gpuc job ")[1].split()[0]
        for line in stderr.splitlines()
        if "as gpuc job" in line
    ]


def test_gpu_rules_become_gpuc_jobs_and_the_rest_run_on_the_controller(
    bootstrapped_home: Path, project: Path, tmp_path: Path
) -> None:
    results = tmp_path / "results"
    results.mkdir()
    done = snakemake(project, results)
    assert done.returncode == 0, done.stderr[-4000:]
    assert (results / "b.txt").read_text() == "alpha 1\nbeta\n"
    assert (results / "summary.txt").read_text().strip() == "2"

    submitted = submitted_ids(done.stderr)
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
    assert "not on gpuc: summary." in done.stderr


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


def killed_mid_job(bootstrapped_home: Path, project: Path, results: Path, seconds: int) -> str:
    """Run a one-job workflow, SIGKILL its controller once the gpuc job is in
    `main`, and return that job's id."""
    with (project / "Snakefile").open("a") as f:
        f.write(
            '\nrule slow:\n    output: R + "/slow.txt"\n    resources: gpu=1\n'
            f'    shell: "sleep {seconds}; echo $GPUC_JOB_ID > {{output}}"\n'
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
    assert job_id, "the controller never submitted"
    wait_until(lambda: state_of(bootstrapped_home, job_id).get("phase") == "main", 60, "phase main")
    controller.kill()
    controller.communicate(timeout=30)
    return job_id


def restart(project: Path, results: Path) -> subprocess.CompletedProcess[str]:
    unlock = subprocess.run(
        [*snakemake_argv(results, str(results / "slow.txt")), "--unlock"],
        cwd=project,
        env=snakemake_env(),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert unlock.returncode == 0, unlock.stderr[-4000:]
    return snakemake(project, results, str(results / "slow.txt"), "--rerun-incomplete")


def test_a_restarted_controller_adopts_a_running_job(
    bootstrapped_home: Path, project: Path, tmp_path: Path
) -> None:
    results = tmp_path / "results"
    results.mkdir()
    job_id = killed_mid_job(bootstrapped_home, project, results, 20)
    assert state_of(bootstrapped_home, job_id).get("status") == "running"

    again = restart(project, results)
    assert again.returncode == 0, again.stderr[-4000:]
    assert submitted_ids(again.stderr) == [], again.stderr[-4000:]
    assert f"Adopted gpuc job {job_id}" in again.stderr, again.stderr[-4000:]
    assert (results / "slow.txt").read_text().strip() == job_id


def test_a_restarted_controller_keeps_what_a_job_finished_while_it_was_down(
    bootstrapped_home: Path, project: Path, tmp_path: Path
) -> None:
    results = tmp_path / "results"
    results.mkdir()
    job_id = killed_mid_job(bootstrapped_home, project, results, 2)
    wait_until(
        lambda: state_of(bootstrapped_home, job_id).get("status") == "succeeded",
        60,
        f"job {job_id} to succeed",
    )

    again = restart(project, results)
    assert again.returncode == 0, again.stderr[-4000:]
    assert submitted_ids(again.stderr) == [], again.stderr[-4000:]
    assert f"gpuc job {job_id}, which already succeeded" in again.stderr, again.stderr[-4000:]
    assert (results / "slow.txt").read_text().strip() == job_id

    settled = snakemake(project, results, str(results / "slow.txt"))
    assert settled.returncode == 0, settled.stderr[-4000:]
    assert "Nothing to be done" in settled.stderr, settled.stderr[-4000:]


def test_a_restarted_controller_replaces_a_job_whose_rule_changed(
    bootstrapped_home: Path, project: Path, tmp_path: Path
) -> None:
    results = tmp_path / "results"
    results.mkdir()
    stale = killed_mid_job(bootstrapped_home, project, results, 60)
    snakefile = project / "Snakefile"
    snakefile.write_text(snakefile.read_text().replace("sleep 60", "sleep 1"))
    subprocess.run(["git", "commit", "-qam", "faster"], cwd=project, check=True)

    again = restart(project, results)
    assert again.returncode == 0, again.stderr[-4000:]
    assert f"Not adopting gpuc job {stale}" in again.stderr, again.stderr[-4000:]
    fresh = submitted_ids(again.stderr)
    assert len(fresh) == 1 and fresh != [stale], again.stderr[-4000:]
    assert (results / "slow.txt").read_text().strip() == fresh[0]
    wait_until(
        lambda: state_of(bootstrapped_home, stale).get("status") == "cancelled",
        60,
        f"job {stale} to be cancelled",
    )


def test_an_earlier_job_is_adopted_unless_it_ended_badly() -> None:
    from snakemake_interface_executor_plugins.executors.base import SubmittedJobInfo

    from snakemake_executor_plugin_gpuc import Executor

    answers = {
        "old-failed": {"status": "failed", "error": None},
        "old-ok": {"status": "succeeded", "error": None},
        "old-run": {"status": "running", "host": "spar", "error": None},
        "old-far": {"status": None, "host_state": "unaskable", "error": "ssh: timed out"},
        "old-lost": {"status": None, "host_state": "answered", "error": "has no job"},
    }
    markers = {
        "done": ["old-failed", "old-ok"],
        "running": ["old-failed", "old-run"],
        "far": ["old-far"],
        "redo": ["old-failed", "old-lost"],
        "fresh": [],
        "stale": ["old-run", "old-far", "old-failed"],
    }
    executor = Executor.__new__(Executor)
    executor.earlier = answers  # type: ignore[assignment]
    executor.earlier_failure = None
    executor.unlike = lambda job, _id: "changed" if job.jobid == "stale" else None  # type: ignore[method-assign]
    cancelled: list[str] = []
    executor.cancel_ids = cancelled.extend  # type: ignore[method-assign]
    executor.in_flight = {}
    executor.in_flight_lock = threading.Lock()
    executor.logger = SimpleNamespace(info=lambda _msg: None)  # type: ignore[assignment]
    executor.workflow = SimpleNamespace(  # type: ignore[assignment]
        persistence=SimpleNamespace(external_jobids=lambda job: markers[job.jobid])
    )
    outcomes: dict[str, str] = {}
    executor.report_job_success = lambda info: outcomes.update({info.job.jobid: "succeeded"})  # type: ignore[method-assign]
    executor.report_job_submission = lambda info: outcomes.update({info.job.jobid: "polled"})  # type: ignore[method-assign]

    for name in markers:
        job = SimpleNamespace(jobid=name)
        info = SubmittedJobInfo(job, aux={})  # type: ignore[arg-type]
        if not executor.adopt(job, info):  # type: ignore[arg-type]
            outcomes[name] = "submit"
        elif name != "fresh":
            outcomes[name] += f" {info.external_jobid}"

    assert outcomes == {
        "done": "succeeded old-ok",
        "running": "polled old-run",
        "far": "polled old-far",
        "redo": "submit",
        "fresh": "submit",
        "stale": "submit",
    }
    assert sorted(executor.in_flight) == ["old-far", "old-run"]
    assert sorted(cancelled) == ["old-far", "old-run"]


def test_a_job_with_an_earlier_one_fails_when_gpuc_cannot_be_asked() -> None:
    from snakemake_interface_executor_plugins.executors.base import SubmittedJobInfo

    from snakemake_executor_plugin_gpuc import Executor

    executor = Executor.__new__(Executor)
    executor.gpuc = lambda *_a, **_k: (_ for _ in ()).throw(GpucError("exit 3"))  # type: ignore[method-assign]
    executor.workflow = SimpleNamespace(  # type: ignore[assignment]
        persistence=SimpleNamespace(external_jobids=lambda job: ["old"] if job.jobid else [])
    )
    marked, unmarked = SimpleNamespace(jobid=1), SimpleNamespace(jobid=0)
    for job in (marked, unmarked):
        job.is_group = lambda: False
    executor.earlier, executor.earlier_failure = executor.earlier_jobs([marked, unmarked])  # type: ignore[list-item]
    errors: list[str] = []
    executor.report_job_error = lambda _info, msg=None, **_k: errors.append(str(msg))  # type: ignore[method-assign]

    assert executor.adopt(marked, SubmittedJobInfo(marked, aux={}))  # type: ignore[arg-type]
    assert not executor.adopt(unmarked, SubmittedJobInfo(unmarked, aux={}))  # type: ignore[arg-type]
    assert len(errors) == 1 and "exit 3" in errors[0]
