"""The sync preflight: prove the uploads work before the job runs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from gpuc.host import jobs, paths, preflight, queue, runner, sync
from gpuc.host.jobs import HostConfig
from gpuc.host.runner import RunnerDeps
from gpuc.host.sync import CommandResult
from tests.conftest import FAKE_GPUS, fake_smi, make_spec


class FakeRunner:
    """Records every command, and fails the ones whose text matches `fail_on`."""

    def __init__(self, fail_on: str | None = None, output: str = "denied") -> None:
        self.fail_on = fail_on
        self.output = output
        self.commands: list[list[str]] = []

    def __call__(
        self, argv: list[str], timeout: float | None = None, env: Mapping[str, str] | None = None
    ) -> CommandResult:
        self.commands.append(argv)
        joined = " ".join(argv)
        if self.fail_on and self.fail_on in joined:
            return CommandResult(argv, 1, self.output)
        return CommandResult(argv, 0, "")


@pytest.fixture
def tools(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sync, "aws_binary", lambda env=None: "/usr/bin/aws")
    monkeypatch.setattr(sync, "hf_binary", lambda env=None: "/usr/bin/hf")


def destinations(commands: Sequence[list[str]]) -> list[str]:
    return [argv[-2] for argv in commands if argv[1:3] == ["s3", "cp"]]


def test_nothing_is_checked_without_outputs_or_a_mirror(gpuc_home: Path, tools: None) -> None:
    fake = FakeRunner()
    assert preflight.run(make_spec(), HostConfig(host="h"), runner=fake, env={}) == []
    assert fake.commands == []


def test_every_s3_output_and_the_host_mirror_get_a_preflight_object(
    gpuc_home: Path, tools: None
) -> None:
    spec = make_spec(outputs=[{"path": "results", "s3": "s3://bucket/exp/{job_id}/results"}])
    config = HostConfig(host="h", s3_prefix="s3://bucket/gpuc/h")
    fake = FakeRunner()
    checked = preflight.run(spec, config, runner=fake, env={})
    assert checked == [
        f"s3://bucket/exp/{spec.job_id}/results",
        f"s3://bucket/gpuc/h/jobs/{spec.job_id}",
    ]
    assert destinations(fake.commands) == [
        f"s3://bucket/exp/{spec.job_id}/results/.preflight",
        f"s3://bucket/gpuc/h/jobs/{spec.job_id}/.preflight",
    ]


def test_a_mirror_alone_is_checked_for_a_job_with_no_outputs(gpuc_home: Path, tools: None) -> None:
    spec = make_spec()
    fake = FakeRunner()
    checked = preflight.run(
        spec, HostConfig(host="h", s3_prefix="s3://bucket/gpuc/h"), runner=fake, env={}
    )
    assert checked == [f"s3://bucket/gpuc/h/jobs/{spec.job_id}"]


def test_a_missing_aws_binary_fails_with_the_command(
    gpuc_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sync, "aws_binary", lambda env=None: None)
    with pytest.raises(preflight.PreflightFailed) as caught:
        preflight.run(
            make_spec(), HostConfig(host="h", s3_prefix="s3://b/p"), runner=FakeRunner(), env={}
        )
    assert "aws" in caught.value.command
    assert "not found" in caught.value.detail


def test_a_denied_write_names_the_destination(gpuc_home: Path, tools: None) -> None:
    fake = FakeRunner(fail_on="s3://bucket/gpuc", output="An error occurred (AccessDenied)")
    with pytest.raises(preflight.PreflightFailed) as caught:
        preflight.run(
            make_spec(), HostConfig(host="h", s3_prefix="s3://bucket/gpuc/h"), runner=fake, env={}
        )
    assert "aws s3 cp" in str(caught.value)
    assert "AccessDenied" in str(caught.value)
    assert "secrets:" in str(caught.value)


def test_hf_outputs_check_the_token_and_write_a_preflight_file(
    gpuc_home: Path, tools: None
) -> None:
    spec = make_spec(outputs=[{"path": "ckpt", "hf": "org/repo", "hf_path": "{job_id}"}])
    fake = FakeRunner()
    checked = preflight.check_hf(spec, runner=fake, env={}, repo_exists=lambda repo, env: True)
    assert checked == ["org/repo"]
    assert fake.commands[0][1:] == ["auth", "whoami"]
    assert fake.commands[-1][1:4] == [
        "upload",
        "org/repo",
        str(paths.job_dir(spec.job_id) / ".preflight"),
    ]
    assert fake.commands[-1][-1] == f"{spec.job_id}/.preflight"


def test_a_bad_hf_token_fails_before_the_job_runs(gpuc_home: Path, tools: None) -> None:
    spec = make_spec(outputs=[{"path": "ckpt", "hf": "org/repo"}])
    fake = FakeRunner(fail_on="auth whoami", output="Invalid user token")
    with pytest.raises(preflight.PreflightFailed) as caught:
        preflight.check_hf(spec, runner=fake, env={}, repo_exists=lambda repo, env: True)
    assert "auth whoami" in caught.value.command
    assert "HF_TOKEN" in caught.value.detail


def test_a_missing_repo_fails_unless_hf_create_is_set(gpuc_home: Path, tools: None) -> None:
    spec = make_spec(outputs=[{"path": "ckpt", "hf": "org/new"}])
    fake = FakeRunner()
    with pytest.raises(preflight.PreflightFailed) as caught:
        preflight.check_hf(spec, runner=fake, env={}, repo_exists=lambda repo, env: False)
    assert "hf_create: true" in caught.value.detail
    assert not any("repos" in " ".join(argv) for argv in fake.commands)


def test_hf_create_creates_the_repo(gpuc_home: Path, tools: None) -> None:
    spec = make_spec(outputs=[{"path": "ckpt", "hf": "org/new", "hf_create": True}])
    fake = FakeRunner()
    preflight.check_hf(spec, runner=fake, env={}, repo_exists=lambda repo, env: False)
    assert fake.commands[1][1:] == ["repos", "create", "org/new", "--type", "model", "--exist-ok"]


def test_a_failed_preflight_fails_the_job_with_the_command_in_the_log(
    gpuc_home: Path, tools: None
) -> None:
    jobs.write_config(HostConfig(host="test-host", s3_prefix="s3://bucket/gpuc/h"))
    spec = make_spec(command="echo SHOULD-NOT-RUN")
    job_id = queue.enqueue(spec)
    jobs.update_state(job_id, status="running", gpus=[FAKE_GPUS[0]])
    fake = FakeRunner(fail_on="s3 cp", output="An error occurred (NoSuchBucket)")

    code = runner.run_job(
        job_id,
        RunnerDeps(smi=fake_smi(), command_runner=fake, preflight=False, poll_interval_s=0.02),
    )

    state = jobs.read_state(job_id)
    log = paths.log_file(job_id).read_text()
    assert (code, state.status, state.reason) == (1, "failed", "sync-preflight")
    assert "SHOULD-NOT-RUN" not in log
    assert "aws s3 cp" in log and "NoSuchBucket" in log


def test_a_healthy_preflight_lets_the_job_run(gpuc_home: Path, tools: None) -> None:
    jobs.write_config(HostConfig(host="test-host", s3_prefix="s3://bucket/gpuc/h"))
    spec = make_spec(command="echo RAN")
    job_id = queue.enqueue(spec)
    jobs.update_state(job_id, status="running", gpus=[FAKE_GPUS[0]])

    code = runner.run_job(
        job_id,
        RunnerDeps(
            smi=fake_smi(), command_runner=FakeRunner(), preflight=False, poll_interval_s=0.02
        ),
    )

    assert (code, jobs.read_state(job_id).status) == (0, "succeeded")
    assert "RAN" in paths.log_file(job_id).read_text()
    assert "sync preflight ok" in paths.log_file(job_id).read_text()
