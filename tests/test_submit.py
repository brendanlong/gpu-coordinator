from __future__ import annotations

import json
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from gpuc.control.config import Settings
from gpuc.control.remote import HostSession
from gpuc.control.s3index import LocalIndex, S3Index, spec_key
from gpuc.control.submit import (
    SubmitError,
    check_gpu_count,
    gather_secrets,
    load_document,
    prepare,
    submit_file,
    submit_spec,
    validate,
)
from gpuc.control.transport import CommandResult
from gpuc.host import jobs
from tests.conftest import host_entry
from tests.fakes3 import FakeS3Client

REMOTE_HOME = "/home/u/.gpuc"


@dataclass
class FakeHost:
    host: str = "gpubox"
    commands: list[str] = field(default_factory=list)
    puts: dict[str, tuple[str, int]] = field(default_factory=dict)
    rsyncs: list[tuple[Path, str, list[str] | None]] = field(default_factory=list)
    excludes: list[str] = field(default_factory=list)

    def run(self, command: str, *, timeout: float = 120.0, check: bool = True) -> CommandResult:
        self.commands.append(command)
        out = ""
        if "gpuc.host enqueue" in command:
            out = json.dumps({"job_id": "unused", "dispatcher_pid": 99})
        return CommandResult(self.host, ["sh", "-c", command], 0, out, "")

    def put_file(self, content: str | bytes, remote_path: str, mode: int = 0o600) -> None:
        text = content.decode() if isinstance(content, bytes) else content
        self.puts[remote_path] = (text, mode)

    def rsync(
        self,
        local_root: Path,
        remote_path: str,
        files: Sequence[str] | None = None,
        excludes: Sequence[str] = (),
    ) -> CommandResult:
        self.rsyncs.append((local_root, remote_path, list(files) if files else None))
        self.excludes = list(excludes)
        return CommandResult(self.host, ["rsync"], 0, "", "")

    def tail(self, remote_path: str, lines: int = 200, follow: bool = False) -> CommandResult:
        return CommandResult(self.host, ["tail"], 0, "", "")

    def argv(self, command: str) -> list[str]:
        return ["ssh", self.host, command]

    def interactive_argv(self, command: str) -> list[str]:
        return ["ssh", "-t", self.host, command]


def session(host: FakeHost) -> HostSession:
    entry = host_entry(
        name="gpubox", kind="ssh", ssh="me@box", gpus=["GPU-a"], python="/usr/bin/py"
    )
    return HostSession(entry, host, REMOTE_HOME, "/usr/bin/py")


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    (root / "src").mkdir(parents=True)
    (root / "src" / "train.py").write_text("print('hi')\n")
    (root / "new.py").write_text("just written, not added\n")
    (root / ".gitignore").write_text(".venv/\nbig.bin\n")
    (root / "big.bin").write_text("ignored junk\n")
    (root / ".venv").mkdir()
    (root / ".venv" / "huge").write_text("x" * 100)
    for argv in (
        ["git", "init", "-q"],
        ["git", "config", "user.email", "t@example.com"],
        ["git", "config", "user.name", "t"],
        ["git", "add", "src/train.py"],
        ["git", "commit", "-qm", "init"],
    ):
        subprocess.run(argv, cwd=root, check=True, capture_output=True)
    return root


def job_document(**overrides: Any) -> dict[str, Any]:
    document: dict[str, Any] = {"name": "t", "command": "python train.py", "gpus": 1}
    document.update(overrides)
    return document


def test_a_yaml_boolean_command_is_caught_with_a_tip() -> None:
    with pytest.raises(SubmitError) as exc:
        validate({"command": True}, "job.yaml")
    assert "command: Input should be a valid string" in str(exc.value)
    assert "quote values YAML reads as booleans" in str(exc.value)


def test_unknown_fields_are_rejected_by_name() -> None:
    with pytest.raises(SubmitError) as exc:
        validate(job_document(gpu=1), "job.yaml")
    assert "job.yaml is not a valid job" in str(exc.value)
    assert "gpu" in str(exc.value)


@pytest.mark.parametrize(
    ("overrides", "needle"),
    [
        ({"command": "  "}, "command"),
        ({"gpus": -1}, "gpus"),
        ({"gpus": 0}, "gpus"),
        ({"priority": 200}, "priority"),
        ({"outputs": [{"path": "r", "bucket": "x"}]}, "outputs.0.bucket"),
    ],
)
def test_bad_values_point_at_the_field(overrides: dict[str, Any], needle: str) -> None:
    with pytest.raises(SubmitError) as exc:
        validate(job_document(**overrides))
    assert needle in str(exc.value)


def test_a_job_must_ask_for_at_least_one_gpu() -> None:
    with pytest.raises(SubmitError) as exc:
        validate(job_document(gpus=0), "job.yaml")
    assert "gpus: Input should be greater than or equal to 1" in str(exc.value)


def test_yaml_and_json_both_load(tmp_path: Path) -> None:
    yaml_file = tmp_path / "job.yaml"
    yaml_file.write_text("name: t\ncommand: echo hi\ngpus: 2\n")
    json_file = tmp_path / "job.json"
    json_file.write_text('{"name": "t", "command": "echo hi", "gpus": 2}')
    assert load_document(yaml_file) == load_document(json_file)


def test_a_missing_job_file_says_so(tmp_path: Path) -> None:
    with pytest.raises(SubmitError, match="no such job file"):
        load_document(tmp_path / "nope.yaml")


def test_missing_secrets_fail_before_the_host_is_touched() -> None:
    with pytest.raises(SubmitError) as exc:
        gather_secrets(["HF_TOKEN", "WANDB_API_KEY"], {"HF_TOKEN": "x"})
    assert "WANDB_API_KEY" in str(exc.value)
    assert "export WANDB_API_KEY" in str(exc.value)


def test_secrets_render_as_an_env_file_the_host_can_parse(tmp_path: Path) -> None:
    body = gather_secrets(["A", "B"], {"A": "1", "B": "two words"})
    path = tmp_path / "s.env"
    path.write_text(body)
    assert jobs.parse_env_file(path) == {"A": "1", "B": "two words"}


def test_submit_expands_job_id_in_output_destinations(control_env: Path, repo: Path) -> None:
    host = FakeHost()
    result = submit_spec(
        host_entry(name="gpubox", gpus=["GPU-a"]),
        validate(
            job_document(
                outputs=[
                    {"path": "results", "s3": "s3://b/exp/{job_id}/results"},
                    {"path": "ckpt", "hf": "org/repo", "hf_path": "{job_id}"},
                ]
            )
        ),
        Settings(),
        workdir=repo,
        session=session(host),
        environ={},
        report=lambda _: None,
    )
    spec = json.loads(host.puts[f"{REMOTE_HOME}/incoming/{result.job_id}/spec.json"][0])
    assert spec["outputs"][0]["s3"] == f"s3://b/exp/{result.job_id}/results"
    assert spec["outputs"][1]["hf_path"] == result.job_id
    assert "{job_id}" not in json.dumps(spec["outputs"])


def test_the_mirror_keeps_the_job_id_unexpanded_so_a_requeue_gets_its_own_namespace(
    control_env: Path, repo: Path
) -> None:
    """The mirror is what `gpuc requeue` submits: the expanded spec there
    would point every re-run at the outputs of the run it came from."""
    client = FakeS3Client()
    host = FakeHost()
    result = submit_spec(
        host_entry(name="gpubox", gpus=["GPU-a"]),
        validate(job_document(outputs=[{"path": "results", "s3": "s3://b/exp/{job_id}/results"}])),
        Settings(s3_bucket="bkt"),
        workdir=repo,
        session=session(host),
        environ={},
        s3=S3Index("bkt", client),
        report=lambda _: None,
    )
    mirrored = json.loads(client.objects[f"bkt/{spec_key(result.job_id)}"])
    assert mirrored["outputs"][0]["s3"] == "s3://b/exp/{job_id}/results"
    assert (mirrored["job_id"], mirrored["attempt"]) == (result.job_id, 1)
    shipped = json.loads(host.puts[f"{REMOTE_HOME}/incoming/{result.job_id}/spec.json"][0])
    assert shipped["outputs"][0]["s3"] == f"s3://b/exp/{result.job_id}/results"


@pytest.mark.parametrize(
    ("output", "key"),
    [
        ({"path": "results", "s3": "s3://b/exp/results"}, "s3"),
        ({"path": "ckpt", "hf": "org/repo", "hf_path": "runs/latest"}, "hf_path"),
    ],
)
def test_submit_refuses_an_output_destination_without_the_job_id(
    control_env: Path, repo: Path, output: dict[str, Any], key: str
) -> None:
    """Every output location includes the job id, so runs never overwrite each
    other; for HF the location is the repo plus the path, and neither has it."""
    host = FakeHost()
    with pytest.raises(SubmitError) as exc:
        submit_spec(
            host_entry(name="gpubox", gpus=["GPU-a"]),
            validate(job_document(outputs=[output])),
            Settings(),
            workdir=repo,
            session=session(host),
            environ={},
            report=lambda _: None,
        )
    message = str(exc.value)
    assert f"output {output['path']}: `{key}: {output[key]}` does not include the job id" in message
    assert "{job_id}" in message
    assert not host.puts and not host.rsyncs


def test_an_hf_repo_per_run_with_a_fixed_path_is_a_unique_location(
    control_env: Path, repo: Path
) -> None:
    host = FakeHost()
    result = submit_spec(
        host_entry(name="gpubox", gpus=["GPU-a"]),
        validate(
            job_document(outputs=[{"path": "ckpt", "hf": "org/run-{job_id}", "hf_path": "weights"}])
        ),
        Settings(),
        workdir=repo,
        session=session(host),
        environ={},
        report=lambda _: None,
    )
    shipped = json.loads(host.puts[f"{REMOTE_HOME}/incoming/{result.job_id}/spec.json"][0])
    assert shipped["outputs"][0]["hf"] == f"org/run-{result.job_id}"
    assert shipped["outputs"][0]["hf_path"] == "weights"


@pytest.mark.parametrize("bad", ["s3://b/{job-id}/x", "s3://b/{jobid}/x", "s3://b/{}/x"])
def test_a_placeholder_this_does_not_know_is_refused_not_a_traceback(
    control_env: Path, repo: Path, bad: str
) -> None:
    host = FakeHost()
    with pytest.raises(SubmitError) as exc:
        submit_spec(
            host_entry(name="gpubox", gpus=["GPU-a"]),
            validate(job_document(outputs=[{"path": "results", "s3": bad}])),
            Settings(),
            workdir=repo,
            session=session(host),
            environ={},
            report=lambda _: None,
        )
    assert "placeholder this does not know" in str(exc.value)
    assert not host.puts


def test_a_destination_carrying_this_jobs_literal_id_is_accepted(
    control_env: Path, repo: Path
) -> None:
    """The expanded string is what is judged, so the id itself is as good as
    the placeholder -- and an *earlier* run's id is not (see the requeue tests)."""
    host = FakeHost()
    job_id = "20260917-000000-abcdef"
    result = submit_spec(
        host_entry(name="gpubox", gpus=["GPU-a"]),
        validate(job_document(outputs=[{"path": "results", "s3": f"s3://b/{job_id}/results"}])),
        Settings(),
        workdir=repo,
        session=session(host),
        environ={},
        job_id=job_id,
        report=lambda _: None,
    )
    assert result.job_id == job_id
    assert f"{REMOTE_HOME}/incoming/{job_id}/spec.json" in host.puts


def test_an_hf_output_without_hf_path_uploads_under_the_job_id_itself(
    control_env: Path, repo: Path
) -> None:
    host = FakeHost()
    result = submit_spec(
        host_entry(name="gpubox", gpus=["GPU-a"]),
        validate(job_document(outputs=[{"path": "ckpt", "hf": "org/repo"}])),
        Settings(),
        workdir=repo,
        session=session(host),
        environ={},
        report=lambda _: None,
    )
    spec = json.loads(host.puts[f"{REMOTE_HOME}/incoming/{result.job_id}/spec.json"][0])
    assert spec["outputs"][0]["hf_path"] is None


def test_prepare_refuses_an_output_without_the_job_id_before_a_pod_is_bought(
    repo: Path,
) -> None:
    with pytest.raises(SubmitError, match="does not include the job id"):
        prepare(
            validate(job_document(outputs=[{"path": "results", "s3": "s3://b/results"}])),
            repo,
            environ={},
        )


def test_submit_ships_tracked_and_untracked_files_but_not_ignored_ones(
    control_env: Path, repo: Path
) -> None:
    (repo / "src" / "train.py").write_text("print('changed')\n")
    host = FakeHost()
    lines: list[str] = []
    result = submit_spec(
        host_entry(name="gpubox", gpus=["GPU-a"]),
        validate(job_document()),
        Settings(),
        workdir=repo,
        session=session(host),
        environ={},
        report=lines.append,
    )
    root, dest, files = host.rsyncs[0]
    assert root == repo
    assert dest == f"{REMOTE_HOME}/incoming/{result.job_id}/workdir"
    assert sorted(files or []) == [".gitignore", "new.py", "src/train.py"]
    assert any(
        line == "syncing 3 files (1 modified, 2 untracked, ignoring .gitignore'd)" for line in lines
    )
    patch, _ = host.puts[f"{REMOTE_HOME}/incoming/{result.job_id}/uncommitted.patch"]
    assert "print('changed')" in patch
    assert "just written, not added" in patch
    source = json.loads(host.puts[f"{REMOTE_HOME}/incoming/{result.job_id}/source.json"][0])
    assert len(source["commit"]) == 40


def test_a_file_deleted_but_still_in_the_index_is_not_sent(control_env: Path, repo: Path) -> None:
    (repo / "src" / "train.py").unlink()
    host = FakeHost()
    submit_spec(
        host_entry(name="gpubox", gpus=["GPU-a"]),
        validate(job_document()),
        Settings(),
        workdir=repo,
        session=session(host),
        environ={},
        report=lambda _: None,
    )
    _, _, files = host.rsyncs[0]
    assert "src/train.py" not in (files or [])


def test_no_git_syncs_everything_except_the_default_excludes(
    control_env: Path, tmp_path: Path
) -> None:
    plain = tmp_path / "not-a-repo"
    (plain / "data").mkdir(parents=True)
    (plain / "data" / "a.txt").write_text("hello\n")
    host = FakeHost()
    lines: list[str] = []
    result = submit_spec(
        host_entry(name="gpubox", gpus=["GPU-a"]),
        validate(job_document()),
        Settings(),
        workdir=plain,
        session=session(host),
        environ={},
        use_git=False,
        report=lines.append,
    )
    root, dest, files = host.rsyncs[0]
    assert (root, dest) == (plain, f"{REMOTE_HOME}/incoming/{result.job_id}/workdir")
    assert files is None
    assert host.excludes == [".venv", "__pycache__", ".git", "*.pyc", "node_modules", ".uv-cache"]
    assert any("WARNING: --no-git" in line for line in lines)
    assert f"{REMOTE_HOME}/incoming/{result.job_id}/uncommitted.patch" not in host.puts


def test_a_non_repo_without_no_git_says_how_to_fix_it(control_env: Path, tmp_path: Path) -> None:
    plain = tmp_path / "plain"
    plain.mkdir()
    with pytest.raises(SubmitError) as caught:
        submit_spec(
            host_entry(name="gpubox", gpus=["GPU-a"]),
            validate(job_document()),
            Settings(),
            workdir=plain,
            session=session(FakeHost()),
            environ={},
            report=lambda _: None,
        )
    assert "--no-git" in str(caught.value)


def test_submit_delivers_secrets_0600_and_never_on_argv(control_env: Path, repo: Path) -> None:
    host = FakeHost()
    result = submit_spec(
        host_entry(name="gpubox", gpus=["GPU-a"]),
        validate(job_document(secrets=["HF_TOKEN"])),
        Settings(),
        workdir=repo,
        session=session(host),
        environ={"HF_TOKEN": "hf_secret_value"},
        report=lambda _: None,
    )
    body, mode = host.puts[f"{REMOTE_HOME}/secrets/{result.job_id}.env"]
    assert body == "HF_TOKEN=hf_secret_value\n"
    assert mode == 0o600
    assert not any("hf_secret_value" in command for command in host.commands)


def test_submit_enqueues_over_stdin_and_records_the_index(control_env: Path, repo: Path) -> None:
    host = FakeHost()
    result = submit_spec(
        host_entry(name="gpubox", gpus=["GPU-a"]),
        validate(job_document(name="lego")),
        Settings(),
        workdir=repo,
        session=session(host),
        environ={},
        report=lambda _: None,
    )
    enqueue = next(c for c in host.commands if "gpuc.host enqueue" in c)
    assert f"enqueue {REMOTE_HOME}/incoming/{result.job_id}/spec.json" in enqueue
    assert f'GPUC_HOME="{REMOTE_HOME}"' in enqueue
    assert f'PYTHONPATH="{REMOTE_HOME}/pkg"' in enqueue
    index = LocalIndex().get(result.job_id)
    assert index is not None
    assert (index.host, index.name, index.attempt) == ("gpubox", "lego", 1)
    assert "s3_bucket is unset" in " ".join(result.notes)


def test_submit_mirrors_the_spec_to_s3_when_a_bucket_is_configured(
    control_env: Path, repo: Path
) -> None:
    client = FakeS3Client()
    host = FakeHost()
    result = submit_spec(
        host_entry(name="gpubox", gpus=["GPU-a"]),
        validate(job_document()),
        Settings(s3_bucket="bkt"),
        workdir=repo,
        session=session(host),
        environ={},
        s3=S3Index("bkt", client),
        report=lambda _: None,
    )
    spec = json.loads(client.objects[f"bkt/{spec_key(result.job_id)}"])
    assert spec["command"] == "python train.py"
    assert result.notes == []
    index = LocalIndex().get(result.job_id)
    assert index is not None and index.spec_uri == f"s3://bkt/{spec_key(result.job_id)}"


def test_a_spec_already_mirrored_is_not_put_again(control_env: Path, repo: Path) -> None:
    """`submit --runpod` mirrors the spec before it buys a pod and passes the
    uri back in; the second PUT was the same object over the wire twice."""
    client = FakeS3Client()
    result = submit_spec(
        host_entry(name="gpubox", gpus=["GPU-a"]),
        validate(job_document()),
        Settings(s3_bucket="bkt"),
        workdir=repo,
        session=session(FakeHost()),
        environ={},
        s3=S3Index("bkt", client),
        spec_uri="s3://bkt/mirrored-earlier.json",
        report=lambda _: None,
    )
    assert f"bkt/{spec_key(result.job_id)}" not in client.objects
    index = LocalIndex().get(result.job_id)
    assert index is not None and index.spec_uri == "s3://bkt/mirrored-earlier.json"


def test_a_job_bigger_than_the_host_is_refused_early(control_env: Path, repo: Path) -> None:
    host = FakeHost()
    with pytest.raises(SubmitError) as exc:
        submit_spec(
            host_entry(name="gpubox", gpus=["GPU-a"]),
            validate(job_document(gpus=4)),
            Settings(),
            workdir=repo,
            session=session(host),
            environ={},
            report=lambda _: None,
        )
    assert str(exc.value).startswith(
        "host gpubox cannot run this job: it needs 4 GPUs, host owns 1"
    )
    assert "Submit to a bigger host" in str(exc.value)
    assert host.rsyncs == []


def submit_to(
    entry: Any, workdir: Path, document: dict[str, Any], host: FakeHost | None = None
) -> Any:
    return submit_spec(
        entry,
        validate(document),
        Settings(),
        workdir=workdir,
        session=session(host or FakeHost()),
        environ={},
        report=lambda _: None,
    )


def test_a_job_that_needs_shared_cards_to_fit_is_accepted(control_env: Path, repo: Path) -> None:
    """Waiting for a card somebody else has is a real plan, and refusing it
    here would make the four-GPUs-on-a-two-GPU-host case impossible."""
    entry = host_entry(name="gpubox", gpus=["GPU-a"], shared_gpus=["GPU-b"])
    assert submit_to(entry, repo, job_document(gpus=2, use_shared=True)).job_id


def test_a_job_that_did_not_ask_is_not_given_the_shared_cards_as_capacity(
    control_env: Path, repo: Path
) -> None:
    """`needs 2 GPUs, host owns 1` would send somebody looking for a bigger
    host when one word in the spec was the answer."""
    entry = host_entry(name="gpubox", gpus=["GPU-a"], shared_gpus=["GPU-b"])
    with pytest.raises(SubmitError) as exc:
        submit_to(entry, repo, job_document(gpus=2))
    assert "host owns 1 and shares 1 this job did not ask for" in str(exc.value)
    assert "use_shared: true" in str(exc.value)


def test_a_job_bigger_than_owned_and_shared_together_is_still_refused(
    control_env: Path, repo: Path
) -> None:
    entry = host_entry(name="gpubox", gpus=["GPU-a"], shared_gpus=["GPU-b"])
    with pytest.raises(SubmitError) as exc:
        submit_to(entry, repo, job_document(gpus=4, use_shared=True))
    assert "needs 4 GPUs, host owns 1 and may borrow 1 shared" in str(exc.value)
    assert "Submit to a bigger host" in str(exc.value)


def test_use_shared_reaches_the_host_in_the_spec(control_env: Path, repo: Path) -> None:
    host = FakeHost()
    entry = host_entry(name="gpubox", gpus=["GPU-a"], shared_gpus=["GPU-b"])
    result = submit_to(entry, repo, job_document(gpus=1, use_shared=True), host)
    staged, _mode = host.puts[f"{REMOTE_HOME}/incoming/{result.job_id}/spec.json"]
    assert json.loads(staged)["use_shared"] is True


def test_submitting_from_a_non_repository_says_what_to_do(
    control_env: Path, tmp_path: Path
) -> None:
    with pytest.raises(SubmitError, match="git init"):
        submit_spec(
            host_entry(name="gpubox", gpus=["GPU-a"]),
            validate(job_document(gpus=1)),
            Settings(),
            workdir=tmp_path,
            session=session(FakeHost()),
            environ={},
            report=lambda _: None,
        )


def test_submit_file_reads_yaml(control_env: Path, repo: Path) -> None:
    (repo / "job.yaml").write_text("name: t\ncommand: echo hi\ngpus: 1\n")
    result = submit_file(
        host_entry(name="gpubox", gpus=["GPU-a"]),
        repo / "job.yaml",
        Settings(),
        workdir=repo,
        session=session(FakeHost()),
        environ={},
        report=lambda _: None,
    )
    assert result.host == "gpubox"
    assert result.attempt == 1


def test_the_gpu_count_of_a_pod_to_be_is_checked_before_it_is_bought() -> None:
    model = validate(job_document(gpus=2))
    with pytest.raises(SubmitError) as caught:
        check_gpu_count(model, 1)
    assert "--gpu-count" in str(caught.value)


def test_a_spec_that_fits_the_pod_or_names_no_count_passes_the_gpu_count_check() -> None:
    check_gpu_count(validate(job_document(gpus=2)), 2)
    check_gpu_count(validate(job_document(gpus=8)), None)


def test_a_long_job_is_fine_on_a_pod(repo: Path) -> None:
    prepared = prepare(validate(job_document(max_runtime_min=6000)), repo, environ={})
    assert prepared.warnings == []


def test_prepare_expands_the_job_id_and_gathers_the_secrets_in_one_call(repo: Path) -> None:
    prepared = prepare(
        validate(
            job_document(
                secrets=["HF_TOKEN"],
                outputs=[{"path": "results", "s3": "s3://b/{job_id}"}],
            )
        ),
        repo,
        job_id="j-fixed",
        attempt=3,
        environ={"HF_TOKEN": "hf_secret"},
    )
    assert prepared.spec.job_id == "j-fixed"
    assert prepared.spec.attempt == 3
    assert prepared.spec.outputs[0].s3 == "s3://b/j-fixed"
    assert prepared.secrets_body == "HF_TOKEN=hf_secret\n"


def test_prepare_refuses_a_missing_secret_before_a_host_is_involved(repo: Path) -> None:
    with pytest.raises(SubmitError, match="HF_TOKEN"):
        prepare(validate(job_document(secrets=["HF_TOKEN"])), repo, environ={})


def test_prepare_refuses_a_workdir_that_is_not_a_repo_unless_git_is_off(tmp_path: Path) -> None:
    model = validate(job_document())
    with pytest.raises(SubmitError, match="not a git repository"):
        prepare(model, tmp_path, environ={})
    assert prepare(model, tmp_path, environ={}, use_git=False).spec.command == model.command


def test_prepare_collects_the_warnings_a_submit_prints(repo: Path) -> None:
    (repo / "results").mkdir()
    (repo / "results" / "old.txt").write_text("stale\n")
    prepared = prepare(
        validate(
            job_document(
                estimated_runtime_min=120,
                max_runtime_min=60,
                outputs=[{"path": "results", "s3": "s3://b/{job_id}"}],
            )
        ),
        repo,
        environ={},
    )
    assert len(prepared.warnings) == 2
    assert "pre-existing file(s) under results/" in prepared.warnings[0]
    assert "expects to be killed as `timeout`" in prepared.warnings[1]


def test_submit_warns_about_files_already_under_an_output_path(
    control_env: Path, repo: Path
) -> None:
    (repo / "results").mkdir()
    (repo / "results" / "report-elephant.md").write_text("from the last run\n")
    lines: list[str] = []
    result = submit_spec(
        host_entry(name="gpubox", gpus=["GPU-a"]),
        validate(job_document(outputs=[{"path": "results", "s3": "s3://b/{job_id}"}])),
        Settings(),
        workdir=repo,
        session=session(FakeHost()),
        environ={},
        report=lines.append,
    )
    assert any("1 pre-existing file(s) under results/" in line for line in lines)
    assert any("pre-existing" in note for note in result.notes)


def test_hf_create_survives_validation_into_the_spec() -> None:
    model = validate(job_document(outputs=[{"path": "ckpt", "hf": "org/repo", "hf_create": True}]))
    spec = model.to_spec("20260915-000000-aaaaaa")
    assert spec.outputs[0].hf_create is True
    assert validate(job_document(outputs=[{"path": "ckpt", "hf": "org/repo"}]))


def test_auto_preempt_survives_validation_into_the_spec() -> None:
    """`gpuc requeue` re-validates the mirrored spec, so a field the model does
    not know is a job that comes back without it -- or not at all."""
    assert validate(job_document(auto_preempt=True)).to_spec("20260915-000000-aaaaaa").auto_preempt
    assert not validate(job_document()).to_spec("20260915-000000-aaaaaa").auto_preempt


def test_the_estimate_fields_survive_validation_into_the_spec() -> None:
    model = validate(
        job_document(
            estimated_runtime_min=360,
            progress_command="cat progress.txt",
            progress_interval_s=30,
        )
    )
    spec = model.to_spec("20260915-000000-aaaaaa")
    assert spec.estimated_runtime_min == 360.0
    assert spec.progress_command == "cat progress.txt"
    assert spec.progress_interval_s == 30.0


def test_a_spec_that_estimates_nothing_leaves_every_estimate_field_unset() -> None:
    spec = validate(job_document()).to_spec("20260915-000000-aaaaaa")
    assert (spec.estimated_runtime_min, spec.progress_command) == (None, None)
    assert spec.progress_interval_s == 60.0


@pytest.mark.parametrize(
    "overrides",
    [
        {"estimated_runtime_min": 0},
        {"estimated_runtime_min": -5},
        {"progress_interval_s": 1},
    ],
)
def test_an_estimate_that_could_not_be_true_is_refused(overrides: dict[str, Any]) -> None:
    with pytest.raises(SubmitError):
        validate(job_document(**overrides))


def test_submit_warns_when_the_estimate_outlives_the_jobs_own_timeout(
    control_env: Path, repo: Path
) -> None:
    lines: list[str] = []
    result = submit_spec(
        host_entry(name="gpubox", gpus=["GPU-a"]),
        validate(job_document(estimated_runtime_min=600, max_runtime_min=120)),
        Settings(),
        workdir=repo,
        session=session(FakeHost()),
        environ={},
        report=lines.append,
    )
    assert any("expects to be killed as `timeout`" in line for line in lines)
    assert any("max_runtime_min" in note for note in result.notes)


def test_an_estimate_inside_the_timeout_is_not_warned_about(control_env: Path, repo: Path) -> None:
    lines: list[str] = []
    submit_spec(
        host_entry(name="gpubox", gpus=["GPU-a"]),
        validate(job_document(estimated_runtime_min=60, max_runtime_min=120)),
        Settings(),
        workdir=repo,
        session=session(FakeHost()),
        environ={},
        report=lines.append,
    )
    assert not any("timeout" in line for line in lines)
