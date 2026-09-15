from __future__ import annotations

import json
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from gpuc.control.config import HostEntry, Settings
from gpuc.control.remote import HostSession
from gpuc.control.s3index import LocalIndex, S3Index, spec_key
from gpuc.control.submit import (
    SubmitError,
    gather_secrets,
    load_document,
    submit_file,
    submit_spec,
    validate,
)
from gpuc.control.transport import CommandResult
from gpuc.host import jobs
from tests.fakes3 import FakeS3Client

REMOTE_HOME = "/home/u/.gpuc"


@dataclass
class FakeHost:
    host: str = "spar"
    commands: list[str] = field(default_factory=list)
    puts: dict[str, tuple[str, int]] = field(default_factory=dict)
    rsyncs: list[tuple[Path, str, list[str] | None]] = field(default_factory=list)

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
        self, local_root: Path, remote_path: str, files: Sequence[str] | None = None
    ) -> CommandResult:
        self.rsyncs.append((local_root, remote_path, list(files) if files else None))
        return CommandResult(self.host, ["rsync"], 0, "", "")

    def tail(self, remote_path: str, lines: int = 200, follow: bool = False) -> CommandResult:
        return CommandResult(self.host, ["tail"], 0, "", "")


def session(host: FakeHost) -> HostSession:
    entry = HostEntry(name="spar", kind="ssh", ssh="me@box", gpus=["GPU-a"], python="/usr/bin/py")
    return HostSession(entry, host, REMOTE_HOME, "/usr/bin/py")


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    (root / "src").mkdir(parents=True)
    (root / "src" / "train.py").write_text("print('hi')\n")
    (root / "big.bin").write_text("not tracked\n")
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
        ({"priority": 200}, "priority"),
        ({"outputs": [{"path": "r", "bucket": "x"}]}, "outputs.0.bucket"),
    ],
)
def test_bad_values_point_at_the_field(overrides: dict[str, Any], needle: str) -> None:
    with pytest.raises(SubmitError) as exc:
        validate(job_document(**overrides))
    assert needle in str(exc.value)


def test_yaml_and_json_both_load(tmp_path: Path) -> None:
    yaml_file = tmp_path / "job.yaml"
    yaml_file.write_text("name: t\ncommand: echo hi\ngpus: 0\n")
    json_file = tmp_path / "job.json"
    json_file.write_text('{"name": "t", "command": "echo hi", "gpus": 0}')
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
        HostEntry(name="spar", gpus=["GPU-a"]),
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
    spec = json.loads(host.puts[f"{REMOTE_HOME}/incoming/{result.job_id}.json"][0])
    assert spec["outputs"][0]["s3"] == f"s3://b/exp/{result.job_id}/results"
    assert spec["outputs"][1]["hf_path"] == result.job_id
    assert "{job_id}" not in json.dumps(spec["outputs"])


def test_submit_ships_only_git_tracked_files_plus_the_patch(control_env: Path, repo: Path) -> None:
    (repo / "src" / "train.py").write_text("print('changed')\n")
    host = FakeHost()
    result = submit_spec(
        HostEntry(name="spar", gpus=["GPU-a"]),
        validate(job_document()),
        Settings(),
        workdir=repo,
        session=session(host),
        environ={},
        report=lambda _: None,
    )
    root, dest, files = host.rsyncs[0]
    assert root == repo
    assert dest == f"{REMOTE_HOME}/jobs/{result.job_id}/workdir"
    assert files == ["src/train.py"]
    patch, _ = host.puts[f"{REMOTE_HOME}/jobs/{result.job_id}/uncommitted.patch"]
    assert "print('changed')" in patch
    source = json.loads(host.puts[f"{REMOTE_HOME}/jobs/{result.job_id}/source.json"][0])
    assert len(source["commit"]) == 40


def test_submit_delivers_secrets_0600_and_never_on_argv(control_env: Path, repo: Path) -> None:
    host = FakeHost()
    result = submit_spec(
        HostEntry(name="spar", gpus=["GPU-a"]),
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
        HostEntry(name="spar", gpus=["GPU-a"]),
        validate(job_document(name="lego")),
        Settings(),
        workdir=repo,
        session=session(host),
        environ={},
        report=lambda _: None,
    )
    enqueue = next(c for c in host.commands if "gpuc.host enqueue" in c)
    assert f"enqueue - < {REMOTE_HOME}/incoming/{result.job_id}.json" in enqueue
    assert f'GPUC_HOME="{REMOTE_HOME}"' in enqueue
    assert f'PYTHONPATH="{REMOTE_HOME}/pkg"' in enqueue
    index = LocalIndex().get(result.job_id)
    assert index is not None
    assert (index.host, index.name, index.attempt) == ("spar", "lego", 1)
    assert "s3_bucket is unset" in " ".join(result.notes)


def test_submit_mirrors_the_spec_to_s3_when_a_bucket_is_configured(
    control_env: Path, repo: Path
) -> None:
    client = FakeS3Client()
    host = FakeHost()
    result = submit_spec(
        HostEntry(name="spar", gpus=["GPU-a"]),
        validate(job_document()),
        Settings(s3_bucket="bkt"),
        workdir=repo,
        session=session(host),
        environ={},
        s3=S3Index("bkt", client),
        report=lambda _: None,
    )
    assert result.spec_uri == f"s3://bkt/{spec_key(result.job_id)}"
    spec = json.loads(client.objects[f"bkt/{spec_key(result.job_id)}"])
    assert spec["command"] == "python train.py"
    assert result.notes == []


def test_a_job_bigger_than_the_host_is_refused_early(control_env: Path, repo: Path) -> None:
    host = FakeHost()
    with pytest.raises(SubmitError) as exc:
        submit_spec(
            HostEntry(name="spar", gpus=["GPU-a"]),
            validate(job_document(gpus=4)),
            Settings(),
            workdir=repo,
            session=session(host),
            environ={},
            report=lambda _: None,
        )
    assert "host spar owns 1" in str(exc.value)
    assert host.rsyncs == []


def test_submitting_from_a_non_repository_says_what_to_do(
    control_env: Path, tmp_path: Path
) -> None:
    with pytest.raises(SubmitError, match="git init"):
        submit_spec(
            HostEntry(name="spar", gpus=["GPU-a"]),
            validate(job_document(gpus=0)),
            Settings(),
            workdir=tmp_path,
            session=session(FakeHost()),
            environ={},
            report=lambda _: None,
        )


def test_submit_file_reads_yaml(control_env: Path, repo: Path) -> None:
    (repo / "job.yaml").write_text("name: t\ncommand: echo hi\ngpus: 0\n")
    result = submit_file(
        HostEntry(name="spar", gpus=["GPU-a"]),
        repo / "job.yaml",
        Settings(),
        workdir=repo,
        session=session(FakeHost()),
        environ={},
        report=lambda _: None,
    )
    assert result.host == "spar"
    assert result.attempt == 1
