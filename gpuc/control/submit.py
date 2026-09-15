"""`gpuc submit`: validate a spec, ship the code and secrets, enqueue on a host.

Order matters. Everything that can fail cheaply (spec validation, missing
secrets in the submitter's environment) fails before we touch the host, and the
queue marker is written last, so a job is only dispatchable once its workdir,
secrets and spec are all in place.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from gpuc.control.config import HostEntry, Settings, utc_now
from gpuc.control.remote import HostSession, open_session
from gpuc.control.s3index import (
    IndexEntry,
    LocalIndex,
    S3Index,
    S3IndexError,
    default_s3_prefix,
)
from gpuc.control.transport import Transport, TransportError, git_tracked_files, uncommitted_patch
from gpuc.host import jobs
from gpuc.host.jobs import JobSpec

Reporter = Callable[[str], None]


class SubmitError(RuntimeError):
    pass


class OutputModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    s3: str | None = None
    hf: str | None = None
    hf_path: str | None = None


class LowUtilModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    window_min: float = 25.0
    floor_pct: float = 5.0
    grace_min: float = 10.0


class JobSpecModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    command: str
    name: str = ""
    setup: str | None = None
    gpus: int = Field(default=1, ge=0)
    env: dict[str, str] = Field(default_factory=dict)
    secrets: list[str] = Field(default_factory=list)
    outputs: list[OutputModel] = Field(default_factory=list)
    sync_interval_s: int = Field(default=180, ge=10)
    priority: int = Field(default=50, ge=0, le=99)
    max_runtime_min: float | None = None
    low_util: LowUtilModel = Field(default_factory=LowUtilModel)
    requires: dict[str, Any] = Field(default_factory=dict)
    cleanup: Literal["on_success", "always", "never"] = jobs.DEFAULT_CLEANUP
    """When the runner deletes the job's `workdir/`. The default keeps a failed
    or cancelled one so it can be inspected."""

    @field_validator("command")
    @classmethod
    def _command_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("command must not be empty")
        return value

    def to_spec(self, job_id: str, attempt: int = 1) -> JobSpec:
        document = self.model_dump()
        document["job_id"] = job_id
        document["attempt"] = attempt
        return JobSpec.from_dict(document)


def load_document(source: str | Path) -> dict[str, Any]:
    if str(source) == "-":
        import sys

        text = sys.stdin.read()
        origin = "stdin"
    else:
        path = Path(source)
        if not path.exists():
            raise SubmitError(f"no such job file: {path}")
        text = path.read_text()
        origin = str(path)
    try:
        document = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise SubmitError(f"{origin} is not valid YAML or JSON:\n{exc}") from exc
    if not isinstance(document, dict):
        raise SubmitError(f"{origin} must contain a mapping of job fields, got {type(document)}")
    return document


def validate(document: Mapping[str, Any], origin: str = "job spec") -> JobSpecModel:
    try:
        return JobSpecModel.model_validate(dict(document))
    except ValidationError as exc:
        problems = "\n".join(
            f"  - {'.'.join(str(p) for p in error['loc']) or '(root)'}: {error['msg']}"
            for error in exc.errors()
        )
        if any(error["type"].endswith("_type") for error in exc.errors()):
            problems += (
                '\n  tip: quote values YAML reads as booleans or numbers, e.g. command: "true"'
            )
        raise SubmitError(f"{origin} is not a valid job:\n{problems}") from exc


def expand_job_id(spec: JobSpec) -> JobSpec:
    for output in spec.outputs:
        if output.s3:
            output.s3 = output.s3.format(job_id=spec.job_id)
        if output.hf:
            output.hf = output.hf.format(job_id=spec.job_id)
        if output.hf_path:
            output.hf_path = output.hf_path.format(job_id=spec.job_id)
    return spec


def gather_secrets(names: list[str], environ: Mapping[str, str] | None = None) -> str:
    environ = environ if environ is not None else os.environ
    missing = [name for name in names if not environ.get(name)]
    if missing:
        raise SubmitError(
            f"these secrets are not set in your environment: {', '.join(missing)}\n"
            f"Export them (for example `export {missing[0]}=...`) or drop them from the "
            f"spec's `secrets:` list, then submit again."
        )
    lines: list[str] = []
    for name in names:
        value = environ[name]
        if "\n" in value:
            raise SubmitError(
                f"secret {name} contains a newline; the env file format cannot hold it"
            )
        lines.append(f"{name}={value}")
    return "".join(f"{line}\n" for line in lines)


def precheck_local(
    model: JobSpecModel,
    workdir: Path,
    *,
    gpu_count: int | None = None,
    environ: Mapping[str, str] | None = None,
) -> None:
    """Everything a submit can fail on without a host, checked before we buy one.

    `submit --runpod` provisions first and enqueues second, so a missing secret
    or a non-git workdir would otherwise be discovered by a pod that is already
    billing and now has nothing to run.
    """
    if gpu_count is not None and model.gpus > gpu_count:
        raise SubmitError(
            f"the spec asks for {model.gpus} GPU(s) but this request would create a pod with "
            f"{gpu_count}.\nRaise --gpu-count, or lower `gpus:` in the spec."
        )
    gather_secrets(model.secrets, environ)
    try:
        git_tracked_files(workdir)
    except TransportError as exc:
        raise SubmitError(
            f"{workdir} is not a git repository, so there is nothing to sync: {exc}\n"
            f"Run `git init && git add -A` there, or submit from your project directory."
        ) from exc


def git_source(workdir: Path) -> dict[str, str]:
    def git(*args: str) -> str:
        proc = subprocess.run(
            ["git", "-C", str(workdir), *args], capture_output=True, text=True, check=False
        )
        return proc.stdout.strip() if proc.returncode == 0 else ""

    return {
        "commit": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "origin": git("config", "--get", "remote.origin.url"),
        "submitted_from": str(workdir),
        "submitted_at": utc_now(),
    }


@dataclass
class SubmitResult:
    job_id: str
    host: str
    attempt: int
    files: int
    spec_uri: str | None = None
    notes: list[str] = field(default_factory=list)

    def render(self) -> str:
        lines = [f"job {self.job_id} queued on host {self.host} (attempt {self.attempt})"]
        lines += [f"  note: {note}" for note in self.notes]
        lines.append(f"  logs: gpuc logs {self.job_id} -f")
        return "\n".join(lines)


def push_workdir(session: HostSession, job_id: str, workdir: Path) -> int:
    try:
        files = git_tracked_files(workdir)
    except TransportError as exc:
        raise SubmitError(
            f"{workdir} is not a git repository, so there is nothing to sync: {exc}\n"
            f"Run `git init && git add -A` there, or submit from your project directory."
        ) from exc
    remote = f"{session.job_dir(job_id)}/workdir"
    session.transport.run(f'mkdir -p "{remote}"', check=True)
    if files:
        session.transport.rsync(workdir, remote, files)
    patch = uncommitted_patch(workdir)
    if patch:
        session.transport.put_file(patch, f"{session.job_dir(job_id)}/uncommitted.patch", 0o644)
    session.transport.put_file(
        json.dumps(git_source(workdir), indent=2) + "\n",
        f"{session.job_dir(job_id)}/source.json",
        0o644,
    )
    return len(files)


def enqueue_spec(session: HostSession, spec: JobSpec) -> dict[str, Any]:
    """Hand the spec to the host over stdin; `enqueue` starts the dispatcher."""
    staged = f"{session.home}/incoming/{spec.job_id}.json"
    session.transport.put_file(json.dumps(spec.to_dict(), indent=2) + "\n", staged, 0o644)
    response = session.host_json(f"enqueue - < {shlex.quote(staged)}")
    session.run(f'rm -f "{staged}"')
    if not isinstance(response, dict):
        raise SubmitError(f"unexpected enqueue response from {session.entry.name}: {response!r}")
    return response


def submit_spec(
    entry: HostEntry,
    spec_model: JobSpecModel,
    settings: Settings | None = None,
    *,
    workdir: Path | None = None,
    transport: Transport | None = None,
    session: HostSession | None = None,
    environ: Mapping[str, str] | None = None,
    attempt: int = 1,
    job_id: str | None = None,
    local_index: LocalIndex | None = None,
    s3: S3Index | None = None,
    report: Reporter = print,
) -> SubmitResult:
    settings = settings or Settings()
    workdir = workdir or Path.cwd()
    notes: list[str] = []

    spec = expand_job_id(spec_model.to_spec(job_id or jobs.new_job_id(), attempt))
    secrets_body = gather_secrets(spec.secrets, environ)
    if spec.gpus > len(entry.gpus):
        raise SubmitError(
            f"job asks for {spec.gpus} GPU(s) but host {entry.name} owns {len(entry.gpus)}.\n"
            f"Submit to a bigger host, or lower `gpus:` in the spec."
        )

    session = session or open_session(entry, settings, transport)
    files = push_workdir(session, spec.job_id, workdir)
    report(f"synced {files} git-tracked files to {session.job_dir(spec.job_id)}/workdir")

    if secrets_body:
        session.transport.put_file(secrets_body, f"{session.home}/secrets/{spec.job_id}.env", 0o600)
        report(f"delivered {len(spec.secrets)} secret(s) as {spec.job_id}.env (0600)")

    s3 = s3 if s3 is not None else S3Index.from_settings(settings)
    spec_uri: str | None = None
    if s3 is None:
        notes.append(
            "s3_bucket is unset, so the spec was not mirrored and `gpuc requeue` "
            "will need --host with the workdir present locally"
        )
    else:
        try:
            spec_uri = s3.put_spec(spec)
        except S3IndexError as exc:
            notes.append(f"could not mirror the spec to S3: {exc}")

    response = enqueue_spec(session, spec)
    index_entry = IndexEntry(
        job_id=spec.job_id,
        host=entry.name,
        name=spec.name,
        attempt=attempt,
        submitted_at=utc_now(),
        s3_prefix=entry.s3_prefix or default_s3_prefix(settings, entry.name),
        spec_uri=spec_uri,
    )
    (local_index or LocalIndex()).record(index_entry)
    if s3 is not None:
        try:
            s3.put_index(index_entry)
        except S3IndexError as exc:
            notes.append(f"could not mirror the index entry to S3: {exc}")
    if not response.get("dispatcher_pid"):
        notes.append(
            "the host reported no dispatcher pid; run `gpuc host bootstrap` if the job stays queued"
        )

    return SubmitResult(
        job_id=spec.job_id,
        host=entry.name,
        attempt=attempt,
        files=files,
        spec_uri=spec_uri,
        notes=notes,
    )


def submit_file(
    entry: HostEntry,
    job_file: str | Path,
    settings: Settings | None = None,
    **kwargs: Any,
) -> SubmitResult:
    document = load_document(job_file)
    return submit_spec(entry, validate(document, str(job_file)), settings, **kwargs)
