"""`gpuc submit`: validate a spec, ship the code and secrets, enqueue on a host.

Order matters. Everything that can fail cheaply (spec validation, missing
secrets in the submitter's environment) fails before we touch the host, and
the job is built under `incoming/` and accepted by one rename, so it is only
dispatchable once its workdir, secrets and spec are all in place.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from gpuc.control.config import HostEntry, Reporter, Settings, utc_now
from gpuc.control.remote import HostSession, open_session
from gpuc.control.s3index import (
    IndexEntry,
    LocalIndex,
    S3Index,
    S3IndexError,
    default_s3_prefix,
)
from gpuc.control.status import placement_unknown
from gpuc.control.transport import (
    NO_GIT_EXCLUDES,
    Transport,
    TransportError,
    git_summary,
    git_tracked_files,
    uncommitted_patch,
)
from gpuc.host import jobs, plan, progress
from gpuc.host.jobs import JobSpec


class SubmitError(RuntimeError):
    pass


class OutputModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    s3: str | None = None
    hf: str | None = None
    hf_path: str | None = None
    hf_create: bool = False
    """Let the sync preflight create this Hugging Face repo if it is missing."""


class JobSpecModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    command: str
    name: str = ""
    setup: str | None = None
    gpus: int = Field(default=1, ge=1)
    use_shared: bool = False
    """Let this job be dispatched to the host's shared cards -- ones gpuc does
    not own and may only borrow while nobody else is on them. Off by default."""
    env: dict[str, str] = Field(default_factory=dict)
    secrets: list[str] = Field(default_factory=list)
    outputs: list[OutputModel] = Field(default_factory=list)
    sync_interval_s: int = Field(default=180, ge=10)
    priority: int = Field(default=50, ge=0, le=99)
    max_runtime_min: float | None = None
    estimated_runtime_min: float | None = Field(default=None, gt=0)
    """Roughly how long this job takes, from the runner's start. Nothing
    enforces it; it is what tells the next person whether to queue behind it."""
    progress_command: str | None = None
    """Run in the workdir every `progress_interval_s` of phase `main`; its last
    line of stdout is how far along the job is, as a fraction of one (`0.42`) or
    a percentage written with a `%` (`42%`)."""
    progress_interval_s: float = Field(default=progress.DEFAULT_INTERVAL_S, ge=5)
    auto_preempt: bool = False
    """Let the host stop this job, as often as it takes, whenever that lets a
    job queued at a lower `priority` number start right away. It re-runs from
    the start, so it belongs to work that is cheap to repeat."""
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
    """Fill `{job_id}` into the output destinations, and refuse any that would
    not carry the id.

    Every output location includes the job id, so runs never overwrite each
    other. The expanded string is what is judged, not the template: a
    destination with a literal id pasted in passes when it is this job's, and
    a mirrored spec an older build wrote with the *previous* run's id in it is
    refused at requeue rather than pointed at that run's outputs. A Hugging
    Face location is the repo plus the path in it, so the id may sit in either;
    an `hf` output with no `hf_path` uploads under the id itself.
    """
    for output in spec.outputs:
        output.s3 = _expand(output, "s3", output.s3, spec.job_id)
        output.hf = _expand(output, "hf", output.hf, spec.job_id)
        output.hf_path = _expand(output, "hf_path", output.hf_path, spec.job_id)
        if output.s3 and spec.job_id not in output.s3:
            raise _no_job_id(output, "s3", output.s3, spec.job_id)
        if (
            output.hf_path
            and spec.job_id not in output.hf_path
            and spec.job_id not in (output.hf or "")
        ):
            raise _no_job_id(output, "hf_path", output.hf_path, spec.job_id)
    return spec


def _expand(output: jobs.Output, key: str, template: str | None, job_id: str) -> str | None:
    if not template:
        return template
    try:
        return template.format(job_id=job_id)
    except (KeyError, IndexError, ValueError) as exc:
        raise SubmitError(
            f"output {output.path}: `{key}: {template}` has a placeholder this does not "
            f"know ({exc}). The only one is {{job_id}}; a literal brace is written {{{{."
        ) from exc


def _no_job_id(output: jobs.Output, key: str, destination: str, job_id: str) -> SubmitError:
    where = "it or in `hf`" if key == "hf_path" else "it"
    return SubmitError(
        f"output {output.path}: `{key}: {destination}` does not include the job id "
        f"({job_id}), so a second run would write over the first.\n"
        f"Put {{job_id}} in {where}, for example `{key}: {destination.rstrip('/')}/"
        f"{{job_id}}`. A mirrored spec that carries an earlier run's id is "
        f"refused for the same reason; edit it out and submit the file again."
    )


def from_mirror(document: dict[str, Any]) -> dict[str, Any]:
    """A mirrored spec as `requeue` submits it: the keys this build knows.

    The mirror holds what some build wrote. The id and attempt are this run's
    to assign, and a key this build does not know, at the top or on an output,
    is not a typo. What is *not* forgiven is a destination carrying an earlier
    run's literal id (builds before the mirror held the template wrote those):
    `expand_job_id` refuses it, because the alternative is a new job writing
    over an old one's outputs.
    """
    known = {k: v for k, v in document.items() if k in JobSpecModel.model_fields}
    outputs = known.get("outputs")
    if isinstance(outputs, list):
        known["outputs"] = [
            {k: v for k, v in o.items() if k in OutputModel.model_fields}
            if isinstance(o, dict)
            else o
            for o in outputs
        ]
    return known


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


@dataclass
class Prepared:
    """A spec checked for everything a submit can fail on without a host."""

    spec: JobSpec
    secrets_body: str
    warnings: list[str] = field(default_factory=list)


def prepare(
    model: JobSpecModel,
    workdir: Path,
    *,
    job_id: str | None = None,
    attempt: int = 1,
    environ: Mapping[str, str] | None = None,
    use_git: bool = True,
) -> Prepared:
    """The checks every submit runs, once, before a host is involved.

    `submit --runpod` provisions first and enqueues second, so a missing
    secret or a non-git workdir found here costs nothing, where the same
    failure discovered by a pod that is already billing costs a pod.
    """
    spec = expand_job_id(model.to_spec(job_id or jobs.new_job_id(), attempt))
    secrets_body = gather_secrets(spec.secrets, environ)
    if use_git:
        try:
            git_tracked_files(workdir)
        except TransportError as exc:
            raise _not_a_repo(workdir, exc) from exc
    warnings = [*preexisting_output_warnings(spec, workdir), *timeout_warnings(spec)]
    return Prepared(spec, secrets_body, warnings)


def check_gpu_count(model: JobSpecModel, gpu_count: int | None) -> None:
    """A rental is bought with `--gpu-count` cards; the spec must fit it."""
    if gpu_count is not None and model.gpus > gpu_count:
        raise SubmitError(
            f"the spec asks for {model.gpus} GPU(s) but this request would create a pod with "
            f"{gpu_count}.\nRaise --gpu-count, or lower `gpus:` in the spec."
        )


def _not_a_repo(workdir: Path, exc: Exception) -> SubmitError:
    return SubmitError(
        f"{workdir} is not a git repository, so there is nothing to sync: {exc}\n"
        f"Run `git init && git add -A` there, submit from your project directory, or pass "
        f"--no-git to rsync the directory as it is."
    )


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
    notes: list[str] = field(default_factory=list)
    session: HostSession | None = field(default=None, repr=False)
    """The connection the enqueue was made over, kept so that looking up where
    the job landed in the queue does not open a second one."""
    placement: dict[str, Any] = field(default_factory=placement_unknown)
    """Where the job landed in the host's queue, looked up after the enqueue:
    see `status.queue_placement`. The default is the "we could not ask" shape,
    so a caller that never looks still emits the document's promised keys as
    nulls rather than leaving them out."""

    def render(self, queue_note: str | None = None) -> str:
        lines = [f"job {self.job_id} queued on host {self.host} (attempt {self.attempt})"]
        lines += [f"  note: {note}" for note in self.notes]
        if queue_note:
            lines.append(queue_note)
        lines.append(f"  logs: gpuc logs {self.job_id} -f")
        return "\n".join(lines)

    def document(self, *, requeued_from: str | None = None) -> dict[str, Any]:
        """`gpuc submit --json` and `gpuc requeue --json`.

        `notes` are the things the text output prints as `note:` -- a spec that
        could not be mirrored, files that were already under an `outputs:` path
        -- and the job is queued regardless. `requeued_from` is null on submit.
        The `queue_*` and `starts_*` fields are the queue as it stood a moment
        after the enqueue, and are null when the host could not be asked again.
        """
        return {
            "job_id": self.job_id,
            "host": self.host,
            "attempt": self.attempt,
            "requeued_from": requeued_from,
            "notes": list(self.notes),
            **self.placement,
        }


def preexisting_output_warnings(spec: JobSpec, workdir: Path) -> list[str]:
    """Say so at submit time when an `outputs:` path is not empty in the checkout.

    Those files are synced to the host with the code and are not this job's
    results; the runner's baseline keeps them out of the upload, but the job
    that wanted them uploaded should hear about it here rather than wonder why
    nothing arrived.
    """
    warnings: list[str] = []
    for output in spec.outputs:
        # The raw `path`, exactly as the spec wrote it and as the host records
        # its baseline: formatting `{job_id}` in here would ask about a
        # directory this job has not created yet, and would raise on any other
        # placeholder the path happens to contain.
        root = workdir / output.path
        if not root.exists():
            continue
        count = sum(1 for path in root.rglob("*") if path.is_file()) if root.is_dir() else 1
        if count:
            warnings.append(
                f"{count} pre-existing file(s) under {output.path}/ are in the checkout and will "
                f"not be uploaded as this job's outputs; use a job-specific output dir "
                f"(for example {output.path}/{{job_id}}/) if you meant them to be"
            )
    return warnings


def timeout_warnings(spec: JobSpec) -> list[str]:
    """An estimate the job's own `max_runtime_min` will not let it reach.

    Both fields are the submitter's, so this is a contradiction in one file,
    and the only place anyone will notice it before the job dies as `timeout`
    hours later.
    """
    if spec.estimated_runtime_min is None or spec.max_runtime_min is None:
        return []
    if spec.estimated_runtime_min <= spec.max_runtime_min:
        return []
    return [
        f"estimated_runtime_min ({spec.estimated_runtime_min:g}) is longer than "
        f"max_runtime_min ({spec.max_runtime_min:g}), so this job expects to be killed as "
        f"`timeout` before it finishes"
    ]


def push_workdir(
    session: HostSession,
    job_id: str,
    workdir: Path,
    *,
    use_git: bool = True,
    report: Reporter = print,
) -> None:
    remote = f"{session.staging_dir(job_id)}/workdir"
    session.transport.run(f'mkdir -p "{remote}"', check=True)
    if not use_git:
        _push_without_git(session, job_id, workdir, remote, report)
        return
    try:
        summary = git_summary(workdir)
    except TransportError as exc:
        raise _not_a_repo(workdir, exc) from exc
    report(summary.render())
    if summary.files:
        session.transport.rsync(workdir, remote, summary.files)
    patch = uncommitted_patch(workdir)
    if patch:
        session.transport.put_file(patch, f"{session.staging_dir(job_id)}/uncommitted.patch", 0o644)
    session.transport.put_file(
        json.dumps(git_source(workdir), indent=2) + "\n",
        f"{session.staging_dir(job_id)}/source.json",
        0o644,
    )


def _push_without_git(
    session: HostSession, job_id: str, workdir: Path, remote: str, report: Reporter
) -> None:
    """`--no-git`: rsync the directory, minus the things that are always junk.

    Loud, because nothing here can tell a 40 GB dataset from a checkpoint
    somebody wants, and `gpuc requeue` cannot rebuild this workdir from git.
    """
    report(
        f"WARNING: --no-git, so all of {workdir} is being synced except "
        f"{', '.join(NO_GIT_EXCLUDES)}. Nothing is read from .gitignore, and `gpuc requeue` "
        f"cannot re-create this workdir from a commit."
    )
    session.transport.rsync(workdir, remote, None, NO_GIT_EXCLUDES)
    source = {"submitted_from": str(workdir), "submitted_at": utc_now(), "git": None}
    session.transport.put_file(
        json.dumps(source, indent=2) + "\n",
        f"{session.staging_dir(job_id)}/source.json",
        0o644,
    )


def enqueue_spec(session: HostSession, spec: JobSpec) -> dict[str, Any]:
    """Put the spec in the staged job dir and ask the host to accept it.

    `enqueue` rewrites the spec normalised, writes the initial state beside
    it, and renames the whole dir into `jobs/`; it also starts the dispatcher.
    """
    staged = f"{session.staging_dir(spec.job_id)}/spec.json"
    session.transport.put_file(json.dumps(spec.to_dict(), indent=2) + "\n", staged, 0o644)
    response = session.host_json(f"enqueue {shlex.quote(staged)}")
    if not isinstance(response, dict):
        raise SubmitError(f"unexpected enqueue response from {session.entry.name}: {response!r}")
    return response


def wont_fit(spec: JobSpec, entry: HostEntry) -> str | None:
    """Why this host could never run this job, or None if it could.

    The dispatcher's own rule (`plan.capacity_failure`), run here against the
    config the host itself answered with a moment ago, so the answer arrives
    before the code is shipped rather than as a failed job -- with the way
    out added, since this is the moment somebody is looking.
    """
    config = entry.config
    failure = plan.capacity_failure(
        spec.gpus,
        len(config.gpus),
        len(config.shared_entries()),
        borrows=config.may_borrow(spec),
    )
    if failure is None:
        return None
    fix = "Submit to a bigger host, or lower `gpus:` in the spec."
    if config.shared_gpus and not config.may_borrow(spec):
        fix = (
            "Add `use_shared: true` to the spec to let it wait for the shared cards, "
            "or lower `gpus:`."
        )
    return f"host {entry.name} cannot run this job: it {failure}.\n{fix}"


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
    s3: S3Index | None = None,
    spec_uri: str | None = None,
    use_git: bool = True,
    report: Reporter = print,
    prepared: Prepared | None = None,
) -> SubmitResult:
    settings = settings or Settings()
    workdir = workdir or Path.cwd()
    notes: list[str] = []

    if prepared is None:
        prepared = prepare(
            spec_model, workdir, job_id=job_id, attempt=attempt, environ=environ, use_git=use_git
        )
    spec, secrets_body = prepared.spec, prepared.secrets_body
    too_big = wont_fit(spec, entry)
    if too_big:
        raise SubmitError(too_big)

    for warning in prepared.warnings:
        report(f"WARNING: {warning}")
        notes.append(warning)

    session = session or open_session(entry, settings, transport)
    push_workdir(session, spec.job_id, workdir, use_git=use_git, report=report)
    report(f"synced to {session.staging_dir(spec.job_id)}/workdir")

    if secrets_body:
        session.transport.put_file(secrets_body, f"{session.home}/secrets/{spec.job_id}.env", 0o600)
        report(f"delivered {len(spec.secrets)} secret(s) as {spec.job_id}.env (0600)")

    s3 = s3 if s3 is not None else S3Index.from_settings(settings)
    if s3 is None:
        notes.append(
            "s3_bucket is unset, so the spec was not mirrored and `gpuc requeue` "
            "will need --host with the workdir present locally"
        )
    elif spec_uri is None:
        # `submit --runpod` mirrors the spec *before* it buys a pod, and passes
        # the uri back in; the same object twice is a wasted round trip. The
        # mirror holds `{job_id}` unexpanded, so a requeue gets its own
        # namespace rather than the one this run wrote into.
        try:
            spec_uri = s3.put_spec(spec_model.to_spec(spec.job_id, attempt))
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
    LocalIndex().record(index_entry)
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
        job_id=spec.job_id, host=entry.name, attempt=attempt, notes=notes, session=session
    )


def with_overrides(document: Mapping[str, Any], **overrides: Any) -> dict[str, Any]:
    """Spec keys a flag set, over what the file said, ignoring the ones it did not.

    Merged into the document before validation rather than onto the model
    after it, so a flag is judged by exactly the rules the same key in the file
    would have been.
    """
    return {**document, **{key: value for key, value in overrides.items() if value is not None}}


def submit_file(
    entry: HostEntry,
    job_file: str | Path,
    settings: Settings | None = None,
    overrides: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> SubmitResult:
    document = with_overrides(load_document(job_file), **dict(overrides or {}))
    return submit_spec(entry, validate(document, str(job_file)), settings, **kwargs)
