"""`snakemake --executor gpuc`: every Snakemake job that needs a GPU becomes
one gpuc job, and every other runs on the controller.

Snakemake finds this package by its name alone (any top-level
`snakemake_executor_plugin_<name>` on `sys.path`), so shipping it in the gpuc
wheel is the whole installation. Snakemake imports it only when Snakemake
runs, which is why gpuc itself does not depend on Snakemake.

It drives the `gpuc` CLI, never gpuc's modules, so it works against whatever
gpuc is on PATH however that was installed, and so every submit, cancel and
status goes through the same checks a person's would.

Each job runs in its own gpuc workdir: a copy of the tree Snakemake was
started from, taken when the job is submitted. That copy is what the job's
`snakemake` reads the Snakefile from, which is why the Snakefile is passed
relative to it rather than as the controller's absolute path. Where outputs
go is the workflow's business: see docs/snakemake.md for the two layouts that
work.
"""

import json
import os
import shlex
import subprocess
import threading
from collections.abc import AsyncGenerator, Mapping
from dataclasses import dataclass, field
from typing import Any, Optional

from snakemake_interface_common.exceptions import WorkflowError
from snakemake_interface_executor_plugins.executors.base import SubmittedJobInfo
from snakemake_interface_executor_plugins.executors.remote import RemoteExecutor
from snakemake_interface_executor_plugins.jobs import JobExecutorInterface
from snakemake_interface_executor_plugins.settings import CommonSettings, ExecutorSettingsBase

FAILED = ("failed", "cancelled")


@dataclass
class ExecutorSettings(ExecutorSettingsBase):
    # Snakemake builds its CLI flags from these annotations at runtime, and
    # reads `Optional[str]`, not `str | None`.
    host: Optional[str] = field(  # noqa: UP045
        default=None,
        metadata={
            "help": "the gpuc host to queue jobs on; a rule's `host` resource overrides it",
            "env_var": False,
            "required": False,
        },
    )
    setup: str = field(
        default="uv sync --frozen",
        metadata={
            "help": "the job spec's `setup`, run in each job's workdir before Snakemake; "
            "an empty string runs none",
            "env_var": False,
            "required": False,
        },
    )
    python: str = field(
        default="uv run --no-sync python",
        metadata={
            "help": "how to run Python inside the job's environment: both the job's "
            "`python -m snakemake` and the spec's `python`",
            "env_var": False,
            "required": False,
        },
    )
    gpuc: str = field(
        default="gpuc",
        metadata={
            "help": "the command that runs gpuc on this machine",
            "env_var": False,
            "required": False,
        },
    )


common_settings = CommonSettings(
    non_local_exec=True,
    implies_no_shared_fs=False,
    # gpuc ships the working tree itself, with every submit.
    job_deploy_sources=False,
    # Snakemake would otherwise put `export VAR='value' &&` in front of the
    # job's command, and the spec forbids secrets on a command line: every
    # variable it would pass goes to gpuc as a secret instead.
    pass_envvar_declarations_to_cmd=False,
    # That would be a `pip install --target` into the job's environment, and
    # a uv environment has no pip. `--gpuc-python "uv run --no-sync --with
    # <plugin> python"` puts one there without it being in the lockfile.
    auto_deploy_default_storage_provider=False,
)


class GpucError(Exception):
    pass


class Executor(RemoteExecutor):
    def __post_init__(self) -> None:
        self.settings: ExecutorSettings = self.executor_settings
        self.gpuc_argv = shlex.split(self.settings.gpuc)
        # Where Snakemake was started. A `workdir:` directive moves the
        # controller's cwd, but the tree gpuc must copy is still this one;
        # `--directory` moves it before this is recorded, hence the refusal.
        if getattr(self.workflow, "overwrite_workdir", None):
            raise WorkflowError(
                "--directory is not supported with --executor gpuc: gpuc copies the directory "
                "snakemake is run from, so run it from the project root and give outputs an "
                "absolute path instead"
            )
        self.project_dir = os.path.abspath(str(self.workflow.workdir_init))
        self.snakefile_in_workdir = os.path.relpath(
            os.path.abspath(str(self.workflow.main_snakefile)), self.project_dir
        )
        if self.snakefile_in_workdir.startswith(os.pardir):
            raise WorkflowError(
                f"the Snakefile {self.workflow.main_snakefile} is outside {self.project_dir}, "
                "so it would not be in the job's copy of that directory; run snakemake "
                "from a directory that contains it"
            )
        self.run_cpu_rules_locally()
        self.unaskable_shown: dict[str, str] = {}
        # Every job submitted and not yet over. Snakemake's own `active_jobs`
        # is emptied for the length of each poll, and a poll here is an ssh
        # round trip to every host: a Ctrl-C landing in one would cancel
        # nothing.
        self.in_flight: dict[str, SubmittedJobInfo] = {}
        self.in_flight_lock = threading.Lock()

    def run_cpu_rules_locally(self) -> None:
        """Mark every rule whose `gpu` is unset or a constant 0 a local rule.

        gpuc runs only jobs that need a GPU, and `gpu=1` on the GPU rules is
        already how a Snakefile written for the local executor says which
        they are. Snakemake routes a job away from the executor only if its
        rule is local, so marking the rule is the one way to keep a job off
        gpuc. A `gpu` given as a function is judged per job, in
        `submission`, where 0 is an error rather than a quiet card.

        `rules` and `localrules` are Snakemake's `Workflow`, beyond the
        executor interface: it offers no way to say a job is local."""
        workflow: Any = self.workflow
        for rule in workflow.rules:
            if is_set(rule.resources.get("gpu")):
                continue
            placed = [key for key in ("host", "runpod") if is_set(rule.resources.get(key))]
            if placed:
                raise WorkflowError(
                    f"rule {rule.name} sets `{placed[0]}` but no `gpu`, so it would run on the "
                    "controller rather than on gpuc; give it `gpu=1` or more"
                )
            workflow.localrules(rule.name)

    # The interface's own annotations on these two are narrower than what it
    # calls them for: `get_snakefile` is inferred as returning None, and
    # `check_active_jobs` is declared a coroutine but consumed with `async for`.
    def get_snakefile(self) -> str:  # pyright: ignore[reportIncompatibleMethodOverride]
        return self.snakefile_in_workdir

    def get_python_executable(self) -> str:
        return self.settings.python

    def run_job(self, job: JobExecutorInterface) -> None:
        job_info = SubmittedJobInfo(job, aux={})
        if job.is_group():
            self.report_job_error(
                job_info,
                msg="gpuc runs one Snakemake job per gpuc job; drop the rule's `group:`, since "
                "a group's summed resources would become its priority and runtime limit\n",
            )
            return
        try:
            spec, argv = self.submission(job)
            answer = self.gpuc([*argv, "--json"], stdin=json.dumps(spec), env=self.secret_env())
        except (GpucError, WorkflowError) as exc:
            self.report_job_error(job_info, msg=f"gpuc submit failed: {exc}\n")
            return
        job_info.external_jobid = answer["job_id"]
        job_info.aux = {"host": answer.get("host")}
        with self.in_flight_lock:
            self.in_flight[answer["job_id"]] = job_info
        self.report_job_submission(job_info)
        self.logger.info(
            f"Submitted job {job.jobid} as gpuc job {answer['job_id']} on {answer.get('host')}."
        )

    def submission(self, job: JobExecutorInterface) -> tuple[dict[str, Any], list[str]]:
        """The spec gpuc gets, and the `gpuc submit` argv that goes with it."""
        resources: Mapping[str, Any] = job.resources
        host = resources.get("host") or self.settings.host
        runpod = resources.get("runpod")
        if not host and not runpod:
            raise WorkflowError(
                "no gpuc host: pass --gpuc-host, or give the rule a `host` or `runpod` resource"
            )
        gpus = int(resources.get("gpu") or 0)
        if gpus < 1:
            raise WorkflowError(
                f"rule {job.name}'s `gpu` came to {gpus} for this job, and a job without a GPU "
                "can only run on the controller, which is decided per rule: make `gpu` a "
                "constant 0 or leave it out, or split the rule"
            )
        spec: dict[str, Any] = {
            "name": job_name(job),
            "command": self.format_job_exec(job),
            "python": self.settings.python,
            "gpus": gpus,
            "secrets": sorted(self.envvars()),
        }
        if self.settings.setup.strip():
            spec["setup"] = self.settings.setup
        for key in ("priority", "max_runtime_min"):
            if resources.get(key) is not None:
                spec[key] = resources[key]
        for key in ("use_shared", "auto_preempt"):
            if resources.get(key) is not None:
                spec[key] = truthy(resources[key])

        argv = ["submit", "-"]
        if runpod:
            argv += ["--runpod", "--gpu", str(runpod), "--gpu-count", str(gpus)]
            if resources.get("vram_gb") is not None:
                argv += ["--min-vram", str(resources["vram_gb"])]
        else:
            argv += ["--host", str(host)]
        return spec, argv

    def secret_env(self) -> dict[str, str]:
        """Snakemake's `--envvars` and its storage plugins' credentials, which
        `gpuc submit` reads from its own environment by name."""
        return {**os.environ, **self.envvars()}

    async def check_active_jobs(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, active_jobs: list[SubmittedJobInfo]
    ) -> AsyncGenerator[SubmittedJobInfo, None]:
        """One `gpuc status` naming every active job, never one per job: a
        sweep has hundreds in flight, and gpuc asks each host once for all of
        its own.

        A job with no `error` is where its status says. One whose host could
        not be asked stays active, with the reason shown. Any other `error`
        is final -- the host has no such job, or is gone and its mirror has
        no end for it -- and the job has failed."""
        if not active_jobs:
            return
        ids = [info.external_jobid or "" for info in active_jobs]
        async with self.status_rate_limiter:
            try:
                document = self.gpuc(["status", "--json", *ids], ok_codes=(0, 1, 4))
            except GpucError as exc:
                self.logger.info(f"gpuc status failed, asking again later: {exc}")
                for info in active_jobs:
                    yield info
                return

        answers = {job.get("job_id"): job for job in document.get("jobs") or []}
        for info, job_id in zip(active_jobs, ids, strict=True):
            job = answers.get(job_id) or {"error": "gpuc status did not mention it"}
            error, status = job.get("error"), job.get("status")
            if error is not None and job.get("host_state") == "unaskable":
                if self.unaskable_shown.get(job_id) != error:
                    self.logger.info(f"gpuc job {job_id}: {error}")
                self.unaskable_shown[job_id] = error
                yield info
            elif error is not None:
                self.report_job_error(info, msg=f"gpuc job {job_id}: {error}\n")
            elif status == "succeeded":
                self.report_job_success(info)
            elif status in FAILED:
                reason = job.get("reason")
                why = f"{status}: {reason}" if reason and reason != status else status
                self.report_job_error(info, msg=f"gpuc job {job_id} {why}; `gpuc logs {job_id}`.\n")
            else:
                self.unaskable_shown.pop(job_id, None)
                yield info

    def report_job_success(self, job_info: SubmittedJobInfo) -> None:
        self.settled(job_info)
        super().report_job_success(job_info)

    def report_job_error(self, job_info: SubmittedJobInfo, msg: Any = None, **kwargs: Any) -> None:
        self.settled(job_info)
        super().report_job_error(job_info, msg, **kwargs)

    def settled(self, job_info: SubmittedJobInfo) -> None:
        with self.in_flight_lock:
            self.in_flight.pop(job_info.external_jobid or "", None)

    def cancel(self) -> None:
        with self.in_flight_lock:
            in_flight = list(self.in_flight.values())
        self.cancel_jobs(in_flight)
        self.shutdown()

    def cancel_jobs(self, active_jobs: list[SubmittedJobInfo]) -> None:
        ids = [str(info.external_jobid) for info in active_jobs if info.external_jobid]
        if not ids:
            return
        try:
            document = self.gpuc(["cancel", "--json", *ids], ok_codes=(0, 1, 4))
        except GpucError as exc:
            self.logger.info(f"gpuc cancel failed: {exc}")
            return
        for error in document.get("errors") or []:
            self.logger.info(f"gpuc cancel: {error}")

    def gpuc(
        self,
        args: list[str],
        *,
        stdin: str | None = None,
        env: Mapping[str, str] | None = None,
        ok_codes: tuple[int, ...] = (0,),
    ) -> dict[str, Any]:
        """Run gpuc in the project directory and return its `--json` document.

        `ok_codes` are the exits that still carry the whole answer: a command
        on several ids exits 1 or 4 for one of them and reports every one."""
        try:
            proc = subprocess.run(
                [*self.gpuc_argv, *args],
                input=stdin,
                capture_output=True,
                text=True,
                cwd=self.project_dir,
                env=dict(env) if env is not None else None,
                check=False,
            )
        except OSError as exc:
            raise GpucError(f"{self.settings.gpuc}: {exc}") from exc
        try:
            document = json.loads(proc.stdout)
        except json.JSONDecodeError:
            document = None
        if not isinstance(document, dict):
            raise GpucError(f"exit {proc.returncode}, no JSON document: {proc.stderr.strip()}")
        if "error" in document:
            raise GpucError(str(document["error"]))
        if proc.returncode not in ok_codes:
            raise GpucError(f"exit {proc.returncode}: {proc.stderr.strip()}")
        return document


def job_name(job: JobExecutorInterface) -> str:
    wildcards = getattr(job, "wildcards_dict", None) or {}
    detail = ",".join(f"{k}={v}" for k, v in wildcards.items())
    return f"{job.name}[{detail}]" if detail else job.name


def is_set(resource: Any) -> bool:
    return resource.is_evaluable() or bool(resource.value)


def truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)
