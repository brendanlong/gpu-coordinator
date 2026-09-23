"""`snakemake --executor gpuc`: every Snakemake job becomes one gpuc job.

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
from collections.abc import AsyncGenerator, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Optional

from snakemake_interface_common.exceptions import WorkflowError
from snakemake_interface_executor_plugins.executors.base import SubmittedJobInfo
from snakemake_interface_executor_plugins.executors.remote import RemoteExecutor
from snakemake_interface_executor_plugins.jobs import JobExecutorInterface
from snakemake_interface_executor_plugins.settings import CommonSettings, ExecutorSettingsBase

RECENT_ALL = 1_000_000
"""`gpuc status --recent`: every finished job the hosts still hold.

The hosts return all of them anyway, and `--recent` only trims the answer; a
job that finished while hundreds of its siblings did must not fall off the
list before this executor has seen it end."""

CANCEL_PARALLELISM = 8
"""How many `gpuc cancel`s run at once when a workflow is stopped. Each is an
ssh round trip, and a Ctrl-C on a sweep should not take minutes."""

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
    # That would be a `pip install --target` into the job's environment, which
    # is a uv project's lockfile's business: the storage plugin is one of the
    # project's dependencies.
    auto_deploy_default_storage_provider=False,
)


class GpucError(Exception):
    pass


class Executor(RemoteExecutor):
    def __post_init__(self) -> None:
        self.settings: ExecutorSettings = self.executor_settings
        self.gpuc_argv = shlex.split(self.settings.gpuc)
        # Where Snakemake was started. A `workdir:` directive moves the
        # controller's cwd, but the tree gpuc must copy is still this one.
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
        self.unaskable_warned: set[str] = set()

    # The interface's own annotations on these two are narrower than what it
    # calls them for: `get_snakefile` is inferred as returning None, and
    # `check_active_jobs` is declared a coroutine but consumed with `async for`.
    def get_snakefile(self) -> str:  # pyright: ignore[reportIncompatibleMethodOverride]
        return self.snakefile_in_workdir

    def get_python_executable(self) -> str:
        return self.settings.python

    def run_job(self, job: JobExecutorInterface) -> None:
        job_info = SubmittedJobInfo(job, aux={})
        try:
            spec, argv = self.submission(job)
            answer = self.gpuc([*argv, "--json"], stdin=json.dumps(spec), env=self.secret_env())
        except (GpucError, WorkflowError) as exc:
            self.report_job_error(job_info, msg=f"gpuc submit failed: {exc}\n")
            return
        job_info.external_jobid = answer["job_id"]
        job_info.aux = {"host": answer.get("host")}
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
        gpus = int(resources.get("gpu") or 1)
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
        """One `gpuc status` for every active job, not one per job: each asks
        every host over ssh, and a sweep has hundreds of jobs in flight."""
        if not active_jobs:
            return
        hosts = {(info.aux or {}).get("host") for info in active_jobs}
        argv = ["status", "--json", "--recent", str(RECENT_ALL)]
        if len(hosts) == 1 and None not in hosts:
            argv += ["--host", str(next(iter(hosts)))]
        async with self.status_rate_limiter:
            try:
                document = self.gpuc(argv, ok_codes=(0, 1))
            except GpucError as exc:
                self.logger.info(f"gpuc status failed, asking again later: {exc}")
                for info in active_jobs:
                    yield info
                return

        jobs, host_states = index_status(document)
        for info in active_jobs:
            job_id = info.external_jobid or ""
            host = (info.aux or {}).get("host")
            seen = jobs.get(job_id)
            if seen is None:
                asked = host_states.values() if host is None else [host_states.get(host)]
                if "unaskable" in asked:
                    if job_id not in self.unaskable_warned:
                        self.unaskable_warned.add(job_id)
                        self.logger.info(f"gpuc job {job_id}: its host could not be asked")
                    yield info
                    continue
                seen = self.settle_missing(job_id, host)
            status, reason = seen
            if status == "succeeded":
                self.report_job_success(info)
            elif status in FAILED or status is None:
                label = status or "lost"
                why = f"{label}: {reason}" if reason else label
                self.report_job_error(info, msg=f"gpuc job {job_id} {why}; `gpuc logs {job_id}`.\n")
            else:
                yield info

    def settle_missing(self, job_id: str, host: str | None) -> tuple[str | None, str | None]:
        """A job no host listed: its host is gone, or no longer has it.

        `gpuc wait` is what reads the S3 mirror for a gone host, and it answers
        at once for a job that has ended, which a job nobody lists has."""
        argv = ["wait", job_id, "--json"]
        if host:
            argv += ["--host", host]
        try:
            document = self.gpuc(argv, ok_codes=(0, 1, 4))
        except GpucError as exc:
            return None, str(exc)
        final = next(iter(document.get("jobs") or []), {})
        if final.get("error"):
            return None, str(final["error"])
        return final.get("status"), final.get("reason")

    def cancel_jobs(self, active_jobs: list[SubmittedJobInfo]) -> None:
        def cancel(info: SubmittedJobInfo) -> str | None:
            argv = ["cancel", str(info.external_jobid), "--json"]
            host = (info.aux or {}).get("host")
            if host:
                argv += ["--host", host]
            try:
                self.gpuc(argv)
            except GpucError as exc:
                return f"gpuc cancel {info.external_jobid}: {exc}"
            return None

        with ThreadPoolExecutor(CANCEL_PARALLELISM) as pool:
            failures = [f for f in pool.map(cancel, active_jobs) if f]
        for failure in failures:
            self.logger.info(failure)

    def gpuc(
        self,
        args: list[str],
        *,
        stdin: str | None = None,
        env: Mapping[str, str] | None = None,
        ok_codes: tuple[int, ...] = (0,),
    ) -> dict[str, Any]:
        """Run gpuc in the project directory and return its `--json` document.

        `ok_codes` are the exits that still carry the whole answer: `status`
        exits 1 for one unreachable host and reports the rest."""
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


def index_status(
    document: Mapping[str, Any],
) -> tuple[dict[str, tuple[str, str | None]], dict[str, str]]:
    """`gpuc status --json` as `{job_id: (status, reason)}` and `{host: state}`."""
    jobs: dict[str, tuple[str, str | None]] = {}
    states: dict[str, str] = {}
    for host in document.get("hosts") or []:
        states[host.get("name")] = host.get("state")
        for key in ("queued", "running", "finished"):
            for job in host.get(key) or []:
                jobs[job["job_id"]] = (job.get("status"), job.get("reason"))
    return jobs, states


def job_name(job: JobExecutorInterface) -> str:
    wildcards = getattr(job, "wildcards_dict", None) or {}
    detail = ",".join(f"{k}={v}" for k, v in wildcards.items())
    return f"{job.name}[{detail}]" if detail else job.name


def truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)
