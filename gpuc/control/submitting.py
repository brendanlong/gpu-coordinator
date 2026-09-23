"""`gpuc submit` and `gpuc requeue`: what they do, for the CLI and the dashboard.

One pipeline for both, whichever way the host arrives: parse and prepare
(everything that needs no host), rent or look up the host, open one session,
ship this build if the host runs another, judge the fit against the host's
own config, stage, enqueue, record. Both return the `SubmitResult` with its
placement filled in; rendering it as text or as the `--json` document stays
with the caller.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from gpuc.control import version as version_mod
from gpuc.control.actions import (
    CliError,
    NotFound,
    UsageError,
    locate,
    make_provider,
    mirror_is_the_answer,
    placement_after,
)
from gpuc.control.bootstrap import DEFAULT_HEALTH, HealthOptions, ensure_build, host_build
from gpuc.control.config import HostEntry, Reporter, Settings, open_registry
from gpuc.control.providers.base import DEFAULT_CUDA_MIN, Cloud, Constraints
from gpuc.control.provision import runpod_host
from gpuc.control.remote import HostSession, open_session
from gpuc.control.s3index import S3Index, S3ObjectMissing
from gpuc.control.submit import (
    Prepared,
    SubmitResult,
    check_gpu_count,
    load_document,
    prepare,
    refuse_unreadable_config,
    submit_spec,
    validate,
    with_overrides,
)
from gpuc.host import jobs

DEFAULT_IDLE_MINUTES = 15.0
"""How long a rental sits with an empty queue before it ends itself, unless
`--idle-min` says otherwise."""


@dataclass
class RentalOptions:
    """`--runpod` and its flags: what to rent, and how, when the job names no host."""

    gpu_names: list[str]
    min_vram_gb: int | None = None
    max_price_usd_hr: float | None = None
    clouds: list[Cloud] = field(default_factory=lambda: list[Cloud](["SECURE"]))
    cuda_min: str = DEFAULT_CUDA_MIN
    gpu_count: int = 1
    reuse: bool = True
    name_hint: str = "job"
    idle_minutes: float = DEFAULT_IDLE_MINUTES
    disk_gb: int | None = None
    image: str | None = None
    health: HealthOptions = DEFAULT_HEALTH

    def constraints(self) -> Constraints:
        if not self.gpu_names:
            raise UsageError(
                "--runpod needs --gpu <name>[,<name>] (for example --gpu A40,RTX4090).\n"
                "Names are matched against the RunPod catalog, short or full."
            )
        return Constraints(
            gpu_names=self.gpu_names,
            min_vram_gb=self.min_vram_gb,
            max_price_usd_hr=self.max_price_usd_hr,
            clouds=self.clouds,
            cuda_min=self.cuda_min,
            gpu_count=self.gpu_count,
        )


def rent_host(rental: RentalOptions, settings: Settings, report: Reporter) -> HostEntry:
    return runpod_host(
        rental.constraints(),
        settings,
        provider=make_provider(settings),
        report=report,
        reuse=rental.reuse,
        name_hint=rental.name_hint,
        idle_minutes=rental.idle_minutes,
        disk_gb=rental.disk_gb if rental.disk_gb is not None else settings.disk_gb,
        image=rental.image or settings.image,
        health_options=rental.health,
    )


def check_placement(*, runpod: bool, host: str | None) -> None:
    """Judge the provisioning flags once, before anything is bought or written."""
    if runpod and host:
        raise UsageError(
            f"--runpod creates a pod and --host {host} names a host that already "
            f"exists, so they cannot be combined. Drop one."
        )


def ensure_package_current(session: HostSession, *, bootstrap: bool, report: Reporter) -> None:
    """Re-ship the package when the host's own config names another build.

    The host's `config.json` is the only copy of what the host is, and the
    session already holds it: the `pkg_commit` judged here is the same read
    the `gpus` the spec is judged against comes from. A host that has never
    been bootstrapped -- no config, or no commit in it -- is refused rather
    than half-installed on the way past: re-shipping the package alone would
    start a dispatcher on a host with no uv and no interpreter of its own, and
    the first anyone would hear of it is the job failing there.
    """
    if not bootstrap:
        return
    refuse_unreadable_config(session)
    if session.config_read.missing or host_build(session) is None:
        raise CliError(
            f"host {session.entry.name} has no gpuc on it yet: its own config records no "
            f"bootstrap.\nRun: gpuc host bootstrap {session.entry.name}"
        )
    host_commit = host_build(session)
    local = version_mod.local_commit()
    if not version_mod.is_other_build(host_commit, local):
        return
    report(
        f"host {session.entry.name} is running gpuc {version_mod.short(host_commit)} and this "
        f"machine has {version_mod.short(local)}: re-syncing the package and restarting the "
        f"dispatcher before enqueueing"
    )
    # Decided above; `always` keeps `ensure_build` from asking the same question.
    ensure_build(session, _quiet, always=True)


def _quiet(_: str) -> None:
    """Swallow a step's progress: the caller has already said what it is doing."""


def enqueue(
    entry: HostEntry,
    prepared: Prepared,
    settings: Settings,
    *,
    workdir: Path,
    use_git: bool,
    bootstrap: bool,
    report: Reporter,
    session: HostSession | None = None,
) -> SubmitResult:
    """The half of a submit that needs a host: one session, then everything
    over it -- the build check, the fit check, the enqueue, and where the job
    landed in the queue. `session` is one a lookup already opened."""
    session = session or open_session(entry, settings)
    ensure_package_current(session, bootstrap=bootstrap, report=report)
    result = submit_spec(
        session, prepared, settings, workdir=workdir, report=report, use_git=use_git
    )
    # The queue is looked up again here rather than inferred from the enqueue:
    # the dispatcher the enqueue started may well have taken the job already,
    # and "position 3 of 5, starts in ~2h" is the thing the submitter actually
    # wants to know and cannot work out from a job id.
    result.placement = placement_after(session, result.job_id, settings)
    return result


def submit_job(
    job_file: str | Path,
    settings: Settings,
    *,
    host: str | None,
    rental: RentalOptions | None = None,
    overrides: dict[str, Any] | None = None,
    workdir: Path,
    use_git: bool = True,
    bootstrap: bool = True,
    report: Reporter = print,
) -> SubmitResult:
    """`gpuc submit`: everything that needs no host first, then the host."""
    check_placement(runpod=rental is not None, host=host)
    if rental is None and not host:
        raise UsageError("submit needs --host <name> (see `gpuc host list`)")
    document = with_overrides(load_document(job_file), **(overrides or {}))
    model = validate(document, str(job_file))
    prepared = prepare(model, workdir, job_id=jobs.new_job_id(), use_git=use_git)
    if rental is not None:
        check_gpu_count(model, rental.gpu_count)
        entry = rent_host(rental, settings, report)
    else:
        entry = open_registry().require(host or "")
    return enqueue(
        entry,
        prepared,
        settings,
        workdir=workdir,
        use_git=use_git,
        bootstrap=bootstrap,
        report=report,
    )


def requeue_job(
    job_id: str,
    settings: Settings,
    *,
    host: str | None,
    rental: RentalOptions | None = None,
    workdir: Path,
    use_git: bool = True,
    bootstrap: bool = True,
    report: Reporter = print,
) -> SubmitResult:
    """`gpuc requeue`: the mirrored spec as a new job, on the host named or
    the one the job ran on."""
    check_placement(runpod=rental is not None, host=host)
    s3 = S3Index.from_settings(settings)
    if s3 is None:
        raise CliError(
            "requeue reads the spec from S3, but s3_bucket is unset in "
            "~/.config/gpu-coordinator/config.toml. Re-submit the job file instead."
        )
    try:
        document = s3.get_spec(job_id)
    except S3ObjectMissing as exc:
        # A job id nobody ever mirrored a spec for does not exist as far as
        # requeue is concerned: exit 4, like every other unknown name.
        raise NotFound(
            f"no mirrored spec for job {job_id}.\n"
            f"Check the id with `gpuc status --all`; only jobs submitted with s3_bucket "
            f"set can be requeued."
        ) from exc
    model = validate(document, f"spec for {job_id}", tolerant=True)
    prepared = prepare(
        model, workdir, job_id=jobs.new_job_id(), requeued_from=job_id, use_git=use_git
    )
    session: HostSession | None = None
    if rental is not None:
        check_gpu_count(model, rental.gpu_count)
        entry = rent_host(rental, settings, report)
    else:
        # `--host` is where it goes, so it has to be one this machine has.
        # Otherwise where the job ran, by the same lookup every other job
        # command uses: the index, then every registered host. A second
        # client with no index of its own still finds it, and an id nobody
        # knows is exit 4.
        read = open_registry()
        if host:
            entry = read.require(host)
        else:
            location = locate(job_id, read.named(), None, settings, skipped=read.skipped)
            trouble = location.trouble
            if location.entry is None or trouble is not None:
                gone = trouble is not None and mirror_is_the_answer(trouble)
                raise CliError(
                    f"job {job_id} ran on host {location.host}, which "
                    f"{'is gone' if gone else 'could not be asked'}: "
                    f"{location.trouble_reason}\n"
                    f"Name another host with --host, or --runpod."
                )
            entry, session = location.entry, location.session
    return enqueue(
        entry,
        prepared,
        settings,
        workdir=workdir,
        use_git=use_git,
        bootstrap=bootstrap,
        report=report,
        session=session,
    )
