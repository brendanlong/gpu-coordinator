"""`gpuc submit` and `gpuc requeue`: what they do, for the CLI and the dashboard.

One pipeline for both: parse and prepare (everything that needs no host),
look up the host, open one session,
ship this build if the host runs another, judge the fit against the host's
own config, stage, enqueue, record. Both return the `SubmitResult` with its
placement filled in; rendering it as text or as the `--json` document stays
with the caller.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from gpuc.control import version as version_mod
from gpuc.control.actions import (
    CliError,
    NotFound,
    UsageError,
    locate,
    mirror_is_the_answer,
    placement_after,
)
from gpuc.control.bootstrap import ensure_build
from gpuc.control.config import HostEntry, Reporter, Settings, open_registry
from gpuc.control.remote import HostSession, open_session
from gpuc.control.s3index import S3Index, S3ObjectMissing
from gpuc.control.submit import (
    Prepared,
    SubmitResult,
    load_document,
    prepare,
    refuse_unreadable_config,
    submit_spec,
    validate,
    with_overrides,
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
    host_commit = session.config.pkg_commit
    if session.config_read.missing or host_commit is None:
        raise CliError(
            f"host {session.entry.name} has no gpuc on it yet: its own config records no "
            f"bootstrap.\nRun: gpuc host bootstrap {session.entry.name}"
        )
    local = version_mod.local_commit()
    if not version_mod.is_other_build(host_commit, local):
        return
    report(
        f"host {session.entry.name} is running gpuc {version_mod.short(host_commit)} and this "
        f"machine has {version_mod.short(local)}: re-syncing the package and restarting the "
        f"dispatcher before enqueueing"
    )
    ensure_build(session, _quiet)


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
    overrides: dict[str, Any] | None = None,
    workdir: Path,
    use_git: bool = True,
    bootstrap: bool = True,
    report: Reporter = print,
) -> SubmitResult:
    """`gpuc submit`: everything that needs no host first, then the host."""
    if not host:
        raise UsageError(
            "submit needs --host <name> (see `gpuc host list`, or `gpuc host rent` for a new one)"
        )
    document = with_overrides(load_document(job_file), **(overrides or {}))
    model = validate(document, str(job_file))
    prepared = prepare(model, workdir, use_git=use_git)
    return enqueue(
        open_registry().require(host),
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
    workdir: Path,
    use_git: bool = True,
    bootstrap: bool = True,
    report: Reporter = print,
) -> SubmitResult:
    """`gpuc requeue`: the mirrored spec as a new job, on the host named or
    the one the job ran on."""
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
    prepared = prepare(model, workdir, requeued_from=job_id, use_git=use_git)
    session: HostSession | None = None
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
                f"Name another host with --host (`gpuc host rent` makes a new one)."
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
