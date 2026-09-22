"""`gpuc submit` and `gpuc requeue`: what they do, for the CLI and the dashboard.

Both return the `SubmitResult` with its placement filled in; rendering it as
text or as the `--json` document stays with the caller.
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
    find_job_host,
    make_provider,
    named_registry,
    placement_after,
)
from gpuc.control.bootstrap import resync_package
from gpuc.control.config import (
    ConfigError,
    HostEntry,
    Reporter,
    Settings,
    registry_transaction,
)
from gpuc.control.providers.base import Cloud, Constraints
from gpuc.control.provision import runpod_host
from gpuc.control.remote import HostSession, RemoteError, open_session
from gpuc.control.s3index import JobIndex, S3Index, S3IndexError, S3ObjectMissing
from gpuc.control.submit import (
    JobSpecModel,
    SubmitResult,
    check_gpu_count,
    from_mirror,
    load_document,
    prepare,
    submit_file,
    submit_spec,
    validate,
    with_overrides,
)
from gpuc.control.transport import TransportError
from gpuc.host import jobs


@dataclass
class RentalOptions:
    """`--runpod` and its flags: what to rent, and how, when the job names no host."""

    gpu_names: list[str]
    min_vram_gb: int | None = None
    max_price_usd_hr: float | None = None
    clouds: list[Cloud] = field(default_factory=lambda: list[Cloud](["SECURE"]))
    cuda_min: str | None = None
    gpu_count: int = 1
    reuse: bool = True
    name_hint: str = "job"
    idle_minutes: float = 15.0
    disk_gb: int | None = None
    image: str | None = None
    health_args: str = ""

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
        health_args=rental.health_args,
    )


def mirror_spec_first(
    model: JobSpecModel, job_id: str, settings: Settings
) -> tuple[str | None, list[str]]:
    """Put the spec in S3 before spending any money, so a lost pod is still requeueable.

    Returns the uri it landed at, so the submit that follows does not PUT the
    same object a second time. `{job_id}` goes up unexpanded: the mirror is
    what `requeue` submits, and that run must land in its own namespace.
    """
    s3 = S3Index.from_settings(settings)
    if s3 is None:
        return None, [
            "s3_bucket is unset, so the spec was not mirrored before provisioning; "
            "`gpuc requeue` will need the job file again"
        ]
    try:
        return s3.put_spec(model.to_spec(job_id)), []
    except S3IndexError as exc:
        return None, [f"could not mirror the spec to S3 before provisioning: {exc}"]


def check_placement(*, runpod: bool, host: str | None) -> None:
    """Judge the provisioning flags once, before anything is bought or written."""
    if runpod and host:
        raise UsageError(
            f"--runpod creates a pod and --host {host} names a host that already "
            f"exists, so they cannot be combined. Drop one."
        )


def ensure_package_current(
    entry: HostEntry, settings: Settings, *, bootstrap: bool = True, report: Reporter = print
) -> HostEntry:
    """Read the host's config, and re-ship the package if it is not this build.

    The host's `config.json` is the only copy of what the host is, so this read
    is also what makes the rest of the submit true: the `gpus` the spec is
    judged against and the `s3_prefix` the job's outputs are recorded under are
    the host's own answer, from a moment ago, not whatever this machine last
    wrote down.

    The commit comes from the same file rather than from this registry, which
    only ever recorded what this machine shipped: two control machines against
    one box -- a laptop and a desktop -- each leave that record describing a
    host the other has since re-bootstrapped, and every submit would then skip
    the check it exists for. An *unrecorded* commit counts as older, because
    the hosts with nothing recorded were bootstrapped by the oldest builds of all.

    Only the package and the dispatcher are re-shipped: uv, the interpreter and
    health cannot have gone stale, and the job is waiting.
    """
    if not bootstrap:
        return entry
    if not (entry.bootstrapped_at or entry.pkg_commit):
        # Nothing has ever installed gpuc on this host -- not this machine, and
        # not whoever else would have left a commit in its config. Re-shipping
        # the package alone would start a dispatcher on a host with no uv and
        # no interpreter of its own, and the first anyone would hear of it is
        # the job failing there.
        raise CliError(
            f"host {entry.name} has no gpuc on it yet: nothing recorded here or in its own "
            f"config says it was ever bootstrapped.\nRun: gpuc host bootstrap {entry.name}"
        )
    if not entry.python:
        return entry
    local = version_mod.local_commit()
    session = _try_session(entry, settings)
    config = session.read_config() if session else None
    # A host that could not be asked keeps the cache it had; the submit right
    # behind this produces the transport error in full.
    if config is None:
        return entry
    if config:
        entry = _record_config(entry, config)
        host_commit = entry.pkg_commit
    else:
        # The host answered and has no config at all: its gpuc home was wiped,
        # taking the package with it. The cache here is the only copy of what
        # that host was, so it is kept rather than overwritten with nothing --
        # and an unrecorded commit re-ships below, which restores both.
        host_commit = None
    if not version_mod.needs_package_sync(local, host_commit):
        return entry
    report(
        f"host {entry.name} is running gpuc {version_mod.short(host_commit)} and this machine "
        f"has {version_mod.short(local)}: re-syncing the package and restarting the "
        f"dispatcher before enqueueing"
    )
    transport = session.transport if session else None
    updated = resync_package(entry, settings, transport=transport, report=_quiet)
    with registry_transaction() as registry:
        registry.put(updated)
    return updated


def _try_session(entry: HostEntry, settings: Settings) -> HostSession | None:
    try:
        return open_session(entry, settings)
    except (ConfigError, RemoteError, TransportError):
        return None


def _record_config(entry: HostEntry, config: dict[str, Any]) -> HostEntry:
    """Cache what the host just said about itself, for the offline commands.

    `gpuc host list` and `gpuc version` never ask a host anything, so this is
    what keeps them from repeating a bootstrap somebody else replaced. Written
    even when the config has not changed, because *when* it was read is half of
    what those commands report. Re-read under the lock and written as one
    field, because this runs on every submit and writing back the whole entry
    read at startup would undo whatever a concurrent `gpuc host probe` learned
    about the same host.
    """
    updated = entry.with_config(config)
    with registry_transaction() as registry:
        current = registry.hosts.get(entry.name)
        registry.put(current.with_config(config) if current else updated)
    return updated


def _quiet(_: str) -> None:
    """Swallow a step's progress: the caller has already said what it is doing."""


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
    check_placement(runpod=rental is not None, host=host)
    overrides = overrides or {}
    if rental is not None:
        document = with_overrides(load_document(job_file), **overrides)
        model = validate(document, str(job_file))
        check_gpu_count(model, rental.gpu_count)
        job_id = jobs.new_job_id()
        prepared = prepare(model, workdir, job_id=job_id, use_git=use_git)
        spec_uri, notes = mirror_spec_first(model, job_id, settings)
        entry = rent_host(rental, settings, report)
        entry = ensure_package_current(entry, settings, bootstrap=bootstrap, report=report)
        result = submit_spec(
            entry,
            model,
            settings,
            workdir=workdir,
            job_id=job_id,
            spec_uri=spec_uri,
            use_git=use_git,
            report=report,
            prepared=prepared,
        )
        result.notes.extend(notes)
        return _placed(result, entry, settings)
    if not host:
        raise UsageError("submit needs --host <name> (see `gpuc host list`)")
    entry = named_registry().require(host)
    entry = ensure_package_current(entry, settings, bootstrap=bootstrap, report=report)
    result = submit_file(
        entry,
        job_file,
        settings,
        overrides,
        workdir=workdir,
        use_git=use_git,
        report=report,
    )
    return _placed(result, entry, settings)


def _placed(result: SubmitResult, entry: HostEntry, settings: Settings) -> SubmitResult:
    """The result, with where the job now sits in the host's queue.

    The queue is looked up again here rather than inferred from the enqueue:
    the dispatcher the enqueue started may well have taken the job already, and
    "position 3 of 5, starts in ~2h" is the thing the submitter actually wants
    to know and cannot work out from a job id.
    """
    result.placement = placement_after(entry, result.job_id, settings, session=result.session)
    return result


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
    check_placement(runpod=rental is not None, host=host)
    registry = named_registry()
    index = JobIndex(settings).get(job_id)
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
    document = from_mirror(document)
    attempt = (index.attempt if index else 1) + 1
    model = validate(document, f"spec for {job_id}")
    prepared = prepare(model, workdir, attempt=attempt, use_git=use_git)
    if rental is not None:
        check_gpu_count(model, rental.gpu_count)
        entry = rent_host(rental, settings, report)
    else:
        # Where the job ran, by the same lookup every other job command uses:
        # the local index, then every registered host. A second client with
        # no index of its own still finds it, and an id nobody knows is exit 4.
        entry, _ = find_job_host(job_id, registry, host)
    entry = ensure_package_current(entry, settings, bootstrap=bootstrap, report=report)
    result = submit_spec(
        entry,
        model,
        settings,
        workdir=workdir,
        attempt=attempt,
        use_git=use_git,
        report=report,
        prepared=prepared,
    )
    return _placed(result, entry, settings)
