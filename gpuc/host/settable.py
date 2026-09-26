"""The job settings a person may change after submit, and the one write that
changes them.

Each is a spec field whose live copy is the job's state: the spec is never
rewritten after enqueue. One table rather than a verb per field, because the
plumbing (host verb, CLI, web endpoint, mirror, `--json`) is identical and
only the rules differ -- and the rules are what this table is for.

Stdlib only, like the rest of `gpuc.host`; the control side imports the
table to build its flags and to refuse a bad value before any host is asked.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from gpuc.host import jobs, paths
from gpuc.host.jobs import JobSpec, JobState

StateCheck = Callable[[str, float, JobState], "str | None"]


@dataclass(frozen=True)
class Settable:
    field: str
    """The JobState (and JobSpec) field, and the key in every document."""
    option: str
    """`gpuc set --<option>`; `--clear-<option>` too when `clearable`."""
    kind: type[int] | type[float]
    statuses: tuple[str, ...]
    """The job statuses it may be changed in."""
    clearable: bool
    help: str
    state_check: StateCheck | None = None
    """Why this job, as it stands, may not take this value; run under the lock."""

    def value_error(self, value: Any) -> str | None:
        """Why `value` is not one this field can hold at all, whatever the job."""
        if value is None:
            return None if self.clearable else f"{self.field} cannot be cleared"
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return f"{self.field} must be a number, got {value!r}"
        if self.kind is int:
            if value != int(value) or not 0 <= value <= 99:
                return f"{self.field} must be 0-99 (lower dispatches first), got {value!r}"
            return None
        if not 0.0 < value < math.inf:
            return f"{self.field} must be a positive number of minutes, got {value!r}"
        if jobs.utc_in(value * 60.0) is None:
            # The `1e10` units typo: no date can hold it, so an estimate would
            # publish no eta and a limit would never be reached.
            return f"{self.field} of {value:g} minutes is too far away to be an end time"
        return None


def _running_for_s(state: JobState) -> float | None:
    if state.started_at is None:
        return None
    try:
        return (datetime.now(UTC) - datetime.fromisoformat(state.started_at)).total_seconds()
    except (ValueError, TypeError):
        return None


def _max_runtime_check(job_id: str, minutes: float | None, state: JobState) -> str | None:
    if state.status != "running":
        return None
    if not state.live_max_runtime:
        # Its runner claimed the job with a build that drops the key, and it
        # enforces the limit it started with: recording a new one would
        # report success for nothing.
        return (
            f"job {job_id} was started by a gpuc build that reads max_runtime_min only at "
            f"start, so it keeps the limit it has; `gpuc preempt` runs it again from the "
            f"start under this build, which would honour a new one"
        )
    elapsed = _running_for_s(state)
    if minutes is not None and elapsed is not None and minutes * 60.0 <= elapsed:
        # Almost certainly a units slip, and it would throw the run away as
        # `timeout` within a minute. Stopping it on purpose is `gpuc cancel`.
        return (
            f"job {job_id} has been running {elapsed / 60.0:.0f} min, so a limit of "
            f"{minutes:g} min would end it at once as `timeout`; `gpuc cancel` stops it"
        )
    return None


FIELDS: tuple[Settable, ...] = (
    Settable(
        "priority",
        "priority",
        int,
        ("queued",),
        clearable=False,
        help="0-99, lower dispatches first; queued jobs only (`preempt --priority` for "
        "a running one)",
    ),
    Settable(
        "estimated_runtime_min",
        "estimate",
        float,
        ("queued", "running"),
        clearable=True,
        help="minutes the job expects to take from the runner's start; informational only",
    ),
    Settable(
        "max_runtime_min",
        "max-runtime",
        float,
        ("queued", "running"),
        clearable=True,
        help="minutes from the runner's start after which the job is killed as `timeout`",
        state_check=_max_runtime_check,
    ),
)
BY_FIELD = {settable.field: settable for settable in FIELDS}


def patch_error(patch: Mapping[str, Any]) -> str | None:
    """Why this patch could not apply to any job, or None."""
    if not patch:
        return "nothing to set"
    for name, value in patch.items():
        settable = BY_FIELD.get(name)
        if settable is None:
            return f"{name} is not something `set` changes; it takes {', '.join(BY_FIELD)}"
        error = settable.value_error(value)
        if error is not None:
            return error
    return None


def _outlives_its_limit(estimate: float | None, limit: float | None) -> str | None:
    """The contradiction `submit` warns about, and the only place anyone will
    see it before the job dies as `timeout`."""
    if estimate is None or limit is None or estimate <= limit:
        return None
    return (
        f"estimated_runtime_min ({estimate:g}) is longer than this job's "
        f"max_runtime_min ({limit:g}), so it expects to be killed as "
        f"`timeout` before it finishes"
    )


def apply(job_id: str, patch: Mapping[str, Any]) -> dict[str, Any]:
    """Change every field in `patch` on one job, or none of them.

    All or nothing: a partial change is hard to report and harder to retry.
    Checked and written under the job's lock, so the status each rule was
    checked against is the one the write lands on. The answer is the job's
    status and each field as the state now holds it, plus `warning`; a
    refusal is `{job_id, error}`, with `missing` for an id this host does not
    have. `patch` has been through `patch_error`.

    A running job's runner re-reads these from the state on a timer
    (`runner.ESTIMATE_REFRESH_S`), so no message passing is needed.
    """
    if not paths.job_dir(job_id).is_dir():
        return {"job_id": job_id, "error": f"no job {job_id} on this host", "missing": True}
    try:
        spec: JobSpec | None = jobs.read_spec(job_id)
    except (RuntimeError, ValueError, OSError):
        spec = None
    try:
        with jobs.locked(job_id):
            state = jobs.read_state(job_id)
            for name, value in patch.items():
                settable = BY_FIELD[name]
                if state.status not in settable.statuses:
                    return {"job_id": job_id, "error": _wrong_status(job_id, settable, state)}
                if settable.state_check is not None:
                    error = settable.state_check(job_id, value, state)
                    if error is not None:
                        return {"job_id": job_id, "error": error}
            fields = dict(patch)
            if "max_runtime_min" in patch:
                # Only with a new limit: a running job whose runner predates
                # the live one must not have its state claim otherwise.
                fields["live_max_runtime"] = True
            state = jobs.apply_fields(state, fields)
            jobs.write_state(job_id, state)
    except (RuntimeError, OSError) as exc:
        return {"job_id": job_id, "error": f"could not change job {job_id}: {exc}"}
    return {
        "job_id": job_id,
        "status": state.status,
        **{name: getattr(state, name) for name in patch},
        "warning": _outlives_its_limit(state.estimated_runtime_min, state.max_runtime(spec)),
    }


def _wrong_status(job_id: str, settable: Settable, state: JobState) -> str:
    wanted = " or ".join(settable.statuses)
    if state.finished:
        return f"job {job_id} has already {state.status}, so its {settable.field} cannot change"
    return f"job {job_id} is {state.status}, and {settable.field} is changed only on a {wanted} job"
