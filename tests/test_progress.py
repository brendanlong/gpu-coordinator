from __future__ import annotations

import contextlib
import os
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import IO

import pytest

from gpuc.host import jobs, paths, progress, queue, runner
from tests.conftest import make_spec


@pytest.mark.parametrize(
    ("stdout", "expected"),
    [
        ("42\n", 42.0),
        ("42.5\n", 42.5),
        ("42%\n", 42.0),
        ("  7 %  \n", 7.0),
        ("0\n", 0.0),
        ("100\n", 100.0),
        ("300/5000\n", 6.0),
        ("0/10\n", 0.0),
        ("10/10\n", 100.0),
        ("noise\nmore noise\n88\n", 88.0),
        ("88\n\n\n", 88.0),
    ],
)
def test_parse_accepts_the_documented_forms(stdout: str, expected: float) -> None:
    assert progress.parse(stdout) == expected


@pytest.mark.parametrize(
    "stdout",
    ["", "   \n", "nearly done\n", "-1\n", "101\n", "nan\n", "inf\n", "5/0\n", "5/-2\n", "a/b\n"],
)
def test_parse_refuses_anything_that_is_not_a_percentage(stdout: str) -> None:
    with pytest.raises(progress.ProgressError):
        progress.parse(stdout)


def test_a_fraction_of_one_is_not_silently_read_as_a_percentage() -> None:
    """`0.42` means 0.42%, not 42%: the ambiguity is resolved in the direction
    that cannot be off by a hundredfold without anyone noticing."""
    assert progress.parse("0.42\n") == 0.4


def test_poll_runs_in_the_workdir_with_the_given_env(tmp_path: Path) -> None:
    (tmp_path / "step").write_text("300\n")
    env = {"TOTAL": "1200", "PATH": os.environ["PATH"]}
    assert progress.poll("echo $(cat step)/$TOTAL", tmp_path, env) == 25.0


def test_poll_reports_a_failing_command(tmp_path: Path) -> None:
    with pytest.raises(progress.ProgressError, match="exited 3"):
        progress.poll("echo broken >&2; exit 3", tmp_path)


def test_poll_reports_unparseable_output(tmp_path: Path) -> None:
    with pytest.raises(progress.ProgressError, match="not a percentage"):
        progress.poll("echo almost-done", tmp_path)


def test_poll_kills_a_command_that_hangs(tmp_path: Path) -> None:
    started = time.monotonic()
    with pytest.raises(progress.ProgressError, match="longer than"):
        progress.poll("sleep 30", tmp_path, timeout_s=0.3)
    assert time.monotonic() - started < 10.0


def test_poll_kills_the_whole_session_not_just_the_shell(tmp_path: Path) -> None:
    """A grandchild left behind would be re-spawned every interval for the rest
    of the job, which is how one broken command becomes a busy host."""
    marker = tmp_path / "still-here"
    with pytest.raises(progress.ProgressError):
        progress.poll(f"(sleep 1; touch {marker}) & wait", tmp_path, timeout_s=0.3)
    time.sleep(2.0)
    assert not marker.exists()


# -- the runner's use of it ---------------------------------------------------


def prepare(**overrides: object) -> str:
    spec = make_spec(**overrides)
    job_id = queue.enqueue(spec)
    queue.remove_marker(job_id)
    jobs.update_state(job_id, status="running")
    paths.ensure_job_layout(job_id)
    return job_id


def deps(**overrides: object) -> runner.RunnerDeps:
    base: dict[str, object] = {"poll_interval_s": 0.02, "preflight": False}
    base.update(overrides)
    return runner.RunnerDeps(**base)  # type: ignore[arg-type]


@contextlib.contextmanager
def live_runner(job_id: str) -> Iterator[tuple[runner.JobRunner, IO[bytes]]]:
    """A runner set up as far as `_record_progress` needs, without a job."""
    started = runner.JobRunner(job_id, deps())
    started.env = {"PATH": os.environ["PATH"]}
    with paths.log_file(job_id).open("ab", buffering=0) as log:
        yield started, log


def seconds_from_now(stamp: str) -> float:
    return (datetime.fromisoformat(stamp) - datetime.now(UTC)).total_seconds()


def test_progress_command_records_a_percentage(gpuc_home: Path) -> None:
    job_id = prepare(command="sleep 0.5", progress_command="echo 25", progress_interval_s=0.05)
    assert runner.run_job(job_id, deps()) == 0
    state = jobs.read_state(job_id)
    assert state.progress_pct == 25.0
    assert state.progress_at and state.progress_error is None
    # The eta is cleared when the job ends: it is only a live job's business,
    # and a surviving one reads as a promise the job is still going.
    assert state.eta is None


def test_the_eta_is_extrapolated_from_the_time_main_has_taken(gpuc_home: Path) -> None:
    job_id = prepare(command="true")
    with live_runner(job_id) as (started, log):
        # A quarter done after ten minutes of `main` means thirty more to go.
        started._record_progress("echo 25", 600.0, log)  # pyright: ignore[reportPrivateUsage]
    state = jobs.read_state(job_id)
    assert state.progress_pct == 25.0
    assert state.eta and 1750.0 < seconds_from_now(state.eta) < 1850.0


def test_a_broken_progress_command_is_recorded_and_never_fails_the_job(gpuc_home: Path) -> None:
    job_id = prepare(command="sleep 0.4", progress_command="exit 9", progress_interval_s=0.05)
    assert runner.run_job(job_id, deps()) == 0
    state = jobs.read_state(job_id)
    assert (state.status, state.progress_pct) == ("succeeded", None)
    assert state.progress_error and "exited 9" in state.progress_error


def test_one_broken_poll_is_logged_once_however_often_it_repeats(gpuc_home: Path) -> None:
    job_id = prepare(
        command="sleep 0.5", progress_command="echo not-a-number", progress_interval_s=0.02
    )
    assert runner.run_job(job_id, deps()) == 0
    assert paths.log_file(job_id).read_text().count("progress command") == 1


def test_a_recovered_poll_clears_the_error(gpuc_home: Path) -> None:
    job_id = prepare(command="true")
    with live_runner(job_id) as (started, log):
        started._record_progress("exit 1", 60.0, log)  # pyright: ignore[reportPrivateUsage]
        assert jobs.read_state(job_id).progress_error
        started._record_progress("echo 40", 60.0, log)  # pyright: ignore[reportPrivateUsage]
    state = jobs.read_state(job_id)
    assert (state.progress_error, state.progress_pct) == (None, 40.0)


def test_zero_percent_leaves_the_submitters_estimate_alone(gpuc_home: Path) -> None:
    """Nothing can be extrapolated from 0%, and an estimate beats no estimate."""
    job_id = prepare(command="true", estimated_runtime_min=120.0)
    jobs.update_state(job_id, eta=jobs.utc_in(3600.0))
    with live_runner(job_id) as (started, log):
        started._record_progress("echo 0", 60.0, log)  # pyright: ignore[reportPrivateUsage]
    state = jobs.read_state(job_id)
    assert state.progress_pct == 0.0
    assert state.eta and 3000.0 < seconds_from_now(state.eta) < 3700.0


def test_estimated_runtime_min_publishes_an_eta_from_the_first_phase(gpuc_home: Path) -> None:
    job_id = prepare(setup="sleep 0.3", command="true", estimated_runtime_min=90.0)
    seen: list[str | None] = []

    def watching_sleep(seconds: float) -> None:
        seen.append(jobs.read_state(job_id).eta)
        time.sleep(seconds)

    assert runner.run_job(job_id, deps(sleep=watching_sleep)) == 0
    published = [eta for eta in seen if eta]
    assert published, "no eta was published while the job was running"
    # Published during `setup` -- the phase somebody most wants an end time for,
    # because a job installing torch looks identical to a wedged one.
    assert 5000.0 < seconds_from_now(published[0]) < 5500.0


def test_no_estimate_and_no_progress_command_means_no_eta(gpuc_home: Path) -> None:
    job_id = prepare(command="true")
    assert runner.run_job(job_id, deps()) == 0
    state = jobs.read_state(job_id)
    assert (state.eta, state.progress_pct, state.progress_error) == (None, None, None)


def test_progress_is_only_polled_during_main(gpuc_home: Path) -> None:
    job_id = prepare(
        setup="sleep 0.4",
        command="sleep 0.4",
        progress_command="echo 50",
        progress_interval_s=0.05,
    )
    phases: list[str | None] = []

    def poller(_command: str, _cwd: Path, _env: dict[str, str]) -> float:
        phases.append(jobs.read_state(job_id).phase)
        return 50.0

    assert runner.run_job(job_id, deps(progress_poller=poller)) == 0
    # During `setup` the command would be reading a file the job has not
    # started writing, and 0% of nothing is not information.
    assert phases and set(phases) == {"main"}
