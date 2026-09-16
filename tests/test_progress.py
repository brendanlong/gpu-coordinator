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
        ("42%\n", 42.0),
        ("42.5%\n", 42.5),
        ("  7 %  \n", 7.0),
        ("0%\n", 0.0),
        ("100%\n", 100.0),
        ("0.42\n", 42.0),
        ("0.0\n", 0.0),
        ("1.0\n", 100.0),
        ("0.06\n", 6.0),
        ("noise\nmore noise\n0.88\n", 88.0),
        ("0.88\n\n\n", 88.0),
    ],
)
def test_parse_accepts_a_fraction_or_a_percentage(stdout: str, expected: float) -> None:
    assert progress.parse(stdout) == expected


@pytest.mark.parametrize(
    "stdout",
    [
        "",
        "   \n",
        "nearly done\n",
        "-0.5\n",
        "1.5\n",
        "-1\n",
        "101%\n",
        "-1%\n",
        "nan\n",
        "inf\n",
        "%\n",
    ],
)
def test_parse_refuses_anything_that_is_not_a_progress_reading(stdout: str) -> None:
    with pytest.raises(progress.ProgressError):
        progress.parse(stdout)


@pytest.mark.parametrize("stdout", ["42\n", "1\n", "0\n", "100\n"])
def test_a_bare_integer_is_refused_rather_than_guessed_at(stdout: str) -> None:
    """`42` could be 42% or an impossible fraction, and `1` could be 1% or a
    finished job. Nothing applied after the fact can tell, so the unit has to
    be in the input -- a `%` or a decimal point."""
    with pytest.raises(progress.ProgressError, match="refuses to guess") as caught:
        progress.parse(stdout)
    assert f"`{stdout.strip()}%`" in str(caught.value)


def test_a_raw_step_counter_fails_from_its_very_first_reading() -> None:
    """The case the decimal point buys: `echo $step` at step 1 would otherwise
    be a fraction, and the job would claim to be finished on its first poll."""
    for step in ("0", "1", "2"):
        with pytest.raises(progress.ProgressError):
            progress.parse(f"{step}\n")


def test_a_percentage_that_floating_point_nudged_over_a_hundred_still_lands() -> None:
    """A job computing its own percentage prints this at the exact moment it
    finishes; erroring there would be absurd."""
    assert progress.parse("100.0000001%\n") == 100.0


def test_a_percentage_genuinely_over_a_hundred_is_still_refused() -> None:
    with pytest.raises(progress.ProgressError, match="between 0% and 100%"):
        progress.parse("100.6%\n")


def test_poll_runs_in_the_workdir_with_the_given_env(tmp_path: Path) -> None:
    (tmp_path / "step").write_text("300\n")
    env = {"TOTAL": "1200", "PATH": os.environ["PATH"]}
    assert progress.poll("echo $(( $(cat step) * 100 / TOTAL ))%", tmp_path, env) == 25.0


def test_poll_reports_a_failing_command(tmp_path: Path) -> None:
    with pytest.raises(progress.ProgressError, match="exited 3"):
        progress.poll("echo broken >&2; exit 3", tmp_path)


def test_poll_reports_unparseable_output(tmp_path: Path) -> None:
    with pytest.raises(progress.ProgressError, match="not a progress reading"):
        progress.poll("echo almost-done", tmp_path)


def test_poll_kills_a_command_that_hangs(tmp_path: Path) -> None:
    started = time.monotonic()
    with pytest.raises(progress.ProgressError, match="longer than"):
        progress.poll("sleep 30", tmp_path, timeout_s=0.3)
    assert time.monotonic() - started < 10.0


def test_poll_is_not_wedged_by_a_grandchild_that_left_the_session(tmp_path: Path) -> None:
    """`setsid` escapes the killpg, and with a pipe it would also hold stdout
    open and block the reap for ever -- taking the runner's cancel, TTL and
    timeout checks down with it. Output goes to a file for exactly this."""
    started = time.monotonic()
    with pytest.raises(progress.ProgressError, match="longer than"):
        progress.poll("setsid sleep 30 & wait", tmp_path, timeout_s=0.3)
    assert time.monotonic() - started < 10.0


def test_poll_reads_only_the_tail_of_a_command_that_floods_stdout(tmp_path: Path) -> None:
    """`cat train.log` next to the documented `tail -1` must cost a bounded
    read, not the whole log in the runner's memory."""
    command = f"head -c {progress.MAX_OUTPUT_BYTES * 4} /dev/zero | tr '\\0' 'x'; echo; echo 0.5"
    assert progress.poll(command, tmp_path) == 50.0


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
    job_id = prepare(command="sleep 0.5", progress_command="echo 25%", progress_interval_s=0.05)
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
        started._record_progress("echo 25%", 600.0, log)  # pyright: ignore[reportPrivateUsage]
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
        started._record_progress("echo 0.40", 60.0, log)  # pyright: ignore[reportPrivateUsage]
    state = jobs.read_state(job_id)
    assert (state.progress_error, state.progress_pct) == (None, 40.0)


def test_zero_percent_leaves_the_submitters_estimate_alone(gpuc_home: Path) -> None:
    """Nothing can be extrapolated from 0%, and an estimate beats no estimate."""
    job_id = prepare(command="true", estimated_runtime_min=120.0)
    jobs.update_state(job_id, eta=jobs.utc_in(3600.0))
    with live_runner(job_id) as (started, log):
        started._record_progress("echo 0%", 60.0, log)  # pyright: ignore[reportPrivateUsage]
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


def test_an_estimate_added_while_the_job_runs_becomes_an_eta(gpuc_home: Path) -> None:
    """The case the whole command exists for: the long job already running when
    the next person arrives is the one nobody could estimate in time."""
    job_id = prepare(command="sleep 0.6")
    seen: list[str | None] = []

    def watching_sleep(seconds: float) -> None:
        if not seen:
            jobs.update_spec(job_id, estimated_runtime_min=90.0)
        seen.append(jobs.read_state(job_id).eta)
        time.sleep(seconds)

    assert runner.run_job(job_id, deps(sleep=watching_sleep, spec_refresh_s=0.0)) == 0
    published = [eta for eta in seen if eta]
    assert published, "the estimate never became an eta"
    assert 5000.0 < seconds_from_now(published[-1]) < 5500.0


def test_clearing_the_estimate_withdraws_the_eta_it_published(gpuc_home: Path) -> None:
    """An estimate somebody decided was wrong must not outlive the decision."""
    job_id = prepare(command="true", estimated_runtime_min=90.0)
    with live_runner(job_id) as (started, _log):
        started._publish_estimated_eta(90.0, 0.0)  # pyright: ignore[reportPrivateUsage]
        assert jobs.read_state(job_id).eta
        started._publish_estimated_eta(None, 0.0)  # pyright: ignore[reportPrivateUsage]
    assert jobs.read_state(job_id).eta is None


def test_an_eta_the_runner_did_not_publish_is_not_withdrawn(gpuc_home: Path) -> None:
    """The re-read runs for the rest of the job; a job with no estimate at all
    must not have its state rewritten every thirty seconds."""
    job_id = prepare(command="true")
    jobs.update_state(job_id, eta=jobs.utc_in(3600.0))
    with live_runner(job_id) as (started, _log):
        started._publish_estimated_eta(None, 0.0)  # pyright: ignore[reportPrivateUsage]
    assert jobs.read_state(job_id).eta


def test_a_measured_eta_is_not_overwritten_by_an_estimate(gpuc_home: Path) -> None:
    """The re-read runs on a timer for the rest of the job; a guess replacing a
    measurement every thirty seconds would be worse than never re-reading."""
    job_id = prepare(
        command="sleep 0.6",
        progress_command="echo 50%",
        progress_interval_s=0.02,
        estimated_runtime_min=600.0,
    )
    etas: list[str] = []

    def watching_sleep(seconds: float) -> None:
        eta = jobs.read_state(job_id).eta
        if eta and jobs.read_state(job_id).progress_pct == 50.0:
            etas.append(eta)
        time.sleep(seconds)

    assert runner.run_job(job_id, deps(sleep=watching_sleep, spec_refresh_s=0.0)) == 0
    assert etas, "no progress eta was published"
    # Half done after a fraction of a second: the measured eta is seconds away,
    # nowhere near the ten hours the spec guesses at.
    assert all(seconds_from_now(eta) < 600.0 for eta in etas)


def test_a_progress_command_added_while_the_job_runs_is_polled(gpuc_home: Path) -> None:
    job_id = prepare(command="sleep 0.8", progress_interval_s=0.02)
    added = False

    def adding_sleep(seconds: float) -> None:
        nonlocal added
        if not added:
            added = True
            jobs.update_spec(job_id, progress_command="echo 25%")
        time.sleep(seconds)

    assert runner.run_job(job_id, deps(sleep=adding_sleep, spec_refresh_s=0.0)) == 0
    assert jobs.read_state(job_id).progress_pct == 25.0


def test_a_spec_that_cannot_be_read_leaves_the_running_job_alone(gpuc_home: Path) -> None:
    """A spec.json being rewritten under us is a transient state, not a reason
    to end a job that is running fine."""
    job_id = prepare(command="sleep 0.4", estimated_runtime_min=90.0)
    truncated = False

    def truncating_sleep(seconds: float) -> None:
        nonlocal truncated
        if not truncated:
            truncated = True
            paths.spec_file(job_id).write_text("{ not json")
        time.sleep(seconds)

    assert runner.run_job(job_id, deps(sleep=truncating_sleep, spec_refresh_s=0.0)) == 0
    assert jobs.read_state(job_id).status == "succeeded"


def test_no_estimate_and_no_progress_command_means_no_eta(gpuc_home: Path) -> None:
    job_id = prepare(command="true")
    assert runner.run_job(job_id, deps()) == 0
    state = jobs.read_state(job_id)
    assert (state.eta, state.progress_pct, state.progress_error) == (None, None, None)


def test_progress_is_only_polled_during_main(gpuc_home: Path) -> None:
    job_id = prepare(
        setup="sleep 0.4",
        command="sleep 0.4",
        progress_command="echo 50%",
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


def test_an_unrepresentable_estimate_is_no_estimate_rather_than_a_dead_job(
    gpuc_home: Path,
) -> None:
    """`estimated_runtime_min: .inf` passes `gt=0` validation and reaches the
    host. Overflowing `timedelta` inside the monitor loop would end the job as
    `runner-died` -- an estimate deciding an outcome it may never touch."""
    for absurd in (float("inf"), 1e15, 5e9):
        job_id = prepare(command="true", estimated_runtime_min=absurd)
        assert runner.run_job(job_id, deps()) == 0
        state = jobs.read_state(job_id)
        assert (state.status, state.eta) == ("succeeded", None)


@pytest.mark.parametrize("interval", [0, -5, float("nan")])
def test_an_unusable_progress_interval_falls_back_to_the_default(interval: float) -> None:
    """Zero or negative would fork a shell on every pass of the runner's loop;
    NaN would silently never poll. The control side's `ge=5` is not in the path
    of a hand-edited spec.json or a staged incoming/<id>.json."""
    spec = make_spec(progress_interval_s=interval)
    assert spec.progress_interval_s == progress.DEFAULT_INTERVAL_S


def test_a_usable_progress_interval_is_left_alone() -> None:
    assert make_spec(progress_interval_s=0.5).progress_interval_s == 0.5


def test_a_job_whose_runner_died_does_not_keep_an_eta(gpuc_home: Path) -> None:
    """Anything keying on `eta_s` would count a dead job as still pending."""
    from gpuc.host import dispatcher

    job_id = prepare(command="true", estimated_runtime_min=360.0)
    jobs.update_state(job_id, eta=jobs.utc_in(3600.0), runner_pid=None)
    dispatcher.Dispatcher()._mark_runner_died(job_id)  # pyright: ignore[reportPrivateUsage]
    state = jobs.read_state(job_id)
    assert (state.status, state.reason, state.eta) == ("failed", "runner-died", None)
