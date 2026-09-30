"""A runner whose queue another machine took over stands down by itself."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from gpuc.host import jobs, owner, paths, procs
from tests.conftest import wait_until
from tests.test_runner import deps, log_of, prepare, run


class StoodDown(Exception):
    pass


def hard_exit(code: int) -> None:
    raise StoodDown(code)


def another_machine_takes_over() -> None:
    jobs.atomic_write_json(paths.owner_file(), {"instance": "another-pod/another-boot"})


def test_a_runner_started_after_the_takeover_claims_nothing(gpuc_home: Path) -> None:
    job_id = prepare()
    another_machine_takes_over()
    assert run(job_id, deps(hard_exit=hard_exit)) == 0
    assert jobs.read_state(job_id).status == "queued"


def test_a_running_job_is_killed_and_nothing_more_is_written(gpuc_home: Path) -> None:
    job_id = prepare(command="echo started; sleep 30")
    taken = False

    def sleep(seconds: float) -> None:
        nonlocal taken
        time.sleep(seconds)
        if not taken and "started" in log_of(job_id):
            another_machine_takes_over()
            taken = True

    start = time.monotonic()
    with pytest.raises(StoodDown):
        run(job_id, deps(sleep=sleep, hard_exit=hard_exit))
    assert time.monotonic() - start < owner.CHECK_S + 10
    state = jobs.read_state(job_id)
    assert (state.status, state.phase, state.ended_at) == ("running", "main", None)
    assert state.pgid is not None
    # Members, not the group: the killed job is this test's unreaped zombie.
    group = procs.JobProcesses(job_pgid=state.pgid)
    wait_until(lambda: not group.members(), what="the job to die")
    assert "STANDING DOWN: another-pod/another-boot" in log_of(job_id)


def test_the_runner_checks_again_before_the_write_that_ends_the_attempt(
    gpuc_home: Path,
) -> None:
    job_id = prepare(command="true")

    def sleep(seconds: float) -> None:
        time.sleep(seconds)
        another_machine_takes_over()

    with pytest.raises(StoodDown):
        run(job_id, deps(sleep=sleep, hard_exit=hard_exit))
    assert jobs.read_state(job_id).status == "running"
