"""`gpuc wait` and `gpuc logs -f`: finding out that a job ended without looking.

Every test here drives a real `gpuc.host status` round trip (see `host_home`),
because the thing being tested is precisely the reading of another process's
state file -- a stubbed poll would test the stub.

Where a test needs the job to end mid-wait it stands in for the sleep between
polls, which is both the hook and what keeps these from being the slowest tests
in the suite. The one that streams a real `tail` uses a real clock instead --
only the actual child process can prove a line reaches the screen -- and asserts
on what is in the output rather than on which line it landed at, so a slow
runner cannot fail it.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from gpuc.control import wait as wait_mod
from gpuc.control.cli import (
    EXIT_ERROR,
    EXIT_INTERRUPTED,
    EXIT_NOT_FOUND,
    EXIT_OK,
    EXIT_USAGE,
    main,
)
from gpuc.control.config import config_file, registry_transaction
from gpuc.control.providers.base import Pod
from gpuc.control.s3index import IndexEntry, LocalIndex
from gpuc.control.transport import TransportError, tail_command
from gpuc.host import jobs, paths
from gpuc.host.jobs import HostConfig, JobSpec, JobState
from tests.conftest import host_entry
from tests.fakeprovider import FakeProvider

GPU = "GPU-2a4bad3b-9fe3-7031-914d-384254e92908"
JOB = "20260915-120000-abc123"


@pytest.fixture
def host_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, control_env: Path) -> Path:
    """A registered `local` host whose GPUC_HOME really answers `gpuc.host status`.

    The `pkg` symlink is what makes the poll a round trip rather than a fake:
    the control side runs `python -m gpuc.host status` with PYTHONPATH pointing
    at it, and reads back whatever that printed.
    """
    home = tmp_path / "host-home"
    monkeypatch.setenv("GPUC_HOME", str(home))
    paths.ensure_layout()
    jobs.write_config(HostConfig(host="local", gpus=[GPU]))
    paths.heartbeat_file().touch()
    monkeypatch.delenv("GPUC_HOME")
    (home / "pkg").symlink_to(Path(__file__).resolve().parents[1])
    with registry_transaction() as registry:
        registry.put(
            host_entry(
                name="local",
                kind="local",
                gpus=[GPU],
                gpuc_home=str(home),
                python=sys.executable,
            )
        )
    return home


def put_job(home: Path, job_id: str = JOB, *, name: str = "lego-s4", **state: Any) -> None:
    """A job on the host, in whatever state the test needs it to be in."""
    with mock.patch.dict(os.environ, {"GPUC_HOME": str(home)}):
        jobs.write_spec(
            JobSpec.from_dict({"job_id": job_id, "name": name, "command": "train", "gpus": 1})
        )
        jobs.write_state(job_id, JobState(**{"status": "running", "phase": "main", **state}))


def after_polls(
    monkeypatch: pytest.MonkeyPatch, rounds: int, then: Callable[[], None]
) -> Callable[[], int]:
    """Do `then` once the wait has polled `rounds` times, and count the polls.

    Hooked onto the poll rather than onto `time.sleep`: that attribute is the
    whole process's, and `Popen.wait(timeout=...)` busy-waits on it, so a
    stand-in there fires from inside subprocess reaping too.
    """
    real = wait_mod.Watch.poll
    seen = [0]

    def poll(self: wait_mod.Watch) -> list[wait_mod.Watched]:
        settled = real(self)
        seen[0] += 1
        if seen[0] == rounds:
            then()
        return settled

    monkeypatch.setattr(wait_mod.Watch, "poll", poll)
    monkeypatch.setattr(wait_mod.time, "sleep", lambda _seconds: None)
    return lambda: seen[0]


# -- `gpuc wait` ---------------------------------------------------------------


def test_wait_on_a_finished_job_returns_its_outcome_at_once(
    host_home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    put_job(host_home, status="succeeded", phase=None, ended_at=jobs.utc_now())
    assert main(["wait", JOB]) == EXIT_OK
    assert f"lego-s4 ({JOB}) on local: succeeded" in capsys.readouterr().out


def test_wait_exits_non_zero_for_a_job_that_failed(
    host_home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The point of the command: a script learns the outcome from `$?`."""
    put_job(
        host_home,
        status="failed",
        phase=None,
        reason="sync-preflight",
        exit_code=1,
        ended_at=jobs.utc_now(),
    )
    assert main(["wait", JOB]) == EXIT_ERROR
    assert "failed (sync-preflight)" in capsys.readouterr().out


def test_wait_exits_non_zero_for_a_cancelled_job(
    host_home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Cancelled is terminal but it is not the work getting done."""
    put_job(host_home, status="cancelled", phase=None, ended_at=jobs.utc_now())
    assert main(["wait", JOB]) == EXIT_ERROR
    # `cancelled (cancelled)` would say nothing twice.
    assert "on local: cancelled" in capsys.readouterr().out


def test_wait_blocks_until_the_job_ends(
    host_home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    put_job(host_home)
    polls = after_polls(
        monkeypatch,
        3,
        lambda: put_job(host_home, status="succeeded", phase=None, ended_at=jobs.utc_now()),
    )
    assert main(["wait", JOB]) == EXIT_OK
    assert polls() == 4, "three polls saw it running, the fourth saw it end"
    assert "succeeded" in capsys.readouterr().out


def test_wait_takes_several_ids_and_fails_if_any_of_them_did(
    host_home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    other = "20260915-130000-def456"
    put_job(host_home, status="succeeded", phase=None, ended_at=jobs.utc_now())
    put_job(
        host_home,
        other,
        name="sweep-2",
        status="failed",
        phase=None,
        reason="timeout",
        ended_at=jobs.utc_now(),
    )
    assert main(["wait", JOB, other]) == EXIT_ERROR
    out = capsys.readouterr().out
    assert "succeeded" in out
    assert "failed (timeout)" in out


def test_wait_says_each_job_once_however_often_it_is_named(
    host_home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    put_job(host_home, status="succeeded", phase=None, ended_at=jobs.utc_now())
    assert main(["wait", JOB, JOB]) == EXIT_OK
    assert capsys.readouterr().out.count(JOB) == 1


def test_wait_json_is_one_document_of_final_states(
    host_home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    put_job(
        host_home,
        status="failed",
        phase=None,
        reason="timeout",
        exit_code=1,
        ended_at=jobs.utc_now(),
    )
    assert main(["wait", JOB, "--json"]) == EXIT_ERROR
    captured = capsys.readouterr()
    document = json.loads(captured.out)
    assert document["schema_version"] == 1
    assert document["errors"] == []
    job = document["jobs"][0]
    assert (job["job_id"], job["host"], job["status"]) == (JOB, "local", "failed")
    assert job["reason"] == "timeout"
    # The outcome line still exists under --json; stdout is the document alone.
    assert "failed (timeout)" in captured.err


def test_wait_on_an_id_no_host_has_is_exit_four(host_home: Path) -> None:
    """`--host` skips the search, so the host itself has to be asked."""
    assert main(["wait", "20260101-000000-aaaaaa", "--host", "local"]) == EXIT_NOT_FOUND


def test_wait_keeps_trying_a_host_it_cannot_reach_and_then_gives_up(
    host_home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """An ssh blip must not end a six-hour wait; a host that has gone must not hang one.

    The interpreter the host is run through is not there, so every poll fails
    the way a dead pod's would -- and the wait keeps asking until the grace
    runs out, which here is a fifth of a second rather than five minutes.
    """
    put_job(host_home)
    with registry_transaction() as registry:
        registry.put(
            host_entry(
                name="local", kind="local", gpus=[GPU], gpuc_home=str(host_home), python="/no/such"
            )
        )
    monkeypatch.setattr(wait_mod, "TROUBLE_GRACE_S", 0.2)
    polls = [0]
    monkeypatch.setattr(wait_mod.time, "sleep", lambda _seconds: polls.__setitem__(0, polls[0] + 1))

    assert main(["wait", JOB, "--host", "local"]) == EXIT_ERROR
    captured = capsys.readouterr()
    assert "could not be asked" in captured.out
    assert polls[0] > 1, "one failed poll is a blip, not a verdict"
    # Said once, not once per poll.
    assert captured.err.count("still waiting") == 1


def test_wait_rejects_an_interval_that_is_not_a_delay(host_home: Path) -> None:
    put_job(host_home)
    assert main(["wait", JOB, "--interval", "0"]) == EXIT_USAGE


def interrupt_after(monkeypatch: pytest.MonkeyPatch, polls: int = 1) -> None:
    """A Ctrl-C landing where one really does: in the middle of the wait.

    Hooked onto the poll rather than onto `time.sleep`, which is the whole
    process's and which `Popen.wait(timeout=...)` busy-waits on -- standing in
    for that would put the interrupt inside the tail's own reaping too.
    """
    real = wait_mod.Watch.poll
    seen = [0]

    def poll(self: wait_mod.Watch) -> list[wait_mod.Watched]:
        seen[0] += 1
        if seen[0] > polls:
            raise KeyboardInterrupt
        return real(self)

    monkeypatch.setattr(wait_mod.Watch, "poll", poll)


def test_an_interrupted_wait_says_so_and_is_not_exit_zero(
    host_home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """This command exists to be abandoned; abandoning it must not read as success."""
    put_job(host_home)
    interrupt_after(monkeypatch)
    assert main(["wait", JOB, "--interval", "0.01"]) == EXIT_INTERRUPTED
    captured = capsys.readouterr()
    assert JOB in captured.err
    assert "interrupted" in captured.err
    assert captured.out == ""


def test_an_interrupted_wait_under_json_still_prints_a_document(
    host_home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """An empty stdout is the one answer --json promises never to give."""
    put_job(host_home)
    interrupt_after(monkeypatch)
    assert main(["wait", JOB, "--json", "--interval", "0.01"]) == EXIT_INTERRUPTED
    document = json.loads(capsys.readouterr().out)
    assert document["exit_code"] == EXIT_INTERRUPTED
    assert JOB in document["error"]


def test_an_interrupt_before_the_first_poll_is_reported_too(
    host_home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Finding a job's host can ask every host in turn, 60s each: it is where a
    mistyped id gets Ctrl-C'd, and it is outside the polling loop."""
    put_job(host_home)

    def interrupted(*_args: Any, **_kwargs: Any) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(wait_mod, "start", interrupted)
    assert main(["wait", JOB, "--json"]) == EXIT_INTERRUPTED
    document = json.loads(capsys.readouterr().out)
    assert document["exit_code"] == EXIT_INTERRUPTED
    assert JOB in document["error"]


def test_wait_reads_the_outcome_from_the_mirror_when_the_host_has_gone(
    host_home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The rental case: the pod finished the job, then idled itself down.

    Reporting "could not be asked" for a run that succeeded would be wrong
    about the one thing the user was waiting for, and the mirror is where the
    spec says to look once a host is gone.
    """
    from tests.fakes3 import FakeS3Client

    put_job(host_home)
    prefix = "s3://bucket/gpuc/local"
    key = f"bucket/gpuc/local/jobs/{JOB}/state.json"
    state = json.dumps({"status": "succeeded", "ended_at": jobs.utc_now(), "exit_code": 0})
    monkeypatch.setattr(
        "gpuc.control.s3index.S3Index.client",
        property(lambda self: FakeS3Client(objects={key: state.encode()})),
    )
    config_file().write_text('s3_bucket = "bucket"\n')
    # The index `gpuc submit` writes here: the mirrored state.json has no name
    # (that lives in the spec), so this is where the line's label comes from.
    LocalIndex().record(IndexEntry(job_id=JOB, host="local", name="lego-s4", s3_prefix=prefix))
    with registry_transaction() as registry:
        registry.put(
            host_entry(
                name="local",
                kind="local",
                gpus=[GPU],
                gpuc_home=str(host_home),
                python="/no/such",
                s3_prefix=prefix,
            )
        )
    monkeypatch.setattr(wait_mod, "TROUBLE_GRACE_S", 0.0)
    monkeypatch.setattr(wait_mod.time, "sleep", lambda _seconds: None)

    assert main(["wait", JOB, "--host", "local"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "succeeded" in out
    assert "from the S3 mirror" in out
    assert f"lego-s4 ({JOB})" in out


def test_wait_reads_a_terminated_rental_from_the_mirror_without_waiting_out_the_grace(
    host_home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A pod the provider says has ended will never answer again, so the
    five-minute grace for an ssh blip would only be five minutes of nothing:
    the mirror is read on the first poll that fails."""
    from tests.fakes3 import FakeS3Client

    prefix = "s3://bucket/gpuc/gpuc-pod"
    key = f"bucket/gpuc/gpuc-pod/jobs/{JOB}/state.json"
    state = json.dumps({"status": "succeeded", "ended_at": jobs.utc_now(), "exit_code": 0})
    monkeypatch.setattr(
        "gpuc.control.s3index.S3Index.client",
        property(lambda self: FakeS3Client(objects={key: state.encode()})),
    )
    config_file().write_text('s3_bucket = "bucket"\n')
    with registry_transaction() as registry:
        registry.put(
            host_entry(
                name="gpuc-pod", kind="rental", pod_id="pod-1", ssh="root@1.2.3.4", s3_prefix=prefix
            )
        )
    ended = Pod(id="pod-1", name="gpuc-pod", status="TERMINATED", cost_usd_hr=0.0)
    monkeypatch.setattr(
        "gpuc.control.actions.make_provider",
        lambda *_a, **_k: FakeProvider(existing=[ended]),
    )

    def unreachable(*_a: Any, **_k: Any) -> Any:
        raise TransportError(message="ssh: connect to 1.2.3.4 port 22: Connection refused")

    monkeypatch.setattr(wait_mod, "open_session", unreachable)
    sleeps = [0]
    monkeypatch.setattr(wait_mod.time, "sleep", lambda _s: sleeps.__setitem__(0, sleeps[0] + 1))

    assert wait_mod.TROUBLE_GRACE_S >= 60, "the grace is left as it ships"
    assert main(["wait", JOB, "--host", "gpuc-pod"]) == EXIT_OK
    captured = capsys.readouterr()
    assert "succeeded" in captured.out
    assert "from the S3 mirror" in captured.out
    assert sleeps[0] == 0, "read on the first failed poll, not after a second one"


def test_the_mirror_still_shouts_when_a_dead_host_lost_the_outputs(
    host_home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The fallback exists for the dead-rental case, which is the same case
    that loses outputs -- so it must not be the one that stays quiet about it.

    `outputs_pending` is the host's own check against the spec and is not in
    the mirrored state, so a mirrored `outputs_lost` has to stand on its own.
    """
    from tests.fakes3 import FakeS3Client

    put_job(host_home)
    key = f"bucket/gpuc/local/jobs/{JOB}/state.json"
    state = json.dumps(
        {"status": "succeeded", "ended_at": jobs.utc_now(), "exit_code": 0, "outputs_lost": True}
    )
    monkeypatch.setattr(
        "gpuc.control.s3index.S3Index.client",
        property(lambda self: FakeS3Client(objects={key: state.encode()})),
    )
    config_file().write_text('s3_bucket = "bucket"\n')
    with registry_transaction() as registry:
        registry.put(
            host_entry(
                name="local",
                kind="local",
                gpus=[GPU],
                gpuc_home=str(host_home),
                python="/no/such",
                s3_prefix="s3://bucket/gpuc/local",
            )
        )
    monkeypatch.setattr(wait_mod, "TROUBLE_GRACE_S", 0.0)
    monkeypatch.setattr(wait_mod.time, "sleep", lambda _seconds: None)

    assert main(["wait", JOB, "--host", "local"]) == EXIT_OK
    assert "OUTPUTS LOST" in capsys.readouterr().out


def test_wait_says_once_when_a_reachable_host_has_no_dispatcher(
    host_home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A queued job there will never start, and silence looks like a busy queue."""
    put_job(host_home, status="queued", phase=None)
    (host_home / "dispatcher.heartbeat").unlink(missing_ok=True)
    after_polls(
        monkeypatch,
        3,
        lambda: put_job(host_home, status="cancelled", phase=None, ended_at=jobs.utc_now()),
    )
    assert main(["wait", JOB]) == EXIT_ERROR
    assert capsys.readouterr().err.count("dispatcher DOWN") == 1


def test_a_dead_dispatcher_is_not_mentioned_while_the_job_is_already_running(
    host_home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A running job is written to its end by its own runner, so the warning
    would be false: nothing here is waiting for the dispatcher to start it."""
    put_job(host_home)
    (host_home / "dispatcher.heartbeat").unlink(missing_ok=True)
    after_polls(
        monkeypatch,
        2,
        lambda: put_job(host_home, status="succeeded", phase=None, ended_at=jobs.utc_now()),
    )
    assert main(["wait", JOB]) == EXIT_OK
    assert "dispatcher DOWN" not in capsys.readouterr().err


def test_wait_json_carries_the_error_and_where_the_answer_came_from(
    host_home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`status` alone cannot say "we never found out"; `error` is the key to check."""
    put_job(host_home)
    with registry_transaction() as registry:
        registry.put(
            host_entry(
                name="local", kind="local", gpus=[GPU], gpuc_home=str(host_home), python="/no/such"
            )
        )
    monkeypatch.setattr(wait_mod, "TROUBLE_GRACE_S", 0.0)
    monkeypatch.setattr(wait_mod.time, "sleep", lambda _seconds: None)

    assert main(["wait", JOB, "--host", "local", "--json"]) == EXIT_ERROR
    document = json.loads(capsys.readouterr().out)
    job = document["jobs"][0]
    assert job["source"] == "host"
    assert job["status"] is None, "never seen, so there is nothing to report as its state"
    assert "could not be asked" in job["error"]
    assert document["errors"] == [job["error"]]


# -- `gpuc logs -f` ------------------------------------------------------------


def write_log(home: Path, job_id: str, text: str) -> Path:
    log = home / "jobs" / job_id / "log.txt"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(text)
    return log


def test_follow_a_finished_job_prints_the_tail_and_exits(
    host_home: Path, capfd: pytest.CaptureFixture[str]
) -> None:
    """`tail -f` on a job that has ended would hang for ever with nothing to say."""
    put_job(host_home, status="succeeded", phase=None, ended_at=jobs.utc_now())
    write_log(host_home, JOB, "epoch 1\nepoch 2\n")
    assert main(["logs", JOB, "-f"]) == EXIT_OK
    out = capfd.readouterr().out
    assert "epoch 2" in out
    assert out.strip().endswith("on local: succeeded")


def test_follow_a_failed_job_exits_with_it(
    host_home: Path, capfd: pytest.CaptureFixture[str]
) -> None:
    put_job(host_home, status="failed", phase=None, reason="gpu-preflight", ended_at=jobs.utc_now())
    write_log(host_home, JOB, "no CUDA device\n")
    assert main(["logs", JOB, "-f"]) == EXIT_ERROR
    assert "failed (gpu-preflight)" in capfd.readouterr().out


def test_follow_streams_a_running_job_and_stops_when_it_ends(
    host_home: Path, monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    """The whole issue in one test: the stream ends, and with the job's verdict.

    Real `tail`, real clock, real interval: the thing worth proving is that a
    line written just before the job ended still reaches the screen, and only
    the actual child process can prove that.
    """
    put_job(host_home)
    log = write_log(host_home, JOB, "starting\n")
    monkeypatch.setattr(wait_mod, "FLUSH_GRACE_S", 0.5)

    def end_the_job() -> None:
        time.sleep(0.3)
        with log.open("a") as handle:
            handle.write("epoch 1\n")
        put_job(host_home, status="succeeded", phase=None, ended_at=jobs.utc_now())

    ending = threading.Thread(target=end_the_job)
    ending.start()
    try:
        assert main(["logs", JOB, "-f", "--interval", "0.05"]) == EXIT_OK
    finally:
        ending.join()
    out = capfd.readouterr().out
    assert "epoch 1" in out
    # Last, because it is printed after the stream has been stopped.
    assert out.strip().splitlines()[-1] == f"lego-s4 ({JOB}) on local: succeeded"


def test_follow_forever_is_still_there_for_anyone_who_wants_it(
    host_home: Path, monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    """It never consults the job's state: a finished job would end `-f` at once."""
    put_job(host_home, status="succeeded", phase=None, ended_at=jobs.utc_now())
    write_log(host_home, JOB, "done\n")
    argv: list[list[str]] = []
    monkeypatch.setattr(
        "gpuc.control.cli.subprocess.call", lambda command: argv.append(command) or 0
    )
    assert main(["logs", JOB, "--follow-forever"]) == EXIT_OK
    assert "tail -F" in " ".join(argv[0])


def test_follow_under_json_is_still_refused(
    host_home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    put_job(host_home)
    assert main(["logs", JOB, "--json", "-f"]) == EXIT_USAGE
    assert main(["logs", JOB, "--json", "--follow-forever"]) == EXIT_USAGE


def test_the_two_follows_are_different_things_and_cannot_both_be_meant(
    host_home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    put_job(host_home)
    assert main(["logs", JOB, "-f", "--follow-forever"]) == EXIT_USAGE


def test_an_interval_is_refused_where_nothing_polls(host_home: Path) -> None:
    """It paces the check for the job's end, which a plain read never makes."""
    put_job(host_home)
    assert main(["logs", JOB, "--interval", "5"]) == EXIT_USAGE


def test_an_interrupted_follow_is_never_exit_zero(
    host_home: Path, monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    """0 means the job succeeded, so a Ctrl-C must not be able to say it."""
    put_job(host_home)
    write_log(host_home, JOB, "starting\n")
    interrupt_after(monkeypatch)
    assert main(["logs", JOB, "-f", "--interval", "0.01"]) == EXIT_INTERRUPTED
    assert "interrupted" in capfd.readouterr().err


def test_a_follow_retries_a_log_that_is_not_written_yet_and_a_read_does_not() -> None:
    """A plain read must fail on a missing log: that is what sends it to S3."""
    assert tail_command("/j/log.txt", 10, follow=True, retry=True).startswith("tail -F ")
    assert tail_command("/j/log.txt", 10, follow=True).startswith("tail -f ")
    assert tail_command("/j/log.txt", 10).startswith("tail -n ")
