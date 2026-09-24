"""A job whose host is gone: every command reads the mirror, the same way.

Gone is a pod the provider reports terminated, a name the index gives that
this machine has no entry for, and a `--host` this machine has no entry for.
Each test here drives one command against one of those and checks it gets
the mirror's answer -- or, with no mirror, says the job is lost.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from gpuc.control import wait as wait_mod
from gpuc.control.cli import EXIT_ERROR, EXIT_NOT_FOUND, EXIT_OK, main
from gpuc.control.config import config_file, registry_transaction
from gpuc.control.providers.base import Pod
from gpuc.control.s3index import IndexEntry, LocalIndex
from tests.conftest import host_entry, load_registry
from tests.fakeprovider import FakeProvider
from tests.fakes3 import FakeS3Client

JOB = "20260915-120000-abc123"
POD = "gpuc-pod"
PREFIX = f"s3://bucket/gpuc/{POD}"


def ago(**delta: float) -> str:
    return (datetime.now(UTC) - timedelta(**delta)).isoformat()


def mirror(
    monkeypatch: pytest.MonkeyPatch,
    jobs: dict[str, dict[str, Any]],
    *,
    host: str = POD,
    logs: dict[str, str] | None = None,
) -> None:
    """An S3 mirror holding each job's state.json (and log), with each job in
    the local index as `gpuc submit` recorded it."""
    objects = {
        f"bucket/gpuc/{host}/jobs/{job_id}/state.json": json.dumps(state).encode()
        for job_id, state in jobs.items()
    }
    objects.update(
        {
            f"bucket/gpuc/{host}/jobs/{job_id}/log.txt": text.encode()
            for job_id, text in (logs or {}).items()
        }
    )
    client = FakeS3Client(objects=objects)
    monkeypatch.setattr("gpuc.control.s3index.S3Index.client", property(lambda self: client))
    config_file().write_text('s3_bucket = "bucket"\n')
    for job_id in jobs:
        LocalIndex().record(
            IndexEntry(
                job_id=job_id, host=host, name="lego-s4", s3_prefix=f"s3://bucket/gpuc/{host}"
            )
        )


def succeeded(**when: float) -> dict[str, Any]:
    return {"status": "succeeded", "ended_at": ago(**when), "exit_code": 0}


@pytest.fixture
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(wait_mod.time, "sleep", lambda _s: None)


# -- naming the host must not make the answer worse -----------------------------


@pytest.mark.parametrize("named", [False, True])
def test_logs_of_a_job_on_a_forgotten_host_read_the_mirror(
    control_env: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    named: bool,
) -> None:
    mirror(monkeypatch, {JOB: succeeded(minutes=5)}, logs={JOB: "loss 0.1\n"})
    host = ["--host", POD] if named else []
    assert main(["logs", JOB, *host, "--json"]) == EXIT_OK
    document = json.loads(capsys.readouterr().out)
    assert (document["source"], document["lines"]) == ("s3", ["loss 0.1"])


@pytest.mark.parametrize("named", [False, True])
def test_wait_on_a_job_on_a_forgotten_host_reads_the_mirror(
    control_env: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    no_sleep: None,
    named: bool,
) -> None:
    mirror(monkeypatch, {JOB: succeeded(minutes=5)})
    host = ["--host", POD] if named else []
    assert main(["wait", JOB, *host]) == EXIT_OK
    assert "from the S3 mirror" in capsys.readouterr().out


# -- the per-job verbs ----------------------------------------------------------


@pytest.mark.parametrize("named", [False, True])
def test_cancel_of_a_job_on_a_gone_host_says_how_it_ended(
    control_env: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    named: bool,
) -> None:
    """What a live host says to a cancel of a finished job: its status, exit 0."""
    mirror(monkeypatch, {JOB: {"status": "failed", "ended_at": ago(minutes=5), "exit_code": 3}})
    host = ["--host", POD] if named else []
    assert main(["cancel", JOB, *host, "--json"]) == EXIT_OK
    (job,) = json.loads(capsys.readouterr().out)["jobs"]
    assert (job["host"], job["status"], job["source"]) == (POD, "failed", "mirror")


def test_reorder_of_a_job_on_a_gone_host_is_refused_with_how_it_ended(
    control_env: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    mirror(monkeypatch, {JOB: succeeded(minutes=5)})
    assert main(["reorder", JOB, "--priority", "3"]) == EXIT_ERROR
    err = capsys.readouterr().err
    assert "cannot reorder" in err and "ended succeeded" in err and "is gone" in err


# -- with no mirror, gone is lost -----------------------------------------------


def test_with_no_bucket_a_job_on_a_gone_host_is_lost_and_every_command_says_so(
    control_env: Path, capsys: pytest.CaptureFixture[str], no_sleep: None
) -> None:
    LocalIndex().record(IndexEntry(job_id=JOB, host=POD))
    for command in (["wait", JOB], ["cancel", JOB], ["logs", JOB]):
        assert main(command) == EXIT_ERROR, command
        captured = capsys.readouterr()
        assert "s3_bucket is unset" in captured.out + captured.err, command


def test_a_mirror_with_no_final_state_is_a_lost_job_not_a_guess(
    control_env: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    no_sleep: None,
) -> None:
    mirror(monkeypatch, {JOB: {"status": "running", "started_at": ago(hours=1)}})
    assert main(["wait", JOB]) == EXIT_ERROR
    assert "no final state" in capsys.readouterr().out
    assert main(["cancel", JOB]) == EXIT_ERROR
    assert "no final state" in capsys.readouterr().err


# -- status ---------------------------------------------------------------------


def test_status_of_a_named_gone_host_lists_its_jobs_from_the_mirror(
    control_env: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    mirror(monkeypatch, {JOB: succeeded(days=30)})
    assert main(["status", "--host", POD]) == EXIT_OK
    out = capsys.readouterr().out
    assert f"host {POD}: GONE" in out
    assert f"lego-s4 ({JOB}) succeeded" in out and "from the S3 mirror" in out

    assert main(["status", "--host", POD, "--json"]) == EXIT_OK
    host = json.loads(capsys.readouterr().out)["hosts"][0]
    assert host["state"] == "gone"
    assert [job["status"] for job in host["finished"]] == ["succeeded"]


def test_status_all_gives_a_gone_hosts_jobs_their_final_status(
    control_env: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    mirror(monkeypatch, {JOB: succeeded(days=3)})
    assert main(["status", "--all", "--json"]) == EXIT_OK
    [job] = json.loads(capsys.readouterr().out)["unhosted"]
    assert (job["host_state"], job["status"]) == ("gone", "succeeded")


def test_status_lists_a_terminated_rentals_jobs_from_the_mirror_and_forgets_it(
    control_env: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    mirror(monkeypatch, {JOB: succeeded(minutes=5)})
    with registry_transaction() as registry:
        registry.put(
            host_entry(
                name=POD, kind="rental", pod_id="pod-1", ssh="root@1.2.3.4", s3_prefix=PREFIX
            )
        )
    ended = Pod(id="pod-1", name=POD, status="TERMINATED", cost_usd_hr=0.0)
    monkeypatch.setattr(
        "gpuc.control.actions.make_provider", lambda *_a, **_k: FakeProvider(existing=[ended])
    )
    assert main(["status"]) == EXIT_OK
    captured = capsys.readouterr()
    assert f"host {POD} [rental]" in captured.out and "GONE" in captured.out
    assert f"({JOB}) succeeded" in captured.out
    assert f"forgetting host {POD}" in captured.err
    assert POD not in load_registry().hosts

    # Forgotten, it is the other kind of gone, and the answer is the same.
    assert main(["status", "--host", POD]) == EXIT_OK
    assert f"({JOB}) succeeded" in capsys.readouterr().out


def test_ssh_to_a_job_on_a_host_this_machine_does_not_have_is_still_no_such_host(
    control_env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A shell needs a host to open: there `--host` names a destination."""
    assert main(["ssh", "--host", "nope", "--print", JOB]) == 4
    assert "no host named 'nope'" in capsys.readouterr().err


# -- a typo is not a rental that ended ------------------------------------------


def test_a_typod_host_for_a_job_the_index_puts_elsewhere_is_no_such_host(
    control_env: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    no_sleep: None,
) -> None:
    """Otherwise the mirror's stale copy of a job still running on its real
    host would be printed as the answer, or the job reported lost."""
    mirror(monkeypatch, {JOB: {"status": "running", "started_at": ago(hours=1)}}, host="box")
    with registry_transaction() as registry:
        registry.put(host_entry(name="box", kind="ssh", ssh="me@box"))
    for command in ("logs", "wait", "cancel"):
        assert main([command, JOB, "--host", "boxx"]) == EXIT_NOT_FOUND, command
        captured = capsys.readouterr()
        said = captured.out + captured.err
        assert "no host named 'boxx'" in said and "on host box" in said, command


def test_status_of_a_name_nothing_knows_is_no_such_host(
    control_env: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    mirror(monkeypatch, {JOB: succeeded(minutes=5)})
    assert main(["status", "--host", "gpuc-pdo"]) == EXIT_NOT_FOUND
    assert "no host named 'gpuc-pdo'" in capsys.readouterr().err


def test_status_says_which_of_a_gone_hosts_jobs_went_with_it(
    control_env: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    mirror(monkeypatch, {JOB: {"status": "running", "started_at": ago(hours=1)}})
    assert main(["status", "--host", POD]) == EXIT_OK
    out = capsys.readouterr().out
    assert f"lost    1 job(s) ({JOB})" in out and "went with the host" in out
    assert main(["status", "--host", POD, "--json"]) == EXIT_OK
    host = json.loads(capsys.readouterr().out)["hosts"][0]
    assert host["lost"]["jobs"] == [JOB] and host["finished"] == []


def test_with_no_bucket_status_says_a_gone_hosts_jobs_are_lost(
    control_env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    LocalIndex().record(IndexEntry(job_id=JOB, host=POD))
    assert main(["status", "--host", POD]) == EXIT_OK
    assert "s3_bucket is unset" in capsys.readouterr().out
