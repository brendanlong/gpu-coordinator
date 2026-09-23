"""What `gpuc` promises automation: exit codes, `status --json`, and that one
bad host entry never takes the CLI with it.

The incident these come from: a registry another session could not validate
made every subcommand fail, and a non-zero `gpuc status` was read as "0 jobs
running". Both halves of that have to be impossible now.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from gpuc.control import status as status_mod
from gpuc.control.cli import (
    EXIT_ERROR,
    EXIT_INTERRUPTED,
    EXIT_LOCAL_STATE,
    EXIT_NOT_FOUND,
    EXIT_OK,
    EXIT_USAGE,
    main,
)
from gpuc.control.config import (
    HostEntry,
    backup_path,
    config_file,
    hosts_file,
    pod_known_hosts_file,
    read_registry,
)
from gpuc.control.s3index import S3IndexError
from gpuc.control.status import HostState, HostView
from tests.conftest import (
    FAKE_GPUS,
    accept_job,
    host_entry,
    install_fake_nvidia_smi,
    load_registry,
    register_host,
)
from tests.fakehost import FakeHost

GPU = FAKE_GPUS[0]

GOOD_ENTRY = {
    "name": "good",
    "kind": "local",
    "cache": {"config": {"host": "good", "gpus": [GPU], "idle_minutes": 15.0}},
}
BAD_ENTRY = {"name": "bad", "kind": "a kind that does not exist", "port": "twenty-two"}


def write_hosts(document: object) -> Path:
    path = hosts_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document))
    return path


# -- a broken registry must not brick the CLI ---------------------------------


def test_one_bad_host_entry_is_skipped_and_the_rest_still_work(
    control_env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_hosts({"hosts": {"good": GOOD_ENTRY, "bad": BAD_ENTRY}})
    read = read_registry()
    assert set(read.registry.hosts) == {"good"}
    assert read.unreadable is False
    assert "skipping host 'bad'" in read.errors[0]
    assert read.skipped["bad"] == BAD_ENTRY

    # Exit 1: one host is missing from the answer. The rest of it is still here.
    assert main(["host", "list"]) == EXIT_ERROR
    captured = capsys.readouterr()
    assert "good" in captured.out
    assert "skipping host 'bad'" in captured.err


LEGACY_RENTAL = {"name": "gpuc-old", "kind": "runpod", "ssh": "root@1.2.3.4", "pod_id": "podOLD"}
"""A rental as a build before `rental: {provider, pod_id}` wrote one."""


def test_a_rental_an_earlier_build_registered_is_refused_not_read_as_ssh(
    control_env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Read tolerantly it parses as an ssh host, and then nothing about it being
    a rental works: no terminate, no reuse, never forgotten when its pod ends.
    So it is one more entry this build cannot read -- skipped with the reason
    and the way back, written back untouched, exit 1 -- not a quiet ssh host."""
    write_hosts({"hosts": {"good": GOOD_ENTRY, "gpuc-old": LEGACY_RENTAL}})
    read = read_registry()
    assert set(read.registry.hosts) == {"good"}
    assert read.skipped["gpuc-old"] == LEGACY_RENTAL
    assert "earlier build" in read.errors[0]
    assert "gpuc host add <name> --pod podOLD" in read.errors[0]

    assert main(["host", "list"]) == EXIT_ERROR
    captured = capsys.readouterr()
    assert "good" in captured.out
    assert "gpuc host add <name> --pod podOLD" in captured.err
    # A rental spelled the one way this build spells it is still a rental.
    spelled = HostEntry.model_validate(
        {"name": "n", "rental": {"provider": "runpod", "pod_id": "p"}}
    )
    assert spelled.kind == "rental"


def test_a_write_puts_back_the_entry_this_build_could_not_read(
    control_env: Path, fake_host: FakeHost
) -> None:
    """It is another session's host, not ours to delete on the next `host set`."""
    write_hosts({"hosts": {"good": GOOD_ENTRY, "bad": BAD_ENTRY}})
    assert main(["host", "set", "good", "--idle-min", "9"]) == EXIT_OK
    document = json.loads(hosts_file().read_text())
    assert document["hosts"]["bad"] == BAD_ENTRY
    assert document["hosts"]["good"]["cache"]["config"]["idle_minutes"] == 9.0
    assert document["schema_version"] == 1
    assert fake_host.config is not None and fake_host.config["idle_minutes"] == 9.0


def test_an_unparseable_registry_says_so_takes_a_bak_and_exits_three(
    control_env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = hosts_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{ this is not json")
    assert main(["status"]) == EXIT_LOCAL_STATE
    err = capsys.readouterr().err
    assert str(path) in err
    assert ".bak" in err
    assert backup_path(path).read_text() == "{ this is not json"


def test_status_does_not_call_an_unreadable_registry_an_empty_one(
    control_env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_hosts(["not", "an", "object"])
    assert main(["status"]) == EXIT_LOCAL_STATE
    assert "this is not `no jobs running`" in capsys.readouterr().err


def test_a_mutation_refuses_to_overwrite_a_registry_it_could_not_read(
    control_env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = hosts_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{ this is not json")
    assert main(["host", "set", "local", "--idle-min", "9"]) == EXIT_LOCAL_STATE
    assert "not a readable host registry" in capsys.readouterr().err
    assert path.read_text() == "{ this is not json"


def test_a_job_command_on_an_unreadable_registry_is_unknown_not_missing(
    control_env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Exit 3, not 4: "no host knows that job" would be a lie."""
    path = hosts_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{ this is not json")
    assert main(["logs", "20260101-000000-aaaaaa"]) == EXIT_LOCAL_STATE
    assert main(["cancel", "20260101-000000-aaaaaa"]) == EXIT_LOCAL_STATE
    assert main(["host", "list"]) == EXIT_LOCAL_STATE
    assert "not a readable host registry" in capsys.readouterr().err


def test_a_missing_registry_is_simply_no_hosts(control_env: Path) -> None:
    assert read_registry().unreadable is False
    assert load_registry().hosts == {}
    assert main(["status"]) == EXIT_OK


# -- exit codes ---------------------------------------------------------------


def test_status_reports_an_unreachable_host_and_exits_one(
    control_env: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The host it could not read is the failure; the ones it could are the answer."""
    register_host(name="gpubox", kind="ssh", ssh="me@nowhere.invalid", gpus=GPU)
    register_host(name="local", gpus=GPU)

    def only_local(entry: HostEntry, *args: object, **kwargs: object) -> HostView:
        if entry.name == "local":
            return HostView(entry=entry, state=HostState.ANSWERED)
        return HostView(
            entry=entry, state=HostState.UNREACHABLE, error="ssh: could not resolve hostname"
        )

    monkeypatch.setattr(status_mod, "gather", only_local)
    assert main(["status"]) == EXIT_ERROR
    out = capsys.readouterr().out
    assert "UNREACHABLE" in out
    assert "host local" in out


def test_status_reports_a_bad_entry_per_host_and_exits_one(
    control_env: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    write_hosts({"hosts": {"good": GOOD_ENTRY, "bad": BAD_ENTRY}})
    monkeypatch.setattr(
        status_mod, "gather", lambda entry, *a, **k: HostView(entry=entry, state=HostState.ANSWERED)
    )
    assert main(["status"]) == EXIT_ERROR
    captured = capsys.readouterr()
    assert "host good" in captured.out
    assert "skipping host 'bad'" in captured.err


def test_a_bad_entry_does_not_fail_a_status_asked_about_another_host(
    control_env: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--host good` was never asked about the entry that would not parse."""
    write_hosts({"hosts": {"good": GOOD_ENTRY, "bad": BAD_ENTRY}})
    monkeypatch.setattr(
        status_mod, "gather", lambda entry, *a, **k: HostView(entry=entry, state=HostState.ANSWERED)
    )
    assert main(["status", "--host", "good"]) == EXIT_OK
    assert main(["status", "--host", "good", "--json"]) == EXIT_OK


def test_status_all_json_carries_the_jobs_only_the_index_knows(
    control_env: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The one list of what was on a host that lost its state, in the form a
    script reads: `--json` used to drop it while the text form printed it."""
    from gpuc.control.s3index import IndexEntry, LocalIndex

    register_host(name="gpubox", kind="ssh", ssh="me@box")
    monkeypatch.setattr(status_mod, "gather", _answered_with_no_jobs)
    LocalIndex().record(
        IndexEntry(
            job_id="20260101-000000-aaaaaa",
            host="gone-box",
            name="lost",
            requeued_from="20250101-000000-000000",
            submitted_at="2026-01-01T00:00:00+00:00",
        )
    )
    capsys.readouterr()
    assert main(["status", "--all", "--json"]) == EXIT_OK
    document = status_json(capsys)
    (entry,) = document["unhosted"]
    assert (entry["job_id"], entry["host"], entry["name"]) == (
        "20260101-000000-aaaaaa",
        "gone-box",
        "lost",
    )
    assert entry["requeued_from"] == "20250101-000000-000000"
    assert entry["outputs_lost"] is False
    assert main(["status", "--json"]) == EXIT_OK
    assert status_json(capsys)["unhosted"] == []


def _answered_with_no_jobs(entry: HostEntry, *a: object, **k: object) -> HostView:
    return HostView(entry=entry, state=HostState.ANSWERED, heartbeat_age_s=1.0)


def test_status_all_is_one_when_the_index_it_needs_cannot_be_read(
    control_env: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--all` exists to list what a lost host had, and a short list is not that."""
    register_host(name="local", gpus=GPU)
    monkeypatch.setattr(
        status_mod, "gather", lambda entry, *a, **k: HostView(entry=entry, state=HostState.ANSWERED)
    )

    class _Unreadable:
        def list_index(self) -> list[object]:
            raise S3IndexError("no credentials")

    monkeypatch.setattr(
        "gpuc.control.s3index.S3Index.from_settings", lambda settings: _Unreadable()
    )
    assert main(["status", "--all"]) == EXIT_ERROR
    assert "could not read the S3 index" in capsys.readouterr().err


def test_version_exits_one_for_an_entry_it_could_not_read(
    control_env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_hosts({"hosts": {"good": GOOD_ENTRY, "bad": BAD_ENTRY}})
    assert main(["version"]) == EXIT_ERROR
    assert main(["version", "--json"]) == EXIT_ERROR
    assert "skipping host 'bad'" in capsys.readouterr().err


def test_a_named_host_that_does_not_exist_is_four(
    control_env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["status", "--host", "nope"]) == EXIT_NOT_FOUND
    assert "no host named 'nope'" in capsys.readouterr().err


def test_a_bad_duration_is_usage(control_env: Path, capsys: pytest.CaptureFixture[str]) -> None:
    register_host(name="local", gpus=GPU)
    assert main(["status", "--since", "soon"]) == EXIT_USAGE
    assert "--since" in capsys.readouterr().err


def test_an_unreadable_config_file_is_three(
    control_env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (control_env / "config" / "config.toml").write_text("this is not = = toml\n")
    assert main(["status"]) == EXIT_LOCAL_STATE
    assert "not readable TOML" in capsys.readouterr().err


# -- status --json ------------------------------------------------------------


def status_json(capsys: pytest.CaptureFixture[str]) -> dict[str, Any]:
    document = json.loads(capsys.readouterr().out)
    assert isinstance(document, dict)
    return document


RUNNING_JOB = "20260915-120000-abc123"
FINISHED_JOB = "20260915-100000-def456"


@pytest.fixture
def real_local_host(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, control_env: Path) -> Path:
    """A registered `local` host whose GPUC_HOME really answers `gpuc.host status`.

    `gather()` shells out to the on-host package with `PYTHONPATH=<home>/pkg`,
    so the symlink is what makes this a real round trip rather than a fake: the
    JSON below is produced by `gpuc.host.__main__.cmd_status`, parsed by
    `gather`, and rendered by `status.document` with nothing stubbed.
    """
    from gpuc.host import jobs, paths
    from gpuc.host.jobs import HostConfig, JobSpec, JobState

    home = tmp_path / "host-home"
    monkeypatch.setenv("GPUC_HOME", str(home))
    # The host's own card, answered by the fake driver whatever this machine
    # has: the projection below holds only for a card nvidia-smi reports.
    install_fake_nvidia_smi(tmp_path / "bin", [GPU])
    monkeypatch.setenv("PATH", f"{tmp_path / 'bin'}{os.pathsep}{os.environ['PATH']}")
    paths.ensure_layout()
    jobs.write_config(HostConfig(host="local", gpus=[GPU]))
    paths.heartbeat_file().touch()

    started = (datetime.now(UTC) - timedelta(seconds=30)).isoformat()
    ended = (datetime.now(UTC) - timedelta(minutes=30)).isoformat()
    accept_job(
        JobSpec.from_dict(
            {"job_id": RUNNING_JOB, "name": "lego-s4", "command": "train", "gpus": 1}
        ),
        status="running",
        phase="main",
        gpus=[GPU],
        started_at=started,
        util_recent=[90.0, 95.0],
        isolation="cgroup",
    )
    accept_job(
        JobSpec.from_dict(
            {
                "job_id": FINISHED_JOB,
                "name": "probe",
                "command": "probe",
                # Declared but never confirmed uploaded: this is what makes the
                # host report `outputs_pending`, and only the host can know it.
                "outputs": [{"path": "results", "s3": "s3://bucket/{job_id}"}],
            }
        )
    )
    # With something actually under `results/`: a declared output that was
    # never written is not a result anyone can lose.
    (paths.workdir(FINISHED_JOB) / "results").mkdir(parents=True, exist_ok=True)
    (paths.workdir(FINISHED_JOB) / "results" / "loss.json").write_text("{}")
    jobs.write_state(
        FINISHED_JOB, JobState(status="failed", reason="timeout", ended_at=ended, exit_code=1)
    )
    monkeypatch.delenv("GPUC_HOME")

    (home / "pkg").symlink_to(Path(__file__).resolve().parents[1])
    entry = host_entry(
        name="local", kind="local", gpus=[GPU], gpuc_home=str(home), python=sys.executable
    )
    write_hosts({"hosts": {"local": json.loads(entry.model_dump_json())}})
    return home


def test_status_json_is_one_document_with_the_promised_shape(
    real_local_host: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["status", "--json"]) == EXIT_OK
    document = status_json(capsys)
    assert document["schema_version"] == 1
    assert document["errors"] == []
    host = document["hosts"][0]
    assert host["name"] == "local"
    assert host["kind"] == "local"
    assert host["reachable"] is True
    assert host["dispatcher"]["alive"] is True
    assert 0 <= host["dispatcher"]["heartbeat_age_s"] < 60
    assert host["queued"] == []

    job = next(j for j in host["running"] if j["job_id"] == RUNNING_JOB)
    assert set(job) == {
        "job_id",
        "name",
        "status",
        "reason",
        "problems",
        "upload_errors",
        "exit_code",
        "phase",
        "elapsed_s",
        "util",
        "progress_pct",
        "eta",
        "eta_s",
        "estimated_runtime_min",
        "auto_preempt",
        "progress_error",
        "gpus",
        "gpus_requested",
        "use_shared",
        "starts_in_s",
        "starts_at",
        "starts_unknown",
        "iso",
        "ended_at",
        "outputs_pending",
        "outputs_lost",
        "priority",
        "attempt",
        "requeued_from",
        "started_at",
        "workdir_bytes",
        "outputs",
        "links",
    }
    assert (job["name"], job["status"], job["phase"]) == ("lego-s4", "running", "main")
    assert job["util"] == 95.0
    assert job["gpus"] == [GPU]
    assert job["iso"] == "cgroup"
    assert job["elapsed_s"] > 0
    assert (job["outputs"], job["links"]) == ([], [])

    done = next(j for j in host["finished"] if j["job_id"] == FINISHED_JOB)
    assert done["reason"] == "timeout"
    assert done["outputs_pending"] is True
    # Where the results were meant to go, and a console link to open it: the
    # dashboard's anchors come from here, and the text view has no room for them.
    assert done["outputs"] == [
        {
            "path": "results",
            "s3": "s3://bucket/{job_id}",
            "hf": None,
            "hf_path": None,
            "hf_create": False,
        }
    ]
    (link,) = done["links"]
    assert (link["kind"], link["path"], link["target"]) == ("s3", "results", "s3://bucket/{job_id}")
    assert link["url"].startswith("https://s3.console.aws.amazon.com/s3/buckets/bucket?prefix=")
    assert host["pod"] is None
    assert (host["draining"], host["pod_gone"]) == (False, False)

    # One row for the owned card. Which shape it takes says whether nvidia-smi
    # on *this* machine could resolve it, which is not what this test is about.
    (row,) = host["gpus"]
    if row.get("available") is False:
        assert row == {"owned_as": GPU, "available": False}
    else:
        assert set(row) == {"index", "uuid", "name", "vram_mib", "busy_job"}
        assert row["uuid"] == GPU
        assert row["busy_job"] == RUNNING_JOB


def test_config_show_json_is_the_effective_settings(
    control_env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_file().write_text('s3_bucket = "bucket"\ndisk_gb = 5\n')
    assert main(["config", "show", "--json"]) == EXIT_OK
    document = status_json(capsys)
    assert document["schema_version"] == 1
    assert document["config_file"] == str(config_file())
    assert document["config_file_exists"] is True
    assert document["settings"]["s3_bucket"] == "bucket"
    assert document["settings"]["disk_gb"] == 5
    assert document["notes"] == []


def test_status_json_says_unreachable_rather_than_empty(
    control_env: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    register_host(name="gpubox", kind="ssh", ssh="me@box", gpus=GPU)
    capsys.readouterr()
    monkeypatch.setattr(
        status_mod,
        "gather",
        lambda entry, *a, **k: HostView(
            entry=entry, state=HostState.UNREACHABLE, error="ssh timed out"
        ),
    )
    assert main(["status", "--json"]) == EXIT_ERROR
    host = status_json(capsys)["hosts"][0]
    assert host["reachable"] is False
    assert host["running"] == []
    assert host["errors"] == ["ssh timed out"]


def test_status_json_on_an_unreadable_registry_is_exit_three_and_still_json(
    control_env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = hosts_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{ not json")
    assert main(["status", "--json"]) == EXIT_LOCAL_STATE
    document = status_json(capsys)
    assert document["hosts"] == []
    assert "could not be read" in document["errors"][-1]


def test_status_json_carries_the_skipped_entry_as_a_top_level_error(
    control_env: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    write_hosts({"hosts": {"good": GOOD_ENTRY, "bad": BAD_ENTRY}})
    monkeypatch.setattr(
        status_mod, "gather", lambda entry, *a, **k: HostView(entry=entry, state=HostState.ANSWERED)
    )
    assert main(["status", "--json"]) == EXIT_ERROR
    document = status_json(capsys)
    assert [host["name"] for host in document["hosts"]] == ["good"]
    assert "skipping host 'bad'" in document["errors"][0]


def test_every_option_says_what_it_does() -> None:
    """`--help` is the only documentation most of these flags will ever get.

    A bare `--gpu-count N` tells a reader nothing about whether it is the pod's
    GPUs or the job's.
    """
    import argparse

    from gpuc.control.cli import build_parser

    def walk(parser: argparse.ArgumentParser, path: str) -> list[str]:
        missing: list[str] = []
        for action in parser._actions:
            if isinstance(action, argparse._SubParsersAction):
                for name, sub in action.choices.items():
                    missing += walk(sub, f"{path} {name}")
                continue
            if action.option_strings and not action.help:
                missing.append(f"{path} {action.option_strings[0]}")
        return missing

    assert walk(build_parser(), "gpuc") == []


def test_ssh_help_says_where_it_lands_and_whose_exit_code_it_returns() -> None:
    from gpuc.control.cli import build_parser

    ssh = build_parser()._subparsers._group_actions[0].choices["ssh"]  # type: ignore[union-attr]
    text = ssh.format_help()
    assert "workdir" in text and "job dir" in text
    assert "exit code" in text


# -- version ------------------------------------------------------------------


def test_version_prints_the_version_the_commit_and_each_host(
    control_env: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from gpuc.control import version as version_mod

    register_host(
        name="gpubox",
        kind="ssh",
        ssh="me@box",
        pkg_commit="b" * 40,
        bootstrapped_at="2026-09-15T20:00:00+00:00",
    )
    capsys.readouterr()
    monkeypatch.setattr(version_mod, "local_commit", lambda: "a" * 40)
    monkeypatch.setattr(version_mod, "installed_commit", lambda: "a" * 40)
    assert main(["version"]) == EXIT_OK
    out = capsys.readouterr().out
    assert version_mod.__version__ in out
    assert "a" * 12 in out
    assert "b" * 12 in out
    assert "DIFFERS" in out
    assert "gpuc host bootstrap <host>" in out


def test_the_commit_status_judges_is_the_hosts_own_not_the_registrys(
    control_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The registry records what *this* machine shipped. A second control
    machine (a laptop on another build) makes that record describe a host it no
    longer matches, so the warning has to come from the host's answer."""
    from gpuc.control import version as version_mod

    monkeypatch.setattr(version_mod, "local_commit", lambda: "a" * 40)
    # Registry agrees with this build; the host says otherwise, and wins.
    entry = host_entry(name="gpubox", kind="ssh", ssh="me@box", pkg_commit="a" * 40)
    view = HostView(entry=entry, state=HostState.ANSWERED, pkg_commit="b" * 40)
    warnings = status_mod.host_warnings(view)
    assert len(warnings) == 1
    assert "host gpubox is running gpuc " + "b" * 12 in warnings[0]
    assert "gpuc host bootstrap gpubox" in warnings[0]
    assert status_mod.render(view).count("WARNING") == 1
    assert status_mod.host_json(view)["pkg_commit"] == "b" * 40
    # A warning, apart from the errors that decide the exit code.
    assert status_mod.host_json(view)["warnings"] == warnings
    assert status_mod.host_json(view)["errors"] == []


def test_a_host_running_this_build_or_one_we_could_not_ask_says_nothing(
    control_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gpuc.control import version as version_mod

    monkeypatch.setattr(version_mod, "local_commit", lambda: "a" * 40)
    entry = host_entry(name="s", pkg_commit="b" * 40)
    current = HostView(entry=entry, state=HostState.ANSWERED, pkg_commit="a" * 40)
    assert status_mod.host_warnings(current) == []
    # Unreachable: "we could not ask" is not evidence of anything.
    assert status_mod.host_warnings(HostView(entry=entry, pkg_commit="b" * 40)) == []


def test_host_list_reports_this_machines_own_record_and_says_that_is_what_it_is(
    control_env: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """`host list` never asks a host anything, so the one thing it must not do
    is imply it did: what it has is the commit *this* machine last shipped."""
    from gpuc.control import version as version_mod

    register_host(
        name="gpubox",
        kind="ssh",
        ssh="me@box",
        pkg_commit="b" * 40,
        bootstrapped_at="2026-09-15T20:00:00+00:00",
    )
    monkeypatch.setattr(version_mod, "local_commit", lambda: "a" * 40)
    capsys.readouterr()

    assert main(["host", "list"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "pkg     " + "b" * 12 + " on the host, as of " in out
    assert "NOTE host gpubox was last seen running gpuc " + "b" * 12 in out
    assert "gpuc status" in out

    assert main(["host", "list", "--json"]) == EXIT_OK
    (host,) = json.loads(capsys.readouterr().out)["hosts"]
    assert host["warnings"] == [
        version_mod.shipped_commit_note("gpubox", "b" * 40, "a" * 40),
    ]

    # On this build it has nothing to say, and says nothing.
    monkeypatch.setattr(version_mod, "local_commit", lambda: "b" * 40)
    assert main(["host", "list", "--json"]) == EXIT_OK
    (host,) = json.loads(capsys.readouterr().out)["hosts"]
    assert host["warnings"] == []


def test_a_host_too_old_to_say_which_build_it_runs_is_still_warned_about(
    control_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A *reachable* host that answered without a commit is a host on a build
    from before `status` reported one -- which is the oldest code of all, and
    exactly what `submit` re-ships on every run. Staying quiet about it would
    have `status` calling those hosts current while every submit disagreed."""
    from gpuc.control import version as version_mod

    monkeypatch.setattr(version_mod, "local_commit", lambda: "a" * 40)
    entry = host_entry(name="gpubox", kind="ssh", ssh="me@box", pkg_commit="a" * 40)
    (warning,) = status_mod.host_warnings(HostView(entry=entry, state=HostState.ANSWERED))
    assert "a build that named no commit" in warning
    assert "gpuc host bootstrap gpubox" in warning
    # With nothing to compare against: a gpuc that cannot name its own
    # commit has no business telling a host it is behind.
    monkeypatch.setattr(version_mod, "local_commit", lambda: None)
    assert status_mod.host_warnings(HostView(entry=entry, state=HostState.ANSWERED)) == []


def test_a_dispatcher_older_than_the_package_it_dispatches_is_a_warning(
    control_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The host has this build on disk and a dispatcher from before it still
    serving the queue, which is the one build mismatch nothing here can see
    from `pkg_commit` alone -- and the one that had a `use_shared` job waiting
    on two idle cards the running code did not know the host had."""
    from gpuc.control import version as version_mod

    monkeypatch.setattr(version_mod, "local_commit", lambda: "a" * 40)
    entry = host_entry(name="gpubox", kind="ssh", ssh="me@box", pkg_commit="a" * 40)
    view = HostView(
        entry=entry,
        state=HostState.ANSWERED,
        pkg_commit="a" * 40,
        dispatcher_pkg_commit="b" * 40,
        heartbeat_age_s=2.0,
    )
    (warning,) = status_mod.host_warnings(view)
    assert "running dispatcher was started on gpuc " + "b" * 12 in warning
    assert "gpuc host bootstrap gpubox" in warning
    assert status_mod.host_json(view)["dispatcher"]["pkg_commit"] == "b" * 40

    # The same build, and a host whose dispatcher is not running at all: the
    # second has nothing to be behind.
    same = replace(view, dispatcher_pkg_commit="a" * 40)
    assert status_mod.host_warnings(same) == []
    assert status_mod.host_warnings(replace(view, heartbeat_age_s=None)) == []


def test_a_host_behind_on_its_package_is_told_that_once_not_twice(
    control_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A host whose *package* is behind is already being told to re-bootstrap,
    and which build its dispatcher happens to be on does not change the fix."""
    from gpuc.control import version as version_mod

    monkeypatch.setattr(version_mod, "local_commit", lambda: "a" * 40)
    entry = host_entry(name="gpubox", kind="ssh", ssh="me@box", pkg_commit="a" * 40)
    view = HostView(entry=entry, state=HostState.ANSWERED, pkg_commit="b" * 40, heartbeat_age_s=2.0)
    assert len(status_mod.host_warnings(view)) == 1


def test_a_config_this_machine_has_not_caught_up_with_is_not_a_warning(
    control_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The host owns its config, so a cache that disagrees with it is stale and
    not a conflict -- and the cards printed above are the host's own answer."""
    from gpuc.control import version as version_mod

    monkeypatch.setattr(version_mod, "local_commit", lambda: "a" * 40)
    entry = host_entry(
        name="gpubox", kind="ssh", ssh="me@box", gpus=["2", "3"], pkg_commit="a" * 40
    )
    view = HostView(entry=entry, state=HostState.ANSWERED, pkg_commit="a" * 40, owned=["0", "1"])
    assert status_mod.host_warnings(view) == []


def test_the_installed_commit_comes_from_direct_url_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from importlib.metadata import PathDistribution

    from gpuc.control import version as version_mod

    dist_info = tmp_path / "gpu_coordinator-0.1.0.dist-info"
    dist_info.mkdir()
    (dist_info / "direct_url.json").write_text(
        json.dumps(
            {
                "url": "https://github.com/brendanlong/gpu-coordinator",
                "vcs_info": {"vcs": "git", "commit_id": "c" * 40},
            }
        )
    )
    monkeypatch.setattr(
        version_mod.Distribution, "from_name", staticmethod(lambda _: PathDistribution(dist_info))
    )
    assert version_mod.installed_commit() == "c" * 40


# -- `--json` on every command -------------------------------------------------


def document_of(capsys: pytest.CaptureFixture[str]) -> dict[str, Any]:
    """stdout must be exactly one JSON object, whatever else the command said."""
    document = json.loads(capsys.readouterr().out)
    assert isinstance(document, dict)
    assert document["schema_version"] == 1
    return document


JSON_COMMANDS = [
    ["status"],
    ["submit"],
    ["requeue"],
    ["logs"],
    ["wait"],
    ["cancel"],
    ["preempt"],
    ["reorder"],
    ["estimate"],
    ["pods"],
    ["version"],
    ["clean"],
    ["host", "list"],
    ["host", "probe"],
    ["host", "add"],
    ["host", "set"],
    ["host", "bootstrap"],
    ["host", "clean"],
    ["host", "remove"],
    ["host", "terminate"],
    ["config", "init"],
]


@pytest.mark.parametrize("command", JSON_COMMANDS, ids=lambda c: " ".join(c))
def test_every_command_that_promises_json_has_the_flag(command: list[str]) -> None:
    import argparse

    from gpuc.control.cli import build_parser

    parser: argparse.ArgumentParser = build_parser()
    for name in command:
        action = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
        parser = action.choices[name]
    assert "--json" in parser.format_help()


def test_a_failure_under_json_is_a_document_and_the_same_exit_code(
    control_env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A caller parsing stdout must never be handed nothing at all."""
    assert main(["cancel", "20260101-000000-aaaaaa", "--json"]) == EXIT_NOT_FOUND
    captured = capsys.readouterr()
    document = json.loads(captured.out)
    assert document["exit_code"] == EXIT_NOT_FOUND
    assert "no registered host knows job" in document["error"]
    assert "no registered host knows job" in captured.err


def test_a_ctrl_c_is_exit_130_and_a_document_whoever_was_running(
    control_env: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The boundary in `main`, not the blocking commands' own handling.

    `status` has no idea about interrupts, which is the point: a command that
    blocks cannot forget to be 130, and cannot leave `--json` with the empty
    stdout the flag promises never to give.
    """
    from gpuc.control import cli

    def interrupted(*_args: Any, **_kwargs: Any) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "open_registry", interrupted)
    assert main(["status"]) == 130
    assert "interrupted" in capsys.readouterr().err
    assert main(["status", "--json"]) == 130
    document = document_of(capsys)
    assert (document["exit_code"], document["error"]) == (130, "interrupted")


def test_a_usage_error_under_json_is_a_document_too(
    control_env: Path, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    job = tmp_path / "job.yaml"
    job.write_text('command: "true"\n')
    assert main(["submit", str(job), "--json"]) == EXIT_USAGE
    assert json.loads(capsys.readouterr().out)["exit_code"] == EXIT_USAGE


def test_a_command_line_argparse_rejects_still_prints_a_document(
    control_env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """argparse exits before any command runs; stdout must not be empty."""
    with pytest.raises(SystemExit) as exit_info:
        main(["reorder", "20260101-000000-aaaaaa", "--priority", "soon", "--json"])
    assert exit_info.value.code == EXIT_USAGE
    captured = capsys.readouterr()
    assert json.loads(captured.out)["exit_code"] == EXIT_USAGE
    assert "--priority" in captured.err


def test_help_under_json_is_not_an_error_document(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["status", "--json", "--help"])
    assert exit_info.value.code == EXIT_OK
    assert "error" not in capsys.readouterr().out


def test_host_list_json_carries_the_entries_and_the_skipped_ones(
    control_env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_hosts({"hosts": {"good": GOOD_ENTRY, "bad": BAD_ENTRY}})
    assert main(["host", "list", "--json"]) == EXIT_ERROR
    document = document_of(capsys)
    (host,) = document["hosts"]
    assert host["name"] == "good"
    assert host["kind"] == "local"
    assert host["gpus"] == [GPU]
    assert host["remote_home"] == "$HOME/.gpuc"
    assert host["ephemeral"] is False
    assert "skipping host 'bad'" in document["errors"][0]


def test_host_add_json_is_the_host_as_list_reports_it_plus_what_add_did(
    control_env: Path, fake_host: FakeHost, capsys: pytest.CaptureFixture[str]
) -> None:
    """The registry entry `host add` just wrote, in `host list --json`'s shape."""
    assert main(["host", "add", "gpubox", "--ssh", "me@box", "--json"]) == EXIT_OK
    document = document_of(capsys)
    assert (document["name"], document["kind"], document["ssh"]) == ("gpubox", "ssh", "me@box")
    assert document["gpus"] == ["GPU-a", "GPU-b"]
    assert document["adopted"] is False
    assert document["config_path"] == fake_host.config_path
    assert document["warnings"] == []
    assert isinstance(document["changes"], list)
    assert fake_host.config is not None


def test_host_add_json_carries_the_owns_nothing_warning_in_the_document(
    control_env: Path, fake_host: FakeHost, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["host", "add", "none", "--ssh", "me@none", "--gpus", "", "--json"]) == EXIT_OK
    document = document_of(capsys)
    assert document["gpus"] == []
    assert any("owns no GPUs (--gpus '' asked for none)" in w for w in document["warnings"])


def test_host_add_json_failure_is_a_document_with_the_text_forms_exit_code(
    control_env: Path, fake_host: FakeHost, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = ["host", "add", "gpubox", "--ssh", "me@box", "--gpus", "GPU-a", "--shared-gpus", "GPU-a"]
    code = main(argv)
    captured = capsys.readouterr()
    assert code == 1 and captured.out == ""
    assert main([*argv, "--json"]) == code
    assert document_of(capsys)["exit_code"] == code


def test_host_set_json_is_the_host_after_the_change(
    control_env: Path, fake_host: FakeHost, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["host", "add", "gpubox", "--ssh", "me@box", "--gpus", "0"]) == EXIT_OK
    capsys.readouterr()
    assert main(["host", "set", "gpubox", "--shared-gpus", "1", "--json"]) == EXIT_OK
    document = document_of(capsys)
    assert document["name"] == "gpubox"
    assert document["shared_gpus"] == ["1"]
    assert document["adopted"] is True
    assert document["changes"] and any("shared_gpus" in change for change in document["changes"])
    assert document["address"] == {}
    assert document["warnings"] == []
    assert load_registry().require("gpubox").config.shared_gpus == ["1"]


def test_host_set_json_reports_an_address_change_made_here(
    control_env: Path, fake_host: FakeHost, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["host", "add", "gpubox", "--ssh", "me@box"]) == EXIT_OK
    capsys.readouterr()
    assert main(["host", "set", "gpubox", "--persistent-root", "/vol/me", "--json"]) == EXIT_OK
    document = document_of(capsys)
    assert document["address"] == {"persistent_root": "/vol/me"}
    assert document["changes"] == []
    assert document["persistent_root"] == "/vol/me"
    assert document["remote_home"] == "/vol/me/gpuc"


def test_host_set_json_with_nothing_to_set_is_usage_in_both_forms(
    control_env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    register_host(name="local", gpus=GPU)
    assert main(["host", "set", "local"]) == EXIT_USAGE
    capsys.readouterr()
    assert main(["host", "set", "local", "--json"]) == EXIT_USAGE
    assert document_of(capsys)["exit_code"] == EXIT_USAGE


def test_host_remove_json_says_what_was_forgotten_and_that_a_pod_is_not_touched(
    control_env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    register_host(name="pod", kind="rental", ssh="root@1.2.3.4", pod_id="p1")
    register_host(name="local", gpus=GPU)
    pinned = pod_known_hosts_file("pod")
    pinned.parent.mkdir(parents=True, exist_ok=True)
    pinned.write_text("[1.2.3.4]:22 ssh-ed25519 AAAA\n")
    capsys.readouterr()
    assert main(["host", "remove", "pod", "--json"]) == EXIT_OK
    document = document_of(capsys)
    assert (document["host"], document["kind"], document["pod_id"]) == ("pod", "rental", "p1")
    assert any("p1" in text and "not terminated" in text for text in document["notes"])
    # RunPod recycles host:port, so the next pod under this name must not be
    # checked against this one's key.
    assert not pinned.exists()
    assert main(["host", "remove", "local", "--json"]) == EXIT_OK
    document = document_of(capsys)
    assert (document["host"], document["kind"], document["pod_id"]) == ("local", "local", None)
    assert document["notes"] == []
    assert load_registry().hosts == {}
    assert main(["host", "remove", "local"]) == EXIT_NOT_FOUND
    capsys.readouterr()
    assert main(["host", "remove", "local", "--json"]) == EXIT_NOT_FOUND
    assert document_of(capsys)["exit_code"] == EXIT_NOT_FOUND


def test_host_terminate_json_is_what_was_ended_and_what_it_was_doing(
    control_env: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.fakeprovider import FakeProvider, PodScript, running_pod

    monkeypatch.setenv("RUNPOD_API_KEY", "test-key")
    provider = FakeProvider()
    provider.adopt(running_pod("gpuc-e2e-aaa", "pod1"), PodScript(ssh_after_polls=0))
    monkeypatch.setattr("gpuc.control.cli.make_provider", lambda *a, **k: provider)
    register_host(name="gpuc-e2e-aaa", kind="rental", ssh="root@1.2.3.4", pod_id="pod1")
    capsys.readouterr()

    assert main(["host", "terminate", "gpuc-e2e-aaa", "--force", "--json"]) == EXIT_OK

    document = document_of(capsys)
    assert (document["host"], document["pod_id"]) == ("gpuc-e2e-aaa", "pod1")
    assert (document["terminated"], document["forgotten"], document["checked"]) == (
        True,
        True,
        False,
    )
    assert document["cost_usd_hr"] == 0.49
    assert provider.terminated == ["pod1"]
    assert load_registry().hosts == {}
    assert main(["host", "terminate", "nothing-like-this", "--json"]) == EXIT_NOT_FOUND
    assert document_of(capsys)["exit_code"] == EXIT_NOT_FOUND


def test_host_bootstrap_json_is_what_the_bootstrap_left_on_the_host(
    control_env: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.test_cli import bootstrapping

    register_host(name="local", gpus=GPU)
    bootstrapping(monkeypatch)
    capsys.readouterr()
    assert main(["host", "bootstrap", "local", "--json"]) == EXIT_OK
    captured = capsys.readouterr()
    document = json.loads(captured.out)
    assert document == {
        "schema_version": 1,
        "host": "local",
        "home": "/root/.gpuc",
        "files": 20,
        "pkg_commit": None,
        "dispatcher_pid": 4242,
        "warnings": [],
    }
    # The step-by-step progress bootstrap prints is on stderr, off the document.
    assert "fake bootstrap of local" in captured.err
    assert load_registry().require("local").bootstrapped_at


def test_host_bootstrap_all_json_is_the_tally_per_host_as_data(
    control_env: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from gpuc.control.bootstrap import BootstrapError
    from tests.test_cli import bootstrapping

    register_host(name="gpubox", kind="ssh", ssh="me@box")
    register_host(name="pod", kind="rental", ssh="root@1.2.3.4", pod_id="p1")
    document = json.loads(hosts_file().read_text())
    document["hosts"]["bad"] = BAD_ENTRY
    hosts_file().write_text(json.dumps(document))
    bootstrapping(monkeypatch, fail={"pod": BootstrapError("ssh to pod failed")})
    capsys.readouterr()

    # Exit 1 because a host failed, exactly as without --json.
    assert main(["host", "bootstrap", "--all", "--json"]) == 1
    captured = capsys.readouterr()
    tally = json.loads(captured.out)
    assert tally["schema_version"] == 1
    assert (tally["total"], tally["bootstrapped"], tally["failed"]) == (2, ["gpubox"], ["pod"])
    assert tally["unreadable"] == ["bad"]
    assert tally["interrupted"] is False
    assert "skipping host 'bad'" in tally["errors"][0]
    by_name = {host["name"]: host for host in tally["hosts"]}
    assert set(by_name) == {"gpubox", "pod"}
    assert by_name["gpubox"]["outcome"] == "bootstrapped"
    assert by_name["gpubox"]["error"] is None
    assert by_name["gpubox"]["dispatcher_pid"] == 4242
    assert by_name["pod"]["outcome"] == "failed"
    assert by_name["pod"]["error"] == "ssh to pod failed"
    assert by_name["pod"]["ephemeral"] is True
    assert by_name["pod"]["dispatcher_pid"] is None
    assert "== gpubox (1/2) ==" in captured.err
    assert "warning: host pod: ssh to pod failed" in captured.err


def test_host_bootstrap_all_json_interrupted_names_the_hosts_never_reached(
    control_env: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.test_cli import bootstrapping

    register_host(name="abox", kind="ssh", ssh="me@a")
    register_host(name="bbox", kind="ssh", ssh="me@b")
    register_host(name="cbox", kind="ssh", ssh="me@c")
    bootstrapping(monkeypatch, fail={"bbox": KeyboardInterrupt()})
    capsys.readouterr()
    # A Ctrl-C is 130 like everywhere else, and the error document still
    # carries the tally: what got through is true, and what did not is named.
    assert main(["host", "bootstrap", "--all", "--json"]) == EXIT_INTERRUPTED
    tally = document_of(capsys)
    assert tally["interrupted"] is True
    assert "interrupted during bbox" in tally["error"]
    assert [(h["name"], h["outcome"]) for h in tally["hosts"]] == [
        ("abox", "bootstrapped"),
        ("bbox", "interrupted"),
        ("cbox", "not_attempted"),
    ]


def test_host_bootstrap_all_json_with_no_hosts_is_an_empty_tally(
    control_env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["host", "bootstrap", "--all", "--json"]) == EXIT_OK
    tally = document_of(capsys)
    assert (tally["hosts"], tally["total"]) == ([], 0)


def test_host_bootstrap_all_json_on_a_registry_that_breaks_mid_run_is_the_error_document(
    control_env: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from gpuc.control.config import LocalStateUnreadable
    from tests.test_cli import bootstrapping

    register_host(name="gpubox", kind="ssh", ssh="me@box")
    register_host(name="local", gpus=GPU)
    bootstrapping(monkeypatch, fail={"local": LocalStateUnreadable("hosts.json is not json")})
    capsys.readouterr()
    assert main(["host", "bootstrap", "--all", "--json"]) == EXIT_LOCAL_STATE
    captured = capsys.readouterr()
    document = json.loads(captured.out)
    assert document["exit_code"] == EXIT_LOCAL_STATE
    assert "hosts.json is not json" in document["error"]
    assert "1/2 host(s) bootstrapped" in captured.err


class PruningTransport:
    """A host whose `uv cache prune` reports the cache size either side."""

    def __init__(self, output: str) -> None:
        self.output = output

    def run(self, command: str, *, timeout: float = 120.0, check: bool = True) -> Any:
        from gpuc.control.transport import CommandResult

        return CommandResult("fake", ["sh", "-c", command], 0, self.output, "")


def test_host_clean_json_is_the_cache_and_what_the_prune_freed(
    control_env: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    register_host(name="local", gpus=GPU, python="/py")
    host = PruningTransport("before_kib=18874368\nafter_kib=11534336\ndir=/home/u/.cache/uv\n")
    monkeypatch.setattr("gpuc.control.remote.transport_for", lambda entry, settings=None: host)
    assert main(["host", "clean", "local", "--uv-cache"]) == EXIT_OK
    assert "pruned 18.0 GiB -> 11.0 GiB" in capsys.readouterr().out
    assert main(["host", "clean", "local", "--uv-cache", "--json"]) == EXIT_OK
    document = document_of(capsys)
    assert document == {
        "schema_version": 1,
        "host": "local",
        "cache_dir": "/home/u/.cache/uv",
        "before": "18.0 GiB",
        "after": "11.0 GiB",
        "before_bytes": 18874368 * 1024,
        "after_bytes": 11534336 * 1024,
        "freed_bytes": (18874368 - 11534336) * 1024,
    }


def test_host_clean_json_without_the_flag_is_usage_in_both_forms(
    control_env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    register_host(name="local", gpus=GPU)
    assert main(["host", "clean", "local"]) == EXIT_USAGE
    capsys.readouterr()
    assert main(["host", "clean", "local", "--json"]) == EXIT_USAGE
    assert document_of(capsys)["exit_code"] == EXIT_USAGE


def test_config_init_json_is_the_path_and_whether_it_was_there(
    control_env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["config", "init", "--json"]) == EXIT_OK
    document = document_of(capsys)
    assert document == {"schema_version": 1, "config_file": str(config_file()), "existed": False}
    assert config_file().exists()
    # Refusing to clobber is exit 1 in both forms, with the reason as `error`.
    assert main(["config", "init"]) == 1
    capsys.readouterr()
    assert main(["config", "init", "--json"]) == 1
    document = document_of(capsys)
    assert document["exit_code"] == 1
    assert "already exists" in document["error"]
    assert main(["config", "init", "--force", "--json"]) == EXIT_OK
    assert document_of(capsys)["existed"] is True


def test_host_list_json_on_an_unreadable_registry_is_exit_three_and_still_json(
    control_env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = hosts_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{ not json")
    assert main(["host", "list", "--json"]) == EXIT_LOCAL_STATE
    document = document_of(capsys)
    assert document["hosts"] == []
    assert document["errors"]


def test_version_json_says_which_hosts_are_current(
    control_env: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from gpuc.control import version as version_mod

    entries = {
        name: json.loads(
            host_entry(
                name=name,
                kind="ssh",
                ssh="me@box",
                pkg_commit=commit,
                bootstrapped_at="2026-09-15T20:00:00+00:00",
            ).model_dump_json()
        )
        for name, commit in (("fresh", "a" * 40), ("stale", "b" * 40))
    }
    write_hosts({"hosts": entries})
    monkeypatch.setattr(version_mod, "local_commit", lambda: "a" * 40)
    monkeypatch.setattr(version_mod, "installed_commit", lambda: "a" * 40)
    assert main(["version", "--json"]) == EXIT_OK
    document = document_of(capsys)
    assert document["version"] == version_mod.__version__
    assert document["commit"] == "a" * 40
    assert document["source"] == "installed"
    assert {host["name"]: host["current"] for host in document["hosts"]} == {
        "fresh": True,
        "stale": False,
    }


def test_logs_json_carries_the_lines_and_where_they_came_from(
    real_local_host: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    log = real_local_host / "jobs" / RUNNING_JOB / "log.txt"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text("first\nsecond\n")
    assert main(["logs", RUNNING_JOB, "--json"]) == EXIT_OK
    document = document_of(capsys)
    assert document["job_id"] == RUNNING_JOB
    assert document["host"] == "local"
    assert document["source"] == "host"
    assert document["location"] == str(log)
    assert document["lines"] == ["first", "second"]
    assert document["notes"] == []


def test_logs_json_refuses_to_follow(
    real_local_host: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`-f` has no end, and a document has to be complete."""
    assert main(["logs", RUNNING_JOB, "--json", "-f"]) == EXIT_USAGE
    assert "cannot follow" in json.loads(capsys.readouterr().out)["error"]


def test_logs_json_says_when_it_fell_back_to_the_mirror(
    real_local_host: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.fakes3 import FakeS3Client

    (config_file()).write_text('s3_bucket = "bucket"\n')
    uri = f"bucket/gpuc/local/jobs/{RUNNING_JOB}/log.txt"
    monkeypatch.setattr(
        "gpuc.control.s3index.S3Index.client",
        property(lambda self: FakeS3Client(objects={uri: b"mirrored\n"})),
    )
    entry = load_registry().require("local")
    entry = entry.with_config({**entry.cache.config, "s3_prefix": "s3://bucket/gpuc/local"})
    write_hosts({"hosts": {"local": json.loads(entry.model_dump_json())}})
    # The host lost the log; the mirror still has it.
    (real_local_host / "jobs" / RUNNING_JOB / "log.txt").unlink()

    assert main(["logs", RUNNING_JOB, "--json"]) == EXIT_OK
    document = document_of(capsys)
    assert document["source"] == "s3"
    assert document["location"] == f"s3://{uri}"
    assert document["lines"] == ["mirrored"]
    assert any("could not read" in note for note in document["notes"])


def test_status_json_says_what_order_the_queue_runs_in(
    real_local_host: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The whole round trip for dispatch order: the host's queue markers and
    specs, through `gather`, into the document automation is told to prefer.

    Priority is the one field that explains the order, so a queue you cannot
    sort by it is a list you cannot act on.
    """
    from gpuc.host import jobs, paths, queue
    from gpuc.host.jobs import JobSpec

    monkeypatch.setenv("GPUC_HOME", str(real_local_host))
    paths.ensure_layout()
    # The card is held for another hour, which is what gives the jobs waiting
    # for it a start time at all.
    jobs.update_state(RUNNING_JOB, eta=(datetime.now(UTC) + timedelta(hours=1)).isoformat())
    for job_id, name, priority, gpus, use_shared in (
        ("20260915-140000-bbbbbb", "sweep", 90, 1, False),
        ("20260915-140100-cccccc", "urgent", 10, 2, True),
    ):
        queue.enqueue(
            JobSpec.from_dict(
                {
                    "job_id": job_id,
                    "name": name,
                    "command": "train",
                    "priority": priority,
                    "gpus": gpus,
                    "use_shared": use_shared,
                    "estimated_runtime_min": 30.0,
                }
            )
        )
    monkeypatch.delenv("GPUC_HOME")

    assert main(["status", "--json"]) == EXIT_OK
    queued = status_json(capsys)["hosts"][0]["queued"]
    assert [(job["name"], job["priority"]) for job in queued] == [("urgent", 10), ("sweep", 90)]
    assert queued == sorted(queued, key=lambda job: job["priority"])
    assert (queued[0]["gpus_requested"], queued[1]["gpus_requested"]) == (2, 1)
    # Which of them may be dispatched onto a card the host only borrows: the
    # spec's own answer, and the other half of explaining this queue.
    assert (queued[0]["use_shared"], queued[1]["use_shared"]) == (True, False)
    # `urgent` wants two cards and the host owns one, so it never fits; `sweep`
    # takes the card the running job gives back in an hour.
    assert queued[0]["starts_in_s"] is None
    assert 3500 < queued[1]["starts_in_s"] < 3600
    assert queued[1]["starts_at"] > datetime.now(UTC).isoformat()

    assert main(["status"]) == EXIT_OK
    text = capsys.readouterr().out
    assert "queued  urgent (20260915-140100-cccccc) prio=10 needs 2 gpus est 30m" in text
    assert "starts in ~1h00m" in text
