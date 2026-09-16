"""What `gpuc` promises automation: exit codes, `status --json`, and that one
bad host entry never takes the CLI with it.

The incident these come from: a registry another session could not validate
made every subcommand fail, and a non-zero `gpuc status` was read as "0 jobs
running". Both halves of that have to be impossible now.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from gpuc.control import status as status_mod
from gpuc.control.cli import (
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
    load_registry,
    read_registry,
)
from gpuc.control.status import HostView

GPU = "GPU-2a4bad3b-9fe3-7031-914d-384254e92908"

GOOD_ENTRY = {
    "name": "good",
    "kind": "local",
    "gpus": [GPU],
    "idle_minutes": 15.0,
    "ttl_hours": None,
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

    assert main(["host", "list"]) == EXIT_OK
    captured = capsys.readouterr()
    assert "good" in captured.out
    assert "skipping host 'bad'" in captured.err


def test_a_write_puts_back_the_entry_this_build_could_not_read(control_env: Path) -> None:
    """It is another session's host, not ours to delete on the next `host set`."""
    write_hosts({"hosts": {"good": GOOD_ENTRY, "bad": BAD_ENTRY}})
    assert main(["host", "set", "good", "--idle-min", "9"]) == EXIT_OK
    document = json.loads(hosts_file().read_text())
    assert document["hosts"]["bad"] == BAD_ENTRY
    assert document["hosts"]["good"]["idle_minutes"] == 9.0
    assert document["schema_version"] == 1


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
    assert main(["host", "add", "local", "--gpus", GPU]) == EXIT_LOCAL_STATE
    assert "Nothing was written" in capsys.readouterr().err
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


def test_status_is_zero_even_when_every_host_is_unreachable(
    control_env: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    main(["host", "add", "gpubox", "--ssh", "me@nowhere.invalid", "--gpus", GPU])

    def unreachable(entry: HostEntry, *args: object, **kwargs: object) -> HostView:
        return HostView(entry=entry, reachable=False, error="ssh: could not resolve hostname")

    monkeypatch.setattr(status_mod, "gather", unreachable)
    assert main(["status"]) == EXIT_OK
    assert "UNREACHABLE" in capsys.readouterr().out


def test_status_reports_a_bad_entry_per_host_without_failing(
    control_env: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    write_hosts({"hosts": {"good": GOOD_ENTRY, "bad": BAD_ENTRY}})
    monkeypatch.setattr(
        status_mod, "gather", lambda entry, *a, **k: HostView(entry=entry, reachable=True)
    )
    assert main(["status"]) == EXIT_OK
    captured = capsys.readouterr()
    assert "host good" in captured.out
    assert "skipping host 'bad'" in captured.err


def test_a_named_host_that_does_not_exist_is_four(
    control_env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["status", "--host", "nope"]) == EXIT_NOT_FOUND
    assert "no host named 'nope'" in capsys.readouterr().err


def test_a_bad_duration_is_usage(control_env: Path, capsys: pytest.CaptureFixture[str]) -> None:
    main(["host", "add", "local", "--gpus", GPU])
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
    paths.ensure_layout()
    jobs.write_config(HostConfig(host="local", gpus=[GPU]))
    paths.heartbeat_file().touch()

    started = (datetime.now(UTC) - timedelta(seconds=30)).isoformat()
    ended = (datetime.now(UTC) - timedelta(minutes=30)).isoformat()
    jobs.write_spec(
        JobSpec.from_dict({"job_id": RUNNING_JOB, "name": "lego-s4", "command": "train", "gpus": 1})
    )
    jobs.write_state(
        RUNNING_JOB,
        JobState(
            status="running",
            phase="main",
            gpus=[GPU],
            started_at=started,
            util_recent=[90.0, 95.0],
            isolation="cgroup",
        ),
    )
    jobs.write_spec(
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
        FINISHED_JOB, JobState(status="failed", reason="low-util", ended_at=ended, exit_code=1)
    )
    monkeypatch.delenv("GPUC_HOME")

    (home / "pkg").symlink_to(Path(__file__).resolve().parents[1])
    entry = HostEntry(
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
        "phase",
        "elapsed_s",
        "util",
        "progress_pct",
        "eta",
        "eta_s",
        "estimated_runtime_min",
        "progress_error",
        "gpus",
        "iso",
        "ended_at",
        "outputs_pending",
        "outputs_lost",
        "suspect",
        "priority",
        "attempt",
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
    assert done["reason"] == "low-util"
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
    assert (host["draining"], host["paused"], host["pod_gone"]) == (False, False, False)

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
    config_file().write_text('s3_bucket = "bucket"\nmax_pods = 5\n')
    assert main(["config", "show", "--json"]) == EXIT_OK
    document = status_json(capsys)
    assert document["schema_version"] == 1
    assert document["config_file"] == str(config_file())
    assert document["config_file_exists"] is True
    assert document["settings"]["s3_bucket"] == "bucket"
    assert document["settings"]["max_pods"] == 5
    assert document["notes"] == []


def test_status_json_says_unreachable_rather_than_empty(
    control_env: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    main(["host", "add", "gpubox", "--ssh", "me@box", "--gpus", GPU])
    capsys.readouterr()
    monkeypatch.setattr(
        status_mod,
        "gather",
        lambda entry, *a, **k: HostView(entry=entry, reachable=False, error="ssh timed out"),
    )
    assert main(["status", "--json"]) == EXIT_OK
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
        status_mod, "gather", lambda entry, *a, **k: HostView(entry=entry, reachable=True)
    )
    assert main(["status", "--json"]) == EXIT_OK
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

    main(["host", "add", "gpubox", "--ssh", "me@box"])
    with_commit = (
        load_registry()
        .require("gpubox")
        .model_copy(update={"pkg_commit": "b" * 40, "bootstrapped_at": "2026-09-15T20:00:00+00:00"})
    )
    write_hosts({"hosts": {"gpubox": json.loads(with_commit.model_dump_json())}})
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
    entry = HostEntry(name="gpubox", kind="ssh", ssh="me@box", pkg_commit="a" * 40)
    view = HostView(entry=entry, reachable=True, pkg_commit="b" * 40)
    warnings = status_mod.host_warnings(view)
    assert len(warnings) == 1
    assert "host gpubox is running gpuc " + "b" * 12 in warnings[0]
    assert "gpuc host bootstrap gpubox" in warnings[0]
    assert status_mod.render(view).count("WARNING") == 1
    assert status_mod.host_json(view)["pkg_commit"] == "b" * 40
    assert warnings[0] in status_mod.host_json(view)["errors"]


def test_a_host_running_this_build_or_one_we_could_not_ask_says_nothing(
    control_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gpuc.control import version as version_mod

    monkeypatch.setattr(version_mod, "local_commit", lambda: "a" * 40)
    entry = HostEntry(name="s", pkg_commit="b" * 40)
    current = HostView(entry=entry, reachable=True, pkg_commit="a" * 40)
    assert status_mod.host_warnings(current) == []
    # Unreachable: "we could not ask" is not evidence of anything.
    assert status_mod.host_warnings(HostView(entry=entry, pkg_commit="b" * 40)) == []


def test_host_list_reports_this_machines_own_record_and_says_that_is_what_it_is(
    control_env: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """`host list` never asks a host anything, so the one thing it must not do
    is imply it did: what it has is the commit *this* machine last shipped."""
    from gpuc.control import version as version_mod

    main(["host", "add", "gpubox", "--ssh", "me@box"])
    shipped = (
        load_registry()
        .require("gpubox")
        .model_copy(update={"pkg_commit": "b" * 40, "bootstrapped_at": "2026-09-15T20:00:00+00:00"})
    )
    write_hosts({"hosts": {"gpubox": json.loads(shipped.model_dump_json())}})
    monkeypatch.setattr(version_mod, "local_commit", lambda: "a" * 40)
    capsys.readouterr()

    assert main(["host", "list"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "pkg     " + "b" * 12 + " shipped from here" in out
    assert "NOTE host gpubox was last given gpuc " + "b" * 12 + " from this machine" in out
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
    entry = HostEntry(name="gpubox", kind="ssh", ssh="me@box", pkg_commit="a" * 40)
    (warning,) = status_mod.host_warnings(HostView(entry=entry, reachable=True))
    assert "a build too old to say which" in warning
    assert "gpuc host bootstrap gpubox" in warning
    # With nothing to compare against: a gpuc that cannot name its own
    # commit has no business telling a host it is behind.
    monkeypatch.setattr(version_mod, "local_commit", lambda: None)
    assert status_mod.host_warnings(HostView(entry=entry, reachable=True)) == []


def test_a_config_only_another_control_machine_could_have_written_is_flagged(
    control_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gpuc.control import version as version_mod

    monkeypatch.setattr(version_mod, "local_commit", lambda: "a" * 40)
    entry = HostEntry(name="gpubox", kind="ssh", ssh="me@box", gpus=["2", "3"], pkg_commit="a" * 40)
    view = HostView(
        entry=entry,
        reachable=True,
        pkg_commit="a" * 40,
        configured={"host": "gpubox", "gpus": ["0", "1"]},
    )
    (warning,) = status_mod.host_warnings(view)
    assert "gpus 0,1 -> 2,3" in warning
    assert "gpuc host bootstrap gpubox" in warning


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
    ["cancel"],
    ["reorder"],
    ["estimate"],
    ["pods"],
    ["version"],
    ["clean"],
    ["reconcile"],
    ["host", "list"],
    ["host", "probe"],
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
    assert main(["host", "list", "--json"]) == EXIT_OK
    document = document_of(capsys)
    (host,) = document["hosts"]
    assert host["name"] == "good"
    assert host["kind"] == "local"
    assert host["gpus"] == [GPU]
    assert host["remote_home"] == "$HOME/.gpuc"
    assert host["ephemeral"] is False
    assert "skipping host 'bad'" in document["errors"][0]


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
            HostEntry(
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
    entry = (
        load_registry().require("local").model_copy(update={"s3_prefix": "s3://bucket/gpuc/local"})
    )
    write_hosts({"hosts": {"local": json.loads(entry.model_dump_json())}})

    assert main(["logs", RUNNING_JOB, "--json"]) == EXIT_OK
    document = document_of(capsys)
    assert document["source"] == "s3"
    assert document["location"] == f"s3://{uri}"
    assert document["lines"] == ["mirrored"]
    assert any("could not read" in note for note in document["notes"])
