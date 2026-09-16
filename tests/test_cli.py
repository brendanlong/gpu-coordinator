from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

from gpuc.control import reconcile as reconcile_mod
from gpuc.control.clean import purge_host
from gpuc.control.cli import (
    EXIT_LOCAL_STATE,
    EXIT_NOT_FOUND,
    EXIT_USAGE,
    build_parser,
    main,
)
from gpuc.control.config import (
    HostEntry,
    LocalStateUnreadable,
    Settings,
    config_file,
    hosts_file,
    load_registry,
    load_settings,
    registry_transaction,
)
from gpuc.control.providers.base import Constraints
from gpuc.control.remote import HostSession
from gpuc.control.submit import SubmitResult
from tests.fakeprovider import FakeProvider, fake_bootstrap, running_pod
from tests.fakes3 import FakeS3Client

GPU = "GPU-2a4bad3b-9fe3-7031-914d-384254e92908"


def test_host_add_local_records_the_uuids(
    control_env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["host", "add", "local", "--gpus", GPU]) == 0
    entry = load_registry().require("local")
    assert (entry.kind, entry.gpus, entry.ssh) == ("local", [GPU], None)
    assert "gpuc host bootstrap local" in capsys.readouterr().out


def test_host_add_ssh_records_the_target_and_port(control_env: Path) -> None:
    main(["host", "add", "gpubox", "--ssh", "me@box", "--port", "2222", "--gpus", "GPU-a,GPU-b"])
    entry = load_registry().require("gpubox")
    assert (entry.kind, entry.ssh, entry.port) == ("ssh", "me@box", 2222)
    assert entry.gpus == ["GPU-a", "GPU-b"]


def test_host_list_and_remove(control_env: Path, capsys: pytest.CaptureFixture[str]) -> None:
    main(["host", "add", "local", "--gpus", GPU])
    assert main(["host", "list"]) == 0
    assert GPU in capsys.readouterr().out
    assert main(["host", "remove", "local"]) == 0
    assert load_registry().hosts == {}
    assert main(["host", "list"]) == 0
    assert "no hosts registered" in capsys.readouterr().out


def test_commands_on_an_unknown_host_explain_themselves(
    control_env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["host", "bootstrap", "nope"]) == EXIT_NOT_FOUND
    assert "no host named 'nope'" in capsys.readouterr().err


def test_submit_without_a_host_says_which_flag_is_missing(
    control_env: Path, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    job = tmp_path / "job.yaml"
    job.write_text("command: true\n")
    assert main(["submit", str(job)]) == EXIT_USAGE
    assert "--host" in capsys.readouterr().err


def test_status_with_no_hosts_is_not_an_error(
    control_env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["status"]) == 0
    assert "no hosts registered" in capsys.readouterr().out


def test_cancel_for_an_unknown_job_tells_you_where_to_look(
    control_env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["cancel", "20260101-000000-aaaaaa"]) == EXIT_NOT_FOUND
    assert "no registered host knows job" in capsys.readouterr().err


def test_requeue_without_an_s3_bucket_explains_the_gap(
    control_env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    main(["host", "add", "local", "--gpus", GPU])
    assert main(["requeue", "20260101-000000-aaaaaa", "--host", "local"]) == 1
    assert "s3_bucket is unset" in capsys.readouterr().err


def test_submit_runpod_without_a_gpu_name_says_which_flag_is_missing(
    control_env: Path,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RUNPOD_API_KEY", "test-key")
    job = tmp_path / "job.yaml"
    job.write_text('command: "true"\n')
    assert main(["submit", str(job), "--runpod"]) == EXIT_USAGE
    assert "--gpu <name>" in capsys.readouterr().err


def test_submit_runpod_passes_the_flags_through_and_mirrors_the_spec_first(
    control_env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = tmp_path / "job.yaml"
    job.write_text('command: "true"\ngpus: 1\n')
    monkeypatch.setenv("RUNPOD_API_KEY", "test-key")
    (Path(control_env) / "config/config.toml").write_text('s3_bucket = "bucket"\n')
    s3 = FakeS3Client()
    monkeypatch.setattr("gpuc.control.s3index.S3Index.client", property(lambda self: s3))

    seen: dict[str, object] = {}

    def fake_runpod_host(constraints: Constraints, settings: Settings, **kwargs: object):
        seen.update(kwargs)
        seen["constraints"] = constraints
        seen["mirrored_before_provisioning"] = sorted(s3.objects)
        return HostEntry(name="gpuc-e2e-1", kind="runpod", gpus=["GPU-1"])

    def fake_submit_spec(entry: HostEntry, model: object, *args: object, **kwargs: object):
        seen["host"] = entry.name
        seen["job_id"] = kwargs["job_id"]
        return SubmitResult(job_id=str(kwargs["job_id"]), host=entry.name, attempt=1)

    monkeypatch.setattr("gpuc.control.cli.runpod_host", fake_runpod_host)
    monkeypatch.setattr("gpuc.control.cli.submit_spec", fake_submit_spec)

    assert (
        main(
            [
                "submit",
                str(job),
                "--runpod",
                "--gpu",
                "A40,RTX4090",
                "--min-vram",
                "24",
                "--max-price",
                "0.60",
                "--cloud",
                "any",
                "--cuda-min",
                "12.8",
                "--idle-min",
                "2",
                "--ttl-hours",
                "1",
                "--disk",
                "20",
                "--no-reuse",
                "--name-hint",
                "e2e",
            ]
        )
        == 0
    )
    constraints = seen["constraints"]
    assert isinstance(constraints, Constraints)
    assert constraints.gpu_names == ["A40", "RTX4090"]
    assert (constraints.min_vram_gb, constraints.max_price_usd_hr) == (24, 0.60)
    assert constraints.clouds == ["SECURE", "COMMUNITY"]
    assert (seen["idle_minutes"], seen["ttl_hours"], seen["disk_gb"]) == (2.0, 1.0, 20)
    assert (seen["reuse"], seen["name_hint"]) == (False, "e2e")
    mirrored = seen["mirrored_before_provisioning"]
    assert isinstance(mirrored, list) and mirrored == [f"bucket/gpuc/specs/{seen['job_id']}.json"]


def test_reconcile_once_fails_closed_without_desired_state(
    control_env: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RUNPOD_API_KEY", "test-key")
    provider = FakeProvider()
    monkeypatch.setattr("gpuc.control.cli.make_provider", lambda settings: provider)
    assert main(["reconcile", "--once"]) == 1
    assert "does not exist" in capsys.readouterr().out
    assert provider.terminated == []


def test_pods_lists_ours_and_counts_the_others(
    control_env: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RUNPOD_API_KEY", "test-key")
    provider = FakeProvider(existing=[running_pod("other-tenant", "podF")])
    provider.adopt(running_pod("gpuc-e2e-aaa", "pod1"))
    monkeypatch.setattr("gpuc.control.cli.make_provider", lambda settings: provider)
    assert main(["pods", "--no-heartbeat"]) == 0
    out = capsys.readouterr().out
    assert "gpuc-e2e-aaa" in out
    assert "1 other pod(s) in the account, never touched: other-tenant (RUNNING)" in out


def test_requeue_runpod_reads_the_spec_from_s3_and_provisions(
    control_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RUNPOD_API_KEY", "test-key")
    (Path(control_env) / "config/config.toml").write_text('s3_bucket = "bucket"\n')
    s3 = FakeS3Client()
    s3.objects["bucket/gpuc/specs/20260101-000000-aaaaaa.json"] = json.dumps(
        {"job_id": "20260101-000000-aaaaaa", "command": "true", "gpus": 1, "attempt": 1}
    ).encode()
    monkeypatch.setattr("gpuc.control.s3index.S3Index.client", property(lambda self: s3))
    seen: dict[str, object] = {}

    def fake_runpod_host(constraints: Constraints, settings: Settings, **kwargs: object):
        seen["gpu_names"] = constraints.gpu_names
        return HostEntry(name="gpuc-e2e-1", kind="runpod", gpus=["GPU-1"])

    def fake_submit_spec(entry: HostEntry, model: object, *args: object, **kwargs: object):
        seen["attempt"] = kwargs["attempt"]
        return SubmitResult(job_id="new", host=entry.name, attempt=2)

    monkeypatch.setattr("gpuc.control.cli.runpod_host", fake_runpod_host)
    monkeypatch.setattr("gpuc.control.cli.submit_spec", fake_submit_spec)

    assert main(["requeue", "20260101-000000-aaaaaa", "--runpod", "--gpu", "A40"]) == 0
    assert seen == {"gpu_names": ["A40"], "attempt": 2}


# -- first-run experience -------------------------------------------------------


def test_config_init_writes_a_commented_file_that_reloads_to_the_defaults(
    control_env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["config", "init"]) == 0
    path = config_file()
    text = path.read_text()
    for key in (
        "s3_bucket",
        "runpod_pod_prefix",
        "max_pods",
        "max_total_usd_per_hour",
        "ssh_key",
        "image",
        "disk_gb",
    ):
        assert key in text
    assert load_settings() == Settings()
    assert str(path) in capsys.readouterr().out


def test_config_init_refuses_to_clobber_without_force(control_env: Path) -> None:
    assert main(["config", "init"]) == 0
    config_file().write_text("max_pods = 9\n")
    assert main(["config", "init"]) == 1
    assert load_settings().max_pods == 9
    assert main(["config", "init", "--force"]) == 0
    assert load_settings().max_pods == 3


def test_config_show_works_without_a_config_file(
    control_env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["config", "show"]) == 0
    out = capsys.readouterr().out
    assert "does not exist; using defaults" in out
    assert "no s3_bucket" in out


def test_commands_work_with_no_config_file_and_say_how_to_make_one(
    control_env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert not config_file().exists()
    assert main(["host", "list"]) == 0
    captured = capsys.readouterr()
    assert "gpuc config init" in captured.err
    assert captured.err.count("gpuc config init") == 1
    assert "no hosts registered" in captured.out


def test_no_note_once_a_config_file_exists(
    control_env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    main(["config", "init"])
    capsys.readouterr()
    assert main(["host", "list"]) == 0
    assert "gpuc config init" not in capsys.readouterr().err


def test_runpod_commands_fail_fast_without_an_api_key(
    control_env: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Before the spec is mirrored, before a host is picked, before any spend."""
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    for argv in (
        ["submit", "job.yaml", "--runpod", "--gpu", "A40"],
        ["pods"],
        ["reconcile", "--once"],
    ):
        assert main(argv) == 1
        err = capsys.readouterr().err
        assert err.strip().splitlines() == [
            "error: RUNPOD_API_KEY is not set; export it before using --runpod, "
            "`gpuc pods` or `gpuc reconcile`"
        ]


def test_a_non_runpod_command_does_not_need_the_api_key(
    control_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    assert main(["host", "list"]) == 0


def test_reconcile_install_does_not_need_the_api_key(
    control_env: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Writing unit files is setup, not a provider call."""
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    # systemd_dir() is $HOME-relative, so it has to be redirected explicitly or
    # the test would install units into the developer's real session.
    units = tmp_path / "systemd"
    monkeypatch.setattr(reconcile_mod, "systemd_dir", lambda: units)
    assert main(["reconcile", "--install"]) == 0
    assert {p.name for p in units.iterdir()} == {
        "gpuc-reconcile.service",
        "gpuc-reconcile.timer",
    }


def test_submit_runpod_refuses_a_too_big_spec_before_creating_a_pod(
    control_env: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A spec that cannot run on the pod we would buy must fail before we buy it."""
    job = tmp_path / "job.yaml"
    job.write_text('command: "true"\ngpus: 4\n')
    monkeypatch.setenv("RUNPOD_API_KEY", "test-key")
    created: list[object] = []
    monkeypatch.setattr(
        "gpuc.control.cli.runpod_host",
        lambda *a, **k: created.append(a) or HostEntry(name="gpuc-x", kind="runpod"),
    )

    assert main(["submit", str(job), "--runpod", "--gpu", "A40"]) == 1
    assert "--gpu-count" in capsys.readouterr().err
    assert created == []


def test_submit_runpod_refuses_missing_secrets_before_creating_a_pod(
    control_env: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = tmp_path / "job.yaml"
    job.write_text('command: "true"\ngpus: 1\nsecrets: ["GPUC_DEFINITELY_UNSET"]\n')
    monkeypatch.setenv("RUNPOD_API_KEY", "test-key")
    created: list[object] = []
    monkeypatch.delenv("GPUC_DEFINITELY_UNSET", raising=False)
    monkeypatch.setattr(
        "gpuc.control.cli.runpod_host",
        lambda *a, **k: created.append(a) or HostEntry(name="gpuc-x", kind="runpod"),
    )

    assert main(["submit", str(job), "--runpod", "--gpu", "A40"]) == 1
    assert "GPUC_DEFINITELY_UNSET" in capsys.readouterr().err
    assert created == []


# -- clean --purge --verify ---------------------------------------------------


class StubSession:
    """A host that answers `purge` with whatever the test wants."""

    def __init__(self, payloads: list[dict[str, object]]) -> None:
        self.payloads = payloads
        self.calls: list[str] = []
        self.checked: list[bool] = []

    def host_json(self, args: str, *, timeout: float = 0.0, check: bool = True) -> object:
        self.calls.append(args)
        self.checked.append(check)
        return self.payloads.pop(0)


def purged_entry(job_id: str, prefix: str | None = "s3://bucket/gpuc/gpubox") -> dict[str, object]:
    return {
        "job_id": job_id,
        "status": "succeeded",
        "bytes": 1024,
        "age_days": 30.0,
        "meta_synced_at": "2026-01-01T00:00:00+00:00",
        "meta_synced_to": prefix,
        "forced": False,
    }


def as_session(session: StubSession) -> HostSession:
    return cast("HostSession", session)


def test_verify_purges_only_the_jobs_whose_mirror_answers(control_env: Path) -> None:
    client = FakeS3Client(objects={"bucket/gpuc/gpubox/jobs/kept/log.txt": b"hello\n"})
    entry = HostEntry(name="gpubox", kind="ssh", ssh="me@gpubox", python="/usr/bin/python3")
    session = StubSession(
        [
            {"dry_run": True, "purged": [purged_entry("kept"), purged_entry("gone")]},
            {"dry_run": False, "purged": [purged_entry("kept")], "freed_bytes": 1024},
        ]
    )
    report = purge_host(
        entry,
        Settings(),
        session=as_session(session),
        older_than_days=7.0,
        verify=True,
        s3_client=client,
    )
    assert report.verified == ["kept"]
    assert [job["job_id"] for job in report.purged] == ["kept"]
    assert any("no mirrored log" in str(job["why"]) for job in report.purge_skipped)
    assert "--only kept" in session.calls[1]


def test_a_confirmed_horizon_zero_purge_says_what_it_is_doing(control_env: Path) -> None:
    """`--purge --all-finished --yes` is allowed, and never silent about it."""
    from gpuc.control.clean import clean_host

    entry = HostEntry(name="gpubox", kind="ssh", ssh="me@gpubox", python="/usr/bin/python3")
    session = StubSession([{"dry_run": False, "purged": [purged_entry("old")], "freed_bytes": 1}])
    report = clean_host(
        entry,
        Settings(),
        session=as_session(session),
        purge=True,
        all_finished=True,
        yes=True,
    )
    assert "--older-than 0.0" in session.calls[0]
    assert "purging every finished job (horizon 0)" in report.render()


def test_verify_with_force_purges_even_an_unmirrored_job(control_env: Path) -> None:
    client = FakeS3Client()
    entry = HostEntry(name="gpubox", kind="ssh", ssh="me@gpubox", python="/usr/bin/python3")
    session = StubSession(
        [
            {"dry_run": True, "purged": [purged_entry("gone")]},
            {"dry_run": False, "purged": [purged_entry("gone")], "freed_bytes": 1024},
        ]
    )
    report = purge_host(
        entry,
        Settings(),
        session=as_session(session),
        older_than_days=7.0,
        force=True,
        verify=True,
        s3_client=client,
    )
    assert [job["job_id"] for job in report.purged] == ["gone"]
    assert any("purged anyway because --force" in note for note in report.notes)


def test_verify_without_purge_is_refused(
    control_env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    main(["host", "add", "gpubox", "--ssh", "me@box"])
    assert main(["clean", "--host", "gpubox", "--verify", "--all-finished"]) == EXIT_USAGE
    assert "only mean something with --purge" in capsys.readouterr().err


def test_clean_with_no_selection_and_no_purge_exits(
    control_env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    main(["host", "add", "gpubox", "--ssh", "me@box"])
    assert main(["clean", "--host", "gpubox"]) == EXIT_USAGE
    assert "--all-finished" in capsys.readouterr().err


# -- clean --only -------------------------------------------------------------


def test_only_purges_the_named_jobs_at_horizon_zero_and_scopes_the_sweep(
    control_env: Path,
) -> None:
    """Naming ids is the confirmation `--purge --all-finished` needs `--yes` for."""
    from gpuc.control.clean import clean_host

    entry = HostEntry(name="gpubox", kind="ssh", ssh="me@gpubox", python="/usr/bin/python3")
    session = StubSession([{"dry_run": False, "purged": [purged_entry("a")], "freed_bytes": 1}])
    report = clean_host(
        entry,
        Settings(),
        session=as_session(session),
        purge=True,
        only=["a", "b"],
    )
    assert session.calls == ["purge --older-than 0.0 --only a,b --sweep-only a,b"]
    assert "--only a,b" in report.render()
    # A host that exits 1 has still said what it deleted; the report is the
    # point of the call, so `clean` must not let the exit code discard it.
    assert session.checked == [False]


def test_only_without_purge_cleans_just_those_workdirs(control_env: Path) -> None:
    from gpuc.control.clean import clean_host

    entry = HostEntry(name="gpubox", kind="ssh", ssh="me@gpubox", python="/usr/bin/python3")
    session = StubSession([{"dry_run": False, "removed": [], "freed_bytes": 0}])
    clean_host(entry, Settings(), session=as_session(session), only=["a"])
    assert session.calls == ["clean --only a"]


def test_only_with_verify_purges_what_answered_and_sweeps_what_was_asked(
    control_env: Path,
) -> None:
    client = FakeS3Client(objects={"bucket/gpuc/gpubox/jobs/kept/log.txt": b"hello\n"})
    entry = HostEntry(name="gpubox", kind="ssh", ssh="me@gpubox", python="/usr/bin/python3")
    session = StubSession(
        [
            {"dry_run": True, "purged": [purged_entry("kept"), purged_entry("gone")]},
            {"dry_run": False, "purged": [purged_entry("kept")], "freed_bytes": 1024},
        ]
    )
    report = purge_host(
        entry,
        Settings(),
        session=as_session(session),
        only=["kept", "gone"],
        verify=True,
        s3_client=client,
    )
    assert report.verified == ["kept"]
    assert "--only kept,gone --sweep-only kept,gone" in session.calls[0]
    # The job whose mirror never answered keeps its dir and still loses its venv.
    assert session.calls[1].endswith("--only kept --sweep-only kept,gone")


def test_a_dry_run_says_the_workdirs_verification_dropped_will_still_go(
    control_env: Path,
) -> None:
    """The host sized its sweep over the dirs it expected to purge, so the
    workdirs of the jobs we then drop are in neither total."""
    client = FakeS3Client()
    entry = HostEntry(name="gpubox", kind="ssh", ssh="me@gpubox", python="/usr/bin/python3")
    session = StubSession([{"dry_run": True, "purged": [purged_entry("gone")]}])
    report = purge_host(
        entry,
        Settings(),
        session=as_session(session),
        only=["gone"],
        verify=True,
        dry_run=True,
        s3_client=client,
    )
    assert report.purged == []
    assert any("still reclaims their workdirs" in note for note in report.notes)


def test_purge_only_needs_no_yes(control_env: Path) -> None:
    from gpuc.control.clean import check_flags

    check_flags(purge=True, only=["a"])


@pytest.mark.parametrize("extra", [["--all-finished"], ["--older-than", "7"]])
def test_only_cannot_be_combined_with_an_age_horizon(
    control_env: Path, capsys: pytest.CaptureFixture[str], extra: list[str]
) -> None:
    """argparse rejects the command line; `check_flags` judges it for callers."""
    from gpuc.control.clean import CleanUsageError, check_flags

    main(["host", "add", "gpubox", "--ssh", "me@box"])
    with pytest.raises(SystemExit) as caught:
        main(["clean", "--host", "gpubox", "--only", "a", *extra])
    assert caught.value.code == EXIT_USAGE
    assert "not allowed with argument --only" in capsys.readouterr().err
    with pytest.raises(CleanUsageError, match="cannot be combined"):
        check_flags(
            only=["a"],
            all_finished="--all-finished" in extra,
            older_than_days=None if "--all-finished" in extra else 7.0,
        )


def test_an_empty_only_is_a_usage_error_not_a_silent_no_op(
    control_env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The host reads `--only ''` as "purge nothing"; a user never means that."""
    main(["host", "add", "gpubox", "--ssh", "me@box"])
    assert main(["clean", "--host", "gpubox", "--purge", "--only", " , "]) == EXIT_USAGE
    assert "at least one job id" in capsys.readouterr().err


def test_retention_days_is_stored_and_cleared(control_env: Path) -> None:
    assert main(["host", "add", "gpubox", "--ssh", "me@box", "--retention-days", "14"]) == 0
    assert load_registry().require("gpubox").retention_days == 14.0
    assert load_registry().require("gpubox").host_config().retention_days == 14.0
    assert main(["host", "set", "gpubox", "--retention-days", ""]) == 0
    assert load_registry().require("gpubox").retention_days is None


def test_a_bad_retention_value_is_rejected(
    control_env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert (
        main(["host", "add", "gpubox", "--ssh", "me@box", "--retention-days", "soon"]) == EXIT_USAGE
    )
    assert "wants a number of days" in capsys.readouterr().err


def test_the_index_listing_flags_jobs_whose_outputs_were_lost(control_env: Path) -> None:
    from gpuc.control.cli import _outputs_lost_ids
    from gpuc.control.s3index import IndexEntry, S3Index

    client = FakeS3Client(
        objects={
            "bucket/gpuc/pod/jobs/lost/state.json": b'{"outputs_lost": true}',
            "bucket/gpuc/pod/jobs/fine/state.json": b'{"outputs_lost": false}',
        }
    )
    entries = [
        IndexEntry(job_id=job_id, host="pod", s3_prefix="s3://bucket/gpuc/pod")
        for job_id in ("lost", "fine", "missing")
    ]
    assert _outputs_lost_ids(S3Index("bucket", client), entries) == {"lost"}


def test_submit_and_requeue_both_take_no_git() -> None:
    parser = build_parser()
    assert parser.parse_args(["submit", "job.yaml", "--host", "h", "--no-git"]).no_git
    assert parser.parse_args(["requeue", "id", "--host", "h", "--no-git"]).no_git
    assert not parser.parse_args(["submit", "job.yaml", "--host", "h"]).no_git


def test_ttl_is_unset_unless_asked_for() -> None:
    parser = build_parser()
    assert parser.parse_args(["host", "add", "h"]).ttl_hours is None
    assert parser.parse_args(["submit", "j", "--runpod", "--gpu", "A40"]).ttl_hours is None
    assert parser.parse_args(["host", "add", "h", "--ttl-hours", "6"]).ttl_hours == 6.0


def test_a_negative_ttl_clears_the_cap(control_env: Path) -> None:
    with registry_transaction() as registry:
        registry.put(HostEntry(name="h", gpus=[], ttl_hours=6.0))
    assert main(["host", "set", "h", "--ttl-hours", "-1"]) == 0
    assert load_registry().hosts["h"].ttl_hours is None


def test_submit_reships_the_package_to_a_host_on_another_commit(
    control_env: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A host on older code dispatches the job with code that does not match
    the spec this machine just wrote."""
    job = tmp_path / "job.yaml"
    job.write_text('command: "true"\n')
    main(["host", "add", "gpubox", "--ssh", "me@box", "--gpus", GPU])
    _set_host(python="/py", pkg_commit="a" * 40)
    monkeypatch.setattr("gpuc.control.cli.version_mod.local_commit", lambda: "b" * 40)
    resynced: list[str] = []

    def fake_resync(entry: HostEntry, settings: object = None, **kwargs: object) -> HostEntry:
        resynced.append(entry.name)
        return entry.model_copy(update={"pkg_commit": "b" * 40})

    monkeypatch.setattr("gpuc.control.cli.resync_package", fake_resync)
    monkeypatch.setattr(
        "gpuc.control.cli.submit_file",
        lambda *a, **k: SubmitResult(job_id="j", host="gpubox", attempt=1),
    )

    assert main(["submit", str(job), "--host", "gpubox"]) == 0
    out = capsys.readouterr().out
    assert resynced == ["gpubox"]
    assert sum("re-syncing the package" in line for line in out.splitlines()) == 1
    assert load_registry().require("gpubox").pkg_commit == "b" * 40

    # Now the host is on this build, so nothing is shipped.
    resynced.clear()
    assert main(["submit", str(job), "--host", "gpubox"]) == 0
    assert resynced == []

    # An unrecorded commit counts as different: those hosts are the oldest.
    _set_host(pkg_commit=None)
    assert main(["submit", str(job), "--host", "gpubox", "--no-bootstrap"]) == 0
    assert resynced == []
    assert main(["submit", str(job), "--host", "gpubox"]) == 0
    assert resynced == ["gpubox"]


def _set_host(**changes: object) -> None:
    with registry_transaction() as registry:
        registry.put(registry.require("gpubox").model_copy(update=changes))


def test_a_negative_ttl_on_add_means_no_ttl_not_an_expired_host(
    control_env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """-1 stored as a TTL is a host the next reaper pass terminates."""
    assert main(["host", "add", "h", "--ssh", "me@box", "--ttl-hours", "-1"]) == 0
    assert load_registry().hosts["h"].ttl_hours is None
    assert main(["host", "add", "z", "--ssh", "me@box", "--ttl-hours", "0"]) == EXIT_USAGE
    assert "would expire the host the moment it exists" in capsys.readouterr().err


def test_runpod_and_host_together_are_a_usage_error(
    control_env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One picks a host that exists, the other buys one. Not both."""
    monkeypatch.setenv("RUNPOD_API_KEY", "test-key")
    job = tmp_path / "job.yaml"
    job.write_text('command: "true"\n')
    main(["host", "add", "gpubox", "--ssh", "me@box", "--gpus", GPU])
    assert main(["submit", str(job), "--runpod", "--gpu", "A40", "--host", "gpubox"]) == EXIT_USAGE
    assert main(["requeue", "job-1", "--host", "gpubox", "--runpod", "--gpu", "A40"]) == EXIT_USAGE


def test_requeue_of_an_unknown_job_is_exit_four(
    control_env: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    (Path(control_env) / "config/config.toml").write_text('s3_bucket = "bucket"\n')
    monkeypatch.setattr(
        "gpuc.control.s3index.S3Index.client", property(lambda self: FakeS3Client())
    )
    main(["host", "add", "gpubox", "--ssh", "me@box", "--gpus", GPU])
    assert main(["requeue", "20260101-000000-nosuch", "--host", "gpubox"]) == EXIT_NOT_FOUND
    assert "no mirrored spec for job" in capsys.readouterr().err


def test_purging_every_finished_job_has_to_be_asked_for_twice(
    control_env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`--purge --all-finished` is an age horizon of 0: it deletes the job dir
    of something that ended a minute ago, log and all."""
    main(["host", "add", "gpubox", "--ssh", "me@box"])
    assert main(["clean", "--host", "gpubox", "--purge", "--all-finished"]) == EXIT_USAGE
    assert "Add --yes to confirm" in capsys.readouterr().err


def test_host_add_takes_gpu_indices_and_stores_them_as_given(control_env: Path) -> None:
    """Ownership of part of a shared box is an agreement in nvidia-smi
    numbering, so resolving it here would freeze this boot's mapping into the
    registry; the host redoes it every dispatch pass."""
    assert main(["host", "add", "box", "--ssh", "me@box", "--gpus", f"2,3,{GPU}"]) == 0
    assert load_registry().require("box").gpus == ["2", "3", GPU]


def test_host_add_and_set_refuse_a_gpus_value_that_is_neither(
    control_env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["host", "add", "box", "--gpus", "A40,2"]) == EXIT_USAGE
    assert "gpuc host probe" in capsys.readouterr().err
    assert load_registry().hosts == {}

    assert main(["host", "add", "box", "--gpus", "2"]) == 0
    assert main(["host", "set", "box", "--gpus", "GPU-a,nonsense"]) == EXIT_USAGE
    assert load_registry().require("box").gpus == ["2"]


# -- `--json` on every command ------------------------------------------------


def one_document(capsys: pytest.CaptureFixture[str]) -> dict[str, object]:
    document = json.loads(capsys.readouterr().out)
    assert isinstance(document, dict)
    assert document["schema_version"] == 1
    return document


def test_submit_json_is_the_queued_job_and_its_notes(
    control_env: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = tmp_path / "job.yaml"
    job.write_text('command: "true"\ngpus: 0\n')
    main(["host", "add", "local", "--gpus", GPU])

    def fake_submit_file(entry: HostEntry, *args: object, **kwargs: object) -> SubmitResult:
        # The progress a text submit prints inline must not land on the document.
        report = kwargs["report"]
        assert callable(report)
        report("syncing 3 files")
        return SubmitResult(
            job_id="20260915-120000-abc123", host=entry.name, attempt=1, notes=["s3_bucket unset"]
        )

    monkeypatch.setattr("gpuc.control.cli.submit_file", fake_submit_file)
    capsys.readouterr()
    assert main(["submit", str(job), "--host", "local", "--json"]) == 0
    captured = capsys.readouterr()
    document = json.loads(captured.out)
    assert document == {
        "schema_version": 1,
        "job_id": "20260915-120000-abc123",
        "host": "local",
        "attempt": 1,
        "requeued_from": None,
        "notes": ["s3_bucket unset"],
    }
    assert "syncing 3 files" in captured.err


def test_requeue_json_names_the_job_it_came_from(
    control_env: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    (Path(control_env) / "config/config.toml").write_text('s3_bucket = "bucket"\n')
    s3 = FakeS3Client()
    s3.objects["bucket/gpuc/specs/20260101-000000-aaaaaa.json"] = json.dumps(
        {"job_id": "20260101-000000-aaaaaa", "command": "true", "gpus": 0}
    ).encode()
    monkeypatch.setattr("gpuc.control.s3index.S3Index.client", property(lambda self: s3))
    monkeypatch.setattr(
        "gpuc.control.cli.submit_spec",
        lambda entry, *a, **k: SubmitResult(job_id="new", host=entry.name, attempt=2),
    )
    main(["host", "add", "local", "--gpus", GPU])
    capsys.readouterr()

    assert main(["requeue", "20260101-000000-aaaaaa", "--host", "local", "--json"]) == 0
    document = one_document(capsys)
    assert document["requeued_from"] == "20260101-000000-aaaaaa"
    assert (document["job_id"], document["attempt"]) == ("new", 2)


def test_cancel_json_is_the_hosts_own_answer(
    control_env: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    main(["host", "add", "local", "--gpus", GPU])
    monkeypatch.setattr(
        "gpuc.control.cli.open_session",
        lambda *a, **k: as_session(StubSession([{"job_id": "j", "status": "cancelling"}])),
    )
    capsys.readouterr()
    assert main(["cancel", "20260101-000000-aaaaaa", "--host", "local", "--json"]) == 0
    document = one_document(capsys)
    assert document["status"] == "cancelling"
    assert (document["job_id"], document["host"]) == ("20260101-000000-aaaaaa", "local")


def test_reorder_json_repeats_the_priority_it_set(
    control_env: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    class Moved:
        def host_cli(self, args: str, *, check: bool = True) -> object:
            return type("Result", (), {"returncode": 0})()

    main(["host", "add", "local", "--gpus", GPU])
    monkeypatch.setattr("gpuc.control.cli.open_session", lambda *a, **k: Moved())
    capsys.readouterr()
    argv = ["reorder", "20260101-000000-aaaaaa", "--priority", "10", "--host", "local", "--json"]
    assert main(argv) == 0
    assert one_document(capsys)["priority"] == 10


def test_estimate_json_repeats_what_the_host_recorded(
    control_env: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    main(["host", "add", "local", "--gpus", GPU])
    session = StubSession(
        [{"job_id": "j", "estimated_runtime_min": 150.0, "status": "running", "warning": None}]
    )
    monkeypatch.setattr("gpuc.control.cli.open_session", lambda *a, **k: as_session(session))
    capsys.readouterr()
    argv = ["estimate", "20260101-000000-aaaaaa", "--minutes", "150", "--host", "local", "--json"]
    assert main(argv) == 0
    document = one_document(capsys)
    assert document["estimated_runtime_min"] == 150.0 and document["status"] == "running"
    # `check=False`: the host's refusal is a document, and raising on the exit
    # code would throw away the reason it gave.
    assert session.checked == [False]
    assert session.calls == ["estimate 20260101-000000-aaaaaa 150.0"]


def test_estimate_reports_the_hosts_refusal(
    control_env: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    main(["host", "add", "local", "--gpus", GPU])
    payload: dict[str, object] = {"job_id": "j", "error": "job j has already succeeded"}
    monkeypatch.setattr(
        "gpuc.control.cli.open_session", lambda *a, **k: as_session(StubSession([payload]))
    )
    capsys.readouterr()
    assert main(["estimate", "20260101-000000-aaaaaa", "--minutes", "5", "--host", "local"]) == 1
    assert "already succeeded" in capsys.readouterr().err


def test_estimate_clear_asks_the_host_to_clear_it(
    control_env: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    main(["host", "add", "local", "--gpus", GPU])
    session = StubSession(
        [{"job_id": "j", "estimated_runtime_min": None, "status": "queued", "warning": None}]
    )
    monkeypatch.setattr("gpuc.control.cli.open_session", lambda *a, **k: as_session(session))
    capsys.readouterr()
    assert main(["estimate", "20260101-000000-aaaaaa", "--clear", "--host", "local"]) == 0
    assert session.calls == ["estimate 20260101-000000-aaaaaa --clear"]
    assert "no longer estimates" in capsys.readouterr().out


def test_estimate_refuses_a_host_that_did_not_say_what_it_recorded(
    control_env: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Otherwise a build that does not know this command -- or any document
    without the key -- reads as a successful *clear* of a job it never touched.
    """
    main(["host", "add", "local", "--gpus", GPU])
    monkeypatch.setattr(
        "gpuc.control.cli.open_session",
        lambda *a, **k: as_session(StubSession([{"job_id": "j", "status": "running"}])),
    )
    capsys.readouterr()
    assert main(["estimate", "20260101-000000-aaaaaa", "--minutes", "5", "--host", "local"]) == 1
    assert "did not say what estimate it recorded" in capsys.readouterr().err


def test_estimate_updates_the_mirrored_spec_so_requeue_carries_it(
    control_env: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """`requeue` submits what the mirror holds, so an estimate left only on the
    host would be dropped by a re-run without a word."""
    from gpuc.control.s3index import S3Index

    main(["host", "add", "local", "--gpus", GPU])
    (Path(control_env) / "config/config.toml").write_text('s3_bucket = "bucket"\n')
    s3 = FakeS3Client()
    monkeypatch.setattr("gpuc.control.s3index.S3Index.client", property(lambda self: s3))
    S3Index("bucket", s3).put_spec_document(
        "20260101-000000-aaaaaa", {"command": "true", "some_future_field": 1}
    )
    monkeypatch.setattr(
        "gpuc.control.cli.open_session",
        lambda *a, **k: as_session(
            StubSession([{"job_id": "j", "estimated_runtime_min": 150.0, "status": "running"}])
        ),
    )
    capsys.readouterr()
    assert main(["estimate", "20260101-000000-aaaaaa", "--minutes", "150", "--host", "local"]) == 0
    mirrored = S3Index("bucket", s3).get_spec("20260101-000000-aaaaaa")
    assert mirrored["estimated_runtime_min"] == 150.0
    assert mirrored["some_future_field"] == 1


def test_estimate_says_so_when_the_mirror_kept_the_old_estimate(
    control_env: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The host has it, so the command succeeded; but a silent divergence is
    exactly what `requeue` would fall into later."""
    main(["host", "add", "local", "--gpus", GPU])
    (Path(control_env) / "config/config.toml").write_text('s3_bucket = "bucket"\n')
    monkeypatch.setattr(
        "gpuc.control.s3index.S3Index.client", property(lambda self: FakeS3Client())
    )
    monkeypatch.setattr(
        "gpuc.control.cli.open_session",
        lambda *a, **k: as_session(
            StubSession([{"job_id": "j", "estimated_runtime_min": 150.0, "status": "running"}])
        ),
    )
    capsys.readouterr()
    argv = ["estimate", "20260101-000000-aaaaaa", "--minutes", "150", "--host", "local", "--json"]
    assert main(argv) == 0
    warnings = one_document(capsys)["warnings"]
    assert isinstance(warnings, list) and "requeue" in warnings[0]


@pytest.mark.parametrize(
    "flags", [[], ["--minutes", "5", "--clear"], ["--minutes", "0"], ["--minutes", "1e10"]]
)
def test_estimate_refuses_a_bad_invocation_before_asking_any_host(
    control_env: Path, capsys: pytest.CaptureFixture[str], flags: list[str]
) -> None:
    main(["host", "add", "local", "--gpus", GPU])
    capsys.readouterr()
    assert main(["estimate", "20260101-000000-aaaaaa", "--host", "local", *flags]) == EXIT_USAGE


def test_pods_json_separates_ours_from_everyone_elses(
    control_env: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RUNPOD_API_KEY", "test-key")
    provider = FakeProvider(existing=[running_pod("subrep-other", "podF")])
    provider.adopt(running_pod("gpuc-e2e-aaa", "pod1"))
    monkeypatch.setattr("gpuc.control.cli.make_provider", lambda settings: provider)
    capsys.readouterr()
    assert main(["pods", "--no-heartbeat", "--json"]) == 0
    document = one_document(capsys)
    (pod,) = document["pods"]  # type: ignore[misc]
    assert pod["name"] == "gpuc-e2e-aaa"
    assert pod["desired"] is False
    assert pod["heartbeat_age_s"] is None
    assert document["others"] == [{"id": "podF", "name": "subrep-other", "status": "RUNNING"}]


def test_reconcile_once_json_reports_the_error_it_exits_one_for(
    control_env: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The commentary it would print goes to stderr; stdout is the document."""
    monkeypatch.setenv("RUNPOD_API_KEY", "test-key")
    monkeypatch.setattr("gpuc.control.cli.make_provider", lambda settings: FakeProvider())
    capsys.readouterr()
    assert main(["reconcile", "--once", "--json"]) == 1
    captured = capsys.readouterr()
    document = json.loads(captured.out)
    assert document["terminated"] == []
    assert "does not exist" in document["errors"][0]
    assert "does not exist" in captured.err


@pytest.mark.parametrize("extra", [[], ["--install"], ["--once", "--install"]])
def test_reconcile_json_needs_once_and_nothing_else(
    control_env: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    extra: list[str],
) -> None:
    """`--install` writes unit files and a systemd blurb, which is not a document."""
    monkeypatch.setenv("RUNPOD_API_KEY", "test-key")
    units = tmp_path / "systemd"
    monkeypatch.setattr(reconcile_mod, "systemd_dir", lambda: units)
    assert main(["reconcile", "--json", *extra]) == EXIT_USAGE
    assert "--once" in json.loads(capsys.readouterr().out)["error"]
    assert not units.exists()


def test_clean_json_carries_what_went_and_what_was_kept(
    control_env: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from gpuc.control.clean import CleanReport

    main(["host", "add", "local", "--gpus", GPU])
    report = CleanReport(
        host="local",
        removed=[{"job_id": "a", "status": "succeeded", "bytes": 2048, "age_days": 9.0}],
        skipped=[{"job_id": "b", "why": "still running"}],
        freed_bytes=2048,
    )
    monkeypatch.setattr("gpuc.control.cli.clean_host", lambda *a, **k: report)
    capsys.readouterr()
    assert main(["clean", "--host", "local", "--all-finished", "--json"]) == 0
    document = one_document(capsys)
    assert document["freed_bytes"] == 2048
    assert document["removed"] == report.removed
    assert document["skipped"] == report.skipped
    assert document["errors"] == []


def test_clean_json_still_exits_one_when_the_host_reported_errors(
    control_env: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from gpuc.control.clean import CleanReport

    main(["host", "add", "local", "--gpus", GPU])
    monkeypatch.setattr(
        "gpuc.control.cli.clean_host",
        lambda *a, **k: CleanReport(host="local", errors=["could not remove workdir"]),
    )
    capsys.readouterr()
    assert main(["clean", "--host", "local", "--all-finished", "--json"]) == 1
    assert one_document(capsys)["errors"] == ["could not remove workdir"]


def test_host_probe_json_keeps_the_raw_sections_and_the_notes(
    control_env: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from gpuc.control.probe import parse_probe

    main(["host", "add", "gpubox", "--ssh", "me@box"])
    sample = (
        "===driver===\n580.173.02\n"
        "===gpus===\n0, GPU-1111, NVIDIA A40, 46068 MiB\n"
        "===uv===\nnot installed\n"
    )
    monkeypatch.setattr(
        "gpuc.control.cli.probe_host", lambda *a, **k: parse_probe("gpubox", sample)
    )
    capsys.readouterr()
    assert main(["host", "probe", "gpubox", "--json"]) == 0
    document = one_document(capsys)
    assert document["driver_version"] == "580.173.02"
    assert document["has_nvidia_smi"] is True
    assert document["sections"]["uv"] == "not installed"  # type: ignore[index]
    assert document["gpus"] == [
        {
            "uuid": "GPU-1111",
            "name": "NVIDIA A40",
            "vram_mib": 46068,
            "index": 0,
            "assigned": False,
        }
    ]
    assert any("uv is missing" in note for note in document["notes"])  # type: ignore[union-attr]


def test_host_probe_shows_only_assigned_gpus_unless_all_gpus_is_asked_for(
    control_env: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from gpuc.control.config import load_registry
    from gpuc.control.probe import parse_probe

    main(["host", "add", "gpubox", "--ssh", "me@box", "--gpus", "1"])
    sample = (
        "===driver===\n580.173.02\n"
        "===gpus===\n0, GPU-1111, NVIDIA A40, 46068 MiB\n1, GPU-2222, NVIDIA A40, 46068 MiB\n"
    )
    monkeypatch.setattr(
        "gpuc.control.cli.probe_host",
        lambda entry, *a, **k: parse_probe(entry.name, sample, entry.root, entry.gpus),
    )

    capsys.readouterr()
    assert main(["host", "probe", "gpubox"]) == 0
    default = capsys.readouterr().out
    assert "1 of 2 assigned to gpubox" in default
    assert "GPU-2222" in default and "GPU-1111" not in default

    assert main(["host", "probe", "gpubox", "--all-gpus"]) == 0
    everything = capsys.readouterr().out
    assert "GPU-1111" in everything
    assert "GPU-2222  NVIDIA A40  46068 MiB  (assigned)" in everything

    # Both cards are recorded either way, so `host set --gpus 0` can name one.
    assert set(load_registry().hosts["gpubox"].gpu_info) == {"GPU-1111", "GPU-2222"}


def test_host_probe_json_is_written_after_the_registry_write_it_can_fail_on(
    control_env: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failing write prints an error document, and stdout may hold only one."""
    from gpuc.control.config import ConfigError
    from gpuc.control.probe import parse_probe

    main(["host", "add", "gpubox", "--ssh", "me@box"])
    monkeypatch.setattr(
        "gpuc.control.cli.probe_host",
        lambda *a, **k: parse_probe(
            "gpubox", "===driver===\n580.173.02\n===gpus===\n0, GPU-1111, NVIDIA A40, 46068 MiB\n"
        ),
    )

    def locked() -> object:
        raise ConfigError("state lock is held by another gpuc")

    monkeypatch.setattr("gpuc.control.cli.registry_transaction", locked)
    capsys.readouterr()
    assert main(["host", "probe", "gpubox", "--json"]) == 1
    document = one_document(capsys)
    assert "state lock" in str(document["error"])


def test_host_list_json_reports_the_host_env_by_name_only(
    control_env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`--env` is where somebody hand-sets a token, and this document travels."""
    main(["host", "add", "local", "--gpus", GPU, "--env", "HF_TOKEN=hf_secret"])
    capsys.readouterr()
    assert main(["host", "list", "--json"]) == 0
    document = one_document(capsys)
    (host,) = document["hosts"]  # type: ignore[misc]
    assert host["env"] == {"HF_TOKEN": "<set>"}
    assert "hf_secret" not in json.dumps(document)


def bootstrapping(
    monkeypatch: pytest.MonkeyPatch, *, fail: dict[str, BaseException] | None = None
) -> list[tuple[str, object]]:
    """Record every (host, health_args) bootstrap was asked for; raise for the named hosts."""
    attempted: list[tuple[str, object]] = []

    def fake(entry: HostEntry, settings: Settings | None = None, **kwargs: Any):
        attempted.append((entry.name, kwargs.get("health_args")))
        error = (fail or {}).get(entry.name)
        if error is not None:
            raise error
        return fake_bootstrap(entry, settings, **kwargs)

    monkeypatch.setattr("gpuc.control.cli.bootstrap_host", fake)
    return attempted


def test_host_bootstrap_all_does_every_registered_host(
    control_env: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    main(["host", "add", "local", "--gpus", GPU])
    main(["host", "add", "gpubox", "--ssh", "me@box"])
    attempted = bootstrapping(monkeypatch)
    capsys.readouterr()

    assert main(["host", "bootstrap", "--all", "--health-args", "--min-mbps 0.1"]) == 0
    # The registry reads back in name order, which is the order `host list` shows.
    assert attempted == [("gpubox", "--min-mbps 0.1"), ("local", "--min-mbps 0.1")]
    out = capsys.readouterr().out
    assert "== gpubox (1/2) ==" in out
    assert "== local (2/2) ==" in out
    assert "2/2 host(s) bootstrapped" in out
    registry = load_registry()
    assert all(registry.require(name).bootstrapped_at for name in ("local", "gpubox"))


def test_host_bootstrap_all_carries_on_past_a_host_that_fails(
    control_env: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """One unreachable pod must not cost the upgrade of every other host."""
    from gpuc.control.bootstrap import BootstrapError

    main(["host", "add", "gpubox", "--ssh", "me@box"])
    main(["host", "add", "zbox", "--ssh", "me@zbox"])
    with registry_transaction() as registry:
        registry.put(HostEntry(name="pod", kind="runpod", ssh="root@1.2.3.4", pod_id="p1"))
    attempted = bootstrapping(monkeypatch, fail={"pod": BootstrapError("ssh to pod failed")})
    capsys.readouterr()

    assert main(["host", "bootstrap", "--all"]) == 1
    assert [name for name, _ in attempted] == ["gpubox", "pod", "zbox"]
    captured = capsys.readouterr()
    assert "error: host pod: ssh to pod failed" in captured.err
    assert "2/3 host(s) bootstrapped" in captured.out
    assert "failed: pod" in captured.out
    # The one failure that is somebody else's job to clean up says whose.
    assert "gpuc reconcile --once" in captured.out
    registry = load_registry()
    assert all(registry.require(name).bootstrapped_at for name in ("gpubox", "zbox"))
    assert registry.require("pod").bootstrapped_at is None


def test_host_bootstrap_all_counts_the_hosts_it_could_not_read(
    control_env: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A host this build cannot parse was not bootstrapped either: never say "all"."""
    main(["host", "add", "gpubox", "--ssh", "me@box"])
    document = json.loads(hosts_file().read_text())
    document["hosts"]["bad"] = {"name": "bad", "kind": "not a kind", "port": "twenty-two"}
    hosts_file().write_text(json.dumps(document))
    bootstrapping(monkeypatch)
    capsys.readouterr()

    assert main(["host", "bootstrap", "--all"]) == 0
    captured = capsys.readouterr()
    assert "skipping host 'bad'" in captured.err
    assert "1/1 host(s) bootstrapped" in captured.out
    assert "1 host(s) in the registry could not be read" in captured.out


def test_host_bootstrap_all_stops_when_the_registry_stops_being_readable(
    control_env: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exit 3, after saying how far it got: the next write would be a guess."""
    main(["host", "add", "gpubox", "--ssh", "me@box"])
    main(["host", "add", "local", "--gpus", GPU])
    attempted = bootstrapping(
        monkeypatch, fail={"local": LocalStateUnreadable("hosts.json is not json")}
    )
    capsys.readouterr()

    assert main(["host", "bootstrap", "--all"]) == EXIT_LOCAL_STATE
    assert [name for name, _ in attempted] == ["gpubox", "local"]
    captured = capsys.readouterr()
    assert "1/2 host(s) bootstrapped" in captured.out
    assert "hosts.json is not json" in captured.err


def test_host_bootstrap_all_interrupted_still_says_what_it_got_through(
    control_env: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Five minutes a host for health alone makes this one people do Ctrl-C."""
    main(["host", "add", "gpubox", "--ssh", "me@box"])
    main(["host", "add", "local", "--gpus", GPU])
    bootstrapping(monkeypatch, fail={"local": KeyboardInterrupt()})
    capsys.readouterr()

    assert main(["host", "bootstrap", "--all"]) == 1
    out = capsys.readouterr().out
    assert "interrupted during local" in out
    assert "1/2 host(s) bootstrapped" in out


def test_host_bootstrap_all_with_no_hosts_is_not_an_error(
    control_env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["host", "bootstrap", "--all"]) == 0
    assert "no hosts registered" in capsys.readouterr().out


def test_host_bootstrap_wants_a_name_or_all_but_not_both(
    control_env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    main(["host", "add", "local", "--gpus", GPU])
    assert main(["host", "bootstrap"]) == EXIT_USAGE
    assert "--all" in capsys.readouterr().err
    assert main(["host", "bootstrap", "local", "--all"]) == EXIT_USAGE
    assert "not both" in capsys.readouterr().err
