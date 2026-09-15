from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from gpuc.control import reconcile as reconcile_mod
from gpuc.control.clean import purge_host
from gpuc.control.cli import EXIT_NOT_FOUND, EXIT_USAGE, build_parser, main
from gpuc.control.config import (
    HostEntry,
    Settings,
    config_file,
    load_registry,
    load_settings,
    registry_transaction,
)
from gpuc.control.providers.base import Constraints
from gpuc.control.remote import HostSession
from gpuc.control.submit import SubmitResult
from tests.fakeprovider import FakeProvider, running_pod
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
    main(["host", "add", "spar", "--ssh", "me@box", "--port", "2222", "--gpus", "GPU-a,GPU-b"])
    entry = load_registry().require("spar")
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


def test_reconcile_install_writes_units_without_enabling_them(
    control_env: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    assert main(["reconcile", "--install", "--interval", "45"]) == 0
    out = capsys.readouterr().out
    assert "gpuc-reconcile.timer" in out and "systemctl --user enable --now" in out


def test_pods_lists_ours_and_counts_the_others(
    control_env: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RUNPOD_API_KEY", "test-key")
    provider = FakeProvider(existing=[running_pod("subrep-other", "podF")])
    provider.adopt(running_pod("gpuc-e2e-aaa", "pod1"))
    monkeypatch.setattr("gpuc.control.cli.make_provider", lambda settings: provider)
    assert main(["pods", "--no-heartbeat"]) == 0
    out = capsys.readouterr().out
    assert "gpuc-e2e-aaa" in out
    assert "1 other pod(s) in the account, never touched: subrep-other (RUNNING)" in out


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

    def host_json(self, args: str, *, timeout: float = 0.0) -> object:
        self.calls.append(args)
        return self.payloads.pop(0)


def purged_entry(job_id: str, prefix: str | None = "s3://bucket/gpuc/spar") -> dict[str, object]:
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
    client = FakeS3Client(objects={"bucket/gpuc/spar/jobs/kept/log.txt": b"hello\n"})
    entry = HostEntry(name="spar", kind="ssh", ssh="me@spar", python="/usr/bin/python3")
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

    entry = HostEntry(name="spar", kind="ssh", ssh="me@spar", python="/usr/bin/python3")
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
    entry = HostEntry(name="spar", kind="ssh", ssh="me@spar", python="/usr/bin/python3")
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
    main(["host", "add", "spar", "--ssh", "me@box"])
    assert main(["clean", "--host", "spar", "--verify", "--all-finished"]) == EXIT_USAGE
    assert "only mean something with --purge" in capsys.readouterr().err


def test_clean_with_no_selection_and_no_purge_exits(
    control_env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    main(["host", "add", "spar", "--ssh", "me@box"])
    assert main(["clean", "--host", "spar"]) == EXIT_USAGE
    assert "--all-finished" in capsys.readouterr().err


def test_retention_days_is_stored_and_cleared(control_env: Path) -> None:
    assert main(["host", "add", "spar", "--ssh", "me@box", "--retention-days", "14"]) == 0
    assert load_registry().require("spar").retention_days == 14.0
    assert load_registry().require("spar").host_config().retention_days == 14.0
    assert main(["host", "set", "spar", "--retention-days", ""]) == 0
    assert load_registry().require("spar").retention_days is None


def test_a_bad_retention_value_is_rejected(
    control_env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert (
        main(["host", "add", "spar", "--ssh", "me@box", "--retention-days", "soon"]) == EXIT_USAGE
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
    main(["host", "add", "spar", "--ssh", "me@box", "--gpus", GPU])
    _set_host(python="/py", pkg_commit="a" * 40)
    monkeypatch.setattr("gpuc.control.cli.version_mod.local_commit", lambda: "b" * 40)
    resynced: list[str] = []

    def fake_resync(entry: HostEntry, settings: object = None, **kwargs: object) -> HostEntry:
        resynced.append(entry.name)
        return entry.model_copy(update={"pkg_commit": "b" * 40})

    monkeypatch.setattr("gpuc.control.cli.resync_package", fake_resync)
    monkeypatch.setattr(
        "gpuc.control.cli.submit_file",
        lambda *a, **k: SubmitResult(job_id="j", host="spar", attempt=1),
    )

    assert main(["submit", str(job), "--host", "spar"]) == 0
    out = capsys.readouterr().out
    assert resynced == ["spar"]
    assert sum("re-syncing the package" in line for line in out.splitlines()) == 1
    assert load_registry().require("spar").pkg_commit == "b" * 40

    # Now the host is on this build, so nothing is shipped.
    resynced.clear()
    assert main(["submit", str(job), "--host", "spar"]) == 0
    assert resynced == []

    # An unrecorded commit counts as different: those hosts are the oldest.
    _set_host(pkg_commit=None)
    assert main(["submit", str(job), "--host", "spar", "--no-bootstrap"]) == 0
    assert resynced == []
    assert main(["submit", str(job), "--host", "spar"]) == 0
    assert resynced == ["spar"]


def _set_host(**changes: object) -> None:
    with registry_transaction() as registry:
        registry.put(registry.require("spar").model_copy(update=changes))


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
    main(["host", "add", "spar", "--ssh", "me@box", "--gpus", GPU])
    assert main(["submit", str(job), "--runpod", "--gpu", "A40", "--host", "spar"]) == EXIT_USAGE
    assert main(["requeue", "job-1", "--host", "spar", "--runpod", "--gpu", "A40"]) == EXIT_USAGE


def test_requeue_of_an_unknown_job_is_exit_four(
    control_env: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    (Path(control_env) / "config/config.toml").write_text('s3_bucket = "bucket"\n')
    monkeypatch.setattr(
        "gpuc.control.s3index.S3Index.client", property(lambda self: FakeS3Client())
    )
    main(["host", "add", "spar", "--ssh", "me@box", "--gpus", GPU])
    assert main(["requeue", "20260101-000000-nosuch", "--host", "spar"]) == EXIT_NOT_FOUND
    assert "no mirrored spec for job" in capsys.readouterr().err


def test_purging_every_finished_job_has_to_be_asked_for_twice(
    control_env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`--purge --all-finished` is an age horizon of 0: it deletes the job dir
    of something that ended a minute ago, log and all."""
    main(["host", "add", "spar", "--ssh", "me@box"])
    assert main(["clean", "--host", "spar", "--purge", "--all-finished"]) == EXIT_USAGE
    assert "Add --yes to confirm" in capsys.readouterr().err


def test_status_filters_have_defaults() -> None:
    args = build_parser().parse_args(["status"])
    assert (args.recent, args.since) == (5, None)


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
