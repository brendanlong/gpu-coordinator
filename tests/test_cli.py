from __future__ import annotations

import json
from pathlib import Path

import pytest

from gpuc.control.cli import main
from gpuc.control.config import HostEntry, Settings, load_registry
from gpuc.control.providers.base import Constraints
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
    assert main(["host", "bootstrap", "nope"]) == 1
    assert "no host named 'nope'" in capsys.readouterr().err


def test_submit_without_a_host_says_which_flag_is_missing(
    control_env: Path, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    job = tmp_path / "job.yaml"
    job.write_text("command: true\n")
    assert main(["submit", str(job)]) == 1
    assert "--host" in capsys.readouterr().err


def test_status_with_no_hosts_is_not_an_error(
    control_env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["status"]) == 0
    assert "no hosts registered" in capsys.readouterr().out


def test_cancel_for_an_unknown_job_tells_you_where_to_look(
    control_env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["cancel", "20260101-000000-aaaaaa"]) == 1
    assert "no registered host knows job" in capsys.readouterr().err


def test_requeue_without_an_s3_bucket_explains_the_gap(
    control_env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    main(["host", "add", "local", "--gpus", GPU])
    assert main(["requeue", "20260101-000000-aaaaaa", "--host", "local"]) == 1
    assert "s3_bucket is unset" in capsys.readouterr().err


def test_submit_runpod_without_a_gpu_name_says_which_flag_is_missing(
    control_env: Path, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    job = tmp_path / "job.yaml"
    job.write_text('command: "true"\n')
    assert main(["submit", str(job), "--runpod"]) == 1
    assert "--gpu <name>" in capsys.readouterr().err


def test_submit_runpod_passes_the_flags_through_and_mirrors_the_spec_first(
    control_env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = tmp_path / "job.yaml"
    job.write_text('command: "true"\ngpus: 1\n')
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
        return SubmitResult(job_id=str(kwargs["job_id"]), host=entry.name, attempt=1, files=0)

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
        return SubmitResult(job_id="new", host=entry.name, attempt=2, files=0)

    monkeypatch.setattr("gpuc.control.cli.runpod_host", fake_runpod_host)
    monkeypatch.setattr("gpuc.control.cli.submit_spec", fake_submit_spec)

    assert main(["requeue", "20260101-000000-aaaaaa", "--runpod", "--gpu", "A40"]) == 0
    assert seen == {"gpu_names": ["A40"], "attempt": 2}
