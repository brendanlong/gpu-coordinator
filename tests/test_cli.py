from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from gpuc.control import reconcile as reconcile_mod
from gpuc.control.clean import purge_host
from gpuc.control.cli import (
    EXIT_ERROR,
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
    read_desired,
    registry_transaction,
)
from gpuc.control.providers.base import Constraints
from gpuc.control.remote import HostSession, RemoteError
from gpuc.control.status import placement_unknown
from gpuc.control.submit import SubmitResult
from tests.conftest import host_entry, register_host
from tests.fakehost import FakeHost
from tests.fakeprovider import FakeProvider, fake_bootstrap, running_pod
from tests.fakes3 import FakeS3Client

GPU = "GPU-2a4bad3b-9fe3-7031-914d-384254e92908"


def test_host_add_writes_the_first_config_of_a_host_that_has_none(
    control_env: Path, fake_host: FakeHost, capsys: pytest.CaptureFixture[str]
) -> None:
    """The host owns its config, so `add` is where a host that has none gets
    one -- and the only place this machine decides what a host is."""
    assert main(["host", "add", "local", "--gpus", GPU]) == 0
    entry = load_registry().require("local")
    assert (entry.kind, entry.gpus, entry.ssh) == ("local", [GPU], None)
    assert fake_host.config is not None
    assert (fake_host.config["host"], fake_host.config["gpus"]) == ("local", [GPU])
    assert fake_host.config["created_at"]
    out = capsys.readouterr().out
    assert "wrote its first config" in out
    assert "gpuc host bootstrap local" in out


def test_host_add_adopts_the_config_a_host_already_has(
    control_env: Path, fake_host: FakeHost, capsys: pytest.CaptureFixture[str]
) -> None:
    """The second control machine's path, and the ordinary one: what the host
    is was decided by whoever set it up, and this machine takes it as it is."""
    theirs = {
        "host": "gpubox",
        "gpus": ["2", "3"],
        "s3_prefix": "s3://theirs/gpuc/gpubox",
        "retention_days": 30.0,
        "env": {"HF_HOME": "/big"},
    }
    fake_host.put_file(json.dumps(theirs), "/home/u/.gpuc/config.json")
    assert main(["host", "add", "gpubox", "--ssh", "me@box"]) == 0
    entry = load_registry().require("gpubox")
    assert entry.gpus == ["2", "3"]
    assert entry.s3_prefix == "s3://theirs/gpuc/gpubox"
    assert entry.retention_days == 30.0
    assert entry.env == {"HF_HOME": "/big"}
    assert entry.seen_at
    assert fake_host.config == theirs  # nothing was written to the host
    assert "adopted the config on the host" in capsys.readouterr().out


def test_host_add_registers_a_host_under_the_name_it_calls_itself(
    control_env: Path, fake_host: FakeHost, capsys: pytest.CaptureFixture[str]
) -> None:
    """`config.json`'s `host` is what the host answers as, and what its
    `s3_prefix` was derived from; a second local name for it would split the
    two machines' view of one box in half."""
    fake_host.put_file('{"host": "gpubox", "gpus": ["0"]}', "/home/u/.gpuc/config.json")
    assert main(["host", "add", "other-name", "--ssh", "me@box"]) == 0
    assert set(load_registry().hosts) == {"gpubox"}
    assert "calls itself 'gpubox'" in capsys.readouterr().out


def test_a_flag_on_an_adopted_host_is_an_override_and_says_so(
    control_env: Path, fake_host: FakeHost, capsys: pytest.CaptureFixture[str]
) -> None:
    fake_host.put_file(
        '{"host": "gpubox", "gpus": ["0"], "retention_days": 30.0}', "/home/u/.gpuc/config.json"
    )
    assert main(["host", "add", "gpubox", "--ssh", "me@box", "--retention-days", "7"]) == 0
    assert "host <- retention_days 30.0 -> 7.0" in capsys.readouterr().out
    assert fake_host.config is not None and fake_host.config["retention_days"] == 7.0
    assert fake_host.config["gpus"] == ["0"]


def test_host_add_refuses_a_gpu_list_that_overlaps_the_hosts_own(
    control_env: Path, fake_host: FakeHost, capsys: pytest.CaptureFixture[str]
) -> None:
    """Two machines each believing they own part of an overlapping share hand
    one card to two jobs, which no warning would undo."""
    fake_host.put_file('{"host": "gpubox", "gpus": ["0", "1"]}', "/home/u/.gpuc/config.json")
    assert main(["host", "add", "gpubox", "--ssh", "me@box", "--gpus", "1,2"]) == EXIT_ERROR
    assert "shares 1 with that list without matching it" in capsys.readouterr().err
    assert load_registry().hosts == {}
    assert fake_host.config is not None and fake_host.config["gpus"] == ["0", "1"]
    # A disjoint list is a deliberate reassignment, and goes through.
    assert main(["host", "add", "gpubox", "--ssh", "me@box", "--gpus", "2,3"]) == 0
    assert fake_host.config["gpus"] == ["2", "3"]


def test_host_set_writes_the_shared_cards(
    control_env: Path, fake_host: FakeHost, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["host", "add", "gpubox", "--ssh", "me@box", "--gpus", "0"]) == 0
    capsys.readouterr()
    assert main(["host", "set", "gpubox", "--shared-gpus", "1"]) == 0
    assert fake_host.config is not None and fake_host.config["shared_gpus"] == ["1"]
    assert "shared_gpus" in capsys.readouterr().out
    # Clearable, which is the point of taking it as a string.
    assert main(["host", "set", "gpubox", "--shared-gpus", ""]) == 0
    assert fake_host.config["shared_gpus"] == []


def test_a_card_cannot_be_both_owned_and_shared(
    control_env: Path, fake_host: FakeHost, capsys: pytest.CaptureFixture[str]
) -> None:
    """The two lists say opposite things about a card, and the refusal fires
    where it was typed -- in either spelling, and against what the host already
    holds rather than only against the flags in one command."""
    assert main(["host", "add", "gpubox", "--ssh", "me@box", "--gpus", "0"]) == 0
    assert main(["host", "set", "gpubox", "--shared-gpus", "GPU-a"]) == EXIT_ERROR
    assert "both --gpus and --shared-gpus" in capsys.readouterr().err
    assert fake_host.config is not None and fake_host.config["shared_gpus"] == []

    assert main(["host", "set", "gpubox", "--gpus", "0", "--shared-gpus", "0,1"]) == EXIT_ERROR
    assert "both --gpus and --shared-gpus" in capsys.readouterr().err
    assert fake_host.config["shared_gpus"] == []
    # Disjoint is the ordinary case and goes through.
    assert main(["host", "set", "gpubox", "--shared-gpus", "1"]) == 0
    assert fake_host.config["shared_gpus"] == ["1"]


def test_the_gpu_overlap_refusal_sees_through_index_and_uuid_spellings(
    control_env: Path, fake_host: FakeHost, capsys: pytest.CaptureFixture[str]
) -> None:
    """`--gpus 0` and `--gpus GPU-a` can be the same card, and the index is the
    spelling somebody copies off `gpuc host probe`."""
    fake_host.put_file(
        '{"host": "gpubox", "gpus": ["GPU-a", "GPU-b"]}', "/home/u/.gpuc/config.json"
    )
    assert main(["host", "add", "gpubox", "--ssh", "me@box", "--gpus", "0"]) == EXIT_ERROR
    assert "shares GPU-a with that list without matching it" in capsys.readouterr().err
    assert fake_host.config is not None and fake_host.config["gpus"] == ["GPU-a", "GPU-b"]
    # The same two cards by their other name is not a difference at all.
    assert main(["host", "add", "gpubox", "--ssh", "me@box", "--gpus", "0,1"]) == 0
    assert fake_host.config["gpus"] == ["0", "1"]


def test_an_adopted_name_never_replaces_a_different_host_registered_here(
    control_env: Path, fake_host: FakeHost, capsys: pytest.CaptureFixture[str]
) -> None:
    """A box somebody set up as `local` on their own machine calls itself
    `local` here too, and this machine has one of those."""
    register_host(name="local", gpus=GPU)
    fake_host.put_file('{"host": "local", "gpus": ["0"]}', "/home/u/.gpuc/config.json")
    assert main(["host", "add", "desktop", "--ssh", "me@desktop", "--idle-min", "99"]) == EXIT_ERROR
    err = capsys.readouterr().err
    assert "calls itself 'local'" in err
    assert "gpuc host remove local" in err
    assert load_registry().require("local").ssh is None
    assert set(load_registry().hosts) == {"local"}
    # Refused before the flags reached the host, not after.
    assert fake_host.config is not None and "idle_minutes" not in fake_host.config


def test_a_second_machine_registers_the_same_box_and_agrees_with_the_first(
    control_env: Path, fake_host: FakeHost, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole point of the split. One box, two control machines, one
    configuration -- and a `host set` from either is what the other sees."""
    assert main(["host", "add", "gpubox", "--ssh", "me@box", "--gpus", "2,3"]) == 0
    first = load_registry().require("gpubox")

    # The laptop: its own registry, the same host, and no flags to retype.
    monkeypatch.setenv("GPUC_STATE_DIR", str(control_env / "laptop-state"))
    assert main(["host", "add", "gpubox", "--ssh", "me@box"]) == 0
    second = load_registry().require("gpubox")
    assert second.gpus == first.gpus == ["2", "3"]
    assert second.created_at == first.created_at

    # And a change from the laptop is the host's, so the desktop sees it.
    assert main(["host", "set", "gpubox", "--gpus", "2,3,4"]) == 0
    monkeypatch.setenv("GPUC_STATE_DIR", str(control_env / "state"))
    assert main(["host", "probe", "gpubox"]) == 0
    assert load_registry().require("gpubox").gpus == ["2", "3", "4"]


def test_host_set_keeps_the_cache_dir_a_new_env_did_not_mention(
    control_env: Path, fake_host: FakeHost
) -> None:
    """`--env` replaces what somebody set by hand. `UV_CACHE_DIR` is not that:
    bootstrap derives it from the host's own filesystem, and dropping it costs
    every job on that host a full copy of every wheel."""
    fake_host.put_file(
        json.dumps({"host": "gpubox", "gpus": ["0"], "env": {"UV_CACHE_DIR": "/vol/uv"}}),
        "/home/u/.gpuc/config.json",
    )
    assert main(["host", "add", "gpubox", "--ssh", "me@box"]) == 0
    assert main(["host", "set", "gpubox", "--env", "HF_HOME=/big"]) == 0
    assert fake_host.config is not None
    assert fake_host.config["env"] == {"HF_HOME": "/big", "UV_CACHE_DIR": "/vol/uv"}
    # `--cache-dir ''` is what clears it, and says so.
    assert main(["host", "set", "gpubox", "--cache-dir", ""]) == 0
    assert fake_host.config["env"] == {"HF_HOME": "/big"}


def test_host_set_that_changes_nothing_does_not_rewrite_the_hosts_config(
    control_env: Path, fake_host: FakeHost, capsys: pytest.CaptureFixture[str]
) -> None:
    """A dispatcher is reading that file; repeating what it already says is no
    reason to replace it."""
    assert main(["host", "add", "gpubox", "--ssh", "me@box", "--gpus", "0"]) == 0
    before = list(fake_host.commands)
    capsys.readouterr()
    assert main(["host", "set", "gpubox", "--gpus", "0"]) == 0
    assert "nothing changed" in capsys.readouterr().out
    written = [
        c for c in fake_host.commands[len(before) :] if "config --merge" in c or "mv -f" in c
    ]
    assert written == []


def test_host_add_needs_gpus_for_a_host_with_no_config_and_lists_the_cards(
    control_env: Path, fake_host: FakeHost, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["host", "add", "gpubox", "--ssh", "me@box"]) == EXIT_ERROR
    err = capsys.readouterr().err
    assert "has no config of its own yet" in err
    assert "GPU-a" in err  # the probe's own card list, to copy from
    assert load_registry().hosts == {}


def test_host_add_ssh_records_the_target_and_port(control_env: Path) -> None:
    register_host(name="gpubox", kind="ssh", ssh="me@box", port=2222, gpus="GPU-a,GPU-b")
    entry = load_registry().require("gpubox")
    assert (entry.kind, entry.ssh, entry.port) == ("ssh", "me@box", 2222)
    assert entry.gpus == ["GPU-a", "GPU-b"]


def test_host_list_and_remove(control_env: Path, capsys: pytest.CaptureFixture[str]) -> None:
    register_host(name="local", gpus=GPU)
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


def test_use_shared_is_an_override_of_the_spec_and_only_when_it_is_passed(
    control_env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A flag that was not typed is not an opinion: a spec that already says
    `use_shared: true` must keep saying it."""
    seen: list[dict[str, Any]] = []

    def capture(entry: Any, job_file: Any, settings: Any, overrides: Any = None, **_: Any) -> Any:
        seen.append(dict(overrides or {}))
        return SubmitResult(job_id="j", host="gpubox", attempt=1)

    monkeypatch.setattr("gpuc.control.cli.submit_file", capture)
    monkeypatch.setattr("gpuc.control.cli.ensure_package_current", lambda entry, *a, **k: entry)
    monkeypatch.setattr("gpuc.control.cli.placement_after", lambda *a, **k: placement_unknown())
    register_host(name="gpubox", kind="ssh", ssh="me@box", gpus=GPU)
    job = tmp_path / "job.yaml"
    job.write_text('command: "true"\n')

    assert main(["submit", str(job), "--host", "gpubox"]) == 0
    assert main(["submit", str(job), "--host", "gpubox", "--use-shared"]) == 0
    assert seen == [{"use_shared": None}, {"use_shared": True}]


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
    register_host(name="local", gpus=GPU)
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
        return host_entry(name="gpuc-e2e-1", kind="runpod", gpus=["GPU-1"])

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
        return host_entry(name="gpuc-e2e-1", kind="runpod", gpus=["GPU-1"])

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
        ["host", "add", "rented", "--pod", "pod1"],
    ):
        assert main(argv) == 1
        err = capsys.readouterr().err
        assert err.strip().splitlines() == [
            "error: RUNPOD_API_KEY is not set; export it before using --runpod, "
            "`gpuc host add --pod`, `gpuc pods` or `gpuc reconcile`"
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
        lambda *a, **k: created.append(a) or host_entry(name="gpuc-x", kind="runpod"),
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
        lambda *a, **k: created.append(a) or host_entry(name="gpuc-x", kind="runpod"),
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
    entry = host_entry(name="gpubox", kind="ssh", ssh="me@gpubox", python="/usr/bin/python3")
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

    entry = host_entry(name="gpubox", kind="ssh", ssh="me@gpubox", python="/usr/bin/python3")
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
    entry = host_entry(name="gpubox", kind="ssh", ssh="me@gpubox", python="/usr/bin/python3")
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
    register_host(name="gpubox", kind="ssh", ssh="me@box")
    assert main(["clean", "--host", "gpubox", "--verify", "--all-finished"]) == EXIT_USAGE
    assert "only mean something with --purge" in capsys.readouterr().err


def test_clean_with_no_selection_and_no_purge_exits(
    control_env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    register_host(name="gpubox", kind="ssh", ssh="me@box")
    assert main(["clean", "--host", "gpubox"]) == EXIT_USAGE
    assert "--all-finished" in capsys.readouterr().err


# -- clean --only -------------------------------------------------------------


def test_only_purges_the_named_jobs_at_horizon_zero_and_scopes_the_sweep(
    control_env: Path,
) -> None:
    """Naming ids is the confirmation `--purge --all-finished` needs `--yes` for."""
    from gpuc.control.clean import clean_host

    entry = host_entry(name="gpubox", kind="ssh", ssh="me@gpubox", python="/usr/bin/python3")
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

    entry = host_entry(name="gpubox", kind="ssh", ssh="me@gpubox", python="/usr/bin/python3")
    session = StubSession([{"dry_run": False, "removed": [], "freed_bytes": 0}])
    clean_host(entry, Settings(), session=as_session(session), only=["a"])
    assert session.calls == ["clean --only a"]


def test_only_with_verify_purges_what_answered_and_sweeps_what_was_asked(
    control_env: Path,
) -> None:
    client = FakeS3Client(objects={"bucket/gpuc/gpubox/jobs/kept/log.txt": b"hello\n"})
    entry = host_entry(name="gpubox", kind="ssh", ssh="me@gpubox", python="/usr/bin/python3")
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
    entry = host_entry(name="gpubox", kind="ssh", ssh="me@gpubox", python="/usr/bin/python3")
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

    register_host(name="gpubox", kind="ssh", ssh="me@box")
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
    register_host(name="gpubox", kind="ssh", ssh="me@box")
    assert main(["clean", "--host", "gpubox", "--purge", "--only", " , "]) == EXIT_USAGE
    assert "at least one job id" in capsys.readouterr().err


def test_retention_days_is_stored_and_cleared(control_env: Path, fake_host: FakeHost) -> None:
    add = ["host", "add", "gpubox", "--ssh", "me@box", "--gpus", "", "--retention-days", "14"]
    assert main(add) == 0
    assert load_registry().require("gpubox").retention_days == 14.0
    assert fake_host.config is not None and fake_host.config["retention_days"] == 14.0
    assert main(["host", "set", "gpubox", "--retention-days", ""]) == 0
    assert load_registry().require("gpubox").retention_days is None
    assert fake_host.config["retention_days"] is None


def test_a_bad_retention_value_is_rejected(
    control_env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert (
        main(["host", "add", "gpubox", "--ssh", "me@box", "--retention-days", "soon"]) == EXIT_USAGE
    )
    assert "wants a number of days" in capsys.readouterr().err


def test_a_host_getting_its_first_config_sweeps_workdirs_after_a_day(
    control_env: Path, fake_host: FakeHost
) -> None:
    assert main(["host", "add", "gpubox", "--ssh", "me@box", "--gpus", ""]) == 0
    assert fake_host.config is not None and fake_host.config["workdir_days"] == 1.0
    assert load_registry().require("gpubox").workdir_days == 1.0
    # ...and it is the only horizon that is on without being asked for.
    assert fake_host.config["retention_days"] is None


def test_an_adopted_config_is_not_given_a_sweep_it_never_had(
    control_env: Path, fake_host: FakeHost
) -> None:
    """A host that has been getting along without one: meeting it is not the
    moment to start deleting there."""
    fake_host.put_file('{"host": "gpubox", "gpus": ["0"]}', "/home/u/.gpuc/config.json")
    assert main(["host", "add", "gpubox", "--ssh", "me@box"]) == 0
    assert fake_host.config is not None
    assert fake_host.config.get("workdir_days") is None


def test_workdir_days_is_stored_and_cleared(control_env: Path, fake_host: FakeHost) -> None:
    add = ["host", "add", "gpubox", "--ssh", "me@box", "--gpus", "", "--workdir-days", "3.5"]
    assert main(add) == 0
    assert fake_host.config is not None and fake_host.config["workdir_days"] == 3.5
    assert main(["host", "set", "gpubox", "--workdir-days", "0"]) == 0
    # Zero is a real horizon -- "reclaim it as soon as it finishes" -- and must
    # not be confused with the empty string that turns the sweep off.
    assert fake_host.config["workdir_days"] == 0.0
    assert load_registry().require("gpubox").workdir_days == 0.0
    assert main(["host", "set", "gpubox", "--workdir-days", ""]) == 0
    assert fake_host.config["workdir_days"] is None
    assert load_registry().require("gpubox").workdir_days is None


def test_a_bad_workdir_days_value_is_rejected(
    control_env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["host", "add", "gpubox", "--ssh", "me@box", "--workdir-days", "-1"]) == EXIT_USAGE
    assert "--workdir-days cannot be negative" in capsys.readouterr().err


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
        registry.put(host_entry(name="h", gpus=[], ttl_hours=6.0))
    assert main(["host", "set", "h", "--ttl-hours", "-1"]) == 0
    assert load_registry().hosts["h"].ttl_hours is None


def _fake_host_build(monkeypatch: pytest.MonkeyPatch, config: dict[str, Any] | None) -> list[str]:
    """Answer `submit`'s "what build is this host running" with `config`.

    ``None`` is a host that could not be asked at all, and it is faked at the
    ssh boundary -- `open_session` raising, which is what an unreachable host
    really does -- so that the fallback *and* the catching are both exercised.
    Returns the list every re-ship appends its host to.
    """
    resynced: list[str] = []
    fake_session = SimpleNamespace(
        transport=SimpleNamespace(host="gpubox"), read_config=lambda: config
    )

    def fake_resync(entry: HostEntry, settings: object = None, **kwargs: object) -> HostEntry:
        resynced.append(entry.name)
        # The session opened to ask the host is the one the re-ship rides on:
        # a second `open_session` here would be a second ssh handshake.
        assert kwargs.get("transport") is fake_session.transport
        # As the real one does: the host's config, or the last one seen where
        # the host has none, plus the commit just shipped.
        restored = config or entry.initial_config().to_dict()
        return entry.with_config({**restored, "pkg_commit": "b" * 40})

    monkeypatch.setattr("gpuc.control.cli.resync_package", fake_resync)

    def open_or_fail(*_: object, **__: object) -> Any:
        if config is None:
            raise RemoteError("gpubox", "printf %s", "could not reach host gpubox")
        return fake_session

    monkeypatch.setattr("gpuc.control.cli.open_session", open_or_fail)
    monkeypatch.setattr(
        "gpuc.control.cli.submit_file",
        lambda *a, **k: SubmitResult(job_id="j", host="gpubox", attempt=1),
    )
    return resynced


def test_submit_reships_the_package_to_a_host_on_another_commit(
    control_env: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A host on other code dispatches the job with code that does not match
    the spec this machine just wrote."""
    job = tmp_path / "job.yaml"
    job.write_text('command: "true"\n')
    register_host(name="gpubox", kind="ssh", ssh="me@box", gpus=GPU)
    _set_host(python="/py", pkg_commit="a" * 40)
    monkeypatch.setattr("gpuc.control.cli.version_mod.local_commit", lambda: "b" * 40)
    host_config: dict[str, Any] = {"pkg_commit": "a" * 40}
    resynced = _fake_host_build(monkeypatch, host_config)

    assert main(["submit", str(job), "--host", "gpubox"]) == 0
    out = capsys.readouterr().out
    assert resynced == ["gpubox"]
    assert sum("re-syncing the package" in line for line in out.splitlines()) == 1
    assert load_registry().require("gpubox").pkg_commit == "b" * 40

    # Now the host is on this build, so nothing is shipped.
    resynced.clear()
    host_config["pkg_commit"] = "b" * 40
    assert main(["submit", str(job), "--host", "gpubox"]) == 0
    assert resynced == []

    # An unrecorded commit counts as different: those hosts are the oldest.
    del host_config["pkg_commit"]
    assert main(["submit", str(job), "--host", "gpubox", "--no-bootstrap"]) == 0
    assert resynced == []
    assert main(["submit", str(job), "--host", "gpubox"]) == 0
    assert resynced == ["gpubox"]


def test_submit_asks_the_host_which_build_it_runs_not_this_machines_record(
    control_env: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The registry only ever recorded what *this* machine shipped.

    Register the same box from a laptop too and that record describes a host
    the laptop has since re-bootstrapped: both machines would then agree with
    themselves and skip the check forever.
    """
    job = tmp_path / "job.yaml"
    job.write_text('command: "true"\n')
    register_host(name="gpubox", kind="ssh", ssh="me@box", gpus=GPU)
    _set_host(python="/py", pkg_commit="b" * 40)
    monkeypatch.setattr("gpuc.control.cli.version_mod.local_commit", lambda: "b" * 40)
    resynced = _fake_host_build(monkeypatch, {"pkg_commit": "c" * 40})

    assert main(["submit", str(job), "--host", "gpubox"]) == 0
    assert resynced == ["gpubox"]
    assert "host gpubox is running gpuc " + "c" * 12 in capsys.readouterr().out
    # The fake re-ship records what it shipped, as the real one does.
    assert load_registry().require("gpubox").pkg_commit == "b" * 40

    # And when there is nothing to re-ship, the record still stops repeating a
    # bootstrap somebody else replaced: `host list` and `version` have only it.
    resynced.clear()
    monkeypatch.setattr("gpuc.control.cli.version_mod.local_commit", lambda: "c" * 40)
    assert main(["submit", str(job), "--host", "gpubox"]) == 0
    assert resynced == []
    assert load_registry().require("gpubox").pkg_commit == "c" * 40


def test_submit_records_the_hosts_commit_without_clobbering_the_rest_of_the_entry(
    control_env: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """It runs on every submit, so it writes the one field under the lock: the
    entry it read at startup is stale the moment a `gpuc host probe` in another
    session writes what it learned about the same host."""
    job = tmp_path / "job.yaml"
    job.write_text('command: "true"\n')
    register_host(name="gpubox", kind="ssh", ssh="me@box", gpus=GPU)
    _set_host(python="/py", pkg_commit="b" * 40)
    monkeypatch.setattr("gpuc.control.cli.version_mod.local_commit", lambda: "c" * 40)
    _fake_host_build(monkeypatch, {"pkg_commit": "c" * 40})

    def concurrent_probe() -> dict[str, Any]:
        # Between reading the entry and recording what the host said, another
        # session records a driver version against the same host.
        _set_host(driver_version="580.173.02")
        return {"pkg_commit": "c" * 40}

    monkeypatch.setattr(
        "gpuc.control.cli.open_session",
        lambda *a, **k: SimpleNamespace(
            transport=SimpleNamespace(host="gpubox"), read_config=concurrent_probe
        ),
    )
    assert main(["submit", str(job), "--host", "gpubox"]) == 0
    entry = load_registry().require("gpubox")
    assert (entry.pkg_commit, entry.driver_version) == ("c" * 40, "580.173.02")


def test_submit_takes_the_hosts_config_as_it_finds_it(
    control_env: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The host owns its config, so a cache that disagrees with it is stale,
    not a conflict: the read before the enqueue is what the job is judged by,
    and it is what the registry then holds."""
    job = tmp_path / "job.yaml"
    job.write_text('command: "true"\n')
    register_host(name="gpubox", kind="ssh", ssh="me@box", gpus="2,3", retention_days=7.0)
    _set_host(python="/py", pkg_commit="b" * 40)
    monkeypatch.setattr("gpuc.control.cli.version_mod.local_commit", lambda: "b" * 40)
    theirs = {
        "pkg_commit": "b" * 40,
        "gpus": ["0", "1"],
        "s3_prefix": "s3://theirs/gpuc/gpubox",
        "retention_days": None,
    }
    resynced = _fake_host_build(monkeypatch, theirs)

    assert main(["submit", str(job), "--host", "gpubox"]) == 0
    out = capsys.readouterr().out
    assert resynced == []
    # Nothing to warn about, and nothing to reconcile by hand.
    assert "WARNING" not in out
    entry = load_registry().require("gpubox")
    assert entry.gpus == ["0", "1"]
    assert entry.s3_prefix == "s3://theirs/gpuc/gpubox"
    assert entry.retention_days is None


def test_submit_json_keeps_its_document_alone_and_its_warnings_on_stderr(
    control_env: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    job = tmp_path / "job.yaml"
    job.write_text('command: "true"\n')
    register_host(name="gpubox", kind="ssh", ssh="me@box", gpus="2,3")
    _set_host(python="/py", pkg_commit="a" * 40)
    monkeypatch.setattr("gpuc.control.cli.version_mod.local_commit", lambda: "b" * 40)
    _fake_host_build(monkeypatch, {"pkg_commit": "c" * 40, "gpus": ["0", "1"]})
    capsys.readouterr()

    assert main(["submit", str(job), "--host", "gpubox", "--json"]) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out)["job_id"] == "j"
    assert "re-syncing the package" in captured.err


def test_submit_refuses_a_host_nobody_has_ever_bootstrapped(
    control_env: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`host add` writes a config; it does not install anything. Re-shipping
    the package to such a host would start a dispatcher with no uv under it,
    and the job would fail there instead of here."""
    job = tmp_path / "job.yaml"
    job.write_text('command: "true"\n')
    register_host(name="bare", kind="ssh", ssh="me@bare", gpus=GPU, bootstrapped_at=None)
    _fake_host_build(monkeypatch, {})

    assert main(["submit", str(job), "--host", "bare"]) == EXIT_ERROR
    err = capsys.readouterr().err
    assert "no gpuc on it yet" in err
    assert "gpuc host bootstrap bare" in err


def test_submit_keeps_the_cached_config_of_a_host_that_has_lost_its_own(
    control_env: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A host whose gpuc home was wiped answers with no config at all. That is
    not an answer about what the host is -- the cache here is the only copy
    left, and the re-ship this triggers is what puts it back."""
    job = tmp_path / "job.yaml"
    job.write_text('command: "true"\n')
    register_host(name="gpubox", kind="ssh", ssh="me@box", gpus=GPU, s3_prefix="s3://b/gpuc/gpubox")
    _set_host(python="/py", pkg_commit="b" * 40)
    monkeypatch.setattr("gpuc.control.cli.version_mod.local_commit", lambda: "b" * 40)
    resynced = _fake_host_build(monkeypatch, {})

    assert main(["submit", str(job), "--host", "gpubox"]) == 0
    assert resynced == ["gpubox"]
    entry = load_registry().require("gpubox")
    assert entry.gpus == [GPU]
    assert entry.s3_prefix == "s3://b/gpuc/gpubox"


def test_submit_leaves_this_machines_record_alone_when_the_host_cannot_be_asked(
    control_env: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unreachable host is the submit behind this one's error to report, in
    full; here it only means the record we have is the best there is -- and
    that the ssh failure never leaves `ensure_package_current` as a traceback."""
    job = tmp_path / "job.yaml"
    job.write_text('command: "true"\n')
    register_host(name="gpubox", kind="ssh", ssh="me@box", gpus=GPU)
    _set_host(python="/py", pkg_commit="b" * 40)
    monkeypatch.setattr("gpuc.control.cli.version_mod.local_commit", lambda: "b" * 40)
    resynced = _fake_host_build(monkeypatch, None)

    assert main(["submit", str(job), "--host", "gpubox"]) == 0
    assert resynced == []
    assert load_registry().require("gpubox").pkg_commit == "b" * 40


def _set_host(
    *, python: str | None = None, driver_version: str | None = None, **config: object
) -> None:
    """Stage what this machine has cached about `gpubox`, as a connect would."""
    with registry_transaction() as registry:
        entry = registry.require("gpubox")
        if python or driver_version:
            entry = entry.with_cache(python=python, driver_version=driver_version)
        if config:
            entry = entry.with_config({**entry.cache.config, **config})
        registry.put(entry)


def test_a_negative_ttl_on_add_means_no_ttl_not_an_expired_host(
    control_env: Path, fake_host: FakeHost, capsys: pytest.CaptureFixture[str]
) -> None:
    """-1 stored as a TTL is a host the next reaper pass terminates."""
    assert main(["host", "add", "h", "--ssh", "me@box", "--gpus", "", "--ttl-hours", "-1"]) == 0
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
    register_host(name="gpubox", kind="ssh", ssh="me@box", gpus=GPU)
    assert main(["submit", str(job), "--runpod", "--gpu", "A40", "--host", "gpubox"]) == EXIT_USAGE
    assert main(["requeue", "job-1", "--host", "gpubox", "--runpod", "--gpu", "A40"]) == EXIT_USAGE


def test_requeue_of_an_unknown_job_is_exit_four(
    control_env: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    (Path(control_env) / "config/config.toml").write_text('s3_bucket = "bucket"\n')
    monkeypatch.setattr(
        "gpuc.control.s3index.S3Index.client", property(lambda self: FakeS3Client())
    )
    register_host(name="gpubox", kind="ssh", ssh="me@box", gpus=GPU)
    assert main(["requeue", "20260101-000000-nosuch", "--host", "gpubox"]) == EXIT_NOT_FOUND
    assert "no mirrored spec for job" in capsys.readouterr().err


def test_purging_every_finished_job_has_to_be_asked_for_twice(
    control_env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`--purge --all-finished` is an age horizon of 0: it deletes the job dir
    of something that ended a minute ago, log and all."""
    register_host(name="gpubox", kind="ssh", ssh="me@box")
    assert main(["clean", "--host", "gpubox", "--purge", "--all-finished"]) == EXIT_USAGE
    assert "Add --yes to confirm" in capsys.readouterr().err


def test_host_add_takes_gpu_indices_and_stores_them_as_given(
    control_env: Path, fake_host: FakeHost
) -> None:
    """Ownership of part of a shared box is an agreement in nvidia-smi
    numbering, so resolving it here would freeze this boot's mapping into the
    host's config; the host redoes it every dispatch pass."""
    assert main(["host", "add", "box", "--ssh", "me@box", "--gpus", f"2,3,{GPU}"]) == 0
    assert load_registry().require("box").gpus == ["2", "3", GPU]
    assert fake_host.config is not None and fake_host.config["gpus"] == ["2", "3", GPU]


def test_host_add_and_set_refuse_a_gpus_value_that_is_neither(
    control_env: Path, fake_host: FakeHost, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["host", "add", "box", "--gpus", "A40,2"]) == EXIT_USAGE
    assert "gpuc host probe" in capsys.readouterr().err
    assert load_registry().hosts == {}
    assert fake_host.commands == []  # judged before the host was touched

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
    register_host(name="local", gpus=GPU)

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
        # Nothing here can answer a `status`, so the queue fields are the
        # "we could not ask" shape. Null is not `not queued`: the submit above
        # already happened.
        "queue_position": None,
        "queue_length": None,
        "dispatched": None,
        "starts_in_s": None,
        "starts_at": None,
        "starts_unknown": None,
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
    register_host(name="local", gpus=GPU)
    capsys.readouterr()

    assert main(["requeue", "20260101-000000-aaaaaa", "--host", "local", "--json"]) == 0
    document = one_document(capsys)
    assert document["requeued_from"] == "20260101-000000-aaaaaa"
    assert (document["job_id"], document["attempt"]) == ("new", 2)


def test_cancel_json_is_the_hosts_own_answer(
    control_env: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    register_host(name="local", gpus=GPU)
    monkeypatch.setattr(
        "gpuc.control.actions.open_session",
        lambda *a, **k: as_session(StubSession([{"job_id": "j", "status": "cancelling"}])),
    )
    capsys.readouterr()
    assert main(["cancel", "20260101-000000-aaaaaa", "--host", "local", "--json"]) == 0
    document = one_document(capsys)
    assert document["status"] == "cancelling"
    assert (document["job_id"], document["host"]) == ("20260101-000000-aaaaaa", "local")


def test_preempt_asks_the_host_and_repeats_what_it_said(
    control_env: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    register_host(name="local", gpus=GPU)
    session = StubSession([{"job_id": "j", "status": "preempting", "priority": 50}])
    monkeypatch.setattr("gpuc.control.actions.open_session", lambda *a, **k: as_session(session))
    capsys.readouterr()
    assert main(["preempt", "20260101-000000-aaaaaa", "--host", "local", "--json"]) == 0
    document = one_document(capsys)
    assert (document["status"], document["priority"]) == ("preempting", 50)
    assert (document["job_id"], document["host"]) == ("20260101-000000-aaaaaa", "local")
    assert session.calls == ["preempt 20260101-000000-aaaaaa"]
    # The host's refusal is a document, so the exit code must not pre-empt it.
    assert session.checked == [False]


def test_preempt_with_a_priority_passes_it_on_and_re_mirrors_the_spec(
    control_env: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same trap as `reorder`: `requeue` submits what the mirror holds, and
    would hand the job back at the priority it was pushed off the GPU from."""
    from gpuc.control.s3index import S3Index

    main(["host", "add", "local", "--gpus", GPU])
    (Path(control_env) / "config/config.toml").write_text('s3_bucket = "bucket"\n')
    s3 = FakeS3Client()
    monkeypatch.setattr("gpuc.control.s3index.S3Index.client", property(lambda self: s3))
    S3Index("bucket", s3).put_spec_document(
        "20260101-000000-aaaaaa", {"command": "true", "priority": 50}
    )
    session = StubSession([{"job_id": "j", "status": "preempting", "priority": 90}])
    monkeypatch.setattr("gpuc.control.actions.open_session", lambda *a, **k: as_session(session))
    capsys.readouterr()
    argv = ["preempt", "20260101-000000-aaaaaa", "--priority", "90", "--host", "local", "--json"]
    assert main(argv) == 0
    assert session.calls == ["preempt 20260101-000000-aaaaaa --priority 90"]
    assert S3Index("bucket", s3).get_spec("20260101-000000-aaaaaa")["priority"] == 90


def test_preempt_reports_the_hosts_refusal_to_free_the_host_for_nothing(
    control_env: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The host owns this judgement -- it is the only side that knows what is
    in its queue -- so the control side's job is to carry the reason back."""
    register_host(name="local", gpus=GPU)
    refusal = "nothing else is queued on this host, so preempting job j would stop it"
    monkeypatch.setattr(
        "gpuc.control.actions.open_session",
        lambda *a, **k: as_session(StubSession([{"job_id": "j", "error": refusal}])),
    )
    capsys.readouterr()
    assert main(["preempt", "20260101-000000-aaaaaa", "--host", "local"]) == EXIT_ERROR
    assert "nothing else is queued" in capsys.readouterr().err


def test_preempt_reports_the_hosts_refusal(
    control_env: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    register_host(name="local", gpus=GPU)
    payload: dict[str, object] = {"job_id": "j", "error": "job j is queued, not running"}
    monkeypatch.setattr(
        "gpuc.control.actions.open_session", lambda *a, **k: as_session(StubSession([payload]))
    )
    capsys.readouterr()
    assert main(["preempt", "20260101-000000-aaaaaa", "--host", "local"]) == EXIT_ERROR
    assert "queued, not running" in capsys.readouterr().err


def test_a_host_too_old_to_know_preempt_is_never_reported_as_success(
    control_env: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Otherwise the caller frees GPUs that are still busy and submits on top."""
    register_host(name="local", gpus=GPU)
    monkeypatch.setattr(
        "gpuc.control.actions.open_session",
        lambda *a, **k: as_session(StubSession([{"job_id": "j"}])),
    )
    capsys.readouterr()
    assert main(["preempt", "20260101-000000-aaaaaa", "--host", "local"]) == EXIT_ERROR
    assert "did not say what it did" in capsys.readouterr().err


def test_preempt_refuses_a_priority_outside_the_range(
    control_env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    register_host(name="local", gpus=GPU)
    argv = ["preempt", "20260101-000000-aaaaaa", "--priority", "100", "--host", "local"]
    assert main(argv) == EXIT_USAGE
    assert "0-99" in capsys.readouterr().err


def test_reorder_json_repeats_the_priority_it_set_and_where_the_job_landed(
    control_env: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A move you cannot see is a move you have to re-check by hand: the
    document says which position the job is in now, and when it should run."""
    moved = "20260101-000000-aaaaaa"

    class Moved:
        def host_cli(self, args: str, *, check: bool = True) -> object:
            return type("Result", (), {"returncode": 0})()

        def host_json(self, args: str, *, timeout: float = 0.0, check: bool = True) -> object:
            return {
                "host": "local",
                "gpus": [GPU],
                "dispatcher_heartbeat_age_s": 1.0,
                "queue": [{"priority": 10, "job_id": moved}, {"priority": 50, "job_id": "other"}],
                "jobs": [
                    {"job_id": moved, "status": "queued", "gpus_requested": 1},
                    {"job_id": "other", "status": "queued", "gpus_requested": 1},
                ],
            }

    register_host(name="local", gpus=GPU)
    monkeypatch.setattr("gpuc.control.actions.open_session", lambda *a, **k: Moved())
    capsys.readouterr()
    argv = ["reorder", moved, "--priority", "10", "--host", "local", "--json"]
    assert main(argv) == 0
    document = one_document(capsys)
    assert document["priority"] == 10
    assert (document["queue_position"], document["queue_length"]) == (1, 2)
    assert (document["dispatched"], document["starts_in_s"]) == (False, 0.0)


def test_estimate_json_repeats_what_the_host_recorded(
    control_env: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    register_host(name="local", gpus=GPU)
    session = StubSession(
        [{"job_id": "j", "estimated_runtime_min": 150.0, "status": "running", "warning": None}]
    )
    monkeypatch.setattr("gpuc.control.actions.open_session", lambda *a, **k: as_session(session))
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
    register_host(name="local", gpus=GPU)
    payload: dict[str, object] = {"job_id": "j", "error": "job j has already succeeded"}
    monkeypatch.setattr(
        "gpuc.control.actions.open_session", lambda *a, **k: as_session(StubSession([payload]))
    )
    capsys.readouterr()
    assert main(["estimate", "20260101-000000-aaaaaa", "--minutes", "5", "--host", "local"]) == 1
    assert "already succeeded" in capsys.readouterr().err


def test_estimate_clear_asks_the_host_to_clear_it(
    control_env: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    register_host(name="local", gpus=GPU)
    session = StubSession(
        [{"job_id": "j", "estimated_runtime_min": None, "status": "queued", "warning": None}]
    )
    monkeypatch.setattr("gpuc.control.actions.open_session", lambda *a, **k: as_session(session))
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
    register_host(name="local", gpus=GPU)
    monkeypatch.setattr(
        "gpuc.control.actions.open_session",
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

    register_host(name="local", gpus=GPU)
    (Path(control_env) / "config/config.toml").write_text('s3_bucket = "bucket"\n')
    s3 = FakeS3Client()
    monkeypatch.setattr("gpuc.control.s3index.S3Index.client", property(lambda self: s3))
    S3Index("bucket", s3).put_spec_document(
        "20260101-000000-aaaaaa", {"command": "true", "some_future_field": 1}
    )
    monkeypatch.setattr(
        "gpuc.control.actions.open_session",
        lambda *a, **k: as_session(
            StubSession([{"job_id": "j", "estimated_runtime_min": 150.0, "status": "running"}])
        ),
    )
    capsys.readouterr()
    assert main(["estimate", "20260101-000000-aaaaaa", "--minutes", "150", "--host", "local"]) == 0
    mirrored = S3Index("bucket", s3).get_spec("20260101-000000-aaaaaa")
    assert mirrored["estimated_runtime_min"] == 150.0
    assert mirrored["some_future_field"] == 1


def test_reorder_updates_the_mirrored_spec_so_requeue_carries_the_new_priority(
    control_env: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same trap as the estimate: `requeue` submits what the mirror holds, so a
    move left only on the host comes back at the priority it was submitted at."""
    from gpuc.control.s3index import S3Index

    class Moved:
        def host_cli(self, args: str, *, check: bool = True) -> object:
            return type("Result", (), {"returncode": 0})()

        def host_json(self, args: str, *, timeout: float = 0.0, check: bool = True) -> object:
            raise RemoteError("local", "status", "host is busy")

    main(["host", "add", "local", "--gpus", GPU])
    (Path(control_env) / "config/config.toml").write_text('s3_bucket = "bucket"\n')
    s3 = FakeS3Client()
    monkeypatch.setattr("gpuc.control.s3index.S3Index.client", property(lambda self: s3))
    S3Index("bucket", s3).put_spec_document(
        "20260101-000000-aaaaaa", {"command": "true", "priority": 50, "some_future_field": 1}
    )
    monkeypatch.setattr("gpuc.control.actions.open_session", lambda *a, **k: Moved())
    capsys.readouterr()
    argv = ["reorder", "20260101-000000-aaaaaa", "--priority", "5", "--host", "local", "--json"]
    assert main(argv) == 0
    mirrored = S3Index("bucket", s3).get_spec("20260101-000000-aaaaaa")
    assert mirrored["priority"] == 5
    assert mirrored["some_future_field"] == 1
    # The host could not be asked where the job landed, which is a document of
    # nulls and never a failed reorder.
    document = one_document(capsys)
    assert (document["priority"], document["queue_position"]) == (5, None)


def test_reorder_says_so_when_the_mirror_kept_the_old_priority(
    control_env: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    class Moved:
        def host_cli(self, args: str, *, check: bool = True) -> object:
            return type("Result", (), {"returncode": 0})()

        def host_json(self, args: str, *, timeout: float = 0.0, check: bool = True) -> object:
            raise RemoteError("local", "status", "host is busy")

    main(["host", "add", "local", "--gpus", GPU])
    (Path(control_env) / "config/config.toml").write_text('s3_bucket = "bucket"\n')
    monkeypatch.setattr(
        "gpuc.control.s3index.S3Index.client", property(lambda self: FakeS3Client())
    )
    monkeypatch.setattr("gpuc.control.actions.open_session", lambda *a, **k: Moved())
    capsys.readouterr()
    argv = ["reorder", "20260101-000000-aaaaaa", "--priority", "5", "--host", "local", "--json"]
    assert main(argv) == 0
    warnings = one_document(capsys)["warnings"]
    assert isinstance(warnings, list) and "requeue" in warnings[0]


def test_estimate_says_so_when_the_mirror_kept_the_old_estimate(
    control_env: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The host has it, so the command succeeded; but a silent divergence is
    exactly what `requeue` would fall into later."""
    register_host(name="local", gpus=GPU)
    (Path(control_env) / "config/config.toml").write_text('s3_bucket = "bucket"\n')
    monkeypatch.setattr(
        "gpuc.control.s3index.S3Index.client", property(lambda self: FakeS3Client())
    )
    monkeypatch.setattr(
        "gpuc.control.actions.open_session",
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
    register_host(name="local", gpus=GPU)
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

    register_host(name="local", gpus=GPU)
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

    register_host(name="local", gpus=GPU)
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

    register_host(name="gpubox", kind="ssh", ssh="me@box")
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
            "shared": False,
        }
    ]
    assert any("uv is missing" in note for note in document["notes"])  # type: ignore[union-attr]


def test_host_probe_shows_only_assigned_gpus_unless_all_gpus_is_asked_for(
    control_env: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from gpuc.control.config import load_registry
    from gpuc.control.probe import parse_probe

    register_host(name="gpubox", kind="ssh", ssh="me@box", gpus="1")
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

    register_host(name="gpubox", kind="ssh", ssh="me@box")
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
    register_host(name="local", gpus=GPU, env={"HF_TOKEN": "hf_secret"})
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
    register_host(name="local", gpus=GPU)
    register_host(name="gpubox", kind="ssh", ssh="me@box")
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

    register_host(name="gpubox", kind="ssh", ssh="me@box")
    register_host(name="zbox", kind="ssh", ssh="me@zbox")
    with registry_transaction() as registry:
        registry.put(
            host_entry(
                name="pod",
                kind="runpod",
                ssh="root@1.2.3.4",
                pod_id="p1",
                bootstrapped_at=None,
            )
        )
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
    register_host(name="gpubox", kind="ssh", ssh="me@box")
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
    register_host(name="gpubox", kind="ssh", ssh="me@box")
    register_host(name="local", gpus=GPU)
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
    register_host(name="gpubox", kind="ssh", ssh="me@box")
    register_host(name="local", gpus=GPU)
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
    register_host(name="local", gpus=GPU)
    assert main(["host", "bootstrap"]) == EXIT_USAGE
    assert "--all" in capsys.readouterr().err
    assert main(["host", "bootstrap", "local", "--all"]) == EXIT_USAGE
    assert "not both" in capsys.readouterr().err


# -- adopting a pod another machine rented -------------------------------------


def adoptable(monkeypatch: pytest.MonkeyPatch, fake_host: FakeHost) -> FakeProvider:
    """A pod on the account, set up by a machine this one knows nothing about."""
    monkeypatch.setenv("RUNPOD_API_KEY", "test-key")
    provider = FakeProvider()
    provider.adopt(running_pod("gpuc-e2e-aaa", "pod1"))
    monkeypatch.setattr("gpuc.control.cli.make_provider", lambda settings: provider)
    fake_host.put_file(
        json.dumps(
            {
                "host": "gpuc-e2e-aaa",
                "gpus": ["GPU-1111"],
                "ttl_hours": 4.0,
                "provider": {"kind": "runpod", "pod_id": "pod1", "created_at": "2026-09-15T12:00"},
            }
        ),
        "/home/u/.gpuc/config.json",
    )
    return provider


def test_host_add_pod_adopts_a_pod_another_machine_created(
    control_env: Path,
    fake_host: FakeHost,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Its address comes from the provider, everything else from the pod."""
    adoptable(monkeypatch, fake_host)
    assert main(["host", "add", "rented", "--pod", "pod1"]) == 0

    entry = load_registry().require("gpuc-e2e-aaa")
    assert (entry.kind, entry.pod_id, entry.ssh, entry.port) == (
        "runpod",
        "pod1",
        "root@1.2.3.4",
        22000,
    )
    assert entry.gpus == ["GPU-1111"] and entry.ttl_hours == 4.0
    out = capsys.readouterr().out
    assert "adopted the config on the host" in out
    # This machine now watches it too, without having created it.
    assert "recorded it in desired/gpuc-e2e-aaa.json" in out
    desired = read_desired("gpuc-e2e-aaa")
    assert desired is not None and desired.pod_id == "pod1"


def test_host_add_pod_refuses_a_pod_that_is_gone(
    control_env: Path,
    fake_host: FakeHost,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adoptable(monkeypatch, fake_host)
    assert main(["host", "add", "rented", "--pod", "podX"]) == EXIT_ERROR
    assert "nothing to add" in capsys.readouterr().err
    assert load_registry().hosts == {}


def test_host_add_pod_and_ssh_are_the_same_question_twice(
    control_env: Path, fake_host: FakeHost, monkeypatch: pytest.MonkeyPatch
) -> None:
    adoptable(monkeypatch, fake_host)
    assert main(["host", "add", "rented", "--pod", "pod1", "--ssh", "me@box"]) == EXIT_USAGE


def test_host_add_pod_says_what_watching_an_unbootstrapped_pod_means(
    control_env: Path,
    fake_host: FakeHost,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Adopting a pod nobody has set up arms the silence rule against it: it has
    no dispatcher to beat, so `reconcile` here will end it. Say so."""
    monkeypatch.setenv("RUNPOD_API_KEY", "test-key")
    provider = FakeProvider()
    provider.adopt(running_pod("gpuc-e2e-aaa", "pod1"))
    monkeypatch.setattr("gpuc.control.cli.make_provider", lambda settings: provider)

    assert main(["host", "add", "rented", "--pod", "pod1", "--gpus", "GPU-1111"]) == 0

    out = capsys.readouterr().out
    assert "wrote its first config" in out
    assert "nothing has bootstrapped this pod" in out
    assert "terminates it in 30 min" in out
    assert "gpuc host bootstrap rented" in out


def test_an_existing_gpu_overlap_does_not_block_every_other_host_set(
    control_env: Path, fake_host: FakeHost
) -> None:
    """A host already in that state -- hand-edited, or written by a build
    without the check -- must still be reachable by a command about something
    else entirely."""
    fake_host.put_file(
        '{"host": "gpubox", "gpus": ["0"], "shared_gpus": ["0", "1"]}',
        "/home/u/.gpuc/config.json",
    )
    assert main(["host", "add", "gpubox", "--ssh", "me@box"]) == 0
    assert main(["host", "set", "gpubox", "--idle-min", "30"]) == 0
    assert fake_host.config is not None and fake_host.config["idle_minutes"] == 30.0
