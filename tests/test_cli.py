from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from gpuc.control import version as version_mod
from gpuc.control.bootstrap import BootstrapResult, HealthOptions
from gpuc.control.clean import purge_host
from gpuc.control.cli import (
    EXIT_ERROR,
    EXIT_INTERRUPTED,
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
    load_settings,
    registry_transaction,
    utc_now,
)
from gpuc.control.gpuinfo import owned_entries
from gpuc.control.providers.base import Constraints, Pod
from gpuc.control.remote import HostConfigRead, HostSession, RemoteError
from gpuc.control.status import placement_unknown
from gpuc.control.submit import JobSpecModel, Prepared, SubmitResult, expand_job_id
from gpuc.host.jobs import HostConfig
from tests.conftest import host_entry, load_registry, register_host
from tests.fakehost import FakeHost
from tests.fakeprovider import FakeProvider, running_pod
from tests.fakes3 import FakeS3Client

GPU = "GPU-2a4bad3b-9fe3-7031-914d-384254e92908"


def test_host_add_writes_the_first_config_of_a_host_that_has_none(
    control_env: Path, fake_host: FakeHost, capsys: pytest.CaptureFixture[str]
) -> None:
    """The host owns its config, so `add` is where a host that has none gets
    one -- and the only place this machine decides what a host is."""
    assert main(["host", "add", "local", "--gpus", GPU]) == 0
    entry = load_registry().require("local")
    assert (entry.kind, entry.config.gpus, entry.ssh) == ("local", [GPU], None)
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
    fake_host.put_file(json.dumps(theirs), fake_host.config_path)
    assert main(["host", "add", "gpubox", "--ssh", "me@box"]) == 0
    entry = load_registry().require("gpubox")
    assert entry.config.gpus == ["2", "3"]
    assert entry.config.s3_prefix == "s3://theirs/gpuc/gpubox"
    assert entry.config.retention_days == 30.0
    assert entry.config.env == {"HF_HOME": "/big"}
    assert entry.seen_at
    assert fake_host.config == theirs  # nothing was written to the host
    assert "adopted the config on the host" in capsys.readouterr().out


def test_host_add_registers_a_host_under_the_name_it_calls_itself(
    control_env: Path, fake_host: FakeHost, capsys: pytest.CaptureFixture[str]
) -> None:
    """`config.json`'s `host` is what the host answers as, and what its
    `s3_prefix` was derived from; a second local name for it would split the
    two machines' view of one box in half."""
    fake_host.put_file('{"host": "gpubox", "gpus": ["0"]}', fake_host.config_path)
    assert main(["host", "add", "other-name", "--ssh", "me@box"]) == 0
    assert set(load_registry().hosts) == {"gpubox"}
    assert "calls itself 'gpubox'" in capsys.readouterr().out


def test_a_flag_on_an_adopted_host_is_an_override_and_says_so(
    control_env: Path, fake_host: FakeHost, capsys: pytest.CaptureFixture[str]
) -> None:
    fake_host.put_file(
        '{"host": "gpubox", "gpus": ["0"], "retention_days": 30.0}', fake_host.config_path
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
    fake_host.put_file('{"host": "gpubox", "gpus": ["0", "1"]}', fake_host.config_path)
    assert main(["host", "add", "gpubox", "--ssh", "me@box", "--gpus", "1,2"]) == EXIT_ERROR
    assert "shares 1 with that list without matching it" in capsys.readouterr().err
    assert load_registry().hosts == {}
    assert fake_host.config is not None and fake_host.config["gpus"] == ["0", "1"]
    # A disjoint list is a deliberate reassignment, and goes through.
    assert main(["host", "add", "gpubox", "--ssh", "me@box", "--gpus", "2,3"]) == 0
    assert fake_host.config["gpus"] == ["2", "3"]


def test_host_add_gpus_all_is_refused_where_the_same_cards_listed_would_be(
    control_env: Path, fake_host: FakeHost, capsys: pytest.CaptureFixture[str]
) -> None:
    """`all` on this box is `0,1`, which overlaps a host that owns 0 without
    matching it; spelling it `all` is no way round the refusal."""
    fake_host.put_file('{"host": "gpubox", "gpus": ["0"]}', fake_host.config_path)
    assert main(["host", "add", "gpubox", "--ssh", "me@box", "--gpus", "0,1"]) == EXIT_ERROR
    capsys.readouterr()
    assert main(["host", "add", "gpubox", "--ssh", "me@box", "--gpus", "all"]) == EXIT_ERROR
    assert "--gpus all shares 0 with that list without matching it" in capsys.readouterr().err
    assert fake_host.config is not None and fake_host.config["gpus"] == ["0"]
    # On a host that already owns every card it matches, and is adopted as is.
    fake_host.put_file('{"host": "gpubox", "gpus": null}', fake_host.config_path)
    assert main(["host", "add", "gpubox", "--ssh", "me@box", "--gpus", "all"]) == 0
    assert fake_host.config["gpus"] is None


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
    fake_host.put_file('{"host": "gpubox", "gpus": ["GPU-a", "GPU-b"]}', fake_host.config_path)
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
    fake_host.put_file('{"host": "local", "gpus": ["0"]}', fake_host.config_path)
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
    assert second.config.gpus == first.config.gpus == ["2", "3"]
    assert second.config.created_at == first.config.created_at

    # And a change from the laptop is the host's, so the desktop sees it.
    assert main(["host", "set", "gpubox", "--gpus", "2,3,4"]) == 0
    monkeypatch.setenv("GPUC_STATE_DIR", str(control_env / "state"))
    assert main(["host", "probe", "gpubox"]) == 0
    assert load_registry().require("gpubox").config.gpus == ["2", "3", "4"]


def test_host_set_keeps_the_cache_dir_a_new_env_did_not_mention(
    control_env: Path, fake_host: FakeHost
) -> None:
    """`--env` replaces what somebody set by hand. `UV_CACHE_DIR` is not that:
    bootstrap derives it from the host's own filesystem, and dropping it costs
    every job on that host a full copy of every wheel."""
    fake_host.put_file(
        json.dumps({"host": "gpubox", "gpus": ["0"], "env": {"UV_CACHE_DIR": "/vol/uv"}}),
        fake_host.config_path,
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
    written = [c for c in fake_host.commands[len(before) :] if "mv -f" in c]
    assert written == []


def test_host_add_owns_every_card_by_default_on_a_host_with_no_config(
    control_env: Path, fake_host: FakeHost, capsys: pytest.CaptureFixture[str]
) -> None:
    """The spec's default: a host owns every card it has -- written as null, not
    as the cards the probe saw, so one added later is owned too."""
    assert main(["host", "add", "gpubox", "--ssh", "me@box"]) == 0
    assert fake_host.config is not None
    assert "gpus" in fake_host.config and fake_host.config["gpus"] is None
    assert fake_host.config.get("shared_gpus", []) == []
    entry = load_registry().require("gpubox")
    assert entry.config.gpus is None
    assert owned_entries(entry.config.gpus, entry.config.shared_gpus, entry.gpu_info) == [
        "GPU-a",
        "GPU-b",
    ]
    out = capsys.readouterr().out
    assert "with 2 GPU(s)" in out
    assert "wrote its first config" in out
    assert "owns no GPUs" not in out

    assert main(["host", "list"]) == 0
    assert "gpus 2 (all: " in capsys.readouterr().out


def test_host_add_shared_gpus_alone_owns_everything_else(
    control_env: Path, fake_host: FakeHost
) -> None:
    """A card is owned or borrowed, never both, so `--shared-gpus 1` on its own
    is "everything else is mine" -- in either spelling of the card."""
    assert main(["host", "add", "gpubox", "--ssh", "me@box", "--shared-gpus", "1"]) == 0
    assert fake_host.config is not None
    assert (fake_host.config["gpus"], fake_host.config["shared_gpus"]) == (None, ["1"])
    entry = load_registry().require("gpubox")
    assert owned_entries(entry.config.gpus, entry.config.shared_gpus, entry.gpu_info) == ["GPU-a"]
    fake_host.wipe()
    assert main(["host", "add", "other", "--ssh", "me@other", "--shared-gpus", "GPU-a"]) == 0
    assert (fake_host.config["gpus"], fake_host.config["shared_gpus"]) == (None, ["GPU-a"])
    entry = load_registry().require("other")
    assert owned_entries(entry.config.gpus, entry.config.shared_gpus, entry.gpu_info) == ["GPU-b"]


def test_host_set_gpus_all_owns_every_card_again(
    control_env: Path, fake_host: FakeHost, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["host", "add", "gpubox", "--ssh", "me@box", "--gpus", "0"]) == 0
    capsys.readouterr()
    assert main(["host", "set", "gpubox", "--gpus", "all"]) == 0
    assert "gpus 0 -> all" in capsys.readouterr().out
    assert fake_host.config is not None and fake_host.config["gpus"] is None
    assert load_registry().require("gpubox").config.gpus is None

    assert main(["host", "set", "gpubox", "--gpus", " all "]) == 0
    assert "nothing changed" in capsys.readouterr().out
    assert main(["host", "set", "gpubox", "--gpus", "1"]) == 0
    assert "gpus all -> 1" in capsys.readouterr().out
    assert fake_host.config["gpus"] == ["1"]


def test_host_add_without_gpus_keeps_what_a_configured_host_has(
    control_env: Path, fake_host: FakeHost, capsys: pytest.CaptureFixture[str]
) -> None:
    """The default is for a host with no config. One that has one is adopted as
    it stands: an omitted `--gpus` there is not "reset to every card"."""
    fake_host.put_file('{"host": "gpubox", "gpus": ["0"]}', fake_host.config_path)
    assert main(["host", "add", "gpubox", "--ssh", "me@box"]) == 0
    assert fake_host.config is not None and fake_host.config["gpus"] == ["0"]
    assert load_registry().require("gpubox").config.gpus == ["0"]
    assert "host <- gpus" not in capsys.readouterr().out


def test_host_add_registers_a_host_with_no_cards_and_says_so(
    control_env: Path,
    fake_host: FakeHost,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A host that reports no cards is not given "owns nothing" as its first
    config by default: a driver still coming up would make that stick. Said
    explicitly with `--gpus ''` it is a host that can hold a config and
    nothing else, and the same line says so when an all-covering
    `--shared-gpus` is what left it with nothing."""
    fake_host.set_gpus(None)
    assert main(["host", "add", "cpubox", "--ssh", "me@cpu"]) == 1
    assert fake_host.config is None
    assert "cpubox" not in load_registry().hosts
    assert "reports no GPUs" in capsys.readouterr().err
    assert main(["host", "add", "cpubox", "--ssh", "me@cpu", "--gpus", ""]) == 0
    assert fake_host.config is not None and fake_host.config["gpus"] == []
    assert load_registry().require("cpubox").config.gpus == []
    out = capsys.readouterr().out
    assert "owns no GPUs (--gpus '' asked for none)" in out
    assert "gpuc host set cpubox --gpus <list>" in out

    fake_host.set_gpus(["GPU-a", "GPU-b"])
    fake_host.wipe()
    assert main(["host", "add", "none", "--ssh", "me@none", "--gpus", ""]) == 0
    assert "owns no GPUs (--gpus '' asked for none)" in capsys.readouterr().out
    fake_host.wipe()
    assert main(["host", "add", "lent", "--ssh", "me@lent", "--shared-gpus", "0,1"]) == 0
    assert fake_host.config["gpus"] is None
    assert "owns no GPUs (every card it has is shared)" in capsys.readouterr().out


def test_host_add_ssh_records_the_target_and_port(control_env: Path) -> None:
    register_host(name="gpubox", kind="ssh", ssh="me@box", port=2222, gpus="GPU-a,GPU-b")
    entry = load_registry().require("gpubox")
    assert (entry.kind, entry.ssh, entry.port) == ("ssh", "me@box", 2222)
    assert entry.config.gpus == ["GPU-a", "GPU-b"]


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
    seen: list[bool] = []

    def capture(entry: Any, prepared: Prepared, *a: Any, **k: Any) -> Any:
        seen.append(prepared.spec.use_shared)
        return SubmitResult(job_id="j", host=entry.name)

    monkeypatch.setattr("gpuc.control.submitting.enqueue", capture)
    register_host(name="gpubox", kind="ssh", ssh="me@box", gpus=GPU)
    job = tmp_path / "job.yaml"
    job.write_text('command: "true"\n')

    assert main(["submit", str(job), "--host", "gpubox"]) == 0
    assert main(["submit", str(job), "--host", "gpubox", "--use-shared"]) == 0
    assert seen == [False, True]
    job.write_text('command: "true"\nuse_shared: true\n')
    assert main(["submit", str(job), "--host", "gpubox"]) == 0
    assert seen[-1] is True


def test_an_invalid_spec_never_touches_the_host(
    control_env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Everything that needs no host is judged before any host is asked: a
    typo in the job file costs no ssh round trip, and no pod."""
    register_host(name="gpubox", kind="ssh", ssh="me@box", gpus=GPU)
    monkeypatch.setattr(
        "gpuc.control.remote.open_session",
        lambda *a, **k: pytest.fail("an invalid spec must not open a session"),
    )
    job = tmp_path / "job.yaml"
    job.write_text('command: "true"\nmax_runtime_mins: 5\n')
    assert main(["submit", str(job), "--host", "gpubox"]) == EXIT_ERROR
    job.write_text('command: "true"\nsecrets: [NO_SUCH_SECRET_HERE]\n')
    monkeypatch.delenv("NO_SUCH_SECRET_HERE", raising=False)
    assert main(["submit", str(job), "--host", "gpubox"]) == EXIT_ERROR


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


def test_submit_runpod_passes_the_flags_through(
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
        return host_entry(name="gpuc-e2e-1", kind="rental", gpus=["GPU-1"])

    def fake_enqueue(entry: HostEntry, prepared: Prepared, *args: object, **kwargs: object):
        seen["host"] = entry.name
        seen["job_id"] = prepared.spec.job_id
        return SubmitResult(job_id=prepared.spec.job_id, host=entry.name)

    monkeypatch.setattr("gpuc.control.submitting.runpod_host", fake_runpod_host)
    monkeypatch.setattr("gpuc.control.submitting.enqueue", fake_enqueue)

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
    assert (seen["idle_minutes"], seen["disk_gb"]) == (2.0, 20)
    assert (seen["reuse"], seen["name_hint"]) == (False, "e2e")
    # The spec is mirrored once, after the host has accepted it: nothing is
    # written to S3 before a pod is bought.
    assert seen["mirrored_before_provisioning"] == []


def test_pods_lists_ours_and_counts_the_others(
    control_env: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RUNPOD_API_KEY", "test-key")
    provider = FakeProvider(existing=[running_pod("other-tenant", "podF")])
    provider.adopt(running_pod("gpuc-e2e-aaa", "pod1"))
    monkeypatch.setattr("gpuc.control.cli.make_provider", lambda *a, **k: provider)
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
        return host_entry(name="gpuc-e2e-1", kind="rental", gpus=["GPU-1"])

    def fake_enqueue(entry: HostEntry, prepared: Prepared, *args: object, **kwargs: object):
        seen["requeued_from"] = prepared.requeued_from
        return SubmitResult(job_id="new", host=entry.name, requeued_from=prepared.requeued_from)

    monkeypatch.setattr("gpuc.control.submitting.runpod_host", fake_runpod_host)
    monkeypatch.setattr("gpuc.control.submitting.enqueue", fake_enqueue)

    assert main(["requeue", "20260101-000000-aaaaaa", "--runpod", "--gpu", "A40"]) == 0
    assert seen == {"gpu_names": ["A40"], "requeued_from": "20260101-000000-aaaaaa"}


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
        "ssh_key",
        "image",
        "disk_gb",
    ):
        assert key in text
    assert load_settings() == Settings()
    assert str(path) in capsys.readouterr().out


def test_config_init_refuses_to_clobber_without_force(control_env: Path) -> None:
    assert main(["config", "init"]) == 0
    config_file().write_text("disk_gb = 9\n")
    assert main(["config", "init"]) == 1
    assert load_settings().disk_gb == 9
    assert main(["config", "init", "--force"]) == 0
    assert load_settings().disk_gb == 50


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
        ["host", "add", "rented", "--pod", "pod1"],
        ["host", "terminate", "gpuc-e2e-aaa"],
    ):
        assert main(argv) == 1
        err = capsys.readouterr().err
        assert err.strip().splitlines() == [
            "error: RUNPOD_API_KEY is not set; export it before using --runpod, "
            "`gpuc host add --pod`, `gpuc host terminate` or `gpuc pods`"
        ]


def test_a_non_runpod_command_does_not_need_the_api_key(
    control_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    assert main(["host", "list"]) == 0


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
        "gpuc.control.submitting.runpod_host",
        lambda *a, **k: created.append(a) or host_entry(name="gpuc-x", kind="rental"),
    )

    assert main(["submit", str(job), "--runpod", "--gpu", "A40"]) == 1
    assert "--gpu-count" in capsys.readouterr().err
    assert created == []


def test_submit_runpod_refuses_a_pod_with_no_gpu_even_for_a_job_that_needs_none(
    control_env: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = tmp_path / "job.yaml"
    job.write_text('command: "true"\ngpus: 0\n')
    created: list[object] = []
    monkeypatch.setattr(
        "gpuc.control.submitting.runpod_host",
        lambda *a, **k: created.append(a) or host_entry(name="gpuc-x", kind="rental"),
    )
    with pytest.raises(SystemExit) as exited:
        main(["submit", str(job), "--runpod", "--gpu-count", "0"])
    assert exited.value.code == EXIT_USAGE
    assert "--gpu-count: must be at least 1" in capsys.readouterr().err
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
        "gpuc.control.submitting.runpod_host",
        lambda *a, **k: created.append(a) or host_entry(name="gpuc-x", kind="rental"),
    )

    assert main(["submit", str(job), "--runpod", "--gpu", "A40"]) == 1
    assert "GPUC_DEFINITELY_UNSET" in capsys.readouterr().err
    assert created == []


# -- clean --purge --verify ---------------------------------------------------


class StubSession:
    """A host that answers `purge` (or anything else) with whatever the test wants."""

    def __init__(self, payloads: list[dict[str, object]]) -> None:
        self.payloads = payloads
        self.calls: list[str] = []
        self.checked: list[bool] = []
        self.entry = mirrored_host()
        self.config = self.entry.config

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
        "mirrored_at": "2026-01-01T00:00:00+00:00",
        "mirror": None if prefix is None else f"{prefix}/jobs/{job_id}",
        "forced": False,
    }


def test_a_verified_list_too_long_for_an_argument_travels_as_a_file(
    control_env: Path,
) -> None:
    from gpuc.control import clean as clean_mod

    ids = [f"20260101-000000-{i:06x}" for i in range(3000)]
    client = FakeS3Client(
        objects={f"bucket/gpuc/gpubox/jobs/{job_id}/log.txt": b"x" for job_id in ids}
    )
    puts: dict[str, str] = {}

    class FileSession(StubSession):
        home = "/home/u/.gpuc"
        transport = SimpleNamespace(
            put_file=lambda content, path, mode=0o600: puts.__setitem__(path, content)
        )

    session = FileSession([{"purged": [], "purge_skipped": [], "removed": []}])
    clean_mod.purge_host(
        mirrored_host(), Settings(), session=as_session(session), verify=True, s3_client=client
    )
    (call,) = session.calls
    assert "--verified-file " in call and "--verified " not in call
    ((path, content),) = puts.items()
    assert path in call and content.strip().split(",") == ids


def mirrored_host() -> HostEntry:
    return host_entry(
        name="gpubox",
        ssh="me@gpubox",
        python="/usr/bin/python3",
        s3_prefix="s3://bucket/gpuc/gpubox",
    )


def as_session(session: StubSession) -> HostSession:
    return cast("HostSession", session)


NO_MIRRORED_LOG = "not backed up: the mirror has no log for it"


def test_verify_is_one_host_call_carrying_the_ids_the_mirror_has_a_log_for(
    control_env: Path,
) -> None:
    """The listing happens first, here; the host is asked once, with the answer."""
    client = FakeS3Client(
        objects={
            "bucket/gpuc/gpubox/jobs/kept/log.txt": b"hello\n",
            "bucket/gpuc/gpubox/jobs/kept/state.json": b"{}",
            # Another host's prefix, and a job with a state but no log: neither counts.
            "bucket/gpuc/other/jobs/elsewhere/log.txt": b"hello\n",
            "bucket/gpuc/gpubox/jobs/nolog/state.json": b"{}",
        }
    )
    entry = mirrored_host()
    session = StubSession(
        [
            {
                "dry_run": False,
                "purged": [purged_entry("kept")],
                "purge_skipped": [{"job_id": "gone", "why": NO_MIRRORED_LOG}],
                "freed_bytes": 1024,
            }
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
    assert session.calls == ["purge --older-than 7.0 --verified kept"]
    assert session.checked == [False]
    assert report.verified == ["kept"]
    assert [job["job_id"] for job in report.purged] == ["kept"]
    assert any(job["why"] == NO_MIRRORED_LOG for job in report.purge_skipped)
    assert "SKIPPED gone  " + NO_MIRRORED_LOG in report.render()


def test_verify_with_nothing_mirrored_still_passes_an_empty_verified_list(
    control_env: Path,
) -> None:
    """`--verified ''` is "none of them are backed up", which must not collapse
    into leaving the flag off and letting the host trust its own records."""
    entry = mirrored_host()
    session = StubSession([{"dry_run": True, "purged": []}])
    report = purge_host(
        entry,
        Settings(),
        session=as_session(session),
        only=["gone"],
        verify=True,
        dry_run=True,
        s3_client=FakeS3Client(),
    )
    assert session.calls == ["purge --older-than 0.0 --dry-run --only gone --verified ''"]
    assert report.purged == []
    assert report.verified == []


def test_verify_asks_the_host_nothing_when_the_mirror_cannot_be_listed(
    control_env: Path,
) -> None:
    """ "Could not check" must not read as "nothing is backed up", and it must
    not reach the host as a purge either."""
    from gpuc.control.clean import CleanError

    class Refusing(FakeS3Client):
        def list_objects_v2(self, **_: Any) -> dict[str, Any]:
            raise RuntimeError("AccessDenied")

    entry = mirrored_host()
    session = StubSession([])
    with pytest.raises(CleanError, match="AccessDenied"):
        purge_host(
            entry,
            Settings(),
            session=as_session(session),
            older_than_days=7.0,
            verify=True,
            s3_client=Refusing(),
        )
    assert session.calls == []


def test_verified_mirrors_are_the_ids_with_a_log_under_the_hosts_prefix(
    control_env: Path,
) -> None:
    from gpuc.control.clean import verified_mirrors

    client = FakeS3Client(
        objects={
            "bucket/gpuc/gpubox/jobs/b/log.txt": b"",
            "bucket/gpuc/gpubox/jobs/a/log.txt": b"",
            "bucket/gpuc/gpubox/jobs/a/state.json": b"{}",
            "bucket/gpuc/gpubox/jobs/nolog/state.json": b"{}",
            "bucket/gpuc/gpubox/jobs/deep/outputs/log.txt": b"",
            "bucket/gpuc/gpubox-2/jobs/sibling/log.txt": b"",
            "bucket/gpuc/other/jobs/elsewhere/log.txt": b"",
        },
        page_size=2,
    )
    assert verified_mirrors(mirrored_host().config.s3_prefix, Settings(), client=client) == [
        "a",
        "b",
    ]
    assert {call["Prefix"] for call in client.list_calls} == {"gpuc/gpubox/jobs/"}
    assert len(client.list_calls) > 1  # followed the continuation tokens


def test_verified_mirrors_of_a_host_with_no_prefix_is_nothing(control_env: Path) -> None:
    from gpuc.control.clean import verified_mirrors

    client = FakeS3Client(objects={"bucket/gpuc/gpubox/jobs/a/log.txt": b""})
    assert verified_mirrors(None, Settings(), client=client) == []
    assert client.list_calls == []


def test_verified_mirrors_raises_when_the_listing_fails(control_env: Path) -> None:
    from gpuc.control.clean import CleanError, verified_mirrors

    class Refusing(FakeS3Client):
        def list_objects_v2(self, **_: Any) -> dict[str, Any]:
            raise RuntimeError("AccessDenied")

    with pytest.raises(CleanError, match=r"s3://bucket/gpuc/gpubox/jobs/.*AccessDenied"):
        verified_mirrors(mirrored_host().config.s3_prefix, Settings(), client=Refusing())


# -- locate ----------------------------------------------------------------------


def _probes(monkeypatch: pytest.MonkeyPatch, answers: dict[str, object]) -> list[str]:
    """Stand in for `open_session` in `locate`: each host answers its
    `status <id>` with `answers[name]`, or cannot be reached when its answer
    is an exception. Returns the hosts that were asked."""
    asked: list[str] = []

    def session(entry: HostEntry, *_: object, **__: object) -> object:
        asked.append(entry.name)
        answer = answers.get(entry.name, {"jobs": []})
        if isinstance(answer, Exception):
            raise answer
        return SimpleNamespace(entry=entry, config=HostConfig(), host_json=lambda *a, **k: answer)

    monkeypatch.setattr("gpuc.control.remote.open_session", session)
    return asked


def test_a_second_client_asks_the_host_the_mirror_names_first(
    control_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """This machine never submitted the job, so its local index is empty; the
    S3 index entry the submitting machine wrote names the host -- by *that*
    machine's name for it, so it is asked first rather than believed."""
    from gpuc.control.actions import locate
    from gpuc.control.s3index import IndexEntry, S3Index

    register_host(name="first", ssh="me@first")
    register_host(name="gpubox", ssh="me@gpubox")
    client = FakeS3Client()
    S3Index("bkt", client).put_index(
        IndexEntry(job_id="j1", host="gpubox", s3_prefix="s3://bkt/gpuc/gpubox")
    )
    monkeypatch.setattr("gpuc.control.s3index.S3Index.client", property(lambda self: client))
    asked = _probes(monkeypatch, {"gpubox": {"jobs": [{"job_id": "j1"}]}})

    location = locate("j1", load_registry(), None, Settings(s3_bucket="bkt"))
    assert location.host == "gpubox"
    assert location.index is not None and location.index.s3_prefix == "s3://bkt/gpuc/gpubox"
    assert asked == ["gpubox"]
    # The session that found it is the one the caller goes on using.
    assert location.session is not None and location.trouble is None


def test_a_mirror_index_naming_a_host_that_does_not_know_the_job_is_not_believed(
    control_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Another client's `gpu1` may be this client's `lab`."""
    from gpuc.control.actions import locate
    from gpuc.control.s3index import IndexEntry, S3Index

    register_host(name="gpu1", ssh="me@mine")
    register_host(name="lab", ssh="me@theirs")
    client = FakeS3Client()
    S3Index("bkt", client).put_index(IndexEntry(job_id="j1", host="gpu1"))
    monkeypatch.setattr("gpuc.control.s3index.S3Index.client", property(lambda self: client))
    asked = _probes(monkeypatch, {"lab": {"jobs": [{"job_id": "j1"}]}})

    assert locate("j1", load_registry(), None, Settings(s3_bucket="bkt")).host == "lab"
    assert asked == ["gpu1", "lab"]


def test_a_job_the_mirror_has_no_entry_for_is_found_by_asking_the_hosts(
    control_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing index object is "not indexed", not an error: the probe goes on."""
    from gpuc.control.actions import locate

    register_host(name="first", ssh="me@first")
    register_host(name="gpubox", ssh="me@gpubox")
    client = FakeS3Client()
    monkeypatch.setattr("gpuc.control.s3index.S3Index.client", property(lambda self: client))
    asked = _probes(monkeypatch, {"gpubox": {"jobs": [{"job_id": "j1"}]}})

    location = locate("j1", load_registry(), None, Settings(s3_bucket="bkt"))
    assert location.host == "gpubox"
    assert location.index is None
    assert "gpubox" in asked


def test_a_verb_on_many_jobs_is_one_lookup_and_one_call_per_host(
    control_env: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Stopping a sweep must not cost a round trip per job: each host is asked
    once which of the ids it has, and sent one call naming all of its own."""
    register_host(name="a", ssh="me@a")
    register_host(name="b", ssh="me@b")
    holds = {"a": ["j1", "j2"], "b": ["j3"]}
    calls: dict[str, list[str]] = {"a": [], "b": []}

    def session(entry: HostEntry, *_: object, **__: object) -> object:
        def host_json(args: str, **_: object) -> object:
            calls[entry.name].append(args)
            verb, *ids = args.split()
            mine = [job_id for job_id in ids if job_id in holds[entry.name]]
            if verb == "status":
                return {"jobs": [{"job_id": job_id, "status": "running"} for job_id in mine]}
            return {"jobs": [{"job_id": job_id, "status": "cancelling"} for job_id in mine]}

        return SimpleNamespace(entry=entry, config=HostConfig(), host_json=host_json)

    monkeypatch.setattr("gpuc.control.remote.open_session", session)
    capsys.readouterr()
    assert main(["cancel", "j1", "j2", "j3", "j4", "--json"]) == EXIT_NOT_FOUND
    assert calls == {
        "a": ["status j1 j2 j3 j4", "cancel j1 j2"],
        "b": ["status j1 j2 j3 j4", "cancel j3"],
    }
    document = one_document(capsys)
    jobs = cast("list[dict[str, Any]]", document["jobs"])
    assert [(job["job_id"], job["host"], job.get("status")) for job in jobs] == [
        ("j1", "a", "cancelling"),
        ("j2", "a", "cancelling"),
        ("j3", "b", "cancelling"),
        ("j4", None, None),
    ]
    assert document["errors"] == [jobs[3]["error"]]


def test_status_asks_each_host_for_the_window_it_will_show(
    control_env: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The host trims its history, not the client after the whole of it has
    crossed the wire -- except for `--all`, which lists every job a host did
    not send as one it no longer has."""
    register_host(name="gpubox", ssh="me@gpubox")
    asked: list[str] = []

    def session(entry: HostEntry, *_: object, **__: object) -> object:
        def host_json(args: str, **_: object) -> object:
            asked.append(args)
            return {"jobs": []}

        return SimpleNamespace(entry=entry, config=HostConfig(), host_json=host_json)

    monkeypatch.setattr("gpuc.control.remote.open_session", session)
    assert main(["status", "--json"]) == 0
    assert main(["status", "--recent", "3", "--since", "2h", "--json"]) == 0
    assert main(["status", "--all", "--json"]) == 0
    assert asked == ["status --recent 5", "status --recent 3 --since 7200.0", "status"]


def test_status_of_ids_reports_every_one_when_one_cannot_be_placed(
    control_env: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """An id no answering host has while another host is down may be on that
    host: exit 1 with the reason -- after the ids that were found."""
    register_host(name="up", ssh="me@up")
    register_host(name="down", ssh="me@down")
    _probes(
        monkeypatch,
        {
            "up": {"jobs": [{"job_id": "j1", "status": "running"}]},
            "down": RemoteError("down", "printf %s", "ssh timed out"),
        },
    )
    capsys.readouterr()
    assert main(["status", "j1", "j2", "--json"]) == EXIT_ERROR
    jobs = cast("list[dict[str, Any]]", one_document(capsys)["jobs"])
    assert [(job["job_id"], job["status"], job["error"] is None) for job in jobs] == [
        ("j1", "running", True),
        ("j2", None, False),
    ]
    assert "down: ssh timed out" in jobs[1]["error"]


def test_a_job_no_index_and_no_host_knows_is_not_found(
    control_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gpuc.control.actions import NotFound, locate

    register_host(name="gpubox", ssh="me@gpubox")
    asked = _probes(monkeypatch, {})
    with pytest.raises(NotFound, match="no registered host knows job j1"):
        locate("j1", load_registry(), None, Settings())
    assert asked == ["gpubox"]


def test_a_host_that_could_not_be_asked_is_a_failure_with_its_reason_not_a_missing_job(
    control_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The job may well be on the host that is down. "No such job" told over a
    connection error is a lie, and exit 4 would send a script the wrong way."""
    from gpuc.control.actions import CliError, NotFound, locate

    register_host(name="up", ssh="me@up")
    register_host(name="down", ssh="me@down")
    down = RemoteError("down", "printf %s", "could not reach host down: ssh timed out")
    _probes(monkeypatch, {"down": down})
    with pytest.raises(CliError) as caught:
        locate("j1", load_registry(), None, Settings())
    assert not isinstance(caught.value, NotFound)
    assert "down: could not reach host down: ssh timed out" in str(caught.value)


def test_a_job_the_index_puts_on_an_unreachable_host_stays_on_it(
    control_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The index says where the job is; the host cannot confirm it. The
    location carries the trouble for the caller to judge -- `logs` reads the
    mirror, a verb refuses with the reason -- and never hides it."""
    from gpuc.control.actions import locate
    from gpuc.control.s3index import IndexEntry, S3Index

    register_host(name="gpubox", ssh="me@gpubox")
    register_host(name="other", ssh="me@other")
    client = FakeS3Client()
    S3Index("bkt", client).put_index(IndexEntry(job_id="j1", host="gpubox"))
    monkeypatch.setattr("gpuc.control.s3index.S3Index.client", property(lambda self: client))
    _probes(monkeypatch, {"gpubox": RemoteError("gpubox", "printf %s", "no route to host")})
    location = locate("j1", load_registry(), None, Settings(s3_bucket="bkt"))
    assert location.host == "gpubox"
    assert location.trouble is not None and "no route to host" in location.trouble_reason


def test_a_verb_on_a_job_whose_host_is_down_is_exit_one_with_the_reason(
    control_env: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    register_host(name="gpubox", ssh="me@gpubox")
    _probes(monkeypatch, {"gpubox": RemoteError("gpubox", "printf %s", "no route to host")})
    capsys.readouterr()
    assert main(["cancel", "20260101-000000-aaaaaa", "--json"]) == EXIT_ERROR
    document = one_document(capsys)
    assert "no route to host" in str(document["errors"])


def _mirrored_log(monkeypatch: pytest.MonkeyPatch, host: str, job_id: str) -> FakeS3Client:
    """A bucket holding `job_id`'s log and state under `host`'s prefix, plus the
    S3 index entry the submitting machine wrote for it."""
    from gpuc.control.s3index import IndexEntry, S3Index

    prefix = f"s3://bkt/gpuc/{host}"
    client = FakeS3Client(
        objects={
            f"bkt/gpuc/{host}/jobs/{job_id}/log.txt": b"epoch 1\nepoch 2\n",
            f"bkt/gpuc/{host}/jobs/{job_id}/state.json": b'{"status": "succeeded"}',
        }
    )
    S3Index("bkt", client).put_index(IndexEntry(job_id=job_id, host=host, s3_prefix=prefix))
    monkeypatch.setattr("gpuc.control.s3index.S3Index.client", property(lambda self: client))
    config_file().write_text('s3_bucket = "bkt"\n')
    return client


def test_logs_of_a_job_on_a_host_this_machine_has_forgotten_are_the_mirrors(
    control_env: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The rental ended and `status` forgot it; the index still names it. That
    host is gone, so the mirror is the answer and nothing failed: exit 0."""
    job_id = "20260101-000000-aaaaaa"
    register_host(name="gpubox", ssh="me@gpubox")
    _mirrored_log(monkeypatch, "gpuc-pod", job_id)
    _probes(monkeypatch, {})
    capsys.readouterr()
    assert main(["logs", job_id, "--json"]) == 0
    document = one_document(capsys)
    assert (document["source"], document["host"]) == ("s3", "gpuc-pod")
    assert document["lines"] == ["epoch 1", "epoch 2"]
    assert any(
        "not registered on this machine" in note for note in cast("list[str]", document["notes"])
    )


def test_logs_of_a_job_on_an_unreachable_host_print_the_mirror_and_still_fail(
    control_env: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A host that is only unreachable may hold a newer log than its mirror:
    what the mirror has is printed (as much as can be done) and the command
    exits 1 with the reason, never 0 as if that were the answer."""
    job_id = "20260101-000000-aaaaaa"
    register_host(name="gpubox", ssh="me@gpubox")
    _mirrored_log(monkeypatch, "gpubox", job_id)
    _probes(monkeypatch, {"gpubox": RemoteError("gpubox", "printf %s", "no route to host")})
    capsys.readouterr()
    assert main(["logs", job_id]) == EXIT_ERROR
    captured = capsys.readouterr()
    assert "epoch 2" in captured.out
    assert "could not read the host log: no route to host" in captured.err
    assert "falling back to the S3 mirror" in captured.err


def test_logs_of_a_job_whose_rental_has_ended_are_the_mirrors_and_exit_zero(
    control_env: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    job_id = "20260101-000000-aaaaaa"
    register_host(name="gpuc-pod", kind="rental", pod_id="pod-1", ssh="root@1.2.3.4")
    _mirrored_log(monkeypatch, "gpuc-pod", job_id)
    monkeypatch.setattr("gpuc.control.actions.make_provider", lambda *a, **k: FakeProvider())
    _probes(monkeypatch, {"gpuc-pod": RemoteError("gpuc-pod", "printf %s", "must not be asked")})
    capsys.readouterr()
    assert main(["logs", job_id]) == 0
    captured = capsys.readouterr()
    assert "epoch 2" in captured.out
    assert "this rental has ended" in captured.err


def test_requeue_of_a_job_whose_host_was_forgotten_needs_another_host(
    control_env: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exit 1 naming the gone host and the way on, not exit 4: the spec is in
    the mirror and the job is real, it just has nowhere to go back to."""
    from gpuc.control.s3index import S3Index

    job_id = "20260101-000000-aaaaaa"
    register_host(name="gpubox", ssh="me@gpubox")
    client = _mirrored_log(monkeypatch, "gpuc-pod", job_id)
    S3Index("bkt", client).put_spec_document(job_id, {"command": "train", "gpus": 1})
    _probes(monkeypatch, {})
    capsys.readouterr()
    assert main(["requeue", job_id]) == EXIT_ERROR
    err = capsys.readouterr().err
    assert "gpuc-pod, which is gone" in err and "--host" in err


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


def test_verify_with_force_passes_both_and_reports_the_forced_purge(control_env: Path) -> None:
    client = FakeS3Client()
    entry = mirrored_host()
    forced = {**purged_entry("gone"), "forced": True}
    session = StubSession([{"dry_run": False, "purged": [forced], "freed_bytes": 1024}])
    report = purge_host(
        entry,
        Settings(),
        session=as_session(session),
        older_than_days=7.0,
        force=True,
        verify=True,
        s3_client=client,
    )
    assert session.calls == ["purge --older-than 7.0 --force --verified ''"]
    assert [job["job_id"] for job in report.purged] == ["gone"]
    assert report.verified == []
    assert "FORCED" in report.render()


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
    assert session.calls == ["purge --older-than 0.0 --only a,b"]
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


def test_only_with_verify_names_the_jobs_and_what_the_mirror_vouches_for(
    control_env: Path,
) -> None:
    """The host scopes purge and sweep to `--only`, and judges the backup by
    `--verified`: one call, both lists."""
    client = FakeS3Client(objects={"bucket/gpuc/gpubox/jobs/kept/log.txt": b"hello\n"})
    entry = mirrored_host()
    session = StubSession(
        [
            {
                "dry_run": False,
                "purged": [purged_entry("kept")],
                "purge_skipped": [{"job_id": "gone", "why": NO_MIRRORED_LOG}],
                "freed_bytes": 1024,
            }
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
    assert session.calls == ["purge --older-than 0.0 --only kept,gone --verified kept"]
    assert report.verified == ["kept"]


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
    assert load_registry().require("gpubox").config.retention_days == 14.0
    assert fake_host.config is not None and fake_host.config["retention_days"] == 14.0
    assert main(["host", "set", "gpubox", "--retention-days", ""]) == 0
    assert load_registry().require("gpubox").config.retention_days is None
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
    assert load_registry().require("gpubox").config.workdir_days == 1.0
    # ...and it is the only horizon that is on without being asked for.
    assert fake_host.config["retention_days"] is None


def test_an_adopted_config_is_not_given_a_sweep_it_never_had(
    control_env: Path, fake_host: FakeHost
) -> None:
    """A host that has been getting along without one: meeting it is not the
    moment to start deleting there."""
    fake_host.put_file('{"host": "gpubox", "gpus": ["0"]}', fake_host.config_path)
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
    assert load_registry().require("gpubox").config.workdir_days == 0.0
    assert main(["host", "set", "gpubox", "--workdir-days", ""]) == 0
    assert fake_host.config["workdir_days"] is None
    assert load_registry().require("gpubox").config.workdir_days is None


def test_a_bad_workdir_days_value_is_rejected(
    control_env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["host", "add", "gpubox", "--ssh", "me@box", "--workdir-days", "-1"]) == EXIT_USAGE
    assert "--workdir-days cannot be negative" in capsys.readouterr().err


def test_the_index_listing_flags_jobs_whose_outputs_were_lost(control_env: Path) -> None:
    from gpuc.control.actions import _mirrored_states
    from gpuc.control.s3index import IndexEntry, JobIndex, S3Index

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
    index = JobIndex(Settings(s3_bucket="bucket"))
    index.s3 = S3Index("bucket", client)
    states = _mirrored_states(index, entries)
    assert {job_id for job_id, state in states.items() if state.get("outputs_lost")} == {"lost"}
    # No bucket configured: no mirror to read, so nothing is flagged.
    assert _mirrored_states(JobIndex(Settings()), entries) == {}


def test_submit_and_requeue_both_take_no_git() -> None:
    parser = build_parser()
    assert parser.parse_args(["submit", "job.yaml", "--host", "h", "--no-git"]).no_git
    assert parser.parse_args(["requeue", "id", "--host", "h", "--no-git"]).no_git
    assert not parser.parse_args(["submit", "job.yaml", "--host", "h"]).no_git


def _host_config(host: FakeHost, config: dict[str, Any]) -> None:
    """What the host's own `config.json` says from now on; `{}` is a host
    that has none, as a wiped home answers."""
    if config:
        host.put_file(json.dumps(config), host.config_path, 0o644)
    else:
        host.wipe()


def _fake_host_build(
    monkeypatch: pytest.MonkeyPatch, host: FakeHost, config: dict[str, Any] | None
) -> list[str]:
    """A host whose `config.json` is `config` ({} for none, None for a host
    that cannot be reached), a re-ship that records itself, and an enqueue
    that does nothing. Returns the hosts re-shipped to."""
    resynced: list[str] = []
    if config is None:

        def unreachable(*_: object, **__: object) -> Any:
            raise RemoteError("gpubox", "printf %s", "could not reach host gpubox")

        monkeypatch.setattr("gpuc.control.remote.resolve_home", unreachable)
    else:
        _host_config(host, config)

    def fake_ensure_build(session: HostSession, report: Any, **_: Any) -> int:
        resynced.append(session.entry.name)
        # As the real one does: the commit just shipped, and nothing else.
        session.write_config({"pkg_commit": version_mod.local_commit()})
        return 1

    monkeypatch.setattr("gpuc.control.submitting.ensure_build", fake_ensure_build)
    monkeypatch.setattr(
        "gpuc.control.submitting.submit_spec",
        lambda session, prepared, *a, **k: SubmitResult(job_id="j", host=session.entry.name),
    )
    monkeypatch.setattr(
        "gpuc.control.submitting.placement_after", lambda *a, **k: placement_unknown()
    )
    return resynced


def test_submit_reships_the_package_to_a_host_on_another_commit(
    control_env: Path,
    fake_host: FakeHost,
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
    resynced = _fake_host_build(monkeypatch, fake_host, host_config)

    assert main(["submit", str(job), "--host", "gpubox"]) == 0
    out = capsys.readouterr().out
    assert resynced == ["gpubox"]
    assert sum("re-syncing the package" in line for line in out.splitlines()) == 1
    assert load_registry().require("gpubox").config.pkg_commit == "b" * 40

    # Now the host is on this build, so nothing is shipped.
    resynced.clear()
    _host_config(fake_host, {"pkg_commit": "b" * 40})
    assert main(["submit", str(job), "--host", "gpubox"]) == 0
    assert resynced == []

    # A host whose config names no commit was never bootstrapped by any build
    # that records one: refused, not half-installed on the way past.
    _host_config(fake_host, {"host": "gpubox"})
    assert main(["submit", str(job), "--host", "gpubox", "--no-bootstrap"]) == 0
    assert resynced == []
    assert main(["submit", str(job), "--host", "gpubox"]) == EXIT_ERROR
    assert resynced == []


def test_submit_asks_the_host_which_build_it_runs_not_this_machines_record(
    control_env: Path,
    fake_host: FakeHost,
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
    resynced = _fake_host_build(monkeypatch, fake_host, {"pkg_commit": "c" * 40})

    assert main(["submit", str(job), "--host", "gpubox"]) == 0
    assert resynced == ["gpubox"]
    assert "host gpubox is running gpuc " + "c" * 12 in capsys.readouterr().out
    # The fake re-ship records what it shipped, as the real one does.
    assert load_registry().require("gpubox").config.pkg_commit == "b" * 40

    # The host now runs `b`; a client on `c` re-ships again, and the record
    # follows the host each time: `host list` and `version` have only it.
    resynced.clear()
    monkeypatch.setattr("gpuc.control.cli.version_mod.local_commit", lambda: "c" * 40)
    assert main(["submit", str(job), "--host", "gpubox"]) == 0
    assert resynced == ["gpubox"]
    assert load_registry().require("gpubox").config.pkg_commit == "c" * 40
    resynced.clear()
    assert main(["submit", str(job), "--host", "gpubox"]) == 0
    assert resynced == []


def test_submit_refuses_a_host_whose_config_cannot_be_read_and_says_why(
    control_env: Path,
    fake_host: FakeHost,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A config that is there and does not parse is neither "no gpuc yet" nor a
    host that owns no cards: the reason it could not be read is the answer."""
    job = tmp_path / "job.yaml"
    job.write_text('command: "true"\n')
    register_host(name="gpubox", kind="ssh", ssh="me@box", gpus=GPU)
    _set_host(python="/py", pkg_commit="b" * 40)
    monkeypatch.setattr(
        "gpuc.control.remote.read_config",
        lambda *a, **k: HostConfigRead(unreadable="config.json is there but holds no JSON object"),
    )
    assert main(["submit", str(job), "--host", "gpubox"]) == EXIT_ERROR
    err = capsys.readouterr().err
    assert "could not be read" in err and "holds no JSON object" in err
    assert "no gpuc on it yet" not in err and "cannot run this job" not in err


def test_submit_records_the_hosts_commit_without_clobbering_the_rest_of_the_entry(
    control_env: Path,
    fake_host: FakeHost,
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
    _fake_host_build(monkeypatch, fake_host, {"pkg_commit": "c" * 40})

    def concurrent_probe() -> dict[str, Any]:
        # Between reading the entry and recording what the host said, another
        # session records a driver version against the same host.
        _set_host(driver_version="580.173.02")
        return {"pkg_commit": "c" * 40}

    monkeypatch.setattr(
        "gpuc.control.remote.read_config",
        lambda *a, **k: HostConfigRead(concurrent_probe()),
    )
    assert main(["submit", str(job), "--host", "gpubox"]) == 0
    entry = load_registry().require("gpubox")
    assert (entry.config.pkg_commit, entry.driver_version) == ("c" * 40, "580.173.02")


def test_submit_takes_the_hosts_config_as_it_finds_it(
    control_env: Path,
    fake_host: FakeHost,
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
    resynced = _fake_host_build(monkeypatch, fake_host, theirs)

    assert main(["submit", str(job), "--host", "gpubox"]) == 0
    out = capsys.readouterr().out
    assert resynced == []
    # Nothing to warn about, and nothing to reconcile by hand.
    assert "WARNING" not in out
    entry = load_registry().require("gpubox")
    assert entry.config.gpus == ["0", "1"]
    assert entry.config.s3_prefix == "s3://theirs/gpuc/gpubox"
    assert entry.config.retention_days is None


def test_submit_json_keeps_its_document_alone_and_its_warnings_on_stderr(
    control_env: Path,
    fake_host: FakeHost,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    job = tmp_path / "job.yaml"
    job.write_text('command: "true"\n')
    register_host(name="gpubox", kind="ssh", ssh="me@box", gpus="2,3")
    _set_host(python="/py", pkg_commit="a" * 40)
    monkeypatch.setattr("gpuc.control.cli.version_mod.local_commit", lambda: "b" * 40)
    _fake_host_build(monkeypatch, fake_host, {"pkg_commit": "c" * 40, "gpus": ["0", "1"]})
    capsys.readouterr()

    assert main(["submit", str(job), "--host", "gpubox", "--json"]) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out)["job_id"] == "j"
    assert "re-syncing the package" in captured.err


def test_submit_refuses_a_host_nobody_has_ever_bootstrapped(
    control_env: Path,
    fake_host: FakeHost,
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
    resynced = _fake_host_build(monkeypatch, fake_host, {"host": "bare", "gpus": [GPU]})

    assert main(["submit", str(job), "--host", "bare"]) == EXIT_ERROR
    err = capsys.readouterr().err
    assert "no gpuc on it yet" in err
    assert "gpuc host bootstrap bare" in err
    assert resynced == []


def test_submit_refuses_a_host_that_has_lost_its_config(
    control_env: Path,
    fake_host: FakeHost,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A host whose gpuc home was wiped answers with no config at all. Nothing
    says what cards it owns, so nothing is enqueued: `gpuc host bootstrap`
    restores the last config seen, and is what the refusal names."""
    job = tmp_path / "job.yaml"
    job.write_text('command: "true"\n')
    register_host(name="gpubox", kind="ssh", ssh="me@box", gpus=GPU, s3_prefix="s3://b/gpuc/gpubox")
    _set_host(python="/py", pkg_commit="b" * 40)
    monkeypatch.setattr("gpuc.control.cli.version_mod.local_commit", lambda: "b" * 40)
    resynced = _fake_host_build(monkeypatch, fake_host, {})

    assert main(["submit", str(job), "--host", "gpubox"]) == EXIT_ERROR
    assert "gpuc host bootstrap gpubox" in capsys.readouterr().err
    assert resynced == []
    # The cache is the only copy left, and it is left alone.
    entry = load_registry().require("gpubox")
    assert entry.config.gpus == [GPU]
    assert entry.config.s3_prefix == "s3://b/gpuc/gpubox"


def test_submit_leaves_this_machines_record_alone_when_the_host_cannot_be_asked(
    control_env: Path,
    fake_host: FakeHost,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unreachable host is the submit's own failure, reported in full and
    exit 1; the record this machine has of the host is left as it was."""
    job = tmp_path / "job.yaml"
    job.write_text('command: "true"\n')
    register_host(name="gpubox", kind="ssh", ssh="me@box", gpus=GPU)
    _set_host(python="/py", pkg_commit="b" * 40)
    monkeypatch.setattr("gpuc.control.cli.version_mod.local_commit", lambda: "b" * 40)
    resynced = _fake_host_build(monkeypatch, fake_host, None)

    assert main(["submit", str(job), "--host", "gpubox"]) == EXIT_ERROR
    assert resynced == []
    assert load_registry().require("gpubox").config.pkg_commit == "b" * 40


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
    assert load_registry().require("box").config.gpus == ["2", "3", GPU]
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
    assert load_registry().require("box").config.gpus == ["2"]


# -- `--json` on every command ------------------------------------------------


def one_document(capsys: pytest.CaptureFixture[str]) -> dict[str, object]:
    document = json.loads(capsys.readouterr().out)
    assert isinstance(document, dict)
    assert document["schema_version"] == 1
    return document


def one_job(capsys: pytest.CaptureFixture[str]) -> dict[str, Any]:
    """The one entry of a job verb's `{jobs, errors}` document."""
    document = one_document(capsys)
    jobs = document["jobs"]
    assert isinstance(jobs, list) and len(jobs) == 1
    return cast("dict[str, Any]", jobs[0])


def test_submit_json_is_the_queued_job_and_its_notes(
    control_env: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = tmp_path / "job.yaml"
    job.write_text('command: "true"\n')
    register_host(name="local", gpus=GPU)

    def fake_enqueue(entry: HostEntry, prepared: Prepared, *args: object, **kwargs: object):
        # The progress a text submit prints inline must not land on the document.
        report = kwargs["report"]
        assert callable(report)
        report("syncing 3 files")
        return SubmitResult(
            job_id="20260915-120000-abc123", host=entry.name, notes=["s3_bucket unset"]
        )

    monkeypatch.setattr("gpuc.control.submitting.enqueue", fake_enqueue)
    capsys.readouterr()
    assert main(["submit", str(job), "--host", "local", "--json"]) == 0
    captured = capsys.readouterr()
    document = json.loads(captured.out)
    assert document == {
        "schema_version": 1,
        "job_id": "20260915-120000-abc123",
        "host": "local",
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
        {"job_id": "20260101-000000-aaaaaa", "command": "true", "gpus": 1}
    ).encode()
    monkeypatch.setattr("gpuc.control.s3index.S3Index.client", property(lambda self: s3))
    monkeypatch.setattr(
        "gpuc.control.submitting.enqueue",
        lambda entry, prepared, *a, **k: SubmitResult(
            job_id="new", host=entry.name, requeued_from=prepared.requeued_from
        ),
    )
    register_host(name="local", gpus=GPU)
    capsys.readouterr()

    assert main(["requeue", "20260101-000000-aaaaaa", "--host", "local", "--json"]) == 0
    document = one_document(capsys)
    assert document["requeued_from"] == "20260101-000000-aaaaaa"
    assert document["job_id"] == "new"


class _KnowsJobs:
    """A session on a host that knows exactly the jobs it is given."""

    def __init__(self, entry: HostEntry, known: set[str], asked: list[str]) -> None:
        self.entry, self.name, self.known, self.asked = entry, entry.name, known, asked
        self.config = entry.config

    def host_json(self, args: str, *, timeout: float = 0.0, check: bool = True) -> object:
        self.asked.append(f"{self.name}: {args}")
        job_id = args.split()[-1]
        return {"jobs": [{"job_id": job_id}] if job_id in self.known else []}


def _requeue_setup(
    control_env: Path, monkeypatch: pytest.MonkeyPatch, known: dict[str, set[str]]
) -> tuple[list[str], list[str]]:
    """A mirrored spec, no local index entry (this is a second client), and
    one fake session per registered host. Returns (hosts asked, hosts submitted to)."""
    (Path(control_env) / "config/config.toml").write_text('s3_bucket = "bucket"\n')
    s3 = FakeS3Client()
    s3.objects["bucket/gpuc/specs/20260101-000000-aaaaaa.json"] = json.dumps(
        {"job_id": "20260101-000000-aaaaaa", "command": "true", "gpus": 1}
    ).encode()
    monkeypatch.setattr("gpuc.control.s3index.S3Index.client", property(lambda self: s3))
    asked: list[str] = []
    submitted: list[str] = []
    monkeypatch.setattr(
        "gpuc.control.remote.open_session",
        lambda entry, *a, **k: _KnowsJobs(entry, known.get(entry.name, set()), asked),
    )
    monkeypatch.setattr("gpuc.control.submitting.ensure_package_current", lambda session, **k: None)
    monkeypatch.setattr(
        "gpuc.control.submitting.placement_after", lambda *a, **k: placement_unknown()
    )

    def fake_submit(session: Any, prepared: Prepared, *a: Any, **k: Any) -> SubmitResult:
        submitted.append(session.entry.name)
        return SubmitResult(
            job_id="new", host=session.entry.name, requeued_from=prepared.requeued_from
        )

    monkeypatch.setattr("gpuc.control.submitting.submit_spec", fake_submit)
    for name in known:
        register_host(name=name, kind="ssh", ssh=f"me@{name}", gpus=GPU)
    return asked, submitted


def test_requeue_without_a_host_finds_the_job_on_a_host_this_client_never_submitted_to(
    control_env: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A second client has no index entry for the job, so it asks every host,
    the same lookup every other job command makes."""
    asked, submitted = _requeue_setup(
        control_env, monkeypatch, {"a": set(), "b": {"20260101-000000-aaaaaa"}}
    )
    capsys.readouterr()

    assert main(["requeue", "20260101-000000-aaaaaa", "--json"]) == 0
    document = one_document(capsys)
    assert submitted == ["b"]
    assert document["host"] == "b"
    assert document["requeued_from"] == "20260101-000000-aaaaaa"
    assert "b: status 20260101-000000-aaaaaa" in asked


def test_requeue_without_a_host_of_a_job_no_host_knows_is_exit_four(
    control_env: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    asked, submitted = _requeue_setup(control_env, monkeypatch, {"a": set(), "b": set()})
    capsys.readouterr()

    assert main(["requeue", "20260101-000000-aaaaaa"]) == EXIT_NOT_FOUND
    assert "no registered host knows job 20260101-000000-aaaaaa" in capsys.readouterr().err
    assert submitted == []
    assert len(asked) == 2


def test_requeue_ignores_spec_keys_an_older_build_mirrored(
    control_env: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The mirror holds what some build wrote; `low_util` was in every spec
    before the watchdog went, and a submit's typo check is the wrong tool for
    a file no person typed."""
    (Path(control_env) / "config/config.toml").write_text('s3_bucket = "bucket"\n')
    s3 = FakeS3Client()
    s3.objects["bucket/gpuc/specs/20260101-000000-aaaaaa.json"] = json.dumps(
        {
            "job_id": "20260101-000000-aaaaaa",
            "command": "true",
            "gpus": 1,
            "low_util": {"enabled": False, "window_min": 25, "floor_pct": 5, "grace_min": 10},
            "from_the_future": {"unknown": True},
            "outputs": [{"path": "out", "s3": "s3://b/{job_id}/out", "hf_private": True}],
        }
    ).encode()
    monkeypatch.setattr("gpuc.control.s3index.S3Index.client", property(lambda self: s3))
    submitted: list[JobSpecModel] = []

    def fake_enqueue(entry: Any, prepared: Prepared, *a: Any, **k: Any) -> SubmitResult:
        submitted.append(prepared.model)
        return SubmitResult(job_id="new", host=entry.name)

    monkeypatch.setattr("gpuc.control.submitting.enqueue", fake_enqueue)
    register_host(name="local", gpus=GPU)
    capsys.readouterr()

    assert main(["requeue", "20260101-000000-aaaaaa", "--host", "local"]) == 0
    assert [m.command for m in submitted] == ["true"]


def test_requeue_runpod_refuses_an_output_without_the_job_id_before_provisioning(
    control_env: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RUNPOD_API_KEY", "test-key")
    (Path(control_env) / "config/config.toml").write_text('s3_bucket = "bucket"\n')
    s3 = FakeS3Client()
    s3.objects["bucket/gpuc/specs/20260101-000000-aaaaaa.json"] = json.dumps(
        {
            "job_id": "20260101-000000-aaaaaa",
            "command": "true",
            "gpus": 1,
            "outputs": [{"path": "results", "s3": "s3://b/exp/results"}],
        }
    ).encode()
    monkeypatch.setattr("gpuc.control.s3index.S3Index.client", property(lambda self: s3))
    monkeypatch.setattr(
        "gpuc.control.submitting.runpod_host",
        lambda *a, **k: pytest.fail("no pod may be bought for a spec that is refused"),
    )
    assert main(["requeue", "20260101-000000-aaaaaa", "--runpod", "--gpu", "A40"]) == 1
    assert "does not include the job id" in capsys.readouterr().err


def test_requeue_expands_the_mirrored_template_with_the_new_id(
    control_env: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    (Path(control_env) / "config/config.toml").write_text('s3_bucket = "bucket"\n')
    s3 = FakeS3Client()
    s3.objects["bucket/gpuc/specs/20260101-000000-aaaaaa.json"] = json.dumps(
        {
            "job_id": "20260101-000000-aaaaaa",
            "command": "true",
            "gpus": 1,
            "outputs": [{"path": "results", "s3": "s3://b/exp/{job_id}/results"}],
        }
    ).encode()
    monkeypatch.setattr("gpuc.control.s3index.S3Index.client", property(lambda self: s3))
    submitted: list[Prepared] = []

    def fake_enqueue(entry: Any, prepared: Prepared, *a: Any, **k: Any) -> SubmitResult:
        submitted.append(prepared)
        return SubmitResult(job_id=prepared.spec.job_id, host=entry.name)

    monkeypatch.setattr("gpuc.control.submitting.enqueue", fake_enqueue)
    register_host(name="local", gpus=GPU)
    capsys.readouterr()

    assert main(["requeue", "20260101-000000-aaaaaa", "--host", "local"]) == 0
    (prepared,) = submitted
    assert prepared.spec.job_id != "20260101-000000-aaaaaa"
    assert prepared.spec.outputs[0].s3 == f"s3://b/exp/{prepared.spec.job_id}/results"
    # And the mirror keeps the template, so the next requeue gets its own too.
    assert prepared.mirrored()["outputs"][0]["s3"] == "s3://b/exp/{job_id}/results"
    spec = prepared.model.to_spec("20260102-000000-bbbbbb")
    assert expand_job_id(spec).outputs[0].s3 == "s3://b/exp/20260102-000000-bbbbbb/results"


def test_cancel_json_is_the_hosts_own_answer(
    control_env: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    register_host(name="local", gpus=GPU)
    monkeypatch.setattr(
        "gpuc.control.remote.open_session",
        lambda *a, **k: as_session(
            StubSession([{"jobs": [{"job_id": "20260101-000000-aaaaaa", "status": "cancelling"}]}])
        ),
    )
    capsys.readouterr()
    assert main(["cancel", "20260101-000000-aaaaaa", "--host", "local", "--json"]) == 0
    document = one_job(capsys)
    assert document["status"] == "cancelling"
    assert (document["job_id"], document["host"]) == ("20260101-000000-aaaaaa", "local")


def test_preempt_asks_the_host_and_repeats_what_it_said(
    control_env: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    register_host(name="local", gpus=GPU)
    session = StubSession(
        [{"jobs": [{"job_id": "20260101-000000-aaaaaa", "status": "preempting", "priority": 50}]}]
    )
    monkeypatch.setattr("gpuc.control.remote.open_session", lambda *a, **k: as_session(session))
    capsys.readouterr()
    assert main(["preempt", "20260101-000000-aaaaaa", "--host", "local", "--json"]) == 0
    document = one_job(capsys)
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
    session = StubSession(
        [{"jobs": [{"job_id": "20260101-000000-aaaaaa", "status": "preempting", "priority": 90}]}]
    )
    monkeypatch.setattr("gpuc.control.remote.open_session", lambda *a, **k: as_session(session))
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
        "gpuc.control.remote.open_session",
        lambda *a, **k: as_session(
            StubSession([{"jobs": [{"job_id": "20260101-000000-aaaaaa", "error": refusal}]}])
        ),
    )
    capsys.readouterr()
    assert main(["preempt", "20260101-000000-aaaaaa", "--host", "local"]) == EXIT_ERROR
    assert "nothing else is queued" in capsys.readouterr().err


def test_preempt_reports_the_hosts_refusal(
    control_env: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    register_host(name="local", gpus=GPU)
    payload: dict[str, object] = {
        "jobs": [{"job_id": "20260101-000000-aaaaaa", "error": "job j is queued, not running"}]
    }
    monkeypatch.setattr(
        "gpuc.control.remote.open_session", lambda *a, **k: as_session(StubSession([payload]))
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
        "gpuc.control.remote.open_session",
        lambda *a, **k: as_session(StubSession([{"jobs": [{"job_id": "20260101-000000-aaaaaa"}]}])),
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
        entry = host_entry(name="local", gpus=[GPU])
        config = HostConfig(host="local", gpus=[GPU])

        def host_json(self, args: str, *, timeout: float = 0.0, check: bool = True) -> object:
            if args.startswith("reorder"):
                return {"jobs": [{"job_id": moved, "status": "queued", "priority": 10}]}
            return {
                "host": "local",
                "gpus": [GPU],
                "dispatcher_heartbeat_age_s": 1.0,
                "queue": [{"priority": 10, "job_id": moved}, {"priority": 50, "job_id": "other"}],
                "jobs": [
                    {"job_id": moved, "status": "queued", "priority": 10, "starts_in_s": 0.0},
                    {"job_id": "other", "status": "queued", "priority": 50, "starts_in_s": 0.0},
                ],
            }

    register_host(name="local", gpus=GPU)
    monkeypatch.setattr("gpuc.control.remote.open_session", lambda *a, **k: Moved())
    capsys.readouterr()
    argv = ["reorder", moved, "--priority", "10", "--host", "local", "--json"]
    assert main(argv) == 0
    document = one_job(capsys)
    assert document["priority"] == 10
    assert (document["queue_position"], document["queue_length"]) == (1, 2)
    assert (document["dispatched"], document["starts_in_s"]) == (False, 0.0)


def test_estimate_json_repeats_what_the_host_recorded(
    control_env: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    register_host(name="local", gpus=GPU)
    session = StubSession(
        [
            {
                "jobs": [
                    {
                        "job_id": "20260101-000000-aaaaaa",
                        "estimated_runtime_min": 150.0,
                        "status": "running",
                        "warning": None,
                    }
                ]
            }
        ]
    )
    monkeypatch.setattr("gpuc.control.remote.open_session", lambda *a, **k: as_session(session))
    capsys.readouterr()
    argv = ["estimate", "20260101-000000-aaaaaa", "--minutes", "150", "--host", "local", "--json"]
    assert main(argv) == 0
    document = one_job(capsys)
    assert document["estimated_runtime_min"] == 150.0 and document["status"] == "running"
    # `check=False`: the host's refusal is a document, and raising on the exit
    # code would throw away the reason it gave.
    assert session.checked == [False]
    assert session.calls == ["estimate 20260101-000000-aaaaaa --minutes 150.0"]


def test_estimate_reports_the_hosts_refusal(
    control_env: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    register_host(name="local", gpus=GPU)
    payload: dict[str, object] = {
        "jobs": [{"job_id": "20260101-000000-aaaaaa", "error": "job j has already succeeded"}]
    }
    monkeypatch.setattr(
        "gpuc.control.remote.open_session", lambda *a, **k: as_session(StubSession([payload]))
    )
    capsys.readouterr()
    assert main(["estimate", "20260101-000000-aaaaaa", "--minutes", "5", "--host", "local"]) == 1
    assert "already succeeded" in capsys.readouterr().err


def test_estimate_clear_asks_the_host_to_clear_it(
    control_env: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    register_host(name="local", gpus=GPU)
    session = StubSession(
        [
            {
                "jobs": [
                    {
                        "job_id": "20260101-000000-aaaaaa",
                        "estimated_runtime_min": None,
                        "status": "queued",
                        "warning": None,
                    }
                ]
            }
        ]
    )
    monkeypatch.setattr("gpuc.control.remote.open_session", lambda *a, **k: as_session(session))
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
        "gpuc.control.remote.open_session",
        lambda *a, **k: as_session(
            StubSession([{"jobs": [{"job_id": "20260101-000000-aaaaaa", "status": "running"}]}])
        ),
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
        "gpuc.control.remote.open_session",
        lambda *a, **k: as_session(
            StubSession(
                [
                    {
                        "jobs": [
                            {
                                "job_id": "20260101-000000-aaaaaa",
                                "estimated_runtime_min": 150.0,
                                "status": "running",
                            }
                        ]
                    }
                ]
            )
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
        entry = host_entry(name="local", gpus=[GPU])
        config = HostConfig(host="local", gpus=[GPU])

        def host_json(self, args: str, *, timeout: float = 0.0, check: bool = True) -> object:
            if args.startswith("reorder"):
                return {
                    "jobs": [
                        {"job_id": "20260101-000000-aaaaaa", "status": "queued", "priority": 5}
                    ]
                }
            raise RemoteError("local", "status", "host is busy")

    main(["host", "add", "local", "--gpus", GPU])
    (Path(control_env) / "config/config.toml").write_text('s3_bucket = "bucket"\n')
    s3 = FakeS3Client()
    monkeypatch.setattr("gpuc.control.s3index.S3Index.client", property(lambda self: s3))
    S3Index("bucket", s3).put_spec_document(
        "20260101-000000-aaaaaa", {"command": "true", "priority": 50, "some_future_field": 1}
    )
    monkeypatch.setattr("gpuc.control.remote.open_session", lambda *a, **k: Moved())
    capsys.readouterr()
    argv = ["reorder", "20260101-000000-aaaaaa", "--priority", "5", "--host", "local", "--json"]
    assert main(argv) == 0
    mirrored = S3Index("bucket", s3).get_spec("20260101-000000-aaaaaa")
    assert mirrored["priority"] == 5
    assert mirrored["some_future_field"] == 1
    # The host could not be asked where the job landed, which is a document of
    # nulls and never a failed reorder.
    document = one_job(capsys)
    assert (document["priority"], document["queue_position"]) == (5, None)


def test_reorder_says_so_when_the_mirror_kept_the_old_priority(
    control_env: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    class Moved:
        entry = host_entry(name="local", gpus=[GPU])
        config = HostConfig(host="local", gpus=[GPU])

        def host_json(self, args: str, *, timeout: float = 0.0, check: bool = True) -> object:
            if args.startswith("reorder"):
                return {
                    "jobs": [
                        {"job_id": "20260101-000000-aaaaaa", "status": "queued", "priority": 5}
                    ]
                }
            raise RemoteError("local", "status", "host is busy")

    main(["host", "add", "local", "--gpus", GPU])
    (Path(control_env) / "config/config.toml").write_text('s3_bucket = "bucket"\n')
    monkeypatch.setattr(
        "gpuc.control.s3index.S3Index.client", property(lambda self: FakeS3Client())
    )
    monkeypatch.setattr("gpuc.control.remote.open_session", lambda *a, **k: Moved())
    capsys.readouterr()
    argv = ["reorder", "20260101-000000-aaaaaa", "--priority", "5", "--host", "local", "--json"]
    assert main(argv) == 0
    warnings = one_job(capsys)["warnings"]
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
        "gpuc.control.remote.open_session",
        lambda *a, **k: as_session(
            StubSession(
                [
                    {
                        "jobs": [
                            {
                                "job_id": "20260101-000000-aaaaaa",
                                "estimated_runtime_min": 150.0,
                                "status": "running",
                            }
                        ]
                    }
                ]
            )
        ),
    )
    capsys.readouterr()
    argv = ["estimate", "20260101-000000-aaaaaa", "--minutes", "150", "--host", "local", "--json"]
    assert main(argv) == 0
    warnings = one_job(capsys)["warnings"]
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
    monkeypatch.setattr("gpuc.control.cli.make_provider", lambda *a, **k: provider)
    capsys.readouterr()
    assert main(["pods", "--no-heartbeat", "--json"]) == 0
    document = one_document(capsys)
    (pod,) = document["pods"]  # type: ignore[misc]
    assert pod["name"] == "gpuc-e2e-aaa"
    assert pod["host"] is None
    assert pod["heartbeat_age_s"] is None
    assert document["others"] == [{"id": "podF", "name": "subrep-other", "status": "RUNNING"}]


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
            "assigned": True,
            "shared": False,
        }
    ]
    assert document["assigned_gpus"] is None
    assert any("uv is missing" in note for note in document["notes"])  # type: ignore[union-attr]


def test_host_probe_shows_only_assigned_gpus_unless_all_gpus_is_asked_for(
    control_env: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from gpuc.control.probe import parse_probe

    register_host(name="gpubox", kind="ssh", ssh="me@box", gpus="1")
    sample = (
        "===driver===\n580.173.02\n"
        "===gpus===\n0, GPU-1111, NVIDIA A40, 46068 MiB\n1, GPU-2222, NVIDIA A40, 46068 MiB\n"
    )
    monkeypatch.setattr(
        "gpuc.control.cli.probe_host",
        lambda entry, *a, **k: parse_probe(entry.name, sample, entry.root, entry.config.gpus),
    )

    capsys.readouterr()
    assert main(["host", "probe", "gpubox"]) == 0
    default = capsys.readouterr().out
    assert "1 of 2 assigned to gpubox" in default
    assert "GPU-2222" in default and "GPU-1111" not in default

    assert main(["host", "probe", "gpubox", "--all-gpus"]) == 0
    everything = capsys.readouterr().out
    assert "GPU-1111" in everything
    assert "NVIDIA A40  46068 MiB  GPU-2222  (assigned)" in everything

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

    monkeypatch.setattr("gpuc.control.cli.update_cache", lambda *a, **k: locked())
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
    """Record every (host, health options) bootstrap was asked for; raise for the named hosts."""
    attempted: list[tuple[str, object]] = []

    def fake(entry: HostEntry, settings: Settings | None = None, **kwargs: Any):
        attempted.append((entry.name, kwargs.get("health_options")))
        error = (fail or {}).get(entry.name)
        if error is not None:
            raise error
        kwargs.get("report", print)(f"fake bootstrap of {entry.name}")
        updated = entry.with_cache(python="/root/python", uv="/root/uv").model_copy(
            update={"bootstrapped_at": utc_now()}
        )
        return updated, BootstrapResult(
            host=entry.name, home="/root/.gpuc", files=20, dispatcher_pid=4242
        )

    monkeypatch.setattr("gpuc.control.hosts.bootstrap_host", fake)
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
    asked = HealthOptions(min_mbps=0.1)
    assert attempted == [("gpubox", asked), ("local", asked)]
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
                kind="rental",
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
    assert "warning: host pod: ssh to pod failed" in captured.err
    assert "2/3 host(s) bootstrapped" in captured.out
    assert "failed: pod" in captured.out
    registry = load_registry()
    assert all(registry.require(name).bootstrapped_at for name in ("gpubox", "zbox"))
    assert registry.require("pod").bootstrapped_at is None


class _NoPods:
    """A provider whose account has no pod by that id any more."""

    def __init__(self, pod: Pod | None = None) -> None:
        self._pod = pod

    def get(self, pod_id: str) -> Pod | None:
        return self._pod

    def is_gone(self, pod: Pod | None) -> bool:
        return pod is None or pod.status == "TERMINATED"

    def is_dead(self, pod: Pod | None) -> bool:
        return pod is None or pod.status in ("EXITED", "TERMINATED")


def test_host_bootstrap_all_forgets_a_rental_the_provider_no_longer_has(
    control_env: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rental that ended itself is how one is meant to die, so exit 0."""
    from gpuc.control.bootstrap import BootstrapError

    register_host(name="gpubox", kind="ssh", ssh="me@box")
    with registry_transaction() as registry:
        registry.put(host_entry(name="pod", kind="rental", ssh="root@1.2.3.4", pod_id="p1"))
    bootstrapping(monkeypatch, fail={"pod": BootstrapError("ssh to pod failed")})
    monkeypatch.setattr("gpuc.control.actions.make_provider", lambda *a, **k: _NoPods())
    capsys.readouterr()

    assert main(["host", "bootstrap", "--all"]) == 0
    out = capsys.readouterr().out
    assert "1/2 host(s) bootstrapped" in out
    assert "forgotten, their pods are gone: pod" in out
    assert "pod" not in load_registry().hosts


def test_host_bootstrap_all_forgets_a_terminated_rental_too(
    control_env: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The case #68 describes: the pod is still in the account, TERMINATED."""
    from gpuc.control.bootstrap import BootstrapError

    with registry_transaction() as registry:
        registry.put(host_entry(name="pod", kind="rental", ssh="root@1.2.3.4", pod_id="p1"))
    bootstrapping(monkeypatch, fail={"pod": BootstrapError("ssh to pod failed")})
    terminated = Pod(id="p1", name="gpuc-pod", status="TERMINATED", cost_usd_hr=0.0)
    monkeypatch.setattr("gpuc.control.actions.make_provider", lambda *a, **k: _NoPods(terminated))
    capsys.readouterr()

    assert main(["host", "bootstrap", "--all"]) == 0
    assert "p1 is TERMINATED" in capsys.readouterr().out
    assert load_registry().hosts == {}


def test_host_bootstrap_all_keeps_a_rental_the_provider_cannot_be_asked_about(
    control_env: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """No answer is not `it ended`: the failure stands and the entry stays."""
    from gpuc.control.bootstrap import BootstrapError
    from gpuc.control.providers.base import ProviderError

    with registry_transaction() as registry:
        registry.put(host_entry(name="pod", kind="rental", ssh="root@1.2.3.4", pod_id="p1"))
    bootstrapping(monkeypatch, fail={"pod": BootstrapError("ssh to pod failed")})

    def unavailable(*a: object, **k: object) -> object:
        raise ProviderError("RUNPOD_API_KEY is not set")

    monkeypatch.setattr("gpuc.control.actions.make_provider", unavailable)
    capsys.readouterr()

    assert main(["host", "bootstrap", "--all"]) == 1
    captured = capsys.readouterr()
    assert "failed: pod" in captured.out
    assert "pod status unavailable" in captured.out
    assert "pod" in load_registry().hosts


def test_host_bootstrap_all_counts_the_hosts_it_could_not_read(
    control_env: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A host this build cannot parse was not bootstrapped either: never say
    "all", and never exit 0 over it."""
    register_host(name="gpubox", kind="ssh", ssh="me@box")
    document = json.loads(hosts_file().read_text())
    document["hosts"]["bad"] = {"name": "bad", "kind": "not a kind", "port": "twenty-two"}
    hosts_file().write_text(json.dumps(document))
    bootstrapping(monkeypatch)
    capsys.readouterr()

    assert main(["host", "bootstrap", "--all"]) == 1
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

    assert main(["host", "bootstrap", "--all"]) == EXIT_INTERRUPTED
    err = capsys.readouterr().err
    assert "interrupted during local" in err
    assert "1/2 host(s) bootstrapped" in err


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
    monkeypatch.setattr("gpuc.control.hosts.make_provider", lambda *a, **k: provider)
    fake_host.put_file(
        json.dumps(
            {
                "host": "gpuc-e2e-aaa",
                "gpus": ["GPU-1111"],
                "idle_minutes": 4.0,
                "provider": {"kind": "runpod", "pod_id": "pod1", "created_at": "2026-09-15T12:00"},
            }
        ),
        fake_host.config_path,
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
        "rental",
        "pod1",
        "root@1.2.3.4",
        22000,
    )
    assert entry.config.gpus == ["GPU-1111"] and entry.config.idle_minutes == 4.0
    out = capsys.readouterr().out
    assert "adopted the config on the host" in out
    # A pod with a dispatcher ends itself; there is nothing to warn about.
    assert "nothing has bootstrapped this pod" not in out


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


def test_host_add_pod_says_an_unbootstrapped_pod_will_never_end_itself(
    control_env: Path,
    fake_host: FakeHost,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pod nobody has set up has no dispatcher, so nothing idles it out: it
    bills until bootstrap gives it one or a person ends it. Say so."""
    monkeypatch.setenv("RUNPOD_API_KEY", "test-key")
    provider = FakeProvider()
    provider.adopt(running_pod("gpuc-e2e-aaa", "pod1"))
    monkeypatch.setattr("gpuc.control.hosts.make_provider", lambda *a, **k: provider)

    assert main(["host", "add", "rented", "--pod", "pod1", "--gpus", "GPU-1111"]) == 0

    out = capsys.readouterr().out
    assert "wrote its first config" in out
    assert "nothing has bootstrapped this pod" in out
    assert "gpuc host bootstrap rented" in out


# -- ending a pod on purpose ---------------------------------------------------


def rented_host(monkeypatch: pytest.MonkeyPatch) -> FakeProvider:
    monkeypatch.setenv("RUNPOD_API_KEY", "test-key")
    provider = FakeProvider()
    provider.adopt(running_pod("gpuc-e2e-aaa", "pod1"))
    monkeypatch.setattr("gpuc.control.cli.make_provider", lambda *a, **k: provider)
    register_host(name="gpuc-e2e-aaa", kind="rental", ssh="root@1.2.3.4", pod_id="pod1")
    return provider


def test_host_terminate_ends_the_pod_and_says_what_it_cost(
    control_env: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = rented_host(monkeypatch)
    monkeypatch.setattr(
        "gpuc.control.remote.open_session",
        lambda *a, **k: as_session(StubSession([{"host": "gpuc-e2e-aaa", "jobs": []}])),
    )

    assert main(["host", "terminate", "gpuc-e2e-aaa"]) == 0

    assert provider.terminated == ["pod1"]
    out = capsys.readouterr().out
    assert "terminated gpuc-e2e-aaa (pod1) (was $0.490/h)" in out
    assert "forgotten here" in out


def test_host_terminate_of_a_busy_pod_is_exit_1_and_says_how_to_insist(
    control_env: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = rented_host(monkeypatch)
    payload = {
        "host": "gpuc-e2e-aaa",
        "jobs": [{"job_id": "j1", "name": "train", "status": "running", "phase": "main"}],
    }
    monkeypatch.setattr(
        "gpuc.control.remote.open_session", lambda *a, **k: as_session(StubSession([payload]))
    )

    assert main(["host", "terminate", "gpuc-e2e-aaa"]) == EXIT_ERROR

    assert provider.terminated == []
    assert "gpuc-e2e-aaa" in load_registry().hosts
    err = capsys.readouterr().err
    assert "train (j1)" in err and "--force" in err


def test_an_existing_gpu_overlap_is_refused_until_it_is_fixed(
    control_env: Path, fake_host: FakeHost, capsys: pytest.CaptureFixture[str]
) -> None:
    """A hand-edited config with one card in both lists is judged on every
    write, whatever the command was about: the fix is the one way on."""
    fake_host.put_file(
        '{"host": "gpubox", "gpus": ["0"], "shared_gpus": ["0", "1"]}',
        fake_host.config_path,
    )
    assert main(["host", "add", "gpubox", "--ssh", "me@box"]) == 0
    assert main(["host", "set", "gpubox", "--idle-min", "30"]) == EXIT_ERROR
    assert "both --gpus and --shared-gpus" in capsys.readouterr().err
    assert main(["host", "set", "gpubox", "--idle-min", "30", "--shared-gpus", "1"]) == 0
    assert fake_host.config is not None and fake_host.config["idle_minutes"] == 30.0


def test_reorder_refuses_to_report_a_move_the_host_did_not_say_it_made(
    control_env: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A host answer with no `status` (an older build's `{"reordered": true}`
    shape included) is an error: success for a job the host may never have
    touched is worse than a failure."""

    class Older:
        def host_json(self, args: str, *, timeout: float = 0.0, check: bool = True) -> object:
            if args.startswith("reorder"):
                return {"jobs": [{"job_id": "20260101-000000-aaaaaa", "reordered": True}]}
            raise RemoteError("local", "status", "host is busy")

    register_host(name="local", gpus=GPU)
    monkeypatch.setattr("gpuc.control.remote.open_session", lambda *a, **k: Older())
    capsys.readouterr()
    argv = ["reorder", "20260101-000000-aaaaaa", "--priority", "5", "--host", "local", "--json"]
    assert main(argv) == EXIT_ERROR
    job = one_job(capsys)
    assert "did not say what it did with 20260101-000000-aaaaaa" in str(job["error"])
    assert "priority" not in job


def test_the_commands_the_dashboard_could_want_live_outside_the_cli() -> None:
    """cli.py is argparse and text; what a command *does* is somewhere the web can call.

    A structural pin, not a behavioural one: the logic that used to live in
    these `cmd_*` bodies must stay importable without the CLI.
    """
    import inspect

    from gpuc.control import actions, cli, hosts, submitting

    moved = {
        submitting: ["submit_job", "requeue_job", "ensure_package_current", "enqueue"],
        hosts: ["add_host", "set_host", "bootstrap_and_record", "bootstrap_every_host"],
        actions: ["unhosted_jobs", "status", "locate", "read_log"],
    }
    for module, names in moved.items():
        for name in names:
            assert hasattr(module, name), f"{module.__name__} lost {name}"
    for name in [
        "ensure_package_current",
        "check_runpod_args",
        "runpod_target",
        "BootstrapTally",
        "_print_unhosted",
        "_outputs_lost_ids",
        "_pod_address",
        "_refuse_a_taken_name",
    ]:
        assert not hasattr(cli, name), f"cli.py defines {name} again"
    calls = {
        cli.cmd_submit: "submit_job",
        cli.cmd_requeue: "requeue_job",
        cli.cmd_host_add: "add_host",
        cli.cmd_host_set: "set_host",
        cli.cmd_host_bootstrap: "bootstrap_every_host",
        cli.cmd_status: "status",
    }
    for command, callee in calls.items():
        assert f"{callee}(" in inspect.getsource(command), f"{command.__name__} skips {callee}"
