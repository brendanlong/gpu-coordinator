"""`--persistent-root` (gpuc home on a volume that survives a restart), the
per-host `env` plumbing it shares wiring with, and `gpuc host set`.

The env chain under test is registry entry -> on-host config.json `env` ->
dispatcher child env -> runner job env -> what the job's own command sees.
Nothing populates that env automatically: uv, its caches and the aws bundle
live in `$HOME` on every host, because bootstrap reinstalls them in seconds and
these shared volumes are slower than the local disk.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from gpuc.control.bootstrap import BootstrapError, bootstrap_host, env_prefix, remote_path
from gpuc.control.cli import EXIT_NOT_FOUND, EXIT_USAGE, main
from gpuc.control.config import HostEntry, load_registry
from gpuc.control.remote import host_command
from gpuc.host import dispatcher, health, jobs, paths, queue, runner
from gpuc.host.jobs import HostConfig, JobState
from tests.conftest import FAKE_GPUS, fake_smi, make_spec
from tests.test_bootstrap import ScriptedHost
from tests.test_runner import deps, log_of, prepare

ROOT = "/mnt/ssd-2/brendan"
HOST_ENV = {"UV_CACHE_DIR": "/mnt/ssd-2/brendan/uv-cache", "HF_HOME": "/scratch/hf"}


def rooted(**overrides: object) -> HostEntry:
    return HostEntry.model_validate({"name": "spar", "persistent_root": ROOT, **overrides})


# -- the registry entry ---------------------------------------------------


def test_no_persistent_root_changes_nothing() -> None:
    entry = HostEntry(name="plain")
    assert entry.root is None
    assert entry.remote_home == "$HOME/.gpuc"
    assert entry.host_config().env == {}
    assert remote_path(entry) == 'PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"'
    assert env_prefix(entry) == ""


def test_a_persistent_root_moves_only_gpuc_home() -> None:
    entry = rooted()
    assert entry.remote_home == f"{ROOT}/gpuc"
    # Not the caches, not uv, not the aws bundle: a root is for the state that
    # cannot be reinstalled, and /mnt is the slow disk.
    assert entry.host_config().env == {}
    assert env_prefix(entry) == ""


def test_a_trailing_slash_does_not_double_up() -> None:
    assert rooted(persistent_root=f"{ROOT}/").remote_home == f"{ROOT}/gpuc"


def test_an_explicit_gpuc_home_still_wins_over_the_root() -> None:
    assert rooted(gpuc_home="/tmp/elsewhere").remote_home == "/tmp/elsewhere"


def test_a_hand_set_env_reaches_the_host_config() -> None:
    config = rooted(gpus=["GPU-a"], env=dict(HOST_ENV)).host_config()
    assert config.env == HOST_ENV
    assert HostConfig.from_dict(json.loads(json.dumps(config.to_dict()))).env == HOST_ENV


# -- HostConfig.apply_env -------------------------------------------------


def test_apply_env_merges_the_env_and_fronts_path_with_any_bin_dir() -> None:
    config = HostConfig(env={"UV_TOOL_BIN_DIR": "/opt/bin", "HF_HOME": "/scratch/hf"})
    env = config.apply_env({"HOME": "/home/brendan", "PATH": "/usr/bin"})
    assert env["HF_HOME"] == "/scratch/hf"
    assert env["PATH"].split(os.pathsep)[0] == "/opt/bin"
    assert env["PATH"].endswith("/usr/bin")


def test_apply_env_is_a_noop_without_a_host_env() -> None:
    env = HostConfig().apply_env({"HOME": "/home/brendan", "PATH": "/usr/bin"})
    assert env == {"HOME": "/home/brendan", "PATH": "/usr/bin"}


def test_an_env_that_names_no_bin_dir_leaves_path_alone() -> None:
    config = HostConfig(env=dict(HOST_ENV))
    assert config.bin_dirs() == []
    assert config.apply_env({"HOME": "/h", "PATH": "/usr/bin"})["PATH"] == "/usr/bin"


def test_a_bin_dir_is_never_listed_twice() -> None:
    config = HostConfig(env={"UV_INSTALL_DIR": "/opt/bin", "UV_TOOL_BIN_DIR": "/opt/bin"})
    once = config.apply_env({"HOME": "/h", "PATH": "/usr/bin"})["PATH"]
    assert once == "/opt/bin:/usr/bin"
    assert config.apply_env({"HOME": "/h", "PATH": once})["PATH"] == once


# -- dispatcher and runner ------------------------------------------------


def with_host_env(gpus: list[str] | None = None) -> None:
    jobs.write_config(HostConfig(host="spar", gpus=gpus or list(FAKE_GPUS), env=dict(HOST_ENV)))


def test_the_dispatcher_child_env_carries_the_host_env(gpuc_home: Path) -> None:
    with_host_env()
    env = dispatcher._child_env(Path("/pkg"))
    assert env["UV_CACHE_DIR"] == HOST_ENV["UV_CACHE_DIR"]
    assert env["HF_HOME"] == "/scratch/hf"
    assert env["PYTHONPATH"].startswith("/pkg")


def test_a_missing_config_leaves_the_dispatcher_env_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GPUC_HOME", str(tmp_path / "nothing-here"))
    assert "UV_CACHE_DIR" not in dispatcher._child_env(Path("/pkg"))


def test_build_env_gives_the_job_the_host_env(gpuc_home: Path) -> None:
    with_host_env()
    spec = make_spec(job_id="j1")
    jobs.write_state("j1", JobState())
    env = runner.build_env(spec, [FAKE_GPUS[0]])
    assert env["UV_CACHE_DIR"] == HOST_ENV["UV_CACHE_DIR"]
    assert env["HF_HOME"] == "/scratch/hf"


def test_a_job_can_override_the_host_env(gpuc_home: Path) -> None:
    with_host_env()
    spec = make_spec(job_id="j2", env={"UV_CACHE_DIR": "/tmp/mine"})
    jobs.write_state("j2", JobState())
    env = runner.build_env(spec, [])
    assert env["UV_CACHE_DIR"] == "/tmp/mine"
    assert env["HF_HOME"] == "/scratch/hf"


def test_the_job_command_really_sees_the_host_env(gpuc_home: Path) -> None:
    with_host_env()
    job_id = prepare(command='echo "cache=$UV_CACHE_DIR"')
    assert runner.run_job(job_id, deps()) == 0
    assert f"cache={HOST_ENV['UV_CACHE_DIR']}" in log_of(job_id)


def test_a_dispatched_job_sees_it_too(gpuc_home: Path) -> None:
    """The dispatcher spawns the runner as a real child, so this covers the
    whole chain (config -> dispatcher env -> runner -> bash) in one go."""
    with_host_env()
    queue.enqueue(make_spec(command='echo "cache=$UV_CACHE_DIR"', gpus=0))
    loop = dispatcher.Dispatcher(dispatcher.DispatcherDeps(smi=fake_smi()))
    loop.run_once()
    job_id = jobs.list_job_ids()[0]
    deadline = time.monotonic() + 60
    while not jobs.read_state(job_id).finished and time.monotonic() < deadline:
        time.sleep(0.05)
        loop.run_once()
    assert jobs.read_state(job_id).status == "succeeded"
    assert f"cache={HOST_ENV['UV_CACHE_DIR']}" in log_of(job_id)


def test_health_measures_the_gpuc_home_filesystem(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The disk floor must be about the volume that actually holds the jobs.

    With a persistent root, gpuc home is on the shared volume, and `check_disk`
    measures `paths.home()` -- so the number bootstrap prints is the root's
    free space, not the overlay's.
    """
    monkeypatch.setenv("GPUC_HOME", str(tmp_path / "ssd" / "gpuc"))
    jobs.write_config(HostConfig(host="spar"))
    check = health.check_disk(min_free_gb=0.0)
    assert check.ok
    assert str(tmp_path / "ssd" / "gpuc") in check.detail
    assert paths.home() == tmp_path / "ssd" / "gpuc"


# -- bootstrap ------------------------------------------------------------


class RootedHost(ScriptedHost):
    """A scripted host whose gpuc home is under the persistent root."""

    def _answer(self, command: str) -> tuple[int, str]:
        if command.startswith("set -e; root="):
            return 0, f"drwx------ 2 brendan brendan 4096 Jan  1 00:00 {ROOT}\n"
        if command.startswith("printf %s"):
            return 0, f"{ROOT}/gpuc"
        return super()._answer(command)


def bootstrapped(**overrides: object) -> tuple[RootedHost, HostEntry]:
    host = RootedHost()
    updated, _ = bootstrap_host(
        rooted(gpus=["GPU-a"], **overrides), transport=host, report=lambda _: None
    )
    return host, updated


def test_bootstrap_creates_the_root_before_anything_else(control_env: Path) -> None:
    host, _ = bootstrapped()
    setup = host.events[0]
    assert setup.startswith(f"set -e; root={ROOT};")
    assert 'if [ ! -d "$root" ]; then mkdir -p "$root"; chmod 700 "$root"; fi' in setup


def test_bootstrap_does_not_touch_an_existing_roots_mode(control_env: Path) -> None:
    host, _ = bootstrapped()
    # The only chmod is inside the "it did not exist" branch.
    assert host.events[0].count("chmod 700") == 1


def test_a_host_without_a_root_gets_no_root_step(control_env: Path) -> None:
    host = ScriptedHost()
    bootstrap_host(HostEntry(name="h", gpus=["GPU-a"]), transport=host, report=lambda _: None)
    assert not any(e.startswith("set -e; root=") for e in host.events)


def test_the_package_and_config_land_under_the_root(control_env: Path) -> None:
    host, _ = bootstrapped()
    assert host.rsyncs[0][1] == f"{ROOT}/gpuc/pkg"
    config = json.loads(host.puts[f"{ROOT}/gpuc/config.json"][0])
    assert (config["gpus"], config["env"]) == (["GPU-a"], {})


def test_uv_and_the_aws_bundle_stay_in_home(control_env: Path) -> None:
    host = RootedHost(uv_present=False, aws_present=False)
    bootstrap_host(rooted(gpus=["GPU-a"]), transport=host, report=lambda _: None)
    install = next(e for e in host.events if "awscli-exe" in e)
    assert '-i "$HOME/.local/aws-cli" -b "$HOME/.local/bin"' in install
    assert ROOT not in install
    assert next(e for e in host.events if "astral.sh/uv" in e).endswith("| sh")


def test_the_aws_bundle_can_be_unpacked_without_unzip(control_env: Path) -> None:
    host = RootedHost(aws_present=False)
    bootstrap_host(rooted(gpus=["GPU-a"]), transport=host, report=lambda _: None)
    install = next(e for e in host.events if "awscli-exe" in e)
    # A slim container image has no unzip and no sudo to install one.
    assert "command -v unzip" in install
    assert 'python3 -m zipfile -e "$tmp/awscliv2.zip" "$tmp"' in install
    assert 'chmod -R u+x "$tmp/aws"' in install


def test_every_host_package_invocation_pins_the_root_home(control_env: Path) -> None:
    host, _ = bootstrapped()
    for event in host.events:
        if "gpuc.host" in event or "from gpuc.host import" in event:
            assert f'GPUC_HOME="{ROOT}/gpuc"' in event
            assert f'PYTHONPATH="{ROOT}/gpuc/pkg"' in event


def test_a_root_we_cannot_create_fails_bootstrap_with_advice(control_env: Path) -> None:
    class ReadOnly(RootedHost):
        def _answer(self, command: str) -> tuple[int, str]:
            if command.startswith("set -e; root="):
                return 1, "mkdir: cannot create directory: Permission denied"
            return super()._answer(command)

    host = ReadOnly()
    with pytest.raises(BootstrapError) as exc:
        bootstrap_host(rooted(gpus=["GPU-a"]), transport=host, report=lambda _: None)
    assert "Permission denied" in str(exc.value)
    assert "gpuc host set spar --persistent-root" in str(exc.value)
    assert not any("spawn_detached_dispatcher" in e for e in host.events)


def test_a_hand_set_env_is_given_to_every_remote_step(control_env: Path) -> None:
    host = RootedHost(python_present=False)
    bootstrap_host(
        rooted(gpus=["GPU-a"], env=dict(HOST_ENV)), transport=host, report=lambda _: None
    )
    for event in host.events:
        if "python find" in event or "python install" in event or "-m gpuc.host" in event:
            assert 'HF_HOME="/scratch/hf"' in event
    started = next(e for e in host.events if "spawn_detached_dispatcher" in e)
    assert 'HF_HOME="/scratch/hf"' in started


def test_the_dispatcher_is_started_with_any_env_bin_dir_first(control_env: Path) -> None:
    host = RootedHost()
    bootstrap_host(
        rooted(gpus=["GPU-a"], env={"UV_TOOL_BIN_DIR": "/opt/bin"}),
        transport=host,
        report=lambda _: None,
    )
    started = next(e for e in host.events if "spawn_detached_dispatcher" in e)
    assert started.startswith('PATH="/opt/bin:$HOME/.local/bin:$HOME/.cargo/bin:$PATH"')


def test_host_command_without_an_env_is_unchanged() -> None:
    assert host_command("/py", "/h/.gpuc", "status").startswith('GPUC_HOME="/h/.gpuc"')


# -- the CLI --------------------------------------------------------------


def test_host_add_records_a_persistent_root(
    control_env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["host", "add", "spar", "--ssh", "spar", "--persistent-root", ROOT]) == 0
    assert load_registry().require("spar").remote_home == f"{ROOT}/gpuc"
    assert f"persistent root {ROOT}" in capsys.readouterr().out


def test_host_add_records_env_pairs(control_env: Path) -> None:
    main(["host", "add", "spar", "--ssh", "spar", "--env", "HF_HOME=/scratch/hf", "--env", "A=b"])
    assert load_registry().require("spar").env == {"HF_HOME": "/scratch/hf", "A": "b"}


def test_a_malformed_env_pair_is_rejected(
    control_env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["host", "add", "spar", "--ssh", "spar", "--env", "HF_HOME"]) == EXIT_USAGE
    assert "--env wants KEY=VALUE" in capsys.readouterr().err


def test_host_set_edits_one_field_and_leaves_the_rest(control_env: Path) -> None:
    main(["host", "add", "spar", "--ssh", "spar", "--gpus", "GPU-a,GPU-b", "--idle-min", "7"])
    assert main(["host", "set", "spar", "--persistent-root", ROOT]) == 0
    entry = load_registry().require("spar")
    assert entry.persistent_root == ROOT
    assert entry.gpus == ["GPU-a", "GPU-b"]
    assert entry.idle_minutes == 7.0
    assert entry.ssh == "spar"


def test_host_set_replaces_the_gpu_list(control_env: Path) -> None:
    main(["host", "add", "spar", "--ssh", "spar", "--gpus", "GPU-a"])
    main(["host", "set", "spar", "--gpus", "GPU-b,GPU-c"])
    assert load_registry().require("spar").gpus == ["GPU-b", "GPU-c"]


def test_host_set_can_hand_every_gpu_back(control_env: Path) -> None:
    main(["host", "add", "spar", "--ssh", "spar", "--gpus", "GPU-a"])
    assert main(["host", "set", "spar", "--gpus", ""]) == 0
    assert load_registry().require("spar").gpus == []


def test_host_set_can_clear_the_root(control_env: Path) -> None:
    main(["host", "add", "spar", "--ssh", "spar", "--persistent-root", ROOT])
    main(["host", "set", "spar", "--persistent-root", ""])
    entry = load_registry().require("spar")
    assert entry.persistent_root is None
    assert entry.remote_home == "$HOME/.gpuc"


def test_host_set_replaces_the_whole_env(control_env: Path) -> None:
    main(["host", "add", "spar", "--ssh", "spar", "--env", "A=1", "--env", "B=2"])
    main(["host", "set", "spar", "--env", "B=3"])
    assert load_registry().require("spar").env == {"B": "3"}
    main(["host", "set", "spar", "--env", ""])
    assert load_registry().require("spar").env == {}


def test_host_set_changes_the_timers(control_env: Path) -> None:
    main(["host", "add", "spar", "--ssh", "spar"])
    main(["host", "set", "spar", "--idle-min", "3", "--ttl-hours", "0.5"])
    entry = load_registry().require("spar")
    assert (entry.idle_minutes, entry.ttl_hours) == (3.0, 0.5)


def test_host_set_with_no_flags_says_so(
    control_env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    main(["host", "add", "spar", "--ssh", "spar"])
    assert main(["host", "set", "spar"]) == EXIT_USAGE
    assert "changes nothing" in capsys.readouterr().err


def test_host_set_on_an_unknown_host_names_the_known_ones(
    control_env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["host", "set", "nope", "--idle-min", "1"]) == EXIT_NOT_FOUND
    assert "no host named 'nope'" in capsys.readouterr().err


def test_host_set_says_the_host_is_untouched_until_bootstrap(
    control_env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    main(["host", "add", "spar", "--ssh", "spar"])
    main(["host", "set", "spar", "--persistent-root", ROOT])
    assert "gpuc host bootstrap spar" in capsys.readouterr().out


def test_host_list_shows_the_root(control_env: Path, capsys: pytest.CaptureFixture[str]) -> None:
    main(["host", "add", "spar", "--ssh", "spar", "--persistent-root", ROOT])
    capsys.readouterr()
    main(["host", "list"])
    assert f"persistent root {ROOT} (gpuc home {ROOT}/gpuc)" in capsys.readouterr().out


# -- finding what was on a host that lost its state -----------------------


def index_job(job_id: str, host: str) -> None:
    from gpuc.control.s3index import IndexEntry, LocalIndex

    LocalIndex().record(IndexEntry(job_id=job_id, host=host, name=f"n-{job_id}"))


def test_status_all_lists_index_jobs_per_host(
    control_env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """After a wiped $HOME the host answers but knows nothing, so the index is
    the only list of what to `gpuc requeue`."""
    index_job("20260101-000000-aaaaaa", "spar")
    index_job("20260101-000001-bbbbbb", "other")
    capsys.readouterr()
    assert main(["status", "--all"]) == 0
    out = capsys.readouterr().out
    assert "20260101-000000-aaaaaa n-20260101-000000-aaaaaa host=spar" in out
    assert "host=other" in out
    assert "gpuc requeue 20260101-000000-aaaaaa --host spar" in out


def test_status_all_can_be_narrowed_to_the_host_being_recovered(
    control_env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    main(["host", "add", "spar", "--ssh", "spar"])
    index_job("20260101-000000-aaaaaa", "spar")
    index_job("20260101-000001-bbbbbb", "other")
    capsys.readouterr()
    main(["status", "--host", "spar", "--all"])
    out = capsys.readouterr().out
    assert "20260101-000000-aaaaaa" in out
    assert "20260101-000001-bbbbbb" not in out
