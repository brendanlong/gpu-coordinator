"""The shared uv cache: nothing may break uv's linking, and bootstrap must
restore it on a host where gpuc home and `$HOME` are different filesystems.

uv materialises a venv by reflinking or hardlinking wheels out of its cache.
Both only work inside one filesystem, and `UV_LINK_MODE=copy` disables them
outright, so the two ways to lose ~6.5 GB per torch job are setting that
variable and letting the cache drift onto another volume.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gpuc.control.bootstrap import bootstrap_host, cache_dir_beside, resolve_cache_dir
from gpuc.control.probe import parse_probe
from gpuc.control.remote import HostSession, env_prefix
from gpuc.host import dispatcher, health, jobs, runner
from gpuc.host.jobs import HostConfig
from tests.conftest import host_entry, make_spec
from tests.test_bootstrap import ScriptedHost, entry

# -- (a) nothing we ship may defeat uv's linking ------------------------------

# Anchored at the repo, not at the cwd: `pytest` from anywhere but the project
# root would otherwise read nothing and pass.
REPO = Path(__file__).resolve().parents[1]
SHIPPED = [
    REPO / "gpuc/host/runner.py",
    REPO / "gpuc/host/dispatcher.py",
    REPO / "gpuc/host/jobs.py",
    REPO / "gpuc/host/paths.py",
    REPO / "gpuc/control/submit.py",
]


def test_nothing_in_the_job_path_ever_sets_uv_link_mode() -> None:
    for path in SHIPPED:
        assert path.is_file(), path
        assert "UV_LINK_MODE" not in path.read_text(), path


def test_the_runner_does_not_invent_a_uv_cache_dir(
    gpuc_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("UV_CACHE_DIR", raising=False)
    monkeypatch.delenv("UV_LINK_MODE", raising=False)
    env = runner.build_env(make_spec(), [], jobs.read_config())
    assert "UV_CACHE_DIR" not in env
    assert "UV_LINK_MODE" not in env


def test_the_dispatcher_does_not_invent_a_uv_cache_dir(
    gpuc_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("UV_CACHE_DIR", raising=False)
    monkeypatch.delenv("UV_LINK_MODE", raising=False)
    env = dispatcher._child_env(Path("/pkg"))
    assert "UV_CACHE_DIR" not in env
    assert "UV_LINK_MODE" not in env


def test_the_host_config_env_is_the_only_thing_that_sets_the_cache(gpuc_home: Path) -> None:
    config = HostConfig(host="h", env={"UV_CACHE_DIR": "/vol/me/.cache/uv"})
    jobs.write_config(config)
    assert runner.build_env(make_spec(), [], config)["UV_CACHE_DIR"] == "/vol/me/.cache/uv"
    assert dispatcher._child_env(Path("/pkg"))["UV_CACHE_DIR"] == "/vol/me/.cache/uv"


def test_a_job_can_still_override_the_host_cache(gpuc_home: Path) -> None:
    config = HostConfig(host="h", env={"UV_CACHE_DIR": "/vol/me/.cache/uv"})
    spec = make_spec(env={"UV_CACHE_DIR": "/tmp/mine"})
    assert runner.build_env(spec, [], config)["UV_CACHE_DIR"] == "/tmp/mine"


# -- the host's UV_CACHE_DIR reaches everything that runs on the host ---------


def test_the_hosts_cache_dir_reaches_every_remote_step() -> None:
    """One variable in the host's own `env`, which is where it lives: the
    registry's `cache_dir` is a reading of it, not a second copy."""
    host = host_entry(name="h", cache_dir="/vol/me/.cache/uv")
    assert host.cache_dir == "/vol/me/.cache/uv"
    assert host.env["UV_CACHE_DIR"] == "/vol/me/.cache/uv"
    assert 'UV_CACHE_DIR="/vol/me/.cache/uv"' in env_prefix(host.env)
    session = HostSession(host, ScriptedHost(), "/vol/me/gpuc", "/usr/bin/python3")
    assert session.env["UV_CACHE_DIR"] == "/vol/me/.cache/uv"


def test_no_cache_dir_means_no_variable() -> None:
    assert "UV_CACHE_DIR" not in host_entry(name="h").env
    assert host_entry(name="h").cache_dir is None


# -- (b) the bootstrap rule ---------------------------------------------------


def test_the_cache_goes_beside_gpuc_home_not_inside_it() -> None:
    # Inside would make `rm -rf` of gpuc home take the cache with it.
    assert cache_dir_beside("/workspace/me/gpuc") == "/workspace/me/.cache/uv"
    assert cache_dir_beside("/workspace/me/gpuc/") == "/workspace/me/.cache/uv"
    assert cache_dir_beside("/home/u/.gpuc") == "/home/u/.cache/uv"
    assert cache_dir_beside("/gpuc") == "/gpuc/uv-cache"


def test_one_filesystem_leaves_the_cache_alone(control_env: Path) -> None:
    host = ScriptedHost(cache_dev="66", home_dev="66")
    assert resolve_cache_dir(host, entry(), "/uv", "/home/u/.gpuc", lambda _: None) is None


def test_two_filesystems_move_the_cache_next_to_gpuc_home(control_env: Path) -> None:
    host = ScriptedHost(uv_cache="/root/.cache/uv", cache_dev="66", home_dev="99")
    picked = resolve_cache_dir(host, entry(), "/uv", "/workspace/me/gpuc", lambda _: None)
    assert picked == "/workspace/me/.cache/uv"


def test_an_unreadable_filesystem_changes_nothing(control_env: Path) -> None:
    host = ScriptedHost(cache_dev="unknown", home_dev="99")
    assert resolve_cache_dir(host, entry(), "/uv", "/workspace/me/gpuc", lambda _: None) is None


def test_a_cache_the_host_already_names_is_never_overridden(control_env: Path) -> None:
    """However it got there -- `--cache-dir`, `--env`, or an earlier bootstrap
    -- the host's own config wins. This is the one key bootstrap fills in
    itself, and only when it is empty."""
    host = ScriptedHost(cache_dev="66", home_dev="99")
    for pinned in (entry(cache_dir="/mnt/big/uv"), entry(env={"UV_CACHE_DIR": "/mnt/big/uv"})):
        assert resolve_cache_dir(host, pinned, "/uv", "/workspace/me/gpuc", lambda _: None) is None


def test_bootstrap_writes_the_cache_dir_into_the_hosts_config(control_env: Path) -> None:
    host = ScriptedHost(cache_dev="66", home_dev="99")
    updated, _ = bootstrap_host(entry(), transport=host, report=lambda _: None)
    assert updated.cache_dir == "/home/u/.cache/uv"
    assert host.config is not None
    assert host.config["env"] == {"UV_CACHE_DIR": "/home/u/.cache/uv"}


def test_bootstrap_on_one_filesystem_writes_no_cache_dir(control_env: Path) -> None:
    host = ScriptedHost(cache_dev="66", home_dev="66")
    updated, _ = bootstrap_host(entry(), transport=host, report=lambda _: None)
    assert updated.cache_dir is None
    assert host.config is not None
    assert host.config["env"] == {}


def test_the_cache_is_resolved_before_uv_tool_install(control_env: Path) -> None:
    """`uv tool install` should already be filling the cache this host will use."""
    host = ScriptedHost(hf_present=False, cache_dev="66", home_dev="99")
    bootstrap_host(entry(), transport=host, report=lambda _: None)
    assert host.index_of("cache dir") < host.index_of("tool install huggingface_hub")
    install = host.events[host.index_of("tool install huggingface_hub")]
    assert 'UV_CACHE_DIR="/home/u/.cache/uv"' in install


# -- (c) probe and health both report it --------------------------------------


def test_health_reports_the_cache_size_and_a_shared_filesystem(gpuc_home: Path) -> None:
    cache = gpuc_home.parent / "uv-cache"
    cache.mkdir()
    (cache / "wheel.bin").write_bytes(b"w" * 40_000)
    check = health.check_uv_cache(HostConfig(host="h", env={"UV_CACHE_DIR": str(cache)}))
    assert check.ok and not check.warn
    assert "same filesystem" in check.detail
    assert isinstance(check.value, float | int) and check.value > 0


def test_health_warns_loudly_when_the_cache_is_on_another_filesystem(
    gpuc_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(health, "same_filesystem", lambda _a, _b: False)
    check = health.check_uv_cache(HostConfig(host="gpubox"))
    assert check.ok and check.warn
    assert "DIFFERENT filesystem" in check.detail
    assert "gpuc host set gpubox --cache-dir" in check.detail


def test_health_never_fails_a_host_over_its_cache(gpuc_home: Path) -> None:
    config = HostConfig(host="h", env={"UV_CACHE_DIR": "/nowhere/at/all"})
    assert health.check_uv_cache(config).ok


def test_the_configured_cache_dir_wins_over_the_process_environment(
    gpuc_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("UV_CACHE_DIR", "/from/environ")
    config = HostConfig(host="h", env={"UV_CACHE_DIR": "/from/config"})
    assert health.uv_cache_dir(config) == Path("/from/config")
    assert health.uv_cache_dir(HostConfig(host="h")) == Path("/from/environ")


PROBE_SHARED = """===uv_cache===
dir=/home/u/.cache/uv
size=18G
gpuc_home=/home/u/.gpuc
cache_dev=66
home_dev=66
"""

PROBE_SPLIT = """===uv_cache===
dir=/root/.cache/uv
size=12G
gpuc_home=/workspace/me/gpuc
cache_dev=66
home_dev=99
"""


def test_probe_reports_the_cache_size_and_that_it_is_shared() -> None:
    report = parse_probe("gpubox", PROBE_SHARED)
    assert report.uv_cache["size"] == "18G"
    assert report.cache_shares_gpuc_home_fs is True
    rendered = report.render()
    assert "uv_cache: /home/u/.cache/uv size 18G" in rendered
    assert "same filesystem as gpuc home /home/u/.gpuc: yes" in rendered
    assert "different filesystems" not in rendered


def test_probe_calls_out_a_split_cache_and_names_the_fix() -> None:
    report = parse_probe("pod", PROBE_SPLIT)
    assert report.cache_shares_gpuc_home_fs is False
    rendered = report.render()
    assert "same filesystem as gpuc home /workspace/me/gpuc: NO" in rendered
    assert "copies each one instead" in rendered
    assert "gpuc host bootstrap pod" in rendered


def test_probe_says_unknown_rather_than_guessing() -> None:
    report = parse_probe("h", "===uv_cache===\ndir=/x\nsize=absent\ncache_dev=unknown\n")
    assert report.cache_shares_gpuc_home_fs is None
    assert "unknown" in report.render()


# -- gpuc host clean --uv-cache -----------------------------------------------


class PruningHost(ScriptedHost):
    def _answer(self, command: str) -> tuple[int, str]:
        if "cache prune" in command:
            return 0, "before=18G\nafter=11G\ndir=/home/u/.cache/uv\n"
        return super()._answer(command)


def test_host_clean_prunes_rather_than_cleans(control_env: Path) -> None:
    from gpuc.control.clean import prune_uv_cache

    host = PruningHost()
    report = prune_uv_cache(entry(uv="/home/u/.local/bin/uv"), transport=host)
    assert "pruned 18G -> 11G" in report.render()
    # Sizes in bytes are what a script wants, and a host that printed none
    # (an older prune script, a `du` that failed) leaves them null, not zero.
    assert report.document() == {
        "host": "h",
        "cache_dir": "/home/u/.cache/uv",
        "before": "18G",
        "after": "11G",
        "before_bytes": None,
        "after_bytes": None,
        "freed_bytes": None,
    }
    pruned = next(e for e in host.events if "cache prune" in e)
    # `uv cache clean` would throw away the wheels the next job wants to link.
    assert "cache clean" not in pruned
    assert "/home/u/.local/bin/uv" in pruned


def test_host_clean_passes_the_hosts_cache_dir(control_env: Path) -> None:
    from gpuc.control.clean import prune_uv_cache

    host = PruningHost()
    prune_uv_cache(entry(cache_dir="/vol/me/.cache/uv"), transport=host)
    pruned = next(e for e in host.events if "cache prune" in e)
    assert 'UV_CACHE_DIR="/vol/me/.cache/uv"' in pruned


def test_a_failed_prune_is_an_error_not_a_silent_no_op(control_env: Path) -> None:
    from gpuc.control.clean import CleanError, prune_uv_cache

    class Broken(ScriptedHost):
        def _answer(self, command: str) -> tuple[int, str]:
            if "cache prune" in command:
                return 2, "disk is read-only"
            return super()._answer(command)

    with pytest.raises(CleanError, match="read-only"):
        prune_uv_cache(entry(), transport=Broken())
