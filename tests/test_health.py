from __future__ import annotations

import json
import time
import urllib.error
from pathlib import Path

import pytest

from gpuc.host import __main__ as cli
from gpuc.host import health, jobs
from gpuc.host.jobs import HostConfig
from tests.conftest import FAKE_GPUS, fake_smi


def slow_downloader(url: str, max_bytes: int, timeout: float) -> int:
    time.sleep(0.05)
    return 1000


def fast_downloader(url: str, max_bytes: int, timeout: float) -> int:
    return max_bytes


def dead_downloader(url: str, max_bytes: int, timeout: float) -> int:
    return 0


def failing_downloader(url: str, max_bytes: int, timeout: float) -> int:
    raise urllib.error.URLError("name resolution failed")


def test_driver_check_reports_the_version() -> None:
    check = health.check_driver(fake_smi())
    assert check.ok and check.value == "580.173.02"


def test_driver_check_fails_loudly_without_nvidia_smi() -> None:
    def broken(args: list[str]) -> str:
        from gpuc.host.gpus import GpuError

        raise GpuError("nvidia-smi not found on PATH")

    check = health.check_driver(broken)
    assert not check.ok
    assert "nvidia-smi not found" in check.detail


def test_uuid_check_fails_when_an_owned_uuid_is_absent() -> None:
    check = health.check_gpu_uuids(["GPU-gone"], fake_smi())
    assert not check.ok
    assert "GPU-gone" in check.detail


def test_uuid_check_resolves_owned_indices_and_names_the_ones_that_are_gone() -> None:
    """A host may own its share of a box by index, and the index that no longer
    exists is exactly the failure this check is for."""
    ok = health.check_gpu_uuids(["0", "1"], fake_smi())
    assert ok.ok and ok.value == 2

    check = health.check_gpu_uuids(["0", "7"], fake_smi())
    assert not check.ok
    assert "7" in check.detail
    assert f"0={FAKE_GPUS[0]}" in check.detail


def test_disk_check_measures_the_gpuc_volume(gpuc_home: Path) -> None:
    assert health.check_disk(min_free_gb=0.0).ok
    failing = health.check_disk(min_free_gb=1e9)
    assert not failing.ok
    assert str(gpuc_home) in failing.detail


def test_download_check_reports_mbps_and_enforces_the_floor() -> None:
    ok = health.check_download("http://x", downloader=fast_downloader, min_mbps=1.0)
    assert ok.ok
    assert "MB/s" in ok.detail
    assert isinstance(ok.value, float)

    slow = health.check_download("http://x", downloader=slow_downloader, min_mbps=1000.0)
    assert not slow.ok


def test_download_check_treats_zero_bytes_as_a_dead_network() -> None:
    check = health.check_download("http://x", downloader=dead_downloader)
    assert not check.ok
    assert "0 bytes" in check.detail


def test_a_download_error_is_a_warning_not_a_failed_host() -> None:
    check = health.check_download("http://x", downloader=failing_downloader)
    assert check.ok and check.warn
    assert "could not measure throughput" in check.detail
    assert "name resolution failed" in check.detail


def test_a_download_error_leaves_the_report_green_but_warned(gpuc_home: Path) -> None:
    report = health.run_checks(
        smi=fake_smi(), downloader=failing_downloader, min_free_gb=0.0, url="http://x"
    )
    assert report["ok"]
    assert any("name resolution failed" in w for w in report["warnings"])


def test_zero_bytes_is_still_fatal(gpuc_home: Path) -> None:
    report = health.run_checks(
        smi=fake_smi(), downloader=dead_downloader, min_free_gb=0.0, url="http://x"
    )
    assert not report["ok"]


def test_the_default_download_url_is_a_stable_sized_endpoint() -> None:
    assert health.DEFAULT_DOWNLOAD_URL.startswith(
        "https://github.com/astral-sh/uv/releases/latest/"
    )


def test_the_disk_floor_message_says_what_to_do(gpuc_home: Path) -> None:
    check = health.check_disk(min_free_gb=1e9)
    assert not check.ok
    assert "floor 1000000000.0 GB" in check.detail
    assert "--min-free-gb" in check.detail


def test_run_checks_emits_json_with_every_check(gpuc_home: Path) -> None:
    report = health.run_checks(
        smi=fake_smi(), downloader=fast_downloader, min_free_gb=0.0, url="http://x"
    )
    assert report["ok"]
    assert report["gpus"] == FAKE_GPUS
    assert [c["name"] for c in report["checks"]] == [
        "driver",
        "gpu_uuids",
        "disk",
        "uv_cache",
        "hf_cache",
        "data_dir",
        "download",
    ]
    json.dumps(report)


def test_run_checks_is_not_red_on_a_host_with_no_cards(gpuc_home: Path) -> None:
    jobs.write_config(HostConfig(host="cpu-box", gpus=[]))
    report = health.run_checks(
        smi=fake_smi(uuids=[]), downloader=fast_downloader, min_free_gb=0.0, url="http://x"
    )
    assert report["ok"]


def test_health_main_exit_code_follows_the_report(
    gpuc_home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(health, "http_download", fast_downloader)
    monkeypatch.setattr(health.gpus, "run_nvidia_smi", fake_smi())
    code = cli.main(["health", "--min-free-gb", "0", "--download-url", "http://x"])
    report = json.loads(capsys.readouterr().out)
    assert code == 0
    assert report["host"] == "test-host"


def test_the_uuid_check_covers_the_shared_cards_too() -> None:
    """A `--shared-gpus` entry that names nothing is the same typo as an owned
    one, and would otherwise be a card gpuc silently never borrows."""
    ok = health.check_gpu_uuids(["0"], fake_smi(), shared=["1"])
    assert ok.ok and ok.value == 1
    assert "1 shared" in ok.detail

    check = health.check_gpu_uuids(["0"], fake_smi(), shared=["7"])
    assert not check.ok
    assert "config.shared_gpus entries not present" in check.detail


def test_a_card_that_is_both_owned_and_shared_fails_the_check() -> None:
    """The two lists say opposite things about a card, so one in both is never
    what anybody meant -- and bootstrap is where it should be found."""
    check = health.check_gpu_uuids(["0", "1"], fake_smi(), shared=["0"])
    assert not check.ok
    assert FAKE_GPUS[0] in check.detail
    assert "not both" in check.detail


def test_a_host_that_only_borrows_still_checks_its_driver(gpuc_home: Path) -> None:
    """`gpus: []` with shared cards is a real configuration -- everything on
    this box belongs to somebody else -- and it needs a driver like any other."""
    jobs.write_config(HostConfig(host="h", gpus=[], shared_gpus=list(FAKE_GPUS)))
    report = health.run_checks(smi=fake_smi(), downloader=fast_downloader)
    checks = {check["name"]: check for check in report["checks"]}
    assert checks["driver"]["value"] == "580.173.02"
    assert checks["gpu_uuids"]["ok"]
    assert report["shared_gpus"] == FAKE_GPUS
