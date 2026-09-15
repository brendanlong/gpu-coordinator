from __future__ import annotations

import json
import time
import urllib.error
from pathlib import Path

import pytest

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
    assert health.DEFAULT_DOWNLOAD_URL == "https://speed.cloudflare.com/__down?bytes=50000000"
    assert health.DEFAULT_MIN_FREE_GB == 5.0


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
        "download",
    ]
    json.dumps(report)


def test_run_checks_is_not_red_on_a_cpu_only_host(gpuc_home: Path) -> None:
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
    code = health.main(["--min-free-gb", "0", "--download-url", "http://x"])
    report = json.loads(capsys.readouterr().out)
    assert code == 0
    assert report["host"] == "test-host"
