"""Host preflight: driver, assigned UUIDs, free disk, network throughput.

Prints JSON. Every check has a timeout, because the failure shape we care
about (a host whose network or driver is dead) hangs rather than errors.
"""

from __future__ import annotations

import json
import shutil
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from gpuc.host import USER_AGENT, gpus, jobs, paths
from gpuc.host.gpus import SmiRunner

DEFAULT_DOWNLOAD_URL = (
    "https://download.pytorch.org/whl/cpu/torch-2.5.1%2Bcpu-cp311-cp311-linux_x86_64.whl"
)
DEFAULT_DOWNLOAD_BYTES = 50 * 1024 * 1024
DEFAULT_MIN_MBPS = 1.0
DEFAULT_MIN_FREE_GB = 20.0
DEFAULT_DOWNLOAD_TIMEOUT_S = 120.0

Downloader = Callable[[str, int, float], int]
"""(url, max_bytes, timeout) -> bytes actually read. Injectable for tests."""


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    value: float | str | None = None


def check_driver(smi: SmiRunner = gpus.run_nvidia_smi) -> Check:
    try:
        version = gpus.driver_version(smi)
    except gpus.GpuError as exc:
        return Check("driver", False, str(exc))
    return Check("driver", True, f"nvidia driver {version}", version)


def check_gpu_uuids(owned: Sequence[str], smi: SmiRunner = gpus.run_nvidia_smi) -> Check:
    if not owned:
        return Check("gpu_uuids", True, "no GPUs owned by this host", 0)
    try:
        gpus.assert_uuids_present(owned, smi)
    except gpus.GpuError as exc:
        return Check("gpu_uuids", False, str(exc))
    return Check("gpu_uuids", True, f"{len(owned)} owned UUID(s) present", len(owned))


def check_disk(min_free_gb: float = DEFAULT_MIN_FREE_GB) -> Check:
    root = paths.home()
    root.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(root).free / 1e9
    ok = free_gb >= min_free_gb
    return Check(
        "disk",
        ok,
        f"{free_gb:.1f} GB free on the {root} volume (floor {min_free_gb:.0f} GB)",
        round(free_gb, 1),
    )


def http_download(url: str, max_bytes: int, timeout: float) -> int:
    request = urllib.request.Request(
        url,
        headers={
            "Range": f"bytes=0-{max_bytes - 1}",
            "User-Agent": USER_AGENT,
        },
    )
    read = 0
    with urllib.request.urlopen(request, timeout=timeout) as response:
        while read < max_bytes:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            read += len(chunk)
    return read


def check_download(
    url: str = DEFAULT_DOWNLOAD_URL,
    *,
    max_bytes: int = DEFAULT_DOWNLOAD_BYTES,
    min_mbps: float = DEFAULT_MIN_MBPS,
    timeout: float = DEFAULT_DOWNLOAD_TIMEOUT_S,
    downloader: Downloader = http_download,
) -> Check:
    start = time.monotonic()
    try:
        read = downloader(url, max_bytes, timeout)
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        return Check("download", False, f"GET {url} failed: {exc}")
    elapsed = max(time.monotonic() - start, 1e-6)
    mbps = (read / 1e6) / elapsed
    if read == 0:
        return Check("download", False, f"GET {url} returned 0 bytes in {elapsed:.1f}s", 0.0)
    ok = mbps >= min_mbps
    return Check(
        "download",
        ok,
        f"{read / 1e6:.0f} MB in {elapsed:.1f}s = {mbps:.1f} MB/s "
        f"(floor {min_mbps:.1f} MB/s) from {url}",
        round(mbps, 2),
    )


def run_checks(
    *,
    smi: SmiRunner | None = None,
    downloader: Downloader | None = None,
    url: str = DEFAULT_DOWNLOAD_URL,
    min_mbps: float = DEFAULT_MIN_MBPS,
    min_free_gb: float = DEFAULT_MIN_FREE_GB,
    download_timeout: float = DEFAULT_DOWNLOAD_TIMEOUT_S,
) -> dict[str, Any]:
    config = jobs.read_config()
    smi = smi or gpus.run_nvidia_smi
    downloader = downloader or http_download
    checks = [
        check_driver(smi) if config.gpus else Check("driver", True, "no GPUs owned", None),
        check_gpu_uuids(config.gpus, smi),
        check_disk(min_free_gb),
        check_download(url, min_mbps=min_mbps, timeout=download_timeout, downloader=downloader),
    ]
    return {
        "host": config.host,
        "gpus": config.gpus,
        "ok": all(c.ok for c in checks),
        "checks": [asdict(c) for c in checks],
    }


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="gpuc.host health")
    parser.add_argument("--download-url", default=DEFAULT_DOWNLOAD_URL)
    parser.add_argument("--min-mbps", type=float, default=DEFAULT_MIN_MBPS)
    parser.add_argument("--min-free-gb", type=float, default=DEFAULT_MIN_FREE_GB)
    parser.add_argument("--download-timeout", type=float, default=DEFAULT_DOWNLOAD_TIMEOUT_S)
    args = parser.parse_args(list(argv) if argv is not None else None)
    report = run_checks(
        url=args.download_url,
        min_mbps=args.min_mbps,
        min_free_gb=args.min_free_gb,
        download_timeout=args.download_timeout,
    )
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1
