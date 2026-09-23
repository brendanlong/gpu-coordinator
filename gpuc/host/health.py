"""Host preflight: driver, the GPUs this host owns, free disk, network throughput.

Prints JSON. Every check has a timeout, because the failure shape we care
about (a host whose network or driver is dead) hangs rather than errors.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from gpuc.host import USER_AGENT, cleanup, gpus, jobs, paths
from gpuc.host.gpus import SmiRunner

# The uv release tarball: ~19 MB, served by GitHub without per-IP rate limits,
# and a URL that survives releases. speed.cloudflare.com returned 429 after a
# day of bootstraps from one machine, and a pinned wheel URL rots when the
# version is yanked.
DEFAULT_DOWNLOAD_URL = (
    "https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-unknown-linux-gnu.tar.gz"
)
DEFAULT_DOWNLOAD_BYTES = 50 * 1024 * 1024
DEFAULT_MIN_MBPS = 1.0
DEFAULT_MIN_FREE_GB = 5.0
DEFAULT_DOWNLOAD_TIMEOUT_S = 120.0

Downloader = Callable[[str, int, float], int]
"""(url, max_bytes, timeout) -> bytes actually read. Injectable for tests."""


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    value: float | str | None = None
    warn: bool = False


def check_driver(smi: SmiRunner = gpus.run_nvidia_smi) -> Check:
    try:
        version = gpus.driver_version(smi)
    except gpus.GpuError as exc:
        return Check("driver", False, str(exc))
    return Check("driver", True, f"nvidia driver {version}", version)


def check_gpu_uuids(
    owned: Sequence[str], smi: SmiRunner = gpus.run_nvidia_smi, *, shared: Sequence[str] = ()
) -> Check:
    """Every entry in `config.gpus` and `config.shared_gpus` -- index or UUID --
    names a card that is here, and no card is named twice.

    An index that does not resolve is the failure this exists to catch early: a
    shared box renumbered, or the agreement moved, and the host would otherwise
    just quietly have fewer cards to hand out than anyone thinks. A card named
    twice is the other way round: as an index and its own UUID it is a promise
    of two cards, and in both lists it would be handed out as ours *and* have
    its usage second-guessed as somebody else's. `gpus.resolve` is the rule,
    the same one the dispatcher hands cards out by.
    """
    if not owned and not shared:
        return Check("gpu_uuids", True, "no GPUs owned by this host", 0)
    try:
        table = gpus.list_gpus(smi)
    except gpus.GpuError as exc:
        return Check("gpu_uuids", False, str(exc))
    cards = gpus.resolve(owned, table, shared)
    problems: list[str] = []
    if cards.missing:
        problems.append(f"config.gpus entries not present on this host: {', '.join(cards.missing)}")
    if cards.shared_missing:
        problems.append(
            f"config.shared_gpus entries not present on this host: "
            f"{', '.join(cards.shared_missing)}"
        )
    if cards.duplicates:
        problems.append(
            f"entries naming a card already named: {', '.join(cards.duplicates)}. An index "
            f"and its own UUID are one card, and a card is either ours to hand out or "
            f"somebody else's to borrow, not both"
        )
    if problems:
        return Check(
            "gpu_uuids",
            False,
            f"{'; '.join(problems)}; nvidia-smi reports: {gpus.describe_table(table)}",
        )
    detail = f"{len(cards.owned)} owned GPU(s) present"
    if cards.shared:
        detail += f", {len(cards.shared)} shared"
    return Check("gpu_uuids", True, detail, len(cards.owned))


def check_disk(min_free_gb: float = DEFAULT_MIN_FREE_GB) -> Check:
    root = paths.home()
    root.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(root).free / 1e9
    ok = free_gb >= min_free_gb
    detail = f"{free_gb:.1f} GB free on the {root} volume (floor {min_free_gb:.1f} GB)"
    if not ok:
        detail += (
            "; jobs sync outputs through this volume, so free space here or raise "
            "--min-free-gb if you know the job is small"
        )
    return Check("disk", ok, detail, round(free_gb, 1))


def uv_cache_dir(config: jobs.HostConfig | None = None) -> Path:
    """Where uv will cache wheels for this host's jobs.

    The host config's `env` first, because that is what the dispatcher exports
    to every job; then this process's own environment; then uv's own default
    (`$XDG_CACHE_HOME/uv`, else `~/.cache/uv`). The one place this question
    is answered: the health check reports it, and bootstrap asks the host's
    own code for it (`uv_cache_placement`) before deciding `UV_CACHE_DIR`.
    """
    configured = (config.env.get("UV_CACHE_DIR") if config else None) or os.environ.get(
        "UV_CACHE_DIR"
    )
    if configured:
        return Path(configured).expanduser()
    xdg = os.environ.get("XDG_CACHE_HOME")
    return (Path(xdg) if xdg else Path.home() / ".cache") / "uv"


def _nearest_existing(path: Path) -> Path:
    for candidate in [path, *path.parents]:
        if candidate.exists():
            return candidate
    return Path("/")


def same_filesystem(left: Path, right: Path) -> bool | None:
    """Do these two paths live on one filesystem? None when it cannot be read.

    Compared at the nearest existing ancestor, because the interesting case is
    a cache directory that has not been created yet.
    """
    try:
        return _nearest_existing(left).stat().st_dev == _nearest_existing(right).stat().st_dev
    except OSError:
        return None


def uv_cache_placement(config: jobs.HostConfig | None = None) -> dict[str, Any]:
    """The uv cache's location, size, and whether it shares gpuc home's
    filesystem (None when that could not be read). What `check_uv_cache`
    reports and what bootstrap decides `UV_CACHE_DIR` on, so both agree."""
    cache = uv_cache_dir(config)
    home = paths.home()
    return {
        "dir": str(cache),
        "gpuc_home": str(home),
        "shares_gpuc_home_fs": same_filesystem(cache, home),
        "size_bytes": cleanup.dir_size(cache) if cache.is_dir() else 0,
    }


def check_uv_cache(config: jobs.HostConfig | None = None) -> Check:
    """Report the uv cache's size and whether uv can link out of it into a venv.

    Always a warning, never a failure: uv falls back to copying when the cache
    and the venv are on different filesystems, so this costs a full ~6.5 GB
    torch venv per job in disk and minutes in wall clock, but nothing breaks.
    """
    placement = uv_cache_placement(config)
    cache = Path(placement["dir"])
    home = paths.home()
    shared = placement["shares_gpuc_home_fs"]
    size = placement["size_bytes"]
    where = f"{cache} holds {cleanup.human_bytes(size)}" if cache.is_dir() else f"{cache} is empty"
    if shared is True:
        return Check("uv_cache", True, f"{where}, on the same filesystem as {home}", size)
    if shared is None:
        return Check("uv_cache", True, f"{where}; could not compare filesystems", size, warn=True)
    return Check(
        "uv_cache",
        True,
        f"{where}, on a DIFFERENT filesystem from gpuc home {home}, so uv cannot "
        f"hardlink or reflink into a job's venv and copies every wheel instead. "
        f"Set a cache on gpuc home's volume: "
        f"gpuc host set {config.host if config else '<host>'} "
        f"--cache-dir {home.parent}/.cache/uv && gpuc host bootstrap ...",
        size,
        warn=True,
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
        # A warning, not a failure: one unreachable measurement endpoint is not
        # evidence that this host cannot reach S3 or Hugging Face, and failing
        # bootstrap over it strands a pod we are already paying for.
        return Check(
            "download",
            True,
            f"could not measure throughput, GET {url} failed: {exc}",
            None,
            warn=True,
        )
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
        check_driver(smi)
        if (config.gpus or config.shared_gpus)
        else Check("driver", True, "no GPUs owned", None),
        check_gpu_uuids(config.gpus, smi, shared=config.shared_gpus),
        check_disk(min_free_gb),
        check_uv_cache(config),
        check_download(url, min_mbps=min_mbps, timeout=download_timeout, downloader=downloader),
    ]
    return {
        "host": config.host,
        "gpus": config.gpus,
        "shared_gpus": config.shared_gpus,
        "ok": all(c.ok for c in checks),
        "warnings": [c.detail for c in checks if c.warn],
        "checks": [asdict(c) for c in checks],
    }


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--download-url", default=DEFAULT_DOWNLOAD_URL)
    parser.add_argument("--min-mbps", type=float, default=DEFAULT_MIN_MBPS)
    parser.add_argument("--min-free-gb", type=float, default=DEFAULT_MIN_FREE_GB)
    parser.add_argument("--download-timeout", type=float, default=DEFAULT_DOWNLOAD_TIMEOUT_S)


def main(args: argparse.Namespace) -> int:
    report = run_checks(
        url=args.download_url,
        min_mbps=args.min_mbps,
        min_free_gb=args.min_free_gb,
        download_timeout=args.download_timeout,
    )
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1
