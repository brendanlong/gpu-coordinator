"""A curated fleet, served through the real dashboard code, for README screenshots.

Nothing here touches a registry or a host: it builds `HostView`s directly and
hands them to the same `status.document()` the API returns, so what the page
renders is the production rendering of a made-up fleet.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from typing import Any
from urllib.parse import urlsplit

from gpuc.control import status as status_mod
from gpuc.control import version as version_mod
from gpuc.control.actions import host_document
from gpuc.control.config import HostCache, HostEntry, Rental
from gpuc.control.gpuinfo import GpuInfo
from gpuc.control.providers.base import Pod
from gpuc.control.status import HostState, HostView, JobView, SharedGpu

NOW = datetime.now(UTC)


def at(**delta: float) -> str:
    return (NOW + timedelta(**delta)).isoformat()


def uuid(n: int) -> str:
    return f"GPU-0000000{n}-0000-0000-0000-00000000000{n}"


def entry(
    name: str,
    kind: str,  # the call sites still say it; the entry derives it from the address
    *,
    ssh: str | None = None,
    gpus: list[str],
    shared: list[str] | None = None,
    s3_prefix: str | None = None,
    model: str,
    vram_mib: int,
    pod_id: str | None = None,
) -> HostEntry:
    config: dict[str, Any] = {
        "host": name,
        "gpus": gpus,
        "shared_gpus": shared or [],
        "retention_days": 14,
        "idle_minutes": 15,
        "pkg_commit": version_mod.local_commit(),
    }
    if s3_prefix:
        config["s3_prefix"] = s3_prefix
    return HostEntry(
        name=name,
        ssh=ssh,
        rental=Rental(pod_id=pod_id) if pod_id else None,
        cache=HostCache(
            read_at=at(minutes=-3),
            driver_version="580.65.06",
            gpu_info={g: GpuInfo(name=model, vram_mib=vram_mib) for g in [*gpus, *(shared or [])]},
            config=config,
        ),
    )


def views() -> list[HostView]:
    desktop = HostView(
        entry=entry(
            "desktop",
            "local",
            gpus=[uuid(1)],
            s3_prefix="s3://my-bucket/gpuc/desktop",
            model="NVIDIA GeForce RTX 4090",
            vram_mib=24564,
        ),
        state=HostState.ANSWERED,
        pkg_commit=version_mod.local_commit(),
        dispatcher_pkg_commit=version_mod.local_commit(),
        heartbeat_age_s=2.0,
        owned=[uuid(1)],
        indices={uuid(1): 0},
        running=[
            JobView(
                job_id="20260918-142201-9f31ac",
                name="sft-qwen3-4b",
                status="running",
                phase="main",
                gpus=[uuid(1)],
                gpus_requested=1,
                priority=50,
                started_at=at(hours=-1, minutes=-52),
                util_recent=[94.0, 97.0, 96.0],
                progress_pct=63.0,
                eta=at(hours=1, minutes=4),
                estimated_runtime_min=180.0,
            )
        ],
        finished=[
            JobView(
                job_id="20260918-104512-3ac8e1",
                name="sft-qwen3-4b-lr1e5",
                status="succeeded",
                started_at=at(hours=-7),
                ended_at=at(hours=-4, minutes=-6),
                outputs=[{"s3": "s3://my-bucket/runs/20260918-104512-3ac8e1"}],
            )
        ],
    )

    lab = HostView(
        entry=entry(
            "lab",
            "ssh",
            ssh="me@lab-gpu-03",
            gpus=[uuid(2), uuid(3)],
            shared=[uuid(4)],
            s3_prefix="s3://my-bucket/gpuc/lab",
            model="NVIDIA A40",
            vram_mib=49140,
        ),
        state=HostState.ANSWERED,
        pkg_commit=version_mod.local_commit(),
        dispatcher_pkg_commit=version_mod.local_commit(),
        heartbeat_age_s=1.0,
        owned=[uuid(2), uuid(3)],
        indices={uuid(2): 0, uuid(3): 1, uuid(4): 2},
        shared=[
            SharedGpu(uuid=uuid(4), index=2, memory_mib=38210.0, utilization_pct=99.0, unused=False)
        ],
        running=[
            JobView(
                job_id="20260918-131055-7b02de",
                name="grid-lr3e4-wd01",
                status="running",
                phase="main",
                gpus=[uuid(2)],
                gpus_requested=1,
                priority=40,
                started_at=at(hours=-3, minutes=-11),
                util_recent=[88.0, 91.0, 90.0],
                progress_pct=71.0,
                eta=at(hours=1, minutes=17),
                estimated_runtime_min=270.0,
            ),
            JobView(
                job_id="20260918-131103-1d44f0",
                name="grid-lr1e4-wd01",
                status="running",
                phase="main",
                gpus=[uuid(3)],
                gpus_requested=1,
                priority=40,
                started_at=at(hours=-3, minutes=-11),
                util_recent=[89.0, 92.0, 93.0],
                progress_pct=68.0,
                eta=at(hours=1, minutes=33),
                estimated_runtime_min=270.0,
            ),
        ],
        queue=[
            JobView(
                job_id="20260918-134402-55c9ab",
                name="grid-lr3e5-wd01",
                status="queued",
                priority=40,
                gpus_requested=1,
                use_shared=False,
                estimated_runtime_min=270.0,
            ),
            JobView(
                job_id="20260918-151217-c07b93",
                name="eval-checkpoints",
                status="queued",
                priority=70,
                gpus_requested=1,
                use_shared=True,
                auto_preempt=True,
                estimated_runtime_min=45.0,
            ),
        ],
        finished=[
            JobView(
                job_id="20260918-092230-4e1b77",
                name="grid-lr1e3-wd01",
                status="failed",
                reason="job",
                exit_code=1,
                started_at=at(hours=-9),
                ended_at=at(hours=-8, minutes=-41),
            ),
        ],
    )

    rented = HostView(
        entry=entry(
            "a100-burst",
            "runpod",
            ssh="root@1.2.3.4",
            pod_id="k7q2m9x4v1",
            gpus=[uuid(5), uuid(6)],
            s3_prefix="s3://my-bucket/gpuc/a100-burst",
            model="NVIDIA A100 80GB PCIe",
            vram_mib=81920,
        ),
        state=HostState.ANSWERED,
        pkg_commit=version_mod.local_commit(),
        dispatcher_pkg_commit=version_mod.local_commit(),
        heartbeat_age_s=3.0,
        owned=[uuid(5), uuid(6)],
        indices={uuid(5): 0, uuid(6): 1},
        pod=Pod(
            id="k7q2m9x4v1",
            name="gpuc-a100-burst",
            status="RUNNING",
            cost_usd_hr=1.64,
            gpu_name="A100 80GB PCIe",
            gpu_count=2,
            cuda_version="12.8",
            gpu_utils=[98, 97],
            created_at=NOW - timedelta(minutes=48),
        ),
        running=[
            JobView(
                job_id="20260918-145812-b6e330",
                name="pretrain-ablation",
                status="running",
                phase="main",
                gpus=[uuid(5), uuid(6)],
                gpus_requested=2,
                priority=30,
                started_at=at(minutes=-41),
                util_recent=[97.0, 98.0, 98.0],
                progress_pct=22.0,
                eta=at(hours=2, minutes=26),
                estimated_runtime_min=180.0,
            )
        ],
    )

    return [desktop, lab, rented]


LOG = """\
[setup] uv sync --frozen: 214 packages in 6.31s
[check] gpu preflight ok: torch 2.11.0 cuda 12.8 devices 1 NVIDIA GeForce RTX 4090
[check] s3://my-bucket/runs/20260918-142201-9f31ac writable
[main] loading Qwen/Qwen3-4B in bf16
[main] train tokens/s 18412 | step 2400/3800 | loss 1.284 | lr 1.83e-05
[main] train tokens/s 18390 | step 2450/3800 | loss 1.271 | lr 1.79e-05
[main] eval  step 2450 | loss 1.302 | ppl 3.68
[main] train tokens/s 18437 | step 2500/3800 | loss 1.266 | lr 1.75e-05
[sync] uploaded checkpoints/step2500 (2.1 GiB) to s3://my-bucket/runs/20260918-142201-9f31ac
[main] train tokens/s 18405 | step 2550/3800 | loss 1.259 | lr 1.71e-05
"""


def static(name: str) -> bytes:
    return (resources.files("gpuc.control.web") / "static" / name).read_bytes()


def status_json() -> dict[str, Any]:
    document = status_mod.document(views())
    document["gathered_at"] = NOW.isoformat()
    return document


ROUTES_JSON = {
    "/api/status": status_json,
    "/api/hosts": lambda: {
        "hosts": [host_document(v.entry) for v in views()],
        "errors": [],
    },
    "/api/config": lambda: {
        "config_file": "/home/me/.config/gpu-coordinator/config.toml",
        "config_file_exists": True,
        "state_dir": "/home/me/.local/share/gpu-coordinator",
        "settings": {"s3_bucket": "my-bucket", "s3_prefix": "gpuc"},
        "notes": [],
    },
    "/api/version": lambda: {
        "version": version_mod.__version__,
        "commit": version_mod.local_commit(),
        "source": "installed",
        "dirty": False,
        "hosts": [],
        "errors": [],
    },
}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:
        pass

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        if path == "/":
            return self.send(static("index.html"), "text/html; charset=utf-8")
        if path.startswith("/static/"):
            name = path.rsplit("/", 1)[-1]
            kind = "text/javascript" if name.endswith(".js") else "text/css"
            return self.send(static(name), f"{kind}; charset=utf-8")
        if path in ROUTES_JSON:
            return self.send(json.dumps(ROUTES_JSON[path]()).encode(), "application/json")
        if path.startswith("/api/jobs/") and path.endswith("/logs"):
            job_id = path.split("/")[3]
            body = {
                "job_id": job_id,
                "host": "desktop",
                "source": "host",
                "location": f"/home/me/.gpuc/jobs/{job_id}/log.txt",
                "lines": LOG.splitlines(),
                "notes": [],
            }
            return self.send(json.dumps(body).encode(), "application/json")
        self.send(b"not found\n", "text/plain", code=404)

    def send(self, body: bytes, content_type: str, code: int = 200) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    import sys

    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8747
    print(f"demo dashboard on http://127.0.0.1:{port}/", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
