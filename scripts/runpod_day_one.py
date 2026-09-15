"""Day-one RunPod test: create one A40 pod, ssh in, check pod-scoped self-terminate, tear down.

Run once with `uv run python scripts/runpod_day_one.py`. It creates a billable pod
(a few cents) and always terminates it, including on failure.
"""

from __future__ import annotations

import json
import secrets
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from gpuc.control.providers.base import Constraints, Offer, Pod, ProviderError
from gpuc.control.providers.runpod import RunPodProvider

IMAGE = "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"
READY_CEILING_S = 900.0
SSH_CEILING_S = 300.0
SELF_TERMINATE_WATCH_S = 90.0
CUDA_MIN = "12.8"

# RunPod puts RUNPOD_POD_ID / RUNPOD_API_KEY in /etc/rp_environment, not in the
# environment a non-interactive ssh session inherits.
RP_ENV = "[ -f /etc/rp_environment ] && . /etc/rp_environment; "

READ_ACCOUNT_PODS = (
    'curl -s -o /dev/null -w "%{http_code}" '
    '-H "Authorization: Bearer $RUNPOD_API_KEY" https://api.runpod.io/v2/pods'
)

SELF_TERMINATE = (
    'curl -s -o /dev/null -w "%{http_code}" -X POST '
    '-H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json" '
    '-d \'{"action":"terminate"}\' '
    "https://api.runpod.io/v2/pods/$RUNPOD_POD_ID/action"
)

start = time.monotonic()


def log(message: str) -> None:
    print(f"[{time.monotonic() - start:7.1f}s] {message}", flush=True)


def local_public_key() -> Path:
    for name in ("id_ed25519.pub", "id_ecdsa.pub", "id_rsa.pub"):
        path = Path.home() / ".ssh" / name
        if path.exists():
            return path
    raise SystemExit("no public key found in ~/.ssh")


def ssh_command(pod: Pod, identity: Path, known_hosts: Path, remote: str) -> list[str]:
    assert pod.ssh_direct is not None
    return [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        f"UserKnownHostsFile={known_hosts}",
        "-o",
        "ConnectTimeout=15",
        "-i",
        str(identity),
        "-p",
        str(pod.ssh_direct.port),
        f"{pod.ssh_direct.username}@{pod.ssh_direct.host}",
        remote,
    ]


def wait_for_ssh_direct(provider: RunPodProvider, pod_id: str) -> Pod:
    deadline = time.monotonic() + READY_CEILING_S
    last = ""
    while time.monotonic() < deadline:
        pod = provider.get(pod_id)
        if pod is None:
            raise SystemExit(f"pod {pod_id} vanished before it was ready")
        state = f"{pod.status} ssh.direct={'yes' if pod.ssh_direct else 'no'}"
        if state != last:
            log(f"pod {pod_id}: {state} cuda={pod.cuda_version}")
            last = state
        if pod.status == "RUNNING" and pod.ssh_direct is not None:
            return pod
        if pod.status in ("ERROR", "TERMINATED"):
            raise SystemExit(f"pod {pod_id} reached {pod.status}")
        time.sleep(10.0)
    raise SystemExit(
        f"pod {pod_id} did not reach RUNNING with ssh.direct in {READY_CEILING_S:.0f}s"
    )


def wait_for_ssh(pod: Pod, identity: Path, known_hosts: Path) -> float:
    deadline = time.monotonic() + SSH_CEILING_S
    attempt = 0
    while time.monotonic() < deadline:
        attempt += 1
        probe = subprocess.run(
            ssh_command(pod, identity, known_hosts, "true"),
            capture_output=True,
            text=True,
            timeout=30,
        )
        if probe.returncode == 0:
            return time.monotonic() - start
        if attempt % 5 == 1:
            log(f"ssh not up yet (rc={probe.returncode}): {probe.stderr.strip()[:160]}")
        time.sleep(10.0)
    raise SystemExit(f"ssh never succeeded within {SSH_CEILING_S:.0f}s")


def remote(pod: Pod, identity: Path, known_hosts: Path, command: str) -> str:
    done = subprocess.run(
        ssh_command(pod, identity, known_hosts, RP_ENV + command),
        capture_output=True,
        text=True,
        timeout=120,
    )
    if done.returncode != 0:
        log(f"remote command failed rc={done.returncode}: {done.stderr.strip()[:300]}")
    return done.stdout.strip()


def pick_offer(provider: RunPodProvider) -> Offer:
    offers = provider.offers(
        Constraints(
            gpu_names=["A40"],
            clouds=["SECURE", "COMMUNITY"],
            max_price_usd_hr=0.60,
            cuda_min=CUDA_MIN,
        )
    )
    for offer in offers:
        log(
            f"offer {offer.gpu_id} {offer.cloud} ${offer.price_usd_hr}/h "
            f"availability={offer.availability} cuda={offer.cuda_versions}"
        )
    if not offers:
        raise SystemExit("no A40 offer under $0.60/h with an available CUDA >= 12.8")
    return offers[0]


def teardown(provider: RunPodProvider, pod_id: str | None, name: str) -> None:
    if pod_id is not None:
        pod = provider.get(pod_id)
        if pod is not None and not pod.name.startswith(provider.caps.prefix):
            raise SystemExit(f"refusing to terminate {pod.name!r}: not ours")
        if pod is not None and pod.status != "TERMINATED":
            log(f"terminating {pod_id} from outside")
            provider.terminate(pod_id)
        final = provider.get(pod_id)
        log(f"after terminate: {final.status if final else '404 (gone)'}")
    ours = provider.list_ours()
    log(f"list() with prefix {provider.caps.prefix!r}: {[p.name for p in ours]}")
    if pod_id is not None:
        billing = provider.billing(pod_id)
        log(f"billing for {name} ({pod_id}): {json.dumps(billing.get('metadata', {}))}")


def main() -> int:
    provider = RunPodProvider()
    public_key = local_public_key()
    identity = public_key.with_suffix("")
    added = provider.ensure_ssh_key(public_key.read_text())
    log(f"ssh key {public_key.name}: {'registered now' if added else 'already registered'}")

    offer = pick_offer(provider)
    name = f"gpuc-dayone-{secrets.token_hex(3)}"
    pod_id: str | None = None
    try:
        created_at = time.monotonic()
        pod = provider.create(
            offer,
            name,
            image=IMAGE,
            disk_gb=20,
            env={"HF_HUB_ENABLE_HF_TRANSFER": "0"},
            cuda_min=CUDA_MIN,
        )
        pod_id = pod.id
        log(f"created {name} -> {pod.id} on {offer.cloud} at ${offer.price_usd_hr}/h")

        pod = wait_for_ssh_direct(provider, pod.id)
        assert pod.ssh_direct is not None
        log(
            f"ssh.direct after {time.monotonic() - created_at:.0f}s: "
            f"{pod.ssh_direct.username}@{pod.ssh_direct.host}:{pod.ssh_direct.port} "
            f"host cudaVersion={pod.cuda_version}"
        )

        with tempfile.TemporaryDirectory() as tmp:
            known_hosts = Path(tmp) / "known_hosts"
            wait_for_ssh(pod, identity, known_hosts)
            log(f"first successful ssh {time.monotonic() - created_at:.0f}s after create")
            log("ssh command shape: " + " ".join(ssh_command(pod, identity, known_hosts, "<cmd>")))

            log(
                "nvidia-smi:\n"
                + remote(
                    pod,
                    identity,
                    known_hosts,
                    "nvidia-smi --query-gpu=index,uuid,driver_version --format=csv",
                )
            )
            log("RUNPOD_POD_ID: " + remote(pod, identity, known_hosts, "echo $RUNPOD_POD_ID"))
            log(
                "RUNPOD_API_KEY in pod env: "
                + remote(
                    pod,
                    identity,
                    known_hosts,
                    'if [ -n "$RUNPOD_API_KEY" ]; then echo set; else echo unset; fi',
                )
            )
            log(
                "pod-scoped key reading GET /pods (account scope check): HTTP "
                + remote(pod, identity, known_hosts, READ_ACCOUNT_PODS)
            )
            code = remote(pod, identity, known_hosts, SELF_TERMINATE)
            log(f"pod-scoped self-terminate HTTP code: {code}")

        watch_deadline = time.monotonic() + SELF_TERMINATE_WATCH_S
        while time.monotonic() < watch_deadline:
            observed = provider.get(pod.id)
            if observed is None or observed.status == "TERMINATED":
                log("pod-scoped key DID terminate the pod")
                break
            time.sleep(10.0)
        else:
            log("pod-scoped key did NOT terminate the pod; account key must do it")
        return 0
    except (ProviderError, SystemExit) as error:
        log(f"FAILED: {error}")
        return 1
    finally:
        teardown(provider, pod_id, name)


if __name__ == "__main__":
    sys.exit(main())
