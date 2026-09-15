# Requirements review: GPU provisioning and queueing tool

Date: 2026-09-14. Input: the requirements list, the wiki's
`gpu-job-runner-failure-catalog` and `skypilot-runpod-gotchas` pages, the
2026-09-14 scoping journal, the experiments repo's `skypilot/lib.sh`,
`shared/gpu.py`, `robust_launch.sh` and `train.sh`, the RunPod v2 OpenAPI
spec, a live RunPod catalog query, the Vast CLI's offer-filter list, and the
`huggingface_hub` upload docs.

Verdict up front: **nothing in the list is infeasible, and no two requirements
conflict outright.** Three of them pull against each other and need an
explicit decision (auto-down vs. no long-lived launcher, "no daemon" vs.
automatic re-provisioning, incremental upload vs. files that get rewritten).
Two are risky as phrased (automatic kill-and-move on health failure; "logs go to
S3, not local"). About ten things are missing, most of them the boring
interlocks that the failure catalog says were the actual cost.

## 1. Requirement-by-requirement

### 1.1 Provision RunPod with filters (GPU, driver, etc.), extensible to Vast

Feasible, and easier than under SkyPilot. Verified against the v2 API today:

- `GET /v2/catalog/gpus?include=AVAILABILITY&product=POD` returns, per GPU
  type, list price per cloud, a stock level, and **per-host-CUDA-version
  availability**. Today's A40 row: `$0.49/h, HIGH, 12.8/13.0/13.2 all
  available`. RTX 3090: `$0.50, LOW, only 13.0 in stock`.
- `POST /v2/pods` takes `gpu.minCudaVersion` or `gpu.allowedCudaVersions`,
  `gpu.minRamPerGpu`, `gpu.minVcpuCountPerGpu`, `dataCenterIds`, `cloud`
  (SECURE/COMMUNITY), `disk`, `ports`, `env`, `image` or `templateId`.
- The response carries `cudaVersion` for the host actually drawn, `cost`,
  `runtime.gpus[].util`, and `ssh.direct` (host, port) when `22/tcp` is in
  `ports`. The proxy SSH endpoint is interactive-only; **rsync needs the
  direct port**, so every pod must expose `22/tcp`.

Two corrections to the failure catalog:

- The catalog says create-pod "takes `gpuTypeIds` as a list". That is the MCP
  tool's parameter. The v2 REST endpoint takes **one** `gpu.id` and its own
  description says it "does not search for capacity, and it does not fall back
  to a different GPU". The tool has to iterate candidates itself. That is a
  five-line loop, not a problem, but it means the "pick cheapest" logic is ours.
- RunPod filters by **host CUDA version**, not driver version. They map 1:1
  (12.8 → 570, 13.0 → 580), so this is the same filter under a different name.

Vast: `vastai search offers` filters on `driver_version`, `cuda_vers`,
`gpu_name`, `gpu_ram`, `dph`, `reliability`, `inet_down`, `inet_up`,
`disk_space`, `verified`, `direct_port_count`. So a provider interface of
"constraints in, priced offers out; create(image, onstart, ssh key); status;
destroy" fits both. Recommendation: define that interface now, implement
RunPod only. Vast has been broken under SkyPilot for months and one working
provider is what proves the design.

**Missing from both providers' filters: network health.** The dead-network
B200 (0 bytes received over 10 s, 266 open connections) is not something you
can filter for. That has to be a post-provision health check (section 1.5).

### 1.2 Auto-down after the last queued job ends

Feasible but it is the single highest-risk requirement, because **neither
provider has a native TTL or idle-stop.** The create-pod body has no such
field; SkyPilot's `-i 15 --down` is implemented by a daemon SkyPilot installs
on the host. We have to build the same thing, and it must not depend on the
launcher surviving.

What works:

- The on-host queue dispatcher (which has to exist anyway, section 1.3) is the
  natural dead-man switch: it knows when the queue has been empty for N
  minutes, and it can call the provider's terminate API on its own pod.
- Belt and braces: a hard TTL on the pod set at creation (`sleep TTL; terminate`
  in the container, or the dispatcher refusing to start jobs past a deadline)
  and a local reaper that lists provider pods with our name prefix and kills
  anything not in the desired-state file.

One thing to verify on day one of the RunPod work: RunPod injects a pod-scoped
`RUNPOD_API_KEY`, and there are reports of it returning 403 on terminate. If
so, the tool must inject a **scoped account API key** (RunPod supports scoped
keys) with terminate permission as an env var. Test this with a 1-minute pod
before building anything on it.

Semantic tension to decide: "after its last queued job ends" is judged by the
host, but new jobs are enqueued from the local machine. Between the host
deciding to terminate and a new enqueue arriving there is a race. Resolution:
the dispatcher writes a `draining` marker under its lock before terminating;
an enqueue that sees `draining` fails and the local side treats the host as
gone and re-provisions. The window is seconds and the cost is one extra
provision, which is acceptable.

### 1.3 Per-host user-level queue, local and over SSH, GPU subset by UUID

Feasible with nothing outside `$HOME`. Constraints checked:

- `flock` (util-linux), `setsid`, `nohup` are on every Linux host. `uv`
  installs to `~/.local/bin` and brings its own CPython, so the on-host part
  can be a Python package with **zero third-party dependencies** and still run
  anywhere `uv` can be curl-installed.
- `CUDA_VISIBLE_DEVICES=<uuid>` works and fails closed (verified in the
  catalog: a stale UUID gives `device_count()==0`). Config per host is a list
  of owned UUIDs; a job requests 0..N of them; the runner asserts the
  `nvidia-smi --query-gpu=index,uuid` mapping before launch.
- Zero-GPU jobs are just jobs that are assigned an empty set. No special case.

Tension: the failure catalog argues for "no daemon" (a `flock` loop so nothing
depends on an SSH session surviving), but reorder, cancel and status all want
something to talk to. The reconciliation: a **per-host dispatcher process**
that any `enqueue` starts idempotently under a lock (`flock -n` on a pidfile;
if held, someone is already dispatching), detached with `setsid nohup`, that
exits when the queue is empty. State is a directory of job files, so
"reorder" is renaming a priority prefix and "cancel" is writing a marker plus
killing the job's process group. There is no long-lived process to lose, and
whoever enqueues next restarts it. This gets the robustness of the flock loop
with the features of a daemon.

Two things to probe on the SPAR box before designing around it, because they
vary by distro config and cannot be assumed:

- Whether detached processes survive SSH logout (`KillUserProcesses` in
  logind; the default is `no`, but some admins flip it).
- Whether `systemd-run --user --scope` works (needs a user manager and cgroup
  delegation). It does on brendan-desktop as the `claude` user. If it works,
  cancel can kill the whole cgroup, which catches double-forked children.
  Otherwise fall back to process-group kill and accept that daemonized
  grandchildren escape.

The host-probe script offered in the scoping journal (driver, owned UUIDs,
disk quota, network throughput to S3/HF, existing scheduler, the two items
above) should be the first thing written.

### 1.4 Status, logging, enqueue / cancel / reorder

Feasible. The only subtlety is the exit-code discipline the catalog documents:
the job's status must be the job's exit code, captured before any cleanup, and
a failed final upload must fail an otherwise-green job. Copy the existing
`lib.sh` semantics; they were measured, not guessed.

Logs: S3 has no append, so "stream logs to S3" means re-uploading the log
file every N seconds (fine up to tens of MB) or rotating into numbered chunks
(needed for multi-hour runs with verbose output). Keep the log on the host
too; see section 2.5.

### 1.5 Health checks: driver, GPU availability, speed test; kill and re-place on failure

Feasible, with two qualifications.

The checks themselves:

- Driver: a real GPU op in the torch the job will actually use, not
  `nvidia-smi` (which passes on driver-too-old hosts). `shared/gpu.py` is
  this; port it as-is.
- GPU availability: UUID count matches the assignment; `torch.cuda.device_count()`
  equals the expected number.
- Speed: a timed download of ~100 MB from S3 and from HF (catches the
  dead-network host), a timed disk write, and a small matmul benchmark
  compared against a per-GPU-type floor (catches a throttled or shared card).
- Disk: free space on the working volume, and `HF_HOME` pointed somewhere with
  room (the `/workspace/.cache` fill is in the catalog).

Qualification one: this check belongs in **two** places. At provision time it
gates the host. At job start it runs again inside the job's venv, because the
sub-venv CPU-torch failure passes a host-level check. The catalog is explicit
on this.

Qualification two, and this is the "bad idea as phrased" item: **automatic
kill-and-move must be limited to provision-time failures with a hard
signature.** Brendan's 2026-07-24 guidance in the incident-mining journal is
that detectors must be signal-based, not wall-clock-based, and that mid-run
"broken vs slow vs degraded" is left to judgment. The `robust_launch.sh`
12-minute deadline that killed four healthy clusters is the canonical
counterexample. So: pre-job preflight fails → destroy, re-place, requeue,
automatically. Mid-job stall → flag it loudly in `status`, do not act, except
for the hard TTL. Every timer must exceed the slowest legitimate phase
(cold `uv sync` with a torch download is 5-10 minutes on a healthy host).

"Move the jobs" also needs a resume policy (section 2.3).

### 1.6 Python via uv only; use a typical docker image where possible

Feasible, and the two halves do not conflict as long as the image carries
**only system-level things**. On RunPod use `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`
(or the cu130 sibling) because it gives sshd honoring `PUBLIC_KEY`, rsync,
and the CUDA toolkit. Its bundled torch is irrelevant since `uv sync` installs
the project's own. On SPAR and local there is no container and the same
runner script runs bare. Anything project-specific baked into the image would
make the three targets diverge, which is exactly the layer-split failure the
catalog warns about.

Choose the image's CUDA version to be at most the `minCudaVersion` you request,
and set `HF_HUB_ENABLE_HF_TRANSFER=0` unless `hf_transfer` is installed.

### 1.7 List of options, pick cheapest

Feasible: query the catalog with availability, filter by constraints, sort by
price, try in order, advance on a capacity error. Two policy knobs worth
exposing from the start, both from Brendan's own guidance: a max price cap (so
"wait for the cheap one at $0" is the default and "take the next tier" is
opt-in), and a preference for SECURE over COMMUNITY at equal price. Cross-
provider price comparison can wait for Vast.

### 1.8 Saved outputs uploaded to HF and/or S3 as produced

Feasible. Details that matter:

- S3: periodic `sync` of an output directory, as `lib.sh` does now. Prefer
  `boto3` inside the tool's own venv over depending on an `aws` binary, since
  SPAR has no sudo and the tool already has a venv.
- HF: `huggingface_hub.CommitScheduler` uploads a folder every N minutes in a
  background thread. It is **append-only by contract**: overwriting or
  deleting files "can corrupt the repository", and every push is a git commit,
  so the docs recommend at least 5 minutes between pushes. `upload_folder`
  is resumable and multi-commit and is the right call for checkpoints.
- Partial files: a sync can pick up a half-written checkpoint. Require atomic
  writes in jobs (write to a temp name, rename), and have the syncer skip files
  modified in the last few seconds.
- The collision problem: the catalog documents that mid-run sync into the
  canonical prefix trips the overwrite guard at the end of the run. The fix is
  in section 2.1: make the namespace unique by construction so there is no
  guard.

### 1.9 Robust to reboots, including of the local service

Feasible if you accept one framing: **make every local process a restartable
reconciler, not a stateful controller.** Concretely:

- Job specs and status live on each host (authoritative, because only the host
  knows) and are mirrored to S3 (durable index). The local CLI is stateless:
  it rebuilds its view from S3 plus SSH.
- The one thing that needs to be "awake" on the local side is provisioning and
  the reaper. Run it as a `systemd --user` unit under the `claude` user, which
  already has linger enabled (verified: `Linger=yes`). It loops: read desired
  state, list provider pods, act on the difference. Every action is idempotent
  (pod names embed a host id, so a duplicate create is detectable) and safe to
  interrupt anywhere.
- Auto-down never depends on the local side (section 1.2).

This is the opposite of the current setup, where a wedged controller blocks
every launch. Note that "the main local service" in the requirement should
end up being about 200 lines and hold no state that is not also in S3.

## 2. What is missing

### 2.1 Job identity and output namespace

Assign a unique job id at enqueue (timestamp plus random suffix, or ULID) and
put **everything** under `<prefix>/<job-id>/`: logs, outputs, metadata,
attempts. A retry is a new attempt under the same id. Collisions become
impossible by construction, which deletes the three-layer overwrite guard, the
`in-progress/` subprefix workaround, and `ALLOW_OVERWRITE=1`. Human-readable
names go in metadata and in an index file, not in the path.

### 2.2 Credentials

- Never on argv. Ship secrets to hosts via the SSH session environment or a
  mode-0600 file (`/proc/<pid>/cmdline` is world-readable).
- Write-test S3 and HF before the expensive work starts (`s3_preflight`
  already does S3). A missing credential should fail at enqueue time or
  preflight, never at the final upload.
- wandb is optional and unavailability must degrade, not kill.
- On RunPod, an account-scoped key for self-termination (section 1.2).

### 2.3 Resume policy for moved or retried jobs

"Move the jobs" and "requeue on failure" need the job spec to say what a
restart means: from scratch (default), or resume from the last checkpoint at
`<job-id>/attempts/<n-1>/`. Without this, moving a job that had run for two
hours silently discards two hours. Also record which attempt produced which
output.

### 2.4 Cost interlocks

- Hard TTL per host, set at creation, independent of everything else.
- Max runtime per job.
- Max $/hour per provision request.
- A reaper that lists all pods carrying our name prefix and terminates any not
  in desired state, and that **never touches pods without our prefix**. Two
  A40 pods from another session (`subrep-*`) are running in the account right
  now; the current rule that agents do not touch pods they did not start must
  survive into the tool.
- Verify teardown against the provider (`GET /v2/pods`), not our own state,
  and remember billing lags about an hour.

### 2.5 Logs and state on the host as well as S3

"Logs go to S3 and not the local machine" should read "S3 is the durable copy".
The host must keep the log file (the job writes there, the syncer reads it),
and local `status` should be able to `tail` a host directly for the seconds
between syncs. What should not exist is local-only state that a reboot loses.

### 2.6 Torch build vs host driver, closed at the source

The tool can infer the CUDA build the project's lockfile pins (the torch wheel
tag in `uv.lock`, or `torch.version.cuda` in the synced env) and pass it as
`minCudaVersion`. That turns failure class #1 from "preflight catches it after
provisioning" into "we never draw that host". Today's fleet is 12.8 / 13.0 /
13.2, so cu128 and cu130 wheels both have stock; cu126 is unnecessary.

### 2.7 Code sync and reproducibility

Rsync only git-tracked files (`git ls-files`), record the commit hash and the
diff of uncommitted changes in job metadata, and never rsync data. Record the
host's actual GPU, CUDA version, and provider id per attempt.

### 2.8 Environment build cost

A cold `uv sync` per job is 5-10 minutes on a good host and is the phase most
often mistaken for a hang. Keep `UV_CACHE_DIR` and the project venv on the
host across jobs; the dispatcher only re-syncs when the lockfile hash changes.
Use `uv sync --frozen` and `uv run --no-sync` so a verified env cannot be
re-resolved at run time (the `ncclCommResume` incident).

### 2.9 Multiple local sessions

Several agent sessions run as the same user and will enqueue concurrently.
Local state goes in `~/.local/share/gpu-coordinator/`, not a worktree, and the
reconciler is single-instance under a lock. On-host enqueue is already
serialized by the host's lock.

### 2.10 SSH hygiene for ephemeral hosts

Pin the host key on first contact and store it per pod id, use
`ControlMaster` to avoid reconnect storms, retry with backoff during the
STARTING window, and time out individual commands.

### 2.11 Observability that answers the money question

`status` should show, per host: provider state, `cost`, `runtime.gpus[].util`
from the provider, queue depth, current job and its last log lines, and
minutes since the last output file changed. A `--suspects` filter for
"RUNNING, billing, util ~0, no output progress" is the single most useful
view the catalog asks for. Alert (not kill) on suspects and on any pod older
than its TTL.

### 2.12 Disk hygiene between jobs

Point `HF_HOME` and `UV_CACHE_DIR` at the large volume, report free space in
health checks, and offer a per-job "clean HF model cache after" flag.

## 3. Things to leave out

- Vast implementation (keep the interface), spot recovery, multi-node,
  Docker on non-RunPod hosts, a web UI, cross-provider price optimization.
- Any use of the local Kind cluster or SkyPilot API server. The local GPU is
  world-readable at `/dev/nvidia0` and the local target is just a host with
  one owned UUID.
- A scheduler smarter than "first runnable job in priority order whose GPU
  request fits the free owned UUIDs".

## 4. Build order

Same as the scoping journal, with one addition:

1. Host probe script; run it on the SPAR box and locally.
2. On-host package: queue, dispatcher, runner, UUID assignment, preflight,
   periodic sync, exit-code discipline. Test on the local GPU.
3. Same package over SSH on SPAR.
4. Provider interface plus RunPod driver, reconciler, dead-man switch, reaper,
   TTL. First test: provision the cheapest available GPU, run a 1-minute job,
   verify teardown against `GET /v2/pods`, with a $1 cap.
5. Only then: health-check-driven re-placement, HF upload, Vast.

The money-safety code is written last on purpose, after the runner and queue
are proven on two free targets, and it is exercised first with a one-minute
job under a hard cap.
