# Requirements review: GPU provisioning and queueing tool

> **Snapshot, not the contract.** This is the reasoning as it stood on
> **2026-09-14**, kept because the *why* behind most decisions has not changed.
> The code has moved since. Where this document disagrees with
> [ARCHITECTURE.md](ARCHITECTURE.md), **ARCHITECTURE wins** — most visibly on
> the hard per-host TTL proposed below, which was replaced by the idle
> terminate plus the dead-dispatcher rule (`--ttl-hours` still exists, but it is
> opt-in and off by default). For what the tool does today, read
> [usage.md](usage.md); for how it is built, ARCHITECTURE.md.

Date: 2026-09-14. Input: the requirements list, the wiki's
`gpu-job-runner-failure-catalog` and `skypilot-runpod-gotchas` pages, the
2026-07-24 incident-mining journal (including its "final state" section, which
supersedes the earlier guidance in the same file), the 2026-09-14 scoping
journal, the experiments repo's `skypilot/lib.sh`, `shared/gpu.py`,
`robust_launch.sh` and `train.sh`, the RunPod v2 OpenAPI spec, live RunPod
catalog and pod queries, the Vast CLI's offer-filter list, the
`huggingface_hub` upload docs, and an independent review pass over a first
draft of this document.

## Verdict

Every requirement is buildable. One is best effort rather than guaranteed:
neither RunPod nor Vast offers a pod TTL or idle-stop (verified against the
v2 spec; the only `idleTimeout` is serverless-only), and anything that runs
on the pod cannot fire when the container never starts. Decision
(2026-09-14): auto-down is **best effort**. The local side owns the host
until it has proven healthy, the host owns itself after that, and the
residual exposure is named rather than hidden.

Two other tensions need a decision ("no daemon" vs. automatic
re-provisioning; incremental HF upload vs. files that get rewritten). Two
requirements are risky as phrased ("logs not local", and automatic
kill-and-move, which is now scoped: re-place only before any job has run,
and never on low utilization). About fifteen things are missing, most of
them the interlocks the failure catalog says were the actual cost.

## 1. Requirement-by-requirement

### 1.1 Provision RunPod with filters (GPU, driver, etc.), extensible to Vast

Feasible, and easier than under SkyPilot. Verified against the v2 API today:

- `GET /v2/catalog/gpus?include=AVAILABILITY&product=POD&cloud=<tier>`
  returns, per GPU type, list price per cloud, a stock level, and
  **per-host-CUDA-version availability**. The stock fields are scoped by the
  `cloud` query parameter (default SECURE), so a query per tier is needed
  before comparing community prices against community stock.
- `POST /v2/pods` takes `gpu.minCudaVersion` or `gpu.allowedCudaVersions`,
  `gpu.minRamPerGpu`, `gpu.minVcpuCountPerGpu`, `dataCenterIds`, `cloud`,
  `disk`, `ports`, `env`, `image` or `templateId`, and `startSsh`.
- The pod response carries `cudaVersion` for the host actually drawn, `cost`,
  `runtime.gpus[].util` (verified live: 100% on a running A40), the pod's
  `env`, and `ssh.direct` (host, port).
- `GET /v2/pods/{id}/logs` streams container and system logs over SSE with
  `tail`, `since`, and `Last-Event-ID`. This is the pod-log view the catalog
  says SkyPilot could not see, and it is where the `/dev/dri/cardN`
  crash-loop is visible.

SSH is the part most likely to produce an unreachable pod:

- `startSsh: true` injects `PUBLIC_KEY` from the account's registered keys
  and **does nothing if no keys are registered** (`PUT /v2/account/ssh-keys`).
  Alternatively set `env.PUBLIC_KEY` explicitly.
- The proxy endpoint is interactive-only. Rsync needs `ssh.direct`, which is
  populated only when `22/tcp` is in `ports` **and** the pod has been assigned
  a public port. Exposing the port is necessary, not sufficient. "RUNNING
  for N seconds with no direct endpoint" must count as a placement failure.

Corrections to the failure catalog, all verified:

- create-pod does not take `gpuTypeIds` as a list. That is the MCP tool. The
  v2 endpoint takes **one** `gpu.id` and its description says it "does not
  search for capacity, and it does not fall back". The tool iterates.
- There is no `sshPublicKey` field in v2. Use `startSsh` plus registered keys.
- Live GPU utilization is in REST v2 (`runtime.gpus[].util`). The GraphQL
  dependency is no longer needed.

RunPod filters by **host CUDA version**, not driver version. The two move
together (12.8 is driver 570, 13.0 is driver 580), so this is the same filter
under a different name; see section 2.6 for how tight to set it.

Vast: `vastai search offers` filters on `driver_version`, `cuda_vers`,
`gpu_name`, `gpu_ram`, `dph`, `reliability`, `inet_down`, `inet_up`,
`disk_space`, `verified`, `direct_port_count`. A provider interface of
"constraints in, priced offers out; create(image, onstart, ssh key); status;
logs; destroy" fits both. Recommendation: define that interface now,
implement RunPod only. Vast has been broken under SkyPilot for months and one
working provider is what proves the design.

**No provider can filter for network health.** The dead-network B200 (0 bytes
received over 10 s, 266 open connections) is only catchable by a
post-provision timed download (section 1.5).

### 1.2 Auto-down after the last queued job ends

Best effort, by decision. The ownership split that makes it simple:

**Before the host has proven healthy, the local reconciler owns it.** A new
pod must reach RUNNING, answer SSH on its direct port, and pass the on-host
preflight (driver op, network download, disk) within a few minutes. The
reconciler polls for that; if it does not happen by the ceiling (about
15 minutes from create, matching the final policy in the 2026-07-24
journal) or a broken-host signature appears in the pod log, the reconciler
terminates and re-places. This is the only layer that covers a pod stuck in
PROVISIONING or crash-looping in STARTING, and it needs nothing on the pod.
The on-host preflight itself is run by the dispatcher on first start; on
failure the dispatcher writes `unhealthy` to S3 and terminates its own pod,
and the reconciler re-places either way.

**After the host has proven healthy, the host owns itself.** The dispatcher
terminates its own pod once the queue has been empty for N minutes and the
last job's status and outputs have been flushed to S3. A hard TTL, also
enforced by the dispatcher, bounds a stuck job. The low-utilization watchdog
(section 1.5) is the third host-side trigger. None of this depends on the
desktop being up, which is the property the requirement is really asking
for: a local reboot must neither kill a healthy job nor leak a pod.

**Residual exposure:** a pod that passed health checks and then lost its
network or had its container die cannot terminate itself, and the local
reaper (list pods with our prefix, compare to desired state, terminate
strays past TTL) only covers that while the desktop is up. Acceptable; an
off-host copy of the reaper (a scheduled GitHub Actions job holding the key)
is cheap to add later if this ever bites.

Two failure paths to handle explicitly:

- **Terminate can fail.** The pod-scoped `RUNPOD_API_KEY` is reported to
  return 403 on terminate. If the on-host terminate fails, the dispatcher
  must clear its `draining` marker and go back to accepting work, and the
  reconciler must alert. Otherwise the host refuses jobs and bills forever
  with an empty queue. Test on day one whether the pod-scoped key works; if
  not, an account key has to reach the pod over SSH after boot (section 2.2).
- **Draining plus re-provision can double-bill.** The local side only
  provisions a replacement after the provider reports the old pod
  `TERMINATED` (or after the reaper has issued the terminate itself), never
  on the strength of a `draining` marker.

If an account-level key has to sit on the pod for self-terminate, its blast
radius is every pod in the account, including any pod another session or
another person is running. The v2 spec exposes no per-pod scoping.
Mitigations: hard-code the pod's own id into the terminate path, prefer the
pod-scoped key if it works, and treat this as the one place where the
never-touch-others rule rests on code rather than on permissions. Also note
`GET /v2/pods` returns the pod's full `env`, so any key placed there at
create time is readable by any holder of an account key and persists in the
pod record (section 2.2).

### 1.3 Per-host user-level queue, local and over SSH, GPU subset by UUID

Feasible with nothing outside `$HOME`. Constraints checked:

- `flock` (util-linux), `setsid`, `nohup` are on every Linux host. `uv`
  installs to `~/.local/bin` and brings its own CPython.
- `CUDA_VISIBLE_DEVICES=<uuid>` works and fails closed (verified in the
  catalog: a stale UUID gives `device_count()==0`). Config per host is a list
  of owned UUIDs; a job requests 0..N of them; the runner asserts the
  `nvidia-smi --query-gpu=index,uuid` mapping before launch.
- Zero-GPU jobs are jobs assigned an empty set. No special case.

Dependency boundary, which the first draft blurred: the **queue core**
(enqueue, lock, dispatch, cancel, reorder, exit-code discipline) is stdlib
only, so a broken environment cannot take out the queue. The **runner
helpers** (preflight matmul, S3 and HF sync, health checks) live in a
separate uv-managed venv owned by the tool. A sync-venv failure fails the job,
not the host.

Tension: the failure catalog argues for "no daemon" (a `flock` loop so nothing
depends on an SSH session surviving), but reorder, cancel and status all want
something to talk to. The reconciliation is a **per-host dispatcher** that any
`enqueue` starts idempotently, detached with `setsid nohup`, that exits when
the queue is empty. Rules that keep it from becoming the wedged controller of
failure class #5:

- `enqueue` writes the job file first and never depends on acquiring the
  dispatcher lock. A wedged dispatcher can never lose an enqueue.
- The dispatcher holds `flock` on an open fd (released by the kernel on
  death, so a crash frees it) **and** touches a heartbeat file every few
  seconds. A new dispatcher that finds the lock held but the heartbeat stale
  kills the holder's process group and takes over.
- State is a directory of job files. "Reorder" is renaming a priority prefix.
  "Cancel" is a marker plus a kill of the job's process group.

"No long-lived process to lose" was a misleading phrase in the first draft.
The dispatcher lives as long as the queue does. What is true is that no
process's death loses state, and whoever enqueues next restarts it.

Cancel semantics vary by host. On the local desktop `systemd-run
--user --scope` works (verified), so cancel can kill a whole cgroup and catch
double-forked children. RunPod containers have no systemd, and an arbitrary
ssh box is unknown. **Process-group kill is the baseline**; cgroup kill is an
upgrade where available.

Two things to probe on an ssh host before designing around it:

- Whether detached processes survive SSH logout (`KillUserProcesses` in
  logind; the default is `no`, but admins flip it).
- Whether a user systemd manager exists at all.

The host-probe script offered in the scoping journal (driver, owned UUIDs,
disk quota, network throughput to S3 and HF, existing scheduler, the two items
above, and whether another user already has a process on an "owned" card)
should be the first thing written.

### 1.4 Status, logging, enqueue / cancel / reorder

Feasible. The exit-code discipline the catalog documents is the important
part: the job's status is the job's exit code, captured before any cleanup,
and a failed final upload fails an otherwise-green job. Copy the `lib.sh`
semantics; they were measured, not guessed.

Logs: S3 has no append, so streaming means either re-uploading the file every
N seconds or rotating into numbered chunks. Both cost a PUT per tick, and a
`sync` costs a LIST per run. The August 2026 bill spike (2.48M Tier-1
requests, 624 GB egress, from one replication loop) is the warning. Budget
it: sync every 2 to 5 minutes, chunk logs so each tick uploads only the new
chunk, and have `status` tail the host over SSH rather than read S3. Keep
the log on the host; see section 2.5.

### 1.5 Health checks: driver, GPU availability, speed test; kill and re-place on failure

Feasible, with the automation boundary drawn carefully. Brendan's final
policy from the 2026-07-24 journal (its last section, which supersedes the
"judgment only" framing above it):

- **Pre-RUNNING is cheap but not free.** Be patient about capacity, do not
  escalate to a pricier GPU over a few minutes' wait, but **provisioning
  should not take more than about 15 minutes**. Past that, something is
  wrong. The gotchas page independently says INIT beyond ~10 minutes is
  failed.
- **RUNNING and billing:** watch progress, do not auto-kill on a clock.

So the automation is:

- **Provision phase (local reconciler):** destroy and re-place on a hard
  signature in the pod logs (`card[0-9]`, `device nodes`, `OCI runtime`,
  `runc create`, `failed to create shim`), on a provision log with no new
  lines for 15 minutes, or on the ceiling without RUNNING plus a reachable
  direct SSH port.
- **Preflight phase (host dispatcher, first start):** destroy and re-place on
  a failed check. Checks: a real GPU op in the torch the job will use
  (`shared/gpu.py`, ported as-is); `device_count()` equals the assignment;
  timed ~100 MB downloads from S3 and from HF with a generous floor (the
  dead-network signature is 0 bytes in 10 s, not "slow"); a timed disk
  write; a small matmul against a per-GPU-type floor; free space on the
  working volume; `HF_HOME` on the large volume.
- **Job phase:** the same preflight runs again **inside the job's venv**
  (a sub-venv CPU torch passes a host-level check; the catalog is explicit).
  A failure here is a code or environment problem, so it **fails the job and
  leaves the host alone**.
- **Low-utilization watchdog (host side, per job):** the dispatcher samples
  `nvidia-smi` utilization on the job's assigned UUIDs every ~30 s. Once the
  job's own setup has finished (the runner knows when it hands off to the
  user command) plus a grace period for model loading, if the rolling mean
  over a long window stays below a floor, the watchdog kills the job with
  status `failed: low-util`, and normal idle terminate follows if the queue
  is empty. Defaults should be conservative and per-job overridable: a
  20 to 30 minute window and a floor around 5%. Some jobs are legitimately
  bursty, so the spec needs an opt-out. To stop a broken commit draining a
  whole queue one job at a time, two consecutive low-util failures on a
  host pause its queue and terminate it.

**Low-util does not re-provision.** After a passed health check the likely
cause is the code (a CPU-torch sub-venv, a data-loader bottleneck, a hung
download inside the job), and re-placing would replay the same job on a
fresh host and burn the same money again. The right outcome is: job marked
failed with the reason, host terminated, nothing requeued, loud entry in
`status`. Re-placement is reserved for failures that happen before any job
has run, where the host is the only variable. On ssh and local hosts the
watchdog behaves identically except that "terminate host" is a no-op.

Every timer must exceed the slowest legitimate phase: a cold `uv sync` with
a torch download is 5 to 10 minutes on a healthy host.

"Move the jobs" needs two things the first draft missed: a resume policy for
the running job (section 2.3) and **moving the queued backlog**, which
otherwise dies with the host. That is why the queue of an ephemeral host must
be authoritative in S3, not on the host (section 1.9).

### 1.6 Python via uv only; use a typical docker image where possible

Feasible, and the two halves do not conflict as long as the image carries
**only system-level things**. On RunPod use
`runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` (or the cu130 sibling): it
gives sshd honoring `PUBLIC_KEY`, rsync, and the CUDA toolkit. Its bundled
torch is irrelevant since `uv sync` installs the project's own. On ssh and
local hosts there is no container and the same runner runs bare. Anything
project-specific baked into the image would make the three targets diverge,
which is the layer-split failure the catalog warns about.

The image carries its own `cuda>=X.Y` requirement (the "unsatisfied
condition" init failure in the catalog is this). Request `minCudaVersion`
at least as high as the image's, and set `HF_HUB_ENABLE_HF_TRANSFER=0` unless
`hf_transfer` is installed.

### 1.7 List of options, pick cheapest

Feasible: query the catalog per cloud tier with availability, filter, sort by
price, try in order, advance on a capacity error or on a pod that never gets
a direct SSH port. The policy question the first draft dodged: **community is
30 to 55% cheaper than secure** (A40 $0.35 vs $0.49, RTX 3090 $0.22 vs
$0.50, 4090 $0.34 vs $0.74 today), and the catalog's own observation is that
cheap pools hit more host-side failures. Expose the tier as a job-level
choice with a default, and record per-tier failure counts in the job
metadata so the default can be revisited with data. Also expose a max price
cap so "wait for the cheap one at $0" is the default and "take the next tier"
is opt-in. Cross-provider price comparison waits for Vast.

### 1.8 Saved outputs uploaded to HF and/or S3 as produced

Feasible. Details that matter:

- S3: periodic sync of an output directory. The `aws` CLI v2 is a
  self-contained bundle that installs into `$HOME` without sudo, so it is
  usable on a host you have no root on; `boto3` has no `sync`, so choosing
  it means writing the size-and-mtime diff. Either is fine; pick one and say so.
- HF: `huggingface_hub.CommitScheduler` uploads a folder every N minutes.
  It is **append-only by contract**: overwriting or deleting files "can
  corrupt the repository", each push is a git commit, and the docs
  recommend at least 5 minutes between pushes. `upload_folder` is resumable
  and multi-commit and is the right call for checkpoints. Only one uploader
  process per folder, which matters with several sessions (section 2.9).
- HF repos created by the library are **public unless the org default is
  private**. Logs and checkpoints must go to explicitly private repos.
  Repo limits (100k files, 10k per folder, 50 GB per file hard) will be
  reached by per-job namespacing; plan one repo per experiment, not one
  global one.
- Partial files: a sync can pick up a half-written checkpoint. Require
  atomic writes in jobs (temp name, then rename) and have the syncer skip
  files modified in the last few seconds.
- The collision problem the catalog documents (mid-run sync tripping the
  overwrite guard at the end) is removed by section 2.1.

### 1.9 Robust to reboots, including of the local service

Feasible if every local process is a **restartable reconciler, not a
stateful controller**, and if authority is placed correctly:

- **For ephemeral hosts, S3 is authoritative** for the job queue and job
  status, because the host will be destroyed. The host holds a working copy
  and flushes status changes before acting on them (in particular before
  self-terminating). For ssh and local hosts, the host is authoritative and
  S3 is the mirror.
- The local CLI is stateless: it rebuilds its view from S3 plus SSH.
- The reconciler (provision, reaper, heartbeat) runs as a `systemd --user`
  unit under your own account, with linger enabled. It loops: read
  desired state, list provider pods, act on the difference. Every action is
  idempotent and safe to interrupt. It is single-instance under an fd lock
  with a heartbeat and staleness takeover, same rule as the dispatcher, so
  a hung holder cannot block provisioning.
- Auto-down never depends on the local side (section 1.2), and the worst
  case has an off-host reaper.

## 2. What is missing

### 2.1 Job identity and output namespace

Assign a unique job id at enqueue and put everything under
`<prefix>/<job-id>/`: logs, outputs, metadata, attempts. A retry is a new
attempt under the same id. Collisions become impossible by construction,
which deletes the three-layer overwrite guard, the `in-progress/` subprefix
workaround, and `ALLOW_OVERWRITE=1`. Human-readable names go in metadata and
an index file, not in the path.

### 2.2 Credentials

- Never on argv (`/proc/<pid>/cmdline` is world-readable).
- **Not in pod `env` at create time either**: it is returned by
  `GET /v2/pods` and persists in the pod record. Push credentials over SSH
  after boot, as a mode-0600 file written from stdin. `SendEnv` does not
  work on a box whose sshd you do not administer, because `AcceptEnv` needs
  an sshd config change.
- Write-test S3 and HF at enqueue time locally and again in preflight on the
  host. A missing credential fails before the expensive work, never at the
  final upload.
- wandb is optional and unavailability degrades, never kills.
- The on-pod terminate key's blast radius is stated in section 1.2.

### 2.3 Resume policy for moved or retried jobs

The job spec says what a restart means: from scratch (default), or resume
from the last checkpoint at `<job-id>/attempts/<n-1>/`. Without it, moving a
job that ran for two hours silently discards two hours. Record which attempt
produced which output.

### 2.4 Cost interlocks

- Hard TTL per host; max runtime per job; max $/hour per provision request.
- The local reaper is the safety net for hosts that never proved healthy
  and for strays past TTL while the desktop is up; it is not the primary
  teardown path (section 1.2).
- **Account-wide caps**, because several sessions provision independently:
  max concurrent pods with our prefix and max total $/hour, checked against
  `GET /v2/pods` (with `includeClusterPods=true`) before every create.
- The reaper **fails closed**: if desired state is unreadable or empty it
  does nothing and alerts. It must never interpret "no state" as "terminate
  everything with our prefix", which would be failure class #4 again.
- The reaper never touches pods without our prefix.
- Verify teardown against the provider, not our own state, and remember
  billing lags about an hour.

### 2.5 Logs and state on the host as well as S3

"Logs go to S3 and not the local machine" should read "S3 is the durable
copy". The host keeps the log file, `status` tails the host directly, and
nothing local-only survives a reboot by design because nothing local-only
exists.

### 2.6 Torch build vs host driver, closed at the source

The tool should derive a CUDA floor from the project's lockfile and pass it
as `minCudaVersion`, so the driver-lottery host is never drawn. The floor
should be the **max of the image's `cuda>=` requirement and the wheel's CUDA
major**, not the wheel's exact minor: within a major, driver minor-version
compatibility lets a cu128 wheel run on a 12.4 host for ordinary ops, and an
exact-minor floor throws away stock (A100 80GB has 12.4 hosts today). The
major boundary (cu13 needs 13.0, driver 580) is hard. Verify the minor-
compat claim with the preflight on first use; the preflight is the backstop
either way. Today's fleet is mostly 12.8 / 13.0 / 13.2, so the practical
impact is small until it isn't.

### 2.7 Code sync and reproducibility

Rsync only git-tracked files (`git ls-files`), record the commit hash and the
diff of uncommitted changes in job metadata, never rsync data. Record the
host's GPU, CUDA version, provider id, tier, and attempt outcome per attempt.

### 2.8 Environment build cost

A cold `uv sync` per job is 5 to 10 minutes on a good host and is the phase
most often mistaken for a hang. Keep `UV_CACHE_DIR` and the project venv on
the host across jobs; re-sync only when the lockfile hash changes. Use
`uv sync --frozen` and `uv run --no-sync` so a verified env cannot be
re-resolved at run time (the `ncclCommResume` incident).

### 2.9 Multiple local sessions

Several agent sessions run as the same user and will enqueue concurrently.
Local state goes in `~/.local/share/gpu-coordinator/`, not a worktree. The
reconciler is single-instance with staleness takeover (section 1.9), the
HF uploader is single-instance per folder (section 1.8), and account-wide
caps (section 2.4) are what stop N sessions from provisioning N pods.

### 2.10 SSH hygiene for ephemeral hosts

Pin the host key on first contact per pod id, use `ControlMaster`, retry
with backoff during STARTING, time out individual commands, and treat
"no direct port" as a placement failure (section 1.1).

### 2.11 Observability that answers the money question

`status` shows, per host: provider state, `cost`, `runtime.gpus[].util`
from the provider, queue depth, current job and phase, its last log lines,
and minutes since the last output file changed. A `--suspects` filter for
"billing, util ~0, no output progress" must be **phase-aware**: during
`uv sync` and model download that signature is normal. It applies only once
a job's preflight has passed and its main phase has started. Alert, do not
kill, on suspects and on any pod older than its TTL.

### 2.12 Shared-box citizenship

Pinning UUIDs is not enough on a box you share with other people. Cap
dataloader workers and thread counts explicitly (a `os.cpu_count()`-sized
loader is antisocial there), respect the disk quota, and have the probe and
preflight detect another user's process on an "owned" card before assigning
it.

### 2.13 Disk hygiene between jobs

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

Same as the scoping journal, with the money-safety work last and exercised
first under a cap:

1. Host probe script; run it on an ssh host and locally.
2. On-host queue core (stdlib only) and runner venv: dispatcher with fd lock
   and heartbeat, UUID assignment, preflight, periodic sync, exit-code
   discipline. Test on the local GPU.
3. Same package over SSH on a shared box.
4. Provider interface plus RunPod driver, reconciler, the pre-healthy
   ceiling, on-host idle terminate and TTL, local reaper, account caps. First test: register an SSH key, provision the
   cheapest available GPU with `startSsh` and `22/tcp`, confirm
   `ssh.direct`, run a 1-minute job, verify teardown against `GET /v2/pods`,
   with a $1 cap. Second test: whether the pod-scoped key can terminate its
   own pod.
5. Only then: the low-utilization watchdog, HF upload, an off-host reaper if
   ever needed, Vast.
