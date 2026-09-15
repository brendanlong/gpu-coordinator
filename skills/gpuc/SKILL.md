---
name: gpuc
description: Run GPU jobs with gpu-coordinator (gpuc) on the local desktop, the SPAR shared box, or a RunPod pod. Use when asked to train, evaluate, or run anything on a GPU, to check on or cancel a GPU job, or to read its logs.
---

# Running GPU jobs with gpuc

`gpuc` is a per-host job queue plus RunPod provisioning. One submit path for
three kinds of host. State is shared by every session of this user, so hosts
registered once stay registered.

## Setup (once per machine)

```bash
uv tool install "git+https://github.com/brendanlong/gpu-coordinator@main"
gpuc status            # lists registered hosts; if empty, see "Hosts" below
```

If `gpuc` is not on PATH, run it as `uv run gpuc` from a checkout.

## Pick a host

| host | when | notes |
|---|---|---|
| `local` | anything that fits in 8 GB VRAM | free, shared with the desktop; one GPU |
| `spar` | up to 2× A40 48 GB, no cost | shared pod; only our two cards are assigned; home is wiped on restart |
| `--runpod` | needs more, or both are busy | costs money; provisions the cheapest matching pod, idles down after 15 min |

Prefer `local`, then `spar`, then RunPod. Check `gpuc status` first: a busy
host queues your job behind the running one, which is usually fine.

## Write a job spec

Run `gpuc submit` from inside the project checkout: the working directory is
rsynced to the host (git-tracked files, their current contents, plus a
patch of uncommitted changes). Large data must come from S3 or HF inside
the job, never from the checkout.

```yaml
name: lego-s4                      # label only
setup: uv sync --frozen            # phase "setup"; venv is cached across jobs on the host
command: uv run --no-sync python -m experiments.lego.train --k-max 6 --device cuda
gpus: 1                            # 0 never waits for a GPU
env:
  REQUIRE_CUDA: "1"
  PYTHONUNBUFFERED: "1"
secrets: [AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, HF_TOKEN, WANDB_API_KEY]
outputs:
  - path: results                  # relative to the workdir
    s3: s3://brendanlong-experiments/<experiment>/{job_id}/results
  - path: checkpoints
    hf: brendanlong/<repo>
    hf_path: "{job_id}"
sync_interval_s: 180               # outputs upload every 3 min while running, and at the end
max_runtime_min: 720               # optional wall-clock cap
low_util:                          # kills a job whose GPU sits idle; defaults are conservative
  enabled: true
cleanup: on_success                # workdir deleted after a successful run
```

Rules that avoid the classic failures:

- Always list the `secrets` your outputs need. They come from your own
  shell environment and are delivered to the host as a 0600 file. A job
  with S3 or HF outputs and no credentials fails at preflight, in seconds.
- `{job_id}` in output destinations makes every run's namespace unique.
  Never reuse a fixed prefix.
- Write results incrementally (per checkpoint, per sweep point). The
  periodic sync bounds what a killed host can lose to one interval.
- Write files atomically (temp name, then rename), so a sync never uploads
  a half-written checkpoint.
- Pass `--device cuda` explicitly and keep `REQUIRE_CUDA=1`. The runner
  runs a real GPU op inside the job's venv before `main`; a CPU-only torch
  fails the job as `gpu-preflight` rather than crawling for hours.
- Torch builds: cu128 works on every host we use (local driver 580, SPAR
  driver 535, RunPod filtered to CUDA ≥ 12.8). cu130 does not work on SPAR.

## Submit, watch, finish

```bash
gpuc submit job.yaml --host local
gpuc submit job.yaml --host spar
gpuc submit job.yaml --runpod --gpu A40 --max-price 0.60        # or --gpu A40,RTX4090 --cloud any

gpuc status                      # every host: queue, running job + phase, recent results
gpuc status --suspects           # billing pods with idle GPUs, jobs whose outputs did not upload
gpuc logs <jobid> [-f]           # tails the host; falls back to the S3 copy
gpuc cancel <jobid>              # SIGTERM then SIGKILL of the job's process tree; final sync still runs
gpuc reorder <jobid> --priority 10
gpuc requeue <jobid> --host spar # re-run from the mirrored spec, new attempt
gpuc pods                        # RunPod: every pod we own, cost, age, util
```

A job's status is its exit code. `failed: <reason>` reasons you will see:
`gpu-preflight` (no working CUDA in the venv), `sync-preflight` (aws/hf or
credentials missing), `low-util`, `timeout`, `sync` (final upload failed;
results exist only on the host), `no-outputs` (the output path was never
written).

Never fire-and-forget. After submitting, confirm the job reaches phase
`main` and that its first log lines look right, then check back on a
timer. Do not kill a job on a wall-clock guess; `--suspects` shows the
signals to judge from.

## RunPod specifics

- Provisioning to job start is about a minute. If it fails before the host
  proves healthy it re-places automatically onto the next offer.
- The pod terminates itself 15 minutes after its queue empties
  (`--idle-min`). Submitting again within that window reuses it.
- There is no overall pod lifetime; per-job `max_runtime_min` is the cap.
- `gpuc reconcile --once` cleans up registry entries for gone pods and
  reaps strays. Run it if `status` says `POD GONE`.
- Only act on pods named `gpuc-*`. Others belong to other people.

## Housekeeping

- `gpuc clean --host <host> --all-finished` removes finished jobs'
  workdirs (venvs). Records and logs stay until `--purge`, which only
  removes jobs whose log, state and outputs are confirmed mirrored.
- After a SPAR pod restart: re-copy the SSH key if needed, then
  `gpuc host bootstrap spar`, then `gpuc status --host spar --all` and
  `gpuc requeue` whatever was in flight.

## Hosts (rarely needed)

```bash
gpuc host list
gpuc host probe <name>                       # driver, GPU UUIDs, disk, network, home-fs persistence
gpuc host add <name> --ssh user@host --gpus GPU-uuid,GPU-uuid
gpuc host bootstrap <name>                   # idempotent: uv, package, health check, dispatcher
gpuc host set <name> --gpus ... --retention-days 14 --s3-prefix s3://bucket/gpuc/<name>
```

Full reference: `README.md`, `docs/ARCHITECTURE.md` (the contract) and
`docs/requirements-review.md` (why it is built this way) in the repo.
