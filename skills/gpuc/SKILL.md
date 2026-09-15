---
name: gpuc
description: Run GPU jobs with gpu-coordinator (gpuc) on the local desktop, the SPAR shared box, or a RunPod pod. Use when asked to train, evaluate, or run anything on a GPU, to check on or cancel a GPU job, or to read its logs.
---

# Running GPU jobs with gpuc

`gpuc` is a per-host job queue plus RunPod provisioning. One submit path for
three kinds of host. The host is authoritative for its own queue and state; S3
is a mirror. State is shared by every session of this user, so hosts registered
once stay registered.

## Before you trust any of this

```bash
gpuc version     # this build's commit, and each host's package commit
gpuc status      # registered hosts, their queues, and what is running
```

`gpuc skill` prints this file, and `gpuc skill --install [DIR]` writes a copy
to `DIR/.claude/skills/gpuc/SKILL.md`.

If `gpuc` is not on PATH, run it as `uv run gpuc` from a checkout. A host marked
`OLDER: re-bootstrap` needs nothing from you: `gpuc submit` and `gpuc requeue`
re-sync the package and restart that host's dispatcher before enqueueing (pass
`--no-bootstrap` to skip it). Registering, bootstrapping and configuring hosts
is `docs/setup.md` in the repo, not this guide.

## Pick a host

| host | when | notes |
|---|---|---|
| `local` | anything that fits in 8 GB VRAM | free, shared with the desktop; one GPU |
| `spar` | up to 2× A40 48 GB, no cost | shared box; only the two cards assigned to us (by nvidia-smi index) are ever used; home is wiped on restart |
| `--runpod` | needs more, or both are busy | costs money; provisions the cheapest matching pod, idles down after 15 min |

Prefer `local`, then `spar`, then RunPod. Check `gpuc status` first: a busy host
queues your job behind the running one, which is usually fine.

## Write a job spec

Run `gpuc submit` from inside the project checkout: the working directory is
rsynced to the host (git-tracked *and* untracked files, `.gitignore` obeyed,
plus a patch of uncommitted changes). Large data must come from S3 or HF inside
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
sync_interval_s: 180               # upload cadence while running, and at the end; minimum 10
priority: 50                       # 0 first, 99 last
max_runtime_min: 720               # optional wall-clock cap
low_util:                          # kills a job whose GPU sits idle; defaults are conservative
  enabled: true                    # {window_min: 25, floor_pct: 5, grace_min: 10}
cleanup: on_success                # workdir deleted after a successful run
```

Rules that avoid the classic failures:

- Always list the `secrets` your outputs need. They come from your own shell
  environment and are delivered to the host as a 0600 file. A job with S3 or HF
  outputs and no credentials fails at preflight, in seconds.
- `{job_id}` in output destinations makes every run's namespace unique. Never
  reuse a fixed prefix.
- Point `outputs:` at a directory the job creates. Files that were already there
  in the checkout are never uploaded as your results, and a path holding only
  those counts as `no-outputs`.
- Write results incrementally (per checkpoint, per sweep point). The periodic
  sync bounds what a killed host can lose to one interval.
- Write files atomically (temp name, then rename), so a sync never uploads a
  half-written checkpoint.
- Pass `--device cuda` explicitly and keep `REQUIRE_CUDA=1`. The runner runs a
  real GPU op inside the job's venv before `main`; a CPU-only torch fails the
  job as `gpu-preflight` rather than crawling for hours.
- Torch builds: cu128 works on every host we use (local driver 580, SPAR driver
  535, RunPod filtered to CUDA ≥ 12.8). cu130 does not work on SPAR.

## Submit, watch, finish

```bash
gpuc submit job.yaml --host local
gpuc submit job.yaml --host spar
gpuc submit job.yaml --runpod --gpu A40 --max-price 0.60        # or --gpu A40,RTX4090 --cloud any

gpuc status                      # every host: queue, running job + phase, recent results
gpuc status --json               # the same, machine-readable (see "Exit codes" below)
gpuc status --suspects           # running jobs that are billing but idle, judged by each job's
                                 # own low_util window/floor/grace, plus pods past a TTL they have;
                                 # it never kills anything
gpuc status --all                # adds jobs only the index knows (a host that lost its state)
gpuc logs <jobid> [-f]           # tails the host; falls back to the S3 mirror only if that job
                                 # has an s3_prefix (its own or the host's) and s3_bucket is set
gpuc ssh <host|jobid>            # a shell there (a job id lands in its workdir)
gpuc ssh <host|jobid> -- ls -la  # one command, run by a login bash there; gpuc exits with that
                                 # command's own exit code
gpuc ssh <host|jobid> --print    # just print the ssh line, to copy
gpuc cancel <jobid>              # SIGTERM then SIGKILL of the job's process tree; final sync still runs
gpuc reorder <jobid> --priority 10          # queued jobs only
gpuc requeue <jobid> --host spar # re-run from the mirrored spec, attempt+1; needs s3_bucket set,
                                 # and re-syncs the workdir from your current directory
gpuc pods                        # RunPod: every pod we own, cost, age, util, wanted?
```

A job's status is its exit code. `failed: <reason>` reasons you will see:
`gpu-preflight` (no working CUDA in the venv), `sync-preflight` (aws/hf or
credentials missing), `low-util` (idle GPU), `low-util-pause` (the host paused
after two low-util failures and stopped this job so it could drain), `timeout`
(`max_runtime_min`), `ttl` (the host's opt-in lifetime cap ran out), `sync`
(final upload failed; results exist only on the host), `no-outputs` (the output
path was never written), `terminated`, `runner-died`.

`gpuc status` also flags jobs, not just failures: `outputs not uploaded` means
the results are still only on that host, and **`OUTPUTS LOST`** means an
ephemeral host retried the upload three times while draining and gave up — those
results are gone, and only re-running the job brings them back.

Never fire-and-forget. After submitting, confirm the job reaches phase `main`
and that its first log lines look right, then check back on a timer. Do not kill
a job on a wall-clock guess; `--suspects` shows the signals to judge from.

## Exit codes and `--json` (read this before scripting anything)

| code | meaning |
| --- | --- |
| 0 | ok — **including** a host that is unreachable or whose dispatcher is down; that is reported per host, not as a failure |
| 1 | the command failed (transport, provider, refused submit) |
| 2 | usage: a bad or missing flag |
| 3 | local state (`hosts.json`, `config.toml`) is unreadable, so the answer is **unknown** |
| 4 | the job or host named does not exist |

```bash
gpuc status --json | jq -r '.hosts[] | "\(.name) reachable=\(.reachable) running=\(.running | length)"'
gpuc status --json | jq '[.hosts[].running[] | {job_id, name, phase, elapsed_s, util}]'
```

The document is `{schema_version, hosts: [...], errors: [...]}`. Each host has
`name, kind, reachable, pkg_commit, dispatcher{alive, heartbeat_age_s},
provider_util, gpus, queued, running, finished, errors`; each job in those three
lists has `job_id, name, status, reason, phase, elapsed_s, util, gpus, iso,
ended_at, outputs_pending`.

Rules, and they are not optional:

- **Key on the JSON `running` list**, never on scraped text and never on the
  exit code alone.
- **Exit 3 means "unknown", never "nothing is running".** Stop and say so;
  something local is broken, and jobs may well be running. The same goes for a
  non-empty top-level `errors`, and for `"reachable": false` on the host you
  care about — we could not ask it.
- A job's `util` is the host's own nvidia-smi sampler (shown as
  `util 98% (host)`); a pod's `provider_util` is RunPod's reading for the whole
  pod (`provider util 71%`). They differ legitimately; do not compare them.
- Ignore keys you do not recognise; more will be added. (`gpuc logs` has no
  `--json`; it is a byte stream.)

## RunPod specifics

- Provisioning to job start is about a minute. If it fails before the host
  proves healthy it re-places automatically onto the next offer, cheapest first.
- The pod terminates itself 15 minutes after its queue empties (`--idle-min`),
  draining its uploads first.
- An existing gpuc pod is reused instead of a new one when its recorded offer
  still matches the request (GPU name, VRAM, price, tier, CUDA floor), it owns
  enough cards, the provider says it is RUNNING, its dispatcher heartbeat is
  under 30 s old, and it is neither draining nor paused. `--no-reuse` forces a
  new one.
- There is **no overall pod lifetime by default**; per-job `max_runtime_min` is
  the cap. `--ttl-hours` is opt-in and *does* kill a running job when it expires.
- A pod that stops answering (dead dispatcher, no ssh) with nothing running is
  terminated after 30 minutes by `gpuc reconcile`.
- `gpuc reconcile --once` cleans up registry entries for gone pods and reaps
  strays. Run it if `status` says `POD GONE`.
- Only act on pods named `gpuc-*`. Others belong to other people.

## Housekeeping

- `gpuc clean --host <host> --all-finished` removes finished jobs' workdirs
  (venvs). Records and logs stay until `--purge`, which only removes jobs whose
  log, state and outputs are confirmed mirrored.
- After a SPAR restart: re-copy the SSH key if needed, then `gpuc host bootstrap
  spar`, then `gpuc status --host spar --all` and `gpuc requeue` whatever was in
  flight.

Full reference in the repo: `README.md`, `docs/setup.md` (install, hosts,
credentials), `docs/usage.md` (every command and failure mode),
`docs/ARCHITECTURE.md` (the contract).
