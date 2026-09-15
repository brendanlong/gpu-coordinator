# gpu-coordinator architecture (prototype spec)

Companion to `requirements-review.md`, which holds the reasoning. This file
is the contract implementers build against. Where the two disagree, this
file wins for the prototype.

## Goals

One submit path for three kinds of host: the local desktop (1 GPU), a shared
SSH box with no sudo (a subset of its GPUs), and ephemeral RunPod pods. Not
finicky, not buggy, and never leaks a paid pod in the normal path.

Non-goals for the prototype: Vast, multi-node, spot, a web UI, S3 as the
authoritative queue (host is authoritative, S3 is the mirror; `gpuc requeue`
resubmits from the S3 spec if a host dies).

## Package layout

One uv project, Python >= 3.11, package `gpuc`, CLI entry point `gpuc`.

```
gpuc/
  host/        # runs ON hosts. STDLIB ONLY. No imports outside stdlib.
    __main__.py   # `python -m gpuc.host <cmd>`; the same CLI as `gpuc host-*`
    paths.py      # ~/.gpuc layout
    jobs.py       # job id, spec/state read/write, atomic writes
    queue.py      # enqueue, list, reorder, cancel markers
    dispatcher.py # lock+heartbeat, pick next runnable, launch runner, idle/TTL terminate
    runner.py     # one job: env, CUDA_VISIBLE_DEVICES, preflight, watchdog, sync, exit code
    gpus.py       # nvidia-smi parsing, UUID<->index assertion, utilization sampling
    sync.py       # periodic upload loop (shells out to `aws` or `hf`; see Sync)
    health.py     # host preflight: driver, UUIDs, disk, network download timing
    terminate.py  # self-terminate via provider API (urllib), key from ~/.gpuc/secrets
  control/     # runs on the local machine. May use third-party deps.
    cli.py        # `gpuc` top-level commands
    config.py     # ~/.local/share/gpu-coordinator/ layout, hosts registry
    transport.py  # LocalTransport / SshTransport: run, rsync, put_file(0600), tail
    bootstrap.py  # install uv + this package on a host, write config, run host preflight
    providers/
      base.py     # Provider interface: offers(constraints), create, get, logs, terminate, list
      runpod.py   # v2 REST implementation
    reconcile.py  # desired state vs provider; ceiling; reaper; caps
    submit.py     # resolve target -> host -> enqueue; provision if needed
    s3index.py    # mirror of specs/state for `requeue` and `status --all`
tests/
```

`gpuc.host` must be importable and runnable with a bare interpreter: the
bootstrap rsyncs the package to the host and runs it with the uv-managed
Python, with no `uv sync` needed for the queue to work. Upload helpers shell
out to binaries the bootstrap installs into `$HOME` (`aws` CLI v2 bundle,
`hf` from `uv tool install huggingface_hub`), and a missing binary fails the
*job's* sync step with a clear message, never the queue.

## On-host state: `~/.gpuc/`

```
config.json          # {"host": "<name>", "gpus": ["GPU-uuid", ...], "provider": null | {"kind":"runpod","pod_id":..},
                     #  "idle_minutes": 15, "ttl_hours": 24, "s3_prefix": "s3://bucket/gpuc/<host>"}
secrets/<name>       # 0600 files delivered over SSH after boot. Never in argv, never in pod env.
queue/<prio>-<jobid> # empty marker files; lexical order is dispatch order. prio is 2 digits, default 50.
jobs/<jobid>/
  spec.json          # the submitted JobSpec (immutable)
  state.json         # {"status": queued|running|succeeded|failed|cancelled, "attempt": n,
                     #  "reason": str|null, "exit_code": int|null, "gpus": [...], "started_at", "ended_at",
                     #  "phase": setup|preflight|main|sync, "pid": int|null, "pgid": int|null,
                     #  "runner_pid": int|null, "runner_boot_id": str|null,
                     #  "runner_starttime": str|null, "sync_error": str|null}
                     # pgid is the *job's* group, published by the runner when it spawns a phase.
                     # It is absent during the launch window; cancel is the marker alone until then.
  workdir/           # rsynced code (git-tracked files only)
  log.txt            # combined stdout/stderr of setup + command, line-buffered
  outputs/           # default output root; JobSpec.outputs paths are relative to workdir
dispatcher.lock      # fd flock held by the running dispatcher
dispatcher.heartbeat # mtime touched every 5 s by the dispatcher
dispatcher.log
draining             # present while the host is shutting itself down
```

All state writes are atomic (write temp in same dir, `os.replace`).

## JobSpec (JSON; `gpuc submit` accepts YAML or JSON and normalizes)

```
{
  "name": "lego-s4",                    # human label, not an identifier
  "command": "uv run python -m experiments.lego.train --k-max 6",
  "setup": "uv sync --frozen",          # optional; runs before command, phase=setup
  "gpus": 1,                            # 0..N owned GPUs
  "env": {"REQUIRE_CUDA": "1"},
  "secrets": ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "HF_TOKEN", "WANDB_API_KEY"],
                                        # read from the submitter's env, delivered as ~/.gpuc/secrets/<jobid>.env (0600),
                                        # sourced into the job's environment by the runner
  "outputs": [{"path": "results", "s3": "s3://bucket/exp/{job_id}/results"},
              {"path": "checkpoints", "hf": "org/repo", "hf_path": "{job_id}"}],
  "sync_interval_s": 180,
  "priority": 50,
  "max_runtime_min": null,
  "low_util": {"enabled": true, "window_min": 25, "floor_pct": 5, "grace_min": 10},
  "requires": {"cuda_min": "12.8"}      # informs provisioning only
}
```

`{job_id}` is expanded in output destinations. Output namespaces are unique
by construction; there is no overwrite guard anywhere.

Job id: `YYYYMMDD-HHMMSS-<6 hex>`, assigned by `gpuc submit`.

## Dispatcher (`python -m gpuc.host dispatch`)

- Started by every `enqueue` (and by bootstrap) with `setsid nohup ... &`.
  Takes `flock(LOCK_EX|LOCK_NB)` on `dispatcher.lock`. If held **and** the
  heartbeat is younger than 30 s, exit 0 silently. If held and the heartbeat
  is stale, kill the holder's process group (pgid recorded in the lock file
  body), then take over.
- Loop every 2 s: read `queue/` in lexical order; for each entry whose
  `spec.gpus` fits the free owned UUIDs, assign UUIDs, remove the queue
  marker, set state running, spawn the runner in its own process group
  (`start_new_session=True`), record pid/pgid. Multiple jobs may run at
  once if GPUs allow; a `gpus: 0` job never waits.
- Cancel: `queue.cancel(jobid)` writes `jobs/<id>/cancel`; the dispatcher
  sends SIGTERM to the pgid, waits 15 s, SIGKILL. Queued jobs are cancelled
  by removing the marker and setting state.
- Reorder: `queue.reorder(jobid, prio)` renames the marker.
- Idle terminate (only when `config.provider` is set): if no running jobs
  and the queue has been empty for `idle_minutes`, or `ttl_hours` has
  elapsed since `config.created_at` and nothing is running: write
  `draining`, run one final sync of every job's state and log to
  `s3_prefix`, then call `terminate.self_terminate()`. On failure: remove
  `draining`, log loudly, keep dispatching, retry every 10 minutes.
- Two consecutive `failed: low-util` jobs: stop dispatching, log, and (if
  ephemeral) drain and terminate.
- Exit when the queue is empty, nothing is running, and the host is not
  ephemeral. Ephemeral hosts keep the dispatcher alive until terminate.

## Runner (one process per job)

1. Export `CUDA_VISIBLE_DEVICES=<assigned UUIDs comma-joined>` (empty string
   when `gpus: 0`), the spec `env`, and the secrets file. Assert via
   `nvidia-smi --query-gpu=index,uuid` that every assigned UUID is present
   on the host; fail the job otherwise.
2. `phase=setup`: run `spec.setup` in `workdir` with `bash -eo pipefail`.
3. GPU preflight **inside the job's environment**: if `gpus > 0`, run
   `uv run --no-sync python -c "<real op>"` (the `shared/gpu.py` probe:
   `is_available()` then a tensor add + `.item()`), and assert
   `device_count()` equals `gpus`. Failure -> `failed: gpu-preflight`.
4. Start the sync loop (background thread or subprocess) for `outputs`,
   every `sync_interval_s`, skipping files modified in the last 10 s.
   Also upload `log.txt` and `state.json` to `s3_prefix/jobs/<id>/` on
   the same cadence.
5. `phase=main`: run `spec.command`, stdout+stderr appended to `log.txt`.
   Start the low-util watchdog after `grace_min`: sample assigned GPUs'
   utilization every 30 s; if the rolling mean over `window_min` is below
   `floor_pct`, SIGTERM the process group, then SIGKILL after 15 s, status
   `failed: low-util`. `max_runtime_min` is enforced the same way with
   reason `timeout`.
6. Capture the exit code **before** any cleanup. Stop the sync loop and run
   one final sync; a failed final sync makes a succeeded job `failed:
   sync`. Write final state. Exit code of the runner = job exit code.

## Control side: `gpuc` CLI

```
gpuc host add local  --gpus GPU-uuid[,..]                          # this machine
gpuc host add spar   --ssh user@host [--port N] --gpus GPU-uuid,.. # shared box
gpuc host bootstrap <host>        # install uv + package, write config, run host preflight, start dispatcher
gpuc host probe <host>            # print driver, GPUs+UUIDs, disk, logind KillUserProcesses, systemd --user, network timing
gpuc host list | remove <host>

gpuc submit job.yaml --host <host>                                  # existing host
gpuc submit job.yaml --runpod --gpu A40[,RTX4090] [--min-vram 24] [--max-price 0.60] [--cloud secure|community]
                     [--cuda-min 12.8] [--idle-min 15] [--ttl-hours 24] [--reuse]   # provision or reuse a gpuc pod
gpuc status [--host H] [--all] [--suspects]
gpuc logs <jobid> [-f]           # tail from the host over transport; S3 fallback with a note
gpuc cancel <jobid>
gpuc reorder <jobid> --priority N
gpuc requeue <jobid> [--host H | --runpod ...]   # resubmit from the S3 spec, new attempt
gpuc reconcile [--once]          # the loop; installable as a systemd --user service via `gpuc reconcile --install`
gpuc pods                        # provider view: every pod with our prefix, cost, util, age, desired?
```

Local state: `~/.local/share/gpu-coordinator/` with `hosts.json`,
`desired/<host>.json` for ephemeral hosts, and a lock for reconcile.
Config file `~/.config/gpu-coordinator/config.toml`: `s3_bucket`,
`runpod_pod_prefix = "gpuc-"`, `max_pods = 3`, `max_total_usd_per_hour = 3.0`,
`ssh_key = "~/.ssh/id_ed25519"`.

## Transport

`LocalTransport` runs subprocesses directly. `SshTransport` uses the system
`ssh`/`rsync` with `-o BatchMode=yes -o ConnectTimeout=15`, a `ControlMaster`
socket under the state dir, per-command timeouts, and host keys pinned on
first contact into `~/.local/share/gpu-coordinator/known_hosts`. Code sync is
`rsync` of `git ls-files` output from the submitter's working directory
(plus a `git diff` saved as `uncommitted.patch` in the job dir). `put_file`
writes 0600 content via stdin (`cat > path && chmod 600 path`); secrets
never touch argv.

## Bootstrap (any host, idempotent)

1. `curl -LsSf https://astral.sh/uv/install.sh | sh` if `~/.local/bin/uv`
   is missing; `uv python install 3.12` if no suitable interpreter.
2. rsync the `gpuc` package to `~/.gpuc/pkg/`; install `aws` CLI v2 bundle
   into `~/.local/aws-cli` and `uv tool install huggingface_hub` (both
   skipped if present; failures are warnings).
3. Write `~/.gpuc/config.json` from the host registry entry.
4. Run `python -m gpuc.host health` and fail bootstrap on a failed check.
5. Start the dispatcher.

## RunPod provider (v2 REST, `https://api.runpod.io/v2`, bearer `RUNPOD_API_KEY`)

- `offers(constraints)`: `GET /catalog/gpus?include=AVAILABILITY&product=POD&cloud=<tier>&minCudaVersion=<x>`
  once per requested tier; filter by name list / min VRAM / max price /
  availability != NONE and at least one `cudaVersions[].available`; sort by
  price. Pass GPU ids exactly as the catalog returns them.
- `create(offer)`: `POST /pods` with `name="gpuc-<host>"`, `image`
  (default `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`), `gpu.id`,
  `gpu.minCudaVersion`, `cloud`, `disk`, `ports: ["22/tcp"]`,
  `startSsh: true`, `env: {"HF_HUB_ENABLE_HF_TRANSFER": "0"}`. Before the
  first create ever, ensure the local public key is in
  `PUT /account/ssh-keys` (merge, do not replace others). One capacity
  error -> next offer. Never set secrets in `env`.
- `get(id)`: status, `cudaVersion`, `cost`, `runtime.gpus[].util`,
  `ssh.direct`. `logs(id)`: `GET /pods/{id}/logs` (SSE; read with urllib
  and a timeout). `terminate(id)`: `POST /pods/{id}/action {"action":"terminate"}`
  then poll `get` until `TERMINATED` or 404. `list()`:
  `GET /pods?includeClusterPods=true`.
- Caps: before create, count pods with our prefix and sum their `cost`;
  refuse if `max_pods` or `max_total_usd_per_hour` would be exceeded.

## Provisioning flow (`gpuc submit --runpod`)

1. Write the spec to S3 (`s3://bucket/gpuc/specs/<jobid>.json`) first.
2. If `--reuse` (default on): pick an existing desired host whose offer
   matches constraints and whose dispatcher is alive; enqueue there.
3. Else for each offer in order: check caps; `create`; record
   `desired/<host>.json` with pod id, offer, created_at, ceiling; poll
   `get` until RUNNING **and** `ssh.direct` present; poll SSH until a
   trivial command succeeds; run bootstrap (which runs host health and
   starts the dispatcher); deliver the RunPod key as
   `~/.gpuc/secrets/runpod` (0600) for self-terminate. Enqueue. On any
   broken-host signature in `logs`, or the 15-minute ceiling, or a health
   failure: `terminate`, wait for TERMINATED, try the next offer.
4. Day-one test to run before relying on it: does the pod-scoped
   `RUNPOD_API_KEY` inside the pod terminate its own pod? If yes, do not
   deliver an account key at all.

## Reconcile loop

Every 60 s under a lock: for each `desired/` host, `get` its pod; if
missing or TERMINATED, mark the desired entry gone and note any jobs that
were running there (for `requeue`). For every provider pod with our prefix
not in `desired/`, or older than its TTL, terminate and log. Never touch a
pod without the prefix. If `desired/` is unreadable, do nothing and log an
error (fail closed). `--install` writes a `systemd --user` service + timer.

## Status output

Per host: kind, reachable?, dispatcher alive?, GPUs (owned/free), queue
(id, name, prio), running (id, name, phase, minutes, last util), recent
finished (id, status, reason). For ephemeral hosts also: pod status, $/h,
age, provider util. `--suspects`: running jobs in `phase=main` past
`grace_min` with mean util below floor over the last 10 min, and any pod
older than TTL. Never kills anything.

## Testing rules

- Unit tests run without a GPU (mock `nvidia-smi` output, temp `~/.gpuc`).
- Local GPU integration tests use tiny tensors (`torch.zeros(8)`), never
  more than ~100 MB VRAM; other people's jobs share the card.
- RunPod integration: A40 only, `--max-price 0.60`, a job whose command
  is under two minutes, `--idle-min 2`, `--ttl-hours 1`, and the test
  asserts teardown via `list()` and prints the final `GET /billing/pods`
  for the pod. Two pods named `subrep-*` belong to someone else: read them
  in `list()`, never act on them. Every test that creates a pod has a
  `finally` that terminates it.

## Code conventions

uv, ruff, pyright strict-ish, type annotations everywhere, pydantic on the
control side only. No comments that restate the next line; explain *why*
where it is non-obvious. Prefer good names over docstrings. Errors carry
the command, the host, and the last lines of output.
