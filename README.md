# gpu-coordinator

One submit path for three kinds of GPU host:

- **local** — this desktop and its cards,
- **ssh** — a shared box you have no sudo on, using a subset of its GPUs,
- **runpod** — an ephemeral pod, provisioned for the job and torn down after it.

Each host runs a small stdlib-only dispatcher out of `~/.gpuc`; the host is
authoritative for its own queue, and S3 is an optional mirror. `gpuc submit`
is the same command for all three.

Design contract: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).
Reasoning behind it: [`docs/requirements-review.md`](docs/requirements-review.md).

## Install

```sh
uv tool install .                  # from a checkout, puts `gpuc` on your PATH
uv run gpuc --help                 # or just run it out of the checkout
```

Nothing has to be installed on a host by hand: `gpuc host bootstrap` installs
uv, a Python, the `gpuc.host` package, and the `aws`/`hf` upload helpers into
`$HOME` over ssh.

## Configuration

`gpuc` works with no config file at all (no S3 mirror, defaults for everything
else) and says so once per run. To write a commented one:

```sh
gpuc config init      # ~/.config/gpu-coordinator/config.toml
gpuc config show      # the effective settings, file or not
```

Keys: `s3_bucket`, `runpod_pod_prefix`, `max_pods`, `max_total_usd_per_hour`,
`ssh_key`, `image`, `disk_gb`.

`--runpod`, `gpuc pods` and `gpuc reconcile` need `RUNPOD_API_KEY` exported;
without it they fail immediately with one line rather than part-way through.

## Quick start: local

```sh
nvidia-smi --query-gpu=index,uuid --format=csv     # pick the UUIDs to hand over
gpuc host add local --gpus GPU-2a4bad3b-...
gpuc host bootstrap local
gpuc submit job.example.yaml --host local
gpuc status
gpuc logs <jobid> -f
gpuc cancel <jobid>
```

## Quick start: a shared SSH box

```sh
gpuc host add spar --ssh me@spar --port 22 --gpus GPU-aaa,GPU-bbb
gpuc host probe spar        # driver, every GPU as `[index] UUID name`, disk, systemd --user, network speed
gpuc host bootstrap spar
gpuc submit job.yaml --host spar
```

Run `probe` **before** `host add` if you do not know the UUIDs: it prints every
card with its index and UUID. Only the UUIDs you list are ever assigned, so the
other users of the box keep the rest; jobs get `CUDA_VISIBLE_DEVICES` set to
UUIDs, not indices, which cannot drift when someone else's job starts.

## Quick start: RunPod

```sh
export RUNPOD_API_KEY=...
gpuc submit job.yaml --runpod --gpu A40 --max-price 0.6 --idle-min 15 --ttl-hours 24
gpuc pods                   # every pod with our prefix: cost, util, age, is it wanted?
gpuc reconcile --once       # terminate leaked or expired pods now
gpuc reconcile --install    # write a systemd --user service + timer (it prints how to enable it)
```

Provisioning mirrors the spec to S3 first, then walks the catalog offers
cheapest-first, creates a pod, waits for a direct SSH endpoint, bootstraps it,
runs the host health check, and enqueues. Any failure terminates the pod and
tries the next offer. `--reuse` is the default: an existing gpuc pod that
matches the constraints and whose dispatcher is alive gets the job instead.

**Auto-down.** The pod terminates itself when the queue has been empty and
nothing has run for `--idle-min`, or when `--ttl-hours` has elapsed with
nothing running, or after two consecutive `low-util` failures. It drains
(final sync of every job's log and state) before calling the provider.

What that does **not** guarantee: self-terminate needs the pod to still be
reachable and the provider API to answer. If the pod wedges, loses network, or
the API call fails, it keeps billing. `gpuc reconcile` is the backstop — it
terminates pods with our prefix that no `desired/` record wants or that are
past their TTL — and it only runs when you run it, so install the timer.
`--install` writes the units but deliberately does not enable them; it prints
the `systemctl --user` lines and where to put `RUNPOD_API_KEY` for the service.
It never touches a pod without the configured prefix, and if `desired/` is
unreadable it terminates nothing at all.

## The job spec

See [`job.example.yaml`](job.example.yaml) for a fully commented example with
S3 and Hugging Face outputs and secrets. Fields:

| field | default | meaning |
| --- | --- | --- |
| `command` | required | run in the workdir under `bash -eo pipefail`, phase `main` |
| `name` | `""` | a label for `status`; not an identifier |
| `setup` | none | run before `command`, phase `setup` |
| `gpus` | `1` | how many of the host's owned GPUs to assign; `0` never waits |
| `env` | `{}` | plain environment for the job |
| `secrets` | `[]` | names read from *your* shell, delivered 0600 as `~/.gpuc/secrets/<jobid>.env` |
| `outputs` | `[]` | `{path, s3}` and/or `{path, hf, hf_path}`; `path` is relative to the workdir |
| `sync_interval_s` | `180` | background upload cadence |
| `priority` | `50` | 0 first, 99 last |
| `max_runtime_min` | none | wall clock cap; over it the job is `failed: timeout` |
| `low_util` | on | `{enabled, window_min: 25, floor_pct: 5, grace_min: 10}` |
| `requires` | `{}` | e.g. `cuda_min: "12.8"`; informs provisioning only |

`{job_id}` expands in output destinations. Output namespaces are unique by
construction, so nothing guards against overwriting.

Secrets never appear in argv, in a pod's provider-visible env, or in any log.
The runner loads them into the job's environment *and* into the environment the
upload commands run with, so `secrets: [AWS_ACCESS_KEY_ID, ...]` is enough to
push outputs to S3 — no credential file on the host. The file is removed after
the final sync, not before it.

## Low-util and cancel

The runner samples the assigned GPUs every 30 s, but only during phase `main`,
so model downloads and compiles in `setup` can never look idle. After
`grace_min` minutes of `main`, if the rolling mean over `window_min` is below
`floor_pct`, the job's whole process group gets SIGTERM, then SIGKILL 15 s
later, and the job is `failed: low-util`. `gpuc status --suspects` shows jobs
that are heading that way and never kills anything.

`gpuc cancel` writes a marker the runner sees; the runner kills the job's
process group (SIGTERM, 15 s, SIGKILL), runs the final sync, and records
`cancelled`. A job cancelled before its first phase starts never starts one.
A queued job is cancelled by removing its queue marker.

## Where state lives

On each host, `~/.gpuc/` (0700): `config.json`, `queue/`, `jobs/<id>/` with
`spec.json`, `state.json`, `log.txt`, `workdir/`, `outputs/`,
`secrets/<id>.env`, plus `dispatcher.lock`, `dispatcher.heartbeat` and
`dispatcher.log`. All writes are atomic.

Locally, `~/.config/gpu-coordinator/config.toml` for settings and
`~/.local/share/gpu-coordinator/` for state: `hosts.json`, `desired/<host>.json`
for pods we want alive, `jobs/` (the local job index), `known_hosts` plus
`known_hosts.d/<pod>` per pod, and `state.lock`. SSH ControlMaster sockets go
in `$XDG_RUNTIME_DIR/gpuc/` (or `/tmp/gpuc-<uid>/`), because a unix socket path
must fit in 108 bytes.

With `s3_bucket` set, specs go to `s3://<bucket>/gpuc/specs/<jobid>.json` and
each job's `log.txt` and `state.json` are mirrored under the host's prefix.

## Troubleshooting

| symptom | what it means | what to do |
| --- | --- | --- |
| `status` says `dispatcher DOWN` | nothing is holding the host's lock, or its heartbeat is over 30 s old | `gpuc host bootstrap <host>` (idempotent); any `gpuc submit` also restarts it |
| provisioning gives up with "no direct SSH endpoint" | RunPod never exposed port 22 within the 15-minute ceiling — usually a bad placement | the pod was already terminated; re-run the submit, or widen `--gpu` / `--cloud any` |
| `ssh ... cannot create its ControlMaster socket` | the socket path is over the 108-byte unix limit | point `XDG_RUNTIME_DIR` at a short directory, or unset it to use `/tmp/gpuc-<uid>` |
| bootstrap fails with "host health failed" | the driver, disk or network check on the host said no | read the named check; fix the host (free disk, load the driver) and re-run bootstrap |
| job is `failed: low-util` | the GPU sat under `floor_pct` for `window_min` of phase `main` | raise `low_util.grace_min`, lower `floor_pct`, or set `low_util.enabled: false` for genuinely CPU-bound work |
| job is `failed: sync` (or `...+sync`) | the final upload failed; the run itself may have been fine | check the tail of `gpuc logs <jobid>`; the usual cause is missing `secrets:` for the destination, or no `aws`/`hf` on the host (re-run bootstrap) |
| job is `failed: no-outputs` | the `outputs` path was never written | check the job actually wrote to that path, relative to the workdir |
| `status` says `POD GONE` | the pod is terminated or missing but the registry still lists it | `gpuc reconcile --once` |
