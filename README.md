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
(`gpuc reconcile --install`, which only writes unit files, does not.)

## Commands

```
gpuc host add <name> [--ssh user@host] [--port N] [--gpus UUID,..] [--gpuc-home PATH]
                     [--persistent-root PATH] [--env K=V] [--s3-prefix s3://..]
                     [--idle-min N] [--ttl-hours N]
gpuc host set <name> [--gpus UUID,..] [--persistent-root PATH] [--gpuc-home PATH]
                     [--env K=V] [--s3-prefix s3://..] [--idle-min N] [--ttl-hours N]
gpuc host bootstrap <name> [--health-args "..."]     # idempotent; also restarts the dispatcher
gpuc host probe <name>                               # driver, GPUs+UUIDs, disk+fs type, systemd, network
gpuc host list | gpuc host remove <name>             # remove forgets locally; the host is untouched
gpuc submit <job.yaml|-> --host <name>               # or --runpod ... (flags below)
gpuc status [--host H] [--all] [--suspects]          # --all adds jobs only the index knows
gpuc logs <job-id> [-f] [-n LINES] [--host H]        # host first, S3 mirror as a noted fallback
gpuc cancel <job-id> [--host H]
gpuc reorder <job-id> --priority N [--host H]        # queued jobs only
gpuc requeue <job-id> [--host H | --runpod ...]      # re-reads the spec from S3, attempt+1
gpuc reconcile [--once] [--interval S] [--install]
gpuc pods [--no-heartbeat]                           # --no-heartbeat skips the per-pod ssh check
gpuc config init [--force] | gpuc config show
```

`--host` is optional on `logs`, `cancel` and `reorder`: the local job index is
tried first, then every registered host is asked whether it knows the id.
A job file of `-` is read from stdin.

`gpuc host set` edits one registered host in place — handing over two more
GPUs, or moving its state to a persistent root — without `remove`+`add`, which
would drop everything else about the entry. Only the flags you pass change;
`--gpus ''` hands every card back. Nothing on the host changes until the next
`gpuc host bootstrap <name>`.

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
                            # and whether $HOME is on an overlay (see below)
gpuc host bootstrap spar
gpuc submit job.yaml --host spar
```

Run `probe` **before** `host add` if you do not know the UUIDs: it prints every
card with its index and UUID. Only the UUIDs you list are ever assigned, so the
other users of the box keep the rest; jobs get `CUDA_VISIBLE_DEVICES` set to
UUIDs, not indices, which cannot drift when someone else's job starts.

## Hosts whose home directory is wiped on restart

A container-backed host (a Kubernetes pod, most cloud notebooks) usually has
`$HOME` on the image's throwaway upper layer: `df -T $HOME` says `overlay`, and
everything in it is gone the next time the pod is rescheduled — uv, the
`gpuc.host` package, the queue, every job dir and every venv. `gpuc host probe`
says so and names the flag:

```
  home_fs: overlay overlay 3748906852 105410796 3643496056   3% /
  note: $HOME is on an overlay filesystem, so it is a container's throwaway
        upper layer and is wiped on every restart.
        Point this host at a volume that survives:
        gpuc host set spar --persistent-root /mnt/<volume>/$USER
```

`--persistent-root R` moves **gpuc home** — and only gpuc home — to `R/gpuc`:
`config.json`, `queue/`, every `jobs/<id>/` with its spec, state, log and
workdir. Those are the things that cannot be reinstalled. Everything that can
be stays in `$HOME`: uv, its cache and managed Pythons, `uv tool` installs and
the `aws` CLI bundle, because `gpuc host bootstrap` puts them back in seconds
and these shared volumes are much slower than a container's local disk — a
venv or a dataset cache on one is a bad trade. `R` is created 0700 if we create
it (an existing `R` keeps its mode: it may be your own directory with other
things in it), because these volumes are usually world-writable with one
directory per user.

```sh
gpuc host add spar --ssh spar --persistent-root /mnt/ssd-2/$USER --gpus GPU-aaa,GPU-bbb
gpuc host set spar --persistent-root /mnt/ssd-2/$USER      # or move an existing host
gpuc host bootstrap spar
```

With a root, a restart costs one `gpuc host bootstrap` and the queue picks up
where it left off. Without one — which is the right choice when the shared
volume is slow and restarts are rare — the host comes back empty, and the
recovery is a re-submit:

**Runbook: the pod restarted.** The symptom is `gpuc status` reporting
`dispatcher DOWN`, or ssh failing outright.

1. If your SSH key or `authorized_keys` lived in the wiped home, put it back
   (`ssh-copy-id -i ~/.ssh/id_ed25519_spar user@host`, or however that host is
   provisioned), then check `gpuc host probe spar` answers at all.
2. `gpuc host bootstrap spar` — idempotent, and the whole of the host-side
   recovery: uv, a Python, the package, `config.json`, the health check and the
   dispatcher.
3. Find what was on it and re-submit:

   ```sh
   gpuc status --host spar --all      # what the job index says was there
   gpuc requeue <job-id> --host spar  # one line per job you still want
   ```

   `--all` lists jobs the index knows but the host does not, with their host
   name, and prints a ready-made `gpuc requeue` line. `requeue` re-reads the
   spec from the S3 mirror, so this needs `s3_bucket` set; without a mirror,
   re-submit the job file by hand.
4. With a `--persistent-root`, step 3 is unnecessary: the queue directory came
   back with the volume, so queued jobs start again as soon as the dispatcher
   does. Jobs that were *running* when the pod died are failed on the
   dispatcher's next start (their runner is gone) and are the only ones to
   `requeue`.

The health check's disk floor is measured on gpuc home's filesystem — the
overlay by default, `R`'s volume when a root is set — so it is always about the
volume the jobs actually fill. These shared volumes are often near full;
`--health-args "--min-free-gb 20"` demands more. The download floor is about
the host's network (this pod measures ~25 MB/s, the floor is 1 MB/s).

Per-host environment, if some path really does belong elsewhere:

```sh
gpuc host add|set spar --env HF_HOME=/mnt/ssd-2/$USER/hf --env WANDB_MODE=offline
gpuc host bootstrap spar     # the env reaches the host in config.json
```

Nothing populates that automatically. It is applied by the dispatcher and by
the runner to every job's environment *before* the job's own `env:`, so a job
can override any of it, and `UV_INSTALL_DIR`/`UV_TOOL_BIN_DIR` in it also go on
the front of `PATH`.

## Quick start: RunPod

```sh
export RUNPOD_API_KEY=...
gpuc submit job.yaml --runpod --gpu A40 --max-price 0.6 --idle-min 15 --ttl-hours 24
gpuc pods                   # every pod with our prefix: cost, util, age, is it wanted?
gpuc reconcile --once       # terminate leaked or expired pods now
gpuc reconcile --install    # write a systemd --user service + timer (it prints how to enable it)
```

Flags for `--runpod` (the same set on `gpuc submit` and `gpuc requeue`):

| flag | default | meaning |
| --- | --- | --- |
| `--gpu A40[,RTX4090]` | required with `--runpod` | catalog names, matched case-insensitively against both the short name (`A40`) and the catalog id (`NVIDIA A40`) |
| `--gpu-count N` | `1` | GPUs in the pod |
| `--min-vram GB` | none | per-GPU VRAM floor |
| `--max-price USD` | none | **whole pod** per hour, so at `--gpu-count 2` it is compared against twice the per-GPU price |
| `--cloud secure\|community\|any` | `secure` | `any` queries both tiers and sorts the merged list by price |
| `--cuda-min X.Y` | `12.8` | host CUDA floor, passed to the catalog query and to `create` |
| `--idle-min N` / `--ttl-hours N` | `15` / `24` | the pod's own auto-down timers |
| `--disk GB` / `--image REF` | from `config.toml` | container disk and pod image |
| `--no-reuse` | reuse is on | always create a new pod |
| `--name-hint TEXT` | `job` | goes into the pod name after the prefix |
| `--health-args "..."` | none | extra flags for the on-host health check, e.g. `--min-mbps 0.1` |

Provisioning checks what it can locally first (spec validity, `secrets:`
present in your shell, a git workdir, `gpus:` against `--gpu-count`), mirrors
the spec to S3, then walks the catalog offers cheapest-first: create, wait for
a direct SSH endpoint, bootstrap, host health check, enqueue. Any failure
terminates that pod and tries the next offer; a cap that the cheapest offer
trips aborts the whole submit instead, since every later offer costs more.

Reuse is the default. An existing gpuc pod is used instead of a new one when
its recorded offer still matches the request, it owns enough GPUs, the
provider says the pod is `RUNNING`, its dispatcher heartbeat is under 30 s
old, and it is neither draining nor paused. A registry entry whose pod the
provider no longer has is forgotten on the spot rather than dialled.

**Auto-down.** The pod terminates itself when the queue has been empty and
nothing has run for `--idle-min`, or when `--ttl-hours` has elapsed with
nothing running, or after two consecutive `low-util` failures. It drains
(final sync of every job's log and state) before calling the provider.

What that does **not** guarantee: self-terminate needs the pod to still be
reachable and the provider API to answer. If the pod wedges, loses network, or
the API call fails, it keeps billing. `gpuc reconcile` is the backstop — it
terminates pods with our prefix that no `desired/` record wants or that are
past their TTL (TTL measured from the provider's own `createdAt`, not from
when this machine first heard of the pod) — and it only runs when you run it,
so install the timer. `--install` writes the units but deliberately does not
enable them; it prints the `systemctl --user` lines and where to put
`RUNPOD_API_KEY` for the service.

The reaper fails closed in every direction: it never touches a pod without the
configured prefix, it terminates nothing at all if `desired/` is unreadable,
it leaves a prefixed pod that has no `desired/` record alone until it is older
than the 15-minute provisioning ceiling (another session may be mid-create),
and it leaves one alone entirely if the provider does not report a creation
time. A terminate that fails keeps its `desired/` record, is logged loudly,
and makes `gpuc reconcile --once` exit non-zero.

**Caps.** `max_pods` and `max_total_usd_per_hour` are account-wide: they are
checked against the provider's pod list (every pod with our prefix, whoever
created it), not against this machine's registry, immediately before `create`
and with the local state lock held. So two `gpuc submit --runpod` in two
shells on this machine cannot both slip past the caps. Two *different
machines* sharing one account still can — the window is the length of one
`create` call.

**A pod is never created without a record of it.** `desired/<host>.json` is
written under the same lock as the create, and every exit from provisioning
between `create` and the final registry write — including Ctrl-C, a failed
`desired/` write, and errors nothing thought to catch — terminates the pod
before unwinding.

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

On each host, `~/.gpuc/` — or `<persistent-root>/gpuc/` — (0700):
`config.json`, `queue/`, `jobs/<id>/` with
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
| everything on a host is suddenly gone | the container restarted and `$HOME` was on the overlay | `gpuc host bootstrap <host>`, then `gpuc status --host <host> --all` and `gpuc requeue` what you still want ([runbook](#hosts-whose-home-directory-is-wiped-on-restart)) |
