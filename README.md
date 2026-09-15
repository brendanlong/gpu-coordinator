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
`ssh_key`, `image`, `disk_gb`, `dead_dispatcher_minutes` (how long an ephemeral
host may be silent before `gpuc reconcile` terminates it; 30 by default).

`--runpod`, `gpuc pods` and `gpuc reconcile` need `RUNPOD_API_KEY` exported;
without it they fail immediately with one line rather than part-way through.
(`gpuc reconcile --install`, which only writes unit files, does not.)

## Commands

```
gpuc host add <name> [--ssh user@host] [--port N] [--gpus UUID,..] [--gpuc-home PATH]
                     [--persistent-root PATH] [--env K=V] [--cache-dir PATH]
                     [--s3-prefix s3://..] [--retention-days N] [--idle-min N] [--ttl-hours N]
gpuc host set <name> [--gpus UUID,..] [--persistent-root PATH] [--gpuc-home PATH]
                     [--env K=V] [--cache-dir PATH] [--s3-prefix s3://..]
                     [--retention-days N] [--idle-min N] [--ttl-hours N]
gpuc host bootstrap <name> [--health-args "..."]     # idempotent; also restarts the dispatcher
gpuc host probe <name>                               # driver, GPUs+UUIDs, disk+fs type, systemd, network
gpuc host clean <name> --uv-cache                    # `uv cache prune` on the host
gpuc host list | gpuc host remove <name>             # remove forgets locally; the host is untouched
gpuc submit <job.yaml|-> --host <name> [--no-git]    # or --runpod ... (flags below)
gpuc status [--host H] [--all] [--suspects]          # --all adds jobs only the index knows
           [--recent N] [--since 24h|7d|90m]         # how much of the finished list to show
gpuc logs <job-id> [-f] [-n LINES] [--host H]        # host first, S3 mirror as a noted fallback
gpuc cancel <job-id> [--host H]
gpuc clean --host H (--all-finished | --older-than DAYS) [--dry-run]   # free finished workdirs
gpuc clean --host H --purge [--older-than DAYS] [--verify] [--force] [--dry-run]  # whole job dirs
gpuc reorder <job-id> --priority N [--host H]        # queued jobs only
gpuc requeue <job-id> [--host H | --runpod ...] [--no-git]   # re-reads the spec from S3, attempt+1
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

`gpuc host list` and `gpuc status` name the cards once a host has been probed or
bootstrapped, so a registry of UUIDs is readable at a glance:

```
spar   ssh   spar_cluster   gpus=2 (2x NVIDIA A40 45 GB, driver 580.65.06) python=... bootstrapped=...
  NVIDIA A40                   45 GB   GPU-80646905-50a9-afc1-4375-43ca475b15e4
  NVIDIA A40                   45 GB   GPU-83123e65-fe58-7831-6b21-1814b07c25f7

host spar [ssh] spar_cluster  dispatcher 2s ago  gpus 1/2 free (2x NVIDIA A40 45 GB, driver 580.65.06)
  gpu     NVIDIA A40 45 GB  GPU-80646905-...  busy 20260915-195133-be2d4a
  gpu     NVIDIA A40 45 GB  GPU-83123e65-...  free
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
be stays in `$HOME`: uv, its managed Pythons, `uv tool` installs and the `aws`
CLI bundle, because `gpuc host bootstrap` puts them back in seconds and these
shared volumes are much slower than a container's local disk. The one
exception is uv's *cache*, which bootstrap moves to `R/.cache/uv` when `R` is
on a different filesystem from `$HOME`: the venv is inside a workdir under
gpuc home either way, so a cache on the other volume does not keep it off the
slow disk — it just makes uv copy 6.5 GB instead of linking it
([details](#disk-workdirs-and-the-uv-cache)). `R` is created 0700 if we create
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

Nothing populates `--env` automatically. It is applied by the dispatcher and by
the runner to every job's environment *before* the job's own `env:`, so a job
can override any of it, and `UV_INSTALL_DIR`/`UV_TOOL_BIN_DIR` in it also go on
the front of `PATH`. (`--cache-dir` is the one setting bootstrap *does* choose
for you; see [the uv cache](#disk-workdirs-and-the-uv-cache).)

## Quick start: RunPod

```sh
export RUNPOD_API_KEY=...
gpuc submit job.yaml --runpod --gpu A40 --max-price 0.6 --idle-min 15
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
| `--idle-min N` | `15` | idle minutes before the pod terminates itself |
| `--ttl-hours N` | none | optional hard cap on the pod's life; see [auto-down](#auto-down) |
| `--disk GB` / `--image REF` | from `config.toml` | container disk and pod image |
| `--no-reuse` | reuse is on | always create a new pod |
| `--name-hint TEXT` | `job` | goes into the pod name after the prefix |
| `--health-args "..."` | none | extra flags for the on-host health check, e.g. `--min-mbps 0.1` |

Provisioning checks what it can locally first (spec validity, `secrets:`
present in your shell, a git workdir unless `--no-git`, `gpus:` against
`--gpu-count`, and `max_runtime_min` against any `--ttl-hours`), mirrors
the spec to S3, then walks the catalog offers cheapest-first: create, wait for
a direct SSH endpoint, bootstrap, host health check, enqueue. Any failure
terminates that pod and tries the next offer; a cap that the cheapest offer
trips aborts the whole submit instead, since every later offer costs more.

Reuse is the default. An existing gpuc pod is used instead of a new one when
its recorded offer still matches the request, it owns enough GPUs, the
provider says the pod is `RUNNING`, its dispatcher heartbeat is under 30 s
old, and it is neither draining nor paused. A registry entry whose pod the
provider no longer has is forgotten on the spot rather than dialled.

<a name="auto-down"></a>
**Auto-down.** The pod terminates itself when the queue has been empty and
nothing has run for `--idle-min`, or after two consecutive `low-util` failures.
It drains (final sync of every job's log and state) before calling the provider.

**There is no overall TTL by default.** `--ttl-hours` is an opt-in hard cap: a
wall clock that kills a training run at hour 24 is a worse failure than a pod
that idles for fifteen minutes first. When you do set one, the *dispatcher*
enforces it — past the cap it stops the running job with reason `ttl`, lets the
runner sync its outputs, then drains and terminates — and the reaper enforces it
as a backstop from the provider's own `createdAt`. `gpuc submit --runpod`
refuses a job whose `max_runtime_min` is longer than the TTL you asked for,
rather than letting the TTL kill it halfway. `gpuc host set <host> --ttl-hours
-1` takes a cap back off.

**What stops a forgotten pod instead.** `gpuc reconcile` terminates a
bootstrapped pod whose dispatcher heartbeat has been dead — or whose ssh has not
answered at all — for `dead_dispatcher_minutes` (30 by default) with no job
running, and says so loudly. A pod running a job is never touched, however old
it is; a pod that cannot tell us it is running one cannot idle-terminate itself
either, and it is billing all the same.

What that does **not** guarantee: self-terminate needs the pod to still be
reachable and the provider API to answer. If the pod wedges, loses network, or
the API call fails, it keeps billing. `gpuc reconcile` is the backstop — it
terminates pods with our prefix that no `desired/` record wants, that have gone
silent for `dead_dispatcher_minutes`, or that are past a TTL you set (measured
from the provider's own `createdAt`, not from when this machine first heard of
the pod) — and it only runs when you run it,
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
| `outputs` | `[]` | `{path, s3}` and/or `{path, hf, hf_path, hf_create}`; `path` is relative to the workdir |
| `sync_interval_s` | `180` | background upload cadence |
| `priority` | `50` | 0 first, 99 last |
| `max_runtime_min` | none | wall clock cap; over it the job is `failed: timeout` |
| `low_util` | on | `{enabled, window_min: 25, floor_pct: 5, grace_min: 10}` |
| `requires` | `{}` | e.g. `cuda_min: "12.8"`; informs provisioning only |
| `cleanup` | `on_success` | when to delete `workdir/`: `on_success`, `always` or `never` |

`{job_id}` expands in output destinations. Output namespaces are unique by
construction, so nothing guards against overwriting. `hf_create: true` on an
output lets the sync preflight create a Hugging Face repo that does not exist
yet; without it, a missing repo fails the job in seconds instead of creating
`org/typo`.

**Files already under an output path are not your job's outputs.** A checkout
usually ships committed files where the results go — `results/report-elephant.md`
from the last run, a figure in `figures/`. Those arrive in the workdir with the
code, so before `setup` the runner records every declared output path's contents
(relative path, size, mtime) in `jobs/<id>/outputs_baseline.json`. Every upload
skips files that still match it, and a path that ends up holding *only* those
files counts as having produced nothing (`failed: no-outputs`). `gpuc submit`
says so up front:

```
WARNING: 3 pre-existing file(s) under results/ are in the checkout and will not be
         uploaded as this job's outputs; use a job-specific output dir
         (for example results/{job_id}/) if you meant them to be
```

**The sync preflight.** Before `main`, and after the GPU preflight, the runner
proves the uploads can actually happen: with S3 outputs (or a host `s3_prefix`)
the `aws` binary must resolve and a small `.preflight` object must upload to
every destination and to the host's mirror prefix; with HF outputs, `hf` must
resolve, `hf auth whoami` must succeed with the job's token, and a `.preflight`
file must upload to each repo. A failure is `failed: sync-preflight` with the
command and its error in the log — seconds in, rather than after four hours of
training with the only copy of a checkpoint on a pod that is about to go away.
A job with no outputs on a host with no mirror checks nothing.

Secrets never appear in argv, in a pod's provider-visible env, or in any log.
The runner loads them into the job's environment *and* into the environment the
upload commands run with, so `secrets: [AWS_ACCESS_KEY_ID, ...]` is enough to
push outputs to S3 — no credential file on the host. The file is removed after
the final sync, not before it.

## Disk: workdirs and the uv cache

A job's `workdir/` is the rsynced code *and* whatever the job builds in it —
usually a venv, and a torch venv measures about 6.5 GB. It is also the only
part of a job dir that can be recreated, since `gpuc requeue` re-syncs it from
git, so it is the one part gpuc will delete.

**Per job: `cleanup:`.** The runner applies the policy after the final sync and
the final state write — never before, because `outputs:` paths live *inside*
the workdir — and records `workdir_removed` in `state.json`. `spec.json`,
`state.json` and `log.txt` always stay, so `gpuc logs` and `gpuc status` keep
working on a cleaned job.

| `cleanup:` | succeeded | failed | cancelled |
| --- | --- | --- | --- |
| `on_success` (default) | removed | **kept** | **kept** |
| `always` | removed | removed | removed |
| `never` | kept | kept | kept |

The default keeps a failed or cancelled workdir precisely so you can ssh in and
look at it.

**After the fact: `gpuc clean`.**

```sh
gpuc clean --host spar --all-finished --dry-run   # what would go, and how big
gpuc clean --host spar --all-finished             # every succeeded/failed/cancelled job
gpuc clean --host spar --older-than 7             # only jobs that ended over 7 days ago
```

It fails closed: a job that is running or queued, a job whose `state.json` is
missing or unreadable, and (under `--older-than`) a job with no usable
`ended_at` are all skipped and listed as kept. It also removes stale
`incoming/<id>.json` staged specs — ones whose job has finished, or orphans
over an hour old — and never anything else. Sizes are `du`-style allocated
blocks, so a venv sharing extents with the uv cache reads as an upper bound.

### Retention: what is kept, and what `--purge` removes

`clean` never touches `spec.json`, `state.json` or `log.txt` — a cleaned job
still answers `gpuc logs` and `gpuc status`. Those three files are small, but
they are kept *forever*, and on a long-lived box "forever" eventually shows up
in `ls`. `gpuc clean --purge` is the one command that removes them:

| | `clean` | `clean --purge` |
| --- | --- | --- |
| `workdir/` (code + venv + outputs) | removed | removed |
| `spec.json`, `state.json`, `log.txt` | **kept** | removed |
| stale `incoming/<id>.json` | removed | removed |
| a stray queue marker for the job | — | removed |
| running or queued jobs | never touched | never touched |

```sh
gpuc clean --host spar --purge --dry-run              # default: ended over 7 days ago
gpuc clean --host spar --purge --older-than 30        # a month instead
gpuc clean --host spar --purge --older-than 30 --verify   # HEAD each mirrored log first
gpuc clean --host spar --purge --older-than 0 --force     # delete unmirrored records too
```

**The precondition: it must be backed up.** After a job's last upload, the
runner writes `meta_synced_at` and `meta_synced_to` into `state.json` — set
only when the upload actually returned 0 — and then re-uploads `state.json`
once more so the mirror matches. A purge refuses any job without that record:

```
  SKIPPED 20260901-101500-a1b2c3  not backed up: no s3_prefix on this host
  SKIPPED 20260902-090000-d4e5f6  not backed up: final upload failed
```

Outputs are the second precondition. `outputs:` paths live *inside* the
workdir, and a failed or cancelled job keeps its workdir, so a job that ended
`failed: sync` can hold the only copy of a checkpoint. The runner records
`outputs_synced_at` only when the *final* output upload succeeded, and a purge
skips anything unconfirmed with `outputs not confirmed uploaded` (a spec with
no `outputs:`, or a workdir that is already gone, has nothing to confirm).
`gpuc status` flags those jobs so you can `gpuc requeue` them or copy them off:

```
  done    20260902-090000-d4e5f6 bulky failed (sync)  outputs not uploaded
  outputs 1 finished job(s) produced outputs that never reached S3/HF: 20260902-090000-d4e5f6; ...
```

An ephemeral host retries those uploads while it drains — three tries a minute
apart, five minutes at most — and if they still fail it records
`outputs_lost: true` and terminates anyway (the pod is billing, and whatever
sent it away — an idle timer, a TTL, the reaper — does not pause). `gpuc status --all` shows `OUTPUTS LOST` against such a job.

`--force` overrides both preconditions, removes the record anyway, and says so
loudly per job. `--verify` (control side only) HEADs each candidate's
`<s3_prefix>/jobs/<id>/log.txt` with your own credentials before deleting the
original; without it the host's `meta_synced_at` is trusted, which is the only
answer a host with no credentials of ours can give.

**Automatic retention.** `--retention-days N` makes the host's dispatcher do
this by itself — never with `--force`:

```sh
gpuc host set spar --retention-days 14
gpuc host bootstrap spar        # the setting reaches the host in config.json
gpuc host set spar --retention-days ''      # back to keeping everything (the default)
```

The dispatcher purges once at startup and then at most once an hour. On an
ephemeral host that is the whole of its life; on a **non-ephemeral host the
dispatcher only lives while jobs are queued or running**, so the sweep happens
on your next `gpuc submit` rather than on a timer. Each pass also does the
ordinary workdir clean over the same horizon, which needs no mirror — so on a
host with no `--s3-prefix`, `--retention-days` reclaims old venvs and nothing
else.

For any of this to ever delete anything, the host itself must have an
`s3_prefix` in its `config.json`:

```sh
gpuc host set spar --s3-prefix s3://my-bucket/gpuc/spar
gpuc host bootstrap spar        # the prefix only reaches the host here
```

A pod from `gpuc submit --runpod` gets one derived from `s3_bucket`
automatically. A `local` or `ssh` host does **not**: `s3_bucket` on its own
mirrors specs and the job index from this machine, not each job's log and state
from the host, so `--s3-prefix` is the flag that matters. (Set it only on a host
whose jobs can actually authenticate to that bucket — `secrets:` or an instance
role — since a final sync that fails turns an otherwise green job into
`failed: sync`.) Without a prefix every job stays `not backed up: no s3_prefix
on this host` and `--retention-days` is a no-op, which is the intended failure
mode: nothing is deleted that nothing else has a copy of.

After a purge, `gpuc status --all` still lists the job from the S3 index, and
`gpuc logs <id>` says it was purged from the host and falls back to the mirror:

```
note: job 20260901-101500-a1b2c3 was purged from host spar (gpuc clean --purge removes the
      whole job dir once it is mirrored)
note: falling back to the S3 mirror at s3://bucket/gpuc/spar/jobs/20260901-101500-a1b2c3/log.txt
```

`gpuc status` adds one line per host once finished workdirs hold more than 1 GiB:

```
  disk    12.9 GiB still in 2 finished job workdir(s); free it with: gpuc clean --host spar --all-finished
```

**The uv cache is shared, and must stay linkable.** uv caches wheels in
`~/.cache/uv` and materialises a venv by reflinking or hardlinking out of it —
so a second job that needs the same torch build costs seconds and almost no
disk. Both mechanisms only work *within one filesystem*, and `UV_LINK_MODE=copy`
disables them outright. gpuc therefore never sets `UV_LINK_MODE` and never sets
`UV_CACHE_DIR` unless the host's own config does.

The case that breaks it is a host whose gpuc home is not on `$HOME`'s
filesystem — a RunPod pod or any container with `--persistent-root
/workspace/$USER`, where gpuc home is on the network volume and `~/.cache` is
on the container's overlay. There uv copies every wheel into every venv, at the
full size of the venv, onto the slowest disk the host has. `gpuc host
bootstrap` checks for exactly this, with one comparison:

> if gpuc home and uv's cache are on different filesystems, set
> `UV_CACHE_DIR` to `<parent of gpuc home>/.cache/uv`.

Beside gpuc home, not inside it, so `gpuc clean` and an `rm -rf` of gpuc home
cannot take the cache with them. It is written into `HostConfig.env`, which the
dispatcher and the runner apply to every job. Pin it yourself with `gpuc host
add|set <host> --cache-dir PATH`, or override it entirely with `--env
UV_CACHE_DIR=...`; bootstrap never overrides either.

Both `gpuc host probe` and the bootstrap health check report the cache's size
and whether it shares a filesystem with gpuc home:

```
  uv_cache: /home/brendan/.cache/uv size 18G (same filesystem as gpuc home /home/brendan/.gpuc: yes)
```

`gpuc host clean <host> --uv-cache` runs `uv cache prune` there, which drops
unused and unreachable entries but keeps the wheels a venv still links to.

## What gets synced to the host

`gpuc submit` rsyncs `git ls-files --cached --others --exclude-standard`: every
tracked file **and** every untracked one git would keep. A file you have just
written and not yet `git add`ed is part of the experiment, and leaving it behind
used to mean running the wrong code on the GPU; `.gitignore` is still obeyed, so
`.venv/` and caches stay home. Files in the index but deleted on disk are
dropped rather than sent. One line says what happened:

```
syncing 43 files (2 modified, 1 untracked, ignoring .gitignore'd)
```

`uncommitted.patch` in the job dir is `git diff HEAD` taken against a copy of
the index with `git add -N` applied, so it contains your untracked files too and
your real staging area is never touched.

For a directory that is not a repository at all, `--no-git` rsyncs all of it
except `.venv`, `__pycache__`, `.git`, `*.pyc`, `node_modules` and `.uv-cache`,
with a warning — nothing reads `.gitignore` in that mode, and `gpuc requeue`
cannot rebuild that workdir from a commit.

## How a job is killed

A job's phases run inside a transient `systemd --user` scope wherever the host
has one (this desktop does; a RunPod pod and most shared boxes do not). That
matters for one reason: a grandchild that double-forks (`setsid`, `nohup`, a
daemonising server) escapes the job's process group and survives a
`kill -- -PGID`, holding a GPU the next job is about to be given — but it cannot
leave its cgroup, so `systemctl --user stop <unit>` reaps the whole tree.

`gpuc status` says which mode a host is in per job (`isolation: cgroup` or
`pgid` in `state.json`). Under `pgid` the daemonised-grandchild hole is real and
not fixed; `gpuc cancel` still stops everything in the job's process group.

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

## Who we say we are

Every request gpuc makes to somebody else's service carries one user agent,
from `gpuc/_version.py`:

```
gpuc/0.1.0 (+https://github.com/brendanlong/gpu-coordinator; self@brendanlong.com)
```

The RunPod API calls, a pod's self-terminate, the health check's download
measurement, bootstrap's `curl` for the uv installer, every boto3 client on this
machine, and `HF_HUB_USER_AGENT_ORIGIN` in each job's environment (so `hf`
sends it too; a job can override it in `env:`). The one thing that cannot carry
it is the **`aws` CLI on a host**: its User-Agent is not overridable, so S3
requests made by the CLI are anonymous. That is a limitation, not an oversight.

## Where state lives

On each host, `~/.gpuc/` — or `<persistent-root>/gpuc/` — (0700):
`config.json`, `queue/`, `jobs/<id>/` with
`spec.json`, `state.json`, `log.txt`, `workdir/` (deleted per `cleanup:`),
`outputs/`,
`secrets/<id>.env`, plus `dispatcher.lock`, `dispatcher.heartbeat` and
`dispatcher.log`. All writes are atomic.

Locally, `~/.config/gpu-coordinator/config.toml` for settings and
`~/.local/share/gpu-coordinator/` for state: `hosts.json`, `desired/<host>.json`
for pods we want alive, `jobs/` (the local job index), `known_hosts` plus
`known_hosts.d/<pod>` per pod, and `state.lock`. SSH ControlMaster sockets go
in `$XDG_RUNTIME_DIR/gpuc/` (or `/tmp/gpuc-<uid>/`), because a unix socket path
must fit in 108 bytes.

With `s3_bucket` set, specs go to `s3://<bucket>/gpuc/specs/<jobid>.json` and
the job index to `gpuc/index/`. Each job's `log.txt` and `state.json` are
mirrored by the *host*, which needs its own `--s3-prefix` (pods get one
derived from `s3_bucket`; a `local` or `ssh` host is told to set it).
`state.json` records `meta_synced_at`/`meta_synced_to` once that mirror is
confirmed and `outputs_synced_at` once the final output upload is, which is
what `gpuc clean --purge` requires before it deletes a job dir
([Retention](#retention-what-is-kept-and-what---purge-removes)).

## Troubleshooting

| symptom | what it means | what to do |
| --- | --- | --- |
| `status` says `dispatcher DOWN` | nothing is holding the host's lock, or its heartbeat is over 30 s old | `gpuc host bootstrap <host>` (idempotent); any `gpuc submit` also restarts it |
| provisioning gives up with "no direct SSH endpoint" | RunPod never exposed port 22 within the 15-minute ceiling — usually a bad placement | the pod was already terminated; re-run the submit, or widen `--gpu` / `--cloud any` |
| `ssh ... cannot create its ControlMaster socket` | the socket path is over the 108-byte unix limit | point `XDG_RUNTIME_DIR` at a short directory, or unset it to use `/tmp/gpuc-<uid>` |
| bootstrap fails with "host health failed" | the driver, disk or network check on the host said no | read the named check; fix the host (free disk, load the driver) and re-run bootstrap |
| job is `failed: low-util` | the GPU sat under `floor_pct` for `window_min` of phase `main` | raise `low_util.grace_min`, lower `floor_pct`, or set `low_util.enabled: false` for genuinely CPU-bound work |
| job is `failed: sync` (or `...+sync`) | the final upload failed; the run itself may have been fine | check the tail of `gpuc logs <jobid>`; the usual cause is missing `secrets:` for the destination, or no `aws`/`hf` on the host (re-run bootstrap) |
| job is `failed: sync-preflight` | the upload the job would do at the end cannot work: no `aws`/`hf`, a missing secret, a bucket or repo that is not writable | the log names the exact command and error; fix the credential, the destination, or add `hf_create: true`, then re-submit |
| job is `failed: ttl` | this host has an opt-in `--ttl-hours` cap and it ran out; the outputs were synced before the host went away | raise or drop the cap (`gpuc host set <host> --ttl-hours -1`), then `gpuc requeue <id>` |
| job is `failed: no-outputs` | the `outputs` path was never written, or everything in it came with the checkout | check the job actually wrote to that path, relative to the workdir |
| a host is out of disk, or `status` shows a `disk` line | finished jobs' workdirs (usually venvs) are still there | `gpuc clean --host <host> --all-finished`, and set `cleanup: always` on jobs you never need to inspect |
| `clean --purge` skips everything as "not backed up" | the host has no `s3_prefix`, so nothing is mirrored and deleting a job dir would lose its log | `gpuc host set <host> --s3-prefix s3://bucket/gpuc/<host>` + `gpuc host bootstrap`, or accept the loss with `--force` |
| `status` says a job's `outputs not uploaded` | the final upload of its `outputs:` failed, so the results exist only on that host | copy them off (`gpuc logs` shows the sync error), or `gpuc requeue <id>`; a purge will not remove it until they are confirmed |
| a job is `OUTPUTS LOST` | an ephemeral host drained, retried the upload three times and gave up before terminating | the results are gone; `gpuc requeue <id>` re-runs it, and fix the credential or bucket first |
| `--retention-days` never deletes anything | the dispatcher only lives while a non-ephemeral host has work, and it never purges an unmirrored job | check `gpuc status` for `not backed up`, and remember the sweep runs on the next submit |
| `uv sync` re-downloads torch on every job | uv's cache is on a different filesystem from gpuc home, so it copies instead of linking | `gpuc host bootstrap <host>` (it sets `UV_CACHE_DIR` for you), or pin one with `--cache-dir` |
| `status` says `POD GONE` | the pod is terminated or missing but the registry still lists it | `gpuc reconcile --once` |
| `reconcile` reports `DEAD DISPATCHER` and terminates a pod | its dispatcher stopped beating (or ssh stopped answering) for `dead_dispatcher_minutes` with nothing running | expected: that pod could no longer stop itself. Raise `dead_dispatcher_minutes` if your hosts go quiet legitimately |
| everything on a host is suddenly gone | the container restarted and `$HOME` was on the overlay | `gpuc host bootstrap <host>`, then `gpuc status --host <host> --all` and `gpuc requeue` what you still want ([runbook](#hosts-whose-home-directory-is-wiped-on-restart)) |
