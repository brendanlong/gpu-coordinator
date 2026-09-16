# Setup

Installing `gpuc`, telling it about a bucket and some hosts, and taking it all
down again. What to *do* with it afterwards is [usage.md](usage.md); the
contract the code keeps is [ARCHITECTURE.md](ARCHITECTURE.md).

## Prerequisites

**This machine**

- Python >= 3.11 and [uv](https://docs.astral.sh/uv/).
- `ssh` and `rsync` on the PATH. gpuc drives the system binaries; it does not
  read your `~/.ssh/config`, so everything it needs is in the registry.
- An ssh key that reaches your hosts. Point `ssh_key` in `config.toml` at the
  private key; its `.pub` is what gets uploaded to the RunPod account before the
  first pod is created. Unset means ssh picks its own key.
- `systemd --user`, only if you want `gpuc reconcile` on a timer.

**Each host**

- `ssh` and `rsync`, an account you can log into with that key, and outbound
  HTTPS (bootstrap fetches uv, a Python, the `aws` CLI and `hf`).
- The NVIDIA driver and `nvidia-smi` for any host that should run GPU work. A
  host without it is registered fine and can only run `gpus: 0` jobs.
- Nothing else installed by hand: `gpuc host bootstrap` puts uv, a Python
  (floor 3.11, it installs 3.12), the `gpuc.host` package, the `aws` CLI v2
  bundle and `hf` into `$HOME` over ssh, and starts the dispatcher.
- **systemd is optional and buys one thing**: with a `systemd --user` session
  that has cgroup delegation, each job phase runs in a transient scope, so
  stopping it reaps the whole tree — including a grandchild that double-forked
  out of the process group and would otherwise sit on a GPU. Without it (every
  RunPod pod, most shared boxes and containers) the kill is by process group and
  that hole is real. `gpuc host probe` reports which you get.

## Install

```sh
uv tool install "git+https://github.com/brendanlong/gpu-coordinator@main"
gpuc version
```

Or from a checkout, with no install at all: `uv run gpuc ...`.

`gpuc skill --install` drops the agent guide into a project.

## Settings

`gpuc` runs with no config file: every key has a working default, and without
`s3_bucket` there is simply no mirror. To write a commented one:

```sh
gpuc config init      # ~/.config/gpu-coordinator/config.toml (--force to overwrite)
gpuc config show      # the effective settings, file or not
```

| key | default | meaning |
| --- | --- | --- |
| `s3_bucket` | unset | the mirror's bucket. Unset means no mirror at all, so no `gpuc requeue` and no `gpuc logs` after a host is gone |
| `runpod_pod_prefix` | `"gpuc-"` | only pods whose name starts with this are ever read, reaped or terminated |
| `max_pods` | `3` | refuse to create a pod past this count (account-wide, every pod with the prefix) |
| `max_total_usd_per_hour` | `3.0` | the same for the summed hourly cost |
| `ssh_key` | unset | private key for ssh and rsync; its `.pub` goes to the RunPod account |
| `image` | `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` | default pod image (`--image` per submit) |
| `disk_gb` | `50` | default container disk (`--disk` per submit) |
| `dead_dispatcher_minutes` | `30.0` | how long an ephemeral host may be silent, with nothing running, before `gpuc reconcile` terminates it |

**`s3_bucket` and `--s3-prefix` are two different mirrors.** `s3_bucket` is
written by *this machine*: job specs to `s3://<bucket>/gpuc/specs/<job-id>.json`
and the job index to `s3://<bucket>/gpuc/index/`, which is what `gpuc requeue`
and `gpuc status --all` read. A host's `--s3-prefix` is written by *the host*:
each job's `log.txt` and `state.json` to `<prefix>/jobs/<job-id>/`, which is
what survives the host and what `gpuc clean --purge` requires before it deletes
anything. A pod from `gpuc submit --runpod` gets `s3://<bucket>/gpuc/<pod-name>`
derived automatically; a `local` or `ssh` host does **not** — set it by hand,
and only on a host whose jobs can authenticate to that bucket.

## Credentials

**This machine (boto3).** The S3 mirror uses boto3's default credential chain:
environment variables, `~/.aws/credentials`, an instance role — gpuc never reads
a key out of its own config. On the bucket it needs `s3:PutObject` and
`s3:GetObject` under `gpuc/*`, `s3:ListBucket` (for `gpuc status --all`, which
lists `gpuc/index/`), and, for `gpuc clean --purge --verify`, permission to HEAD
each mirrored `log.txt`.

**The hosts.** A host uploads with whatever the *job* carries: list the names in
the spec's `secrets:` and they are read from your shell at submit time and
delivered to the host as `~/.gpuc/secrets/<job-id>.env`, mode 0600, never in
argv and never in a pod's provider-visible environment. `secrets:
[AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY]` is all an S3 output — or the host's
own `--s3-prefix` mirror — needs. A provisioned pod is additionally given
`~/.aws/credentials` (0600) built from `AWS_ACCESS_KEY_ID`,
`AWS_SECRET_ACCESS_KEY` and `AWS_SESSION_TOKEN` in your shell when the host has
an `s3_prefix`, with the region from `AWS_REGION`, `AWS_DEFAULT_REGION`, else
`us-east-1`.

**RunPod.** Export `RUNPOD_API_KEY`. `gpuc submit --runpod`, `gpuc pods` and
`gpuc reconcile` check it first and exit 1 with one line if it is missing
(`gpuc reconcile --install`, which only writes unit files, does not). The key is
delivered to each pod as `~/.gpuc/secrets/runpod` so it can terminate itself.

**Hugging Face.** Put `HF_TOKEN` (or `HUGGING_FACE_HUB_TOKEN`) in the job's
`secrets:`. The sync preflight runs `hf auth whoami` with it and fails the job
in seconds if it cannot write the repo.

## Registering hosts

```sh
gpuc host add local --gpus 0                               # this machine
gpuc host add gpubox --ssh me@gpubox --port 22 --gpus 2,3  # a box you reach over ssh
gpuc host probe gpubox       # driver, the cards assigned to this host as `[index] uuid name`,
                             # disk, $HOME's filesystem, systemd --user, uv cache, network speed.
                             # Needs the host registered, so add it first (with no --gpus if you do
                             # not know them yet) and `gpuc host set gpubox --gpus …` once you can
                             # read them off
gpuc host probe gpubox --all-gpus   # every card in the box, `(assigned)` on the ones this host owns
gpuc host bootstrap gpubox   # idempotent; run it again after any `host set`
```

RunPod hosts are never added by hand: `gpuc submit --runpod ...` creates the
pod, registers it as kind `runpod`, and bootstraps it
([usage.md](usage.md#runpod)).

| flag (`host add`, and `host set` to change one) | default | meaning |
| --- | --- | --- |
| `--ssh user@host` / `--port N` | this machine / `22` | omit `--ssh` for a `local` host |
| `--gpus 2,3` or `--gpus GPU-8064…,3` | none | what this host may use: nvidia-smi **indices**, UUIDs, or a mix, stored exactly as typed. Indices are how a share of a shared box is agreed; the host re-resolves them to UUIDs on every dispatch pass and pins jobs with `CUDA_VISIBLE_DEVICES=<uuid>`, so a renumbered driver cannot hand your job somebody else's card. An owned card the host cannot see is reported `UNAVAILABLE` and jobs wait for it |
| `--gpuc-home PATH` | `$HOME/.gpuc` | override where gpuc home lives on the host |
| `--cache-dir PATH` | bootstrap decides | uv's cache for this host (`UV_CACHE_DIR`). Bootstrap sets one on gpuc home's filesystem when they differ, because uv only reflinks or hardlinks a venv out of its cache within one filesystem; an explicit `--env UV_CACHE_DIR=…` always wins |
| `--persistent-root R` | none | gpuc home moves to `R/gpuc` (below) |
| `--env K=V` (repeatable) | none | extra environment for every job on this host, applied *before* the job's own `env:`. Nothing populates it automatically |
| `--s3-prefix s3://…` | none | this host's own log/state mirror |
| `--retention-days N` | none | the host's dispatcher auto-purges job dirs older than this, but only ones whose log and state it has confirmed mirrored — so with no `--s3-prefix` it deletes nothing. `''` goes back to keeping everything |
| `--idle-min N` | `15` | how long an ephemeral host may sit with an empty queue before terminating itself. **Inert on `local` and `ssh` hosts**, which never terminate themselves |
| `--ttl-hours N` | none | opt-in hard cap on the host's life; past it the dispatcher kills the running job with reason `ttl`, syncs, and terminates. `-1` means no TTL, on `host add` and `host set` alike (a stored `-1` would be a host already past its TTL). `0` is refused |

On a box you share, `--gpus` is the whole of what gpuc may touch, so `host
probe` lists only those cards and says how many it hid (`2 of 8 assigned to
gpubox`); `--all-gpus` shows the box as nvidia-smi sees it. Either way the probe
records **every** card's name and VRAM, so `gpuc host set gpubox --gpus 5` names
something the registry already knows. An assigned entry no card answers to is
called out: jobs needing it would queue forever.

`gpuc host set` edits one entry in place — only the flags you pass — instead of
`remove` + `add`, which would drop everything else about the host. **Nothing on
the host changes until the next `gpuc host bootstrap <name>`**, which is what
rewrites its `config.json`. `gpuc host list` shows what is registered, with each
card as `[index] name vram uuid` once the host has been probed or bootstrapped.

### Hosts whose `$HOME` is wiped on restart

A container-backed host (a Kubernetes pod, most cloud notebooks) usually has
`$HOME` on the image's throwaway upper layer: `gpuc host probe` reports
`home_fs: overlay` and says so. `--persistent-root R` moves **gpuc home and only
gpuc home** to `R/gpuc`: `config.json`, `queue/` and every `jobs/<id>/` with its
spec, state, log and workdir — the things that cannot be reinstalled. uv, its
Pythons and the `aws` bundle stay in `$HOME`, because bootstrap puts them back
in seconds and these shared volumes are much slower than local disk. `R` is
created 0700 if gpuc creates it; an existing `R` keeps its mode.

```sh
gpuc host add gpubox --ssh gpubox --persistent-root /mnt/ssd-2/$USER --gpus GPU-aaa,GPU-bbb
gpuc host set gpubox --persistent-root /mnt/ssd-2/$USER      # or move an existing host
gpuc host bootstrap gpubox
```

**Runbook: the host restarted and came back empty.** The symptom is
`dispatcher DOWN` in `gpuc status`, or ssh failing outright.

1. If your key or `authorized_keys` lived in the wiped home, put it back, and
   check `gpuc host probe <host>` answers at all.
2. `gpuc host bootstrap <host>` — idempotent, and the whole of the host-side
   recovery.
3. With a `--persistent-root`, stop here: the queue came back with the volume,
   so queued jobs start again as soon as the dispatcher does, and only jobs that
   were *running* need resubmitting (their runner is gone, so the dispatcher
   fails them on its next start).
4. Without one, find what was on it and resubmit:

   ```sh
   gpuc status --host <host> --all      # what the job index says was there
   gpuc requeue <job-id> --host <host>  # one line per job you still want
   ```

   `requeue` re-reads the spec from the S3 mirror, so this needs `s3_bucket`
   set; without one, submit the job file again by hand.

## The reconcile timer

`gpuc reconcile` is the only thing that terminates pods nothing wants any more,
and it only runs when something runs it. Install it as a `systemd --user` timer
(60 s by default, `--interval` to change it):

```sh
gpuc reconcile --install     # writes the units; deliberately does not enable them
install -m 600 /dev/null ~/.config/gpu-coordinator/env
echo RUNPOD_API_KEY=... >> ~/.config/gpu-coordinator/env
systemctl --user daemon-reload
systemctl --user enable --now gpuc-reconcile.timer
systemctl --user list-timers gpuc-reconcile.timer
journalctl --user -u gpuc-reconcile.service -f
```

The service runs `gpuc reconcile --once` with `GPUC_CONFIG_DIR` and
`GPUC_STATE_DIR` pinned to this user's directories and reads `RUNPOD_API_KEY`
from that env file, which `--install` does not create. What it terminates, and
what it refuses to touch, is in [usage.md](usage.md#reconcile).

## Upgrading

A host's package is a *copy*, not a link, so upgrading here does not upgrade it.

```sh
uv tool upgrade gpu-coordinator
uv tool install --reinstall "git+https://github.com/brendanlong/gpu-coordinator@<commit>"   # or pin
gpuc version                 # this build's commit, and each host's
gpuc host bootstrap <host>   # for every host `version` marks OLDER
gpuc host bootstrap --all    # or all of them, in one command
```

`--all` takes every registered host in turn, including ephemeral ones. A host
that fails does not stop the others — a pod that has already gone away is the
ordinary case, and `gpuc reconcile --once` is what forgets it — so the run ends
with a tally naming each failure and exits 1, while the hosts that did upgrade
stay upgraded. The tally also counts any host entry this build could not read
(skipped with a warning), because that host was not upgraded either.

`gpuc submit` and `gpuc requeue` do this themselves when the host they are about
to enqueue on is not on this commit — including a host with no commit recorded,
which means it was bootstrapped by a build old enough not to write one. They
re-sync the package and restart the dispatcher first, print one line saying so,
and `--no-bootstrap` skips it. Re-bootstrapping is safe at any time: **running
jobs are not disturbed and do not block it.** A dispatcher that is already alive
keeps the lock and finishes on its own (older) code; every new runner uses the
new package, and whichever dispatcher takes over adopts the running jobs from
their `state.json`. Only the dispatcher is ever replaced, never a runner.

Before you push a change: `./check.sh` runs ruff, pyright and the test suite,
which is exactly what CI runs on every pull request (`--fast` skips the sync).

Two sessions on different builds are fine as long as both are recent: every file
the two sides share is read with unknown keys ignored and a `null` for a
non-optional field taken as that field's default.

## Teardown

```sh
gpuc host remove <name>        # forgets it locally; nothing on the host changes
gpuc host clean <name> --uv-cache   # `uv cache prune` there, if you want the disk back first
```

`host remove` does not stop anything: to clear a host's own state, delete its
gpuc home (`gpuc ssh <host> -- rm -rf ~/.gpuc`, or the persistent root's `gpuc`
directory) while nothing is running.

**Terminating a pod deliberately.** There is no terminate command; the pod
terminates itself once its queue has been empty for `--idle-min`, which is the
normal path. To do it now, remove the record that says you still want it and let
the reaper treat it as a stray:

```sh
gpuc pods                                              # confirm the name and that it is idle
rm ~/.local/share/gpu-coordinator/desired/<name>.json
gpuc host remove <name>
gpuc reconcile --once                                  # terminates prefixed pods with no record
```

The stray rule only fires once the pod is over 15 minutes old, so a pod created
minutes ago has to age out (or be terminated in the RunPod console). Leaving the
`desired/` record in place and only removing the host has the same end effect
more slowly: the reaper can no longer reach it, so it terminates it after
`dead_dispatcher_minutes`.

**Disabling the timer.**

```sh
systemctl --user disable --now gpuc-reconcile.timer
rm ~/.config/systemd/user/gpuc-reconcile.{timer,service}
systemctl --user daemon-reload
```

With the timer off, nothing reaps a leaked pod: run `gpuc reconcile --once` by
hand, and check `gpuc pods` before you walk away.
