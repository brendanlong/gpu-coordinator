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
- `systemd --user`, only if you want `gpuc reconcile` on a timer or the web
  dashboard as a service.

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

## The web dashboard

```sh
gpuc web set-password        # prompts twice; writes a bcrypt hash to ~/.config/gpu-coordinator/web-password (0600)
gpuc web serve               # http://127.0.0.1:8646/ until Ctrl-C
```

One password guards every page and API call. `--bind 0.0.0.0` makes it
reachable from other machines; there is no TLS, so do that only on a VPN
interface or behind a proxy that terminates TLS. What it shows and does is in
[usage.md](usage.md#the-web-dashboard).

To keep it running, `--install` writes a `systemd --user` service that serves
with the same `--bind` and `--port`, and deliberately does not enable it:

```sh
gpuc web serve --bind 0.0.0.0 --port 8646 --install
systemctl --user daemon-reload
systemctl --user enable --now gpuc-web.service
journalctl --user -u gpuc-web.service -f
```

The unit pins `GPUC_CONFIG_DIR` and `GPUC_STATE_DIR` to this user's
directories and reads `RUNPOD_API_KEY` from the same `config_dir()/env` file
the [reconcile timer](#the-reconcile-timer) uses, so RunPod hosts show their
pod line; without it they still render. It restarts on failure, and it needs
`loginctl enable-linger` to outlive your session, exactly like the timer.
`--install` refuses nothing: with no password set the service starts, logs
the `gpuc web set-password` line and exits, systemd retries it five times
over five minutes and then leaves it `failed`, and `--install` says so.
Disabling it again is `systemctl --user disable --now gpuc-web.service` and
removing the unit file, exactly as for the timer below.

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
(`gpuc reconcile --install` and `gpuc web serve --install`, which only write
unit files, do not). The key is
delivered to each pod as `~/.gpuc/secrets/runpod` so it can terminate itself.

**Hugging Face.** Put `HF_TOKEN` (or `HUGGING_FACE_HUB_TOKEN`) in the job's
`secrets:`. The sync preflight runs `hf auth whoami` with it and fails the job
in seconds if it cannot write the repo.

## Registering hosts

```sh
gpuc host add local --gpus 0                               # this machine
gpuc host add gpubox --ssh me@gpubox --port 22 --gpus 2,3  # a box you reach over ssh
gpuc host add gpubox --ssh me@gpubox                       # …one somebody already set up: adopt it
gpuc host probe gpubox       # driver, the cards assigned to this host as `[index] uuid name`,
                             # disk, $HOME's filesystem, systemd --user, uv cache, network speed
gpuc host probe gpubox --all-gpus   # every card in the box, `(assigned)` on the ones this host owns
gpuc host bootstrap gpubox   # installs uv, the package and the dispatcher; idempotent
```

**`host add` is a connect.** It opens the host, probes it, and reads
`~/.gpuc/config.json`:

- the host **has** a config — it was set up from another machine, or from this
  one before — and that config is adopted as it stands. Only the address
  (`--ssh`, `--port`, `--gpuc-home` / `--persistent-root`) is recorded here.
  Flags you pass are explicit overrides, written through to the host and
  reported field by field (`host <- retention_days 30.0 -> 7.0`). A `--gpus`
  that claims *some* of the cards the host already has is refused rather than
  warned about, because that is the one difference that can hand one card to
  two jobs — in whichever spelling, since `--gpus 0` and `--gpus GPU-…` can be
  the same card; a disjoint list is a reassignment and goes through, and
  `--force` overrides the refusal. The host is registered under the name it
  calls itself, unless a *different* host is already registered here under that
  name, which is refused rather than replaced.
- the host has **no** config — nothing has been set up there yet — so this is
  where one is written, and `--gpus` is required (`--gpus ''` for a host whose
  cards gpuc may not use). The probe's card list is printed if you leave it out.

So a host is registered by asking it what it is, and a second machine
connecting to a box the first one set up is the ordinary path. Such a host is
usable from that machine at once — `gpuc status`, `gpuc host set`, `gpuc logs`
— because the probe records an interpreter to run the on-host package with;
`gpuc host bootstrap` is still what ships *this* build's package to it.

A `config.json` that is there but does not parse stops all of this: it is a
file the host is running on, so nothing replaces it, and the error says to fix
or delete it.

RunPod pods are never *created* by hand: `gpuc submit --runpod ...` creates the
pod, registers it as kind `runpod`, and bootstraps it
([usage.md](usage.md#runpod)). A pod that already exists is adopted the same way
any other host is, from any machine holding the API key:

```sh
gpuc pods                                  # the account's pods, with their ids
gpuc host add rented --pod <pod-id>        # its address from the provider, its config from the pod
```

That is what makes a pod the laptop queued usable from the desktop: the pod owns
its `config.json` — cards, mirror, TTL, and the record of what it was rented as —
so nothing about the machine that created it matters afterwards. It is also
recorded in `desired/` here, so this machine's `gpuc reconcile` watches it.

The address is the top two rows; every other flag is the host's own config,
which `host set` writes through to it.

| flag (`host add`, and `host set` to change one) | default | meaning |
| --- | --- | --- |
| `--ssh user@host` / `--port N` | this machine / `22` | omit `--ssh` for a `local` host |
| `--pod POD_ID` (`host add`) | none | adopt a pod the account is renting instead of naming an ssh target; the provider says where it is. Needs `RUNPOD_API_KEY` |
| `--gpus 2,3` or `--gpus GPU-8064…,3` | none | what this host may use: nvidia-smi **indices**, UUIDs, or a mix, stored exactly as typed. Indices are how a share of a shared box is agreed; the host re-resolves them to UUIDs on every dispatch pass and pins jobs with `CUDA_VISIBLE_DEVICES=<uuid>`, so a renumbered driver cannot hand your job somebody else's card. An owned card the host cannot see is reported `UNAVAILABLE` and jobs wait for it |
| `--gpuc-home PATH` | `$HOME/.gpuc` | override where gpuc home lives on the host |
| `--cache-dir PATH` | bootstrap decides | uv's cache for this host, which is `UV_CACHE_DIR` in its `env`. Bootstrap sets one on gpuc home's filesystem when they differ, because uv only reflinks or hardlinks a venv out of its cache within one filesystem — but only when the host's config names none, however it got there |
| `--persistent-root R` | none | gpuc home moves to `R/gpuc` (below) |
| `--env K=V` (repeatable) | none | extra environment for every job on this host, applied *before* the job's own `env:`. Nothing populates it automatically. It replaces the whole set, except `UV_CACHE_DIR`, which is bootstrap's and `--cache-dir`'s |
| `--s3-prefix s3://…` | none | this host's own log/state mirror |
| `--retention-days N` | none | the host's dispatcher auto-purges job dirs older than this, but only ones whose log and state it has confirmed mirrored — so with no `--s3-prefix` it deletes nothing. `''` goes back to keeping everything |
| `--workdir-days N` | `1` on a host being configured for the first time | the host's dispatcher reclaims a finished job's `workdir/` — the checkout and the venv, never its log or state — once it ended this long ago. No mirror needed: `gpuc requeue` rebuilds a workdir from git, so this is the horizon worth having short. `''` keeps workdirs until you run `gpuc clean`. A host whose config already exists keeps whatever it says, including nothing |
| `--idle-min N` | `15` | how long an ephemeral host may sit with an empty queue before terminating itself. **Inert on `local` and `ssh` hosts**, which never terminate themselves |
| `--ttl-hours N` | none | opt-in hard cap on the host's life; past it the dispatcher kills the running job with reason `ttl`, syncs, and terminates. `-1` means no TTL, on `host add` and `host set` alike (a stored `-1` would be a host already past its TTL). `0` is refused |

On a box you share, `--gpus` is the whole of what gpuc may touch, so `host
probe` lists only those cards and says how many it hid (`2 of 8 assigned to
gpubox`); `--all-gpus` shows the box as nvidia-smi sees it. Either way the probe
records **every** card's name and VRAM, so `gpuc host set gpubox --gpus 5` names
something already known. An assigned entry no card answers to is
called out, as are two entries naming one card: `gpuc host bootstrap` fails its
`gpu_uuids` check on both, so the probe is where you want to find them.

`gpuc host set` changes one field at a time, and where it writes depends on
which field: `--gpus`, `--env`, `--cache-dir`, `--s3-prefix`,
`--retention-days`, `--workdir-days`, `--idle-min` and `--ttl-hours` are the **host's own**
config, so they are written through to its `config.json` immediately — the host
has to answer, and every change is reported as `host <- …`. `--persistent-root`
and `--gpuc-home` are *addresses*, kept here (`here <- …`) and applied to the
host by the next `gpuc host bootstrap`.

`gpuc host list` shows what is registered, one block per host, with each card
as `gpu [index] name vram uuid` and a `pkg` line naming the commit the host was
running **when it was last read** (`as of 3m ago`). It never asks a host
anything: everything but the address is a cache of the last answer, which is
why it is labelled with its age and why `gpuc status` is what asks. `gpuc host
probe` refreshes that cache (and nothing else). The interpreter path bootstrap
chose is in `gpuc host list --json` and `gpuc host probe`.

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
   recovery. A wiped home takes `config.json` with it, so this is the one case
   where bootstrap writes one: the last config this machine read off that host
   is restored, which is why the cache is kept.
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

`gpuc reconcile` is the safety net for the states a pod cannot get itself out
of -- it never bootstrapped, its dispatcher died, or it is past a TTL its own
config carries -- and it only runs when something runs it. (A healthy pod needs
none of this: it drains and terminates itself once its queue has been empty for
`--idle-min`.) Install it as a `systemd --user` timer
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

**On more than one machine is fine.** Each pass asks every pod with your prefix
what it is, and a pod holding a gpuc config is left alone whichever machine
created it — so the desktop can watch the pod the laptop queued, with the laptop
shut. To *watch* a pod (rather than only report it) that machine needs an ssh
key the pod accepts, and RunPod injects the account's keys when the pod is
**created**: a key you register later is not on a pod that already exists. So
put both machines' keys on the account before you provision, or accept that each
pod is watched from the machines whose keys it was born with. Nothing is lost
either way — a pod this machine cannot place is reported every pass and never
terminated. To drive a pod as well as watch it, adopt it: `gpuc host add <name>
--pod <pod-id>`.

## Upgrading

A host's package is a *copy*, not a link, so upgrading here does not upgrade it.

```sh
uv tool upgrade gpu-coordinator
uv tool install --reinstall "git+https://github.com/brendanlong/gpu-coordinator@<commit>"   # or pin
gpuc version                 # this build's commit, and what each host was last given from here
gpuc host bootstrap <host>   # for every host `version` marks DIFFERS
gpuc host bootstrap --all    # or all of them, in one command
```

`gpuc version` and `gpuc host list` never ssh, so what they show is the commit
the host was running when this machine last read it, labelled with its age.
`gpuc status` asks each host what it is running now, and that is the answer
that counts.

`--all` takes every registered host in turn, including ephemeral ones. A host
that fails does not stop the others — a pod that has already gone away is the
ordinary case, and `gpuc reconcile --once` is what forgets it — so the run ends
with a tally naming each failure and exits 1, while the hosts that did upgrade
stay upgraded. The tally also counts any host entry this build could not read
(skipped with a warning), because that host was not upgraded either.

A host nothing has ever installed gpuc on -- registered with `gpuc host add`
and not bootstrapped, by this machine or any other -- is refused by `gpuc
submit` rather than half-installed on the way past: shipping the package to it
would start a dispatcher with no uv under it, and the job would fail there
instead of here.

`gpuc submit` and `gpuc requeue` do this themselves when the host they are about
to enqueue on is not on this commit — read from the host's own `config.json`,
which is also the read that tells them what the host's cards and mirror are,
and including a host with no commit recorded, which means it was bootstrapped
by a build old enough not to write one. They re-sync
the package and restart the dispatcher first, print one line saying so, and
`--no-bootstrap` skips it. Re-bootstrapping is safe at any time: **running
jobs are not disturbed and do not block it.** A dispatcher that is already alive
keeps the lock and finishes on its own (older) code; every new runner uses the
new package, and whichever dispatcher takes over adopts the running jobs from
their `state.json`. Only the dispatcher is ever replaced, never a runner.

Before you push a change: `./check.sh` runs ruff, pyright and the test suite,
which is exactly what CI runs on every pull request (`--fast` skips the sync).

Two sessions on different builds are fine as long as both are recent: every file
the two sides share is read with unknown keys ignored and a `null` for a
non-optional field taken as that field's default.

### The same host from two machines

A host is the host's own: its queue, its job state and its logs live there, so
registering one box from a desktop *and* a laptop works — each machine's
`gpuc status`, `logs` and `cancel` see every job on it, whoever submitted it,
and the dispatcher orders them all by priority as usual.

What the host **is** — its cards, its mirror, its env, its timers — lives in
`config.json` on the host and nowhere else, so there is nothing to keep in
step: on the second machine,

```sh
gpuc host add gpubox --ssh me@gpubox     # reads what the host already says it is
```

and that is all. `gpuc host set` on either machine writes through to the same
file; `gpuc submit` reads it before it enqueues, so the cards a job is judged
against are always the host's own answer.

What is per-machine is the **address**: `--ssh`, `--port`, and `--gpuc-home` /
`--persistent-root`, which is how this machine reaches the host and finds that
config. Get the last two wrong and you have pointed at a second, empty gpuc
home on the same box rather than at the host — the one thing worth copying from
`gpuc host list --json` on the first machine.

Everything else the registry holds is a **cache** of what the host last said,
kept so `gpuc host list` and `gpuc version` have something to print offline.
They label it with its age (`as of 3m ago`); anything that decides something
reads the host.

The exception is the reconcile timer: until [#36](https://github.com/brendanlong/gpu-coordinator/issues/36)
lands, running `gpuc reconcile` on a second machine can terminate a pod the
first one created, because "is this pod ours" is still answered from local
state only.

## Teardown

```sh
gpuc host remove <name>        # forgets it locally; nothing on the host changes
gpuc host clean <name> --uv-cache   # `uv cache prune` there, if you want the disk back first
```

`host remove` does not stop anything: to clear a host's own state, delete its
gpuc home (`gpuc ssh <host> -- rm -rf ~/.gpuc`, or the persistent root's `gpuc`
directory) while nothing is running.

**Terminating a pod deliberately.** There is no terminate command, and deleting
local state is not one: nothing here terminates a pod for having no record. Tell
the pod instead — it is the thing that can stop itself:

```sh
gpuc pods                                       # confirm the name and that it is idle
gpuc host set <name> --idle-min 0               # stop as soon as the queue is empty
gpuc host set <name> --ttl-hours 0.1            # or: stop in six minutes, killing a running job
```

`--idle-min 0` reaches the host's own `config.json`, so the dispatcher drains
(retrying unconfirmed outputs, mirroring every job's log and state) and
terminates on its next pass with nothing running. `--ttl-hours` is the one that
does not wait for the job — it asks each runner to stop with reason `ttl`, lets
it sync, and then terminates. For a pod that has stopped answering ssh
altogether, the reaper gets it after `dead_dispatcher_minutes`; for one that
answers nothing at all and belongs to nobody, the RunPod console is the tool.

**Disabling the timer, or the dashboard service.**

```sh
systemctl --user disable --now gpuc-reconcile.timer
rm ~/.config/systemd/user/gpuc-reconcile.{timer,service}
systemctl --user disable --now gpuc-web.service      # if you installed the dashboard
rm ~/.config/systemd/user/gpuc-web.service
systemctl --user daemon-reload
```

With the timer off, nothing reaps a leaked pod: run `gpuc reconcile --once` by
hand, and check `gpuc pods` before you walk away.
