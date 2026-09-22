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
- `systemd --user`, only if you want the web dashboard as a service.

**Each host**

- `ssh` and `rsync`, an account you can log into with that key, and outbound
  HTTPS (bootstrap fetches uv, a Python, the `aws` CLI and `hf`).
- The NVIDIA driver and `nvidia-smi`. A host without them can be registered
  and probed, but every job needs a GPU, so nothing can be submitted to it.
- Nothing else installed by hand: `gpuc host bootstrap` puts uv, a Python
  (floor 3.11, it installs 3.12), the `gpuc.host` package, the `aws` CLI v2
  bundle and `hf` into `$HOME` over ssh, and starts the dispatcher.
- **systemd is optional**: with a `systemd --user` session a cancel reaps
  the whole process tree, without one a double-forked grandchild can escape
  (see [how a job is killed](usage.md#how-a-job-is-killed)). `gpuc host probe`
  reports which you get.

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
| `runpod_pod_prefix` | `"gpuc-"` | only pods whose name starts with this are ever read or terminated |
| `ssh_key` | unset | private key for ssh and rsync; its `.pub` goes to the RunPod account |
| `image` | `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` | default pod image (`--image` per submit) |
| `disk_gb` | `50` | default container disk (`--disk` per submit) |

Nothing limits how many pods an account runs or what they cost; `gpuc pods`
shows what is billing.

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
directories and reads `RUNPOD_API_KEY` from `~/.config/gpu-coordinator/env`
if that file exists, so RunPod hosts show their pod line; without it they
still render. Create it as
`install -m 600 /dev/null ~/.config/gpu-coordinator/env` and add one
`RUNPOD_API_KEY=...` line; `--install` does not create it. The service
restarts on failure, and it needs `loginctl enable-linger` to outlive your
session. `--install` refuses nothing: with no password set the service
starts, logs the `gpuc web set-password` line and exits, systemd retries it
five times over five minutes and then leaves it `failed`, and `--install`
says so. Disabling it again is under [teardown](#teardown) below.

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
`gpuc host add --pod` check it first and exit 1 with one line if it is
missing. The key is delivered to each pod as `~/.gpuc/secrets/runpod` so it
can terminate itself.

**Hugging Face.** Put `HF_TOKEN` (or `HUGGING_FACE_HUB_TOKEN`) in the job's
`secrets:`. The sync preflight runs `hf auth whoami` with it and fails the job
in seconds if it cannot write the repo.

## Registering hosts

```sh
gpuc host add local                                        # this machine, every card it has
gpuc host add gpubox --ssh me@gpubox --port 22 --gpus 2,3  # a box you reach over ssh, two of its cards
gpuc host add gpubox --ssh me@gpubox                       # …one somebody already set up: adopt it
gpuc host set gpubox --shared-gpus 4,5                     # two more it may borrow while nobody else is on them
gpuc host probe gpubox       # driver, the cards assigned to this host as `[index] name vram uuid`,
                             # disk, $HOME's filesystem, systemd --user, uv cache, network speed
gpuc host probe gpubox --all-gpus   # every card in the box, `(assigned)` on the ones this host owns
gpuc host bootstrap gpubox   # installs uv, the package and the dispatcher; idempotent
```

**`host add` is a connect.** It opens the host, probes it, and reads
`~/.gpuc/config.json`:

- the host **has** a config — set up from another machine, or from this one
  before — and it is adopted as it stands; only the address (`--ssh`, `--port`,
  `--gpuc-home` / `--persistent-root`) is recorded here. Flags you pass are
  overrides, written through to the host and reported field by field
  (`host <- retention_days 30.0 -> 7.0`). A `--gpus` that claims *some* of the
  cards the host already has is refused (that is the one difference that can
  hand a card to two jobs, and `--gpus 0` and `--gpus GPU-…` may be the same
  card); a disjoint list is a reassignment and goes through, and `--force`
  overrides. The host is registered under the name it calls itself, unless a
  *different* host already has that name here.
- the host has **no** config — nothing has been set up there yet — so this is
  where one is written, and by default it **owns every card** nvidia-smi
  reports there, by UUID. `--gpus` narrows that to the cards named (`--gpus ''`
  for a host whose cards gpuc may not use), and `--shared-gpus` takes its cards
  out of the owned set, so `--shared-gpus 1` alone owns everything but card 1.
  A host with no nvidia-smi, or nothing left after those flags, is registered
  owning nothing, and `host add` says that nothing can be submitted to it until
  `gpuc host set <name> --gpus <list>` assigns some. Only this path has the
  default: an omitted `--gpus` on a host that already has a config keeps what
  the host has (above), never resets it to every card. A host that reports no
  cards at all (no nvidia-smi, or a driver still coming up) is refused rather
  than given "owns nothing" as its first config; `--gpus ''` says so on
  purpose.

A second machine connecting to a box the first one set up is the ordinary path,
and such a host is usable from it at once — `gpuc status`, `gpuc host set`,
`gpuc logs`. `gpuc host bootstrap` is what ships *this* build's package to it.

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

The pod owns its `config.json` — cards, mirror, idle timer, and what it was
rented as — so nothing about the machine that created it matters afterwards. A
pod nobody has bootstrapped has no dispatcher and so will never end itself;
`host add` says so, and `gpuc host bootstrap <name>` gives it one.

The address is the top two rows, kept here (`here <- …`) and applied to the
host by the next `gpuc host bootstrap`. Every other flag is the host's own
config: `host set` writes it through to the host's `config.json` at once (the
host has to answer) and reports each change as `host <- …`.

| flag (`host add`, and `host set` to change one) | default | meaning |
| --- | --- | --- |
| `--ssh user@host` / `--port N` | this machine / `22` | omit `--ssh` for a `local` host |
| `--pod POD_ID` (`host add`) | none | adopt a pod the account is renting instead of naming an ssh target; the provider says where it is. Needs `RUNPOD_API_KEY`. Add `--gpuc-home` if that pod keeps gpuc somewhere other than `$HOME/.gpuc` |
| `--gpus 2,3` or `--gpus GPU-8064…,3` | every card nvidia-smi reports, on a host with no config; what the host has, on one that does | nvidia-smi **indices**, UUIDs, or a mix, stored as typed; the host re-resolves indices to UUIDs on every dispatch pass, so a renumbered driver cannot hand your job somebody else's card. An owned card the host cannot see is `UNAVAILABLE` and jobs wait for it |
| `--shared-gpus 4,5` | none | cards gpuc may **borrow** but does not own, spelled like `--gpus` and never overlapping it; see [shared GPUs](usage.md#shared-gpus) |
| `--gpuc-home PATH` | `$HOME/.gpuc` | override where gpuc home lives on the host |
| `--cache-dir PATH` | bootstrap decides | `UV_CACHE_DIR` in the host's `env`. Bootstrap sets one on gpuc home's filesystem when they differ (uv only links a venv out of its cache within one filesystem), and never overrides one the config already names |
| `--persistent-root R` | none | gpuc home moves to `R/gpuc` (below) |
| `--env K=V` (repeatable) | none | extra environment for every job on this host, applied *before* the job's own `env:`. Nothing populates it automatically. It replaces the whole set, except `UV_CACHE_DIR`, which is bootstrap's and `--cache-dir`'s |
| `--s3-prefix s3://…` | none | this host's own log/state mirror |
| `--retention-days N` | none | auto-purge whole job dirs this old, only ones whose log and state are confirmed mirrored — so with no `--s3-prefix` it deletes nothing. `''` turns it off |
| `--workdir-days N` | `1` on a host being configured for the first time | auto-sweep a finished job's `workdir/` (never its log or state) once it ended this long ago; no mirror needed. `''` turns it off. A host whose config already exists keeps whatever it says |
| `--idle-min N` | `15` | how long an ephemeral host may sit with an empty queue before terminating itself. **Inert on `local` and `ssh` hosts**, which never terminate themselves |

On a box you share, pass `--gpus`: it is the whole of what gpuc may touch, so
`host probe` lists only those cards and says how many it hid (`2 of 8 assigned to
gpubox`); `--all-gpus` shows the box as nvidia-smi sees it. Either way the probe
records **every** card's name and VRAM, so `gpuc host set gpubox --gpus 5` names
something already known. An assigned entry no card answers to is
called out, as are two entries naming one card: `gpuc host bootstrap` fails its
`gpu_uuids` check on both, so the probe is where you want to find them.

`gpuc host list` shows what is registered, one block per host, with each card as
`gpu [index] name vram uuid` and a `pkg` line naming the commit the host was
running **when it was last read** (`as of 3m ago`). It never asks a host
anything: everything but the address is a cache of the last answer, labelled
with its age. `gpuc status` is what asks. `gpuc host probe` refreshes that
cache (and nothing else). The interpreter path bootstrap chose is in
`gpuc host list --json` and `gpuc host probe`.

### Hosts whose `$HOME` is wiped on restart

A container-backed host (a Kubernetes pod, most cloud notebooks) usually has
`$HOME` on the image's throwaway upper layer, which `gpuc host probe` reports as
`home_fs: overlay`. That is not by itself a problem to fix. A host's queue is
expected to be disposable — S3 is the mirror and `gpuc requeue` the recovery —
and moving gpuc home puts every job's venv and workdir on a network volume,
which drags uv's cache there too (the `--cache-dir` row above), since a cache on
the other filesystem is copied into every venv rather than linked. Where that
trade is worth it, `--persistent-root R` moves **gpuc home and only
gpuc home** to `R/gpuc`: `config.json`, `queue/` and every `jobs/<id>/` with its
spec, state, log and workdir — the things that cannot be reinstalled. uv, its
Pythons and the `aws` bundle stay in `$HOME`. `R` is created 0700 if gpuc
creates it; an existing `R` keeps its mode.

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

Shipping the package is only half an upgrade: a dispatcher imports its code
once, so the dispatcher bootstrap starts also takes over from the incumbent,
adopting the running jobs — nothing is interrupted. `gpuc status` reports the
*running* dispatcher's commit alongside the package's and warns if they come
apart.

`--all` takes every registered host in turn, including ephemeral ones. A host
that fails does not stop the others: the run ends with a tally naming each
failure, including host entries this build could not read, and exits 1, while
the hosts that did upgrade stay upgraded. A rental the provider no longer has is
not a failure -- it ended itself, so the entry is forgotten and the tally says
so. `--json` prints that tally as one entry per host
([usage.md](usage.md#--json-everywhere-else)).

A host nothing has ever bootstrapped is refused by `gpuc submit` rather than
half-installed on the way past: it has no uv to run a dispatcher with.

`gpuc submit` and `gpuc requeue` do this themselves when the host they are about
to enqueue on is not on this commit — read from the host's own `config.json`,
which is also the read that tells them what the host's cards and mirror are, and
including a host with no commit recorded. They re-sync the package and restart
the dispatcher first, print one line saying so, and `--no-bootstrap` skips it.
Re-bootstrapping is safe at any time: **running jobs are not disturbed and do
not block it.** A dispatcher that is already alive keeps the lock and finishes
on its own (older) code; every new runner uses the new package, and whichever
dispatcher takes over adopts the running jobs from their `state.json`. Only the
dispatcher is ever replaced, never a runner.

Two sessions on different builds are fine: each ignores fields it does not know.

### The same host from two machines

A host is the host's own -- queue, job state, logs and `config.json` all live
there -- so registering one box from a desktop *and* a laptop is the ordinary
`gpuc host add gpubox --ssh me@gpubox` on the second machine, and every command
on either sees every job. The one thing to get right is the **address**: get
`--gpuc-home` / `--persistent-root` wrong and you have pointed at a second,
empty gpuc home on the same box -- copy them from `gpuc host list --json` on
the first machine.

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
gpuc cancel <job-id>                            # and end a job you are not waiting for
```

`--idle-min 0` reaches the host's own `config.json`, so the dispatcher drains
(retrying unconfirmed outputs, mirroring every job's log and state) and
terminates on its next pass with nothing running. Nothing stops a pod out from
under a running job: cancel the job first if you do not want to wait for it.
**Nothing on this machine watches a pod after it is set up.** One whose
dispatcher has died, or that has stopped answering ssh altogether, bills until
you end it: `gpuc pods` shows it, with its hourly cost and how long ago its
dispatcher last beat, and the RunPod console terminates it. Check `gpuc pods`
before you walk away.

**Disabling the dashboard service.**

```sh
systemctl --user disable --now gpuc-web.service
rm ~/.config/systemd/user/gpuc-web.service
systemctl --user daemon-reload
```
