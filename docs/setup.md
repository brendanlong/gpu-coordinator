# Setup

Installing `gpuc`, telling it about a bucket and some hosts, and taking it all
down again. What to *do* with it afterwards is [usage.md](usage.md); the
contract the code keeps is [ARCHITECTURE.md](ARCHITECTURE.md).

## Prerequisites

**This machine**

- Python >= 3.11 and [uv](https://docs.astral.sh/uv/).
- `ssh` and `rsync` on the PATH. gpuc does not read your `~/.ssh/config`.
- An ssh key that reaches your hosts. Point `ssh_key` in `config.toml` at the
  private key; its `.pub` is uploaded to the RunPod account before the first
  pod is created. Unset means ssh picks its own key.
- `systemd --user`, only if you want the web dashboard as a service.

**Each host**

- `ssh` and `rsync`, an account you can log into with that key, and outbound
  HTTPS.
- The NVIDIA driver and `nvidia-smi`. A host without them can be registered
  and probed, but nothing can be submitted to it.
- Nothing else installed by hand: `gpuc host bootstrap` puts uv, a Python
  (floor 3.11, it installs 3.12), the `gpuc.host` package, the `aws` CLI v2
  bundle and `hf` into `$HOME` over ssh, and starts the dispatcher.
- `systemd --user` is optional: with it a cancel reaps the whole process tree,
  without it a double-forked grandchild can escape (see [how a job is
  killed](usage.md#how-a-job-is-killed)). `gpuc host probe` reports which you
  get.

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
and the job index to `s3://<bucket>/gpuc/index/`, read by `gpuc requeue` and
`gpuc status --all`. A host's `--s3-prefix` is written by *the host*: each
job's `log.txt` and `state.json` to `<prefix>/jobs/<job-id>/`, read by `gpuc
logs` and `gpuc wait` once the host is gone and required by `gpuc clean
--purge`. A pod from `gpuc submit --runpod` gets `s3://<bucket>/gpuc/<pod-name>`
automatically; a `local` or `ssh` host does **not**: set it by hand, on a host
whose jobs can authenticate to that bucket.

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
if that file exists (`install -m 600 /dev/null ~/.config/gpu-coordinator/env`,
then one `RUNPOD_API_KEY=...` line; `--install` does not create it). The
service restarts on failure, needs `loginctl enable-linger` to outlive your
session, and with no password set fails after five tries over five minutes.
Disabling it again is under [teardown](#teardown) below.

## Credentials

**This machine (boto3).** The S3 mirror uses boto3's default credential chain:
environment variables, `~/.aws/credentials`, an instance role. On the bucket it
needs `s3:PutObject` and `s3:GetObject` under `gpuc/*`, and `s3:ListBucket`
for `gpuc status --all` and `gpuc clean --purge --verify`.

**The hosts.** A host uploads with whatever the *job* carries: the names in the
spec's `secrets:` are read from your shell at submit time and delivered to the
host as `~/.gpuc/secrets/<job-id>.env`, mode 0600. `secrets:
[AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY]` is all an S3 output, or the host's
own `--s3-prefix` mirror, needs. A provisioned pod with an `s3_prefix` is also
given `~/.aws/credentials` (0600) from `AWS_ACCESS_KEY_ID`,
`AWS_SECRET_ACCESS_KEY` and `AWS_SESSION_TOKEN` in your shell, with the region
from `AWS_REGION`, `AWS_DEFAULT_REGION`, else `us-east-1`.

**RunPod.** Export `RUNPOD_API_KEY`. `gpuc submit --runpod`, `gpuc pods`,
`gpuc host add --pod` and `gpuc host terminate` exit 1 with one line if it is
missing. A pod terminates itself with the pod-scoped key RunPod leaves in its
own `/etc/rp_environment`; nothing of yours is delivered.

**Hugging Face.** Put `HF_TOKEN` (or `HUGGING_FACE_HUB_TOKEN`) in the job's
`secrets:`.

## Registering hosts

```sh
gpuc host add local                                        # this machine, every card it has
gpuc host add gpubox --ssh me@gpubox --port 22 --gpus 2,3  # a box you reach over ssh, two of its cards
gpuc host add gpubox --ssh me@gpubox                       # …one somebody already set up: adopt it
gpuc host set gpubox --shared-gpus 4,5                     # two more it may borrow while nobody else is on them
gpuc host probe gpubox       # driver, the cards assigned to this host as `[index] name vram uuid`,
                             # disk, $HOME's filesystem, systemd --user, uv, python3
gpuc host probe gpubox --all-gpus   # every card in the box, `(assigned)` on the ones this host owns
gpuc host bootstrap gpubox   # installs uv, the package and the dispatcher; idempotent
```

**`host add` is a connect.** It opens the host, probes it, and reads
`~/.gpuc/config.json`:

- the host **has** a config (set up from another machine, or from this one
  before): it is adopted as it stands, and only the address (`--ssh`, `--port`,
  `--gpuc-home` / `--persistent-root`) is recorded here. Other flags are
  overrides, written through to the host and reported field by field
  (`host <- retention_days 30.0 -> 7.0`). A `--gpus` that claims *some* of the
  cards the host already has is refused unless `--force`; a disjoint list goes
  through. The host is registered under the name it calls itself, unless a
  *different* host already has that name here.
- the host has **no** config: one is written, owning **every card** nvidia-smi
  reports there, by UUID. `--gpus` narrows that (`--gpus ''` for a host whose
  cards gpuc may not use), and `--shared-gpus` takes its cards out of the owned
  set. A host that reports no cards at all is refused unless `--gpus ''` says
  so on purpose; a host left owning nothing is registered, and nothing can be
  submitted to it until `gpuc host set <name> --gpus <list>` assigns some.

A `config.json` that is there but does not parse stops all of this; the error
says to fix or delete it.

RunPod pods are never *created* by hand: `gpuc submit --runpod ...` creates the
pod, registers it as a `rental`, and bootstraps it
([usage.md](usage.md#runpod)). A pod that already exists is adopted the same way
any other host is, from any machine holding the API key:

```sh
gpuc pods                                  # the account's pods, with their ids
gpuc host add rented --pod <pod-id>        # its address from the provider, its config from the pod
```

A pod nobody has bootstrapped has no dispatcher and so will never end itself;
`host add` says so, and `gpuc host bootstrap <name>` gives it one.

A rental registered by an earlier build of gpuc is not read: every command
warns `registered by an earlier build` for that entry, works with the rest,
and exits 1 until you run `gpuc host add <name> --pod <pod-id>` again.

The address is the top two rows, kept here (`here <- …`) and applied to the
host by the next `gpuc host bootstrap`. Every other flag is the host's own
config: `host set` writes it through to the host's `config.json` at once (the
host has to answer) and reports each change as `host <- …`.

| flag (`host add`, and `host set` to change one) | default | meaning |
| --- | --- | --- |
| `--ssh user@host` / `--port N` | this machine / `22` | omit `--ssh` for a `local` host |
| `--pod POD_ID` (`host add`) | none | adopt a pod the account is renting; the provider says where it is. Needs `RUNPOD_API_KEY`. Add `--gpuc-home` if that pod keeps gpuc somewhere other than `$HOME/.gpuc` |
| `--gpus 2,3` or `--gpus GPU-8064…,3` | every card nvidia-smi reports, on a host with no config; what the host has, on one that does | nvidia-smi **indices**, UUIDs, or a mix, stored as typed and re-resolved to UUIDs on every dispatch pass. An owned card the host cannot see is `UNAVAILABLE` and jobs wait for it |
| `--shared-gpus 4,5` | none | cards gpuc may **borrow** but does not own, spelled like `--gpus` and never overlapping it; see [shared GPUs](usage.md#shared-gpus) |
| `--gpuc-home PATH` | `$HOME/.gpuc` | where gpuc home lives on the host |
| `--cache-dir PATH` | bootstrap decides | `UV_CACHE_DIR` in the host's `env`, the same as `--env UV_CACHE_DIR=PATH`. Bootstrap sets one beside gpuc home when gpuc home and `~/.cache` are on different filesystems, and never overrides one the config names. Under `--persistent-root`, `HF_HOME` is set beside gpuc home the same way |
| `--persistent-root R` | none | gpuc home moves to `R/gpuc` (below) |
| `--env K=V` (repeatable) | none | extra environment for every job on this host, applied *before* the job's own `env:`. It replaces the whole set except `UV_CACHE_DIR`, `HF_HOME` and `GPUC_DATA_DIR`, which survive an `--env` that does not name them; `--env K=` removes a key, those included. `GPUC_DATA_DIR` moves the host's [data directory](usage.md#the-hosts-data-directory) |
| `--s3-prefix s3://…` | `s3://<s3_bucket>/gpuc/<name>` on a host being configured for the first time, when `s3_bucket` is set; none otherwise | this host's own log/state mirror; `''` turns it off |
| `--retention-days N` | none | auto-purge whole job dirs this old, only ones whose log and state are confirmed mirrored; `''` turns it off |
| `--workdir-days N` | `1` on a host being configured for the first time | auto-sweep a finished job's `workdir/` once it ended this long ago; `''` turns it off |
| `--idle-min N` | `15` | how long a rental may sit with an empty queue before terminating itself. Inert on `local` and `ssh` hosts |

On a box you share, pass `--gpus`: `host probe` lists only those cards and says
how many it hid (`2 of 8 assigned to gpubox`); `--all-gpus` shows the box as
nvidia-smi sees it. Either way the probe records every card's name and VRAM.
An assigned entry no card answers to is called out, as are two entries naming
one card; `gpuc host bootstrap` fails its `gpu_uuids` check on both.

`gpuc host list` shows what is registered, one block per host, with each card as
`gpu [index] name vram uuid` and a `pkg` line naming the commit the host was
running **when it was last read** (`as of 3m ago`). It never asks a host
anything; `gpuc status` is what asks, and `gpuc host probe` refreshes the
cache. The interpreter path bootstrap chose is in `gpuc host list --json` and
`gpuc host probe`.

### Hosts whose `$HOME` is wiped on restart

A container-backed host (a Kubernetes pod, most cloud notebooks) usually has
`$HOME` on the image's throwaway upper layer, which `gpuc host probe` reports as
`home_fs: overlay`. That is not by itself a problem: S3 is the mirror and `gpuc
requeue` the recovery. Where keeping the queue is worth a network volume,
`--persistent-root R` moves **gpuc home and only gpuc home** to `R/gpuc`
(`config.json` and every `jobs/<id>/`) and puts the uv and Hugging Face caches
beside it; uv, its Pythons and the `aws` bundle stay in `$HOME`. `R` is
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
2. `gpuc host bootstrap <host>`: the whole of the host-side recovery. It
   restores the last `config.json` this machine read off that host.
3. With a `--persistent-root`, stop here: queued jobs start again as soon as
   the dispatcher does, and only jobs that were *running* need resubmitting.
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

`gpuc version` and `gpuc host list` never ssh: they show the commit the host
was running when this machine last read it, labelled with its age. `gpuc
status` asks each host what it is running now, reports the *running*
dispatcher's commit alongside the package's, and warns if they differ.

Re-bootstrapping is safe at any time: **running jobs are not disturbed and do
not block it.** The dispatcher bootstrap starts takes over from the incumbent
and adopts the running jobs; only the dispatcher is ever replaced, never a
runner.

`--all` takes every registered host in turn, including rentals. A host that
fails does not stop the others: the run ends with a tally naming each failure,
including host entries this build could not read, and exits 1. A rental the
provider no longer has is forgotten and the tally says so. `--json` prints
that tally as one entry per host ([usage.md](usage.md#--json-everywhere-else)).

`gpuc submit` and `gpuc requeue` re-ship the package and restart the dispatcher
themselves when the host's own `config.json` names another commit, print one
line saying so, and `--no-bootstrap` skips it. A config that names no commit
is a host nothing has bootstrapped, and is refused. A checkout with
uncommitted changes is its own build (`<commit>-dirty-<hash>`), so the next
edit to the tree is re-shipped too.

Two sessions on different builds are fine: each ignores fields it does not know.

### The same host from two machines

Registering one box from a desktop *and* a laptop is the ordinary `gpuc host
add gpubox --ssh me@gpubox` on the second machine, and every command on either
sees every job. Get the **address** right: a wrong `--gpuc-home` /
`--persistent-root` points at a second, empty gpuc home on the same box. Copy
them from `gpuc host list --json` on the first machine.

## Teardown

```sh
gpuc host remove <name>        # forgets it locally; nothing on the host changes
gpuc host clean <name> --uv-cache   # `uv cache prune` there, if you want the disk back first
gpuc host clean <name> --hf-cache   # `hf cache prune`: detached revisions and partial downloads
gpuc host clean <name> --data lego-v3   # delete $GPUC_DATA_DIR/lego-v3
gpuc host terminate <name>     # a rental only: ends it at the provider, then forgets it
```

`host remove` does not stop anything: to clear a host's own state, delete its
gpuc home (`gpuc ssh <host> -- rm -rf ~/.gpuc`, or the persistent root's `gpuc`
directory) while nothing is running.

**Terminating a pod deliberately.**

```sh
gpuc host terminate <name>             # end the rental now, and forget it here
gpuc host terminate <pod-id> --force   # one that cannot answer: do not ask it anything
```

Without `--force` the host has to say it is idle; the refusals are in
[usage.md](usage.md#terminate). Either way the terminate is confirmed with the
provider before this machine forgets the host, and a pod that may still be
billing keeps its registry entry.

**To let the pod finish first**, tell it to stop itself instead:

```sh
gpuc pods                           # confirm the name and what it is doing
gpuc host set <name> --idle-min 0   # stop as soon as the queue is empty
gpuc cancel <job-id>                # and end a job you are not waiting for
```

**Nothing on this machine watches a pod after it is set up.** One whose
dispatcher has died will never idle out: `gpuc pods` shows it with its hourly
cost and its heartbeat, and `gpuc host terminate <pod-id> --force` ends it.
Check `gpuc pods` before you walk away.

**Disabling the dashboard service.**

```sh
systemctl --user disable --now gpuc-web.service
rm ~/.config/systemd/user/gpuc-web.service
systemctl --user daemon-reload
```
