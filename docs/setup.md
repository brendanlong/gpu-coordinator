# Setup

Installing `gpuc` and giving it somewhere to run jobs. Running them is
[usage.md](usage.md). Every command takes `--help` for its full list of flags.

## Install

You need Python 3.11+, [uv](https://docs.astral.sh/uv/), and `ssh` and `rsync`
on the PATH.

```sh
uv tool install "git+https://github.com/brendanlong/gpu-coordinator@main"
gpuc version
```

`gpuc skill --install` drops the agent guide into the current project.

## Add a host

There are three kinds of host. Pick whichever you have.

### This machine

```sh
gpuc host add local
```

### A machine you reach over SSH

The machine needs `ssh`, `rsync`, outbound HTTPS and the NVIDIA driver. No
sudo, and nothing to install by hand.

```sh
gpuc host add gpubox --ssh me@gpubox
```

`host add` records the address, writes the host's config, and bootstraps it:
installs uv, Python and gpuc under your home directory there and checks the
driver, disk and network. A host already on this build is not bootstrapped
again. If the bootstrap fails the host stays registered; fix what it names and
run `gpuc host bootstrap gpubox`.

Adding a host that was already set up from another of your machines adopts its
config and queue as they are, and brings it to this machine's gpuc build. Give the same `--persistent-root` or `--gpuc-home` the first machine
used, if any (`gpuc host list --json` there), or you will get a second, empty
queue.

gpuc does not read `~/.ssh/config`: give the real `user@hostname`, `--port`
if it is not 22, and `ssh_key` in the [config file](#configuration) if your
default key is not the one that gets in.

**On a box you share with other people**, say which cards are yours. By
default a host uses every card it has.

```sh
gpuc host add gpubox --ssh me@gpubox --gpus 2,3               # only cards 2 and 3
gpuc host set gpubox --shared-gpus 4,5                        # may borrow 4 and 5 when idle
gpuc host probe gpubox --all-gpus                             # what the box has
```

Cards can be named by nvidia-smi index or UUID. Shared cards are only used by
jobs that ask for them ([usage.md](usage.md#shared-gpus)).

### A RunPod rental

There is nothing to register: `gpuc submit --runpod` rents a pod, sets it up
and queues the job on it, and the pod shuts itself down once its queue has been
empty for 15 minutes.

```sh
export RUNPOD_API_KEY=...
gpuc submit job.yaml --runpod --gpu A40 --max-price 0.60
```

Your `ssh_key`'s public half is added to the RunPod account the first time.
Choosing GPUs, reuse and ending a pod early are in
[usage.md](usage.md#rentals).

To drive a pod that another of your machines rented: `gpuc pods` lists them,
and `gpuc host add <name> --pod <pod-id>` adopts one.

## Configuration

gpuc works with no config file. To write a commented one to
`~/.config/gpu-coordinator/config.toml`:

```sh
gpuc config init
gpuc config show     # the settings in effect
```

| key | default | meaning |
| --- | --- | --- |
| `s3_bucket` | unset | an S3 bucket gpuc mirrors job specs, logs and state to, so you can read a job's log and resubmit it after its host is gone. Strongly recommended with rentals |
| `ssh_key` | unset | the private key used to reach hosts |
| `image` | a RunPod PyTorch image | the default pod image for rentals |
| `disk_gb` | `50` | the default pod disk for rentals |
| `runpod_pod_prefix` | `"gpuc-"` | gpuc only ever touches pods whose name starts with this |

This machine reaches the bucket with your usual AWS credentials (environment,
`~/.aws/credentials`, or an instance role). Rentals get mirrored to it
automatically, and so does any host added after `s3_bucket` is set. For a host
added before, run `gpuc host set <host> --s3-prefix s3://<bucket>/gpuc/<host>`.

## Credentials for jobs

A job gets secrets by naming them in its spec; their values are read from your
shell when you submit:

```yaml
secrets: [AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, HF_TOKEN, WANDB_API_KEY]
```

Uploading outputs to S3, and a host mirroring logs to your bucket, need the two
AWS keys; uploading to Hugging Face needs `HF_TOKEN`. Secrets are stored on the
host in a file only you can read, and deleted when the job is done.

## The web dashboard

An optional page showing the same thing as `gpuc status`, with buttons to
cancel, preempt and reprioritize jobs.

```sh
gpuc web set-password
gpuc web serve                      # http://127.0.0.1:8646/
```

It has no TLS: only use `--bind 0.0.0.0` on a VPN or behind a proxy.
`gpuc web serve --install` writes a systemd user service for it (not enabled;
the command prints how).

## Upgrading

```sh
uv tool upgrade gpu-coordinator
gpuc host bootstrap --all     # optional: gpuc submit does this per host as needed
```

Upgrading never disturbs running jobs.

## Removing a host

```sh
gpuc host remove <name>       # forget it here; nothing on the host changes
gpuc host terminate <name>    # a rental: end the pod now, then forget it
```

To clear gpuc off a host entirely, remove `~/.gpuc` there while nothing is
running.

## Hosts that lose `$HOME` on restart

Container-based hosts (a Kubernetes pod, most cloud notebooks) often wipe
`$HOME` when they restart, taking gpuc and its queue with them. If the host has
a persistent volume, keep gpuc's state there:

```sh
gpuc host add gpubox --ssh me@gpubox --persistent-root /mnt/data/$USER
```

After a restart, `gpuc host bootstrap <host>` reinstalls gpuc. With a
persistent root the queue then carries on, and the jobs that were running are
queued again and restart from the beginning. Without one, `gpuc status --host
<host> --all` lists what was there and, run from the project's directory,
`gpuc requeue <job-id>` resubmits each (this needs `s3_bucket`).

A persistent root holds the jobs' workdirs too: each checkout and the venv
`setup` builds in it. If the volume is slow (a network filesystem) or small,
keep workdirs on the local disk instead:

```sh
gpuc host set gpubox --scratch-dir '~/gpuc-scratch' && gpuc host bootstrap gpubox
```

The queue, logs and data directory stay on the persistent root. Each job's
checkout is archived there at submit and unpacked onto scratch when it runs,
so a restart that wipes scratch costs a job its workdir, not its code. Jobs
with kept outputs (an `outputs:` path with no `s3` or `hf`) are refused on such
a host, since a restart would take them. Change `--scratch-dir` only while
nothing is queued or running.
