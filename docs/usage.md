# Usage

Running jobs once you have a host ([setup.md](setup.md)). This covers what
each part does; `gpuc <command> --help` lists every flag. A pipeline of many
dependent jobs is [snakemake.md](snakemake.md).

## Quick start

```yaml
# job.yaml
name: train-small
setup: uv sync
command: uv run python train.py --out results
gpus: 1
secrets: [AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY]
outputs:
  - path: results
    s3: s3://my-bucket/runs/{job_id}
```

```sh
gpuc submit job.yaml --host gpubox    # prints the job id and its place in the queue
gpuc status                           # every host: cards, queue, running and recent jobs
gpuc logs <job-id> -f                 # follow the log until the job ends
gpuc wait <job-id> ...                # block until the jobs end; exits 0 only if all succeeded
gpuc fetch <job-id>                   # copy its outputs to ./<job-id>/ while it is still on the host
```

The job runs in a copy of the directory you submit from: every file git
tracks plus untracked ones that are not `.gitignore`d, uncommitted changes
included. (`--no-git` for a directory that is not a repository.) Keep large
data out of it; see [the data directory](#the-hosts-data-directory).

## The job spec

YAML or JSON. `job.example.yaml` in the repo root is a commented example of
every field.

| field | default | meaning |
| --- | --- | --- |
| `command` | required | bash to run. Anything longer than a line or two belongs in a script: `command: bash run.sh` |
| `setup` | none | bash to run first, e.g. `uv sync` |
| `name` | none | a label for `gpuc status` |
| `gpus` | `1` | how many GPUs. `0` for a CPU-only job |
| `priority` | `50` | `0`–`99`, **lower runs first** |
| `env` | none | plain environment variables |
| `secrets` | none | names of environment variables to copy from your shell at submit |
| `outputs` | none | what to keep: see [outputs](#outputs) |
| `max_runtime_min` | none | kill the job after this long |
| `estimated_runtime_min` | none | your guess, used to show when queued jobs will start |
| `progress_command` | none | prints how far along the job is: see [progress](#progress-and-estimates) |
| `auto_preempt` | `false` | let the job be bumped by higher-priority work: see [preemption](#preemption) |
| `use_shared` | `false` | may use the host's [shared GPUs](#shared-gpus) |
| `cleanup` | `on_success` | when the job's checkout is deleted: `on_success`, `always` or `never` |

Before `command` runs, gpuc checks inside the job's environment that its GPUs
work (this expects a uv project with torch; set `python:` if yours is
different) and that every output destination is writable. A job fails early,
with the reason in its log, rather than after hours of training.

## Outputs

```yaml
outputs:
  - path: checkpoints                          # relative to the job's directory
    s3: s3://my-bucket/lego/{job_id}/ckpt      # uploaded as the job runs, and at the end
  - path: model
    hf: me/lego-models                         # a Hugging Face repo, under <job-id>/
  - path: plots                                # no destination: kept on the host
```

- `s3` destinations must contain `{job_id}`, so runs never overwrite each other.
- Uploads happen every few minutes while the job runs, not just at the end.
- Files that were already in your checkout are not treated as outputs.
- An output with no destination stays on the host until you delete it;
  `gpuc fetch <job-id>` copies it to you. Rentals refuse these, since the pod
  goes away.
- `gpuc fetch` works on any job whose directory is still on its host, running
  or finished, and `--path` copies any other file from it.

## The host's data directory

Each job gets `$GPUC_DATA_DIR`, a directory on the host shared by every job
and never cleaned automatically. Download datasets there once:

```sh
data="$GPUC_DATA_DIR/lego-v3"
[ -d "$data" ] || aws s3 sync s3://my-bucket/datasets/lego-v3 "$data"
```

`gpuc host clean <host> --data lego-v3` deletes an entry. Hugging Face
downloads already share one cache per host.

## Priority and the queue

Lower `priority` runs first, strictly: a job that is waiting for cards to free
up holds them, and lower-priority jobs will not take them even if that leaves
cards idle for a while. If you would rather a big job wait than block smaller
ones, give it a higher number. Jobs with `gpus: 0` are never blocked.

`gpuc set <job-id> --priority N` reorders a queued job.

### Preemption

`gpuc preempt <job-id>` stops a running job and puts it back in the queue, so
something ahead of it can start. Queue the job you are making room for first,
at a lower priority number, then preempt.

A job with `auto_preempt: true` may run on cards a higher-priority job is
waiting for, and is preempted automatically when that job can start.

A preempted job **starts over from the beginning** (`setup` included) in the
same directory. If it is long, have it save checkpoints and resume from them.

### Shared GPUs

A shared card (`gpuc host set <host> --shared-gpus 4,5`) is one gpuc may borrow
but does not own. Only jobs with `use_shared: true` (or `gpuc submit
--use-shared`) use one, only after every owned card is busy, and only while
nvidia-smi shows nobody else on it. Once borrowed it is held until the job
ends, even if its owner comes back.

## Progress and estimates

`gpuc status` shows when running jobs should end and queued jobs should start,
if it has something to go on. Either give an estimate:

```yaml
estimated_runtime_min: 240
```

or have the job report its progress and let gpuc extrapolate:

```yaml
progress_command: "tail -1 progress.txt"    # the job writes 0.37, or 37%
```

`gpuc set <job-id> --estimate N` changes the estimate on a queued or running
job.

## Managing jobs

```sh
gpuc cancel <job-id> ...                     # stop and don't retry
gpuc preempt <job-id> ...                    # stop and requeue
gpuc set <job-id> ... --priority 10          # also --estimate, --max-runtime
gpuc requeue <job-id>                        # submit a finished job again, from your current directory
gpuc ssh <job-id>                            # a shell in the job's directory
```

`cancel`, `preempt`, `set`, `wait`, `fetch` and `status` take any number of job
ids. `--host` is only needed when gpuc cannot find a job itself.

**How a job is stopped.** Cancel, preempt and `max_runtime_min` send the job
SIGTERM, then SIGKILL 15 seconds later; trap SIGTERM if you want to save
something. Its outputs are uploaded one last time afterwards. Anything the job
started (dataloader workers, servers) is stopped with it, except that on a
host without systemd user sessions (RunPod included) a process that
daemonizes itself can escape.

`gpuc requeue` needs `s3_bucket` set, since it reads the original spec from
there. It can send the job to a different host (`--host`, or `--runpod`).

## Rentals

```sh
gpuc submit job.yaml --runpod --gpu A40,RTX4090 --max-price 0.60
```

`--gpu` takes one or more GPU names and the cheapest available match is
rented; `--gpu-count` and `--cloud community` (cheaper, less
reliable) narrow the choice, and `gpuc submit --help` lists the rest. If a pod
fails to come up it is terminated and the next offer tried, for up to 15
minutes. A running pod that already matches is reused rather than renting
another (`--no-reuse` to always rent).

The pod terminates itself once its queue has been empty for `--idle-min`
(default 15), after finishing its uploads. Nothing on your machine needs to
stay running. A rental's disk goes with it, so give every output an `s3` or
`hf` destination and set `s3_bucket` so its logs outlive it.

```sh
gpuc pods                            # every gpuc pod on the account, with its cost
gpuc host terminate <name>           # end one now; refuses unless it is idle (--force)
gpuc host set <name> --idle-min 0    # or: let it finish its queue, then stop
```

A pod whose setup was interrupted, or whose gpuc died, will not stop itself.
`gpuc pods` shows them with no heartbeat; end them with `gpuc host terminate
<pod-id> --force`.

## Disk cleanup

A finished job's directory holds its checkout and whatever it built there
(usually a virtualenv). Its log and state are kept; the rest is deleted:

- when the job succeeds (`cleanup: on_success`, the default),
- or a day after it ends, whatever happened (unless `cleanup: never`) (`gpuc host set <host>
  --workdir-days N` to change that),

but never while outputs are waiting to upload, and never outputs that are kept
on the host. To clean up by hand:

```sh
gpuc clean --host gpubox --all-finished --dry-run
gpuc clean --host gpubox --all-finished
```

`--purge` also deletes logs and state, for jobs whose logs are safely in S3.

## Scripting

Every command that reports something takes `--json` and prints exactly one
JSON document on stdout. Fields are added over time, never changed; ignore
ones you don't know. Exit codes:

| code | meaning |
| --- | --- |
| 0 | everything worked. For `wait` and `logs -f`: the job succeeded |
| 1 | something failed (the command says what), or for `wait`/`logs -f`, the job did |
| 2 | bad usage |
| 3 | gpuc's local state is unreadable, so the answer is unknown |
| 4 | no such job or host |

A command reports everything it can: one unreachable host or bad job id is
reported, the rest carry on, and the exit code is non-zero.

```sh
id=$(gpuc submit job.yaml --host gpubox --json | jq -r .job_id)
gpuc wait "$id" || echo "failed"
```

## When a job fails

`gpuc status <job-id>` gives the reason and `gpuc logs <job-id>` the detail.

| reason | meaning |
| --- | --- |
| `exit N` / `setup` | your command, or `setup`, exited non-zero |
| `gpu-assert` | a card assigned to the job has disappeared from nvidia-smi |
| `gpu-preflight` | torch in the job's environment could not use its GPUs; check the torch build against the host's driver |
| `sync-preflight` | an output destination is not writable: usually a missing secret or a typo in the bucket |
| `sync` | the final upload failed; `gpuc fetch <job-id>` gets the outputs from the host |
| `no-outputs` | an `outputs:` path was never written; if the job wrote elsewhere, `gpuc fetch <job-id> --path <where>` |
| `timeout` | it ran past `max_runtime_min` |
| `needs N GPUs, host owns M` | the host lost cards after the job was queued |
| `runner-died`, `spawn-failed`, `terminated` | gpuc itself was killed or crashed on the host |
| `checkout-lost` | the host keeps workdirs on scratch, and the job's code could not be put back there: the archive is gone, or scratch is full or not writable |
| `host-restarted` | the host kept restarting under the job; each earlier restart queued it again from the start |

Other things that go wrong:

| symptom | what to do |
| --- | --- |
| `gpuc status` says `dispatcher DOWN` | `gpuc host bootstrap <host>`. Running jobs are not affected |
| a warning that a host runs a different gpuc build | `gpuc host bootstrap <host>`, or just submit to it |
| `uv sync` re-downloads everything on each job | `gpuc host bootstrap <host>` puts uv's cache somewhere it can be reused |
| a host is out of disk | `gpuc clean --host <host> --all-finished` |
| a host shows `GONE` | the rental ended; with `s3_bucket` set, its jobs' logs can still be read and the jobs requeued |
| a host shows `UNASKABLE` | gpuc could not reach it; if it is a rental, it may still be billing |
| a rental's outputs show `OUTPUTS LOST` | it could not upload them before shutting down; fix the credentials and `gpuc requeue` |
