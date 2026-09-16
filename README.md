# gpu-coordinator

One submit path (`gpuc submit`) for three kinds of GPU host:

- **local** — this machine and the cards you hand it,
- **ssh** — a shared box you have no sudo on, using a subset of its GPUs,
- **runpod** — an ephemeral pod, provisioned for the job and torn down after it.

## The model

Each host runs a small stdlib-only dispatcher out of `~/.gpuc` (or
`<persistent-root>/gpuc`), and **the host is authoritative** for its own queue,
its job state and its logs: `gpuc` asks it over ssh and prints what it says. S3
is a **mirror**, never the queue — with `s3_bucket` set, specs and the job index
go up from here and each host mirrors its own logs and state, which is what
makes `gpuc requeue`, `gpuc logs` after a pod is gone, and `gpuc clean --purge`
possible. Nothing runs in the background on the control side except the
optional `gpuc reconcile` timer, which terminates pods nothing wants any more.

## Quick start

```sh
uv tool install "git+https://github.com/brendanlong/gpu-coordinator@main"
gpuc host add local --gpus 0                 # nvidia-smi index or GPU-… UUID
gpuc host bootstrap local                    # installs uv, the package, the dispatcher
gpuc submit job.example.yaml --host local    # or --runpod --gpu A40 --max-price 0.60
gpuc status                                  # queues, running jobs, recent results
gpuc logs <job-id> -f
gpuc web set-password && gpuc web serve   # the same, in a browser at http://127.0.0.1:8646/
```

## Documentation

| | |
| --- | --- |
| [`docs/setup.md`](docs/setup.md) | install, config keys, credentials, registering hosts, the reconcile timer, upgrading, teardown |
| [`docs/usage.md`](docs/usage.md) | the job spec, every command, RunPod, how a job is killed, exit codes and `--json`, cleanup and retention, troubleshooting |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | the contract: on-host state, dispatcher and runner behaviour, transport, reconcile rules, testing rules |
| [`docs/requirements-review.md`](docs/requirements-review.md) | the 2026-09-14 reasoning behind it (a snapshot; ARCHITECTURE wins where they differ) |
| [`skills/gpuc/SKILL.md`](skills/gpuc/SKILL.md) | the agent-facing quick guide, self-contained |
| [`job.example.yaml`](job.example.yaml) | a fully commented job spec |

Run `gpuc --help`, and `--help` on any subcommand, for the authoritative flags
and defaults.

## Who we say we are

Every request gpuc makes to somebody else's service carries one user agent,
from `gpuc/_version.py`:

```
gpuc/0.1.0 (+https://github.com/brendanlong/gpu-coordinator; self@brendanlong.com)
```

The RunPod API, a pod's self-terminate, the health check's download, bootstrap's
`curl` for the uv installer, every control-side boto3 client, and
`HF_HUB_USER_AGENT_ORIGIN` in each job's environment (so `hf` sends it too; a
job may override it). The one exception is the `aws` CLI on a host, whose
User-Agent cannot be overridden.
