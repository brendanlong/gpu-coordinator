# gpu-coordinator

One submit path (`gpuc submit`) for three kinds of GPU host:

- **local** — this machine and the cards you hand it,
- **ssh** — a shared box you have no sudo on, using a subset of its GPUs,
- **runpod** — an ephemeral pod, provisioned for the job and torn down after it.

![A terminal session: cat job.yaml shows a five-line spec; gpuc submit sends it
to the host workstation, which starts it at once; a second submit queues
eval-checkpoints behind it; gpuc status prints the card, the running job with
its estimated finish, the queued job with its priority and projected start, and
the job that finished before them; gpuc logs -f streams the training output as
it is written; and a last gpuc status has the running job at 100% utilization
and 29% done.](docs/media/cli.gif)

The same recording as text, to copy from:
[asciinema.org/a/fu8jgOVnwDi6dlYd](https://asciinema.org/a/fu8jgOVnwDi6dlYd).

## The model

Each host runs a small stdlib-only dispatcher out of `~/.gpuc` (or
`<persistent-root>/gpuc`), and **the host is authoritative** for its own queue,
its job state, its logs and its own configuration — which cards it may use,
where it mirrors, what environment its jobs get: `gpuc` asks it over ssh and
prints what it says. The registry here is an address book (how to reach a host)
plus a cache of what it last said, so a second machine picks a box up with
`gpuc host add <name> --ssh …` and adopts what is already there. S3
is a **mirror**, never the queue — with `s3_bucket` set, specs and the job index
go up from here and each host mirrors its own logs and state, which is what
makes `gpuc requeue`, `gpuc logs` after a pod is gone, and `gpuc clean --purge`
possible. Nothing runs in the background on the control side except the
optional web dashboard; a rented pod ends itself once its queue has been idle,
`gpuc pods` shows what is still billing, and `gpuc host terminate` ends one
outright.

## Quick start

```sh
uv tool install "git+https://github.com/brendanlong/gpu-coordinator@main"
gpuc host add local                          # every card nvidia-smi reports; --gpus 0
                                             # (an index or GPU-… UUID) narrows it
gpuc host bootstrap local                    # installs uv, the package, the dispatcher
gpuc submit job.example.yaml --host local    # or --runpod --gpu A40 --max-price 0.60
gpuc status                                  # queues, running jobs, recent results
gpuc logs <job-id> -f                        # streams until the job ends, then exits with it
gpuc wait <job-id> [<job-id> ...]            # the same wait with no log, for a sweep
gpuc web set-password && gpuc web serve   # the same, in a browser at http://127.0.0.1:8646/
```

## The dashboard

`gpuc web serve` puts every host behind one password: the cards and what each
is doing, the queue in dispatch order, and the actions the CLI has — logs,
estimate, reorder, preempt, cancel. It asks the hosts, so it shows what they
say and nothing a command could not tell you.

![The gpuc dashboard listing three hosts. desktop (local) has one RTX 4090
running a fine-tune at 96% utilization. lab (ssh) has two A40s running a grid
search, a third card marked shared and in use by somebody else, two jobs queued
with their priorities and projected starts, and one that failed. a100-burst
(runpod) is a rented pod billing $1.64 an hour for a job across both of its
A100s. A running job's row carries Logs, Estimate, Preempt and Cancel; a queued
one carries its priority, Reorder, and no Preempt.](docs/media/web-dashboard.png)

A job's log opens in a panel over the list, and with `follow` ticked the page
re-fetches its tail as the host writes it:

![The log panel open over the host list, showing a running job's log: uv sync
in the setup phase, the GPU and S3 checks that run before the job's main phase,
then training steps and a checkpoint upload to S3.](docs/media/web-logs.png)

## Documentation

| | |
| --- | --- |
| [`docs/SPEC.md`](docs/SPEC.md) | the goals and non-goals every change is checked against |
| [`docs/setup.md`](docs/setup.md) | install, config keys, credentials, registering hosts, upgrading, teardown |
| [`docs/usage.md`](docs/usage.md) | the job spec, every command, RunPod, how a job is killed, exit codes and `--json`, cleanup and retention, troubleshooting |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | the contract: on-host state, dispatcher and runner behaviour, transport, provisioning rules, testing rules |
| [`skills/gpuc/SKILL.md`](skills/gpuc/SKILL.md) | the agent-facing quick guide, self-contained |
| [`job.example.yaml`](job.example.yaml) | a fully commented job spec |

Run `gpuc --help`, and `--help` on any subcommand, for the authoritative flags
and defaults.
