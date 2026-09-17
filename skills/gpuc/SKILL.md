---
name: gpuc
description: Run GPU jobs with gpu-coordinator (gpuc) on this machine, a box you reach over ssh, or a RunPod pod. Use when asked to train, evaluate, or run anything on a GPU, to check on or cancel a GPU job, or to read its logs.
---

# Running GPU jobs with gpuc

`gpuc` is a per-host job queue plus RunPod provisioning. One submit path for
three kinds of host. The host is authoritative for its own queue and state; S3
is a mirror. State is shared by every session of this user, so hosts registered
once stay registered.

## Before you trust any of this

```bash
gpuc version     # this build's commit, and the one each host was last seen running
gpuc status      # registered hosts, their queues, and what is running
```

`gpuc status` is the one that asks the hosts themselves; a `WARNING` line under
a host means it is running a build other than this one (in either direction),
and `gpuc host bootstrap <host>` is the fix.
Jobs still queue and run either way. What a host *is* -- its cards, its mirror,
its env -- is the host's own `config.json`, so there is nothing to keep in step
between machines and nothing to warn about.

`gpuc skill` prints this file, and `gpuc skill --install [DIR]` writes a copy
to `DIR/.claude/skills/gpuc/SKILL.md`.

If `gpuc` is not on PATH, run it as `uv run gpuc` from a checkout. A host that
`gpuc version` marks `DIFFERS: re-bootstrap` needs nothing from you: `gpuc submit` and `gpuc requeue`
re-sync the package and restart that host's dispatcher before enqueueing (pass
`--no-bootstrap` to skip it). Registering, bootstrapping and configuring hosts
is `docs/setup.md` in the repo, not this guide.

## Pick a host

`gpuc host list` is the list that matters: every registered host with its kind
and its cards, each as `[index] name vram uuid` — `?` for name and VRAM until
that host has been probed or bootstrapped. Which of them are busy, and what is
holding them, is `gpuc status`.

| kind | when | notes |
|---|---|---|
| `local` | the job fits on this machine's own cards | free, and shared with everything else using that GPU |
| `ssh` | a bigger or shared box already registered | free to you; only the cards registered to that host are ever used, and some such boxes wipe `$HOME` on restart |
| `runpod` (`--runpod`) | nothing registered is big enough, or they are all busy | costs money; provisions the cheapest matching pod, idles down after 15 min |

Prefer a host you already have over a pod you pay for, and check `gpuc status`
first: a busy host queues your job behind the running one, which is usually
fine.

## Write a job spec

Run `gpuc submit` from inside the project checkout: the working directory is
rsynced to the host (git-tracked *and* untracked files, `.gitignore` obeyed,
plus a patch of uncommitted changes). Large data must come from S3 or HF inside
the job, never from the checkout.

```yaml
name: lego-s4                      # label only
setup: uv sync --frozen            # phase "setup"; venv is cached across jobs on the host
command: uv run --no-sync python -m experiments.lego.train --k-max 6 --device cuda
gpus: 1                            # at least 1
use_shared: false                  # also use cards the host borrows rather than owns
env:
  REQUIRE_CUDA: "1"
  PYTHONUNBUFFERED: "1"
secrets: [AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, HF_TOKEN, WANDB_API_KEY]
outputs:
  - path: results                  # relative to the workdir
    s3: s3://my-bucket/lego/{job_id}/results
  - path: checkpoints
    hf: my-org/lego-checkpoints
    hf_path: "{job_id}"
sync_interval_s: 180               # upload cadence while running, and at the end; minimum 10
priority: 50                       # 0 first, 99 last. The queue is taken strictly in this
                                   # order: a job that does not fit HOLDS the free cards it is
                                   # waiting for, so a wide job at the front idles a card rather
                                   # than losing it to a narrow job behind it
max_runtime_min: 720               # optional wall-clock cap
estimated_runtime_min: 480         # optional; what `gpuc status` shows the next person
progress_command: "tail -1 results/progress.txt"   # optional; last stdout line is a percentage
progress_interval_s: 60            # prints `0.42` or `42%`; a bare `42` is refused; min 5
auto_preempt: false                # true: the host stops this job whenever that lets a job
                                   # queued at a LOWER priority number start now, and queues it
                                   # again as attempt+1. It RE-RUNS FROM THE START, any number
                                   # of times, and may wait for ever on a busy host
cleanup: on_success                # workdir deleted after a successful run
```

Rules that avoid the classic failures:

- Always list the `secrets` your outputs need. They come from your own shell
  environment and are delivered to the host as a 0600 file. A job with S3 or HF
  outputs and no credentials fails at preflight, in seconds.
- `{job_id}` in output destinations makes every run's namespace unique. Never
  reuse a fixed prefix.
- Point `outputs:` at a directory the job creates. Files that were already there
  in the checkout are never uploaded as your results, and a path holding only
  those counts as `no-outputs`.
- `use_shared: true` (or `gpuc submit --use-shared`) lets the job onto a host's
  **shared** cards: ones gpuc does not own and takes only while nvidia-smi says
  nobody else is on them. It is how a `gpus: 4` job runs on a host that owns two
  and shares two, and how a queue drains onto idle cards you do not own. Owned
  cards are always used first. A host with none configured ignores it, and
  `gpuc status` shows shared cards on their own `shared` lines.
- Write results incrementally (per checkpoint, per sweep point). The periodic
  sync bounds what a killed host can lose to one interval.
- Write files atomically (temp name, then rename), so a sync never uploads a
  half-written checkpoint.
- Pass `--device cuda` explicitly and keep `REQUIRE_CUDA=1`. The runner runs a
  real GPU op inside the job's venv before `main`; a CPU-only torch fails the
  job as `gpu-preflight` rather than crawling for hours.
- Match the torch build to the host's driver, which `gpuc host probe <host>`
  prints: a cu13x wheel needs a newer driver than a cu128 one, and an older
  shared box is usually the binding constraint. cu128 is the safe default; a
  `--runpod` pod is filtered to CUDA >= 12.8 (`--cuda-min` to change it).
- Say how long the job will take. On a shared host the next person's only
  alternative to `estimated_runtime_min` is guessing or paying for a pod. If
  the job already writes its progress anywhere, point `progress_command` at it
  and `gpuc status` shows a live end time instead of your guess; a progress
  command that breaks is recorded and ignored, never fatal. Forgetting it at
  submit time is recoverable: `gpuc estimate <jobid> --minutes N` sets it on a
  job that is already queued or running.

## Submit, watch, finish

```bash
gpuc submit job.yaml --host local
gpuc submit job.yaml --host <host>                              # any name from `gpuc host list`
gpuc submit job.yaml --runpod --gpu A40 --max-price 0.60        # or --gpu A40,RTX4090 --cloud any

gpuc status                      # every host: free cards, queue, running job + phase, eta,
                                 # recent results; each job as `name (job-id)`, each running
                                 # job's cards as `gpu=2,3`
gpuc status --json               # the same, machine-readable; --json is on every command
                                 # that has an answer (see "Exit codes" below). A human
                                 # wants `gpuc web serve`: the same in a browser
gpuc status --all                # adds jobs only the index knows (a host that lost its state)
gpuc logs <jobid> [-f]           # tails the host; falls back to the S3 mirror only if that job
                                 # has an s3_prefix (its own or the host's) and s3_bucket is set
gpuc ssh <host|jobid>            # a shell there (a job id lands in its workdir)
gpuc ssh <host|jobid> -- ls -la  # one command, run by a login bash there; gpuc exits with that
                                 # command's own exit code
gpuc ssh <host|jobid> --print    # just print the ssh line, to copy
gpuc cancel <jobid>              # SIGTERM then SIGKILL of the job's process tree; final sync still runs
gpuc reorder <jobid> --priority 10          # queued jobs only; prints the new position and
                                 # when the job is now expected to start
gpuc preempt <jobid> --priority 60          # running jobs only: stop it and queue it again as
                                 # attempt+1 under the same id, so a more important job gets the
                                 # GPUs. It RE-RUNS FROM THE START, in the same workdir the
                                 # stopped attempt left behind, so only preempt a job that
                                 # tolerates that. QUEUE THE OTHER JOB FIRST: this is refused
                                 # (exit 1) unless something already waiting would be dispatched
                                 # ahead of the preempted job. It keeps its own priority, and at
                                 # the SAME priority it wins the tie (older job id sorts first),
                                 # so --priority is how you put it behind the job that is waiting
gpuc estimate <jobid> --minutes 150
                                 # set estimated_runtime_min on a queued or running job
                                 # (--clear removes it); a running job picks it up within a minute
gpuc requeue <jobid> --host <host>
                                 # re-run from the mirrored spec, attempt+1; needs s3_bucket set,
                                 # and re-syncs the workdir from your current directory
gpuc pods                        # RunPod: every pod we own, cost, age, util, wanted?
```

A job's status is its exit code. `failed: <reason>` reasons you will see:
`gpu-preflight` (no working CUDA in the venv), `sync-preflight` (aws/hf or
credentials missing), `timeout` (`max_runtime_min`), `preempted`
(`gpuc preempt` -- or the job's own `auto_preempt` -- stopped that attempt; the job is
queued again as the next one), `sync`
(final upload failed; results exist only on the host), `no-outputs` (the output
path was never written), `terminated`, `runner-died`.

`gpuc status` also flags jobs, not just failures: `outputs not uploaded` means
the results are still only on that host, and **`OUTPUTS LOST`** means an
ephemeral host retried the upload three times while draining and gave up — those
results are gone, and only re-running the job brings them back.

Never fire-and-forget. After submitting, confirm the job reaches phase `main`
and that its first log lines look right, then check back on a timer. Do not kill
a job on a wall-clock guess; `gpuc status` shows the phase, utilization and
estimate to judge from.

## Exit codes and `--json` (read this before scripting anything)

| code | meaning |
| --- | --- |
| 0 | ok — **including** a host that is unreachable or whose dispatcher is down; that is reported per host, not as a failure |
| 1 | the command failed (transport, provider, refused submit) |
| 2 | usage: a bad or missing flag |
| 3 | local state (`hosts.json`, `config.toml`) is unreadable, so the answer is **unknown** |
| 4 | the job or host named does not exist |

```bash
gpuc status --json | jq -r '.hosts[] | "\(.name) reachable=\(.reachable) running=\(.running | length)"'
gpuc status --json | jq '[.hosts[].running[] | {job_id, name, phase, elapsed_s, util}]'
```

The document is `{schema_version, hosts: [...], errors: [...]}`. Each host has
`name, kind, reachable, pkg_commit, dispatcher{alive, heartbeat_age_s},
provider_util, gpus, shared_gpus, queued, running, finished, errors`; each job in those three
lists has `job_id, name, status, reason, phase, priority, elapsed_s, util,
progress_pct, eta, eta_s, estimated_runtime_min, progress_error, gpus,
gpus_requested, use_shared, starts_in_s, starts_at, iso, ended_at,
outputs_pending` (`starts_*` are null unless the job is queued). Each entry in
`shared_gpus` adds `memory_mib`, `utilization_pct` and `unused` — the host's own
verdict on whether gpuc would borrow that card right now.

`priority` (0-99, **lower runs first**) is the field that explains queue order,
and it is on running jobs too. `queued` is already in dispatch order, so
`jq '.hosts[].queued | sort_by(.priority)'` reproduces it, and the host takes it
strictly in that order -- a job that does not fit holds the cards it is waiting
for, so a free card next to a queued job does not mean that job is next.
`starts_in_s` is when that job's turn is expected to come, projected from the
estimates of the jobs ahead of it; it is null when one of them estimated
nothing, so absent means "not known", never "not soon".

**Every command that has an answer takes `--json`**, and means the same thing by
it: stdout is one object with `schema_version`, everything else the command says
goes to stderr, and the exit code is unchanged by the flag. Prefer it to
scraping any of the text output.

| command | the document |
| --- | --- |
| `submit`, `requeue` | `{job_id, host, attempt, requeued_from, notes[], queue_position, queue_length, dispatched, starts_in_s, starts_at, starts_unknown}`; the queue fields are looked up just after the enqueue, and are all null when the host could not be asked again (the job is queued regardless). `starts_unknown` is why there is no start time — a draining host, a job ahead that estimated nothing — and is null when there is one |
| `logs` | `{job_id, host, source, location, lines[], notes[]}`; `source` is `host` or `s3`. Not with `-f` (exit 2) |
| `cancel` | `{job_id, host, status}` |
| `preempt` | `{job_id, host, status, priority, warnings[]}`; `status` is `preempting` and `priority` is what it will be queued again at |
| `reorder` | `{job_id, host, priority, warnings[]}` plus the same queue fields as `submit`, so you can see the move take effect. A `warnings` entry means the mirrored spec kept the old priority, so a `requeue` would not carry the move |
| `estimate` | `{job_id, host, estimated_runtime_min, status, warnings[]}` |
| `pods` | `{pods[], hourly_usd, others[], notes[]}` |
| `version` | `{version, commit, source, dirty, python, executable, hosts[], errors[]}` |
| `host list` | `{hosts[], errors[]}` |
| `host probe` | `{host, sections{}, driver_version, gpus[] each with assigned, assigned_gpus[], uv_cache{}, notes[], ...}` |
| `clean` | `{host, dry_run, purge, freed_bytes, removed[], skipped[], purged[], errors[], ...}` |

```bash
id=$(gpuc submit job.yaml --host gpubox --json | jq -r .job_id)
gpuc logs "$id" --json | jq -r '.lines[-20:][]'
```

Rules, and they are not optional:

- **Key on the JSON `running` list**, never on scraped text and never on the
  exit code alone.
- **Order a queue by `priority`, not by position in the file you read it from**
  — and never by `eta` or `estimated_runtime_min`, which say how long a job
  takes, not when it is taken.
- **Exit 3 means "unknown", never "nothing is running".** Stop and say so;
  something local is broken, and jobs may well be running. The same goes for a
  non-empty top-level `errors`, and for `"reachable": false` on the host you
  care about — we could not ask it.
- A command that failed still prints a document: `{schema_version, error,
  exit_code}`. `error` (singular) means it did not do what you asked; `errors`
  (plural) is trouble it survived and does not imply a non-zero exit on its own.
- A job's `util` is the host's own nvidia-smi sampler; a pod's `provider_util`
  is RunPod's reading for the whole pod. They differ legitimately; do not
  compare them. (In the text output the job's is tagged `util 98% (host)` only
  on a host that also shows a pod's `provider util 71%`.)
- Ignore keys you do not recognise; more will be added.

## RunPod specifics

- Provisioning to job start is about a minute. If it fails before the host
  proves healthy it re-places automatically onto the next offer, cheapest first.
- The pod terminates itself 15 minutes after its queue empties (`--idle-min`),
  draining its uploads first.
- An existing gpuc pod is reused instead of a new one when its recorded offer
  still matches the request (GPU name, VRAM, price, tier, CUDA floor), it owns
  enough cards, the provider says it is RUNNING, its dispatcher heartbeat is
  under 30 s old, and it is not draining. `--no-reuse` forces a new one.
- There is **no overall pod lifetime**; per-job `max_runtime_min` is the cap,
  and the idle timer (`--idle-min`) is the only thing that ends a healthy pod.
- **Nothing on this machine watches or terminates a pod after it is set up.**
  A pod whose dispatcher dies bills until a person ends it: `gpuc pods` shows
  every pod with our prefix, its hourly cost and its heartbeat, and the pod
  ends in the RunPod console. `gpuc host set <host> --idle-min 0` hurries a
  pod that still has a dispatcher.
- `status` says `POD GONE` for a registry entry whose pod is terminated:
  `gpuc host remove <name>` forgets it (the next `submit --runpod` does so on
  its own).
- `gpuc host add <name> --pod <pod-id>` adopts a pod this machine did not
  create, reading the config the pod already has.
- Only act on pods named `gpuc-*`. Others belong to other people.

## Housekeeping

- A host sweeps finished jobs' workdirs itself a day after they end
  (`--workdir-days`, 1 on a host configured for the first time; `gpuc host set
  <host> --workdir-days N` changes it and reaches the host at once). Logs,
  state and specs are never swept. So a failed run is yours to inspect for a day, and `gpuc requeue`
  rebuilds a workdir from git whenever it is gone.
- That sweep refuses a job whose spec says `cleanup: never` and one whose
  `outputs:` have not reached S3 or HF — so a workdir holding results that
  never uploaded is never taken from under you. `gpuc status` names them.
- `gpuc clean --host <host> --all-finished` does that sweep now, at any age.
  Records and logs stay until `--purge`, which only removes jobs whose
  log, state and outputs are confirmed mirrored.
- `gpuc clean --host <host> --only <job-id>[,<job-id>]` does the same for named
  jobs only, whatever their age, and touches no other job. Add `--purge` for
  their whole job dirs (plus `--force` for a job with no confirmed mirror, which
  deletes its only copy). A job id the host does not know refuses the whole
  selection.
- After a host restarts with its `$HOME` wiped (`dispatcher DOWN`, or ssh
  failing outright): re-copy the SSH key if needed, then `gpuc host bootstrap
  <host>`, then `gpuc status --host <host> --all` and `gpuc requeue` whatever
  was in flight. A host with a `--persistent-root` keeps its queue, so only
  jobs that were running need resubmitting.

Full reference in the repo: `README.md`, `docs/setup.md` (install, hosts,
credentials), `docs/usage.md` (every command and failure mode),
`docs/ARCHITECTURE.md` (the contract).
