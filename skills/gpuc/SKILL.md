---
name: gpuc
description: Run GPU jobs with gpu-coordinator (gpuc) on this machine, a box you reach over ssh, or a RunPod pod. Use when asked to train, evaluate, or run anything on a GPU, to check on or cancel a GPU job, or to read its logs.
---

# Running GPU jobs with gpuc

`gpuc` is a per-host job queue plus RunPod provisioning. The host is
authoritative for its own queue and state; S3 is a mirror. Hosts registered
once stay registered for every session of this user.

## Before you trust any of this

```bash
gpuc version     # this build's commit, and the one each host was last seen running
gpuc status      # registered hosts, their queues, and what is running
```

`gpuc status` asks the hosts themselves. A `WARNING` line under a host means it
is running another build; `gpuc submit` and `gpuc requeue` re-ship the package
and restart that host's dispatcher before enqueueing (`--no-bootstrap` skips
it), so nothing is needed from you. A host whose config names no build was
never bootstrapped, and `submit` refuses it until `gpuc host bootstrap <host>`
has run.

`gpuc skill` prints this file, and `gpuc skill --install [DIR]` writes a copy
to `DIR/.claude/skills/gpuc/SKILL.md`. If `gpuc` is not on PATH, run it as `uv
run gpuc` from a checkout. Registering, bootstrapping and configuring hosts is
`docs/setup.md` in the repo, not this guide.

## Pick a host

`gpuc host list` is every registered host with its kind and its cards, each as
`[index] name vram uuid`. Which are busy is `gpuc status`.

| kind | when | notes |
|---|---|---|
| `local` | the job fits on this machine's own cards | free, and shared with everything else using that GPU |
| `ssh` | a bigger or shared box already registered | free to you; only the cards registered to that host are ever used |
| `rental` (`--runpod`) | nothing registered is big enough, or they are all busy | costs money; provisions the cheapest matching pod, idles down after 15 min |

Prefer a host you already have over a pod you pay for; a busy host queues your
job behind the running one, which is usually fine. If nothing is registered,
`gpuc host add local` then `gpuc host bootstrap local` gives you this machine
with every card nvidia-smi reports.

## Write a job spec

Run `gpuc submit` from inside the project checkout: the working directory is
rsynced to the host (git-tracked *and* untracked files, `.gitignore` obeyed,
plus a patch of uncommitted changes). Large data must come from S3 or HF inside
the job, never from the checkout. Download it into `$GPUC_DATA_DIR/<name>`
and skip the download when it is already there: that directory outlives the
job, every job on the host sees it, and only `gpuc host clean <host> --data
<name>` removes anything from it.

```yaml
name: lego-s4                      # label only
setup: uv sync --frozen            # phase "setup"; venv is cached across jobs on the host
command: uv run --no-sync python -m experiments.lego.train --k-max 6 --device cuda
# python: .venv/bin/python        # only for a repo that is not a uv project: how the
                                  # GPU check before `main` runs Python in the job's env
gpus: 1                            # 0 for a CPU-only job: never waits, sees no card
use_shared: false                  # also use cards the host borrows rather than owns
env:
  REQUIRE_CUDA: "1"
  PYTHONUNBUFFERED: "1"
secrets: [AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, HF_TOKEN, WANDB_API_KEY]
outputs:
  - path: results                  # relative to the workdir
    s3: s3://my-bucket/lego/{job_id}/results
  - path: figures                  # no s3/hf: KEPT on the host, `gpuc fetch <id>` gets it
  - path: checkpoints
    hf: my-org/lego-checkpoints
    hf_path: "{job_id}"
sync_interval_s: 180               # upload cadence while running, and at the end; minimum 10
priority: 50                       # 0 first, 99 last. The queue is taken strictly in this
                                   # order: a job that does not fit HOLDS the free cards it is
                                   # waiting for (a `gpus: 0` job needs none, so goes ahead)
max_runtime_min: 720               # optional wall-clock cap
estimated_runtime_min: 480         # optional; what `gpuc status` shows the next person
progress_command: "tail -1 results/progress.txt"   # optional; last stdout line is a percentage
progress_interval_s: 60            # prints `0.42` or `42%`; a bare `42` is refused; min 5
auto_preempt: false                # true: the host stops this job whenever that lets a job
                                   # queued AHEAD of it start now, and queues it again. It
                                   # RE-RUNS FROM THE START, any number of times. In return it
                                   # may run on cards HELD for a job ahead, until that job can start
cleanup: on_success                # workdir deleted after a successful run
```

Rules that avoid the classic failures:

- Always list the `secrets` your outputs need. A job with S3 or HF outputs and
  no credentials fails at preflight, in seconds.
- An output with no `s3` or `hf` is kept on the host in the job's workdir;
  the sweeps delete the checkout around it, and only a forced purge
  (`gpuc clean --host H --purge --force --only <id>`) deletes it. A rental refuses one at submit. Use it for
  results you will `gpuc fetch`, not for anything that must survive the host.
- `{job_id}` is required in every output destination: an `s3` without it is
  refused at submit, and so is an `hf` output with it in neither `hf` nor
  `hf_path` (`hf_path` left out means the job id itself).
- Point `outputs:` at a directory the job creates. Files that were already
  there in the checkout are never uploaded, and a path holding only those
  counts as `no-outputs`.
- `use_shared: true` (or `gpuc submit --use-shared`) lets the job onto a host's
  **shared** cards, which gpuc takes only while nvidia-smi says nobody else is
  on them. Owned cards are always used first.
- A phase's background processes are stopped when it ends: one `setup`
  starts is gone before `main` runs, and whatever `main` leaves running is
  stopped before its cards are freed. Start a server the job needs inside
  `command`. On hosts without lingering user systemd (every RunPod pod) a `setsid` or
  `nohup` daemon escapes this and keeps its GPU, so do not leave one.
- Write results incrementally and atomically (temp name, then rename), so the
  periodic sync never uploads a half-written checkpoint.
- Pass `--device cuda` explicitly and keep `REQUIRE_CUDA=1`. A real GPU op runs
  inside the job's venv before `main`; a CPU-only torch fails the job as
  `gpu-preflight` rather than crawling for hours.
- Match the torch build to the host's driver, which `gpuc host probe <host>`
  prints. cu128 is the safe default; a `--runpod` pod is filtered to CUDA >=
  12.8 (`--cuda-min` to change it).
- Say how long the job will take. `progress_command` gives `gpuc status` a
  live end time instead of your guess; a progress command that breaks is
  ignored, never fatal. `gpuc estimate <jobid> --minutes N` sets an estimate
  on a job that is already queued or running.

## Submit, watch, finish

```bash
gpuc submit job.yaml --host local
gpuc submit job.yaml --host <host>                              # any name from `gpuc host list`
gpuc submit job.yaml --runpod --gpu A40 --max-price 0.60        # or --gpu A40,RTX4090 --cloud any

gpuc status                      # every host: free cards, queue, running job + phase, eta,
                                 # recent results; each job as `name (job-id)`, each running
                                 # job's cards as `gpu=2,3`
gpuc status --json               # the same, machine-readable; see "Exit codes" below
gpuc status --all                # adds jobs only the index knows (a host that lost its state)
gpuc status <jobid> [<jobid> ...]
                                 # exactly these jobs, whatever state they are in; one
                                 # question per host however many ids
gpuc logs <jobid>                # tails the host; the S3 mirror when the host is gone
gpuc logs <jobid> -f             # STOPS when the job does: the outcome is the last line and
                                 # gpuc exits 0 only if the job succeeded. --follow-forever
                                 # is the never-ending stream
gpuc wait <jobid> [<jobid> ...]  # block until every job named has ended; one line each,
                                 # exit 0 only if ALL of them succeeded
gpuc ssh <host|jobid>            # a shell there (a job id lands in its workdir)
gpuc ssh <host|jobid> -- ls -la  # one command, run by a login bash there; gpuc exits with that
                                 # command's own exit code
gpuc cancel <jobid> [<jobid> ...]
                                 # SIGTERM then SIGKILL of each job's process tree; final sync
                                 # still runs. cancel, reorder, preempt and estimate all take
                                 # several ids: one call per host, one outcome per id
gpuc reorder <jobid> --priority 10          # queued jobs only; prints the new position and
                                 # when the job is now expected to start
gpuc preempt <jobid> --priority 60          # running jobs only: stop it and queue it again
                                 # under the same id. It RE-RUNS FROM THE START in the same
                                 # workdir. QUEUE THE OTHER JOB FIRST: this is refused (exit 1)
                                 # unless a job waiting for cards would be dispatched ahead of the
                                 # preempted job, and at the SAME priority the preempted job
                                 # wins the tie, so --priority is how you put it behind
gpuc estimate <jobid> --minutes 150
                                 # set estimated_runtime_min on a queued or running job
                                 # (--clear removes it)
gpuc requeue <jobid> --host <host>
                                 # re-run from the mirrored spec as a new job; needs s3_bucket
                                 # set, and re-syncs the workdir from your current directory
gpuc pods                        # RunPod: every pod we own, cost, age, util, host name
```

`failed: <reason>` reasons you will see: `gpu-preflight` (no working CUDA in
the venv), `sync-preflight` (aws/hf or credentials missing), `timeout`
(`max_runtime_min`), `sync` (final upload failed; results exist only on the
host), `no-outputs` (the output path was never written), `terminated`,
`runner-died`. A job that ended for a reason of its own and *also* lost its
upload keeps its reason and lists `sync` or `no-outputs` under `problems`. A
preempted job goes straight back to `queued` and never shows `preempted`
unless its state was edited by hand.

`gpuc status` also flags jobs: `UPLOAD FAILING` on a running job means an
output is not reaching its destination (usually an `outputs:` path that does
not exist yet; check before the job runs for hours); `outputs not uploaded`
means a finished job's results are still only on that host; `OUTPUTS LOST`
means an ephemeral host gave up on them before terminating, and only
re-running the job brings them back.

Never fire-and-forget. After submitting, confirm the job reaches phase `main`
and that its first log lines look right, then check back on a timer. Do not kill
a job on a wall-clock guess; `gpuc status` shows the phase, utilization and
estimate to judge from.

**Do not write a polling loop around `gpuc status`**: `gpuc wait` is that loop,
and it is the one to use whenever the next thing you do depends on how a job
turned out:

```bash
id=$(gpuc submit job.yaml --host gpubox --json | jq -r .job_id)
gpuc wait "$id" || gpuc logs "$id" -n 50     # exit 1 means it did not succeed
```

It backs off from 2s to 30s and takes several ids, so a sweep is one command.
`gpuc logs <jobid> -f` is the same wait for a single job with the log on
screen. Both exit with the **job's** outcome: 0 only if every job succeeded, 1
if any did not, 130 on Ctrl-C. Neither is a background job, and killing one
leaves the run alone.

**A host that cannot be asked** (ssh failed, or the provider says nothing can
run on its pod) is exit 1 with the reason from every command, and may still
hold its jobs; `gpuc wait` keeps asking for five minutes before it reads the
job's final state from the S3 mirror. **A host that is gone** (its rental
ended, or it is no longer registered on this machine) is answered from the
mirror at once, exit 0; a job the mirror has no final state for is exit 1.

**A graph of jobs** (train, then evaluate, for every seed) is a Snakefile, not
a script around `gpuc wait`: `snakemake --executor gpuc --gpuc-host <host>`
submits each Snakemake job whose rule sets `gpu=1` or more as a gpuc job and
runs rules without a `gpu` on the controller. `uv run --with
"gpu-coordinator @ git+https://github.com/brendanlong/gpu-coordinator"
snakemake ...` loads the plugin without adding it to the project. Run that
controller inside tmux, not as a background shell job: it has no persistence of its own, and one that dies
leaves its gpuc jobs running and submits them again when restarted. Outputs
must be in a directory the controller can see, or in object storage;
`docs/snakemake.md` in the repo has both layouts.

## Exit codes and `--json` (read this before scripting anything)

| code | meaning |
| --- | --- |
| 0 | everything the command was asked to do worked. For `gpuc wait` and `gpuc logs -f`, the job succeeded |
| 1 | something failed: transport, provider, a refused submit, **a host that could not be asked**. Whatever did work is still reported, so read the output before retrying. For `gpuc wait` and `gpuc logs -f` it is the **job** that did not succeed |
| 2 | usage: a bad or missing flag |
| 3 | local state (`hosts.json`, `config.toml`) is unreadable, so the answer is **unknown** |
| 4 | the job or host named does not exist, decided only once every host answered. A command given several ids carries out and reports the ones it found and exits 4 for the unknown one |
| 130 | a Ctrl-C |

```bash
gpuc status --json | jq -r '.hosts[] | "\(.name) reachable=\(.reachable) running=\(.running | length)"'
gpuc status --json | jq '[.hosts[].running[] | {job_id, name, phase, elapsed_s, util}]'
```

The document is `{schema_version, hosts: [...], unhosted: [...], errors: [...]}`.
Each host has `name, kind, state, reachable, pkg_commit,
dispatcher{alive, heartbeat_age_s}, provider_util, gpus, shared_gpus, queued,
running, finished, errors, warnings`; `state` is `answered`, `unaskable` (a
failure, with the reason in `errors`; it may still hold its jobs) or `gone`
(the rental ended: not a failure). Each job in the three lists has `job_id,
name, status, reason, problems, upload_errors, phase, priority, attempt,
requeued_from, elapsed_s, util, progress_pct, eta, eta_s,
estimated_runtime_min, progress_error, gpus, gpus_requested, use_shared,
starts_in_s, starts_at, starts_unknown, iso, ended_at, outputs_pending`
(`starts_*` are null unless the job is queued). `unhosted` is `--all`'s list of
jobs only the index knows, each `{job_id, name, host, host_state, status,
requeue, requeued_from, submitted_at, s3_prefix, outputs_lost}`. **Requeue one only if
`requeue` is true**: an `unaskable` host may still be running that job, and a
second copy is not recovery. Each entry in `shared_gpus` adds `memory_mib`,
`utilization_pct` and `unused`.

`priority` (0-99, **lower runs first**) is the field that explains queue order,
and `queued` is already in dispatch order. `starts_in_s` is when that job's
turn is expected, projected from the estimates of the jobs ahead of it; null
means "not known", never "not soon".

**Every command that has an answer takes `--json`**: stdout is one object with
`schema_version`, everything else goes to stderr, and the exit code is
unchanged by the flag. Prefer it to scraping any of the text output.

| command | the document |
| --- | --- |
| `submit`, `requeue` | `{job_id, host, requeued_from, notes[], queue_position, queue_length, dispatched, starts_in_s, starts_at, starts_unknown}`; the queue fields are all null when the host could not be asked again (the job is queued regardless), and `starts_unknown` is why there is no start time |
| `logs` | `{job_id, host, source, location, lines[], notes[]}`; `source` is `host` or `s3`. Not with `-f` (exit 2) |
| `wait` | `{jobs[], errors[]}`, once every job has ended. Each of `jobs[]` is that job's final state in the shape `status --json` uses, plus `host`, `source` (`host` or `mirror`) and `error`. **Check `error`, not `status`**: when it is not null, `status` is only the last thing its host managed to say. Exit 1 unless every job succeeded |
| `status <jobid> ...` | `wait`'s document as things stand now. Exit 4 if an id is unknown, 1 if any other job has an `error`, else 0 however the jobs went |
| `cancel`, `preempt`, `reorder`, `estimate` | `{jobs[], errors[]}`, one entry per id: `{job_id, host, source, error, warnings[]}` plus `status` and the verb's own fields (`priority`; `estimated_runtime_min`; `reorder` adds the queue fields of `submit`). **Check `error`**: when it is set the host did not confirm the change, and the verb's fields are absent. A `warnings` entry means the mirrored spec kept the old value, so a `requeue` would not carry it. Exit 4 if an id is unknown, else 1 if any `error` |
| `pods` | `{pods[], hourly_usd, others[], notes[]}` |
| `version` | `{version, commit, source, dirty, python, executable, hosts[], errors[]}` |
| `host list` | `{hosts[], errors[]}` |
| `host probe` | `{host, sections{}, driver_version, gpus[] each with assigned, assigned_gpus[], notes[], ...}` |
| `host add`, `host set` | one `host list` entry as the registry now holds it, plus `adopted`, `config_path`, `changes[]`, `warnings[]` (`set` adds `address{}`) |
| `host bootstrap` | `{host, home, files, pkg_commit, dispatcher_pid, warnings[]}`; with `--all`, `{hosts[], total, bootstrapped[], failed[], gone[], unreadable[], interrupted, errors[]}` where each of `hosts[]` is `{name, outcome, error, ...}` and `outcome` is `bootstrapped`, `failed`, `gone`, `interrupted` or `not_attempted` |
| `host clean --uv-cache` | `{host, cache_dir, before, after, before_bytes, after_bytes, freed_bytes}` |
| `host remove` | `{host, kind, pod_id, notes[]}`; a rental is not terminated by this |
| `host terminate` | `{host, pod_id, pod_name, pod_status, cost_usd_hr, checked, running[], queued[], outputs_pending[], terminated, forgotten, notes[]}`; `checked` false means the host could not be asked, so the three lists are empty for want of an answer |
| `config init` | `{config_file, existed}` |
| `clean` | `{host, dry_run, purge, freed_bytes, removed[], skipped[], purged[], errors[], ...}` |

Rules, and they are not optional:

- **Key on the JSON `running` list**, never on scraped text and never on the
  exit code alone.
- **Order a queue by `priority`**, never by `eta` or `estimated_runtime_min`,
  which say how long a job takes, not when it is taken.
- **Exit 3 means "unknown", never "nothing is running".** Stop and say so. The
  same goes for a non-empty top-level `errors`, and for `"reachable": false`
  on the host you care about.
- A command that failed still prints a document: `{schema_version, error,
  exit_code}`. `error` (singular) means it did not do what you asked; `errors`
  (plural) is trouble it survived.
- A job's `util` is the host's own nvidia-smi sampler; a pod's `provider_util`
  is RunPod's reading for the whole pod. Do not compare them.
- Ignore keys you do not recognise; more will be added.

## RunPod specifics

- Provisioning to job start is about a minute. A pod that fails before it
  proves healthy is terminated and the next offer tried, cheapest first,
  inside one 15-minute ceiling for the whole attempt.
- The pod terminates itself 15 minutes after its queue empties (`--idle-min`),
  draining its uploads first. There is **no overall pod lifetime**; per-job
  `max_runtime_min` is the cap.
- An existing gpuc pod is reused when its recorded offer still matches the
  request, it owns enough cards, the provider says it is RUNNING, its
  dispatcher heartbeat is under 30 s old, and it is not draining. `--no-reuse`
  forces a new one.
- **Nothing on this machine watches a pod after it is set up, or ends one on
  its own.** A pod whose dispatcher dies never idles out and bills until a
  person ends it: `gpuc pods` shows every pod with our prefix, its hourly cost
  and its heartbeat.
- `gpuc host terminate <host|pod-id>` ends a rental now and forgets it here.
  **Without `--force` the host has to say it is idle**: work in flight, a host
  that did not answer, and a pod registered nowhere here are all exit 1,
  naming what it found. `--force` asks nothing, which is how you end a pod
  whose dispatcher is dead. To let the pod finish instead, `gpuc host set
  <host> --idle-min 0`. **Do not `--force` past a refusal on your own**: ask
  the user; those jobs are not yours to discard.
- A rental that ended itself is forgotten when it is found: `status` and `host
  bootstrap --all` drop the registry entry and say so. A pod that is merely
  stopped is `UNASKABLE`, exit 1, and stays until `gpuc host terminate <name>`
  ends it or `gpuc host remove <name>` forgets it and leaves it billing.
- `gpuc host add <name> --pod <pod-id>` adopts a pod this machine did not
  create. It is also the fix when a command warns `skipping host ... registered
  by an earlier build`: that entry is ignored (and the command exits 1) until
  it is re-added.
- Only act on pods named `gpuc-*`. Others belong to other people.

## Housekeeping

- A host sweeps finished jobs' workdirs itself a day after they end
  (`--workdir-days`; `gpuc host set <host> --workdir-days N` changes it).
  Logs, state and specs are never swept, and `gpuc requeue` rebuilds a workdir
  from git whenever it is gone.
- That sweep refuses a job whose spec says `cleanup: never` and one whose
  `outputs:` have not reached S3 or HF; `gpuc status` names them.
- `gpuc fetch <job-id>` copies a job's `outputs:` (less what came with the
  checkout) from its workdir to `./<job-id>/`, running or finished. It is the
  way out of `failed: sync`, and `--path <dir>` recovers what a mistyped
  `outputs:` missed. `--list` shows what it would copy.
- `gpuc clean --host <host> --all-finished` does that sweep now, at any age.
  `--only <job-id>[,<job-id>]` does it for named jobs only. Add `--purge` for
  whole job dirs, which only removes jobs whose log, state and outputs are
  confirmed mirrored (`--force` deletes a job's only copy).
- `gpuc host clean <host> --uv-cache --hf-cache` prunes the host's caches
  without losing anything a job would need again. `--data <name>` deletes
  from its data directory, which nothing else ever does.
- After a host restarts with its `$HOME` wiped (`dispatcher DOWN`, or ssh
  failing outright): re-copy the SSH key if needed, then `gpuc host bootstrap
  <host>`, then `gpuc status --host <host> --all` and `gpuc requeue` whatever
  was in flight. A host with a `--persistent-root` keeps its queue, so only
  jobs that were running need resubmitting.

Full reference in the repo: `README.md`, `docs/setup.md` (install, hosts,
credentials), `docs/usage.md` (every command and failure mode),
`docs/snakemake.md` (workflows), `docs/ARCHITECTURE.md` (the contract).
