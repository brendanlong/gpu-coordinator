# Usage

Installing gpuc and registering hosts is [setup.md](setup.md). `gpuc --help`, and
`--help` on every subcommand, is the authoritative list of flags; this page is
what they mean together.

## The job spec

A job is one YAML (or JSON) document. `gpuc submit job.yaml --host H` runs it in
the working directory you submit from. `job.example.yaml` in the repo root is a
commented example; `-` as the file name reads the spec from stdin.

| field | default | meaning |
| --- | --- | --- |
| `command` | **required** | run in `workdir/` as phase `main`; a blank one is refused at submit |
| `name` | `""` | a label for `status`; not an identifier |
| `setup` | none | run first, as phase `setup` |
| `python` | `uv run --no-sync python` | how to run Python inside the job's own environment, for the GPU check before `main`. A repo that keeps its stack outside a uv project names its own interpreter: `.venv/bin/python`, `python` |
| `gpus` | `1` | how many of the host's GPUs to assign, at least 1. More than the host can ever provide is refused at submit |
| `use_shared` | `false` | also let this job onto the host's **shared** GPUs — cards gpuc does not own and takes only while nobody else is on them. See [shared GPUs](#shared-gpus). `gpuc submit --use-shared` sets it from the command line |
| `env` | `{}` | plain environment for the job, applied after the host's `--env` |
| `secrets` | `[]` | names read from *your* shell at submit time and delivered to the host as `~/.gpuc/secrets/<job-id>.env` (0600). Missing from your shell is a refused submit |
| `outputs` | `[]` | `{path, s3}` and/or `{path, hf, hf_path, hf_create}`; `path` is relative to the workdir. `s3` **must contain `{job_id}`**, and so must `hf` or `hf_path`, or the submit is refused; `hf_path` left out is the job id itself |
| `sync_interval_s` | `180` | background upload cadence; **minimum 10** |
| `priority` | `50` | `0`–`99`, lower dispatches first, and the queue is taken strictly in that order — see [priority is not advisory](#priority-is-not-advisory) |
| `max_runtime_min` | none | wall clock from the runner's start; over it the job is `failed: timeout` |
| `estimated_runtime_min` | none | roughly how long you expect it to take, measured the same way. Nothing enforces it; see [job length estimates](#job-length-estimates) |
| `progress_command` | none | run in the workdir during phase `main`; its last line of stdout is how far along the job is |
| `progress_interval_s` | `60` | how often to run it; **minimum 5** |
| `auto_preempt` | `false` | let the host stop this job whenever that lets a job queued at a **lower** `priority` number start right away; see [automatic preemption](#automatic-preemption) |
| `requires` | `{}` | e.g. `cuda_min: "12.8"`. **Informs provisioning only**; the host never checks it |
| `cleanup` | `on_success` | when the runner deletes `workdir/`: `on_success`, `always`, `never` |
| `attempt` | `1` | set by `gpuc requeue` and by a preempt, never by you; a submitter's value is ignored |

Unknown keys are refused at submit.

**Every output destination must name the job.** `{job_id}` expands in `s3`, `hf`
and `hf_path`; a destination that does not contain it after expansion is refused
at submit, as is any other `{...}`. An `hf` output with no `hf_path` uploads
under the id itself. `hf_create: true` lets the sync preflight create a Hugging
Face repo that does not exist. Nothing looks at what a destination already
holds, so this is the only overwrite guard there is.

**Files already under an output path are not your results.** The runner records
what each declared path holds before `setup`; uploads skip files that still
match, and a path holding only those is `failed: no-outputs`. Above **500**
pre-existing files under one path the exclusion is dropped, with a warning.
`gpuc submit` warns about them before the job is queued.

**The sync preflight** runs after the GPU check and before `main`, and a job
with no outputs on a host with no mirror checks nothing. With S3
outputs (or a host `s3_prefix`), `aws` must resolve and a `.preflight` object
must upload to every destination and to the host's mirror prefix; with HF
outputs, `hf` must resolve, `hf auth whoami` must succeed with the job's token,
and a `.preflight` file must upload to each repo. Failure is `failed:
sync-preflight`, seconds in, with the command and its error in the log.

## Priority is not advisory

Dispatch order is `<priority>-<job id>`, lower first, and the host takes the
queue **in order**: a job that cannot start yet holds the free cards it is
waiting for, and nothing behind it may take them. A two-card job at priority 10
does not lose its card to a one-card job at 50 that happens to fit.

**It costs utilization**, and on a rented pod that is billed. Queue a big job at
a **higher** number if you would rather it waited than have a card sit idle for
it.

A job waiting for a [shared card](#shared-gpus) somebody else is using does not
hold: the queue behind it runs. A job short of an *owned* card holds even when
that card has dropped off `nvidia-smi`; `gpuc status` marks the card
`UNAVAILABLE` and the job's `starts_unknown` names it.

## Shared GPUs

A box may hand you some cards outright and leave the rest to other people.
`--shared-gpus` says which are the second kind: cards gpuc may *borrow*, never
ones it owns.

```sh
gpuc host set spar --gpus 2,3 --shared-gpus 4,5
gpuc submit job.yaml --host spar --use-shared   # or `use_shared: true` in the spec
```

A job reaches a shared card only when both hold:

- it asked — `use_shared: true`, or `gpuc submit --use-shared`. Off by default;
- **nobody else is on the card**: nvidia-smi reports 0 MiB used and 0% util for
  it right now. A card that could not be read counts as in use.

Owned cards come first: a job takes every free owned card it can use and borrows
only the shortfall. A borrowed card is held until the job ends, so if the card's
real owner starts something there you are sharing it, and `gpuc preempt` is the
way out.

`gpuc status` shows borrowed cards on their own `shared` lines, and
`gpuc host bootstrap` refuses a card listed as both owned and shared.

## Job length estimates

Both ways of answering "when will this be done" are optional, and neither ever
affects a job's outcome.

`estimated_runtime_min` is your own guess, measured from the runner's start.
`gpuc estimate <job-id> --minutes N` sets it on a job that is already **queued or
running**, and `--clear` takes it off; a running job's runner picks it up within
a minute, and a finished job is refused (exit 1). It changes the job's state,
never the spec it was submitted with; the mirrored spec is updated too, so a
later `gpuc requeue` carries the estimate, and if the mirror cannot be written
the command still succeeds and says so.

`progress_command` replaces the guess with a measurement. It runs in the workdir,
with the job's own environment, every `progress_interval_s` of phase `main`, and
the **last line of its stdout** is how far along the job is:

```yaml
progress_command: "tail -1 results/progress.txt"   # the job appends `0.37` as it goes
progress_interval_s: 60
```

| printed | means |
| --- | --- |
| `0.42` | a fraction of one — 42% |
| `42%` | a percentage — 42% |
| `42`, `1`, `0` | **refused**: a bare integer could be either |

If you are dividing, `step / total` prints the decimal for free; in shell,
`echo "$((step * 100 / total))%"`.

Only the last line of the last 64 KiB is read, so the command may be a pipeline
that also logs. The estimate extrapolates from phase `main` alone, so a slow
`uv sync` is never charged to it. A progress command that exits non-zero, prints
nonsense or takes longer than 10 seconds is killed and ignored: once in
`log.txt`, and as `progress_error` in `gpuc status --json`.

`gpuc status` shows an `eta` on each running job tagged with where it came from —
`(42%)` measured, `(est)` your guess — an `est` on each queued job, and, on a
host with no free card, when the next is expected:

```
  running lego-s4 (20260915-120000-a1b2c3) phase=main 1h36m util 98% gpu=0 eta 3h20m (37%)
  queued  sweep (20260915-130000-d4e5f6) prio=50 est 6h00m starts in ~3h20m
  free    next card in ~3h20m (20260915-120000-a1b2c3)
```

A queued job's `starts` is the dispatch rule run forward over the estimates the
host has, and is absent rather than invented: a job whose turn depends on one
that estimated nothing has no start time, and neither does anything on a
draining host. A job waiting for more than one card says so (`needs 2 gpus`).

## Automatic preemption

`auto_preempt: true` lets the host stop a job whenever that lets a job queued at
a **lower** `priority` number start right away, and queue it again as its next
attempt. It re-runs from the start in the workdir it left behind, so mark work
that is cheap to repeat — a sweep point, an eval, a job that checkpoints — and
not a run whose `setup:` would trip over its own leftovers.

A job is stopped only when that starts a waiting one immediately, and never
while a host is draining. A borrowed card is never freed for a job that did not
ask to borrow. The stopped job is queued again at its own priority, behind the
job it made room for, and there is no limit on how often one job gives way.

`gpuc status` shows `auto-preempt` on those jobs, and the dispatcher log and the
job's own log name the job each preempt made room for.

## What gets synced to the host

`gpuc submit` rsyncs `git ls-files --cached --others --exclude-standard`: every
tracked file **and** every untracked one git would keep. `.gitignore` is obeyed,
so venvs and caches stay home; files in the index but deleted on disk are dropped
rather than sent. One line says what happened:

```
syncing 43 files (2 modified, 1 untracked, ignoring .gitignore'd)
```

Alongside the workdir go `uncommitted.patch` — `git diff HEAD -- .`, untracked
files included, taken without touching your staging area — and `source.json`
with the commit, branch, origin and submitting directory.

`--no-git` rsyncs a directory that is not a repository, minus `.venv`,
`__pycache__`, `.git`, `*.pyc`, `node_modules` and `.uv-cache`, with a warning:
nothing reads `.gitignore` in that mode, and **`gpuc requeue` cannot rebuild a
`--no-git` workdir**.

## Commands

**`gpuc submit <job.yaml|-> --host NAME`** — validate, sync the workdir, deliver
secrets, enqueue. `--no-git` is above; `--no-bootstrap` enqueues without first
re-shipping the package to a host on an older build (see
[setup.md](setup.md#upgrading)); `--use-shared` is `use_shared: true` from the
command line; `--runpod` and its flags are [below](#runpod).

It then asks the host where the job landed:

```
job 20260915-233000-112233 queued on host spar (attempt 1)
  queue: position 2 of 5; starts in ~3h20m
  logs: gpuc logs 20260915-233000-112233 -f
```

Where the start time cannot be [projected](#job-length-estimates) the line says
why instead, and it is absent when the host could not be asked again — the job
is queued either way. A job the dispatcher got to first prints `dispatched
already; it is running now`. `gpuc reorder` prints the same line.

**`gpuc status`** — per host: kind, reachability, free cards, dispatcher
heartbeat, one line per owned card (`free` / `busy` / `UNAVAILABLE`) and per
[shared](#shared-gpus) one, the queue, running jobs with phase, elapsed time,
last util, the cards they hold (`gpu=2,3`) and any
[end-time estimate](#job-length-estimates), and recent finished jobs. Every job
is `name (job-id)`.

```
host local [local]  gpus 0/1 free (driver 580.173.02)
  dispatcher 0s ago
  gpu     [0] busy NVIDIA GeForce RTX 3060 Ti 8 GB
  running lego-s4 (20260915-231241-f880d9) phase=main 27m util 100% gpu=0 eta 45m (37%)
  done    hello (20260915-074344-1d4db4) succeeded 15h ago
host spar [ssh]  gpus 0/2 free (driver 535.309.01)
  dispatcher 2s ago
  gpu     [2] busy NVIDIA A40 45 GB
  gpu     [3] busy NVIDIA A40 45 GB
  shared  [4] free NVIDIA A40 45 GB
  shared  [5] IN USE NVIDIA A40 45 GB (somebody else: 21504 MiB, 98% util)
  running paper-diff (20260915-222409-7a2b60) phase=main 1h16m util 100% gpu=2
  running paper-plain (20260915-224057-9f10c3) phase=main 59m util 100% gpu=3
  queued  sweep (20260915-233000-112233) prio=50 est 6h00m
```

On a `shared` line, `free` means gpuc would take the card right now, `busy` means
one of *our* jobs has it, and `IN USE` means somebody else does, with their
memory and utilization.

`--host H` narrows it; `--recent N` (default 5) and `--since 24h|7d|90m` (a bare
number means hours) choose how much of the finished list to show; `--all` adds
jobs only the local index and the S3 index know, which is how you find what was
on a host that lost its state, and is exit 1 if the S3 index could not be read;
`--json` is [below](#exit-codes-and---json). Card
UUIDs are in `gpuc host list`, and everything a host can say about itself is in
`gpuc host probe`.

**`gpuc logs <job-id> [-f] [-n N] [--host H]`** — tails `log.txt` on the host
(`-n` defaults to 200). If the host cannot produce it, gpuc says why — including
"was purged" when the whole job dir is gone — and falls back to the S3 mirror,
which needs `s3_bucket` set here **and** an `s3_prefix` for that job.

**`-f` follows until the job ends**, prints the job's outcome as its last line,
and exits 0 only if the job succeeded. A job that has already finished prints
its tail and exits; a job still queued is followed until its log appears.
A line or two of the runner's own cleanup can land after the follow has stopped;
the log itself always has them. `--follow-forever` streams until you stop it;
`--interval SECONDS` pins the poll (2s backing off to 30s by default). Neither
follow can be combined with `--json`, or with the other.

**`gpuc wait <job-id> [<job-id> ...] [--host H]`** — the same wait without the
log, for several jobs at once: one line per job as it ends, and exit 0 only if
**all** of them succeeded.

```sh
gpuc submit job.yaml --host spar --json | jq -r .job_id | xargs gpuc wait || echo "it did not work"
```

Killing either wait leaves the run alone: the host owns the job.

A host that stops answering does not end the wait: gpuc keeps asking for five
minutes, then reads the job's final state from the S3 mirror and says so. Only
if the mirror has nothing is that job exit 1. A reachable host whose dispatcher
is down is reported once and waited through — nothing there will start a queued
job until `gpuc host bootstrap <host>` restarts it. An id no host has is exit 4.
`--interval` is as above, and `--json` is [below](#exit-codes-and---json).

**`gpuc ssh <host|job-id> [--print] [-- CMD ...]`** — an ssh with gpuc's own key,
port, `known_hosts` and ControlMaster socket, none of which are in your
`~/.ssh/config`. A host name lands in its gpuc home; a job id lands in that job's
`workdir/`, falling back to the job dir once the workdir is gone. The words after
`--` are run by a **login bash** in that directory, so `gpuc ssh <job-id> -- 'ls
| wc -l'` is a pipeline, and gpuc exits with the remote command's own exit code.
`--print` prints the command line instead of running it. A `local` host gets your
own `$SHELL` with no ssh at all.

**`gpuc cancel <job-id>`** — see [how a job is killed](#how-a-job-is-killed).

**`gpuc reorder <job-id> --priority N`** — queued jobs only (a running or
finished job is exit 1). It prints the job's new queue position and start time,
and records the priority in the job's spec on the host and in its S3 mirror, so
`gpuc requeue` carries the move.

**`gpuc preempt <job-id>`** — stop a *running* job and queue it again as its next
attempt, so something more important can have its GPUs. It **starts over**:
`setup:` and `command:` run again from the top, in the workdir the stopped
attempt left behind, part-written checkpoints and all. The job never leaves its
host and nothing is re-synced from here.

It comes back at its own priority unless `--priority N` changes it, and at the
same priority it takes its own cards straight back — so **queue the job you are
making room for first, at a lower number, then preempt**. A preempt that would
only re-run the same job is refused (exit 1), as are queued and finished jobs,
which `gpuc reorder` and `gpuc requeue` are for. The stopped attempt is not
queued again if it had already ended on its own, was cancelled while stopping,
lost its workdir, or the host is draining. `auto_preempt: true` has the host do
this with no command at all ([above](#automatic-preemption)).

**`gpuc estimate <job-id> --minutes N`** — set (or `--clear`) a queued or running
job's `estimated_runtime_min`; see [job length estimates](#job-length-estimates).

**`gpuc requeue <job-id>`** — re-read the spec from the S3 mirror and submit it
again as attempt+1, with the workdir re-synced from your *current* directory. It
therefore **needs `s3_bucket`** and cannot rebuild a `--no-git` workdir.
`--host H` sends it somewhere else, `--runpod` provisions for it, and with
neither it goes back to the host the local index names. The new run gets its own
output namespace. A mirrored spec this build will not accept is refused rather
than queued to fail; submitting the job file again is the way round it. Keys
this build does not know are dropped rather than refused.

`--host` is optional on `logs`, `wait`, `cancel`, `preempt`, `reorder`,
`estimate` and `requeue`: the local job index is tried first, then every
registered host is asked whether it knows the id. An unknown job or host is
exit 4.

**`gpuc skill`** — prints the agent guide
([`skills/gpuc/SKILL.md`](../skills/gpuc/SKILL.md)) to stdout. `--install [DIR]`
writes it to `DIR/.claude/skills/gpuc/SKILL.md` instead (default: the current
directory) and refuses to overwrite without `--force`.

**`gpuc version`** — this build, its commit, and the commit this machine last
shipped to each bootstrapped host, marking the ones to re-bootstrap. It reads
the registry's cache and never touches a host; what a host is *running* is
`gpuc status`.

**`gpuc clean`**, **`gpuc pods`** and the host commands have their own sections
below and in [setup.md](setup.md). **`gpuc web serve`** is the
[web dashboard](#the-web-dashboard).

## The web dashboard

`gpuc web serve` is `gpuc status`, `gpuc host list` and `gpuc config show` on one
page, refreshed every 15 seconds, with a button for each of `gpuc cancel`,
`gpuc preempt`, `gpuc reorder` and `gpuc estimate` and a **Logs** panel that
tails `gpuc logs`. Every job links to where its `outputs:` went, to its W&B run
when the job's `env` names `WANDB_ENTITY` and `WANDB_PROJECT`, and to its
mirrored log. The links are what the job declared and are never checked: an
`outputs not uploaded` flag beside one means the link is empty.

```sh
gpuc web set-password          # once; prompts twice, stores a bcrypt hash 0600
gpuc web serve                 # http://127.0.0.1:8646/
gpuc web serve --bind 0.0.0.0 --port 8646   # reachable from other machines
gpuc web serve --bind 0.0.0.0 --install     # the same, as a systemd --user service (see setup.md)
```

Every page and every API document is behind that one password, and the server
refuses to start until one is set. Sessions live in the server's memory, so a
restart logs everyone out. There is **no TLS**: bind to localhost or a VPN
interface, or put it behind a TLS-terminating proxy.

The API is the same `--json` documents, over plain HTTP once the session cookie
is held:

| endpoint | the document |
| --- | --- |
| `GET /api/status?host=H&recent=N&since=24h` | `gpuc status --json`, plus `gathered_at` |
| `GET /api/hosts` | `gpuc host list --json` |
| `GET /api/config` | `gpuc config show --json` |
| `GET /api/version` | `gpuc version --json` |
| `GET /api/jobs/<id>/logs?lines=N&host=H` | `gpuc logs --json` (no `-f`; the page re-fetches the tail instead) |
| `POST /api/jobs/<id>/cancel` `{host?}` | `gpuc cancel --json` |
| `POST /api/jobs/<id>/reorder` `{priority, host?}` | `gpuc reorder --json` |
| `POST /api/jobs/<id>/preempt` `{priority?, host?}` | `gpuc preempt --json` |
| `POST /api/jobs/<id>/estimate` `{minutes}` or `{clear: true}` | `gpuc estimate --json` |

A failure is the same `{schema_version, error, exit_code}` document the CLI
prints, with the exit code mapped onto the status: 2 is 400, 3 is 503, 4 is 404,
1 is 500; a request with no session is 401.

## RunPod

```sh
export RUNPOD_API_KEY=...
gpuc submit job.yaml --runpod --gpu A40 --max-price 0.60
gpuc pods                   # every pod with our prefix: cost, util, age, which host it is here
gpuc host add rented --pod <pod-id>   # drive a pod another machine rented
```

The same flags work on `gpuc submit` and `gpuc requeue`:

| flag | default | meaning |
| --- | --- | --- |
| `--gpu A40[,RTX4090]` | required with `--runpod` | catalog names, matched case-insensitively against both the short name and the catalog id; cheapest match wins |
| `--gpu-count N` | `1` | GPUs on the pod; the spec's `gpus:` must fit in it |
| `--min-vram GB` | none | skip offers with less VRAM per GPU |
| `--max-price USD` | none | **whole pod** per hour, so at `--gpu-count 2` it is compared against twice the per-GPU price |
| `--cloud secure\|community\|any` | `secure` | which tier to buy from; community is cheaper and less reliable; `any` merges both and sorts by price |
| `--cuda-min X.Y` | `12.8` | the floor sent to `create`; passing it explicitly also filters the catalog query |
| `--idle-min N` | `15` | terminate the pod once its queue has been empty this long |
| `--disk GB` / `--image REF` | `config.toml` | container disk and pod image |
| `--no-reuse` | reuse is on | always create a new pod |
| `--name-hint TEXT` | `job` | goes into the pod name after the prefix |
| `--health-args "..."` | none | extra flags for the on-host health check, e.g. `--min-mbps 0.1` |

**Offers.** Matching offers are tried cheapest first: create, wait for a direct
SSH endpoint, bootstrap, health check, enqueue. Any failure terminates that pod
and moves to the next offer, all inside a **15-minute ceiling**.

**Reuse** is the default: a registered `runpod` host whose recorded offer still
matches the request, that owns at least `--gpu-count` cards, whose pod is
`RUNNING` with a dispatcher that beat in the last 30 s, and that is not
draining. A reused pod keeps the image, disk and `--idle-min` it was created
with. `--no-reuse` always creates a new one.

**The pod is not tied to the machine that bought it.** Its cards, its mirror and
its idle timer live in its own `config.json`, so `gpuc host add <name> --pod
<pod-id>` registers it from a second machine.

**This machine owns a pod only until it proves healthy.** Every exit from
provisioning between `create` and the final registry write, Ctrl-C included,
terminates the pod first. A terminate that fails is reported loudly and the
entry kept, so `gpuc status` and `gpuc pods` still show the pod.

<a name="auto-down"></a>
**Auto-down.** The pod terminates itself once nothing is running and the queue
has been empty for `--idle-min`. It drains first — retrying unconfirmed outputs
and mirroring every job's log and state — but only a failed *terminate* stops
the shutdown, which then retries in 10 minutes. There is no overall pod
lifetime; a job's own cap is `max_runtime_min`.

<a name="pods"></a>
**After that the pod owns itself, and nothing here watches it.** A pod whose
dispatcher dies, or whose provisioning client was killed before it could clean
up, bills until a person ends it. **`gpuc pods`** shows every pod in the account
with our prefix — name, id, status, GPU, `$/h`, CUDA, age, util, the `HOST` name
it is registered under here, and how long ago its dispatcher last beat — with
the hourly total, plus other people's pods by name only, never touched. A pod
with a fresh heartbeat will end itself; one with none, and nothing running, will
not: `gpuc host set <host> --idle-min 0` hurries one that still has a
dispatcher, and `gpuc host terminate` ends either. `--no-heartbeat` skips the
per-pod ssh check. A pod registered nowhere here is named at the bottom with how
to adopt or end it, and one younger than the 15-minute provisioning ceiling is
flagged as possibly still being set up by a `submit` elsewhere. `gpuc pods`
never terminates anything.

<a name="terminate"></a>
**`gpuc host terminate <host|pod-id|pod-name> [--force]`** — end a rental now
and forget it here.

**Without `--force`, the host has to say it is idle.**

| what happened | the refusal (exit 1) says |
| --- | --- |
| a job is running or queued, or a finished job's outputs are not confirmed uploaded | which jobs, and both ways on |
| the host did not answer | why, and that nothing is known about its queue |
| the pod is not registered | that, plus `gpuc host add --pod` to adopt it first |

```
gpuc-sweep-3f21aa (rzk1n8x) is not idle:
  running   lego-s4 (20260915-231241-f880d9)
  unsynced  hello (20260915-074344-1d4db4) (its outputs are not confirmed uploaded)
Terminating now kills those jobs and loses anything not already uploaded.
  gpuc host set gpuc-sweep-3f21aa --idle-min 0   let it finish, then stop by itself
  gpuc host terminate gpuc-sweep-3f21aa --force   end it now anyway
```

`--force` does not ask at all, and is the way past all three.

A pod the provider calls dead — `TERMINATED`, `EXITED`, `ERROR` or missing — is
never refused over: the command ends what is left of the rental and drops the
registry entry with no flag.

The terminate is retried and confirmed with the provider before the registry
entry goes. One that still cannot be confirmed is exit 1 and **keeps** the
entry, so a pod that may be billing stays in `gpuc status` and `gpuc pods`. A
`local` or `ssh` host has no rental to end and is refused; `gpuc host remove`
forgets one of those.

A rental that ended itself — a pod the provider reports missing or `TERMINATED`
— is **forgotten where it is found**: `gpuc status`, `gpuc host bootstrap --all`
and the next `submit --runpod` reuse pass each drop the registry entry and say
so. A pod the provider still has but has stopped (`EXITED`, `ERROR`) is a host
nothing can run on: `POD GONE` in `gpuc status`, exit 1, and it stays until
`gpuc host remove <name>`.

## How a job is killed

`gpuc cancel` records the request in the job's state, which the runner checks
before every phase and on every poll. The runner stops the job's systemd scope,
SIGTERMs its process group and SIGKILLs it 15 s later, then runs the final sync
and writes the final state. A queued job is cancelled on the spot. If the
runner does not act the dispatcher escalates: the scope and a SIGKILL of the
job's group at 15 s, a SIGTERM of the runner at 30 s, a SIGKILL of its group at
45 s. `gpuc preempt` is the same request with a different ending.

Each phase runs in a transient `systemd --user` scope where the host has one and
in its own process group where it does not (`isolation: cgroup` or `pgid` in
`gpuc status`). Under `pgid` — every RunPod pod, most shared boxes — a
grandchild that double-forks (`setsid`, `nohup`, a daemonising server) survives
the kill and holds its GPU. It cannot leave a cgroup.

A preempted job is briefly `failed: preempted` before it is queued again as the
next attempt. Its workdir and secrets file are kept whatever `cleanup:` says.

Utilization is sampled on the assigned cards every 30 s **during phase `main`
only**, so downloads and compiles in `setup` never show as idle. A sample
nvidia-smi could not produce is unknown, never 0%.

Every `failed: <reason>`:

| reason | what happened |
| --- | --- |
| `exit <N>` | `command` exited non-zero and nothing else killed it |
| `setup` | the `setup` phase exited non-zero |
| `gpu-assert` | an assigned GPU (index or UUID) is not present in the host's `nvidia-smi`, or the job was started with no GPU assigned at all |
| `gpu-preflight` | a real GPU op inside the job's venv failed, or `device_count()` did not match `gpus:` — usually a CPU-only torch |
| `sync-preflight` | the uploads the job would do at the end cannot work (no `aws`/`hf`, a missing secret, an unwritable bucket or repo) |
| `timeout` | `max_runtime_min` elapsed |
| `preempted` | `gpuc preempt`, or the job's own `auto_preempt`, stopped this attempt; the job is queued again as the next one, and this is the record of the attempt that was stopped |
| `terminated` | the runner itself was signalled (and the job was not cancelled) |
| `sync` | the final upload failed; the run itself may have been fine. A succeeded job becomes `failed: sync`; a job that was already over for a reason of its own keeps that reason and lists `sync` in its `problems` |
| `no-outputs` | an `outputs:` path was never written, or holds only files that came with the checkout. A problem beside the reason, the same way |
| `bad-spec` | the queued spec could not be read |
| `needs N GPUs, host owns M` | the host's ownership shrank after the job was queued. On a host with [shared cards](#shared-gpus) it counts the ones this job asked for, and says so when it asked for none |
| `needs at least 1 GPU, asked for 0` | the spec on the host asks for no card. `gpuc submit` refuses that, so it is a spec an older build queued or one edited by hand |
| `spawn-failed` | the dispatcher could not start a runner process |
| `runner-died` | the runner vanished without writing final state; the dispatcher kills anything it left behind before freeing its GPUs |

`cancelled` is a status of its own, not a failure.

## Exit codes and `--json`

| code | meaning |
| --- | --- |
| 0 | everything the command was asked to do worked. For `gpuc wait` and `gpuc logs -f`, the job succeeded |
| 1 | something failed: a transport, a provider, a refused submit, a host that could not be read, a `clean` the host reported errors for, a host `host bootstrap --all` failed on. For `gpuc wait` and `gpuc logs -f` it is the **job** that did not succeed |
| 2 | usage: a bad flag, a missing required one, a bad `--since` |
| 3 | local state is unreadable (`hosts.json` or `config.toml`), so the answer is **unknown** |
| 4 | the job or host named on the command line does not exist |
| 130 | a Ctrl-C |

**A failure never costs you the rest of the answer.** An unreachable host is
reported in its own block, every other host is reported in full, and the command
exits 1. A registry entry this build cannot parse is the same: a warning on
stderr, the entry written back untouched, and exit 1 from the commands reporting
on every host (`status`, `host list`, `version`) — not from one given a single
host or job. A rental whose pod has ended is not a failure at all; gpuc forgets
that host. Only a `hosts.json` that cannot be parsed at all is exit 3, which
prints the error, the path, and that a `.bak` was kept.

```sh
gpuc status --json | jq '.hosts[] | {name, reachable, running: (.running | length)}'
gpuc status --json | jq '[.hosts[].running[] | {job_id, name, phase, elapsed_s, util}]'
```

```json
{
  "schema_version": 1,
  "hosts": [
    {
      "name": "gpubox",
      "kind": "ssh",
      "reachable": true,
      "pkg_commit": "8f1c2d0a9b34",
      "dispatcher": { "alive": true, "heartbeat_age_s": 2.0 },
      "provider_util": null,
      "gpus": [
        { "index": 0, "uuid": "GPU-8064...", "name": "NVIDIA A40", "vram_mib": 46068,
          "busy_job": "20260915-120000-abc123" }
      ],
      "shared_gpus": [
        { "index": 4, "uuid": "GPU-aaaa...", "name": "NVIDIA A40", "vram_mib": 46068,
          "busy_job": null, "memory_mib": 0.0, "utilization_pct": 0.0, "unused": true }
      ],
      "queued": [
        { "job_id": "20260915-130000-d4e5f6", "name": "sweep", "status": "queued",
          "priority": 50, "gpus_requested": 2, "use_shared": true, "gpus": [],
          "estimated_runtime_min": 360.0,
          "starts_in_s": 12060.0, "starts_at": "2026-09-15T16:31:00+00:00" }
      ],
      "running": [
        { "job_id": "20260915-120000-abc123", "name": "lego-s4", "status": "running",
          "reason": null, "phase": "main", "priority": 50, "elapsed_s": 4210.5, "util": 96.0,
          "progress_pct": 37.0, "eta": "2026-09-15T16:31:00+00:00", "eta_s": 12060.0,
          "estimated_runtime_min": 480.0, "progress_error": null,
          "gpus": ["GPU-8064..."], "gpus_requested": 1, "use_shared": false,
          "iso": "pgid", "ended_at": null,
          "outputs_pending": false }
      ],
      "finished": [],
      "errors": []
    }
  ],
  "errors": []
}
```

The job objects in `queued`, `running` and `finished` all carry the same fields.
Beyond what the example shows:

- `priority` (0–99, lower first) is on **every** job, and `queued` is already in
  dispatch order. Null only when the host did not say.
- `gpus_requested` is what the spec asked for; a queued job holds no `gpus` yet.
  `use_shared` is whether it may take one of `shared_gpus`. Null, like `priority`
  and `auto_preempt`, means the host did not say — never `false`.
- `starts_in_s` / `starts_at` are when a queued job's turn is expected (see [job
  length estimates](#job-length-estimates)); null for anything else, and for a
  queued job whose turn cannot be dated, when `starts_unknown` says why. The
  host works both out; a client only prints them.
- `problems` lists what else went wrong on the way out of a finished job
  (`sync`, `no-outputs`) beside its `reason`. `upload_errors` is the last
  failure standing at each of the job's upload destinations; a **running** job
  with one is an output that is not reaching its destination, and its line
  says `UPLOAD FAILING`.
- `eta` is absolute and `eta_s` the same instant in seconds from now (negative
  once overdue); both null without a length estimate and once the job is
  finished. `progress_pct` survives the job; `progress_error` is why the last
  poll produced nothing.
- `attempt`, `started_at`, `exit_code`, `outputs_lost`, `workdir_bytes`,
  `outputs` (the spec's, as the host holds them) and `links` — one
  `{kind, path, target, url}` per place the results, W&B run or mirrored log can
  be opened, derived from what the job declared and never checked.
- A job's `util` is its **last** sample from the host's own nvidia-smi; a pod's
  `provider_util` is the provider's per-GPU reading for the whole pod, null
  elsewhere. Two measurements that will differ.

Per host: `target`, `draining`, `pod_gone`, `pod_terminated` (that pod is gone
for good, so the next `gpuc status` forgets the host), `pod` (the provider's
view of an ephemeral host's pod) and `pkg_commit`, the host's own answer for the build it
runs — `null` means it did not say, never "up to date". A card the host cannot
see appears in `gpus` as `{"owned_as": "3", "available": false}`; `shared_gpus`
has the same shape plus `unused` (no memory held, no work running) and
`busy_job` (one of *our* jobs has it), and a missing one is
`{"shared_as": "5", "available": false}`. `--recent` and `--since` apply to
`--json`; `--all` does not.

Rules for anything automated:

- **Key on `hosts[].running`.** It is the host's own answer; an empty list means
  the host said nothing is running.
- **Read queue order from `priority`**, not from `eta` or
  `estimated_runtime_min`: those say how long a job takes, not when it is taken.
- **Treat exit 3 as "unknown", never as "nothing is running".** So is a non-empty
  top-level `errors`, and so is `"reachable": false` for the host you care about.
- Unknown keys will be added over time; ignore the ones you do not know.

### `--json` everywhere else

`status`, `submit`, `requeue`, `logs`, `wait`, `cancel`, `preempt`, `reorder`,
`estimate`, `pods`, `version`, `clean`, `config show`, `config init`,
`host list`, `host probe`, `host add`, `host set`, `host bootstrap`,
`host clean`, `host remove` and `host terminate` take `--json`: **stdout is exactly one JSON
object**, it carries `schema_version`, and everything the text output would print
alongside it — progress, warnings, `note:` lines — goes to stderr. Exit codes are
unchanged by the flag. The commands without it have no answer to give: `ssh`
opens a shell, `skill` prints the guide, `web serve` runs a server, and
`web set-password` and `skill --install` write one file.

**A command that failed prints a document too**, so a caller parsing stdout is
never handed nothing at all:

```json
{ "schema_version": 1, "error": "no registered host knows job 20260915-120000-abc123.\n...",
  "exit_code": 4 }
```

`error` (singular) is the whole answer: the command did not do what it was asked.
A command line argparse itself rejects gets the same document, with the reason on
stderr where argparse wrote it. `errors` (plural) is different — per-host or
per-job trouble the command reported rather than stopped for.

| command | the document |
| --- | --- |
| `submit`, `requeue` | `{job_id, host, attempt, requeued_from, notes[], queue_position, queue_length, dispatched, starts_in_s, starts_at, starts_unknown}`. `requeued_from` is null on `submit`; `notes` are the text output's `note:` lines. The queue fields are the host's answer just after the enqueue: `queue_position` is 1-based in dispatch order, `dispatched` is true for a job the host started before we could look, and `starts_unknown` says why there is no start time (null when there is one). All of them are null when the host could not be asked again — never a reason to think the job was not queued |
| `logs` | `{job_id, host, source, location, lines[], notes[]}`. `source` is `"host"` or `"s3"` and `location` is the remote path or the `s3://` uri it was read from; `lines` is the log with no trailing newlines. **Not with either follow** (exit 2): the document is printed once and a follow is a stream, so `gpuc wait --json` is the JSON form of waiting |
| `wait` | `{jobs[], errors[]}`, printed once every job has ended. Each of `jobs[]` is that job's final state in the shape `status --json` gives a job, plus `host`, `source` (`"host"`, or `"mirror"` for a state read from S3 after the host went away) and `error`. **Check `error`, not `status`**: when it is not null, `status` is only the last thing its host managed to say — `"running"` for a host that vanished mid-run, null for a job nothing was heard about, which carries only `job_id`, `host`, `source`, `error` and a null `status`. Every `error` is in `errors[]` too. Under `"source": "mirror"` the `outputs_*` fields come from the mirrored state rather than the host's own check. The per-job outcome lines go to stderr. Exit 1 unless every job succeeded |
| `cancel` | `{job_id, host, status}` — the host's own word, `cancelled` for a queued job or `cancelling` for a running one |
| `preempt` | `{job_id, host, status, priority, warnings[]}`. `status` is the host's own word (`preempting`); `priority` is what it will be queued again at, which is the job's own unless `--priority` changed it. `warnings` carries a mirrored spec that could not be updated, exactly as `reorder` does |
| `reorder` | `{job_id, host, priority, warnings[]}` plus the same `queue_position`, `queue_length`, `dispatched`, `starts_in_s`, `starts_at` and `starts_unknown` as `submit`, so a move can be checked without a second call. `warnings` carries a mirrored spec that could not be updated, which means `gpuc requeue` would re-run the job at its old priority |
| `estimate` | `{job_id, host, estimated_runtime_min, status, warnings[]}`. `estimated_runtime_min` is what the job's state holds now (null after `--clear`) and `status` is the job's, since only a queued or running one can be set; `warnings` carries a `max_runtime_min` contradiction and a mirrored spec that could not be updated |
| `pods` | `{pods[], hourly_usd, others[], notes[]}`. Each pod is `{id, name, status, gpu_name, gpu_count, cost_usd_hr, cuda_version, age_s, created_at, gpu_utils[], host, heartbeat_age_s}`; `host` is the registry name this machine drives it under, null if none; `others` are pods without our prefix, `{id, name, status}` only, because we never touch them |
| `version` | `{version, commit, source, dirty, python, executable, hosts[], errors[]}`, each host `{name, pkg_commit, seen_at, current}`. `pkg_commit` here is the commit the host was running when this machine last read it, not what it runs now — that is `status --json`'s `pkg_commit`. Exit 1 for an entry that could not be parsed, 3 if the whole registry is unreadable |
| `config show` | `{config_file, config_file_exists, state_dir, settings{}, notes[]}` — the effective settings, file or not |
| `host list` | `{hosts[], errors[]}` — each registry entry: the address (`name`, `kind`, `ssh`, `port`, `gpuc_home`, `persistent_root`, `pod_id`), the host's own config as last read (`gpus`, `s3_prefix`, `env`, `cache_dir`, `idle_minutes`, `retention_days`, `pkg_commit`) flattened beside it with `config_seen_at`, the raw `cache` it came from, plus `remote_home`, `ephemeral` and `warnings[]`. Nothing here asks the host. The host's `env` is reported by **name only** (`{"HF_TOKEN": "<set>"}`). A skipped entry is an `errors` string, not a host, and exit 1. Exit 3 if the whole registry is unreadable |
| `host probe` | `{host, sections{}, driver_version, has_nvidia_smi, gpus[], assigned_gpus[], assigned_missing[], home_fs_type, home_is_overlay, persistent_root, uv_cache{}, notes[]}`. `gpus` is **every** card the host has whatever `--all-gpus` said, each one `{uuid, name, vram_mib, index, assigned}`; `assigned_gpus` is this host's `--gpus` as registered and `assigned_missing` the entries in it no card answered to. `sections` is the probe script's raw output section by section, so anything this build does not interpret is still there |
| `clean` | `{host, dry_run, purge, freed_bytes, removed[], skipped[], purged[], purge_skipped[], incoming_removed[], verified[], notes[], errors[]}`. Job objects are `{job_id, status, bytes, age_days, mirrored_at, mirror}`, plus `why` on the skipped ones and `forced` on a purged job that had no confirmed backup; `incoming_removed` names job dirs a submit never finished |
| `host add`, `host set` | the host as `host list --json` reports one entry (the address, the host's own config flattened beside it, `cache`, `remote_home`, `ephemeral`), as the registry holds it once the command is done, plus `adopted` (the host already had a config, which `add` took as it stood), `config_path` (that config on the host), `changes[]` (one line per config field this command wrote through to the host, empty when it held that already) and `warnings[]` (`host list`'s re-bootstrap note, and for `add` a host that owns no card or a pod nothing has bootstrapped). `host set` adds `address{}`: the fields it changed here rather than on the host (`persistent_root`, `gpuc_home`), by name and new value |
| `host remove` | `{host, kind, pod_id, notes[]}` — what was forgotten here. Nothing on the host changes, and a rental is **not** terminated: it bills until it idles out, and `notes` says so, naming `host terminate` |
| `host terminate` | `{host, pod_id, pod_name, pod_status, cost_usd_hr, checked, running[], queued[], outputs_pending[], terminated, forgotten, notes[]}` — `pod_status` and `cost_usd_hr` are what the provider said **before** the terminate, so `"TERMINATED"` with `terminated: false` is a pod that was already gone. `checked` is whether the host itself answered: false means `running`, `queued` and `outputs_pending` are empty because nothing could be asked, not because there was nothing there. `forgotten` is whether the registry entry went. A refusal is the error document, exit 1 |
| `host bootstrap` | `{host, home, files, pkg_commit, dispatcher_pid, warnings[]}` — the gpuc home the package went to, how many files, the commit the host now runs, the dispatcher started, and every warning the run printed. With `--all`: `{hosts[], total, bootstrapped[], failed[], gone[], unreadable[], interrupted, errors[]}` — one `hosts[]` entry per registered host, `{name, outcome, error, ephemeral}` plus the single-host fields (null unless it was bootstrapped). `outcome` is `bootstrapped`, `failed` (with `error` saying why), `gone` (a rental the provider no longer has, forgotten rather than failed), `interrupted` (the host a Ctrl-C landed in) or `not_attempted` (the ones after it); `unreadable` names entries this build could not read and so never tried. Exit 1 if any host failed or the run was interrupted; a registry that stops being readable mid-run is the error document and exit 3 |
| `host clean --uv-cache` | `{host, cache_dir, before, after, before_bytes, after_bytes, freed_bytes}` — the cache pruned and its size either side, in bytes and as a human-readable string derived from them. All four size fields are null when `du` on the host failed |
| `config init` | `{config_file, existed}` — the path written, and whether a file was already there (only ever true with `--force`; without it an existing file is refused, exit 1) |

```sh
gpuc submit job.yaml --host gpubox --json | jq -r .job_id
gpuc logs "$id" --json | jq -r '.lines[-20:][]'
gpuc wait "$id" --json | jq -r '.jobs[] | "\(.job_id) \(.status) \(.reason // "")"'
gpuc pods --json | jq '[.pods[] | select(.host == null) | .name]'
gpuc clean --host gpubox --all-finished --dry-run --json | jq .freed_bytes
gpuc host bootstrap --all --json | jq -r '.hosts[] | "\(.name) \(.outcome) \(.error // "")"'
```

## Cleanup and retention

A job's `workdir/` is the rsynced code *and* whatever the job builds in it —
usually a venv, and a torch venv is about 6.5 GB. It is the only part of a job
dir gpuc deletes by default: `spec.json`, `state.json` and `log.txt` stay, so
`logs`, `status` and `requeue` keep working on a cleaned job.

**Per job, by the runner**, after the final sync and state write:

| `cleanup:` | succeeded | failed | cancelled |
| --- | --- | --- | --- |
| `on_success` (default) | removed | **kept** | **kept** |
| `always` | removed | removed | removed |
| `never` | kept | kept | kept |

No policy touches a job that is not finished. A workdir the policy keeps is still
swept once it is `--workdir-days` old (below); `cleanup: never` opts out of that
too.

**After the fact: `gpuc clean --host H`.**

```sh
gpuc clean --host gpubox --all-finished --dry-run   # what would go, and how big
gpuc clean --host gpubox --all-finished             # every succeeded/failed/cancelled job
gpuc clean --host gpubox --older-than 7             # only jobs that ended over 7 days ago
gpuc clean --host gpubox --only 20260101-120000-ab12,20260101-130000-cd34
```

|  | `clean` | `clean --purge` |
| --- | --- | --- |
| `workdir/` (code, venv, outputs) | removed | removed |
| `spec.json`, `state.json`, `log.txt` | **kept** | removed |
| the job's `secrets/<id>.env`, if any is left | kept | **removed** |
| job dirs under `incoming/` a submit never finished | removed | removed |
| a stray queue marker for the job | — | removed |
| running or queued jobs, or ones with unreadable state | never touched | never touched |

`--only ID[,ID...]` replaces `--all-finished` and `--older-than`: exactly those
jobs, however recently they ended, and no `--yes` needed for `--purge`, whose
workdir sweep is scoped to the same ids. It does not waive the preconditions
below. An id no job dir matches refuses the whole
selection (exit 1, nothing removed); an empty `--only` is exit 2.

`--purge` removes the whole `jobs/<id>/` of finished jobs and defaults to
`--older-than 7`. It has two preconditions, both read from the job's own
`state.json` on the host:

- **backed up**: the job's upload record for its mirror says the host's own
  final upload of `log.txt` and `state.json` returned 0. Otherwise the skip
  reason is `not backed up: no s3_prefix on this host` or `not backed up: final
  upload failed`.
- **outputs confirmed**: every destination of every declared output has a
  record of the last upload succeeding, or the spec declares no `outputs:`, or
  the workdir is already gone, or nothing was ever written under the declared
  paths. Otherwise `outputs not confirmed uploaded`. Anything unreadable counts
  as content.

`--force` overrides those two and nothing else, per job. `--verify` HEADs each
candidate's mirrored `log.txt` with your own credentials and purges only what
answered; without it the host's own record is trusted. `--purge
--all-finished` is an age horizon of zero, so it needs `--yes` (or `--dry-run`).

**Automatic, by the host: two horizons.** The dispatcher reclaims disk at
startup and then at most once an hour. On a **non-ephemeral host the dispatcher
only lives while there is work**, so this happens on your next submit rather
than on a timer.

| | `--workdir-days N` | `--retention-days N` |
| --- | --- | --- |
| default | `1` | none |
| takes | `workdir/` only | the whole `jobs/<id>/` |
| needs a mirror | no | yes, and never forced |

`--workdir-days` is the one that keeps a busy host from filling up: it takes the
checkout and the venv and leaves everything `logs`, `status` and `requeue` need.
It refuses two jobs `gpuc clean` will take if you name them: one whose spec says
**`cleanup: never`**, and one whose **`outputs:` have not reached S3 or HF** —
the jobs `gpuc status` flags as `outputs not uploaded`. An unreadable
`spec.json` is skipped too. `--workdir-days ''` turns the horizon off.

`--retention-days` is the purge, and deletes the record of the run, so it is
opt-in and only ever acts on jobs whose log and state the host confirmed
mirrored. A purge pass also sweeps workdirs at its own horizon under the same
two refusals; with both set, the shorter horizon wins. Either pass removes a
job dir under `incoming/` that a `gpuc submit` started and never finished,
once it has been untouched for an hour.

Both live in the host's own `config.json`, and `gpuc host set <name>
--workdir-days N` writes through to it, so it takes effect without a bootstrap.
`gpuc host add` on a box that already has a `config.json` adopts what is there
rather than imposing the default.

**Unconfirmed outputs.** `gpuc status` flags a finished job that *produced*
`outputs:` which never reached S3 or HF as `outputs not uploaded`. A job that
only declared outputs and never wrote them is not flagged. An ephemeral host
retries them while it drains (three tries a minute apart, five minutes at most)
and, if they still fail, records `outputs_lost` and terminates anyway. Such a
job shows `OUTPUTS LOST`, and only re-running it recovers the results.

`gpuc status` prints one line per host once finished workdirs hold more than
1 GiB, with the `gpuc clean` line to run. **The figure is what deleting them
would give the filesystem back, not what `du` says they hold** — uv hardlinks or
reflinks a venv out of its wheel cache, so most of those bytes stay when the
workdir goes. It is measured once, when the job ends. On a filesystem that
snapshots your home it reports close to nothing, and the delete really does free
nothing until the snapshot expires.

## Troubleshooting

| symptom | what it means | what to do |
| --- | --- | --- |
| `status` says `dispatcher DOWN` | nothing holds the host's lock, or its heartbeat is over 30 s old | `gpuc host bootstrap <host>` (idempotent); any `gpuc submit` also restarts it |
| provisioning gives up with "no direct SSH endpoint" | RunPod never exposed port 22 within the 15-minute ceiling — usually a bad placement | the pod was already terminated; re-run the submit, or widen `--gpu` / `--cloud any` |
| `ssh ... cannot create its ControlMaster socket` | the socket path would be over the 100-byte limit gpuc enforces | point `XDG_RUNTIME_DIR` at a short directory, or unset it to use `/tmp/gpuc-<uid>` |
| bootstrap fails with "host health failed" | the driver, disk or network check on the host said no | read the named check; fix the host (free disk, load the driver) and re-run bootstrap |
| job is `failed: gpu-preflight` | torch in the job's venv has no working CUDA, or sees the wrong number of devices | check the torch build against the host's driver (`gpuc host probe`), and that `gpus:` matches what the job expects |
| job is `failed: sync` (or lists `sync` in its problems) | the final upload failed; the run itself may have been fine | check the tail of `gpuc logs <job-id>`; usually a missing `secrets:` entry for the destination, or no `aws`/`hf` on the host (re-run bootstrap) |
| job is `failed: sync-preflight` | the uploads the job would do at the end cannot work | the log names the exact command and error; fix the credential or destination, or add `hf_create: true`, then re-submit |
| job is `failed: no-outputs` | the `outputs:` path was never written, or holds only what came with the checkout | check the job writes there, relative to the workdir; use a `{job_id}` subdirectory |
| a host is out of disk, or `status` shows a `disk` line | finished jobs' workdirs (usually venvs) are still there | `gpuc clean --host <host> --all-finished`, and set `cleanup: always` on jobs you never need to inspect |
| one job dir is stuck and the rest of the host is fine | that job's mirror genuinely failed, so an age-based purge either misses it or sweeps up everything else | `gpuc clean --host <host> --purge --only <job-id>` (add `--force` to accept losing its only copy); it leaves every other job alone |
| `clean --purge` skips everything as "not backed up" | the host has no `s3_prefix`, so nothing is mirrored and deleting a job dir would lose its log | `gpuc host set <host> --s3-prefix s3://bucket/gpuc/<host>` (it reaches the host at once), or accept the loss with `--force` |
| `status` says a job's `outputs not uploaded` | the final upload of its `outputs:` failed, so the results exist only on that host | copy them off, or `gpuc requeue <id>`; neither the purge nor the automatic workdir sweep will remove it until they are confirmed |
| a job is `OUTPUTS LOST` | an ephemeral host drained, retried three times and gave up before terminating | the results are gone; fix the credential or bucket, then `gpuc requeue <id>` |
| `--retention-days` never deletes anything | the dispatcher only lives while a non-ephemeral host has work, and it never purges an unmirrored job | check `gpuc status` for `not backed up`, and remember the sweep runs on the next submit |
| finished workdirs pile up anyway | `cleanup: on_success` keeps a failed or cancelled job's workdir on purpose, and the host's own sweep only runs once they are `--workdir-days` old (and only while its dispatcher is alive) | `gpuc clean --host H --all-finished --dry-run` to see them now; `gpuc host set H --workdir-days N` to change the horizon (it reaches the host at once) |
| one workdir never gets swept | the automatic sweep refuses `cleanup: never` and any job whose `outputs:` are still only on the host | `gpuc status` names the second kind; `gpuc clean --host H --only <id>` takes either |
| `requeue` refuses, or rebuilds the wrong code | it reads the spec from S3 (so `s3_bucket` must be set) and re-syncs the workdir from your current directory | run it from the right checkout; a `--no-git` workdir cannot be rebuilt from a commit |
| `--idle-min` did nothing on a shared box | it only applies to ephemeral (RunPod) hosts; `local` and `ssh` hosts never terminate themselves | nothing to do; use `--retention-days` for disk, not `--idle-min` |
| `uv sync` re-downloads torch on every job | uv's cache is on a different filesystem from gpuc home, so it copies instead of linking | `gpuc host bootstrap <host>` (it sets `UV_CACHE_DIR` for you), or pin one with `--cache-dir` |
| `gpuc` exits 3 and names `hosts.json` | the registry could not be parsed at all; a `.bak` was kept beside it | fix or delete the file, then re-add hosts; nothing was written over |
| a warning names one skipped host entry | that entry did not validate; every other host still works and is written back untouched | fix it by hand, or `gpuc host add <name> --ssh ...` to connect to that host again |
| `status` warns `host X is running gpuc <sha> and this machine has <sha>` | the host was last bootstrapped from a different build than this one, in either direction | `gpuc host bootstrap X`, or `gpuc host bootstrap --all` for every host at once — safe while jobs run; the new dispatcher adopts them |
| `status` warns `host X has gpuc <sha> on disk but its running dispatcher was started on <sha>` | the dispatcher outlived the package under it, so nothing shipped since is in effect. A newer dispatcher normally takes over by itself | `gpuc host bootstrap X` — safe while jobs run; the new dispatcher adopts them |
| `status` says `POD GONE` | the provider reports the pod stopped, so nothing can be run on it. A pod that is terminated or missing is the end of the rental, and `status` forgets that entry as it prints it | nothing for a rental that ended; for a stopped pod the provider still has, `gpuc host terminate <name>` ends it and `gpuc host remove <name>` only forgets it |
| `gpuc pods` shows a pod with no heartbeat and nothing running | its dispatcher died, or the machine that was provisioning it was killed before it could clean up; it will never idle out | `gpuc host terminate <pod-id> --force`. A pod that still answers ssh can be re-bootstrapped instead (`gpuc host add <name> --pod <id>`, then `gpuc host bootstrap <name>`) |
| `host terminate` says the pod could not be confirmed gone | the provider refused or did not answer the terminate, three times | the registry entry is kept and the pod may still be billing: run it again, and check the RunPod console if it keeps failing |
| everything on a host is suddenly gone | the container restarted and `$HOME` was on the overlay | the runbook in [setup.md](setup.md#hosts-whose-home-is-wiped-on-restart) |
