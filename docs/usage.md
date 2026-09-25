# Usage

Installing gpuc and registering hosts is [setup.md](setup.md). `gpuc --help`, and
`--help` on every subcommand, is the authoritative list of flags; this page is
what they mean together. A workflow of many dependent jobs is
[snakemake.md](snakemake.md).

## The job spec

A job is one YAML (or JSON) document. `gpuc submit job.yaml --host H` runs it in
the working directory you submit from. `job.example.yaml` in the repo root is a
commented example; `-` as the file name reads the spec from stdin.

| field | default | meaning |
| --- | --- | --- |
| `command` | **required** | run in `workdir/` as phase `main`; blank is refused at submit |
| `name` | `""` | a label for `status`; not an identifier |
| `setup` | none | run first, as phase `setup` |
| `python` | `uv run --no-sync python` | how the GPU check before `main` runs Python inside the job's environment; a repo that is not a uv project names its own (`.venv/bin/python`) |
| `gpus` | `1` | GPUs to assign, at least 1; more than the host can ever provide is refused at submit |
| `use_shared` | `false` | may also use the host's [shared GPUs](#shared-gpus); `gpuc submit --use-shared` sets it |
| `env` | `{}` | plain environment, applied after the host's `--env` |
| `secrets` | `[]` | names read from your shell at submit and delivered to the host as `~/.gpuc/secrets/<job-id>.env` (0600); one missing from your shell is refused |
| `outputs` | `[]` | `{path, s3}` and/or `{path, hf, hf_path, hf_create}`, or `{path}` alone to [keep it on the host](#kept-outputs); `path` is relative to the workdir |
| `sync_interval_s` | `180` | background upload cadence; minimum 10 |
| `priority` | `50` | `0`–`99`, lower first; see [priority](#priority-is-not-advisory) |
| `max_runtime_min` | none | wall clock from the runner's start; over it the job is `failed: timeout` |
| `estimated_runtime_min` | none | your guess, measured the same way; see [estimates](#job-length-estimates) |
| `progress_command` | none | run in the workdir during `main`; its last stdout line is the job's progress |
| `progress_interval_s` | `60` | how often to run it; minimum 5 |
| `auto_preempt` | `false` | see [automatic preemption](#automatic-preemption) |
| `requires` | `{}` | e.g. `cuda_min: "12.8"`; informs provisioning only |
| `cleanup` | `on_success` | when `workdir/` is deleted: `on_success`, `always`, `never` |
| `requeued_from` | none | set by `gpuc requeue`, never by you |

Unknown keys are refused at submit.

**Every output destination must name the job.** `{job_id}` expands in `s3`, `hf`
and `hf_path`; a destination that does not contain it after expansion is refused
at submit, as is any other `{...}`. An `hf` output with no `hf_path` uploads
under the id itself. `hf_create: true` lets the sync preflight create a missing
Hugging Face repo.

<a name="kept-outputs"></a>
**An output with no `s3` or `hf` is kept on the host**, in the job's workdir
where it wrote it. The sweeps delete the checkout around it and leave it;
`gpuc status` shows `kept on host: <path>` and how much the host's kept outputs
hold, and `gpuc fetch <job-id>` copies them here whenever you like. Only a
forced purge that selects the job removes them (`gpuc clean --host H --purge
--force --only <job-id>`, or `--all-finished`). Its `path` must be inside the
workdir and not the workdir itself. A rental refuses kept outputs at submit,
because it terminates itself and they would go with it. A host put back on a
build from before kept outputs (an older client re-ships its own) deletes them
with the workdir.

```yaml
outputs:
  - path: results             # kept on the host; `gpuc fetch <id>` brings it here
  - path: checkpoints
    s3: s3://my-bucket/lego/{job_id}/checkpoints
```

**Files already under an output path are not your results.** Uploads skip files
that were there before `setup`, and a path holding only those is `failed:
no-outputs`. Above 500 pre-existing files under one path the exclusion is
dropped, with a warning. `gpuc submit` warns about them before the job is
queued.

**The sync preflight** runs after the GPU check and before `main`: `aws` (S3) or
`hf` (Hugging Face) must resolve, `hf auth whoami` must succeed with the job's
token, and a `.preflight` object must upload to every destination and to the
host's mirror prefix. Failure is `failed: sync-preflight`, with the command and
its error in the log. A job with no outputs on a host with no mirror checks
nothing.

## Priority is not advisory

Dispatch order is `<priority>-<job id>`, lower first, and the host takes the
queue **in order**: a job that cannot start yet holds the free cards it is
waiting for, and nothing behind it may take them. Queue a big job at a
**higher** number if you would rather it waited than have a card sit idle for
it.

A job waiting for a [shared card](#shared-gpus) somebody else is using does not
hold: the queue behind it runs. A job short of an *owned* card holds even when
that card has dropped off `nvidia-smi`; `gpuc status` marks the card
`UNAVAILABLE` and the job's `starts_unknown` names it.

## Shared GPUs

`--shared-gpus` names cards gpuc may *borrow* but does not own.

```sh
gpuc host set spar --gpus 2,3 --shared-gpus 4,5
gpuc submit job.yaml --host spar --use-shared   # or `use_shared: true` in the spec
```

A job reaches a shared card only when both hold:

- it asked: `use_shared: true`, or `gpuc submit --use-shared`;
- nvidia-smi reports 0 MiB used and 0% util on the card right now. A card that
  could not be read counts as in use.

A job takes every free owned card it can use and borrows only the shortfall. A
borrowed card is held until the job ends; if its real owner starts something
there, `gpuc preempt` is the way out.

`gpuc status` shows borrowed cards on their own `shared` lines, and
`gpuc host bootstrap` refuses a card listed as both owned and shared.

## Job length estimates

Both are optional, and neither affects a job's outcome.

`estimated_runtime_min` is your own guess, measured from the runner's start.
`gpuc estimate <job-id> --minutes N` sets it on a job that is **queued or
running**, and `--clear` takes it off; a running job picks it up within a
minute, and a finished job is refused. The mirrored spec is updated
too, so a later `gpuc requeue` carries the estimate; a mirror that cannot be
written is a warning, not a failure.

`progress_command` replaces the guess with a measurement. It runs in the
workdir, with the job's environment, every `progress_interval_s` of phase
`main`, and the **last line of its stdout** is how far along the job is:

```yaml
progress_command: "tail -1 results/progress.txt"   # the job appends `0.37` as it goes
progress_interval_s: 60
```

| printed | means |
| --- | --- |
| `0.42` | a fraction of one: 42% |
| `42%` | a percentage: 42% |
| `42`, `1`, `0` | **refused**: a bare integer could be either |

Only the last line of the last 64 KiB of output is read. The estimate
extrapolates from phase `main` alone. A progress command that exits non-zero,
prints nonsense or takes longer than 10 seconds is killed and ignored: once in
`log.txt`, and as `progress_error` in `gpuc status --json`.

`gpuc status` shows an `eta` on each running job tagged `(42%)` measured or
`(est)` your guess, an `est` on each queued job, and, on a host with no free
card, when the next is expected:

```
  running lego-s4 (20260915-120000-a1b2c3) phase=main 1h36m util 98% gpu=0 eta 3h20m (37%)
  queued  sweep (20260915-130000-d4e5f6) prio=50 est 6h00m starts in ~3h20m
  free    next card in ~3h20m (20260915-120000-a1b2c3)
```

A queued job's `starts` is absent when its turn depends on a job that estimated
nothing, and on a draining host. A job waiting for more than one card says so
(`needs 2 gpus`).

## Automatic preemption

`auto_preempt: true` lets the host stop the job whenever that lets a job queued
at a **lower** `priority` number start right away, and queue it again as its
next attempt. It re-runs from the start in the workdir it left behind, any
number of times: mark work that is cheap to repeat or that checkpoints.

A job is stopped only when that starts a waiting one immediately, and never
while a host is draining. A borrowed card is never freed for a job that did not
ask to borrow. The stopped job is queued again at its own priority, behind the
job it made room for.

`gpuc status` shows `auto-preempt` on those jobs, and the dispatcher log and the
job's own log name the job each preempt made room for.

## What gets synced to the host

`gpuc submit` rsyncs `git ls-files --cached --others --exclude-standard`: every
tracked file **and** every untracked one git would keep, `.gitignore` obeyed.
Files in the index but deleted on disk are dropped. One line says what happened:

```
syncing 43 files (2 modified, 1 untracked, ignoring .gitignore'd)
```

Alongside the workdir go `uncommitted.patch` (`git diff HEAD -- .`, untracked
files included, without touching your staging area) and `source.json` with the
commit, branch, origin and submitting directory.

`--no-git` rsyncs a directory that is not a repository, minus `.venv`,
`__pycache__`, `.git`, `*.pyc`, `node_modules` and `.uv-cache`, with a warning.
Nothing reads `.gitignore` in that mode, and **`gpuc requeue` cannot rebuild a
`--no-git` workdir**.

## The host's data directory

Every job gets `GPUC_DATA_DIR`: a directory on the host that outlives the job
and that every other job there can see. Download a dataset or model into it
and skip the download when it is already there, and the next job on that
host starts without fetching it again:

```sh
data="$GPUC_DATA_DIR/lego-v3"
[ -d "$data" ] || aws s3 sync s3://my-bucket/datasets/lego-v3 "$data"
```

It is `data/` in the host's gpuc home unless the host names another
(`gpuc host set <host> --env GPUC_DATA_DIR=/big/disk/data`). The runner
creates it. Nothing in gpuc ever deletes from it except `gpuc host clean
<host> --data PATH`, so choose names that say what a thing is, and remove it
yourself when nobody needs it. On a rental it ends with the pod, unless it is
on a `--persistent-root` volume. Hugging Face downloads already share one cache
per host (`HF_HOME`), and `gpuc host clean <host> --hf-cache` prunes it.

`gpuc host bootstrap` prints where the uv cache, the Hugging Face cache and
the data directory are, and how much each holds.

## Commands

**`gpuc submit <job.yaml|-> --host NAME`** — validate, sync the workdir, deliver
secrets, enqueue. `--no-git` is above; `--no-bootstrap` enqueues without first
re-shipping the package to a host on another build (see
[setup.md](setup.md#upgrading)); `--use-shared` is `use_shared: true`;
`--runpod` and its flags are [below](#runpod).

It then asks the host where the job landed:

```
job 20260915-233000-112233 queued on host spar
  queue: position 2 of 5; starts in ~3h20m
  logs: gpuc logs 20260915-233000-112233 -f
```

Where the start time cannot be [projected](#job-length-estimates) the line says
why instead, and it is absent when the host could not be asked again; the job
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
number means hours) choose how much of the finished list each host sends; `--all`
asks each host for every job it has and adds jobs only the local index and the S3
index know, and is exit 1 if the S3 index could not be read; `--json` is [below](#exit-codes-and---json). A host prints
`UNASKABLE` or `GONE` in place of its status ([the host
rule](#exit-codes-and---json)); a gone rental is forgotten as it is printed.
A gone host is shown when it is found gone or named with `--host`.
Card UUIDs are in `gpuc host list`, and everything a host can say about itself
is in `gpuc host probe`.

**`gpuc status <job-id> [<job-id> ...] [--host H]`** — exactly those jobs,
whatever state they are in, one line each: one question per host however many
ids. An id no host has is exit 4 and one that could not be asked about is exit
1, after the rest are reported; how the jobs went is not the exit code (that is
`wait`). A job whose host is gone is answered from the S3 mirror. Not with
`--all`, `--recent` or `--since`.

**`gpuc logs <job-id> [-f] [-n N] [--host H]`** — tails `log.txt` on the host
(`-n` defaults to 200), or the S3 mirror when the host cannot produce it,
which needs `s3_bucket` set here **and** an `s3_prefix` for that job. A job
whose dir was purged reads from the mirror with exit 0.

**`-f` follows until the job ends**, prints the job's outcome as its last line,
and exits 0 only if the job succeeded. A job that has already finished prints
its tail and exits; a job still queued is followed until its log appears. A
line or two of the runner's own cleanup can land after the follow has stopped;
the log itself always has them. `--follow-forever` streams until you stop it;
`--interval SECONDS` pins the poll (2s backing off to 30s by default). Neither
follow can be combined with `--json`, or with the other.

**`gpuc fetch <job-id> [<job-id> ...] [--host H] [--to DIR] [--path P ...] [--list]`**
— copy jobs' files from their workdirs on the host to `DIR/<job-id>/` here
(`DIR` defaults to the current directory), keeping their paths. Without
`--path` it copies each job's `outputs:` paths, less the files that came with
the checkout: the same files an upload would send. `--path` copies everything
under a workdir path instead, and is repeatable; `.` is the whole workdir,
venv included. `--list` prints each file and its size and copies nothing.

It works on a running job (a file being written as it copies arrives
half-written; fetch again) and on a finished one until its workdir is swept.
A job whose workdir is gone, or whose host is gone, is an error for that job
only. An `outputs:` path the job never wrote is a warning.

```sh
gpuc fetch 20260915-233000-112233 --list                  # what it produced so far
gpuc fetch 20260915-233000-112233 --path runs/mx-gdot00   # what a mistyped outputs: missed
```

**`gpuc wait <job-id> [<job-id> ...] [--host H]`** — the same wait without the
log, for several jobs at once: one line per job as it ends, and exit 0 only if
**all** of them succeeded. A reachable host whose dispatcher is down is
reported once and waited through; `gpuc host bootstrap <host>` restarts it.
`--interval` is as above.

```sh
gpuc submit job.yaml --host spar --json | jq -r .job_id | xargs gpuc wait || echo "it did not work"
```

Killing either wait leaves the run alone.

**`gpuc ssh <host|job-id> [--print] [-- CMD ...]`** — an ssh with gpuc's own key,
port, `known_hosts` and ControlMaster socket, none of which are in your
`~/.ssh/config`. A host name lands in its gpuc home; a job id lands in that job's
`workdir/`, falling back to the job dir once the workdir is gone. The words after
`--` are run by a **login bash** in that directory, so `gpuc ssh <job-id> -- 'ls
| wc -l'` is a pipeline, and gpuc exits with the remote command's own exit code.
`--print` prints the command line instead of running it. A `local` host gets your
own `$SHELL` with no ssh at all.

`cancel`, `reorder`, `preempt` and `estimate` take any number of job ids and act
on each, asking each host once for all of its jobs. One line per job it acted
on; each refusal is an `error:` line on stderr. An id no host has is exit 4, and
any other refusal exit 1, once the rest have been carried out.

**`gpuc cancel <job-id> [<job-id> ...]`** — see [how a job is
killed](#how-a-job-is-killed). A finished job is not touched; its status is the
answer, not a refusal.

**`gpuc reorder <job-id> [...] --priority N`** — queued jobs only (a running or
finished job is refused). It prints each job's new queue position and start
time, and records the priority in the job's spec on the host and in its S3
mirror, so `gpuc requeue` carries the move.

**`gpuc preempt <job-id> [...]`** — stop a *running* job and queue it again as its next
attempt. It **starts over**: `setup:` and `command:` run again from the top, in
the workdir the stopped attempt left behind. Nothing is re-synced from here.

It comes back at its own priority unless `--priority N` changes it, and at the
same priority it takes its own cards straight back, so **queue the job you are
making room for first, at a lower number, then preempt**. A preempt that would
only re-run the same job is refused, as are queued and finished jobs.
The stopped attempt is not queued again if it had already ended on its own, was
cancelled while stopping, or the host is draining.

**`gpuc estimate <job-id> [...] --minutes N`** — set (or `--clear`) queued or
running jobs' `estimated_runtime_min`; see [job length estimates](#job-length-estimates).

**`gpuc requeue <job-id>`** — re-read the spec from the S3 mirror and submit it
again as a new job that records where it came from (`requeued_from`), with the
workdir re-synced from your *current* directory. It **needs `s3_bucket`** and
cannot rebuild a `--no-git` workdir. `--host H` sends it somewhere else,
`--runpod` provisions for it, and with neither it goes back to the host that
ran it; a job whose host is gone needs one of them. A mirrored spec this build
will not accept is refused; submitting the job file again is the way round it.
Keys this build does not know are dropped.

`--host` is optional on `status <job-id>`, `logs`, `fetch`, `wait`, `cancel`,
`preempt`, `reorder`, `estimate` and `requeue`: the local job index is tried first, then
every registered host is asked whether it knows the ids. An unknown host is exit 4
unless it is [gone](#exit-codes-and---json) (never for `requeue --host` or `ssh
--host`, which name where to go), and so is a job no host knows once every host
has answered, and a job the host you named answers it does not have.

**`gpuc skill`** — prints the agent guide
([`skills/gpuc/SKILL.md`](../skills/gpuc/SKILL.md)) to stdout. `--install [DIR]`
writes it to `DIR/.claude/skills/gpuc/SKILL.md` instead (default: the current
directory) and refuses to overwrite without `--force`.

**`gpuc version`** — this build, its commit, and the commit each host's config
named when this machine last read it, marking the ones to re-bootstrap. It reads
the registry's cache and never touches a host; what a host is *running* is
`gpuc status`.

**`gpuc clean`**, **`gpuc pods`** and the host commands have their own sections
below and in [setup.md](setup.md). **`gpuc web serve`** is the
[web dashboard](#the-web-dashboard).

## The web dashboard

`gpuc web serve` is `gpuc status`, `gpuc host list` and `gpuc config show` on one
page, refreshed every 15 seconds, with a button for each of `gpuc cancel`,
`gpuc preempt`, `gpuc reorder`, `gpuc estimate` and `gpuc host remove` (on a
host nothing answers from) and a **Logs** panel that tails `gpuc logs`. Every
job links to where its `outputs:` went, to its W&B run when the job's `env`
names `WANDB_ENTITY` and `WANDB_PROJECT`, and to its mirrored log. The links
are what the job declared and are never checked: an `outputs not uploaded`
flag beside one means the link is empty.

```sh
gpuc web set-password          # once; prompts twice, stores a bcrypt hash 0600
gpuc web serve                 # http://127.0.0.1:8646/
gpuc web serve --bind 0.0.0.0 --port 8646   # reachable from other machines
gpuc web serve --bind 0.0.0.0 --install     # the same, as a systemd --user service (see setup.md)
```

Every page and every API document is behind that one password, and the server
refuses to start until one is set. A restart logs everyone out. There is **no
TLS**: bind to localhost or a VPN interface, or put it behind a TLS-terminating
proxy.

The API is the same `--json` documents, over plain HTTP once the session cookie
is held:

| endpoint | the document |
| --- | --- |
| `GET /api/status?host=H&recent=N&since=24h` | `gpuc status --json`, plus `gathered_at` |
| `GET /api/hosts` | `gpuc host list --json` |
| `GET /api/config` | `gpuc config show --json` |
| `GET /api/version` | `gpuc version --json` |
| `GET /api/jobs/<id>/logs?lines=N&host=H` | `gpuc logs --json` (no `-f`; the page re-fetches the tail instead) |
| `POST /api/jobs/<id>/cancel` `{host?}` | `gpuc cancel <id> --json` |
| `POST /api/jobs/<id>/reorder` `{priority, host?}` | `gpuc reorder <id> --json` |
| `POST /api/jobs/<id>/preempt` `{priority?, host?}` | `gpuc preempt <id> --json` |
| `POST /api/jobs/<id>/estimate` `{minutes}` or `{clear: true}` | `gpuc estimate <id> --json` |
| `POST /api/hosts/<name>/remove` | `gpuc host remove --json` |

The HTTP status is the CLI's exit code for the same answer: 0 is 200, 1 is 500,
2 is 400, 3 is 503, 4 is 404, and a request with no session is 401. The
dashboard never forgets a rental; only a typed `gpuc status` does.

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
| `--gpu A40[,RTX4090]` | required with `--runpod` | catalog names, matched case-insensitively against the short name and the catalog id; cheapest match wins |
| `--gpu-count N` | `1` | GPUs on the pod; the spec's `gpus:` must fit in it |
| `--min-vram GB` | none | skip offers with less VRAM per GPU |
| `--max-price USD` | none | **whole pod** per hour |
| `--cloud secure\|community\|any` | `secure` | which tier to buy from; `any` merges both and sorts by price |
| `--cuda-min X.Y` | `12.8` | the CUDA floor: filters the catalog query and is sent to `create` |
| `--idle-min N` | `15` | terminate the pod once its queue has been empty this long |
| `--disk GB` / `--image REF` | `config.toml` | container disk and pod image |
| `--no-reuse` | reuse is on | always create a new pod |
| `--name-hint TEXT` | `job` | goes into the pod name after the prefix |
| `--health-args "..."` | none | extra flags for the on-host health check, e.g. `--min-mbps 0.1` |

**Offers.** Matching offers are tried cheapest first: create, wait for a direct
SSH endpoint, bootstrap, health check, enqueue. A failure of that pod
terminates it and moves to the next offer, all inside **one 15-minute ceiling
for the whole attempt**; an offer the ceiling leaves no time for is listed as
untried. A local ssh misconfiguration, an unreadable key file, or an
`ssh`/`rsync` this machine does not have ends the attempt at the first pod.
Every exit from provisioning before the final registry write, Ctrl-C included,
terminates the pod first; a terminate that fails is reported and the entry
kept.

**Reuse** is the default: a registered rental whose recorded offer still
matches the request, that owns at least `--gpu-count` cards, whose pod is
`RUNNING` with a dispatcher that beat in the last 30 s, and that is not
draining. A reused pod keeps the image, disk and `--idle-min` it was created
with.

<a name="auto-down"></a>
**Auto-down.** The pod terminates itself once nothing is running and the queue
has been empty for `--idle-min`, after draining: retrying unconfirmed outputs
and mirroring every job's log and state. Only a failed *terminate* stops the
shutdown, which then retries in 10 minutes. There is no overall pod lifetime;
a job's own cap is `max_runtime_min`.

<a name="pods"></a>
**After that the pod owns itself, and nothing here watches it.** A pod whose
dispatcher dies, or whose provisioning client was killed before it could clean
up, bills until a person ends it. **`gpuc pods`** shows every pod in the account
with our prefix (name, id, status, GPU, `$/h`, CUDA, age, util, the `HOST` name
it is registered under here, and its dispatcher's heartbeat age) with the
hourly total, plus other people's pods by name only; `--no-heartbeat` skips
the per-pod ssh check. A pod with no heartbeat and nothing running will never
end itself: `gpuc host terminate` ends it. A pod registered nowhere here is
named at the bottom with how to adopt or end it, and one younger than the
15-minute ceiling is flagged as possibly still being set up elsewhere. `gpuc
pods` never terminates anything.

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

`--force` does not ask at all, and is the way past all three. A pod the
provider calls dead (`TERMINATED`, `EXITED`, `ERROR` or missing) is never
refused over.

The terminate is retried and confirmed with the provider before the registry
entry goes. One that still cannot be confirmed is exit 1 and **keeps** the
entry. A `local` or `ssh` host has no rental to end and is refused; `gpuc host
remove` forgets one of those.

## How a job is killed

`gpuc cancel` on a queued job cancels it on the spot. On a running job it
records the request, which the runner acts on before every phase and on every
poll: the job's scope is stopped and its process group SIGTERMed, then
SIGKILLed 15 s later; the final sync, workdir cleanup and mirror follow, and
the job stays `running` in `phase: sync` until they are done. If the runner
does not act the dispatcher escalates: the scope and a SIGKILL of the job's
group at 15 s, a SIGTERM of the runner at 30 s, a SIGKILL of its group at
45 s. A runner already in `phase: sync` is given thirty
minutes before that ladder starts. `gpuc preempt` is the same request with a
different ending; `max_runtime_min` is the same SIGTERM-then-SIGKILL with
`failed: timeout`.

Each phase runs in a transient `systemd --user` scope where the host has one
that lingers (`loginctl enable-linger`), and in its own process group where it
does not (`isolation: cgroup` or `pgid` in `gpuc status`). Under `cgroup` the
dispatcher and each runner also get a scope of their own, so stopping the
scope or service `gpuc submit` ran in stops none of them. Under `pgid`, which is every RunPod pod and most shared boxes, a
grandchild that double-forks (`setsid`, `nohup`, a daemonising server) survives
the kill and holds its GPU; it cannot leave a cgroup.

A phase that exits on its own takes the same stop with it: anything it left
running in its scope or process group (a leaked DataLoader worker, a background
server) is stopped before its cards are freed, and the log says `>>> stopping N
leftover process(es) of the phase`.

A preempted job goes straight from `running` to `queued` as its next attempt;
its log records the attempt that was stopped. Its workdir and secrets file are
kept whatever `cleanup:` says.

Utilization is sampled on the assigned cards every 30 s **during phase `main`
only**. A sample nvidia-smi could not produce is unknown, never 0%.

Every `failed: <reason>`:

| reason | what happened |
| --- | --- |
| `exit <N>` | `command` exited non-zero |
| `setup` | the `setup` phase exited non-zero |
| `gpu-assert` | an assigned GPU is not in the host's `nvidia-smi`, or no GPU was assigned |
| `gpu-preflight` | a GPU op inside the job's environment failed, or `device_count()` did not match `gpus:` |
| `sync-preflight` | a destination cannot be written: no `aws`/`hf`, a missing secret, an unwritable bucket or repo |
| `timeout` | `max_runtime_min` elapsed |
| `preempted` | a preempt stopped this attempt and it could not be queued again because its state had been changed by hand |
| `terminated` | the runner itself was signalled |
| `sync` | the final upload failed; a job already over for a reason of its own keeps that reason and lists `sync` in `problems` |
| `no-outputs` | an `outputs:` path was never written, or holds only files that came with the checkout; listed in `problems` the same way. Never reported for a job whose `main` never started |
| `bad-spec` | the queued spec could not be read, or asks for no GPU |
| `needs N GPUs, host owns M` | the host's ownership shrank after the job was queued; shared cards count only if the job asked for them |
| `spawn-failed` | the dispatcher could not start a runner process |
| `runner-died` | the runner vanished without writing final state; the dispatcher kills anything it left behind |

`cancelled` is a status of its own, not a failure.

## Exit codes and `--json`

| code | meaning |
| --- | --- |
| 0 | everything the command was asked to do worked. For `gpuc wait` and `gpuc logs -f`, the job succeeded |
| 1 | something failed: a transport, a provider, a refused submit, a host that could not be asked, a `clean` the host reported errors for, a host `host bootstrap --all` failed on, or a registry entry it could not read. For `gpuc wait` and `gpuc logs -f` it is the **job** that did not succeed |
| 2 | usage: a bad flag, a missing required one, a bad `--since` |
| 3 | local state is unreadable (`hosts.json` or `config.toml`), so the answer is **unknown** |
| 4 | the job or host named on the command line does not exist |
| 130 | a Ctrl-C |

**A failure never costs you the rest of the answer.** A host that could not be
asked is reported in its own block, every other host in full, and the command
exits 1. A registry entry this build cannot parse is a warning on stderr, the
entry written back untouched, and exit 1 from `status`, `host list` and
`version`; a rental an earlier build registered is one of those, and its
warning says to `gpuc host add <name> --pod <pod-id>` it again. A `hosts.json`
that cannot be parsed at all is exit 3, with the error, the path, and that a
`.bak` was kept.

**The host rule.** A host that is asked is in one of three states, and every
command reports and exits by them:

- **answered**: the host's own word is the answer.
- **unaskable**: it could not be asked (ssh failed; the provider still has the
  pod but nothing can run on it; the provider could not be read), and it may
  still hold its jobs. Exit 1 with the reason. `logs` prints the mirror's copy
  first; `wait` keeps asking for five minutes before reading the mirror.
- **gone**: its rental ended, or the job index says the job ran on a host not
  registered here (named with `--host` or not). Not a failure: the S3 mirror's
  copy of the job is the answer, and a command that would change the job is
  exit 1 saying how it ended. A job the mirror has no final state for, or any
  job with no `s3_bucket`, went with its host: exit 1, saying so. `status` and
  `host bootstrap --all` forget the registry entry and say so.

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
      "state": "answered",
      "reachable": true,
      "pkg_commit": "8f1c2d0a9b34",
      "dispatcher": { "alive": true, "heartbeat_age_s": 2.0, "pkg_commit": "8f1616c..." },
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
      "errors": [],
      "warnings": []
    }
  ],
  "unhosted": [],
  "errors": []
}
```

The job objects in `queued`, `running` and `finished` all carry the same fields.
Beyond what the example shows:

- `priority` (0–99, lower first) is on every job, and `queued` is already in
  dispatch order. Null when the host did not say, as are `use_shared` and
  `auto_preempt`; never `false`.
- `gpus_requested` is what the spec asked for; a queued job holds no `gpus` yet.
- `starts_in_s` / `starts_at`: when a queued job's turn is expected; null for
  anything else, and for a queued job whose turn cannot be dated, when
  `starts_unknown` says why.
- `problems`: what else went wrong on the way out of a finished job (`sync`,
  `no-outputs`). `upload_errors`: the last failure standing at each upload
  destination; a running job with one says `UPLOAD FAILING` in the text.
- `eta` is absolute and `eta_s` the same instant in seconds from now (negative
  once overdue); both null without a length estimate and once the job is
  finished. `progress_pct` survives the job; `progress_error` is why the last
  poll produced nothing.
- `attempt` (launches of this id; a preempt adds one), `requeued_from`,
  `kept_outputs` (the [kept](#kept-outputs) paths a finished job wrote to) and
  `kept_bytes` (what they hold, once the checkout around them is swept),
  `started_at`, `exit_code`, `outputs_lost`, `workdir_bytes`, `outputs` (the
  spec's, as the host holds them) and `links`: one `{kind, path, target, url}`
  per place the results, W&B run or mirrored log can be opened, derived from
  what the job declared and never checked.
- `util` is the job's last sample from the host's own nvidia-smi; a pod's
  `provider_util` is the provider's per-GPU reading for the whole pod, null
  elsewhere.

Per host: `target`, `draining`, `lost` (a gone host's `{jobs, reason}` for the
jobs that went with it, else null), `state` (`answered`, `unaskable` or `gone`, as
above; `errors[]` carries the reason and decides the exit code, `warnings[]`
carries the build mismatch and does not), `pod` (the provider's view of a
rental's pod), `pkg_commit` (the host's own answer for the build it runs; `null`
means it did not say). A card the host cannot see appears in `gpus` as
`{"owned_as": "3", "available": false}`; `shared_gpus` has the same shape plus
`unused` (no memory held, no work running) and `busy_job` (one of *our* jobs
has it), and a missing one is `{"shared_as": "5", "available": false}`.
`--recent` and `--since` apply to `--json`; `--all` fills `unhosted[]` with the
jobs only the index knows, each `{job_id, name, host, host_state, status,
requeue, requeued_from, submitted_at, s3_prefix, outputs_lost}`. `status` is
the mirror's final status when the host is `gone`, else null. `host_state` is the
same three words (`answered` here means the host does not have the job; a host
whose registry entry did not validate is `unaskable`; one not registered here
is `gone`), and `requeue` is whether `gpuc requeue` is the way back: true for
`answered` and `gone`, false for `unaskable`, which may still hold the job.

Rules for anything automated:

- **Key on `hosts[].running`.** It is the host's own answer; an empty list means
  the host said nothing is running.
- **Read queue order from `priority`**, not from `eta` or
  `estimated_runtime_min`: those say how long a job takes, not when it is taken.
- **Treat exit 3 as "unknown", never as "nothing is running".** So is a non-empty
  top-level `errors`, and so is `"reachable": false` for the host you care about.
- Unknown keys will be added over time; ignore the ones you do not know.

### `--json` everywhere else

`status`, `submit`, `requeue`, `logs`, `fetch`, `wait`, `cancel`, `preempt`, `reorder`,
`estimate`, `pods`, `version`, `clean`, `config show`, `config init`,
`host list`, `host probe`, `host add`, `host set`, `host bootstrap`,
`host clean`, `host remove` and `host terminate` take `--json`: **stdout is
exactly one JSON object**, it carries `schema_version`, and everything the text
output would print alongside it (progress, warnings, `note:` lines) goes to
stderr. Exit codes are unchanged by the flag. `ssh`, `skill`, `web serve`,
`web set-password` and `skill --install` have no document.

**A command that failed prints a document too:**

```json
{ "schema_version": 1, "error": "no registered host knows job 20260915-120000-abc123.\n...",
  "exit_code": 4 }
```

`error` (singular) is the whole answer: the command did not do what it was
asked, and a command line argparse rejects gets the same document. `errors`
(plural) is per-host or per-job trouble the command reported rather than
stopped for.

| command | the document |
| --- | --- |
| `submit`, `requeue` | `{job_id, host, requeued_from, notes[], queue_position, queue_length, dispatched, starts_in_s, starts_at, starts_unknown}`. `requeued_from` is null on `submit`; `notes` are the text output's `note:` lines; `queue_position` is 1-based in dispatch order; `dispatched` is true for a job the host started before we could look; `starts_unknown` says why there is no start time. The queue fields are all null when the host could not be asked again, and the job is queued regardless |
| `logs` | `{job_id, host, source, location, lines[], notes[]}`. `source` is `"host"` or `"s3"`, `location` the remote path or `s3://` uri, `lines` the log without trailing newlines, `notes` why the mirror was read. Not with either follow (exit 2) |
| `wait` | `{jobs[], errors[]}`, printed once every job has ended. Each of `jobs[]` is the job's final state in the shape `status --json` gives a job, plus `host`, `host_state` (`answered`, `unaskable` or `gone`, as in [the host rule](#exit-codes-and---json)), `source` (`"host"` or `"mirror"`) and `error`. **Check `error`, not `status`**: when it is not null, `status` is only the last thing its host managed to say, and a job nothing was heard about carries only `job_id`, `host`, `host_state`, `source`, `error` and a null `status`. With an `error`, only a `host_state` of `unaskable` can change if asked again. Every `error` is in `errors[]` too. The per-job outcome lines go to stderr. Exit 1 unless every job succeeded; exit 4 when an id no host has is among them, with the rest still reported |
| `status <job-id> ...` | `wait`'s document, as things stand now rather than once the jobs end. Exit 4 when an id no host has is among them, 1 when any other job has an `error`, else 0 whatever the jobs' own status |
| `fetch` | `{jobs[], errors[]}`, one entry per id: `{job_id, host, source, error, warnings[]}` and, with no `error`, `{status, workdir, files[], bytes, missing[], to}`. Each file is `{path, bytes}`, relative to the workdir; `missing` names `outputs:` paths the job never wrote; `to` is the local directory they were copied to, null under `--list` or when there was nothing to copy. Exit 4 when an id no host has is among them, else 1 when any `error` is set |
| `cancel`, `preempt`, `reorder`, `estimate` | `{jobs[], errors[]}`, one entry per id in the order given: `{job_id, host, source, error, warnings[]}` plus what the host did, below. **Check `error`**: when it is not null the host did not confirm doing what was asked, and none of the fields below are there. `source` is `mirror` for a job whose host is gone, answered from the S3 mirror. `warnings` carries a mirrored spec that could not be updated, so `gpuc requeue` would not carry the change. Every `error` is in `errors[]` too. Exit 4 when an id no host has is among them, else 1 when any `error` is set |
| ↳ `cancel` | `status`: `cancelled` for a queued job, `cancelling` for a running one, a finished job's own status |
| ↳ `preempt` | `status` (`preempting`) and `priority`, what it will be queued again at |
| ↳ `reorder` | `status` (`queued`), `priority`, and the same `queue_position`, `queue_length`, `dispatched`, `starts_in_s`, `starts_at` and `starts_unknown` as `submit` |
| ↳ `estimate` | `status` and `estimated_runtime_min`, what the job's state holds now (null after `--clear`); `warnings` also carries a `max_runtime_min` contradiction |
| `pods` | `{pods[], hourly_usd, others[], notes[]}`. Each pod is `{id, name, status, gpu_name, gpu_count, cost_usd_hr, cuda_version, age_s, created_at, gpu_utils[], host, heartbeat_age_s}`; `host` is the registry name here, null if none; `others` are pods without our prefix, `{id, name, status}` only |
| `version` | `{version, commit, source, dirty, python, executable, hosts[], errors[]}`, each host `{name, pkg_commit, seen_at, current}`. `pkg_commit` is the commit the host was running when this machine last read it. Exit 1 for an entry that could not be parsed, 3 if the whole registry is unreadable |
| `config show` | `{config_file, config_file_exists, state_dir, settings{}, notes[]}`: the effective settings, file or not |
| `host list` | `{hosts[], errors[]}`, each entry the address (`name`, `kind`, `ssh`, `port`, `gpuc_home`, `persistent_root`, `rental` as `{provider, pod_id}` or null, and `pod_id` beside it), the host's own config as last read (`gpus`, `s3_prefix`, `env`, `cache_dir`, `idle_minutes`, `retention_days`, `pkg_commit`) flattened beside it with `config_seen_at`, the raw `cache`, `remote_home`, `ephemeral` and `warnings[]`. Nothing here asks the host. `env` is reported by name only (`{"HF_TOKEN": "<set>"}`). A skipped entry is an `errors` string and exit 1; exit 3 if the whole registry is unreadable |
| `host probe` | `{host, sections{}, driver_version, has_nvidia_smi, gpus[], assigned_gpus[], assigned_missing[], shared_gpus[], shared_missing[], home_fs_type, home_is_overlay, persistent_root, notes[]}`. `gpus` is every card the host has whatever `--all-gpus` said, each `{uuid, name, vram_mib, index, assigned, shared}`; `assigned_gpus` is `--gpus` as registered and `assigned_missing` the entries no card answered to; `shared_gpus` and `shared_missing` the same for `--shared-gpus`; `sections` the probe script's raw output |
| `clean` | `{host, dry_run, purge, freed_bytes, removed[], skipped[], purged[], purge_skipped[], incoming_removed[], verified[], notes[], errors[]}`. Job objects are `{job_id, status, bytes, age_days, mirrored_at, mirror}`, plus `why` on the skipped ones and `forced` on a purged job that had no confirmed backup; `incoming_removed` names job dirs a submit never finished |
| `host add`, `host set` | one `host list` entry as the registry holds it once the command is done, plus `adopted` (the host already had a config), `config_path` (that config on the host), `changes[]` (one line per config field written through to the host) and `warnings[]`. `host set` adds `address{}`: the fields it changed here rather than on the host (`persistent_root`, `gpuc_home`) |
| `host remove` | `{host, kind, pod_id, notes[]}`: what was forgotten here. A rental is **not** terminated, and `notes` says so |
| `host terminate` | `{host, pod_id, pod_name, pod_status, cost_usd_hr, checked, running[], queued[], outputs_pending[], terminated, forgotten, notes[]}`. `pod_status` and `cost_usd_hr` are what the provider said **before** the terminate; `checked` is whether the host itself answered (false means the three lists are empty for want of an answer); `forgotten` is whether the registry entry went. A refusal is the error document, exit 1 |
| `host bootstrap` | `{host, home, files, pkg_commit, dispatcher_pid, warnings[]}`. With `--all`: `{hosts[], total, bootstrapped[], failed[], gone[], unreadable[], interrupted, errors[]}`, one `hosts[]` entry per registered host, `{name, outcome, error, ephemeral}` plus the single-host fields (null unless it was bootstrapped). `outcome` is `bootstrapped`, `failed` (with `error`), `gone` (a rental the provider no longer has, forgotten rather than failed), `interrupted` (the host a Ctrl-C landed in) or `not_attempted` (the ones after it); `unreadable` names entries this build could not read. Exit 1 if any host failed; a Ctrl-C is exit 130 and the error document carries the same tally; a registry that stops being readable mid-run is exit 3 |
| `host clean` | `{host, uv_cache?, hf_cache?, data?}`, one object for each part asked for. `uv_cache` and `hf_cache` are `{cache_dir, before, after, before_bytes, after_bytes, freed_bytes}`, with the size fields null where the host could not measure them, plus `errors[]` on `hf_cache`. `data` is `{data_dir, removed[], freed_bytes, errors[]}`, each removed path `{path, freed_bytes}`; a path that is not there, or is not inside the data directory, is an error for that path only. Exit 1 if any part failed |
| `config init` | `{config_file, existed}`; `existed` is only ever true with `--force`, since an existing file is otherwise refused (exit 1) |

```sh
gpuc submit job.yaml --host gpubox --json | jq -r .job_id
gpuc logs "$id" --json | jq -r '.lines[-20:][]'
gpuc wait "$id" --json | jq -r '.jobs[] | "\(.job_id) \(.status) \(.reason // "")"'
gpuc pods --json | jq '[.pods[] | select(.host == null) | .name]'
gpuc clean --host gpubox --all-finished --dry-run --json | jq .freed_bytes
gpuc host bootstrap --all --json | jq -r '.hosts[] | "\(.name) \(.outcome) \(.error // "")"'
```

## Cleanup and retention

A job's `workdir/` is the rsynced code *and* whatever the job builds in it
(usually a venv). It is the only part of a job dir gpuc deletes by default:
`spec.json`, `state.json` and `log.txt` stay, so `logs`, `status` and `requeue`
keep working on a cleaned job.

**Per job, by the runner**, after the final sync and state write:

| `cleanup:` | succeeded | failed | cancelled |
| --- | --- | --- | --- |
| `on_success` (default) | removed | **kept** | **kept** |
| `always` | removed | removed | removed |
| `never` | kept | kept | kept |

No policy touches a job that is not finished, and none deletes a workdir whose
`outputs:` have not reached their destination. Every removal takes the checkout
and leaves [kept outputs](#kept-outputs) where they are; once only they remain,
there is nothing left to sweep. A workdir the policy keeps is
still swept once it is `--workdir-days` old (below); `cleanup: never` opts out
of that too.

**After the fact: `gpuc clean --host H`.**

```sh
gpuc clean --host gpubox --all-finished --dry-run   # what would go, and how big
gpuc clean --host gpubox --all-finished             # every succeeded/failed/cancelled job
gpuc clean --host gpubox --older-than 7             # only jobs that ended over 7 days ago
gpuc clean --host gpubox --only 20260101-120000-ab12,20260101-130000-cd34
```

|  | `clean` | `clean --purge` |
| --- | --- | --- |
| `workdir/` (code, venv, uploaded outputs) | removed | removed |
| kept outputs in `workdir/` | **kept** | skipped unless `--force` |
| `spec.json`, `state.json`, `log.txt` | **kept** | removed |
| the job's `secrets/<id>.env`, if any is left | kept | **removed** |
| job dirs under `incoming/` a submit never finished | removed | removed |
| running or queued jobs, or ones with unreadable state | never touched | never touched |

`--only ID[,ID...]` replaces `--all-finished` and `--older-than`: exactly those
jobs, however recently they ended, and no `--yes` needed for `--purge`. An id
no job dir matches refuses the whole selection (exit 1, nothing removed); an
empty `--only` is exit 2.

`--purge` removes the whole `jobs/<id>/` of finished jobs and defaults to
`--older-than 7`. It skips a job that is not **backed up** (the host's final
upload of `log.txt` and `state.json` to its mirror did not succeed: `not backed
up: no s3_prefix on this host` or `not backed up: final upload failed`) and
one whose **outputs are not confirmed** (`outputs not confirmed uploaded`: a
declared output was written and some destination has no record of its last
upload succeeding).

It also skips one that **keeps outputs** (`keeps outputs on this host:
<paths>`).

`--force` overrides those three and nothing else, per job. `--verify` also lists
the mirrored logs under the host's prefix with your own credentials and purges
only a job that has one. `--purge --all-finished` needs `--yes` (or
`--dry-run`).

**Automatic, by the host: two horizons.** The dispatcher reclaims disk at
startup and then at most once an hour; on a `local` or `ssh` host it only
lives while there is work, so this happens on your next submit.

| | `--workdir-days N` | `--retention-days N` |
| --- | --- | --- |
| default | `1` | none |
| takes | `workdir/` only | the whole `jobs/<id>/` |
| needs a mirror | no | yes, and never forced |

`--workdir-days` refuses a job whose spec says **`cleanup: never`**, one whose
**`outputs:` have not reached S3 or HF** (flagged `outputs not uploaded` in
`gpuc status`), and one whose `spec.json` is unreadable. `--retention-days`
only ever acts on jobs whose log and state the host confirmed mirrored, and
sweeps workdirs at the shorter of the two horizons under the same refusals.
Either pass removes a job dir under `incoming/` that a `gpuc submit` never
finished, once it has been untouched for an hour. `''` turns either horizon
off.

Both live in the host's own `config.json`; `gpuc host set <name>
--workdir-days N` writes through to it and takes effect without a bootstrap.

**Unconfirmed outputs.** `gpuc status` flags a finished job that *produced*
`outputs:` which never reached S3 or HF as `outputs not uploaded`. An ephemeral
host retries them while it drains (three tries a minute apart) and, if they
still fail, records `outputs_lost` and terminates anyway; such a job shows
`OUTPUTS LOST`, and only re-running it recovers the results.

`gpuc status` prints one line per host once finished workdirs hold more than
1 GiB, with the `gpuc clean` line to run. The figure is what deleting them
would free, not what `du` says they hold, measured once when the job ends.

## Troubleshooting

| symptom | what it means | what to do |
| --- | --- | --- |
| `status` says `dispatcher DOWN` | nothing holds the host's lock, or its heartbeat is over 30 s old | `gpuc host bootstrap <host>`; any `gpuc submit` also restarts it |
| provisioning gives up with "no direct SSH endpoint" | RunPod never exposed port 22 within the 15-minute ceiling | the pod was already terminated; re-run the submit, or widen `--gpu` / `--cloud any` |
| `ssh ... cannot create its ControlMaster socket` | the socket path would be over the 100-byte limit | point `XDG_RUNTIME_DIR` at a short directory, or unset it to use `/tmp/gpuc-<uid>` |
| bootstrap fails with "host health failed" | the driver, disk or network check on the host said no | fix the named check and re-run bootstrap |
| job is `failed: gpu-preflight` | torch in the job's venv has no working CUDA, or sees the wrong number of devices | check the torch build against the host's driver (`gpuc host probe`), and that `gpus:` matches what the job expects |
| job is `failed: sync` (or lists `sync` in `problems`) | the final upload failed | check the tail of `gpuc logs <job-id>`; usually a missing `secrets:` entry, or no `aws`/`hf` on the host (re-run bootstrap). `gpuc fetch <job-id>` copies the outputs here meanwhile |
| job is `failed: sync-preflight` | a destination cannot be written | the log names the command and error; fix the credential or destination, or add `hf_create: true`, then re-submit |
| job is `failed: no-outputs` | the `outputs:` path was never written, or holds only what came with the checkout | write there, relative to the workdir; use a `{job_id}` subdirectory. If the job wrote somewhere else, `gpuc fetch <job-id> --path <where>` gets it |
| a host is out of disk, or `status` shows a `disk` line | finished jobs' workdirs are still there | `gpuc clean --host <host> --all-finished`, and `cleanup: always` on jobs you never need to inspect |
| `clean --purge` skips everything as "not backed up" | the host has no `s3_prefix` | `gpuc host set <host> --s3-prefix s3://bucket/gpuc/<host>`, or accept the loss with `--force` |
| `status` says a job's `outputs not uploaded` | the results exist only on that host | `gpuc fetch <id>`, or `gpuc requeue <id>` |
| `status` says `kept on host` | the job's outputs have no destination, as the spec asked | `gpuc fetch <id>` copies them here; `gpuc clean --host H --purge --force --only <id>` deletes them |
| submit refuses: `is a rental ... would be lost with it` | an output has no `s3` or `hf`, and a rental ends itself | give it a destination, or submit to a host that persists |
| a job is `OUTPUTS LOST` | an ephemeral host retried and gave up before terminating | fix the credential or bucket, then `gpuc requeue <id>` |
| the automatic sweep never takes a job | it runs only while the dispatcher lives, and refuses `cleanup: never`, an unmirrored job dir and outputs still only on the host | `gpuc clean --host H --only <id>` takes the workdir; `--purge --force` takes the rest |
| `requeue` refuses, or rebuilds the wrong code | it reads the spec from S3 (`s3_bucket` must be set) and re-syncs the workdir from your current directory | run it from the right checkout; a `--no-git` workdir cannot be rebuilt |
| `uv sync` re-downloads torch on every job | uv's cache is on a different filesystem from gpuc home | `gpuc host bootstrap <host>` (it sets `UV_CACHE_DIR`), or pin one with `--cache-dir` |
| `gpuc` exits 3 and names `hosts.json` | the registry could not be parsed; a `.bak` was kept beside it | fix or delete the file, then re-add hosts |
| a warning names one skipped host entry | that entry did not validate; every other host still works | fix it by hand, or `gpuc host add <name> --ssh ...` again |
| `status` warns `host X is running gpuc <sha> and this machine has <sha>` | the host was last bootstrapped from a different build | `gpuc host bootstrap X`, or `--all` for every host; safe while jobs run |
| `status` warns `host X has gpuc <sha> on disk but its running dispatcher was started on <sha>` | the dispatcher outlived the package under it | `gpuc host bootstrap X`; safe while jobs run |
| a host `runs a gpuc build that does not understand` a request | it was last bootstrapped from a build older than the command asked of it (`status <ids>`, `status --recent`, the multi-id verbs, `fetch`, `host clean --data` or `--hf-cache`) | `gpuc host bootstrap X`, or `--all`; safe while jobs run |
| `status` says `GONE` or `UNASKABLE` | the [host rule](#exit-codes-and---json); an `UNASKABLE` pod with `ERROR pod <id> is EXITED` may still be billing | `GONE`: nothing. `UNASKABLE` pod: `gpuc host terminate <name>` ends it, `gpuc host remove <name>` only forgets it |
| `gpuc pods` shows a pod with no heartbeat and nothing running | its dispatcher died, or its provisioning client was killed; it will never idle out | `gpuc host terminate <pod-id> --force`, or if it still answers ssh, `gpuc host add <name> --pod <id>` then `gpuc host bootstrap <name>` |
| `host terminate` says the pod could not be confirmed gone | the provider refused or did not answer, three times | the entry is kept and the pod may still be billing: run it again, then check the RunPod console |
| everything on a host is suddenly gone | the container restarted and `$HOME` was on the overlay | the runbook in [setup.md](setup.md#hosts-whose-home-is-wiped-on-restart) |
