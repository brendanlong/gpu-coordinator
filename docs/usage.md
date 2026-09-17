# Usage

Installing gpuc and registering hosts is [setup.md](setup.md). `gpuc --help`,
and `--help` on every subcommand, is the authoritative list of flags; this page
is what they mean together.

## The job spec

A job is one YAML (or JSON) document. `gpuc submit job.yaml --host H` runs it in
the working directory you submit from. `job.example.yaml` in the repo root is a
commented example; `-` as the file name reads the spec from stdin.

| field | default | meaning |
| --- | --- | --- |
| `command` | **required** | run in `workdir/` as phase `main`; a blank one is refused at submit |
| `name` | `""` | a label for `status`; not an identifier |
| `setup` | none | run first, as phase `setup` |
| `gpus` | `1` | how many of the host's GPUs to assign (>= 0). `0` never waits for a card. More than the host can ever provide is refused at submit |
| `use_shared` | `false` | also let this job onto the host's **shared** GPUs — cards gpuc does not own and takes only while nobody else is on them. See [shared GPUs](#shared-gpus). `gpuc submit --use-shared` sets it from the command line |
| `env` | `{}` | plain environment for the job, applied after the host's `--env` |
| `secrets` | `[]` | names read from *your* shell at submit time and delivered to the host as `~/.gpuc/secrets/<job-id>.env` (0600). Missing from your shell is a refused submit |
| `outputs` | `[]` | `{path, s3}` and/or `{path, hf, hf_path, hf_create}`; `path` is relative to the workdir |
| `sync_interval_s` | `180` | background upload cadence; **minimum 10** |
| `priority` | `50` | `0`–`99`, lower dispatches first, and the queue is taken strictly in that order — see [priority is not advisory](#priority-is-not-advisory) |
| `max_runtime_min` | none | wall clock from the runner's start; over it the job is `failed: timeout` |
| `estimated_runtime_min` | none | roughly how long you expect it to take, measured the same way. Nothing enforces it; see [job length estimates](#job-length-estimates) |
| `progress_command` | none | run in the workdir during phase `main`; its last line of stdout is how far along the job is |
| `progress_interval_s` | `60` | how often to run it; **minimum 5** |
| `low_util` | on | `{enabled: true, window_min: 25, floor_pct: 5, grace_min: 10}` — the idle-GPU watchdog |
| `auto_preempt` | `false` | let the host stop this job whenever that lets a job queued at a **lower** `priority` number start right away; see [automatic preemption](#automatic-preemption) |
| `requires` | `{}` | e.g. `cuda_min: "12.8"`. **Informs provisioning only**; the host never checks it |
| `cleanup` | `on_success` | when the runner deletes `workdir/`: `on_success`, `always`, `never` |
| `attempt` | `1` | set by `gpuc requeue` and by a preempt, never by you; a submitter's value is ignored |

Unknown keys are refused at submit, so a typo is an error rather than silence.

`{job_id}` expands in `s3`, `hf` and `hf_path`. Output namespaces are unique by
construction and nothing guards against overwriting a destination you reuse.
`hf_create: true` lets the sync preflight create a Hugging Face repo that does
not exist; without it a missing repo fails the job in seconds instead of
creating `org/typo`.

**Files already under an output path are not your results.** A checkout usually
ships committed files where the results go. Before `setup` the runner records
every declared output path's contents in `jobs/<id>/outputs_baseline.json`;
every upload skips files that still match it, and a path holding *only* those
counts as having produced nothing (`failed: no-outputs`). Above **500**
pre-existing files under one path the exclusion is dropped with a loud warning
in the log, because the exclude list would no longer fit on a command line.
`gpuc submit` warns about pre-existing files before the job is queued.

**The sync preflight** runs after the GPU preflight and before `main`: with S3
outputs (or a host `s3_prefix`) the `aws` binary must resolve and a small
`.preflight` object must upload to every destination *and* to the host's mirror
prefix; with HF outputs, `hf` must resolve, `hf auth whoami` must succeed with
the job's token, and a `.preflight` file must upload to each repo. Failure is
`failed: sync-preflight`, seconds in, with the command and its error in the log.
A job with no outputs on a host with no mirror checks nothing.

## Priority is not advisory

Dispatch order is `<priority>-<job id>`, lower first, and the host takes the
queue **in order**: a job that cannot start yet holds the free cards it is
waiting for, and nothing behind it may take them. A two-card job at priority 10
does not lose its card to a one-card job at 50 that happens to fit.

That matters because the alternative is not "slightly unfair", it is
*indefinite*: while the host walked past a job that did not fit, every card
freed near a wide job at the front went to the narrow jobs behind it, and a
steady supply of them meant the most important job in the queue never ran at
all.

**It costs utilization.** A card waiting for the rest of a job's cards runs
nothing, for as long as the other cards stay busy — and on a rented pod that is
billed. If you would rather a big job waited than have a card sit idle for it,
queue it at a **higher** number than the work you want to keep the host busy
with; priority is the only knob, and it decides both questions at once.

A job only holds cards when the host can supply the **whole** of it from the
cards it owns and can see right now. These are walked past instead, because
holding a card for them would mean waiting on something the host does not
control:

- **`gpus: 0`** — it holds no card, so it can never be the reason anything is
  short of one. It still never waits.
- **a job asking for more cards than the host can currently see** — either a
  card has dropped off `nvidia-smi`, and idling the host until it comes back
  (if it comes back) is worse than letting the queue run; or the job can only
  fit by [borrowing](#shared-gpus), and a shared card comes free when somebody
  else's job ends, which is not this host's to wait for. Neither is failed: the
  host's `config.gpus` says it owns enough. A job bigger than the *configured*
  host, shared cards included, is failed at dispatch as it always was.

## Shared GPUs

A box may hand you some cards outright and leave the rest to other people.
`gpuc host set <name> --shared-gpus 4,5` says which are the second kind: cards
gpuc may *borrow*, never ones it owns.

```sh
gpuc host set spar --gpus 2,3 --shared-gpus 4,5
gpuc submit job.yaml --host spar --use-shared   # or `use_shared: true` in the spec
```

A job reaches a shared card only when both of these hold:

- it asked — `use_shared: true`, or `gpuc submit --use-shared`. Off by default,
  and a job that did not ask is never dispatched to one;
- **nobody else is on the card**: nvidia-smi reports 0 MiB used and 0% util for
  it, right now. Memory is the half that matters — a CUDA context holds
  hundreds of MiB between steps, so 0 MiB means no process on the card, while
  util alone dips to zero between somebody else's epochs. A card that could not
  be read counts as in use.

Owned cards come first, always: a job takes every free owned card it can use
and borrows only the shortfall. That is one rule and it does both jobs shared
cards are for — a one-card job borrows when your own cards are busy, so the
queue drains faster; and a `gpus: 4` job on a host that owns 2 and shares 2
waits for both shared cards to go quiet and then runs across all four.

Start-time estimates count the shared cards that are idle *right now*, so a
borrower whose turn is the next dispatch pass is told `starts now` rather than
being made to wait for one of your own cards. A card somebody else is on is
left out: when they will stop is not something your host can know, so a job
waiting for one has no start time and is told so instead of given an invented
one.

It never gives a card *back*: a job that got one keeps it until it ends, so if
the card's real owner starts something there, the two of you are sharing it and
`gpuc preempt` is the way out.

There is no per-host setting for *which* jobs may borrow, only the per-job
`use_shared`. Borrowing is not a reservation — whatever job gets a shared card
holds it until it ends — so a host-level priority floor would not protect an
important job from a trivial one, it would only pick which trivial jobs wait.
Leave `use_shared` off for the runs that should not take somebody else's card.

`gpuc status` shows the borrowed cards on their own `shared` lines, and
`gpuc host bootstrap` refuses a card listed as both owned and shared.

## Job length estimates

A shared host's queue is a question — *do I wait for this, or do I go and pay
for a pod?* — that nothing but the job itself can answer. Both ways of
answering it are optional, and neither ever affects a job's outcome.

`estimated_runtime_min` is your own guess, and costs nothing:

```yaml
estimated_runtime_min: 480          # about eight hours, from the runner's start
```

You do not have to know it at submit time. `gpuc estimate <job-id> --minutes N`
sets it on a job that is already **queued or running** — the case that matters
most, since the job somebody needs an end time for is the one already running
when they arrive — and `--clear` takes it off again. A running job's runner
re-reads its spec every 30 seconds, so the new estimate reaches `gpuc status`
within a minute; a finished job is refused (exit 1). The job's mirrored spec is
updated too, so a later `gpuc requeue` carries the estimate; if the mirror
cannot be written the command still succeeds and says so, since the estimate is
already recorded where `status` reads it.

`progress_command` replaces the guess with a measurement. It runs in the
workdir, with the job's own environment, every `progress_interval_s` of phase
`main`, and the **last line of its stdout** is how far along the job is:

```yaml
progress_command: "tail -1 results/progress.txt"   # the job appends `0.37` as it goes
progress_interval_s: 60
```

Exactly two forms are accepted, and which one you meant is always written
down:

| printed | means |
| --- | --- |
| `0.42` | a fraction of one — 42% |
| `42%` | a percentage — 42% |
| `42`, `1`, `0` | **refused**: a bare integer could be either |

A bare integer is an error rather than a guess. `42` could be a percentage or
an impossible fraction; `1` could be 1% or a finished job. Reading either the
wrong way is a hundredfold error in an end time somebody is planning around,
and nothing applied after the fact can tell them apart — so the unit goes in
the input, as a `%` or as a decimal point.

The decimal point costs you nothing if you are dividing: `step / total` prints
`0.42`, `0.0` and `1.0` in any language, never a bare integer. What it catches
is `echo $step`, which would otherwise report "finished" on its second poll. In
shell, `echo "$((step * 100 / total))%"` is usually easier than a fraction.

Only the last line is read, so a progress command may be a pipeline that also
logs. gpuc extrapolates from how long phase `main` has taken so far, so the
estimate improves as the job runs and a slow `uv sync` is never charged to it.

A progress command that exits non-zero, prints nonsense, or takes longer than
10 seconds is killed, recorded, and otherwise ignored: the failure appears once
in `log.txt` and in `gpuc status --json` as `progress_error`, and the job runs
on. Only the last 64 KiB of its output is read, so pointing it at a whole log
by mistake costs a bounded read. The runner polls it from the same loop that
watches for a cancel, a TTL and `max_runtime_min`, which is why the timeout is
short: 10 of the 15 seconds the runner gets before the dispatcher escalates a
kill, so a wedged progress command eats most of that budget but never the
whole of it.

`gpuc status` then shows an `eta` on each running job, tagged with where it came
from — `(42%)` for a measurement, `(est)` for your guess — an `est` on each
queued job, and, on a host with no free card where something holding one
estimated an end time, one line saying when the next card is expected:

```
  running lego-s4 (20260915-120000-a1b2c3) phase=main 1h36m util 98% gpu=0 eta 3h20m (37%)
  queued  sweep (20260915-130000-d4e5f6) prio=50 est 6h00m starts in ~3h20m
  free    next card in ~3h20m (20260915-120000-a1b2c3)
```

A queued job's `starts` is the host's own dispatch rule run forward over the
estimates it has: cards come free at the eta of whatever holds them and the
queue is taken in order, because a job that does not fit
[holds the cards it is waiting for](#priority-is-not-advisory). It is evidence
or it is absent: a job whose turn depends on a job that
estimated nothing has no `starts` at all, and a paused or draining host projects
nothing, because nothing is being dispatched. A job waiting for more than one
card says so (`needs 2 gpus`), which is the answer to "there is a card free, why
is it still queued".

Where some of the jobs holding a card offered no end time, the `free` line
appends a count of them, because the real answer can only ever be *sooner* than
it: one of those could finish in a minute. Jobs with `gpus: 0` are ignored throughout —
they hold no card, so they can neither free one nor make the answer sooner. If
*nothing* holding a card estimated an end time there is no line at all, since
the gpu lines above it already say every card is busy.

## Automatic preemption

`auto_preempt: true` in the spec says the same thing the command does, without
anyone being there to type it: the host stops the job whenever doing so lets a
job queued at a **lower** `priority` number start right away, and queues it
again as its next attempt. It re-runs from the start in the workdir it left
behind, exactly as `gpuc preempt` does, so it belongs to work that is cheap to
repeat — a sweep point, an eval, a job that checkpoints and resumes — and not
to a run whose `setup:` would trip over its own leftovers.

The host only does it when it is worth it, and the rules are the command's:

- **It has to be enough.** Freeing one of the two cards the waiting job needs
  would cost an attempt and start nothing, so a job is stopped only when what
  is stopped covers the whole gap. Several are stopped together where one is
  not enough, least important first, and among equals the one that has been
  running the shortest time. Least important first is by `priority` alone, so
  a job may free more cards than the waiting one needs; the surplus goes back
  to the queue like any other card.
- **The waiting job has to be strictly more important.** At the *same*
  priority nothing happens: dispatch order is `<priority>-<job id>` and the
  stopped job's id is the older one, so it would win the tie and take its own
  cards straight back. A job queued at a higher number never preempts anything.
- **The cards stay with the job they were freed for.** A stopped job is queued
  again at its own priority — behind the job that is waiting — and a job that
  does not fit holds the free cards it needs, so the card cannot go back to the
  job that just gave it up. That is the ordinary dispatch rule, not something
  preemption does for itself: see [priority is not
  advisory](#priority-is-not-advisory).
- **The host has to be dispatching.** Paused, draining, past its
  `--ttl-hours` or within five minutes of it, nothing is stopped: the cards
  would go to nobody, and a job stopped that close to the end of a pod's life
  may never be queued again at all.

There is **no limit on how often** one job gives way, and none on how long it
then waits. A host with a steady supply of more important work may never run it
at all; that is what marking it auto-preemptable asked for. `gpuc status` shows
`auto-preempt` on those jobs, queued or running, and the dispatcher log and the
job's own log name the job each preempt made room for.

## What gets synced to the host

`gpuc submit` rsyncs `git ls-files --cached --others --exclude-standard`: every
tracked file **and** every untracked one git would keep, because a file written
and not yet `git add`ed is part of the experiment. `.gitignore` is still obeyed,
so venvs and caches stay home; files in the index but deleted on disk are
dropped rather than sent. One line says what happened:

```
syncing 43 files (2 modified, 1 untracked, ignoring .gitignore'd)
```

Alongside the workdir go `uncommitted.patch` — `git diff HEAD -- .` taken
against a *copy* of the index with `git add -N` applied, so it carries untracked
files too and never touches your staging area — and `source.json` with the
commit, branch, origin and submitting directory.

`--no-git` rsyncs a directory that is not a repository, minus `.venv`,
`__pycache__`, `.git`, `*.pyc`, `node_modules` and `.uv-cache`, with a warning:
nothing reads `.gitignore` in that mode, and **`gpuc requeue` cannot rebuild a
`--no-git` workdir** — it re-syncs whatever your working directory holds at
requeue time.

## Commands

**`gpuc submit <job.yaml|-> --host NAME`** — validate, sync the workdir, deliver
secrets, enqueue. `--no-git` above; `--no-bootstrap` enqueues even when the
host's package is older than this machine's (without it the package is re-synced
and the dispatcher restarted first, see [setup.md](setup.md#upgrading));
`--use-shared` is `use_shared: true` from the command line, see
[shared GPUs](#shared-gpus). `--runpod` and its flags are [below](#runpod).

It then asks the host where the job landed, so the last thing it prints is when
the job is expected to run rather than only that it was queued:

```
job 20260915-233000-112233 queued on host spar (attempt 1)
  queue: position 2 of 5; starts in ~3h20m
  logs: gpuc logs 20260915-233000-112233 -f
```

The start time is [projected](#job-length-estimates) from what the jobs ahead
estimated. Where it cannot be projected the line says so *and why* — `start time
unknown (host spar is paused, so nothing is being dispatched)`, or a job ahead
that gave no estimate, or a job asking for more cards than the host has — and
the whole line is absent when the host could not be asked again, since the job
is queued either way. A dispatcher that got there first prints `dispatched
already; it is running now`. `gpuc reorder` prints the same line, which is how
you check that a move did what you wanted.

**`gpuc status`** — per host: kind, reachability, how many cards are free,
dispatcher heartbeat, one line per owned card (`free` / `busy` / `UNAVAILABLE`,
and what the card is) and per [shared](#shared-gpus) one, the queue, running jobs with phase, elapsed time, last util,
the cards they hold (`gpu=2,3`) and any
[end-time estimate](#job-length-estimates), and recent finished jobs. Every job
is `name (job-id)`. It is the at-a-glance view: card UUIDs are in
`gpuc host list`, everything a host can say about itself is in
`gpuc host probe`, and `--json` carries more than either. `--host H` narrows it;
`--recent N` (default 5) and `--since 24h|7d|90m` (a bare number means hours)
choose how much of the finished list to show; `--all` adds jobs only the local
index and the S3 index know, which is how you find what was on a host that lost
its state; `--json` is [below](#exit-codes-and---json). `--suspects` lists running
jobs that are billing but idle and **never kills anything**: a job is a suspect
only in phase `main`, judged by **its own `low_util` settings** as the host
reports them, so a job that raised its floor is judged by what it asked for and
one with `enabled: false` is never listed. It also flags a pod past a TTL it
actually has.

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

The `shared` lines are the cards this host borrows rather than owns. `free`
means gpuc would take one right now, `busy` means one of *our* jobs has it, and
`IN USE` means somebody else does — with their memory and utilization, because
"there is a card there, why is my job queued" is the question that block gets
asked.

A job's `util` is labelled `(host)` — the host's own nvidia-smi sampler — only
on a host that also prints a pod line, where the provider's `provider util` is
on screen to confuse it with.

**`gpuc logs <job-id> [-f] [-n N] [--host H]`** — tails `log.txt` on the host
(`-n` defaults to 200). If the host cannot produce it, gpuc says why — including
"was purged" when the whole job dir is gone — and falls back to the S3 mirror,
which needs `s3_bucket` set here **and** an `s3_prefix` for that job (from the
job's index entry, else the host's). `-f` follows the host's file and never
falls back, and cannot be combined with `--json`.

**`gpuc ssh <host|job-id> [--print] [-- CMD ...]`** — the hand version of the
transport, with gpuc's own key, port, `known_hosts` and ControlMaster socket,
none of which are in your `~/.ssh/config`. A host name lands in its gpuc home; a
job id lands in that job's `workdir/`, falling back to the job dir when the
workdir has been cleaned away. The words after `--` are joined with spaces and
run by a **login bash** in that directory, exactly as `ssh host CMD` does, so
`gpuc ssh <job-id> -- 'ls | wc -l'` is a pipeline and not a filename — and
**gpuc exits with the remote command's own exit code**. `--print` prints the
equivalent command line instead of running it. A `local` host gets your own
`$SHELL` with no ssh at all.

**`gpuc cancel <job-id>`** — see [how a job is killed](#how-a-job-is-killed).

**`gpuc reorder <job-id> --priority N`** — queued jobs only; a running or
finished job cannot be reordered (exit 1). It prints the job's new queue
position and start time, and the new priority is recorded in the job's spec on
the host and in its S3 mirror — so `gpuc requeue` carries the move — as well as
in the queue marker, so `gpuc status` can still say what priority a job was
dispatched at once the marker is gone.

**`gpuc preempt <job-id>`** — stop a *running* job and queue it again, so
something more important can have its GPUs. Its runner stops it and syncs
whatever it produced, exactly as a cancel does, and then the host queues the
same job id again as its next attempt. Nothing is re-synced from here and the
job never leaves its host; `gpuc requeue` is the command for a finished job, or
for another host.

It **starts over**: the attempt that was stopped keeps nothing but its log, and
its `setup:` and `command:` run again from the top. What it does *not* get is a
fresh workdir — that directory was rsynced from your checkout once, at submit,
and it is still exactly as the stopped attempt left it, part-written
checkpoints and all. So preempt a job that tolerates being re-run over its own
leftovers, and not one whose `setup:` would trip over them. (The `outputs:`
baseline is not re-taken, so anything the stopped attempt produced still counts
as this job's output rather than as a file the checkout came with.)

It comes back at its own priority unless `--priority N` changes it (recorded in
the spec and its S3 mirror, like `gpuc reorder`). The job you are making room
for takes the cards next if it is queued at a **lower** number — the ordinary
case, and no flag is needed for it. At the **same** priority it does not:
dispatch order is `<priority>-<job id>`, and the preempted job was submitted
first, so its id sorts ahead and it takes its own cards straight back.

A job may also volunteer for this: `auto_preempt: true` in its spec has the
host do it whenever something more important is waiting, with no command at
all. See [automatic preemption](#automatic-preemption).

**It only works when something can take its place.** Preempting costs the job
everything it has done, so a preempt that would just re-run the same job is
refused (exit 1) rather than quietly doing that: nothing else queued, nothing
queued that sorts ahead of where this job would land (pass `--priority` above
that job, and the refusal says which one), or a host that is paused or
draining and so is dispatching nothing at all. **Queue the job you want to run
first, then preempt.** Queued and finished jobs are refused too: `gpuc reorder`
moves a queued one, `gpuc requeue` re-runs a finished one.

The host decides the re-queue when the attempt actually stops, and there are
four cases where it does not happen — the job finished, or failed for a reason
of its own, in the seconds before the kill reached it (a re-run would be a
retry nobody asked for); it was cancelled while it was stopping; its workdir is
gone; or the host is draining or past its `--ttl-hours` and is about to stop
existing. In all four the job stays finished, with `gpuc requeue` as the way to
re-run it, and the dispatcher log says which case it was.

**`gpuc estimate <job-id> --minutes N`** — set (or `--clear`) a queued or
running job's `estimated_runtime_min`; see [job length
estimates](#job-length-estimates).

**`gpuc requeue <job-id>`** — re-reads the spec from the S3 mirror and submits
it again as attempt+1, with the workdir re-synced from your *current* directory.
It therefore **needs `s3_bucket`** (without it, submit the job file again) and
cannot rebuild a `--no-git` workdir. `--host H` sends it somewhere else;
`--runpod` provisions for it; with neither, it goes back to the host the local
index says it ran on. It is the other half of the pair with `gpuc preempt`: a
new job id from the mirror, on whichever host you name, for a job that has
already finished.

`--host` is optional on `logs`, `cancel`, `preempt`, `reorder`, `estimate` and `requeue`:
the local job index is tried first, then every registered host is asked whether it knows the
id. An unknown job or host is exit 4.

**`gpuc skill`** — prints the agent guide ([`skills/gpuc/SKILL.md`](../skills/gpuc/SKILL.md))
to stdout, so an agent can read it with `!gpuc skill` without knowing where the
repo is. `--install [DIR]` writes it to `DIR/.claude/skills/gpuc/SKILL.md`
instead (`DIR` defaults to the current directory) and refuses to overwrite an
existing copy without `--force`.

**`gpuc clean`**, **`gpuc pods`**, **`gpuc reconcile`** and the host commands
have their own sections below and in [setup.md](setup.md). `gpuc version` prints
this build, its commit, and the commit this machine last shipped to each
bootstrapped host, marking the ones to re-bootstrap; it reads the registry's
cache and never touches a host, so what a host is *running* is `gpuc status` (see
[the same host from two machines](setup.md#the-same-host-from-two-machines)).

**`gpuc web serve`** — the [web dashboard](#the-web-dashboard): the same
status, host list and config in a browser, with cancel, preempt, re-prioritise,
estimate and a log tail per job.

## The web dashboard

`gpuc web serve` is `gpuc status`, `gpuc host list` and `gpuc config show` on
one page, refreshed every 15 seconds, with a button for each of `gpuc cancel`,
`gpuc preempt`, `gpuc reorder` and `gpuc estimate` and a **Logs** panel that tails
`gpuc logs` (tick *follow* to keep tailing). Every job also links to where its
`outputs:` went — the S3 console for an `s3:` output, the repo tree for an `hf:`
one — to its W&B run when the job's `env` names `WANDB_ENTITY`,
`WANDB_PROJECT` (and `WANDB_RUN_ID`), and to its mirrored log under the host's
`s3_prefix`. The links are derived from what the job *declared*, never checked:
an `outputs not uploaded` flag beside one means the link is empty.

```sh
gpuc web set-password          # once; prompts twice, stores a bcrypt hash 0600
gpuc web serve                 # http://127.0.0.1:8646/
gpuc web serve --bind 0.0.0.0 --port 8646   # reachable from other machines
gpuc web serve --bind 0.0.0.0 --install     # the same, as a systemd --user service (see setup.md)
```

Every page and every API document is behind that one password (only the
stylesheet, the script and the login form itself are served without it). The
server refuses to start until one is set. Sessions live in the server's memory
(a restart logs everyone out) and the cookie is `HttpOnly; SameSite=Strict`, but
there is **no TLS**: bind to localhost or a VPN interface, or put it behind a
TLS-terminating proxy. Wrong guesses are answered one at a time with a growing
pause.

It is a thin wrapper over the same code the CLI runs, and it has no
functionality of its own: what it shows is the `--json` documents, what it can
do is the commands. The API it uses is plain HTTP, once the session cookie is
held:

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
prints, with the exit code mapped onto the status: 2 is 400, 3 is 503, 4 is
404, 1 is 500; a request with no session is 401. The `mirror` link is built
from the host's *current* `s3_prefix`, where `gpuc logs` prefers the prefix
the job's own index entry recorded.

## RunPod

```sh
export RUNPOD_API_KEY=...
gpuc submit job.yaml --runpod --gpu A40 --max-price 0.60
gpuc pods                   # every pod with our prefix: cost, util, age, is it wanted?
gpuc reconcile --once       # forget pods that are gone, enforce TTLs, report what nothing claims
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
| `--ttl-hours N` | none (`-1`) | opt-in hard cap on the pod's life |
| `--disk GB` / `--image REF` | `config.toml` | container disk and pod image |
| `--no-reuse` | reuse is on | always create a new pod |
| `--name-hint TEXT` | `job` | goes into the pod name after the prefix |
| `--health-args "..."` | none | extra flags for the on-host health check, e.g. `--min-mbps 0.1` |

**Offers.** One catalog query per requested tier. An offer is dropped if the
name does not match, availability is `NONE`, its per-GPU VRAM is under
`--min-vram`, the tier has no price, `price × --gpu-count` is over
`--max-price`, the tier cannot supply `--gpu-count` cards, or it reports no
available CUDA version at or above the floor. What is left is sorted by price
ascending (then GPU id, then tier) and tried in that order: create, wait for a
direct SSH endpoint, bootstrap, health check, enqueue. Any failure terminates
that pod and moves to the next offer, all inside a **15-minute ceiling**.

**Caps.** `max_pods` and `max_total_usd_per_hour` are account-wide — checked
against every pod with our prefix, whoever created it, immediately before
`create` and with the local state lock held. Because offers are price-ascending,
a cap the cheapest offer trips aborts the whole submit rather than walking the
list. Two submits on this machine cannot both slip past the caps; two different
machines sharing one account still can, for the length of one `create`.

**Reuse** is the default, and picks the first registered `runpod` host that
satisfies *all* of: it has a `desired/` record; that record's offer still
matches the request (GPU name, `--min-vram`, `--max-price`, tier, CUDA floor);
it owns at least `--gpu-count` cards; the provider says the pod exists and is
`RUNNING`; its dispatcher heartbeat is under 30 s old; it is not draining; it is
not paused. A registered pod the provider no longer has is forgotten on the spot
rather than dialled. Image, disk, `--idle-min` and `--ttl-hours` are *not*
compared: a reused pod keeps the ones it was created with.

**The pod is not tied to the machine that bought it.** What it is — its cards,
its mirror, its TTL, and the record of what it was rented as — lives in its own
`config.json`, so `gpuc host add <name> --pod <pod-id>` on a second machine
registers it from the pod itself and needs nothing the first machine has. That
is also how `gpuc reconcile` tells a pod from a leak wherever it runs
([below](#reconcile)).

**A pod is never created without a record of it.** `desired/<host>.json` is
written under the same lock as the create, and every exit from provisioning
between `create` and the final registry write — Ctrl-C included — terminates the
pod before unwinding.

<a name="auto-down"></a>
**Auto-down.** The pod terminates itself when nothing is running and the queue
has been empty for `--idle-min`, or after two consecutive `low-util` failures.
Either way it drains first: it retries any unconfirmed outputs, mirrors every
job's log and state, and then calls the provider. A failed mirror does **not**
hold up the terminate — the state is already on disk here, and a bucket we
cannot reach is not a reason to keep a paid pod billing. Only a failed
*terminate* stops the shutdown: it logs loudly, keeps dispatching, and retries in
10 minutes.

**There is no overall TTL by default**, because a wall clock that kills a
training run at hour 24 is a worse failure than a pod that idles for fifteen
minutes first. `--ttl-hours` is opt-in and **does kill a running job**: past the
cap the dispatcher asks each runner to stop with reason `ttl`, lets it sync, then
drains and terminates, and `gpuc reconcile` enforces the same cap from the
provider's own `createdAt` as a backstop. `gpuc submit --runpod` refuses a job
whose `max_runtime_min` is longer than the TTL you asked for. `--ttl-hours -1`
means no cap, on `submit`, `host add` and `host set` alike.

<a name="reconcile"></a>
**What stops a forgotten pod.** `gpuc reconcile` (see the timer in
[setup.md](setup.md#the-reconcile-timer)) terminates:

- a bootstrapped pod that has neither beaten its heartbeat nor been seen running
  a job for `dead_dispatcher_minutes` (30 by default) — including one whose ssh
  stopped answering, since that never refreshes `last_seen_at` either. **The
  clock counts silence this machine watched**, not wall clock it was away for:
  after a suspend, a reboot, or a gap of more than five minutes between passes,
  every host gets the full allowance again. A machine that was asleep for three
  days did not see anything, and the pass where it wakes up is the one where its
  own ssh is likeliest to fail;
- a pod past a TTL its host actually has;
- a pod *this machine created* that never bootstrapped by its 15-minute ceiling.
  An adopted pod has a config on it, which is all this machine can see, so the
  dead-dispatcher rule is what judges it instead — including a pod you adopt
  with `--pod` and never bootstrap, which has no dispatcher to beat and is
  terminated after `dead_dispatcher_minutes`. `gpuc host add --pod` says so
  when it registers one.

Those are the three states a pod cannot get itself out of. Everything else it
handles alone: a healthy pod drains and terminates itself once its queue has
been empty for `--idle-min`, and enforces its own TTL, with nothing local
involved.

**Nothing is terminated for the absence of a record.** A pod with our prefix
that this machine has no record of and cannot get an answer out of is reported
every pass, with its age and its hourly cost, and left running — it may be
wedged, it may hold no key of yours, or it may be another machine's `create`
still bootstrapping, and those look identical from here. The report says how to
take it over (`gpuc host add <name> --pod <id>`) or where to end it. That is the
one case that needs you: a pod that never got a config *and* whose creating
machine is never coming back bills until somebody kills it.

**Which pods are "ours" is asked of the pods, not of this machine.**
`desired/<host>.json` is written by whichever machine ran `gpuc submit
--runpod`, so a pod carries the same record itself: its `config.json` holds the
offer it was bought on, when it was created and when it was bootstrapped, under
the `provider` block. Every pass asks each prefixed pod it has no record of what
it is (one ssh session: expand gpuc home, read `config.json`), and a pod holding
a gpuc config is ours whoever created it — it is judged
by the rules above, and its answer is cached in `desired/` here. So the timer
can run on the desktop while the laptop that queued the job is switched off, and
neither machine is special.

A pod with a job running per the host's own state is never touched by the
dead-dispatcher rule, however old it is — but a TTL you set overrides that and
kills the job. The reaper otherwise fails closed in every direction: it never
touches a pod without the configured prefix, it terminates nothing at all if
`desired/` is unreadable, and a terminate that fails keeps its record (so the
next pass retries it) and makes `gpuc reconcile --once` exit non-zero.

**`gpuc pods`** lists every pod in the account: ours (name, id, status, GPU,
`$/h`, CUDA, age, util, `DESIRED`, heartbeat) with the hourly total, and other
people's by name only, never touched. `DESIRED=NO` means nothing *here* wants it
yet — `gpuc reconcile` asks each of those what it is before deciding.
`--no-heartbeat` skips the per-pod dispatcher ssh check, which is what makes the
command slow when a pod is wedged. Nothing here terminates a `DESIRED=NO` pod:
reconcile takes on the ones running gpuc, and the rest are yours to end.

## How a job is killed

Each phase runs in a transient `systemd --user` scope where the host has one,
and in its own process group where it does not (`isolation: cgroup` or `pgid` in
`state.json`, shown by `status`). This matters for exactly one case: a
grandchild that double-forks (`setsid`, `nohup`, a daemonising server) escapes
the process group and survives `kill -- -PGID`, holding a GPU the next job is
about to get — but it cannot leave its cgroup, so `systemctl --user stop <unit>`
reaps the whole tree. Under `pgid` (every RunPod pod, most shared boxes) that
hole is real and not fixed.

The runner owns the kill: it stops the scope, or SIGTERMs the job's process
group and SIGKILLs it 15 s later, then runs the final sync and writes the final
state. `gpuc cancel` writes a marker the runner checks before every phase and on
every poll; a job cancelled before its first phase never starts one, and a
queued job is cancelled by removing its queue marker. The dispatcher escalates
only if the runner does not act: after 15 s it stops the scope and SIGKILLs the
job's group, after 30 s it SIGTERMs the runner itself, after 45 s it SIGKILLs
the runner's group. It never signals the runner's group during the launch
window, when the runner is the only member of it.

`gpuc preempt` (and [`auto_preempt`](#automatic-preemption), which is the same
thing without the command) uses the same machinery with one extra marker:
the runner stops
the job and records `failed: preempted` after its final sync, and the dispatcher
then writes the job's state back to `queued` as the next attempt and puts a
queue marker back. So a preempted job is briefly visible as `failed:
preempted` — that is the record of the attempt that was stopped, and anything
polling `gpuc status --json` will see it for a second or two before the job
reads as `queued` again. The workdir and the job's secrets file are kept
whatever `cleanup:` says, because the next attempt is that same job id and
nothing delivers either a second time.

The low-util watchdog samples the assigned cards every 30 s **during phase
`main` only**, so downloads and compiles in `setup` can never look idle. Once
`grace_min` minutes of `main` have passed, a rolling mean below `floor_pct` over
a full `window_min` window kills the job as `failed: low-util`. A sample
nvidia-smi could not produce is recorded as unknown and never counted as 0%.
Two consecutive low-util failures **pause** the host: it stops dispatching and,
if it is ephemeral, drains and terminates — but never out from under a job.
With anything still running it asks those runners to stop (reason
`low-util-pause`) and drains on a later pass, so their outputs are uploaded.
The pause is a file on the host and survives a dispatcher restart; clearing it
is `gpuc host resume <host>` (below).

Every `failed: <reason>`:

| reason | what happened |
| --- | --- |
| `exit <N>` | `command` exited non-zero and nothing else killed it |
| `setup` | the `setup` phase exited non-zero |
| `gpu-assert` | an assigned GPU (index or UUID) is not present in the host's `nvidia-smi` |
| `gpu-preflight` | a real GPU op inside the job's venv failed, or `device_count()` did not match `gpus:` — usually a CPU-only torch |
| `sync-preflight` | the uploads the job would do at the end cannot work (no `aws`/`hf`, a missing secret, an unwritable bucket or repo) |
| `low-util` | the GPU sat under `floor_pct` for a full `window_min` of `main` |
| `low-util-pause` | the host paused after two low-util failures and asked this job to stop so it could drain |
| `timeout` | `max_runtime_min` elapsed |
| `ttl` | the host's opt-in `--ttl-hours` cap elapsed |
| `preempted` | `gpuc preempt`, or the job's own `auto_preempt`, stopped this attempt; the job is queued again as the next one, and this is the record of the attempt that was stopped |
| `terminated` | the runner itself was signalled (and the job was not cancelled) |
| `sync` | the final upload failed; the run itself may have been fine. A succeeded job becomes `failed: sync`, and any other reason gains `+sync` |
| `no-outputs` | an `outputs:` path was never written, or holds only files that came with the checkout. Appends `+no-outputs` the same way |
| `bad-spec` | the queued spec could not be read |
| `needs N GPUs, host owns M` | the host's ownership shrank after the job was queued. On a host with [shared cards](#shared-gpus) it counts the ones this job asked for, and says so when it asked for none |
| `spawn-failed` | the dispatcher could not start a runner process |
| `incomplete-submit` | `gpuc submit` was interrupted before it finished queueing the job, so this host was never asked to run it. Submit it again |
| `runner-died` | the runner vanished without writing final state; the dispatcher kills anything it left behind before freeing its GPUs. A job whose state never recorded a pid is not automatically this: the dispatcher looks for the runner itself first, and fails the job only when there is none |

`cancelled` is a status of its own, not a failure.

## Exit codes and `--json`

| code | meaning |
| --- | --- |
| 0 | ok. **Includes** a host that is unreachable, has a dead dispatcher, or whose pod is gone: that is data about a host, reported per host, not a failure of the command |
| 1 | the command failed (transport, provider, a refused submit, a `clean` the host reported errors for, a `host bootstrap --all` any host failed — the others still upgraded, and the tally says which) |
| 2 | usage: a bad flag, a missing required one, a bad `--since` |
| 3 | local state is unreadable (`hosts.json` or `config.toml`), so the answer is **unknown** |
| 4 | the job or host named on the command line does not exist |

A single unreadable host entry never reaches these: it is skipped with a warning
on stderr, every other host still works, and the entry is written back untouched
by the next `gpuc host add|set` — it is probably another session's host. Only a
`hosts.json` that cannot be parsed at all is exit 3, and it prints the error,
the path, and that a `.bak` was kept.

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

The job objects in `queued`, `running` and `finished` all carry the same
fields. Beyond what the example shows:

- `priority` (0–99, lower first) is on **every** job, and `queued` is already
  in dispatch order, so `jq '.hosts[].queued | sort_by(.priority)'` reproduces
  the order the host will take them in. Null only when the host did not say
  (a build too old to report it, or a spec it could not read).
- `gpus_requested` is how many cards the spec asked for; a queued job holds
  none yet, so its `gpus` is empty. `use_shared` is whether it may take one of
  `shared_gpus`: with both, two jobs queued on one host can be told apart when
  only one of them is waiting for a card the host owns. Null, like `priority`
  and `auto_preempt`, only when the host did not say — never as a stand-in for
  `false`.
- `starts_in_s` / `starts_at` are when a queued job's turn is expected (see
  [job length estimates](#job-length-estimates)); null for anything not
  queued, and for a queued job whose turn cannot be dated.
- `eta` is absolute and `eta_s` the same instant as seconds from now (negative
  once overdue); both null without a length estimate, and null again once the
  job is finished. `progress_pct` survives the job so you can see how far it
  got; `progress_error` is why the last poll produced nothing.
- `attempt`, `started_at`, `exit_code`, `outputs_lost`, `workdir_bytes`,
  `suspect` (the `--suspects` judgement), `outputs` (the spec's, as the host
  holds them) and `links` — one `{kind, path, target, url}` per place the
  results, W&B run or mirrored log can be opened (`kind` is `s3`, `hf`,
  `wandb` or `mirror`), derived from what the job declared and never checked.
- A job's `util` is its **last** sample from the host's own nvidia-smi; a
  pod's `provider_util` is the provider's per-GPU reading for the whole pod,
  null for any other host. Two measurements that will differ.

Per host: `target`, `draining`, `paused`, `pod_gone`, `pod` (the provider's
view of an ephemeral host's pod, null elsewhere) and `pkg_commit`, the host's
own answer for the build it runs — `null` means the host did not say, never
"up to date", and a reachable host that did not say is one on a build old
enough that it cannot, which `errors` reports like any other mismatch. A card
the host cannot see appears in `gpus` as `{"owned_as": "3", "available":
false}`; `shared_gpus` has the same shape plus `unused` (the host's verdict:
no memory held and no work running) and `busy_job` (one of *our* jobs has it),
and a missing one is `{"shared_as": "5", "available": false}`. `--recent` and
`--since` apply to `--json`; `--suspects` and `--all` do not.

Rules for anything automated:

- **Key on `hosts[].running`.** It is the host's own answer; an empty list means
  the host said nothing is running.
- **Read queue order from `priority`**, not from `eta` or `estimated_runtime_min`:
  those say how long a job takes, not when it is taken.
- **Treat exit 3 as "unknown", never as "nothing is running".** So is a
  non-empty top-level `errors`, and so is `"reachable": false` for the host you
  care about — we could not ask it.
- Unknown keys will be added over time; ignore the ones you do not know.

### `--json` everywhere else

`status`, `submit`, `requeue`, `logs`, `cancel`, `preempt`, `reorder`, `estimate`,
`pods`, `version`, `clean`, `config show`, `host list`, `host probe` and
`reconcile --once` take `--json`, under the same rules: **stdout is exactly one JSON object**, it carries `schema_version`,
and everything the text output would print alongside it — progress, warnings,
`note:` lines — goes to stderr instead. The exit codes are the table above,
unchanged by the flag.

**A command that failed prints a document too**, so a caller parsing stdout is
never handed nothing at all:

```json
{ "schema_version": 1, "error": "no registered host knows job 20260915-120000-abc123.\n...",
  "exit_code": 4 }
```

`error` (singular) is the whole answer: the command did not do what it was
asked. A command line argparse itself rejects (a bad flag, a missing required
one) gets the same document, with the reason on stderr where argparse wrote it. `errors` (plural) is different — per-host or per-job trouble a command
survived, which never implies a non-zero exit by itself (`clean` and
`reconcile --once` are the two that exit 1 on their own `errors`).

| command | the document |
| --- | --- |
| `submit`, `requeue` | `{job_id, host, attempt, requeued_from, notes[], queue_position, queue_length, dispatched, starts_in_s, starts_at, starts_unknown}`. `requeued_from` is the id this run came from, null on `submit`; `notes` are the text output's `note:` lines and do not mean the job was not queued. The queue fields are the host's answer a moment *after* the enqueue: `queue_position` is 1-based in dispatch order, `dispatched` is true for a job the host started before we could look, `starts_unknown` says why there is no start time (a paused or draining host, a job ahead that estimated nothing, a job that asks for more cards than the host has) and is null when there is one, and every one of them is null when the host could not be asked again — never a reason to think the job was not queued |
| `logs` | `{job_id, host, source, location, lines[], notes[]}`. `source` is `"host"` or `"s3"` and `location` is the remote path or the `s3://` uri it was read from; `lines` is the log with no trailing newlines. **Not with `-f`** — a stream has no end, so `--json -f` is exit 2 |
| `cancel` | `{job_id, host, status}` — the host's own word, `cancelled` for a queued job or `cancelling` for a running one |
| `preempt` | `{job_id, host, status, priority, warnings[]}`. `status` is the host's own word (`preempting`); `priority` is what it will be queued again at, which is the job's own unless `--priority` changed it. `warnings` carries a mirrored spec that could not be updated, exactly as `reorder` does |
| `reorder` | `{job_id, host, priority, warnings[]}` plus the same `queue_position`, `queue_length`, `dispatched`, `starts_in_s`, `starts_at` and `starts_unknown` as `submit`, so a move can be checked without a second call. `warnings` carries a mirrored spec that could not be updated, which means `gpuc requeue` would re-run the job at its old priority |
| `estimate` | `{job_id, host, estimated_runtime_min, status, warnings[]}`. `estimated_runtime_min` is what the spec holds now (null after `--clear`) and `status` is the job's, since only a queued or running one can be set; `warnings` carries a `max_runtime_min` contradiction and a mirrored spec that could not be updated |
| `pods` | `{pods[], hourly_usd, others[], notes[]}`. Each pod is `{id, name, status, gpu_name, gpu_count, cost_usd_hr, cuda_version, age_s, created_at, gpu_utils[], desired, heartbeat_age_s}`; `others` are pods without our prefix, `{id, name, status}` only, because we never touch them |
| `version` | `{version, commit, source, dirty, python, executable, hosts[], errors[]}`, each host `{name, pkg_commit, seen_at, current}`. `pkg_commit` here is the commit the host was running when this machine last read it, not what it runs now — that is `status --json`'s `pkg_commit`. Exit 3 if the registry is unreadable |
| `config show` | `{config_file, config_file_exists, state_dir, settings{}, notes[]}` — the effective settings, file or not |
| `host list` | `{hosts[], errors[]}` — each registry entry: the address (`name`, `kind`, `ssh`, `port`, `gpuc_home`, `persistent_root`, `pod_id`), the host's own config as last read (`gpus`, `s3_prefix`, `env`, `cache_dir`, `idle_minutes`, `ttl_hours`, `retention_days`, `pkg_commit`) flattened beside it with `config_seen_at` saying when that was, the raw `cache` it came from, plus `remote_home`, `ephemeral` and `warnings[]` (a re-bootstrap note: nothing here asks the host). The host's `env` is reported by **name only** (`{"HF_TOKEN": "<set>"}`), because `--env` is free-form and this document travels. A skipped entry is an `errors` string, not a host. Exit 3 if the registry is unreadable |
| `host probe` | `{host, sections{}, driver_version, has_nvidia_smi, gpus[], assigned_gpus[], assigned_missing[], home_fs_type, home_is_overlay, persistent_root, uv_cache{}, notes[]}`. `gpus` is **every** card the host has whatever `--all-gpus` said, each one `{uuid, name, vram_mib, index, assigned}`; `assigned_gpus` is this host's `--gpus` as registered and `assigned_missing` the entries in it no card answered to (always empty on a host with no nvidia-smi, which has nothing to answer with). `sections` is the probe script's raw output section by section, so anything this build does not interpret is still there |
| `clean` | `{host, dry_run, purge, freed_bytes, removed[], skipped[], purged[], purge_skipped[], incoming_removed[], verified[], notes[], errors[]}`. The job objects are the host's own: `{job_id, status, bytes, age_days}`, plus `why` on the skipped ones and `forced` on a purged job that had no confirmed backup |
| `reconcile --once` | `{terminated[], forgotten[], kept[], unclaimed[], errors[]}`, host names in the order they were judged — except `unclaimed`, which is *pod* names this machine has no record of and will not terminate. `--json` needs `--once` and nothing else: neither the loop nor `--install` has a document to print |

```sh
gpuc submit job.yaml --host gpubox --json | jq -r .job_id
gpuc logs "$id" --json | jq -r '.lines[-20:][]'
gpuc pods --json | jq '[.pods[] | select(.desired | not) | .name]'
gpuc clean --host gpubox --all-finished --dry-run --json | jq .freed_bytes
```

## Cleanup and retention

A job's `workdir/` is the rsynced code *and* whatever the job builds in it —
usually a venv, and a torch venv is about 6.5 GB. It is also the only part of a
job dir gpuc will delete, because it is the only part that can be recreated.

**Per job, by the runner.** The policy is applied after the final sync and the
final state write — never before, because `outputs:` paths live *inside* the
workdir — and recorded as `workdir_removed`.

| `cleanup:` | succeeded | failed | cancelled |
| --- | --- | --- | --- |
| `on_success` (default) | removed | **kept** | **kept** |
| `always` | removed | removed | removed |
| `never` | kept | kept | kept |

No policy ever touches a job that is not finished. `spec.json`, `state.json` and
`log.txt` always stay, so `logs` and `status` keep working on a cleaned job.

A workdir the policy keeps is not kept forever: the host sweeps it once it is
`--workdir-days` old (a day by default, below). `cleanup: never` is the way to
opt a job out of that too.

**After the fact: `gpuc clean --host H`.**

```sh
gpuc clean --host gpubox --all-finished --dry-run   # what would go, and how big
gpuc clean --host gpubox --all-finished             # every succeeded/failed/cancelled job
gpuc clean --host gpubox --older-than 7             # only jobs that ended over 7 days ago
gpuc clean --host gpubox --only 20260101-120000-ab12,20260101-130000-cd34
```

`--only ID[,ID...]` names the jobs itself, so it replaces `--all-finished` and
`--older-than` rather than combining with them: exactly those jobs and no
others, however recently they ended — and even if the job never recorded when it
ended, which is the shape of a job whose state write was cut short. Naming ids
is its own confirmation, so `--purge --only` needs no `--yes`, and the workdir
sweep `--purge` implies is scoped to the same ids — purging one job does not
reclaim every other finished job's venv on the way past. What naming a job does
*not* waive is the preconditions: a named job with no confirmed mirror or
unconfirmed outputs still needs `--force`, and a running or queued one is never
touched.

An id no job dir on the host matches is a typo, so the whole selection is
refused: nothing is removed, the id is named, and the command exits 1. An empty
`--only` is a usage error (exit 2), never a silent "everything" or a silent
no-op.

|  | `clean` | `clean --purge` |
| --- | --- | --- |
| `workdir/` (code, venv, outputs) | removed | removed |
| `spec.json`, `state.json`, `log.txt` | **kept** | removed |
| the job's `secrets/<id>.env`, if any is left | kept | **removed** |
| stale `incoming/<id>.json` staged specs | removed | removed |
| a stray queue marker for the job | — | removed |
| running or queued jobs, or ones with unreadable state | never touched | never touched |

`--purge` removes the whole `jobs/<id>/` of finished jobs and defaults to
`--older-than 7`; it implies the workdir clean over everything else. It has two
preconditions, both read from the job's own `state.json` on the host:

- **backed up**: `meta_synced_at` is set, which happens only after the host's
  own final upload of `log.txt` and `state.json` returned 0. Otherwise the skip
  reason is `not backed up: no s3_prefix on this host` or `not backed up: final
  upload failed`.
- **outputs confirmed**: `outputs_synced_at` is set, or the spec declares no
  `outputs:`, or the workdir is already gone, or nothing was ever written under
  the declared paths — a job that died before it produced anything, or whose
  output dir holds only files that came with the checkout, has nothing to lose.
  Otherwise `outputs not confirmed uploaded` — `outputs:` paths live inside the
  workdir and a failed job keeps its workdir, so purging one could bin the only
  copy of a checkpoint. That last question is answered conservatively: a path
  that cannot be read, a symlink, or a declared path that cannot even be
  resolved all count as content, so the answer errs towards keeping the dir.

`--force` overrides those two and nothing else, and says so per job.
`--verify` (control side only) HEADs each candidate's mirrored `log.txt` under
the prefix **that job recorded in `meta_synced_to`** (falling back to the host's
registered `s3_prefix`) with your own credentials, and purges only what
answered; without it the host's `meta_synced_at` is trusted, which is the only
answer a host with no credentials of ours can give. `--purge --all-finished` is
an age horizon of zero — it deletes the job dir of something that ended a minute
ago — so it needs `--yes`, or `--dry-run` to see the list first.

**Automatic, by the host: two horizons.** The dispatcher reclaims disk itself,
once at startup and then at most once an hour. On an ephemeral host that is its
whole life; on a **non-ephemeral host the dispatcher only lives while there is
work**, so both happen on your next submit rather than on a timer.

| | `--workdir-days N` | `--retention-days N` |
| --- | --- | --- |
| default | `1` | none |
| takes | `workdir/` only | the whole `jobs/<id>/` |
| needs a mirror | no | yes, and never forced |

`--workdir-days` is the one that keeps a busy host from filling up. A failed or
cancelled job keeps its workdir under the default `cleanup: on_success` so you
can look at it, and a day later you either have or you haven't — so the sweep
takes the checkout and the venv (a torch venv is ~6.5 GB) and leaves
`spec.json`, `state.json` and `log.txt`, which is everything `logs`, `status`
and `requeue` need. `gpuc requeue` re-syncs a workdir from git, so nothing here
is unrecoverable.

Because it runs with nobody watching, it refuses two things `gpuc clean` will
do if you name them:

- a job whose spec says **`cleanup: never`** — that is the one way to ask for a
  workdir to be kept, and a host default may not quietly mean "for a day";
- a job whose **`outputs:` have not reached S3 or HF**, the same precondition
  `--purge` fails closed on. Those paths live inside the workdir, so sweeping
  one would bin the only copy of a checkpoint — exactly the jobs `gpuc status`
  flags as `outputs not uploaded`. Upload them, `gpuc requeue` them, or take
  the workdir yourself with `gpuc clean --host H --only <id>`.

A job whose `spec.json` cannot be read is skipped too: it cannot say it wanted
this kept, but it cannot say it didn't either.

`--workdir-days ''` turns that horizon off. It does **not** mean workdirs are
kept until you run `gpuc clean` if `--retention-days` is also set: each purge
pass sweeps workdirs at *its* horizon (below), so that becomes the only one.

`--retention-days` is the purge, and deletes the record of the run, so it is
opt-in and only ever acts on jobs whose log and state the host has confirmed
mirrored. Each purge pass also does the ordinary workdir clean over *its* own
horizon, under the same two refusals — so on a host with no `--s3-prefix`,
`--retention-days` reclaims old venvs and nothing else. With both set, the
effective workdir horizon is whichever is shorter.

Either pass also clears staged specs (`incoming/<id>.json`) an interrupted
submit left behind, once they are an hour old.

Both live in the host's own `config.json`, and `gpuc host set <name>
--workdir-days N` writes through to it — so it needs the host to answer, and
takes effect without a bootstrap.

`--workdir-days` is also the one timer with a default, and only for a host
being configured for the first time. `gpuc host add` on a box that already has
a `config.json` adopts what is there: a host that has been getting along
without the sweep is not given one by being registered from another machine.

**Unconfirmed outputs.** `gpuc status` flags a finished job that *produced*
`outputs:` which never reached S3 or HF as `outputs not uploaded`, and lists
them per host, because those are the jobs a purge — or a pod going away — would
take with them. A job that only declared outputs and never wrote them is not
flagged: it has nothing to lose, and saying otherwise would both misreport it
and keep its job dir past every retention horizon. Note that `setup` runs after
the baseline is taken, so a job that died in setup *after* writing a checkpoint
is still flagged — that file is this job's doing.
An ephemeral host retries them while it drains (three tries a minute apart, five
minutes at most) and, if they still fail, records `outputs_lost` with the last
error and terminates anyway: the pod is billing, and whatever sent it away does
not pause. Such a job shows `OUTPUTS LOST` in `gpuc status` and in `gpuc status
--all`, and nothing recovers it but re-running the job.

`gpuc status` also prints one line per host once finished workdirs hold more
than 1 GiB, with the `gpuc clean` line to run. **The figure is what deleting
them would give the filesystem back, not what `du` says they hold**, and the
two are nowhere near each other. uv builds a venv out of its wheel cache, so
most of those bytes stay when the workdir goes — the cache still has them.
Measured on two hosts:

| | `du` | actually freed |
| --- | --- | --- |
| hardlinked venv (ext4) | 8.36 GiB | 0.13 GiB |
| reflinked venv (overlay on CoW) | 15.00 GiB | 0.34 GiB |

Both mechanisms are counted, because which one you get is your host's business:
the same uv against the same cache hardlinks on one box and reflinks on the
next. It is measured once, when the job ends, and recorded in its `state.json`
— a finished workdir does not change, and walking every one of them per call
cost `status` four seconds on a host holding sixty. `gpuc clean` measures
afresh, since it is about to delete what it is quoting. A workdir that is
already gone frees nothing and is reported as zero without measuring anything,
so a host that has been idle for a week still answers straight away.

Anything the filesystem will not answer about counts as reclaimable, so no one
file is ever under-counted. The total still can be, in one case worth knowing:
the kernel says an extent is *shared*, not *who with*, so on a filesystem that
snapshots your home (btrfs with snapper or timeshift) every workdir shares
everything with its snapshot and reports close to nothing. The delete really
does free nothing until the snapshot expires, but it is not what the number
looks like it is saying.

## Troubleshooting

| symptom | what it means | what to do |
| --- | --- | --- |
| `status` says `dispatcher DOWN` | nothing holds the host's lock, or its heartbeat is over 30 s old | `gpuc host bootstrap <host>` (idempotent); any `gpuc submit` also restarts it |
| `status` says `PAUSED (low-util)` | two consecutive jobs failed `low-util`, so the host stopped dispatching until told otherwise | fix the jobs (or their `low_util`), then `gpuc host resume <host>`. Re-bootstrapping restarts the dispatcher but does not clear the pause |
| provisioning gives up with "no direct SSH endpoint" | RunPod never exposed port 22 within the 15-minute ceiling — usually a bad placement | the pod was already terminated; re-run the submit, or widen `--gpu` / `--cloud any` |
| `ssh ... cannot create its ControlMaster socket` | the socket path would be over the 100-byte limit gpuc enforces | point `XDG_RUNTIME_DIR` at a short directory, or unset it to use `/tmp/gpuc-<uid>` |
| bootstrap fails with "host health failed" | the driver, disk or network check on the host said no | read the named check; fix the host (free disk, load the driver) and re-run bootstrap |
| job is `failed: gpu-preflight` | torch in the job's venv has no working CUDA, or sees the wrong number of devices | check the torch build against the host's driver (`gpuc host probe`), and that `gpus:` matches what the job expects |
| job is `failed: low-util` | the GPU sat under `floor_pct` for a full `window_min` of phase `main` | raise `low_util.grace_min`, lower `floor_pct`, or set `low_util.enabled: false` for genuinely CPU-bound work |
| job is `failed: sync` (or `...+sync`) | the final upload failed; the run itself may have been fine | check the tail of `gpuc logs <job-id>`; usually a missing `secrets:` entry for the destination, or no `aws`/`hf` on the host (re-run bootstrap) |
| job is `failed: sync-preflight` | the uploads the job would do at the end cannot work | the log names the exact command and error; fix the credential or destination, or add `hf_create: true`, then re-submit |
| job is `failed: ttl` | this host has an opt-in `--ttl-hours` cap and it ran out; the outputs were synced first | raise or drop the cap (`gpuc host set <host> --ttl-hours -1`), then `gpuc requeue <id>` |
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
| `status` says `POD GONE` | the pod is terminated or missing but the registry still lists it | `gpuc reconcile --once` |
| `reconcile` reports `DEAD DISPATCHER` and terminates a pod | it stopped beating (or answering ssh) for `dead_dispatcher_minutes` with nothing running | expected: that pod could no longer stop itself. Raise `dead_dispatcher_minutes` if your hosts go quiet legitimately |
| `reconcile` says a pod is "silent for 0 min" that has been dead for days | the clock counts silence *this machine watched*, and it has just started (a reboot, a resume, or a hand-run after a gap). The line says so | leave the timer running and it goes on the next pass past the limit; to end a pod now, tell the pod — `gpuc host set <host> --idle-min 0` |
| everything on a host is suddenly gone | the container restarted and `$HOME` was on the overlay | the runbook in [setup.md](setup.md#hosts-whose-home-is-wiped-on-restart) |
