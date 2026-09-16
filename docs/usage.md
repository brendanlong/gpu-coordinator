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
| `command` | **required** | run in `workdir/` as phase `main`. Must not be blank |
| `name` | `""` | a label for `status`; not an identifier |
| `setup` | none | run first, as phase `setup` |
| `gpus` | `1` | how many of the host's owned GPUs to assign (>= 0). `0` never waits for a card. More than the host owns is refused at submit |
| `env` | `{}` | plain environment for the job, applied after the host's `--env` |
| `secrets` | `[]` | names read from *your* shell at submit time and delivered to the host as `~/.gpuc/secrets/<job-id>.env` (0600). Missing from your shell is a refused submit |
| `outputs` | `[]` | `{path, s3}` and/or `{path, hf, hf_path, hf_create}`; `path` is relative to the workdir |
| `sync_interval_s` | `180` | background upload cadence; **minimum 10** |
| `priority` | `50` | `0`–`99`, lower dispatches first |
| `max_runtime_min` | none | wall clock from the runner's start; over it the job is `failed: timeout` |
| `estimated_runtime_min` | none | roughly how long you expect it to take, measured the same way. Nothing enforces it; see [job length estimates](#job-length-estimates) |
| `progress_command` | none | run in the workdir during phase `main`; its last line of stdout is how far along the job is |
| `progress_interval_s` | `60` | how often to run it; **minimum 5** |
| `low_util` | on | `{enabled: true, window_min: 25, floor_pct: 5, grace_min: 10}` — the idle-GPU watchdog |
| `requires` | `{}` | e.g. `cuda_min: "12.8"`. **Informs provisioning only**; the host never checks it |
| `cleanup` | `on_success` | when the runner deletes `workdir/`: `on_success`, `always`, `never` |
| `attempt` | `1` | set by `gpuc requeue`, never by you; a submitter's value is ignored |

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
  running lego-s4 (20260915-120000-a1b2c3) phase=main 96.2m util 98% gpu=0 eta 3h20m (37%)
  queued  sweep (20260915-130000-d4e5f6) prio=50 est 6h00m
  free    next card in ~3h20m (20260915-120000-a1b2c3)
```

Where some of the jobs holding a card offered no end time, the `free` line
appends a count of them, because the real answer can only ever be *sooner* than
it: one of those could finish in a minute. Jobs with `gpus: 0` are ignored throughout —
they hold no card, so they can neither free one nor make the answer sooner. If
*nothing* holding a card estimated an end time there is no line at all, since
the gpu lines above it already say every card is busy.

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
and the dispatcher restarted first, see [setup.md](setup.md#upgrading)).
`--runpod` and its flags are [below](#runpod).

**`gpuc status`** — per host: kind, reachability, how many cards are free,
dispatcher heartbeat, one line per owned card (`free` / `busy` / `UNAVAILABLE`,
and what the card is), the queue, running jobs with phase, minutes, last util,
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
  running lego-s4 (20260915-231241-f880d9) phase=main 27.2m util 100% gpu=0 eta 45m (37%)
  done    hello (20260915-074344-1d4db4) succeeded 15h ago
host spar [ssh]  gpus 0/2 free (driver 535.309.01)
  dispatcher 2s ago
  gpu     [2] busy NVIDIA A40 45 GB
  gpu     [3] busy NVIDIA A40 45 GB
  running paper-diff (20260915-222409-7a2b60) phase=main 75.7m util 100% gpu=2
  running paper-plain (20260915-224057-9f10c3) phase=main 58.9m util 100% gpu=3
  queued  sweep (20260915-233000-112233) prio=50 est 6h00m
```

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
finished job cannot be reordered (exit 1).

**`gpuc estimate <job-id> --minutes N`** — set (or `--clear`) a queued or
running job's `estimated_runtime_min`; see [job length
estimates](#job-length-estimates).

**`gpuc requeue <job-id>`** — re-reads the spec from the S3 mirror and submits
it again as attempt+1, with the workdir re-synced from your *current* directory.
It therefore **needs `s3_bucket`** (without it, submit the job file again) and
cannot rebuild a `--no-git` workdir. `--host H` sends it somewhere else;
`--runpod` provisions for it; with neither, it goes back to the host the local
index says it ran on.

`--host` is optional on `logs`, `cancel`, `reorder`, `estimate` and `requeue`:
the local job index is tried first, then every registered host is asked whether it knows the
id. An unknown job or host is exit 4.

**`gpuc skill`** — prints the agent guide ([`skills/gpuc/SKILL.md`](../skills/gpuc/SKILL.md))
to stdout, so an agent can read it with `!gpuc skill` without knowing where the
repo is. `--install [DIR]` writes it to `DIR/.claude/skills/gpuc/SKILL.md`
instead (`DIR` defaults to the current directory) and refuses to overwrite an
existing copy without `--force`.

**`gpuc clean`**, **`gpuc pods`**, **`gpuc reconcile`** and the host commands
have their own sections below and in [setup.md](setup.md). `gpuc version` prints
this build, its commit, and each bootstrapped host's package commit, marking the
ones to re-bootstrap; it reads the registry and never touches a host.

## RunPod

```sh
export RUNPOD_API_KEY=...
gpuc submit job.yaml --runpod --gpu A40 --max-price 0.60
gpuc pods                   # every pod with our prefix: cost, util, age, is it wanted?
gpuc reconcile --once       # terminate leaked or expired pods now
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
  stopped answering, since that never refreshes `last_seen_at` either;
- a pod past a TTL its host actually has;
- a pod that never bootstrapped by its 15-minute ceiling;
- a pod with our prefix that no `desired/` record wants, **once it is over 15
  minutes old** (the grace is there because another session may be mid-create).

A pod with a job running per the host's own state is never touched by the
dead-dispatcher rule, however old it is — but a TTL you set overrides that and
kills the job. The reaper otherwise fails closed in every direction: it never
touches a pod without the configured prefix, it terminates nothing at all if
`desired/` is unreadable or the provider reports no creation time, a terminate
that fails keeps its record (so the next pass retries it) and makes `gpuc
reconcile --once` exit non-zero.

**`gpuc pods`** lists every pod in the account: ours (name, id, status, GPU,
`$/h`, CUDA, age, util, `DESIRED`, heartbeat) with the hourly total, and other
people's by name only, never touched. `DESIRED=NO` means nothing local wants it
and the reaper will take it. `--no-heartbeat` skips the per-pod dispatcher ssh
check, which is what makes the command slow when a pod is wedged.

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
| `terminated` | the runner itself was signalled (and the job was not cancelled) |
| `sync` | the final upload failed; the run itself may have been fine. A succeeded job becomes `failed: sync`, and any other reason gains `+sync` |
| `no-outputs` | an `outputs:` path was never written, or holds only files that came with the checkout. Appends `+no-outputs` the same way |
| `bad-spec` | the queued spec could not be read |
| `needs N GPUs, host owns M` | the host's ownership shrank after the job was queued |
| `spawn-failed` | the dispatcher could not start a runner process |
| `runner-died` | the runner vanished without writing final state; the dispatcher kills anything it left behind before freeing its GPUs |

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
      "queued": [],
      "running": [
        { "job_id": "20260915-120000-abc123", "name": "lego-s4", "status": "running",
          "reason": null, "phase": "main", "elapsed_s": 4210.5, "util": 96.0,
          "progress_pct": 37.0, "eta": "2026-09-15T16:31:00+00:00", "eta_s": 12060.0,
          "estimated_runtime_min": 480.0, "progress_error": null,
          "gpus": ["GPU-8064..."], "iso": "pgid", "ended_at": null,
          "outputs_pending": false }
      ],
      "finished": [],
      "errors": []
    }
  ],
  "errors": []
}
```

The job objects in `queued`, `running` and `finished` all carry those sixteen
fields. `eta` is absolute and `eta_s` is the same instant as seconds from now
(negative once a job is overdue); both are null unless the job has a
[length estimate](#job-length-estimates), and null again once it is finished.
`progress_pct` is null unless the job measures its own, and survives the job so
you can see how far it got; `progress_error` is why the last poll produced
nothing. A card the host cannot see appears in `gpus` as
`{"owned_as": "3", "available": false}` instead. A job's `util` is its **last**
sample from the host's own nvidia-smi over that job's cards; a pod's `provider_util` is the
provider's per-GPU reading for the whole pod, and is null for any other host —
two different measurements that will differ. `--recent` and `--since` apply to
`--json`; `--suspects` and `--all` do not.

Rules for anything automated:

- **Key on `hosts[].running`.** It is the host's own answer; an empty list means
  the host said nothing is running.
- **Treat exit 3 as "unknown", never as "nothing is running".** So is a
  non-empty top-level `errors`, and so is `"reachable": false` for the host you
  care about — we could not ask it.
- Unknown keys will be added over time; ignore the ones you do not know.

### `--json` everywhere else

`submit`, `requeue`, `logs`, `cancel`, `reorder`, `estimate`, `pods`, `version`,
`clean`, `host list`, `host probe` and `reconcile --once` take `--json` too, under the
same rules: **stdout is exactly one JSON object**, it carries `schema_version`,
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
| `submit`, `requeue` | `{job_id, host, attempt, requeued_from, notes[]}`. `requeued_from` is the id this run came from, null on `submit`; `notes` are the text output's `note:` lines and do not mean the job was not queued |
| `logs` | `{job_id, host, source, location, lines[], notes[]}`. `source` is `"host"` or `"s3"` and `location` is the remote path or the `s3://` uri it was read from; `lines` is the log with no trailing newlines. **Not with `-f`** — a stream has no end, so `--json -f` is exit 2 |
| `cancel` | `{job_id, host, status}` — the host's own word, `cancelled` for a queued job or `cancelling` for a running one |
| `reorder` | `{job_id, host, priority}` |
| `estimate` | `{job_id, host, estimated_runtime_min, status, warnings[]}`. `estimated_runtime_min` is what the spec holds now (null after `--clear`) and `status` is the job's, since only a queued or running one can be set; `warnings` carries a `max_runtime_min` contradiction and a mirrored spec that could not be updated |
| `pods` | `{pods[], hourly_usd, others[], notes[]}`. Each pod is `{id, name, status, gpu_name, gpu_count, cost_usd_hr, cuda_version, age_s, created_at, gpu_utils[], desired, heartbeat_age_s}`; `others` are pods without our prefix, `{id, name, status}` only, because we never touch them |
| `version` | `{version, commit, source, dirty, python, executable, hosts[], errors[]}`, each host `{name, pkg_commit, current}`. Exit 3 if the registry is unreadable |
| `host list` | `{hosts[], errors[]}` — each registry entry as stored, plus `remote_home`, `ephemeral` and `warnings[]`. The host's `env` is reported by **name only** (`{"HF_TOKEN": "<set>"}`), because `--env` is free-form and this document travels. A skipped entry is an `errors` string, not a host. Exit 3 if the registry is unreadable |
| `host probe` | `{host, sections{}, driver_version, has_nvidia_smi, gpus[], assigned_gpus[], assigned_missing[], home_fs_type, home_is_overlay, persistent_root, uv_cache{}, notes[]}`. `gpus` is **every** card the host has whatever `--all-gpus` said, each one `{uuid, name, vram_mib, index, assigned}`; `assigned_gpus` is this host's `--gpus` as registered and `assigned_missing` the entries in it no card answered to (always empty on a host with no nvidia-smi, which has nothing to answer with). `sections` is the probe script's raw output section by section, so anything this build does not interpret is still there |
| `clean` | `{host, dry_run, purge, freed_bytes, removed[], skipped[], purged[], purge_skipped[], incoming_removed[], verified[], notes[], errors[]}`. The job objects are the host's own: `{job_id, status, bytes, age_days}`, plus `why` on the skipped ones and `forced` on a purged job that had no confirmed backup |
| `reconcile --once` | `{terminated[], forgotten[], kept[], errors[]}`, host names in the order they were judged. `--json` needs `--once` and nothing else: neither the loop nor `--install` has a document to print |

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

**Automatic retention.** `--retention-days N` on a host makes its dispatcher run
the same purge itself, never forced, once at startup and then at most once an
hour. On an ephemeral host that is its whole life; on a **non-ephemeral host the
dispatcher only lives while there is work**, so the sweep happens on your next
submit rather than on a timer. Each pass also does the ordinary workdir clean
over the same horizon, which needs no mirror — so on a host with no
`--s3-prefix`, `--retention-days` reclaims old venvs and nothing else.

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
than 1 GiB, with the `gpuc clean` line to run.

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
| `gpuc host clean <host>` printed nothing useful | without `--uv-cache` it does nothing at all; it is not `gpuc clean` | `gpuc host clean <host> --uv-cache` prunes uv's cache; `gpuc clean --host <host> ...` is the one that frees job dirs |
| one job dir is stuck and the rest of the host is fine | that job's mirror genuinely failed, so an age-based purge either misses it or sweeps up everything else | `gpuc clean --host <host> --purge --only <job-id>` (add `--force` to accept losing its only copy); it leaves every other job alone |
| `clean --purge` skips everything as "not backed up" | the host has no `s3_prefix`, so nothing is mirrored and deleting a job dir would lose its log | `gpuc host set <host> --s3-prefix s3://bucket/gpuc/<host>` + `gpuc host bootstrap`, or accept the loss with `--force` |
| `status` says a job's `outputs not uploaded` | the final upload of its `outputs:` failed, so the results exist only on that host | copy them off, or `gpuc requeue <id>`; a purge will not remove it until they are confirmed |
| a job is `OUTPUTS LOST` | an ephemeral host drained, retried three times and gave up before terminating | the results are gone; fix the credential or bucket, then `gpuc requeue <id>` |
| `--retention-days` never deletes anything | the dispatcher only lives while a non-ephemeral host has work, and it never purges an unmirrored job | check `gpuc status` for `not backed up`, and remember the sweep runs on the next submit |
| `requeue` refuses, or rebuilds the wrong code | it reads the spec from S3 (so `s3_bucket` must be set) and re-syncs the workdir from your current directory | run it from the right checkout; a `--no-git` workdir cannot be rebuilt from a commit |
| `--idle-min` did nothing on a shared box | it only applies to ephemeral (RunPod) hosts; `local` and `ssh` hosts never terminate themselves | nothing to do; use `--retention-days` for disk, not `--idle-min` |
| `uv sync` re-downloads torch on every job | uv's cache is on a different filesystem from gpuc home, so it copies instead of linking | `gpuc host bootstrap <host>` (it sets `UV_CACHE_DIR` for you), or pin one with `--cache-dir` |
| `gpuc` exits 3 and names `hosts.json` | the registry could not be parsed at all; a `.bak` was kept beside it | fix or delete the file, then re-add hosts; nothing was written over |
| a warning names one skipped host entry | that entry did not validate; every other host still works and is written back untouched | fix it by hand, or `gpuc host add <name> ...` to replace it |
| `status` says `host X runs an older gpuc` | this machine was upgraded and the host's copy of the package was not | `gpuc host bootstrap X`, or `gpuc host bootstrap --all` for every host at once — safe while jobs run; the new dispatcher adopts them |
| `status` says `POD GONE` | the pod is terminated or missing but the registry still lists it | `gpuc reconcile --once` |
| `reconcile` reports `DEAD DISPATCHER` and terminates a pod | it stopped beating (or answering ssh) for `dead_dispatcher_minutes` with nothing running | expected: that pod could no longer stop itself. Raise `dead_dispatcher_minutes` if your hosts go quiet legitimately |
| everything on a host is suddenly gone | the container restarted and `$HOME` was on the overlay | the runbook in [setup.md](setup.md#hosts-whose-home-is-wiped-on-restart) |
