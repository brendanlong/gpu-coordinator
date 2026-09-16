# gpu-coordinator architecture (prototype spec)

Companion to `requirements-review.md`, which holds the reasoning. This file
is the contract implementers build against. Where the two disagree, this
file wins for the prototype.

It is not the CLI reference: flags and defaults are `gpuc --help` and each
subcommand's `--help`, what they mean together is [usage.md](usage.md), and
installing and registering hosts is [setup.md](setup.md).

## Goals

One submit path for three kinds of host: this machine (the cards you give it),
a box reached over SSH with no sudo there (a subset of its GPUs), and ephemeral
RunPod pods. Not finicky, not buggy, and never leaks a paid pod in the normal
path.

Non-goals for the prototype: Vast, multi-node, spot, a web UI, S3 as the
authoritative queue (host is authoritative, S3 is the mirror; `gpuc requeue`
resubmits from the S3 spec if a host dies).

## Package layout

One uv project, Python >= 3.11, package `gpuc`, CLI entry point `gpuc`.

```
gpuc/
  _version.py    # __version__ and user_agent(); STDLIB ONLY, imported by both halves
  host/        # runs ON hosts. STDLIB ONLY. No imports outside stdlib.
    __main__.py   # `python -m gpuc.host <cmd>`; the same CLI as `gpuc host-*`
    paths.py      # ~/.gpuc layout
    jobs.py       # job id, spec/state read/write, atomic writes
    queue.py      # enqueue, list, reorder, cancel and kill markers
    dispatcher.py # lock+heartbeat, pick next runnable, launch runner, idle/TTL terminate
    runner.py     # one job: env, CUDA_VISIBLE_DEVICES, preflights, watchdog, sync, exit code
    scope.py      # systemd --user scope probe/wrap/stop; the cgroup kill path
    preflight.py  # sync preflight: prove `aws`/`hf` can write before the job runs
    baseline.py   # what was already under `outputs:` before the job started
    cleanup.py    # `cleanup:` policy, workdir sizing, the `clean` sweep
    gpus.py       # nvidia-smi parsing, index<->UUID resolution, utilization sampling
    progress.py   # the optional `progress_command`: run it, read a percentage off it
    sync.py       # periodic upload loop (shells out to `aws` or `hf`; see Sync)
    health.py     # host preflight: driver, owned GPUs, disk, network download timing
    terminate.py  # self-terminate via provider API (urllib), key from ~/.gpuc/secrets
  control/     # runs on the local machine. May use third-party deps.
    cli.py        # `gpuc` top-level commands
    config.py     # ~/.local/share/gpu-coordinator/ layout, hosts registry
    transport.py  # LocalTransport / SshTransport: run, rsync, put_file(0600), tail
    bootstrap.py  # install uv + this package on a host, write config, run host preflight
    clean.py      # `gpuc clean` / `gpuc host clean --uv-cache` over the transport
    providers/
      base.py     # Provider interface: offers(constraints), create, get, logs, terminate, list
      runpod.py   # v2 REST implementation
    gpuinfo.py    # per-GPU name/VRAM for the registry and every listing
    reconcile.py  # desired state vs provider; ceiling; reaper; caps
    submit.py     # resolve target -> host -> enqueue; provision if needed
    s3index.py    # mirror of specs/state for `requeue` and `status --all`
tests/
```

Everything the dispatcher spawns -- runners, and through them every job
command -- gets `$HOME/.local/bin` and `$HOME/.cargo/bin` prepended to `PATH`
when they exist. A pod's sshd hands out a PATH with neither, and `uv` lives
there, so without this the runner's own `uv run --no-sync` preflight fails on
every job.

`gpuc.host` must be importable and runnable with a bare interpreter: the
bootstrap rsyncs the package to the host and runs it with the uv-managed
Python, with no `uv sync` needed for the queue to work. Upload helpers shell
out to binaries the bootstrap installs into `$HOME` (`aws` CLI v2 bundle,
`hf` from `uv tool install huggingface_hub`), and a missing binary fails the
*job's* sync step with a clear message, never the queue.

## User agent

One string, from `gpuc/_version.py`: `gpuc/<version> (+<repo url>; <email>)`
(the README quotes it, and a test pins it). `gpuc.host` is stdlib-only, so it
imports `gpuc._version`, which must stay stdlib-only and which bootstrap's rsync
must ship. Every outbound request carries it: the RunPod provider and the host's
self-terminate (urllib), the health check's download, bootstrap's `curl -A`,
every control-side boto3 client (`Config(user_agent_extra=...)`), and
`HF_HUB_USER_AGENT_ORIGIN` in every job's environment, which a job may override.
The `aws` CLI on a host cannot have its User-Agent overridden, so its S3
requests are anonymous to us.

## On-host state: `~/.gpuc/`

```
config.json          # {"schema_version": 1, "host": "<name>", "gpus": ["GPU-uuid" | "<index>", ...],
                     #                              # what this host owns, as it was registered;
                     #                              # see GPU ownership
                     #  "provider": null | {"kind":"runpod","pod_id":..},
                     #  "idle_minutes": 15, "ttl_hours": null | N, "s3_prefix": "s3://bucket/gpuc/<host>",
                     #  "retention_days": null | N,   # auto-purge horizon; null never purges
                     #  "pkg_commit": null | "<sha>", # the commit bootstrap shipped to this host
                     #  "env": {"HF_HOME": ...}}      # host-wide, hand-set; see Persistent root
secrets/<name>       # 0600 files delivered over SSH after boot. Never in argv, never in pod env.
incoming/<jobid>.json # a spec staged 0644 by `submit`, fed to `enqueue -` and deleted
queue/<prio>-<jobid> # empty marker files; lexical order is dispatch order. prio is 2 digits, default 50.
jobs/<jobid>/
  spec.json          # the submitted JobSpec (immutable)
  state.json         # {"status": queued|running|succeeded|failed|cancelled, "attempt": n,
                     #  "reason": str|null, "exit_code": int|null, "gpus": [...], "started_at", "ended_at",
                     #  "phase": setup|preflight|main|sync, "pid": int|null, "pgid": int|null,
                     #  "isolation": "cgroup"|"pgid", "cgroup_unit": str|null,
                     #  "runner_pid": int|null, "runner_boot_id": str|null,
                     #  "runner_starttime": str|null, "sync_error": str|null,
                     #  "util_recent": [float|null, ...], "util_sampled_at": str|null,
                     #  "progress_pct": float|null, "progress_at": str|null,
                     #  "progress_error": str|null, "eta": str|null,
                     #  "workdir_removed": bool,
                     #  "meta_synced_at": str|null, "meta_synced_to": str|null,
                     #  "outputs_synced_at": str|null, "outputs_lost": bool}
                     # meta_synced_* are written only after a *successful* final sync_job_meta
                     # (and state.json goes up once more so the mirror matches); they are the
                     # precondition `purge` checks. outputs_synced_at is set only when the final
                     # upload of `outputs:` succeeded; outputs_lost is set by an ephemeral host's
                     # drain after it retried and gave up.
                     # util_recent is the last 40 main-phase samples; null means nvidia-smi
                     # failed and the sample must not be read as 0%.
                     # eta is null on a queued job and on a finished one; while the job runs
                     # it is the submitter's estimate until progress_pct measures a better
                     # one. progress_pct survives the job so `status` can say how far it got.
                     # pgid is the *job's* group, published by the runner when it spawns a phase.
                     # It is absent during the launch window; cancel is the marker alone until then.
  outputs_baseline.json # per `outputs:` path, the {relpath: [size, mtime_ns]} the
                     # checkout arrived with; those files are never uploaded as this
                     # job's results and never satisfy `outputs:`
  kill               # a kill request with its reason (`ttl`, `low-util-pause`), written by the dispatcher
  workdir/           # rsynced code (git-tracked + untracked, .gitignore obeyed); removed per `cleanup:`
  log.txt            # combined stdout/stderr of setup + command, line-buffered
  outputs/           # default output root; JobSpec.outputs paths are relative to workdir
dispatcher.lock      # fd flock held by the running dispatcher
dispatcher.heartbeat # mtime touched every 5 s by the dispatcher
dispatcher.log
draining             # present while the host is shutting itself down
paused               # present after two consecutive low-util failures; an ephemeral host
                     # then drains once nothing is running, any other host just stops dispatching
```

All state writes are atomic (write temp in same dir, `os.replace`).

## JobSpec (JSON; `gpuc submit` accepts YAML or JSON and normalizes)

```
{
  "name": "lego-s4",                    # human label, not an identifier
  "command": "uv run python -m experiments.lego.train --k-max 6",
  "setup": "uv sync --frozen",          # optional; runs before command, phase=setup
  "gpus": 1,                            # 0..N owned GPUs
  "env": {"REQUIRE_CUDA": "1"},
  "secrets": ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "HF_TOKEN", "WANDB_API_KEY"],
                                        # read from the submitter's env, delivered as ~/.gpuc/secrets/<jobid>.env (0600),
                                        # sourced into the job's environment by the runner
  "outputs": [{"path": "results", "s3": "s3://bucket/exp/{job_id}/results"},
              {"path": "checkpoints", "hf": "org/repo", "hf_path": "{job_id}",
               "hf_create": false}],   # create the repo if the sync preflight finds it missing
  "sync_interval_s": 180,
  "priority": 50,
  "max_runtime_min": null,
  "estimated_runtime_min": null,        # the submitter's own guess, measured from the runner's
                                        # start exactly as max_runtime_min is. Informational only
  "progress_command": null,             # run in workdir/ every progress_interval_s of phase main;
                                        # its last line of stdout is a percentage. See Estimates
  "progress_interval_s": 60,
  "low_util": {"enabled": true, "window_min": 25, "floor_pct": 5, "grace_min": 10},
  "requires": {"cuda_min": "12.8"},     # informs provisioning only
  "cleanup": "on_success",              # on_success | always | never; see Workdir cleanup
  "attempt": 1                          # set by `gpuc requeue`, not by the submitter
}
```

`{job_id}` is expanded in output destinations. Output namespaces are unique
by construction; there is no overwrite guard anywhere.

Job id: `YYYYMMDD-HHMMSS-<6 hex>`, assigned by `gpuc submit`. The
timestamp is second-granular, so two jobs submitted inside the same second
at the same priority tie-break on the random suffix; dispatch order is the
queue's lexical order, not submission order below one second.

## Dispatcher (`python -m gpuc.host dispatch`)

- Started by every `enqueue` (and by bootstrap) with `setsid nohup ... &`.
  Takes `flock(LOCK_EX|LOCK_NB)` on `dispatcher.lock`. If held **and** the
  heartbeat is younger than 30 s, exit 0 silently. If held and the heartbeat
  is stale, kill the holder's process group (pgid recorded in the lock file
  body), then take over.
- Loop every 2 s: resolve `config.gpus` to UUIDs (see GPU ownership), then read
  `queue/` in lexical order; for each entry whose `spec.gpus` fits the free
  owned UUIDs, assign UUIDs, remove the queue
  marker, set state running, spawn the runner in its own process group
  (`start_new_session=True`), record pid/pgid. Multiple jobs may run at
  once if GPUs allow; a `gpus: 0` job never waits.
- Cancel: `queue.cancel(jobid)` writes `jobs/<id>/cancel`. The **runner**
  owns the kill: it checks the marker before each phase and on every poll, and
  stops the phase's scope (`systemctl --user stop <unit>`) or, with no systemd,
  takes down the job's own process group (SIGTERM, 15 s, SIGKILL). The
  dispatcher escalates only once `state.json` publishes a `cgroup_unit`, or a
  pgid that is not the runner's own -- during the launch window the runner *is*
  the only member of its group, and signalling it there would kill the one
  process that can still finish the job cleanly. Queued jobs are cancelled by
  removing the marker and setting state.
- Stop with a reason: `queue.request_kill(jobid, reason)` writes
  `jobs/<id>/kill`. The runner kills the job the same way and ends it
  `failed: <reason>` after a final sync. The TTL uses this; cancel stays its own
  marker, because a TTL stop is not a cancellation anyone asked for.
- Isolation: at startup the dispatcher probes `systemd-run --user --scope
  --collect --quiet -- true` once and hands the answer to every runner it spawns
  as `GPUC_ISOLATION`. See Process isolation.
- Reorder: `queue.reorder(jobid, prio)` renames the marker.
- Idle terminate (only when `config.provider` is set): if no running jobs
  and the queue has been empty for `idle_minutes`: write `draining`, retry any
  unconfirmed outputs, run one final sync of every job's state and log to
  `s3_prefix`, then call `terminate.self_terminate()`. Only a failed
  *terminate* stops the shutdown: remove `draining`, log loudly, keep
  dispatching, retry every 10 minutes. A failed final sync is logged and the
  host terminates anyway -- the state is already on disk here, and a bucket we
  cannot reach is not a reason to keep a paid pod billing forever.
- TTL (`ttl_hours`, **null by default**, an explicit opt-in hard cap): checked
  before the idle logic, so a busy host cannot dodge it. Past the cap with a job
  running, the dispatcher writes a `kill` marker with reason `ttl` for each
  running job and waits: the runner kills it, syncs its outputs, and records
  `failed: ttl`. A runner that has not acted on the marker after `kill_grace_s`
  is escalated exactly like a cancel (scope stop, job group SIGKILL, then the
  runner itself), so a wedged runner cannot keep a TTL'd pod alive. With nothing
  running it drains and terminates as above. With `ttl_hours` null nothing
  terminates on age at all -- see Auto-down and the reaper.
- Two consecutive `failed: low-util` jobs: stop dispatching, log, and (if
  ephemeral) drain and terminate -- but never out from under another job. With
  anything still running it writes a `kill` marker with reason `low-util-pause`
  for each, exactly as the TTL does, and drains on a later pass.
- Exit when the queue is empty, nothing is running, and the host is not
  ephemeral. Ephemeral hosts keep the dispatcher alive until terminate.

## Runner (one process per job)

1. Resolve the assignment against `nvidia-smi --query-gpu=index,uuid` --
   indices and UUIDs both, since either form may be recorded -- and fail the
   job (`gpu-assert`) if an entry names no card that is here. Export
   `CUDA_VISIBLE_DEVICES=<resolved UUIDs comma-joined>` (empty string when
   `gpus: 0`), the spec `env`, and the secrets file.
1b. Snapshot every declared `outputs:` path into `outputs_baseline.json`
   (relative path, size, mtime) -- a checkout routinely ships committed files
   where the outputs go. Before `setup`, because a setup step writing there
   *is* this job's doing. Every upload, periodic and final, excludes files that
   still match the baseline, and a path holding *only* baseline files counts as
   no output at all (`failed: no-outputs`). Above 500 pre-existing files under
   one path the exclusion is dropped with a loud warning in the log, because the
   exclude list would no longer fit on a command line; `gpuc submit` warns about
   pre-existing files before the job is queued.
2. `phase=setup`: run `spec.setup` in `workdir` with `bash -eo pipefail`.
3. `phase=preflight`: GPU preflight **inside the job's environment**. A named
   phase, not a step of `setup`, so `gpuc status` can tell "still installing
   torch" from "proving the card works". If `gpus > 0`, run
   `uv run --no-sync python -c "<real op>"` (the `shared/gpu.py` probe:
   `is_available()` then a tensor add + `.item()`), and assert
   `device_count()` equals `gpus`. Failure -> `failed: gpu-preflight`.
3b. **Sync preflight**, still before `main`, in the job's environment. If the
   spec declares S3 outputs or the host has an `s3_prefix`, the `aws` binary
   must resolve and a `.preflight` object must upload to every destination
   prefix *and* to the host's `jobs/<id>/` mirror prefix. If the spec declares
   HF outputs, `hf` must resolve, `hf auth whoami` must succeed with the job's
   token, and a `.preflight` file must upload to each repo -- creating the repo
   first when that output sets `hf_create: true`, and failing with a message
   naming the flag when it does not exist and `hf_create` is unset. Failure ->
   `failed: sync-preflight`, with the command and its error in `log.txt`, and no
   final output sync (the preflight already proved it cannot work). A job with
   no outputs on a host with no mirror checks nothing.
4. Start the sync loop (background thread) for `outputs`, every
   `sync_interval_s`, skipping files modified in the last 10 s. Also upload
   `log.txt` and `state.json` to `s3_prefix/jobs/<id>/` on the same cadence.
   The upload commands run with the **job's** environment, secrets file
   included, so `secrets: [AWS_ACCESS_KEY_ID, ...]` is sufficient and no
   host-level credential file is needed. The secrets file is unlinked after
   the final sync, never before it.
5. `phase=main`: run `spec.command`, stdout+stderr appended to `log.txt`.
   Each phase runs inside its own transient scope where one is available (see
   Process isolation), and in its own process group where it is not.
   Start the low-util watchdog after `grace_min`: sample assigned GPUs'
   utilization every 30 s; if the rolling mean over `window_min` is below
   `floor_pct`, SIGTERM the process group, then SIGKILL after 15 s, status
   `failed: low-util`. `max_runtime_min` is enforced the same way with
   reason `timeout`. In the same loop, run `spec.progress_command` every
   `progress_interval_s` and record what it says; see Job length estimates.
6. Capture the exit code **before** any cleanup. Stop the sync loop and run
   one final sync; a failed final sync makes a succeeded job `failed: sync`,
   and an output path that was never written makes it `failed: no-outputs`.
   Write final state. Exit code of the runner = job exit code.
7. Apply `spec.cleanup` to `workdir/` -- after the final sync and the final
   state write, never before: `outputs:` paths resolve *inside* the workdir, so
   any earlier removal would delete the run's results on the way past. Record
   `workdir_removed` in `state.json`, then upload state and log. A removal that
   fails is logged and nothing more: the job's outcome is already decided, and
   leftover disk is not worth turning a green run red. That last upload records
   `meta_synced_at`/`meta_synced_to` and puts `state.json` up once more, so the
   mirror includes the record of itself; `outputs_synced_at` is written in the
   final state write when the final output upload succeeded. See Retention.

## Job length estimates

Nothing infers how long a job will take. Two optional, purely informational
inputs answer "queue behind this, or pay for another host?", and neither may
ever change a job's outcome:

- `estimated_runtime_min` -- the submitter's own guess, measured from the
  runner's start exactly as `max_runtime_min` is. The runner publishes it as
  `eta` at the top of *every* phase, not just `main`: a job twenty minutes into
  a `uv sync` is the one somebody most wants an end time for, and it looks
  identical to a wedged one.
- `progress_command` -- run in `workdir/` with the job's own environment, only
  during `main`, every `progress_interval_s`. Its **last non-empty line of
  stdout** says how far along it is, in one of exactly two forms: a fraction
  of one, which carries a decimal point (`0.42`), or a percentage, which
  carries a `%` (`42%`). A bare integer is refused rather than guessed at --
  `42` could be either and `1` could be 1% or a finished job, reading either
  the wrong way is a hundredfold error in a time somebody is planning around,
  and no rule applied after the fact can tell them apart, so the unit has to be
  in the input. `step / total` prints `0.42`, `0.0` and `1.0` in any language,
  so the decimal point costs the intended case nothing and catches `echo
  $step`. Above 0% the runner replaces `eta` with `now + elapsed_main *
  (100 - pct) / pct` -- elapsed *main*, so a slow setup is never charged to the
  first epoch. At 0% the submitter's estimate stands, and `status` tags the eta
  `(est)` accordingly: there is no rate to measure yet.

The poll is synchronous, in the same loop that watches for a cancel, a TTL and
`max_runtime_min`, with a 10 s timeout and a `killpg` of the whole session
behind it, so a wedged progress command delays a kill by at most 10 s of the
15 s the runner gets before the dispatcher escalates -- most of that budget,
which is why the timeout is fixed rather than a spec field. Its output goes to
a temp file, not a pipe: a grandchild that `setsid`s out of the session escapes
the `killpg` *and* would hold a pipe open, blocking the reap for ever and
taking the cancel check down with it. Only the last `MAX_OUTPUT_BYTES` is read
back, so a command pointed at a whole log cannot put it in the runner's memory.

Doing the poll on a thread instead would add a *third* concurrent writer to
`state.json`'s read-modify-write -- the sync loop's `sync_error` is already a
second one, alongside the runner's main thread -- and losing a utilization
sample that way would mislead the low-util watchdog. Bounded latency is the
better trade.

A failed, timed-out or unparseable poll writes `progress_error` and returns.
It is logged the *first* time each distinct message appears, because this runs
every interval for the rest of the job. `eta` is cleared when the job ends;
`progress_pct` is not, because how far it had got when it died is the useful
part. `gpuc status` renders the remaining time relative (`eta 3h20m`), tagged
`(42%)` when it was measured and `(est)` when it was a guess, and adds one
`free` line per fully-busy host saying when its next card is expected -- with a
count of the running jobs that estimated nothing, since the true answer can only
be sooner.

## Process isolation (cgroup scope, else process group)

A grandchild that double-forks (`setsid`, `nohup`, a daemonising server) leaves
the job's process group and survives `kill -- -PGID`, holding a GPU the
dispatcher is about to hand to the next job. A process cannot leave its
**cgroup** without privilege, so where a `systemd --user` session with cgroup
delegation exists each phase runs as:

```
systemd-run --user --scope --collect --quiet -p TimeoutStopSec=15 \
  --unit=gpuc-<jobid>-<phase>.scope -- bash -c 'base64 -d <<<"$1" | bash' _ <b64>
```

The script is base64-encoded on purpose: the words after `--` become a systemd
`ExecStart`, where systemd does its own substitution (`$$` -> `$`, `$VAR` ->
environment) and would silently corrupt any inline shell. The base64 alphabet
has no `$`, and `$1` is digit-led, which systemd leaves alone.

The kill path becomes `systemctl --user stop <unit>` (SIGTERM, then SIGKILL
after `TimeoutStopSec`), with the existing process-group kill kept as a
fallback, and the dispatcher's backstop uses the unit when `state.json` records
one. `isolation` (`cgroup` | `pgid`) and `cgroup_unit` are in `state.json` and
shown by `gpuc status`. The probe runs once per dispatcher and is passed to
runners in `GPUC_ISOLATION`. On a host with no user systemd -- every RunPod pod,
the shared box -- `pgid` is the mode and a daemonised grandchild still escapes;
that is a documented hole, not a fixed one.

## Workdir cleanup

`workdir/` is the only part of a job dir that is recreatable (`gpuc requeue`
re-syncs it from git) and usually the largest -- a torch venv is ~6.5 GB. It is
the only thing gpuc deletes; `spec.json`, `state.json` and `log.txt` always
stay, so `logs` and `status` keep working on a cleaned job.

`on_success` (the default) removes it only after a succeeded job, `always`
after any outcome, `never` not at all (the table is in usage.md). No policy ever
removes the workdir of a job that is not finished, and the removal happens after
the final sync and the final state write, never before.

`python -m gpuc.host clean (--all-finished | --older-than DAYS) [--dry-run]`
is the after-the-fact sweep, driven by `gpuc clean --host H`. It prints JSON:
per-job `bytes` (du-style allocated blocks, deduplicated by inode within the
tree), what was skipped and why, and the staged specs it removed. It fails
closed everywhere: a running or queued job, a job whose `state.json` is missing
or unreadable, and (under `--older-than`) a job with no parseable `ended_at`
are all skipped. It also removes `incoming/<id>.json` staged specs whose job
has finished, or which name no job at all and are over an hour old -- the
window in which that file is load-bearing is one SSH round trip.

## Retention and purge

`clean` keeps `spec.json`, `state.json` and `log.txt` forever. `python -m
gpuc.host purge [--older-than DAYS] [--dry-run] [--force] [--only IDS]` (driven
by `gpuc clean --host H --purge`) removes the whole `jobs/<id>/`, plus any stray
queue marker, secrets file and staged spec, for finished jobs older than DAYS
(default 7, from `ended_at`) that carry two records in their own `state.json`:

- `meta_synced_at` -- the final `sync_job_meta` returned 0, so log and state are
  in S3. The local state is the authority: the host cannot consult the mirror
  without credentials it may not have. Without it, the skip reason is `not
  backed up: no s3_prefix on this host` or `not backed up: final upload failed`.
- confirmed outputs -- `outputs_synced_at` set, a spec with no `outputs:`, a
  workdir that is already gone, or a job that never wrote what it declared.
  `outputs:` paths resolve inside the workdir and a failed job keeps its
  workdir, so a purge could otherwise bin the only copy of a checkpoint.
  Without it: `outputs not confirmed uploaded`. "Never wrote" is the question
  `sync` asks before it refuses to upload -- nothing under the path, or nothing
  there that was not already in the checkout (`outputs_baseline`) -- so a job
  that died before producing anything neither claims a lost result nor sits in
  the job dir forever, unpurgeable. Asked here it is answered more strictly
  than `sync` answers it, because a wrong "nothing here" deletes rather than
  skips an upload: an unreadable path, a symlink (`rglob` does not descend one,
  `aws s3 sync` follows it), a walk that errors, and an `outputs.path` that
  cannot be resolved at all each count as content.

`--force` overrides those two records and nothing else, and marks each removal
`forced` so the report says so loudly. Running, queued and unreadable-state jobs
are never purged, forced or not; `--purge` also runs the ordinary workdir sweep,
which is what reclaims the jobs the purge refused; `--only` narrows what may be
*purged* and nothing else. The purge removes the whole `jobs/<id>/` plus that
job's queue marker, staged spec and `secrets/<id>.env`.

`HostConfig.retention_days` (registry `HostEntry.retention_days`, `gpuc host
add|set --retention-days N`, null by default) makes the dispatcher purge, never
forced, once at startup and then at most once an hour. A non-ephemeral host's
dispatcher only lives while there is work, so in practice that sweep happens on
the next submit.

An ephemeral host's drain retries unconfirmed outputs before terminating --
three attempts a minute apart, five minutes in total, with the job's secrets
file (the runner leaves it in place exactly for this) or the host env -- then
records `outputs_lost` with the last error in `sync_error`, mirrors state, and
terminates anyway: the pod is billing and the TTL that sent us here does not
pause.

The host cannot check the mirror itself -- it may hold no credentials of ours --
so `meta_synced_at` is the authority. The control side's `--verify` is the
exception: it HEADs the mirrored `log.txt` under the prefix each job recorded in
`meta_synced_to` (falling back to the registered `s3_prefix`) before deleting
anything. See usage.md.

The host's `status` reports `workdir_bytes` per *finished* job (a live job's
workdir is still being written to, and walking it on every status call would be
pure cost). `gpuc status` prints one line per host once those exceed 1 GiB.

## The shared uv cache

uv caches wheels under `~/.cache/uv` and materialises a venv by reflinking or
hardlinking out of it, so the second job needing the same torch build costs
seconds and almost no disk. Both mechanisms work only *within one filesystem*,
and `UV_LINK_MODE=copy` disables them outright. Two rules follow:

- Nothing in the job path sets `UV_LINK_MODE`, and neither the dispatcher nor
  the runner sets `UV_CACHE_DIR` unless `HostConfig.env` does.
- `gpuc host bootstrap` compares the filesystem of gpuc home with that of `uv
  cache dir`. If they differ it sets `HostEntry.cache_dir` (surfaced to jobs as
  `UV_CACHE_DIR` via `HostConfig.env`) to `<parent of gpuc home>/.cache/uv` --
  beside gpuc home, not inside it, so `clean` and an `rm -rf` of gpuc home
  cannot take the cache with them. An explicit `--cache-dir` or an `--env
  UV_CACHE_DIR=...` is never overridden, and an unreadable comparison changes
  nothing.

The case that matters is a pod with `--persistent-root /workspace/$USER`: gpuc
home on the network volume, `~/.cache` on the container's overlay. Without the
rule uv copies every wheel into every venv, at the venv's full size, onto the
slowest disk the host has. Measured on an ssh host where gpuc home and the
cache *are* one filesystem, a 6.5 GB torch venv's large `.so` files have
`nlink=1` but `filefrag` reports identical physical extents flagged `shared`:
uv used reflinks, so `du` reads 6.5 GB while the venv costs essentially nothing
beyond the cache. `du` cannot see this, which is why the check compares
filesystems rather than sizes.

`gpuc host probe` and the health check both report the cache's size and whether
it shares a filesystem with gpuc home; the health check is warn-only, because a
split cache costs disk and time, not correctness. `gpuc host clean <host>
--uv-cache` runs `uv cache prune` (not `clean`, which would throw away exactly
the wheels the next job wants to link).

## Control side: `gpuc` CLI

The command surface is `gpuc --help` plus each subcommand's `--help`, and
[usage.md](usage.md) explains it; neither is repeated here. What the CLI must
hold to, whatever the flags:

- `gpuc` runs with no config file at all: every setting has a default, there is
  simply no S3 mirror, and one line on stderr points at `gpuc config init`.
- Anything that talks to RunPod (`--runpod`, `pods`, `reconcile`) checks
  `RUNPOD_API_KEY` first and exits 1 with a single line if it is unset, before
  mirroring a spec or picking a host. `reconcile --install` does not, since it
  only writes unit files.
- A host name is looked up locally; `logs`, `cancel`, `reorder` and `requeue`
  fall back to the job index and then to asking each host, and an id nothing
  knows is exit 4, never a guess.
- Nothing runs in the background on this side except the optional
  `gpuc reconcile` timer.

Local state: `~/.local/share/gpu-coordinator/` with `hosts.json`,
`desired/<host>.json` for ephemeral hosts, `jobs/` (the local job index),
`known_hosts` plus `known_hosts.d/<pod>`, and `state.lock`, which serialises
every registry read-modify-write across concurrent sessions.
`Settings` (`~/.config/gpu-coordinator/config.toml`, all optional):
`s3_bucket = None`, `runpod_pod_prefix = "gpuc-"`, `max_pods = 3`,
`max_total_usd_per_hour = 3.0`, `ssh_key = None` (ssh picks its own key),
`image`, `disk_gb = 50`, `dead_dispatcher_minutes = 30.0`. What each one is for
is setup.md.

## Shared state is read tolerantly, always

Two sessions of one user share `~/.local/share/gpu-coordinator/hosts.json`, and
a host's `config.json` outlives the build that wrote it. So every reader of a
file the two halves share obeys the same two rules, on both sides:

- an unknown key is ignored (a newer writer may add fields);
- an explicit `null` for a field that is **not** declared optional is dropped,
  so the field's default applies. A `null` for a field that *is* optional is a
  real value and round-trips unchanged: `ttl_hours: null` is "never expires",
  `retention_days: null` is "never auto-purge", `s3_prefix: null` is "no
  mirror".

Control side that means `extra="ignore"`, a default on every field, and a
`model_validator(mode="before")` that consults the annotation (`HostEntry`,
`Registry`, `DesiredHost`, `Settings`, `IndexEntry`, `Offer`). Host side, with
no pydantic, the same rules are spelled out in `jobs.from_dict` for
`HostConfig`, `JobSpec`, `JobState` and the dispatcher's lock body: never
`float(None)`, never a `KeyError`, an unusable value means the default.

`hosts.json` and `config.json` both carry `schema_version` (1); readers accept
it missing. `tests/fixtures/schema/` holds today's shape of each file plus a
hand-written older and newer variant, and every one of them must parse.

A host entry that still does not validate is **skipped, not fatal**: `gpuc`
warns, works with the rest, and writes that entry back untouched on the next
registry write -- it is very likely another session's host. Only a `hosts.json`
that cannot be parsed at all stops anything, and then gpuc prints the error and
the path, keeps a `.bak`, and refuses only the commands that would overwrite it.

## Exit codes

The table is in usage.md. The contract behind it: 0 covers an unreachable host,
a dead dispatcher and a pod that is gone (data about a host, reported per host),
3 means local state could not be read at all so the answer is *unknown*, and 4
is a name that does not exist. `gpuc status` never exits non-zero because of one
bad host entry or one unreachable host, and `gpuc status --json` always prints
one document. Automation keys on `hosts[].running` and treats exit 3 as unknown,
never as "nothing running".

`--json` is on every command that has an answer to give, and means the same
thing on each: stdout is one object carrying `schema_version`, everything else
the command has to say goes to stderr, and a failure prints
`{schema_version, error, exit_code}` rather than nothing. The flag never changes
an exit code. `gpuc logs --json` is the tail as `lines[]` plus where it was read
from; with `-f` it is exit 2, because a stream has no end. Each command's schema
is the table in usage.md.

## Transport

`LocalTransport` runs subprocesses directly. `SshTransport` uses the system
`ssh`/`rsync` with `-o BatchMode=yes -o ConnectTimeout=15`, per-command
timeouts, and host keys pinned on first contact into
`~/.local/share/gpu-coordinator/known_hosts` -- except for ephemeral hosts,
which get `known_hosts.d/<pod>`: RunPod recycles `host:port` between pods, so a
shared file plus `accept-new` wedges the *second* pod to land on a reused
endpoint.

The `ControlMaster` socket lives in `$XDG_RUNTIME_DIR/gpuc/` (else
`/tmp/gpuc-<uid>/`, 0700), **not** under the state dir, which under a long
`$HOME` does not fit. `transport.CONTROL_PATH_MAX = 100` is the one limit:
the template with `%C` expanded to its 40 hex characters is checked against it
before ssh runs, leaving room inside a unix socket's 108-byte `sun_path`. Any
ssh failure matching `ControlPath too long|unix_listener` raises immediately
even under `check=False`, because every polling loop here reads a non-zero ssh
as "not up yet" and would otherwise wait out its whole ceiling.

Code sync is `rsync` of `git -c core.quotePath=false ls-files -z --cached
--others --exclude-standard` from the submitter's directory: tracked files *and*
untracked ones git would keep, so a file written and not yet `git add`ed still
reaches the host, while `.gitignore` keeps venvs and caches home. Files in the
index but deleted on disk are dropped from the list rather than sent. `submit`
prints one line: `syncing N files (M modified, K untracked, ignoring
.gitignore'd)`. `uncommitted.patch` is `git diff HEAD -- .` taken against a *copy* of
the index with `git add -N` applied, so it carries untracked files too and never
touches the user's staging. `--no-git` rsyncs a non-repo directory whole, minus
`.venv, __pycache__, .git, *.pyc, node_modules, .uv-cache`, with a warning.
`put_file` writes 0600 content via stdin (`cat > path && chmod 600 path`);
secrets never touch argv.

## Bootstrap (any host, idempotent)

1. `curl -LsSf https://astral.sh/uv/install.sh | sh` if `~/.local/bin/uv`
   is missing; `uv python install 3.12` if no suitable interpreter.
2. rsync the `gpuc` package to `~/.gpuc/pkg/`; install `aws` CLI v2 bundle
   into `~/.local/aws-cli` and `uv tool install huggingface_hub` (both
   skipped if present; failures are warnings -- **except** that a host
   registered with an `s3_prefix` whose `aws` CLI could not be installed fails
   bootstrap outright: every job on it would end `failed: sync-preflight`).
3. Write `~/.gpuc/config.json` from the host registry entry.
4. Run `python -m gpuc.host health` and fail bootstrap on a failed check.
5. Start the dispatcher with `PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"`.
5b. Record the commit this build of gpuc came from (`direct_url.json` of the
   installed dist, else `git rev-parse` of the checkout) as `HostEntry.pkg_commit`
   and in the host's `config.json`. `gpuc host list` and `gpuc version` show it,
   and `gpuc status` warns when a host's differs from this machine's. Bootstrap
   is never blocked by running jobs: the package and config are replaced, an
   already-alive dispatcher keeps the lock until it exits, and whichever
   dispatcher takes over adopts the running jobs from their `state.json`.
6. Record what the host's cards are (`nvidia-smi --query-gpu=index,uuid,name,memory.total`)
   and the driver version from the health report into `HostEntry.gpu_info` /
   `driver_version`, so `gpuc host list` and `gpuc status` can name them. Best
   effort: a host with no nvidia-smi simply lists UUIDs. `gpuc host probe`
   records the same thing before a host is ever bootstrapped, and a RunPod host
   falls back to its offer's GPU name and VRAM.

## GPU ownership: indices in, UUIDs out

`--gpus` takes nvidia-smi indices (`--gpus 2,3`), UUIDs
(`--gpus GPU-8064...`), or a mix, and the registry and `config.json` store
exactly what was given. Indices because that is how a share of a shared box is
agreed and read off `nvidia-smi`; stored verbatim because resolving them at
registration would freeze one boot's numbering into a file nobody looks at
again.

Everything downstream is UUIDs. The runner resolves its assignment again on
the way in and writes the UUIDs back to the job state, and the dispatcher
resolves what it adopts at startup: a dispatcher process from before this and
a runner from after it meet during an in-place upgrade, and an index mistaken
for a busy card's name is a card handed out twice.

The dispatcher re-runs
`nvidia-smi --query-gpu=index,uuid --format=csv,noheader` each pass, maps the
owned indices to whatever the driver is calling those cards now, and assigns,
accounts for and pins jobs by UUID -- `CUDA_VISIBLE_DEVICES` is never an index.
A renumbered box therefore moves a job to the right card rather than silently
handing it someone else's; an owned entry that resolves to nothing is logged,
treated as unavailable (jobs wait, they do not fail), and reported by
`gpuc status` and the health check's `gpu_uuids`. Owning UUIDs only costs no
lookup at all: that mapping is the identity, and the host is not asked.

`gpuc status` and `gpuc host list` show `[index] name vram uuid` per owned
card -- the index from the host for `status`, and from the last probe for the
offline listing. `gpuc host probe` lists the owned cards only, headed `N of M
assigned to <host>`, because on a shared box the rest are somebody else's;
`--all-gpus` lists all M with the owned ones marked. It matches an owned index
against the numbering `nvidia-smi` gave *in that same probe*, records `gpu_info`
for every card either way (so a later `--gpus 5` resolves offline), and notes
owned entries no card answered to.

## Persistent root (a host whose `$HOME` is wiped on restart)

`gpuc host add|set <name> --persistent-root R` moves `GPUC_HOME` to `R/gpuc`,
and nothing else: the queue, specs, state, logs and workdirs are the state that
cannot be reinstalled. uv, its managed Pythons, `uv tool` installs and the
`aws` bundle stay in `$HOME` on every host -- bootstrap reinstalls them in
seconds, and these shared volumes are much slower than a container's local
disk. uv's *cache* is the exception: the venv lives in a workdir under gpuc
home regardless, so keeping the cache on the other volume does not save the
slow disk any writes -- it only stops uv linking. See The shared uv cache.
Bootstrap creates `R` 0700 if it creates it and leaves an existing `R`'s mode alone; everything inside is
gpuc home, which `paths.ensure_layout` already makes 0700. Without a root
nothing changes.

`HostEntry.env` (`--env K=V`, nothing populates it automatically) is written to
`config.json` as `HostConfig.env` and applied to every job's environment
*before* the job's own `env` -- by the dispatcher to every child it spawns, and
by the runner -- and to every `HostSession` invocation of the on-host package.
`UV_INSTALL_DIR`/`UV_TOOL_BIN_DIR` in it are also prepended to `PATH`.
`HostEntry.cache_dir` is the one key bootstrap fills in itself; it reaches
`HostConfig.env` as `UV_CACHE_DIR` and an explicit `env` entry always wins.

`gpuc host probe` reports `$HOME`'s filesystem type (`df -T`, `stat -f`
fallback) and suggests `--persistent-root` when it is an overlay. The health
check's disk floor is measured on `paths.home()`, so it is `R`'s volume when a
root is set. The operator runbook for a host that came back empty is in
setup.md.

## RunPod provider (v2 REST, `https://api.runpod.io/v2`, bearer `RUNPOD_API_KEY`)

- `offers(constraints)`: `GET /catalog/gpus?include=AVAILABILITY&product=POD&cloud=<tier>&minCudaVersion=<x>`
  once per requested tier; filter by name list / min VRAM / max price /
  availability != NONE and at least one `cudaVersions[].available`; sort by
  price. Pass GPU ids exactly as the catalog returns them.
- `create(offer)`: `POST /pods` with `name="gpuc-<host>"`, `image`
  (default `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`), `gpu.id`,
  `gpu.minCudaVersion`, `cloud`, `disk`, `ports: ["22/tcp"]`,
  `startSsh: true`, `env: {"HF_HUB_ENABLE_HF_TRANSFER": "0"}`. Before the
  first create ever, ensure the local public key is in
  `PUT /account/ssh-keys` (merge, do not replace others). One capacity
  error -> next offer. Never set secrets in `env`.
- `get(id)`: status, `cudaVersion`, `cost`, `runtime.gpus[].util`,
  `ssh.direct`. `logs(id)`: `GET /pods/{id}/logs` (SSE; read with urllib
  and a timeout). `terminate(id)`: `POST /pods/{id}/action {"action":"terminate"}`
  then poll `get` until `TERMINATED` or 404. `list()`:
  `GET /pods?includeClusterPods=true`.
- Caps: count pods with our prefix and sum their `cost`; refuse if `max_pods`
  or `max_total_usd_per_hour` would be exceeded. Checked once, in the
  provisioning flow with the state lock held -- unlocked, two concurrent
  sessions would both read "one pod running" and both create.

## Provisioning flow (`gpuc submit --runpod`)

1. Write the spec to S3 (`s3://bucket/gpuc/specs/<jobid>.json`) first.
2. Unless `--no-reuse`: pick an existing desired host whose recorded offer
   still satisfies the constraints, which owns enough cards, whose pod the
   provider reports RUNNING, whose dispatcher heartbeat is fresh, and which is
   neither draining nor paused; enqueue there. A registered pod the provider no
   longer has is forgotten rather than dialled.
3. Else for each offer in order: check caps -- offers are price-ascending, so
   a cap this one trips every later one trips too, and `CapsExceeded` aborts
   the whole submit instead of walking the list; then `create`; record
   `desired/<host>.json` with pod id, offer, created_at, ceiling; poll
   `get` until RUNNING **and** `ssh.direct` present; poll SSH until a
   trivial command succeeds; run bootstrap (which runs host health and
   starts the dispatcher); deliver the RunPod key as
   `~/.gpuc/secrets/runpod` (0600) for self-terminate. Enqueue. On any
   broken-host signature in `logs`, or the 15-minute ceiling, or a health
   failure: `terminate`, wait for TERMINATED, try the next offer. The
   `desired/` record is removed only once the terminate is *confirmed*; a
   terminate that failed leaves the record (and the registry entry) in place,
   because it is the only thing that makes the reaper retry a pod that is
   still billing.
4. Day-one test to run before relying on it: does the pod-scoped
   `RUNPOD_API_KEY` inside the pod terminate its own pod? If yes, do not
   deliver an account key at all.

## Reconcile loop

Every 60 s: the state lock is taken to *read* `desired/` and then for each
local mutation, never across the provider and ssh calls in between -- a
terminate polls for up to five minutes and a concurrent `gpuc submit --runpod`
gives up on the lock after two, and a create that cannot write its `desired/`
record is a leaked, billing pod. Each mutation re-reads the record it is about
to change. For each `desired/` host, `get` its pod; if
missing or TERMINATED, mark the desired entry gone and note any jobs that
were running there (for `requeue`). A pod older than its TTL -- only when that
host has one; the default is none -- is terminated and logged. For every
provider pod with our prefix not in `desired/`, terminate and log -- except that
a prefixed pod with no `desired/` record is left alone until it is older than
the 15-minute provisioning ceiling, so a concurrent session that has created
a pod but not yet written its record cannot have it reaped out from under it.

**The dead-dispatcher rule**, which is what replaced the overall TTL: a
bootstrapped desired host is asked for its status each pass (a 20 s ssh
timeout, so one wedged pod cannot stall the pass). A heartbeat under
`reconcile.HEARTBEAT_FRESH_S = 120` s, or any job the host says is running,
counts as alive and records `last_seen_at` in its `desired/` record. That
constant is the reaper's own and deliberately looser than the dispatcher's 30 s
staleness or the 30 s freshness reuse demands: this one decides whether to
terminate a pod. A host that
has managed neither for `Settings.dead_dispatcher_minutes` (30 by default) --
including one whose ssh never answers, since that never updates `last_seen_at`
either -- is terminated with a loud report: it cannot idle-terminate itself, it
is doing nothing we can see, and it is still billing. A long training run keeps
its host alive indefinitely *under this rule* -- a TTL the host actually has is
checked first and does terminate a pod with a job on it, which is exactly why a
TTL is opt-in. The 15-minute pre-healthy ceiling and the stray-pod rule are
unchanged.
Never touch a pod without the prefix. If `desired/` is unreadable, do nothing
and log an error (fail closed). `--install` writes a `systemd --user` service
and timer but does not enable them, and prints the `systemctl` lines and the
`config_dir()/env` file the service reads `RUNPOD_API_KEY` from.

## Status output

What `status` prints, and every flag, is usage.md. The invariants:

- An ephemeral host whose pod the provider reports missing or TERMINATED is
  `POD GONE`: no ssh is attempted, and the line says to run `gpuc reconcile
  --once` rather than printing a connection error.
- A finished job that produced `outputs:` which never reached S3/HF is flagged
  (`outputs not uploaded`, or `OUTPUTS LOST` once a drain has given up), because
  those are the jobs a purge -- or a pod going away -- would take with them. One
  that declared outputs and never wrote them is not: there is nothing there.
- A pod's `provider_util` is the provider's reading for the whole pod; a job's
  `util` is the host's own nvidia-smi sampler over that job's cards. They are
  labelled separately and never merged.
- `--suspects` judges each running job by *its own* `low_util` window, floor and
  grace as the host reports them, and never kills anything.

## Testing rules

- Unit tests run without a GPU (mock `nvidia-smi` output, temp `~/.gpuc`).
- Local GPU integration tests use tiny tensors (`torch.zeros(8)`), never
  more than ~100 MB VRAM; other people's jobs share the card.
- RunPod integration: A40 only, `--max-price 0.60`, a job whose command
  is under two minutes, `--idle-min 2`, `--ttl-hours 1` (a TTL is opt-in, and a
  test that creates a billable pod is exactly where opting in is right), and the test
  asserts teardown via `list()` and prints the final `GET /billing/pods`
  for the pod. A pod whose name lacks the `runpod_pod_prefix` belongs to
  someone else: read it in `list()`, never act on it. Every test that creates
  a pod has a `finally` that terminates it.

## Code conventions

uv, ruff, pyright strict-ish, type annotations everywhere, pydantic on the
control side only. No comments that restate the next line; explain *why*
where it is non-obvious. Prefer good names over docstrings. Errors carry
the command, the host, and the last lines of output.
