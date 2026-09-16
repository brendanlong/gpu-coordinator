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

Non-goals for the prototype: Vast, multi-node, spot, S3 as the
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
    queue.py      # enqueue, list, reorder, cancel, preempt and kill markers
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
    actions.py    # what the CLI and the dashboard both do: one function per command, returning its --json document
    web/          # `gpuc web`: stdlib http.server, bcrypt login, a static page over the same documents
    config.py     # ~/.local/share/gpu-coordinator/ layout, hosts registry (address + cache)
    connect.py    # `host add` / `host set`: read or write the host's own config.json
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
                     #  "shared_gpus": ["<index>" | "GPU-uuid", ...],  # cards it may borrow while
                     #                              # nobody else is on them; see Shared GPUs
                     #  "shared_min_priority": null | 0-99,  # a floor on how important a job must
                     #                              # be to borrow one (null: no floor)
                     #  "provider": null | {"kind":"runpod","pod_id":..},
                     #  "idle_minutes": 15, "ttl_hours": null | N, "s3_prefix": "s3://bucket/gpuc/<host>",
                     #  "retention_days": null | N,   # auto-purge horizon; null never purges
                     #  "pkg_commit": null | "<sha>", # the commit bootstrap shipped to this host
                     #  "env": {"HF_HOME": ...}}      # host-wide, hand-set; see Persistent root
secrets/<name>       # 0600 files delivered over SSH after boot. Never in argv, never in pod env.
incoming/<jobid>.json # a spec staged 0644 by `submit`, fed to `enqueue -` and deleted
queue/<prio>-<jobid> # empty marker files; lexical order is dispatch order. prio is 2 digits, default 50.
jobs/<jobid>/
  spec.json          # the submitted JobSpec. Only `gpuc estimate` and `gpuc reorder` rewrite it,
                     # and both keep keys another build wrote (see Reorder, Estimate below)
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
                     #  "workdir_bytes": int|null,  # what removing workdir/ would free;
                     #                              # measured once, as the job ended (or by
                     #                              # the first status to find it missing)
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
  kill               # a kill request with its reason (`ttl`, `low-util-pause`, `preempted`)
  preempt            # this job is coming back: `gpuc preempt` wrote it beside the kill request,
                     # and the dispatcher queues the job again once its runner has stopped it.
                     # Removed by whichever of the two decides the job is not coming back
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
  "use_shared": false,                  # may this job also be dispatched to `shared_gpus`?
                                        # see Shared GPUs
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
  "attempt": 1                          # set by `gpuc requeue` and `gpuc preempt`, not by the submitter
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
  once if GPUs allow; a `gpus: 0` job never waits. A job that asked to
  (`use_shared`) and does not fit in the free owned cards makes up the
  shortfall from the shared ones that are idle right now -- see Shared GPUs.
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
- Preempt: `queue.preempt(jobid[, prio])` writes `jobs/<id>/preempt` and then a
  `kill` marker with reason `preempted`; only a *running* job is accepted, and
  only when the host could actually run something else instead --
  `refuse_if_nothing_else_can_run` refuses a paused or draining host, an empty
  queue, and a queue whose every job sorts after `<prio>-<jobid>` (the
  preempted job's id is older than anything queued while it ran, so it wins a
  tie and would take its own cards straight back). Preempting costs the job
  everything it has done, so "it would only re-run the same job" is a refusal,
  not a surprise. The marker is written first and taken back off if the kill
  write fails: a marker nothing will act on re-runs the job the next time it
  fails for any reason at all.
  The runner stops the job and syncs exactly as it does for a TTL, and ends it
  `failed: preempted` -- but keeps `workdir/` whatever `cleanup:` says and keeps
  `secrets/<jobid>.env`, because the next attempt is the same job id and nothing
  delivers either a second time. Both decisions read a snapshot of the marker
  taken at the top of `_finalize`: from the final state write on, a dispatcher
  is entitled to consume the marker, and a runner that asked again afterwards
  would delete the very workdir the next attempt is about to run in. The
  baseline is not re-taken either (`runner._capture_output_baseline`), so the
  stopped attempt's results stay this job's outputs instead of becoming files
  "the checkout arrived with".
  The dispatcher re-queues it in `reap`, and in `adopt_orphans` for one whose
  dispatcher died first -- there, a job that is finished but whose runner is
  *still alive* is adopted rather than re-queued, so nothing is launched into a
  workdir that runner is still writing to. `queue.requeue_preempted` clears the
  `kill` marker, writes a *fresh* `state.json` (`queued`, `attempt+1`, nothing
  of the stopped attempt), writes the queue marker at the spec's priority, and
  removes the `preempt` marker **last**: interrupted, the job is queued and
  still asking to be queued, which the next pass completes as the same attempt
  rather than counting a second one. It does not go back if it is already back,
  if the attempt ended for a reason of its own rather than because something
  stopped it (`queue.STOPPED_BY_US`), if it was cancelled while stopping, if
  its workdir is gone, or if the host is draining or past its TTL -- a host
  that is going away must not queue a job nothing will run, and leaving it
  finished is also what keeps it in the drain's unconfirmed-output retry.
  A kill marker the dispatcher did not write gets a clock in `escalate_kills`
  the first pass it sees one, so a runner that ignores a preempt is escalated
  like any other kill.
- Isolation: at startup the dispatcher probes `systemd-run --user --scope
  --collect --quiet -- true` once and hands the answer to every runner it spawns
  as `GPUC_ISOLATION`. See Process isolation.
- Reorder: `queue.reorder(jobid, prio)` renames the marker *and*
  `jobs.update_spec(jobid, priority=N)`. The marker is the move -- it is what
  the dispatcher orders by -- but it is deleted at dispatch, so the spec is the
  only copy that outlives it and the only thing that can tell `gpuc status` what
  priority a *running* job was dispatched at. A spec that cannot be rewritten
  does not undo the move. The control side re-mirrors the spec for the same
  reason `estimate` does, below.
- Estimate: `jobs.update_spec(jobid, estimated_runtime_min=N)` rewrites
  `spec.json` (raw JSON, so keys another build wrote survive). Queued or
  running, since a running job's runner re-reads the spec; see Job length
  estimates. The control side then re-mirrors the spec, because `requeue`
  submits what S3 holds and would otherwise drop the estimate silently; a
  mirror it cannot write is a warning, not a failure.
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
   `workdir_removed` in `state.json`, and `workdir_bytes` with it -- what the
   workdir would free if it is still there, zero if it is not. The runner is
   the last thing standing in that tree and the job is over, so the figure will
   not change; `status` reads it rather than walking every finished venv on
   every call, which cost that command four seconds on a host holding sixty.
   A job that ended before the field existed, or whose runner died before
   writing it, has `null` there; the first `status` that finds a workdir still
   on disk with no figure walks it and writes one, within a per-call budget so
   the walking cannot cost the host its whole `status`. `gpuc clean` measures
   afresh instead of trusting any of it, because it is about to delete what it
   is quoting. Then upload state and log.
   A removal that fails is logged and nothing more: the job's outcome is
   already decided, and leftover disk is not worth turning a green run red. That last upload records
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
  identical to a wedged one. It may also be set *after* submitting, with
  `gpuc estimate <jobid> --minutes N`, which is only an edit of `spec.json`:
  the monitor loop re-reads the spec every `SPEC_REFRESH_S` and republishes the
  eta from it, so the estimate reaches a job that is already running -- exactly
  the job nobody could have estimated in time. The same re-read picks up a
  `progress_command` added mid-run, and its `progress_interval_s`. Those three
  fields are the whole of it: they describe what the job *reports*, while the
  command, the env and the outputs a run started with are what it ran with.
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
`(42%)` when it was measured and `(est)` when it was a guess -- falling back to
` est <total>` on a running job whose host published no eta, so the text can
never show less than `--json` does -- and adds one
`free` line per fully-busy host saying when its next card is expected -- with a
count of the running jobs that offered no end time, since the true answer can only
be sooner. Only the jobs holding a card are considered, for both halves of that
line; a host where none of them estimated an end time has no time to report, so
it gets no `free` line rather than one saying so.

The same estimates are what `status.queue_start_estimates` projects a queued
job's *start* from (`starts_in_s` / `starts_at`, and ` starts in ~2h10m` on the
queued line). It replays the dispatcher's own rule rather than assuming
strict head-of-queue order -- cards come free at the eta of whatever holds
them, the whole queue is walked at each release, and a job that fits into what
is free before the job ahead of it does starts first, exactly as
`launch_ready` backfills. A card held by a job that published no eta is not
schedulable at all, so a job whose turn depends on one is reported as *unknown*
rather than guessed; that is why a later job can have a start time when an
earlier one does not. A paused or draining host projects nothing, because
nothing is being dispatched. `submit`, `requeue` and `reorder` print the
projection for the job they just touched, alongside its queue position, and say
*why* there is no time when there is none -- "the host is paused" and "a job
ahead of it gave no estimate" are different things to do about it.

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
in `gpuc status --json`; the text view leaves them out, because what a kill
reaps is a question asked while debugging one job, not while scanning a host. The probe runs once per dispatcher and is passed to
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
the final sync and the final state write, never before. A preempted job is the
one exception to the policy: its workdir is what the next attempt re-runs from,
and nothing on this side would rebuild it. That attempt therefore starts in a
dirty tree -- whatever the stopped one wrote is still there -- which is the
trade `gpuc preempt` makes and what its docs tell the submitter.

`python -m gpuc.host clean (--all-finished | --older-than DAYS | --only IDS)
[--dry-run]` is the after-the-fact sweep, driven by `gpuc clean --host H`. It prints JSON:
per-job `bytes` (du-style allocated blocks, deduplicated by inode within the
tree), what was skipped and why, and the staged specs it removed. It fails
closed everywhere: a running or queued job, a job whose `state.json` is missing
or unreadable, and (under `--older-than`) a job with no parseable `ended_at`
are all skipped. It also removes `incoming/<id>.json` staged specs whose job
has finished, or which name no job at all and are over an hour old -- the
window in which that file is load-bearing is one SSH round trip.

## Retention and purge

`clean` keeps `spec.json`, `state.json` and `log.txt` forever. `python -m
gpuc.host purge [--older-than DAYS] [--dry-run] [--force] [--only IDS]
[--sweep-only IDS]` (driven by `gpuc clean --host H --purge`) removes the whole
`jobs/<id>/`, plus any stray queue marker, secrets file and staged spec, for
finished jobs older than DAYS (default 7, from `ended_at`) that carry two
records in their own `state.json`:

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
*purged* and `--sweep-only` narrows that sweep. The purge removes the whole
`jobs/<id>/` plus that job's queue marker, staged spec and `secrets/<id>.env`.

`gpuc clean --host H --only ID[,ID...]` is the user-facing form of both, sent
as `--only` *and* `--sweep-only`, so purging one job does not reclaim every
other finished job's venv on the way past. The two host flags stay separate
because `--verify` needs them to differ: it purges the ids whose mirrored
`log.txt` answered and sweeps the ids the user asked about, which is how a named
job that failed verification still gets its workdir back.

Naming ids *replaces* the age gate rather than tightening it: a named job is
purged and swept whatever `--older-than` says and even if its `state.json`
records no usable `ended_at`, which is the shape a job whose state write was cut
short has -- exactly the stuck kind somebody names. The preconditions are
untouched: a named job with no confirmed mirror or unconfirmed outputs still
needs `--force`, and a running or queued one is never touched at all.

A named id no job dir matches is a typo, and a typo in a delete is not
half-honoured: the host reports it, removes nothing at all, and exits 1. That
exit code is why `clean` and `purge` are the two subcommands the control side
reads with `host_json(check=False)` -- their document *is* the report of the
failure, and raising on the exit code would throw away the account of what did
and did not go. The control side refuses an empty `--only` before it can become
"none of them".

`--only` needs a host package that knows `--sweep-only`; an older one rejects
the command line in argparse, before any subcommand runs, so nothing is deleted
and the error says to re-run `gpuc host bootstrap`.

The dispatcher reclaims disk on two horizons, both once at startup and then at
most once an hour, purge first. A non-ephemeral host's dispatcher only lives
while there is work, so in practice both happen on the next submit.

`HostConfig.retention_days` (`gpuc host add|set --retention-days N`, null by
default) is the purge: whole job dirs, never forced.

`HostConfig.workdir_days` (`gpuc host add|set --workdir-days N`) is the
ordinary `clean` sweep over finished jobs that ended that long ago. It takes
`workdir/` and leaves `spec.json`, `state.json` and `log.txt`, so it has no
mirror precondition and takes nothing a re-run cannot rebuild -- which is why
it may be short where the purge may not.

Its default is `cleanup.DEFAULT_WORKDIR_DAYS` (1), and it lives in
`connect_host`, applied only to a host being configured for the first time --
*not* on the `HostConfig` field, which stays null. Those are different
questions: a host getting its first config should reclaim its venvs, and a host
whose `config.json` predates the key should not start deleting because somebody
shipped it a newer package. An adopted config that says nothing about
`workdir_days` has been getting along without the sweep, and meeting it is not
the moment to start.

Both dispatcher passes go through `cleanup.clean(automatic=True)`, which adds
the two refusals that only make sense for a delete nobody typed: a job whose
spec says `cleanup: never`, and a job whose `outputs:` are not confirmed
elsewhere -- `outputs:` resolve inside `workdir/`, so without that guard the
sweep would bin precisely what `purge` fails closed on and `status` flags as
`outputs not uploaded`. An unreadable `spec.json` is skipped for the same
reason an unreadable `state.json` is. `gpuc clean --only <id>` waives both,
because that is a person naming the job. With both horizons set the effective
workdir horizon is the shorter, since each purge pass sweeps at its own.

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
pure cost). A finished job always gets a number: zero when its workdir is gone,
which is one `is_dir()` and needs nothing recorded, and otherwise the recorded
figure or a walk on the spot, bounded by `cleanup.MEASURING_BUDGET_S` so a
host that has many to do reports the rest as `null` and measures them on the
next call rather than blowing the control side's deadline and reporting nothing
at all. Nothing schedules any of it, so a host with no dispatcher running --
the normal state of an idle one -- answers as well as a busy one. `gpuc status` prints one line per host once the total exceeds 1 GiB.

Every byte figure a `clean`, `purge` or `status` reports is
`cleanup.reclaimable_bytes`: what deleting the tree gives the filesystem back,
not what `du` says it holds. They differ by the venv uv built out of its cache,
which is most of a torch venv, and a number that counts bytes the cache keeps
is one nobody can act on. Measured: 8.36 GiB of `du` returning 0.13 GiB on one
host, 15.00 GiB returning 0.83 GiB on another.

Both of uv's sharing modes count, because which one a host uses is the host's.
Hardlinks fall out of `st_nlink`, which the walk's `stat` already carries.
Reflinks share extents without sharing an inode, so they need `FIEMAP` and its
`FIEMAP_EXTENT_SHARED` flag -- one ioctl per file, which triples the walk. That
is affordable because the walk happens **once per job**, not once per `status`:
the answer goes in `JobState.workdir_bytes` (see the runner's step 7) and
readers read it -- and the job that no longer has a workdir, which is most of
them, is answered without walking anything. Nothing cheaper is exact --
`LOGICAL_INO` costs more per extent, a filesystem scan is O(extents on the
device), and only btrfs qgroups
answer in O(1), per subvolume, with quotas on -- so the trade is to pay it once
and write the number down. Every way it can fail (no FIEMAP, no permission, an
odd filesystem) means "assume it is all yours", so no *file* is ever
under-counted. The figure as a whole still can be: `FIEMAP_EXTENT_SHARED` says
"shared with something", not "shared with something outside this tree", and
finding out which costs a backref walk per extent. On a filesystem where every
extent is shared with a snapshot, a workdir reports close to nothing. See
`reclaimable_bytes` for the full list of what that misses.

`cleanup.dir_size` is the `du` twin and asks none of this; the uv cache's own
size in `health` is the one question that wants it.

## The shared uv cache

uv caches wheels under `~/.cache/uv` and materialises a venv by reflinking or
hardlinking out of it, so the second job needing the same torch build costs
seconds and almost no disk. Both mechanisms work only *within one filesystem*,
and `UV_LINK_MODE=copy` disables them outright. Two rules follow:

- Nothing in the job path sets `UV_LINK_MODE`, and neither the dispatcher nor
  the runner sets `UV_CACHE_DIR` unless `HostConfig.env` does.
- `gpuc host bootstrap` compares the filesystem of gpuc home with that of `uv
  cache dir`. If they differ it sets `UV_CACHE_DIR` in the host's own
  `HostConfig.env` to `<parent of gpuc home>/.cache/uv` -- beside gpuc home,
  not inside it, so `clean` and an `rm -rf` of gpuc home cannot take the cache
  with them. A cache the host's config already names is never overridden,
  however it got there (`--cache-dir`, `--env UV_CACHE_DIR=...`, or an earlier
  bootstrap), and an unreadable comparison changes nothing.

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
- A host name is looked up locally; `logs`, `cancel`, `preempt`, `reorder`,
  `estimate` and `requeue` fall back to the job index and then to asking each host, and an id nothing
  knows is exit 4, never a guess.
- Nothing runs in the background on this side except the optional
  `gpuc reconcile` timer and, if installed, the web dashboard's service.

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
`HostCache`, `Registry`, `DesiredHost`, `Settings`, `IndexEntry`, `Offer`). The
host config a registry entry caches is kept **verbatim** on top of that, so a
key some newer build wrote survives a round trip through this one. Host side, with
no pydantic, the same rules are spelled out in `jobs.from_dict` for
`HostConfig`, `JobSpec`, `JobState` and the dispatcher's lock body: never
`float(None)`, never a `KeyError`, an unusable value means the default.

`hosts.json` and `config.json` both carry `schema_version` (1); readers accept
it missing. `tests/fixtures/schema/` holds today's shape of each file plus a
hand-written older and newer variant -- and, for `hosts.json`, the pre-split
shape that carried each host's config flat beside its address. Every one of
them must parse.

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

## The registry is an address book

`hosts.json` holds two kinds of thing about a host and only two:

- the **address** -- `name`, `kind`, `ssh`, `port`, `gpuc_home` /
  `persistent_root`, `pod_id` -- which is hand-entered, local to this machine,
  and is everything needed to open a session and find `config.json`. Nothing in
  it is a fact about how the host behaves.
- a **cache** of what the host last said: `python`, `uv`, `gpu_info`,
  `driver_version` and a copy of its `config.json`, stamped with `read_at`.
  Offline commands (`host list`, `version`) print it labelled "as of <age>";
  anything that decides something reads the host.

What the host **is** -- `gpus`, `s3_prefix`, `env`, `idle_minutes`,
`ttl_hours`, `retention_days`, `provider`, `pkg_commit` -- lives in
`config.json` on the host and nowhere else. One box driven from a desktop and a
laptop therefore has one configuration, not two, and nothing about the machine
that bootstrapped it first matters afterwards.

- `gpuc host add <name> --ssh ...` is a **connect**: probe, read `config.json`,
  and if it is there adopt it -- registering only the address -- under the name
  the host calls itself. Flags are explicit per-field overrides, written
  through and reported (`host <- retention_days 30.0 -> 7.0`). A `--gpus` that
  overlaps the host's existing set without matching it is **refused**
  (`--force` overrides): every other difference confuses a listing, that one
  hands one card to two jobs. A host with no config is where `--gpus` is
  required and the initial config is written.
- `gpuc host set <name> --gpus ... --env ...` **writes through** to
  `config.json` via `python -m gpuc.host config --merge` (one atomic
  read-modify-write on the host, by the code that reads the file; a host with
  no package yet gets the same merge done here and the file replaced by
  rename). It does not work offline, which is correct: there is no local copy
  to set. `--persistent-root` and `--gpuc-home` are addresses and stay here.
- `gpuc host probe` refreshes the cache and nothing else -- including an
  interpreter to run the on-host package with, so a host somebody else
  bootstrapped answers `status` and `host set` before this machine has
  bootstrapped it.
- "the host has no config" is a marker the host echoes, never the absence of
  parseable output: a `config.json` that is there and does not parse is a file
  the host is running on, so `read_remote_config` returns `None` for it, and
  nothing -- connect, `host set` or bootstrap -- writes over a `None`.
- `gpuc submit`'s pre-enqueue read of `config.json` is the only source for the
  `gpus` a spec is judged against, and it refreshes the cache on the way past.
- Provision is create pod -> wait for ssh -> the same connect, with the initial
  config a pod nobody has configured yet needs.

A registry from before this split still parses: `HostEntry` folds the config
fields it carries into the cache (`cache_dir` becomes `env.UV_CACHE_DIR`, which
is the only place the host ever had it), and the next connect, `host set` or
bootstrap works from the host's own copy. A `config.json` from before it needs
no change at all: it is already the shape the host reads.

## Bootstrap (any host, idempotent)

1. `curl -LsSf https://astral.sh/uv/install.sh | sh` if `~/.local/bin/uv`
   is missing; `uv python install 3.12` if no suitable interpreter.
2. rsync the `gpuc` package to `~/.gpuc/pkg/`; install `aws` CLI v2 bundle
   into `~/.local/aws-cli` and `uv tool install huggingface_hub` (both
   skipped if present; failures are warnings -- **except** that a host
   registered with an `s3_prefix` whose `aws` CLI could not be installed fails
   bootstrap outright: every job on it would end `failed: sync-preflight`).
3. **Never rewrite `~/.gpuc/config.json`.** The host owns it (see The registry
   is an address book), so bootstrap reads it -- the `env` in it decides which
   uv cache the installs populate and which tool directories go on PATH -- and
   merges back exactly two keys, through the host's own `python -m gpuc.host
   config --merge`: the commit it just shipped, and `env.UV_CACHE_DIR` when the
   host names none (`resolve_cache_dir`, decided from the host's own
   filesystem). The one exception is a host with **no** config at all -- one
   registered before this split, or one whose gpuc home was wiped -- where
   there is nothing to preserve and the last config this machine read off it is
   restored, with a warning if it has none either.
4. Run `python -m gpuc.host health` and fail bootstrap on a failed check.
5. Start the dispatcher with `PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"`.
5b. Record the commit this build of gpuc came from (`direct_url.json` of the
   installed dist, else `git rev-parse` of the checkout) in the host's
   `config.json`. That copy is the authoritative one and is what `status`
   reports and what `submit` judges (see Which build is a host running); the
   registry caches it for the offline commands, which say when it was read.
   Bootstrap
   is never blocked by running jobs: the package is replaced, an
   already-alive dispatcher keeps the lock until it exits, and whichever
   dispatcher takes over adopts the running jobs from their `state.json`.
6. Record what the host's cards are (`nvidia-smi --query-gpu=index,uuid,name,memory.total`)
   and the driver version from the health report into the registry's cache, so
   `gpuc host list` and `gpuc status` can name them. Best
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

Shared entries (`--shared-gpus`) go through exactly the same resolution, and
are checked for overlap with the owned ones -- see Shared GPUs.

`gpuc host list` shows `[index] name vram uuid` per owned card, from the last
probe; `gpuc status` shows `[index] free|busy name vram` from the host itself
and names each running job's cards on the job's own line (`gpu=2,3`), because a
UUID per card and a job id per card was most of what made that block unreadable. `gpuc host probe` lists the owned cards only, headed `N of M
assigned to <host>`, because on a shared box the rest are somebody else's;
`--all-gpus` lists all M with the owned ones marked, and a host that owns
nothing (or whose entries resolve to nothing) sees every card, because a probe
with nothing to show is the one that needed the list most. It matches an owned
index against the numbering `nvidia-smi` gave *in that same probe*, records
`gpu_info` for every card either way (so a later `--gpus 5` resolves offline),
and notes the two assignments `check_gpu_uuids` would later refuse to bootstrap:
an entry no card answered to, and two entries folding onto one card.

## Shared GPUs: cards we borrow rather than own

Some boxes hand us a few cards outright and leave the rest to other people.
`config.gpus` is the first kind and `config.shared_gpus` is the second, spelled
the same way (index or UUID) and resolved the same way each pass. The
difference is the whole feature: an owned card is handed out whenever it is
free, and a shared card is only ever *borrowed* -- for one job, and only while
nobody else is on it.

Three gates, all of which have to pass:

1. **The job asked.** `use_shared: true` in the spec, or `gpuc submit
   --use-shared`. Off by default, because taking somebody else's card is a
   decision about a box and not about a job; a job that did not ask is never
   dispatched to one, and shared cards are not counted as capacity for it.
2. **The job is important enough.** `shared_min_priority` on the host, when it
   is set: priorities run 0-99 and lower dispatches first, so it is a *ceiling*
   on the number. Null means no floor.
3. **Nobody else is on the card.** `nvidia-smi --query-gpu=memory.used,
   utilization.gpu` reports 0 MiB *and* 0% for it, right now. Memory is the
   stronger half -- a CUDA context holds hundreds of MiB between steps, so
   0 MiB means no process, while util alone dips to zero between epochs of
   somebody else's run. Every way of not knowing (nvidia-smi failing, a card it
   did not answer about, a `[N/A]` reading) counts as *in use*: this decides
   whether to run on somebody else's GPU, so absence of evidence has to count
   against.

Owned cards first, always. A job takes every free owned card it can use and
borrows only the shortfall, so a shared card is held for the shortest time that
runs the job. That one rule covers both things shared cards are for: a
one-card job borrows when the owned cards are all busy (the queue drains
faster), and a four-card job on a host that owns two and shares two waits until
both of the shared ones go quiet and then runs across all four.

The preflight sample is taken once per dispatch pass and only when a job
actually needs to borrow -- so a host with nothing queued that wants a shared
card runs no extra `nvidia-smi` -- and every job in that pass is judged against
the one reading, which is also what stops two of them being handed one card.

`gpuc status`'s start-time estimate models the shared cards that are idle
*right now*, and only for the jobs allowed onto them, so a borrower whose turn
is the next dispatch pass is told `starts now` rather than being made to wait
for an owned card it will not want. A card somebody else is on is left out of
the model entirely: when they will stop is the one thing this host cannot know,
so a job waiting for one has no start time and is told why.

Two things this deliberately does not do:

- **Yield.** Once a job is running on a borrowed card it keeps it until it
  ends. Someone else starting a job on that card is a collision gpuc does not
  detect and does not resolve; `gpuc preempt` is the manual way out.
- **Fail a job over `shared_min_priority`.** Dropping a job out of the queue is
  permanent, so the dispatcher only does it on a gate a queued job cannot get
  past. `use_shared` is one -- nothing changes it after submit -- and the
  priority floor is not: `gpuc reorder` moves a job over it, `gpuc preempt
  --priority` brings one back above it, and `gpuc host set
  --shared-min-priority` moves the floor under everything already waiting.
  A job held up by the floor alone waits. `gpuc submit` still refuses it up
  front, where nothing has been queued yet and the submitter is looking.

A card in both lists is refused -- by `gpuc host add|set`, where it was typed,
and by the host's own `gpu_uuids` health check at bootstrap. Should one reach a
dispatcher anyway, owning it wins.

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

`HostConfig.env` (`--env K=V`, nothing populates it automatically) is applied to
every job's environment *before* the job's own `env` -- by the dispatcher to
every child it spawns, and by the runner -- and to every `HostSession`
invocation of the on-host package. `UV_INSTALL_DIR`/`UV_TOOL_BIN_DIR` in it are
also prepended to `PATH`. `UV_CACHE_DIR` is the one key bootstrap fills in
itself, and only when the host's config names none; `--cache-dir` is that same
key by another name.

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
host has one; the default is none -- is terminated and logged.

**The pod is the record** (`rented.py`). `desired/<host>.json` exists only on
the machine that ran `gpuc submit --runpod`, so a reaper that trusts it alone
terminates another machine's healthy pod at the ceiling. A pod therefore carries
its own copy: `config.json` -- the file the host owns -- holds `offer`,
`created_at` and `bootstrapped_at` under the `provider` block that already named
its `kind` and `pod_id`. Every pass asks each prefixed pod it has no record of
(one ssh session: expand gpuc home, then read `config.json`), and a pod holding
a gpuc config is *ours*
whoever created it: it is judged by the rules above, and the answer is cached in
this machine's `desired/`, which is what keeps it watched on a later pass that
cannot reach it. So the timer is a watchdog role that any machine holding the
API key can run, and none of them is special.

**Nothing is terminated for the absence of a record.** A prefixed pod this
machine has no record of and cannot get an answer out of is reported every pass,
with its age and its hourly cost, and left running: it may be wedged, it may
hold no key of ours, or it may be another machine's `create` still
bootstrapping, and those are indistinguishable from here. Terminating on that
guess is what took someone's running job, and the guess buys little — the
machine that *does* hold a pod's record still reaps it on TTL and on a dead
dispatcher, and a healthy pod terminates itself on idle. What is left over is a
pod that never got a config whose creating machine never comes back: it bills
behind a report line until a person ends it, which is the deliberate trade.
`gpuc host add <name> --pod <id>` moves that duty here, through the same connect
path as any other host.

**Adoption is permanent and one-way**, and this is the sharpest edge in the
design. After one successful read, this machine holds a `desired/` record for
that pod for as long as the pod exists -- nothing evicts it but a terminate or
the pod going away -- so it will terminate that pod after
`dead_dispatcher_minutes` of it not answering *this* machine, with the machine
that created it never consulted. That is the trade for having a watchdog at all:
the alternative is a wedged pod that bills until a human notices. The half of it
worth knowing is the ssh key (a machine whose key the pod does not hold can
never adopt it, and reports it forever instead), which setup.md says under the
reconcile timer.

Adopting stamps `last_seen_at` on the cached record, because the pod answered in
that same pass: a machine that has only just met a pod
gives it the same `dead_dispatcher_minutes` allowance as one it provisioned
itself, rather than measuring silence from a `bootstrapped_at` days old. The
name on the record comes off the config document rather than the parsed config,
whose default `host` is `local` — a record called `local` would be matched
against this machine's own host on the next pass.

**The dead-dispatcher rule**, which is what replaced the overall TTL: a
bootstrapped desired host is asked for its pulse each pass (a 20 s ssh
timeout, so one wedged pod cannot stall the pass) -- the dispatcher heartbeat's
mtime and a count of the jobs whose `state.json` says `running`, read with
`stat` and `grep` rather than by running the host's package, because the machine
reconciling a pod may never have bootstrapped it and knows no interpreter there.
A heartbeat under
`rented.HEARTBEAT_FRESH_S = 120` s, or any job the host says is running,
counts as alive and records `last_seen_at` in its `desired/` record. That
constant is the reaper's own and deliberately looser than the dispatcher's 30 s
staleness or the 30 s freshness reuse demands: this one decides whether to
terminate a pod. The clock is capped by this machine's own
watching: `reconcile` keeps `watch.json` in the state directory with the time of
the last pass and the start of the current unbroken stretch, and a gap of more
than `WATCH_GAP_MINUTES` (5) resets the stretch. Silence that nothing observed
is not evidence -- a desktop resuming from three days asleep would otherwise
terminate every pod on the first pass whose ssh had not come up yet, and the
timer fires two minutes after boot. The service therefore also `Wants=` the
network target it is `After=`, since `After=` alone does not pull it in. A host
that has managed neither for `Settings.dead_dispatcher_minutes` (30 by default) --
including one whose ssh never answers, since that never updates `last_seen_at`
either -- is terminated with a loud report: it cannot idle-terminate itself, it
is doing nothing we can see, and it is still billing. A long training run keeps
its host alive indefinitely *under this rule* -- a TTL the host actually has is
checked first and does terminate a pod with a job on it, which is exactly why a
TTL is opt-in. The 15-minute pre-healthy ceiling is unchanged, and it only
applies to a record that says the pod was never bootstrapped -- which an adopted
one never does.
Never touch a pod without the prefix. If `desired/` is unreadable, do nothing
and log an error (fail closed). `--install` writes a `systemd --user` service
and timer but does not enable them, and prints the `systemctl` lines and the
`config_dir()/env` file the service reads `RUNPOD_API_KEY` from.

## Web dashboard (`gpuc web serve`)

A thin view, by construction: `gpuc.control.actions` holds one function per
command that returns the document its `--json` form prints, and both `cli.py`
and `web/app.py` call those. Nothing the dashboard shows or does exists only in
the dashboard; a job the CLI refuses to reorder is refused on the wire with the
same words, and `exit_code_for` is the one table mapping an error to an exit
code (CLI) or an HTTP status (2 -> 400, 3 -> 503, 4 -> 404, 1 -> 500). The
API's failure document is `--json`'s: `{schema_version, error, exit_code}`.

The server is stdlib `ThreadingHTTPServer` -- a routing table this size buys
nothing from a framework -- with `bcrypt` the one added dependency. One
password, hashed into `config_dir()/web-password` (0600) by `gpuc web
set-password` and read once at startup; a server with no password refuses to
start. Sessions are random tokens held in memory, `HttpOnly; SameSite=Strict`,
seven days; a POST that carries an `Origin` header must match `Host` (a
browser that honours `SameSite` never sends the cookie cross-site in the
first place, so this is the second lock). Wrong passwords are checked one at
a time under a lock of their own with a growing pause, which a quiet minute
resets; the session table has a separate lock, so a guesser at the door
cannot stall requests from inside. Idle keep-alive connections time out
after 30 s, an oversized or malformed body is refused before it is read, and
a bug in a handler is a 500 with a traceback in the server log, never a
dropped connection. No TLS: it is for localhost or a VPN, or behind a proxy.

The page is static HTML/JS that fetches `/api/status`, `/api/hosts`,
`/api/config` and `/api/version` every 15 s and polls `/api/jobs/<id>/logs`
while a log panel is open with *follow* on. Status is gathered across hosts in
parallel (`actions.gather_all`, also what the text `gpuc status` uses now), so
one wedged host costs its own timeout, not the sum. Realtime updates later mean
an event stream fed by the same gather beside the same documents, not a second
rendering: the page already treats every document as the whole truth on each
refresh, so replacing the timer with a stream changes nothing it draws.

`gpuc web serve --install` writes `gpuc-web.service` to `~/.config/systemd/user`
the way `reconcile --install` writes its timer -- the two share
`control/systemd.py` for the unit directory, an absolute `gpuc` for
`ExecStart` (quoted the way systemd reads it) and writing without enabling.
The unit pins the config and state dirs, reads the timer's env file if it
exists, and restarts on failure under a start limit, so a service with no
password fails after five tries rather than looping for ever.

Host shutdown is deliberately absent, because the CLI has no such command yet
(setup.md's teardown is a hand procedure); the rule is that it lands in `actions`
first and the dashboard calls it.

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
- Every job carries the `priority` it is (or was) ordered by. While a job is
  queued that is its marker's, which is what the dispatcher reads; once the
  marker is gone it is the spec's, which `reorder` keeps current. A host too old
  to report either says `null`, never a default.
- The host's `status` reports each job's `outputs` (the spec's, `{job_id}`
  expanded) and `wandb` (`entity`, `project`, `run_id` from the job's `WANDB_*`
  env, and nothing else of its env). The control side turns those into
  `links[]` in `--json` -- S3 console, HF tree, W&B run, and the host's
  `s3_prefix` mirror of the job -- derived from what the job declared and never
  checked; the text view does not show them.
- Everything said about *what a host is* -- the build it runs, the cards it was
  registered with, what it calls itself -- is the host's own answer, and a host
  that could not be reached produces no claim about any of it. See below.

## Which build is a host running

The package is the one thing a host cannot own, because it is shipped to it: two
machines on different commits, both `gpuc host bootstrap`-ing the same ssh box,
leave it running whichever shipped last, and neither registry can see the
other's. (Everything else about a host it does own -- see The registry is an
address book.)

So the authoritative copy is the host's: `config.json`'s `pkg_commit`, written
by every bootstrap and re-ship, reported back by `python -m gpuc.host status`.

- `gpuc status` warns from that value (`host gpubox is running gpuc <sha> and
  this machine has <sha>`), never from the registry, and says nothing at all
  about a host it could not reach. It judges by the same rule `submit`
  re-ships on, so a host that *answered* and named no commit -- one on a build
  from before `status` reported it -- is warned about rather than passed as
  current. `status --json`'s `pkg_commit` is the host's answer, so `null` there
  means "the host did not say", never "current".
- `gpuc submit` and `gpuc requeue` read the host's `config.json` before they
  enqueue and re-ship the package when it does not match this build -- an
  unrecorded commit included. That same read is what the rest of the submit
  works from: the `gpus` the spec is judged against and the `s3_prefix` the
  job's outputs are recorded under are the host's own answer, from a moment
  ago, and the whole of it replaces the registry's cache on the way past.
- `gpuc host list` and `gpuc version` never ssh, so they report the commit the
  host was running when this machine last read it, labelled with its age
  (`pkg <sha> on the host, as of 3m ago`), and point at `gpuc status` for what
  it is running now.

The build is the only thing left that two machines can disagree about, because
it is the only thing a host cannot own: the package is shipped to it. Its
config is not -- the host holds the only copy, every command that acts reads it
first, and a registry that says something else is simply a stale cache (see The
registry is an address book), so there is nothing to warn about and nothing to
reconcile by hand.

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
