# gpu-coordinator architecture

The contract the code keeps: what lives where, who owns what, and the rules
each half holds to. It is not the CLI reference -- flags and defaults are
`gpuc --help` and each subcommand's `--help`, what they mean together is
[usage.md](usage.md), and installing and registering hosts is
[setup.md](setup.md). Where a section names a function, the *why* lives in
that function's docstring and is not repeated here.

## Goals

One submit path for three kinds of host: this machine (the cards you give it),
a box reached over SSH with no sudo there (a subset of its GPUs), and ephemeral
RunPod pods. Not finicky, not buggy, and never leaks a paid pod in the normal
path.

Non-goals for the prototype: Vast, multi-node, spot, S3 as the
authoritative queue (host is authoritative, S3 is the mirror; `gpuc requeue`
resubmits from the S3 spec if a host dies), and a guaranteed rental teardown.
Nothing on a client watches a pod after handoff: a rental whose dispatcher
dies after it was set up, or whose provisioning client was killed uncleanly
mid-create, bills until a person ends it, and `gpuc pods` is how a person sees
that.

## Package layout

One uv project, Python >= 3.11, package `gpuc`, CLI entry point `gpuc`.

```
gpuc/
  _version.py    # __version__ and user_agent(); STDLIB ONLY, imported by both halves
  host/          # runs ON hosts. STDLIB ONLY: `python -m gpuc.host <cmd>` with a bare interpreter
    __main__.py    # the on-host CLI the control side drives over ssh
    paths.py       # the ~/.gpuc layout
    jobs.py        # job ids, HostConfig/JobSpec/JobState, tolerant readers, atomic writes
    queue.py       # enqueue, list, reorder, cancel, preempt and kill markers
    dispatcher.py  # lock+heartbeat, pick next runnable, launch runner, idle terminate
    runner.py      # one job: env, CUDA_VISIBLE_DEVICES, preflights, watchdog, sync, exit code
    scope.py       # systemd --user scope probe/wrap/stop; the cgroup kill path
    preflight.py   # sync preflight: prove `aws`/`hf` can write before the job runs
    baseline.py    # what was already under `outputs:` before the job started
    cleanup.py     # `cleanup:` policy, workdir sizing, the `clean`/`purge` sweeps
    gpus.py        # nvidia-smi parsing, index<->UUID resolution, utilization sampling
    progress.py    # the optional `progress_command`: run it, read a percentage off it
    sync.py        # periodic upload loop (shells out to `aws` or `hf`)
    health.py      # host preflight: driver, owned GPUs, disk, network download timing
    terminate.py   # self-terminate via provider API (urllib), key from ~/.gpuc/secrets
  control/       # runs on the local machine; may use third-party deps
    cli.py         # argparse and the text output of every `gpuc` command
    actions.py     # one function per command returning its --json document; the CLI and the web call these
    status.py      # gather a host's status, render it, project queue start times
    submit.py      # validate a spec, sync the workdir, deliver secrets, enqueue
    provision.py   # `--runpod`: offers, create, wait for ssh, bootstrap, reuse
    rented.py      # a pod is its own record: the `provider` block in its config.json
    pods.py        # `gpuc pods`
    config.py      # ~/.local/share/gpu-coordinator/ layout, Settings, the hosts registry
    connect.py     # `host add` / `host set`: read or write the host's own config.json
    bootstrap.py   # install uv + this package on a host, run host health, start the dispatcher
    probe.py       # `host probe`: what a host has, before bootstrap
    remote.py      # HostSession: run the on-host package with its environment pinned
    transport.py   # LocalTransport / SshTransport: run, rsync, put_file(0600), tail
    ssh.py         # `gpuc ssh`: the interactive form of the transport
    clean.py       # `gpuc clean` / `gpuc host clean --uv-cache` over the transport
    s3index.py     # the S3 mirror of specs and the job index, and the local index
    gpuinfo.py     # per-GPU name/VRAM for the registry and every listing
    version.py     # this build's commit, and comparing it with a host's
    jsonout.py     # the `--json` rules
    skill.py       # `gpuc skill`: the agent guide, from the wheel or the checkout
    systemd.py     # what `web serve --install` needs of systemd: unit dir, ExecStart, write
    web/           # `gpuc web`: stdlib http.server, bcrypt login, a static page over the same documents
    providers/
      base.py      # Provider interface: offers(constraints), create, get, logs, terminate, list
      runpod.py    # v2 REST implementation
skills/gpuc/SKILL.md  # the agent guide; force-included in the wheel as gpuc/SKILL.md
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

One string, `gpuc/_version.py:user_agent()`, on every outbound request from
either half (the README lists the callers). `gpuc.host` is stdlib-only, so
`gpuc._version` must stay stdlib-only and bootstrap's rsync must ship it. The
`aws` CLI on a host cannot have its User-Agent overridden.

## On-host state: `~/.gpuc/`

```
config.json          # {"schema_version": 1, "host": "<name>", "gpus": ["GPU-uuid" | "<index>", ...],
                     #                              # what this host owns, as it was registered;
                     #                              # see GPU ownership
                     #  "shared_gpus": ["<index>" | "GPU-uuid", ...],  # cards it may borrow while
                     #                              # nobody else is on them; see Shared GPUs
                     #  "provider": null | {"kind":"runpod","pod_id":..},
                     #  "idle_minutes": 15, "s3_prefix": "s3://bucket/gpuc/<host>",
                     #  "retention_days": null | N,   # auto-purge horizon; null never purges
                     #  "workdir_days": null | N,     # auto workdir sweep horizon; see Retention
                     #  "created_at": str,            # when this config was first written
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
                     #  "util_recent": [float|null, ...],
                     #  "progress_pct": float|null, "progress_error": str|null, "eta": str|null,
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
  kill               # a kill request with its reason (`preempted`)
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
```

All state writes are atomic (write temp in same dir, `os.replace`).

## JobSpec (JSON; `gpuc submit` accepts YAML or JSON and normalizes)

```
{
  "name": "lego-s4",                    # human label, not an identifier
  "command": "uv run python -m experiments.lego.train --k-max 6",
  "setup": "uv sync --frozen",          # optional; runs before command, phase=setup
  "gpus": 1,                            # at least 1
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
  "auto_preempt": false,                # let the dispatcher stop this job, as often as it
                                        # takes, whenever that starts a strictly more
                                        # important queued one right away
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

- Started by every `enqueue` (and by bootstrap) in its own session
  (`dispatcher._spawn_host_process`), so it outlives the ssh session.
  Takes `flock(LOCK_EX|LOCK_NB)` on `dispatcher.lock`. If held **and** the
  heartbeat is younger than 30 s, exit 0 silently. If held and the heartbeat
  is stale, kill the holder's process group (pgid recorded in the lock file
  body), then take over.
- **A dispatcher started from the package on disk replaces one that was not**,
  fresh heartbeat or not. The lock body records the `config.pkg_commit` its
  holder read at startup; a dispatcher whose own differs SIGTERMs the holder
  and SIGKILLs it if it has not gone in 30 s. Without this, a dispatcher
  outlived every re-bootstrap of its host: it imports its code once, so `gpuc
  host bootstrap` replaced the package on disk, started a dispatcher that saw
  a fresh heartbeat and exited, and left the queue being served by whatever was
  shipped days ago.
  - *Different*, not newer: a commit id carries no ordering, and the code on
    disk is the code that should be running either way. A holder that recorded
    no commit counts as different -- that is every build from before the field.
    A host with no commit recorded at all replaces nobody: nothing to compare.
  - The SIGTERM is polite only to a holder that has the handler
    (`_stop_on_sigterm`, which finishes the pass and releases the lock). The
    *first* takeover on any host is against a build without it, which dies
    where it stands; `adopt_orphans` picks its jobs back up, including one
    killed mid-launch (see Adoption at startup).
  - Before escalating to SIGKILL the lock is re-read and the holder's pid and
    start time must be unchanged. Two dispatchers start within seconds on every
    `gpuc submit` (the resync starts one, the enqueue another), so "somebody
    else took over while we waited" is ordinary -- and SIGKILLing *them* would
    take out the dispatcher running the code we wanted.
  - `gpuc status` reports the holder's commit as `dispatcher.pkg_commit` and
    warns when it differs from the package on disk.
- Loop every 2 s: resolve `config.gpus` to UUIDs (see GPU ownership), then
  walk `queue/` in lexical order. A job that fits the free owned cards gets
  UUIDs assigned, its marker removed, state `running`, and a runner spawned in
  its own session; a job that asked to (`use_shared`) makes up its shortfall
  from the shared cards that are idle right now (see Shared GPUs).
- **The queue is taken in order** (`launch_ready`): a job that does not fit
  holds the cards it is waiting for, owned and borrowed alike, and nothing
  behind it may take them. Only a job the host could supply *whole* from what
  it owns and can currently see holds; a job that needs a card the host cannot
  see (dropped off nvidia-smi, or shared and in use) is walked past instead,
  and is not failed -- `_capacity_failure` fails a job that asks for no GPU at
  all, or for more than the configured host has, shared cards included. Why
  the obvious rule (dispatch whatever fits) is wrong is
  [usage.md](usage.md#priority-is-not-advisory).
- **A job is submitted when its queue marker exists, and not before.**
  `enqueue` writes the spec, then the state, then the marker, so an interrupted
  `gpuc submit` leaves a job dir this host was never asked to run. It is not
  completed and not dispatched: the client owns everything up to the commit.
  After an hour (`cleanup.INCOMING_STALE_S`, the same horizon `stale_incoming`
  uses, and for the same reason -- a submit that is merely slow must not be
  mistaken for one that died) the dispatcher records it `failed:
  incomplete-submit`, so it neither runs nor sits `queued` for ever, and the
  ordinary retention horizons take the dir afterwards. A job `requeue_preempted`
  is putting back is in the same shape mid-move and is excluded: its preempt
  marker says it is coming back, and `_finish_interrupted_requeue` completes it.
- **`launch_ready` never dispatches on a marker alone** (`_still_queued`): the
  marker says a job was submitted, the state says what has become of it since,
  and only a state of `queued` starts a runner. A marker that disagrees is the
  stale half and is dropped. This is what makes a leftover marker harmless --
  from a `leave_queue` interrupted between its two writes, or a purge that did
  not reach `remove_job_dir`'s marker cleanup -- instead of a second runner in
  the workdir the first one is using.
- **Writes that take a job out of the queue put the state first**
  (`queue.leave_queue`, and `queue.cancel` the same way). The order is the
  design, not a preference: interrupted this way the job is `running` with a
  stale marker, which the rule above already handles; interrupted the other way
  it is `queued` with no marker, which is *exactly* what an uncommitted submit
  looks like. One shape, two opposite right answers, and nothing able to tell
  them apart. Keeping the dispatcher's own interruptions out of that shape is
  what lets an uncommitted submit be recognised at all.
  - The general rule this is an instance of: **do not create an intermediate
    state that is ambiguous with one another actor produces.** Where a record
    cannot be made atomic with the thing it describes -- `state.runner_pid`
    against an actual process -- ask the source instead of guessing from the
    record (below).
- **Adoption at startup** (`adopt_orphans`): every job whose `state.json` says
  `running` is either taken over or failed `runner-died`, and the GPUs of a
  failed one go straight back in the free pool -- so anything it left behind is
  killed first (its `cgroup_unit`, then its `pgid`). "Still running" is the
  recorded `runner_pid` *plus* the boot id and start time recorded with it: a
  bare pid means nothing across a reboot and little after a rollover.
  - A job that names no live runner is **not** judged on that alone.
    `launch_ready` writes `running` before there is a process to name and the
    pid only after the spawn, so a dispatcher killed in that window -- every
    first takeover by a newer build, and both SIGKILL paths -- leaves a live
    runner nothing points at. Failing it would lose the job *and* free the
    cards underneath a process still training on them, with no pgid recorded to
    kill. So /proc is walked once for every live `gpuc.host run <id>`
    (`runner.live_runner_pids`) and those runners are adopted. The same
    lookup answers the preempted-and-still-syncing case below, where the cost
    of getting it wrong is attempt 2 starting in the workdir attempt 1 is
    uploading from.
  - What is found is deliberately *not* written back to `state.json`: the
    runner records its own `runner_pid` moments later, and a read-modify-write
    from the dispatcher would race the one `_resolve_assigned` makes in
    between, whose resolved UUIDs would be the loss.
- Cancel: `queue.cancel(jobid)` writes `jobs/<id>/cancel`. The **runner** owns
  the kill (see Runner); the dispatcher escalates only once `state.json`
  publishes a `cgroup_unit`, or a pgid that is not the runner's own -- during
  the launch window the runner *is* the only member of its group. Queued jobs
  are cancelled by removing the marker and setting state.
- Stop with a reason: `queue.request_kill(jobid, reason)` writes
  `jobs/<id>/kill`; the runner kills the job the same way and ends it
  `failed: <reason>` after a final sync. Preempt uses this; cancel stays its
  own marker, because a stop the host decided on is not a cancellation anyone
  asked for.
- Preempt: `queue.preempt(jobid[, prio])` writes `jobs/<id>/preempt` and then
  a `kill` marker with reason `preempted`, for a *running* job only, and only
  when something else could run instead (`refuse_if_nothing_else_can_run`).
  The marker is written first and taken back off if the kill write fails. The
  runner keeps `workdir/` and `secrets/<jobid>.env` and does not re-take the
  outputs baseline, reading the marker once at the top of `_finalize`
  (afterwards a dispatcher may already have consumed it). The dispatcher
  re-queues in `reap`, and in `adopt_orphans` for one whose dispatcher died
  first; `queue.requeue_preempted` clears `kill`, writes a fresh `queued`
  state at `attempt+1`, writes the queue marker, and removes `preempt`
  **last**, so an interruption is completed as the same attempt. It does not
  go back if the attempt ended for a reason of its own (`queue.STOPPED_BY_US`),
  was cancelled while stopping, has no workdir, or the host is going away.
- Automatic preemption (`preempt_for_waiting`, after `launch_ready`): for each
  queued job that does not fit, stop the set of running `auto_preempt` jobs
  that together cover the *whole* gap (`enough_to_start`), least important
  first and most recently started among equals, and only at a strictly higher
  priority number than the waiting job. Nothing runs on a host that is
  `_going_away`. The stop is `queue.preempt`, so it is the ordinary preempt
  path; cards held by a job that is already stopping count as available, and
  once one stop in a set fails the rest are left alone. Nothing counts how
  often a job has given way.
- Isolation: at startup the dispatcher probes `systemd-run --user --scope
  --collect --quiet -- true` once and hands the answer to every runner it spawns
  as `GPUC_ISOLATION`. See Process isolation.
- Reorder: `queue.reorder(jobid, prio)` renames the marker *and*
  `jobs.update_spec(jobid, priority=N)`, because the marker is deleted at
  dispatch and the spec is the only copy that can tell `gpuc status` what
  priority a running job was dispatched at. The control side re-mirrors the
  spec, for the reason `estimate` does.
- Estimate: `jobs.update_spec(jobid, estimated_runtime_min=N)` rewrites
  `spec.json` (raw JSON, so keys another build wrote survive). Queued or
  running, since a running job's runner re-reads the spec. The control side
  re-mirrors the spec, because `requeue` submits what S3 holds; a mirror it
  cannot write is a warning, not a failure.
- Idle terminate (only when `config.provider` is set): no running jobs and an
  empty queue for `idle_minutes` -> write `draining`, retry unconfirmed
  outputs, mirror every job's state and log to `s3_prefix`, then
  `terminate.self_terminate()`. Only a failed *terminate* stops the shutdown:
  remove `draining`, log loudly, keep dispatching, retry every 10 minutes. A
  failed final sync is logged and the host terminates anyway.
- Exit when the queue is empty, nothing is running, and the host is not
  ephemeral. Ephemeral hosts keep the dispatcher alive until terminate.

## Runner (one process per job)

1. Resolve the assignment against `nvidia-smi --query-gpu=index,uuid` --
   indices and UUIDs both, since either form may be recorded -- and fail the
   job (`gpu-assert`) if it is empty or an entry names no card that is here.
   Export `CUDA_VISIBLE_DEVICES=<resolved UUIDs comma-joined>`, the spec
   `env`, and the secrets file.
1b. Snapshot every declared `outputs:` path into `outputs_baseline.json`
   (relative path, size, mtime) -- a checkout routinely ships committed files
   where the outputs go. Before `setup`, because a setup step writing there
   *is* this job's doing. Every upload, periodic and final, excludes files that
   still match the baseline, and a path holding *only* baseline files counts as
   no output at all (`failed: no-outputs`). Above `baseline.MAX_TRACKED` files
   the exclusion is dropped with a loud warning, because the exclude list would
   no longer fit on a command line.
2. `phase=setup`: run `spec.setup` in `workdir` with `bash -eo pipefail`.
3. `phase=preflight`: GPU preflight **inside the job's environment**. A named
   phase, not a step of `setup`, so `gpuc status` can tell "still installing
   torch" from "proving the card works". Run
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
   Sample the assigned GPUs' utilization every 30 s into `util_recent`, for
   `gpuc status` to show; nothing acts on it, since once the GPU check has
   passed a job that leaves its cards idle is the job's business. Enforce
   `max_runtime_min`: SIGTERM the process group, then SIGKILL after 15 s,
   status `failed: timeout`. In the same loop, run `spec.progress_command`
   every `progress_interval_s` and record what it says; see Job length
   estimates.
6. Capture the exit code **before** any cleanup. Stop the sync loop and run
   one final sync; a failed final sync makes a succeeded job `failed: sync`,
   and an output path that was never written makes it `failed: no-outputs`.
   Write final state. Exit code of the runner = job exit code.
7. Apply `spec.cleanup` to `workdir/` -- after the final sync and the final
   state write, never before: `outputs:` paths resolve *inside* the workdir, so
   any earlier removal would delete the run's results on the way past. Record
   `workdir_removed` and `workdir_bytes` (see Retention and purge). A removal
   that fails is logged and nothing more: the job's outcome is already decided.
   Then upload state and log; that last upload records
   `meta_synced_at`/`meta_synced_to` and puts `state.json` up once more, so the
   mirror includes the record of itself. `outputs_synced_at` is written in the
   final state write when the final output upload succeeded.

## Job length estimates

Nothing infers how long a job will take. Two optional, purely informational
spec fields answer "queue behind this, or pay for another host?", and neither
may ever change a job's outcome. What they mean to a submitter, and how
`status` renders them, is [usage.md](usage.md#job-length-estimates); this is
the mechanics.

- `estimated_runtime_min` is published by the runner as `eta` at the top of
  *every* phase, not just `main`: a job twenty minutes into a `uv sync` is the
  one somebody most wants an end time for. The monitor loop re-reads the spec
  every `SPEC_REFRESH_S`, which is how `gpuc estimate` reaches a running job,
  and the same re-read picks up a `progress_command` and `progress_interval_s`
  added mid-run. Those three fields are the whole of it: they describe what
  the job *reports*, while the command, env and outputs are what it ran with.
- `progress_command` runs in `workdir/` with the job's environment, during
  `main` only, every `progress_interval_s`; `progress.parse` accepts exactly a
  fraction with a decimal point or a percentage with a `%`. Above 0% the
  runner replaces `eta` with `now + elapsed_main * (100 - pct) / pct`; at 0%
  the submitter's estimate stands.
- The poll is synchronous, in the same loop that watches for a cancel and
  `max_runtime_min`, with a 10 s timeout and a `killpg` of the whole
  session behind it: a wedged progress command delays a kill by at most 10 s
  of the 15 s the runner gets before the dispatcher escalates, which is why the
  timeout is fixed rather than a spec field. Output goes to a temp file, not a
  pipe (a grandchild that `setsid`s out of the session would hold a pipe open
  and block the reap for ever), and only the last `MAX_OUTPUT_BYTES` is read.
  A thread instead would add a third concurrent writer to `state.json`'s
  read-modify-write; bounded latency is the better trade.
- A failed, timed-out or unparseable poll writes `progress_error`, logged the
  first time each distinct message appears. `eta` is cleared when the job
  ends; `progress_pct` is not.
- `status.queue_start_estimates` projects a queued job's *start* by replaying
  the dispatcher's own rule: cards come free at the eta of whatever holds
  them, and the queue is taken in order, with the same two exemptions as
  `launch_ready`. A card held by a job that published no eta is not
  schedulable, so a job whose turn depends on it is reported as unknown; a
  draining host projects nothing.

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

## Workdir cleanup, retention and purge

`workdir/` is the only part of a job dir gpuc deletes: it is recreatable
(`gpuc requeue` re-syncs it) and usually the largest. `spec.json`,
`state.json` and `log.txt` always stay. The `cleanup:` policy, the two
automatic horizons, the purge preconditions and every refusal are
[usage.md](usage.md#cleanup-and-retention); the contract behind them:

- No policy touches a job that is not finished, and the runner applies its
  policy after the final sync and state write. A preempted job keeps its
  workdir whatever `cleanup:` says: the next attempt re-runs in it.
- `python -m gpuc.host clean (--all-finished | --older-than DAYS | --only IDS)
  [--sweep-only IDS] [--dry-run]` and `purge [--older-than DAYS] [--only IDS]
  [--sweep-only IDS] [--dry-run] [--force]` print JSON and fail closed: a
  running or queued job, an unreadable `state.json`, and (under an age gate) a
  job with no parseable `ended_at` are skipped. `--only` *replaces* the age
  gate; a named id no job dir matches removes nothing at all and exits 1,
  which is why the control side reads these two with `host_json(check=False)`
  -- the document is the report of the failure. `--purge` also runs the
  workdir sweep; `gpuc clean --only` sends the ids as both `--only` and
  `--sweep-only`, and the two stay separate because `--verify` needs them to
  differ (it purges what the mirror confirmed and sweeps what was named).
- The host cannot consult the mirror (it may hold no credentials of ours), so
  `meta_synced_at` in the job's own `state.json` is the purge's authority.
  `--verify` on the control side is the exception: it HEADs the mirrored
  `log.txt` under `meta_synced_to` before deleting anything.
- The dispatcher runs `cleanup.clean(automatic=True)` on both horizons once at
  startup and then at most once an hour, purge first; the automatic form adds
  two refusals a delete nobody typed needs (`cleanup: never`, outputs not
  confirmed elsewhere). `workdir_days` defaults to
  `cleanup.DEFAULT_WORKDIR_DAYS` only for a host being configured for the
  first time (`connect_host`), never on the `HostConfig` field: a config that
  predates the key must not start deleting because somebody shipped it a
  newer package.
- An ephemeral host's drain retries unconfirmed outputs
  (`OUTPUT_RETRY_ATTEMPTS`, a minute apart) with the job's own secrets file or
  the host env, then records `outputs_lost` and terminates anyway.
- `workdir_bytes` is measured **once**, by the runner as the job ends
  (`reclaimable_bytes`), and read back by `status` rather than walked per
  call, which cost four seconds on a host holding sixty finished venvs. A job
  with no figure (an older build, a runner that died) is walked by the first
  `status` that finds its workdir, within `cleanup.MEASURING_BUDGET_S` per
  call, and reports `null` past the budget. A workdir that is gone is zero
  without a walk. `gpuc clean` measures afresh, since it is about to delete
  what it quotes.
- Every byte figure is what deleting the tree gives the filesystem back, not
  what `du` says: uv hardlinks or reflinks a venv out of its cache, so most of
  a torch venv stays. `reclaimable_bytes` counts both (`st_nlink`, and
  `FIEMAP_EXTENT_SHARED` per file), fails towards "all yours" per file, and
  its docstring lists what the total can still miss. `cleanup.dir_size` is the
  `du` twin, used only for the uv cache's own size in `health`.

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
- Anything that talks to RunPod (`--runpod`, `pods`, `host add --pod`) checks
  `RUNPOD_API_KEY` first and exits 1 with a single line if it is unset, before
  mirroring a spec or picking a host.
- A host name is looked up locally; `logs`, `cancel`, `preempt`, `reorder`,
  `estimate` and `requeue` fall back to the job index and then to asking each host, and an id nothing
  knows is exit 4, never a guess.
- Nothing runs in the background on this side except, if installed, the web
  dashboard's service.

Local state: `~/.local/share/gpu-coordinator/` with `hosts.json`, `jobs/`
(the local job index), `known_hosts` plus `known_hosts.d/<pod>`, and
`state.lock`, which serialises every registry read-modify-write across
concurrent sessions. A `desired/` directory or `watch.json` left there by a
build that had a client-side reaper is ignored. `Settings`
(`~/.config/gpu-coordinator/config.toml`) is all optional and every key is in
[setup.md](setup.md#settings); `dead_dispatcher_minutes`, the reaper's key,
is ignored like any other unknown one.

## Shared state is read tolerantly, always

Two sessions of one user share `~/.local/share/gpu-coordinator/hosts.json`, and
a host's `config.json` outlives the build that wrote it. So every reader of a
file the two halves share obeys the same two rules, on both sides:

- an unknown key is ignored (a newer writer may add fields);
- an explicit `null` for a field that is **not** declared optional is dropped,
  so the field's default applies. A `null` for a field that *is* optional is a
  real value and round-trips unchanged: `retention_days: null` is "never
  auto-purge", `s3_prefix: null` is "no mirror".

Control side that means `extra="ignore"`, a default on every field, and a
`model_validator(mode="before")` that consults the annotation (`HostEntry`,
`HostCache`, `Registry`, `Settings`, `IndexEntry`, `Offer`). The
host config a registry entry caches is kept **verbatim** on top of that, so a
key some newer build wrote survives a round trip through this one. Host side, with
no pydantic, the same rules are spelled out in `jobs.from_dict` for
`HostConfig`, `JobSpec`, `JobState` and the dispatcher's lock body: never
`float(None)`, never a `KeyError`, an unusable value means the default.

`hosts.json` and `config.json` both carry `schema_version` (1); readers accept
it missing. `tests/fixtures/schema/` holds today's shape of each file plus
older, newer and pre-split variants, and every one of them must parse.

A host entry that still does not validate is **skipped, not fatal**: `gpuc`
warns, works with the rest, and writes that entry back untouched on the next
registry write. Only a `hosts.json` that cannot be parsed at all stops
anything (exit 3, a `.bak` kept).

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
--others --exclude-standard` (what that includes is
[usage.md](usage.md#what-gets-synced-to-the-host)); `uncommitted.patch` is
`git diff HEAD -- .` against a *copy* of the index with `git add -N` applied,
so it carries untracked files and never touches the user's staging. `put_file`
writes 0600 content via stdin (`cat > path && chmod 600 path`); secrets never
touch argv.

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
`retention_days`, `provider`, `pkg_commit` -- lives in
`config.json` on the host and nowhere else. One box driven from a desktop and a
laptop therefore has one configuration, not two, and nothing about the machine
that bootstrapped it first matters afterwards.

- `gpuc host add` is a **connect** (`connect_host`): probe, read
  `config.json`, adopt it if it is there under the name the host calls itself,
  else write the initial one. Flags are per-field overrides written through
  to the host; a `--gpus` that overlaps the existing set without matching it
  is refused, because that one difference hands one card to two jobs.
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

Every listing names a card `[index] name vram`; `gpuc host list` adds the
UUID (it is where UUIDs are copied from), `gpuc status` adds free/busy and
names each running job's cards on the job's own line, and `gpuc host probe`
lists the owned cards only unless `--all-gpus`. The probe matches an owned
index against the numbering `nvidia-smi` gave *in that same probe*, records
`gpu_info` for every card either way (so a later `--gpus 5` resolves offline),
and notes the two assignments `check_gpu_uuids` would later refuse to
bootstrap: an entry no card answered to, and two entries folding onto one card.

## Shared GPUs: cards we borrow rather than own

`config.gpus` is what the host owns and `config.shared_gpus` what it may
*borrow*, spelled the same way (index or UUID) and resolved the same way each
pass. An owned card is handed out whenever it is free; a shared card is only
ever borrowed, for one job, and only while nobody else is on it. What that
means for a submitter is [usage.md](usage.md#shared-gpus).

- Two gates: the job asked (`use_shared`, fixed at submit, so the dispatcher
  can fail a job that can never fit rather than leave it waiting), and
  `nvidia-smi --query-gpu=memory.used,utilization.gpu` reports 0 MiB *and* 0%
  for the card right now. Every way of not knowing counts as in use: this
  decides whether to run on somebody else's GPU.
- Owned cards first, always; a job borrows only its shortfall, so a shared
  card is held for the shortest time that runs the job.
- The sample is taken once per dispatch pass and only when a job actually
  needs to borrow, and every job in that pass is judged against that one
  reading, which is also what stops two of them being handed one card.
- No yield: a job that got a borrowed card keeps it until it ends. A collision
  with the card's owner is not detected; `gpuc preempt` is the way out.
- No per-host floor on which jobs may borrow. A version of this had
  `shared_min_priority`, and it was less than it looked (borrowing is not a
  reservation, so a floor never protects an important job from a trivial one)
  and more dangerous (a per-host floor moves under queued jobs, and the
  dispatcher's "this can never run here" is a deletion -- `gpuc reorder
  --priority 99` would have ended the job it was asked to move).
- A card in both lists is refused by `gpuc host add|set` and by the host's own
  `gpu_uuids` health check; should one reach a dispatcher, owning wins.

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
fallback) as a fact and draws no conclusion from it. An ephemeral queue is what
gpuc is built around: a pod gpuc rents gets no network volume, so it has nowhere
durable to put one, and an overlay `$HOME` is not a misconfiguration to warn
about. `--persistent-root` is there for the host where somebody decides the
trade is worth it. The health check's disk floor is measured on
`paths.home()`, so it is `R`'s volume when a root is set. The operator runbook
for a host that came back empty is in setup.md.

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
- Nothing caps how many pods an account runs or what they cost per hour;
  spending limits across rentals are a non-goal. `gpuc pods` is how a person
  sees what is billing.

## Provisioning flow (`gpuc submit --runpod`)

1. Write the spec to S3 (`s3://bucket/gpuc/specs/<jobid>.json`) first.
2. Unless `--no-reuse`: pick an existing registered pod whose own config
   records an offer that still satisfies the constraints, which owns enough
   cards, whose pod the provider reports RUNNING, whose dispatcher heartbeat
   is fresh, and which is not draining; enqueue there. A registered pod the
   provider no longer has is forgotten rather than dialled.
3. Else for each offer in order: `create`; poll `get` until RUNNING **and**
   `ssh.direct` present; poll SSH until a trivial command succeeds; write
   the pod its config,
   whose `provider` block carries the offer and `created_at` (`rented.py`:
   the pod is its own record, and this machine keeps none); run bootstrap
   (which runs host health and starts the dispatcher); deliver the RunPod
   key as `~/.gpuc/secrets/runpod` (0600) for self-terminate. Enqueue. On
   any broken-host signature in `logs`, or the 15-minute ceiling, or a
   health failure -- or a Ctrl-C, or a bug: `terminate`, wait for
   TERMINATED, try the next offer. A terminate that failed is reported
   loudly and leaves the registry entry in place, so `gpuc status` and
   `gpuc pods` keep showing the pod; nothing retries it.
4. The pod-scoped key delivered as `~/.gpuc/secrets/runpod` does terminate
   its own pod; `tests/test_runpod_e2e.py` proves it on every opt-in run.
   From bootstrap on, that is the only thing that ends the pod.

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
parallel (`actions.gather_all`, also what the text `gpuc status` uses), so one
wedged host costs its own timeout, not the sum.

`gpuc web serve --install` writes `gpuc-web.service` to `~/.config/systemd/user`
through `control/systemd.py`: the unit directory, an absolute `gpuc` for
`ExecStart` (quoted the way systemd reads it) and writing without enabling.
The unit pins the config and state dirs, reads `RUNPOD_API_KEY` from
`config_dir()/env` if that file exists, and restarts on failure under a start
limit, so a service with no password fails after five tries rather than
looping for ever.

Anything the dashboard gains lands in `actions` first and the dashboard calls
it.

## Status output

What `status` prints, and every flag, is usage.md. The invariants:

- An ephemeral host whose pod the provider reports missing or TERMINATED is
  `POD GONE`: no ssh is attempted, and the line says to run `gpuc host remove
  <name>` rather than printing a connection error.
- A finished job that produced `outputs:` which never reached S3/HF is flagged
  (`outputs not uploaded`, or `OUTPUTS LOST` once a drain has given up), because
  those are the jobs a purge -- or a pod going away -- would take with them. One
  that declared outputs and never wrote them is not: there is nothing there.
- A pod's `provider_util` is the provider's reading for the whole pod; a job's
  `util` is the host's own nvidia-smi sampler over that job's cards. They are
  labelled separately and never merged.
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

## Testing rules

- `./check.sh` is the whole suite, and what CI runs. A bare `pytest` is safe
  too: `addopts` in `pyproject.toml` excludes the `runpod` marker, so the tests
  that rent hardware never run by accident -- opt in with `pytest -m runpod`.
  Nothing else is excluded by default.
- Unit tests run without a GPU (mock `nvidia-smi` output, temp `~/.gpuc`).
- Local GPU integration tests (`gpu`) **run by default**: they use tiny tensors
  (`torch.zeros(8)`), never more than ~100 MB VRAM because other people's jobs
  share the card, and they skip themselves on a machine whose `nvidia-smi` does
  not report `tests.conftest.LOCAL_GPU_UUID` -- which is every CI runner. A GPU
  queue whose GPU tests are the ones nobody runs is how they rot; two of them
  had, asserting on a `gpuc status` line that had since gained a job name.
- RunPod integration: A40 only, `--max-price 0.60`, a job whose command
  is under two minutes, `--idle-min 2`, and the test asserts teardown via
  `list()` and prints the final `GET /billing/pods`
  for the pod. A pod whose name lacks the `runpod_pod_prefix` belongs to
  someone else: read it in `list()`, never act on it. Every test that creates
  a pod has a `finally` that terminates it.

## Code conventions

uv, ruff, pyright strict-ish, type annotations everywhere, pydantic on the
control side only. No comments that restate the next line; explain *why*
where it is non-obvious. Prefer good names over docstrings. Errors carry
the command, the host, and the last lines of output.
