# gpu-coordinator architecture

The contract the code keeps: what lives where, who owns what, and the rules
each half holds to. What the project is for, and what it deliberately does not
do, is [SPEC.md](SPEC.md). It is not the CLI reference either -- flags and
defaults are `gpuc --help` and each subcommand's `--help`, what they mean
together is [usage.md](usage.md), and installing and registering hosts is
[setup.md](setup.md). Where a section names a function, the *why* lives in
that function's docstring and is not repeated here.

## Package layout

One uv project, Python >= 3.11, package `gpuc`, CLI entry point `gpuc`.

```
gpuc/
  _version.py    # __version__ and user_agent(); STDLIB ONLY, imported by both halves
  host/          # runs ON hosts. STDLIB ONLY: `python -m gpuc.host <cmd>` with a bare interpreter
    __main__.py    # the on-host CLI the control side drives over ssh
    paths.py       # the ~/.gpuc layout
    jobs.py        # job ids, HostConfig/JobSpec/JobState, the per-job lock, tolerant readers, atomic writes
    queue.py       # accept a job, list the queue, claim, cancel, preempt, the next attempt of a preempted job
    plan.py        # the dispatch rule, pure: what a pass does with each queued job, and when each will start
    dispatcher.py  # lock+heartbeat, act on the plan, launch runners, escalate stops, idle terminate
    runner.py      # one job: env, CUDA_VISIBLE_DEVICES, preflights, wall-clock limit, sync, exit code
    scope.py       # systemd --user scope probe/wrap/stop; the cgroup kill path
    procs.py       # process facts, and JobProcesses: the one object every stop goes through
    destinations.py # where files go: one Destination per store (S3, Hugging Face); upload, put_file, preflight
    preflight.py   # sync preflight: prove every destination can be written before the job runs
    baseline.py    # what was already under `outputs:` before the job started
    cleanup.py     # `cleanup:` policy, workdir sizing, the `clean`/`purge` sweeps
    gpus.py        # the one nvidia-smi table parser and the one index/UUID resolver (`resolve`); utilization sampling
    progress.py    # the optional `progress_command`: run it, read a percentage off it
    sync.py        # periodic upload loop, recording each destination's result in the job's state
    health.py      # host preflight: driver, owned GPUs, disk, uv cache placement, the one network throughput test
    storage.py     # the data directory and the HF cache: `host clean --data` and `--hf-cache`
    fetch.py       # which workdir files `gpuc fetch` copies
    terminate.py   # self-terminate via provider API (urllib), key from ~/.gpuc/secrets
  control/       # runs on the local machine; may use third-party deps
    cli.py         # argparse and the text output of every `gpuc` command; `main` emits each answer once
    actions.py     # one function per command returning its Answer (document + text + what failed); the CLI and the web call these
    exits.py       # the exit codes, and the one table that maps them onto HTTP statuses
    hosts.py       # the actions of `host add`, `host set` and `host bootstrap`
    submitting.py  # the actions of `submit` and `requeue`: rent or look up the host, re-ship, enqueue
    status.py      # gather a host's status, render it, project queue start times
    wait.py        # the poll `gpuc wait` and `gpuc logs -f` block on until a job ends
    submit.py      # validate a spec, sync the workdir, deliver secrets, enqueue
    provision.py   # `--runpod`: offers, create, wait for ssh, bootstrap, reuse
    teardown.py    # `host terminate`: end a rental on purpose, and forget it here
    rented.py      # a pod is its own record: the `provider` block in its config.json
    pods.py        # `gpuc pods`
    config.py      # ~/.local/share/gpu-coordinator/ layout, Settings, the hosts registry
    tolerant.py    # the pydantic base every shared-state model reads through
    connect.py     # `host add` / `host set`: read or write the host's own config.json
    bootstrap.py   # install uv + this package on a host, run host health, start the dispatcher; `ensure_build` ships
    probe.py       # `host probe`: what a host has, before bootstrap
    remote.py      # HostSession (opened on a fresh read of config.json) and `ask`: the one way to ask a host
    transport.py   # LocalTransport / SshTransport: run, rsync, pull, put_file(0600), tail
    ssh.py         # `gpuc ssh`: the interactive form of the transport
    clean.py       # `gpuc clean` / `gpuc host clean` over the transport
    fetch.py       # `gpuc fetch`: ask each host for its jobs' files, pull them here
    s3index.py     # the S3 mirror of specs and the job index, and the local index
    gpuinfo.py     # per-GPU name/VRAM for the registry and every listing, over the host's own table and resolver
    version.py     # this build's commit, and comparing it with a host's
    jsonout.py     # the `--json` rules
    skill.py       # `gpuc skill`: the agent guide, from the wheel or the checkout
    systemd.py     # what `web serve --install` needs of systemd: unit dir, ExecStart, write
    web/           # `gpuc web`: stdlib http.server, bcrypt login, a static page over the same documents
    providers/
      base.py      # Provider interface: offers(constraints), create, get, logs, terminate, list
      runpod.py    # v2 REST implementation
snakemake_executor_plugin_gpuc/  # `snakemake --executor gpuc`; a second top-level package in the wheel
skills/gpuc/SKILL.md  # the agent guide; force-included in the wheel as gpuc/SKILL.md
tests/
```

`gpuc.host` must be importable and runnable with a bare interpreter: the
bootstrap rsyncs the package to the host and runs it with the uv-managed
Python, with no `uv sync` needed for the queue to work. Everything the
dispatcher spawns gets `$HOME/.local/bin` and `$HOME/.cargo/bin` prepended to
`PATH` when they exist (`dispatcher._child_env`). Upload helpers shell out to
binaries the bootstrap installs into `$HOME` (`aws` CLI v2 bundle, `hf` from
`uv tool install huggingface_hub`), and a missing binary fails the *job's* sync
step, never the queue.

## User agent

One string, `gpuc/_version.py:user_agent()`, on every outbound request from
either half: the RunPod API, a pod's self-terminate, the health check's
download, bootstrap's `curl`, every control-side boto3 client, and
`HF_HUB_USER_AGENT_ORIGIN` in each job's environment. `gpuc._version` must
stay stdlib-only and bootstrap must ship it. The `aws` CLI on a host cannot
have its User-Agent overridden.

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
                     #  "env": {"HF_HOME": ...}}      # host-wide; see Persistent root
secrets/<name>       # 0600 files delivered over SSH after boot. Never in argv, never in pod env.
data/                # GPUC_DATA_DIR unless the host's env names another; see The data directory
incoming/<jobid>/    # a job `gpuc submit` is still building: the rsynced workdir, then the spec.
                     # `enqueue` renames the whole dir into jobs/, which is the acceptance
jobs/<jobid>/
  .lock              # the flock every read-modify-write of state.json takes
  spec.json          # the submitted JobSpec, written once by `enqueue` and never again
  state.json         # {"status": queued|running|succeeded|failed|cancelled,
                     #  "intent": null | "cancel" | "preempt",   # what somebody asked of a running job;
                     #                                        # cleared by the write that ends it
                     #  "attempt": n,                          # launches of this id: 1, +1 per preempt
                     #  "priority": p,                         # the live priority; the queue is
                     #                                        # every `queued` state in (priority, id) order
                     #  "estimated_runtime_min": null | m,     # the live estimate, as `gpuc estimate` left it
                     #  "reason": str|null, "problems": [str, ...],  # what ended it, and what else went wrong
                     #  "exit_code": int|null, "gpus": [...],   # UUIDs, from the runner's claim
                     #  "started_at", "ended_at",
                     #  "phase": setup|preflight|main|sync,    # `sync` lasts until the write that
                     #                                        # ends the attempt
                     #  "pgid": int|null,                      # the *job's* group, published by the
                     #                                        # runner when it spawns a phase and
                     #                                        # cleared when the phase ends
                     #  "isolation": "cgroup"|"pgid", "cgroup_unit": str|null,
                     #  "runner_pid": int|null, "runner_boot_id": str|null,
                     #  "runner_starttime": str|null,          # the runner, from its own claim
                     #  "util_recent": [float|null, ...],
                     #  "progress_pct": float|null, "progress_error": str|null, "eta": str|null,
                     #  "uploads": [{"to": uri, "output": path|null, "ok_at": str|null, "error": str|null}],
                     #                              # one record per destination, `output` null for
                     #                              # the host's mirror of log.txt and state.json;
                     #                              # the whole account of what is safely elsewhere
                     #  "workdir_bytes": int|null,  # what removing workdir/ would free;
                     #                              # measured once, as the job ended (or by
                     #                              # the first status to find it missing)
                     #  "outputs_lost": bool,       # a drain retried the outputs and gave up
                     #  "ran": bool,                # has `main` started, in any attempt; false
                     #                              # from enqueue, set as `main` begins, never
                     #                              # cleared
                     #  "checkout_removed_at": str|null}  # the checkout went from a workdir/
                     #                              # that still holds kept outputs
                     # util_recent is the last 40 main-phase samples; null means nvidia-smi
                     # failed and must not be read as 0%. eta is null unless the job is running.
                     # progress_pct survives the job.
  outputs_baseline.json # per `outputs:` path, the {relpath: [size, mtime_ns]} the
                     # checkout arrived with; those files are never uploaded as this
                     # job's results and never satisfy `outputs:`
  workdir/           # rsynced code (git-tracked + untracked, .gitignore obeyed); removed per `cleanup:`,
                     # except the job's kept outputs, which stay where it wrote them
  log.txt            # combined stdout/stderr of setup + command, line-buffered
  outputs/           # default output root; JobSpec.outputs paths are relative to workdir
dispatcher.lock      # fd flock held by the running dispatcher
dispatcher.heartbeat # mtime touched every 5 s by the dispatcher
dispatcher.log
draining             # present while the host is shutting itself down
```

A job's `state.json` is the one record of what it is doing and what has been
asked of it; there is no queue file and no marker file, and the queue is
derived from it. Every change to it is one atomic replace (write temp in the
same dir, `os.replace`) under the job's lock. A change of ownership is a
compare-and-set on `status` (`jobs.transition`): the runner claiming the job
it was started for, a cancel ending a queued one, and the write that ends an
attempt. **The runner owns every transition of its job** from the claim
(`queued` -> `running`, naming itself) to the write that ends the attempt (a
terminal status through `jobs.finish`, or `queued` again at the next attempt
for a preempt), so a `running` state always names a runner that existed, and a
job is finished exactly when its runner is gone. Every terminal write goes
through `jobs.finish(job_id, Outcome)`, the dispatcher's own failures
included; it clears the intent, the phase and the processes of the attempt.
An unreadable state is logged and left alone, never written over with
defaults.

## JobSpec (JSON; `gpuc submit` accepts YAML or JSON and normalizes)

```
{
  "name": "lego-s4",                    # human label, not an identifier
  "command": "uv run python -m experiments.lego.train --k-max 6",
  "setup": "uv sync --frozen",          # optional; runs before command, phase=setup
  "python": "uv run --no-sync python",  # how the GPU check runs Python in the job's own env
  "gpus": 1,                            # 0 runs with no card
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
  "priority": 50,                       # copied into the state at enqueue; the state's copy is
                                        # the live one (`gpuc reorder`, `gpuc preempt --priority`)
  "max_runtime_min": null,
  "estimated_runtime_min": null,        # the submitter's own guess, measured from the runner's
                                        # start exactly as max_runtime_min is. Informational only;
                                        # copied into the state at enqueue, where `gpuc estimate` edits it
  "progress_command": null,             # run in workdir/ every progress_interval_s of phase main;
                                        # its last line of stdout is a percentage. See Estimates
  "progress_interval_s": 60,
  "auto_preempt": false,                # let the dispatcher stop this job, as often as it
                                        # takes, whenever that starts a queued one ahead
                                        # of it right away; it may take held cards
  "requires": {"cuda_min": "12.8"},     # informs provisioning only
  "cleanup": "on_success",              # on_success | always | never; see Workdir cleanup
  "requeued_from": null                 # the job `gpuc requeue` resubmitted this one from;
                                        # written by the control side, carried by the host
}
```

`gpuc submit` expands `{job_id}` in output destinations and refuses an `s3` or
`hf_path` whose expanded form does not contain the id (no `hf_path` means the
id itself). Output namespaces are unique by construction; there is no other
overwrite guard anywhere. The S3 mirror holds the spec with `{job_id}`
unexpanded, so `gpuc requeue` expands it with the new id.

Job id: `YYYYMMDD-HHMMSS-<6 hex>`, assigned by `gpuc submit`. Dispatch order
is the queue's lexical order in `(priority, job_id)`; two jobs submitted in
the same second tie-break on the suffix.

## Dispatcher (`python -m gpuc.host dispatch`)

Started by every `enqueue` and by bootstrap, in its own session and, under
`cgroup` isolation, its own transient scope (`dispatcher._spawn_host_process`),
as is each runner: under `cgroup` nothing gpuc runs on a host stays in the
cgroup of the process that started it, so stopping that cgroup reaches no
dispatcher, runner or job. Under `pgid` they share that cgroup and its fate.
The rules it holds to:

- **One dispatcher per host**, by `flock` on `dispatcher.lock` plus a heartbeat.
  A holder whose heartbeat is stale (30 s) is killed by the pgid in the lock
  body and taken over.
- **A dispatcher started from the package on disk replaces one that is not.**
  The lock body records the `pkg_commit` its holder started from; a holder
  whose commit *differs* (`_is_another_build`: different, not older; an
  unrecorded commit counts as different) is SIGTERMed, then SIGKILLed at 30 s
  if the lock still names it. A dispatcher whose own commit is unrecorded
  replaces nobody.
- **One dispatch rule** (`plan.plan`), pure, over one pass's inputs: the queue
  in `(priority, job_id)` order, the owned cards free now, the configured
  counts, and one nvidia-smi reading of the shared cards taken only if a job
  needs to borrow. Per job it decides assigned, holds, stepped over or fails,
  and, for a job that holds, which running `auto_preempt` jobs would cover its
  gap; `launch_ready` acts on it every 2 s, and automatic preemption on that
  set for the first job with a gap. Run forward over the running jobs' etas
  (`plan.project`) it is how the host says when each queued job will start.
- **An `auto_preempt` job may take held cards**, launched like any other,
  unless the holder is *covered*: short of nothing once the cards on their
  way back arrive, or short only of what the preemption set frees. Cards on
  their way back are those of a runner with a stop intent, or one whose last
  state is written (queued again, finished) but which has not been reaped.
- **Acceptance is a rename.** `gpuc submit` builds the job dir under
  `incoming/`; the host's `enqueue` writes the spec and initial state there
  and renames the dir into `jobs/`. A dir left under `incoming/` an hour after
  its last change is a submit that died, and is removed.
- **A launch is a spawn; the runner claims the job.** The dispatcher starts
  `python -m gpuc.host run <id> --gpus <uuids> --attempt <n>` and writes
  nothing; the runner's first act is the compare-and-set from `queued` at that
  attempt to `running`. A runner that dies before claiming is failed
  `runner-died` from `queued`.
- **A card is busy if any runner holds it**: the cards of every runner this
  dispatcher spawned or adopted, and the cards every `running` state on disk
  names (`_launch` re-reads the claims before every spawn).
- **Adoption at startup** (`adopt_orphans`): every `running` state whose
  recorded runner (`runner_pid` with the boot id and start time beside it) is
  alive is adopted; otherwise it is failed `runner-died`, its leftovers killed
  (`cgroup_unit`, then `pgid`) before its cards go back in the pool. Nothing
  is inferred from the process table and nothing is written back.
- **A stop is an intent** in the job's state: `cancel` or `preempt`. The
  runner owns the kill and ends the attempt with its last write; the
  dispatcher escalates only once the grace period has passed
  (`escalate_stops`), one rung per grace period: the job's scope and group,
  then the runner, then the runner's group. A runner in phase `sync` gets
  `SYNC_STOP_PATIENCE_S` first. A queued job is cancelled on the spot, with no
  intent. A cancel overrides a preempt.
- **Preempt** (`queue.preempt`) is for a running job that holds cards, and
  only when a queued job that wants cards would be dispatched ahead of it. The runner's last write is a fresh `queued` state at
  `attempt+1` (`queue.next_attempt`), keeping the live priority and estimate,
  the workdir, the secrets file and the outputs baseline; nothing sees the job
  finished in between. An attempt that ended on its own first, or was
  cancelled while stopping, ends that way instead. A draining host refuses
  one.
- **Automatic preemption** (`preempt_for_waiting`, after `launch_ready`): for
  the one queued job the host is stuck on, stop the set of running
  `auto_preempt` jobs that together cover the gap, least important first, and
  only jobs the waiting one is ahead of in `(priority, job_id)`. Nothing is
  stopped on a host that is going away.
- **Reorder** and **estimate** write the job's state and nothing else; the
  spec is never rewritten after enqueue. The control side re-mirrors the spec
  after both; a mirror it cannot write is a warning.
- **Idle terminate** (only with `config.provider` set): no running jobs and an
  empty queue for `idle_minutes` -> `draining`, retry unconfirmed outputs,
  mirror every job's state and log, then `terminate.self_terminate()`. Only a
  failed *terminate* stops the shutdown: remove `draining`, keep dispatching,
  retry every 10 minutes.
- **Exit** when the queue is empty, nothing is running and the host is not
  ephemeral. An ephemeral host keeps its dispatcher until terminate.

## Runner (one process per job)

The order is the contract; each step is in `runner.py`.

0. Claim the job: one compare-and-set from `queued` to `running` carrying the
   assignment, `started_at`, the isolation mode and the runner's own pid, boot
   id and start time. A claim that fails is a job that is no longer ours; the
   runner exits 0 and writes nothing.
1. Verify the assignment (UUIDs) against `nvidia-smi --query-gpu=index,uuid`;
   fail `gpu-assert` if it names a card that is not here, or is empty for a
   job that asked for cards. Export `CUDA_VISIBLE_DEVICES` as the cards'
   nvidia-smi **indices** with `CUDA_DEVICE_ORDER=PCI_BUS_ID`, as UUIDs only if
   the index table could not be read, and empty -- never unset -- for a job
   with no cards. Then the secrets file and the spec `env`; the host's `env` and PATH
   are the runner's own, from the dispatcher.
1b. Snapshot every declared `outputs:` path into `outputs_baseline.json`,
   **before `setup`** and once per job (a later attempt keeps the first
   baseline). Every upload excludes files that still match, and a path holding
   only those is `failed: no-outputs`. Above `baseline.MAX_TRACKED` files the
   exclusion is dropped with a warning.
2. `phase=setup`: `spec.setup` in `workdir` under `bash -eo pipefail`.
3. `phase=preflight`: the GPU check inside the job's environment, the spec's
   `python` over a probe (`is_available()`, a tensor add and `.item()`,
   `device_count()` equal to `gpus`). Failure -> `failed: gpu-preflight`.
   Skipped for a job with no cards, which then needs no torch.
3b. Sync preflight, still before `main`: every `Destination` the job will
   upload to, the host's `jobs/<id>/` mirror included, proves a `.preflight`
   write works with the job's own environment. Failure ->
   `failed: sync-preflight`, with the command and its error in `log.txt`, and
   no final output sync. A job with no outputs on a host with no mirror checks
   nothing.
4. Start the sync loop (background thread): `outputs` every `sync_interval_s`,
   skipping files modified in the last 10 s, plus `log.txt` and `state.json` to
   `s3_prefix/jobs/<id>/`. Every upload is recorded in the state's `uploads`,
   one record per destination, an output path that does not exist included.
   Uploads run with the job's environment, secrets included.
5. `phase=main`: `spec.command`, stdout and stderr appended to `log.txt`, in its
   own scope or process group (see Process isolation). Sample the assigned GPUs
   every 30 s into `util_recent`; nothing acts on it. Enforce `max_runtime_min`
   (SIGTERM the group, SIGKILL at 15 s, `failed: timeout`), and run
   `spec.progress_command` every `progress_interval_s`.
6. Capture the exit code before any cleanup, set `phase=sync`, stop the sync
   loop, clear every output's success record and upload every output once
   more. A failed upload makes a succeeded job `failed: sync`, an output path
   never written makes it `failed: no-outputs`; a job already over for a reason
   of its own keeps that `reason` and lists the failure under `problems`. Only
   for a job whose `main` started (`Outcome.ran`, the state's `ran`): one
   stopped before it skips the final upload and is never `no-outputs`.
7. Apply `spec.cleanup` to `workdir/` through `cleanup.may_delete`, asked with
   the status about to be written, after the final sync and never before. A
   job on its way back to `queued` keeps it. Record `workdir_bytes`.
8. Mirror the log and state and record the mirror's own upload (the record
   with `output: null`).
9. The write that ends the attempt, as the last act: `jobs.finish` with the
   outcome, or `queue.next_attempt` for a preempt. Then the state once more to
   the mirror. Then the secrets file goes, unless the job is `queued` again or
   an ephemeral host still has its outputs pending for the drain. The runner
   exits with the job's code.

The secrets file goes with **every** terminal write, not only the runner's: a
queued job cancelled, a job the dispatcher failed or marked `runner-died`
loses it in the same act (`cleanup.settle_secrets`), with the one drain
exception above.

## Job length estimates

Nothing infers how long a job will take. Two optional spec fields are purely
informational and may never change a job's outcome; what they mean to a
submitter is [usage.md](usage.md#job-length-estimates).

- `estimated_runtime_min` is published as `eta` from the first phase on. The
  monitor loop re-reads it from the job's state every `ESTIMATE_REFRESH_S`,
  which is how `gpuc estimate` reaches a running job; it is the one thing a
  running job re-reads.
- `progress_command` runs in `workdir/` with the job's environment, during
  `main` only; `progress.parse` accepts a fraction with a decimal point or a
  percentage with a `%`, and nothing else. Above 0% the runner replaces `eta`
  with `now + elapsed_main * (100 - pct) / pct`. The poll is synchronous in
  the loop that watches for a cancel, with a fixed `progress.TIMEOUT_S` and a
  `killpg` behind it, and only the last `MAX_OUTPUT_BYTES` of output is read.
- A failed, timed-out or unparseable poll writes `progress_error`, logged the
  first time each distinct message appears. `eta` is cleared when the job
  ends; `progress_pct` is not.
- The **host** projects each queued job's start (`plan.project`, published by
  `python -m gpuc.host status` as `starts_in_s`, with `starts_unknown` saying
  why not) by running the dispatch rule forward over the running jobs' etas,
  and automatic preemption with it: an `auto_preempt` job's cards come back,
  and it is queued again, when the job it yields to can start. A
  card held by a job that published no eta is not schedulable, so a job whose
  turn depends on it is unknown; a draining host projects nothing. The control
  side renders the answer and computes nothing.

## Process isolation (cgroup scope, else process group)

Where a lingering `systemd --user` instance with cgroup delegation exists each
phase runs as:

```
systemd-run --user --scope --collect --quiet -p TimeoutStopSec=15 \
  --unit=gpuc-<jobid>-<phase>.scope -- bash -c 'base64 -d <<<"$1" | bash' _ <b64>
```

Every stop goes through one object, `procs.JobProcesses` (the scope, the job's
own process group, and the runner), built from the job's state: the runner's
kill, the dispatcher's escalation and the adoption of a dead runner's leftovers
all call it. The kill path is `systemctl --user stop <unit>`, with the
process-group kill as the fallback. The job's group is only ever the one the
runner published for the phase now running, cleared with the unit the moment
the phase ends; the runner's own group is never recorded as the job's. Every
phase that exits on its own is followed by that stop if anything is still in
its scope or group, before the unit is cleared, so the dispatcher never frees
a card a phase's leftovers hold (bar the `pgid` hole below).
`isolation` (`cgroup` | `pgid`) and `cgroup_unit` are in `state.json` and
`gpuc status --json`. The mode is decided once per process
(`scope.isolation()`: what `GPUC_ISOLATION` announces, else one probe) and the
dispatcher announces its answer to every child, so dispatcher, runners and
phases agree on what a kill reaches. A user instance without linger stops at
logout and takes its scopes with it, so the probe answers `pgid` there too,
unless the asking process is already under that instance and would die with
it anyway.
Under `pgid` (no lingering user systemd: every RunPod pod, most shared boxes)
a daemonised grandchild escapes: a documented hole, not a fixed one.

## Workdir cleanup, retention and purge

`workdir/` is the only part of a job dir gpuc deletes; `spec.json`,
`state.json` and `log.txt` always stay. The policy, the two horizons and every
refusal are [usage.md](usage.md#cleanup-and-retention); the contract behind
them:

- No policy touches a job that is not finished. The runner applies its policy
  after the final output sync and before the write that ends the attempt,
  asking with the status it is about to write.
- **One predicate decides every delete**: `cleanup.may_delete(job, what,
  evidence)`, where `what` is the workdir or the whole job dir and `evidence`
  is what the caller can vouch for: `automatic` (nobody typed the command),
  `policy` (the job's own `cleanup:` applied by its runner), `force` (a person
  waived the backup preconditions), or `verified` (which jobs' mirrors the
  caller checked itself). A workdir needs the job finished, under the policy
  also what `cleanup:` says of that status, under an automatic sweep also not
  `cleanup: never`, and under either no outputs pending. A job dir needs the
  record mirrored and no outputs pending, unless forced.
- **Removing a workdir removes the checkout** (`cleanup.remove_workdir`): all
  of it, unless the job has kept outputs (`cleanup.kept_outputs`: outputs with
  no destination that hold something the job wrote, judged like
  `outputs_pending`). Then everything but those paths goes, and
  `checkout_removed_at` in the job's state records it, because the directory
  is still there. Every "is there anything to sweep" question asks
  `cleanup.has_checkout`, never whether `workdir/` exists. A kept path named
  through a symlink keeps both the name and what it resolves to inside the
  workdir. A job dir with kept outputs is purged only when forced.
- **One question about outputs**: `cleanup.outputs_pending(job, spec, state)`,
  the reason a job's outputs are only on this host, or None. It is what the
  drain retries, what keeps a job's secrets file for that drain, what `purge`
  and the automatic sweep refuse over, and what `status` flags. Nothing is
  pending for a job with no `outputs:`, whose `main` never started (`ran`),
  with every destination's last upload recorded, with its workdir gone, or
  with nothing ever written under the declared paths; anything unreadable
  counts as content.
- `python -m gpuc.host clean (--all-finished | --older-than DAYS | --only IDS)
  [--dry-run]` and `purge [--older-than DAYS] [--only IDS] [--verified IDS]
  [--dry-run] [--force]` print JSON and **fail closed**: a running or queued
  job, an unreadable `state.json`, and (under an age gate) a job with no
  parseable `ended_at` are skipped. `--only` replaces the age gate, and a named
  id no job dir matches removes nothing and exits 1; the control side reads
  both with `host_json(check=False)`.
- The host cannot consult the mirror, so the mirror's upload record in the
  job's own `state.json` is the purge's authority. `--verify` on the control
  side lists the mirrored logs under the host's prefix and passes the ids as
  `--verified`; a job must then be in the list *and* have the record.
- The dispatcher reclaims with `Evidence(automatic=True)` at startup and then
  at most once an hour: job dirs older than `retention_days` first, then one
  workdir sweep at the shorter of the two horizons that are set.
  `workdir_days` defaults to `cleanup.DEFAULT_WORKDIR_DAYS` only in the first
  config `config.first_config` builds, never on the `HostConfig` field.
- An ephemeral host's drain retries unconfirmed outputs
  (`OUTPUT_RETRY_ATTEMPTS`, a minute apart), then records `outputs_lost` and
  terminates anyway.
- `workdir_bytes` is measured **once**, by the runner as the job ends, and read
  back by `status`. A job with no figure is walked by the first `status` that
  finds its workdir, within `cleanup.MEASURING_BUDGET_S` per call, and reports
  `null` past the budget. `gpuc clean` measures afresh. Every byte figure is
  what deleting the tree gives the filesystem back (`reclaimable_bytes`), not
  what `du` says; `cleanup.dir_size` is the `du` twin, used only for the uv
  cache in `health`.

## The shared uv cache

uv materialises a venv by reflinking or hardlinking out of `~/.cache/uv`, which
works only *within one filesystem* and not at all under `UV_LINK_MODE=copy`.
Two rules follow:

- Nothing in the job path sets `UV_LINK_MODE`, and neither the dispatcher nor
  the runner sets `UV_CACHE_DIR` unless `HostConfig.env` does.
- `gpuc host bootstrap` compares the filesystem of gpuc home with that of `uv
  cache dir` (`health.uv_cache_placement`, asked of the host's own code once
  the package is there). If they differ it sets `UV_CACHE_DIR` in the host's
  own config to `<parent of gpuc home>/.cache/uv`, beside gpuc home rather
  than inside it. A cache the host's config already names is never
  overridden, and an unreadable comparison changes nothing.

`HostConfig.env` is otherwise opaque, and the keys the tool has an opinion
about are one table, `jobs.MANAGED_ENV`: `UV_CACHE_DIR` and `HF_HOME` are
*sticky* (an `--env` that does not name them keeps them) and *derived* by
bootstrap beside gpuc home when the host names nothing (the uv cache wherever
gpuc home and `$HOME` are on different filesystems, `HF_HOME` only under a
persistent root); `UV_INSTALL_DIR` and `UV_TOOL_BIN_DIR` name directories that
go on every child's PATH. `gpuc host clean <host> --uv-cache` runs `uv cache
prune`, never `clean`, and `--hf-cache` runs `hf cache prune`.

## Fetching a job's files

`gpuc fetch` reads only `jobs/<id>/workdir/`, whatever state the job is in.
The host lists the files (`python -m gpuc.host fetch`), so a job's results are
told from its checkout by the same outputs baseline uploads use, and never by
the client; the client then pulls exactly that list, into `<to>/<job id>/`.
The host verb changes nothing, and nothing is ever pushed from a host. Kept
outputs stay under the same `workdir/` after the checkout around them is
swept, so a fetch looks in one place whatever has been cleaned.

## The data directory

`paths.data_dir()` is `GPUC_DATA_DIR` from the host's `env` (a sticky
`MANAGED_ENV` key), else `data/` in gpuc home. The runner exports it to every
job and creates it 0700 if it can. Nothing on the host deletes from it: not
the workdir sweep, not `purge`, not a rental's drain. Only `python -m
gpuc.host data-remove PATH...` does, and only strictly inside it, resolving
symlinks and `..` first.

## Control side: `gpuc` CLI

The command surface is `gpuc --help` plus each subcommand's `--help`, and
[usage.md](usage.md) explains it; neither is repeated here. What the CLI must
hold to, whatever the flags:

- `gpuc` runs with no config file at all: every setting has a default, there is
  simply no S3 mirror, and one line on stderr points at `gpuc config init`.
- Anything that talks to RunPod (`--runpod`, `pods`, `host add --pod`) checks
  `RUNPOD_API_KEY` first and exits 1 with a single line if it is unset, before
  mirroring a spec or picking a host.
- **One way to ask a host, three answers.** `remote.ask(entry, verb)` answers
  `Answered` (a payload), `Unaskable` (it could not be asked, with the one-line
  reason: ssh failed, the provider still has the pod and nothing can run on
  it, the provider could not be read; the host may still hold its jobs and
  nothing is inferred) or `Gone` (the provider reports the pod terminated or
  missing; nothing failed). Everything that talks to a host consumes those
  three and no other spelling of them; the provider's own vocabulary is
  translated once, in `ask`, and a rental is looked up at its provider there
  and nowhere else. **One rule says what the trouble costs**
  (`actions.mirror_is_the_answer`): a `Gone` host is read from the mirror now,
  and that is the answer, exit 0; an `Unaskable` one may still hold the job,
  so `wait` retries it for `wait.TROUBLE_GRACE_S` before the mirror, `logs`
  prints the mirror's copy but exits 1 with the reason, and every other verb
  is exit 1 with the reason. What a `Gone` host's job became is read by
  `actions.read_mirror` and nothing else: its final state, or why there is
  none, and then the job went with its host and the command fails saying so.
  A `Gone` rental's registry entry is forgotten where it is found
  (`forget_gone_rentals`) by a command somebody typed, never by the
  dashboard's poll.
- A session (`remote.open_session`) reads the host's own `config.json` fresh
  and carries it as `session.config`; nothing that decides anything reads the
  registry's cache. A host with no cached interpreter is asked for one on the
  spot, and what a session reads is recorded in the cache on the way past,
  except by a poll, which never writes the registry.
- A host name is looked up locally; `status <ids>`, `logs`, `wait`, `cancel`,
  `preempt`, `reorder`, `estimate`, `requeue` and `ssh` resolve job ids the
  same way (`actions.locate_many`): the job index (`s3index.JobIndex`, the
  local index then the mirror's), then asking the hosts, each at most twice
  however many ids -- the one the index names, then every host about the ids
  still unplaced. An id no host knows is exit 4 only once every host has
  answered. A host the index names for the job that this machine has no entry
  for is `Gone`, whether or not `--host` named it; one whose registry entry
  this build could not read is `Unaskable`.
- **Every command that takes job ids takes any number**, except `logs`,
  `requeue` and `ssh`, and answers for each: one id it cannot find or act on
  never stops the rest, is exit 4 if no host has it and exit 1 otherwise.
  The verbs (`cancel`, `preempt`, `reorder`, `estimate`) run through
  `actions.job_verbs`: locate the jobs, send each host **one** call naming
  all of its jobs over the session the lookup opened, insist on a verdict
  for each, re-mirror a spec field it changed. The on-host verb takes the
  list and answers `{"jobs": [...]}`, one document per id; a refusal is that
  job's `{job_id, error}`, and one that also says `missing` is "no such
  job".
- What a command does lives in `actions` (with `hosts` and `submitting` for the
  host and submit commands), one function per command returning an `Answer`:
  the document its `--json` form prints, the text form, and what failed.
  `cli.py` holds only argparse, the parsing of flags into plain arguments, and
  the text rendering; `main` emits every answer once and exits with what it
  says. Progress goes to stderr through one reporter under `--json`. The CLI
  and the dashboard call the same functions.
- Nothing runs in the background on this side except, if installed, the web
  dashboard's service.

Local state: `~/.local/share/gpu-coordinator/` with `hosts.json`, `jobs/`
(the local job index), `known_hosts` plus `known_hosts.d/<pod>`, and
`state.lock`, which serialises every registry read-modify-write across
concurrent sessions. `Settings` (`~/.config/gpu-coordinator/config.toml`) is
all optional and every key is in [setup.md](setup.md#settings).

## Snakemake executor plugin

Snakemake finds `snakemake_executor_plugin_gpuc` by its package name, so it
ships in the same wheel and gpuc does not depend on Snakemake.

- It imports nothing from `gpuc`. Every submit, poll and cancel is a `gpuc`
  command with `--json`, run in the directory Snakemake was started from, so
  it reads the same documents under the same compatibility rules as any
  script.
- Only a job that needs a GPU becomes a gpuc job. A rule whose `gpu` is
  unset or a constant 0 is a Snakemake local rule and runs on the controller.
  A `gpu` given as a function always goes to gpuc, and a job it gives 0
  fails rather than getting a card.
- One `gpuc status --json` per poll covers every job in flight. It is never
  one per job.
- A job whose outputs carry Snakemake's incomplete marker is not submitted
  while the gpuc job the marker names is queued, running or on a host that
  could not be asked; that job is polled instead, and one that succeeded is
  the Snakemake job's success. Only a gpuc job whose recorded fingerprint
  (rule code, params, shell command, input and output paths, config) matches
  the job's, with no input newer than its submit and no `-F`/`-R`/`-f` forcing
  the job, is adopted; one that is not is cancelled if live and the job
  submitted again. A job is never submitted again while gpuc could not say
  what became of the earlier one. Asking is one `gpuc status --json` per batch
  of ready jobs.
- Snakemake's `--envvars` and its storage plugins' credentials reach the job
  as gpuc `secrets`, never as exports in the job's command.
- A job runs in its own gpuc workdir. The Snakefile is passed relative to that
  copy, never as the controller's absolute path.

## Shared state is read tolerantly, always

Two sessions of one user share `~/.local/share/gpu-coordinator/hosts.json`, and
a host's `config.json` outlives the build that wrote it. So every reader of a
file the two halves share obeys the same two rules, on both sides:

- an unknown key is ignored (a newer writer may add fields);
- an explicit `null` for a field that is **not** declared optional is dropped,
  so the field's default applies. A `null` for a field that *is* optional is a
  real value and round-trips unchanged: `retention_days: null` is "never
  auto-purge", `s3_prefix: null` is "no mirror".

Control side that means one base, `tolerant.TolerantModel`, which `HostEntry`,
`HostCache`, `Registry`, `Settings`, `IndexEntry` and `Offer` all derive from.
The host config a registry entry caches is kept **verbatim** on top of that,
so a key some newer build wrote survives a round trip through this one. Host
side, with no pydantic, the same rules are spelled out in `jobs.from_dict` for
`HostConfig`, `JobSpec`, `JobState` and the dispatcher's lock body: never
`float(None)`, never a `KeyError`, an unusable value means the default.

`hosts.json` and `config.json` both carry `schema_version` (1); readers accept
it missing. `tests/fixtures/schema/` holds today's shape of each file plus a
newer variant, and every one of them must parse; the shapes earlier builds of
this repository wrote are not kept, pre-release.

A host entry that still does not validate is **skipped, not fatal**: `gpuc`
warns, works with the rest, and writes that entry back untouched on the next
registry write. Only a `hosts.json` that cannot be parsed at all stops
anything (exit 3, a `.bak` kept). One shape is refused on purpose rather than
read: a rental an earlier build spelled with a top-level `pod_id` and no
`rental`, and the warning says to `gpuc host add <name> --pod <id>` it again.

## Exit codes

The table is in usage.md. The contract behind it: **a command reports
everything it found out and exits non-zero if any part of it failed.** Every
action returns an `Answer` carrying its document and its failures, and one
function (`actions.exit_code_of`) turns that into a code: unknown local state
is 3, a relayed outcome (a job's, a remote command's) is itself, any failure
is 1, else 0; `exits.http_status` is the one table from those codes to HTTP
statuses (0 → 200, 1 → 500, 2 → 400, 3 → 503, 4 → 404). What an `Unaskable` or
`Gone` host costs is the one rule under *Control side*; `gpuc status` and
`host bootstrap --all` also exit 1 for a registry entry this build could not
parse.

`--json` is on every command that has an answer to give, and means the same
thing on each: stdout is one object carrying `schema_version`, everything else
goes to stderr, and a failure prints `{schema_version, error, exit_code}` rather
than nothing. The flag never changes an exit code. With either follow,
`gpuc logs --json` is exit 2: a follow is a stream.

`gpuc wait` and `gpuc logs -f` exit with the *job's* outcome rather than their
own, as `gpuc ssh <host> -- cmd` does with the remote command's code. **130** is
a Ctrl-C, raised in one place: `main` turns any `KeyboardInterrupt` into that
exit and the matching `--json` error document, and a command with something to
say about what was in flight raises `Interrupted` to add it.

## Waiting for a job to end (`control/wait.py`)

Purely client-side: nothing on a host knows a client is waiting, so the loop is
free to be killed. Each round sends **one `status` per host** naming every job
still pending there, backing off from 2 s to 30 s unless `--interval` pins it.
`gpuc status <ids>` is one round of it, which settles a host that could not be
asked with its reason at once and never from the mirror.

- A host in trouble is handled by the one rule under *Control side*;
  `actions.read_mirror` is the one reader of a mirrored `state.json`, and
  only if that has no terminal state does the job get an `error`.
- An id whose host answers and does not list it is exit 4, decided after the
  first poll: for `logs -f` at once, for `wait` once the other jobs named have
  been waited for and reported. A job that *was* listed and then vanishes is
  trouble, not a missing id.
- `logs -f` runs `tail -F` as a child writing straight to stdout while the loop
  polls, started on the first poll the host answers, and gives the stream
  `FLUSH_GRACE_S` to catch up before stopping it. `Transport.tail()` keeps
  `-f`, whose non-zero exit on a missing log is what routes `gpuc logs` to the
  mirror.

## Transport

`LocalTransport` runs subprocesses directly. Both expose the same protocol,
including `argv(command)` and `interactive_argv(command)`, so nothing above
them asks which kind it holds. `SshTransport` uses the system `ssh`/`rsync`
with `-o BatchMode=yes -o ConnectTimeout=15`, per-command timeouts, and host
keys pinned on first contact into `~/.local/share/gpu-coordinator/known_hosts`,
except for ephemeral hosts, which get `known_hosts.d/<pod>`.

The `ControlMaster` socket lives in `$XDG_RUNTIME_DIR/gpuc/` (else
`/tmp/gpuc-<uid>/`, 0700), **not** under the state dir.
`transport.CONTROL_PATH_MAX = 100` is the one limit, checked against the
expanded template before ssh runs. Any ssh failure matching `ControlPath too
long|unix_listener` raises immediately even under `check=False`.

Code sync is `rsync` of `git -c core.quotePath=false ls-files -z --cached
--others --exclude-standard` (what that includes is
[usage.md](usage.md#what-gets-synced-to-the-host)); `uncommitted.patch` is
`transport.uncommitted_patch`, a `git diff HEAD -- .` against a throwaway
index so it carries untracked files and never touches the user's staging.
`put_file` writes 0600 content via stdin; secrets never touch argv.

## The registry is an address book

`hosts.json` holds two kinds of thing about a host and only two:

- the **address**: `name`, `ssh`, `port`, `gpuc_home` / `persistent_root`,
  `rental`, hand-entered, local to this machine, and everything needed to open
  a session and find `config.json`. Nothing in it is a fact about how the host
  behaves. A host is a rental exactly when it has a `rental: {provider,
  pod_id}`; `kind` (`local`, `ssh`, `rental`) is derived from the address and
  never stored, and `actions.PROVIDERS` is keyed by `rental.provider`.
- a **cache** of what the host last said: `python`, `uv`, `gpu_info`,
  `driver_version` and a copy of its `config.json`, stamped with `read_at`.
  Offline commands (`host list`, `version`) print it labelled "as of <age>";
  anything that decides something reads the host. Every write of the cache is
  `config.update_cache`: re-read under the lock, merge the fields learned,
  write.

What the host **is** (`gpus`, `s3_prefix`, `env`, `idle_minutes`,
`retention_days`, `provider`, `pkg_commit`) lives in `config.json` on the host
and nowhere else, so one box driven from two machines has one configuration.

- `gpuc host add` is a **connect** (`connect_host`): probe, read
  `config.json`, adopt it if it is there under the name the host calls itself,
  else write `config.first_config`, the one constructor for every kind of host
  (every card seen, `workdir_days`, and the mirror `s3_bucket` implies). Flags
  are per-field overrides written through to the host; a `--gpus` that
  overlaps the existing set without matching it is refused
  (`connect._refuse_overlapping_gpus`). Provision is create pod -> wait for
  ssh -> the same connect.
- `gpuc host set` **writes through** to `config.json`. The client is the one
  writer of that file (`remote.write_config`): read, merge with the host's own
  rule (`jobs.merged_config`), replace by rename. `--persistent-root` and
  `--gpuc-home` are addresses and stay here.
- "the host has no config" is a marker the host echoes, never the absence of
  parseable output: a `config.json` that is there and does not parse is
  reported *unreadable* by `remote.read_config`, and nothing writes over one.
- `gpuc host probe` refreshes the cache and nothing else, including an
  interpreter to run the on-host package with.

## Bootstrap (any host, idempotent)

1. Install `uv` if `~/.local/bin/uv` is missing; `uv python install 3.12` if no
   suitable interpreter.
2. rsync the `gpuc` package to `~/.gpuc/pkg/`; install the `aws` CLI v2 bundle
   and `uv tool install huggingface_hub`, both skipped if present. Their
   failures are warnings, except that a host registered with an `s3_prefix`
   whose `aws` could not be installed fails bootstrap outright. A rental with
   an `s3_prefix` is also given `~/.aws/credentials` (0600, from stdin) out of
   the bootstrapping shell's `AWS_*`; a shared box's `~/.aws` is never touched.
3. **Never rewrite `~/.gpuc/config.json`.** The host owns it, so bootstrap reads
   it and merges back only what it derived (`remote.write_config`): the commit
   just shipped, and the managed env keys the host names none of. The one
   exception is a host with **no** config at all, which gets `first_config`
   with the last config this machine read off it on top.
4. Run `python -m gpuc.host health` and fail bootstrap on a failed check.
5. Start the dispatcher and record the commit this build came from in the
   host's `config.json`. The shipping is `bootstrap.ensure_build`, the one ship
   path, which `submit` and `requeue` also run when the host's config names
   another build. Bootstrap is never blocked by running jobs: the package is
   replaced, and whichever dispatcher takes over adopts them.
6. Record the driver version health reported into the registry's cache. The
   cards are the probe's to record (`gpuc host add`, `gpuc host probe`,
   provisioning), the one look at nvidia-smi the control side takes.

## GPU ownership: indices in, UUIDs out

`--gpus` takes nvidia-smi indices, UUIDs, or a mix, and the registry and
`config.json` store **exactly what was given** (`HostConfig.gpus`).

Everything downstream is UUIDs. **One table and one rule**: `gpus.parse_table`
reads `nvidia-smi --query-gpu=index,uuid,name,memory.total` wherever it is
read, and `gpus.resolve(owned, table, shared)` is the pure function that turns
entries into cards, live on the host and offline on the control side over the
table rebuilt from the registry's cache. The dispatcher resolves once each
pass and assigns and accounts by UUID; the runner resolves its assignment
again on the way in and writes the UUIDs back to the job state; the
dispatcher resolves what it adopts at startup. The one place an index appears
again is the job's own `CUDA_VISIBLE_DEVICES`, translated from the UUIDs by
the runner at that instant and pinned with `CUDA_DEVICE_ORDER=PCI_BUS_ID`.

An entry that resolves to nothing (an index the driver no longer uses or a
UUID it no longer reports) is logged, treated as unavailable (jobs wait, they
do not fail), and reported by `gpuc status` and the health check's
`gpu_uuids`. An entry naming a card already named (an index and its own UUID,
or a card in both lists) is a `duplicate`: the health check and `gpuc host
add|set` refuse it, the dispatcher hands the card out once, and the runner
fails an assignment that carries one.

A host given its first config with no `--gpus` owns every card the probe saw,
as UUIDs, less any named by `--shared-gpus`. A host that already has a config
is never defaulted.

Every listing names a card `[index] name vram`; `gpuc host list` adds the UUID,
`gpuc status` adds free/busy and names each running job's cards, and `gpuc host
probe` lists owned cards unless `--all-gpus`. The probe records `gpu_info` for
every card either way, so a later `--gpus 5` resolves offline.

## Shared GPUs: cards we borrow rather than own

`config.gpus` is what the host owns, `config.shared_gpus` what it may *borrow*,
spelled and resolved the same way. What it means for a submitter is
[usage.md](usage.md#shared-gpus).

- Two gates: the job asked (`use_shared`, fixed at submit), and `nvidia-smi
  --query-gpu=memory.used,utilization.gpu` reports 0 MiB *and* 0% right now.
  Every way of not knowing counts as in use.
- Owned cards first, always; a job borrows only its shortfall.
- The sample is taken once per dispatch pass, and every job in that pass is
  judged against that one reading.
- No yield: a borrowed card is held until the job ends, a collision with its
  owner is not detected, and `gpuc preempt` is the way out.
- No per-host floor on which jobs may borrow (`HostConfig.shared_gpus`).
- A card in both lists is a `duplicate` of `gpus.resolve`, refused by `gpuc
  host add|set` and by the host's own `gpu_uuids` health check; should one
  reach a dispatcher, owning wins.

## Persistent root (a host whose `$HOME` is wiped on restart)

`gpuc host add|set <name> --persistent-root R` moves `GPUC_HOME` to `R/gpuc`,
and nothing else: the queue, specs, state, logs and workdirs are the state that
cannot be reinstalled. uv, its managed Pythons, `uv tool` installs and the `aws`
bundle stay in `$HOME`; uv's *cache* is the exception (see The shared uv
cache). Bootstrap creates `R` 0700 if it creates it, and leaves an existing
`R`'s mode alone.

`HostConfig.env` (`--env K=V`) is applied to every job's environment *before*
the job's own `env`: by the dispatcher to every child it spawns, by the
runner, and to every `HostSession` invocation of the on-host package.
`UV_INSTALL_DIR`/`UV_TOOL_BIN_DIR` in it are also prepended to `PATH`.

`gpuc host probe` reports `$HOME`'s filesystem type as a fact and draws no
conclusion from it. The health check's disk floor is measured on
`paths.home()`, so it is `R`'s volume when a root is set. The runbook for a
host that came back empty is in setup.md.

## Providers

Everything the control side knows about a provider's vocabulary lives on the
`Provider` instance: which pod statuses mean nothing can run (`dead_statuses`),
which mean the rental has ended (`gone_statuses`), which mean it can be
dialled (`running_statuses`), the log signature of a broken host, and how a
terminate is retried and confirmed (`terminate_confirmed`). Nothing outside
`providers/` names a status; `actions.PROVIDERS` maps a rental's `provider`
name to its class. Adding a provider is one class and one table entry.

### RunPod (v2 REST, `https://api.runpod.io/v2`, bearer `RUNPOD_API_KEY`)

- `offers(constraints)`: `GET /catalog/gpus?include=AVAILABILITY&product=POD&cloud=<tier>&minCudaVersion=<x>`
  once per requested tier, `<x>` being the one CUDA floor the create also gets
  (`DEFAULT_CUDA_MIN` unless `--cuda-min`); filter by name list / min VRAM / max price /
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
  spending limits across rentals are a non-goal.

## Provisioning flow (`gpuc submit --runpod`)

1. Parse, validate and prepare the spec (secrets present, `{job_id}`
   expanded) before anything is bought: the same pipeline as `--host`, which
   then rents or looks up the host, opens one session, ships this build if the
   host runs another, judges the fit against the host's own config, stages,
   enqueues and records the job (local index, then the mirror, once).
2. Unless `--no-reuse`: pick an existing registered pod whose own config records
   an offer that still satisfies the constraints, which owns enough cards, whose
   pod the provider reports RUNNING, whose dispatcher heartbeat is fresh, and
   which is not draining; enqueue there. A registered pod the provider no longer
   has is forgotten rather than dialled.
3. Else, for each offer in order: `create`; poll `get` until the pod is
   running **and** `ssh.direct` present; poll SSH until a trivial command
   succeeds; probe the pod once (the same probe `host add` runs); write the pod
   its `first_config`, whose `provider` block carries the offer and
   `created_at` (`rented.py`: the pod is its own record, and this machine
   keeps none); run bootstrap; enqueue. The pod terminates itself with the
   pod-scoped key RunPod leaves in its own `/etc/rp_environment`.
4. **One ceiling bounds the whole attempt** (`CEILING_MINUTES`), every offer
   included, and every failure gets one of three verdicts
   (`provision.verdict`): *keep waiting*, *next offer*, or *abort* (a local
   ssh misconfiguration, an `ssh` or `rsync` this machine does not have, the
   ceiling, a Ctrl-C or a bug: nothing another pod could fix). The pod is
   terminated on the way out of either of the last two, through the one
   `Provider.terminate_confirmed`; a terminate it could not confirm is
   reported loudly and leaves the registry entry in place.
5. From bootstrap on, the only things that end the pod are the pod itself,
   through that pod-scoped key, and a client running `gpuc host terminate`.
   `tests/test_runpod_e2e.py` proves the first on every opt-in run.

## Ending a rental on purpose (`gpuc host terminate`, `teardown.py`)

Client-side: the provider call works whether or not the pod still answers, and
once it returns there is no host left to own any state.

- The target is a registry name, else a pod id, else a pod name; the pod search
  is over `list_ours()`, so a name that is not a registered host only ever
  resolves to a pod with our prefix. A `local` or `ssh` host is refused; an
  unknown name is exit 4.
- Unless `--force`, the host must *say* it is idle: one `status.gather`, then
  exit 1 on work in flight (running, queued, or finished with outputs not
  confirmed uploaded, each named), on a host that did not answer, or on a
  target with no registry entry to ask. `--force` does not ask at all.
- A pod the provider reports dead (`Provider.dead_statuses`) is never refused
  over; one it already reports gone is not an error: nothing is called, and
  the stale entry is dropped.
- The terminate is `Provider.terminate_confirmed`, the same one provisioning
  ends a pod with: the provider must confirm the pod gone before anything
  local changes. Only then is the entry dropped, through `forget_host`, which
  drops it only if it is that pod's and reports whether it went (`forgotten`
  in the document). A terminate that could not be confirmed raises, keeping
  the entry.

## Web dashboard (`gpuc web serve`)

A thin view, by construction: `gpuc.control.actions` holds one function per
command returning the `Answer` its `--json` form prints, and both `cli.py` and
`web/app.py` call those. **Nothing the dashboard shows or does exists only in
the dashboard**, and anything it gains lands in `actions` first. An answer
goes on the wire under the HTTP status its exit code maps to
(`exits.http_status`); the API's failure document is `--json`'s.

The server is stdlib `ThreadingHTTPServer` with `bcrypt` the one added
dependency. One password, hashed into `config_dir()/web-password` (0600) and
read once at startup; a server with no password refuses to start. Sessions are
random tokens held in memory, `HttpOnly; SameSite=Strict`, seven days, and a
POST carrying an `Origin` must match `Host`. Wrong passwords are throttled
(`web/auth.py`). Idle keep-alive connections time out after 30 s, a body over
1 MiB is refused before it is read, and a bug in a handler is a 500 with a
traceback in the server log. No TLS: localhost, a VPN, or behind a proxy.

The page is static HTML/JS polling `/api/status`, `/api/hosts`, `/api/config`
and `/api/version` every 15 s, and `/api/jobs/<id>/logs` while a log panel is
open with *follow* on. Status is gathered across hosts in parallel
(`actions.gather_all`).

`gpuc web serve --install` writes `gpuc-web.service` to `~/.config/systemd/user`
through `control/systemd.py`, without enabling it. The unit pins the config and
state dirs, reads `RUNPOD_API_KEY` from `config_dir()/env` if that file exists,
and restarts on failure under a start limit.

## Status output

What `status` prints, and every flag, is usage.md. The invariants:

- **A host's `status` sends what was asked for, and what it costs grows with
  what the host holds now, not with its history.** Given ids it sends exactly
  those; given none it sends every job that is not finished, the finished
  ones inside `--recent` and `--since`, and every finished job still holding
  a workdir, whose disk and pending outputs the host-level lines count; and
  `finished_count`. `status --all` asks for everything, since a job the index
  knows reads as lost when its host does not send it.
- A host is in one of three states (`status.HostState`), the three answers of
  `remote.ask` under *Control side*, read everywhere else: `UNASKABLE` is
  printed with the reason as its `ERROR` line and is a failure; `GONE` is not,
  its finished jobs are the mirror's, and the entry is forgotten as it is
  printed. For a pod the provider reports
  stopped or gone no ssh is attempted, and the reason is what the provider
  said. One `actions.status` builds the text form, `--json` and the
  dashboard's document, including `--all`'s `unhosted` list, each job in it
  labelled with what the same run found its host to be, and `gpuc requeue`
  offered only where that host cannot still be running it. Per host, `errors`
  decide the exit code and `warnings` (the build) do not.
- Nothing that decides an exit code may be visible only under `--json`: a host
  that answered and still carries an `error` prints it as an `ERROR` line.
- A finished job that produced `outputs:` which never reached S3/HF is flagged
  (`outputs not uploaded`, or `OUTPUTS LOST` once a drain has given up); one
  that declared outputs and never wrote them is not. A *running* job with a
  failure standing at one of its destinations says `UPLOAD FAILING`, from the
  same upload records.
- A pod's `provider_util` and a job's `util` are labelled separately and never
  merged.
- Every job carries the `priority` it is (or was) ordered by, from its state.
  A host that does not report it says `null`, never a default.
- The host's `status` reports each job's `outputs` (the spec's, `{job_id}`
  expanded) and `wandb` (`entity`, `project`, `run_id` from the job's `WANDB_*`
  env, and nothing else of its env). The control side turns those into
  `links[]` in `--json`, derived from what the job declared and never checked;
  the text view does not show them.
- Everything said about *what a host is* (the build it runs, the cards it was
  registered with, what it calls itself) is the host's own answer, and a host
  that could not be reached produces no claim about any of it.

## Which build is a host running

The authoritative copy is the host's `config.json` `pkg_commit`, written by
every bootstrap and reported back by `python -m gpuc.host status`
(`control/version.py`).

- `gpuc status` warns from that value, never from the registry, and says
  nothing about a host it could not reach. A host that *answered* and named no
  commit is warned about rather than passed as current. `status --json`'s
  `pkg_commit` is the host's answer, so `null` means "the host did not say".
- `gpuc submit` and `gpuc requeue` read the host's `config.json` before they
  enqueue and re-ship the package (`ensure_build`) when it does not match this
  build. One comparison, `version.is_other_build`, and it is strict: a host
  that names no commit was never bootstrapped and is refused, and a checkout
  with uncommitted changes is `<commit>-dirty-<hash of the changes>`. That
  same read is what the rest of the submit works from, and it replaces the
  registry's cache on the way past.
- A host on another build is still asked, not shipped to: a request its
  on-host CLI rejects (argparse's exit 2) is `Unaskable` with the reason and
  `gpuc host bootstrap <host>`, never a traceback. What needs only the host's
  cards, dispatcher and queue (reuse, teardown, `pods`, the placement after a
  submit) sends a bare `status`, which every build understands.
- `gpuc host list` and `gpuc version` never ssh: they report the commit the
  host was running when this machine last read it, labelled with its age.

## Testing rules

- `./check.sh` is the whole suite, and what CI runs. A bare `pytest` is safe
  too: `addopts` in `pyproject.toml` excludes the `runpod` marker, so the tests
  that rent hardware never run by accident -- opt in with `pytest -m runpod`.
  Nothing else is excluded by default.
- Unit tests run without a GPU (one fake `nvidia-smi`, `tests/fake_nvidia_smi.py`,
  answering in-process and on `PATH`; temp `~/.gpuc`). The CLI and provisioning
  tests drive a real host in a temporary home (`tests/fakehost.py`): every
  command runs through `bash -c` with `$HOME` moved, so the probe script, the
  host's own config reads and writes, the on-host package and a bootstrap run
  for real. Only nvidia-smi, the `uv`/`aws`/`hf` installs, the provider API
  and S3 are stood in for.
- Local GPU integration tests (`gpu`) **run by default**: they use tiny tensors
  (`torch.zeros(8)`), never more than ~100 MB VRAM because other people's jobs
  share the card, and they skip themselves on a machine whose `nvidia-smi` does
  not report `tests.conftest.LOCAL_GPU_UUID` -- which is every CI runner. A GPU
  queue whose GPU tests are the ones nobody runs is how they rot.
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
