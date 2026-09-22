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
    queue.py       # accept a job, list the queue, cancel, preempt, requeue: every change of a job's status
    plan.py        # the dispatch rule, pure: what a pass does with each queued job, and when each will start
    dispatcher.py  # lock+heartbeat, act on the plan, launch runners, escalate stops, idle terminate
    runner.py      # one job: env, CUDA_VISIBLE_DEVICES, preflights, wall-clock limit, sync, exit code
    scope.py       # systemd --user scope probe/wrap/stop; the cgroup kill path
    destinations.py # where files go: one Destination per store (S3, Hugging Face); upload, put_file, preflight
    preflight.py   # sync preflight: prove every destination can be written before the job runs
    baseline.py    # what was already under `outputs:` before the job started
    cleanup.py     # `cleanup:` policy, workdir sizing, the `clean`/`purge` sweeps
    gpus.py        # nvidia-smi parsing, index<->UUID resolution, utilization sampling
    progress.py    # the optional `progress_command`: run it, read a percentage off it
    sync.py        # periodic upload loop, recording each destination's result in the job's state
    health.py      # host preflight: driver, owned GPUs, disk, network download timing
    terminate.py   # self-terminate via provider API (urllib), key from ~/.gpuc/secrets
  control/       # runs on the local machine; may use third-party deps
    cli.py         # argparse and the text output of every `gpuc` command
    actions.py     # one function per command returning its --json document; the CLI and the web call these
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
either half: the RunPod API, a pod's self-terminate, the health check's
download, bootstrap's `curl`, every control-side boto3 client, and
`HF_HUB_USER_AGENT_ORIGIN` in each job's environment. `gpuc.host` is
stdlib-only, so `gpuc._version` must stay stdlib-only and bootstrap must ship
it. The `aws` CLI on a host cannot have its User-Agent overridden.

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
incoming/<jobid>/    # a job `gpuc submit` is still building: the rsynced workdir, then the spec.
                     # `enqueue` renames the whole dir into jobs/, which is the acceptance
jobs/<jobid>/
  .lock              # the flock every read-modify-write of state.json takes
  spec.json          # the submitted JobSpec, written once by `enqueue` and never again
  state.json         # {"status": queued|running|succeeded|failed|cancelled,
                     #  "intent": null | "cancel" | "preempt",   # what somebody asked of a running job
                     #  "attempt": n, "priority": p,           # the live priority; the queue is
                     #                                        # every `queued` state in (priority, id) order
                     #  "estimated_runtime_min": null | m,     # the live estimate, as `gpuc estimate` left it
                     #  "reason": str|null, "problems": [str, ...],  # what ended it, and what else went wrong
                     #  "exit_code": int|null, "gpus": [...], "started_at", "ended_at",
                     #  "phase": setup|preflight|main|sync, "pid": int|null, "pgid": int|null,
                     #  "isolation": "cgroup"|"pgid", "cgroup_unit": str|null,
                     #  "runner_pid": int|null, "runner_boot_id": str|null,
                     #  "runner_starttime": str|null,
                     #  "util_recent": [float|null, ...],
                     #  "progress_pct": float|null, "progress_error": str|null, "eta": str|null,
                     #  "uploads": [{"to": uri, "output": path|null, "ok_at": str|null, "error": str|null}],
                     #                              # one record per destination, `output` null for
                     #                              # the host's mirror of log.txt and state.json;
                     #                              # the whole account of what is safely elsewhere
                     #  "workdir_removed": bool,
                     #  "workdir_bytes": int|null,  # what removing workdir/ would free;
                     #                              # measured once, as the job ended (or by
                     #                              # the first status to find it missing)
                     #  "outputs_lost": bool}       # a drain retried the outputs and gave up
                     # util_recent is the last 40 main-phase samples; null means nvidia-smi
                     # failed and must not be read as 0%. eta is null unless the job is running.
                     # progress_pct survives the job. pgid is the *job's* group, published by
                     # the runner when it spawns a phase, and absent during the launch window.
  outputs_baseline.json # per `outputs:` path, the {relpath: [size, mtime_ns]} the
                     # checkout arrived with; those files are never uploaded as this
                     # job's results and never satisfy `outputs:`
  workdir/           # rsynced code (git-tracked + untracked, .gitignore obeyed); removed per `cleanup:`
  log.txt            # combined stdout/stderr of setup + command, line-buffered
  outputs/           # default output root; JobSpec.outputs paths are relative to workdir
dispatcher.lock      # fd flock held by the running dispatcher
dispatcher.heartbeat # mtime touched every 5 s by the dispatcher
dispatcher.log
draining             # present while the host is shutting itself down
```

There is no queue file and no marker file: a job's `state.json` is the one
record of what it is doing and what has been asked of it, every change to it
is one atomic replace under the job's lock, and the queue is derived from it.
So there is no order of writes to survive a crash between, and nothing that
reconciles two files that disagree. A change of ownership -- the dispatcher
claiming a queued job, a cancel ending one, a requeue putting one back -- is a
compare-and-set on `status` (`jobs.transition`), so two of them racing on one
job cannot both win.

All state writes are atomic (write temp in same dir, `os.replace`).

## JobSpec (JSON; `gpuc submit` accepts YAML or JSON and normalizes)

```
{
  "name": "lego-s4",                    # human label, not an identifier
  "command": "uv run python -m experiments.lego.train --k-max 6",
  "setup": "uv sync --frozen",          # optional; runs before command, phase=setup
  "python": "uv run --no-sync python",  # how the GPU check runs Python in the job's own env
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
                                        # takes, whenever that starts a strictly more
                                        # important queued one right away
  "requires": {"cuda_min": "12.8"},     # informs provisioning only
  "cleanup": "on_success",              # on_success | always | never; see Workdir cleanup
  "attempt": 1                          # set by `gpuc requeue` and `gpuc preempt`, not by the submitter
}
```

`gpuc submit` expands `{job_id}` in output destinations and refuses an `s3` or
`hf_path` whose expanded form does not contain the id (no `hf_path` means the
id itself). Output namespaces are unique by construction; there is no other
overwrite guard anywhere. The S3 mirror holds the spec with `{job_id}`
unexpanded, so `gpuc requeue` expands it with the new id; a mirrored spec
carrying an earlier run's literal id fails the same check.

Job id: `YYYYMMDD-HHMMSS-<6 hex>`, assigned by `gpuc submit`. The
timestamp is second-granular, so two jobs submitted inside the same second
at the same priority tie-break on the random suffix; dispatch order is the
queue's lexical order, not submission order below one second.

## Dispatcher (`python -m gpuc.host dispatch`)

Started by every `enqueue`, and by bootstrap, in its own session so it outlives
the ssh that started it. The rules it holds to:

- **One dispatcher per host**, by `flock` on `dispatcher.lock` plus a heartbeat.
  A holder whose heartbeat is stale (30 s) is killed by the pgid in the lock
  body and taken over.
- **A dispatcher started from the package on disk replaces one that is not**,
  fresh heartbeat or not: the lock body records the `pkg_commit` its holder
  started from, and a holder whose commit *differs* is SIGTERMed, then SIGKILLed
  at 30 s if the lock still names the same process. *Different*, not newer --
  a holder that recorded no commit counts as different and is evicted, and a
  dispatcher whose own commit is unrecorded replaces nobody. `gpuc status`
  reports the holder's commit as `dispatcher.pkg_commit` and warns when it
  differs from the package on disk.
- **One dispatch rule** (`plan.plan`), pure, over one pass's inputs: the
  queue in `(priority, job_id)` order, the owned cards free now, the configured
  counts, and one nvidia-smi reading of the shared cards taken only if a job
  needs to borrow. It decides one of four things per job -- assigned, holds,
  stepped over, fails -- and `launch_ready` acts on it every 2 s: a job that
  does not fit holds the cards it could take, owned and borrowed alike, and
  nothing behind it may take them. The one exemption is a job short only of a
  shared card somebody else is on, which is stepped over rather than failed. A
  job that asks for no GPU, or for more than the host is configured with
  including the shared cards it may borrow, is failed. The same function,
  run over the cards a stop in flight will hand back, is how automatic
  preemption finds the one job the queue is stuck on, and run forward over the
  running jobs' etas (`plan.project`) it is how the host's `status` says when
  each queued job will start. Three questions, one rule.
- **Acceptance is a rename.** `gpuc submit` builds the job dir under
  `incoming/`; the host's `enqueue` writes the spec and initial state there
  and renames the dir into `jobs/`. A job dir under `jobs/` was accepted by
  construction; a dir left under `incoming/` an hour after its last change is
  a submit that died, and is removed.
- **A queued job is claimed by compare-and-set** (`queue.claim`): only a state
  still `queued` becomes `running`, so a cancel that landed between listing
  the queue and launching costs nothing but a skipped launch.
- **Adoption at startup** (`adopt_orphans`): every job whose state says
  `running` is adopted or failed `runner-died`, and a failed one's leftovers are
  killed (`cgroup_unit`, then `pgid`) before its cards go back in the pool.
  Liveness is the recorded `runner_pid` *with* the boot id and start time
  recorded beside it. A job that names no runner is not failed on that alone --
  `launch_ready` writes `running` before the spawn -- so /proc is walked for
  live `gpuc.host run <id>` processes (`runner.live_runner_pids`) and those are
  adopted. What is found is not written back: the runner records its own pid
  moments later.
- **A stop is an intent** in the job's state: `cancel` or `preempt`. The
  **runner** owns the kill -- it polls its state and ends the attempt as
  `cancelled` or `failed: preempted` after a final sync -- and the dispatcher
  escalates only once the grace period has passed (`escalate_stops`): the
  job's scope and group, then the runner itself, then the runner's group. A
  queued job is cancelled on the spot, with no intent. A cancel overrides a
  preempt.
- **Preempt** (`queue.preempt`) is for a running job, and only when something
  else could run instead. The runner keeps `workdir/` and the secrets file and
  does not re-take the outputs baseline. `queue.requeue_preempted` is one write
  under the job's lock: a fresh `queued` state at `attempt+1`, keeping the
  live priority and estimate. It does not go back if the attempt ended for a
  reason of its own, was cancelled while stopping, has no workdir, or the host
  is going away.
- **Automatic preemption** (`preempt_for_waiting`, after `launch_ready`): for
  the one queued job the host is stuck on, stop the set of running
  `auto_preempt` jobs that together cover the whole gap -- least important
  first, most recently started among equals, and only at a strictly higher
  priority number. Cards held by a job that is already stopping count as
  available and that job is not a candidate again; once one stop in a set
  fails, the rest are left alone. Exactly one queued job is asked per pass.
  Nothing is stopped on a host that is going away, and one pass's nvidia-smi
  reading of the shared cards serves both walks.
- **Reorder** and **estimate** write the job's state and nothing else; the
  spec is never rewritten after enqueue. The control side re-mirrors the spec
  after both; a mirror it cannot write is a warning.
- **Idle terminate** (only with `config.provider` set): no running jobs and an
  empty queue for `idle_minutes` -> `draining`, retry unconfirmed outputs,
  mirror every job's state and log, then `terminate.self_terminate()`. Only a
  failed *terminate* stops the shutdown: remove `draining`, log loudly, keep
  dispatching, retry every 10 minutes.
- **Exit** when the queue is empty, nothing is running and the host is not
  ephemeral. An ephemeral host keeps its dispatcher until terminate.

## Runner (one process per job)

The order is the contract; each step is in `runner.py`.

1. Resolve the assignment against `nvidia-smi --query-gpu=index,uuid`, indices
   and UUIDs both, and fail `gpu-assert` if it is empty or names a card that is
   not here. Export `CUDA_VISIBLE_DEVICES` as the cards' nvidia-smi **indices**
   with `CUDA_DEVICE_ORDER=PCI_BUS_ID`, so the numbers mean the same cards to
   CUDA (vLLM `int()`s each entry; a UUID there fails inside a subprocess with
   an error that points at the model); as UUIDs only if the index table could
   not be read. Then the spec `env` and the secrets file.
1b. Snapshot every declared `outputs:` path into `outputs_baseline.json` (path,
   size, mtime), **before `setup`** -- a setup step writing there is this job's
   doing, a committed file that was already there is not. Every upload excludes
   files that still match, and a path holding only those is `failed:
   no-outputs`. Above `baseline.MAX_TRACKED` files the exclusion is dropped with
   a loud warning.
2. `phase=setup`: `spec.setup` in `workdir` under `bash -eo pipefail`.
3. `phase=preflight`: the GPU check **inside the job's environment** -- the
   spec's `python` (default `uv run --no-sync python`) over a probe
   (`is_available()`, then a tensor add and `.item()`), asserting
   `device_count()` equals `gpus`. Failure ->
   `failed: gpu-preflight`. Its own phase, so `gpuc status` can tell "installing
   torch" from "proving the card works".
3b. **Sync preflight**, still before `main`: every `Destination` the job will
   upload to -- each output's, and the host's `jobs/<id>/` mirror -- proves a
   `.preflight` write works with the job's own environment (S3: `aws` resolves
   and the copy succeeds; Hugging Face: `hf` resolves, `hf auth whoami`
   succeeds with the job's token, the repo exists or `hf_create` makes it, and
   the upload succeeds). Failure -> `failed: sync-preflight`, with the command
   and its error in `log.txt`, and no final output sync. A job with no outputs
   on a host with no mirror checks nothing.
4. Start the sync loop (background thread): `outputs` every `sync_interval_s`,
   skipping files modified in the last 10 s, plus `log.txt` and `state.json` to
   `s3_prefix/jobs/<id>/`. **Every upload is recorded** in the state's
   `uploads`, one record per destination: the last success and the last
   failure, an output path that does not exist included -- which is how a
   running job whose output is not being uploaded shows up in `status` rather
   than in a log line nobody reads. Uploads run with the **job's** environment,
   secrets included, so `secrets: [AWS_ACCESS_KEY_ID, ...]` is enough and no
   host-level credential is needed. The secrets file is unlinked after the
   final sync, never before.
5. `phase=main`: `spec.command`, stdout and stderr appended to `log.txt`, in its
   own scope or process group (see Process isolation). Sample the assigned GPUs
   every 30 s into `util_recent`; nothing acts on it. Enforce
   `max_runtime_min` (SIGTERM the group, SIGKILL at 15 s, `failed: timeout`),
   and run `spec.progress_command` every `progress_interval_s`.
6. Capture the exit code **before** any cleanup, stop the sync loop, clear
   every output's success record and run one final sync, so only the upload
   that includes the last minute's files counts. A failed one makes a succeeded
   job `failed: sync`, and an output path that was never written makes it
   `failed: no-outputs`; a job already over for a reason of its own keeps that
   `reason` and lists the upload failure under `problems`. Write final state.
   The runner exits with the job's code.
7. Apply `spec.cleanup` to `workdir/` **after** the final sync and state write,
   never before -- `outputs:` paths resolve inside the workdir. Record
   `workdir_removed` and `workdir_bytes`; a removal that fails is logged and
   nothing more. Then upload state and log once more and record the mirror's
   own upload (the record with `output: null`), so the mirror includes the
   record of itself.

## Job length estimates

Nothing infers how long a job will take. Two optional spec fields are purely
informational and may never change a job's outcome; what they mean to a
submitter is [usage.md](usage.md#job-length-estimates).

- `estimated_runtime_min` is published as `eta` at the top of *every* phase,
  not just `main`. The monitor loop re-reads it from the job's state every
  `ESTIMATE_REFRESH_S`, which is how `gpuc estimate` reaches a running job.
  It is the one thing a running job re-reads; the spec is never rewritten.
- `progress_command` runs in `workdir/` with the job's environment, during
  `main` only; `progress.parse` accepts a fraction with a decimal point or a
  percentage with a `%`, and nothing else. Above 0% the runner replaces `eta`
  with `now + elapsed_main * (100 - pct) / pct`.
- The poll is synchronous, in the loop that also watches for a cancel and
  `max_runtime_min`, with a fixed 10 s timeout and a `killpg` behind it, so a
  wedged progress command delays a kill by at most 10 s of the 15 s the runner
  has. Output goes to a temp file rather than a pipe, and only the last
  `MAX_OUTPUT_BYTES` is read.
- A failed, timed-out or unparseable poll writes `progress_error`, logged the
  first time each distinct message appears. `eta` is cleared when the job ends;
  `progress_pct` is not.
- The **host** projects each queued job's start (`plan.project`, published by
  `python -m gpuc.host status` as `starts_in_s`, with `starts_unknown` saying
  why not) by running the dispatch rule forward: cards come free at the eta of
  whatever holds them and the queue is taken in order, by the same function a
  pass dispatches with. A card held by a job that published no eta is not
  schedulable, so a job whose turn depends on it is unknown; a draining host
  projects nothing. The control side renders the answer and computes nothing.

## Process isolation (cgroup scope, else process group)

A grandchild that double-forks leaves the job's process group and survives
`kill -- -PGID`, holding a GPU the dispatcher is about to hand on. A process
cannot leave its **cgroup** without privilege, so where a `systemd --user`
session with cgroup delegation exists each phase runs as:

```
systemd-run --user --scope --collect --quiet -p TimeoutStopSec=15 \
  --unit=gpuc-<jobid>-<phase>.scope -- bash -c 'base64 -d <<<"$1" | bash' _ <b64>
```

The script is base64-encoded because the words after `--` become a systemd
`ExecStart`, whose own substitution would corrupt inline shell.

The kill path is then `systemctl --user stop <unit>`, with the process-group
kill kept as a fallback, and the dispatcher's backstop uses the unit when
`state.json` records one. `isolation` (`cgroup` | `pgid`) and `cgroup_unit` are
in `state.json` and `gpuc status --json`. The probe runs once per dispatcher and
reaches runners as `GPUC_ISOLATION`. On a host with no user systemd -- every
RunPod pod, most shared boxes -- `pgid` is the mode and a daemonised grandchild
still escapes: a documented hole, not a fixed one.

## Workdir cleanup, retention and purge

`workdir/` is the only part of a job dir gpuc deletes: it is recreatable and
usually the largest. `spec.json`, `state.json` and `log.txt` always stay. The
policy, the two horizons and every refusal are
[usage.md](usage.md#cleanup-and-retention); the contract behind them:

- No policy touches a job that is not finished, and the runner applies its
  policy after the final sync and state write. A preempted job keeps its
  workdir whatever `cleanup:` says: the next attempt re-runs in it.
- `python -m gpuc.host clean (--all-finished | --older-than DAYS | --only IDS)
  [--sweep-only IDS] [--dry-run]` and `purge [--older-than DAYS] [--only IDS]
  [--sweep-only IDS] [--dry-run] [--force]` print JSON and **fail closed**: a
  running or queued job, an unreadable `state.json`, and (under an age gate) a
  job with no parseable `ended_at` are skipped. `--only` replaces the age gate,
  and a named id no job dir matches removes nothing and exits 1 -- so the
  control side reads these two with `host_json(check=False)`, since the
  document is the report of the failure. `--purge` also runs the workdir sweep,
  and `gpuc clean --only` sends the ids as both `--only` and `--sweep-only`; the
  two stay separate because `--verify` purges what the mirror confirmed and
  sweeps what was named.
- The host cannot consult the mirror, so the mirror's upload record in the
  job's own `state.json` is the purge's authority, and confirmed outputs are
  every output destination's record saying the last upload succeeded.
  `--verify` on the control side is the exception: it HEADs the mirrored
  `log.txt` under the recorded mirror uri first.
- The dispatcher runs `cleanup.clean(automatic=True)` at startup and then at
  most once an hour, purge first. The automatic form adds two refusals a delete
  nobody typed needs: `cleanup: never`, and outputs not confirmed elsewhere.
  `workdir_days` defaults to `cleanup.DEFAULT_WORKDIR_DAYS` only for a host
  being configured for the first time, never on the `HostConfig` field -- a
  config that predates the key must not start deleting because it was shipped a
  newer package.
- An ephemeral host's drain retries unconfirmed outputs
  (`OUTPUT_RETRY_ATTEMPTS`, a minute apart), then records `outputs_lost` and
  terminates anyway.
- `workdir_bytes` is measured **once**, by the runner as the job ends, and read
  back by `status` rather than walked per call. A job with no figure is walked
  by the first `status` that finds its workdir, within
  `cleanup.MEASURING_BUDGET_S` per call, and reports `null` past the budget. A
  workdir that is gone is zero without a walk. `gpuc clean` measures afresh.
- Every byte figure is what deleting the tree gives the filesystem back, not
  what `du` says: uv hardlinks or reflinks a venv out of its cache, so most of
  a torch venv stays. `reclaimable_bytes` counts both (`st_nlink`, and
  `FIEMAP_EXTENT_SHARED` per file) and fails towards "all yours" per file.
  `cleanup.dir_size` is the `du` twin, used only for the uv cache in `health`.

## The shared uv cache

uv materialises a venv by reflinking or hardlinking out of `~/.cache/uv`, which
works only *within one filesystem* and not at all under `UV_LINK_MODE=copy`.
Two rules follow:

- Nothing in the job path sets `UV_LINK_MODE`, and neither the dispatcher nor
  the runner sets `UV_CACHE_DIR` unless `HostConfig.env` does.
- `gpuc host bootstrap` compares the filesystem of gpuc home with that of `uv
  cache dir`. If they differ it sets `UV_CACHE_DIR` in the host's own config to
  `<parent of gpuc home>/.cache/uv` -- beside gpuc home, not inside it, so
  `clean` and an `rm -rf` of gpuc home cannot take the cache with them. A cache
  the host's config already names is never overridden, and an unreadable
  comparison changes nothing.

`HostConfig.env` is otherwise opaque, and the keys the tool has an opinion
about are one table, `jobs.MANAGED_ENV`: `UV_CACHE_DIR` and `HF_HOME` are
*sticky* (an `--env` that does not name them keeps them) and *derived* by
bootstrap beside gpuc home when the host names nothing, so a persistent root
keeps both caches; `UV_INSTALL_DIR` and `UV_TOOL_BIN_DIR` name directories
that go on every child's PATH. Nothing else in `env` means anything to gpuc.

The check compares filesystems rather than sizes because `du` cannot see a
reflink. The case it exists for is a pod with `--persistent-root`: gpuc home on
the network volume, `~/.cache` on the container's overlay.

`gpuc host probe` and the health check both report the cache's size and whether
it shares a filesystem with gpuc home; the health check is warn-only. `gpuc host
clean <host> --uv-cache` runs `uv cache prune`, never `clean`, which would throw
away the wheels the next job wants to link.

## Control side: `gpuc` CLI

The command surface is `gpuc --help` plus each subcommand's `--help`, and
[usage.md](usage.md) explains it; neither is repeated here. What the CLI must
hold to, whatever the flags:

- `gpuc` runs with no config file at all: every setting has a default, there is
  simply no S3 mirror, and one line on stderr points at `gpuc config init`.
- Anything that talks to RunPod (`--runpod`, `pods`, `host add --pod`) checks
  `RUNPOD_API_KEY` first and exits 1 with a single line if it is unset, before
  mirroring a spec or picking a host.
- A host name is looked up locally; `logs`, `wait`, `cancel`, `preempt`,
  `reorder`, `estimate`, `requeue` and `ssh` resolve a job id the same way
  (`find_job_host`): the job index, then asking each host, and an id nothing
  knows is exit 4, never a guess. Every per-job verb runs through
  `actions.job_verb`: find the host, ask it, insist on a verdict, re-mirror a
  spec field it changed. The CLI and the dashboard call the same functions.
- Nothing runs in the background on this side except, if installed, the web
  dashboard's service.

Local state: `~/.local/share/gpu-coordinator/` with `hosts.json`, `jobs/`
(the local job index), `known_hosts` plus `known_hosts.d/<pod>`, and
`state.lock`, which serialises every registry read-modify-write across
concurrent sessions. `Settings` (`~/.config/gpu-coordinator/config.toml`) is
all optional and every key is in [setup.md](setup.md#settings).

## Shared state is read tolerantly, always

Two sessions of one user share `~/.local/share/gpu-coordinator/hosts.json`, and
a host's `config.json` outlives the build that wrote it. So every reader of a
file the two halves share obeys the same two rules, on both sides:

- an unknown key is ignored (a newer writer may add fields);
- an explicit `null` for a field that is **not** declared optional is dropped,
  so the field's default applies. A `null` for a field that *is* optional is a
  real value and round-trips unchanged: `retention_days: null` is "never
  auto-purge", `s3_prefix: null` is "no mirror".

Control side that means one base, `tolerant.TolerantModel` (`extra="ignore"`,
a default on every field, and a `model_validator(mode="before")` that consults
the annotation), which `HostEntry`, `HostCache`, `Registry`, `Settings`,
`IndexEntry` and `Offer` all derive from. The host config a registry entry
caches is kept **verbatim** on top of that, so a key some newer build wrote
survives a round trip through this one. Host side, with
no pydantic, the same rules are spelled out in `jobs.from_dict` for
`HostConfig`, `JobSpec`, `JobState` and the dispatcher's lock body: never
`float(None)`, never a `KeyError`, an unusable value means the default.

`hosts.json` and `config.json` both carry `schema_version` (1); readers accept
it missing. `tests/fixtures/schema/` holds today's shape of each file plus
older and newer variants, and every one of them must parse.

A host entry that still does not validate is **skipped, not fatal**: `gpuc`
warns, works with the rest, and writes that entry back untouched on the next
registry write. Only a `hosts.json` that cannot be parsed at all stops
anything (exit 3, a `.bak` kept).

## Exit codes

The table is in usage.md. The contract behind it: **a command reports
everything it found out and exits non-zero if any part of it failed.** `gpuc
status` prints every host that answered *and* exits 1 for one it could not read,
or -- when it was asked about every host rather than one -- for a registry entry
this build could not parse. 3 means local state could not be read at all, so the
answer is *unknown*; 4 is a name that does not exist. Automation keys on
`hosts[].running` and treats exit 3 as unknown, never as "nothing running".

A rental the provider reports missing or TERMINATED is not a failure: `status`
and `host bootstrap --all` forget that registry entry where they find it
(`forget_gone_rentals`, `rental_gone`). A pod the provider still has and nothing
can run on (EXITED, ERROR) is a failure like any other host that could not be
read, and is kept for `gpuc host remove`. Only a command somebody typed
forgets: the dashboard's poll never writes to the registry.

`--json` is on every command that has an answer to give, and means the same
thing on each: stdout is one object carrying `schema_version`, everything else
goes to stderr, and a failure prints `{schema_version, error, exit_code}` rather
than nothing. The flag never changes an exit code. With either follow,
`gpuc logs --json` is exit 2: the document is printed once and a follow is a
stream.

`gpuc wait` and `gpuc logs -f` exit with the *job's* outcome rather than their
own, as `gpuc ssh <host> -- cmd` does with the remote command's code. **130** is
a Ctrl-C, raised in one place: `main` turns any `KeyboardInterrupt` into that
exit and the matching `--json` error document, and a command with something to
say about what was in flight raises `Interrupted` to add it.

## Waiting for a job to end (`control/wait.py`)

Purely client-side: nothing on a host knows a client is waiting, so the loop is
free to be killed. Each round sends **one `status` per host** rather than one
per job, backing off from 2 s to 30 s unless `--interval` pins it. Three rules
that reach outside the module:

- A host that cannot be asked is *trouble*, not an answer: retried for
  `TROUBLE_GRACE_S`, then read from the **S3 mirror** (the spec's "the mirror is
  read only when the host is gone"). A rental whose pod the provider already
  reports gone skips the grace and is read from the mirror at once. Only if
  that has no terminal state does the job get an `error`.
- An id whose host answers and does not list it is exit 4, checked after the
  first poll. A job that *was* listed and then vanishes is trouble, not a
  missing id.
- `logs -f` runs `tail -F` as a child writing straight to stdout while the loop
  polls, and gives the stream `FLUSH_GRACE_S` to catch up before stopping it:
  the runner writes its terminal state before it logs the outcome, so the poll
  is always slightly ahead of the log. `-F` rather than `-f` follows a log that
  does not exist yet, and only from this path -- `Transport.tail()` keeps `-f`,
  whose non-zero exit on a missing log is what routes `gpuc logs` to the mirror.

## Transport

`LocalTransport` runs subprocesses directly. Both expose the same protocol,
including `argv(command)` and `interactive_argv(command)`, so nothing above
them asks which kind it holds: `gpuc ssh`, `gpuc logs -f` and `--print` build
the same command for either. `SshTransport` uses the system
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
  it is a fact about how the host behaves. A host is a rental exactly when it
  has a `pod_id`; `kind` only says which provider.
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
  else write the initial one -- owning every card the probe saw unless
  `--gpus` says otherwise (see GPU ownership). Flags are per-field overrides
  written through to the host; a `--gpus` that overlaps the existing set
  without matching it is refused, because that one difference hands one card
  to two jobs.
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

## Bootstrap (any host, idempotent)

1. Install `uv` if `~/.local/bin/uv` is missing; `uv python install 3.12` if no
   suitable interpreter.
2. rsync the `gpuc` package to `~/.gpuc/pkg/`; install the `aws` CLI v2 bundle
   and `uv tool install huggingface_hub`, both skipped if present. Their
   failures are warnings -- **except** that a host registered with an
   `s3_prefix` whose `aws` could not be installed fails bootstrap outright,
   since every job on it would end `failed: sync-preflight`.
3. **Never rewrite `~/.gpuc/config.json`.** The host owns it, so bootstrap reads
   it and merges back only what it derived, through the host's own `config
   --merge`: the commit just shipped, and the managed env keys the host names
   none of (`UV_CACHE_DIR` by filesystem comparison, `HF_HOME` beside gpuc
   home). The one exception is a host with **no** config at all, where the last
   config this machine read off it is restored.
4. Run `python -m gpuc.host health` and fail bootstrap on a failed check.
5. Start the dispatcher with `$HOME/.local/bin` and `$HOME/.cargo/bin` on PATH,
   and record the commit this build came from in the host's `config.json` --
   the authoritative copy, which `status` reports and `submit` judges. Bootstrap
   is never blocked by running jobs: the package is replaced, and whichever
   dispatcher takes over adopts them from their `state.json`.
6. Record the host's cards and driver version into the registry's cache, best
   effort, so `gpuc host list` and `gpuc status` can name them. `gpuc host
   probe` records the same before a host is bootstrapped, and a RunPod host
   falls back to its offer's GPU name and VRAM.

## GPU ownership: indices in, UUIDs out

`--gpus` takes nvidia-smi indices, UUIDs, or a mix, and the registry and
`config.json` store **exactly what was given**: an index is how a share of a
shared box is agreed, and resolving at registration would freeze one boot's
numbering into a file nobody looks at again.

Everything downstream is UUIDs. The dispatcher re-runs `nvidia-smi
--query-gpu=index,uuid` each pass, maps the owned entries to whatever the driver
calls those cards now, and assigns and accounts by UUID. The runner resolves
its assignment again on the way in and writes the UUIDs back to the job
state, and the dispatcher resolves what it adopts at startup: an index
mistaken for a busy card's name is a card handed out twice. The one place an
index appears again is the job's own `CUDA_VISIBLE_DEVICES`, translated from
the UUIDs by the runner at that instant and pinned with
`CUDA_DEVICE_ORDER=PCI_BUS_ID`, because that is the form every CUDA stack
accepts.

An owned entry that resolves to nothing is logged, treated as unavailable (jobs
wait, they do not fail), and reported by `gpuc status` and the health check's
`gpu_uuids`. Shared entries resolve the same way and are checked for overlap
with the owned ones.

A host given its first config with no `--gpus` owns every card the probe saw, as
UUIDs, less any named by `--shared-gpus`. A host that already has a config is
never defaulted.

Every listing names a card `[index] name vram`; `gpuc host list` adds the UUID,
`gpuc status` adds free/busy and names each running job's cards, and `gpuc host
probe` lists owned cards unless `--all-gpus`. The probe records `gpu_info` for
every card either way, so a later `--gpus 5` resolves offline.

## Shared GPUs: cards we borrow rather than own

`config.gpus` is what the host owns, `config.shared_gpus` what it may *borrow*,
spelled and resolved the same way. What it means for a submitter is
[usage.md](usage.md#shared-gpus).

- Two gates: the job asked (`use_shared`, fixed at submit, so a job that can
  never fit is failed rather than left waiting), and `nvidia-smi
  --query-gpu=memory.used,utilization.gpu` reports 0 MiB *and* 0% right now.
  Every way of not knowing counts as in use.
- Owned cards first, always; a job borrows only its shortfall.
- The sample is taken once per dispatch pass, and every job in that pass is
  judged against that one reading -- which is also what stops two of them being
  handed one card.
- No yield: a borrowed card is held until the job ends, a collision with its
  owner is not detected, and `gpuc preempt` is the way out.
- No per-host floor on which jobs may borrow. Borrowing is not a reservation, so
  a floor never protects an important job from a trivial one.
- A card in both lists is refused by `gpuc host add|set` and by the host's own
  `gpu_uuids` health check; should one reach a dispatcher, owning wins.

## Persistent root (a host whose `$HOME` is wiped on restart)

`gpuc host add|set <name> --persistent-root R` moves `GPUC_HOME` to `R/gpuc`,
and nothing else: the queue, specs, state, logs and workdirs are the state that
cannot be reinstalled. uv, its managed Pythons, `uv tool` installs and the `aws`
bundle stay in `$HOME` -- bootstrap reinstalls them in seconds, and these shared
volumes are much slower than a container's local disk. uv's *cache* is the
exception (see The shared uv cache). Bootstrap creates `R` 0700 if it creates
it, and leaves an existing `R`'s mode alone.

`HostConfig.env` (`--env K=V`) is applied to every job's environment *before*
the job's own `env` -- by the dispatcher to every child it spawns, by the
runner, and to every `HostSession` invocation of the on-host package.
`UV_INSTALL_DIR`/`UV_TOOL_BIN_DIR` in it are also prepended to `PATH`.
`UV_CACHE_DIR` is the one key bootstrap fills in itself, and only when the
host's config names none; `--cache-dir` is that same key.

`gpuc host probe` reports `$HOME`'s filesystem type as a fact and draws no
conclusion from it: an ephemeral queue is what gpuc is built around, and an
overlay `$HOME` is not a misconfiguration to warn about. The health check's disk
floor is measured on `paths.home()`, so it is `R`'s volume when a root is set.
The runbook for a host that came back empty is in setup.md.

## Providers

Everything the control side knows about a provider's vocabulary lives on the
`Provider` instance: which pod statuses mean nothing can run (`dead_statuses`),
which mean the rental has ended (`gone_statuses`), the log signature of a
broken host, and how a terminate is retried. Provisioning, `status`, `wait`
and `host terminate` read those and name no status themselves;
`actions.PROVIDERS` maps a `provider.kind` to its class. Adding a provider is
one class and one table entry.

### RunPod (v2 REST, `https://api.runpod.io/v2`, bearer `RUNPOD_API_KEY`)

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
2. Unless `--no-reuse`: pick an existing registered pod whose own config records
   an offer that still satisfies the constraints, which owns enough cards, whose
   pod the provider reports RUNNING, whose dispatcher heartbeat is fresh, and
   which is not draining; enqueue there. A registered pod the provider no longer
   has is forgotten rather than dialled.
3. Else, for each offer in order: `create`; poll `get` until RUNNING **and**
   `ssh.direct` present; poll SSH until a trivial command succeeds; write the
   pod its config, whose `provider` block carries the offer and `created_at`
   (`rented.py`: the pod is its own record, and this machine keeps none); run
   bootstrap; enqueue. The pod terminates itself with the pod-scoped key
   RunPod leaves in its own `/etc/rp_environment`. On any broken-host signature in `logs`, the
   15-minute ceiling, a health failure, a Ctrl-C or a bug: `terminate`, wait for
   TERMINATED, try the next offer. A terminate that failed is reported loudly
   and leaves the registry entry in place; nothing retries it.
4. From bootstrap on, the only things that end the pod are the pod itself,
   through that pod-scoped key, and a client running
   `gpuc host terminate`. `tests/test_runpod_e2e.py` proves the first on every
   opt-in run.

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
- A pod the provider reports in `DEAD_STATUSES` is never refused over.
- The terminate is retried (`TERMINATE_ATTEMPTS`) and the provider must confirm
  TERMINATED before anything local changes. Only then is the entry dropped,
  through `forget_host`, which drops it only if it is that pod's and reports
  whether it went -- what `forgotten` in the document means. A terminate that
  could not be confirmed raises, keeping the entry.
- A pod the provider already reports TERMINATED is not an error: nothing is
  called, and the stale entry is dropped.

## Web dashboard (`gpuc web serve`)

A thin view, by construction: `gpuc.control.actions` holds one function per
command returning the document its `--json` form prints, and both `cli.py` and
`web/app.py` call those. **Nothing the dashboard shows or does exists only in
the dashboard**, and anything it gains lands in `actions` first. `exit_code_for`
is the one table mapping an error to an exit code (CLI) or an HTTP status (2 ->
400, 3 -> 503, 4 -> 404, 1 -> 500), and the API's failure document is `--json`'s.

The server is stdlib `ThreadingHTTPServer` with `bcrypt` the one added
dependency. One password, hashed into `config_dir()/web-password` (0600) and
read once at startup; a server with no password refuses to start. Sessions are
random tokens held in memory, `HttpOnly; SameSite=Strict`, seven days, and a
POST carrying an `Origin` must match `Host`. Wrong passwords are checked one at
a time under their own lock with a growing pause that a quiet minute resets, so
a guesser at the door cannot stall requests from inside. Idle keep-alive
connections time out after 30 s, a body over 1 MiB is refused before it is read,
and a bug in a handler is a 500 with a traceback in the server log, never a
dropped connection. No TLS: localhost, a VPN, or behind a proxy.

The page is static HTML/JS polling `/api/status`, `/api/hosts`, `/api/config`
and `/api/version` every 15 s, and `/api/jobs/<id>/logs` while a log panel is
open with *follow* on. Status is gathered across hosts in parallel
(`actions.gather_all`), so one wedged host costs its own timeout, not the sum.

`gpuc web serve --install` writes `gpuc-web.service` to `~/.config/systemd/user`
through `control/systemd.py`, without enabling it. The unit pins the config and
state dirs, reads `RUNPOD_API_KEY` from `config_dir()/env` if that file exists,
and restarts on failure under a start limit, so a service with no password fails
after five tries rather than looping for ever.

## Status output

What `status` prints, and every flag, is usage.md. The invariants:

- A host is in one of four states (`status.HostState`), decided once in
  `gather` and read everywhere else: it answered; it could not be reached; its
  pod is dead (the provider still has it and nothing can run on it: a failure,
  kept for `gpuc host terminate` or `gpuc host remove`); its pod is gone (the
  rental has ended: not a failure, and the entry is forgotten as it is
  printed). For either pod state no ssh is attempted, and the line says what
  the provider said rather than printing a connection error.
- A host that answered and still carries an `error` -- the provider could not be
  asked about its pod -- prints it as an `ERROR` line under the header. Nothing
  that decides an exit code may be visible only under `--json`.
- A finished job that produced `outputs:` which never reached S3/HF is flagged
  (`outputs not uploaded`, or `OUTPUTS LOST` once a drain has given up), because
  those are the jobs a purge -- or a pod going away -- would take with them. One
  that declared outputs and never wrote them is not: there is nothing there. A
  *running* job with a failure standing at one of its destinations says
  `UPLOAD FAILING` on its line, from the same upload records.
- A pod's `provider_util` is the provider's reading for the whole pod; a job's
  `util` is the host's own nvidia-smi sampler over that job's cards. They are
  labelled separately and never merged.
- Every job carries the `priority` it is (or was) ordered by, from its state,
  which is the one copy there is. A host too old to report it says `null`,
  never a default.
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

The package is the one thing a host cannot own, because it is shipped to it:
two machines on different commits bootstrapping the same box leave it running
whichever shipped last, and neither registry can see the other's. So the
authoritative copy is the host's `config.json` `pkg_commit`, written by every
bootstrap and reported back by `python -m gpuc.host status`.

- `gpuc status` warns from that value, never from the registry, and says
  nothing about a host it could not reach. A host that *answered* and named no
  commit is warned about rather than passed as current. `status --json`'s
  `pkg_commit` is the host's answer, so `null` means "the host did not say".
- `gpuc submit` and `gpuc requeue` read the host's `config.json` before they
  enqueue and re-ship the package when it does not match this build. That same
  read is what the rest of the submit works from -- the `gpus` the spec is
  judged against, the `s3_prefix` its outputs are recorded under -- and it
  replaces the registry's cache on the way past.
- `gpuc host list` and `gpuc version` never ssh: they report the commit the
  host was running when this machine last read it, labelled with its age.

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
