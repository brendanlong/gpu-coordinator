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

**`gpuc status`** — per host: kind, reachability, dispatcher heartbeat, owned
cards (`free` / `busy <job-id>` / `UNAVAILABLE`), the queue, running jobs with
phase, minutes and last util, and recent finished jobs. `--host H` narrows it;
`--recent N` (default 5) and `--since 24h|7d|90m` (a bare number means hours)
choose how much of the finished list to show; `--all` adds jobs only the local
index and the S3 index know, which is how you find what was on a host that lost
its state; `--json` is [below](#exit-codes-and---json). `--suspects` lists running
jobs that are billing but idle and **never kills anything**: a job is a suspect
only in phase `main`, judged by **its own `low_util` settings** as the host
reports them, so a job that raised its floor is judged by what it asked for and
one with `enabled: false` is never listed. It also flags a pod past a TTL it
actually has.

**`gpuc logs <job-id> [-f] [-n N] [--host H]`** — tails `log.txt` on the host
(`-n` defaults to 200). If the host cannot produce it, gpuc says why — including
"was purged" when the whole job dir is gone — and falls back to the S3 mirror,
which needs `s3_bucket` set here **and** an `s3_prefix` for that job (from the
job's index entry, else the host's). `-f` follows the host's file and never
falls back. There is no `--json`: a log is a byte stream.

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

**`gpuc requeue <job-id>`** — re-reads the spec from the S3 mirror and submits
it again as attempt+1, with the workdir re-synced from your *current* directory.
It therefore **needs `s3_bucket`** (without it, submit the job file again) and
cannot rebuild a `--no-git` workdir. `--host H` sends it somewhere else;
`--runpod` provisions for it; with neither, it goes back to the host the local
index says it ran on.

`--host` is optional on `logs`, `cancel`, `reorder` and `requeue`: the local job
index is tried first, then every registered host is asked whether it knows the
id. An unknown job or host is exit 4.

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
| `gpu-assert` | an assigned UUID is not present in the host's `nvidia-smi` |
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
| 1 | the command failed (transport, provider, a refused submit, a `clean` the host reported errors for) |
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

The job objects in `queued`, `running` and `finished` all carry those eleven
fields. A card the host cannot see appears in `gpus` as `{"owned_as": "3",
"available": false}` instead. A job's `util` is its **last** sample from the
host's own nvidia-smi over that job's cards; a pod's `provider_util` is the
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
```

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
  `outputs:`, or the workdir is already gone. Otherwise `outputs not confirmed
  uploaded` — `outputs:` paths live inside the workdir and a failed job keeps
  its workdir, so purging one could bin the only copy of a checkpoint.

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

**Unconfirmed outputs.** `gpuc status` flags a finished job whose `outputs:`
never reached S3 or HF as `outputs not uploaded`, and lists them per host,
because those are the jobs a purge — or a pod going away — would take with them.
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
| `clean --purge` skips everything as "not backed up" | the host has no `s3_prefix`, so nothing is mirrored and deleting a job dir would lose its log | `gpuc host set <host> --s3-prefix s3://bucket/gpuc/<host>` + `gpuc host bootstrap`, or accept the loss with `--force` |
| `status` says a job's `outputs not uploaded` | the final upload of its `outputs:` failed, so the results exist only on that host | copy them off, or `gpuc requeue <id>`; a purge will not remove it until they are confirmed |
| a job is `OUTPUTS LOST` | an ephemeral host drained, retried three times and gave up before terminating | the results are gone; fix the credential or bucket, then `gpuc requeue <id>` |
| `--retention-days` never deletes anything | the dispatcher only lives while a non-ephemeral host has work, and it never purges an unmirrored job | check `gpuc status` for `not backed up`, and remember the sweep runs on the next submit |
| `requeue` refuses, or rebuilds the wrong code | it reads the spec from S3 (so `s3_bucket` must be set) and re-syncs the workdir from your current directory | run it from the right checkout; a `--no-git` workdir cannot be rebuilt from a commit |
| `--idle-min` did nothing on a shared box | it only applies to ephemeral (RunPod) hosts; `local` and `ssh` hosts never terminate themselves | nothing to do; use `--retention-days` for disk, not `--idle-min` |
| `uv sync` re-downloads torch on every job | uv's cache is on a different filesystem from gpuc home, so it copies instead of linking | `gpuc host bootstrap <host>` (it sets `UV_CACHE_DIR` for you), or pin one with `--cache-dir` |
| `gpuc` exits 3 and names `hosts.json` | the registry could not be parsed at all; a `.bak` was kept beside it | fix or delete the file, then re-add hosts; nothing was written over |
| a warning names one skipped host entry | that entry did not validate; every other host still works and is written back untouched | fix it by hand, or `gpuc host add <name> ...` to replace it |
| `status` says `host X runs an older gpuc` | this machine was upgraded and the host's copy of the package was not | `gpuc host bootstrap X` — safe while jobs run; the new dispatcher adopts them |
| `status` says `POD GONE` | the pod is terminated or missing but the registry still lists it | `gpuc reconcile --once` |
| `reconcile` reports `DEAD DISPATCHER` and terminates a pod | it stopped beating (or answering ssh) for `dead_dispatcher_minutes` with nothing running | expected: that pod could no longer stop itself. Raise `dead_dispatcher_minutes` if your hosts go quiet legitimately |
| everything on a host is suddenly gone | the container restarted and `$HOME` was on the overlay | the runbook in [setup.md](setup.md#hosts-whose-home-is-wiped-on-restart) |
