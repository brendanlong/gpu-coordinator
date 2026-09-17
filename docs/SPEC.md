# gpu-coordinator specification

The canonical statement of what this project does and does not do. Every
change is checked against it; architecture, design and code follow from it.
It says *what* must hold, not *how*. Mechanisms, file layouts and defaults are
[ARCHITECTURE.md](ARCHITECTURE.md), [setup.md](setup.md), [usage.md](usage.md)
and `--help`. Where this document and any of those disagree, this one wins and
the other is wrong.

## Purpose

Let one user (1) provision access to hosts with GPUs and (2) queue work to run
on those GPUs, from any of their machines, with a rented host shutting itself
down when its work is done and nobody watching.

## Terms

- **Client**: the machine `gpuc` is run from. A user may have several, and any
  of them may drive any host.
- **Host**: a machine that runs jobs. Three kinds: **local** (the client
  itself), **ssh** (a machine reached over SSH, typically shared with other
  people and with no sudo), and **rental** (a machine rented from a provider
  for as long as there is work; RunPod today, others later). A local host may
  have no GPU at all.
- **Dispatcher**: the process on a host that owns its queue.
- **Owned GPU**: a card the host may use unconditionally. **Shared GPU**: a
  card the host may borrow under the conditions in *Queueing*. Both are named
  by nvidia-smi index or by UUID. Any other card on the machine is never used,
  and is reported only where the user is choosing cards or debugging.
- **Job**: one spec (command, GPU count, priority, options) submitted to one
  host, identified by an id that is unique across all hosts.
- **Backup destination**: S3 or Hugging Face today; others may be added.
- **Mirror**: an S3 location holding a copy of what must survive a host: job
  specs, the job index, and each job's log and state. It is a copy, never the
  queue.

## Preconditions

- Clients run Linux or macOS. Hosts run Linux. Other Unixes are nice to have;
  Windows is not a target.
- No sudo anywhere. Everything installed on a host lives under the user's home
  (or a chosen persistent directory), and nothing runs as a system service on
  a host.
- A host needs ssh, rsync, outbound HTTPS, and the NVIDIA driver with
  `nvidia-smi` for any GPU it will run jobs on. Bootstrap installs everything
  else, including uv and a Python (3.11 or newer) if none is present. The
  on-host component itself needs nothing but that interpreter: no packages,
  no virtual environment.
- A client needs Python 3.11 or newer and the system `ssh` and `rsync`.
- The tool is built around Python but does not dictate what a job runs, with
  one exception: a job that asks for GPUs is assumed to be a uv project with
  torch installed, which is what the runner's GPU check exercises.

## Hosts and provisioning

- **The client owns provisioning until the host proves healthy.** Any failure
  before that point is the client's to clean up; for a rental that means
  terminating it, then trying the next offer within a time ceiling.
- **Health is checked before any job runs on a host**: the driver, every
  owned and shared GPU resolves to a present card and none is in both lists,
  free disk, and a basic network throughput test.
- **After that the host owns itself**: its queue, its job state, its logs, its
  configuration (which cards it owns and shares, its mirror, its environment)
  and, for a rental, its own shutdown. A rental terminates itself once its
  queue has been empty for a configured idle period, or past an opt-in
  lifetime cap, after draining its uploads.
- **Any client whose SSH key reaches a host can connect to it without
  conflict** and sees the same queue, jobs and configuration. Nothing else
  about the client that set a host up matters afterwards. That includes a
  rental another machine rented.
- The client's record of a host is an address plus a cache. Anything that
  decides something asks the host; output from the cache is labelled with its
  age.
- Connecting selects which cards are owned and which are shared. A rental
  owns every card it has.
- **A client-side reaper is a safety net, not the owner.** It may terminate a
  rental only in states the rental cannot leave on its own: it never
  bootstrapped, its dispatcher has been silent with nothing running, or it is
  past a lifetime cap. Any client holding the provider key may take on a
  rental it did not create by asking the rental what it is, and thereafter
  judges it by those rules; a rental the client has no record of and cannot
  get an answer from is reported, never terminated. Rentals that are not
  ours are never touched, and unreadable local state means the reaper does
  nothing.
- Rentals are bounded by account-wide caps on count and hourly cost, and an
  existing matching rental is reused before a new one is created.

## Queueing

- **A job is queued on the host the user names**, or on a rental provisioned
  for it.
- **The client owns the job until it is committed to the queue**, then the
  host owns it: running, cleanup, and its record. There is one moment at
  which a job becomes the host's; before it the host never runs the job, and
  a submit that dies before it leaves a job the host eventually marks failed
  and cleans up, never one it runs or holds forever.
- A job states its requirements as a number of GPUs. Zero is allowed and
  never waits for a card.
- **Priority is numeric, lower first, and strict.** The queue is taken in
  order: a job that does not yet fit holds the free cards it is waiting for,
  and nothing behind it may take them, even at the cost of idle cards. Two
  kinds of job are stepped over instead of holding cards: one that needs no
  GPU, and one that needs a card the host cannot currently see, which
  includes a shared card someone else is using.
- Priorities of queued jobs can be changed, and the queue reorders
  accordingly.
- **A running job can be preempted** so that a job ahead of it in dispatch
  order can run, by command or automatically for jobs that opt in. Automatic
  preemption fires only for a strictly higher-priority job and only when the
  cards it frees are enough to start it. Preemption restarts the job from the
  beginning in its existing working tree; checkpointing is the job's
  business. A preempt is refused when nothing waiting would be dispatched
  ahead of the preempted job, because it would discard work for nothing.
- **Shared GPUs** are used only by jobs that opt in, only after every free
  owned card, and only while nvidia-smi reports zero memory and zero
  utilization on the card. Owned cards are assumed to have no other users and
  this is never verified. A borrowed card is treated as owned until the job
  ends; a later collision with its real owner is not detected.
- A queued job bigger than the host's configuration, because the
  configuration shrank after submit, fails rather than waits; shared cards
  count only for a job that opted into them. A card that is merely missing
  right now makes a job wait.

## Running a job

- The job's code is the working tree the user submits from, including
  untracked and uncommitted changes. Large data comes from a backup
  destination inside the job, never through the checkout.
- **Checks happen as early as they can.** At submit: the spec is valid, every
  secret it names is present, and its GPU count fits the host. At job start,
  before the main phase: the GPUs work inside the job's own environment, and
  every backup destination is writable with the job's own credentials.
- A job runs as setup, those checks, main, and a final upload, and may
  report progress or an estimated remaining time. Estimates are informational
  and never change a job's outcome; nothing infers a job's length.
- **Outputs are backed up continuously** during the run, not only at the end,
  and so are logs. Files that were already in the checkout are never uploaded
  as results. Output locations can include the job id so runs do not
  overwrite each other; the tool does not otherwise guard against reuse.
- On success, uploads finish and the working tree is deleted. On failure or
  cancellation it is kept for a configurable period for debugging. On a
  rental, the rental's shutdown overrides that period, and a rental that has
  retried its uploads and still cannot deliver them terminates anyway: an
  expensive machine is not kept alive for a bucket we cannot reach. A job
  whose outputs are not confirmed backed up is never deleted automatically on
  a host that persists.
- A job whose GPUs sit idle for too long is killed. After repeated such
  failures a host stops dispatching until told to resume, and a rental shuts
  down. Jobs may opt out or tune this. A job may set a wall-clock limit.
- Cancel, preempt and every other kill reap the job's whole process tree,
  using a cgroup where the host provides one and a process group otherwise.
- A finished or lost job can be resubmitted from the mirror as a new job on
  any host.

## Backups and secrets

- Backups need credentials the job carries as named environment variables.
- **Secrets travel only in files with restrictive permissions**: never in
  command lines, never in a provider's pod environment, never in logs, and
  never in anything synced to a backup destination. A job's secrets are
  deleted once the job is finished and its uploads are confirmed.
- Plain, non-secret environment is part of the job spec and may be mirrored.

## Monitoring

- Everything about hosts and jobs is visible from the CLI: every host's
  cards and dispatcher; every queued job in dispatch order with its projected
  start; every running job with its phase, cards, utilization and estimate;
  recent results; and each job's log.
- **Every command that reports something supports JSON output**, and exit
  codes distinguish "failed", "usage", "local state unreadable, so unknown"
  and "no such job or host". An unreachable host is data, not a failure.
- **The web app shows the same information and offers the same actions as
  the CLI, and nothing else.** A capability exists in the CLI first, and the
  web app calls the same code.
- Monitoring asks the host. The mirror is read only when the host is gone.

## Upgrades and compatibility

- **The tool can be upgraded at any time without disturbing running jobs.**
  After an upgrade, the queue is served by the new build: a dispatcher on a
  different build hands over to the one shipped to the host, and running jobs
  are adopted, never restarted. Submitting to a host on another build ships
  the current one first.
- Clients and hosts on different builds interoperate: a build ignores fields
  it does not know and never fails on a file another build wrote.

## Deployment and agents

- Nothing runs in the background on a client except the reaper and the web
  app, both optional. Each ships as a systemd user unit (Linux) that the tool
  writes but does not enable.
- The repository includes a skill describing how to use the tool, and the
  tool can print it.
- Tests never rent hardware unless explicitly asked to, and tests that need a
  local GPU are not excluded by default.

## Non-goals

- A scheduler that picks a host for a job, or moves jobs between hosts.
- Requirements beyond a GPU count: no VRAM, CPU or memory matching. A CUDA
  floor informs rental selection only.
- Multi-node jobs, spot or interruptible instances, running jobs in
  containers.
- A guaranteed rental teardown. Termination is best effort from the host,
  backed by the client-side reaper; a rental whose container never starts
  needs the reaper or a person.
- Automatic re-placement of a job after it has started running: a failure at
  that point is more likely the job's than the host's.
- Multi-user or multi-tenant operation: one user's hosts, one password on the
  web app, no TLS of its own.
- Perfect process cleanup on a host without user systemd: a daemonized
  grandchild can escape a process-group kill there.
