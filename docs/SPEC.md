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
  of them may drive any host. A client need not have a GPU of its own.
- **Host**: a machine that runs jobs. Three kinds: **local** (the client
  itself), **ssh** (a machine reached over SSH, typically shared with other
  people and with no sudo), and **rental** (a machine rented from a provider
  for as long as there is work).
- **Dispatcher**: the process on a host that owns its queue.
- **Owned GPU**: a card the host may use unconditionally. **Shared GPU**: a
  card the host may borrow under the conditions in *Queueing*. Both are named
  by nvidia-smi index or by UUID. Any other card on the machine is never used,
  and is reported only where the user is choosing cards or debugging.
- **Job**: one spec (command, GPU count, priority, options) submitted to one
  host, identified by an id that is unique across all hosts.
- **Backup destination**: a remote store a job's outputs are copied to.
- **Kept output**: an output with no backup destination. It stays on its
  host, where the job wrote it.
- **Data directory**: a directory on a host that every job there can read and
  write, for what one job keeps for a later one: a dataset it downloaded, a
  checkpoint the next step starts from.
- **Mirror**: an S3 location holding a copy of what must survive a host: job
  specs, the job index, and each job's log and state. It is a copy, never the
  queue.

RunPod is the first rental provider, and S3 and Hugging Face the first backup
destinations. Adding another of either changes nothing else in this document.

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
  one exception: the GPU check at job start runs inside the job's own
  environment and expects a uv project with torch.

## Hosts and provisioning

- **The client owns provisioning until the host proves healthy.** Any failure
  before that point is the client's to clean up; for a rental that means
  terminating it, then trying the next offer within a time ceiling.
- **Health is checked before any job runs on a host**: the driver, every
  owned and shared GPU resolves to a present card and none is in both lists,
  free disk, and a basic network throughput test.
- **After that the host owns itself**: its queue, its job state, its logs, its
  configuration and, for a rental, its own shutdown. A rental terminates
  itself once its queue has been empty for a configured idle period, after
  draining its uploads. Nothing on a client watches a rental after handoff.
- **A user can end a rental**, in one command, whether or not it's still busy.
- **A rental that has ended is a state, not a failure.** A client that finds
  the pod gone forgets its record of that host and says so.
- **Any client whose SSH key reaches a host can drive it, without conflict**,
  and sees the same queue, jobs and configuration. Nothing about the client that set the host up
  matters afterwards, including for a rental another machine rented.
- The client's record of a host is an address plus a cache. Anything that
  decides something asks the host; output from the cache is labelled with its
  age.
- Connecting selects which cards are owned and which are shared. By default
  a host owns every card it has.
- An existing matching rental is reused before a new one is created. Rentals
  that are not ours are never touched.

## Queueing

- **A job is queued on the host the user names.**
- **A job has exactly one owner at a time.** The client owns it until the
  host has accepted it into its queue; from then on the host owns it. Before
  acceptance the host never runs the job. A submit that dies before
  acceptance leaves nothing the host will run, and what it staged is
  removed.
- A job states its requirements as a number of GPUs, which may be none. A job
  asking for more cards than the host is configured with, counting shared
  cards only if the job opted into them, is refused at submit and fails at
  dispatch if the configuration shrinks afterwards.
- **Priority is numeric, lower first, and strict.** The queue is taken in
  order: a job that does not yet fit holds the free cards it is waiting for,
  and nothing behind it may take them, even at the cost of idle cards. What is
  held is those cards, not the queue: a job behind that needs none of them
  goes ahead. A job waiting for a shared card
  someone else is using is stepped over instead.
- Priorities of queued jobs can be changed, and the queue reorders
  accordingly.
- **A running job can be preempted** so that a job ahead of it in dispatch
  order can run, by command or automatically for jobs that opt in -- and
  automatically only for a strictly higher-priority job. Preemption restarts
  the job from the beginning in its existing working tree; checkpointing is the
  job's business. A preempt is refused when nothing waiting for its cards
  would be dispatched ahead of the preempted job.
- **Shared GPUs** are used only by jobs that opt in, only after every free
  owned card, and only while nvidia-smi reports the card idle. Owned cards are
  trusted to have no other users. A borrowed card is held until the job ends;
  nothing detects a later collision.

## Running a job

- The job's code is the working tree the user submits from, including
  untracked and uncommitted changes. The checkout is code, not data: a job
  fetches datasets itself, and may keep them in the host's data directory for
  the jobs after it. Nothing empties the data directory except a person.
- **Checks happen as early as they can.** At submit: the spec is valid, every
  secret it names is present, and its GPU count fits the host. At job start,
  before the main phase: its GPUs work inside the job's own environment, and
  every backup destination is writable with the job's own credentials.
- A job runs as setup, those checks, main, and a final upload, and may
  report progress or an estimated remaining time. Estimates are informational
  and never change a job's outcome.
- **Outputs with a backup destination are backed up continuously** during the
  run, not only at the end, and so are logs. Files that were already in the
  checkout are never uploaded as results. Every output location includes the
  job id, so runs never overwrite each other.
- **Kept outputs stay on their host until a person removes them.** A rental
  refuses them at submit.
- On success, uploads finish and the checkout is deleted; kept outputs stay
  where the job wrote them. On failure or cancellation the checkout is kept
  for a configurable period. A rental's shutdown
  overrides that period, and one that has retried its uploads and still cannot
  deliver them terminates anyway. A job whose outputs are not confirmed backed
  up is never deleted automatically on a host that persists.
- A job may set a wall-clock limit and will be terminated if it exceeds that
  time.
- Cancel, preempt and every other kill reap the job's whole process tree,
  using a cgroup where the host provides one and a process group otherwise.
  So does the end of each phase: nothing a phase started outlives it.
- A finished or lost job can be resubmitted from the mirror as a new job on
  any host, with its secrets read from the submitter's shell again.

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
  recent results, each with its mean utilization over its main phase; and
  each job's log.
- **A user can copy a job's outputs from its host to their machine**, while
  it runs or after, for as long as its workdir is on the host.
- **A user can wait for jobs to end**, in one command that reports what
  happened to each and exits with their outcome.
- **Asking after, fetching, waiting for, cancelling, preempting, reordering
  and estimating jobs each take any number of job ids** in one command, at a
  round trip per host rather than per job.
- **A command does as much as it can, says what it could not do, and exits
  non-zero if anything failed.** One host that cannot be reached never stops
  the others being reported, one job id that cannot be found or acted on never
  stops the others, and the reason is never hidden.
- **Every command that reports something supports JSON output**, and exit
  codes distinguish "failed", "usage", "local state unreadable, so unknown"
  and "no such job or host".
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

- Nothing runs in the background on a client except the optional web app,
  which ships as a systemd user unit (Linux) that the tool writes but does
  not enable.
- The repository includes a skill describing how to use the tool, and the
  tool can print it.
- The distribution includes a Snakemake executor plugin that queues each
  Snakemake job that needs a GPU as a job on a named host, and leaves the
  rest to run where Snakemake runs. It drives the CLI and does nothing the
  CLI cannot.
- Tests never rent hardware unless explicitly asked to, and tests that
  require a local GPU skip if there is none available. Local GPU tests must
  use minimal resources.

## Non-goals

- A scheduler that picks a host for a job, or moves jobs between hosts.
- Requirements beyond a GPU count: no VRAM, CPU or memory matching. A CUDA
  floor informs rental selection only.
- Multi-node jobs, spot or interruptible instances, running jobs in
  containers.
- A guaranteed rental teardown. A rental ends itself when idle; one whose
  provisioning client died before terminating it, or whose dispatcher dies
  after handoff, bills until a person ends it.
- Judging whether a job used its GPU efficiently. Its utilization is reported;
  what that number should have been is the user's call.
- Spending limits across rentals.
- Automatic re-placement of a job after it has started running: a failure at
  that point is more likely the job's than the host's.
- Multi-user or multi-tenant operation: one user's hosts, one password on the
  web app, no TLS of its own.
- Perfect process cleanup on a host without user systemd: a daemonized
  grandchild can escape a process-group kill there.
