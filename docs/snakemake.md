# Snakemake

gpuc ships a Snakemake executor plugin. `snakemake --executor gpuc` submits
every Snakemake job that needs a GPU as a gpuc job and runs the rest on the
controller, so a workflow of train, evaluate and retrain for every seed is a
Snakefile, not a job graph of your own. Snakemake decides what to run and in
what order; gpuc decides which cards each job gets.

## Install

Nothing gpuc-specific has to be in the project. The plugin is part of the
`gpu-coordinator` distribution, and Snakemake finds it in whatever
environment the controller runs in:

```sh
uv run --with "gpu-coordinator @ git+https://github.com/brendanlong/gpu-coordinator" \
  snakemake --executor gpuc --gpuc-host spar --jobs 20
```

That also puts `gpuc` on the controller's `PATH`. Putting `gpu-coordinator`
in the project with `uv add --dev` works as well.

Each GPU job runs `snakemake` again, inside its own gpuc workdir and the
project's environment, so **Snakemake has to be a dependency of the project**
(a dev dependency is fine: `uv sync --frozen` installs those). Anything else
the job-side `snakemake` needs, such as a storage plugin, can come the same
way as the plugin: `--gpuc-python "uv run --no-sync --with <package> python"`
([object storage](#several-hosts-object-storage) below).

Add `.snakemake/` to `.gitignore`. Each submit copies the working tree the way
`gpuc submit` always does, and without the entry that copy includes
Snakemake's metadata and logs.

## Running a workflow

Run Snakemake from the project root, without `--directory` (which is
refused):

```sh
uv run snakemake --executor gpuc --gpuc-host spar --jobs 20
```

(with `--with` as above if gpuc is not in the project).

**Run it in tmux** (or another session that outlives your terminal). The
controller is an ordinary foreground process: nothing in gpuc keeps it alive,
and a controller that dies leaves its jobs behind (see [a killed
controller](#a-killed-controller) below).

`--jobs` is how many gpuc jobs are queued or running at once. The host's
queue decides how many of those actually run.

| setting | default | meaning |
| --- | --- | --- |
| `--gpuc-host NAME` | none | the host to queue jobs on. Required unless every rule sets `host` or `runpod` |
| `--gpuc-setup CMD` | `uv sync --frozen` | each job's `setup`; `""` runs none |
| `--gpuc-python CMD` | `uv run --no-sync python` | runs the job's `snakemake` and is the spec's `python` |
| `--gpuc-gpuc CMD` | `gpuc` | how to run gpuc on this machine, e.g. `"uv run gpuc"` |

**A rule goes to gpuc only if it needs a GPU**, which is how a Snakefile
written for the local executor already says it: `resources: gpu=1` or more.
A rule with no `gpu`, or `gpu=0`, runs on the controller the way a
`localrule: True` does, with no gpuc job, card or copy of the tree. A rule
that sets `host` or `runpod` without `gpu` is refused before anything runs,
though a dry run (`-n`) does not check. A `gpu` given as a function is judged per job, and one that comes to 0 fails
that job: whether a rule runs on the controller is decided per rule.

The resources of a GPU rule become job spec fields:

| resource | becomes |
| --- | --- |
| `gpu` | `gpus` |
| `host` | `--host`, overriding `--gpuc-host` |
| `runpod` | `gpuc submit --runpod --gpu <value> --gpu-count <gpu>` |
| `vram_gb` | `--min-vram`, with `runpod` |
| `priority` | `priority` |
| `max_runtime_min` | `max_runtime_min` |
| `use_shared`, `auto_preempt` | the spec fields; `1`, `true`, `yes` and `on` are true |

```python
rule train:
    output: "checkpoints/{seed}/model.pt"
    resources: gpu=1, priority=40, max_runtime_min=600
    shell: "uv run --no-sync python -m train --seed {wildcards.seed} --out {output}"
```

Each gpuc job is named after its rule and wildcards (`train[seed=3]`), so
`gpuc status` and `gpuc logs` say which it is. A failed gpuc job fails its
Snakemake job, and the error names the gpuc job id to pass to `gpuc logs`.
Ctrl-C on Snakemake runs `gpuc cancel` on every job it has in flight.

**Secrets.** `snakemake --envvars NAME` reaches the job as a gpuc
[secret](usage.md#the-job-spec): the value is read from your shell at submit
and delivered to the host in a 0600 file. It never appears in the job's
command. A storage plugin's credentials are passed the same way.

**Status.** Every poll is one `gpuc status --json` naming every job in
flight, and Ctrl-C is one `gpuc cancel` naming them all. A job whose host
could not be asked stays in flight, and the reason is printed. A job its host
no longer has, or whose host is gone with no end recorded in the mirror, has
failed. A job on a rental that ended after finishing gets its outcome from the
mirror.

<a name="a-killed-controller"></a>
**A killed controller.** Ctrl-C cancels the workflow's gpuc jobs, but a
controller that is killed outright leaves them queued and running. Started
again, it submits them again. Cancel the old ones first; `gpuc status` lists
them by rule name.

## Where files live

Each gpuc job runs in its own copy of the project, taken when the job is
submitted. Snakemake's controller still looks for each job's outputs after the
job ends, so a relative output path written inside the job's copy is never
seen. Two layouts work.

### One machine: a shared directory

Run the controller on the GPU host, with that host registered there as
`local`, and give every input and output an absolute path outside the job
workdirs. Registering it on itself is safe on a host another machine already
drives: `gpuc host add local` is a [connect](setup.md#registering-hosts), so
it adopts the same config and queue. Rules without a GPU run there too, next
to the outputs.

```python
R = "/home/me/myproject-results"

rule train:
    output: R + "/checkpoints/{seed}/model.pt"
    shell: "uv run --no-sync python -m train --seed {wildcards.seed} --out {output}"
```

Rule commands still run in the job's own copy of the project, so
`uv run` finds the job's environment and code. Don't use a `workdir:`
directive for this: it would run every rule command in the results directory
instead. A results directory inside the project has to be in `.gitignore`, or
every submit copies it. This layout needs no storage plugin and nothing leaves
the machine.

### Several hosts: object storage

Use a Snakemake storage plugin for inputs and outputs and tell Snakemake there
is no shared filesystem, so each job downloads its inputs and uploads its
outputs itself:

```sh
export SNAKEMAKE_STORAGE_S3_ACCESS_KEY=... SNAKEMAKE_STORAGE_S3_SECRET_KEY=...
uv run --with "gpu-coordinator @ git+https://github.com/brendanlong/gpu-coordinator" \
       --with snakemake-storage-plugin-s3 \
  snakemake --executor gpuc --gpuc-host spar --jobs 20 \
  --gpuc-python "uv run --no-sync --with snakemake-storage-plugin-s3 python" \
  --shared-fs-usage none \
  --default-storage-provider s3 --default-storage-prefix s3://my-bucket/myproject
```

The storage plugin is needed on both sides: the first `--with` is the
controller's, and `--gpuc-python` gives each job its own. The plugin never
installs one into the job by itself. Pin the same version in both, or put the
storage plugin in the project instead.

This works across local, ssh and RunPod hosts in one workflow, at the cost of
moving every checkpoint through the bucket.

## Limits

- Every GPU job is one `gpuc submit`, and each submit copies the working tree
  and runs `setup`.
- Job groups (`group:`) are refused, CPU rules included: Snakemake never runs
  a grouped rule on the controller. Each Snakemake job is its own gpuc job.
- The controller runs wherever you start it, and a controller that dies
  leaves its jobs behind. gpuc runs only jobs that need a GPU, so it cannot
  host the controller.
