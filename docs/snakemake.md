# Snakemake

gpuc ships a Snakemake executor plugin. `snakemake --executor gpuc` submits
every Snakemake job as a gpuc job, so a workflow of train, evaluate and
retrain for every seed is a Snakefile, not a job graph of your own. Snakemake
decides what to run and in what order; gpuc decides which cards each job gets.

## Install

The plugin is part of the `gpu-coordinator` distribution. Put gpuc and
Snakemake in the project:

```sh
uv add --dev "gpu-coordinator @ git+https://github.com/brendanlong/gpu-coordinator" snakemake
```

Each job runs `snakemake` again, inside its own gpuc workdir and the project's
environment, so **Snakemake has to be a dependency of the project** (a dev
dependency is fine: `uv sync --frozen` installs those). So does any storage
plugin the workflow uses: the plugin never `pip install`s one into the job.

Add `.snakemake/` to `.gitignore`. Each submit copies the working tree the way
`gpuc submit` always does, and without the entry that copy includes
Snakemake's metadata and logs.

## Running a workflow

Run Snakemake from the project root:

```sh
uv run snakemake --executor gpuc --gpuc-host spar --jobs 20
```

`--jobs` is how many gpuc jobs are queued or running at once. The host's
queue decides how many of those actually run.

| setting | default | meaning |
| --- | --- | --- |
| `--gpuc-host NAME` | none | the host to queue jobs on. Required unless every rule sets `host` or `runpod` |
| `--gpuc-setup CMD` | `uv sync --frozen` | each job's `setup`; `""` runs none |
| `--gpuc-python CMD` | `uv run --no-sync python` | runs the job's `snakemake` and is the spec's `python` |
| `--gpuc-gpuc CMD` | `gpuc` | how to run gpuc on this machine, e.g. `"uv run gpuc"` |

Rule resources become job spec fields:

| resource | becomes |
| --- | --- |
| `gpu` | `gpus` (default 1) |
| `host` | `--host`, overriding `--gpuc-host` |
| `runpod` | `gpuc submit --runpod --gpu <value> --gpu-count <gpu>` |
| `vram_gb` | `--min-vram`, with `runpod` |
| `priority` | `priority` |
| `max_runtime_min` | `max_runtime_min` |
| `use_shared`, `auto_preempt` | the spec fields; `1`, `true` and `yes` are true |

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

**Status.** Every poll is one `gpuc status --json` covering every job in
flight, restricted with `--host` when they are all on one host. A job whose
host could not be asked is waited for. A job no host lists any more is looked
up with `gpuc wait`, which reads the S3 mirror for a rental that has ended.

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
`local`, and put every input and output in a directory outside the job
workdirs:

```python
workdir: "/home/me/myproject/results"
```

A `workdir:` directive (or absolute paths) points the controller and every
job at the same directory. The code each job runs is still its own copy of the
tree Snakemake was started from. This needs no storage plugin and nothing
leaves the machine.

### Several hosts: object storage

Use a Snakemake storage plugin for inputs and outputs and tell Snakemake there
is no shared filesystem, so each job downloads its inputs and uploads its
outputs itself:

```sh
uv add --dev snakemake-storage-plugin-s3
export SNAKEMAKE_STORAGE_S3_ACCESS_KEY=... SNAKEMAKE_STORAGE_S3_SECRET_KEY=...
uv run snakemake --executor gpuc --gpuc-host spar --jobs 20 \
  --shared-fs-usage none \
  --default-storage-provider s3 --default-storage-prefix s3://my-bucket/myproject
```

This works across local, ssh and RunPod hosts in one workflow, at the cost of
moving every checkpoint through the bucket. Don't use a `workdir:` directive
in this mode. The directory it names would have to exist on every host.

## Limits

- Every Snakemake job is one `gpuc submit`, and each submit copies the working
  tree.
- Every job needs a GPU. Mark a rule that doesn't need one `localrule: True`,
  so the controller runs it itself.
