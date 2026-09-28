#!/usr/bin/env bash
# Builds $DEMO for `record_cli.sh`: a scratch registry, a local host with its
# own gpuc home, a project with the two specs `demo.sh` submits, and one
# finished job so `gpuc status` has a `done` line. Needs a local GPU.
#
# The gpuc it puts on PATH is this checkout's, so the recording shows the code
# in the tree and bootstrap ships it. Rerunning starts over, so do it once
# `record_cli.sh` has cancelled the jobs it left and the dispatcher has idled out.
set -euo pipefail

REPO=$(cd "$(dirname "$0")/../.." && pwd)
DEMO=${DEMO:-$HOME/gpuc-demo}

rm -rf "$DEMO"
mkdir -p "$DEMO/project"
(cd "$REPO" && uv sync --frozen -q)

cat >"$DEMO/env.sh" <<EOF
export GPUC_CONFIG_DIR=$DEMO/config
export XDG_DATA_HOME=$DEMO/data
export PATH=$REPO/.venv/bin:\$PATH
EOF
# shellcheck disable=SC1091
source "$DEMO/env.sh"

cd "$DEMO/project"
cat >pyproject.toml <<'EOF'
[project]
name = "demo"
version = "0"
requires-python = ">=3.11"
dependencies = ["torch"]
EOF

# The job has to keep the card busy, with no sleep in its loop, or `gpuc
# status` reports low utilization and the recording looks like an idle GPU.
cat >train.py <<'EOF'
import math
import sys
from pathlib import Path

import torch

STEPS, TAG = int(sys.argv[1]), sys.argv[2]
Path("results").mkdir(exist_ok=True)
w = torch.randn(2048, 2048, device="cuda")
x = torch.randn(2048, 2048, device="cuda")
for step in range(1, STEPS + 1):
    for _ in range(250):
        x = torch.tanh(x @ w) * 0.5
    torch.cuda.synchronize()
    loss = 0.2 + 2.3 * math.exp(-step / 70)
    print(f"[{TAG}] step {step}/{STEPS} loss {loss:.4f}", flush=True)
    Path("results/progress.txt").write_text(f"{step / STEPS:.3f}\n")
EOF

spec() {
  cat <<EOF
name: $1
command: uv run --no-sync python train.py $2 $3
setup: uv sync --frozen
gpus: 1
priority: $4
estimated_runtime_min: $5
progress_command: "tail -1 results/progress.txt"
progress_interval_s: 5
EOF
}
spec sft-qwen3-4b 300 sft 50 2 >job.yaml
# A higher priority than job.yaml, so it queues behind the running job rather
# than starting beside it on a free card.
spec eval-checkpoints 60 eval 30 1 >eval.yaml
spec sft-qwen3-4b-lr1e5 20 sft 50 1 >seed.yaml
printf 'results/\n.venv/\n' >.gitignore
uv lock -q
git init -q
git add -A
git commit -qm demo

gpuc config init >/dev/null
gpuc host add workstation --gpuc-home "$DEMO/gpuc-home"
gpuc host bootstrap workstation
seed=$(gpuc submit seed.yaml --host workstation --json | python3 -c 'import json, sys; print(json.load(sys.stdin)["job_id"])')
gpuc wait "$seed"
