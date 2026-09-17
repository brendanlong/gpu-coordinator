#!/usr/bin/env bash
# Lint, typecheck and test. CI runs this same file (.github/workflows/ci.yml),
# one step per job, so the two cannot drift.
#
#   ./check.sh                  sync, then lint + typecheck + test
#   ./check.sh --fast           the same three, without `uv sync --frozen`
#   ./check.sh --fast lint      one step only (lint | typecheck | test)
set -euo pipefail
cd "$(dirname "$0")"

sync=1
steps=()
for arg in "$@"; do
  case "$arg" in
    --fast) sync=0 ;;
    lint | typecheck | test) steps+=("$arg") ;;
    *)
      echo "usage: $0 [--fast] [lint|typecheck|test ...]" >&2
      exit 2
      ;;
  esac
done
if [ ${#steps[@]} -eq 0 ]; then
  steps=(lint typecheck test)
fi

run() {
  echo "+ $*" >&2
  "$@"
}

if [ "$sync" -eq 1 ]; then
  run uv sync --frozen
fi

for step in "${steps[@]}"; do
  case "$step" in
    lint)
      run uv run --frozen ruff check .
      run uv run --frozen ruff format --check .
      ;;
    typecheck)
      run uv run --frozen pyright gpuc tests
      ;;
    test)
      # No `-m` here: `pyproject.toml` already excludes the `runpod` tests,
      # which rent hardware and are opt-in (`uv run pytest -m runpod`). The
      # `gpu` tests run when this machine has the card they name and skip
      # themselves when it does not, which is every CI runner.
      run uv run --frozen pytest -q --durations=10
      ;;
  esac
done
