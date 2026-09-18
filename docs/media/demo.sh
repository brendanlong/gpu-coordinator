#!/usr/bin/env bash
# The session `cli.gif` records. Run it under asciinema; see README.md here.
#
# $DEMO holds a scratch registry (GPUC_CONFIG_DIR, XDG_DATA_HOME), a host added
# with `--gpuc-home $DEMO/gpuc-home`, and a project with two job specs. Keep the
# path short: `gpuc submit` prints where it synced to, and a long one wraps.
set -u

DEMO=${DEMO:-$HOME/gpuc-demo}
# shellcheck disable=SC1091
source "$DEMO/env.sh"
cd "$DEMO/project" || exit 1

PROMPT='\033[1;32m$\033[0m '

type_out() {
  printf "$PROMPT"
  local text="$1" i
  for ((i = 0; i < ${#text}; i++)); do
    printf '%s' "${text:i:1}"
    sleep 0.035
  done
  printf '\n'
  sleep 0.4
}

run() {
  type_out "$1"
  eval "$1"
  sleep "${2:-1.5}"
  printf '\n'
}

clear
run 'cat job.yaml' 3
run 'gpuc submit job.yaml --host workstation' 1.5

JOB_ID=$(gpuc status --json | python3 -c '
import json, sys
hosts = json.load(sys.stdin)["hosts"]
jobs = [j for h in hosts for j in h["running"] + h["queued"]]
print(jobs[0]["job_id"] if jobs else "")
')

run 'gpuc submit eval.yaml --host workstation' 1.5
run 'gpuc status' 4

# Recorded as idle and compressed by asciinema's -i: it is here so the host has
# sampled utilization by the time the last status prints.
sleep 25

run "timeout 11 gpuc logs $JOB_ID -f -n 12" 1.5
run 'gpuc status' 4
