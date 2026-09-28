#!/usr/bin/env bash
# Records `demo.sh` into cli.cast and renders cli.gif, after `setup_demo.sh`.
#
# `demo.sh` sleeps 25s so the host has sampled utilization before the last
# `gpuc status`. `agg --idle-time-limit` is what shortens that gap for the
# viewer: `asciinema rec -i` only writes `idle_time_limit` into the cast
# header, and the recorded timestamps keep the full 25 seconds.
#
# The cast is written outside the checkout and copied in afterwards: the build
# id of a dirty tree hashes its changes, so a file growing in it mid-recording
# makes every `gpuc` command see a new build and re-ship it to the host. Record
# from a clean tree for the same reason.
set -euo pipefail
cd "$(dirname "$0")"
DEMO=${DEMO:-$HOME/gpuc-demo}
export TERM=xterm-256color

asciinema rec --overwrite --cols 100 --rows 32 -i 1.5 \
  --title 'gpuc: submit a job, watch the queue' -c ./demo.sh "$DEMO/cli.cast"
cp "$DEMO/cli.cast" cli.cast
agg --theme github-dark --font-size 15 --fps-cap 12 --speed 1.15 \
  --idle-time-limit 1.2 --last-frame-duration 4 cli.cast cli.gif

# The recording ends with both jobs unfinished; free the card.
# shellcheck disable=SC1091
source "$DEMO/env.sh"
gpuc status --host workstation --json | python3 -c '
import json, sys
for h in json.load(sys.stdin)["hosts"]:
    for j in h["running"] + h["queued"]:
        print(j["job_id"])
' | xargs -r gpuc cancel
