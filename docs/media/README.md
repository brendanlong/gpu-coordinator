# The README's pictures, and how to make them again

## `cli.gif`, `cli.cast`

A real session against a real host: a throwaway `GPUC_CONFIG_DIR`, a host added
with `--gpuc-home` somewhere scratch, and a job whose `command` keeps one card
busy so `gpuc status` has a utilization to show. Recorded and rendered with:

```sh
asciinema rec --cols 100 --rows 32 -i 1.5 -c ./demo.sh cli.cast
agg --theme github-dark --font-size 15 --fps-cap 12 --speed 1.15 \
    --idle-time-limit 1.2 --last-frame-duration 4 cli.cast cli.gif
```

`demo.sh` types each command a character at a time and pauses after it; the
commands themselves are ordinary `gpuc`. Sleeps between them are recorded as
idle and compressed by `-i`, so waiting for the host to sample utilization
costs the viewer a second.

## `web-dashboard.png`, `web-logs.png`

A fleet nobody owns: three hosts, one of each kind, with job names and a bucket
that are made up. `demo_state.py` builds `HostView`s and serves
`status.document()` for them beside the dashboard's own static files, so the
page renders a fabricated world with the shipped code.

```sh
uv run python docs/media/demo_state.py 8747     # then screenshot 127.0.0.1:8747
```

Both were taken at 1280 wide with the browser in dark mode, and `optipng -o3`
run over the result.
