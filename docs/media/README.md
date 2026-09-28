# The README's pictures, and how to make them again

## `cli.gif`, `cli.cast`

A real session against a real host: this machine, with a GPU. From a clean
checkout (commit first; see `record_cli.sh` for why):

```sh
docs/media/setup_demo.sh     # builds $GPUC_DEMO, default ~/gpuc-demo, and seeds one finished job
docs/media/record_cli.sh     # records demo.sh, renders the gif, cancels what it left running
```

`setup_demo.sh` keeps everything in `$GPUC_DEMO` — its own registry
(`GPUC_CONFIG_DIR`, `XDG_DATA_HOME`), a host added with its own `--gpuc-home`,
and the project the specs run from — so nothing touches the registry you
actually use. Keep that path short: `gpuc submit` prints where it synced to,
and a long one wraps at 100 columns. The demo host owns one card,
`$GPUC_DEMO_GPU` (default 0), and no other dispatcher knows it is there, so pick
one nobody is using. Running the script again cancels the last run's jobs and
starts over.

`demo.sh` is the session itself: it types each command a character at a time
and pauses after it. It needs `asciinema` and `agg`, the asciinema project's
renderer, whose prebuilt `agg-x86_64-unknown-linux-gnu` needs no toolchain.

## `web-dashboard.png`, `web-logs.png`

A fleet nobody owns: three hosts, one of each kind, with job names, an IP and a
bucket that are made up. `demo_state.py` builds `HostView`s and serves
`status.document()` for them beside the dashboard's own static files, so the
page renders a fabricated world with the shipped code. From the repo root:

```sh
uv run --with playwright python docs/media/screenshot_web.py
uv run python docs/media/demo_state.py 8747     # to look at it yourself
```

Curating a state means curating every field the page reads. Leaving one null
does not leave a blank: it prints a re-bootstrap warning on every host, an
`idle undefinedm`, and a `borrowing unknown`. `docs/media` is in pyright's
`include` for the same reason: a field a model does not declare is dropped
silently at runtime, and pyright is what catches it.
