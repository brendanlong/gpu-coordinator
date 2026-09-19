# The README's pictures, and how to make them again

## `cli.gif`, `cli.cast`

A real session against a real host. Everything below happens in `$DEMO`
(`~/gpuc-demo` when `demo.sh` is not told otherwise) so nothing touches the
registry you actually use, and the path stays short: `gpuc submit` prints where
it synced to, and a long one wraps at 100 columns.

```sh
export DEMO=~/gpuc-demo
mkdir -p "$DEMO"
cat > "$DEMO/env.sh" <<'EOF'
export GPUC_CONFIG_DIR=$HOME/gpuc-demo/config
export XDG_DATA_HOME=$HOME/gpuc-demo/data
EOF
source "$DEMO/env.sh"
gpuc config init                                       # else every command prints a note
gpuc host add workstation --gpuc-home "$DEMO/gpuc-home"
gpuc host bootstrap workstation
```

`$DEMO/project` is a git repo with a `pyproject.toml` depending on `torch`, a
`train.py`, and the two specs `demo.sh` submits. The job has to keep a card
*busy*: the first take's loop slept between steps, so `gpuc status` honestly
reported `util 9%` and the recording looked like an idle GPU.

```python
# train.py: STEPS and TAG come from argv.
w = torch.randn(2048, 2048, device="cuda")
x = torch.randn(2048, 2048, device="cuda")
for step in range(1, STEPS + 1):
    for _ in range(250):  # ~0.6s of work, no sleep
        x = torch.tanh(x @ w) * 0.5
    torch.cuda.synchronize()
    print(f"[{TAG}] step {step}/{STEPS} loss {loss:.4f}", flush=True)
    Path("results/progress.txt").write_text(f"{step / STEPS:.3f}\n")
```

```yaml
# job.yaml; eval.yaml is the same with `priority: 30` and a shorter run, so it
# queues behind this one instead of starting beside it.
name: sft-qwen3-4b
command: uv run --no-sync python train.py 300 sft
setup: uv sync --frozen
gpus: 1
priority: 50
estimated_runtime_min: 2
progress_command: "tail -1 results/progress.txt"
progress_interval_s: 5
```

Seed one finished job first (submit the same spec with a small step count and
let it end), so `gpuc status` has a `done` line to show. Then, from this
directory:

```sh
asciinema rec --cols 100 --rows 32 -i 1.5 -c ./demo.sh cli.cast
agg --theme github-dark --font-size 15 --fps-cap 12 --speed 1.15 \
    --idle-time-limit 1.2 --last-frame-duration 4 cli.cast cli.gif
```

`demo.sh` types each command a character at a time and pauses after it. The
`sleep 25` in it is what gives the host time to sample utilization before the
last `gpuc status`; the viewer does not wait through it, because `agg
--idle-time-limit` shortens the gap at render time. (`asciinema rec -i` only
writes `idle_time_limit` into the cast header — the recorded timestamps keep
the full 25 seconds.)

`agg` is the asciinema project's own renderer; its prebuilt
`agg-x86_64-unknown-linux-gnu` needs no toolchain.

## `web-dashboard.png`, `web-logs.png`

A fleet nobody owns: three hosts, one of each kind, with job names, an IP and a
bucket that are made up. `demo_state.py` builds `HostView`s and serves
`status.document()` for them beside the dashboard's own static files, so the
page renders a fabricated world with the shipped code. From the repo root:

```sh
uv run python docs/media/demo_state.py 8747     # then screenshot 127.0.0.1:8747
```

Both were taken at 1280 wide with the browser in dark mode (Playwright's
`page.emulateMedia({colorScheme: 'dark'})`, before `goto`), and `optipng -o3`
run over the result.

Curating a state means curating every field the page reads. Leaving one null
does not leave a blank: it printed a re-bootstrap warning on every host, an
`idle undefinedm`, and a `borrowing unknown`. `docs/media` is in pyright's
`include` for the same reason — a `GpuInfo(memory_mib=…)` that pydantic
silently dropped took the VRAM off every card in the first version of these
screenshots.
