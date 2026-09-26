"""The whole control side against this machine as a `local` host, no GPU needed.

Every job needs a card, so the host is given one: a fake `nvidia-smi` on PATH
answers for it, and a stand-in `torch.py` in the checkout passes the runner's
GPU preflight. Everything else is redirected too: GPUC_CONFIG_DIR,
GPUC_STATE_DIR and the host's GPUC_HOME, so the real ~/.gpuc and the real
registry are never touched.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import subprocess
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from gpuc.control.cli import main
from gpuc.control.s3index import LocalIndex
from gpuc.host import scope
from tests.conftest import FAKE_GPUS, install_fake_nvidia_smi, install_fake_torch, load_registry

HEALTH_ARGS = "--min-mbps 0.05 --min-free-gb 1"


def wait_until(predicate: Callable[[], bool], timeout: float, what: str) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(0.2)
    raise AssertionError(f"timed out after {timeout}s waiting for {what}")


def state_of(home: Path, job_id: str) -> dict[str, object]:
    path = home / "jobs" / job_id / "state.json"
    if not path.exists():
        return {}
    try:
        document: dict[str, object] = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}
    return document


def log_tail(home: Path, job_id: str, lines: int = 25) -> str:
    path = home / "jobs" / job_id / "log.txt"
    return "\n".join(path.read_text().splitlines()[-lines:]) if path.exists() else "(no log)"


def finished(home: Path, job_id: str) -> bool:
    return state_of(home, job_id).get("status") in ("succeeded", "failed", "cancelled")


def wait_for_main_phase(home: Path, job_id: str, timeout: float = 600.0) -> None:
    """Used by the GPU e2e module, which has to wait out a torch venv sync."""
    wait_until(
        lambda: state_of(home, job_id).get("phase") == "main" or finished(home, job_id),
        timeout,
        f"job {job_id} to reach phase=main",
    )
    assert state_of(home, job_id)["status"] == "running", log_tail(home, job_id)


@pytest.fixture
def workdir(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    root.mkdir()
    (root / "hello.txt").write_text("hello\n")
    install_fake_torch(root)
    for argv in (
        ["git", "init", "-q"],
        ["git", "config", "user.email", "t@example.com"],
        ["git", "config", "user.name", "t"],
        ["git", "add", "-A"],
        ["git", "commit", "-qm", "init"],
    ):
        subprocess.run(argv, cwd=root, check=True, capture_output=True)
    return root


SHARED_UV_CACHE = str(Path.home() / ".cache/uv")
"""Pin the tests to this machine's real uv cache.

A temp gpuc home is on /tmp, so bootstrap's rule would otherwise give each test
a fresh cache there and re-download torch into a tmpfs. `--cache-dir` is the
documented way to opt out of the rule, and exercising it here keeps that path
covered.
"""


@pytest.fixture(scope="module")
def bootstrapped_home(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """One bootstrapped `local` host for the whole module.

    `$HOME` and the XDG dirs are redirected too. Bootstrap installs uv, a
    Python, the aws CLI and `hf` into `$HOME/.local` when it cannot find them,
    and a test suite has no business writing there; with `$HOME` temporary it
    finds all four on `$PATH` instead and installs nothing.

    Module-scoped because a bootstrap costs ~5s and nothing below needs a
    pristine host: every test keys on the job id it submitted. A test that does
    need a fresh one re-bootstraps itself (`--health-args` makes that cheap).
    """
    root = tmp_path_factory.mktemp("control-e2e")
    fake_home = root / "home"
    (fake_home / ".local" / "bin").mkdir(parents=True)
    patch = pytest.MonkeyPatch()
    patch.setenv("HOME", str(fake_home))
    # The host's one card, which `host add` owns by default. This machine's
    # own, if it has any, are not this suite's to use.
    install_fake_nvidia_smi(fake_home / ".local" / "bin", [FAKE_GPUS[0]])
    patch.setenv("PATH", f"{fake_home / '.local' / 'bin'}{os.pathsep}{os.environ['PATH']}")
    patch.setenv("XDG_CONFIG_HOME", str(fake_home / ".config"))
    patch.setenv("XDG_DATA_HOME", str(fake_home / ".local" / "share"))
    patch.setenv("XDG_CACHE_HOME", str(fake_home / ".cache"))
    patch.setenv("GPUC_CONFIG_DIR", str(root / "config"))
    patch.setenv("GPUC_STATE_DIR", str(root / "state"))
    patch.delenv("GPUC_HOME", raising=False)
    patch.setenv(scope.ISOLATION_ENV, scope.PGID)
    (root / "config").mkdir()
    (root / "state").mkdir()

    home = root / "gpuc-home"
    try:
        assert (
            main(
                [
                    "host",
                    "add",
                    "local",
                    "--gpuc-home",
                    str(home),
                    "--cache-dir",
                    SHARED_UV_CACHE,
                ]
            )
            == 0
        )
        assert main(["host", "bootstrap", "local", "--health-args", HEALTH_ARGS]) == 0
        yield home
    finally:
        _stop_dispatcher(home)
        patch.undo()


def _stop_dispatcher(home: Path) -> None:
    """Leave nothing running on this shared machine: jobs first, then the dispatcher."""
    for state_file in sorted(home.glob("jobs/*/state.json")):
        state = state_of(home, state_file.parent.name)
        pgid = state.get("pgid") if state.get("status") == "running" else None
        runner = state.get("runner_pid") if state.get("status") == "running" else None
        for group in (pgid, runner):
            if isinstance(group, int) and group > 1:
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    os.killpg(group, 9)
    lock = home / "dispatcher.lock"
    if not lock.exists():
        return
    body = lock.read_text().strip()
    if body.isdigit():
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(int(body), 15)


def submit(workdir: Path, document: str, name: str = "job.yaml") -> str:
    """Submit from `workdir` as the user would, and return the new job id."""
    job = workdir / name
    job.write_text(document)
    before = _indexed_job_ids()
    cwd = Path.cwd()
    os.chdir(workdir)
    try:
        assert main(["submit", str(job), "--host", "local"]) == 0
    finally:
        os.chdir(cwd)
    new = _indexed_job_ids() - before
    assert len(new) == 1, f"submit recorded {len(new)} index entries"
    return new.pop()


def _indexed_job_ids() -> set[str]:
    from gpuc.control.s3index import LocalIndex

    return {entry.job_id for entry in LocalIndex().list()}


def test_bootstrap_installs_the_package_and_records_the_interpreter(
    bootstrapped_home: Path,
) -> None:
    home = bootstrapped_home
    assert (home / "pkg/gpuc/host/dispatcher.py").exists()
    config = json.loads((home / "config.json").read_text())
    assert config["host"] == "local"
    # `host add` was given no `--gpus`: every card, written as null.
    assert "gpus" in config and config["gpus"] is None
    assert (home / "secrets").stat().st_mode & 0o777 == 0o700
    entry = load_registry().require("local")
    assert FAKE_GPUS[0] in entry.gpu_info
    assert entry.python and Path(entry.python).exists()
    assert entry.bootstrapped_at


def test_probe_reports_this_machine(
    bootstrapped_home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["host", "probe", "local"]) == 0
    out = capsys.readouterr().out
    assert "host local" in out
    assert "systemd_scope:" in out


def test_submit_runs_a_job_and_logs_and_status_find_it(
    bootstrapped_home: Path, workdir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    home = bootstrapped_home
    job_id = submit(workdir, "name: hi\ncommand: cat hello.txt\ncleanup: never\n")
    capsys.readouterr()

    wait_until(lambda: finished(home, job_id), 120, f"job {job_id} to finish")
    state = state_of(home, job_id)
    assert (state["status"], state["exit_code"], state["attempt"]) == ("succeeded", 0, 1)
    assert (home / "jobs" / job_id / "workdir" / "hello.txt").exists()

    assert main(["logs", job_id]) == 0
    assert "hello" in capsys.readouterr().out

    assert main(["status", "--host", "local"]) == 0
    status = capsys.readouterr().out
    assert job_id in status
    assert "succeeded" in status


def test_a_job_writes_to_the_data_dir_and_host_clean_removes_it(
    bootstrapped_home: Path, workdir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    home = bootstrapped_home
    job_id = submit(workdir, 'name: keep\ncommand: echo kept > "$GPUC_DATA_DIR/kept.txt"\n')
    wait_until(lambda: finished(home, job_id), 120, f"job {job_id} to finish")
    assert state_of(home, job_id)["status"] == "succeeded", log_tail(home, job_id)
    assert (home / "data" / "kept.txt").read_text() == "kept\n"
    capsys.readouterr()

    code = main(["host", "clean", "local", "--data", "kept.txt", "--data", "absent", "--json"])
    document = json.loads(capsys.readouterr().out)
    assert code == 1
    assert document["data"]["removed"] == [{"path": "kept.txt", "freed_bytes": 5}]
    assert document["data"]["errors"] == [f"absent: not in {home / 'data'}"]
    assert not (home / "data" / "kept.txt").exists()


def test_fetch_copies_a_jobs_results_and_not_what_came_with_the_checkout(
    bootstrapped_home: Path, workdir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    home = bootstrapped_home
    (workdir / "results").mkdir()
    (workdir / "results" / "old.md").write_text("from the checkout\n")
    subprocess.run(["git", "add", "-A"], cwd=workdir, check=True)
    subprocess.run(["git", "commit", "-qm", "old"], cwd=workdir, check=True)
    job_id = submit(
        workdir,
        "name: produce\n"
        "command: echo new > results/new.txt\n"
        "outputs: [{path: results}]\n"
        "cleanup: never\n",
    )
    wait_until(lambda: finished(home, job_id), 120, f"job {job_id} to finish")
    assert state_of(home, job_id)["status"] == "succeeded", log_tail(home, job_id)
    capsys.readouterr()

    unknown = "20260101-000000-aaaaaa"
    to = tmp_path / "fetched"
    assert main(["fetch", job_id, unknown, "--host", "local", "--to", str(to), "--json"]) == 4
    document = json.loads(capsys.readouterr().out)
    fetched, missing = document["jobs"]
    assert [f["path"] for f in fetched["files"]] == ["results/new.txt"]
    assert fetched["to"] == str(to / job_id)
    assert missing["error"]
    assert (to / job_id / "results" / "new.txt").read_text() == "new\n"
    assert not (to / job_id / "results" / "old.md").exists()

    assert main(["fetch", job_id, "--path", "results", "--list"]) == 0
    listed = capsys.readouterr().out
    assert "results/old.md" in listed and "results/new.txt" in listed


def test_a_kept_output_outlives_the_sweep_and_status_and_fetch_find_it(
    bootstrapped_home: Path, workdir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    home = bootstrapped_home
    job_id = submit(
        workdir,
        "name: keeper\ncommand: mkdir -p results && echo kept > results/r.txt\n"
        "outputs: [{path: results}]\n",
    )
    wait_until(lambda: finished(home, job_id), 120, f"job {job_id} to finish")
    assert state_of(home, job_id)["status"] == "succeeded", log_tail(home, job_id)
    job_workdir = home / "jobs" / job_id / "workdir"
    # `cleanup: on_success` took the checkout and left the kept output.
    assert sorted(p.name for p in job_workdir.iterdir()) == ["results"]
    assert "removed the checkout, keeping results" in log_tail(home, job_id, 50)
    capsys.readouterr()

    assert main(["status", "--host", "local"]) == 0
    out = capsys.readouterr().out
    assert "kept on host: results" in out
    assert "keep outputs on this host" in out
    assert main(["fetch", job_id, "--to", str(tmp_path)]) == 0
    assert (tmp_path / job_id / "results" / "r.txt").read_text() == "kept\n"


def test_wait_blocks_on_a_real_job_and_exits_with_its_outcome(
    bootstrapped_home: Path, workdir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`gpuc wait` is the `wait_until` loop above, made a command.

    Against the real dispatcher, because what it has to read correctly is the
    state file another process writes as the job ends.
    """
    job_id = submit(workdir, "name: doomed\ncommand: sh -c 'echo working; exit 3'\n")
    capsys.readouterr()

    assert main(["wait", job_id, "--interval", "0.5"]) == 1
    out = capsys.readouterr().out
    assert job_id in out
    assert "failed" in out
    assert finished(bootstrapped_home, job_id)


def test_following_a_real_job_stops_when_the_job_does(
    bootstrapped_home: Path, workdir: Path, capfd: pytest.CaptureFixture[str]
) -> None:
    """The issue itself: `logs -f` used to stream happily past the end of the run.

    `capfd`, not `capsys`: the tail is a child process writing to the real
    descriptor, and it is the whole point of the test.
    """
    job_id = submit(
        workdir, "name: chatty\ncommand: sh -c 'echo line-one; sleep 1; echo line-two'\n"
    )
    capfd.readouterr()

    assert main(["logs", job_id, "-f", "--interval", "0.5"]) == 0
    out = capfd.readouterr().out
    assert "line-two" in out, out
    # The outcome is the last line, because it is printed after the stream has
    # been stopped. `removed workdir` is somewhere above it: the runner logs
    # that *after* writing the terminal state this wait read, so its presence
    # is the flush grace doing its job. Not asserted at a fixed position -- a
    # loaded machine can take longer over that cleanup than the grace allows,
    # and losing a line from the stream is not a failure of anything.
    lines = out.strip().splitlines()
    assert lines[-1].startswith(f"chatty ({job_id}) on local: succeeded after "), lines[-3:]
    assert "removed workdir" in out


def test_submit_says_where_in_the_queue_the_job_landed(
    bootstrapped_home: Path, workdir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The submitter's actual question -- when does it run -- answered by the
    host it was just queued on, rather than left to a second command."""
    capsys.readouterr()
    # Long enough that it is still queued or running when the placement is
    # looked up: a job that had already *finished* would be neither, and this
    # test would pass on an empty line.
    job_id = submit(workdir, "name: placed\ncommand: sleep 300\n")
    out = capsys.readouterr().out
    assert f"job {job_id} queued on host local" in out
    # The card is free, so the job is dispatchable the moment it is queued and
    # both answers are honest: the dispatcher the enqueue started may have
    # taken it already.
    assert "queue: position 1 of 1; starts now" in out or "dispatched already" in out
    # The host is shared with every other test in this module: leave it idle.
    assert main(["cancel", job_id]) == 0
    wait_until(lambda: finished(bootstrapped_home, job_id), 120, f"job {job_id} to finish")


def test_a_running_job_can_be_cancelled(
    bootstrapped_home: Path, workdir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    home = bootstrapped_home
    job_id = submit(workdir, "name: sleepy\ncommand: sleep 300\npriority: 50\n")
    capsys.readouterr()

    wait_until(
        lambda: state_of(home, job_id).get("status") == "running", 60, "the job to start running"
    )
    assert main(["cancel", job_id]) == 0
    wait_until(lambda: finished(home, job_id), 120, "the job to be cancelled")
    assert state_of(home, job_id)["status"] == "cancelled"


def test_preempt_refuses_when_it_would_only_re_run_the_same_job(
    bootstrapped_home: Path, workdir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Nothing is waiting, so this would stop the job and start it again --
    losing everything it had done, to no end. (Where it *is* worth doing needs
    a card to contend over: see the GPU end-to-end module.)"""
    home = bootstrapped_home
    job_id = submit(workdir, "name: lonely\ncommand: sleep 300\n")
    wait_until(
        lambda: state_of(home, job_id).get("status") == "running", 60, "the job to start running"
    )
    capsys.readouterr()

    assert main(["preempt", job_id]) == 1
    assert "nothing else is queued" in capsys.readouterr().err
    assert state_of(home, job_id)["status"] == "running"
    assert not (home / "jobs" / job_id / "preempt").exists()

    assert main(["cancel", job_id]) == 0
    wait_until(lambda: finished(home, job_id), 120, "the job to be cancelled")


def test_estimate_reaches_a_running_job_and_status_and_json_agree(
    bootstrapped_home: Path, workdir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The bug this command exists for: a job submitted without an estimate,
    given one while it runs. The text and `--json` must say the same thing --
    a scripted caller seeing an estimate the operator cannot is the worst of it.
    """
    home = bootstrapped_home
    job_id = submit(workdir, "name: sleepy\ncommand: sleep 300\n")
    wait_until(
        lambda: state_of(home, job_id).get("status") == "running", 60, "the job to start running"
    )
    capsys.readouterr()

    assert main(["estimate", job_id, "--minutes", "150"]) == 0
    assert "150 min" in capsys.readouterr().out
    assert state_of(home, job_id)["estimated_runtime_min"] == pytest.approx(150.0)

    assert main(["status", "--host", "local"]) == 0
    text = capsys.readouterr().out
    assert main(["status", "--host", "local", "--json"]) == 0
    document = json.loads(capsys.readouterr().out)
    (running,) = document["hosts"][0]["running"]
    assert running["estimated_runtime_min"] == 150.0
    assert "2h30m" in text

    assert main(["cancel", job_id]) == 0
    wait_until(lambda: finished(home, job_id), 120, "the job to be cancelled")
    assert main(["estimate", job_id, "--minutes", "10"]) == 1
    assert "already cancelled" in capsys.readouterr().err


def test_a_failing_job_keeps_its_exit_code(bootstrapped_home: Path, workdir: Path) -> None:
    home = bootstrapped_home
    job_id = submit(workdir, "name: nope\ncommand: exit 23\n")
    wait_until(lambda: finished(home, job_id), 120, "the job to fail")
    state = state_of(home, job_id)
    assert (state["status"], state["exit_code"]) == ("failed", 23)


@pytest.fixture
def s3_bucket_configured() -> Iterator[None]:
    """`s3_bucket = "bkt"`, undone afterwards.

    The config dir is shared by the whole module now, and the purge tests below
    assert this host has *no* S3 mirror to fall back on.
    """
    from gpuc.control import config

    path = config.config_file()
    before = path.read_text() if path.exists() else None
    path.write_text('s3_bucket = "bkt"\n')
    try:
        yield
    finally:
        if before is None:
            path.unlink(missing_ok=True)
        else:
            path.write_text(before)


def test_requeue_resubmits_from_the_s3_spec_as_a_new_job_that_names_its_origin(
    bootstrapped_home: Path,
    workdir: Path,
    s3_bucket_configured: None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from tests.fakes3 import FakeS3Client

    fake = FakeS3Client()
    monkeypatch.setattr("boto3.client", lambda service, **_: fake)

    home = bootstrapped_home
    first = submit(workdir, "name: hi\ncommand: cat hello.txt\n")
    wait_until(lambda: finished(home, first), 120, f"job {first} to finish")
    assert f"bkt/gpuc/specs/{first}.json" in fake.objects

    before = _indexed_job_ids()
    cwd = Path.cwd()
    os.chdir(workdir)
    try:
        assert main(["requeue", first, "--host", "local"]) == 0
    finally:
        os.chdir(cwd)
    assert f"requeued from {first}" in capsys.readouterr().out
    second = (_indexed_job_ids() - before).pop()
    assert second != first
    # The new job records where it came from, in the index and in the mirror
    # the next requeue would read.
    indexed = LocalIndex().get(second)
    assert indexed is not None and indexed.requeued_from == first
    mirrored = json.loads(fake.objects[f"bkt/gpuc/specs/{second}.json"])
    assert (mirrored["job_id"], mirrored["requeued_from"]) == (second, first)

    wait_until(lambda: finished(home, second), 120, f"job {second} to finish")
    state = state_of(home, second)
    # The host's `attempt` counts the launches of *this* id: a requeue is a
    # new job, so it is 1 there however many times the spec has been run.
    assert (state["status"], state["attempt"]) == ("succeeded", 1)
    assert "hello" in log_tail(home, second)


def test_a_secret_never_reaches_the_log_or_gpuc_logs(
    bootstrapped_home: Path,
    workdir: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The job may read its secret; nothing we write down may contain it."""
    canary = "gpuc-canary-9f1c2b7e-do-not-log"
    monkeypatch.setenv("WANDB_API_KEY", canary)
    home = bootstrapped_home
    job_id = submit(
        workdir,
        "name: secretive\n"
        'command: test -n "$WANDB_API_KEY" && echo the job saw its secret\n'
        "secrets: [WANDB_API_KEY]\n",
    )
    capsys.readouterr()
    wait_until(lambda: finished(home, job_id), 120, f"job {job_id} to finish")
    assert state_of(home, job_id)["status"] == "succeeded"

    assert main(["logs", job_id]) == 0
    captured = capsys.readouterr()
    assert "the job saw its secret" in captured.out
    assert canary not in captured.out + captured.err

    for path in (
        home / "jobs" / job_id / "log.txt",
        home / "jobs" / job_id / "state.json",
        home / "jobs" / job_id / "spec.json",
        home / "dispatcher.log",
    ):
        assert canary not in path.read_text(), path
    # The runner unlinks the secrets file *after* it writes the final state, so
    # `finished()` above does not mean it is gone yet.
    wait_until(
        lambda: not (home / "secrets" / f"{job_id}.env").exists(),
        30,
        f"the secrets file for {job_id} to be removed",
    )


# -- gpuc clean ---------------------------------------------------------------


def big_file_job(home: Path, workdir: Path, *, cleanup: str = "never") -> str:
    """A job that leaves something measurable in its workdir."""
    job_id = submit(
        workdir,
        f"name: bulky\ncommand: dd if=/dev/zero of=blob.bin bs=1M count=4 2>/dev/null\n"
        f"cleanup: {cleanup}\n",
    )
    wait_until(lambda: finished(home, job_id), 120, f"job {job_id} to finish")
    return job_id


def test_clean_dry_run_then_real(
    bootstrapped_home: Path, workdir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    home = bootstrapped_home
    job_id = big_file_job(home, workdir)
    blob = home / "jobs" / job_id / "workdir" / "blob.bin"
    assert blob.exists()
    capsys.readouterr()

    assert main(["clean", "--host", "local", "--all-finished", "--dry-run"]) == 0
    dry = capsys.readouterr().out
    assert job_id in dry
    assert "dry run, nothing was deleted" in dry
    assert "MiB" in dry
    assert blob.exists(), "a dry run must delete nothing"

    assert main(["clean", "--host", "local", "--all-finished"]) == 0
    real = capsys.readouterr().out
    assert job_id in real and "freed" in real
    assert not (home / "jobs" / job_id / "workdir").exists()
    assert state_of(home, job_id)["workdir_bytes"] == 0
    assert (home / "jobs" / job_id / "log.txt").exists()


def test_status_mentions_leftover_workdirs_and_clean_clears_it(
    bootstrapped_home: Path, workdir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    home = bootstrapped_home
    job_id = submit(
        workdir,
        "name: hog\ncommand: dd if=/dev/zero of=blob.bin bs=1M count=1100 2>/dev/null\n"
        "cleanup: never\n",
    )
    wait_until(lambda: finished(home, job_id), 300, f"job {job_id} to finish")
    capsys.readouterr()

    assert main(["status", "--host", "local"]) == 0
    assert "gpuc clean --host local --all-finished" in capsys.readouterr().out

    assert main(["clean", "--host", "local", "--all-finished"]) == 0
    capsys.readouterr()
    assert main(["status", "--host", "local"]) == 0
    assert "gpuc clean --host local --all-finished" not in capsys.readouterr().out


def test_clean_removes_a_leftover_incoming_dir(
    bootstrapped_home: Path, workdir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    home = bootstrapped_home
    job_id = submit(workdir, 'name: ok\ncommand: "true"\n')
    wait_until(lambda: finished(home, job_id), 120, f"job {job_id} to finish")
    staged = home / "incoming" / "20200101-000000-dead00"
    (staged / "workdir").mkdir(parents=True)
    old = time.time() - 2 * 3600
    for path in [staged, staged / "workdir"]:
        os.utime(path, (old, old))
    capsys.readouterr()

    assert main(["clean", "--host", "local", "--all-finished"]) == 0
    assert "leftover staged" in capsys.readouterr().out
    assert not staged.exists()


# -- purge --------------------------------------------------------------------


def mark_mirrored(home: Path, job_id: str, prefix: str = "s3://bucket/gpuc/local") -> None:
    """Stand in for a successful final meta sync on a host with a prefix."""
    # Under the job's own lock: the runner is still writing its last fields
    # (workdir size, the meta sync) when `finished()` first says so, and an
    # unlocked write here would be lost to its next read-modify-write.
    lock = home / "jobs" / job_id / ".lock"
    with lock.open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        path = home / "jobs" / job_id / "state.json"
        document = json.loads(path.read_text())
        document["uploads"] = [
            {"to": f"{prefix}/jobs/{job_id}", "output": None, "ok_at": document.get("ended_at")}
        ]
        path.write_text(json.dumps(document, indent=2) + "\n")


def test_purge_removes_a_mirrored_job_whole(
    bootstrapped_home: Path, workdir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    home = bootstrapped_home
    job_id = big_file_job(home, workdir)
    mark_mirrored(home, job_id)
    capsys.readouterr()

    assert main(["clean", "--host", "local", "--purge", "--older-than", "0", "--dry-run"]) == 0
    assert "WOULD PURGE" in capsys.readouterr().out
    assert (home / "jobs" / job_id / "log.txt").exists()

    assert main(["clean", "--host", "local", "--purge", "--older-than", "0"]) == 0
    assert "PURGED" in capsys.readouterr().out
    assert not (home / "jobs" / job_id).exists()


def test_purge_only_takes_one_job_and_leaves_the_rest_of_the_host_alone(
    bootstrapped_home: Path, workdir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    home = bootstrapped_home
    go = big_file_job(home, workdir)
    keep = big_file_job(home, workdir)
    mark_mirrored(home, go)
    mark_mirrored(home, keep)
    capsys.readouterr()

    # No --yes: naming the id is the confirmation.
    assert main(["clean", "--host", "local", "--purge", "--only", go]) == 0
    assert "PURGED" in capsys.readouterr().out
    assert not (home / "jobs" / go).exists()
    # Not even the venv of the job that was not named.
    assert (home / "jobs" / keep / "workdir" / "blob.bin").exists()


def test_a_typo_in_only_refuses_the_whole_selection_and_still_reports(
    bootstrapped_home: Path, workdir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The host exits 1 here, and its report is the answer -- not a transport error."""
    home = bootstrapped_home
    job_id = big_file_job(home, workdir)
    mark_mirrored(home, job_id)
    capsys.readouterr()

    selection = f"{job_id},20260101-000000-typo11"
    assert main(["clean", "--host", "local", "--purge", "--only", selection, "--json"]) == 1
    document = json.loads(capsys.readouterr().out)
    assert document["purged"] == []
    assert any("20260101-000000-typo11" in error for error in document["errors"])
    assert (home / "jobs" / job_id / "log.txt").exists()


def test_logs_and_status_after_a_purge(
    bootstrapped_home: Path, workdir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    home = bootstrapped_home
    job_id = big_file_job(home, workdir)
    mark_mirrored(home, job_id)
    assert main(["clean", "--host", "local", "--purge", "--older-than", "0"]) == 0
    capsys.readouterr()

    # No S3 mirror is configured locally, so `logs` can only say where it went.
    assert main(["logs", job_id]) == 1
    captured = capsys.readouterr()
    assert "purged from host local" in captured.err
    assert "no S3 mirror to fall back on" in captured.err
    assert "purged from host local" in captured.err

    assert main(["status", "--all"]) == 0
    out = capsys.readouterr().out
    assert "jobs known only to the index" in out
    assert job_id in out


def test_retention_days_reaches_the_host_config(
    bootstrapped_home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    home = bootstrapped_home
    # Through the host's own CLI, on the host, with no bootstrap in between:
    # `config.json` is the only copy of this setting.
    assert main(["host", "set", "local", "--retention-days", "14"]) == 0
    assert json.loads((home / "config.json").read_text())["retention_days"] == 14.0
    assert load_registry().require("local").config.retention_days == 14.0
    assert main(["host", "set", "local", "--retention-days", ""]) == 0
    assert json.loads((home / "config.json").read_text())["retention_days"] is None
    assert load_registry().require("local").config.retention_days is None


def test_workdir_days_reaches_the_host_config(
    bootstrapped_home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    home = bootstrapped_home
    assert json.loads((home / "config.json").read_text())["workdir_days"] == 1.0
    # No bootstrap between these: `host set` is a write-through to the host's
    # own config, and there is no local copy for it to have changed instead.
    assert main(["host", "set", "local", "--workdir-days", "3"]) == 0
    assert json.loads((home / "config.json").read_text())["workdir_days"] == 3.0
    assert main(["host", "set", "local", "--workdir-days", ""]) == 0
    assert json.loads((home / "config.json").read_text())["workdir_days"] is None
    assert load_registry().require("local").config.workdir_days is None


def test_logs_on_a_named_host_for_an_id_it_never_had_is_not_a_purge(
    bootstrapped_home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # A job's `name` passed where an id belongs: nothing has checked it exists,
    # and "purged" would send the user looking for a job that never was.
    assert main(["logs", "--host", "local", "ekfac-smoke"]) == 4
    err = capsys.readouterr().err
    assert "host local has no job ekfac-smoke, and that is not a job id" in err
    assert "purged" not in err
