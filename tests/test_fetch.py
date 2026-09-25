"""`gpuc fetch`, host side: which files of a job's workdir are its results.

The same answer uploads give -- a file that came with the checkout is not
fetched unless a person names its path -- so a fetch and an S3 upload of one
job hold the same files."""

from __future__ import annotations

from pathlib import Path

import pytest

from gpuc.host import baseline, fetch, jobs, paths
from gpuc.host.jobs import JobSpec
from tests.conftest import make_spec


def job_with(tmp_files: dict[str, str], **spec: object) -> JobSpec:
    made = make_spec(**spec)
    workdir = paths.workdir(made.job_id)
    for rel, text in tmp_files.items():
        (workdir / rel).parent.mkdir(parents=True, exist_ok=True)
        (workdir / rel).write_text(text)
    return made


def names(listing: dict[str, object]) -> list[str]:
    files = listing["files"]
    assert isinstance(files, list)
    return [f["path"] for f in files]


def test_outputs_are_fetched_less_what_came_with_the_checkout(gpuc_home: Path) -> None:
    spec = job_with({"results/old.md": "checked in"}, outputs=[{"path": "results"}])
    baseline.capture(spec, paths.workdir(spec.job_id), spec.job_id)
    (paths.workdir(spec.job_id) / "results" / "new.pt").write_text("weights")
    found = fetch.listing(spec.job_id, spec, [])
    assert names(found) == ["results/new.pt"]
    assert found["bytes"] == len("weights")


def test_a_named_path_is_fetched_whole(gpuc_home: Path) -> None:
    spec = job_with({"results/old.md": "x", "runs/a/log": "y"}, outputs=[{"path": "results"}])
    baseline.capture(spec, paths.workdir(spec.job_id), spec.job_id)
    found = fetch.listing(spec.job_id, spec, ["results", "runs"])
    assert names(found) == ["results/old.md", "runs/a/log"]


def test_a_file_output_and_a_missing_one(gpuc_home: Path) -> None:
    spec = job_with({"model.pt": "w"}, outputs=[{"path": "model.pt"}, {"path": "never"}])
    found = fetch.listing(spec.job_id, spec, [])
    assert names(found) == ["model.pt"]
    assert found["missing"] == ["never"]


def test_symlinks_are_listed_not_followed(gpuc_home: Path) -> None:
    spec = job_with({"results/real/a": "a"}, outputs=[{"path": "results"}])
    (paths.workdir(spec.job_id) / "results" / "alias").symlink_to("real")
    assert sorted(names(fetch.listing(spec.job_id, spec, []))) == [
        "results/alias",
        "results/real/a",
    ]


@pytest.mark.parametrize("path", ["/etc", "../..", "results/../../x"])
def test_nothing_outside_the_workdir(gpuc_home: Path, path: str) -> None:
    spec = job_with({"results/a": "a"}, outputs=[{"path": "results"}])
    with pytest.raises(fetch.NotFetchable):
        fetch.listing(spec.job_id, spec, [path])


def test_a_job_with_no_outputs_needs_a_path(gpuc_home: Path) -> None:
    spec = job_with({"a": "a"})
    with pytest.raises(fetch.NotFetchable, match="--path"):
        fetch.listing(spec.job_id, spec, [])
    assert names(fetch.listing(spec.job_id, spec, ["."])) == ["a"]


def test_a_swept_workdir_says_so(gpuc_home: Path) -> None:
    spec = make_spec(outputs=[{"path": "results"}])
    with pytest.raises(fetch.NotFetchable, match="workdir is gone"):
        fetch.listing(spec.job_id, spec, [])


def test_the_host_verb_answers_every_id(
    gpuc_home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import json

    from gpuc.host import __main__ as cli
    from gpuc.host.jobs import JobState

    spec = job_with({"results/a": "a"}, outputs=[{"path": "results"}])
    paths.spec_file(spec.job_id).write_text(json.dumps(spec.to_dict()))
    jobs.write_state(spec.job_id, JobState(status="running"))
    assert cli.main(["fetch", spec.job_id, "20260101-000000-aaaaaa"]) == 1
    answers = json.loads(capsys.readouterr().out)["jobs"]
    assert answers[0]["status"] == "running" and names(answers[0]) == ["results/a"]
    assert answers[1]["missing"] is True
