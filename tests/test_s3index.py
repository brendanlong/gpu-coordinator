from __future__ import annotations

import re
from pathlib import Path

import pytest

from gpuc.control.config import Settings, default_s3_prefix
from gpuc.control.s3index import (
    IndexEntry,
    JobIndex,
    LocalIndex,
    S3Index,
    S3IndexError,
    S3ObjectMissing,
    job_log_uri,
    split_uri,
)
from gpuc.host.jobs import JobSpec
from tests.fakes3 import FakeS3Client


def spec(job_id: str = "20260101-000000-abc123") -> JobSpec:
    return JobSpec.from_dict({"job_id": job_id, "command": "true", "name": "t"})


def test_local_index_round_trips(control_env: Path) -> None:
    index = LocalIndex()
    entry = IndexEntry(job_id="j1", host="gpubox", name="t", requeued_from="j0")
    index.record(entry)
    assert index.get("j1") == entry
    assert index.get("missing") is None
    assert [e.job_id for e in index.list()] == ["j1"]


def test_local_index_ignores_a_corrupt_file(control_env: Path) -> None:
    index = LocalIndex()
    index.record(IndexEntry(job_id="j1", host="gpubox"))
    (index.directory / "j2.json").write_text("{not json")
    assert [e.job_id for e in index.list()] == ["j1"]


def test_s3_spec_round_trips_for_requeue() -> None:
    s3 = S3Index("bkt", FakeS3Client())
    uri = s3.put_spec_document(spec().job_id, spec().to_dict())
    assert uri == "s3://bkt/gpuc/specs/20260101-000000-abc123.json"
    document = s3.get_spec("20260101-000000-abc123")
    assert document["command"] == "true"


def test_s3_index_lists_entries_for_status_all() -> None:
    s3 = S3Index("bkt", FakeS3Client())
    s3.put_index(IndexEntry(job_id="b", host="gpubox"))
    s3.put_index(IndexEntry(job_id="a", host="local"))
    assert [e.job_id for e in s3.list_index()] == ["a", "b"]


def test_a_write_failure_says_what_to_check() -> None:
    s3 = S3Index("bkt", FakeS3Client(fail_put=True))
    with pytest.raises(S3IndexError) as exc:
        s3.put_spec_document(spec().job_id, spec().to_dict())
    assert "AccessDenied" in str(exc.value)
    assert "Check AWS credentials" in str(exc.value)


def test_missing_key_is_an_error_not_an_empty_spec() -> None:
    # Its own class, so `gpuc requeue` can answer exit 4 (no such job) for a
    # spec nobody ever mirrored, and exit 1 for an S3 that would not answer.
    with pytest.raises(
        S3ObjectMissing, match=re.escape("no object at s3://bkt/gpuc/specs/nope.json")
    ):
        S3Index("bkt", FakeS3Client()).get_spec("nope")
    assert issubclass(S3ObjectMissing, S3IndexError)


def test_log_fallback_uri_matches_what_the_host_uploads() -> None:
    prefix = default_s3_prefix(Settings(s3_bucket="bkt"), "gpubox")
    assert prefix == "s3://bkt/gpuc/gpubox"
    assert job_log_uri(prefix or "", "j1") == "s3://bkt/gpuc/gpubox/jobs/j1/log.txt"
    assert split_uri("s3://bkt/gpuc/gpubox/jobs/j1/log.txt") == (
        "bkt",
        "gpuc/gpubox/jobs/j1/log.txt",
    )


def test_no_bucket_means_no_mirror() -> None:
    assert default_s3_prefix(Settings(), "gpubox") is None
    assert S3Index.from_settings(Settings()) is None


def test_list_index_follows_continuation_tokens() -> None:
    client = FakeS3Client(page_size=2)
    s3 = S3Index("bkt", client)
    for index in range(7):
        s3.put_index(IndexEntry(job_id=f"j{index}", host="gpubox"))
    client.list_calls.clear()
    entries = s3.list_index()
    assert [e.job_id for e in entries] == [f"j{index}" for index in range(7)]
    assert len(client.list_calls) == 4
    assert client.list_calls[1]["Token"]


def test_list_index_stops_at_the_limit() -> None:
    client = FakeS3Client(page_size=2)
    s3 = S3Index("bkt", client)
    for index in range(7):
        s3.put_index(IndexEntry(job_id=f"j{index}", host="gpubox"))
    assert len(s3.list_index(limit=3)) == 3


def test_a_list_failure_names_the_prefix() -> None:
    class Broken(FakeS3Client):
        def list_objects_v2(self, **_: object) -> dict[str, object]:
            raise RuntimeError("AccessDenied")

    with pytest.raises(S3IndexError, match="gpuc/index"):
        S3Index("bkt", Broken()).list_index()


# -- JobIndex: the local index, then the mirror --------------------------------


def job_index(tmp_path: Path, client: FakeS3Client | None) -> JobIndex:
    """A `JobIndex` over a local index under `tmp_path` and, given a client,
    a mirror in bucket `bkt`."""
    index = JobIndex(
        Settings(s3_bucket="bkt" if client else None), local=LocalIndex(tmp_path / "index")
    )
    if client is not None:
        index.s3 = S3Index("bkt", client)
    return index


def test_job_index_get_falls_back_to_the_mirror(tmp_path: Path) -> None:
    """A second machine never submitted the job, so its local index is empty;
    the mirror is what tells it where the job went."""
    client = FakeS3Client()
    S3Index("bkt", client).put_index(
        IndexEntry(job_id="j1", host="gpubox", s3_prefix="s3://bkt/gpuc/gpubox")
    )
    index = job_index(tmp_path, client)
    found = index.get("j1")
    assert found is not None and found.host == "gpubox"
    assert index.get("missing") is None


def test_job_index_get_prefers_the_local_copy(tmp_path: Path) -> None:
    client = FakeS3Client()
    S3Index("bkt", client).put_index(IndexEntry(job_id="j1", host="from-mirror"))
    index = job_index(tmp_path, client)
    index.local.record(IndexEntry(job_id="j1", host="from-local"))
    client.objects.clear()  # would the mirror be read, it would now say nothing
    found = index.get("j1")
    assert found is not None and found.host == "from-local"


def test_job_index_get_without_a_bucket_is_the_local_index_alone(tmp_path: Path) -> None:
    index = job_index(tmp_path, None)
    assert index.s3 is None
    assert index.get("j1") is None
    index.local.record(IndexEntry(job_id="j1", host="gpubox"))
    found = index.get("j1")
    assert found is not None and found.host == "gpubox"


def test_job_index_get_treats_an_unreadable_mirror_as_no_entry(tmp_path: Path) -> None:
    class Refusing(FakeS3Client):
        def get_object(self, **_: object) -> dict[str, object]:
            raise RuntimeError("AccessDenied")

    assert job_index(tmp_path, Refusing()).get("j1") is None


def test_job_index_all_merges_both_indexes(tmp_path: Path) -> None:
    client = FakeS3Client()
    S3Index("bkt", client).put_index(IndexEntry(job_id="remote", host="pod"))
    index = job_index(tmp_path, client)
    index.local.record(IndexEntry(job_id="local", host="gpubox"))
    entries, short = index.all()
    assert sorted(entries) == ["local", "remote"]
    assert short is None


def test_job_index_all_is_incomplete_when_the_mirror_cannot_be_listed(tmp_path: Path) -> None:
    """ "Could not read the mirror" must not read as "these are all the jobs"."""

    class Broken(FakeS3Client):
        def list_objects_v2(self, **_: object) -> dict[str, object]:
            raise RuntimeError("AccessDenied")

    index = job_index(tmp_path, Broken())
    index.local.record(IndexEntry(job_id="local", host="gpubox"))
    entries, short = index.all()
    assert list(entries) == ["local"]
    assert short is not None and "AccessDenied" in short


def test_job_index_mirror_prefix_is_the_jobs_own_before_the_hosts(tmp_path: Path) -> None:
    from tests.conftest import host_entry

    index = job_index(tmp_path, None)
    host = host_entry(name="gpubox", ssh="me@box", s3_prefix="s3://bkt/gpuc/new")
    index.local.record(IndexEntry(job_id="old", host="gpubox", s3_prefix="s3://bkt/gpuc/old"))
    assert index.mirror_prefix("old", host) == "s3://bkt/gpuc/old"
    assert index.mirror_prefix("unindexed", host) == "s3://bkt/gpuc/new"


def test_job_index_mirrored_state_is_a_document_or_nothing(tmp_path: Path) -> None:
    client = FakeS3Client(
        objects={
            "bkt/gpuc/h/jobs/good/state.json": b'{"status": "succeeded"}',
            "bkt/gpuc/h/jobs/list/state.json": b"[1, 2]",
            "bkt/gpuc/h/jobs/torn/state.json": b"{not json",
        }
    )
    index = job_index(tmp_path, client)
    prefix = "s3://bkt/gpuc/h"
    assert index.mirrored_state("good", prefix) == {"status": "succeeded"}
    for job_id in ("list", "torn", "missing"):
        assert index.mirrored_state(job_id, prefix) is None
    assert index.mirrored_state("good", None) is None
    assert job_index(tmp_path, None).mirrored_state("good", prefix) is None
