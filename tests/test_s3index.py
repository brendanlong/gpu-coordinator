from __future__ import annotations

import re
from pathlib import Path

import pytest

from gpuc.control.config import Settings
from gpuc.control.s3index import (
    IndexEntry,
    LocalIndex,
    S3Index,
    S3IndexError,
    S3ObjectMissing,
    default_s3_prefix,
    job_log_uri,
    split_uri,
)
from gpuc.host.jobs import JobSpec
from tests.fakes3 import FakeS3Client


def spec(job_id: str = "20260101-000000-abc123") -> JobSpec:
    return JobSpec.from_dict({"job_id": job_id, "command": "true", "name": "t", "attempt": 2})


def test_local_index_round_trips(control_env: Path) -> None:
    index = LocalIndex()
    entry = IndexEntry(job_id="j1", host="gpubox", name="t", attempt=3)
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
    uri = s3.put_spec(spec())
    assert uri == "s3://bkt/gpuc/specs/20260101-000000-abc123.json"
    document = s3.get_spec("20260101-000000-abc123")
    assert document["command"] == "true"
    assert document["attempt"] == 2


def test_s3_index_lists_entries_for_status_all() -> None:
    s3 = S3Index("bkt", FakeS3Client())
    s3.put_index(IndexEntry(job_id="b", host="gpubox"))
    s3.put_index(IndexEntry(job_id="a", host="local"))
    assert [e.job_id for e in s3.list_index()] == ["a", "b"]


def test_a_write_failure_says_what_to_check() -> None:
    s3 = S3Index("bkt", FakeS3Client(fail_put=True))
    with pytest.raises(S3IndexError) as exc:
        s3.put_spec(spec())
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
