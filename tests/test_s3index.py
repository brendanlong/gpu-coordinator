from __future__ import annotations

from pathlib import Path

import pytest

from gpuc.control.config import Settings
from gpuc.control.s3index import (
    IndexEntry,
    LocalIndex,
    S3Index,
    S3IndexError,
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
    entry = IndexEntry(job_id="j1", host="spar", name="t", attempt=3)
    index.record(entry)
    assert index.get("j1") == entry
    assert index.get("missing") is None
    assert [e.job_id for e in index.list()] == ["j1"]


def test_local_index_ignores_a_corrupt_file(control_env: Path) -> None:
    index = LocalIndex()
    index.record(IndexEntry(job_id="j1", host="spar"))
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
    s3.put_index(IndexEntry(job_id="b", host="spar"))
    s3.put_index(IndexEntry(job_id="a", host="local"))
    assert [e.job_id for e in s3.list_index()] == ["a", "b"]


def test_a_write_failure_says_what_to_check() -> None:
    s3 = S3Index("bkt", FakeS3Client(fail_put=True))
    with pytest.raises(S3IndexError) as exc:
        s3.put_spec(spec())
    assert "AccessDenied" in str(exc.value)
    assert "Check AWS credentials" in str(exc.value)


def test_missing_key_is_an_error_not_an_empty_spec() -> None:
    with pytest.raises(S3IndexError, match="NoSuchKey"):
        S3Index("bkt", FakeS3Client()).get_spec("nope")


def test_log_fallback_uri_matches_what_the_host_uploads() -> None:
    prefix = default_s3_prefix(Settings(s3_bucket="bkt"), "spar")
    assert prefix == "s3://bkt/gpuc/spar"
    assert job_log_uri(prefix or "", "j1") == "s3://bkt/gpuc/spar/jobs/j1/log.txt"
    assert split_uri("s3://bkt/gpuc/spar/jobs/j1/log.txt") == (
        "bkt",
        "gpuc/spar/jobs/j1/log.txt",
    )


def test_no_bucket_means_no_mirror() -> None:
    assert default_s3_prefix(Settings(), "spar") is None
    assert S3Index.from_settings(Settings()) is None
