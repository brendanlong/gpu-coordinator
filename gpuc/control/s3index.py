"""The S3 mirror: specs for `gpuc requeue`, a flat index for `gpuc status --all`.

The host is authoritative; this is a mirror, so every read here is best-effort
and every failure is reported as a note rather than an error. The local index
under the state dir answers "which host has job X" without touching S3 at all.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from gpuc._version import user_agent
from gpuc.control.config import Settings, index_dir
from gpuc.control.tolerant import TolerantModel
from gpuc.host.jobs import JobSpec

if TYPE_CHECKING:
    from mypy_boto3_s3.client import S3Client

SPEC_PREFIX = "gpuc/specs"
INDEX_PREFIX = "gpuc/index"


class S3IndexError(RuntimeError):
    pass


class S3ObjectMissing(S3IndexError):
    """The object is not there, as opposed to S3 refusing to answer.

    The difference is the exit code: a spec nobody ever mirrored is "no such
    job", while a timeout or a denied request is a failure of the command.
    """


MISSING_CODES = ("NoSuchKey", "404")


def _is_missing(exc: Exception) -> bool:
    """Is this a 404? boto3 builds its error classes at runtime, so the code is
    read off the response the exception carries, with its text as the fallback."""
    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        error = response.get("Error")
        code = error.get("Code") if isinstance(error, dict) else None
        return str(code) in MISSING_CODES
    return any(code in str(exc) for code in MISSING_CODES)


class IndexEntry(TolerantModel):
    """One job, as the index remembers it. Tolerant like the registry: this
    file is read by whichever session is doing the recovery, not necessarily
    the build that wrote it."""

    job_id: str = ""
    host: str = ""
    name: str = ""
    attempt: int = 1
    submitted_at: str = ""
    s3_prefix: str | None = None
    spec_uri: str | None = None


def spec_key(job_id: str) -> str:
    return f"{SPEC_PREFIX}/{job_id}.json"


def index_key(job_id: str) -> str:
    return f"{INDEX_PREFIX}/{job_id}.json"


def make_s3_client() -> S3Client:
    """The one place boto3 is constructed, so every request carries our agent.

    `user_agent_extra` appends to botocore's own string rather than replacing
    it; AWS support reads the whole line, and the SDK half of it is the half
    they ask for.
    """
    import boto3
    from botocore.config import Config

    return boto3.client("s3", config=Config(user_agent_extra=user_agent()))


def default_s3_prefix(settings: Settings, host: str) -> str | None:
    if not settings.s3_bucket:
        return None
    return f"s3://{settings.s3_bucket}/gpuc/{host}"


def job_uri(s3_prefix: str, job_id: str, name: str = "") -> str:
    """Where a host mirrors one job under its `s3_prefix`, or one file of it."""
    uri = f"{s3_prefix.rstrip('/')}/jobs/{job_id}"
    return f"{uri}/{name}" if name else uri


def job_log_uri(s3_prefix: str, job_id: str) -> str:
    return job_uri(s3_prefix, job_id, "log.txt")


def split_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise S3IndexError(f"not an s3 uri: {uri}")
    bucket, _, key = uri[len("s3://") :].partition("/")
    return bucket, key


@dataclass
class LocalIndex:
    """One small JSON file per job, so concurrent sessions never collide."""

    root: Path | None = None

    @property
    def directory(self) -> Path:
        return self.root or index_dir()

    def record(self, entry: IndexEntry) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"{entry.job_id}.json"
        tmp = path.parent / f".{path.name}.{os.getpid()}.tmp"
        tmp.write_text(entry.model_dump_json(indent=2) + "\n")
        os.replace(tmp, path)

    def get(self, job_id: str) -> IndexEntry | None:
        path = self.directory / f"{job_id}.json"
        if not path.exists():
            return None
        try:
            return IndexEntry.model_validate_json(path.read_text())
        except (OSError, ValidationError):
            return None

    def list(self) -> list[IndexEntry]:
        if not self.directory.is_dir():
            return []
        entries = [self.get(p.stem) for p in sorted(self.directory.glob("*.json"))]
        return [e for e in entries if e is not None]


class S3Index:
    def __init__(self, bucket: str, client: Any | None = None) -> None:
        self.bucket = bucket
        self._client = client

    @staticmethod
    def from_settings(settings: Settings, client: Any | None = None) -> S3Index | None:
        if not settings.s3_bucket:
            return None
        return S3Index(settings.s3_bucket, client)

    @property
    def client(self) -> S3Client:
        if self._client is None:
            self._client = make_s3_client()
        return self._client

    def _put(self, key: str, body: str) -> str:
        try:
            self.client.put_object(
                Bucket=self.bucket, Key=key, Body=body.encode(), ContentType="application/json"
            )
        except Exception as exc:
            raise S3IndexError(
                f"could not write s3://{self.bucket}/{key}: {exc}\n"
                f"Check AWS credentials, or unset s3_bucket in the config to skip the mirror."
            ) from exc
        return f"s3://{self.bucket}/{key}"

    def _get(self, key: str) -> str:
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=key)
            return response["Body"].read().decode("utf-8", "replace")
        except Exception as exc:
            if _is_missing(exc):
                raise S3ObjectMissing(f"no object at s3://{self.bucket}/{key}") from exc
            raise S3IndexError(f"could not read s3://{self.bucket}/{key}: {exc}") from exc

    def put_spec(self, spec: JobSpec) -> str:
        return self._put(spec_key(spec.job_id), json.dumps(spec.to_dict(), indent=2) + "\n")

    def put_spec_document(self, job_id: str, document: dict[str, Any]) -> str:
        """Re-mirror a spec as raw JSON, for an edit to a spec already up there.

        Raw, rather than through `JobSpec`: this is what `requeue` will submit,
        and a round trip would drop the keys a newer build wrote.
        """
        return self._put(spec_key(job_id), json.dumps(document, indent=2) + "\n")

    def get_spec(self, job_id: str) -> dict[str, Any]:
        document = json.loads(self._get(spec_key(job_id)))
        if not isinstance(document, dict):
            raise S3IndexError(f"spec for {job_id} is not a JSON object")
        return document

    def put_index(self, entry: IndexEntry) -> str:
        return self._put(index_key(entry.job_id), entry.model_dump_json(indent=2) + "\n")

    def list_keys(self, prefix: str, limit: int) -> list[str]:
        """Every key under `prefix`, following continuation tokens.

        A single list_objects_v2 call returns at most 1000 keys, so an index
        with more jobs than that silently lost its tail.
        """
        keys: list[str] = []
        token: str | None = None
        while len(keys) < limit:
            request: dict[str, Any] = {
                "Bucket": self.bucket,
                "Prefix": prefix,
                "MaxKeys": min(1000, limit - len(keys)),
            }
            if token:
                request["ContinuationToken"] = token
            try:
                response = self.client.list_objects_v2(**request)
            except Exception as exc:
                raise S3IndexError(f"could not list s3://{self.bucket}/{prefix}: {exc}") from exc
            keys += [key for item in response.get("Contents", []) if (key := item.get("Key"))]
            token = response.get("NextContinuationToken")
            if not response.get("IsTruncated") or not token:
                break
        return keys[:limit]

    def list_index(self, limit: int = 200) -> list[IndexEntry]:
        entries: list[IndexEntry] = []
        for key in self.list_keys(f"{INDEX_PREFIX}/", limit):
            if not key.endswith(".json"):
                continue
            try:
                entries.append(IndexEntry.model_validate_json(self._get(key)))
            except (S3IndexError, ValidationError):
                continue
        return sorted(entries, key=lambda e: e.job_id)

    def get_uri(self, uri: str) -> str:
        bucket, key = split_uri(uri)
        if bucket != self.bucket:
            return S3Index(bucket, self._client)._get(key)
        return self._get(key)
