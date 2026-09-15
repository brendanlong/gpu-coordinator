"""An in-memory stand-in for the three boto3 S3 calls s3index makes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class _Body:
    def __init__(self, data: bytes) -> None:
        self._data = data

    def read(self) -> bytes:
        return self._data


@dataclass
class FakeS3Client:
    objects: dict[str, bytes] = field(default_factory=dict)
    fail_put: bool = False
    page_size: int | None = None
    list_calls: list[dict[str, Any]] = field(default_factory=list)

    def put_object(self, *, Bucket: str, Key: str, Body: bytes, **_: Any) -> dict[str, Any]:
        if self.fail_put:
            raise RuntimeError("AccessDenied")
        self.objects[f"{Bucket}/{Key}"] = Body
        return {}

    def get_object(self, *, Bucket: str, Key: str, **_: Any) -> dict[str, Any]:
        try:
            return {"Body": _Body(self.objects[f"{Bucket}/{Key}"])}
        except KeyError as exc:
            raise RuntimeError("NoSuchKey") from exc

    def list_objects_v2(
        self,
        *,
        Bucket: str,
        Prefix: str = "",
        MaxKeys: int = 1000,
        ContinuationToken: str | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        self.list_calls.append({"Prefix": Prefix, "MaxKeys": MaxKeys, "Token": ContinuationToken})
        keys = [
            key.split("/", 1)[1]
            for key in sorted(self.objects)
            if key.startswith(f"{Bucket}/{Prefix}")
        ]
        start = keys.index(ContinuationToken) if ContinuationToken else 0
        size = min(MaxKeys, self.page_size or MaxKeys)
        page, rest = keys[start : start + size], keys[start + size :]
        response: dict[str, Any] = {"Contents": [{"Key": key} for key in page]}
        if rest:
            response["IsTruncated"] = True
            response["NextContinuationToken"] = rest[0]
        return response
