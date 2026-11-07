from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import boto3
import pytest
from botocore.stub import Stubber

from gpuc.control import awskeys
from gpuc.control.awskeys import AwsKeyError, access_policy, deliver_mirror_key, narrowed
from gpuc.host.jobs import HostConfig
from tests.test_bootstrap import ScriptedHost
from tests.test_submit import FakeSts

RENTED = {"kind": "runpod", "pod_id": "p1"}
KEYS = {"AWS_ACCESS_KEY_ID": "AKIA", "AWS_SECRET_ACCESS_KEY": "shhh"}


def test_listing_is_allowed_only_under_the_prefixes_used() -> None:
    policy = access_policy(["s3://a/x/1", "s3://a/y/2/", "s3://b/z"])
    listing = {
        statement["Resource"]: statement["Condition"]["StringLike"]["s3:prefix"]
        for statement in policy["Statement"]
        if statement["Action"] == "s3:ListBucket"
    }
    assert listing == {
        "arn:aws:s3:::a": ["x/1", "x/1/*", "y/2", "y/2/*"],
        "arn:aws:s3:::b": ["z", "z/*"],
    }


@pytest.fixture
def stubbed_sts(monkeypatch: pytest.MonkeyPatch) -> Stubber:
    """The real `federation_token`, over a botocore client that answers from a script."""
    client = boto3.client("sts", region_name="us-east-1")
    monkeypatch.setattr(boto3, "client", lambda *args, **kwargs: client)
    return Stubber(client)


def test_the_key_is_asked_for_by_name_for_36_hours_under_the_policy(stubbed_sts: Stubber) -> None:
    expires = datetime(2026, 11, 9, tzinfo=UTC)
    policy = access_policy(["s3://b/gpuc/h/jobs/20261107-120000-abcdef"])
    stubbed_sts.add_response(
        "get_federation_token",
        {
            "Credentials": {
                "AccessKeyId": "ASIAEXAMPLEKEY12",
                "SecretAccessKey": "s",
                "SessionToken": "t",
                "Expiration": expires,
            }
        },
        {
            "Name": "gpuc-20261107-120000-abcdef",
            "Policy": json.dumps(policy, separators=(",", ":")),
            "DurationSeconds": 129_600,
        },
    )
    with stubbed_sts:
        key = narrowed(
            "20261107-120000-abcdef", ["s3://b/gpuc/h/jobs/20261107-120000-abcdef"], KEYS
        )
    assert (key.access_key_id, key.session_token, key.expires) == ("ASIAEXAMPLEKEY12", "t", expires)


@pytest.mark.parametrize(
    ("code", "grant"), [("AccessDenied", True), ("PackedPolicyTooLarge", False)]
)
def test_only_a_refusal_is_blamed_on_the_missing_grant(
    stubbed_sts: Stubber, code: str, grant: bool
) -> None:
    stubbed_sts.add_client_error("get_federation_token", service_error_code=code)
    with stubbed_sts, pytest.raises(AwsKeyError) as exc:
        narrowed("h", ["s3://b/x"], KEYS)
    assert code in str(exc.value)
    assert ("sts:GetFederationToken" in str(exc.value)) is grant
    assert "--trusted" in str(exc.value)


def test_an_untrusted_rental_mirrors_with_a_key_narrowed_to_its_own_jobs() -> None:
    host, sts = ScriptedHost(), FakeSts()
    config = HostConfig(host="gpuc-x", s3_prefix="s3://bucket/gpuc/gpuc-x", provider=RENTED)
    progress: list[str] = []
    environ = {**KEYS, "AWS_REGION": "eu-west-1"}
    assert (
        deliver_mirror_key(host, config, progress.append, trusted=False, environ=environ, mint=sts)
        is None
    )
    [(name, policy, *_)] = sts.minted
    assert name == "gpuc-gpuc-x"
    assert policy["Statement"][0]["Resource"] == ["arn:aws:s3:::bucket/gpuc/gpuc-x/jobs/*"]
    ((path, (body, mode)),) = host.puts.items()
    assert Path(path).name == "credentials" and mode == 0o600
    assert "shhh" not in body and "aws_session_token = narrow-token" in body
    assert "region = eu-west-1" in body
    assert not any("shhh" in line or "narrow-secret" in line for line in progress)


def test_a_trusted_rental_mirrors_with_the_keys_themselves() -> None:
    host, sts = ScriptedHost(), FakeSts()
    config = HostConfig(host="gpuc-x", s3_prefix="s3://b/x", provider=RENTED)
    assert (
        deliver_mirror_key(host, config, lambda _: None, trusted=True, environ=KEYS, mint=sts)
        is None
    )
    ((_, (body, _)),) = host.puts.items()
    assert "aws_secret_access_key = shhh" in body
    assert sts.minted == []


@pytest.mark.parametrize(("environ", "sts"), [(KEYS, FakeSts(refuses=True)), ({}, FakeSts())])
def test_an_untrusted_rental_with_no_narrowed_key_is_left_with_no_aws_keys(
    environ: dict[str, str], sts: FakeSts
) -> None:
    """Rather than with whatever is there: an earlier build put the real keys there."""
    host = ScriptedHost()
    config = HostConfig(host="gpuc-x", s3_prefix="s3://b/x", provider=RENTED)
    warning = deliver_mirror_key(
        host, config, lambda _: None, trusted=False, environ=environ, mint=sts
    )
    assert warning and "has no AWS keys" in warning
    assert host.puts == {}
    assert any("rm -f" in command and ".aws/credentials" in command for command in host.events)


def test_no_mirror_or_a_shared_box_is_never_touched() -> None:
    host = ScriptedHost()
    no_mirror = HostConfig(host="gpuc-x", provider=RENTED)
    shared_box = HostConfig(host="gpubox", s3_prefix="s3://b/x")
    for config in (no_mirror, shared_box):
        for trusted in (True, False):
            assert (
                deliver_mirror_key(host, config, lambda _: None, trusted=trusted, environ=KEYS)
                is None
            )
    assert host.puts == {} and host.events == []


def test_the_session_name_fits_what_sts_takes(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []

    def mint(name: str, *_: Any) -> dict[str, Any]:
        seen.append(name)
        return FakeSts()(name, {}, "", "")

    narrowed("gpuc-a-very-long-host-name/with spaces-é", ["s3://b/x"], KEYS, mint)
    [name] = seen
    assert len(name) <= 32 and name.startswith("gpuc-")
    assert all(c.isascii() and (c.isalnum() or c in "_+=,.@-") for c in name)
    assert awskeys.MAX_DURATION_S == 36 * 3600
