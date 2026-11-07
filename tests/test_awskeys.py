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
from gpuc.control.transport import LocalTransport
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


def mirror(
    host: Any,
    config: HostConfig,
    *,
    trusted: bool,
    environ: dict[str, str] = KEYS,
    sts: FakeSts | None = None,
    progress: list[str] | None = None,
) -> str | None:
    return deliver_mirror_key(
        host,
        config,
        (progress if progress is not None else []).append,
        trusted=trusted,
        expected_prefix=config.s3_prefix,
        environ=environ,
        mint=sts or FakeSts(),
    )


RENTAL = HostConfig(host="gpuc-x", s3_prefix="s3://bucket/gpuc/gpuc-x", provider=RENTED)


def test_an_untrusted_rental_mirrors_with_a_key_narrowed_to_its_own_jobs() -> None:
    host, sts = ScriptedHost(), FakeSts()
    progress: list[str] = []
    environ = {**KEYS, "AWS_REGION": "eu-west-1"}
    assert mirror(host, RENTAL, trusted=False, environ=environ, sts=sts, progress=progress) is None
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
    assert mirror(host, RENTAL, trusted=True, sts=sts) is None
    ((_, (body, _)),) = host.puts.items()
    assert "aws_secret_access_key = shhh" in body
    assert sts.minted == []


@pytest.mark.parametrize(
    ("environ", "sts"), [(KEYS, FakeSts(refuses=True)), ({}, FakeSts())], ids=["refused", "no-keys"]
)
def test_a_mirror_key_that_cannot_be_renewed_is_left_as_it_is(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, environ: dict[str, str], sts: FakeSts
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    credentials = tmp_path / ".aws" / "credentials"
    credentials.parent.mkdir()
    credentials.write_text("[default]\naws_session_token = earlier\n")
    warning = mirror(LocalTransport(), RENTAL, trusted=False, environ=environ, sts=sts)
    assert warning and "not renewed" in warning
    assert credentials.read_text() == "[default]\naws_session_token = earlier\n"


def test_a_host_with_its_own_idea_of_its_prefix_gets_no_mirror_key() -> None:
    host, sts = ScriptedHost(), FakeSts()
    warning = deliver_mirror_key(
        host,
        RENTAL,
        lambda _: None,
        trusted=False,
        expected_prefix="s3://bucket/gpuc/somebody-else",
        environ=KEYS,
        mint=sts,
    )
    assert warning and "does not choose where its keys reach" in warning
    assert sts.minted == [] and host.puts == {}


def test_a_wildcard_never_reaches_a_policy() -> None:
    for uri in ("s3://*", "s3://bucket/gpuc/?/x"):
        with pytest.raises(AwsKeyError, match="wildcard"):
            access_policy([uri])


def test_no_mirror_or_a_shared_box_is_never_touched() -> None:
    host = ScriptedHost()
    no_mirror = HostConfig(host="gpuc-x", provider=RENTED)
    shared_box = HostConfig(host="gpubox", s3_prefix="s3://b/x")
    for config in (no_mirror, shared_box):
        for trusted in (True, False):
            assert mirror(host, config, trusted=trusted) is None
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


def test_a_destination_is_granted_once_and_with_its_key_exactly() -> None:
    policy = access_policy(["s3://b//runs/j1", "s3://b//runs/j1/", "s3://b//runs/j1"])
    [objects, listing] = policy["Statement"]
    assert objects["Resource"] == ["arn:aws:s3:::b//runs/j1/*"]
    assert listing["Condition"]["StringLike"]["s3:prefix"] == ["/runs/j1", "/runs/j1/*"]
