from __future__ import annotations

from pathlib import Path

from gpuc.control.awskeys import deliver_mirror_key, write_policy
from gpuc.host.jobs import HostConfig
from tests.test_bootstrap import ScriptedHost
from tests.test_submit import FakeSts

RENTED = {"kind": "runpod", "pod_id": "p1"}


def test_listing_is_allowed_only_under_the_prefixes_written() -> None:
    policy = write_policy(["s3://a/x/1", "s3://a/y/2/", "s3://b/z"])
    listing = {
        statement["Resource"]: statement["Condition"]["StringLike"]["s3:prefix"]
        for statement in policy["Statement"]
        if statement["Action"] == "s3:ListBucket"
    }
    assert listing == {
        "arn:aws:s3:::a": ["x/1", "x/1/*", "y/2", "y/2/*"],
        "arn:aws:s3:::b": ["z", "z/*"],
    }


def test_a_rental_mirrors_with_a_key_narrowed_to_its_own_jobs() -> None:
    host, sts = ScriptedHost(), FakeSts()
    config = HostConfig(host="gpuc-x", s3_prefix="s3://bucket/gpuc/gpuc-x", provider=RENTED)
    progress: list[str] = []
    environ = {
        "AWS_ACCESS_KEY_ID": "AKIA",
        "AWS_SECRET_ACCESS_KEY": "shhh",
        "AWS_REGION": "eu-west-1",
    }
    assert deliver_mirror_key(host, config, progress.append, environ, sts) is None
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
    config = HostConfig(host="gpuc-x", s3_prefix="s3://b/x", provider={**RENTED, "trusted": True})
    environ = {"AWS_ACCESS_KEY_ID": "AKIA", "AWS_SECRET_ACCESS_KEY": "shhh"}
    assert deliver_mirror_key(host, config, lambda _: None, environ, sts) is None
    ((_, (body, _)),) = host.puts.items()
    assert "aws_secret_access_key = shhh" in body
    assert sts.minted == []


def test_a_mirror_key_that_cannot_be_narrowed_is_a_warning_and_nothing_is_sent() -> None:
    host = ScriptedHost()
    config = HostConfig(host="gpuc-x", s3_prefix="s3://b/x", provider=RENTED)
    environ = {"AWS_ACCESS_KEY_ID": "AKIA", "AWS_SECRET_ACCESS_KEY": "shhh"}
    warning = deliver_mirror_key(host, config, lambda _: None, environ, FakeSts(refuses=True))
    assert warning and "sts:GetFederationToken" in warning
    assert host.puts == {}


def test_no_keys_is_a_warning_and_no_mirror_or_a_shared_box_is_nothing() -> None:
    host = ScriptedHost()
    rented = HostConfig(host="gpuc-x", s3_prefix="s3://b/x", provider=RENTED)
    warning = deliver_mirror_key(host, rented, lambda _: None, {})
    assert warning and "AWS_ACCESS_KEY_ID" in warning
    environ = {"AWS_ACCESS_KEY_ID": "AKIA", "AWS_SECRET_ACCESS_KEY": "shhh"}
    no_mirror = HostConfig(host="gpuc-x", provider=RENTED)
    shared_box = HostConfig(host="gpubox", s3_prefix="s3://b/x")
    for config in (no_mirror, shared_box):
        assert deliver_mirror_key(host, config, lambda _: None, environ) is None
    assert host.puts == {}
