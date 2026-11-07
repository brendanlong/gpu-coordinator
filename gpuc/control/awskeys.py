"""The AWS keys a host is given: yours, or ones narrowed to what it writes.

A rental is somebody else's machine -- on RunPod's community cloud or a Vast
host, a stranger's -- and whoever has root there can read every key on it. So
unless it was rented `--trusted`, a rental never sees the submitter's own AWS
keys. It gets keys from STS `GetFederationToken` instead, which the
submitter's IAM user mints with its own long-term keys: they last 36 hours at
most, and the session policy passed in leaves them able to do nothing but
list and write under the prefixes this host uploads to. Stolen, they let the
thief write into those prefixes until they expire, and nothing else.

`GetFederationToken` rather than the alternatives: `GetSessionToken` needs no
permission but cannot be narrowed, only shortened, and `AssumeRole` needs a
role an administrator creates, with keys that last 12 hours at most. This
needs `sts:GetFederationToken` granted to the user once, and no more.

What a host writes is already per job: a job's `outputs:` all carry its id,
and its mirror is `{s3_prefix}/jobs/{job_id}`. So a job's keys cover exactly
the targets its own sync preflight checks, and the host's own mirror key
covers `{s3_prefix}/jobs` -- this host's jobs and no other host's.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from gpuc._version import user_agent
from gpuc.control.config import Reporter
from gpuc.control.transport import Transport
from gpuc.host import destinations, preflight
from gpuc.host.jobs import HostConfig, JobSpec

MAX_DURATION_S = 129_600
"""36 hours, the longest `GetFederationToken` will issue keys for."""
KEY_VARS = ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN")
GRANT = (
    "The IAM user whose keys these are needs `sts:GetFederationToken` on "
    "`arn:aws:sts::<account-id>:federated-user/gpuc-*`, and they must be its long-term "
    "keys rather than temporary ones. Or rent the host with `--trusted` to send the keys "
    "themselves."
)


class AwsKeyError(RuntimeError):
    pass


@dataclass(frozen=True)
class NarrowedKey:
    access_key_id: str
    secret_access_key: str
    session_token: str
    expires: datetime
    prefixes: tuple[str, ...]
    """The `s3://` locations it may write under, for saying so."""

    def env(self) -> dict[str, str]:
        return dict(
            zip(
                KEY_VARS,
                (self.access_key_id, self.secret_access_key, self.session_token),
                strict=True,
            )
        )


Mint = Callable[[str, dict[str, Any], str, str], dict[str, Any]]
"""`(name, policy, access_key_id, secret_access_key)` -> STS's `Credentials`."""


def federation_token(
    name: str, policy: dict[str, Any], access_key_id: str, secret_access_key: str
) -> dict[str, Any]:
    import json

    import boto3
    from botocore.config import Config
    from botocore.exceptions import BotoCoreError, ClientError

    client = boto3.client(
        "sts",
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
        region_name=os.environ.get("AWS_REGION")
        or os.environ.get("AWS_DEFAULT_REGION")
        or "us-east-1",
        config=Config(user_agent_extra=user_agent()),
    )
    try:
        response = client.get_federation_token(
            Name=name, Policy=json.dumps(policy), DurationSeconds=MAX_DURATION_S
        )
    except (BotoCoreError, ClientError) as exc:
        raise AwsKeyError(f"could not mint a narrowed AWS key: {exc}\n{GRANT}") from exc
    return dict(response["Credentials"])


def _split(uri: str) -> tuple[str, str]:
    bucket, _, key = uri.removeprefix("s3://").partition("/")
    return bucket, key.strip("/")


def write_policy(uris: Iterable[str]) -> dict[str, Any]:
    """List and write under each of `uris`, and nothing else: no reads, no deletes.

    `aws s3 sync` lists the destination to see what is already there, and a
    multipart upload that fails is aborted, so those two are allowed too.
    """
    objects: list[str] = []
    listed: dict[str, list[str]] = {}
    for uri in uris:
        bucket, key = _split(uri)
        objects.append(f"arn:aws:s3:::{bucket}/{key}/*")
        listed.setdefault(bucket, []).extend([key, f"{key}/*"])
    statements: list[dict[str, Any]] = [
        {
            "Effect": "Allow",
            "Action": ["s3:PutObject", "s3:AbortMultipartUpload"],
            "Resource": objects,
        }
    ]
    statements += [
        {
            "Effect": "Allow",
            "Action": "s3:ListBucket",
            "Resource": f"arn:aws:s3:::{bucket}",
            "Condition": {"StringLike": {"s3:prefix": prefixes}},
        }
        for bucket, prefixes in listed.items()
    ]
    return {"Version": "2012-10-17", "Statement": statements}


def narrowed(
    name: str,
    uris: list[str],
    credentials: Mapping[str, str],
    mint: Mint = federation_token,
) -> NarrowedKey:
    """A key minted from `credentials` that can write under `uris` and nothing else."""
    access_key_id = credentials.get("AWS_ACCESS_KEY_ID")
    secret_access_key = credentials.get("AWS_SECRET_ACCESS_KEY")
    if not (access_key_id and secret_access_key):
        raise AwsKeyError(
            "a narrowed AWS key is minted from AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY, "
            "and both are needed"
        )
    issued = mint(_session_name(name), write_policy(uris), access_key_id, secret_access_key)
    return NarrowedKey(
        access_key_id=issued["AccessKeyId"],
        secret_access_key=issued["SecretAccessKey"],
        session_token=issued["SessionToken"],
        expires=issued["Expiration"],
        prefixes=tuple(uris),
    )


def _session_name(name: str) -> str:
    """STS's `Name`: 2 to 32 of `[\\w+=,.@-]`, and the `gpuc-` the grant names."""
    return re.sub(r"[^\w+=,.@-]", "-", f"gpuc-{name}")[:32]


def job_secrets(
    spec: JobSpec,
    config: HostConfig,
    secrets: Mapping[str, str],
    mint: Mint = federation_token,
) -> tuple[dict[str, str], NarrowedKey | None]:
    """The secrets this job's host is given: as named, unless the host narrows
    AWS keys and the job carries any, in which case those are swapped for keys
    that can write only where this job uploads. None: nothing was swapped."""
    if not config.narrows_aws or not any(name in secrets for name in KEY_VARS):
        return dict(secrets), None
    uris = [
        target.uri
        for target in preflight.targets(spec, config)
        if isinstance(target, destinations.S3)
    ]
    key = narrowed(spec.job_id, uris, secrets, mint)
    kept = {name: value for name, value in secrets.items() if name not in KEY_VARS}
    return {**kept, **key.env()}, key


def deliver_mirror_key(
    transport: Transport,
    config: HostConfig,
    report: Reporter,
    environ: Mapping[str, str] | None = None,
    mint: Mint = federation_token,
) -> str | None:
    """Give a *rented* host AWS keys of its own, for the mirror.

    Each job's sync runs with the job's own `secrets:`, but the dispatcher's
    drain mirrors every job's log.txt and state.json after the jobs -- and
    their secrets files -- are gone, with only its own environment to do it.
    So the pod gets a `~/.aws/credentials` from this shell, written from stdin
    at 0600, never via argv and never in the pod env (`GET /pods` returns the
    env to any holder of an account key). A rental only: its `$HOME` is ours,
    and a shared box's user has their own.

    Bootstrap delivers it, so a pod whose home was wiped gets it back with
    everything else, and so does every submit, since a narrowed key lasts 36
    hours and the rental may outlive that. Returns a warning when the mirror
    will go without.
    """
    if not config.ephemeral or not config.s3_prefix:
        return None
    environ = dict(os.environ) if environ is None else dict(environ)
    if not (environ.get("AWS_ACCESS_KEY_ID") and environ.get("AWS_SECRET_ACCESS_KEY")):
        return (
            f"no AWS credentials in this environment, so the pod cannot mirror logs to "
            f"{config.s3_prefix} with keys from here; export AWS_ACCESS_KEY_ID and "
            f"AWS_SECRET_ACCESS_KEY if you want the mirror"
        )
    keys = {name: environ[name] for name in KEY_VARS if environ.get(name)}
    described = "S3 credentials"
    if config.narrows_aws:
        mirror = f"{config.s3_prefix.rstrip('/')}/jobs"
        try:
            key = narrowed(config.host, [mirror], environ, mint)
        except AwsKeyError as exc:
            return f"the pod cannot mirror logs to {config.s3_prefix}: {exc}"
        keys = key.env()
        described = (
            f"an AWS key that can only write {mirror}, until {key.expires:%Y-%m-%d %H:%M %Z},"
        )
    region = environ.get("AWS_REGION") or environ.get("AWS_DEFAULT_REGION") or "us-east-1"
    body = "[default]\n" + f"region = {region}\n"
    body += "".join(f"{name.lower()} = {value}\n" for name, value in keys.items())
    home = transport.run('printf %s "$HOME"', check=True).stdout.strip()
    transport.put_file(body, f"{home}/.aws/credentials", 0o600)
    report(f"delivered {described} to {home}/.aws/credentials (0600) for the log mirror")
    return None
