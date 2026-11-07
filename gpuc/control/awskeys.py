"""The AWS keys a host is given: yours, or ones narrowed to what it uploads.

A rental is somebody else's machine -- on RunPod's community cloud or a Vast
host, a stranger's -- and whoever has root there can read every key on it. So
a host this machine does not trust (`HostEntry.trusted`) never sees the
submitter's own AWS keys. It gets keys from STS `GetFederationToken` instead,
which the submitter's IAM user mints with its own long-term keys: they last
36 hours at most, and the session policy passed in leaves them able to list,
read and write under the prefixes the host uploads to and nothing else.
Stolen, they let the thief read and overwrite what that host uploaded until
they expire. Reading is allowed because what is there came from the host in
the first place, and a job may want its own checkpoints back.

`GetFederationToken` rather than the alternatives: `GetSessionToken` needs no
permission but cannot be narrowed, only shortened, and `AssumeRole` needs a
role an administrator creates, with keys that last 12 hours at most. This
needs `sts:GetFederationToken` granted to the user once, and no more.

What a host writes is already per job: a job's `outputs:` all carry its id,
and its mirror is `{s3_prefix}/jobs/{job_id}`. So a job's keys cover exactly
the targets its own sync preflight checks, and a rental's own mirror key
covers `{s3_prefix}/jobs` -- that host's jobs and no other host's.
"""

from __future__ import annotations

import json
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
TRUST = (
    "or trust the host with your own keys: `gpuc host rent --trusted`, or `gpuc host add` "
    "without `--untrusted`"
)
GRANT = (
    "The IAM user whose keys these are needs `sts:GetFederationToken` on "
    "`arn:aws:sts::<account-id>:federated-user/gpuc-*`, and they must be its long-term "
    f"keys rather than temporary ones; {TRUST}."
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
    """The `s3://` locations it may use, for saying so."""

    def env(self) -> dict[str, str]:
        values = (self.access_key_id, self.secret_access_key, self.session_token)
        return dict(zip(KEY_VARS, values, strict=True))


Mint = Callable[[str, dict[str, Any], str, str], dict[str, Any]]
"""`(name, policy, access_key_id, secret_access_key)` -> STS's `Credentials`."""


def federation_token(
    name: str, policy: dict[str, Any], access_key_id: str, secret_access_key: str
) -> dict[str, Any]:
    import boto3
    from botocore.config import Config
    from botocore.exceptions import BotoCoreError, ClientError

    region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1"
    client = boto3.client(
        "sts",
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
        region_name=region,
        config=Config(user_agent_extra=user_agent()),
    )
    try:
        response = client.get_federation_token(
            Name=name,
            Policy=json.dumps(policy, separators=(",", ":")),
            DurationSeconds=MAX_DURATION_S,
        )
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code")
        fix = GRANT if code == "AccessDenied" else f"Fix that, {TRUST}."
        raise AwsKeyError(f"could not mint a narrowed AWS key: {exc}\n{fix}") from exc
    except BotoCoreError as exc:
        raise AwsKeyError(f"could not mint a narrowed AWS key: {exc}\nFix that, {TRUST}.") from exc
    return dict(response["Credentials"])


def _split(uri: str) -> tuple[str, str]:
    bucket, _, key = uri.removeprefix("s3://").partition("/")
    return bucket, key.rstrip("/")


def access_policy(uris: Iterable[str]) -> dict[str, Any]:
    """List, read and write under each of `uris`, and nothing else.

    No deletes: the host never deletes (`aws s3 sync` runs without
    `--delete`). A multipart upload that fails is aborted, so that is allowed.
    A location with a `*` or `?` is refused rather than escaped: it is never a
    real bucket or prefix, and one from a host's config is a widening attempt.
    """
    objects: list[str] = []
    listed: dict[str, list[str]] = {}
    # Once each: STS caps the policy at 2048 characters, and outputs often share one.
    for uri in dict.fromkeys(uri.rstrip("/") for uri in uris):
        if any(char in uri for char in "*?"):
            raise AwsKeyError(f"{uri} has a `*` or `?`, which a policy reads as a wildcard")
        bucket, key = _split(uri)
        objects.append(f"arn:aws:s3:::{bucket}/{key}/*")
        listed.setdefault(bucket, []).extend([key, f"{key}/*"])
    statements: list[dict[str, Any]] = [
        {
            "Effect": "Allow",
            "Action": ["s3:PutObject", "s3:GetObject", "s3:AbortMultipartUpload"],
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
    """A key minted from `credentials` that can use `uris` and nothing else."""
    access_key_id = credentials.get("AWS_ACCESS_KEY_ID")
    secret_access_key = credentials.get("AWS_SECRET_ACCESS_KEY")
    if not (access_key_id and secret_access_key):
        raise AwsKeyError(
            "a narrowed AWS key is minted from AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY, "
            "and both are needed"
        )
    if credentials.get("AWS_SESSION_TOKEN") or access_key_id.startswith("ASIA"):
        raise AwsKeyError(
            "these AWS keys are temporary (from SSO or a role), and a narrowed key can only be "
            f"minted from an IAM user's long-term keys; use those, {TRUST}"
        )
    issued = mint(_session_name(name), access_policy(uris), access_key_id, secret_access_key)
    return NarrowedKey(
        access_key_id=issued["AccessKeyId"],
        secret_access_key=issued["SecretAccessKey"],
        session_token=issued["SessionToken"],
        expires=issued["Expiration"],
        prefixes=tuple(uris),
    )


def _session_name(name: str) -> str:
    """STS's `Name`: 2 to 32 of `[\\w+=,.@-]`, and the `gpuc-` the grant names."""
    return re.sub(r"[^\w+=,.@-]", "-", f"gpuc-{name}", flags=re.ASCII)[:32]


def check_prefix(config: HostConfig, expected: str | None) -> None:
    """Refuse an untrusted host's `s3_prefix` unless it is the one this machine
    would give it (`config.default_s3_prefix`).

    The prefix is the host's own config, and its keys are scoped from it: a
    host that named another host's prefix, or another bucket, would have this
    machine mint keys that reach those.
    """
    given = config.s3_prefix.rstrip("/") if config.s3_prefix else None
    wanted = expected.rstrip("/") if expected else None
    if given != wanted:
        fix = f"--s3-prefix {wanted}" if wanted else "--s3-prefix ''"
        raise AwsKeyError(
            f"the host's s3_prefix is {given or 'unset'}, not {wanted or 'unset'} as this machine "
            f"would set it, and an untrusted host does not choose where its keys reach. Set it "
            f"back with `gpuc host set {config.host} {fix}`, {TRUST}"
        )


def job_secrets(
    spec: JobSpec,
    config: HostConfig,
    secrets: Mapping[str, str],
    *,
    trusted: bool,
    expected_prefix: str | None,
    mint: Mint = federation_token,
) -> tuple[dict[str, str], NarrowedKey | None]:
    """The secrets this job's host is given: as named on a trusted host, and
    otherwise with any AWS keys swapped for ones that can use only where this
    job uploads. None: nothing was swapped.

    Refused here rather than at job start: a job that uploads to S3 naming no
    AWS keys, since an untrusted host has none of the submitter's to fall back
    on, and one that names them with nowhere on S3 to upload, since a narrowed
    key could do nothing at all.
    """
    if trusted:
        return dict(secrets), None
    named = any(name in secrets for name in KEY_VARS)
    if not named and any(output.s3 for output in spec.outputs):
        raise AwsKeyError(
            "the job uploads to S3 but its `secrets:` name no AWS keys, and an untrusted host "
            f"has none of yours to fall back on. Add AWS_ACCESS_KEY_ID and "
            f"AWS_SECRET_ACCESS_KEY to `secrets:`, {TRUST}."
        )
    if not named:
        return dict(secrets), None
    uris = [
        target.uri
        for target in preflight.targets(spec, config)
        if isinstance(target, destinations.S3)
    ]
    if not uris:
        raise AwsKeyError(
            "the job names AWS keys in `secrets:` but uploads nothing to S3, and the keys an "
            f"untrusted host is given can only use where it uploads. Drop them, {TRUST}."
        )
    check_prefix(config, expected_prefix)
    key = narrowed(spec.job_id, uris, secrets, mint)
    kept = {name: value for name, value in secrets.items() if name not in KEY_VARS}
    return {**kept, **key.env()}, key


def deliver_mirror_key(
    transport: Transport,
    config: HostConfig,
    report: Reporter,
    *,
    trusted: bool,
    expected_prefix: str | None,
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
    hours and the rental may outlive that. Only ever a refresh: a key that
    cannot be renewed leaves whatever is there, since deleting it would stop
    a mirror that still works. Returns a warning when it was not renewed.
    """
    if not config.ephemeral or not config.s3_prefix:
        return None
    environ = dict(os.environ) if environ is None else dict(environ)
    home = transport.run('printf %s "$HOME"', check=True).stdout.strip()
    path = f"{home}/.aws/credentials"
    keys = {name: environ[name] for name in KEY_VARS if environ.get(name)}
    described = "S3 credentials"
    problem = None
    if not (keys.get("AWS_ACCESS_KEY_ID") and keys.get("AWS_SECRET_ACCESS_KEY")):
        problem = (
            "there are no AWS credentials in this environment (export AWS_ACCESS_KEY_ID and "
            "AWS_SECRET_ACCESS_KEY)"
        )
    elif not trusted:
        mirror = f"{config.s3_prefix.rstrip('/')}/jobs"
        try:
            check_prefix(config, expected_prefix)
            key = narrowed(config.host, [mirror], environ, mint)
            keys = key.env()
            described = f"an AWS key for {mirror} only, until {key.expires:%Y-%m-%d %H:%M %Z},"
        except AwsKeyError as exc:
            problem = str(exc)
    if problem is not None:
        return f"the pod's mirror key for {config.s3_prefix} was not renewed: {problem}"
    region = environ.get("AWS_REGION") or environ.get("AWS_DEFAULT_REGION") or "us-east-1"
    body = "[default]\n" + f"region = {region}\n"
    body += "".join(f"{name.lower()} = {value}\n" for name, value in keys.items())
    transport.put_file(body, path, 0o600)
    report(f"delivered {described} to {path} (0600) for the log mirror")
    return None
