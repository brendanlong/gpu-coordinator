"""Job secrets encrypted to the host's key, opened by the host through
pyrage under `uv run` (a real one: uv fetches pyrage once into its cache)."""

from __future__ import annotations

import stat
from pathlib import Path

import pyrage
import pytest

from gpuc.host import cleanup, jobs, paths, sealed
from tests.test_runner import deps, log_of, prepare, run


def seal(job_id: str, body: str) -> None:
    recipient = pyrage.x25519.Recipient.from_str(sealed.recipient())
    paths.sealed_env_file(job_id).write_bytes(
        pyrage.encrypt(body.encode(), [recipient], armored=True)
    )


def test_the_host_makes_its_key_once_and_keeps_it_private(gpuc_home: Path) -> None:
    first = sealed.recipient()
    assert first.startswith("age1")
    assert sealed.recipient() == first
    mode = stat.S_IMODE(paths.host_identity_file().stat().st_mode)
    assert mode == 0o600


def test_a_job_gets_its_sealed_secrets_and_nobody_sees_them_in_plain_text(
    gpuc_home: Path,
) -> None:
    job_id = prepare(command='echo "token=$HF_TOKEN"', secrets=["HF_TOKEN"])
    seal(job_id, "HF_TOKEN=hf_secret_value\n")
    assert "hf_secret_value" not in paths.sealed_env_file(job_id).read_text()
    assert run(job_id, deps()) == 0
    assert "token=hf_secret_value" in log_of(job_id)
    assert not paths.sealed_env_file(job_id).exists()


def test_plain_secrets_from_an_older_client_still_reach_the_job(gpuc_home: Path) -> None:
    job_id = prepare(command='echo "token=$HF_TOKEN"', secrets=["HF_TOKEN"])
    paths.job_env_file(job_id).write_text("HF_TOKEN=plain_value\n")
    assert run(job_id, deps()) == 0
    assert "token=plain_value" in log_of(job_id)


def test_secrets_the_host_cannot_open_fail_the_job_before_it_runs(gpuc_home: Path) -> None:
    job_id = prepare(command="touch ran", secrets=["HF_TOKEN"])
    other = pyrage.x25519.Identity.generate().to_public()
    paths.sealed_env_file(job_id).write_bytes(pyrage.encrypt(b"HF_TOKEN=x\n", [other]))
    sealed.recipient()
    assert run(job_id, deps()) != 0
    state = jobs.read_state(job_id)
    assert (state.status, state.reason, state.ran) == ("failed", "secrets", False)
    assert "could not open the job's secrets" in log_of(job_id)


def test_removing_a_jobs_secrets_removes_both_kinds(gpuc_home: Path) -> None:
    paths.sealed_env_file("j").write_text("sealed")
    paths.job_env_file("j").write_text("plain")
    cleanup.remove_secrets("j")
    assert not paths.sealed_env_file("j").exists()
    assert not paths.job_env_file("j").exists()


def test_a_host_without_uv_says_so(gpuc_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sealed.shutil, "which", lambda *_a, **_k: None)
    paths.sealed_env_file("j").write_text("sealed")
    with pytest.raises(sealed.SealedError, match="uv is not installed"):
        sealed.job_secrets("j")


def test_a_key_whose_public_half_was_never_written_is_derived_again(gpuc_home: Path) -> None:
    first = sealed.recipient()
    paths.host_recipient_file().unlink()
    assert sealed.recipient() == first
