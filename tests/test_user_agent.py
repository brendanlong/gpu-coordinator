"""One user agent, one version, everywhere we talk to somebody else's service."""

from __future__ import annotations

import tomllib
from pathlib import Path

from gpuc._version import __version__, user_agent
from gpuc.control.bootstrap import package_files, uv_installer_command
from gpuc.control.providers import runpod
from gpuc.host import USER_AGENT
from gpuc.host.runner import build_env

from .conftest import make_spec

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED = (
    f"gpuc/{__version__} (+https://github.com/brendanlong/gpu-coordinator; self@brendanlong.com)"
)


def test_version_matches_pyproject() -> None:
    document = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    assert document["project"]["version"] == __version__


def test_user_agent_is_exactly_the_documented_string() -> None:
    assert user_agent() == EXPECTED


def test_every_caller_uses_the_same_string() -> None:
    assert USER_AGENT == EXPECTED
    assert runpod.USER_AGENT == EXPECTED


def test_bootstrap_ships_the_version_module_and_sends_the_agent() -> None:
    assert "gpuc/_version.py" in package_files()
    assert f"-A '{EXPECTED}'" in uv_installer_command()


def test_boto3_clients_add_the_agent(monkeypatch) -> None:
    seen: dict[str, object] = {}

    def fake_client(service: str, **kwargs: object) -> str:
        seen.update({"service": service, **kwargs})
        return "client"

    monkeypatch.setattr("boto3.client", fake_client)
    from gpuc.control.s3index import make_s3_client

    assert make_s3_client() == "client"
    assert seen["service"] == "s3"
    assert seen["config"].user_agent_extra == EXPECTED  # type: ignore[attr-defined]


def test_a_job_tells_hugging_face_who_it_is(gpuc_home) -> None:
    env = build_env(make_spec(), [])
    assert env["HF_HUB_USER_AGENT_ORIGIN"] == EXPECTED


def test_a_job_may_override_the_hf_origin(gpuc_home) -> None:
    env = build_env(make_spec(env={"HF_HUB_USER_AGENT_ORIGIN": "mine"}), [])
    assert env["HF_HUB_USER_AGENT_ORIGIN"] == "mine"
