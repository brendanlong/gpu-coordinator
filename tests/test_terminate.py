from __future__ import annotations

import io
import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, cast

import pytest

from gpuc.host import paths, terminate
from gpuc.host.jobs import HostConfig

POD_CONFIG = HostConfig(host="pod-1", provider={"kind": "runpod", "pod_id": "abc123"})


@pytest.fixture
def no_pod_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    monkeypatch.delenv("RUNPOD_POD_ID", raising=False)
    rp = tmp_path / "rp_environment"
    monkeypatch.setenv("GPUC_RP_ENVIRONMENT", str(rp))
    return rp


def test_api_key_prefers_the_secrets_file_override(
    gpuc_home: Path, no_pod_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RUNPOD_API_KEY", "from-env")
    no_pod_env.write_text("export RUNPOD_API_KEY=from-file\n")
    paths.secrets_file("runpod").write_text("from-secrets\n")
    assert terminate.read_api_key() == "from-secrets"


def test_api_key_falls_back_to_the_pod_env(
    gpuc_home: Path, no_pod_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RUNPOD_API_KEY", "from-env")
    assert terminate.read_api_key() == "from-env"


def test_api_key_falls_back_to_rp_environment(gpuc_home: Path, no_pod_env: Path) -> None:
    no_pod_env.write_text(
        "export RUNPOD_POD_ID=pod-xyz\n"
        'export RUNPOD_API_KEY="rpa_secret"\n'
        "# a comment\n"
        "NOT_AN_ASSIGNMENT\n"
    )
    assert terminate.read_api_key() == "rpa_secret"


def test_rp_environment_parsing_strips_quotes_and_export(no_pod_env: Path) -> None:
    no_pod_env.write_text("export A=1\nB='two'\nexport C=\"three\"\n\n#D=4\nE=with=equals\n")
    assert terminate.rp_environment() == {
        "A": "1",
        "B": "two",
        "C": "three",
        "E": "with=equals",
    }


def test_rp_environment_missing_file_is_empty(no_pod_env: Path) -> None:
    assert terminate.rp_environment() == {}


def test_pod_id_resolution_order(
    gpuc_home: Path, no_pod_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = {"kind": "runpod", "pod_id": "from-config"}
    assert terminate.read_pod_id(provider) == "from-config"
    no_pod_env.write_text("export RUNPOD_POD_ID=from-rp-env\n")
    assert terminate.read_pod_id(provider) == "from-rp-env"
    monkeypatch.setenv("RUNPOD_POD_ID", "from-env")
    assert terminate.read_pod_id(provider) == "from-env"


def test_missing_api_key_is_a_clear_error(gpuc_home: Path, no_pod_env: Path) -> None:
    paths.secrets_file("runpod").write_text("   \n")
    with pytest.raises(terminate.TerminateError) as excinfo:
        terminate.read_api_key()
    assert "RUNPOD_API_KEY is unset" in str(excinfo.value)
    assert str(no_pod_env) in str(excinfo.value)


def test_missing_pod_id_is_a_clear_error(gpuc_home: Path, no_pod_env: Path) -> None:
    with pytest.raises(terminate.TerminateError) as excinfo:
        terminate.read_pod_id({"kind": "runpod"})
    assert "RUNPOD_POD_ID is unset" in str(excinfo.value)


def test_self_terminate_uses_the_injected_call(
    gpuc_home: Path, no_pod_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RUNPOD_API_KEY", "k")
    seen: list[tuple[str, str]] = []

    def fake(pod_id: str, key: str) -> str:
        seen.append((pod_id, key))
        return "{}"

    terminate.self_terminate(POD_CONFIG, terminate_call=fake)
    assert seen == [("abc123", "k")]


def test_self_terminate_prefers_the_pods_own_id(
    gpuc_home: Path, no_pod_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RUNPOD_API_KEY", "k")
    no_pod_env.write_text("export RUNPOD_POD_ID=real-pod\n")
    seen: list[tuple[str, str]] = []
    terminate.self_terminate(POD_CONFIG, terminate_call=lambda p, k: seen.append((p, k)) or "")
    assert seen == [("real-pod", "k")]


def test_self_terminate_refuses_a_non_provider_host() -> None:
    with pytest.raises(terminate.TerminateError) as excinfo:
        terminate.self_terminate(HostConfig(host="desktop"))
    assert "no provider" in str(excinfo.value)


def test_runpod_terminate_posts_the_v2_action(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class FakeResponse:
        def read(self) -> bytes:
            return b'{"status":"ok"}'

        def __enter__(self) -> FakeResponse:
            return self

        def __exit__(self, *args: object) -> None:
            return None

    def fake_urlopen(request: urllib.request.Request, timeout: float = 0.0) -> FakeResponse:
        captured["url"] = request.full_url
        captured["method"] = request.get_method()
        captured["body"] = json.loads(cast("bytes", request.data))
        captured["auth"] = request.get_header("Authorization")
        captured["agent"] = request.get_header("User-agent")
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    body = terminate.runpod_terminate("pod-9", "secret")
    assert body == '{"status":"ok"}'
    assert captured["url"] == "https://api.runpod.io/v2/pods/pod-9/action"
    assert captured["method"] == "POST"
    assert captured["body"] == {"action": "terminate"}
    assert captured["auth"] == "Bearer secret"
    # Cloudflare 403s urllib's default User-Agent.
    assert captured["agent"] == terminate.USER_AGENT
    assert captured["timeout"] > 0


def test_runpod_terminate_surfaces_http_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(request: urllib.request.Request, timeout: float = 0.0) -> None:
        raise urllib.error.HTTPError(
            request.full_url,
            401,
            "Unauthorized",
            {},  # type: ignore[arg-type]
            io.BytesIO(b"bad key"),
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(terminate.TerminateError) as excinfo:
        terminate.runpod_terminate("pod-9", "secret")
    assert "HTTP 401" in str(excinfo.value)
    assert "bad key" in str(excinfo.value)
