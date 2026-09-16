"""The dashboard: one password in front of the CLI's own documents and actions.

Served on a real socket on a background thread, driven with `http.client`, so
what is tested is the wire: cookies, redirects, status codes, and that every
API document is the one `--json` prints.
"""

from __future__ import annotations

import http.client
import io
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlencode, urlsplit

import pytest

from gpuc.control import status as status_mod
from gpuc.control import web
from gpuc.control.cli import EXIT_USAGE, main
from gpuc.control.config import HostEntry, Settings, hosts_file
from gpuc.control.remote import HostSession, RemoteError
from gpuc.control.status import HostView
from gpuc.control.web.app import Dashboard, ServerThread
from gpuc.control.web.auth import (
    SESSION_COOKIE,
    Sessions,
    hash_password,
    password_file,
    read_password_hash,
    verify,
)

PASSWORD = "correct horse battery"
GPU = "GPU-2a4bad3b-9fe3-7031-914d-384254e92908"
RUNNING_JOB = "20260915-120000-abc123"
FINISHED_JOB = "20260915-110000-def456"


class Client:
    """A cookie-keeping HTTP client, one connection per request."""

    def __init__(self, url: str) -> None:
        parts = urlsplit(url)
        self.host = parts.hostname or "127.0.0.1"
        self.port = parts.port or 80
        self.cookies: dict[str, str] = {}

    def request(
        self,
        method: str,
        path: str,
        body: bytes | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, dict[str, str], bytes]:
        connection = http.client.HTTPConnection(self.host, self.port, timeout=30)
        sent = dict(headers or {})
        if self.cookies:
            sent["Cookie"] = "; ".join(f"{k}={v}" for k, v in self.cookies.items())
        connection.request(method, path, body=body, headers=sent)
        response = connection.getresponse()
        data = response.read()
        got = {key.lower(): value for key, value in response.getheaders()}
        for header, value in response.getheaders():
            if header.lower() == "set-cookie":
                name, _, rest = value.partition("=")
                token = rest.split(";", 1)[0]
                if token:
                    self.cookies[name] = token
                else:
                    self.cookies.pop(name, None)
        connection.close()
        return response.status, got, data

    def get(self, path: str) -> tuple[int, dict[str, str], bytes]:
        return self.request("GET", path)

    def get_json(self, path: str) -> tuple[int, dict[str, Any]]:
        status, _, data = self.get(path)
        return status, json.loads(data)

    def post_json(
        self, path: str, body: dict[str, Any] | None = None, **headers: str
    ) -> tuple[int, dict[str, Any]]:
        status, _, data = self.request(
            "POST",
            path,
            json.dumps(body or {}).encode(),
            {"Content-Type": "application/json", **headers},
        )
        return status, json.loads(data)

    def login(self, password: str = PASSWORD) -> tuple[int, dict[str, str], bytes]:
        return self.request(
            "POST",
            "/login",
            urlencode({"password": password}).encode(),
            {"Content-Type": "application/x-www-form-urlencoded"},
        )


@pytest.fixture(scope="module")
def password_hash() -> str:
    # bcrypt is slow by design; one hash for the module is plenty.
    return hash_password(PASSWORD)


@pytest.fixture
def app(password_hash: str) -> Dashboard:
    return Dashboard(Sessions(password_hash, sleep=lambda _: None), settings=Settings)


@pytest.fixture
def client(app: Dashboard, control_env: Path):
    with ServerThread(app) as server:
        yield Client(server.url)


@pytest.fixture
def logged_in(client: Client) -> Client:
    status, _, _ = client.login()
    assert status == 303
    return client


def minutes_ago(minutes: float) -> str:
    return (datetime.now(UTC) - timedelta(minutes=minutes)).isoformat()


def fake_host_view(entry: HostEntry, *a: object, **k: object) -> HostView:
    view = HostView(entry=entry, reachable=True, owned=[GPU], heartbeat_age_s=2.0)
    view.queue, view.running, view.finished = status_mod.job_views(
        {
            "queue": [{"priority": 50, "job_id": "20260915-130000-aaaaaa"}],
            "jobs": [
                {
                    "job_id": "20260915-130000-aaaaaa",
                    "name": "next",
                    "status": "queued",
                    "estimated_runtime_min": 60,
                },
                {
                    "job_id": RUNNING_JOB,
                    "name": "lego-s4",
                    "status": "running",
                    "phase": "main",
                    "gpus": [GPU],
                    "started_at": minutes_ago(30),
                    "util_recent": [90.0],
                    "outputs": [{"path": "results", "s3": f"s3://bucket/lego/{RUNNING_JOB}"}],
                    "wandb": {"entity": "me", "project": "lego", "run_id": "r1"},
                },
                {
                    "job_id": FINISHED_JOB,
                    "name": "probe",
                    "status": "failed",
                    "reason": "low-util",
                    "ended_at": minutes_ago(60),
                },
            ],
        }
    )
    return view


@pytest.fixture
def one_host(control_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    main(["host", "add", "gpubox", "--ssh", "me@box", "--gpus", GPU])
    monkeypatch.setattr(status_mod, "gather", fake_host_view)


# -- the password -------------------------------------------------------------


def test_set_password_writes_a_bcrypt_hash_0600(
    control_env: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "stdin", io.StringIO(f"{PASSWORD}\n"))
    assert main(["web", "set-password", "--stdin"]) == 0
    path = password_file()
    assert path.stat().st_mode & 0o777 == 0o600
    assert PASSWORD not in path.read_text()
    assert verify(PASSWORD, read_password_hash())
    assert not verify("something else", read_password_hash())
    assert "gpuc web serve" in capsys.readouterr().out


def test_set_password_refuses_a_short_one(
    control_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "stdin", io.StringIO("short\n"))
    assert main(["web", "set-password", "--stdin"]) == EXIT_USAGE
    assert not password_file().exists()


def test_serve_refuses_to_start_without_a_password(
    control_env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["web", "serve", "--port", "0"]) == 1
    assert "gpuc web set-password" in capsys.readouterr().err


def test_a_password_over_bcrypts_limit_is_wrong_not_an_exception(password_hash: str) -> None:
    assert verify("x" * 100, password_hash) is False


def test_sessions_throttle_a_run_of_wrong_passwords(password_hash: str) -> None:
    slept: list[float] = []
    sessions = Sessions(password_hash, sleep=slept.append)
    assert sessions.login("wrong") is None
    assert sessions.login("wrong") is None
    assert sessions.login("wrong") is None
    assert slept == [0.5, 1.0]
    token = sessions.login(PASSWORD)
    assert token is not None and sessions.check(token)
    assert sessions.failures == 0


def test_sessions_expire(password_hash: str) -> None:
    now = [0.0]
    sessions = Sessions(password_hash, ttl_s=10.0, clock=lambda: now[0], sleep=lambda _: None)
    token = sessions.login(PASSWORD)
    assert sessions.check(token)
    now[0] = 11.0
    assert not sessions.check(token)


# -- the front door -------------------------------------------------------------


def test_the_page_redirects_to_login_without_a_session(client: Client) -> None:
    status, headers, _ = client.get("/")
    assert (status, headers["location"]) == (303, "/login")
    status, _, body = client.get("/login")
    assert status == 200
    assert b'name="password"' in body


def test_the_api_refuses_without_a_session_rather_than_redirecting(client: Client) -> None:
    status, document = client.get_json("/api/status")
    assert status == 400
    assert document["error"] == "not logged in"
    assert document["schema_version"] == 1


def test_a_wrong_password_is_told_so_and_gets_no_cookie(client: Client) -> None:
    status, _, body = client.login("not it")
    assert status == 401
    assert b"wrong password" in body
    assert SESSION_COOKIE not in client.cookies


def test_login_sets_a_cookie_that_opens_the_page_and_the_api(client: Client) -> None:
    status, headers, _ = client.login()
    assert (status, headers["location"]) == (303, "/")
    cookie = headers["set-cookie"]
    assert "HttpOnly" in cookie and "SameSite=Strict" in cookie
    status, _, body = client.get("/")
    assert status == 200
    assert b"gpuc dashboard" in body
    status, _, body = client.get("/static/app.js")
    assert status == 200 and b"/api/status" in body
    status, document = client.get_json("/api/config")
    assert status == 200
    assert document["settings"]["max_pods"] == 3
    assert "config_file" in document


def test_logout_ends_the_session(logged_in: Client) -> None:
    status, headers, _ = logged_in.request("POST", "/logout")
    assert (status, headers["location"]) == (303, "/login")
    status, _, _ = logged_in.get("/")
    assert status == 303


def test_a_post_from_another_origin_is_refused(logged_in: Client) -> None:
    status, document = logged_in.post_json(
        f"/api/jobs/{RUNNING_JOB}/cancel", {}, Origin="http://evil.example"
    )
    assert status == 400
    assert "cross-origin" in document["error"]


# -- the documents are the CLI's ----------------------------------------------


def test_status_is_the_status_json_document(logged_in: Client, one_host: None) -> None:
    status, document = logged_in.get_json("/api/status")
    assert status == 200
    assert document["schema_version"] == 1 and document["errors"] == []
    assert document["gathered_at"]
    (host,) = document["hosts"]
    assert (host["name"], host["kind"], host["target"], host["reachable"]) == (
        "gpubox",
        "ssh",
        "me@box",
        True,
    )
    assert host["dispatcher"]["alive"] is True
    assert [j["job_id"] for j in host["running"]] == [RUNNING_JOB]
    running = host["running"][0]
    assert running["util"] == 90.0
    kinds = {link["kind"]: link for link in running["links"]}
    assert kinds["s3"]["url"] == (
        f"https://s3.console.aws.amazon.com/s3/buckets/bucket?prefix=lego/{RUNNING_JOB}/"
    )
    assert kinds["wandb"]["url"] == "https://wandb.ai/me/lego/runs/r1"
    assert host["queued"][0]["priority"] == 50
    assert host["finished"][0]["reason"] == "low-util"


def test_status_narrows_to_one_host_and_refuses_a_bad_since(
    logged_in: Client, one_host: None
) -> None:
    status, document = logged_in.get_json("/api/status?host=gpubox&recent=1&since=24h")
    assert status == 200 and len(document["hosts"]) == 1
    status, document = logged_in.get_json("/api/status?host=nope")
    assert status == 404 and "no host named 'nope'" in document["error"]
    status, document = logged_in.get_json("/api/status?since=soon")
    assert status == 400 and document["exit_code"] == EXIT_USAGE


def test_hosts_and_version_are_the_list_and_version_documents(
    logged_in: Client, one_host: None
) -> None:
    status, document = logged_in.get_json("/api/hosts")
    assert status == 200
    (host,) = document["hosts"]
    assert (host["name"], host["ssh"], host["remote_home"]) == ("gpubox", "me@box", "$HOME/.gpuc")
    status, document = logged_in.get_json("/api/version")
    assert status == 200 and document["version"]


def test_an_unreadable_registry_is_503_with_the_reason(logged_in: Client) -> None:
    path = hosts_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{ not json")
    status, document = logged_in.get_json("/api/status")
    assert status == 503
    assert document["hosts"] == []
    assert "could not be read" in document["errors"][-1]


class StubSession:
    """A `HostSession` that answers `host_json`/`host_cli`/`tail` from a script."""

    def __init__(self, answers: list[dict[str, Any]], *, log: str = "line 1\nline 2\n") -> None:
        self.answers = answers
        self.commands: list[str] = []
        self.log = log
        self.transport = self

    def host_json(self, args: str, **_: Any) -> Any:
        self.commands.append(args)
        return self.answers.pop(0)

    def host_cli(self, args: str, **_: Any) -> Any:
        self.commands.append(args)
        answer = self.answers.pop(0)
        return type("Result", (), {"returncode": answer.get("returncode", 0), "output": ""})()

    def job_dir(self, job_id: str) -> str:
        return f"/home/me/.gpuc/jobs/{job_id}"

    def tail(self, remote: str, *, lines: int) -> Any:
        self.commands.append(f"tail -n {lines} {remote}")
        return type("Result", (), {"returncode": 0, "stdout": self.log, "output": self.log})()


@pytest.fixture
def stub(monkeypatch: pytest.MonkeyPatch, one_host: None) -> StubSession:
    session = StubSession([])
    monkeypatch.setattr(
        "gpuc.control.actions.open_session", lambda *a, **k: cast("HostSession", session)
    )
    return session


def test_cancel_is_the_cancel_command(logged_in: Client, stub: StubSession) -> None:
    stub.answers.append({"job_id": RUNNING_JOB, "status": "cancelling"})
    status, document = logged_in.post_json(f"/api/jobs/{RUNNING_JOB}/cancel", {"host": "gpubox"})
    assert status == 200
    assert document == {
        "schema_version": 1,
        "job_id": RUNNING_JOB,
        "host": "gpubox",
        "status": "cancelling",
    }
    assert stub.commands == [f"cancel {RUNNING_JOB}"]


def test_reorder_is_the_reorder_command_and_checks_the_range(
    logged_in: Client, stub: StubSession
) -> None:
    stub.answers.append({"returncode": 0})
    status, document = logged_in.post_json(
        f"/api/jobs/{RUNNING_JOB}/reorder", {"host": "gpubox", "priority": 7}
    )
    assert status == 200 and document["priority"] == 7
    assert stub.commands == [f"reorder {RUNNING_JOB} 7"]
    status, document = logged_in.post_json(
        f"/api/jobs/{RUNNING_JOB}/reorder", {"host": "gpubox", "priority": 100}
    )
    assert status == 400 and "0-99" in document["error"]
    status, document = logged_in.post_json(
        f"/api/jobs/{RUNNING_JOB}/reorder", {"host": "gpubox", "priority": "7"}
    )
    assert status == 400


def test_a_refused_reorder_is_the_clis_refusal(logged_in: Client, stub: StubSession) -> None:
    stub.answers.append({"returncode": 1})
    status, document = logged_in.post_json(
        f"/api/jobs/{RUNNING_JOB}/reorder", {"host": "gpubox", "priority": 7}
    )
    assert status == 500
    assert "cannot be reordered" in document["error"] and document["exit_code"] == 1


def test_estimate_sets_and_clears(logged_in: Client, stub: StubSession) -> None:
    stub.answers.append(
        {"job_id": RUNNING_JOB, "estimated_runtime_min": 90.0, "status": "running", "warning": None}
    )
    status, document = logged_in.post_json(
        f"/api/jobs/{RUNNING_JOB}/estimate", {"host": "gpubox", "minutes": 90}
    )
    assert status == 200 and document["estimated_runtime_min"] == 90.0
    stub.answers.append(
        {"job_id": RUNNING_JOB, "estimated_runtime_min": None, "status": "running", "warning": None}
    )
    status, document = logged_in.post_json(
        f"/api/jobs/{RUNNING_JOB}/estimate", {"host": "gpubox", "clear": True}
    )
    assert status == 200 and document["estimated_runtime_min"] is None
    assert stub.commands == [f"estimate {RUNNING_JOB} 90.0", f"estimate {RUNNING_JOB} --clear"]
    status, document = logged_in.post_json(
        f"/api/jobs/{RUNNING_JOB}/estimate", {"host": "gpubox", "minutes": -5}
    )
    assert status == 400


def test_logs_is_the_logs_document(logged_in: Client, stub: StubSession) -> None:
    status, document = logged_in.get_json(f"/api/jobs/{RUNNING_JOB}/logs?host=gpubox&lines=2")
    assert status == 200
    assert document["lines"] == ["line 1", "line 2"]
    assert (document["source"], document["host"]) == ("host", "gpubox")
    assert document["location"].endswith(f"/jobs/{RUNNING_JOB}/log.txt")
    assert stub.commands == [f"tail -n 2 /home/me/.gpuc/jobs/{RUNNING_JOB}/log.txt"]


def test_a_job_id_that_is_not_one_is_refused_before_any_host(
    logged_in: Client, stub: StubSession
) -> None:
    status, document = logged_in.get_json("/api/jobs/%3Bls/logs?host=gpubox")
    assert status == 400 and "not a job id" in document["error"]
    assert stub.commands == []


def test_an_unknown_job_is_404(
    logged_in: Client, one_host: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The host is asked and cannot answer, so the job is nowhere: exit 4, HTTP 404."""

    def down(*a: object, **k: object) -> Any:
        raise RemoteError("gpubox", "status", "ssh timed out")

    monkeypatch.setattr("gpuc.control.actions.open_session", down)
    status, document = logged_in.post_json("/api/jobs/20260101-000000-aaaaaa/cancel", {})
    assert status == 404
    assert "no registered host knows job" in document["error"]


def test_a_bad_body_is_400(logged_in: Client, one_host: None) -> None:
    status, _, data = logged_in.request(
        "POST", f"/api/jobs/{RUNNING_JOB}/cancel", b"[]", {"Content-Type": "application/json"}
    )
    assert status == 400 and b"JSON object" in data
    status, _, data = logged_in.request(
        "POST", f"/api/jobs/{RUNNING_JOB}/cancel", b"{", {"Content-Type": "application/json"}
    )
    assert status == 400


def test_unknown_paths(logged_in: Client) -> None:
    status, document = logged_in.get_json("/api/nothing")
    assert status == 404 and "no such endpoint" in document["error"]
    status, _, _ = logged_in.get("/nothing")
    assert status == 404
    status, _, _ = logged_in.request("POST", "/api/status")
    assert status == 405


def test_serve_binds_and_stops(
    control_env: Path, monkeypatch: pytest.MonkeyPatch, password_hash: str
) -> None:
    path = password_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(password_hash + "\n")
    server = web.make_server("127.0.0.1", 0)
    try:
        assert server.server_address[1] > 0
    finally:
        server.server_close()
