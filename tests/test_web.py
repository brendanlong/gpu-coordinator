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
from gpuc.control.status import HostState, HostView
from gpuc.control.web.app import Dashboard, ServerThread
from gpuc.control.web.auth import (
    SESSION_COOKIE,
    Sessions,
    hash_password,
    password_file,
    read_password_hash,
    verify,
)
from gpuc.host.jobs import HostConfig
from tests.conftest import host_entry, register_host

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
    view = HostView(entry=entry, state=HostState.ANSWERED, owned=[GPU], heartbeat_age_s=2.0)
    view.queue, view.running, view.finished = status_mod.job_views(
        {
            "queue": [{"priority": 50, "job_id": "20260915-130000-aaaaaa"}],
            "jobs": [
                {
                    "job_id": "20260915-130000-aaaaaa",
                    "name": "next",
                    "status": "queued",
                    "priority": 50,
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
                    "reason": "timeout",
                    "ended_at": minutes_ago(60),
                },
            ],
        }
    )
    return view


@pytest.fixture
def one_host(control_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    register_host(name="gpubox", kind="ssh", ssh="me@box", gpus=GPU)
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
    assert status == 401
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
    assert document["settings"]["disk_gb"] == 50
    assert "config_file" in document


def test_logout_ends_the_session(logged_in: Client) -> None:
    status, headers, _ = logged_in.request("POST", "/logout")
    assert (status, headers["location"]) == (303, "/login")
    status, _, _ = logged_in.get("/")
    assert status == 303
    logged_in.cookies[SESSION_COOKIE] = "a-token-the-server-forgot"
    status, _, _ = logged_in.get("/api/status")
    assert status == 401


def test_a_quiet_minute_forgets_the_run_of_failures(password_hash: str) -> None:
    now = [0.0]
    slept: list[float] = []
    sessions = Sessions(password_hash, clock=lambda: now[0], sleep=slept.append)
    assert sessions.login("wrong") is None
    assert sessions.login("wrong") is None
    now[0] = 61.0
    assert sessions.login(PASSWORD) is not None
    assert slept == [0.5]


def test_every_bad_request_still_gets_a_status_line(client: Client) -> None:
    """A dropped connection tells a browser nothing; these each used to be one."""
    status, _, _ = client.get("/static/missing.js")
    assert status == 404
    status, _, data = client.request("POST", "/login", b"x", {"Content-Length": "abc"})
    assert status == 400 and b"Content-Length" in data
    # A declared 2 MiB with one byte actually sent: the server must answer
    # from the header alone, before reading a body it is about to refuse. (A
    # client that sends the whole body sees the same 413, or a broken pipe
    # once the server has closed on it, depending on the race.)
    status, _, data = client.request(
        "POST",
        "/login",
        b"a",
        {"Content-Type": "application/x-www-form-urlencoded", "Content-Length": str(2 << 20)},
    )
    assert status == 413 and b"over" in data
    status, _, _ = client.get("/login")
    assert status == 200


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
    assert host["finished"][0]["reason"] == "timeout"


def test_status_narrows_to_one_host_and_refuses_a_bad_since(
    logged_in: Client, one_host: None
) -> None:
    status, document = logged_in.get_json("/api/status?host=gpubox&recent=1&since=24h")
    assert status == 200 and len(document["hosts"]) == 1
    status, document = logged_in.get_json("/api/status?host=nope")
    assert status == 200 and document["hosts"][0]["state"] == "gone"
    status, document = logged_in.get_json("/api/status?since=soon")
    assert status == 400 and document["exit_code"] == EXIT_USAGE


def test_all_is_a_flag_and_zero_means_no(logged_in: Client, one_host: None) -> None:
    """`bool("0")` is True: `?all=0` used to list every index-only job."""
    from gpuc.control.s3index import IndexEntry, LocalIndex

    LocalIndex().record(IndexEntry(job_id="20260101-000000-aaaaaa", host="gone-box", name="lost"))
    for query, listed in [
        ("all=1", True),
        ("all=true", True),
        ("all=0", False),
        ("all=false", False),
        ("all=", False),
        ("", False),
    ]:
        status, document = logged_in.get_json(f"/api/status?{query}")
        assert status == 200, query
        assert bool(document["unhosted"]) is listed, query


def test_hosts_and_version_are_the_list_and_version_documents(
    logged_in: Client, one_host: None
) -> None:
    status, document = logged_in.get_json("/api/hosts")
    assert status == 200
    (host,) = document["hosts"]
    assert (host["name"], host["ssh"], host["remote_home"]) == ("gpubox", "me@box", "$HOME/.gpuc")
    status, document = logged_in.get_json("/api/version")
    assert status == 200 and document["version"]


def test_a_host_that_could_not_be_read_is_the_clis_exit_one_on_the_wire(
    logged_in: Client, control_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One table from exit code to HTTP status: a status the CLI exits 1 on is
    a 500 that still carries every host, not a 200 that hides the failure."""
    from gpuc.control.exits import EXIT_ERROR, http_status

    register_host(name="gpubox", kind="ssh", ssh="me@box", gpus=GPU)

    def unreachable(entry: HostEntry, *a: object, **k: object) -> HostView:
        return HostView(entry=entry, state=HostState.UNASKABLE, error="ssh timed out")

    monkeypatch.setattr(status_mod, "gather", unreachable)
    status, document = logged_in.get_json("/api/status")
    assert status == http_status(EXIT_ERROR) == 500
    (host,) = document["hosts"]
    assert host["reachable"] is False and host["errors"] == ["ssh timed out"]
    assert "error" not in document


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
        self.entry = host_entry(name="gpubox", kind="ssh", ssh="me@box", gpus=[GPU])
        self.config = HostConfig(host="gpubox", gpus=[GPU])

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
        "gpuc.control.remote.open_session", lambda *a, **k: cast("HostSession", session)
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
        "source": "host",
    }
    assert stub.commands == [f"cancel {RUNNING_JOB}"]


def test_preempt_is_the_preempt_command(logged_in: Client, stub: StubSession) -> None:
    stub.answers.append({"job_id": RUNNING_JOB, "status": "preempting", "priority": 50})
    status, document = logged_in.post_json(f"/api/jobs/{RUNNING_JOB}/preempt", {"host": "gpubox"})
    assert status == 200
    assert (document["status"], document["priority"]) == ("preempting", 50)
    assert stub.commands == [f"preempt {RUNNING_JOB}"]
    status, document = logged_in.post_json(
        f"/api/jobs/{RUNNING_JOB}/preempt", {"host": "gpubox", "priority": "7"}
    )
    assert status == 400 and "0-99" in document["error"]


def test_reorder_is_the_reorder_command_and_checks_the_range(
    logged_in: Client, stub: StubSession
) -> None:
    stub.answers.append({"job_id": RUNNING_JOB, "status": "queued", "priority": 7})
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
    stub.answers.append({"job_id": RUNNING_JOB, "error": "only a queued job can be reordered"})
    status, document = logged_in.post_json(
        f"/api/jobs/{RUNNING_JOB}/reorder", {"host": "gpubox", "priority": 7}
    )
    assert status == 500
    assert "only a queued job can be reordered" in document["error"]
    assert document["exit_code"] == 1


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


def test_an_unknown_job_is_404(logged_in: Client, stub: StubSession) -> None:
    """Every host answered and none has it: exit 4, HTTP 404."""
    stub.answers.append({"jobs": []})
    status, document = logged_in.post_json("/api/jobs/20260101-000000-aaaaaa/cancel", {})
    assert status == 404
    assert "no registered host knows job" in document["error"]


def test_a_host_that_cannot_be_asked_is_a_failure_not_a_missing_job(
    logged_in: Client, one_host: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The one host that may hold the job is down: exit 1 and HTTP 500 with
    the reason, never "no such job" told over a connection error."""

    def down(*a: object, **k: object) -> Any:
        raise RemoteError("gpubox", "status", "ssh timed out")

    monkeypatch.setattr("gpuc.control.remote.open_session", down)
    status, document = logged_in.post_json("/api/jobs/20260101-000000-aaaaaa/cancel", {})
    assert status == 500 and document["exit_code"] == 1
    assert "gpubox: ssh timed out" in document["error"]
    assert "no registered host knows" not in document["error"]


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


# -- `serve --install` ----------------------------------------------------------


def test_serve_install_writes_the_unit_and_does_not_enable_it(
    control_env: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from gpuc.control.web import service as service_mod

    units = tmp_path / "systemd-user"
    monkeypatch.setattr(service_mod, "systemd_dir", lambda: units)

    def refuse(*a: object, **k: object) -> Any:
        raise AssertionError("enabling is the user's call: nothing may spawn a process")

    # Everything in `subprocess` funnels through Popen; and `--install` that
    # fell through to the server would sit in serve_forever, not fail.
    monkeypatch.setattr("subprocess.Popen", refuse)
    monkeypatch.setattr(web, "make_server", refuse)
    assert main(["web", "serve", "--bind", "0.0.0.0", "--port", "9000", "--install"]) == 0
    unit = (units / "gpuc-web.service").read_text()
    assert "ExecStart=/" in unit
    assert "WantedBy=default.target" in unit
    exec_line = next(line for line in unit.splitlines() if line.startswith("ExecStart="))
    assert exec_line.endswith("web serve --bind 0.0.0.0 --port 9000")
    assert f"Environment=GPUC_CONFIG_DIR={control_env / 'config'}" in unit
    assert f"Environment=GPUC_STATE_DIR={control_env / 'state'}" in unit
    assert f"EnvironmentFile=-{control_env / 'config'}/env" in unit
    assert "Restart=on-failure" in unit
    assert "StartLimitBurst=5" in unit
    out = capsys.readouterr().out
    assert "not enabled" in out
    assert "systemctl --user enable --now gpuc-web.service" in out
    assert "gpuc web set-password" in out, "no password is set, and the unit would only crash"


def test_serve_install_is_quiet_about_the_password_once_one_is_set(
    control_env: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    password_hash: str,
) -> None:
    from gpuc.control.web import service as service_mod

    monkeypatch.setattr(service_mod, "systemd_dir", lambda: tmp_path / "units")
    path = password_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(password_hash + "\n")
    assert main(["web", "serve", "--install"]) == 0
    out = capsys.readouterr().out
    assert "set-password" not in out
    assert (tmp_path / "units" / "gpuc-web.service").exists()


def test_serve_install_refuses_a_bind_that_would_rewrite_the_unit(
    control_env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gpuc.control.web import service as service_mod

    monkeypatch.setattr(service_mod, "systemd_dir", lambda: tmp_path / "units")
    bad = "0.0.0.0\nExecStart=/bin/echo pwned"
    assert main(["web", "serve", "--bind", bad, "--install"]) == EXIT_USAGE
    assert not (tmp_path / "units").exists()


def test_exec_start_words_are_quoted_the_way_systemd_reads_them() -> None:
    from gpuc.control.systemd import quote

    assert quote("/home/me/.local/bin/gpuc") == "/home/me/.local/bin/gpuc"
    assert quote("--bind") == "--bind"
    assert quote("/home/my name/gpuc") == '"/home/my name/gpuc"'
    assert quote("/home/100%/gpuc") == '"/home/100%%/gpuc"'
    assert quote('a"b\\c') == '"a\\"b\\\\c"'
