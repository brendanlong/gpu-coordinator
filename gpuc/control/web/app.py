"""The dashboard's HTTP surface: a login page, a static page, and a JSON API.

Every API document is one a `--json` command already prints, built by the same
function in `gpuc.control.actions`, and every action is one the CLI has. The
page is static HTML and JavaScript that polls the API, so realtime updates
later mean adding an event stream beside the same documents, not a second
rendering of the world.

Stdlib `http.server` on purpose: the control side may use third-party
packages, but a framework buys nothing a routing table this size needs, and
`ThreadingHTTPServer` already handles a slow host's ssh on one request without
blocking the next.
"""

from __future__ import annotations

import html
import json
import re
import sys
import threading
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from http import HTTPStatus
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from typing import Any
from urllib.parse import parse_qs, urlsplit

from gpuc.control import status as status_mod
from gpuc.control.actions import (
    EXIT_LOCAL_STATE,
    EXIT_NOT_FOUND,
    EXIT_USAGE,
    UsageError,
    cancel_job,
    check_estimate,
    config_document,
    estimate_job,
    exit_code_for,
    failure_message,
    hosts_document,
    preempt_job,
    read_log,
    reorder_job,
    status_document,
    version_document,
)
from gpuc.control.config import Settings, load_settings, read_registry, utc_now
from gpuc.control.web.auth import SESSION_COOKIE, SESSION_TTL_S, Sessions, read_password_hash
from gpuc.host.jobs import SCHEMA_VERSION

DEFAULT_BIND = "127.0.0.1"
DEFAULT_PORT = 8646
MAX_BODY_BYTES = 1 << 20
MAX_LOG_LINES = 5000
JOB_ID = re.compile(r"^[A-Za-z0-9._-]+$")
"""Job ids are `YYYYMMDD-HHMMSS-hex`; anything outside this set is not one and
is refused before it can reach a shell, quoted or not."""

HTTP_FOR_EXIT = {
    EXIT_USAGE: HTTPStatus.BAD_REQUEST,
    EXIT_LOCAL_STATE: HTTPStatus.SERVICE_UNAVAILABLE,
    EXIT_NOT_FOUND: HTTPStatus.NOT_FOUND,
}
"""The CLI's exit codes, on the wire. Anything else the CLI would exit 1 on --
an unreachable host, a refused reorder -- is a 500: the command failed."""

STATIC_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
}


@dataclass
class Request:
    method: str
    path: str
    query: dict[str, list[str]]
    headers: dict[str, str]
    body: bytes = b""
    cookies: dict[str, str] = field(default_factory=dict)
    match: re.Match[str] | None = None

    def param(self, name: str) -> str | None:
        values = self.query.get(name)
        return values[0] if values else None

    def json(self) -> dict[str, Any]:
        """The request body as an object; a usage error if it is anything else."""
        if not self.body:
            return {}
        try:
            document = json.loads(self.body)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise UsageError(f"the request body is not JSON: {exc}") from exc
        if not isinstance(document, dict):
            raise UsageError("the request body must be a JSON object")
        return document

    @property
    def session_token(self) -> str | None:
        return self.cookies.get(SESSION_COOKIE)

    @property
    def wants_json(self) -> bool:
        return self.path.startswith("/api/")


@dataclass
class Response:
    status: int = HTTPStatus.OK
    body: bytes = b""
    content_type: str = "text/plain; charset=utf-8"
    headers: list[tuple[str, str]] = field(default_factory=list)

    @staticmethod
    def json(document: dict[str, Any], status: int = HTTPStatus.OK) -> Response:
        payload = {"schema_version": SCHEMA_VERSION, **document}
        return Response(
            status,
            json.dumps(payload, indent=2).encode(),
            "application/json; charset=utf-8",
            [("Cache-Control", "no-store")],
        )

    @staticmethod
    def error(message: str, exit_code: int) -> Response:
        """The same document `--json` prints when a command fails, with an HTTP status."""
        status = HTTP_FOR_EXIT.get(exit_code, HTTPStatus.INTERNAL_SERVER_ERROR)
        return Response.json({"error": message, "exit_code": exit_code}, status)

    @staticmethod
    def redirect(location: str) -> Response:
        return Response(HTTPStatus.SEE_OTHER, b"", headers=[("Location", location)])

    @staticmethod
    def html(text: str, status: int = HTTPStatus.OK) -> Response:
        return Response(
            status, text.encode(), STATIC_TYPES[".html"], [("Cache-Control", "no-store")]
        )


Handler = Callable[["Dashboard", Request], Response]


def static_file(name: str) -> bytes:
    return resources.files("gpuc.control.web").joinpath("static", name).read_bytes()


def login_page(message: str | None = None) -> str:
    template = static_file("login.html").decode()
    notice = f'<p class="error" role="alert">{html.escape(message)}</p>' if message else ""
    return template.replace("<!--MESSAGE-->", notice)


class Dashboard:
    """The routes, and the one rule they share: nothing without a session.

    `settings` is read per request, so an edited `config.toml` shows up on the
    next refresh rather than after a restart.
    """

    def __init__(
        self, sessions: Sessions, *, settings: Callable[[], Settings] = load_settings
    ) -> None:
        self.sessions = sessions
        self.load_settings = settings
        self.routes: list[tuple[str, re.Pattern[str], Handler, bool]] = [
            ("GET", re.compile(r"^/$"), Dashboard.index, True),
            ("GET", re.compile(r"^/login$"), Dashboard.login_form, False),
            ("POST", re.compile(r"^/login$"), Dashboard.login, False),
            ("POST", re.compile(r"^/logout$"), Dashboard.logout, True),
            ("GET", re.compile(r"^/static/([a-z]+\.(?:js|css))$"), Dashboard.static, False),
            ("GET", re.compile(r"^/favicon\.ico$"), Dashboard.no_icon, False),
            ("GET", re.compile(r"^/api/status$"), Dashboard.api_status, True),
            ("GET", re.compile(r"^/api/hosts$"), Dashboard.api_hosts, True),
            ("GET", re.compile(r"^/api/config$"), Dashboard.api_config, True),
            ("GET", re.compile(r"^/api/version$"), Dashboard.api_version, True),
            ("GET", re.compile(r"^/api/jobs/([^/]+)/logs$"), Dashboard.api_logs, True),
            ("POST", re.compile(r"^/api/jobs/([^/]+)/cancel$"), Dashboard.api_cancel, True),
            ("POST", re.compile(r"^/api/jobs/([^/]+)/reorder$"), Dashboard.api_reorder, True),
            ("POST", re.compile(r"^/api/jobs/([^/]+)/preempt$"), Dashboard.api_preempt, True),
            ("POST", re.compile(r"^/api/jobs/([^/]+)/estimate$"), Dashboard.api_estimate, True),
        ]

    def handle(self, request: Request) -> Response:
        for method, pattern, handler, protected in self.routes:
            match = pattern.match(request.path)
            if match is None:
                continue
            if method != request.method:
                continue
            request.match = match
            if request.method == "POST" and not same_origin(request):
                return Response.error("cross-origin request refused", EXIT_USAGE)
            if protected and not self.sessions.check(request.session_token):
                if request.wants_json:
                    # 401, not the exit-code table's 400: the page keys its
                    # "go and log in again" on the status, not on the words.
                    return Response.json(
                        {"error": "not logged in", "exit_code": EXIT_USAGE},
                        HTTPStatus.UNAUTHORIZED,
                    )
                return Response.redirect("/login")
            try:
                return handler(self, request)
            except Exception as exc:
                code = exit_code_for(exc)
                if code is None:
                    raise
                return Response.error(failure_message(exc), code)
        if any(pattern.match(request.path) for _, pattern, _, _ in self.routes):
            return Response(HTTPStatus.METHOD_NOT_ALLOWED, b"method not allowed\n")
        if request.wants_json:
            return Response.error(f"no such endpoint: {request.path}", EXIT_NOT_FOUND)
        return Response(HTTPStatus.NOT_FOUND, b"not found\n")

    # -- pages -----------------------------------------------------------------

    def index(self, request: Request) -> Response:
        return Response.html(static_file("index.html").decode())

    def static(self, request: Request) -> Response:
        assert request.match is not None
        name = request.match.group(1)
        suffix = name[name.rfind(".") :]
        try:
            body = static_file(name)
        except FileNotFoundError:
            return Response(HTTPStatus.NOT_FOUND, b"not found\n")
        return Response(HTTPStatus.OK, body, STATIC_TYPES[suffix])

    def no_icon(self, request: Request) -> Response:
        """Browsers ask for one unprompted; an empty answer keeps the console clean."""
        return Response(HTTPStatus.NO_CONTENT)

    def login_form(self, request: Request) -> Response:
        if self.sessions.check(request.session_token):
            return Response.redirect("/")
        return Response.html(login_page())

    def login(self, request: Request) -> Response:
        form = parse_qs(request.body.decode("utf-8", "replace"))
        password = form.get("password", [""])[0]
        token = self.sessions.login(password)
        if token is None:
            return Response.html(login_page("wrong password"), HTTPStatus.UNAUTHORIZED)
        response = Response.redirect("/")
        response.headers.append(("Set-Cookie", session_cookie(token)))
        return response

    def logout(self, request: Request) -> Response:
        self.sessions.logout(request.session_token)
        response = Response.redirect("/login")
        response.headers.append(("Set-Cookie", session_cookie("", clear=True)))
        return response

    # -- the API: `--json` documents over HTTP ---------------------------------

    def api_status(self, request: Request) -> Response:
        settings = self.load_settings()
        read = read_registry()
        recent = int_param(request, "recent", status_mod.RECENT_FINISHED)
        since = request.param("since")
        try:
            since_s = status_mod.parse_duration(since) if since else None
        except ValueError as exc:
            raise UsageError(f"since: {exc}") from exc
        document = status_document(
            read, settings, host=request.param("host") or None, recent=recent, since_s=since_s
        )
        document["gathered_at"] = utc_now()
        status = HTTPStatus.SERVICE_UNAVAILABLE if read.unreadable else HTTPStatus.OK
        return Response.json(document, status)

    def api_hosts(self, request: Request) -> Response:
        read = read_registry()
        status = HTTPStatus.SERVICE_UNAVAILABLE if read.unreadable else HTTPStatus.OK
        return Response.json(hosts_document(read), status)

    def api_config(self, request: Request) -> Response:
        return Response.json(config_document(self.load_settings()))

    def api_version(self, request: Request) -> Response:
        read = read_registry()
        status = HTTPStatus.SERVICE_UNAVAILABLE if read.unreadable else HTTPStatus.OK
        return Response.json(version_document(read), status)

    def api_logs(self, request: Request) -> Response:
        job_id = job_id_of(request)
        lines = min(int_param(request, "lines", 200), MAX_LOG_LINES)
        entry, log = read_log(job_id, request.param("host") or None, lines, self.load_settings())
        return Response.json(log.document(job_id, entry.name))

    def api_cancel(self, request: Request) -> Response:
        job_id = job_id_of(request)
        body = request.json()
        return Response.json(cancel_job(job_id, host_of(body), self.load_settings()))

    def api_reorder(self, request: Request) -> Response:
        job_id = job_id_of(request)
        body = request.json()
        priority = body.get("priority")
        if isinstance(priority, bool) or not isinstance(priority, int):
            raise UsageError("reorder needs an integer `priority` (0-99)")
        return Response.json(reorder_job(job_id, priority, host_of(body), self.load_settings()))

    def api_preempt(self, request: Request) -> Response:
        job_id = job_id_of(request)
        body = request.json()
        priority = body.get("priority")
        if priority is not None and (isinstance(priority, bool) or not isinstance(priority, int)):
            raise UsageError("preempt takes an integer `priority` (0-99), or none at all")
        return Response.json(preempt_job(job_id, priority, host_of(body), self.load_settings()))

    def api_estimate(self, request: Request) -> Response:
        job_id = job_id_of(request)
        body = request.json()
        minutes = body.get("minutes")
        if minutes is not None and (
            isinstance(minutes, bool) or not isinstance(minutes, (int, float))
        ):
            raise UsageError("estimate needs a number of `minutes`, or `clear: true`")
        wanted = check_estimate(
            None if minutes is None else float(minutes), clear=bool(body.get("clear"))
        )
        return Response.json(estimate_job(job_id, wanted, host_of(body), self.load_settings()))


def job_id_of(request: Request) -> str:
    assert request.match is not None
    job_id = request.match.group(1)
    if not JOB_ID.match(job_id):
        raise UsageError(f"{job_id!r} is not a job id")
    return job_id


def host_of(body: dict[str, Any]) -> str | None:
    host = body.get("host")
    if host is None or host == "":
        return None
    if not isinstance(host, str):
        raise UsageError("`host` must be a host name")
    return host


def int_param(request: Request, name: str, default: int) -> int:
    raw = request.param(name)
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise UsageError(f"{name} must be an integer, got {raw!r}") from exc
    if value < 0:
        raise UsageError(f"{name} cannot be negative")
    return value


def same_origin(request: Request) -> bool:
    """A POST from another site is refused, whatever cookie it carries.

    The cookie is `SameSite=Strict`, so a browser never sends it cross-site in
    the first place; this is the second lock, for a browser that does not
    honour that, and it costs one header comparison.
    """
    origin = request.headers.get("origin")
    if origin is None:
        return True
    host = request.headers.get("host", "")
    return urlsplit(origin).netloc == host


def session_cookie(token: str, *, clear: bool = False) -> str:
    age = 0 if clear else int(SESSION_TTL_S)
    return f"{SESSION_COOKIE}={token}; Path=/; HttpOnly; SameSite=Strict; Max-Age={age}"


IDLE_TIMEOUT_S = 30
"""How long a kept-alive connection may sit silent before its thread is
released. Without it every idle browser connection pins a thread for ever."""


class BadRequest(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


def make_handler(app: Dashboard) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        timeout = IDLE_TIMEOUT_S

        def do_GET(self) -> None:
            self._serve("GET")

        def do_POST(self) -> None:
            self._serve("POST")

        def _serve(self, method: str) -> None:
            try:
                response = app.handle(self._request(method))
            except BadRequest as exc:
                # The body was not read, so the bytes left on the socket must
                # never be parsed as the next request.
                self.close_connection = True
                response = Response(exc.status, f"{exc}\n".encode())
            except Exception:
                # Every request gets a status line, even for a bug: a dropped
                # connection tells the browser nothing and hides the traceback
                # from the next request's log line.
                traceback.print_exc()
                self.close_connection = True
                response = Response(HTTPStatus.INTERNAL_SERVER_ERROR, b"internal error\n")
            self._write(response)

        def _write(self, response: Response) -> None:
            self.send_response(response.status)
            self.send_header("Content-Type", response.content_type)
            self.send_header("Content-Length", str(len(response.body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            for name, value in response.headers:
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(response.body)

        def _request(self, method: str) -> Request:
            parts = urlsplit(self.path)
            raw_length = self.headers.get("Content-Length") or "0"
            try:
                length = int(raw_length)
            except ValueError:
                raise BadRequest(HTTPStatus.BAD_REQUEST, "bad Content-Length") from None
            if length < 0:
                raise BadRequest(HTTPStatus.BAD_REQUEST, "bad Content-Length")
            if length > MAX_BODY_BYTES:
                raise BadRequest(
                    HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                    f"request body over {MAX_BODY_BYTES} bytes",
                )
            body = self.rfile.read(length) if length else b""
            cookies = SimpleCookie(self.headers.get("Cookie", ""))
            return Request(
                method=method,
                path=parts.path,
                query=parse_qs(parts.query),
                headers={key.lower(): value for key, value in self.headers.items()},
                body=body,
                cookies={name: morsel.value for name, morsel in cookies.items()},
            )

        def log_message(self, format: str, *args: Any) -> None:
            # One line per request on stderr, like the default, minus the
            # date every line repeats.
            print(f"{self.address_string()} {format % args}", file=sys.stderr)

    return Handler


def make_server(
    bind: str = DEFAULT_BIND, port: int = DEFAULT_PORT, app: Dashboard | None = None
) -> ThreadingHTTPServer:
    """A server ready for `serve_forever()`; refuses to exist without a password."""
    if app is None:
        app = Dashboard(Sessions(read_password_hash()))
    server = ThreadingHTTPServer((bind, port), make_handler(app))
    server.daemon_threads = True
    return server


class ServerThread:
    """A server on a background thread, for tests and for embedding."""

    def __init__(self, app: Dashboard, bind: str = DEFAULT_BIND) -> None:
        self.server = make_server(bind, 0, app)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        host, port = self.server.server_address[:2]
        return f"http://{host}:{port}"

    def __enter__(self) -> ServerThread:
        self.thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.server.shutdown()
        self.server.server_close()
