"""
Local fileserver to access local files through HTTP.

This module implements local file servers, that can be used to server
local files to a local neuroglancer instance.

Classes
-------
LocalFileServer
    A fileserver that serves local files
LocalFileServerInBackground
    A fileserver that runs in a background process
"""
# stdlib
import _thread
import atexit
import logging
import os.path as op
import re
import wsgiref.util
from dataclasses import dataclass
from typing import Callable, Literal
from wsgiref.headers import Headers
from wsgiref.simple_server import WSGIRequestHandler, make_server

# internals
from ngtools.utils import find_available_port

LOG = logging.getLogger(__name__)


class LocalFileServer:
    """
    A fileserver that serves local files.

    All paths should be absolute!
    """

    def __init__(
        self,
        port: int = 0,
        ip: str = "",
        app: Callable | None = None,
        start: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        port : int
            Port number to use
        ip : str
            IP address
        app : callable
            Application
        start : bool
            Start straight away.
        """
        port, ip = find_available_port(port, ip)
        self.port = port
        self.ip = ip

        if isinstance(app, (list, tuple)):
            app = Application(app)
        elif not app:
            app = Application()

        self.server = make_server(self.ip, self.port, app,
                                  handler_class=_NoLogRequestHandler)
        self.thread = _Thread(target=self.server.serve_forever)
        if start:
            self.start()

    @property
    def app(self) -> Callable:  # noqa: D102
        return self.server.get_app()

    @app.setter
    def app(self, value: Callable) -> Callable:
        return self.server.set_app(value)

    def get_url(self) -> str | None:
        """URL of the fileserver."""
        return f'http://{self.ip}:{self.port}/'

    def start(self) -> None:
        """Start server and server forever."""
        if not self.thread.is_alive():
            self.thread.start()
            atexit.register(self.stop)

    def is_running(self) -> bool:
        """Check if server is running."""
        if not getattr(self, "thread", None):
            return False
        return self.thread.is_alive()

    def stop(self) -> None:
        """Shutdown server."""
        if self.is_running():
            self.server.shutdown()
            self.server.server_close()
        self.thread.join()
        atexit.unregister(self.stop)

    def __del__(self) -> None:
        self.stop()


class _Thread:
    """
    My own implementation of threading.Thread.

    threading.Thread sometimes hangs when exiting IPython, whereas
    this implementation does not. Let's hope I am not doing anything
    stupid.
    """

    def __init__(
        self, target: Callable, args: tuple = (), kwargs: dict = {}
    ) -> None:
        self.lock: _thread.LockType = _thread.allocate_lock()
        self.target: Callable = target
        self.args: tuple = args
        self.kwargs: dict = kwargs
        self.id: int | None = None

    def start(self) -> None:

        if self.is_alive():
            return

        def run(lock: _thread.LockType) -> object | type:
            try:
                lock.acquire()
                return self.target(*self.args, **self.kwargs)
            finally:
                lock.release()

        self.id = _thread.start_new_thread(run, (self.lock,))

    def is_alive(self) -> bool:
        return self.lock.locked()

    def join(self) -> None:
        self.lock.acquire()
        self.lock.release()


class _NoLogRequestHandler(WSGIRequestHandler):

    def log_message(self, *args, **kwargs) -> None:
        pass


class Application:
    """An application."""

    def __init__(self, handlers: list | None = None) -> None:
        self.handlers = list(handlers or [])

    def __call__(
        self, environ: dict, start_response: Callable
    ) -> list[bytes | str | dict]:
        """Serve by deferring to appropriate handlers."""
        LOG.debug(f"Application: {environ['PATH_INFO']}")
        environ = Environ.from_environ(environ)
        method = environ.method.lower()

        for pattern, kls, *kw in self.handlers:
            kw = kw[0] if kw else {}
            match = re.fullmatch(pattern, environ.path)
            if not match:
                continue
            handler: Handler = kls(self, environ, **kw)
            try:
                handler.prepare()
                getattr(handler, method)(*match.groups())
                handler.on_finish()
            except Exception as e:
                handler = ServerErrorHandler(self, e)
            return self.make_response(start_response, handler)

        handler = FileNotFoundHandler(self, environ.request_uri())
        return self.make_response(start_response, handler)

    @classmethod
    def make_response(
        cls, start_response: Callable, handler: "Handler"
    ) -> list:
        """Make WSGI response."""
        handler.headers["Access-Control-Allow-Origin"] = "*"
        handler.headers["Access-Control-Allow-Methods"] = "GET, HEAD, OPTIONS"
        handler.headers["Access-Control-Allow-Headers"] = "Content-Type"
        start_response(str(handler.status), list(handler.headers.items()))
        return [handler.body] if handler.body is not None else []


class Status(str):
    """Well formatted HTTP status."""

    STATUS_MAP = {}

    def __new__(self, *args, **kwargs) -> "Status":  # noqa: D102
        object = kwargs.pop("object", args[0] if args else None)
        try:
            object = int(object)
        except ValueError:
            if isinstance(object, str):
                for value in self.STATUS_MAP.values():
                    if object in (value, value[4:]):
                        return value
            raise ValueError("Unknown code:", object)
        else:
            return self.STATUS_MAP[object]


STATUSES = {
    # 1xx informational response
    100: "100 Continue",
    101: "101 Switching Protocols",
    102: "102 Processing",
    103: "103 Early Hints",
    # 2xx success
    200: "200 OK",
    201: "201 Created",
    202: "202 Accepted",
    203: "203 Non-Authoritative Information",
    204: "204 No Content",
    205: "205 Reset Content",
    206: "206 Partial Content",
    207: "207 Multi-Status",
    208: "208 Already Reported",
    226: "226 IM Used",
    # 3xx redirection
    300: "300 Multiple Choices",
    301: "301 Moved Permanently",
    302: "302 Found",
    303: "303 See Other",
    304: "304 Not Modified",
    305: "305 Use Proxy",
    306: "306 Switch Proxy",
    307: "307 Temporary Redirect",
    308: "308 Permanent Redirect",
    # 4xx client errors
    400: "400 Bad Request",
    401: "401 Unauthorized",
    402: "402 Payment Required",
    403: "403 Forbidden",
    404: "404 Not Found",
    405: "405 Method Not Allowed",
    406: "406 Not Acceptable",
    407: "407 Proxy Authentication Required",
    408: "408 Request Timeout",
    409: "409 Conflict",
    410: "410 Gone",
    411: "411 Length Required",
    412: "412 Precondition Failed",
    413: "413 Payload Too Large",
    414: "414 URI Too Long",
    415: "415 Unsupported Media Type",
    416: "416 Range Not Satisfiable",
    417: "417 Expectation Failed",
    418: "418 I'm a teapot",
    421: "421 Misdirected Request",
    422: "422 Unprocessable Content",
    423: "423 Locked",
    424: "424 Failed Dependency",
    425: "425 Too Early",
    426: "426 Upgrade Required",
    428: "428 Precondition Required",
    429: "429 Too Many Requests",
    431: "431 Request Header Fields Too Large",
    451: "451 Unavailable For Legal Reasons",
    # 5xx server errors
    500: "500 Internal Server Error",
    501: "501 Not Implemented",
    502: "502 Bad Gateway",
    503: "503 Service Unavailable",
    504: "504 Gateway Timeout",
    505: "505 HTTP Version Not Supported",
    506: "506 Variant Also Negotiates",
    507: "507 Insufficient Storage",
    508: "508 Loop Detected",
    510: "510 Not Extended",
    511: "511 Network Authentication Required",
}


for code, name in STATUSES.items():
    status = str.__new__(Status, name)
    status.code = code
    Status.STATUS_MAP[code] = status


@dataclass
class Environ:
    """WSGI environ, as a dataclass."""

    _environ: dict | None = None
    """Saved environ dict"""

    method: Literal[
        "GET", "HEAD", "POST", "DELETE", "PATCH", "PUT", "OPTIONS"
    ] | None = None
    """
    The HTTP request method, such as GET or POST.
    This cannot ever be an empty string, and so is always required.
    """

    app: str | None = None
    """
    The initial portion of the request URL's "path" that corresponds to
    the application object, so that the application knows its virtual
    "location". This may be an empty string, if the application corresponds
    to the root" of the server.
    """

    path: str | None = None
    """
    The remainder of the request URL's "path", designating the virtual
    “location” of the request's target within the application.
    This may be an empty string, if the request URL targets the application
    root and does not have a trailing slash.
    """

    query: str | None = None
    """
    The portion of the request URL that follows the "?", if any.
    May be empty or absent.
    """

    content_type: str | None = None
    """
    The contents of any Content-Type fields in the HTTP request.
    May be empty or absent.
    """

    content_length: int | None = None
    """
    The contents of any Content-Length fields in the HTTP request.
    May be empty or absent.
    """

    server_name: str | None = None
    server_port: str | None = None
    """
    When combined with SCRIPT_NAME and PATH_INFO, these variables can be
    used to complete the URL. Note, however, that HTTP_HOST, if present,
    should be used in preference to SERVER_NAME for reconstructing the
    request URL. See the URL Reconstruction section below for more detail.
    SERVER_NAME and SERVER_PORT can never be empty strings, and so are
    always required.
    """

    server_protocol: str | None = None
    """
    The version of the protocol the client used to send the request.
    Typically this will be something like "HTTP/1.0" or "HTTP/1.1" and
    may be used by the application to determine how to treat any HTTP
    request headers.
    (This variable should probably be called REQUEST_PROTOCOL, since it
    denotes the protocol used in the request, and is not necessarily the
    protocol that will be used in the server's response. However, for
    compatibility with CGI we have to keep the existing name.)
    """

    headers: Headers = Headers()
    """
    Variables corresponding to the client-supplied HTTP request headers
    (i.e., variables whose names begin with HTTP_).
    The presence or absence of these variables should correspond with the
    presence or absence of the appropriate HTTP header in the request.
    """

    @classmethod
    def from_environ(cls, environ: dict) -> "Environ":
        """Convert environ dictionary to `Environ` object."""
        key_map = {
            "method": "REQUEST_METHOD",
            "app": "SCRIPT_NAME",
            "path": "PATH_INFO",
            "query": "QUERY_STRING",
            "content_type": "CONTENT_TYPE",
            "content_length": "CONTENT_LENGTH",
            "server_name": "SERVER_NAME",
            "server_port": "SERVER_PORT",
            "server_protocol": "SERVER_PROTOCOL",
        }

        def to_http_header(key: str) -> str:
            return "-".join(key[5:].split("_")).capitalize()

        headers = {
            to_http_header(key): value for key, value in environ.items()
            if key.startswith("HTTP_")
        }
        headers = Headers(list(headers.items()))

        public = {
            key: environ[mapped_key]
            for key, mapped_key in key_map.items()
            if mapped_key in environ
        }
        return Environ(**public, headers=headers, _environ=environ)

    def to_environ(self) -> dict:
        """Convert `Environ` object to environ dictionary."""
        environ = self._environ

        key_map = {
            "method": "REQUEST_METHOD",
            "app": "SCRIPT_NAME",
            "path": "PATH_INFO",
            "query": "QUERY_STRING",
            "content_type": "CONTENT_TYPE",
            "content_length": "CONTENT_LENGTH",
            "server_name": "SERVER_NAME",
            "server_port": "SERVER_PORT",
            "server_protocol": "SERVER_PROTOCOL",
        }

        def from_http_header(key: str) -> str:
            return "_".join(key[5:].split("-")).upper()

        headers = {
            from_http_header(key): value
            for key, value in self.headers.items()
        }
        environ.update({
            mapped_key: getattr(self, key)
            for key, mapped_key in key_map.items()
            if getattr(self, key) is not None
        })
        environ.update(headers)
        return environ

    def request_uri(self) -> str:
        """
        Return the full request URI, optionally including the query string,
        using the algorithm found in the “URL Reconstruction” section of
        PEP 3333. If include_query is false, the query string is not included
        in the resulting URI.
        """
        return wsgiref.util.request_uri(self.to_environ())

    def shift_path(self) -> str:
        """
        Shift a single name from PATH_INFO to SCRIPT_NAME and return the name.
        The environ dictionary is modified in-place; use a copy if you need
        to keep the original PATH_INFO or SCRIPT_NAME intact.

        If there are no remaining path segments in PATH_INFO, None is returned.
        """
        environ = self.to_environ()
        name = wsgiref.util.shift_path_info()
        self.path = environ["PATH_INFO"]
        self.app = environ["SCRIPT_NAME"]
        self._environ = environ
        return name


def _method_not_allowed(name: str) -> Callable:

    def mock(self: "Handler", *args, **kwargs) -> None:
        self.send_error(405, name)

    return mock


class Handler:
    """Class that handles requests."""

    def __init__(
        self, app: "Application", environ: Environ = None, **kwargs
    ) -> None:
        self.app = app
        self.environ = environ
        self.headers: Headers = Headers()
        self._status: Status = Status(500)  # so that we raise if forgot to set
        self.body: bytes | str | dict | None = None
        self._finished: bool = False
        self.initialize(**kwargs)

    @property
    def status(self) -> Status:
        """Status, as a well formatted string."""
        return self._status

    @status.setter
    def status(self, value: str | int | None) -> Status:
        if value is None:
            self._status = None
        else:
            self._status = Status(value)

    def initialize(self, **kwargs) -> None:
        """Initialize handler."""
        pass

    def prepare(self) -> None:
        """Peform steps before the main request."""
        pass

    def on_finish(self) -> None:
        """Peform steps after the main request."""
        pass

    def send_error(self, status: int | str | Status, msg: str | None) -> None:
        """Make the returned request an error request."""
        self.status = status
        self.headers["Content-type"] = "text/html"
        if msg is not None:
            self.body = (
                "<html><body>%s: %s</body></html>" % (status, str(msg))
            ).encode("utf-8")
        else:
            self.body = (
                "<html><body>%s</body></html>" % status
            ).encode("utf-8")

    get = _method_not_allowed("get")
    head = _method_not_allowed("head")
    post = _method_not_allowed("post")
    delete = _method_not_allowed("delete")
    patch = _method_not_allowed("patch")
    put = _method_not_allowed("put")
    options = _method_not_allowed("options")


class StaticFileHandler(Handler):
    """Serves static files."""

    def _request(self, method: str, path: str) -> None:
        range = self.environ.headers.get("Range", "")
        if range.startswith('bytes='):
            range = range.split('=')[1].strip().split('-')
        else:
            range = None

        if not op.isfile(path):
            return self.send_error(404, path)

        length = lengthrange = op.getsize(path)

        if range:
            start, end = range
            if not start:
                start = max(0, length - int(end))
                end = None
            start, end = int(start or 0), int(end or length - 1)
            end = min(end, length - 1)
            lengthrange = end - start + 1

        header = {
            "Content-type": "application/octet-stream",
            "Content-Length": str(lengthrange),
            "Accept-Ranges": "bytes",
        }

        if method.lower() == "get":
            try:
                with open(path, 'rb') as f:
                    if range:
                        f.seek(start)
                        self.body = f.read(lengthrange)
                        header["Content-Range"] = f"{start}-{end}/{length}"
                    else:
                        self.body = f.read()
            except Exception as e:
                return self.send_error(400, e)

        self.status = Status(206 if range else 200)

        for key, value in header.items():
            self.headers.add_header(key, value)

    def head(self, path: str) -> None:  # noqa: D102
        return self._request("head", path)

    def get(self, path: str) -> None:  # noqa: D102
        return self._request("get", path)

    def options(self, path: str) -> None:  # noqa: D102
        self.status = 200


class ErrorHandler(Handler):
    """Mock handler for HTTP errors."""

    def __init__(
        self, app: "Application", code: int | str, msg: str | None = None
    ) -> None:
        super().__init__(app)
        self.send_error(code, msg)


class MethodNotAllowedHandler(ErrorHandler):
    """Mock handler for error 405."""

    def __init__(self, app: "Application", msg: str | None = None) -> None:
        super().__init__(app, 405, msg)


class ServerErrorHandler(ErrorHandler):
    """Mock handler for error 500."""

    def __init__(self, app: "Application", msg: str | None = None) -> None:
        super().__init__(app, 500, msg)


class FileNotFoundHandler(ErrorHandler):
    """Mock handler for error 405."""

    def __init__(self, app: "Application", msg: str | None = None) -> None:
        super().__init__(app, 404, msg)
