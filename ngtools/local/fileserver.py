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
import _thread
import atexit
import os.path as op
import socket
import sys
from wsgiref.simple_server import WSGIRequestHandler, make_server


class _NoLoggingWSGIRequestHandler(WSGIRequestHandler):
    """Overloaded handler that does not log anything."""

    def log_message(self, format, *args):  # noqa: ANN001, ANN002, ANN202
        pass


def find_available_port(port: int = 0, ip: str = "") -> tuple[int, str]:
    """Return an available port and the local IP."""
    try:
        s = socket.socket()
        s.bind((ip, port))
        ip = s.getsockname()[0]
        port = s.getsockname()[1]
        s.close()
    except OSError:
        port0 = port
        s = socket.socket()
        s.bind((ip, 0))
        ip = s.getsockname()[0]
        port = s.getsockname()[1]
        s.close()
        print(f'Port {port0} already in use. Use port {port} instead.',
              file=sys.stderr)
    return port, ip


class _Thread:
    """
    My own implementation of threading.Thread.

    threading.Thread sometimes hangs when exiting IPython, whereas
    this implementation does not. Let's hope I am not doing anything
    stupid.
    """

    def __init__(
        self, target: callable, args: tuple = (), kwargs: dict = {}
    ) -> None:
        self.lock: _thread.LockType = _thread.allocate_lock()
        self.target: callable = target
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


class LocalFileServer:
    """
    A fileserver that serves local files.

    All paths should be absolute!
    """

    def __init__(self, port: int = 0, ip: str = "") -> None:
        """
        Parameters
        ----------
        port : int
            Port number to use
        ip : str
            IP address
        """
        port, ip = find_available_port(port, ip)
        self.port = port
        self.ip = ip
        self.thread = None

        self.server = make_server(self.ip, self.port, self._serve,
                                  handler_class=_NoLoggingWSGIRequestHandler)
        self.thread = _Thread(target=self.server.serve_forever)

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
        return self.thread.is_alive()

    def stop(self) -> None:
        """Shutdown server."""
        if self.is_running():
            self.server.shutdown()
            self.server.server_close()
        self.thread.join()
        atexit.unregister(self.stop)

    @staticmethod
    def _file_not_found(dest: str, start_response: callable) -> list:
        """Response when file is not found."""
        start_response(
            "404 Not found",
            [
                ("Content-type", "text/html"),
                ("Access-Control-Allow-Origin", "*"),
            ],
        )
        return [
            ("<html><body>%s not found</body></html>" % dest).encode("utf-8")
        ]

    @staticmethod
    def _method_not_allowed(method: str, start_response: callable) -> list:
        """Response when method is not allowed."""
        start_response(
            "405 Method Not Allowed",
            [
                ("Content-type", "text/html"),
                ("Access-Control-Allow-Origin", "*"),
            ],
        )
        return [
            (
                "<html><body>%s not allowed</body></html>" % method
             ).encode("utf-8")
        ]

    FORMATS = {
        'tracts': 'tracts',
        'trk': 'tracts',
        'tck': 'tracts',
    }

    def _serve(self, environ: dict, start_response: callable) -> list:
        """Serve function passed to the server."""
        path_info = environ['PATH_INFO']

        method = environ['REQUEST_METHOD']
        if method not in ('GET', 'HEAD'):
            return self._method_not_allowed(method, start_response)

        range = environ.get('HTTP_RANGE', '')
        if range.startswith('bytes='):
            range = range.split('=')[1].strip().split('-')
        else:
            range = None

        # TODO There may be a leading protocol indicating file format

        if path_info.endswith(('.zarr', '.zarr/')):
            for name in (".zgroup", ".zarray", "zarr.json"):
                if op.exists(op.join(path_info, name)):
                    path_info = op.join(path_info, name)
                    break

        if not op.isfile(path_info):
            return self._file_not_found(path_info, start_response)

        length = lengthrange = op.getsize(path_info)

        if range:
            start, end = range
            if not start:
                start = max(0, length - int(end))
                end = None
            start, end = int(start or 0), int(end or length - 1)
            end = min(end, length - 1)
            lengthrange = end - start + 1

        header = [
            ("Content-type", "application/octet-stream"),
            ("Content-Length", str(lengthrange)),
            ("Access-Control-Allow-Origin", "*"),
            ("Accept-Ranges", "bytes"),
        ]

        body = []
        if method == 'GET':
            with open(path_info, 'rb') as f:
                if range:
                    f.seek(start)
                    body = [f.read(lengthrange)]
                    header.append(
                        ("Content-Range", f"{start}-{end}/{length}")
                    )
                else:
                    body = [f.read()]

        start_response("200 OK", header)
        return body
