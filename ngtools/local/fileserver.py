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
import os
import socket
import sys
from multiprocessing import Process
from threading import Thread
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


class LocalFileServer:
    """
    A fileserver that serves local files.

    All paths should be absolute!
    """

    def __init__(
        self, port: int = 0, ip: str = "", interrupt: bool = True
    ) -> None:
        """
        Parameters
        ----------
        port : int
            Port number to use
        ip : str
            IP address
        interrupt : bool
            Whether we can keyboard-interrupt the server.
            If instantiated inside a background process, useful to set
            to False so that the exception is handled in the main thread.
        """
        port, ip = find_available_port(port, ip)
        self.port = port
        self.ip = ip
        self.thread = None

        if interrupt is True:
            interrupt = KeyboardInterrupt
        if not interrupt:
            interrupt = tuple()
        if not isinstance(interrupt, (list, tuple)):
            interrupt = (interrupt,)
        interrupt = tuple(interrupt)
        self.interrupt = interrupt

        self.server = make_server(self.ip, self.port, self._serve,
                                  handler_class=_NoLoggingWSGIRequestHandler)

    def _serve_forever(self) -> None:
        """Run the server forever."""
        while True:
            try:
                self.server.serve_forever()
            except self.interrupt as e:
                raise e
            finally:
                continue

    def start(self) -> None:
        """Start server and server forever."""
        if not self.thread:
            self.thread = Thread(target=self._serve_forever)
            self.thread.start()
        elif not self.thread.is_alive():
            self.thread = None
            self.start()

    def stop(self) -> None:
        """Shutdown server."""
        self.server.shutdown()
        self.server.server_close()

    def __del__(self) -> None:
        try:
            self.stop()
        except Exception:
            pass

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
            path_info = os.path.join(path_info, '.zgroup')

        if not os.path.isfile(path_info):
            return self._file_not_found(path_info, start_response)

        length = lengthrange = os.path.getsize(path_info)

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
