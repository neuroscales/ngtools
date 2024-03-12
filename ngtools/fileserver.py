import os
import sys
import socket
from multiprocessing import Process
from wsgiref.simple_server import make_server
from wsgiref.simple_server import WSGIRequestHandler


class NoLoggingWSGIRequestHandler(WSGIRequestHandler):
    def log_message(self, format, *args):
        pass


class LocalFileServer:
    """
    A fileserver that serves local files.

    All paths should be absolute!
    """

    def __init__(self, port=0, ip='', interrupt=True):
        """
        Parameters
        ----------
        port : int
            Port number to use
        ip : str
            IP address
        interrupt : bool
            Whether we can keyboard-interrupt the server.
            If instanrtiated inside a background process, useful to set
            to False so that the exception is handled in the main thread.
        """

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

        self.port = port
        self.ip = ip

        if interrupt is True:
            interrupt = KeyboardInterrupt
        if not interrupt:
            interrupt = tuple()
        if not isinstance(interrupt, (list, tuple)):
            interrupt = (interrupt,)
        interrupt = tuple(interrupt)
        self.interrupt = interrupt

        self.server = make_server(self.ip, self.port, self.serve,
                                  handler_class=NoLoggingWSGIRequestHandler)

    def serve_forever(self):
        while True:
            try:
                self.server.serve_forever()
            except self.interrupt as e:
                raise e
            finally:
                continue

    @staticmethod
    def file_not_found(dest, start_response):
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
    def method_not_allowed(method, start_response):
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

    def serve(self, environ, start_response):
        path_info = environ['PATH_INFO']

        method = environ['REQUEST_METHOD']
        if method not in ('GET', 'HEAD'):
            return self.method_not_allowed(method, start_response)

        range = environ.get('HTTP_RANGE', '')
        if range.startswith('bytes='):
            range = range.split('=')[1].strip().split('-')
        else:
            range = None

        # TODO There may be a leading protocol indicating file format

        if path_info.endswith(('.zarr', '.zarr/')):
            path_info = os.path.join(path_info, '.zgroup')

        if not os.path.isfile(path_info):
            return self.file_not_found(path_info, start_response)

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


class LocalFileServerInBackground:
    """
    A fileserver that runs in a background process
    """

    def __init__(self, port=0, ip='', interrupt=False):
        """
        Parameters
        ----------
        port : int
            Port number to use
        ip : str
            IP address
        interrupt : bool or [tuple of] type
            Exceptions that do interrupt the fileserver.
            If any other exception happens, the fileserver is restarted.
            If True, interupt on KeyboardInterrupt.
        """
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

        self.port = port
        self.ip = ip
        self.process = None
        self.interrupt = interrupt

    @classmethod
    def _start_and_server_forever(cls, port, ip, interupt):
        server = LocalFileServer(port, ip, interrupt=interupt)
        server.serve_forever()

    def start_and_serve_forever(self):
        if not self.process:
            self.process = Process(
                target=self._start_and_server_forever,
                args=(self.port, self.ip, self.interrupt)
            )
        if not self.process.is_alive():
            self.process.start()

    def stop(self):
        if self.process and self.process.is_alive():
            self.process.terminate()

    def __del__(self):
        self.stop()
