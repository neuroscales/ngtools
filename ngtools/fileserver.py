import os
from multiprocessing import Process
from wsgiref.simple_server import make_server
from wsgiref.simple_server import WSGIRequestHandler


class NoLoggingWSGIRequestHandler(WSGIRequestHandler):
    def log_message(self, format, *args):
        pass


class LocalFileServer:
    """
    A fileserver that serves local files
    """

    def __init__(self, cwd=None, port=9123, ip='127.0.01', interrupt=True):
        """
        Parameters
        ----------
        cwd : str or Path
            Current working directory.
            All relative paths will be relative to this directory.
        port : int
            Port number to use
        ip : str
            IP address
        interrupt : bool
            Whether we can keyboard-interrupt the server.
            If instanrtiated inside a background process, useful to set
            to False so that the exception is handled in the main thread.
        """
        self.cwd = cwd or os.getcwd()
        self.port = port
        self.ip = ip
        self.interrupt = interrupt
        self.server = make_server(self.ip, self.port, self.serve,
                                  handler_class=NoLoggingWSGIRequestHandler)

    def serve_forever(self):
        try:
            self.server.serve_forever()
        except KeyboardInterrupt as e:
            if self.interrupt:
                raise e

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

    def serve(self, environ, start_response):

        path_info = environ["PATH_INFO"]
        # There is always a leading /, this means that relative paths
        # have the form
        #   /relative/path/from/cwd/file.ext
        # while absolute paths have the form
        #   //absolute/path/from/systemroot/file.ext
        # We remove the leading /, and prepend cwd if there is no more /
        if path_info.startswith('/'):
            path_info = path_info[1:]
        if path_info.startswith('root://'):
            # absolute path, keep one leading /
            path_info = path_info[6:]
        else:
            # relative path, prepend working dir
            path_info = os.path.join(self.cwd, path_info)
        if path_info.endswith(('.zarr', '.zarr/')):
            path_info = os.path.join(path_info, '.zgroup')
        if not os.path.isfile(path_info):
            return self.file_not_found(path_info, start_response)
        with open(path_info, 'rb') as f:
            data = f.read()
        start_response(
            "200 OK",
            [
                ("Content-type", "application/octet-stream"),
                ("Content-Length", str(len(data))),
                ("Access-Control-Allow-Origin", "*"),
            ],
        )
        return [data]


class LocalFileServerInBackground:
    """
    A fileserver that runs in a background process
    """

    def __init__(self, cwd=None, port=9123, ip='127.0.01'):
        """
        Parameters
        ----------
        cwd : str or Path
            Current working directory.
            All relative paths will be relative to this directory.
        port : int
            Port number to use
        ip : str
            IP address
        """
        self.cwd = cwd or os.getcwd()
        self.port = port
        self.ip = ip
        self.process = None

    @classmethod
    def _start_and_server_forever(cls, cwd, port, ip):
        server = LocalFileServer(cwd, port, ip, interrupt=False)
        server.serve_forever()

    def start_and_serve_forever(self):
        if not self.process:
            self.process = Process(target=self._start_and_server_forever,
                                   args=(self.cwd, self.port, self.ip))
        if not self.process.is_alive():
            self.process.start()

    def stop(self):
        if self.process and self.process.is_alive():
            self.process.terminate()

    def __del__(self):
        self.stop()
