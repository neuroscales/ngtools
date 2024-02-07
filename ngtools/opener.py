import fsspec
# from contextlib import contextmanager
from os import PathLike
from pathlib import Path
from indexed_gzip import IndexedGzipFile
from bz2 import BZ2File


def stringify_path(filename):
    """Ensure that the input is a str or a file-like object"""
    if isinstance(filename, PathLike):
        return filename.__fspath__()
    if isinstance(filename, Path):
        return str(filename)
    return filename


REMOTE_PROTOCOLS = ["http", "https", "s3"]


def remote_protocols():
    """List of accepeted remote protocols"""
    return tuple(x + '://' for x in REMOTE_PROTOCOLS)


class open:

    def __init__(self, fileobj, mode='rb', compression=None):
        """
        Open a file from a path or url or an opened file object

        Parameters
        ----------
        fileobj : path or url or file-object
            Input file
        mode : str
            Opening mode
        compression : {'zip', 'bz2', 'gzip', 'lzma', 'xz', 'infer'}
            Compression mode.
            If 'infer', guess from magic number (if mode 'r') or
            filename (if mode 'w').

        Returns
        -------
        fileobj
            Opened file
        """
        self.fileobj = fileobj
        self.mode = mode
        self.compression = compression
        self.fileobjs = []
        self._is_inside = False

    def __enter__(self):
        if not self._is_inside:
            self._is_inside = True
            return self._open()
        return self.fileobjs[-1] if self.fileobjs else self.fileobj

    def __exit__(self, type=None, value=None, traceback=None):
        for fileobj in reversed(self.fileobjs):
            fileobj.close()
        self.fileobjs = []
        self._is_inside = False

    def __del__(self):
        self.__exit__()

    def open(self):
        return self.__enter__()

    def close(self):
        return self.__exit__()

    def read(self, *a, **k):
        return self.__enter__().read(*a, **k)

    def readline(self, *a, **k):
        return self.__enter__().readline(*a, **k)

    def readinto(self, *a, **k):
        return self.__enter__().readinto(*a, **k)

    def write(self, *a, **k):
        return self.__enter__().write(*a, **k)

    def writeline(self, *a, **k):
        return self.__enter__().writeline(*a, **k)

    def seek(self, *a, **k):
        return self.__enter__().seek(*a, **k)

    def tell(self, *a, **k):
        return self.__enter__().tell(*a, **k)

    def _open(self):
        fileobj = self.fileobj
        if isinstance(fileobj, str):
            fileobj = self.fsspec_open(fileobj)
        if self.compression == 'infer' and 'r' in self.mode:
            fileobj = self.infer(fileobj)
        return fileobj

    def infer(self, fileobj):
        pos = fileobj.tell()
        magic = fileobj.read(2)
        fileobj.seek(pos)
        if magic == b'\x1f\x8b':
            fileobj = IndexedGzipFile(fileobj)
            self.fileobjs.append(fileobj)
        if magic == b'BZh':
            fileobj = BZ2File(fileobj)
            self.fileobjs.append(fileobj)
        return fileobj

    def fsspec_open(self, fileobj):
        fileobj = stringify_path(fileobj)
        opt = dict()
        if not (self.compression == 'infer' and 'r' in self.mode):
            opt['compression'] = self.compression
        # if fileobj.startswith(remote_protocols()):
        #     opt['block_size'] = 0
        fileobj = fsspec.open(fileobj, self.mode, **opt)
        self.fileobjs.append(fileobj)
        fileobj = fileobj.open()
        self.fileobjs.append(fileobj)
        return fileobj
