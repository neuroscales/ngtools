"""General-purpose byte stream opener."""
# stdlib
import json
from bz2 import BZ2File
from io import BufferedReader, BytesIO
from os import PathLike, environ
from pathlib import Path
from types import TracebackType
from typing import IO

# externals
import fsspec
import numpy as np
import requests
from indexed_gzip import IndexedGzipFile
from typing_extensions import Buffer

# internals
from ngtools.protocols import FORMATS, LAYERS, PROTOCOLS

LINCBRAIN_API: str = "https://api.lincbrain.org/api"


def stringify_path(filename: IO | PathLike | str) -> IO | str:
    """Ensure that the input is a str or a file-like object."""
    if isinstance(filename, PathLike):
        return filename.__fspath__()
    if isinstance(filename, Path):
        return str(filename)
    return filename


_LINC_AUTH_CACHE = {}


def linc_auth_opt(token: str | None = None) -> dict:
    """
    Create fsspec authentication options to access lincbrain data.

    Parameters
    ----------
    token : str
        Your LINCBRAIN_API_KEY

    Returns
    -------
    opt : dict
        options to pass to `fsspec`'s `HTTPFileSystem`.
    """
    token = token or environ.get('LINCBRAIN_API_KEY', None)
    if not token:
        return {}
    if token in _LINC_AUTH_CACHE:
        return _LINC_AUTH_CACHE[token]
    # Check that credential is correct
    session = requests.Session()
    session.get(
        f"{LINCBRAIN_API}/auth/token",
        headers={"Authorization": f"token {token}"}
    )
    # Get cookies
    response = session.get(
        f"{LINCBRAIN_API}/permissions/s3/",
        headers={"Authorization": f"token {token}"}
    )
    cookies = response.cookies.get_dict()
    # Pass cookies to FileSystem
    opt = dict(client_kwargs={'cookies': cookies})
    _LINC_AUTH_CACHE[token] = opt
    return opt


def remote_protocols() -> tuple[str]:
    """List of accepeted remote protocols."""
    return tuple(x + '://' for x in PROTOCOLS if x != "file")


def parse_protocols(url: str | PathLike) -> tuple[str, str, str, str]:
    """
    Parse protocols out of a url.

    Parameters
    ----------
    url : str | PathLike
        URL with or without protocols

    Returns
    -------
    layer_type : str | None
        Layer type (e.g. `"volume"`, `"labels"`, etc.)
    format : str | None
        Format (e.g., `"zarr"`, `"nifti"`, etc.)
    stream : str | None
        Stream/communication protocol (e.g., `"file"`, `"http"`, etc.)
    url : str
        URL with stream protocol but without format or layer type.
    """
    url = str(url)
    protocol = layert_type = format = None
    *parts, path = str(url).split("://")
    for part in parts:
        if part in PROTOCOLS:
            if protocol is not None:
                raise ValueError("Too many streaming procols:", protocol, part)
            protocol = part
        elif part in LAYERS:
            if layert_type is not None:
                raise ValueError("Too many layer procols:", layert_type, part)
            layert_type = part
        elif part in FORMATS:
            if format is not None:
                raise ValueError("Too many format procols:", format, part)
            format = part
        else:
            raise ValueError("Unknown protocol:", part)
    protocol = protocol or "file"
    if protocol != "file":
        path = protocol + "://" + path
    return layert_type, format, protocol, path


def filesystem(
    protocol: str | PathLike | fsspec.AbstractFileSystem,
    **opt
) -> fsspec.AbstractFileSystem:
    """Return the filesystem corresponding to a protocol or URI."""
    if isinstance(protocol, fsspec.AbstractFileSystem):
        return protocol
    protocol = stringify_path(protocol)
    protocol = parse_protocols(protocol)[-2]
    try:
        opt.update(linc_auth_opt())
    except Exception:
        pass
    return fsspec.filesystem(protocol, **opt)


def exists(uri: str | PathLike, **opt) -> bool:
    """Check that the file or directory pointed by a URI exists."""
    uri = parse_protocols(uri)[-1]
    fs = filesystem(uri, **opt)
    return fs.exists(uri)


def read_json(fileobj: str | PathLike | IO, *args, **kwargs) -> dict:
    """Read local or remote JSON file."""
    with open(fileobj, "rb") as f:
        return json.load(f, *args, **kwargs)


def write_json(
    obj: dict,
    fileobj: str | PathLike | IO,
    *args,
    **kwargs
) -> dict:
    """Read local or remote JSON file."""
    with open(fileobj, "w") as f:
        return json.dump(obj, f, *args, **kwargs)


def update_json(
    obj: dict,
    fileobj: str | PathLike | IO,
    *args,
    **kwargs
) -> dict:
    """Read local or remote JSON file."""
    write_json(read_json(fileobj).update(obj), fileobj, *args, **kwargs)


def fsopen(
    uri: str | PathLike,
    mode: str = "rb",
    compression: str | None = None
) -> fsspec.spec.AbstractBufferedFile:
    """Open a file with fsspec (authentify if needed)."""
    fs = filesystem(uri)
    return fs.open(uri, mode, compression=compression)


class chain:
    """Chain multiple file objects."""

    # FIXME: never tested and not really used in the end

    def __init__(self, *files, **kwargs) -> None:
        self._files = list(map(lambda x: open(files, **kwargs)))
        self._file_index = 0

    def _open(self) -> None:
        self._open(self._file_index)

    def _close(self) -> None:
        self._close(self._file_index)

    def __enter__(self) -> IO:
        self._open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._close()

    def read(self, n: int | None = None, *a, **k) -> bytes | str:
        out = None
        while self._file_index < len(self._files):
            with self._files[self._file_index] as f:
                out1 = f.read(n, *a, **k)
                if n != 0 and len(out1) == 0:
                    self._file_index += 1
                if n is not None:
                    n -= len(out1)
                out = out1 if out is None else (out + out1)
                if n <= 0:
                    break
        return out

    def readline(self, n: int | None = None, *a, **k) -> bytes | str:
        out = []
        while self._file_index < len(self._files):
            with self._files[self._file_index] as f:
                out1 = f.readlines(n, *a, **k)
                if n != 0 and len(out1) == 0:
                    self._file_index += 1
                if n is not None:
                    n -= len(out1)
                out += out1
                if n <= 0:
                    break
        return out

    def readinto(self, buf: Buffer, *a, **k) -> None:
        # FIXME: I am not sure this implementation works
        buf0 = None
        if isinstance(buf, np.ndarray):
            buf0, buf = buf, np.empty_like(buf, shape=[buf.numel()])

        count = 0
        while self._file_index < len(self._files):
            with self._files[self._file_index] as f:
                count1 = f.readinto(buf[count:], *a, **k)
                if len(buf) != 0 and count1 in (0, None):
                    self._file_index += 1
                if count1 is not None:
                    count += count1
                if count >= len(buf):
                    break

        if buf0 is not None:
            buf0[...] = buf.reshape(buf0.shape)
            buf = buf0
        return count

    def write(self, *a, **k) -> int:
        raise NotImplementedError

    def writeline(self, *a, **k) -> int:
        raise NotImplementedError

    def peek(self, *a, **k) -> bytes | str:
        head = self.read(*a, **k)
        self._files.insert(self._file_index, BytesIO(head))
        return head

    def seek(self, *a, **k) -> int:
        raise NotImplementedError

    def tell(self, *a, **k) -> int:
        raise NotImplementedError


class open:
    """
    General-purpose stream opener.

    Handles:

    * paths (local or url)
    * opened file-objects
    * anything that `fsspec` handles
    * compressed streams
    """

    def __init__(
        self,
        fileobj: str | PathLike | IO,
        mode: str = 'rb',
        compression: str | None = None,
        open: bool = True,
    ) -> None:
        """
        Open a file from a path or url or an opened file object.

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
        auth : dict
            Authentification options to pass to `fsspec`.

        Returns
        -------
        fileobj
            Opened file
        """
        self.fileobj: str | IO = stringify_path(fileobj)
        """The input path or file-like object"""
        self.fileobjs: list[IO] = []
        """Additional file-like objects that may be created upon opening."""
        self.mode: str = mode
        """Opening mode."""
        self.compression: str = compression
        """Compression mode."""

        # open
        self._is_open = False
        if open:
            self._open()

    @property
    def _effective_fileobj(self) -> IO:
        # This is the most final file-like object
        return self.fileobjs[-1] if self.fileobjs else self.fileobj

    def _close(self) -> None:
        # close all the file-like objects that we've created
        for fileobj in reversed(self.fileobjs):
            fileobj.close()
        self.fileobjs = []
        self._is_open = False

    def _open(self) -> None:
        if self._is_open:
            return
        if isinstance(self.fileobj, str):
            self._fsspec_open()
        if self.compression == 'infer' and 'r' in self.mode:
            self._infer()
        self._is_open = True

    def _fsspec_open(self) -> None:
        # Take the last file object (most likely the input fileobj), which
        # must be a path, and open it with `fsspec.open()`.
        uri = stringify_path(self._effective_fileobj)
        # Ensure that input is a string
        if not isinstance(uri, str):
            raise TypeError("Expected a URI, but got:", uri)
        # Set compression option
        opt = dict()
        if not (self.compression == 'infer' and 'r' in self.mode):
            opt['compression'] = self.compression
        # Open with fsspec
        fileobj = fsopen(uri, mode=self.mode, **opt)
        self.fileobjs.append(fileobj)
        # -> returns a fsspec object that's not always fully opened
        #    so open again to be sure.
        if hasattr(fileobj, 'open'):
            fileobj = fileobj.open()
            self.fileobjs.append(fileobj)

    def _infer(self) -> None:
        # Take the last file object (which at this point must be a byte
        # stream) and infer its compression from its magic bytes.
        fileobj = self._effective_fileobj
        if not hasattr(fileobj, "peek"):
            fileobj = BufferedReader(fileobj)
            self.fileobjs.append(fileobj)
        try:
            magic = fileobj.peek(2)
        except Exception:
            return
        if magic == b'\x1f\x8b':
            fileobj = IndexedGzipFile(fileobj)
            self.fileobjs.append(fileobj)
        if magic == b'BZh':
            fileobj = BZ2File(fileobj)
            self.fileobjs.append(fileobj)

    # ------------------------------------------------------------------
    # open() / IO API
    # ------------------------------------------------------------------

    def __enter__(self) -> IO:
        self._open()
        return self._effective_fileobj

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._close()

    def __del__(self) -> None:
        self._close()

    def open(self) -> IO:
        if not self._is_open:
            self._open()
        return self

    def close(self) -> None:
        self._close()

    def read(self, *a, **k) -> bytes | str:
        return self._effective_fileobj.read(*a, **k)

    def readline(self, *a, **k) -> bytes | str:
        return self._effective_fileobj.readline(*a, **k)

    def readinto(self, *a, **k) -> None:
        return self._effective_fileobj.readinto(*a, **k)

    def write(self, *a, **k) -> int:
        return self._effective_fileobj.write(*a, **k)

    def writeline(self, *a, **k) -> int:
        return self._effective_fileobj.writeline(*a, **k)

    def peek(self, *a, **k) -> bytes | str:
        return self._effective_fileobj.peek(*a, **k)

    def seek(self, *a, **k) -> int:
        return self._effective_fileobj.seek(*a, **k)

    def tell(self, *a, **k) -> int:
        return self._effective_fileobj.tell(*a, **k)
