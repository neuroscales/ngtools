"""General-purpose byte stream opener.

Classes
-------
open
    General-purpose stream opener.
chain
    Chain multiple file objects.

Functions
---------
stringify_path
    Ensure that the input is a str or a file-like object.
parse_protocols
    Parse protocols out of a url.
fsopen
    Open a file with fsspec (authentify if needed).
filesystem
    Return the fsspec filesystem corresponding to a protocol or URI.
exists
    Check that the file or directory pointed by a URI exists.
linc_auth_opt
    Create fsspec authentication options to access lincbrain data.
read_json
    Read local or remote JSON file.
write_json
    Write local or remote JSON file.
update_json
    Update local or remote JSON file.
"""
# stdlib
import json
import logging
import time
from bz2 import BZ2File
from collections import namedtuple
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

LOG = logging.getLogger(__name__)

LINCBRAIN_API: str = "https://api.lincbrain.org/api"

# Cache for LINC cookies
_LINC_AUTH_CACHE = {}

# Cache for fsspec filesystems
_FILESYSTEMS_CACHE = {}


def stringify_path(filename: IO | PathLike | str) -> IO | str:
    """Ensure that the input is a str or a file-like object."""
    if isinstance(filename, PathLike):
        return filename.__fspath__()
    if isinstance(filename, Path):
        return str(filename)
    return filename


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


# def remote_protocols() -> tuple[str]:
#     """List of accepeted remote protocols."""
#     return tuple(x + '://' for x in PROTOCOLS if x != "file")


parsed_protocols = namedtuple(
    "parsed_protocols",
    ["layer", "format", "stream", "url"]
)


def parse_protocols(url: str | PathLike) -> tuple[str, str, str, str]:
    """
    Parse protocols out of a url.

    Returns a named tuple with fields "layer", "format", "stream", "url".

    Parameters
    ----------
    url : str | PathLike
        URL with or without protocols

    Returns
    -------
    layer : str | None
        Layer type (e.g. `"volume"`, `"labels"`, etc.)
    format : str | None
        Format (e.g., `"zarr"`, `"nifti"`, etc.)
    stream : str
        Stream/communication protocol (e.g., `"file"`, `"http"`, etc.)
    url : str
        URL with stream protocol but without format or layer type.
    """
    url = str(url)
    layer = format = stream = None
    *parts, path = str(url).split("://")
    for part in parts:
        if part in PROTOCOLS:
            if stream is not None:
                raise ValueError("Too many streaming protocols:", stream, part)
            stream = part
        elif part in LAYERS:
            if layer is not None:
                raise ValueError("Too many layer protocols:", layer, part)
            layer = part
        elif part in FORMATS:
            if format is not None:
                raise ValueError("Too many format protocols:", format, part)
            format = part
        else:
            raise ValueError("Unknown protocol:", part)
    stream = stream or "file"
    if stream != "file":
        path = stream + "://" + path
    return parsed_protocols(layer, format, stream, path)


def filesystem(
    protocol: str | PathLike | fsspec.AbstractFileSystem,
    **opt
) -> fsspec.AbstractFileSystem:
    """Return the filesystem corresponding to a protocol or URI."""
    if isinstance(protocol, fsspec.AbstractFileSystem):
        return protocol
    protocol = stringify_path(protocol)
    if protocol not in PROTOCOLS:
        protocol = parse_protocols(protocol).stream
    if protocol in _FILESYSTEMS_CACHE:
        return _FILESYSTEMS_CACHE[protocol]
    try:
        opt.update(linc_auth_opt())
    except Exception:
        pass
    fs = fsspec.filesystem(protocol, **opt)
    _FILESYSTEMS_CACHE[protocol] = fs
    return fs


def exists(uri: str | PathLike, **opt) -> bool:
    """Check that the file or directory pointed by a URI exists."""
    tic = time.time()
    uri0 = uri
    uri = parse_protocols(uri)[-1]
    fs = filesystem(uri, **opt)
    exists = fs.exists(uri)
    toc = time.time()
    LOG.debug(f"exists({uri0}): {toc-tic} s")
    return exists


def read_json(fileobj: str | PathLike | IO, *args, **kwargs) -> dict:
    """Read local or remote JSON file."""
    tic = time.time()
    with open(fileobj, "rb") as f:
        data = json.load(f, *args, **kwargs)
    toc = time.time()
    LOG.debug(f"read_json({fileobj}): {toc-tic} s")
    return data


def write_json(
    obj: dict,
    fileobj: str | PathLike | IO,
    *args,
    **kwargs
) -> None:
    """Write local or remote JSON file."""
    tic = time.time()
    with open(fileobj, "w") as f:
        json.dump(obj, f, *args, **kwargs)
    toc = time.time()
    LOG.debug(f"write_json(..., {fileobj}): {toc-tic} s")
    return


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
        """Read characters ("t") or byes ("b")."""
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
        """Read lines ("t")."""
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
        """Read bytes into an existing buffer."""
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
        """Write characters ("t") or bytes ("b")."""
        raise NotImplementedError

    def writeline(self, *a, **k) -> int:
        """Write lines."""
        raise NotImplementedError

    def peek(self, *a, **k) -> bytes | str:
        """Read tge first bytes without moving the cursor."""
        head = self.read(*a, **k)
        self._files.insert(self._file_index, BytesIO(head))
        return head

    def seek(self, *a, **k) -> int:
        """Move the cursor."""
        raise NotImplementedError

    def tell(self, *a, **k) -> int:
        """Return the cursor position."""
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
        """Open the file."""
        if not self._is_open:
            self._open()
        return self

    def close(self) -> None:
        """Close the file object."""
        self._close()

    def read(self, *a, **k) -> bytes | str:
        """Read characters ("t") or byes ("b")."""
        return self._effective_fileobj.read(*a, **k)

    def readline(self, *a, **k) -> bytes | str:
        """Read lines ("t")."""
        return self._effective_fileobj.readline(*a, **k)

    def readinto(self, *a, **k) -> None:
        """Read bytes into an existing buffer."""
        return self._effective_fileobj.readinto(*a, **k)

    def write(self, *a, **k) -> int:
        """Write characters ("t") or bytes ("b")."""
        return self._effective_fileobj.write(*a, **k)

    def writeline(self, *a, **k) -> int:
        """Write lines."""
        return self._effective_fileobj.writeline(*a, **k)

    def peek(self, *a, **k) -> bytes | str:
        """Read tge first bytes without moving the cursor."""
        return self._effective_fileobj.peek(*a, **k)

    def seek(self, *a, **k) -> int:
        """Move the cursor."""
        return self._effective_fileobj.seek(*a, **k)

    def tell(self, *a, **k) -> int:
        """Return the cursor position."""
        return self._effective_fileobj.tell(*a, **k)
