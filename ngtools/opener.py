import fsspec
from contextlib import contextmanager
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


@contextmanager
def open(fileobj, mode='rb', compression='infer'):
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
    fileobj = stringify_path(fileobj)
    if not hasattr(fileobj, 'read'):
        opt = dict()
        if not (compression == 'infer' and 'r' in mode):
            opt['compression'], compression = compression, None
        if fileobj.startswith(remote_protocols()):
            opt['block_size'] = 0
        with fsspec.open(fileobj, mode, **opt) as f:
            with open(f, mode, compression) as ff:
                yield ff
        return

    if compression == 'infer' and 'r' in mode \
            and not isinstance(fileobj, (IndexedGzipFile, BZ2File)):
        pos = fileobj.tell()
        magic = fileobj.read(2)
        fileobj.seek(pos)
        if magic == b'\x1f\x8b':
            with IndexedGzipFile(fileobj) as f:
                yield f
            return
        if magic == b'BZh':
            with BZ2File(fileobj) as f:
                yield f
            return

    yield fileobj
