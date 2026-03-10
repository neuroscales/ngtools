
# stdlib
import os.path as op
from os import PathLike

# externals
import neuroglancer as ng

# internals
from ngtools.local.fileserver import CurrentFileserver
from ngtools.opener import filesystem
from ngtools.protocols import parse_protocols

_LocalType = (ng.local_volume.LocalVolume, ng.skeleton.SkeletonSource)
_LocalLike = ng.local_volume.LocalVolume | ng.skeleton.SkeletonSource
_DataSourceLike = (
    str |
    ng.local_volume.LocalVolume |
    ng.skeleton.SkeletonSource |
    ng.LayerDataSource
)
_DataSourcesLike = (
    _DataSourceLike |
    list[_DataSourceLike | dict | None] |
    (ng.LayerDataSources | dict | None)
)
_LayerLike = ng.Layer | dict | None
_LayerArg = (
    _DataSourceLike |
    list[_DataSourceLike | dict | None] |
    ng.LayerDataSources |
    (ng.Layer | dict | None)
)
_AsType = type | tuple[type]


def get_layer_url(arg: _LayerArg, **kwargs) -> str | _LocalLike | None:
    if kwargs.get("source", None):
        return get_layer_url(kwargs.pop("source"), **kwargs)
    if kwargs.get("url", None):
        return get_layer_url(kwargs.pop("url"), **kwargs)
    if isinstance(arg, (str, PathLike)):
        return adapt_url(arg)
    if isinstance(arg, _LocalType):
        if hasattr(arg, "_url"):
            return arg._url
        return arg
    if hasattr(arg, "source"):
        return get_layer_url(arg.source)
    if hasattr(arg, "url"):
        return get_layer_url(arg.url)
    if isinstance(arg, dict):
        if arg.get("source", None):
            return get_layer_url(arg.get("source", arg))
        if arg.get("url", None):
            return get_layer_url(arg.get("url", arg))
    if hasattr(arg, "__iter__"):
        for arg1 in arg:
            url = get_layer_url(arg1, **kwargs)
            if url is not None:
                return url
    return None


def get_layer_source(arg: _LayerArg, **kwargs) -> _DataSourcesLike:
    if kwargs.get("source", None):
        return get_layer_url(kwargs.pop("source"), **kwargs)
    if isinstance(arg, (str, PathLike)):
        return adapt_url(arg)
    if isinstance(arg, _LocalType):
        return arg
    if hasattr(arg, "source"):
        return get_layer_url(arg.source)
    if hasattr(arg, "url"):
        return arg
    if isinstance(arg, dict):
        if arg.get("source", None):
            return get_layer_url(arg.get("source", arg))
        if "url" in arg:
            return arg
    if hasattr(arg, "__iter__"):
        return arg
    if isinstance(arg, (ng.LayerDataSource, ng.LayerDataSources)):
        return arg
    return None


def get_datasource_url(
    arg: _DataSourceLike, LocalSource: _AsType = tuple(), **kwargs
) -> str | _LocalLike | None:
    if kwargs.get("url", ""):
        url = kwargs["url"]
        if isinstance(url, (str, PathLike)):
            url = kwargs["url"] = adapt_url(url)
    elif isinstance(arg, (str, PathLike)):
        url = arg = adapt_url(arg)
    elif hasattr(arg, "url"):
        url = arg.url
        if isinstance(url, (str, PathLike)):
            url = arg.url = adapt_url(url)
    elif isinstance(arg, dict) and "url" in arg:
        url = arg["url"]
        if isinstance(url, (str, PathLike)):
            url = arg["url"] = adapt_url(arg["url"])
    elif not isinstance(arg, LocalSource):
        raise ValueError("Missing data source url")
    return url, arg, kwargs


def adapt_url(uri: str) -> str:
    with CurrentFileserver() as fileserver:
        uri = str(uri).rstrip("/")
        parsed = parse_protocols(uri)
        short_uri = parsed.url
        basename = op.basename(short_uri)

        # extension-based hint
        if not parsed.format:
            if basename.endswith(".zarr"):
                parsed = parsed.with_format("zarr")
            elif basename.endswith(".n5"):
                parsed = parsed.with_format("n5")
            elif basename.endswith((".nii", ".nii.gz")):
                parsed = parsed.with_format("nifti")

        if parsed.stream == "dandi":
            # neuroglancer does not understand dandi:// uris,
            # so we use the s3 url instead.
            short_uri = filesystem(short_uri).s3_url(short_uri)
            parsed = parsed.with_part(stream="https", url=short_uri)

        elif parsed.stream == "file":
            # neuroglancer does not understand file:// uris,
            # so we serve it over http using a local fileserver.
            if not fileserver:
                raise ValueError(
                    "Cannot load local files without a fileserver"
                )
            short_uri = fileserver + "/local/" + op.abspath(short_uri)
            parsed = parsed.with_part(stream="http", url=short_uri)

        if fileserver:
            # if local viewer and data is on linc
            # -> redirect to our handler that deals with credentials
            linc_prefix = "https://neuroglancer.lincbrain.org/"
            if parsed.url.startswith(linc_prefix):
                path = parsed.url[len(linc_prefix):]
                local_url = fileserver + "/linc/" + path
                parsed = parsed.with_part(stream="http", url=local_url)

        uri = str(parsed).rstrip("/")

    return uri
