"""
Wrappers around neuroglancer classes that can compute metadata and
optional attributes that neuroglancer typically delegates to the frontend.

Dependencies
------------
* nifti://          -> nibabel
* zarr://           -> zarr
* precomputed://    -> cloudvolume
"""
# stdlib
import functools
import json
import logging
import time
from itertools import product
from os import PathLike
from types import EllipsisType
from typing import Callable, Iterator

# externals
import neuroglancer as ng
import nibabel as nib
import numpy as np
import zarr
import zarr.storage
from cloudvolume import CloudVolume
from nibabel.spatialimages import HeaderDataError
from numpy.typing import ArrayLike
from upath import UPath

# internals
import ngtools.transforms as T
import ngtools.units as U
from ngtools.opener import (
    exists,
    filesystem,
    open,  # !!! hides builtin open
    parse_protocols,
    read_json,
)
from ngtools.utils import Wraps

LOG = logging.getLogger(__name__)


class _SizedCache(dict):
    def __init__(self, max_size: int | None = None) -> None:
        self._max_size = max_size
        self._keys = []

    def __setitem__(self, key: object, value: object) -> None:
        if key in self:
            del self[key]
        super().__setitem__(key, value)
        if self._max_size:
            keys = list(self.keys())
            while len(self) > self._max_size:
                del self[keys.pop(0)]


_DATASOURCE_REGISTRY = {}
_DATASOURCE_INFO_CACHE = _SizedCache(128)


def _quantiles(q: float | list[float], data: ArrayLike) -> np.ndarray[float]:
    """Compute (efficiently) intensity quantiles."""
    tic = time.time()
    steps = [max(1, x//64) for x in data.shape]
    slicer = (Ellipsis,) + tuple(slice(None, None, step) for step in steps)
    data = np.asarray(data[slicer])
    data = data[np.isfinite(data) & (data != 0)]
    quantiles = np.quantile(data, q)
    toc = time.time()
    LOG.info(f"Compute quantiles: {toc-tic} s")
    return quantiles


def datasource(formats: str | list[str]) -> Callable[[type], type]:
    """
    Register a format-specific datasource class.

    Parameters
    ----------
    formats : str | list[str]
        Formats that should fallback to the decorated class.
    """
    if isinstance(formats, str):
        formats = [formats]

    def decorator(cls: type[LayerDataSource]) -> type[LayerDataSource]:
        for format in formats:
            _DATASOURCE_REGISTRY[format] = cls
        cls.PROTOCOLS = list(formats)
        return cls

    return decorator


SourceType = (
    dict |                                          # json
    str | PathLike |                                # uri
    ng.LayerDataSource |                            # data source
    ng.LocalVolume | ng.skeleton.SkeletonSource     # local source
)


class LayerDataSources(Wraps(ng.LayerDataSources)):
    """List of data sources."""

    _DataSourceLike = (
        str |
        ng.local_volume.LocalVolume |
        ng.skeleton.SkeletonSource |
        ng.LayerDataSource
    )
    _DataSourcesLike = (
        _DataSourceLike |
        list[_DataSourceLike | dict | None] |
        ng.LayerDataSources |
        dict |
        None
    )

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # Trigger data source computation so that defaults that differ
        # between ngtools and neuroglancer are set according to ngtools.
        for i, val in enumerate(self):
            self[i] = val

    @functools.wraps(ng.LayerDataSources.__iter__)
    def __iter__(self) -> Iterator[ng.LayerDataSource]:
        for source in super().__iter__():
            yield LayerDataSource(source)

    @functools.wraps(ng.LayerDataSources.__getitem__)
    def __getitem__(self, *args, **kwargs) -> "LayerDataSource":
        return LayerDataSource(super().__getitem__(*args, **kwargs))

    @functools.wraps(ng.LayerDataSources.pop)
    def pop(self, *args, **kwargs) -> "LayerDataSource":  # noqa: D102
        return LayerDataSource(super().pop(*args, **kwargs))


class _LayerDataSourceFactory(type):
    """Factory for LayerDataSource objects."""

    _LocalSource = (ng.local_volume.LocalVolume, ng.skeleton.SkeletonSource)
    _DataSourceLike = (
        str |
        ng.local_volume.LocalVolume |
        ng.skeleton.SkeletonSource |
        ng.LayerDataSource |
        dict |
        None
    )

    def __call__(
        cls,
        arg: _DataSourceLike = None,
        *args,
        **kwargs
    ) -> "LayerDataSource":
        # Only use the factory if it is not called from a subclass
        if cls is not LayerDataSource:
            obj = super().__call__(arg, *args, **kwargs)
            return obj

        # Get url
        if kwargs.get("url", ""):
            url = kwargs["url"]
            if isinstance(url, (str, PathLike)):
                url = kwargs["url"] = str(url)
        elif isinstance(arg, (str, PathLike)):
            url = arg = str(arg)
        elif hasattr(arg, "url"):
            url = arg.url
        elif isinstance(arg, dict) and "url" in arg:
            url = arg["url"]
        elif not isinstance(arg, cls._LocalSource):
            raise ValueError("Missing data source url")

        # If local object -> delegate
        if isinstance(url, ng.LocalVolume):
            return LocalVolumeDataSource(arg, *args, **kwargs)
        if isinstance(url, ng.skeleton.SkeletonSource):
            return LocalSkeletonDataSource(arg, *args, **kwargs)

        parsed_url = parse_protocols(url)

        # If python:// url -> local object, but retrieved from the viewer.
        if parsed_url.stream == "python":

            # Get local object
            path = parsed_url.url
            source_type = path.split("://")[-1].strip("/").split("/")[0]
            kls = {
                "volume": LocalVolumeDataSource,
                "skeleton": LocalSkeletonDataSource
            }[source_type]

            # Defer to LocalDataSource factory
            # TODO: build a `LocalDataSourceFactory` to defer even more?
            return kls(arg, *args, **kwargs)

        # If local:// url -> local annotations
        if parsed_url.url == "local://annotations":
            return LocalAnnotationDataSource(arg, *args, **kwargs)

        # If format protocol provided, use it
        if parsed_url.format in _DATASOURCE_REGISTRY:
            format = parsed_url.format
            LOG.debug(f"LayerDataSource - use format hint: {format}")
            return _DATASOURCE_REGISTRY[format](arg, *args, **kwargs)

        # Otherwise, check for extensions
        for format, kls in _DATASOURCE_REGISTRY.items():
            if parsed_url.url.endswith((format, format+".gz", format+".bz2")):
                LOG.debug(f"LayerDataSource - found extension: {format}")
                try:
                    obj = kls(arg, *args, **kwargs)
                    LOG.debug(f"LayerDataSource - {format} (success)")
                    return obj
                except Exception:
                    continue

        if False:  # Do not use for now -- need to implement sniffers
            # Otherwise, try all
            for format, kls in _DATASOURCE_REGISTRY.items():
                LOG.debug(f"LayerDataSource - try format: {format}")
                try:
                    obj = kls(arg, *args, **kwargs)
                    LOG.debug(f"LayerDataSource - {format} (success)")
                    return obj
                except Exception:
                    continue

        # Otherwise. build a simple source
        LOG.debug("LayerDataSource - Fallback to simple LayerDataSource")
        return super().__call__(arg, *args, **kwargs)


class DataSourceInfo:
    """Base class for source-specific metadata."""

    ...


class LayerDataSource(Wraps(ng.LayerDataSource),
                      metaclass=_LayerDataSourceFactory):
    """A wrapper around `ng.LayerDataSource` that computes metadata."""

    PROTOCOLS: list[str] = []

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # ensure that the url has the right format
        if isinstance(self.url, str):
            url = parse_protocols(self.url).url
            if self.PROTOCOLS:
                url = self.PROTOCOLS[0] + "://" + url
            self.url = url

    def _compute_info(self) -> DataSourceInfo:
        """Compute format-specific metadata."""
        ...

    def to_json(self) -> dict:  # noqa: D102
        # ensure the transform that we guess is saved in the ng object
        self.transform = self.transform
        return super().to_json()

    @property
    def info(self) -> DataSourceInfo:
        """Lazy access to format-specific metadata."""
        if not getattr(self, '_info', None):
            LOG.debug("Trigger _compute_info")
            url = self.local_url
            if url:
                url = parse_protocols(url).url
            key = (type(self), url)
            if key not in _DATASOURCE_INFO_CACHE:
                LOG.debug("Recompute info")
                try:
                    _DATASOURCE_INFO_CACHE[key] = self._compute_info()
                except Exception:
                    LOG.warning(f"Could not compute info for: {key}")
            else:
                LOG.debug("Use cached info")
            self._info = _DATASOURCE_INFO_CACHE.get(key, None)
        return self._info

    @classmethod
    def _raise_not_implemented_error(cls, name: str) -> None:
        raise NotImplementedError(
            f"`{name}` not implemented for this format ({cls.__name__})."
        )

    def __set__url__(self, value: str) -> str:
        LocalObject = (ng.local_volume.LocalVolume, ng.skeleton.SkeletonSource)
        if not isinstance(value, LocalObject):
            format = parse_protocols(value)[1]
            if format not in self.PROTOCOLS:
                raise ValueError(
                    "Cannot assign uri with protocol", format, "in "
                    "data source of type:", type(self)
                )
            self._info = None
        return value

    @property
    def local_url(self) -> str:
        """Path to url, even if file is local."""
        url = self.url
        if isinstance(url, (ng.LocalVolume, ng.skeleton.SkeletonSource)):
            if hasattr(self, '_url'):
                return self._url
            if hasattr(url, 'url'):
                return url.url
            return None
        return url

    def __get_transform__(self) -> ng.CoordinateSpaceTransform:
        """
        If a transform has been explicitely set, return it,
        else, return the transform implicitely defined by neuroglancer.
        """
        assigned_trf = self._wrapped.transform
        if (
            assigned_trf is None or
            assigned_trf.matrix is None or
            assigned_trf.input_dimensions is None or
            assigned_trf.output_dimensions is None
        ):
            tic = time.time()
            implicit_trf = getattr(self, '_transform', None)
            if implicit_trf is None:
                return assigned_trf
            if assigned_trf is None:
                return implicit_trf
            toc = time.time()
            LOG.debug(
                f"{type(self).__name__}: "
                "Compute implicit transform: "
                f"{toc - tic} s"
            )
            matrix = assigned_trf.matrix
            if matrix is None:
                matrix = implicit_trf.matrix
            idims = (
                assigned_trf.input_dimensions or
                implicit_trf.input_dimensions
            )
            odims = (
                assigned_trf.output_dimensions or
                implicit_trf.input_dimensions
            )
            self._wrapped.transform = ng.CoordinateSpaceTransform(
                matrix=matrix,
                input_dimensions=idims,
                output_dimensions=odims,
            )
            return self._wrapped.transform
        else:
            return assigned_trf

    @property
    def shape(self) -> list[int]:
        """Return the shape of the array (spatial dimensions first)."""
        if hasattr(self, "_shape"):
            return self._shape
        self._raise_not_implemented_error("shape")

    @property
    def rank(self) -> int:
        """Return the number of dimensions in the array."""
        return len(self.shape)

    ndim = rank

    @property
    def dtype(self) -> np.dtype:
        """Return the data type of the array."""
        if hasattr(self, "_dtype"):
            return self._dtype
        self._raise_not_implemented_error("dtype")

    @property
    def slope_inter(self) -> tuple[float, float]:
        """Affine scaling to apply to the stored intensity."""
        if hasattr(self, "_slope_inter"):
            return self._slope_inter
        return (1.0, 0.0)

    @property
    def dataobj(self) -> ArrayLike:
        """Return an array-like object pointing to the data."""
        return self.get_dataobj(0)

    @property
    def nb_levels(self) -> int:
        """Number of pyramid levels."""
        if hasattr(self, "_nb_levels"):
            return self._nb_levels
        return 1

    def get_dataobj(self, level: int = 0, mode: str = "r") -> ArrayLike:
        """Return an array-like object pointing to a pyramid level."""
        if level < 0:
            level = self.nb_levels + level
        return self._get_dataobj(level, mode)

    def _get_dataobj(self, level: int = 0, mode: str = "r") -> ArrayLike:
        # Implementation classes should implement this method.
        self._raise_not_implemented_error("get_dataobj")

    _IndexType = int | bool | slice | ArrayLike | EllipsisType
    _SlicerType = _IndexType | tuple[_IndexType]

    def __getitem__(self, slicer: _SlicerType) -> ArrayLike:
        return self.dataobj[slicer]

    def quantiles(self, q: float | list[float]) -> np.ndarray:
        """Compute data quantiles."""
        spatial_dims = self.transform.input_dimensions
        spatial_dims = [
            i for i, unit in enumerate(spatial_dims.units)
            if unit.endswith("m")
        ]
        for i in range(self.nb_levels):
            dat = self.get_dataobj(i, mode="r")
            if all([dat.shape[i] <= 64 for i in spatial_dims]):
                break
        return _quantiles(q, dat)

    def apply_transform(self, *args: ng.CoordinateSpaceTransform
                        ) -> "LayerDataSource":
        """Apply an additional transform in model space."""
        self.transform = T.compose(*args, self.transform)
        return self

    @property
    def input_dimensions(self) -> ng.CoordinateSpace:
        """Input dimensions."""
        return self.transform.input_dimensions

    inputDimensions = input_dimensions

    @property
    def output_dimensions(self) -> ng.CoordinateSpace:
        """Output dimensions."""
        return self.transform.output_dimensions

    outputDimensions = output_dimensions

    @property
    def input_bbox(self) -> list[list[float]]:
        """Bounding box, in input_dimensions space and units."""
        shape = self.shape
        # FIXME: should there be a half voxel shift?
        min = [0.0] * self.rank
        max = shape
        return [min, max]

    @property
    def output_bbox(self) -> list[list[float]]:
        """Bounding box, in output_dimensions space and units."""
        bbox = self.input_bbox

        mn = np.full([self.rank], +float('inf'))
        mx = np.full([self.rank], -float('inf'))

        for corner in product([0, 1], repeat=self.rank):
            coord = [bbox[j][i] for i, j in enumerate(corner)]

            mat = np.eye(self.rank+1)[:-1]
            mat[:, -1] = coord
            coord = ng.CoordinateSpaceTransform(
                input_dimensions=self.input_dimensions,
                output_dimensions=self.input_dimensions,
                matrix=mat,
            )

            coord = T.compose(self.transform, coord)
            coord = coord.matrix[:, -1]

            mn = np.minimum(mn, coord)
            mx = np.maximum(mx, coord)

        bbox = [mn.tolist(), mx.tolist()]
        return bbox

    @property
    def input_center(self) -> list[float]:
        """Center of the field of view in input dimensions space and units."""
        bbox = np.asarray(self.input_bbox)
        return ((bbox[0] + bbox[1]) / 2).tolist()

    @property
    def output_center(self) -> list[float]:
        """Center of the field of view in output dimensions space and units."""
        center = self.input_center
        mat = np.eye(self.rank+1)[:-1]
        mat[:, -1] = center
        center = ng.CoordinateSpaceTransform(
            input_dimensions=self.input_dimensions,
            output_dimensions=self.input_dimensions,
            matrix=mat,
        )
        center = T.compose(self.transform, center)
        center = center.matrix[:, -1]
        return center

    @property
    def input_bbox_size(self) -> list[float]:
        """Center of the field of view in input dimensions space and units."""
        bbox = np.asarray(self.input_bbox)
        return (bbox[1] - bbox[0]).tolist()

    @property
    def output_bbox_size(self) -> list[float]:
        """Center of the field of view in output dimensions space and units."""
        bbox = np.asarray(self.output_bbox)
        return (bbox[1] - bbox[0]).tolist()

    @property
    def output_voxel_size(self) -> list[float]:
        """Voxel size in model space."""
        mat = np.eye(self.rank+1)[:-1]
        mat[:, -1] = 1
        size = ng.CoordinateSpaceTransform(
            input_dimensions=self.input_dimensions,
            output_dimensions=self.input_dimensions,
            matrix=mat,
        )
        size = T.compose(self.transform, size)
        size = size.matrix[range(self.rank), range(self.rank)]
        return size.tolist()


class VolumeInfo(DataSourceInfo):
    """Base class for volume metadata."""

    def __init__(self, url: str) -> None:
        """
        Parameters
        ----------
        url : str
            URL to the file
        """
        self.url = str(url)

    def getNbLevels(self) -> int:
        """Compute number of dimensions."""
        return 1

    def getShape(self, level: int = 0) -> list[int]:
        """Compute array shape."""
        raise NotImplementedError

    def getRank(self) -> int:
        """Compute number of dimensions."""
        return len(self.getShape())

    def getSlopeInter(self) -> tuple[float, float]:
        """Affine intensity scaling."""
        return (1.0, 0.0)

    def getInputNames(self) -> list[str]:
        """Compute names of input axes."""
        raise NotImplementedError

    def getOutputNames(self) -> list[str]:
        """Compute names of output axes."""
        raise NotImplementedError

    def getInputUnits(self) -> list[str]:
        """Compute all units."""
        raise NotImplementedError

    def getOutputUnits(self) -> list[str]:
        """Compute all units."""
        raise NotImplementedError

    def getInputScales(self) -> list[float]:
        """Compute input scales."""
        raise NotImplementedError

    def getOutputScales(self) -> list[float]:
        """Compute output scales."""
        raise NotImplementedError

    def getDataType(self) -> np.dtype:
        """Compute numpy data type."""
        raise NotImplementedError

    def getInputDimensions(self) -> ng.CoordinateSpace:
        """Compute input dimensions."""
        names = self.getInputNames()
        units = self.getInputUnits()
        scales = self.getInputScales()
        return ng.CoordinateSpace({
            name: [scale, unit]
            for name, scale, unit in zip(names, scales, units)
        })

    def getOutputDimensions(self) -> ng.CoordinateSpace:
        """Compute output dimensions."""
        names = self.getOutputNames()
        units = self.getOutputUnits()
        scales = self.getOutputScales()
        return ng.CoordinateSpace({
            name: [scale, unit]
            for name, scale, unit in zip(names, scales, units)
        })

    def getMatrix(self) -> np.ndarray:
        """Compute transformation matrix."""
        rank = self.getRank()
        return np.eye(rank+1)[:-1]

    def getTransform(self) -> ng.CoordinateSpaceTransform:
        """Compute neuroglancer transform."""
        return ng.CoordinateSpaceTransform(
            matrix=self.getMatrix(),
            input_dimensions=self.getInputDimensions(),
            output_dimensions=self.getOutputDimensions(),
        )


class VolumeDataSource(LayerDataSource):
    """Wrapper for volume source."""

    info: VolumeInfo

    @property
    def _transform(self) -> list[int]:
        return self.info.getTransform()

    @property
    def _nb_levels(self) -> list[int]:
        return self.info.getNbLevels()

    @property
    def _shape(self) -> list[int]:
        return self.info.getShape()

    @property
    def _dtype(self) -> np.dtype:
        return self.info.getDataType()

    @property
    def _slope_inter(self) -> tuple[float, float]:
        return self.info.getSlopeInter()


class SkeletonDataSource(LayerDataSource):
    """Wrapper for skeleton source."""

    ...


class MeshDataSource(LayerDataSource):
    """Wrapper for mesh source."""

    ...


class AnnotationDataSource(LayerDataSource):
    """Wrapper for annotation source."""

    ...


class LocalDataSource(LayerDataSource):
    """Wrapper for local data sources."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._format_specific_init()

    def _format_specific_init(self) -> None:
        pass


class LocalAnnotationDataSource(AnnotationDataSource, LocalDataSource):
    """Wrapper for local annotation sources."""

    ...


class _LocalSkeletonDataSourceFactory(_LayerDataSourceFactory):

    def __call__(
        cls,
        arg: _LayerDataSourceFactory._DataSourceLike = None,
        *args,
        **kwargs
    ) -> "LocalSkeletonDataSource":
        # Only use the factory if it is not called from a subclass
        if cls is not LocalSkeletonDataSource:
            obj = super().__call__(arg, *args, **kwargs)
            return obj

        # Get url
        if kwargs.get("url", ""):
            url = kwargs["url"]
            if isinstance(url, (str, PathLike)):
                url = kwargs["url"] = str(url)
        elif isinstance(arg, (str, PathLike)):
            url = arg = str(arg)
        elif hasattr(arg, "url"):
            url = arg.url
        elif isinstance(arg, dict) and "url" in arg:
            url = arg["url"]
        elif not isinstance(arg, cls._LocalSource):
            raise ValueError("Missing data source url")

        # If local object -> delegate
        if isinstance(url, ng.LocalVolume):
            raise ValueError(
                "Non skeleton local object passed to a skeleton source"
            )
        if isinstance(url, ng.skeleton.SkeletonSource):
            return super().__call__(arg, *args, **kwargs)

        parsed_url = parse_protocols(url)

        # If python:// url -> local object, but retrieved from the viewer.
        if parsed_url.stream == "python":

            # Get local object
            path = parsed_url.url
            source_type, token = path.split("://")[-1].strip("/").split("/")
            if source_type != "skeleton":
                raise ValueError(
                    "Non skeleton local object passed to a skeleton source"
                )
            vol = ng.server.global_server.get_volume(token)

            # Replace url with object
            if kwargs.get("url", ""):
                kwargs["url", vol]
            elif isinstance(arg, (str, PathLike)):
                arg = vol
            elif hasattr(arg, "url"):
                arg.url = vol
            elif isinstance(arg, dict) and "url" in arg:
                arg["url"] = vol

            # Return object
            kls = getattr(vol, "DataSourceType", LocalSkeletonDataSource)
            if kls is LocalSkeletonDataSource:
                return super().__call__(arg, *args, **kwargs)
            else:
                return kls(arg, *args, **kwargs)

        # If format protocol provided, use it
        if parsed_url.format in _DATASOURCE_REGISTRY:
            format = parsed_url.format
            LOG.debug(f"LocalSkeletonDataSource - use format hint: {format}")
            return _DATASOURCE_REGISTRY[format](arg, *args, **kwargs)

        # Otherwise, check for extensions
        for format, kls in _DATASOURCE_REGISTRY.items():
            if parsed_url.url.endswith((format, format+".gz", format+".bz2")):
                LOG.debug(f"LocalSkeletonDataSource - extension: {format}")
                try:
                    obj = kls(arg, *args, **kwargs)
                    LOG.debug(f"LocalSkeletonDataSource - {format} (success)")
                    return obj
                except Exception:
                    continue

        # Otherwise. build a simple source
        LOG.debug("LocalSkeletonDataSource - Fallback")
        return super().__call__(arg, *args, **kwargs)


class LocalSkeletonDataSource(SkeletonDataSource, LocalDataSource,
                              metaclass=_LocalSkeletonDataSourceFactory):
    """Wrapper for data source that wraps a `SkeletonSource`."""

    @property
    def local_skeleton(self) -> ng.LocalVolume | None:
        """Points to the underlying `SkeletonSource`, if any."""
        url = super().url
        if isinstance(self, ng.skeleton.SkeletonSource):
            return url
        return None

    ...


class _LocalVolumeDataSourceFactory(_LayerDataSourceFactory):

    def __call__(
        cls,
        arg: _LayerDataSourceFactory._DataSourceLike = None,
        *args,
        **kwargs
    ) -> "LocalVolumeDataSource":
        # Only use the factory if it is not called from a subclass
        if cls is not LocalVolumeDataSource:
            obj = super().__call__(arg, *args, **kwargs)
            return obj

        # Get url
        if kwargs.get("url", ""):
            url = kwargs["url"]
            if isinstance(url, (str, PathLike)):
                url = kwargs["url"] = str(url)
        elif isinstance(arg, (str, PathLike)):
            url = arg = str(arg)
        elif hasattr(arg, "url"):
            url = arg.url
        elif isinstance(arg, dict) and "url" in arg:
            url = arg["url"]
        elif not isinstance(arg, cls._LocalSource):
            raise ValueError("Missing data source url")

        # If local object -> delegate
        if isinstance(url, ng.LocalVolume):
            return LocalVolumeDataSource(arg, *args, **kwargs)
        if isinstance(url, ng.skeleton.SkeletonSource):
            raise ValueError(
                "Non volume local object passed to a volume source"
            )

        parsed_url = parse_protocols(url)

        # If python:// url -> local object, but retrieved from the viewer.
        if parsed_url.stream == "python":

            # Get local object
            path = parsed_url.url
            source_type, token = path.split("://")[-1].strip("/").split("/")
            if source_type != "volume":
                raise ValueError(
                    "Non volume local object passed to a volume source"
                )
            vol = ng.server.global_server.get_volume(token)

            # Replace url with object
            if kwargs.get("url", ""):
                kwargs["url", vol]
            elif isinstance(arg, (str, PathLike)):
                arg = vol
            elif hasattr(arg, "url"):
                arg.url = vol
            elif isinstance(arg, dict) and "url" in arg:
                arg["url"] = vol

            # Return object
            kls = getattr(vol, "DataSourceType", LocalVolumeDataSource)
            if kls is LocalVolumeDataSource:
                return super().__call__(arg, *args, **kwargs)
            else:
                return kls(arg, *args, **kwargs)

        # If format protocol provided, use it
        if parsed_url.format in _DATASOURCE_REGISTRY:
            format = parsed_url.format
            LOG.debug(f"LocalVolumeDataSource - use format hint: {format}")
            return _DATASOURCE_REGISTRY[format](arg, *args, **kwargs)

        # Otherwise, check for extensions
        for format, kls in _DATASOURCE_REGISTRY.items():
            if parsed_url.url.endswith((format, format+".gz", format+".bz2")):
                LOG.debug(f"LocalVolumeDataSource - extension: {format}")
                try:
                    obj = kls(arg, *args, **kwargs)
                    LOG.debug(f"LocalVolumeDataSource - {format} (success)")
                    return obj
                except Exception:
                    continue

        # Otherwise. build a simple source
        LOG.debug("LocalVolumeDataSource - Fallback")
        return super().__call__(arg, *args, **kwargs)


class LocalVolumeDataSource(VolumeDataSource, LocalDataSource,
                            metaclass=_LocalVolumeDataSourceFactory):
    """Wrapper for data source that wraps a `LocalVolume`."""

    def __init__(self, *args, **kwargs) -> None:
        self._layer_type = kwargs.pop("layer_type", None)
        super().__init__(*args, **kwargs)

    @property
    def local_volume(self) -> ng.LocalVolume | None:
        """Points to the underlying `LocalVolume`, if any."""
        url = super().url
        if isinstance(self, ng.LocalVolume):
            return url
        return None

    @property
    def _nb_levels(self) -> int:
        return 1

    @property
    def _dtype(self) -> np.dtype:
        return self.local_volume.data.data.dtype

    @property
    def _shape(self) -> np.dtype:
        return self.local_volume.data.shape

    def _get_dataobj(self, level: int = 0, mode: str = "r") -> ArrayLike:
        return self.local_volume.data.data


class NiftiVolumeInfo(VolumeInfo):
    """Metadata of a NIfTI volume."""

    def __init__(
        self,
        url: str | PathLike,
        align_corner: bool = False,
        affine: str = "best",
    ) -> None:
        """
        Parameters
        ----------
        url : str
            URL to the file
        align_corner : bool
            If True, use neuroglancer's native behavior when computing
            the transform, which is to asume that (0, 0, 0) points to
            the corner of the first voxel. If False, use NIfTI's spec,
            which is to assume the (0, 0, 0) points to the center of the
            first voxel.
        affine : {"qform", "sform", "best", "base"}
            Which orientation matrix to use.
            Neuroglancer uses the qform by default.
        """
        super().__init__(url)
        self._align_corner = align_corner
        self._affine = affine
        self._nib_header: nib.nifti1.Nifti1Header = self._load()

    def _load(self) -> nib.nifti1.Nifti1Header | nib.nifti2.Nifti2Header:
        NiftiHeaders = (nib.nifti1.Nifti1Header, nib.nifti2.Nifti2Header)
        tic = time.time()
        url = parse_protocols(self.url).url
        for compression in ('infer', 'gzip', None):
            fileobj = open(url, compression=compression)
            for hdr_klass in NiftiHeaders:
                try:
                    try:
                        hdr = hdr_klass.from_fileobj(fileobj, check=True)
                    except HeaderDataError:
                        # Check if failure is due to "old" nifti-zarr magic
                        hdr = hdr_klass.from_fileobj(fileobj, check=False)
                        if hdr["magic"].item().decode() not in ("nz1", "nz2"):
                            raise
                    toc = time.time()
                    LOG.info(f"Load nii header: {toc-tic} s")
                    return hdr
                except Exception:
                    pass
        toc = time.time()
        LOG.info(f"Failed to load nii header: {toc-tic} s")
        raise RuntimeError("Failed to load file.")

    _allSourceNames = ["i", "j", "k", "m", "c^", "c1^", "c2^"]
    _allViewNames = ["x", "y", "z", "t", "c^", "c1^", "c2^"]

    def getShape(self) -> list[int]:
        """Compute array shape."""
        return list(self._nib_header.get_data_shape())

    def getInputNames(self) -> list[str]:
        """Compute names of input axes."""
        return self._allSourceNames[:self.getRank()]

    def getOutputNames(self) -> list[str]:
        """Compute names of output axes."""
        return self._allViewNames[:self.getRank()]

    def getSpaceUnit(self) -> str:
        """Compute spatial unit."""
        if not getattr(self, "_space_unit", None):
            unit = self._nib_header.get_xyzt_units()[0]
            if unit == "unknown":
                unit = "mm"
            unit = U.as_neuroglancer_unit(unit)
            self._space_unit = unit
        return self._space_unit

    def getTimeUnit(self) -> str:
        """Compute time unit."""
        if not getattr(self, "_time_unit", None):
            unit = self._nib_header.get_xyzt_units()[1]
            if unit == "unknown":
                unit = "sec"
            unit = U.as_neuroglancer_unit(unit)
            self._time_unit = unit
        return self._time_unit

    def getUnits(self) -> list[str]:
        """Compute all units."""
        all_units = (
            [self.getSpaceUnit()] * 3 +
            [self.getTimeUnit()] +
            [""] * 3
        )
        return all_units[:self.getRank()]

    getInputUnits = getUnits
    getOutputUnits = getUnits

    def getInputScales(self) -> list[float]:
        """Compute input scales."""
        return list(map(float, self._nib_header.get_zooms()))

    def getOutputScales(self) -> list[float]:
        """Compute output scales."""
        return [1] * self.getRank()

    def getDescription(self) -> str:
        """Compute description string."""
        return self._nib_header["description"]

    def getDataType(self) -> np.dtype:
        """Compute numpy data type."""
        return self._nib_header.get_data_dtype()

    def getSlopeInter(self) -> tuple[float, float]:
        """Compute intensity transform."""
        scale, slope = self._nib_header.get_slope_inter()
        if scale is None:
            scale = 1.0
        if slope is None:
            slope = 0.0
        return scale, slope

    def getMatrix(self) -> np.ndarray:
        """Compute transformation matrix."""
        # Neuroglancer uses the qform (rotation), but without
        # scaling it by voxel size. The linear part therefore maps
        # spatial units to spatial units (m, mm or um)
        # and the offset is expressed in spatial units.
        #
        # Nibabel returns a voxel-to-spatial-unit transform, so we need
        # to rescale it
        #
        # If `_align_corner=False`, we introduce an additional shift
        # to account for the fact that neuroglancer assumes that coordinate
        # (0, 0, 0) of the input space is at the corner of the first voxel.
        if getattr(self, "_matrix", None) is None:
            srank = min(3, self.getRank())
            for affine in (self._affine, "best", "sform", "qform", "base"):
                if affine == "base":
                    matrix = self._nib_header.get_base_affine()[:-1]
                elif affine == "best":
                    matrix = self._nib_header.get_best_affine()[:-1]
                elif affine == "sform":
                    matrix = self._nib_header.get_sform()[:-1]
                else:
                    matrix = self._nib_header.get_qform()[:-1]
                scales = (matrix[:3, :3] ** 2).sum(0) ** 0.5
                if scales.max() == 0:
                    continue
                else:
                    break
            if not self._align_corner:
                matrix[:3, -1] -= matrix[:3, :-1] @ ([0.5]*srank)
            matrix[:3, :-1] /= scales
            fullmatrix = np.eye(self.getRank()+1)[:-1]
            fullmatrix[:srank, :srank] = matrix[:srank, :srank]
            fullmatrix[:srank, -1] = matrix[:srank, -1]
            self._matrix = fullmatrix
        return self._matrix


@datasource(["nifti", "nii"])
class NiftiDataSource(VolumeDataSource):
    """
    Wrapper for nifti sources.

    Note that by default, neuroglancer assumes that the coordinate
    (0, 0, 0) in voxel space corresponds to the _corner_ of the first
    voxel, whereas the nifti spec is that it corresponds to the _center_
    of the first voxel. This wrapper follows the nifti convention by default.
    The neuroglancer default behavior can be recovered with the option
    `align_corner=True`.

    Also, by default, neuroglancer uses the `qform` orientation matrix,
    whereas the `intent` code must be used to choose between the `sform`
    and `qform` according to the spec (as implemented in e.g. nibabel).
    This wrapper follows the specification, unless the affine to use
    is explictly specified with the option `affine="sform"`,
    `affine="qform"`, or `affine="base"`.
    """

    def __init__(self, *args, **kwargs) -> None:
        """
        Parameters
        ----------
        json_data : dict | NiftiDataSource | str, optional
            URL to the file

        Other Parameters
        ----------------
        url : str
            URL to the file
        align_corner : bool, default=False
            If True, use neuroglancer's native behavior when computing
            the transform, which is to asume that (0, 0, 0) points to
            the corner of the first voxel. If False, use NIfTI's spec,
            which is to assume the (0, 0, 0) points to the center of the
            first voxel.
        affine : {"qform", "sform", "best", "base"}
            Which orientation matrix to use.
            Neuroglancer uses the qform by default.
        """
        self._align_corner = kwargs.pop("align_corner", False)
        self._select_affine = kwargs.pop("affine", "best")
        self._nib_image = None
        self._stream = None
        super().__init__(*args, **kwargs)
        # Trigger transform computation because we do not trust
        # neuroglancer's behaviour completely.
        self.transform = self.transform

    def _compute_info(self) -> NiftiVolumeInfo:
        return NiftiVolumeInfo(
            self.url,
            align_corner=self._align_corner,
            affine=self._select_affine,
        )

    def _load_image(self, mode: str = "r") -> nib.nifti1.Nifti1Image:
        tic = time.time()
        NiftiImages = (nib.nifti1.Nifti1Image, nib.nifti2.Nifti2Image)

        url = parse_protocols(self.url).url
        for compression in ('infer', 'gzip', None):
            self._stream = open(url, compression=compression)
            for img_klass in NiftiImages:
                try:
                    hdr = img_klass.from_stream(self._stream)
                    toc = time.time()
                    LOG.info(f"Loaded nifti file: {toc-tic} s")
                    return hdr
                except Exception:
                    pass
            self._stream = None
        toc = time.time()
        LOG.error(f"Failed to load nifti file {url}: {toc-tic} s")
        raise RuntimeError(f"Failed to load nifti file {url}.")

    def __del__(self) -> None:
        try:
            self._stream.close()
        except Exception:
            pass

    def _get_dataobj(self, *a, **k) -> ArrayLike:
        # NOTE: we return the raw (unscaled) data, like neuroglancer
        if getattr(self, "_nib_image", None) is None:
            self._nib_image = self._load_image()
        return self._nib_image.dataobj


class ZarrVolumeInfo(VolumeInfo):
    """Base class + common methods for Zarr metadata."""

    nifti: NiftiVolumeInfo | None

    def getShape(self, level: int = 0) -> list[int]:
        """Array shape at a given level."""
        return self._zarray[level]["shape"]

    def getNbLevels(self) -> int:
        """Return the number of levels in the pyramid."""
        try:
            return len(self.getMultiscalesOME()[0]["datasets"])
        except Exception:
            return 1

    def hasOME(self) -> bool:
        """Return `True` if Zarr has OME metadata."""
        return "multiscales" in self._zattrs or "ome" in self._zattrs

    def hasNIfTI(self) -> bool:
        """Return `True` if Zarr has NIfTI metadata."""
        return self.nifti is not None

    def getMultiscalesOME(self) -> list[dict]:
        """Return content of OME metadata (JSON)."""
        if "multiscales" in self._zattrs:
            return self._zattrs["multiscales"]
        else:
            return self._zattrs["ome"]["multiscales"]

    def getAxesOME(self) -> list[dict] | None:
        """Return OME axes (JSON)."""
        try:
            return self.getMultiscalesOME()[0]["axes"]
        except Exception:
            return None

    def getDatasetOME(self, level: int = 0) -> dict | None:
        """Return OME dataset (JSON)."""
        try:
            ome = self.getMultiscalesOME()[0]
            return ome["datasets"][level]
        except Exception:
            return None

    def getNames(self) -> list[str]:
        """Return axis names."""
        axes = self.getAxesOME()
        if axes is None:
            if isinstance(self, Zarr2VolumeInfo):
                return [f"d{i}" for i in range(self.getRank())]
            else:
                assert isinstance(self, Zarr3VolumeInfo)
                return [f"dim{i}" for i in range(self.getRank())]
        return [
            axis["name"] + ("'" if axis["type"] == "channel" else "")
            for axis in axes
        ]

    getInputNames = getOutputNames = getNames

    def getAxisTypes(self) -> list[str]:
        """Return axis type (space, time, channel)."""
        axes = self.getAxesOME()
        if axes is None:
            return [""] * self.getRank()
        return [axis["type"] for axis in axes]

    def getUnits(self) -> list[str]:
        """Return units."""
        axes = self.getAxesOME()
        if axes is None:
            return [""] * self.getRank()
        return [
            U.as_neuroglancer_unit(axis.get("unit", ""))
            for axis in axes
        ]

    getInputUnits = getOutputUnits = getUnits

    def getScales(self) -> list[float]:
        """Return scales."""
        if getattr(self, "_scales", None) is not None:
            return list(self._scales)

        scale = [1.0] * self.getRank()
        # Read common scale (applies to time)
        try:
            for trf in self.getMultiscalesOME()["coordinateTransformations"]:
                if "scale" in trf:
                    scale = trf["scale"]
                    break
        except Exception:
            pass
        # Read level 0 scale (applies to space)
        try:
            level0 = self.getDatasetOME()
            for transform in level0.get("coordinateTransformations", []):
                if "scale" in transform:
                    scale = [
                        scale1 if scale1 != 1 else scale0
                        for scale0, scale1 in zip(scale, transform["scale"])
                    ]
                    break
        except Exception:
            pass
        self._scales = scale
        return list(scale)

    getInputScales = getOutputScales = getScales

    def getTranslations(self) -> list[float]:
        """
        Return translations (in spatial units).

        Although OME-NGFF v0.4 requires translations to appear after
        scales so that translations are expressed in world units,
        neuroglancer accepts transformations that have the opposite
        order, i.e., where translations are expressed in voxels. We also
        handle this case and rescale translations to world units.
        """
        if getattr(self, "_translations", None) is not None:
            return list(self._translations)

        trans = [0.0] * self.getRank()
        # Read common scale (applies to time)
        try:
            root = self.getMultiscalesOME()[0]
            for trf in root["coordinateTransformations"]:
                if "scale" in trf:
                    scale = trf["scale"]
                    trans = [t0*s for t0, s in zip(trans, scale)]
                elif "translation" in trf:
                    trans = [t0+t for t0, t in zip(trans, trf["translation"])]
        except Exception:
            pass
        # Read level 0 scale (applies to space)
        try:
            level0 = self.getDatasetOME()
            for trf in level0.get("coordinateTransformations", []):
                if "scale" in trf:
                    scale = trf["scale"]
                    trans = [t0*s for t0, s in zip(trans, scale)]
                elif "translation" in trf:
                    trans = [t0+t for t0, t in zip(trans, trf["translation"])]
        except Exception:
            pass
        # OME defines translations with respect to the center of the corner
        # voxel, but neuroglancer defines them with respect to its corner.
        # We must therefore introduce an extra shift. See:
        # https://github.com/google/neuroglancer/blob/master/src/datasource/zarr/ome.ts#L244
        if self.hasOME:
            scales = self.getScales()
            types = self.getAxisTypes()
            trans = [
                t - 0.5 * (s if type == "space" else 0)
                for t, s, type in zip(trans, scales, types)
            ]
        self._translations = trans
        return list(trans)

    def getMatrix(self) -> np.ndarray:
        """Return transformation matrix."""
        if getattr(self, "_matrix", None) is None:
            matrix = np.eye(self.getRank()+1)[:-1]
            matrix[:, -1] = self.getTranslations()
            self._matrix = matrix
        return self._matrix

    def getZarrTransform(self) -> ng.CoordinateSpaceTransform:
        """Return the pure zarr transform."""
        return ng.CoordinateSpaceTransform(
            matrix=self.getMatrix(),
            input_dimensions=self.getInputDimensions(),
            output_dimensions=self.getOutputDimensions(),
        )

    def getNiftiInputDimensions(self) -> ng.CoordinateSpace:
        """Return nifti dimensions, but using OME axis names."""
        names = list(reversed(self.getNames()))
        units = self.nifti.getUnits()
        scales = self.nifti.getInputScales()
        return ng.CoordinateSpace({
            name: [scale, unit]
            for name, scale, unit in zip(names, scales, units)
        })

    def getNiftiOutputDimensions(self) -> ng.CoordinateSpace:
        """Return nifti output dimensions."""
        return self.nifti.getOutputDimensions()

    def getNiftiMatrix(self) -> np.ndarray:
        """Return nifti transformation matrix."""
        return self.nifti.getMatrix()

    def getNiftiTransform(self) -> ng.CoordinateSpaceTransform:
        """Return nifti transform."""
        return ng.CoordinateSpaceTransform(
            matrix=self.getNiftiMatrix(),
            input_dimensions=self.getNiftiInputDimensions(),
            output_dimensions=self.getNiftiOutputDimensions(),
        )

    def getNiftiZarrTransform(self) -> ng.CoordinateSpaceTransform:
        """
        Return the transform to apply to a Zarr layer so that it's
        oriented according to the nifti metadata.
        """
        return T.compose(
            self.getNiftiTransform(),
            ng.CoordinateSpaceTransform(
                input_dimensions=self.getInputDimensions(),
                output_dimensions=self.getInputDimensions(),
            ),
        )

    def getTransform(self) -> ng.CoordinateSpaceTransform:
        """
        Return the nifti-zarr transform if it exists, else the ome-zarr
        transform.
        """
        return (
            self.getNiftiZarrTransform() if self.hasNIfTI() else
            self.getZarrTransform()
        )


class Zarr2VolumeInfo(ZarrVolumeInfo):
    """Volume info for Zarr v2."""

    def __init__(self, url: str, nifti: bool | None = None) -> None:
        super().__init__(url)
        url = UPath(parse_protocols(self.url).url)

        if exists(url / ".zgroup"):
            is_group = True
        elif exists(url / ".zarray"):
            is_group = False
            self._zarray = [read_json(url / ".zarray")]
        else:
            raise ValueError("Not a zarr")
        if exists(url / ".zattrs"):
            self._zattrs = read_json(url / ".zattrs")
        else:
            self._zattrs = {}

        if is_group:
            self._zarray = []
            try:
                zattrs = self._zattrs
                if "ome" in self._zattrs:
                    zattrs = zattrs["ome"]
                datasets = zattrs["multiscales"][0]["datasets"]
                for dataset in datasets:
                    path = dataset["path"]
                    zpath = url / path / ".zarray"
                    if exists(zpath):
                        self._zarray.append(read_json(zpath))
                    else:
                        raise ValueError("Missing array in OME zarr:", path)
            except Exception:
                raise ValueError("Missing or invalid OME metadata.")

        self.nifti = None
        if nifti is not False:
            if exists(url / "nifti" / ".zarray"):
                url = url / "nifti" / "0"
                try:
                    self.nifti = NiftiVolumeInfo(
                        url, affine="best", align_corner=True
                    )
                except Exception:
                    if nifti is True:
                        raise
            elif nifti is True:
                raise FileNotFoundError("Cannot find nifti group in zarr.")

    def getDataType(self) -> np.dtype:
        """Array shape at a given level."""
        return np.dtype(self._zarray[0]["dtype"])


class Zarr3VolumeInfo(ZarrVolumeInfo):
    """Volume info for Zarr v3."""

    def __init__(self, url: str, nifti: bool | None = None) -> None:
        super().__init__(self)
        url = UPath(parse_protocols(self.url).url)

        if not exists(url / "zarr.json"):
            raise ValueError("Not a zarr")

        zbase = read_json(url / "zarr.json")
        self._zattrs = zbase.get("attributes", {})
        if zbase["node_type"] == "group":
            is_group = True
        else:
            assert zbase["node_type"] == "array"
            is_group = False
            self._zarray = zbase

        if is_group:
            try:
                self._zarray = []
                zattrs = self._zattrs
                if "ome" in self._zattrs:
                    zattrs = zattrs["ome"]
                datasets = zattrs["multiscales"][0]["datasets"]
                for dataset in datasets:
                    path = dataset["path"]
                    zpath = url / path / "zarr.json"
                    if exists(zpath):
                        self._zarray.append(read_json(zpath))
                    else:
                        raise ValueError("Missing array in OME zarr:", path)
            except Exception:
                raise ValueError("Missing or invalid OME metadata.")

        self.nifti = None
        if nifti is not False:
            if exists(url / "nifti" / "zarr.json"):
                url = url / "nifti" / "c0"
                self.nifti = NiftiVolumeInfo(
                    url, affine="best", align_corner=True
                )
            elif nifti is True:
                raise FileNotFoundError("Cannot find nifti group in zarr.")

    def getDataType(self) -> np.dtype:
        """Array shape at a given level."""
        return np.dtype(self._zarray[0]["data_type"])


class _ZarrDataSourceFactory(_LayerDataSourceFactory):
    _DataSourceLike = _LayerDataSourceFactory._DataSourceLike

    def __call__(
            cls, arg: _DataSourceLike = None, *args, **kwargs
    ) -> "ZarrDataSource":
        # Only use the factory if it is not called from a subclass
        if cls is not ZarrDataSource:
            obj = super().__call__(arg, *args, **kwargs)
            return obj

        # Get url
        if kwargs.get("url", ""):
            url = kwargs["url"]
            if isinstance(url, (str, PathLike)):
                url = kwargs["url"] = str(url)
        elif isinstance(arg, (str, PathLike)):
            url = arg = str(arg)
        elif hasattr(arg, "url"):
            url = arg.url
        elif isinstance(arg, dict) and "url" in arg:
            url = arg["url"]
        elif isinstance(arg, cls._LocalSource):
            raise TypeError(f"Cannot convert {type(arg)} to ZarrDataSource")
        else:
            raise ValueError("Missing data source url")

        format = parse_protocols(url).format
        LOG.debug(f"ZarrDataSource - hint: {format}")

        if format == "zarr2":
            LOG.debug("ZarrDataSource -> Zarr2DataSource")
            return Zarr2DataSource(arg, *args, **kwargs)

        if format == "zarr3":
            LOG.debug("ZarrDataSource -> Zarr3DataSource")
            return Zarr3DataSource(arg, *args, **kwargs)

        LOG.debug("ZarrDataSource - guess version...")
        version = cls.guess_version(url)
        LOG.debug(f"ZarrDataSource - guess version: {version}")

        if version == 2:
            LOG.debug("ZarrDataSource -> Zarr2DataSource")
            return Zarr2DataSource(arg, *args, **kwargs)

        elif version == 3:
            LOG.debug("ZarrDataSource -> Zarr3DataSource")
            return Zarr3DataSource(arg, *args, **kwargs)

        LOG.debug("ZarrDataSource - fallback to super")
        return super().__call__(arg, *args, **kwargs)


@datasource(["zarr", "zarr2", "zarr3"])
class ZarrDataSource(VolumeDataSource, metaclass=_ZarrDataSourceFactory):
    """Wrapper for Zarr data sources.

    This wrapper will automatically check for the presence of a `nifti`
    group in the zarr object, which indicates that the file really has
    the nifti-zarr format.

    This behavior can be changed by setting the option `nifti=False`
    (to load the file as a pure OME-Zarr) or `nifti=True` (which raises
    an error if the nifti group is not found).
    """

    info: ZarrVolumeInfo

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._nifti = kwargs.pop('nifti', None)
        self._align_corner = kwargs.pop("align_corner", False)
        # trigger computation of the transform, in case it's a nifti-zarr
        # TODO: check if nifti zarr first.
        self.transform = self.transform

    @classmethod
    def guess_version(cls, url: str) -> int:
        """Guess zarr version."""
        url = UPath(parse_protocols(url).url)
        if exists(url / ".zgroup"):
            return 2
        if exists(url / ".zarray"):
            return 2
        if exists(url / "zarr.json"):
            return 3
        return 0

    def _compute_info(self) -> Zarr2VolumeInfo:
        version = self.guess_version(self.url)
        if version == 2:
            return Zarr2VolumeInfo(self.url)
        elif version == 3:
            return Zarr3VolumeInfo(self.url)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _get_dataobj(self, level: int = 0, mode: str = "r") -> ArrayLike:
        """Return an array-like object pointing to a pyramid level."""
        tic = time.time()
        url = parse_protocols(self.url).url
        fs = filesystem(url)
        store = zarr.storage.FSStore(url, fs=fs, mode=mode)
        if not self.info.hasOME():
            dataobj = zarr.open(store, mode)
        else:
            path = self.info.getDatasetOME(level)["path"]
            dataobj = zarr.open(store, mode, path=path)
        toc = time.time()
        LOG.info(f"Map zarr: {toc - tic} s")
        return dataobj


@datasource("zarr2")
class Zarr2DataSource(ZarrDataSource):
    """Wrapper for Zarr v2 data sources."""

    def _compute_info(self) -> Zarr2VolumeInfo:
        return Zarr2VolumeInfo(self.url)


@datasource("zarr3")
class Zarr3DataSource(ZarrDataSource):
    """Wrapper for Zarr v3 data sources."""

    def _compute_info(self) -> Zarr3VolumeInfo:
        return Zarr3VolumeInfo(self.url)


class N5VolumeInfo(VolumeInfo):
    """Base class + common methods for N5 metadata."""

    def __init__(self, url: str) -> None:
        super().__init__(url)
        url = UPath(parse_protocols(self.url).url)
        if exists(url / "attributes.json"):
            self._attributes = read_json(url / "attributes.json")
        else:
            self._attributes = {}

        if "dimensions" in self._attributes:
            self._is_multiscale = False
            self._arrays = [self._attributes]
            self._attributes = {}
        else:
            self._is_multiscale = True

        if self._is_multiscale:
            self._arrays = []
            for i in range(self.getNbLevels()):
                url_i = url / f"s{i}" / "attributes.json"
                if exists(url_i):
                    self._arrays.append(read_json(url_i))
                else:
                    self._arrays.append({})

    def getDataType(self) -> np.dtype:
        """Array data type."""
        try:
            return np.dtype(self._attributes["dataType"])
        except Exception:
            return np.dtype(self._arrays[0]["dataType"])

    def getShape(self, level: int = 0) -> list[int]:
        """Array shape at a given level."""
        return list(self._arrays[level]["dimensions"])

    def getNbLevels(self) -> int:
        """Return the number of levels in the pyramid."""
        try:
            return len(self._attributes["downsamplingFactors"])
        except Exception:
            return 1

    def getNames(self) -> list[str]:
        """Return axis names."""
        rank = self.getRank()
        try:
            names = list(self._attributes["axes"])
        except Exception:
            names = list(self._arrays[0]["axes"])
        if len(names) != rank:
            raise ValueError(f"Rank mismatch: {len(names)} != {rank}")
        return names

    getInputNames = getOutputNames = getNames

    def getUnits(self) -> list[str]:
        """Return units."""
        rank = self.getRank()
        units = ["nm"] * self.getRank()
        attrs = [self._attributes] + self._arrays[:1]
        for attr in attrs:
            if "units" in attr:
                units = list(attr["units"])
                break
            elif "pixelResolution" in attr:
                unit = attr["pixelResolution"].get("unit", "")
                unit = unit or "nm"
                units = [unit] * rank
                break
        if len(units) != rank:
            raise ValueError(f"Rank mismatch: {len(units)} != {rank}")
        return units

    getInputUnits = getOutputUnits = getUnits

    def getScales(self) -> list[float]:
        """Return scales."""
        rank = self.getRank()
        scales = [1.0] * rank
        attrs = [self._attributes] + self._arrays[:1]
        for attr in attrs:
            if "resolution" in attr:
                scales = list(attr["resolution"])
                break
            elif "pixelResolution" in attr:
                scales = list(attr["pixelResolution"]["dimensions"])
                break
        if len(scales) != rank:
            raise ValueError(f"Rank mismatch: {len(scales)} != {rank}")
        return scales

    getInputScales = getOutputScales = getScales

    def getTranslations(self) -> list[float]:
        """Return translations (in spatial units)."""
        return [0.0] * self.getRank()


@datasource(["n5"])
class N5DataSource(VolumeDataSource):
    """Wrapper for N5 data sources."""

    info: N5VolumeInfo

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def _compute_info(self) -> Zarr2VolumeInfo:
        return N5VolumeInfo(self.url)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _get_dataobj(self, level: int = 0, mode: str = "r") -> ArrayLike:
        """Return an array-like object pointing to a pyramid level."""
        tic = time.time()
        url = parse_protocols(self.url).url
        fs = filesystem(url)

        try:
            from zarr.n5 import N5FSStore
        except (ImportError, ModuleNotFoundError):
            if int(zarr.__version__.split(".")[0]) > 2:
                raise ImportError(
                    f"N5 is only available in zarr v2, but you have "
                    f"v{zarr.__version__} installed.")

        store = N5FSStore(url, fs=fs, mode=mode)
        path = str(level)
        if path[:1] != "s":
            path = "s" + path
        dataobj = dataobj = zarr.open(store, mode, path=path)
        toc = time.time()
        LOG.info(f"Map n5: {toc - tic} s")
        return dataobj


class _PrecomputedInfoFactory(type):
    def __call__(cls, url: str | dict) -> "PrecomputedInfo":
        info = cls._load_dict(url)
        return {
            "neuroglancer_multiscale_volume": PrecomputedVolumeInfo,
            "neuroglancer_multilod_draco": PrecomputedMeshInfo,
            "neuroglancer_legacy_mesh": PrecomputedLegacyMeshInfo,
            "neuroglancer_skeletons": PrecomputedSkeletonInfo,
            "neuroglancer_annotations_v1": PrecomputedAnnotationInfo,
        }[info["@type"]](info)


class PrecomputedInfo(metaclass=_PrecomputedInfoFactory):
    """Base class for metadate about precomputed objects."""

    def __init__(self, url: str | dict) -> None:
        self._info = self._load_dict(url)

    @classmethod
    def _load_dict(cls, url: str | dict) -> dict:
        if not isinstance(url, dict):
            if url.startswith("precomputed://"):
                url = url[14:]
            with open(url, "rb") as f:
                info = json.load(f)
        else:
            info = url
        return info


class PrecomputedMeshInfo(PrecomputedInfo):
    """Metadata of a precomputed mesh."""

    ...


class PrecomputedLegacyMeshInfo(PrecomputedInfo):
    """Metadata of a precomputed mesh (legacy)."""

    ...


class PrecomputedSkeletonInfo(PrecomputedInfo):
    """Metadata of a precomputed skeleton."""

    ...


class PrecomputedAnnotationInfo(PrecomputedInfo):
    """Metadata of a precomputed annotation."""

    ...


class PrecomputedVolumeInfo(PrecomputedInfo):
    """Metadata of a precomputed volume."""

    def getDataType(self) -> np.ndarray:
        """TODO."""
        return np.dtype(self._info["data_type"])

    def getShape(self, level: int = 0) -> list[int]:
        """TODO."""
        if level < 0:
            level = self.getNbLevels() + level
        if level < 0 or level >= self.getNbLevels():
            raise IndexError("Pyramid level not available:", level)

        shape = list(self._info["scales"][level]["size"])
        if self._info["num_channels"] > 1:
            shape += [self._info["num_channels"]]
        return shape

    def getRank(self) -> int:
        """TODO."""
        return 4 if self._info["num_channels"] > 1 else 3

    def getScales(self) -> list[float]:
        """TODO."""
        scales = list(self._info["scales"][0]["resolution"])
        scales += max(0, self.getRank()-3) * [1.0]
        return scales

    def getUnits(self) -> list[str]:
        """TODO."""
        return ["nm"] * 3 + max(0, self.getRank()-3) * [""]

    def getNames(self) -> list[str]:
        """TODO."""
        names = ["x", "y", "z"]
        if self.getRank() == 4:
            names += ["c"]
        return names

    def getDimensions(self) -> ng.CoordinateSpace:
        """TODO."""
        names = self.getNames()
        scales = self.getScales()
        units = self.getUnits()
        return ng.CoordinateSpace({
            name: [scale, unit]
            for name, scale, unit in zip(names, scales, units)
        })

    def getTransform(self) -> ng.CoordinateSpaceTransform:
        """TODO."""
        dimensions = self.getDimensions()
        return ng.CoordinateSpaceTransform(
            matrix=np.eye(self.getRank()+1)[:-1],
            input_dimensions=dimensions,
            output_dimensions=dimensions,
        )

    def getNbLevels(self) -> int:
        """TODO."""
        return len(self._info["scales"])

    def getVoxelOffset(self) -> list[int]:
        """TODO."""
        return list(self._info["voxel_offset"])

    def getChunkSize(self, level: int = 0, choice: int = 0) -> list[int]:
        """TODO."""
        return self._info["scales"][level]["chunk_sizes"][choice]


class _PrecomputedDataSourceFactory(_LayerDataSourceFactory):
    def __call__(cls, url: str | dict) -> "PrecomputedDataSource":
        info = cls._load_dict(url)
        return {
            "neuroglancer_multiscale_volume": PrecomputedVolumeDataSource,
            "neuroglancer_multilod_draco": PrecomputedMeshDataSource,
            "neuroglancer_legacy_mesh": PrecomputedLegacyMeshDataSource,
            "neuroglancer_skeletons": PrecomputedSkeletonDataSource,
            "neuroglancer_annotations_v1": PrecomputedAnnotationDataSource,
        }[info["@type"]](info)


@datasource("precomputed")
class PrecomputedDataSource(LayerDataSource,
                            metaclass=_PrecomputedDataSourceFactory):
    """Base wrapper for precomputed data sources."""

    ...


class PrecomputedMeshDataSource(MeshDataSource, PrecomputedDataSource):
    """Base wrapper for precomputed mesh sources."""

    ...


class PrecomputedLegacyMeshDataSource(MeshDataSource, PrecomputedDataSource):
    """Base wrapper for precomputed mesh (legacy) sources."""

    ...


class PrecomputedSkeletonDataSource(SkeletonDataSource, PrecomputedDataSource):
    """Base wrapper for precomputed skeleton sources."""

    ...


class PrecomputedAnnotationDataSource(
    AnnotationDataSource, PrecomputedDataSource
):
    """Base wrapper for precomputed annotations sources."""

    ...


class PrecomputedVolumeDataSource(VolumeDataSource, PrecomputedDataSource):
    """Wrapper for precomputed volumes."""

    def _compute_info(self) -> PrecomputedVolumeInfo:
        return PrecomputedVolumeInfo(self.url)

    def _get_dataobj(self, level: int = 0, mode: str = "r") -> ArrayLike:
        url = parse_protocols(self.url).url
        return CloudVolume(url, mip=level)
