"""
Wrappers around neuroglancer classes that can compute metadata and
optional attributes that neuroglancer typically delegates to the frontend.
"""
# stdlib
import json
from itertools import product
from os import PathLike
from typing import Callable, Iterator

# externals
import neuroglancer as ng
import nibabel as nib
import numpy as np
import zarr
import zarr.storage
from cloudvolume import CloudVolume
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

_DATASOURCE_REGISTRY = {}


def _quantiles(q: float | list[float], data: ArrayLike) -> np.ndarray[float]:
    """Compute (efficiently) intensity quantiles."""
    steps = [max(1, x//64) for x in data.shape]
    slicer = (Ellipsis,) + tuple(slice(None, None, step) for step in steps)
    return np.quantile(np.asarray(data[slicer]), q)


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


class LayerDataSources(ng.LayerDataSources):
    """List of data sources."""

    def __init__(self, other: list[ng.LayerDataSource]) -> None:
        if not isinstance(other, (list, tuple, ng.LayerDataSources)):
            other = [other]
        other = list(map(LayerDataSource, other))
        super().__init__(other)

    def append(self, x: ng.LayerDataSource) -> None:  # noqa: D102
        super().append(LayerDataSource(x))

    def extend(self, other: list[ng.LayerDataSource]) -> None:  # noqa: D102
        super().extend(map(LayerDataSource, other))

    def __setitem__(self, k: int, v: ng.LayerDataSource) -> None:
        super().__setitem__(k, LayerDataSource(v))

    def __getitem__(self, *a, **k) -> "LayerDataSource":
        return LayerDataSource(super().__getitem__(*a, **k))

    def __iter__(self) -> Iterator["LayerDataSource"]:
        for source in super().__iter__():
            yield LayerDataSource(source)

    def pop(self, *a, **k) -> ng.LayerDataSource:  # noqa: D102
        return LayerDataSource(super().pop(*a, **k))


class _LayerDataSourceFactory(type):
    """Factory for LayerDataSource objects."""

    def __call__(
        cls,
        json_data: dict | ng.LayerDataSource | None = None,
        *args,
        **kwargs
    ) -> "LayerDataSource":
        # If (single) input already a LayerDataSource, return it as is.
        #   LayerDataSource(inp: LayerDataSource) -> LayerDataSource
        if not args and not kwargs:
            if isinstance(json_data, cls) and cls is not LayerDataSource:
                return json_data
        # only use the factory if it is not called from a subclass
        if isinstance(json_data, ng.LayerDataSource):
            json_data = json_data.to_json()
        if cls is not LayerDataSource:
            obj = super().__call__(json_data, *args, **kwargs)
            return obj
        # Use  ng.LayerDataSource to get url
        #   (deals with json_data, keywords, local volumes, etc.)
        url = ng.LayerDataSource(json_data, *args, **kwargs).url
        # Local objects -> delegate
        if isinstance(url, ng.LocalVolume):
            return LocalVolumeDataSource(json_data, *args, **kwargs)
        if isinstance(url, ng.skeleton.SkeletonSource):
            return LocalSkeletonDataSource(json_data, *args, **kwargs)
        # If format protocol provided, use it
        format = parse_protocols(url)[1]
        if format in _DATASOURCE_REGISTRY:
            return _DATASOURCE_REGISTRY[format](json_data, *args, **kwargs)
        # Otherwise. build a simple source
        return super().__call__(json_data, *args, **kwargs)


class DataSourceInfo:
    """Base class for source-specific metadata."""

    ...


class LayerDataSource(ng.LayerDataSource, metaclass=_LayerDataSourceFactory):
    """A wrapper around `ng.LayerDataSource` that computes metadata."""

    PROTOCOLS: list[str]

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._format_specific_init()

    def _format_specific_init(self) -> None:
        """Additional format-specific initialization."""
        ...

    def _compute_info(self) -> DataSourceInfo:
        """Compute format-specific metadata."""
        ...

    @property
    def info(self) -> DataSourceInfo:
        """Lazy access to format-specific metadata."""
        if not getattr(self, '_info', None):
            self._info = self._compute_info()
        return self._info

    @classmethod
    def _raise_not_implemented_error(cls, name: str) -> None:
        raise NotImplementedError(
            f"`{name}` not implemented for this format ({cls.__name__})."
        )

    @property
    def url(self) -> str:  # noqa: D102
        return super().url

    @url.setter
    def url(self, value: str) -> str:
        format = parse_protocols(value)[1]
        if format not in self.PROTOCOLS:
            raise ValueError(
                "Cannot assign uri with protocol", format, "in "
                "data source of type:", type(self)
            )
        ng.LayerDataSource.url.fset(self, value)
        self._format_specific_init()

    @property
    def local_url(self) -> str:
        """Path to url, even if file is local."""
        url = self.url
        if isinstance(url, ng.LocalVolume, ng.skeleton.SkeletonSource):
            if hasattr(self, '_url'):
                return self._url
            if hasattr(url, 'url'):
                return url.url
            return None
        return url

    @property
    def transform(self) -> ng.CoordinateSpaceTransform:
        """
        If a transform has been explicitely set, return it,
        else, return the transform implcitely defined by neuroglancer.
        """
        assigned_trf = super().transform
        implicit_trf = getattr(self, '_transform', None)
        if not assigned_trf:
            return implicit_trf
        elif not implicit_trf:
            return assigned_trf
        else:
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
            return ng.CoordinateSpaceTransform(
                matrix=matrix,
                input_dimensions=idims,
                output_dimensions=odims,
            )

    @transform.setter
    def transform(self, value: ng.CoordinateSpaceTransform | dict) -> None:
        ng.LayerDataSource.transform.fset(self, value)

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
        self._raise_not_implemented_error("get_dataobj")

    _IndexType = int | bool | slice | ArrayLike | type(Ellipsis)
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

    def apply_transform(self, transform: ng.CoordinateSpaceTransform) -> None:
        """Apply an additional transform in model space."""
        self.transform = T.compose(transform, self.transform)

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
        scales = self.input_dimensions.scales
        min = [0.0] * self.rank
        max = [x * scl for x, scl in zip(shape, scales)]
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
                output_dimensions=self.output_dimensions,
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
        return ((bbox[0] + bbox[1]) / 2). tolist()

    @property
    def output_center(self) -> list[float]:
        """Center of the field of view in output dimensions space and units."""
        center = self.input_center
        mat = np.eye(self.rank+1)[:-1]
        mat[:, -1] = center
        center = ng.CoordinateSpaceTransform(
            input_dimensions=self.input_dimensions,
            output_dimensions=self.output_dimensions,
            matrix=mat,
        )
        center = T.compose(self.transform, center)
        center = center.matrix[:, -1]
        return center


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

    ...


class LocalSkeletonDataSource(VolumeDataSource, LocalDataSource):
    """Wrapper for data source that wraps a `SkeletonSource`."""

    @property
    def local_skeleton(self) -> ng.LocalVolume | None:
        """Points to the underlying `SkeletonSource`, if any."""
        url = super().url
        if isinstance(self, ng.skeleton.SkeletonSource):
            return url
        return None

    ...


class LocalVolumeDataSource(SkeletonDataSource, LocalDataSource):
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
        affine: str = "qform",
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
        """
        super().__init__(url)
        self._align_corner = align_corner
        self._affine = affine
        self._nib_header: nib.nifti1.Nifti1Header = self._load()

    def _load(self) -> nib.nifti1.Nifti1Header | nib.nifti2.Nifti2Header:
        NiftiHeaders = (nib.nifti1.Nifti1Header, nib.nifti2.Nifti2Header)
        url = parse_protocols(self.url)[-1]
        for compression in ('infer', 'gzip', None):
            fileobj = open(url, compression=compression)
            for image_klass in NiftiHeaders:
                try:
                    return image_klass.from_fileobj(fileobj)
                except Exception:
                    pass
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
                unit = "s"
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
            if self._affine == "base":
                matrix = self._nib_header.get_base_affine()[:-1]
            elif self._affine == "best":
                matrix = self._nib_header.get_best_affine()[:-1]
            elif self._affine == "sform":
                matrix = self._nib_header.get_sform()[:-1]
            else:
                matrix = self._nib_header.get_qform()[:-1]
            scales = self._nib_header['pixdim'][1:4]
            if not self._align_corner:
                matrix[:3, -1] -= matrix[:3, :-1] @ ([0.5]*srank)
            matrix[:3, :-1] /= scales
            fullmatrix = np.eye(self.getRank()+1)[:-1]
            fullmatrix[:srank, :srank] = matrix[:srank, :srank]
            fullmatrix[:srank, -1] = matrix[:srank, -1]
            self._matrix = fullmatrix
        return self._matrix


@datasource("nifti")
class NiftiDataSource(VolumeDataSource):
    """
    Wrapper for nifti sources.

    Note that by default, neuroglancer assumes that the coordinate
    (0, 0, 0) in voxel space corresponds to the _corner_ of the first
    voxel, whereas the nifti spec is that it corresponds to the _center_
    of the first voxel. This wrapper follows the nifti convention by default.
    The neuroglancer default behavior can be recovered with the option
    `align_corner=True`.
    """

    def __init__(self, *args, **kwargs) -> None:
        self._align_corner = kwargs.pop("align_corner", False)
        self._select_affine = kwargs.pop("affine", False)
        super().__init__(*args, **kwargs)

    def _compute_info(self) -> NiftiVolumeInfo:
        return NiftiVolumeInfo(
            self.url,
            align_corner=self._align_corner,
            affine=self._select_affine,
        )

    def _format_specific_init(self) -> None:
        self._nib_image = None
        self._stream = None

    def _load_image(self, mode: str = "r") -> nib.nifti1.Nifti1Image:
        url = self.url[8:]
        self._stream = open(url, compression='infer', mode=mode)
        for image_klass in (nib.nifti1.Nifti1Image, nib.nifti2.Nifti2Image):
            try:
                return image_klass.from_stream(self._stream)
            except Exception:
                pass
        raise RuntimeError("Failed to load file.")

    def __del__(self) -> None:
        try:
            self._stream.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _get_dataobj(self, level: int = 0, mode: str = "r") -> ArrayLike:
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
        return [axis["name"] for axis in axes]

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
        url = UPath(parse_protocols(self.url)[-1])

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
            try:
                self._zarray = []
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
                self.nifti = NiftiVolumeInfo(url, affine="best")
            elif nifti is True:
                raise FileNotFoundError("Cannot find nifti group in zarr.")

    def getDataType(self) -> np.dtype:
        """Array shape at a given level."""
        return np.dtype(self._zarray[0]["dtype"])


class Zarr3VolumeInfo(ZarrVolumeInfo):
    """Volume info for Zarr v3."""

    def __init__(self, url: str, nifti: bool | None = None) -> None:
        super().__init__(self)
        url = UPath(parse_protocols(self.url)[-1])

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
                self.nifti = NiftiVolumeInfo(url, affine="best")
            elif nifti is True:
                raise FileNotFoundError("Cannot find nifti group in zarr.")

    def getDataType(self) -> np.dtype:
        """Array shape at a given level."""
        return np.dtype(self._zarray[0]["data_type"])


class _ZarrDataSourceFactory(_LayerDataSourceFactory):
    def __call__(cls, *args, **kwargs) -> "ZarrDataSource":
        # If (single) input already a LayerDataSource, return it as is.
        #   LayerDataSource(inp: LayerDataSource) -> LayerDataSource
        if len(args) == 1 and not kwargs:
            if isinstance(args[0], cls) and cls is not ZarrDataSource:
                return args[0]
        # only use the factory if it is not called from a subclass
        if cls is not ZarrDataSource:
            return super().__call__(*args, **kwargs)
        url = ng.LayerDataSource(*args, **kwargs).url
        format = parse_protocols(url)[1]
        if format == "zarr2":
            return Zarr2DataSource(*args, **kwargs)
        if format == "zarr3":
            return Zarr3DataSource(*args, **kwargs)
        version = cls.guess_version(url)
        if version == 2:
            return Zarr2DataSource(*args, **kwargs)
        elif version == 3:
            return Zarr3DataSource(*args, **kwargs)
        raise ValueError("Unsupporter zarr version:", version)


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
        self._nifti = kwargs.pop('nifti', None)
        self._align_corner = kwargs.pop("align_corner", False)
        super().__init__(*args, **kwargs)

    @classmethod
    def guess_version(cls, url: str) -> int:
        """Guess zarr version."""
        url = UPath(parse_protocols(url)[-1])
        if exists(url / "zarr.json"):
            return 3
        if exists(url / ".zarray"):
            return 2
        if exists(url / ".zgroup"):
            return 2
        return 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _get_dataobj(self, level: int = 0, mode: str = "r") -> ArrayLike:
        """Return an array-like object pointing to a pyramid level."""
        url = parse_protocols(self.url)[-1]
        fs = filesystem(url)
        store = zarr.storage.FSStore(url, fs=fs, mode=mode)
        if not self.info.hasOME():
            return zarr.open(store, mode)
        else:
            path = self.info.getDatasetOME(level)["path"]
            return zarr.open(store, mode, path=path)


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
        url = parse_protocols(self.url)[-1]
        return CloudVolume(url, mip=level)
