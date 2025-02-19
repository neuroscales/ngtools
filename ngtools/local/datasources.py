"""Objects that represent local multidimensional arrays."""
# stdlib
import os
from os import PathLike
from typing import IO
from xml.etree import ElementTree as etree

# externals
import neuroglancer as ng
import nibabel as nib
import numpy as np
import tifffile
from nibabel.filebasedimages import ImageFileError
from nibabel.imageclasses import all_image_classes
from nibabel.spatialimages import SpatialImage

# internals
from ngtools.datasources import (
    LocalVolumeDataSource,
    VolumeInfo,
    datasource,
)
from ngtools.opener import open, parse_protocols, stringify_path


def _babel_load(
    fileobj: str | PathLike | IO,
    classes: list[SpatialImage] = tuple(all_image_classes)
) -> SpatialImage:
    fileobj = stringify_path(fileobj)
    if isinstance(fileobj, str):
        protocol, url = parse_protocols(fileobj)[-2:]
        fileobj = open(url, compression='infer')

        if protocol == "file":
            try:
                stat_result = os.stat(url)
            except OSError:
                raise FileNotFoundError(
                    f"No such file or no access: '{url}'")
            if stat_result.st_size <= 0:
                raise ImageFileError(f"Empty file: '{url}'")

            sniff = None
            for image_class in classes:
                is_valid, sniff \
                    = image_class.path_maybe_image(fileobj, sniff)
                if is_valid:
                    img = image_class.from_filename(fileobj)
                    return img

        else:
            for image_class in classes:
                try:
                    return image_class.from_url(fileobj)
                except Exception:
                    pass
    else:
        fileobj = open(fileobj, compression='infer')
        for image_class in classes:
            try:
                return image_class.from_stream(fileobj)
            except Exception:
                pass
    raise ImageFileError(f'Cannot work out file type of "{fileobj}"')


class BabelVolumeInfo(VolumeInfo):
    """Generic NiBabel metadata."""

    def __init__(self, fileobj: str | PathLike | IO | SpatialImage) -> None:
        if not isinstance(fileobj, SpatialImage):
            fileobj = _babel_load(fileobj)
        self._affine = fileobj.affine
        self._dtype = fileobj.get_data_dtype()
        self._shape = fileobj.shape
        self._header = getattr(fileobj, 'header', None)
        self._footer = getattr(fileobj, 'footer', None)

    _allInputNames = ["i", "j", "k", "m", "c^", "c1^", "c2^"]
    _allOutputNames = ["x", "y", "z", "t", "c^", "c1^", "c2^"]

    def getDataType(self) -> np.dtype:
        """TODO."""
        return self._dtype

    def getShape(self, level: int = 0) -> list[int]:
        """TODO."""
        return self._shape

    def getInputNames(self) -> list[float]:
        """TODO."""
        return self._allInputNames[:self.getRank()]

    def getOutputNames(self) -> list[float]:
        """TODO."""
        return self._allOutputNames[:self.getRank()]

    def getInputScales(self) -> list[float]:
        """TODO."""
        vx = (self._affine[:-1, :-1] ** 2).sum() ** 0.5
        vx = vx[:self.getRank()].tolist()
        vx += max(0, self.getRank() - len(vx)) * [1]
        return vx

    def getOutputScales(self) -> list[float]:
        """TODO."""
        return [1.0] * self.getRank()

    def getUnits(self) -> list[str]:
        """TODO."""
        units = ["mm"] * max(self.getRank(), 3)
        units += max(0, self.getRank() - len(units)) * [""]
        return units

    getInputUnits = getOutputUnits = getUnits

    def getMatrix(self) -> np.ndarray:
        """TODO."""
        if getattr(self, "_matrix", None) is None:
            affine = np.copy(self._affine)
            srank = min(3, self.getRank())
            scales = self.getInputScales()[:srank]
            affine[:srank, :srank] -= affine[:srank, :srank] @ ([0.5]*srank)
            affine[:srank, :srank] /= scales
            mat = np.eye(self.getRank()+1)[:-1]
            mat[:srank, :srank] = affine[:srank, :srank]
            mat[:srank, -1] = affine[:srank, -1]
            self._matrix = mat
        return self._matrix


@datasource(["nibabel", "mgh", "mgz"])
class BabelDataSource(LocalVolumeDataSource):
    """A local volume read using nibabel."""

    IMAGE_CLASSES = all_image_classes

    class LocalVolume(ng.LocalVolume):
        """LocalVolume that knows it is in a BabelDataSource."""

        ...

    def _format_specific_init(self) -> None:
        self._url = self.url
        format, _, _, url = parse_protocols(self.url)
        self._layer_type = self._layer_type or (
            "segmentation" if format in ("labels", "segmentation") else
            "image"
        )
        fileobj = _babel_load(url)
        self._info = BabelVolumeInfo(fileobj)
        self.url = self.LocalVolume(
            np.asarray(fileobj.dataobj),
            self.info.getInputDimensions(),
            volume_type=self._layer_type,
        )


BabelDataSource.LocalVolume.DataSourceType = BabelDataSource


@datasource(["mgh", "mgz"])
class MGHDataSource(BabelDataSource):
    """Wrapper for MGH/MGZ volume sources."""

    IMAGE_CLASSES = [nib.freesurfer.mghformat.MGHImage]

    class LocalVolume(BabelDataSource.LocalVolume):
        """LocalVolume that knows it is in a MGHDataSource."""

        ...


MGHDataSource.LocalVolume.DataSourceType = MGHDataSource


class TiffVolumeInfo(VolumeInfo):
    """Wrapper for local tiff volumes."""

    def __init__(
        self, url: str, fileobj: IO | tifffile.TiffFile | None = None
    ) -> None:
        self.url = str(url)
        url = parse_protocols(url)[-1]

        fileobj_is_mine = False
        if not isinstance(fileobj, tifffile.TiffFile):
            if not fileobj:
                fileobj = open(url, "rb")
                fileobj_is_mine = True
            mappedfile = tifffile.TiffFile(fileobj)
        else:
            mappedfile = fileobj

        self._shape = mappedfile.shaped_metadata
        self._dtype = mappedfile.series[0].dtype
        self._axes = mappedfile.series[0].axes
        self._ome = mappedfile.ome_metadata

        if fileobj_is_mine:
            fileobj.close()

        if self._ome:
            self._scales, self._units, self._names \
                = self.getZoomsOME(mappedfile.ome_metadata)
        else:
            self._scales = [1] * len(self._shape)
            self._units = [""] * len(self._shape)

    def getShape(self, level: int = 0) -> list[int]:
        """TODO."""
        assert level in (0, -1)
        return list(self._shape)

    def getDataType(self) -> np.dtype:
        """TODO."""
        return self._dtype

    def getScales(self) -> list[float]:
        """TODO."""
        return list(self._scales)

    getInputScales = getOutputScales = getScales

    def getUnits(self) -> list[float]:
        """TODO."""
        return ["" if unit == "pixel" else unit for unit in self._units]

    getInputUnits = getOutputUnits = getUnits

    def getNames(self) -> list[str]:
        """TODO."""
        return list(self._names)

    getInputNames = getOutputNames = getNames

    def getDimensions(self) -> ng.CoordinateSpace:
        """TODO."""
        names = self.getNames()
        scales = self.getScales()
        units = self.getUnits()
        return ng.CoordinateSpace({
            name: (scale, unit)
            for name, scale, unit in zip(names, scales, units)
        })

    getInputDimensions = getOutputDimensions = getDimensions

    def getTransform(self) -> ng.CoordinateSpaceTransform:
        """TODO."""
        return ng.CoordinateSpaceTransform(
            input_dimensions=self.getInputDimensions(),
            output_dimensions=self.getOutputDimensions(),
            transform=np.ndarray(self.getRank()+1)[:-1]
        )

    @staticmethod
    def getZoomsOME(
        omexml: str | bytes,
        series: int | list[int] | None = None
    ) -> tuple[list[float], list[str], str]:
        """Extract zoom factors (i.e., voxel size) from OME metadata.

        This function returns the zoom levels *at the highest resolution*.
        Zooms at subsequent resolution (_in the in-plane direction only_)
        can be obtained by multiplying the zooms at the previous resolution
        by 2. (This assumes that pyramid levels are built using a sliding
        window).

        If more than one series is requested, the returned variables
        are wrapped in a tuple E.g.:
        ```python
        >>> zooms, units, axes = ome_zooms(omexml)
        >>> zooms_series1 = zooms[1]
        >>> zooms_series1
        (10., 10., 5.)
        >>> units_series1 = units[1]
        >>> units_series1
        ('mm', 'mm', 'mm')
        ```

        Parameters
        ----------
        omexml : str or bytes
            OME-XML metadata
        series : int or list[int] or None, default=all
            Series ID(s)

        Returns
        -------
        zooms : tuple[float]
        units : tuple[str]
        axes : str

        """
        if not isinstance(omexml, (str, bytes)) or omexml[-4:] != 'OME>':
            return None, None, None

        # Open XML parser (copied from tifffile)
        try:
            root = etree.fromstring(omexml)
        except etree.ParseError:
            try:
                omexml = omexml.decode(errors='ignore').encode()
                root = etree.fromstring(omexml)
            except Exception:
                return None

        single_series = True
        if series is not None:
            single_series = isinstance(series, int)
            if not isinstance(series, (list, tuple)):
                series = [series]
            series = list(series)

        all_zooms = []
        all_units = []
        all_axes = []
        n_image = -1
        for image in root:
            # Any number [0, inf) of `image` elements
            if not image.tag.endswith('Image'):
                continue
            n_image += 1

            if series is not None and n_image not in series:
                all_zooms.append(None)
                all_units.append(None)
                all_axes.append(None)
                continue

            for pixels in image:
                # exactly one `pixels` element per image
                if not pixels.tag.endswith('Pixels'):
                    continue

                attr = pixels.attrib
                axes = ''.join(reversed(attr['DimensionOrder']))
                physical_axes = [ax for ax in axes
                                 if 'PhysicalSize' + ax in attr]
                zooms = [float(attr['PhysicalSize' + ax])
                         for ax in physical_axes]
                units = [attr.get('PhysicalSize' + ax + 'Unit', '')
                         for ax in physical_axes]

                all_zooms.append(tuple(zooms))
                all_units.append(tuple(units))
                all_axes.append(''.join(physical_axes))

        # reorder series
        if series is not None:
            all_zooms = [all_zooms[d] for d in series]
            all_units = [all_units[d] for d in series]
            all_axes = [all_axes[d] for d in series]
        if single_series:
            return all_zooms[0], all_units[0], all_axes[0]
        else:
            return tuple(all_zooms), tuple(all_units), tuple(all_axes)


@datasource(["tiff", "tif"])
class TiffDataSource(LocalVolumeDataSource):
    """A local tiff volume, read with tifffile."""

    class LocalVolume(ng.LocalVolume):
        """LocalVolume that knows it is in a TiffDataSource."""

        ...

    def _format_specific_init(self) -> None:
        self._url = self.url
        format, _, _, url = parse_protocols(self.url)
        self._layer_type = self._layer_type or (
            "segmentation" if format in ("labels", "segmentation") else
            "image"
        )

        fileobj = open(url)
        mappedfile = tifffile.TiffFile(fileobj)
        self._info = TiffVolumeInfo(url, fileobj=mappedfile)
        self.url = self.LocalVolume(
            mappedfile.asarray(),
            self.info.getInputDimensions(),
            volume_type=self._layer_type,
        )
        fileobj.close()


TiffDataSource.LocalVolume.DataSourceType = TiffDataSource
