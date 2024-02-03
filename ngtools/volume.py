import os
import io
import neuroglancer as ng
import nibabel as nib
import numpy as np
import tifffile
import zarr.hierarchy
import zarr.storage
import dask.array
import base64
from contextlib import contextmanager
from xml.etree import ElementTree as etree
from nibabel.filebasedimages import ImageFileError
from nibabel.imageclasses import all_image_classes
from .opener import open, stringify_path


class BabelMixin:
    """Mixin for 3D neuroimaging formats that have a voxel2world affine"""

    @property
    def dimensions(self):
        scales = np.sqrt(np.square(self.affine[:3, :3]).sum(0))
        return ng.CoordinateSpace(
            names=["i", "j", "k"],
            scales=scales.tolist(),
            units=['mm']*3,
        )

    outputDimensions = ng.CoordinateSpace(
        names=["x", "y", "z"],
        scales=[1]*3,
        units=['mm']*3,
    )

    @property
    def transform(self):
        scales = np.sqrt(np.square(self.affine[:3, :3]).sum(0))
        affine = np.copy(self.affine)
        affine[:3, :3] /= scales
        return ng.CoordinateSpaceTransform(
            matrix=affine[:3, :4],
            input_dimensions=self.dimensions,
            output_dimensions=self.outputDimensions,
        )


def _quantiles(q, data):
    """Compute (efficiently) intensity quantiles"""
    steps = [max(1, x//64) for x in data.shape]
    slicer = (Ellipsis,) + tuple(slice(None, None, step) for step in steps)
    return np.quantile(np.asarray(data[slicer]), q)


class RemoteSource(ng.LayerDataSource):
    """A remote data source"""

    @classmethod
    def from_filename(cls, filename, *args, **kwargs):
        if filename.startswith(('zarr://', 'zarr2://', 'zarr3://')):
            klass = RemoteZarr
        elif filename.startswith('nifti://'):
            klass = RemoteNifti
        else:
            klass = RemoteSource
        return klass(filename, *args, **kwargs)


class RemoteZarr(RemoteSource):
    """A remote zarr source"""

    def aszarr(self):
        url = '://'.join(self.url.split('://')[1:])
        store = zarr.storage.FSStore(url)
        group = zarr.group(store=store)
        return group

    def quantiles(self, q):
        group = self.aszarr()
        multiscales = group.attrs['multiscales'][0]
        nb_levels = len(multiscales['datasets'])

        for level in reversed(range(nb_levels)):
            shape = group[f'{level}'].shape
            shape = [shape[i] for i in range(len(shape))
                     if multiscales['axes'][i]['type'] == 'space']
            if any(s > 64 for s in shape):
                break

        data = dask.array.from_zarr(group[f'{level}'])
        return _quantiles(q, data)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        group = self.aszarr()
        multiscales = group.attrs['multiscales'][0]
        axistypes = [axis['type'] for axis in multiscales['axes']]
        self.names = [axis['name'] for axis in multiscales['axes']]
        self.names = ["c'" if n == 'c' else n for n in self.names]
        units = [axis.get('unit', '') for axis in multiscales['axes']]
        scales = multiscales['datasets'][0]['coordinateTransformations'][0]['scale']  # noqa: E501
        if 'coordinateTransformations' in multiscales:
            scalestime = multiscales['coordinateTransformations'][0]['scale']
            scales = [
                s0 if t == 'spatial' else s1
                for t, s0, s1 in zip(axistypes, scales, scalestime)
            ]
        scales_units = [self.units2ng(x, u) for x, u in zip(scales, units)]
        self.scales = [x[0] for x in scales_units]
        self.units = [x[1] for x in scales_units]
        self.affine = None
        if 'nifti' in group.attrs:
            binheader = base64.b64decode(group.attrs['nifti']['base64'])
            self.affine = nib.Nifti1Header.from_fileobj(
                io.BytesIO(binheader), check=False).get_sform()

    @property
    def dimensions(self):
        return ng.CoordinateSpace(
            names=self.names,
            scales=self.scales,
            units=self.units,
        )

    @property
    def outputDimensions(self):
        if self.affine is not None:
            return ng.CoordinateSpace(
                names=self.names[:-3] + ["x", "y", "z"],
                scales=self.scales[:-3] + [1]*3,
                units=self.units[:-3] + ['mm']*3,
            )
        else:
            return self.dimensions

    @property
    def transform(self):
        if self.affine is None:
            return None
        affine = np.copy(self.affine)
        affine[:3, :3] /= self.scales[-3:]
        fullaffine = np.eye(len(self.names)+1)[:-1]
        fullaffine[-3:, -4:-1] = self.affine[:3, :3][:, ::-1]
        fullaffine[-3:, -1] = self.affine[:3, -1]
        return ng.CoordinateSpaceTransform(
            matrix=fullaffine,
            input_dimensions=self.dimensions,
            output_dimensions=self.outputDimensions,
        )

    @staticmethod
    def units2ng(value, unit):
        NONSI = dict(
            # space
            angstrom=lambda x: (0.1*x, "nm"),
            foot=lambda x: (3.048*x, "dm"),
            inch=lambda x: (2.54*x, "cm"),
            mile=lambda x: (1.609*x, "m"),
            yard=lambda x: (0.9144*x, "m"),
            parsec=lambda x: (30.8568*x, "Pm"),
            # time
            day=lambda x: (86400*x, "s"),
            hour=lambda x: (3600*x, "s"),
            minute=lambda x: (60*x, "s"),
        )
        SI = dict(
            # space
            attometer="am",
            centimeter="cm",
            decimeter="dm",
            exameter="Em",
            femtometer="fm",
            gigameter="Gm",
            hectometer="Hm",
            kilometer="Km",
            megameter="Mm",
            meter="m",
            micrometer="um",
            millimeter="mm",
            nanometer="nm",
            petameter="Pm",
            picometer="pm",
            terameter="Tm",
            yoctometer="Ym",
            yottameter="ym",
            zeptometer="Zm",
            zettameter="zm",
            # time
            attosecond="as",
            centisecond="cs",
            decisecond="ds",
            exasecond="Es",
            femtosecond="fs",
            gigasecond="Gs",
            hectosecond="Hs",
            kilosecond="Ks",
            megasecond="Ms",
            microsecond="us",
            millisecond="ms",
            nanosecond="ns",
            petasecond="Ps",
            picosecond="ps",
            second="s",
            terasecond="Ts",
            yoctosecond="Ys",
            yottasecond="ys",
            zeptosecond="Zs",
            zettasecond="zs",
        )
        if unit in SI:
            return value, SI[unit]
        elif unit in NONSI:
            return NONSI[unit](value)
        else:
            return value, ''


class RemoteNifti(BabelMixin, RemoteSource):
    """A remote NIfTI source"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        with self.asnibabel() as f:
            self.affine = f.affine

    @contextmanager
    def asnibabel(self):
        url = '://'.join(self.url.split('://')[1:])
        with open(url, compression='gzip') as stream:
            yield nib.Nifti1Image.from_stream(stream)

    def quantiles(self, q):
        with self.asnibabel() as f:
            self.affine = f.affine
            return _quantiles(q, f.get_fdata(dtype='float32'))


class LocalSource(ng.LocalVolume):
    """Extends LocalVolume with utility functions"""

    @classmethod
    def from_filename(cls, filename, *args, **kwargs):
        if filename.startswith(('mgh://', 'mgz://')) \
                or filename.endswith(('.mgh', '.mgz')):
            klass = LocalMGH
        elif filename.startswith('nifti://') \
                or filename.endswith(('.nii', '.nii.gz')):
            klass = LocalNifti
        elif filename.startswith('tiff://') \
                or filename.endswith(('.tif', '.tiff')):
            klass = LocalTiff
        else:
            klass = LocalSource
        if '://' in filename:
            filename = '://'.join(filename.split('://')[1:])
        return klass(filename, *args, **kwargs)

    def quantiles(self, q):
        steps = [max(1, x//64) for x in self.data.shape]
        slicer = tuple(slice(None, None, step) for step in steps)
        return np.quantile(self.data[slicer], q).tolist()


class LocalBabel(BabelMixin, LocalSource):
    """A local volume read using nibabel"""

    possible_image_classes = all_image_classes

    def __init__(self, fileobj, layer_type=None):
        """
        Parameters
        ----------
        fileobj : str or path or file-like
            Input file
        layer_type : {'volume', 'labels', None}
            Type of volume
        """
        mappedfile = self._tryload(fileobj)
        self.affine = mappedfile.affine
        self.layer_type = layer_type
        if layer_type == 'volume':
            layer_type = 'image'
        elif layer_type == 'labels':
            layer_type = 'segmentation'
        super.__init__(
            np.asarray(mappedfile.dataobj) if layer_type == 'segmentation'
            else mappedfile.get_fdata(dtype='float32'),
            self.dimensions,
            volume_type=layer_type,
        )

    @classmethod
    def _tryload(cls, fileobj):
        fileobj = stringify_path(fileobj)
        if isinstance(fileobj, str):
            if not fileobj.startswith(('http://', 'https://')):
                try:
                    stat_result = os.stat(fileobj)
                except OSError:
                    raise FileNotFoundError(
                        f"No such file or no access: '{fileobj}'")
                if stat_result.st_size <= 0:
                    raise ImageFileError(f"Empty file: '{fileobj}'")

                sniff = None
                for image_klass in cls.possible_image_classes:
                    is_valid, sniff \
                        = image_klass.path_maybe_image(fileobj, sniff)
                    if is_valid:
                        img = image_klass.from_filename(fileobj)
                        return img

            else:
                for image_klass in cls.possible_image_classes:
                    try:
                        return image_klass.from_url(fileobj)
                    except Exception:
                        pass
        else:
            fileobj = open(fileobj)
            for image_klass in cls.possible_image_classes:
                try:
                    return image_klass.from_stream(fileobj)
                except Exception:
                    pass
        raise ImageFileError(f'Cannot work out file type of "{fileobj}"')


class LocalMGH(LocalBabel):
    possible_image_classes = [
        nib.freesurfer.mghformat.MGHImage,
    ]


class LocalNifti(LocalBabel):
    possible_image_classes = [
        nib.nifti1.Nifti1Image,
        nib.nifti2.Nifti2Image,
    ]


class LocalTiff(LocalSource):
    """A local tiff volume, read with tifffile"""

    @property
    def dimensions(self):
        return ng.CoordinateSpace(
            names=self.names,
            scales=self.scales,
            units=self.units,
        )

    @property
    def outputDimensions(self):
        return self.dimensions

    @property
    def transform(self):
        return None

    def __init__(self, fileobj, layer_type=None):
        """
        Parameters
        ----------
        fileobj : str or path or file-like
            Input file
        layer_type : {'volume', 'labels', None}
            Type of volume
        """
        fileobj = open(fileobj)
        mappedfile = tifffile.TiffFile(fileobj)

        if mappedfile.ome_metadata:
            self.scales, self.units, self.names \
                = self.ome_zooms(mappedfile.ome_metadata)
        else:
            self.scales = [1] * len(mappedfile.shape)
            self.units = [''] * len(mappedfile.shape)
            self.names = tifffile.axes
        self.names = list(map(lambda x: x.lower(), self.axes))
        self.layer_type = layer_type

        if layer_type == 'volume':
            layer_type = 'image'
        elif layer_type == 'labels':
            layer_type = 'segmentation'
        super.__init__(
            mappedfile.asarray(),
            self.dimensions,
            volume_type=layer_type,
        )

    @staticmethod
    def parse_unit(unit):
        """Parse a unit string

        Parameters
        ----------
        unit : str
            String describing a unit (e.g., 'mm')

        Returns
        -------
        factor : float
            Factor that relates the base unit to the parsed unit.
            E.g., `parse_unit('mm')[0] -> 1e-3`
        baseunit : str
            Physical unit without the prefix.
            E.g., `parse_unit('mm')[1] -> 'm'`

        """
        if unit is None or len(unit) == 0:
            return 1.
        if unit == 'pixel':
            return 1., unit
        unit_type = unit[-1]
        unit_scale = unit[:-1]
        mu1 = '\u00B5'
        mu2 = '\u03BC'
        unit_map = {
            'Y': 1E24, 'Z': 1E21, 'E': 1E18, 'P': 1E15, 'T': 1E12, 'G': 1E9,
            'M': 1E6, 'k': 1E3, 'h': 1e2, 'da': 1E1, 'd': 1E-1, 'c': 1E-2,
            'm': 1E-3, mu1: 1E-6, mu2: 1E-6, 'u': 1E-6, 'n': 1E-9,
            'p': 1E-12, 'f': 1E-15, 'a': 1E-18, 'z': 1E-21, 'y': 1E-24,
        }
        if len(unit_scale) == 0:
            return 1., unit_type
        else:
            return unit_map[unit_scale], unit_type

    @staticmethod
    def ome_zooms(omexml, series=None):
        """Extract zoom factors (i.e., voxel size) from OME metadata

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

        ```

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

        single_series = False
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
