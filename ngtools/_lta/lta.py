"""
MGH file format for affine transformations.

https://surfer.nmr.mgh.harvard.edu/fswiki/FsTutorial/LtaFormat

A bit must be said on the FreeSurfer space conventions:
- Any 3D volume is associated with a "physical" coordinate system.
  This system only depends on the voxel size and volume shape.
  This coordinate system is millimetric and ensures that the origin is
  in the center of the field-of-view.
- There is also a more generic "RAS" system. MGH images store
  an additional transform in an xform (= columns of a 3x4  matrix).
  This transform maps from physical to RAS space. When on-disk
  dimensions are not ordered as RAS, the xform includes shifts and
  permutations, to ensure that the output coordinates follow the RAS
  orientation.
- A NIfTI-like orientation matrix (named vox2ras in the FS nomenclature)
  can be recovered by composing the voxel-to-physical and the
  physical-to-ras matrices.
- Note that before registration (when images are mis-registered), their
  RAS might differ. Registration typically finds a ras-to-ras
  transform that brings the two dissociate spaces in alignment.
- The LTA file stores such a world-to-world transform, along with the
  orientation matrices of the original source and destination images.
  The voxel-to-physical and physical-to-ras components of these
  orientation matrices are stored as well.
- However, the LTA can also store transforms between different kinds of
  spaces (voxel-to-voxel, physical-to-physical, ras-to-ras or even
  rsa-to-rsa).
"""
# stdlib
import os
from dataclasses import dataclass
from enum import IntEnum
from typing import Iterator, Literal, TextIO, Union

# externals
import numpy as np

# internals
from .fsutils import (
    affine_matmul,
    affine_to_fs,
    fs_to_affine,
    read_key,
    read_values,
    write_key,
    write_values,
)


class Constants(IntEnum):
    """Defines used in FreeSurfer's C code."""

    # Affine transformation types
    LINEAR_VOX_TO_VOX = 0
    LINEAR_VOXEL_TO_VOXEL = LINEAR_VOX_TO_VOX
    LINEAR_RAS_TO_RAS = 1
    LINEAR_PHYSVOX_TO_PHYSVOX = 2
    LINEAR_CORONAL_RAS_TO_CORONAL_RAS = 21
    LINEAR_COR_TO_COR = LINEAR_CORONAL_RAS_TO_CORONAL_RAS
    # Transformation file types
    TRANSFORM_ARRAY_TYPE = 10
    MORPH_3D_TYPE = 11
    MNI_TRANSFORM_TYPE = 12
    MATLAB_ASCII_TYPE = 13


# Value format (= sequence of types) associated to each key
lta_keys = {
    'type': int,
    'nxforms': int,
    'mean': (float,) * 3,
    'sigma': float,
    'valid': int,
    'filename': str,
    'volume': (int,) * 3,
    'voxelsize': (float,) * 3,
    'xras': (float,) * 3,
    'yras': (float,) * 3,
    'zras': (float,) * 3,
    'cras': (float,) * 3,
}


def nested_update(old_dict: dict, new_dict: dict) -> None:
    """
    Update a dictionary in place.

    Nested dictionaries are updated too instead of being replaced.

    Parameters
    ----------
    old_dict : dict
        Dictionary updated in-place
    new_dict : dict
        New values

    """
    for key, value in new_dict.items():
        if (key in old_dict and isinstance(old_dict[key], dict) and
                isinstance(value, dict)):
            nested_update(old_dict[key], value)
        else:
            old_dict[key] = value


@dataclass
class LTAStruct:
    """Structure encoding an LTA file.

    This representation mimics the representation on dist and is not
    exposed to the user.
    """

    type: int = Constants.LINEAR_VOX_TO_VOX     # Affine type
    nxforms: int = 1                            # Number of affines stored
    mean: tuple = (0., 0., 0.)                  # ?
    sigma: float = 1.                           # ?
    affine: np.ndarray = None                   # Affine(s)

    @dataclass
    class VolumeInfo:
        valid: bool = 1                         #
        filename: str = None                    # Filename of the volume
        volume: tuple = None                    # 3D shape
        voxelsize: tuple = None                 # Voxel size
        xras: tuple = None                      # Columns of the xform
        yras: tuple = None                      # "
        zras: tuple = None                      # "
        cras: tuple = None                      # "

    src: VolumeInfo = None                      # Source volume
    dst: VolumeInfo = None                      # Destination volume

    def __str__(self) -> str:
        s = '\n'.join(list(self.to_lines()))
        s = 'LTAStruct\n' + '---------\n' + s
        return s

    __repr__ = __str__

    @classmethod
    def from_filename(cls, fname: str) -> "LTAStruct":
        """Build from path to LTA file."""
        with open(fname, 'r') as f:
            return cls.from_lines(f)

    @classmethod
    def from_lines(cls, f: list[str]) -> "LTAStruct":
        """Build from an iterator over lines."""
        lta = LTAStruct()
        section = 'header'
        affine = []
        affine_shape = (1, 4, 4)
        for line in f:
            if hasattr(line, 'decode'):
                line = line.decode()
            line = line.split('\r\n')[0]  # remove eol (windows)
            line = line.split('\n')[0]    # remove eol (unix)
            line = line.split('#')[0]     # remove hanging comments
            line = line.rstrip()          # remove trailing whitespaces
            if not line:
                continue
            if line.startswith('src volume info'):
                section = 'src0'
                continue
            elif line.startswith('dst volume info'):
                section = 'dst0'
                continue
            elif section == 'header':
                if '=' in line:
                    try:
                        key, value = read_key(line, lta_keys)
                        setattr(lta, key, value)
                    finally:
                        continue
                # else we should be in the affine section
                try:
                    affine_shape = read_values(line, [int] * 3)
                    if not affine_shape:
                        affine_shape = (1, 4, 4)
                    section = 'affine'
                finally:
                    continue
            elif section == 'affine':
                try:
                    row = read_values(line, [float] * affine_shape[-1])
                    if not affine:
                        affine.append([])
                    if len(affine[-1]) == affine_shape[-2]:
                        affine.append([])
                    affine[-1].append(list(row))
                finally:
                    continue
            else:
                if section.endswith('0'):
                    section = section[:-1]
                    setattr(lta, section, cls.VolumeInfo())
                vol = section
                try:
                    key, value = read_key(line, lta_keys)
                    setattr(getattr(lta, vol), key, value)
                finally:
                    continue
        if affine is not None:
            lta.affine = np.asarray(affine).reshape(affine_shape)
        return lta

    def to_lines(self) -> Iterator[str]:
        """Return an iterator over lines."""
        # header
        attributes = ('type', 'nxforms', 'mean', 'sigma')
        for attr in attributes:
            val = getattr(self, attr)
            if val is not None:
                row = write_key(attr, val)
                if attr == 'type':
                    row = row + '  # ' + Constants(val).name
                yield row
        # affine
        if self.affine is not None:
            yield write_values(self.affine.shape)
            affine = self.affine.reshape(-1, self.affine.shape[-1])
            for row in affine:
                yield write_values(row)
        # volume info
        volumes = ('src', 'dst')
        attributes = ('valid', 'filename', 'volume', 'voxelsize',
                      'xras', 'yras', 'zras', 'cras')
        for volume in volumes:
            block = getattr(self, volume)
            if block is None:
                continue
            yield f'{volume} volume info'
            for attr in attributes:
                val = getattr(block, attr)
                if val is not None:
                    yield write_key(attr, val)
        return

    def to_file_like(self, f: TextIO) -> None:
        """Write to file-like object."""
        for line in self.to_lines():
            f.write(line + os.linesep)
        return

    def to_filename(self, fname: str) -> None:
        """Write to file."""
        with open(fname, 'wt') as f:
            return self.to_file_like(f)

    def to(self, thing: str | TextIO) -> None:
        """Write to something."""
        if isinstance(thing, str):
            return self.to_filename(thing)
        else:
            return self.to_file_like(thing)


class LinearTransformArray:
    """MGH format for affine transformations."""

    @classmethod
    def possible_extensions(cls) -> tuple[str]:
        return ('.lta',)

    def __init__(
        self,
        file_like: str | TextIO | None = None,
        mode: str = 'r'
    ) -> None:
        """

        Parameters
        ----------
        file_like : str of file object
            File to map
        mode : {'r', 'r+'}, default='r'
            Read in read-only ('r') or read-and-write ('r+') mode.
            Modifying the file in-place is only possible in 'r+' mode.
        """
        self.filename = None
        self.mode = mode
        if file_like is None:
            self._struct = LTAStruct()
        elif isinstance(file_like, LTAStruct):
            self._struct = file_like
        else:
            self.filename = file_like
            if 'r' in mode:
                if isinstance(file_like, str):
                    self._struct = LTAStruct.from_filename(file_like)
                else:
                    self._struct = LTAStruct.from_lines(file_like)
            else:
                self._struct = LTAStruct()

    @property
    def shape(self) -> list[int]:
        if self._struct.affine is not None:
            return tuple(self._struct.affine.shape)
        else:
            return tuple()

    def matrix(
        self,
        source: Literal['voxel', 'physical', 'ras'] = 'ras',
        dest: Literal['voxel', 'physical', 'ras'] = 'ras',
        dtype: np.dtype | None = None
    ) -> np.ndarray:
        """Return <source>2<dest> matrix.

        Parameters
        ----------
        source : {'voxel', 'physical', 'ras'}, default='ras'
        dest : {'voxel', 'physical', 'ras'}, default='ras'
        dtype : np.dtype, optional

        Returns
        -------
        affine : (D+1, D+1) array[dtype]

        """
        backend = dict(dtype=dtype)
        affine = self.raw_matrix(**backend)
        if affine is None:
            return None

        # we may need to convert from a weird space to RAS space
        if self.type()[0] != source or self.type()[1] != dest:
            src, _ = self.source_space(source, self.type()[0], **backend)
            dst, _ = self.destination_space(self.type()[1], dest, **backend)
            if src is not None and dst is not None:
                affine = affine_matmul(dst, affine_matmul(affine, src))
        return affine

    def raw_matrix(self, dtype: np.dtype | None = None) -> np.ndarray:
        """Return raw matrix."""
        if self._struct.affine is None:
            return None
        return self._struct.affine.astype(dtype)

    def source_space(
        self,
        source: Literal['voxel', 'physical', 'ras'] = 'voxel',
        dest: Literal['voxel', 'physical', 'ras'] = 'ras',
        dtype: np.dtype | None = None
    ) -> tuple[np.ndarray, list[int]]:
        """Return the space (affine + shape) of the source image.

        Parameters
        ----------
        source : {'voxel', 'physical', 'ras'}, default='voxel'
            Source space of the affine
        dest : {'voxel', 'physical', 'ras'}, default='ras'
            Destination space of the affine
        dtype : torch.dtype, optional

        Returns
        -------
        affine : (4, 4) array
            A voxel to world affine matrix
        shape : (3,) tuple[int]
            The spatial shape of the image

        """
        if self._struct.src is not None:
            affine = fs_to_affine(
                self._struct.src.volume,
                self._struct.src.voxelsize,
                self._struct.src.xras,
                self._struct.src.yras,
                self._struct.src.zras,
                self._struct.src.cras,
                source=source, dest=dest)
            shape = tuple(self._struct.src.volume)
            affine = affine.astype(dtype)
            return affine, shape
        return None, None

    def destination_space(
        self,
        source: Literal['voxel', 'physical', 'ras'] = 'voxel',
        dest: Literal['voxel', 'physical', 'ras'] = 'ras',
        dtype: np.dtype | None = None
    ) -> tuple[np.ndarray, list[int]]:
        """Return the space (affine + shape) of the destination image.

        Parameters
        ----------
        source : {'voxel', 'physical', 'ras'}, default='voxel'
            Source space of the affine
        dest : {'voxel', 'physical', 'ras'}, default='ras'
            Destination space of the affine
        dtype : torch.dtype, optional

        Returns
        -------
        affine : (4, 4) array
            A voxel to world affine matrix
        shape : (3,) tuple[int]
            The spatial shape of the image

        """
        if self._struct.dst is not None:
            affine = fs_to_affine(
                self._struct.dst.volume,
                self._struct.dst.voxelsize,
                self._struct.dst.xras,
                self._struct.dst.yras,
                self._struct.dst.zras,
                self._struct.dst.cras,
                source=source, dest=dest)
            shape = tuple(self._struct.dst.volume)
            affine = affine.astype(dtype)
            return affine, shape
        return None, None

    def set_source_space(
        self,
        affine: np.ndarray,
        shape: list[int],
        source: Literal['voxel', 'physical', 'ras'] = 'voxel',
        dest: Literal['voxel', 'physical', 'ras'] = 'ras',
    ) -> "LinearTransformArray":
        """Set the source space of the transform.

        Parameters
        ----------
        affine : (4, 4) array
            Affine matrix
        shape : sequence of int
            Volume shape
        source : {'voxel', 'physical', 'ras'}, default='voxel'
            Source space of the affine
        dest : {'voxel', 'physical', 'ras'}, default='ras'
            Destination space of the affine

        Returns
        -------
        self

        """
        if not self._struct.src:
            self._struct.src = self._struct.VolumeInfo()
        if not self._struct.dst:
            self._struct.dst = self._struct.VolumeInfo()
        vx, x, y, z, c = affine_to_fs(affine, shape, source, dest)
        self._struct.src.volume = tuple(shape)
        self._struct.src.voxelsize = vx
        self._struct.src.xras = x
        self._struct.src.yras = y
        self._struct.src.zras = z
        self._struct.src.cras = c
        return self

    def set_destination_space(
        self,
        affine: np.ndarray,
        shape: list[int],
        source: Literal['voxel', 'physical', 'ras'] = 'voxel',
        dest: Literal['voxel', 'physical', 'ras'] = 'ras',
    ) -> "LinearTransformArray":
        """Set the destination space of the transform.

        Parameters
        ----------
        affine : (4, 4) array
            Affine matrix
        shape : sequence of int
            Volume shape
        source : {'voxel', 'physical', 'ras'}, default='voxel'
            Source space of the affine
        dest : {'voxel', 'physical', 'ras'}, default='ras'
            Destination space of the affine

        Returns
        -------
        self

        """
        vx, x, y, z, c = affine_to_fs(affine, shape, source, dest)
        self._struct.dst.volume = tuple(shape)
        self._struct.dst.voxelsize = vx
        self._struct.dst.xras = x
        self._struct.dst.yras = y
        self._struct.dst.zras = z
        self._struct.dst.cras = c
        return self

    def type(self) -> Constants:
        if self._struct.type == Constants.LINEAR_VOX_TO_VOX:
            return 'voxel', 'voxel'
        elif self._struct.type == Constants.LINEAR_RAS_TO_RAS:
            return 'ras', 'ras'
        elif self._struct.type == Constants.LINEAR_COR_TO_COR:
            return 'rsa', 'rsa'
        elif self._struct.type == Constants.LINEAR_PHYSVOX_TO_PHYSVOX:
            return 'physical', 'physical'
        else:
            raise TypeError(f'Don\'t know what to do with {self._struct.type}')

    def metadata(self, keys: list[str] | None = None) -> dict:
        """Read additional metadata.

        Parameters
        ----------
        keys : sequence of str, optional
            Keys should be in {'type', 'mean', 'sigma', 'src',
                'src',
                'src.valid', 'src.filename', 'src.volume', 'src.voxelsize',
                'src.xras', 'src.yras', 'src.zras', 'src.cras',
                'dst',
                'dst.valid', 'dst.filename', 'dst.volume', 'dst.voxelsize',
                'dst.xras', 'dst.yras', 'dst.zras', 'dst.cras'
            }

        Returns
        -------
        dict
            type : {'ras', 'rsa', 'physical', 'voxel'}
                Transformation type
            mean : (3,) sequence of float
                Mean image intensity
            sigma : float
                Standard deviation of the image intensity
            src : dict
                Source volume information
            dst : dict
                Destination volume information
            src|dst.valid : bool
                Is the volume valid
            src|dst.filename : str
                Filename of the source volume
            src|dst.volume : tuple[int]
                3D shape
            src|dst.voxelsize : tuple[float]
                Voxel size
            src|dst.xras : tuple[float]
            src|dst.yras : tuple[float]
            src|dst.zras : tuple[float]
            src|dst.cras : tuple[float]

        """
        known_keys = ('type', 'mean', 'sigma')
        known_superkeys = ('src', 'dst')
        known_subkeys = ('valid', 'filename', 'volume', 'voxelsize',
                         'xras', 'yras', 'zras', 'cras')
        all_keys = known_keys + tuple(sup + '.' + sub
                                      for sub in known_subkeys
                                      for sup in known_superkeys)
        keys = keys or all_keys
        meta = dict()
        for key in keys:
            if key in known_keys:
                meta[key] = getattr(self._struct, key)
            else:
                sup, *sub = key.split('.')
                if sup in known_superkeys and sub and sub[0] in known_subkeys:
                    sub = sub[0]
                    meta[key] = getattr(getattr(self._struct, sup), sub)
                else:
                    meta[key] = None
        return meta

    def set_fdata(self, affine: np.ndarray) -> "LinearTransformArray":
        affine = np.asarray(affine)
        backend = dict(dtype=affine.dtype)
        if affine.shape[-2:] != (4, 4):
            raise ValueError('Expected a batch of 4x4 matrix')

        # we may need to convert from RAS space to a weird space
        afftype = self.type()[0]
        if afftype != 'ras':
            src, _ = self.source_space(afftype, 'ras', **backend)
            dst, _ = self.destination_space('ras', afftype, **backend)
            if src is not None and dst is not None:
                affine = affine_matmul(dst, affine_matmul(affine, src))

        affine = np.asarray(affine).reshape([-1, 4, 4])
        self._struct.affine = affine
        self._struct.nxform = affine.shape[0]
        return self

    def set_data(self, affine: np.ndarray) -> "LinearTransformArray":
        affine = np.asarray(affine)
        if affine.shape != (4, 4):
            raise ValueError('Expected a 4x4 matrix')
        affine = affine.reshape([-1, 4, 4])
        self._struct.affine = affine
        self._struct.nxform = affine.shape[0]
        return self

    @classmethod
    def _set_metadata(cls, struct: LTAStruct, **meta) -> None:
        """Set LTA fields (in-place) from a dictionary.

        Parameters
        ----------
        struct : LTAStruct
        meta : dict

        """
        if 'type' in meta and not isinstance(meta['type'], int):
            ltatype = meta['type']
            if isinstance(ltatype, (list, tuple)):
                srctype, dsttype = ltatype
            else:
                srctype = dsttype = ltatype
            if srctype == dsttype == 'ras':
                ltatype = Constants.LINEAR_RAS_TO_RAS
            elif srctype == dsttype == 'rsa':
                ltatype = Constants.LINEAR_COR_TO_COR
            elif srctype == dsttype == 'voxel':
                ltatype = Constants.LINEAR_VOX_TO_VOX
            elif srctype == dsttype == 'physical':
                ltatype = Constants.LINEAR_PHYSVOX_TO_PHYSVOX
            else:
                raise ValueError('Unsupported LTA type:', ltatype)
            meta['type'] = ltatype

        def tupleof(typ: type) -> callable:
            return (lambda x: tuple(typ(y) for y in x))

        known_keys = dict(type=int, mean=tupleof(float), sigma=float)
        known_superkeys = ('src', 'dst')
        known_subkeys = dict(valid=int, filename=str, volume=tupleof(int),
                             voxelsize=tupleof(float), xras=tupleof(float),
                             yras=tupleof(float), zras=tupleof(float),
                             cras=tupleof(float))
        for key, value in meta.items():
            if key in tuple(known_keys.keys()):
                conv = known_keys[key]
                setattr(struct, key, conv(value))
            elif key in known_superkeys and isinstance(value, dict):
                sup = getattr(struct, key)
                for subkey, subval in value.items():
                    if subkey in known_subkeys:
                        conv = known_subkeys[subkey]
                        setattr(sup, subkey, conv(subval))

    def set_metadata(
        self,
        dict_like: dict | None = None,
        **meta
    ) -> "LinearTransformArray":
        """Set additional metadata.

        Parameters
        ----------
        type : {'ras', 'rsa', 'physical', 'voxel'}
            Transformation type
        mean : (3,) sequence of float
            Mean image intensity
        sigma : float
            Standard deviation of the image intensity
        src, dst : dict
            Source/Destination volume information with keys:
            valid : bool
                Is the volume valid
            filename : str
                Filename of the source volume
            volume : tuple[int]
                3D shape
            voxelsize : tuple[float]
                Voxel size
            xras : tuple[float]
            yras : tuple[float]
            zras : tuple[float]
            cras : tuple[float]

        Returns
        -------
        self

        """
        dict_like = dict_like or {}
        dict_like.update(meta)
        self._set_metadata(self._struct, **dict_like)
        return self

    def save(
        self, file_like: str | TextIO | None = None, *args, **meta
    ) -> "LinearTransformArray":
        if '+' not in self.mode and 'w' not in self.mode:
            raise RuntimeError('Cannot write into read-only volume. '
                               'Re-map in mode "r+" to allow in-place '
                               'writing.')
        self.set_metadata(**meta)
        file_like = file_like or self.filename or self.file_like
        self._struct.to(file_like)
        return self

    @classmethod
    def save_new(
        cls,
        affine: Union[np.ndarray, "LinearTransformArray"],
        file_like: str | TextIO | None = None,
        like: Union["LinearTransformArray", str, TextIO] | None = None,
        *args,
        **meta
    ) -> "LinearTransformArray":
        if isinstance(affine, LinearTransformArray):
            if like is None:
                like = affine
            affine = affine.data()
        affine = np.asanyarray(affine)
        if like is not None and not isinstance(like, LinearTransformArray):
            like = LinearTransformArray(like)

        if affine.shape[-2:] != (4, 4):
            raise ValueError('Expected a batch of 4x4 matrix')
        affine = affine.reshape([-1, 4, 4])

        struct = LTAStruct()
        struct.affine = affine
        metadata = like.metadata() if like is not None else dict()
        nested_update(metadata, meta)
        cls._set_metadata(struct, **metadata)
        struct.to(file_like)
