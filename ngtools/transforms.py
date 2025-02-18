"""
Utilities to manipulate transforms (`ng.CoordinateSpaceTransform`) in
neuroglancer.

Also provide tools to read and convert arbitrary coordinate transforms into
neuroglancer format.

Functions
---------
compose
    Compose two `ng.CoordinateSpaceTransform`.
convert_transform
    Convert the units of a transform, while keeping the transform equivalent.
ensure_same_scale
    Convert the units of a transform, so that all axes have the same
    unit and scale, while keeping the transform equivalent.
get_matrix
    Get matrix from transform, creating one if needed.
inverse
    Invert a `ng.CoordinateSpaceTransform`.
load_affine
    Load an affine transform from a file or stream.
normalize_transform
    Ensure that input and output spaces have SI units.
subtransform
    Keep only axes whose units are of a given kind.
to_square
    Ensure that an affine matrix is square (with its homogeneous row).


Note on tranforms
-----------------

A `CoordinateSpaceTransform` has the following fields

    transform : np.ndarray
        Homogeneous affine transform that maps input coordinates
        (columns) to output coordinates (row)
    inputDimensions : CoordinateSpace
        A description of the input space.
        Sometimes referred to as the "source" space in the ng sourcecode.
    outputDimensions : CoordinateSpace
        A description of the output space.
        Sometimes referred to as the "view" space in the ng sourcecode.

A `CoordinateSpace` has the following fields:

    names : list[str]
        The name of each dimension.
    scales : list[float]
        The scale (voxel size) of each dimension.
    units : list[str]
        The unit in which scale is expressed.

Two important points must be noted, as they differ from how
transforms are encoded in other frameworks:

    * The _linear part_ of the transform is ___always___ unit-less.
        That is, it maps from/to the same isotropic unit.

    * The _translation_ component of the transform is ___always___
        expressed in output units.

A nifti-like "voxel to scaled-world" matrix would therefore be
constructed via

```
    [ x ]   [ 1/vx     ]   [ Axi Axj Axk ]   [ sx       ]   [ i ]   [ Tx ]
    [ y ] = [   1/vy   ] * [ Ayi Ayj Ayk ] * [    sy    ] * [ j ] + [ Ty ]
    [ z ]   [     1/vz ]   [ Azi Azj Azk ]   [       sz ]   [ k ]   [ Tz ]
```

where `v` is the view/output voxel size and `s` is the source/input
voxel size. In that context, [x, y, z] is expressed in "scaled world" units
and [i, j, k] is expressed in "voxel" units.

Alternatively, a "scaled-voxel to world" matrix can be constructed via

```
    [ x ]   [ Axi Axj Axk ]   [ sx       ]   [ i ]   [ vx       ]   [ Tx ]
    [ y ] = [ Ayi Ayj Ayk ] * [    sy    ] * [ j ] + [    vy    ] * [ Ty ]
    [ z ]   [ Azi Azj Azk ]   [       sz ]   [ k ]   [       vz ]   [ Tz ]
```
"""

# stdlib
import logging
import math
from numbers import Number
from os import PathLike
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import IO

# externals
import neuroglancer as ng
import numpy as np
from nitransforms.io import afni, fsl, itk, lta

# internals
import ngtools.spaces as S
import ngtools.units as U
from ngtools._lta.lta import LinearTransformArray
from ngtools.opener import open, parse_protocols, stringify_path

LOG = logging.getLogger(__name__)

AFFINE_FORMATMAP = {
    'afni': afni.AFNILinearTransform,
    'fsl': fsl.FSLLinearTransform,
    'itk': itk.ITKLinearTransform,
    'lta': getattr(lta, 'LinearTransformArray',
                   getattr(lta, 'FSLinearTransform')),
}


def matrix_to_quaternion(mat: np.ndarray) -> np.ndarray:
    """Convert a rotation matrix to a quaternion.

    We follow neuroglancer and assume that quaternions are ordered as
    [*v, r] (or again, [i, j, k, r]). This differs form wikipedia
    which defines them as [r, *v] (or [r, i, j, k]).
    """
    # https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation#Fitting_quaternions
    mat = np.asarray(mat)
    if tuple(mat.shape) != (3, 3):
        raise ValueError("Expected a 3x3 matrix")
    if np.linalg.det(mat) < 0:
        # left-handed: flip first axis
        mat = np.copy(mat)
        mat[0, :] *= -1
    [[Qxx, Qxy, Qxz], [Qyx, Qyy, Qyz], [Qzx, Qzy, Qzz]] = mat
    K = np.asarray([
        [Qxx - Qyy - Qzz, Qyx + Qxy, Qzx + Qxz, Qzy - Qyz],
        [Qyx + Qxy, Qyy - Qxx - Qzz, Qzy + Qyz, Qxz - Qzx],
        [Qzx + Qxz, Qzy + Qyz, Qzz - Qxx - Qyy, Qyx - Qxy],
        [Qzy - Qyz, Qxz - Qzx, Qyx - Qxy, Qxx + Qyy + Qzz],
    ]) / 3
    val, vec = np.linalg.eig(K)
    vec = vec[:, val.argmax()]
    # vec = np.concatenate([vec[1:], vec[:1]])  # wiki to neuroglancer
    return vec


def quaternion_to_matrix(q: np.ndarray) -> np.ndarray:
    """Convert a quaternion to a rotation matrix."""
    # https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation#From_a_quaternion_to_an_orthogonal_matrix
    q = np.asarray(q).flatten()
    if q.shape != (4,):
        raise ValueError("Expected a vector of length 4")
    q = np.concatenate([q[-1:], q[:-1]])  # neuroglancer to wiki
    a, b, c, d = q
    s = 2 / (a*a + b*b + c*c + d*d)
    bs = b * s
    cs = c * s
    ds = d * s
    ab = a * bs
    ac = a * cs
    ad = a * ds
    bb = b * bs
    bc = b * cs
    bd = b * ds
    cc = c * cs
    cd = c * ds
    dd = d * ds
    m = np.asarray([
        [1 - cc - dd, bc - ad, bd + ac],
        [bc + ad, 1 - bb - dd, cd - ab],
        [bd - ac, cd + ab, 1 - bb - cc],
    ])
    return m


def compose_quaternions(a: np.ndarray, b: np.ndarray, *others) -> np.ndarray:
    """Compose quaternions."""
    a = np.asarray(a).flatten()
    b = np.asarray(b).flatten()
    av, ar = a[:-1], a[-1:]
    bv, br = b[:-1], b[-1:]
    cr = ar * br - np.dot(av, bv)
    cv = ar * bv + br * av + np.cross(av, bv)
    c = np.concatenate([cv, cr])
    if others:
        c = compose_quaternions(c, *others)
    return c


def inverse_quaternions(a: np.ndarray) -> np.ndarray:
    """Inverse quaternion."""
    a = np.asarray(a).flatten()
    av, ar = a[:-1], a[-1:]
    nrm = ar*ar + (av*av).sum()
    ar = +ar / nrm
    av = -av / nrm
    a = np.concatenate([av, ar])
    return a


def load_affine(
    fileobj: IO | PathLike | str,
    format: str | None = None,
    moving: PathLike | str | None = None,
    fixed: PathLike | str | None = None,
    names: list[str] = ("x", "y", "z"),
) -> ng.CoordinateSpaceTransform:
    """
    Load an affine transform from a file.

    Parameters
    ----------
    fileobj : file_like or str
        Input file. May start with a format protocol such as
        `"afni://path/to/file"`. Accepeted protocols are

        * `"afni://"`   Affine generated by AFNI
        * `"fsl://"`    Affine generated by FSL FLIRT
        * `"itk://"`    Affine in ITK format (e.g., generated by ANTs)
        * `"lta://"`    Affine in FreeSurfer LTA format (mri_robust_register)
    format : {"afni", "fsl", "itk", "lta"}, optional
        Alternative way to provide a format hint.

    Returns
    -------
    affine : (4, 4) np.ndarray
        A RAS2RAS affine transform.
    """
    opt = dict(moving=moving, fixed=fixed, names=names)

    fileobj = stringify_path(fileobj)

    if isinstance(fileobj, str):
        parsed = parse_protocols(fileobj)
        format = format or parsed.format
        fileobj = parsed.url

        if not format and fileobj.endswith('.lta'):
            format = 'lta'

        # If remote file, open with fsspec
        if parsed.stream != "file":
            with open(fileobj) as f:
                return load_affine(f, format=format, **opt)

    # If open file object, write to local file
    if hasattr(fileobj, 'read'):
        with TemporaryDirectory() as tmp:
            # NOTE: I use a TemporaryDirectory rather than a
            #       NamedTemporaryFile because I've encountered
            #       issues where the file was not yet written
            #       but was already being read by the transform reader.
            tmpname = Path(tmp) / "affine.lta"
            with tmpname.open("wb") as f:
                f.write(fileobj.read())
            out = load_affine(tmpname, format=format, **opt)
            return out

    def _read(klass: type) -> ng.CoordinateSpaceTransform:
        """Try to read with a nitransforms type."""
        ras2ras = klass.from_filename(fileobj).to_ras(moving, fixed)
        if isinstance(ras2ras, list):
            ras2ras = ras2ras[0]
        return _make_transform(ras2ras)

    def _make_transform(ras2ras: np.ndarray) -> ng.CoordinateSpaceTransform:
        dims = ng.CoordinateSpace(names=names, scales=[1]*3, units=["mm"]*3)
        return ng.CoordinateSpaceTransform(
            matrix=ras2ras[:3],
            input_dimensions=dims,
            output_dimensions=dims,
        )

    if format == "lta":
        # Try my reader first
        # (nitransforms does not like nitorch's cropped LTAs)
        try:
            out = LinearTransformArray(fileobj).matrix()[0]
            out = _make_transform(out)
            LOG.debug(f'Succesfully read format "{format}".')
            return out
        except Exception as e:
            LOG.info(f'Tried format "{format}" (our impl) with no success.')
            LOG.debug(e)
            raise e

    if format:
        try:
            out = _read(AFFINE_FORMATMAP[format])
            LOG.debug(f'Succesfully read format "{format}".')
            return out
        except Exception as e:
            LOG.warning(f'Tried format "{format}" with no success.')
            LOG.debug(e)

    hint_format = format
    for format, klass in AFFINE_FORMATMAP.items():
        if format == hint_format:
            continue
        try:
            out = _read(klass)
            LOG.info(f'Succesfully read format "{format}".')
            return out
        except Exception as e:
            LOG.info(f'Tried format "{format}" with no success.')
            LOG.debug(e)

    if hint_format != "lta":
        # Try my reader last
        # (nitransforms does not like nitorch's cropped LTAs)
        try:
            out = LinearTransformArray(fileobj).matrix()[0]
            LOG.debug(f'Succesfully read format "{format}".')
            return out
        except Exception as e:
            LOG.warning(f'Tried format "{format}" (our impl) with no success.')
            LOG.debug(e)

    LOG.error(f"Failed to load {fileobj}.")
    raise RuntimeError(f"Failed to load {fileobj}.")


def to_square(affine: np.ndarray) -> np.ndarray:
    """Ensure an affine matrix is in square (homogeneous) form."""
    if affine.shape[0] == affine.shape[1]:
        return affine
    new_affine = np.eye(affine.shape[-1])
    new_affine[:-1, :] = affine
    return new_affine


def get_matrix(
    trf: ng.CoordinateSpaceTransform,
    square: bool = False
) -> np.ndarray:
    """Get (a copy of) the transfomation's matrix -- create one if needed."""
    rank = len(trf.input_dimensions.names)
    matrix = trf.matrix
    if matrix is None:
        matrix = np.eye(rank+1)[:-1]
    if square:
        matrix = to_square(np.copy(matrix))
    if not square and matrix.shape[0] == matrix.shape[1]:
        matrix = matrix[:-1]
    return matrix


def inverse(
    transform: ng.CoordinateSpaceTransform
) -> ng.CoordinateSpaceTransform:
    """Invert a transform."""
    # Invert normalized transform
    #
    #   In normalized units we have
    #
    #       [out] = linear @ [inp] + out_scale @ [offset]
    #
    #   yielding the inverse
    #
    #       [inp] = inv(linear) @ [out] - inv(linear) @ out_scale @ [offset]
    #
    #   Assuming that we reuse the same input/output dimensions, but swapped,
    #   we build the inverse transform by
    #
    #       linear = inv(linear)
    #       offset = - inv(inp_scale) @ inv(linear) @ out_scale @ [offset]
    #
    #   In our case, we've already ensured that inp_scale == out_scale == 1.

    matrix = get_matrix(transform, square=False)
    offset = matrix[:, -1:]
    linear = matrix[:, :-1]

    inp_scales = np.asarray([
        U.normalize_unit(x.scale, x.unit)
        for x in transform.input_dimensions
    ])[:, None]
    out_scales = np.asarray([
        U.normalize_unit(x.scale, x.unit)
        for x in transform.output_dimensions
    ])[:, None]

    inv_linear = np.linalg.inv(linear)
    inv_offset = -(inv_linear @ (offset * out_scales)) / inp_scales
    inv_matrix = np.concatenate([inv_linear, inv_offset], -1)

    inv_transform = ng.CoordinateSpaceTransform(
        input_dimensions=transform.output_dimensions,
        output_dimensions=transform.input_dimensions,
        matrix=inv_matrix
    )
    return inv_transform


def compose(
    *transforms,
    adapt: bool = False,
) -> ng.CoordinateSpaceTransform:
    """
    Compose two transforms.

    Parameters
    ----------
    *transforms : ng.CoordinateSpaceTransform
        Transforms to compose
    adapt : bool
        Try to adapt midspaces if they are different neurospaces.

    Returns
    -------
    composition : ng.CoordinateSpaceTransform
    """
    if len(transforms) == 0:
        return None
    if len(transforms) == 1:
        return transforms[0]
    left, right, *transforms = transforms
    if transforms:
        opt = dict(adapt=adapt)
        return compose(compose(left, right, **opt), *transforms, **opt)

    # ------------------------------------------------------------------
    # Adapt midspace
    #
    #   The left and right transforms may map from/to spaces that are
    #   compatible but different (for example, different neurospaces
    #   such as "ras" and "lpi"). We therefore convert the intermediate
    #   spaces (right output and left input) to the same neurospace
    #   (xyz == ras) first.
    if adapt:
        space_right = S.space_to_name(right.output_dimensions, compact=True)
        space_left = S.space_to_name(left.input_dimensions, compact=True)
        sorted_right = list(sorted(space_right))
        sorted_left = list(sorted(space_left))
        if sorted_right != ["x", "y", "z"][:len(sorted_right)]:
            right = compose(S.neurotransforms[("xyz", space_right)], right)
        if sorted_left != ["x", "y", "z"][:len(sorted_left)]:
            left = compose(left, S.neurotransforms[(space_left, "xyz")])

    # ------------------------------------------------------------------
    # Get parts
    li_dims = left.input_dimensions.to_json()
    lo_dims = left.output_dimensions.to_json()
    ri_dims = right.input_dimensions.to_json()
    ro_dims = right.output_dimensions.to_json()

    l_matrix = get_matrix(left, square=True)
    r_matrix = get_matrix(right, square=True)

    # ------------------------------------------------------------------
    # Filter left transform
    #
    #   Any input axis of the left transform that does not correspond
    #   to an output axis of the right transform must be removed.
    #
    #   This is because it is not possible to insert an dimension that
    #   did not exist in the input tensor in neuroglancer (AFAIK!).
    #
    #   However, the dimension that we drop should not be mixed in
    #   any way with the dimensions that are preserved. To check this we:
    #
    #       1. Find all output dimensions that depend (have nonzero weight)
    #          on one of the deleted dimensions.
    #       2. Check that each of these output dimensions _only_ depends
    #          (has non zero weight) on deleted dimensions.
    #
    #   In other words, if an output axis depends on both a deleted and
    #   a preserved axis, we have a problem.

    li_names = list(li_dims.keys())
    lo_names = list(lo_dims.keys())

    # List of input names to remove from left transform
    li_names_rm = [name for name in li_dims if name not in ro_dims]
    li_indcs_rm = [li_names.index(name) for name in li_names_rm]

    # Check mixing across removed and preserved axes
    lo_indcs_rm = set()
    for li_indx in li_indcs_rm:
        lo_indces = l_matrix[:-1, li_indx].nonzero()[0].tolist()
        for lo_indx in lo_indces:
            li_indces = l_matrix[lo_indx, :-1].nonzero()[0].tolist()
            if set(li_indces) - set(li_indcs_rm):
                raise RuntimeError(
                    "Transforms are not compatible. The left transform "
                    "mixes axes that are present in the right transform "
                    "with axes that are missing from the right transform."
                )
        lo_indcs_rm |= set(lo_indces)
    lo_names_rm = [lo_names[indx] for indx in lo_indcs_rm]

    # Delete rows and columns from matrix
    li_keep = [i for i in range(len(li_names)) if i not in li_indcs_rm]
    lo_keep = [i for i in range(len(lo_names)) if i not in lo_indcs_rm]
    l_matrix = l_matrix[li_keep + [-1], :][:, lo_keep + [-1]]

    # Filter dimensions
    li_dims = dict(filter(lambda x: x[0] not in li_names_rm, li_dims.items()))
    lo_dims = dict(filter(lambda x: x[0] not in lo_names_rm, lo_dims.items()))

    # ------------------------------------------------------------------
    # Populate left transform
    #
    #   We also need to insert into the left transform an identity
    #   sub-transform that corresponds to axes that are present in the
    #   output dimensions of the right axis but not in the input dimensions
    #   of the left transform.

    li_names = list(li_dims.keys())
    ro_names = list(ro_dims.keys())

    # Find axes that exist in the input space but not in the output space
    li_names_add = [name for name in ro_dims if name not in li_dims]

    # Check that these axis names do not already exist in the ouput
    # dimensions of the right matrix.
    if any(name in lo_dims for name in li_names_add):
        raise RuntimeError(
            "Transforms are not compatible. The left transform defines "
            "an output axis whose name conflicts with an output axis "
            "of the right transform."
        )

    # Add axes to the input/output dimensions of the right transform
    li_dims.update({name: ro_dims[name] for name in li_names_add})
    lo_dims.update({name: ro_dims[name] for name in li_names_add})

    # Insert identity matrix in the left matrix
    l_ndims = len(l_matrix) - 1
    r_ndims = len(r_matrix) - 1
    l_matrix0, l_matrix = l_matrix, np.eye(r_ndims+1)
    # -> copy compatible part
    l_matrix[:l_ndims, :l_ndims] = l_matrix0[:-1, :-1]
    l_matrix[:l_ndims, -1:] = l_matrix0[:-1, -1:]

    # Reorder right side of the matrix to conform to the right matrix
    li_names = list(li_dims.keys())
    l_matrix = l_matrix[:, [li_names.index(n) for n in ro_names] + [-1]]

    # ------------------------------------------------------------------
    # Compose matrices
    #
    #   A voxel to world matrix is constructed from a ng transform by:
    #
    #   [ x ]   [ 1/vx     ]   [ Axi Axj Axk ]   [ sx       ]   [ i ]   [ Tx ]
    #   [ y ] = [   1/vy   ] * [ Ayi Ayj Ayk ] * [    sy    ] * [ j ] + [ Ty ]
    #   [ z ]   [     1/vz ]   [ Azi Azj Azk ]   [       sz ]   [ k ]   [ Tz ]
    #
    #   where `v` is the view/output voxel size and `s` is the source/input
    #   voxel size. In this context, `[x, y, z]` is in "view" scaled unit and
    #   `[i, j, k]` is in "source" scaled unit.
    #
    #   If we're working in unscaled SI units, we instead have
    #
    #   [ x ]   [ Axi Axj Axk ]   [ i ]   [ vx       ]   [ Tx ]
    #   [ y ] = [ Ayi Ayj Ayk ] * [ j ] + [    vy    ] * [ Ty ]
    #   [ z ]   [ Azi Azj Azk ]   [ k ]   [       vz ]   [ Tz ]
    #
    #   where both `[x, y, z]` and `[i, j, k]` are in unscaled SI units
    #   (i.e., "m" or "s").
    #
    #   To compose two transforms
    #
    #   [mid] = rlinear @ [inp] + rout_scale @ [roffset]
    #   [out] = llinear @ [mid] + lout_scale @ [loffset]
    #
    #   where `out`, `inp` and `mid` are in unscaled SI units,
    #   one therefore does
    #
    #   [out] = llinear @ rlinear    @ [inp]
    #         + llinear @ rout_scale @ [roffset]
    #         +           lout_scale @ [loffset]
    #
    #   assuming the the input scale of the composed transform is `rinp_scale`
    #   and its output scale is `lout_scale`, the matrix is therefore
    #   constructed by
    #
    #   linear = llinear @ rlinear
    #   offset = loffset + inv(lout_scale) @ llinear @ rout_scale @ roffset

    l_linear = l_matrix[:-1, :-1]
    r_linear = r_matrix[:-1, :-1]
    l_offset = l_matrix[:-1, -1:]
    r_offset = r_matrix[:-1, -1:]

    ro_scles = np.asarray([
        U.normalize_unit(scale, unit)
        for (scale, unit) in ro_dims.values()
    ])[:, None]
    lo_scles = np.asarray([
        U.normalize_unit(scale, unit)
        for (scale, unit) in lo_dims.values()
    ])[:, None]

    matrix = np.eye(len(l_matrix))
    matrix[:-1, :-1] = l_linear @ r_linear
    matrix[:-1, -1:] = l_offset + (l_linear @ (ro_scles * r_offset)) / lo_scles

    return ng.CoordinateSpaceTransform(
        input_dimensions=ng.CoordinateSpace(ri_dims),
        output_dimensions=ng.CoordinateSpace(lo_dims),
        matrix=matrix[:-1],
    )


AxisConversionSpecification = (
    ng.DimensionScale |
    str | list[str] |                               # unit(s)
    tuple[float, str] | list[tuple[float, str]]     # scale(s) + unit(s)
)
SpaceConversionSpecification = (
    ng.CoordinateSpace |
    AxisConversionSpecification |
    dict[str, AxisConversionSpecification]
)


def convert_transform(
    transform: ng.CoordinateSpaceTransform,
    input: SpaceConversionSpecification | None = None,
    output: SpaceConversionSpecification | None = None,
) -> ng.CoordinateSpaceTransform:
    """
    Convert units of a transform.

    !!! example
        === "A"
            __Ensure that all axes have SI unit (without prefix)__
            ```python
            new_space = convert_transform(space, "base", "base")
            # or
            base = ["m", "s", "hz", "rad/s"]
            new_space = convert_space(space, base, base)
            # or
            new_space = normalize_transform(space)
            ```
        === "B"
            __Ensure that all axes have unit-scale SI unit (without prefix)__
            ```python
            base = [[1, "m"], [1, "s"], [1, "hz"], [1, "rad/s"]]
            new_space = convert_space(space, base, base)
            # or
            new_space = normalize_transform(space, unit_scale=True)
            ```
        === "C"
            __Ensure that input spatial axes have millimeter unit__
            ```python
            new_space = convert_space(space, "mm")
            # or
            new_space = convert_space(space, "mm", None)
            ```
        === "D"
            __Ensure that all spatial axes have millimeter unit__
            ```python
            new_space = convert_space(space, "mm", "mm")
            ```
        === "E"
            __Ensure that specific input axes have specific unit__
            ```python
            new_space = convert_space(space, {"x": "mm", "y": "mm", "t": "ms"})
            # or
            new_space = convert_space(space, {("x", "y"): "mm", "t": "ms"})
            # or
            new_space = convert_space(space, {("x", "y", "t"): ["mm", "ms"]})
            ```
        === "F"
            __Ensure that specific input axes have specific scales__
            ```python
            new_space = convert_space(space, {"x": [2, "mm"], "y": [2, "mm"], "t": [3, "ms"]})
            # or
            new_space = convert_space(space, {("x", "y"): [2, "mm"], "t": [3, "ms"]})
            # or
            new_space = convert_space(space, {("x", "y", "t"): [[2, "mm"], [3, "ms"]]})
            # or
            new_space = convert_space(space,
            ```

    !!! info "See also: `normalize_transform`."

    Parameters
    ----------
    transform : ng.CoordinateSpaceTransform
        Transform
    input : ng.CoordinateSpace | str | tuple[float, str] | list | dict | None
        Either a coordinate space to use in-place of the existing one,
        or a new (list of) unit(s), or a mapping from axis name(s) to
        new unit(s).
    output : ng.CoordinateSpace | str | tuple[float, str] | list | dict | None
        Either a coordinate space to use in-place of the existing one,
        or a new (list of) unit(s), or a mapping from axis name(s) to
        new unit(s).

    Returns
    -------
    transform : ng.CoordinateSpaceTransform
    """  # noqa: E501
    rank = len(transform.input_dimensions.names)
    old_input: ng.CoordinateSpace = transform.input_dimensions
    old_output: ng.CoordinateSpace = transform.output_dimensions

    def get_spec_unit(
        spec: SpaceConversionSpecification
    ) -> dict[str, float]:
        """Extract the unit part of the specification."""
        if isinstance(spec, (list, tuple)) \
                and not isinstance(spec, ng.DimensionScale):
            if isinstance(spec[0], Number):
                spec = ng.DimensionScale.from_json(list(spec))
            else:
                return [get_spec_unit(spec1) for spec1 in spec]
        if isinstance(spec, ng.DimensionScale):
            return spec.unit
        if isinstance(spec, str):
            return spec
        if isinstance(spec, dict):
            return {
                name1: get_spec_unit(spec1)
                for name1, spec1 in spec.items()
            }
        raise TypeError("Unknown specification type:", type(spec))

    def apply_spec_scale(
        space: ng.CoordinateSpace,
        spec: SpaceConversionSpecification,
        names: list[str] | None = None,
    ) -> ng.CoordinateSpace:
        """Apply the scaling part of the specification."""
        if names is None:
            names = space.names
        if isinstance(spec, (list, tuple)) \
                and not isinstance(spec, ng.DimensionScale):
            if isinstance(spec[0], Number):
                spec = ng.DimensionScale.from_json(list(spec))
            else:
                for spec1 in spec:
                    space = apply_spec_scale(space, spec1)
                return space
        if isinstance(spec, ng.DimensionScale):
            space = space.to_json()
            for name in names:
                if U.same_unit_kind(space[name][1], spec.unit):
                    space[name] = [spec.scale, spec.unit]
            space = ng.CoordinateSpace(space)
            return space
        if isinstance(spec, str):
            return space
        if isinstance(spec, dict):
            for name, spec1 in spec.items():
                space = apply_spec_scale(space, spec1, _ensure_list(name))
            return space
        raise TypeError("Unknown specification type:", type(spec))

    input = input or old_input
    if not isinstance(input, ng.CoordinateSpace):
        spec = input
        old_input = ng.CoordinateSpace(old_input.to_json())  # copy
        input = S.convert_space(old_input, get_spec_unit(spec))
        input = apply_spec_scale(input, spec)

    output = output or old_output
    if not isinstance(output, ng.CoordinateSpace):
        spec = output
        old_output = ng.CoordinateSpace(old_output.to_json())  # copy
        output = S.convert_space(old_output, get_spec_unit(spec))
        output = apply_spec_scale(output, spec)

    matrix = get_matrix(transform, square=True)

    for i in range(rank):
        # input (scale columns of the linear component)
        old_scale = old_input.scales[i]
        new_scale = input.scales[i]
        old_unit = old_input.units[i]
        new_unit = input.units[i]
        old_scale = U.convert_unit(old_scale, old_unit, new_unit)
        matrix[:-1, i] *= (old_scale / new_scale)
        # output (scale the translation)
        old_scale = old_output.scales[i]
        old_unit = old_output.units[i]
        new_scale = output.scales[i]
        new_unit = output.units[i]
        old_scale = U.convert_unit(old_scale, old_unit, new_unit)
        matrix[i, -1] *= (old_scale / new_scale)

    return ng.CoordinateSpaceTransform(
        input_dimensions=input,
        output_dimensions=output,
        matrix=matrix[:-1, :],
    )


def normalize_transform(
    transform: ng.CoordinateSpaceTransform,
    unit_scale: bool = False,
) -> ng.CoordinateSpaceTransform:
    """Ensure that input and output spaces have SI units.

    !!! info "See also: `convert_transform`, `ensure_same_scale`."

    Parameters
    ----------
    transform : ng.CoordinateSpaceTransform
        Transformation
    unit_scale : bool
        If `True`, make input and output space to have unit scale.
        Otherwise, simply convert them to SI unit.

    Returns
    -------
    transform : ng.CoordinateSpaceTransform
        Converted transformation
    """
    spec = ["m", "s", "hz", "rad/s"]
    if unit_scale:
        spec = [(1, unit) for unit in spec]
    return convert_transform(transform, spec, spec)


def ensure_same_scale(
    transform: ng.CoordinateSpaceTransform
) -> ng.CoordinateSpaceTransform:
    """
    Ensure that all axes and both input and output spaces use the same
    unit and the same scaling.

    This function is quite related to `normalize_transform(*, unit_scale=True)`
    except that the units closest to input units are used, instead of
    prefixless SI units.

    !!! info "See also: `normalize_transform`."
    """
    # Find all scales and prefixes in the input/output spaces
    #   The `unitmap` dictionary will map unit kinds present in the
    #   transform's dimensions to a list of possible exponents.
    unitmap = {}
    for dimensions in (transform.outputDimensions, transform.inputDimensions):
        dimensions: ng.CoordinateSpace
        for unit, scale in zip(dimensions.units, dimensions.scales):
            prefix, kind = U.split_unit(unit)
            exponent = U.SI_PREFIX_EXPONENT[prefix]
            exponent += int(round(math.log10(scale)))
            unitmap.setdefault(kind, [])
            unitmap[kind].append(exponent)

    # Choose the unit and scale that are closest/most common
    #   The `scalemap` dictionary will map unit kinds present in the
    #   transform's dimensions to a list of (scale, unit) to convert to.
    scalemap = {}
    for kind, exponents in dict(unitmap).items():
        # find most common exponent
        exponent, count = None, 0
        for maybe_exponent in set(exponents):
            if exponents.count(maybe_exponent) > count:
                exponent = maybe_exponent
        # find SI prefix closest to exponent and distance to exponent
        prefix, dist = None, float("inf")
        for maybe_prefix, maybe_exponent in U.SI_PREFIX_EXPONENT.items():
            if abs(exponent - maybe_exponent) < abs(dist):
                dist = exponent - maybe_exponent
                prefix = maybe_prefix
        # map unit kinf
        scalemap[kind] = [10 ** dist, prefix + kind]

    # convert the transform
    for kind, scale in scalemap.items():
        spec = ng.DimensionScale(*scale)
        transform = convert_transform(transform, spec, spec)
    return transform


def subtransform(
    transform: ng.CoordinateSpaceTransform,
    unit: str
) -> ng.CoordinateSpaceTransform:
    """
    Generate a subtransform by keeping only axes of a certain unit kind.

    Parameters
    ----------
    transform : ng.CoordinateSpaceTransform
        Transformation
    unit : str
        Unit kind to keep.

    Returns
    -------
    transform : ng.CoordinateSpaceTransform
        Converted transformation

    """
    kind = U.split_unit(unit)[1]
    idim: ng.CoordinateSpace = transform.input_dimensions
    odim: ng.CoordinateSpace = transform.output_dimensions
    matrix = transform.matrix
    if matrix is None:
        matrix = np.eye(len(idim.names)+1)[:-1]
    # filter input axes
    ikeep = []
    for i, unit in enumerate(idim.units):
        unit = U.split_unit(unit)[1]
        if unit == kind:
            ikeep.append(i)
    idim = ng.CoordinateSpace({
        idim.names[i]: [idim.scales[i], idim.units[i]]
        for i in ikeep
    })
    matrix = matrix[:, ikeep + [-1]]
    # filter output axes
    okeep = []
    for i, unit in enumerate(odim.units):
        unit = U.split_unit(unit)[1]
        if unit == kind:
            okeep.append(i)
    odim = ng.CoordinateSpace({
        odim.names[i]: [odim.scales[i], odim.units[i]]
        for i in okeep
    })
    matrix = matrix[okeep, :]

    return ng.CoordinateSpaceTransform(
        matrix=matrix,
        input_dimensions=idim,
        output_dimensions=odim,
    )


def _ensure_list(x: object, n: int | None = None, **kwargs) -> list:
    if isinstance(x, np.ndarray):
        x = x.tolist()
    if not isinstance(x, (list, tuple)):
        x = [x]
    x = list(x)
    if n is not None:
        if len(x) < n:
            default = kwargs.get("default", x[-1])
            x += [default] * (n - len(x))
        elif len(x) > n:
            x = x[:n]
    return x


def _ensure_tuple(x: object, n: int | None = None, **kwargs) -> tuple:
    return tuple(_ensure_list(x, n, **kwargs))
