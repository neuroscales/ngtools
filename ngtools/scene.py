"""A neuroglancer scene with a programmatic interface."""
# stdlib
import functools
import json
import logging
import sys
from io import BytesIO
from os import PathLike
from typing import Literal, Sequence
from urllib.parse import quote as urlquote
from urllib.parse import unquote as urlunquote
from urllib.parse import urlparse

# externals
import neuroglancer as ng
import numpy as np
from neuroglancer.viewer_state import wrapped_property
from numpy.typing import ArrayLike
from upath import UPath

# import to trigger datasource registration
import ngtools.local.datasources  # noqa: F401
import ngtools.local.tracts  # noqa: F401

# internals
import ngtools.spaces as S
import ngtools.transforms as T
from ngtools.datasources import LayerDataSource, LayerDataSources
from ngtools.layers import Layer, Layers
from ngtools.local.iostream import StandardIO
from ngtools.opener import exists, open, parse_protocols
from ngtools.shaders import colormaps, shaders
from ngtools.units import convert_unit, split_unit
from ngtools.utils import DEFAULT_URL, Wraps

# monkey-patch Layer state to expose channelDimensions
ng.Layer.channel_dimensions = ng.Layer.channelDimensions = wrapped_property(
    "channelDimensions", ng.CoordinateSpace
)
ng.Layer.local_dimensions = ng.Layer.localDimensions = ng.Layer.layerDimensions


LOG = logging.getLogger(__name__)


def _ensure_list(x: object) -> list:
    """Ensure that an object is a list. Make one if needed."""
    if isinstance(x, np.ndarray):
        x = x.tolist()
    if not isinstance(x, (list, tuple)):
        x = [x]
    return list(x)


URILike = str | PathLike
SourceType = ng.LayerDataSource | ng.LocalVolume | ng.skeleton.SkeletonSource


class ViewerState(Wraps(ng.ViewerState)):
    """Smart ng.ViewerState that knows default values set in the frontend."""

    def __get_layers(self) -> Layers:
        return Layers(getattr(super(), "layers"))

    def _default_dimensions(self) -> ng.CoordinateSpace:
        dims = {}
        for layer in self.layers:
            layer: ng.ManagedLayer
            if layer.name.startswith("__"):
                continue
            layer = layer.layer
            source = LayerDataSources(layer.source)
            transform = source[0].transform
            odims = transform.output_dimensions
            dims.update({
                name: [1, split_unit(unit)[1]]
                for name, (_, unit) in odims.to_json().items()
                if not name.endswith(("^", "'"))
            })
        return dims

    def __set_dimensions(
        self, value: ng.CoordinateSpace | None
    ) -> ng.CoordinateSpace:
        default_value = self._default_dimensions()
        if value is None:
            value = default_value
        value = value.to_json()
        default_value = default_value.to_json()
        value = {
            name: val for name, val in value.to_json()
            if name in default_value
        }
        value.update({
            name: val for name, val in default_value.items()
            if name not in value
        })
        return value

    def __get_dimensions(self) -> ng.CoordinateSpace:
        """All non-local dimensions."""
        self.dimensions = getattr(super(), "dimensions")
        return getattr(super(), "dimensions")

    @property
    def spatial_dimensions(self) -> ng.CoordinateSpace:
        """All spatial dimensions (with meter-like unit)."""
        return ng.CoordinateSpace({
            key: [scale, unit]
            for key, (scale, unit) in self.dimensions.to_json()
            if unit[-1:] == "m"
        })

    @property
    def _space(self) -> str:
        """Current space."""
        return "".join(x[:1].lower() for x in self.spatial_dimensions.names)

    def __set_relative_display_scales(
        self, value: dict[str, float] | None
    ) -> dict[str, float]:
        dimensions = self.dimensions
        value = value or {}
        value = {
            name: val for name, val in value.items()
            if name in dimensions
        }
        value.update({
            name: 1.0 for name in self.dimensions
            if name not in value
        })
        return value

    def __get_relative_display_scales(self) -> dict[str, float]:
        """Relative display scales."""
        self.relative_display_scales \
            = getattr(super(), "relative_display_scales")
        return getattr(super(), "relative_display_scales")

    __get_relativeDisplayScales = __get_relative_display_scales

    # def __get_display_dimensions(self) -> list[str]:
    #     """Name of (up to three) displayed dimensions."""
    #     self.display_dimensions \
    #         = super().__getattribute__("display_dimensions")
    #     return super().__getattribute__("display_dimensions")

    def __set_display_dimensions(self, value: list[str] | None) -> None:
        dimensions = self.dimensions.items()
        if value is None:
            value = []
        value = [name for name in value if name in dimensions]
        dimensions = [name for name in dimensions if name not in value]
        value = (value + dimensions)[:3]
        return value

    # def __get_cross_section_orientation(self) -> np.ndarray:
    #     """Orientation of the cross section view."""
    #     self.cross_section_orientation \
    #         = super().__getattribute__("cross_section_orientation")
    #     return super().__getattribute__("cross_section_orientation")

    # def __set_cross_section_orientation(self, value: ArrayLike) -> None:
    #     if value is None:
    #         value = [0, 0, 0, 1]
    #     value = np.ndarray(value).tolist()
    #     value = value + max(0, 4 - len(value)) * [0]
    #     value = value[:4]
    #     value = np.ndarray(value)
    #     value /= (value**2).sum()**0.5
    #     super().cross_section_orientation = value

    # __get_crossSectionOrientation = __get_cross_section_orientation
    # __set_crossSectionOrientation = __set_cross_section_orientation

    # def _default_position(self) -> list[float]:
    #     """
    #     Compute a smart default position (center of the fist).

    #     NOTE: positions are expressed in "model scaled space". That is,
    #     it the "z" dimension listed in `dimensions` has scale (0.5, "mm"),
    #     a position increment of 1 will correspond to an effective increment
    #     of 0.5 mm.
    #     """
    #     pos = [0.0] * len(self.dimensions.names)
    #     for layer in self.layers:
    #         layer: ng.ManagedLayer
    #         if not layer.visible:
    #             continue
    #         layer = layer.layer
    #         if getattr(layer, "source", []) == 0:
    #             continue
    #         source = layer.source[0]
    #         if not hasattr(source, "output_center"):
    #             continue
    #         center = source.output_center
    #         for i, (name, (scale, unit)) \
    #                 in enumerate(self.dimensions.to_json().items()):
    #             if name not in source.output_dimensions.names:
    #                 continue
    #             j = source.output_dimensions.names.index(name)
    #             unit0 = source.output_dimensions.units[j]
    #             value = convert_unit(center[j], unit0, unit)
    #             pos[i] = value / scale
    #     return pos

    # def __get_position(self) -> list[float]:
    #     if not super().__getattribute__("position"):
    #         self.position = self._default_position()
    #     return super().__getattribute__("position")


def autolog(func: callable) -> callable:
    """Decorate a function to automatically log its usage."""

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):  # noqa: ANN001, ANN202
        if LOG.level <= logging.DEBUG or self.stdio._level <= logging.DEBUG:
            pargs = ", ".join(
                f'"{arg}"' if isinstance(arg, str) else str(arg)
                for arg in args
            )
            pkwargs = ", ".join(
                f"{key}=" + f'"{val}"' if isinstance(val, str) else str(val)
                for key, val in kwargs.items()
            )
            if not pargs:
                pargskwargs = pkwargs
            elif not pkwargs:
                pargskwargs = pargs
            else:
                pargskwargs = pargs + ", " + pkwargs
            self.stdio.debug(f"{func.__name__}({pargskwargs})")
        try:
            return func(self, *args, **kwargs)
        except Exception as e:
            if (
                LOG.level <= logging.DEBUG or
                self.stdio._level <= logging.DEBUG
            ):
                self.stdio.debug(str(e))
            raise e

    return wrapper


class Scene(ViewerState):
    """A neuroglancer scene with a programmatic interface."""

    def __init__(self, *args, **kwargs) -> None:
        """

        Other Parameters
        ----------------
        stdout : TextIO | str
            Output stream.
        stderr : TextIO  | str
            Error stream.
        level : {"debug", "info", "warning", "error", "any"} | int | None
            Level of printing.
            * If None:      no printing
            * If "error":   print errors
            * If "warning": print errors and warnings
            * If "info":    print errors, warnings and infos
            * If "debug":   print errors, warnings, infos and debug messages.
            * If "any:      print any message.
        """
        stdio = kwargs.pop("stdio", None)
        if stdio is None:
            stdin = kwargs.pop("stdout", sys.stdin)
            stdout = kwargs.pop("stdout", sys.stdout)
            stderr = kwargs.pop("stderr", sys.stderr)
            level = kwargs.pop("level", "info")
            stdio = StandardIO(
                stdin=stdin, stdout=stdout, stderr=stderr,
                level=level, logger=LOG
            )
        super().__init__(*args, **kwargs)
        self.stdio = stdio

    @autolog
    def load(
        self,
        uri: URILike | list[URILike] | dict[str, URILike],
        transform: ArrayLike | list[float] | URILike | None = None,
        **kwargs
    ) -> None:
        """
        Load file(s).

        Parameters
        ----------
        uri : str | list[str] | dict[str, str]
            Paths or URL, eventually prepended with "type" and "format"
            protocols. Ex: `"labels://nifti://http://cloud.com/path/to/nii.gz"`
            If a dictionary, maps layer names to file names.
        transform : array_like | list[float] | str | None
            Affine transform to apply to the loaded volume.

        Other Parameters
        ----------------
        name : str | list[str]
            Alternative way of providing layer names.
            If used, `uri` cannot be a `dict`.
        """
        # prepare names and URLs
        names = []
        if isinstance(uri, dict):
            names = list(uri.keys())
            uri = list(uri.values())
        else:
            names = kwargs.pop("name", [])
        uris = _ensure_list(uri or [])
        names = _ensure_list(names or [])

        # nb_layers_0 = len(self.layers)

        # load layers
        onames = []
        for n, uri in enumerate(uris):
            uri = str(uri).rstrip("/")
            short_uri = parse_protocols(uri)[-1]
            name = names[n] if names else UPath(short_uri).name
            onames.append(name)
            layer = Layer(uri, **kwargs)
            self.layers.append(name=name, layer=layer)

        # rename axes according to current naming scheme
        self.rename_axes(self.world_axes(), layer=onames)

        # apply transform
        if transform is not None:
            self.transform(transform, layer=onames)

        # if nb_layers_0 == 0:
        #     self.position = self._default_position()

    @autolog
    def unload(
        self,
        layer: str | list[str] | None = None,
    ) -> None:
        """
        Unload layers.

        Parameters
        ----------
        layer : str or list[str]
            Layer(s) to unload
        """
        layers = layer or [layer.name for layer in self.layers]
        for name in _ensure_list(layers):
            del self.layers[name]

    @autolog
    def rename(self, src: str, dst: str) -> None:
        """
        Rename a layer.

        Parameters
        ----------
        src : str
            Current name
        dst : str
            New name
        """
        for layer in self.layers:
            layer: ng.ManagedLayer
            if layer.name == src:
                layer.name = dst
                return
        raise ValueError('No layer named', src)

    @autolog
    def world_axes(
        self, axes: dict[str, str] | list[str] | str | None = None,
        **kwargs,
    ) -> dict[str, str]:
        """
        Map native model axes (`"x"`, `"y"`, `"z"`, `"t"`) to
        neuroanatomical or arbitrary names.

        Parameters
        ----------
        axes : dict[str, str] | list[str] | str | None
            Mapping from native to user names.
            If None, simply return current mapping.

        Other Parameters
        ----------------
        src, dst : str | list[str]
            Native/New axes names. If used, `axes` cannot be used.

        Returns
        -------
        axes : dict[str, str]
            Mapping from native to user names.
        """
        def make_local_annot() -> ng.AnnotationLayer:
            coord = ng.CoordinateSpace()
            return ng.AnnotationLayer(ng.LocalAnnotationLayer(coord))

        # check if keywords were used
        if "dst" in kwargs:
            if axes is not None:
                msg = "Cannot use `axes` if `src` is used."
                self.stdio.error(msg)
                raise ValueError(msg)
            axes = _ensure_list(kwargs["dst"])
        if "src" in kwargs:
            if isinstance(axes, dict):
                msg = "Cannot use `axes` if `dst` is used."
                self.stdio.error(msg)
                raise ValueError(msg)
            if axes is None:
                msg = "Missing user-defined names for axes."
                self.stdio.error(msg)
                raise ValueError(msg)
            native = _ensure_list(kwargs["src"])
            axes = {src: dst for src, dst in zip(native, axes)}

        # We save the mapping using the output dimensions of two
        # different local annotation layers.
        if "__world_axes_native__" not in self.layers:
            self.layers.append(ng.ManagedLayer(
                name="__world_axes_native__",
                layer=make_local_annot(),
                archived=True
            ))
        if "__world_axes_current__" not in self.layers:
            self.layers.append(ng.ManagedLayer(
                name="__world_axes_current__",
                layer=make_local_annot(),
                archived=True
            ))
        world_axes_native = self.layers["__world_axes_native__"]
        world_axes_current = self.layers["__world_axes_current__"]

        old_axes = {"x": "x", "y": "y", "z": "z", "t": "t"}
        old_axes.update({
            native: current
            for native, current in zip(
                world_axes_native.source[0].transform.output_dimensions.names,
                world_axes_current.source[0].transform.output_dimensions.names
            )
        })

        if not axes:
            return old_axes

        new_axes = axes
        axes = dict(old_axes)

        old2nat = {v: k for k, v in old_axes.items()}
        self.rename_axes(old2nat)

        if isinstance(new_axes, str):
            new_axes = S.name_compact2full(new_axes)
        if isinstance(new_axes, (list, tuple)):
            new_axes = {
                native: new_axis
                for native, new_axis in zip("xyz", new_axes)
            }
        axes.update(new_axes)

        if any(axis in axes.values() and axes[axis] != axis for axis in axes):
            raise ValueError(
                "Renaming scheme cannot involve permuting native axes."
            )

        world_axes_native.source[0].transform = ng.CoordinateSpaceTransform(
            output_dimensions=ng.CoordinateSpace({
                name: [1, ""] for name in axes.keys()
            })
        )
        world_axes_current.source[0].transform = ng.CoordinateSpaceTransform(
            output_dimensions=ng.CoordinateSpace({
                name: [1, ""] for name in axes.values()
            })
        )

        self.rename_axes(axes)
        return axes

    @autolog
    def rename_axes(
        self,
        axes: str | list[str] | dict[str],
        layer: str | list[str] | None = None,
        **kwargs
    ) -> dict[str]:
        """
        Rename world axes.

        Parameters
        ----------
        axes : str | list[str] | dict[str]
            A dictionary that maps old names to new names, or a known
            neurospace (_e.g._, `"xyz", `["x", "y", "z"]`, `"ras"`,
            `["right", "anterior", "posterior"]`)
        layer : str | list[str] | None
            Layers to rename. By default, all.

        Other Parameters
        ----------------
        src, dst : str | list[str]
            Old/New axes names. If used, `axes` cannot be used.

        Returns
        -------
        axes : dict[str]
            Mapping from old names to new names
        """
        # check if keywords were used
        if "dst" in kwargs:
            if axes is not None:
                msg = "Cannot use `axes` if `src` is used."
                self.stdio.error(msg)
                raise ValueError(msg)
            axes = _ensure_list(kwargs["dst"])
        if "src" in kwargs:
            if isinstance(axes, dict):
                msg = "Cannot use `axes` if `dst` is used."
                self.stdio.error(msg)
                raise ValueError(msg)
            if axes is None:
                msg = "Missing user-defined names for axes."
                self.stdio.error(msg)
                raise ValueError(msg)
            native = _ensure_list(kwargs["src"])
            axes = {src: dst for src, dst in zip(native, axes)}

        # format axis names
        if isinstance(axes, str):
            axes = S.name_compact2full(axes)
        if isinstance(axes, (list, tuple)):
            model_axes = self.world_axes()
            axes = {
                model_axes[native_axis]: new_axis
                for native_axis, new_axis in zip("xyz", axes)
            }

        layers = _ensure_list(layer)
        for named_layer in self.layers:
            if layers and named_layer.name not in layers:
                continue
            layer = named_layer.layer
            transform = layer.source[0].transform
            transform.output_dimensions = ng.CoordinateSpace({
                axes.get(name): scl
                for name, scl in transform.output_dimensions.to_json().items()
            })
            layer.source[0].transform = transform

        return axes

    @autolog
    def space(
        self,
        mode: Literal["radio", "neuro"] = "radio",
        layer: str | None = None
    ) -> str:
        """
        Rotate the view such that the cross-sections are aligned with
        the voxel axes of a layer. Also, switch between radiological and
        neurological orientation.

        Parameters
        ----------
        mode
            Neurological (patient's point of view) or radiological
            (facing the patient) orientation.
        layer
            Name of a layer or `"world"`.
        """
        neuro = np.asarray([0, -1, 1, 0])
        radio = np.asarray([1, 0, 0, -1])
        current = np.asarray(self.cross_section_orientation or [0, 0, 0, 1])
        current_mode = ["neuro", "radio"][int(
            np.linalg.norm(current - neuro)
            <
            np.linalg.norm(current - radio)
        )]

        # If first input is a layer name, switch
        if layer is None and mode in self.layers:
            layer, mode = mode, None

        # If no input arguments, simply return known dimensions
        if mode is None and layer is None:
            self.stdio.info(current_mode)
            return current_mode
        mode = mode or current_mode

        if layer:
            if layer not in self.layers:
                raise ValueError("No layer named:", layer)

            #   1. Get voxel2world matrix
            source = LayerDataSource(self.layers[layer].source[0])
            transform = T.subtransform(source.transform, 'm')
            transform = T.ensure_same_scale(transform)
            matrix = T.get_matrix(transform, square=True)[:-1, :-1]
            #   2.remove scales and shears
            u, _, vh = np.linalg.svd(matrix)
            rot = u @ vh
            #   3. preserve permutations and flips
            #      > may not work if exactly 45 deg rotation so add a tiny
            #        bit of noise
            eps = np.random.randn(*rot.shape) * 1E-8 * rot.abs().max()
            orient = (rot + eps).round()
            assert np.allclose(orient @ orient.T, np.eye(len(orient)))
            rot = rot @ orient.T
            if np.linalg.det(rot) < 0:
                # flip left-right
                rot[0, :] *= -1
            to_layer = rot.T
            to_layer = T.matrix_to_quaternion(rot)
        else:
            to_layer = [0, 0, 0, 1]  # identity

        # canonical neuro/radio axes
        to_view = neuro if mode[0].lower() == "n" else radio
        to_layer = T.compose_quaternions(to_view, to_layer)

        # TODO
        # At some point I wanted to show data in the layer voxel space
        # closest to the current view, but it's a bit more difficult
        # than I thought (I need to compose the world2voxel rotation
        # with the world2view transform -- but the latter depends
        # on the displayed dimension so I'll need to do some filtering/
        # permutation). I might come back to it at some point. In the
        # meantime I always reset the displayed dimensions to [x, y, z].

        # Display spatial dimensions (reorder if needed)
        display_dimensions_order = [
            "x", "r", "right", "y", "a", "anterior", "z", "s", "superior"
        ]
        display_dimensions = self.spatial_dimensions.names
        display_dimensions = list(sorted(
            display_dimensions,
            key=lambda x: (
                display_dimensions_order.index(x)
                if x in display_dimensions_order else float('inf')
            )
        ))
        self.display_dimensions = display_dimensions

        # Set transform
        self.cross_section_orientation = to_layer.tolist()
        return mode

    @autolog
    def display(
        self,
        dimensions: str | list[str] | None = None,
    ) -> None:
        """
        Change displayed dimensions and/or change neurospace.

        Parameters
        ----------
        dimensions : str or list[str]
            The three dimensions to display.
            Each dimension can be one of:

            * `"left"` or `"right"`
            * `"posterior"` or `"anterior"`
            * `"inferior"` or `"superior"`

            Otherwise, dimensions can be native axis names of the loaded
            data, such as `"x"`, `"y"`, `"z"`.

            A compact representation (`"RAS"`, or `"zyx"`) can also be
            provided.
        """
        dimensions = _ensure_list(dimensions)
        if len(dimensions) > 3:
            raise ValueError('display takes at most three axis names')
        if len(dimensions) == 1:
            dimensions = dimensions[0]
        if isinstance(dimensions, str):
            dimensions = [S.letter2full.get(letter.lower(), letter)
                          for letter in dimensions]
        self.display_dimensions = dimensions

    _TransformLike = (
        ng.CoordinateSpaceTransform | ArrayLike | str | PathLike | BytesIO
    )

    @autolog
    def transform(
        self,
        transform: _TransformLike,
        layer: str | list[str] | None = None,
        inv: bool = False,
        *,
        mov: str = None,
        fix: str = None,
    ) -> None:
        """
        Apply an affine transform.

        Parameters
        ----------
        transform : list[float] or np.ndarray or fileobj
            Affine transform (RAS+)
        layer : str or list[str]
            Layer(s) to transform
        inv : bool
            Invert the transform

        Other Parameters
        ----------------
        mov : str
            Moving/Floating image (required by some affine formats)
        fix : str
            Fixed/Reference image (required by some affine formats)
        """
        # save current axes
        display_dimensions = self.display_dimensions

        # rename axes to xyz
        world_axes = self.world_axes()
        self.world_axes({"x": "x", "y": "y", "z": "z"})

        # prepare transformation matrix
        transform = self._load_transform(transform, inv, mov=mov, fix=fix)

        # apply transform
        self._apply_transform(transform, layer)

        # go back to original axis names
        self.world_axes(world_axes)
        self.display(display_dimensions)

    def _apply_transform(
        self,
        transform: ng.CoordinateSpaceTransform,
        layer: str | list[str] | None = None,
        adapt: bool = False,
    ) -> None:
        """
        Apply a transform to one or more layers.

        This function expects that the output dimensions of the layer(s)
        match the input dimensions of the transform. I.e., they must
        already be in the correct neurospace and layerspace.
        """
        transform = self._load_transform(transform)

        layer_names = layer
        if isinstance(layer_names, str):
            layer_names = [layer_names]

        for layer in self.layers:
            layer: ng.ManagedLayer
            layer_name = layer.name
            if layer_names and layer_name not in layer_names:
                continue
            if layer_name.startswith('__'):
                continue
            layer = layer.layer
            if isinstance(layer, ng.ImageLayer):
                for source in layer.source:
                    source: LayerDataSource
                    source.transform = T.compose(
                        transform, source.transform, adapt=adapt
                    )

    @staticmethod
    def _load_transform(
        transform: (
            ng.CoordinateSpaceTransform | ArrayLike | str | PathLike | BytesIO
        ),
        inv: bool = False,
        *,
        mov: str = None,
        fix: str = None,
    ) -> ng.CoordinateSpaceTransform:
        if isinstance(transform, ng.CoordinateSpaceTransform):
            return transform

        transform = _ensure_list(transform)
        if len(transform) == 1:
            transform = transform[0]
        if isinstance(transform, str):
            transform = T.load_affine(transform, moving=mov, fixed=fix)
        if not isinstance(transform, ng.CoordinateSpaceTransform):
            transform = np.asarray(transform, dtype='float64')
            if transform.ndim == 1:
                if len(transform) == 12:
                    transform = transform.reshape([3, 4])
                elif len(transform) == 16:
                    transform = transform.reshape([4, 4])
                else:
                    n = int(np.sqrt(1 + 4 * len(transform)).item()) // 2
                    transform = transform.reshape([n, n+1])
            elif transform.ndim > 2:
                raise ValueError('Transforms must be matrices')
            transform = T.to_square(transform)
            if inv:
                transform = np.linalg.inv(transform)
            transform = transform[:-1]
            return ng.CoordinateSpaceTransform(
                matrix=transform,
                input_dimensions=S.neurospaces["xyz"],
                output_dimensions=S.neurospaces["xyz"],
            )
        elif inv:
            transform = T.inverse(transform)
        return transform

    @autolog
    def channel_mode(
        self,
        mode: Literal["local", "channel", "global"],
        layer: str | list[str] | None = None,
        dimension: str | list[str] = 'c',
    ) -> None:
        """
        Change the mode (local or intensity) of an axis.

        Parameters
        ----------
        mode : {"local", "channel", "global"}
            How to interpret this dimension:

            - "local": enables a switch/slider to change which channel
              is displayed (e.g., time).
            - "channel": enables the joint use of all channels to
              display a single intensity/color (e.g., RGB).
            - "global": enables the display of this axis in the
              cross-view.
        layer : [list of] str
            Names of layers to process
        dimension : [list of] str
            Names of dimensions to process
        """
        # NOTE
        #  localDimensions is named layerDimensions in the neuroglancer
        #   python code
        dimensions = _ensure_list(dimension)
        layers = _ensure_list(layer or [])

        mode = mode[0].lower()
        if mode not in ('l', 'c', 'g'):
            raise ValueError('Unknown channel mode. Should be one of '
                             '{local, channel, global}')

        def rename_key(
            space: ng.CoordinateSpace,
            src: str,
            dst: str,
        ) -> ng.CoordinateSpace:
            space = space.to_json()
            newspace = {}
            for key, val in space.items():
                if key == src:
                    key = dst
                newspace[key] = val
            return ng.CoordinateSpace(newspace)

        def update_transform(
            transform: ng.CoordinateSpaceTransform,
            olddim: str,
            newdim: str,
        ) -> ng.CoordinateSpaceTransform:
            if newdim.endswith(("'", "^")):
                sdim = newdim[:-1]
            else:
                sdim = newdim
            cdim = sdim + '^'
            transform.outputDimensions = rename_key(
                transform.outputDimensions, olddim, newdim)
            odims = list(transform.outputDimensions.to_json().keys())
            if newdim == cdim:
                # to channel dimension -> half unit shift
                shift = 0.5
            elif olddim == cdim:
                # from channel dimension -> remove half unit shift
                shift = -0.5
            else:
                shift = 0
            if shift:
                if transform.matrix is not None:
                    transform.matrix[odims.index(newdim), -1] += shift
                else:
                    matrix = np.eye(len(odims)+1)[:-1]
                    matrix[odims.index(newdim), -1] = shift
                    transform.matrix = matrix
            return transform

        def create_transform(
            scale: list[float], olddim: str, newdim: str
        ) -> ng.CoordinateSpaceTransform:
            if newdim.endswith(("'", "^")):
                sdim = newdim[:-1]
            else:
                sdim = newdim
            cdim = sdim + '^'
            if newdim == cdim:
                # to channel dimension -> half unit shift
                shift = 0.5
            elif olddim == cdim:
                # from channel dimension -> remove half unit shift
                shift = -0.5
            else:
                shift = 0
            transform = ng.CoordinateSpaceTransform(
                matrix=np.asarray([[1, shift]]),
                inputDimensions=ng.CoordinateSpace(
                    names=[olddim],
                    scales=[scale[0]],
                    units=[scale[1]],
                ),
                outputDimensions=ng.CoordinateSpace(
                    names=[newdim],
                    scales=[scale[0]],
                    units=[scale[1]],
                )
            )
            return transform

        for layer in self.layers:
            layer: ng.ManagedLayer
            if layers and layer.name not in layers:
                continue
            layer: ng.Layer = layer.layer
            for dimension in dimensions:
                ldim = dimension + "'"
                cdim = dimension + "^"
                sdim = dimension
                localDimensions = layer.localDimensions.to_json()
                if layer.localPosition:
                    localPosition = layer.localPosition.tolist()
                else:
                    localPosition = []
                channelDimensions = layer.channelDimensions.to_json()
                transform = None
                for source in layer.source:
                    if getattr(source, 'transform', {}):
                        transform = source.transform
                        break
                else:
                    source = layer.source[0]

                if mode == 'l':     # LOCAL
                    if ldim in localDimensions:
                        continue
                    was_channel = False
                    if cdim in channelDimensions:
                        was_channel = True
                        scale = channelDimensions.pop(cdim)
                    else:
                        scale = [1, ""]
                    localDimensions[ldim] = scale
                    localPosition = [*(localPosition or []), 0]
                    if transform:
                        update_transform(
                            transform, cdim if was_channel else sdim, ldim)
                    else:
                        source.transform = create_transform(
                            scale, cdim if was_channel else sdim, ldim)

                elif mode == 'c':   # CHANNEL
                    if cdim in channelDimensions:
                        continue
                    was_local = False
                    if ldim in localDimensions:
                        was_local = True
                        for i, key in enumerate(localDimensions.keys()):
                            if key == ldim:
                                break
                        scale = localDimensions.pop(ldim)
                        if i < len(localPosition):
                            localPosition.pop(i)
                    else:
                        scale = [1, ""]
                    channelDimensions[cdim] = scale
                    if transform:
                        update_transform(
                            transform, ldim if was_local else sdim, cdim)
                    else:
                        source.transform = create_transform(
                            scale, ldim if was_local else sdim, cdim)

                elif mode == 'g':   # SPATIAL
                    if cdim not in channelDimensions and \
                            ldim not in localDimensions:
                        continue
                    scale = [1, ""]
                    was_channel = False
                    if cdim in channelDimensions:
                        scale = channelDimensions.pop(cdim)
                        was_channel = True
                    if ldim in localDimensions:
                        for i, key in enumerate(localDimensions.keys()):
                            if key == ldim:
                                break
                        scale = localDimensions.pop(ldim)
                        if i < len(localPosition):
                            localPosition.pop(i)
                    if transform:
                        update_transform(
                            transform, cdim if was_channel else ldim, sdim)
                    else:
                        source.transform = create_transform(
                            scale, cdim if was_channel else ldim, sdim)
                    if sdim not in self.dimensions.to_json():
                        dimensions = self.dimensions.to_json()
                        dimensions[sdim] = scale
                        self.dimensions = ng.CoordinateSpace(dimensions)
                layer.localDimensions = ng.CoordinateSpace(localDimensions)
                layer.localPosition = np.asarray(localPosition)
                layer.channelDimensions = ng.CoordinateSpace(channelDimensions)

    @autolog
    def position(
        self,
        coord: float | list[float],
        dimensions: str | list[str] | None = None,
        unit: str | None = None,
        **kwargs,
    ) -> list[float]:
        """
        Change cursor position.

        Parameters
        ----------
        coord : [list of] float
            New position
        dimensions : [list of] str
            Axis of each coordinate. Can be a compact name like 'RAS'.
            Default: Currently displayed axes.
        unit : str
            Units of the coordinates. Default: Unit of current axes.

        Returns
        -------
        coord : list[float]
            Current cursor position.
        """
        if kwargs.pop('reset', not isinstance(coord, Sequence) and coord == 0):
            return coord([0] * len(self.dimensions))

        if not self.dimensions:
            raise RuntimeError(
                'Dimensions not known. Are you running the app in windowless '
                'mode? If yes, you must open a neuroglancer window to access '
                'or modifiy the cursor position')

        dim = self.dimensions

        # No argument -> print current position
        if not coord:
            string = []
            position = list(map(float, self.position))
            for x, d, s, u in zip(position, dim.names, dim.scales, dim.units):
                x = float(x) * float(s)
                string += [f'{d}: {x:g} {u}']
            self.stdio.info(', '.join(string))
            return position

        # Preproc dimensions
        if isinstance(dimensions, str):
            dimensions = [dimensions]
        dimensions = dimensions or list(map(str, dim.names))
        if len(dimensions) == 1 and len(dimensions[0]) > 1:
            dimensions = S.name_compact2full(dimensions[0])
        dimensions = dimensions[:len(coord)]

        # Convert unit
        unitmap = {n: u for u, n in zip(dim.units, dim.names)}
        current_units = [unitmap[d] for d in dimensions]
        coord = convert_unit(coord, unit, current_units)

        # Sort coordinate in same order as dim
        coord = {n: x for x, n in zip(coord, dimensions)}
        for x, n in zip(self.position, dim.names):
            coord.setdefault(n, x)
        coord = [coord[n] for n in dim.names]

        # Assign new coord
        self.position = list(coord.values())

        return list(map(float, self.position))

    @autolog
    def shader(
        self,
        shader: str | PathLike,
        layer: str | list[str] | None = None,
    ) -> str:
        """
        Apply a shader (that is, a colormap or lookup table).

        Parameters
        ----------
        shader : str
            A known shader name (from `ngtools.shaders`), or some
            user-defined shader code, or a LUT file.
        layer : str or list[str], optional
            Apply the shader to these layers. Default: all layers.
        """
        layer_names = _ensure_list(layer or [])

        if shader.lower() == 'rgb':
            for layer in self.layers:
                layer: ng.ManagedLayer
                layer_name = layer.name
                if layer_names and layer_name not in layer_names:
                    continue
                layer: ng.Layer = layer.layer
                if not layer.channelDimensions.to_json():
                    self.channel_mode('channel', layer=layer_name)

        if hasattr(shaders, shader):
            shader = getattr(shaders, shader)
        elif hasattr(colormaps, shader):
            shader = shaders.colormap(shader)
        elif 'main()' not in shader:
            # assume it's a path
            shader = shaders.lut(shader)
        for layer in self.layers:
            if layer_names and layer.name not in layer_names:
                continue
            layer = layer.layer
            if hasattr(layer, "shader"):
                layer.shader = shader

        self.stdio.info(shader)
        return shader

    @autolog
    def layout(
        self,
        layout: str | None = None,
        stack: Literal["row", "column"] | None = None,
        layer: str | list[str] | None = None,
        *,
        flex: float = 1,
        append: bool | int | list[int] | str | None = None,
        insert: int | list[int] | str | None = None,
        remove: int | list[int] | str | None = None,
    ) -> object:
        """
        Change layout.

        Parameters
        ----------
        layout : [list of] {"xy", "yz", "xz", "xy-3d", "yz-3d", "xz-3d", "4panel", "3d"}
            Layout(s) to set or insert. If list, `stack` must be set.
        stack : {"row", "column"}, optional
            Insert a stack of layouts.
        layer : [list of] str
            Set of layers to include in the layout.
            By default, all layers are included (even future ones).

        Other Parameters
        ----------------
        flex : float, default=1
            ???
        append : bool or [list of] int or str
            Append the layout to an existing stack.
            If an integer or list of integer, they are used to navigate
            through the nested stacks of layouts.
            Only one of append or insert can be used.
        insert : int or [list of] int or str
            Insert the layout into an existing stack.
            If an integer or list of integer, they are used to navigate
            through the nested stacks of layouts.
            Only one of append or insert can be used.
        remove : int or [list of] int or str
            Remove the layout in an existing stack.
            If an integer or list of integer, they are used to navigate
            through the nested stacks of layouts.
            If `remove` is used, `layout` should be `None`.

        Returns
        -------
        layout : object
            Current JSON layout
        """  # noqa: E501
        if not layout and (remove is None):
            self.stdio.info(self.layout)
            return self.layout

        layout = _ensure_list(layout or [])

        layer = _ensure_list(layer or [])
        if (len(layout) > 1 or stack) and not layer:
            layer = [_.name for _ in self.layers]

        if layer:
            layout = [ng.LayerGroupViewer(
                layers=layer,
                layout=L,
                flex=flex,
            ) for L in layout]

        if len(layout) > 1 and not stack:
            stack = 'row'
        if not stack and len(layout) == 1:
            layout = layout[0]
        if stack:
            layout = ng.StackLayout(
                type=stack,
                children=layout,
                flex=flex,
            )

        indices = []
        do_append = append is not None
        if do_append:
            indices = _ensure_list(append or [])
            append = do_append

        if insert:
            indices = _ensure_list(insert or [])
            insert = indices.pop(-1)
        else:
            insert = False

        if remove:
            indices = _ensure_list(remove or [])
            remove = indices.pop(-1)
        else:
            remove = False

        if bool(append) + bool(insert) + bool(remove) > 1:
            raise ValueError('Cannot use both append and insert')
        if layout and remove:
            raise ValueError('Do not set `layout` and `remove`')

        if append or (insert is not False) or (remove is not False):
            parent = self.layout
            while indices:
                parent = layout.children[indices.pop(0)]
            if layout and not isinstance(layout, ng.LayerGroupViewer):
                if not layer:
                    if len(parent.children):
                        layer = [L for L in parent.children[-1].layers]
                    else:
                        layer = [_.name for _ in self.layers]
                layout = ng.LayerGroupViewer(
                    layers=layer,
                    layout=layout,
                    flex=flex,
                )
            if append:
                parent.children.append(layout)
            elif insert:
                parent.children.insert(insert, layout)
            elif remove:
                del parent.children[remove]
        else:
            self.layout = layout
        return self.layout

    @autolog
    def zorder(
        self,
        layer: str | list[str],
        steps: int = None,
        **kwargs,
    ) -> None:
        """
        Move or reorder layers.

        In neuroglancer, layers are listed in the order they are loaded,
        with the latest layer appearing on top of the other ones in the
        scene. Counter-intuitively, the latest/topmost layer is listed
        at the bottom of the layer list, while the earliest/bottommost
        layer is listed at the top of the layer list. In this function,
        we list layers in their expected z-order, top to bottom.

        Parameters
        ----------
        layer : str or list[str]
            Name of layers to move.
            If `steps=None`, layers are given the provided order.
            Else, the selected layers are moved by `steps` steps.
        steps : int
            Number of steps to take. Positive steps move layers towards
            the top and negative steps move layers towards the bottom.
        """
        up = kwargs.get('up', 0)
        down = kwargs.get('down', 0)
        if up or down:
            if steps is None:
                steps = 0
            steps += up
            steps -= down

        names = _ensure_list(layer)
        if steps is None:
            # permutation
            names += list(reversed([layer.name for layer in self.layers
                                    if layer.name not in names]))
            layers = {layer.name: layer.layer for layer in self.layers}
            for name in names:
                del self.layers[name]
            for name in reversed(names):
                self.layers[name] = layers[name]
        elif steps == 0:
            return
        else:
            # move up/down
            layers = {layer.name: layer.layer for layer in self.layers}
            indices = {name: n for n, name in enumerate(layers)}
            for name in layers.keys():
                indices[name] += steps * (1 if name in names else -1)
            layers = {
                name: layers[name]
                for name in sorted(layers.keys(), key=lambda x: indices[x])
            }
            for name in layers.keys():
                del self.layers[name]
            for name, layer in layers.items():
                self.layers[name] = layer

    @autolog
    def state(
        self,
        load: str | None = None,
        save: str | None = None,
        url: bool = False,
        print: bool = True,
    ) -> dict:
        """
        Print or save or load the viewer's JSON state.

        Parameters
        ----------
        load : str
            Load state from JSON file, or JSON string (or URL if `url=True`).
        save : str
            Save state to JSON file
        url : bool
            print/load a JSON URL rather than a JSON object
        print : bool
            print the JSON object or URL

        Returns
        -------
        state : dict
            JSON state
        """
        if load:
            if exists(load):
                with open(load, "rb") as f:
                    state = json.load(f)
            elif url:
                if '://' in url:
                    url = urlparse(url).fragment
                    if url[0] != '!':
                        raise ValueError('Neuroglancer URL not recognized')
                    url = url[1:]
                state = json.loads(urlunquote(url))
            else:
                state = json.loads(url)
            self.set_state(state)
        else:
            state = self.to_json()

        if save:
            with open(save, 'wb') as f:
                json.dump(state, f, indent=4)

        if print:
            if url:
                state = urlquote(json.dumps(state))
                state = f'{DEFAULT_URL}#!' + state
                self.stdio.info(state)
            else:
                self.stdio.info(json.dumps(state, indent=4))
        return state
