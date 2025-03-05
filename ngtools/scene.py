"""A neuroglancer scene with a programmatic interface."""
# stdlib
import functools
import json
import logging
import os.path as op
import sys
from copy import deepcopy
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

# import to trigger datasource registration
import ngtools.local.datasources  # noqa: F401
import ngtools.local.tracts  # noqa: F401

# internals
import ngtools.spaces as S
import ngtools.transforms as T
from ngtools.datasources import LayerDataSource
from ngtools.layers import Layer, Layers
from ngtools.local.iostream import StandardIO
from ngtools.opener import exists, filesystem, open, parse_protocols
from ngtools.shaders import colormaps, load_fs_lut, rotate_shader, shaders
from ngtools.units import convert_unit
from ngtools.utils import NG_URLS, Wraps

# monkey-patch Layer state to expose channelDimensions
if not hasattr(ng.Layer, "channelDimensions"):
    ng.Layer.channel_dimensions \
        = ng.Layer.channelDimensions \
        = wrapped_property("channelDimensions", ng.CoordinateSpace)
if not hasattr(ng.Layer, "localDimensions"):
    ng.Layer.local_dimensions \
        = ng.Layer.localDimensions \
        = ng.Layer.layerDimensions


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
    """Smart ng.ViewerState that knows default values set in the frontend.

    Attributes
    ----------
    title : str, default=None
        Window title
    dimensions : CoordinateSpace
        All "global" dimensions.
    relative_display_scales : dict[str, float]
        ???
    display_dimensions : list[str]
        The 2 or 3 dimensions that are displayed in the cross section.
        Their order matters, as they map to the red, green and blue axes.
    position : LinkedType[vector[float]]
        Position in the "global frame".
        This vector must have as many values as there are dimensions.
    velocity : dict[str, DimensionPlaybackVelocity]
        ???
    cross_section_orientation : vector[float], default=(0, 0, 0, 1)
        Orientation of the cross-sections in the "displayed global frame".
        It is a quaternion ordered as `[i, j, k, r]`.
    cross_section_scale : float, default=1
        Zoom level of the cross-sections.
    cross_section_depth : float
        ???
    projection_scale : float
        Zoom level of the 3D window.
    projection_depth : float
        ???
    projection_orientation : vector[float], default=[0, 0, 0, 1]
        Orientation of the 3D window in the "displayed global frame".
        It is a quaternion ordered as `[i, j, k, r]`.
    show_slices : bool, default=True
        Whether to display orthogonal cross sections in the 3D window.
    wire_frame : bool, default=False
        Whether to display mesh wire frams in the 3D window.
    enable_adaptive_downsampling : bool, default=True
        ???
    show_scale_bar : bool, default=True
        Whether to show the scale bar.
    show_default_annotations : bool, default=True
        ???
    gpu_memory_limit : int
        Maximum GPU usage.
    system_memory_limit : int
        Maximum CPU usage.
    concurrent_downloads : int
        Maximum number of concurrent downloads.
    prefetch : bool, default=True
        Prefetch chunks.
    layers : Layers
        List of (named) registered layers.
    layout : StackLayout | LayerGroupViewer | DataPanelLayout | str
        If a string, can be one of
        `{"xy", "yz", "xz", "xy-3d", "yz-3d", "xz-3d", "4panel", "3d"}`.
    cross_section_background_color : str, default="black"
        Background color of cross-sections.
    projection_background_color : str, default="black"
        Background color of 3D window.
    selected_layer : SelectedLayerState
        With fields
        * `layer : str, default=layers[0].name`
            Selected layer
        * `visible : bool, default=False`
            Whether the right panel is visible.
        * `size : int`
            Width of the right panel, in voxels.
    statistics : StatisticsDisplayState
        With fields
        * `visible : bool, default=False`
            Whether the statistics panel is visible.
        * `size : int`
            Height of the panel, in voxels.
    help_panel : HelpPanelState
        With fields
        * `visible : bool, default=False`
            Whether the help panel is visible.
        * `size : int`
            Width of the panel, in voxels.
        * `flex : float, default=1.0`
            Relative height of the panel.
    layer_list_panel : LayerListPanelState
        With fields
        * `visible : bool, default=False`
            Whether the layer list panel is visible.
        * `size : int`
            Width of the panel, in voxels.
        * `flex : float, default=1.0`
            Relative height of the panel.
    partial_viewport : vector[float], default=[0, 0, 1, 1]
        Top-left and bottom-right corner of the visible portion of the
        viewer, where `[0, 0, 1, 1]` corresponds to the entire viewer.
    tool_bindings : dict[str, Tool | str]
        User-specific key bindings.

    """

    # --- non-ng attributes --------------------------------------------

    @property
    def spatial_dimensions(self) -> ng.CoordinateSpace:
        """All spatial dimensions (with meter-like unit)."""
        return ng.CoordinateSpace({
            key: [scale, unit]
            for key, (scale, unit) in self.dimensions.to_json().items()
            if unit[-1:] == "m"
        })

    @property
    def time_dimensions(self) -> ng.CoordinateSpace:
        """All time dimensions (with second-like unit)."""
        return ng.CoordinateSpace({
            key: [scale, unit]
            for key, (scale, unit) in self.dimensions.to_json().items()
            if unit[-1:] == "s"
        })

    @property
    def _space(self) -> str:
        """Current space."""
        return "".join(x[:1].lower() for x in self.spatial_dimensions.names)

    # --- layer --------------------------------------------------------

    def __get_layers__(self) -> Layers:
        return Layers(getattr(super(), "layers"))

    # --- dimensions ---------------------------------------------------

    @property
    def __default_dimensions__(self) -> ng.CoordinateSpace:
        dims = {}
        for layer in self.layers:
            layer: ng.ManagedLayer
            if len(getattr(layer, "source", [])) == 0:
                continue
            transform = layer.source[0].transform
            if transform is None or transform.output_dimensions is None:
                continue
            odims = transform.output_dimensions.to_json()
            for name, scale in odims.items():
                if not name.endswith(("^", "'")):
                    dims.setdefault(name, scale)
        dim_order = ["x", "y", "z", "t", "right", "anterior", "superior"]
        dims = dict(sorted(dims.items(), key=lambda x: (
            dim_order.index(x[0]) if x[0] in dim_order else
            float('inf')
        )))
        return ng.CoordinateSpace(dims)

    def __get_dimensions__(self) -> ng.CoordinateSpace:
        # NOTE: we must define it explicitly because ng's default is not None
        value = self._wrapped.dimensions
        if value is None or len(value.names) == 0:
            self.dimensions = self.__default_dimensions__
        return self._wrapped.dimensions

    def __set_dimensions__(
        self, value: ng.CoordinateSpace | None
    ) -> ng.CoordinateSpace:
        default_value = self.__default_dimensions__
        if value is None:
            value = default_value
        value = value.to_json()
        default_value = default_value.to_json()
        value = {
            name: val for name, val in value.items()
            if name in default_value
        }
        value.update({
            name: val for name, val in default_value.items()
            if name not in value
        })
        return ng.CoordinateSpace(value)

    # --- relative_display_scales --------------------------------------

    @property
    def __default_relative_display_scales__(self) -> dict[str, float]:
        return self.__set_relative_display_scales__(None)

    def __set_relative_display_scales__(
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

    __default__relativeDisplayScales__ = __default_relative_display_scales__
    __set_relativeDisplayScales__ = __set_relative_display_scales__

    # --- display_dimensions -------------------------------------------

    @property
    def __default_display_dimensions__(self) -> list[str]:
        value = self.__set_display_dimensions__(None)
        return value

    def __set_display_dimensions__(self, value: list[str] | None) -> list[str]:
        dimensions = self.dimensions.names
        if value is None:
            value = ['x', 'y', 'z', 't']
        value = [name for name in value if name in dimensions]
        dimensions = [name for name in dimensions if name not in value]
        value = (value + dimensions)[:3]
        return value

    def __get_display_dimensions__(self) -> list[float]:
        value = self._wrapped.display_dimensions
        if hasattr(value, "_data"):
            # NOTE: when assigning into a TypedList attribute (which is
            # the case for ng.ViewerState.display_dimensions), we can
            # only assign basic lists (list, tuple, ndarray), not
            # an existing TypedList.
            # So we return the underlying list instead.
            return value._data
        return value

    __default_displayDimensions__ = __default_display_dimensions__
    __set_displayDimensions__ = __set_display_dimensions__
    __get_displayDimensions__ = __get_display_dimensions__

    # --- cross_section_orientation ------------------------------------

    __default_cross_section_orientation__: list[float] = [0, 0, 0, 1]

    def __set_cross_section_orientation__(self, value: ArrayLike) -> ArrayLike:
        if value is None:
            value = self.__default_cross_section_orientation__
        value = np.asarray(value).tolist()
        value = value + max(0, 4 - len(value)) * [0]
        value = value[:4]
        value = np.asarray(value, dtype="double")
        value /= (value**2).sum()**0.5
        return value

    __default_crossSectionOrientation__ = __default_cross_section_orientation__
    __set_crossSectionOrientation__ = __set_cross_section_orientation__

    # --- cross_section_scale ------------------------------------------

    @property
    def __default_cross_section_scale__(self) -> float:
        """
        Compute a smart default scale (bbox of the fist layer).

        NOTE: scales are expressed in "model scaled space".
        """
        dimensions = self.dimensions
        dimensions = dimensions.to_json()
        scl = {key: 1.0 for key in dimensions}
        for layer in self.layers:
            layer: ng.ManagedLayer
            if not layer.visible:
                continue

            try:
                source = layer.source[0]
                bbox = source.output_bbox_size
                odims = source.transform.output_dimensions.to_json()
                onames = list(odims.keys())
                for name, (scale, unit) in dimensions.items():
                    if name not in onames:
                        continue
                    j = onames.index(name)
                    scale0, unit0 = odims[name]
                    bbox_value = convert_unit(bbox[j], unit0, unit)
                    bbox_value *= scale0 / scale
                    scl[name] = bbox_value / 256  # factor that seems to work

                break
            except Exception:
                continue

        dims = self.display_dimensions[:3]
        if len(dims) == 0:
            scl = 1.0
        else:
            scl = sum(scl[name] for name in dims) / len(dims)
        return scl

    __default_crossSectionScale__ = __default_cross_section_scale__

    # --- projection_scale ------------------------------------------

    @property
    def __default_projection_scale__(self) -> float:
        value = 512 * self.__default_cross_section_scale__
        return value

    __default_projectionScale__ = __default_projection_scale__

    # --- position -----------------------------------------------------

    @property
    def __default_position__(self) -> list[float]:
        """
        Compute a smart default position (center of the fist layer).

        NOTE: positions are expressed in "model scaled space". That is,
        it the "z" dimension listed in `dimensions` has scale (0.5, "mm"),
        a position increment of 1 will correspond to an effective increment
        of 0.5 mm.
        """
        dimensions = self.dimensions.to_json()
        pos = [0.0] * len(dimensions)
        for layer in self.layers:
            layer: ng.ManagedLayer
            if not layer.visible:
                continue

            try:
                source = layer.source[0]
                center = source.output_center
                odims = source.transform.output_dimensions.to_json()
                onames = list(odims.keys())
                for i, (name, (scale, unit)) in enumerate(dimensions.items()):
                    if name not in onames:
                        continue
                    j = onames.index(name)
                    scale0, unit0 = odims[name]
                    scale_ratio = scale0 / scale
                    value = convert_unit(center[j], unit0, unit)
                    value *= scale_ratio
                    pos[i] = value

                break
            except Exception:
                continue

        return pos

    def __get_position__(self) -> list[float]:
        value = self._wrapped.position
        if value is None or len(value) == 0:
            pos = self.__default_position__
            self.position = pos
        return self._wrapped.position

    def __set_position__(self, value: list[float] | None) -> list[float]:
        if value is None:
            value = self.__default_position__
        value = list(value)
        value += [0.0] * max(0, len(self.dimensions.names) - len(value))
        return value


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
                f"{key}=" + (f'"{val}"' if isinstance(val, str) else str(val))
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
                self.stdio.debug(f"{type(e).__name__}({e})")
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
        uri: URILike | list[URILike] | dict[str, URILike] = None,
        transform: ArrayLike | list[float] | URILike | None = None,
        shader: str | None = None,
        inv: bool = False,
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
        inv : bool
            Invert the transform before applying it.

        Other Parameters
        ----------------
        name : str | list[str]
            Alternative way of providing layer names.
            If used, `uri` cannot be a `dict`.
        """
        fileserver = (kwargs.pop("fileserver", "") or "").rstrip("/")

        if kwargs.get("filename", []):
            if uri:
                raise ValueError("filename and uri cannot be used together")
            uri = kwargs.pop("filename")

        # prepare names and URLs
        names = []
        if isinstance(uri, dict):
            names = list(uri.keys())
            uri = list(uri.values())
        else:
            names = kwargs.pop("name", [])
        uris = _ensure_list(uri or [])
        names = _ensure_list(names or [])

        # original number of layers
        # -> will be used later to (re)set the view.
        nb_layers_0 = len(self.layers)

        # load layers
        onames = []
        for n, uri in enumerate(uris):

            # TODO: wrap each file loading in a try/except block?

            uri = str(uri).rstrip("/")
            parsed = parse_protocols(uri)
            short_uri = parsed.url
            basename = op.basename(short_uri)
            name = names[n] if names else basename

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
            layer = Layer(str(parsed), **kwargs)
            self.layers.append(name=name, layer=layer)
            self.stdio.info(f"Loaded: {self.layers[name].to_json()}")
            onames.append(name)

        # rename axes according to current naming scheme
        self.rename_axes(self.world_axes(print=False), layer=onames)

        if transform is not None:
            self.transform(transform, layer=onames, inv=inv)

        if nb_layers_0 == 0:
            # trigger default values
            self.dimensions = None
            self.display_dimensions = None
            self.position = None
            self.cross_section_scale = None
            self.projection_scale = None
            self.space("radio", "world")

        if shader is not None:
            self.shader(shader, layer=onames)

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
            return ng.AnnotationLayer(ng.LocalAnnotationLayer(coord).to_json())

        # check if keywords were used
        if kwargs.get("dst", None):
            if axes is not None:
                msg = "Cannot use `axes` if `src` is used."
                self.stdio.error(msg)
                raise ValueError(msg)
            axes = _ensure_list(kwargs["dst"])
        if kwargs.get("src", None):
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

        native_transform = world_axes_native.source[0].transform
        current_transform = world_axes_current.source[0].transform
        if native_transform:
            native_names = native_transform.output_dimensions.names
        else:
            native_names = []
        if current_transform:
            current_names = current_transform.output_dimensions.names
        else:
            current_names = []

        old_axes = {"x": "x", "y": "y", "z": "z", "t": "t"}
        old_axes.update({
            native: current
            for native, current in zip(native_names, current_names)
        })

        if not axes:
            if kwargs.get("print", False):
                self.stdio.print(old_axes)
            return old_axes

        new_axes = axes
        axes = dict(old_axes)

        old2nat = {v: k for k, v in old_axes.items()}
        self.rename_axes(old2nat)

        if isinstance(new_axes, str):
            new_axes = S.name_compact2full(new_axes)
        if isinstance(new_axes, (list, tuple)):
            if len(new_axes) == 1:
                new_axes = S.name_compact2full(new_axes[0])
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
        # save these so they can be fixed later
        dimensions = self.dimensions.to_json()
        position = self.position

        # check if keywords were used
        if kwargs.get("dst", None):
            if axes is not None:
                msg = "Cannot use `axes` if `src` is used."
                self.stdio.error(msg)
                raise ValueError(msg)
            axes = _ensure_list(kwargs["dst"])
        if kwargs.get("src", None):
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
            model_axes = self.world_axes(print=False)
            axes = {
                model_axes[native_axis]: new_axis
                for native_axis, new_axis in zip("xyz", axes)
            }

        def rename_axis(name: str) -> str:
            for axis_type in ("", "'", "^"):
                if name + axis_type in axes:
                    return axes[name + axis_type] + axis_type
            return name

        layers = _ensure_list(layer or [])
        for named_layer in self.layers:
            if layers and named_layer.name not in layers:
                continue
            if named_layer.name.startswith("__"):
                continue
            if len(getattr(layer, "source", [])) == 0:
                continue
            transform = layer.source[0].transform
            transform.output_dimensions = ng.CoordinateSpace({
                rename_axis(name): scl
                for name, scl in transform.output_dimensions.to_json().items()
            })
            layer.source[0].transform = transform

        self.dimensions = ng.CoordinateSpace({
            rename_axis(name): scl
            for name, scl in dimensions.items()
        })

        self.display_dimensions = [
            rename_axis(name) for name in self.display_dimensions
        ]

        ndim = len(self.dimensions.names)
        position = list(position) + [0.0] * max(0, len(position) - ndim)
        position = position[:ndim]
        self.position = position

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
        norm = np.linalg.norm

        deflt = np.asarray([0., 0., 0., 1.])
        neuro = np.asarray([1., 0., 0., -1.])
        neuro /= norm(neuro)
        radio = np.asarray([0., -1., 1., 0.])
        radio /= norm(radio)
        current = np.asarray(self.cross_section_orientation)  # or [0, 0, 0, 1]
        # -q === q so use sign that's closest to current view
        if norm(current + neuro) < norm(current - neuro):
            neuro = -neuro
        if norm(current + radio) < norm(current - radio):
            radio = -radio
        if norm(current + deflt) < norm(current - deflt):
            deflt = -deflt
        # find layout closest to current view (TODO: try all of them)
        if norm(current - neuro) < norm(current - radio):
            if norm(current - neuro) < norm(current - deflt):
                current_mode = "neuro"
            else:
                current_mode = "default"
        else:
            if norm(current - radio) < norm(current - deflt):
                current_mode = "radio"
            else:
                current_mode = "default"

        # If first input is a layer name, switch
        if layer is None and mode in self.layers:
            layer, mode = mode, None

        # If no input arguments, simply return known dimensions
        if mode is None and layer is None:
            self.stdio.info(current_mode)
            return current_mode
        mode = mode or current_mode

        current_canonical = {
            "radio": radio, "neuro": neuro, "default": deflt
        }[current_mode]

        target_canonical = {
            "radio": radio, "neuro": neuro, "default": deflt
        }[mode]

        # Display spatial dimensions (reorder if needed)
        #
        # TODO
        # At some point I wanted to show data in the layer voxel space
        # closest to the current view, but it's a bit more difficult
        # than I thought (I need to compose the world2voxel rotation
        # with the world2view transform -- but the latter depends
        # on the displayed dimension so I'll need to do some filtering/
        # permutation). I might come back to it at some point. In the
        # meantime I always reset the displayed dimensions to [x, y, z].
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

        # layer-specific rotation
        if layer and (layer.lower() != "world"):
            if layer not in self.layers:
                raise ValueError("No layer named:", layer)
            if len(getattr(self.layers[layer], "source", [])) == 0:
                raise ValueError(f'Layer "{layer} does not have a source')

            #   1. Get voxel2world matrix
            source = self.layers[layer].source[0]
            transform = source.transform
            # transform = T.subtransform(transform, 'm')
            # transform = T.ensure_same_scale(transform)
            matrix = T.get_matrix(transform, square=True)[:-1, :-1]
            #   2.remove scales and shears
            u, _, vh = np.linalg.svd(matrix)
            rot = u @ vh
            #   3. preserve permutations and flips
            #      > may not work if exactly 45 deg rotation so add a tiny
            #        bit of noise
            eps = np.random.randn(*rot.shape) * 1E-8 * (rot*rot).max()
            orient = (rot*rot + eps).round()
            orient = orient * np.sign(rot)
            assert \
                np.allclose(np.abs(orient @ orient.T), np.eye(len(orient))), \
                (orient @ orient.T)
            rot = rot @ orient.T
            #   4. select displayed axes
            ind = [
                transform.output_dimensions.names.index(ax)
                for ax in display_dimensions
            ]
            rot = rot[ind, :][:, ind]
            #   5. To quaternion
            if np.linalg.det(rot) < 0:
                # flip left-right
                assert False, "Negative determinant in voxel-to-model matrix"
            to_layer = T.matrix_to_quaternion(rot)
        elif not layer:
            # keep existing layer axes
            from_canonical = T.inverse_quaternions(current_canonical)
            to_layer = T.compose_quaternions(current, from_canonical)
        else:
            # layer == "world"
            to_layer = [0., 0., 0., 1.]

        # canonical neuro/radio axes
        to_layer = T.compose_quaternions(to_layer, target_canonical)

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
        world_axes = self.world_axes(print=False)
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

            shader = getattr(layer, "shader", None)
            skeleton_shader = getattr(layer, "skeleton_shader", None)

            if shader and shader != "None":
                mat = self._vox2display(transform)
                shader = rotate_shader(shader, mat)
                layer.shader = shader

            elif skeleton_shader and skeleton_shader != "None":
                mat = self._vox2display(transform)
                skeleton_shader = rotate_shader(skeleton_shader, mat)
                layer.skeleton_shader = skeleton_shader

            if shader == "None":
                layer.shader = shaders.default

            if skeleton_shader == "None":
                layer.skeleton_shader = shaders.skeleton.orientation

            for source in getattr(layer, "source", []):
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
                gdim = newdim[:-1]
            else:
                gdim = newdim
            cdim = gdim + '^'

            transform.outputDimensions = rename_key(
                transform.outputDimensions, olddim, newdim)

            odims = list(transform.outputDimensions.names)
            if newdim == cdim:
                # ensure zero offset in channel dim
                if transform.matrix is not None:
                    transform.matrix[odims.index(newdim), -1] = 0
                else:
                    matrix = np.eye(len(odims)+1)[:-1]
                    matrix[odims.index(newdim), -1] = 0
                    transform.matrix = matrix
            return transform

        def create_transform(
            scale: list[float], olddim: str, newdim: str
        ) -> ng.CoordinateSpaceTransform:
            transform = ng.CoordinateSpaceTransform(
                matrix=np.asarray([[1, 0]]),
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

        # Update localDimensions & channelDimensions
        for layer in self.layers:
            layer: ng.ManagedLayer
            if layers and layer.name not in layers:
                continue
            if layer.name[:2] == "__":
                continue
            if len(getattr(layer, "source", [])) == 0:
                continue

            for dimension in dimensions:
                ldim = dimension + "'"
                cdim = dimension + "^"
                gdim = dimension

                # read dimensions
                localDimensions = layer.localDimensions.to_json()
                if layer.localPosition:
                    localPosition = layer.localPosition.tolist()
                else:
                    localPosition = []
                channelDimensions = layer.channelDimensions.to_json()

                # read transform
                transform = None
                for source in layer.source:
                    if getattr(source, 'transform', None):
                        transform = source.transform
                        break
                else:
                    source = layer.source[0]

                # read output dimensions
                odims = None
                if transform:
                    odims = getattr(transform, "outputDimensions", None)

                # --- LOCAL --------------------------------------------
                if mode == 'l':
                    if ldim in localDimensions:
                        continue
                    elif odims and ldim in odims.names:
                        continue

                    was_channel = False
                    if cdim in channelDimensions:
                        was_channel = True
                        scale = channelDimensions.pop(cdim)
                    elif odims and cdim in odims.names:
                        was_channel = True
                        scale = odims.to_json()[cdim]
                    elif odims and gdim in odims.names:
                        scale = odims.to_json()[gdim]
                    elif gdim in self.dimensions:
                        scale = self.dimensions.to_json()[gdim]
                    else:
                        scale = [1, ""]

                    localDimensions[ldim] = scale
                    localPosition = [*(localPosition or []), 0]
                    if transform:
                        update_transform(
                            transform, cdim if was_channel else gdim, ldim)
                    else:
                        source.transform = create_transform(
                            scale, cdim if was_channel else gdim, ldim)

                # --- CHANNEL ------------------------------------------
                elif mode == 'c':
                    if cdim in channelDimensions:
                        continue
                    elif odims and cdim in odims.names:
                        continue

                    was_local = False
                    if ldim in localDimensions:
                        was_local = True
                        i = list(localDimensions.keys()).index(ldim)
                        scale = localDimensions.pop(ldim)
                        if i < len(localPosition):
                            localPosition.pop(i)
                    elif odims and ldim in odims.names:
                        was_local = True
                        scale = odims.to_json()[ldim]
                    else:  # was global
                        try:
                            # get input voxel size
                            idims = transform.inputDimensions
                            matrix = transform.matrix
                            index = odims.names.index(gdim)
                            index = np.abs(matrix[index]).argmax()
                            scale = idims.to_json()[idims.names[index]]
                        except Exception:
                            scale = [1, ""]

                    channelDimensions[cdim] = scale
                    if transform:
                        update_transform(
                            transform, ldim if was_local else gdim, cdim)
                    else:
                        source.transform = create_transform(
                            scale, ldim if was_local else gdim, cdim)

                # --- GLOBAL -------------------------------------------
                elif mode == 'g':
                    if (
                        cdim not in channelDimensions and
                        ldim not in localDimensions
                    ):
                        continue
                    if odims and (
                        cdim not in odims.names and
                        ldim not in odims.names
                    ):
                        continue

                    was_channel = False
                    if cdim in channelDimensions:
                        was_channel = True
                        scale = channelDimensions.pop(cdim)
                    elif odims and cdim in odims.names:
                        was_channel = True
                        scale = odims.to_json()[cdim]
                    elif ldim in localDimensions:
                        i = list(localDimensions.keys()).index(ldim)
                        scale = localDimensions.pop(ldim)
                        if i < len(localPosition):
                            localPosition.pop(i)
                    elif odims and ldim in odims.names:
                        scale = odims.to_json()[ldim]
                    else:
                        scale = [1, ""]

                    if transform:
                        update_transform(
                            transform, cdim if was_channel else ldim, gdim)
                    else:
                        source.transform = create_transform(
                            scale, cdim if was_channel else ldim, gdim)

                    if gdim not in self.dimensions.names:
                        dimensions = self.dimensions.to_json()
                        dimensions[gdim] = scale
                        self.dimensions = ng.CoordinateSpace(dimensions)

                # set position/dimensions
                layer.localDimensions = ng.CoordinateSpace(localDimensions)
                layer.localPosition = np.asarray(localPosition)
                layer.channelDimensions = ng.CoordinateSpace(channelDimensions)

        # Update global dimensions & position
        current_dimensions = self.dimensions.to_json()
        current_position = list(self.position)
        default_dimensions = self.__default_dimensions__.to_json()
        default_position = self.__default_position__
        for gdim in dimensions:
            if mode == "g":
                # add dimension to globals
                if gdim in current_dimensions:
                    continue
                elif gdim in default_dimensions:
                    current_dimensions[gdim] = default_dimensions[gdim]
                    index = list(current_dimensions).index(gdim)
                    current_position.insert(index, default_position[index])
            else:
                # remove dimension from globals
                if gdim not in current_dimensions:
                    continue
                else:
                    index = list(current_dimensions).index(gdim)
                    del current_dimensions[gdim]
                    del current_position[index]

    @autolog
    def move(
        self,
        coord: float | list[float] | dict[str, float] = 0,
        dimensions: str | list[str] | None = None,
        unit: str | None = None,
        absolute: bool = False,
        **kwargs,
    ) -> list[float]:
        """
        Change cursor position.

        Parameters
        ----------
        coord : [list of] float | dict[str, float]
            New position. If a dictionary, maps axes to values.
        dimensions : [list of] str
            Axis of each coordinate. Can be a compact name like 'RAS'.
            Cannot be used when `coord` is a `dict`.
            Default: Currently displayed axes.
        unit : str
            Units of the coordinates. Default: Unit of current axes.
        absolute : bool
            Move to absolute position, rather than relative to current.

        Other Parameters
        ----------------
        reset : bool
            Reset position to default

        Returns
        -------
        coord : list[float]
            Current cursor position.
        """
        if kwargs.pop('reset', not isinstance(coord, Sequence) and coord == 0):
            self.position = self.__default_position__
            return self.position

        if not self.dimensions:
            raise RuntimeError(
                'Dimensions not known. Are you running the app in windowless '
                'mode? If yes, you must open a neuroglancer window to access '
                'or modifiy the cursor position')

        dim = self.dimensions.to_json()

        # No argument -> print current position
        if not coord:
            string = []
            position = list(map(float, self.position))
            for x, d, (s, u) in zip(position, dim.keys(), dim.values()):
                x = float(x) * float(s)
                string += [f'{d}: {x:g} {u}']
            self.stdio.info(', '.join(string))
            return position

        # Preproc dimensions
        if not isinstance(coord, dict):
            if isinstance(dimensions, str):
                dimensions = [dimensions]
            dimensions = dimensions or list(dim.keys())
            if len(dimensions) == 1 and len(dimensions[0]) > 1:
                dimensions = S.name_compact2full(dimensions[0])
            dimensions = dimensions[:len(coord)]
            coord = {n: x for x, n in zip(coord, dimensions)}
        elif dimensions:
            raise ValueError(
                "Cannot use `dimensions` when `coord` is a dictionary"
                )

        # Convert unit
        coord = {
            n: (convert_unit(x, unit, dim[n][1]) / dim[n][0] if unit else x)
            for n, x in coord.items()
        }

        # Sort coordinate in same order as dim
        for n in dim:
            coord.setdefault(n, 0)
        coord = [coord[n] for n in dim]

        # Assign new coord
        if not absolute:
            coord = [(c + p) for c, p in zip(coord, self.position)]
        self.position = coord

        return list(map(float, self.position))

    @autolog
    def zoom(self, factor: float | None = 2.0, reset: bool = False) -> float:
        """Zoom by some factor.

        Parameters
        ----------
        factor : float
            Zoom factor
        reset : bool
            Reset zoom level to default

        Returns
        -------
        scale : float
            Current zoom level
        """
        if reset:
            self.cross_section_scale = self.__default_cross_section_scale__
            return self.cross_section_scale
        scale = self.cross_section_scale
        factor = float(factor or 1.0)
        if factor != 0:
            # cross_section_scale is the _resolution_ of the view,
            # therefore smaller means more zoomed.
            scale /= factor
            self.cross_section_scale = scale
        else:
            self.stdio.print(scale)
        return scale

    @autolog
    def unzoom(self, factor: float | None = 2.0, reset: bool = False) -> float:
        """Unzoom by some factor.

        Parameters
        ----------
        factor : float
            Unzoom factor
        reset : bool
            Reset zoom level to default

        Returns
        -------
        scale : float
            Current zoom level
        """
        if reset:
            self.cross_section_scale = self.__default_cross_section_scale__
            return self.cross_section_scale
        scale = self.cross_section_scale
        factor = float(factor or 1.0)
        if factor != 1:
            # cross_section_scale is the _resolution_ of the view,
            # therefore bigger means less zoomed.
            scale *= factor
            self.cross_section_scale = scale
        else:
            self.stdio.print(scale)
        return scale

    def _vox2display(
        self, transform: ng.CoordinateSpaceTransform | None
    ) -> np.ndarray:
        """
        Compute voxel-to-display matrix.
        Forces voxel order to be (x, y, z) or (i, j, k).
        """
        if transform is None:
            return None

        # get transform components
        idims = getattr(transform, "inputDimensions", None)
        odims = getattr(transform, "outputDimensions", None)
        mat = getattr(transform, "matrix", None)
        if not odims and not idims:
            return None
        if idims and not odims:
            odims = idims
        elif odims and not idims:
            idims = odims
        rank = len(odims.names)
        if mat is None:
            mat = np.eye(rank+1)[:-1]

        # get display and voxel axes
        wnames = self.world_axes()
        onames = self.display_dimensions
        inames = [
            "i" if "i" in idims.names else "x" if idims.names else wnames.get("x", None),  # noqa: E501
            "j" if "j" in idims.names else "y" if idims.names else wnames.get("y", None),  # noqa: E501
            "k" if "k" in idims.names else "z" if idims.names else wnames.get("z", None),  # noqa: E501
        ]
        if not all(inames):
            return None

        # select sub-matrix
        oind = [odims.names.index(name) for name in onames]
        iind = [idims.names.index(name) for name in inames]
        mat = mat[oind, :][:, iind]

        # remove scales and shears
        u, _, vh = np.linalg.svd(mat)
        mat = u @ vh

        return mat

    @autolog
    def shader(
        self,
        shader: str | PathLike,
        layer: str | list[str] | None = None,
        layer_type: str | list[str] | None = None,
        **kwargs
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
        layer_type : str or list[str], optional
            Apply the shader to these layer types. Default: all layers.
        """
        fileserver = (kwargs.pop("fileserver", "") or "").rstrip("/")
        segment_properties = None

        layer_names = _ensure_list(layer or [])
        layer_types = _ensure_list(layer_type or [])

        # Ensure channels have correct channel mode
        if shader.lower() in ('rgb', 'orientation'):
            for layer in self.layers:
                layer: ng.ManagedLayer
                layer_name = layer.name

                if layer_names and layer_name not in layer_names:
                    continue

                if layer_name[:2] == "__":
                    continue

                if len(getattr(layer, "source", [])) == 0:
                    continue

                transform = getattr(layer.source[0], "transform", None)
                odims = getattr(transform, "outputDimensions", None)
                onames = getattr(odims, "names", [])
                if onames and not any(name[-1:] == "^" for name in onames):
                    self.channel_mode('channel', layer=layer_name)

        split_shader = shader.split(".")
        if hasattr(shaders, split_shader[0]):
            shader = shaders
            for name in split_shader:
                shader = getattr(shader, name)
            self.stdio.info(shader)

        elif hasattr(colormaps, shader):
            shader = shaders.colormap(shader)
            self.stdio.info(shader)

        elif 'main()' not in shader:
            # assume it's a path
            path = shader
            lut = load_fs_lut(path)
            shader = shaders.lut(lut)
            f2u8 = lambda x: int(round(x*255))  # noqa: E731
            lut = {
                str(key): f'#{f2u8(r):02x}{f2u8(g):02x}{f2u8(b):02x}'
                for key, (_, (r, g, b, _)) in lut.items()
            }
            if fileserver:
                path = parse_protocols(path).url
                if "://" not in path:
                    path = "file/" + op.abspath(path)
                else:
                    protocol, path = path.split("://")
                    path = protocol + "/" + path

                segment_properties = (
                    "precomputed://" + fileserver + "/lut/" + path
                )

        for layer in self.layers:

            if layer_names and layer.name not in layer_names:
                continue

            if layer.name[:2] == "__":
                continue

            if layer_types and layer.type not in layer_types:
                continue

            # ImageLayer or AnnotationLayer -> Shader
            if hasattr(layer, "shader"):

                layer_shader = shader
                if len(getattr(layer, "source", [])) > 0:
                    transform = getattr(layer.source[0], "transform", None)
                    mat = self._vox2display(transform)
                    layer_shader = rotate_shader(shader, mat)

                layer.shader = layer_shader

            # SegmentationLayer -> LUT
            elif layer.type == "segmentation":

                layer.segment_colors = lut
                for key in lut:
                    if key not in layer.starred_segments:
                        layer.starred_segments[key] = True

                if segment_properties:
                    layer.source.append(
                        ng.LayerDataSource(segment_properties)
                    )

        return shader

    @autolog
    def change_layout(
        self,
        layout: str | None = None,
        stack: Literal["row", "column"] | None = None,
        layer: str | list[str] | None = None,
        *,
        flex: float = 1,
        append: bool | int | list[int] | str | None = None,
        assign: int | list[int] | str | None = None,
        insert: int | list[int] | str | None = None,
        remove: int | list[int] | str | None = None,
    ) -> object:
        """
        Change layout.

        !!! note "Append/Assign/Insert/Remove"

            The requested (stack of) layout(s) can be inserted
            into existing layouts.

            Arguments must be an integer or list of integer, that are
            used to navigate through the existing nested stacks of layouts.

            Only one of {append, assign, insert, remove} can be used.


        Parameters
        ----------
        layout : [list of] {"xy", "yz", "xz", "xy-3d", "yz-3d", "xz-3d", "4panel", "3d"}
            Layout(s) to set or insert. If list, `stack` must be set.
        stack : {"row", "column"}, optional
            Insert a stack of layouts.
            If input layout is a list and `append` is not used,
            default is `"row"`.
        layer : [list of] str
            Set of layers to include in the layout.
            By default, all layers are included (even future ones).

        Other Parameters
        ----------------
        flex : float, default=1
            ???
        append : bool or [list of] int or str
            Append the layout to an existing stack.
        assign : int or [list of] int or str
            Assign the layout into an existing stack.
        insert : int or [list of] int or str
            Insert the layout into an existing stack.
        remove : int or [list of] int or str
            Remove the layout from an existing stack.

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

        # If layer is set, prepare a list of `LayerGroupViewer`s.
        if layer:
            layout = [ng.LayerGroupViewer(
                layers=layer,
                layout=L,
                flex=flex,
            ) for L in layout]

        # If multiple layouts and stack not set, use "row" as default.
        if len(layout) > 1 and not stack:
            stack = 'row'

        # Stack layers
        if stack:
            layout = ng.StackLayout(
                type=stack,
                children=layout,
                flex=flex,
            )

        # Unless there is a single layout
        elif layout:
            layout = layout[0]

        # Prepare append/insert/remove indices

        indices = []

        do_append = append not in (None, False)
        if do_append:
            indices = _ensure_list(append)

        do_assign = assign not in (None, False)
        if do_assign:
            indices = _ensure_list(assign)
            assign = indices.pop(-1)

        do_insert = insert not in (None, False)
        if do_insert:
            indices = _ensure_list(insert)
            insert = indices.pop(-1)

        do_remove = remove not in (None, False)
        if do_remove:
            indices = _ensure_list(remove)
            remove = indices.pop(-1)

        if do_append + do_assign + do_insert + do_remove > 1:
            raise ValueError('Cannot use both append and insert')
        if layout and do_remove:
            raise ValueError('Do not set `layout` and `remove`')
        if do_append + do_assign + do_insert + do_remove == 0:
            # nothing to do
            self.layout = layout
            return self.layout

        # append/insert

        parent = self.layout
        while indices:
            parent = layout.children[indices.pop(0)]

        if do_append:
            parent.children.append(layout)
        elif do_assign:
            parent.children[assign] = layout
        elif do_insert:
            parent.children.insert(insert, layout)
        elif do_remove:
            del parent.children[remove]

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
        instance: str | None = None,
        print: bool = True,
        **kwargs
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
            Print/load/save a JSON URL rather than a JSON object
        open : bool
            Open the URL in the browser.
        instance : {"ng", "linc"}
            Neuroglancer instance to use in the URL.
        print : bool
            print the JSON object or URL

        Returns
        -------
        state : dict
            JSON state
        """
        # --- load -----------------------------------------------------
        if load:
            if exists(load):
                # Load from file
                with open(load, "rb") as f:
                    state = json.load(f)

            elif url:
                # Parse from neuroglancer URL
                if '://' in url:
                    url = urlparse(url).fragment
                    if url[0] != '!':
                        raise ValueError('Neuroglancer URL not recognized')
                    url = url[1:]
                state = json.loads(urlunquote(url))

            else:
                # Parse from JSON string
                state = json.loads(url)

            # Load state in current scene
            scene = ng.ViewerState(state)
            for key in scene.to_json().keys():
                val = getattr(scene, key)
                setattr(self, key, val)

        # --- convert and/or save --------------------------------------
        state = json_state = deepcopy(self.to_json())

        # guess instance
        if instance is None:
            instance = "ng"
            for layer in self.layers:
                for source in getattr(layer, "source", []):
                    url_ = getattr(source, "url", "")
                    if isinstance(url_, str) and (
                        ("lincbrain.org" in url_) or
                        ("/linc/" in url_)
                    ):
                        instance = "linc"
                        break

        # convert linc URLs to neuroglancer.lincbrain.org URLs
        if instance.lower() == "linc":

            def fix_url(url: str) -> str:
                if "/linc/" in url:
                    url = url.split("/linc/")[-1]
                    url = "https://neuroglancer.lincbrain.org/" + url
                return url

            for layer in state.get("layers", []):
                if "source" not in layer:
                    continue
                if not isinstance(layer["source"], list):
                    layer["source"] = [layer["source"]]
                for i, source in enumerate(layer["source"]):
                    if isinstance(source, str):
                        layer["source"][i] = fix_url(source)
                    elif "url" in source:
                        source["url"] = fix_url(source["url"])

        # --- to URL ---
        if url or kwargs.get("open", False):
            state = urlquote(json.dumps(state))
            state = f'{NG_URLS[instance]}#!' + state

            if kwargs.get("open", False):
                import webbrowser
                webbrowser.open(state)

        # --- to JSON string ---
        else:
            state = json.dumps(state, indent=4)

        # --- save -----------------------------------------------------
        if save:
            with open(save, 'wt') as f:
                f.write(state + '\n')

        # --- print ----------------------------------------------------
        if print:
            self.stdio.info(state)

        return json_state
