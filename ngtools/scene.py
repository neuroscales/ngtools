"""A neuroglancer scene with a programmatic interface."""
# stdlib
import json
from io import BytesIO
from os import PathLike
from typing import Any, Literal, Sequence
from urllib.parse import quote as urlquote
from urllib.parse import unquote as urlunquote
from urllib.parse import urlparse

# externals
import neuroglancer as ng
import numpy as np
from neuroglancer.viewer_state import wrapped_property
from numpy.typing import ArrayLike
from upath import UPath

# internals
import ngtools.spaces as S
import ngtools.transforms as T
from ngtools.datasources import (
    LayerDataSource,
    LayerDataSources,
    MeshDataSource,
    SkeletonDataSource,
    VolumeDataSource,
)
from ngtools.local.tracts import TractDataSource
from ngtools.opener import exists, open, parse_protocols
from ngtools.shaders import colormaps, shaders
from ngtools.units import convert_unit, split_unit
from ngtools.utils import DEFAULT_URL

# monkey-patch Layer state to expose channelDimensions
ng.Layer.channel_dimensions = ng.Layer.channelDimensions = wrapped_property(
    "channelDimensions", ng.CoordinateSpace
)
ng.Layer.local_dimensions = ng.Layer.localDimensions = ng.Layer.layerDimensions


def _ensure_list(x: object) -> list:
    """Ensure that an object is a list. Make one if needed."""
    if isinstance(x, np.ndarray):
        x = x.tolist()
    if not isinstance(x, (list, tuple)):
        x = [x]
    return list(x)


_print = print


URILike = str | PathLike
SourceType = ng.LayerDataSource | ng.LocalVolume | ng.skeleton.SkeletonSource


class LayerFactory(type):
    """Metaclass for layers."""

    _JSONData = URILike | dict | ng.LayerDataSource | ng.Layer | None

    def __call__(  # noqa: D102
        cls,
        json_data: _JSONData = None,
        *args, **kwargs
    ) -> "Layer":
        if isinstance(json_data, (str, PathLike)):
            json_data = {"url": str(json_data)}
        if isinstance(json_data, (ng.LayerDataSource, ng.Layer)):
            json_data = json_data.to_json()
        # only use the factory if it is not called from a subclass
        if cls is not Layer:
            return super().__call__(json_data, *args, **kwargs)
        # switch based on uri
        if "source" in kwargs:
            source = LayerDataSources(kwargs["source"])
        else:
            source = LayerDataSources(json_data)
        uri_as_str = source[0].url
        if isinstance(uri_as_str, (str, PathLike)):
            layer_type = parse_protocols(uri_as_str)[0]
            SpecialLayer = {
                'volume': ImageLayer,
                'labels': SegmentationLayer,
                'surface': MeshLayer,
                'mesh': MeshLayer,
                'tracts': TractLayer,
                'skeleton': SkeletonLayer,
            }.get(layer_type, None)
            if SpecialLayer:
                return SpecialLayer(json_data, *args, **kwargs)
        # switch based on data source type
        source = LayerDataSources(json_data, *args, **kwargs)
        if isinstance(source[0], VolumeDataSource):
            return ImageLayer(source=source)
        if isinstance(source[0], SkeletonDataSource):
            return SkeletonLayer(source=source)
        if isinstance(source[0], MeshDataSource):
            return MeshLayer(source=source)
        raise ValueError("Cannot guess layer type for:", source)


class Layer(ng.Layer, metaclass=LayerFactory):
    """Base class for all layers."""

    def __getattribute__(self, name: str) -> Any:
        value = super().__getattribute__(name)
        if name == "source":
            value = LayerDataSources(value)
        return value


class ImageLayer(Layer, ng.ImageLayer):
    """Image layer."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self.source and not self.shaderControls:
            source = self.source[0]
            mn, q0, q1, mx = source.quantiles([0.0, 0.01, 0.99, 1.0])
            self.shaderControls = {
                "normalized": {
                    "range": np.stack([q0, q1]).tolist(),
                    "window": np.stack([mn, mx]).tolist(),
                }
            }


class SegmentationLayer(Layer, ng.SegmentationLayer):
    """Segmentation layer."""

    ...


class SkeletonFactory(LayerFactory):
    """Factory for skeletons."""

    def __call__(self, uri: URILike, *args, **kwargs) -> "Layer":  # noqa: D102
        if isinstance(uri, (str, PathLike)):
            layer_type = parse_protocols(uri)[0]
            layer = {
                'tracts': TractLayer,
                'skeleton': SkeletonLayer,
            }.get(layer_type, lambda *a, **k: None)(uri, *args, **kwargs)
            if layer:
                return layer
        source = LayerDataSource(uri, *args, **kwargs)
        if isinstance(source, TractDataSource):
            return TractLayer(source)
        if isinstance(source, SkeletonDataSource):
            return SkeletonLayer(source)
        raise ValueError("Cannot guess layer type for:", source)


class SkeletonLayer(SegmentationLayer, metaclass=SkeletonFactory):
    """Skeleton layer."""

    ...


class TractLayer(SkeletonLayer):
    """Tract layer."""

    def __init__(self, *args, **kwargs) -> None:
        source = TractDataSource(*args, **kwargs)
        max_tracts = len(source.url.tractfile.streamlines)
        controls = {
            "nbtracts": {
                "min": 0,
                "max": max_tracts,
                "default": source.url.max_tracts,
            }
        }
        super().__init__(
            source=[source],
            skeleton_shader=shaders.trkorient,
            selected_alpha=0,
            not_selected_alpha=0,
            segments=[1],
            shaderControls=controls,
        )


class MeshLayer(Layer, ng.SingleMeshLayer):
    """Mesh layer."""

    ...


class ManagedLayer(ng.ManagedLayer):
    """Named layer."""

    def __setattr__(self, key: str, value: dict | ng.Layer) -> None:
        if key == "layer":
            value = Layer(value)
        return super().__setattr__(key, value)


class Layers(ng.Layers):
    """List of named layers."""

    class _ManagedLayersList(list):

        def __init__(self, other: list[ng.ManagedLayer]) -> None:
            super().__init__(map(ManagedLayer, other))

        def append(self, x: ng.ManagedLayer) -> None:
            super().append(ManagedLayer(x))

        def extend(self, other: list[ng.ManagedLayer]) -> None:
            super().extend(map(ManagedLayer, other))

        def __setitem__(self, k: int, v: ng.ManagedLayer) -> None:
            super().__setitem__(k, ManagedLayer(v))

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._layers = self._ManagedLayersList(self._layers)


class ViewerState(ng.ViewerState):
    """Smart ng.ViewerState that knows default values set in the frontend."""

    def __setattr__(self, name: str, value: object) -> None:
        if hasattr(self, f"__set_{name}"):
            value = getattr(self, f"__set_{name}")(value)
        return super().__setattr__(name, value)

    def __getattribute__(self, name: str) -> object:
        try:
            return super().__getattribute__(f"__get_{name}")()
        except Exception:
            ...
        return super().__getattribute__(name)

    def __set_layer(self, value: ng.Layers) -> Layers:
        return Layers(value)

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
        self.dimensions = super().__getattribute__("dimensions")
        return super().__getattribute__("dimensions")

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
            = super().__getattribute__("relative_display_scales")
        return super().__getattribute__("relative_display_scales")

    __get_relativeDisplayScales = __get_relative_display_scales

    def __get_display_dimensions(self) -> list[str]:
        """Name of (up to three) displayed dimensions."""
        self.display_dimensions \
            = super().__getattribute__("display_dimensions")
        return super().__getattribute__("display_dimensions")

    def __set_display_dimensions(self, value: list[str] | None) -> None:
        dimensions = self.dimensions.items()
        if value is None:
            value = []
        value = [name for name in value if name in dimensions]
        dimensions = [name for name in dimensions if name not in value]
        value = (value + dimensions)[:3]
        return value

    def __get_cross_section_orientation(self) -> np.ndarray:
        """Orientation of the cross section view."""
        self.cross_section_orientation \
            = super().__getattribute__("cross_section_orientation")
        return super().__getattribute__("cross_section_orientation")

    def __set_cross_section_orientation(self, value: ArrayLike) -> None:
        if value is None:
            value = [0, 0, 0, 1]
        value = np.ndarray(value).tolist()
        value = value + max(0, 4 - len(value)) * [0]
        value = value[:4]
        value = np.ndarray(value)
        value /= (value**2).sum()**0.5
        super().cross_section_orientation = value

    __get_crossSectionOrientation = __get_cross_section_orientation
    __set_crossSectionOrientation = __set_cross_section_orientation

    def _default_position(self) -> list[float]:
        """
        Compute a smart default position (center of the fist).

        NOTE: positions are expressed in "model scaled space". That is,
        it the "z" dimension listed in `dimensions` has scale (0.5, "mm"),
        a position increment of 1 will correspond to an effective increment
        of 0.5 mm.
        """
        pos = [0.0] * len(self.dimensions.names)
        for layer in self.layers:
            layer: ng.ManagedLayer
            if not layer.visible:
                continue
            layer = layer.layer
            if getattr(layer, "source", []) == 0:
                continue
            source = layer.source[0]
            if not hasattr(source, "output_center"):
                continue
            center = source.output_center
            for i, (name, (scale, unit)) \
                    in enumerate(self.dimensions.to_json().items()):
                if name not in source.output_dimensions.names:
                    continue
                j = source.output_dimensions.names.index(name)
                unit0 = source.output_dimensions.units[j]
                value = convert_unit(center[j], unit0, unit)
                pos[i] = value / scale
        return pos

    def __get_position(self) -> list[float]:
        if not super().__getattribute__("position"):
            self.position = self._default_position()
        return super().__getattribute__("position")


class Scene(ViewerState):
    """A neuroglancer scene with a programmatic interface."""

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
        """
        # prepare names and URLs
        names = []
        if isinstance(uri, dict):
            names = list(uri.keys())
            uri = uri.values()
        uris = _ensure_list(uri or [])
        names = _ensure_list(names or [])

        nb_layers_0 = len(self.layers)

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
        if transform:
            self.transform(transform, name=onames)

        if nb_layers_0 == 0:
            self.position = self._default_position()

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

    def world_axes(self, axes: dict[str, str] | list[str] | str | None = None
                   ) -> None:
        """
        Map native model axes (`"x"`, `"y"`, `"z"`, `"t"`) to
        neuroanatomical or arbitrary names.

        Parameters
        ----------
        axes : dict[str, str] | list[str] | str | None
            Mapping from native to user names.
            If None, simply return current mapping.

        Returns
        -------
        axes : dict[str, str]
            Mapping from native to user names.
        """
        def make_local_annot() -> ng.AnnotationLayer:
            coord = ng.CoordinateSpace()
            return ng.AnnotationLayer(ng.LocalAnnotationLayer(coord))

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

        new_axes = axes
        axes = {"x": "x", "y": "y", "z": "z", "t": "t"}
        axes.update({
            native: current
            for native, current in zip(
                world_axes_native.source[0].transform.output_dimensions.names,
                world_axes_current.source[0].transform.output_dimensions.names
            )
        })

        if new_axes:
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
        return axes

    def rename_axes(
        self,
        axes: str | list[str] | dict[str],
        layer: str | list[str] | None = None,
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

        Returns
        -------
        axes : dict[str]
            Mapping from old names to new names
        """
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

    def _world2view(
        self, transform: np.ndarray | None = None
    ) -> ng.CoordinateSpaceTransform:
        """Set or get the world-to-view transform (must be a pure rotation)."""
        if "__world_to_view__" not in self.layers:
            self.layers.append(ng.ManagedLayer(
                name="__world_to_view__",
                layer=ng.AnnotationLayer(ng.LocalAnnotationLayer()),
                archived=True
            ))
        __world_to_view__ = self.layers["__world_to_view__"]
        old_transform = __world_to_view__.source[0].transform

        if transform is None:
            return old_transform

        transform = np.asarray(transform)
        rank = len(transform)
        names = ["x", "y", "z"][:rank]
        matrix = np.eye(rank+1)[:-1]
        matrix[:, :-1] = transform
        dimensions = ng.CoordinateSpace({
            name: [1, ""] for name in names
        })
        transform = ng.CoordinateSpaceTransform(
            matrix=matrix,
            input_dimensions=dimensions,
            output_dimensions=dimensions,
        )
        __world_to_view__.source[0].transform = transform
        return transform

    def _neurospace(self, space: str | list[str] | None) -> str:
        """Set or get the neurospace used for display."""
        current_space = S.space_to_name(self.dimensions, compact=True)

        if space is None:
            return current_space

        if isinstance(space, str):
            space = S.name_compact2full(space)
        compact_space = "".join(x[0].lower() for x in space)

        transform = S.neurotransforms[current_space, compact_space]
        self._apply_transform(transform)
        return compact_space

    def space(
        self,
        space: str | list[str] | None = None,
        layer: str | None = None
    ) -> str:
        """
        Rotate the view such that the cross-sections are aligned with
        the voxel axes of a layer.

        Parameters
        ----------
        space
            Name of space to show data in (`"ras"`, `"lpi"`, etc.)
        layer
            Name of a layer or `"world"`.
        """
        # If first input is a layer name, switch
        if layer is None and space in self.layers:
            layer, space = space, None

        # If no input arguments, simply return known dimensions
        if space is None and layer is None:
            return self.dimensions

        # Change neurospace
        space = self._neurospace(space)

        # If it's just a neurospace change, we're done
        if layer is None:
            return space

        # Otherwise, temporarily move to canonical space
        self._neurospace("xyz")

        # Move back to world space
        #   1. get current world2view transform
        world2view = self._world2view()
        #   2. apply its inverse to remove its effect
        view2world = T.inverse(world2view)
        self._apply_transform(view2world)
        #   3. reset world2view to identity
        self._world2view(np.eye(4)[:3])

        # Compute new world2view
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
        view2world = rot
        world2view = rot.T

        # Apply transform
        self._apply_transform(world2view)

        # Save world2view
        self._world2view(world2view[:-1])

        # Reset neurospace
        return self._neurospace(space)

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
        if len(dimensions) == 1:
            dimensions = list(dimensions[0])
        if len(dimensions) > 3:
            raise ValueError('display takes at most three axis names')
        dimensions = [S.letter2full.get(letter.lower(), letter)
                      for letter in dimensions]
        display_dimensions = dimensions

        def compactNames(names: list[str]) -> str:
            names = list(map(lambda x: x[0].lower(), names))
            names = ''.join([d for d in names if d not in 'ct'])
            return names

        names = compactNames(display_dimensions)
        self.space(names)

        self.display_dimensions = display_dimensions

    def transform(
        self,
        transform: (
            ng.CoordinateSpaceTransform | ArrayLike | str | PathLike | BytesIO
        ),
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
        layer_names = layer or []
        if not isinstance(layer_names, (list, tuple)):
            layer_names = [layer_names]

        # save current axes
        dimensions = self.dimensions
        display_dimensions = self.display_dimensions
        world2view = self._world2view()
        view2world = T.inverse(world2view)

        # go back to world xyz space
        self.space("xyz")

        # prepare transformation matrix
        transform = self._load_transform(transform, inv, mov=mov, fix=fix)
        transform = T.compose(world2view, transform, view2world, adapt=True)

        # apply transform
        self._apply_transform(transform, layer, adapt=True)

        # go back to original space
        self.space(S.space_to_name(dimensions))
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

        # make ng transform
        return ng.CoordinateSpaceTransform(
            matrix=transform,
            input_dimensions=S.neurospaces["xyz"],
            output_dimensions=S.neurospaces["xyz"],
        )

    def channel_mode(
        self,
        mode: Literal["local", "channel", "spatial"],
        layer: str | list[str] | None = None,
        dimension: str | list[str] = 'c',
    ) -> None:
        """
        Change the mode (local or intensity) of an axis.

        Parameters
        ----------
        mode : {"local", "channel", "spatial"}
            How to interpret this dimension:

            - "local": enables a switch/slider to change which channel
              is displayed (e.g., time).
            - "channel": enables the joint use of all channels to
              display a single intensity/color (e.g., RGB).
            - "spatial": enables the display of this axis in the
              cross-view.
        layer : [list of] str
            Names of layers to process
        dimension : [list of] str
            Names of dimensions to process
        """
        # NOTE
        #  localDimensions is named localDimensions in the neuroglancer
        #   python code
        dimensions = _ensure_list(dimension)
        layers = _ensure_list(layer or [])

        mode = mode[0].lower()
        if mode not in ('l', 'c', 's'):
            raise ValueError('Unknown channel mode. Should be one of '
                             '{local, channel, spatial}')

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

                elif mode == 's':   # SPATIAL
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

    def position(
        self,
        coord: float | list[float],
        dimensions: str | list[str] | None = None,
        unit: str | None = None,
        world: bool = False,
        layer: bool | str = False,
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
        world : bool
            Coordinate is in world frame.
            Cannot be used at the same time as `layer`.
        layer : bool or str
            Coordinate is in this layer's canonical frame.
            Cannot be used at the same time as `world`.

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
            print(', '.join(string))
            return position

        # Preproc dimensions
        if isinstance(dimensions, str):
            dimensions = [dimensions]
        dimensions = dimensions or list(map(str, dim.names))
        if len(dimensions) == 1 and len(dimensions[0]) > 1:
            dimensions = S.name_compact2full(dimensions[0])
        dimensions = dimensions[:len(coord)]

        # Preproc layer
        if world and layer:
            raise ValueError('Cannot use both --world and --layer')
        if world:
            layer = "world"
        if layer is True:
            layer = self.layers[0].name

        # Change space
        current_dimensions = list(map(str, dim.names))
        change_space = False
        world2view = None
        if layer or any(d not in current_dimensions for d in dimensions):
            world2view = self._world2view()
            self.space(dimensions, layer)
            change_space = True

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

        # Change space back
        if change_space:
            self.space(current_dimensions, "world")

            if world2view is not None:
                self._apply_transform(world2view)
                self._world2view(world2view.matrix)

        return list(map(float, self.position))

    def orient(
        self,
        position: list[float],
        dimensions: str | list[str] | None = None,
        units: str | None = None,
        world: bool = False,
        layer: bool | str = False,
        reset: bool = False,
    ) -> None:
        """NOT IMPLEMETED YET."""
        raise NotImplementedError('Not implemented yet (sorry!)')

    def shader(
        self,
        shader: str | PathLike,
        layer: str | list[str] | None = None,
    ) -> None:
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
            layer.shader = shader

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
            print(self.layout)
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
            Print/load a JSON URL rather than a JSON object
        print : bool
            Print the JSON object or URL

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
                _print(state)
            else:
                _print(json.dumps(state, indent=4))
        return state
