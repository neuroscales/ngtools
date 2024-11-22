"""A neuroglancer scene with a programmatic interface."""
# stdlib
import os.path
from io import BytesIO
from os import PathLike
from typing import Literal, Sequence

# externals
import neuroglancer as ng
import numpy as np
from numpy.typing import ArrayLike
from upath import UPath

# internals
import ngtools.datasources as D
import ngtools.local.datasources as LD
import ngtools.local.tracts as LT
import ngtools.opener as Op
import ngtools.spaces as S
import ngtools.transforms as T
import ngtools.units as U
from ngtools.shaders import colormaps, shaders


def _ensure_list(x: object) -> list:
    """Ensure that an object is a list. Make one if needed."""
    if isinstance(x, np.ndarray):
        x = x.tolist()
    if not isinstance(x, (list, tuple)):
        x = [x]
    return list(x)


URILike = str | PathLike
SourceType = ng.LayerDataSource | ng.LocalVolume | ng.skeleton.SkeletonSource


class LayerFactory(type):
    """Metaclass for layers."""

    def __call__(  # noqa: D102
        self, uri: URILike | ng.LayerDataSource, *args, **kwargs
    ) -> "Layer":
        if isinstance(uri, (str, PathLike)):
            layer_type = Op.parse_protocols(uri)[0]
            layer = {
                'volume': ImageLayer,
                'labels': SegmentationLayer,
                'surface': MeshLayer,
                'mesh': MeshLayer,
                'tracts': TractLayer,
                'skeleton': SkeletonLayer,
            }.get(layer_type, lambda *a, **k: None)(uri, *args, **kwargs)
            if layer:
                return layer
        source = D.LayerDataSource(uri, *args, **kwargs)
        if isinstance(source, D.VolumeDataSource):
            return ImageLayer(source)
        if isinstance(source, D.SkeletonDataSource):
            return SkeletonLayer(source)
        if isinstance(source, D.MeshDataSource):
            return MeshLayer(source)
        raise ValueError("Cannot guess layer type for:", source)


class Layer(ng.Layer, metaclass=LayerFactory):
    """Base class for all layers."""

    def __init__(self, *args, **kwargs):
        source = D.LayerDataSource(*args, **kwargs)
        super().__init__(source)


class ImageLayer(Layer, ng.ImageLayer):
    """Image layer."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        source = D.LayerDataSource(self.source[0])
        mn, q0, q1, mx = source.quantiles([0.0, 0.01, 0.99, 1.0])
        self.shaderControls = {
            "normalized": {
                "range": np.stack([q0, q1]),
                "window": np.stack([mn, mx]),
            }
        }


class SegmentationLayer(Layer, ng.SegmentationLayer):
    """Segmentation layer."""

    ...


class SkeletonFactory(LayerFactory):
    """Factory for skeletons."""

    def __call__(self, uri: URILike, *args, **kwargs) -> "Layer":
        if isinstance(uri, (str, PathLike)):
            layer_type = Op.parse_protocols(uri)[0]
            layer = {
                'tracts': TractLayer,
                'skeleton': SkeletonLayer,
            }.get(layer_type, lambda *a, **k: None)(uri, *args, **kwargs)
            if layer:
                return layer
        source = D.LayerDataSource(uri, *args, **kwargs)
        if isinstance(source, LT.TractDataSource):
            return TractLayer(source)
        if isinstance(source, D.SkeletonDataSource):
            return SkeletonLayer(source)
        raise ValueError("Cannot guess layer type for:", source)


class SkeletonLayer(SegmentationLayer, metaclass=SkeletonFactory):
    """Skeleton layer."""

    ...


class TractLayer(SkeletonLayer):
    """Tract layer."""

    def __init__(self, *args, **kwargs) -> None:
        source = LT.TractDataSource(*args, **kwargs)
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


class ViewerState(ng.ViewerState):
    """Smart ng.ViewerState that knows default values set in the frontend."""

    def _default_dimensions(self) -> ng.CoordinateSpace:
        dims = {}
        for layer in self.layers:
            layer = layer.layer
            source = D.LayerDataSource(layer.source)
            transform = source.transform
            odims = transform.output_dimensions
            dims.update({
                name: [1, U.split_unit(unit)[1]]
                for name, (_, unit) in odims.to_json().items()
                if not name.endswith(("^", "'"))
            })
        return dims

    def _fix_dimensions(self, value) -> ng.CoordinateSpace:
        default_value = self._default_dimensions
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

    @property
    def dimensions(self) -> ng.CoordinateSpace:
        self.value = super().dimensions
        return super().dimensions

    @dimensions.setter
    def dimensions(self, value: ng.CoordinateSpace | None) -> None:
        self.value = self._fix_dimensions(value)

    def _fix_relative_display_scales(self, value) -> dict[str, float]:
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

    @property
    def relative_display_scales(self) -> dict[str, float]:
        self.relative_display_scales = super().relative_display_scales
        return super().relative_display_scales

    @relative_display_scales.setter
    def relative_display_scales(self, value: dict[str, float] | None) -> None:
        value = self._fix_relative_display_scales(value)
        super().relative_display_scales = value

    relativeDisplayScales = relative_display_scales

    @property
    def display_dimensions(self) -> list[str]:
        self.value = super().relative_display_scales
        return super().relative_display_scales

    @display_dimensions.setter
    def display_dimensions(self, value: list[str] | None) -> None:
        dimensions = self.dimensions.items()
        if value is None:
            value = []
        value = [name for name in value if name in dimensions]
        dimensions = [name for name in dimensions if name not in value]
        value = (value + dimensions)[:3]
        super().display_dimensions = value

    @property
    def cross_section_orientation(self) -> np.ndarray:
        """Orientation of the cross section view."""
        self.cross_section_orientation = super().cross_section_orientation
        return super().cross_section_orientation

    @cross_section_orientation.setter
    def cross_section_orientation(self, value: ArrayLike) -> None:
        if value is None:
            value = [0, 0, 0, 1]
        value = np.ndarray(value).tolist()
        value = value + max(0, 4 - len(value)) * [0]
        value = value[:4]
        value = np.ndarray(value)
        value /= (value**2).sum()**0.5
        super().cross_section_orientation = value

    crossSectionOrientation = cross_section_orientation


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

        # load layers
        onames = []
        for n, uri in enumerate(uris):
            uri = str(uri).rstrip("/")
            name = names[n] if names else UPath(uri).name
            onames.append(name)
            layer = Layer(uri, **kwargs)
            self.layers.append(name=name, layer=layer)

        # rename axes according to current naming scheme
        self.rename_axes(self.world_axes(), layer=onames)

        # apply transform
        if transform:
            self.transform(transform, name=onames)

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
        # We save the mapping using the output dimensions of two
        # different local annotation layers.
        if "__world_axes_native__" not in self.layers:
            self.layers.append(ng.ManagedLayer(
                name="__world_axes_native__",
                layer=ng.AnnotationLayer(ng.LocalAnnotationLayer()),
                archived=True
            ))
        if "__world_axes_current__" not in self.layers:
            self.layers.append(ng.ManagedLayer(
                name="__world_axes_current__",
                layer=ng.AnnotationLayer(ng.LocalAnnotationLayer()),
                archived=True
            ))
        __world_axes_native__ = self.layers["__world_axes_native__"]
        __world_axes_current__ = self.layers["__world_axes_current__"]

        new_axes = axes
        axes = {"x": "x", "y": "y", "z": "z", "t": "t"}
        axes.update({
            native: current
            for native, current in zip(
                __world_axes_native__.transform.output_dimensions.names,
                __world_axes_current__.transform.output_dimensions.names
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

        __world_axes_native__.transform = ng.CoordinateSpaceTransform(
            output_dimensions={name: [1, ""] for name in axes.keys()}
        )
        __world_axes_current__.transform = ng.CoordinateSpaceTransform(
            output_dimensions={name: [1, ""] for name in axes.values()}
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
            if layers and layer.name not in layers:
                continue
            layer = named_layer.layer
            transform = layer.transform
            transform.output_dimensions = ng.CoordinateSpace({
                axes.get(name): scale
                for name, scale in transform.output_dimensions.to_json()
            })
            layer.transform = transform

        return axes

    def _world2view(
        self, rotation: np.ndarray | None = None
    ) -> ng.CoordinateSpaceTransform:
        """Set or get the world-to-view transform (must be a pure rotation)."""
        if "__world_to_view__" not in self.layers:
            self.layers.append(ng.ManagedLayer(
                name="__world_to_view__",
                layer=ng.AnnotationLayer(ng.LocalAnnotationLayer()),
                archived=True
            ))
        __world_to_view__ = self.layers["__world_to_view__"]
        old_transform = __world_to_view__.source.transform
        if rotation is None:
            return old_transform

        rotation = np.asarray(rotation)
        rank = len(rotation)
        names = list(self.world_axes().values())[:rank]
        matrix = np.eye(rank+1)[:-1]
        matrix[:, :-1] = rotation
        dimensions = ng.CoordinateSpace({
            name: [1, ""] for name in names
        })
        transform = ng.CoordinateSpaceTransform(
            matrix=matrix,
            output_dimensions=dimensions,
        )
        __world_to_view__.source.transform = transform
        return transform

    def _neurospace(self, space: str | list[str] | None) -> str:
        """Set or get the neurospace used for display."""
        current_space = self.dimensions.to_json().items()
        current_space = [
            name for name, (_, unit) in space
            if U.split_unit(unit)[-1][-1:] == "m"
        ]
        current_space = "".join([name[0].lower() for name in current_space])

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
        space : str
            Name of space to show data in (`"ras"`, `"lpi"`, etc.)
        layer : str
            Name of a layer or `"world"`.
        """
        # If first input is a layer name, switch
        if layer is None and space in self.layers:
            layer, space = space, None

        # If no input arguments, simply return known dimensions
        if space is None and layer is None:
            return self.dimensions

        if space is not None:
            transf

        if layer == "world":
            self.crossSectionOrientation = [0, 0, 0, 1]
            return

        elif layer in S.neurospaces:
            display = list(self.renamed_axes().values())
            this_space = "".join([axis[0].lower() for axis in display])
            matrix = S.neurotransforms[layer, this_space].matrix[:3, :3]
            if matrix.det() < 0:
                display = display[1::-1] + display[-1:]
                this_space = "".join([axis[0].lower() for axis in display])
                matrix = S.neurotransforms[layer, this_space].matrix[:3, :3]
                assert matrix.det() > 0

            u, _, vh = np.linalg.svd(matrix)
            rot = u @ vh
            quat = T.matrix_to_quaternion(rot)
            self.crossSectionOrientation = quat

        else:
            # move to canonical axes
            source = D.LayerDataSource(self.layers[layer].source)
            transform = T.subtransform(source.transform, 'm')
            transform = T.ensure_same_scale(transform)
            matrix = T.get_matrix(transform, square=True)[:-1, :-1]
            # remove scales and shears
            u, _, vh = np.linalg.svd(matrix)
            rot = u @ vh
            # preserve permutations and flips
            # > may not work if exactly 45 deg rotation so add a tiny bit of noise
            eps = np.random.randn(*rot.shape) * 1E-8 * rot.abs().max()
            orient = (rot + eps).round()
            assert np.allclose(orient @ orient.T, np.eye(len(orient)))
            rot = rot @ orient.T
            # convert to quaternion
            quat = T.matrix_to_quaternion(rot)
            self.crossSectionOrientation = quat

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

        def reorient(source: SourceType) -> bool:
            xform0 = D.LayerDataSource(source).transform
            onames = compactNames(xform0.outputDimensions.names)
            if all(name in onames for name in names):
                return False
            xform = S.neurotransforms[(onames, names)]
            source.transform = T.compose(xform, xform0)
            return True

        for layer in self.layers:
            layer = layer.layer
            if isinstance(layer, ng.ImageLayer):
                for source in layer.source:
                    reorient(source)

        self.display_dimensions = display_dimensions

    def transform(
        self,
        transform: ArrayLike[float] | str | PathLike | BytesIO,
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

        display_dimensions = self.display_dimensions
        self.display("ras")

        # prepare transformation matrix
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
        matrix1 = transform = transform[:-1]

        # make ng transform
        xform = ng.CoordinateSpaceTransform(
            matrix=transform,
            input_dimensions=S.neurospaces["ras"],
            output_dimensions=S.neurospaces["ras"],
        )

        def getDimensions(
            source: SourceType,
            _reentrant: bool = False
        ) -> ng.CoordinateSpace:
            if getattr(source, 'transform', None):
                transform = source.transform
                if transform.inputDimensions:
                    return transform.inputDimensions
            if getattr(source, 'dimensions', None):
                return source.dimensions
            if not _reentrant and not isinstance(source, ng.LocalVolume):
                mysource = RemoteSource.from_filename(source.url)
                return getDimensions(mysource, True)
            return None

        def getTransform(source: SourceType) -> ng.CoordinateSpaceTransform:
            if getattr(source, 'transform', None):
                return source.transform
            if not isinstance(source, ng.LocalVolume):
                mysource = RemoteSource.from_filename(source.url)
                if getattr(mysource, 'transform', None):
                    return mysource.transform
            return None

        def composeTransform(source: SourceType) -> bool:
            transform = getTransform(source)
            idims = getDimensions(source)
            matrix = transform.matrix
            if matrix is None:
                matrix = np.eye(4)
            odims = transform.outputDimensions
            xform0 = ng.CoordinateSpaceTransform(
                matrix=matrix,
                input_dimensions=idims,
                output_dimensions=odims,
            )
            if _mode != 'ras2ras':
                xform1 = T.subtransform(ng.CoordinateSpaceTransform(
                    input_dimensions=odims,
                    output_dimensions=odims,
                ), unit='m')
                xform1.matrix = matrix1
            else:
                xform1 = xform
            source.transform = T.compose(xform1, xform0)
            return True

        def applyTransform(source: SourceType) -> bool:
            idims = getDimensions(source)
            if not idims:
                return False
            xform0 = ng.CoordinateSpaceTransform(
                input_dimensions=idims,
                output_dimensions=idims,
            )
            if _mode != 'ras2ras':
                xform1 = T.subtransform(ng.CoordinateSpaceTransform(
                    matrix=matrix1,
                    input_dimensions=idims,
                    output_dimensions=idims,
                ), unit='m')
            else:
                xform1 = xform
            source.transform = T.compose(xform1, xform0)
            return True

        for layer in self.layers:
            if layer_names and layer.name not in layer_names:
                continue
            layer = layer.layer
            if isinstance(layer, ng.ImageLayer):
                for source in layer.source:
                    composeTransform(source)

        self.display(display_dimensions)

    def channel_mode(
        self,
        mode: Literal["local", "channel", "spatial"],
        layer: str | list[str] | None = None,
        dimension: str | list[str] = 'c',
        *,
        state: ng.ViewerState | None = None
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

        for layer in state.layers:
            if layers and layer.name not in layers:
                continue
            layer = layer.layer
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
                    if sdim not in state.dimensions.to_json():
                        dimensions = state.dimensions.to_json()
                        dimensions[sdim] = scale
                        state.dimensions = ng.CoordinateSpace(dimensions)
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
        *,
        state: ng.ViewerState | None = None,
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
            return coord([0] * len(state.dimensions))

        if not state.dimensions:
            raise RuntimeError(
                'Dimensions not known. Are you running the app in windowless '
                'mode? If yes, you must open a neuroglancer window to access '
                'or modifiy the cursor position')

        dim = state.dimensions

        # No argument -> print current position
        if not coord:
            string = []
            position = list(map(float, state.position))
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
            layer = self._WORLD
        if layer is True:
            layer = state.layers[0].name

        # Change space
        current_dimensions = list(map(str, dim.names))
        change_space = False
        to_world = None
        if layer or any(d not in current_dimensions for d in dimensions):
            to_world = self.to_world
            self.display(dimensions, layer, state=state)
            change_space = True

        # Convert unit
        unitmap = {n: u for u, n in zip(dim.units, dim.names)}
        current_units = [unitmap[d] for d in dimensions]
        coord = U.convert_unit(coord, unit, current_units)

        # Sort coordinate in same order as dim
        coord = {n: x for x, n in zip(coord, dimensions)}
        for x, n in zip(state.position, dim.names):
            coord.setdefault(n, x)
        coord = [coord[n] for n in dim.names]

        # Assign new coord
        state.position = list(coord.values())

        # Change space back
        if change_space:
            self.display(current_dimensions, self._WORLD, state=state)

            def lin2aff(x: ArrayLike) -> np.ndarray:
                matrix = np.eye(len(x)+1)
                matrix[:-1, :-1] = x
                return matrix

            if to_world is not None:
                self.transform(lin2aff(to_world.T), _mode='', state=state)
                self.to_world = to_world

        return list(map(float, state.position))

    def orient(
        self,
        position: list[float],
        dimensions: str | list[str] | None = None,
        units: str | None = None,
        world: bool = False,
        layer: bool | str = False,
        reset: bool = False,
        *,
        state: ng.ViewerState | None = None
    ) -> None:
        """NOT IMPLEMETED YET."""
        raise NotImplementedError('Not implemented yet (sorry!)')

    def shader(
        self,
        shader: str | PathLike,
        layer: str | list[str] | None = None,
        *,
        state: ng.ViewerState | None = None
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
            for layer in state.layers:
                layer_name = layer.name
                if layer_names and layer_name not in layer_names:
                    continue
                layer = layer.layer
                if not layer.channelDimensions.to_json():
                    self.channel_mode('channel', layer=layer_name, state=state)

        if hasattr(shaders, shader):
            shader = getattr(shaders, shader)
        elif hasattr(colormaps, shader):
            shader = shaders.colormap(shader)
        elif 'main()' not in shader:
            # assume it's a path
            shader = shaders.lut(shader)
        for layer in state.layers:
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
        state: ng.ViewerState | None = None,
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
            print(state.layout)
            return state.layout

        layout = _ensure_list(layout or [])

        layer = _ensure_list(layer or [])
        if (len(layout) > 1 or stack) and not layer:
            layer = [_.name for _ in state.layers]

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
            parent = state.layout
            while indices:
                parent = layout.children[indices.pop(0)]
            if layout and not isinstance(layout, ng.LayerGroupViewer):
                if not layer:
                    if len(parent.children):
                        layer = [L for L in parent.children[-1].layers]
                    else:
                        layer = [_.name for _ in state.layers]
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
            state.layout = layout
        return state.layout

    def zorder(
        self,
        layer: str | list[str],
        steps: int = None,
        *,
        state: ng.ViewerState | None = None,
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
            names += list(reversed([layer.name for layer in state.layers
                                    if layer.name not in names]))
            layers = {layer.name: layer.layer for layer in state.layers}
            for name in names:
                del state.layers[name]
            for name in reversed(names):
                state.layers[name] = layers[name]
        elif steps == 0:
            return
        else:
            # move up/down
            layers = {layer.name: layer.layer for layer in state.layers}
            indices = {name: n for n, name in enumerate(layers)}
            for name in layers.keys():
                indices[name] += steps * (1 if name in names else -1)
            layers = {
                name: layers[name]
                for name in sorted(layers.keys(), key=lambda x: indices[x])
            }
            for name in layers.keys():
                del state.layers[name]
            for name, layer in layers.items():
                state.layers[name] = layer

    # ==================================================================
    #
    #                            FILEUTILS
    #
    # ==================================================================

    def ensure_url(self, filename: str) -> str:
        """Ensure that a path is a proper URL. Make one if needed."""
        DANDIHINTS = (
            'dandi://',
            'DANDI:',
            'https://identifiers.org/DANDI:',
            'https://dandiarchive.org/',
            'https://gui.dandiarchive.org/',
            'https://api.dandiarchive.org/',
        )
        if filename.startswith(DANDIHINTS):
            return RemoteDandiFileSystem().s3_url(filename)

        if not filename.startswith(remote_protocols()):
            filename = os.path.abspath(filename)
            while filename.endswith('/'):
                filename = filename[:-1]
            prefix = f'http://{self.fileserver.ip}:{self.fileserver.port}/'
            filename = prefix + filename
        return filename

    # Protocols that describe the type of data contained in the file
    LAYERTYPES = [
        'volume',           # Raster data (image or volume)
        'labels',           # Integer raster data, interpreted as labels
        'surface',          # Triangular mesh
        'mesh',             # Other types of mesh ???
        'tracts',           # Set of piecewise curves
        'roi',              # Region of interest ???
        'points',           # Pointcloud
        'transform',        # Spatial transform
        'affine',           # Affine transform
    ]

    # Native neuroglancer formats
    NG_FORMATS = [
        'boss',             # bossDB: Block & Object storage system
        'brainmap',         # Google Brain Maps
        'deepzoom',         # Deep Zoom file-backed data source
        'dvid',             # DVID
        'graphene',         # Graphene Zoom file-backed data source
        'local',            # Local in-memory
        'n5',               # N5 data source
        'nggraph',          # nggraph data source
        'nifti',            # Single NIfTI file
        'obj',              # Wavefront OBJ mesh file
        'precomputed',      # Precomputed file-backed data source
        'render',           # Render
        'vtk',              # VTK mesh file
        'zarr',             # Zarr data source
        'zarr2',            # Zarr v2 data source
        'zarr3',            # Zarr v3 data source
    ]

    # Extra local formats (if not specified, try to guess from file)
    EXTRA_FORMATS = [
        'mgh',              # Freesurfer volume format
        'mgz',              # Freesurfer volume format (compressed)
        'trk',              # Freesurfer streamlines
        'lta',              # Freesurfer affine transform
        'surf',             # Freesurfer surfaces
        'annot',            # Freesurfer surface annotation
        'tck',              # MRtrix streamlines
        'mif',              # MRtrix volume format
        'gii',              # Gifti
        'tiff',             # Tiff volume format
        'niftyreg',         # NiftyReg affine transform
    ]

    def parse_filename(self, filename: str) -> str:
        """
        Parse a filename that may contain protocol hints.

        Parameters
        ----------
        filename : str
            A filename, that may be prepended by:

            1) A layer type protocol, which indicates the kind of object
               that the file contains.
               Examples: `volume://`, `labels://`, `tracts://`.
            2) A format protocol, which indicates the exact file format.
               Examples: `nifti://`, `zarr://`, `mgh://`.
            3) An access protocol, which indices the protocol used to
               access the files.
               Examples: `https://`, `s3://`, `dandi://`.

            All of these protocols are optional. If absent, a guess is
            made using the file extension.

        Returns
        -------
        layertype : str
            Data type protocol. Can be None.
        format : str
            File format protocol. Can be None.
        filename : str
            File path, eventually prepended with an access protocol.
        """
        layertype = None
        for dt in self.LAYERTYPES:
            if filename.startswith(dt + '://'):
                layertype = dt
                filename = filename[len(dt)+3:]
                break

        format = None
        for fmt in self.NG_FORMATS + self.EXTRA_FORMATS:
            if filename.startswith(fmt + '://'):
                format = fmt
                filename = filename[len(fmt)+3:]
                break

        if format is None:
            if filename.endswith('.mgh'):
                format = 'mgh'
            elif filename.endswith('.mgz'):
                format = 'mgz'
            elif filename.endswith(('.nii', '.nii.gz')):
                format = 'nifti'
            elif filename.endswith('.trk'):
                format = 'trk'
            elif filename.endswith('.tck'):
                format = 'tck'
            elif filename.endswith('.lta'):
                format = 'lta'
            elif filename.endswith('.mif'):
                format = 'mif'
            elif filename.endswith(('.tiff', '.tif')):
                format = 'tiff'
            elif filename.endswith('.gii'):
                format = 'gii'
            elif filename.endswith(('.zarr', '.zarr/')):
                format = 'zarr'
            elif filename.endswith('.vtk'):
                format = 'vtk'
            elif filename.endswith('.obj'):
                format = 'obj'
            elif filename.endswith(('.n5', '.n5/')):
                format = 'n5'

        if layertype is None:
            if format in ('tck', 'trk'):
                layertype = 'tracts'
            elif format in ('vtk', 'obj'):
                layertype = 'mesh'
            elif format in ('lta',):
                layertype = 'affine'

        return layertype, format, filename
