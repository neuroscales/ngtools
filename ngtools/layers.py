"""A neuroglancer scene with a programmatic interface."""
# stdlib
import logging
from os import PathLike
from typing import Iterator

# externals
import neuroglancer as ng
import numpy as np

# internals
from ngtools.datasources import (
    LayerDataSources,
    MeshDataSource,
    SkeletonDataSource,
    VolumeDataSource,
)
from ngtools.local.tracts import TractDataSource, TractSkeleton
from ngtools.opener import parse_protocols
from ngtools.shaders import shaders
from ngtools.utils import Wraps

LOG = logging.getLogger(__name__)

URILike = str | PathLike
SourceType = ng.LayerDataSource | ng.LocalVolume | ng.skeleton.SkeletonSource


_LocalType = (ng.local_volume.LocalVolume, ng.skeleton.SkeletonSource)
_LocalLike = ng.local_volume.LocalVolume | ng.skeleton.SkeletonSource
_DataSourceLike = (
    str |
    ng.local_volume.LocalVolume |
    ng.skeleton.SkeletonSource |
    ng.LayerDataSource
)
_DataSourcesLike = (
    _DataSourceLike |
    list[_DataSourceLike | dict | None] |
    (ng.LayerDataSources | dict | None)
)
_LayerLike = ng.Layer | dict | None
_LayerArg = (
    _DataSourceLike |
    list[_DataSourceLike | dict | None] |
    ng.LayerDataSources |
    (ng.Layer | dict | None)
)


def _get_url(arg: _LayerArg, **kwargs) -> str | _LocalLike | None:
    if kwargs.get("source", None):
        return _get_url(kwargs.pop("source"), **kwargs)
    if kwargs.get("url", None):
        return _get_url(kwargs.pop("url"), **kwargs)
    if isinstance(arg, (str, PathLike)):
        return str(arg)
    if isinstance(arg, _LocalType):
        if hasattr(arg, "_url"):
            return arg._url
        return arg
    if hasattr(arg, "source"):
        return _get_url(arg.source)
    if hasattr(arg, "url"):
        return _get_url(arg.url)
    if isinstance(arg, dict):
        if arg.get("source", None):
            return _get_url(arg.get("source", arg))
        if arg.get("url", None):
            return _get_url(arg.get("url", arg))
    if hasattr(arg, "__iter__"):
        for arg1 in arg:
            url = _get_url(arg1, **kwargs)
            if url is not None:
                return url
    print("None")
    return None


def _get_source(arg: _LayerArg, **kwargs) -> _DataSourcesLike:
    if kwargs.get("source", None):
        return _get_url(kwargs.pop("source"), **kwargs)
    if isinstance(arg, (str, PathLike)):
        return str(arg)
    if isinstance(arg, _LocalType):
        return arg
    if hasattr(arg, "source"):
        return _get_url(arg.source)
    if hasattr(arg, "url"):
        return arg
    if isinstance(arg, dict):
        if arg.get("source", None):
            return _get_url(arg.get("source", arg))
        if "url" in arg:
            return arg
    if hasattr(arg, "__iter__"):
        return arg
    if isinstance(arg, (ng.LayerDataSource, ng.LayerDataSources)):
        return arg
    return None


class LayerFactory(type):
    """Metaclass for layers."""

    def __call__(cls, arg: _LayerArg = None, *args, **kwargs) -> "Layer":
        """
        Parameters
        ----------
        arg : ng.Layer or [list of] (str | dict | None)
        """
        arg_json = arg.to_json() if hasattr(arg, "to_json") else arg
        LOG.debug(f"LayerFactory({cls.__name__}, {arg_json})")

        # Only use the factory if it is not called from a subclass
        if cls is not Layer:
            LOG.debug(f"LayerFactory - defer to {cls.__name__}")
            return super().__call__(arg, *args, **kwargs)

        # Special case: pointAnnotation do not have `source`
        if (
            isinstance(arg, ng.PointAnnotationLayer) or
            kwargs.get("type", "") == "pointAnnotation" or
            (
                isinstance(arg, dict) and
                arg.get("type", "") == "pointAnnotation"
            )
        ):
            LOG.debug("LayerFactory - PointAnnotationLayer")
            return PointAnnotationLayer(arg, *args, **kwargs)

        # Check whether the main input is source-like object
        if not (
            arg is None or
            isinstance(arg, ng.Layer) or
            (isinstance(arg, dict) and "url" not in arg)
        ):
            if "source" in kwargs:
                raise ValueError(
                    "source-like object found in both positional "
                    "and keyword arguments."
                )
            kwargs["source"] = arg
            arg = None

        # Switch based on layer protocol
        url = _get_url(arg, **kwargs)
        LOG.debug(f"LayerFactory - url: {url}")

        GuessedLayer = None
        if url and isinstance(url, str):
            if url == "local://annotations":
                if isinstance(arg, ng.LocalAnnotationLayer):
                    odim = arg.source[0].transform.output_dimensions
                    return LocalAnnotationLayer(odim, arg, *args, **kwargs)
                else:
                    return AnnotationLayer(arg, *args, **kwargs)

            layer_type = parse_protocols(url).layer
            LOG.debug(f"LayerFactory - hint: {layer_type}")
            GuessedLayer = {
                "image": ImageLayer,
                "volume": ImageLayer,
                "segmentation": SegmentationLayer,
                "labels": SegmentationLayer,
                "mesh": MeshLayer,
                "surface": MeshLayer,
                "skeleton": SkeletonLayer,
                "tracts": TractLayer,
                "annotation": AnnotationLayer,
            }.get(layer_type, None)
            if GuessedLayer:
                return GuessedLayer(arg, *args, **kwargs)

        # Switch based on type keyword
        if hasattr(arg, "type"):
            layer_type = arg.type
        elif isinstance(arg, dict):
            layer_type = arg.get("type", None)
        else:
            layer_type = None
        LOG.debug(f"LayerFactory - json hint: {layer_type}")

        GuessedLayer = {
            "image": ImageLayer,
            "segmentation": SegmentationLayer,
            "mesh": SingleMeshLayer,
            "annotation": AnnotationLayer,
        }.get(layer_type, None)
        if GuessedLayer:
            return GuessedLayer(arg, *args, **kwargs)

        # Switch based on data source type
        LOG.debug("LayerFactory - build sources")
        sources = LayerDataSources(kwargs.get("source", None))
        kwargs["source"] = sources

        for source in sources:
            LOG.debug(f"LayerFactory - source: {str(type(source))}")
            if isinstance(source, VolumeDataSource):
                LOG.debug("LayerFactory - guess ImageLayer")
                GuessedLayer = ImageLayer
            elif isinstance(source, SkeletonDataSource):
                LOG.debug("LayerFactory - guess SkeletonLayer")
                GuessedLayer = SkeletonLayer
            elif isinstance(source, MeshDataSource):
                LOG.debug("LayerFactory - guess MeshLayer")
                GuessedLayer = MeshLayer
            if GuessedLayer:
                # kwargs["source"] = sources
                return GuessedLayer(*args, **kwargs)

        # Fallback
        LOG.debug("LayerFactory - fallback to simple Layer")
        return super().__call__(arg, *args, **kwargs)


class _SourceMixin:

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # Re-build data sources so that parameters for which
        # ngtools and neuroglancer use different defaults values
        # get set up according to ngtools conventions.
        LOG.debug("Layer - recompute data sources")
        self.source = LayerDataSources(self.source)

    def __get_source__(self) -> LayerDataSources:
        return LayerDataSources(self._wrapped.source)


class Layer(Wraps(ng.Layer), metaclass=LayerFactory):
    """Base class for all layers.

    See `ImageLayer`, `SegmentationLayer`, `MeshLayer` for
    keyword-parameters specific to these layers types.

    Parameters
    ----------
    json_data : dict | Layer | LayerDataSource | str
        Either a Layer (or its JSON representation),
        or a data source (or its JSON representation).

    Other Parameters
    ----------------
    type : {"image", "segmentation", "mesh", "annotation", "pointAnnotation"}
        Layer type -- guessed from source by default.
    source : [list of] LayerDataSource | str
        Data source. Except if layer is a `PointAnnotationLayer`.
    layer_dimensions : ng.CoordinateSpace
        Local dimensions specific to the layer.
    layer_position : ng.CoordinateSpace
        Position within the local dimensions.
    local_velocity : ng.DimensionPlaybackVelocity
        Speed for playback along local dimensions.
    tab : str
    panels : list[ng.LayerSidePanelState]
    pick : bool
    tool_bindings : dict[str, Tool | str]
        Key binding to tools, specific to this layer.
    tool : Tool | str
        Active tool (or tool name).
    """  # noqa: E501

    def __init__(self, *args, **kwargs) -> None:
        LOG.debug("Layer.__init__")
        super().__init__(*args, **kwargs)
        if hasattr(self, "source"):
            # Re-build data sources so that parameters for which
            # ngtools and neuroglancer use different defaults values
            # get set up according to ngtools conventions.
            LOG.debug("Layer - recompute data sources")
            self.source = LayerDataSources(self.source)

    def apply_transform(self, *args: ng.CoordinateSpaceTransform) -> "Layer":
        """Apply an additional transform in model space."""
        for source in getattr(self, "source", []):
            getattr(source, "apply_transform", (lambda *_: None))(*args)
        return self


"""
NOTE: Valid tools are
```
{
    # AnnotationLayer
    "annotatePoint": ng.PlacePointTool,
    "annotateLine": ng.PlaceLineTool,
    "annotateBoundingBox": ng.PlaceBoundingBoxTool,
    "annotateSphere": ng.PlaceEllipsoidTool,
    # ImageLayer
    "blend": ng.BlendTool,
    "opacity": ng.OpacityTool,
    "volumeRendering": ng.VolumeRenderingTool,
    "volumeRenderingGain": ng.VolumeRenderingGainTool,
    "volumeRenderingDepthSamples": ng.VolumeRenderingDepthSamplesTool,
    "crossSectionRenderScale": ng.CrossSectionRenderScaleTool,
    # SegmentationLayer
    "selectedAlpha": ng.SelectedAlphaTool,
    "notSelectedAlpha": ng.NotSelectedAlphaTool,
    "objectAlpha": ng.ObjectAlphaTool,
    "hideSegmentZero": ng.HideSegmentZeroTool,
    "hoverHighlight": ng.HoverHighlightTool,
    "baseSegmentColoring": ng.BaseSegmentColoringTool,
    "ignoreNullVisibleSet": ng.IgnoreNullVisibleSetTool,
    "colorSeed": ng.ColorSeedTool,
    "segmentDefaultColor": ng.SegmentDefaultColorTool,
    # MeshLayer
    "meshRenderScale": ng.MeshRenderScaleTool,
    "meshSilhouetteRendering": ng.MeshSilhouetteRenderingTool,
    "saturation": ng.SaturationTool,
    # SkeletonLayer
    "skeletonRendering.mode2d": ng.SkeletonRenderingMode2dTool,
    "skeletonRendering.mode3d": ng.SkeletonRenderingMode3dTool,
    "skeletonRendering.lineWidth2d": ng.SkeletonRenderingLineWidth2dTool,
    "skeletonRendering.lineWidth3d": ng.SkeletonRenderingLineWidth3dTool,
    # All
    "shaderControl": ng.ShaderControlTool,
    "dimension": ng.DimensionTool,
    # SegmentEdit
    "mergeSegments: ng.MergeSegmentsTool,
    "splitSegments: ng.SplitSegmentsTool,
    "selectSegments: ng.SelectSegmentsTool,
```
"""


class ImageLayer(_SourceMixin, Wraps(ng.ImageLayer), Layer):
    """Image layer.

    See `Layer` for keyword parameters common to all layers.

    Other Parameters
    ----------------
    source : [list of] LayerDataSource | str
    shader : str
        Default shader:
        ```glsl
        #uicontrol invlerp normalized
            void main() {
            emitGrayscale(normalized());
        }
        ```
    shader_controls : dict[str, float | dict | ng.InvlerpParameters | ng.TransferFunctionParameters]
        - InvlerpParameters:
            range: tuple[float, float]
            window: tuple[float, float]
            channel: list[int]
        - TransferFunctionParameters:
            window: tuple[float, float]
            channel: list[int]
            controlPoints : list[list[float]]
            defaultColor : str
    opacity : float, default=0.5
    blend : {"default", "additive"}, default="default"
    volume_rendering_mode : bool | {"off", "on", "max", "min"}, default=False
    volume_rendering_gain : float, default=0
    volume_rendering_depth_samples : float, default=64
    cross_section_render_scale : float, default=1
    """  # noqa: E501

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        if self.shader == "None":
            self.shader = shaders.default

        if self.source and not self.shaderControls:
            source = self.source[0]
            try:
                mn, q0, q1, mx = source.quantiles([0.0, 0.01, 0.99, 1.0])
                self.shaderControls = {
                    "normalized": {
                        "range": np.stack([q0, q1]).tolist(),
                        "window": np.stack([mn, mx]).tolist(),
                    }
                }
            except Exception as e:
                # The source may not be accessible, fail gracefully
                url = getattr(source, "url", None)
                url = getattr(source, "local_url", url)
                LOG.error(f"Could not compute quantiles for {url}: {e}")
                pass


class SegmentationLayer(_SourceMixin, Wraps(ng.SegmentationLayer), Layer):
    """Segmentation layer.

    See `Layer` for keyword parameters common to all layers.

    Other Parameters
    ----------------
    source : [list of] LayerDataSource | str
    starred_segments : StarredSegments | dict[int, bool]
    equivalences : ng.EquivalenceMap | list[list[int]]
    hide_segment_zero : bool, default=True
    hover_highlight : bool, default=True
    base_segment_coloring : bool, default=False
    selected_alpha : float, default=0.5
    not_selected_alpha : float, default=0
    object_alpha : float, default=1
    saturation : float, default=1
    ignore_null_visible_set : bool, default=True
    skeleton_rendering : ng.SkeletonRenderingOptions | dict
        shader : str, optional
        shader_controls : dict[str, float | dict | ng.InvlerpParameters | ng.TransferFunctionParameters]
            - InvlerpParameters:
                range: tuple[float, float]
                window: tuple[float, float]
                channel: list[int]
            - TransferFunctionParameters:
                window: tuple[float, float]
                channel: list[int]
                controlPoints : list[list[float]]
                defaultColor : str
        mode2d : {"lines", "lines_and_points"}, optional
        line_width2d : float, default=2
        mode3d: {"lines", "lines_and_points"}, optional
        line_width3d : float, default=3
    color_seed : int, default=0
    cross_section_render_scale : float, default=1
    mesh_render_scale : float, default=10
    mesh_silhouette_rendering : float, default=0
    segment_query : str, optional
    segment_colors : dict[uint64, str]
    segment_default_color : str, optional
    linked_segmentation_group : str, optional
    linked_segmentation_color_group : str | False
    """  # noqa: E501

    ...


class SkeletonLayerFactory(LayerFactory):
    """Factory for skeletons."""

    def __call__(  # noqa: D102
        cls, arg: _LayerArg = None, *args, **kwargs
    ) -> "SkeletonLayer":
        json_arg = arg.to_json() if hasattr(arg, "to_json") else arg
        LOG.debug(f"SkeletonLayerFactory({cls.__name__}, {json_arg})")

        # Only use the factory if it is not called from a subclass
        if cls is not SkeletonLayer:
            LOG.debug(f"SkeletonLayer - defer to {cls.__name__}")
            return super().__call__(arg, *args, **kwargs)

        # Check whether the main input is source-like object
        if not (
            arg is None or
            isinstance(arg, ng.Layer) or
            (isinstance(arg, dict) and "url" not in arg)
        ):
            if "source" in kwargs:
                raise ValueError(
                    "source-like object found in both positional "
                    "and keyword arguments."
                )
            kwargs["source"] = arg
            arg = None

        # Switch based on layer protocol
        url = _get_url(arg, **kwargs)
        LOG.debug(f"SkeletonLayerFactory - url: {url}")

        GuessedLayer = None
        if url:
            if isinstance(url, str):
                layer_type = parse_protocols(url).layer
                LOG.debug(f"SkeletonLayerFactory - hint: {layer_type}")
                GuessedLayer = {
                    "tracts": TractLayer,
                    "skeleton": SkeletonLayer,
                }.get(layer_type, None)
            elif isinstance(url, TractSkeleton):
                GuessedLayer = TractLayer
            elif isinstance(url, ng.skeleton.SkeletonSource):
                GuessedLayer = SkeletonLayer
            if GuessedLayer:
                if GuessedLayer is SkeletonLayer:
                    return super().__call__(arg, *args, **kwargs)
                else:
                    return GuessedLayer(arg, *args, **kwargs)

        # Switch based on data source type
        sources = LayerDataSources(kwargs.get("source", None))
        for source in sources:
            LOG.debug(f"SkeletonLayerFactory - source: {str(type(source))}")
            if isinstance(source, TractDataSource):
                LOG.debug("SkeletonLayerFactory - guessed TractLayer")
                GuessedLayer = TractLayer
            elif isinstance(source, SkeletonDataSource):
                LOG.debug("SkeletonLayerFactory - guessed SkeletonLayer")
                GuessedLayer = SkeletonLayer
            if GuessedLayer:
                kwargs["source"] = sources
                if GuessedLayer is SkeletonLayer:
                    return super().__call__(arg, *args, **kwargs)
                else:
                    return GuessedLayer(arg, *args, **kwargs)

        msg = f"Cannot guess layer type for {source}"
        LOG.error(f"SkeletonLayerFactory - {msg}")
        raise ValueError(msg)


class SkeletonLayer(SegmentationLayer, metaclass=SkeletonLayerFactory):
    """Skeleton layer."""

    shader = SegmentationLayer.skeleton_shader

    @property
    def shader_controls(self):  # noqa: ANN201, D102
        return self.skeleton_rendering.shader_controls

    @shader_controls.setter
    def shader_controls(self, value) -> None:  # noqa: ANN001
        self.skeleton_rendering.shader_controls = value

    @property
    def mode2d(self):  # noqa: ANN201, D102
        return self.skeleton_rendering.mode2d

    @mode2d.setter
    def mode2d(self, value) -> None:  # noqa: ANN001
        self.skeleton_rendering.mode2d = value

    @property
    def mode3d(self):  # noqa: ANN201, D102
        return self.skeleton_rendering.mode3d

    @mode3d.setter
    def mode3d(self, value) -> None:  # noqa: ANN001
        self.skeleton_rendering.mode3d = value

    @property
    def line_width2d(self):  # noqa: ANN201, D102
        return self.skeleton_rendering.line_width2d

    @line_width2d.setter
    def line_width2d(self, value) -> None:  # noqa: ANN001
        self.skeleton_rendering.line_width2d = value

    @property
    def line_width3d(self):  # noqa: ANN201, D102
        return self.skeleton_rendering.line_width3d

    @line_width3d.setter
    def line_width3d(self, value) -> None:  # noqa: ANN001
        self.skeleton_rendering.line_width3d = value


class TractLayer(SkeletonLayer):
    """Tract layer."""

    def __init__(self, arg: _LayerArg = None, *args, **kwargs) -> None:
        json_arg = arg.to_json() if hasattr(arg, "to_json") else arg
        LOG.debug(f"TractLayer({json_arg})")

        # already a TractLayer -> no deep copy
        if isinstance(arg, TractLayer) and not (args or kwargs):
            LOG.debug(f"TractLayer - defer to {self.__wrapped_class__}")
            super().__init__(arg)
            return

        # get source parameters
        if not (
            arg is None or
            isinstance(arg, ng.Layer) or
            (isinstance(arg, dict) and "url" not in arg)
        ):
            if "source" in kwargs:
                raise ValueError(
                    "source-like object found in both positional "
                    "and keyword arguments."
                )
            kwargs["source"] = arg
            arg = None
            LOG.debug(f"TractLayer - source: {kwargs['source']}")

        # split layer / source keywords
        ksrc = {}
        for key in kwargs:
            if not hasattr(TractLayer, key):
                ksrc[key] = kwargs.pop(key)

        # build source
        LOG.debug("TractLayer - build source...")
        source = kwargs["source"]
        if isinstance(source, (ng.LayerDataSources, list, tuple)):
            if len(source) != 1:
                raise ValueError(f"too many sources: {len(source)}")
            source = source[0]
        kwargs["source"] = TractDataSource(source, **ksrc)
        LOG.debug(f"TractLayer - build source: {kwargs['source'].to_json()}")

        LOG.debug(f"TractLayer - defer to {self.__wrapped_class__}")
        super().__init__(arg, *args, **kwargs)

        if self.skeleton_rendering.shader is None:
            self.skeleton_rendering.shader = shaders.skeleton.orientation
        if self.skeleton_rendering.mode2d is None:
            self.skeleton_rendering.mode2d = "lines"
        if self.skeleton_rendering.lineWidth2d == 2:
            self.skeleton_rendering.lineWidth2d = 0.01
        if self.selected_alpha is None:
            self.selected_alpha = 1
        if self.not_selected_alpha is None:
            self.not_selected_alpha = 0
        if not self.segments:
            self.segments = [1]

        # if "nbtracts" not in self.shader_controls:
        #     max_tracts = len(source.url.tractfile.streamlines)
        #     self.shader_controls["nbtracts"] = {
        #         "min": 0,
        #         "max": max_tracts,
        #         "default": source.url.max_tracts,
        #     }


class MeshLayer(Layer):
    """Common ancestors to all mesh layers.

    See also: `SingleMeshLayer`, `MultiscaleMeshLayer`
    """

    ...


class SingleMeshLayer(_SourceMixin, Wraps(ng.SingleMeshLayer), MeshLayer):
    """
    Single mesh layer.

    See `Layer` for keyword parameters common to all layers.

    See also: `MeshLayer`, `MultiscaleMeshLayer`.

    Other Parameters
    ----------------
    source : [list of] LayerDataSource | str
    vertex_attribute_sources : list[str]
    shader : str, optional
        Default shader:
        ```glsl
        void main() {
            emitGray();
        }
        ```
    vertex_attribute_names : list[str]
    """

    ...


class MultiscaleMeshLayer(SegmentationLayer):
    """Multiscale mesh layer.

    See also: `MeshLayer`, `SingleMeshLayer`.

    Other Parameters
    ----------------
    render_scale : float, default=10
    silhouette_rendering : float, default=0
    """

    render_scale = SegmentationLayer.mesh_render_scale
    silhouette_rendering = SegmentationLayer.mesh_silhouette_rendering


class AnnotationLayer(_SourceMixin, Wraps(ng.AnnotationLayer), Layer):
    """Annotation layer.

    See `Layer` for keyword parameters common to all layers.

    Other Parameters
    ----------------
    source : [list of] LayerDataSource | str
    annotations : list[AnnotationBase | dict]
        Valid annotations:
        ```
        {
            "point": PointAnnotation,
            "line": LineAnnotation,
            "axis_aligned_bounding_box": AxisAlignedBoundingBoxAnnotation,
            "ellipsoid": EllipsoidAnnotation,
        }
        ```
    annotation_properties : list[AnnotationPropertySpec | dict]
        id : str
        type : str
        description : str
        default : float | str
        enum_values : list[int | str]
        enum_labels : list[str]
    annotation_relationships : list[str]
    linked_segmentation_layer : dict[str, str]
    filter_by_segmentation : list[str]
    ignore_null_segment_filter : bool, default=True
    shader : str, optional
        Default shader:
        ```glsl
        void main() {
            setColor(defaultColor());
        }
        ```
    shader_controls : dict[str, float | dict | ng.InvlerpParameters | ng.TransferFunctionParameters]
        - InvlerpParameters:
            range: tuple[float, float]
            window: tuple[float, float]
            channel: list[int]
        - TransferFunctionParameters:
            window: tuple[float, float]
            channel: list[int]
            controlPoints : list[list[float]]
            defaultColor : str
    """  # noqa: E501

    ...


class LocalAnnotationLayer(Wraps(ng.LocalAnnotationLayer), AnnotationLayer):
    """TODO."""

    ...


class PointAnnotationLayer(Wraps(ng.PointAnnotationLayer), Layer):
    """Point annotation layer.

    Parameters
    ----------
    points : list[tuple[float, float, float]]
    """

    ...


class ManagedLayer(Wraps(ng.ManagedLayer), _SourceMixin):
    """Named layer."""

    def __get_layer__(self) -> Layer:
        return Layer(self._wrapped.layer)


class Layers(Wraps(ng.Layers)):
    """List of named layers."""

    def __getitem__(self, *args, **kwargs) -> ManagedLayer:
        return ManagedLayer(super().__getitem__(*args, **kwargs))

    def __iter__(self) -> Iterator[ManagedLayer]:
        for layer in super().__iter__():
            yield ManagedLayer(layer)
