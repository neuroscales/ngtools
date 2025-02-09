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
    LayerDataSource,
    LayerDataSources,
    MeshDataSource,
    SkeletonDataSource,
    VolumeDataSource,
)
from ngtools.local.tracts import TractDataSource
from ngtools.opener import parse_protocols
from ngtools.shaders import shaders
from ngtools.utils import Wraps

LOG = logging.getLogger(__name__)

URILike = str | PathLike
SourceType = ng.LayerDataSource | ng.LocalVolume | ng.skeleton.SkeletonSource


_LayerJSONData = (
    ng.Layer | dict |
    ng.LayerDataSources | ng.LayerDataSource | URILike |
    None
)


def _get_url(json_data: _LayerJSONData) -> str:
    url = None
    if isinstance(json_data, ng.Layer):
        json_data = getattr(json_data, "source", None)
    if isinstance(json_data, dict):
        json_data = json_data.get("source", json_data)
    if isinstance(json_data, (str, PathLike)):
        return str(json_data)
    if isinstance(json_data, (list, tuple)) and json_data:
        if isinstance(json_data[0], (str, PathLike)):
            return str(json_data[0])
    if json_data:
        json_data = LayerDataSources(json_data)
    if json_data:
        url = json_data[0].url
    if not isinstance(url, str):
        url = None
    return url


class LayerFactory(type):
    """Metaclass for layers."""

    def __call__(  # noqa: D102
        cls,
        json_data: _LayerJSONData = None,
        *args, **kwargs
    ) -> "Layer":

        # Special case: pointAnnotation do not have `source`
        if (
            isinstance(json_data, ng.PointAnnotationLayer) or
            kwargs.get("type", "") == "pointAnnotation" or
            (
                isinstance(json_data, dict) and
                json_data.get("type", "") == "pointAnnotation"
            )
        ):
            LOG.debug("Layer - PointAnnotationLayer")
            return PointAnnotationLayer(json_data, *args, **kwargs)

        # Convert to JSON (slower but more robust)
        if hasattr(json_data, "to_json"):
            really_json_data = json_data.to_json()
        else:
            really_json_data = json_data

        # Get source
        if "source" in kwargs:
            source = kwargs.pop("source")
        elif hasattr(json_data, "source"):
            source = json_data.source
        elif "source" in really_json_data:
            source = really_json_data.pop("source")
        else:
            source = json_data
            json_data = None
        kwargs["source"] = source
        LOG.debug(f"Layer - source: {source}")

        # Only use the factory if it is not called from a subclass
        if cls is not Layer:
            LOG.debug(f"Layer - defer to {cls.__name__}")
            return super().__call__(json_data, *args, **kwargs)

        GuessedLayer = None

        # switch based on layer protocol
        uri_as_str = _get_url(source)
        if uri_as_str:
            if uri_as_str == "local://annotations":
                return LocalAnnotationLayer(json_data, *args, **kwargs)

            layer_type = parse_protocols(uri_as_str)[0]
            LOG.debug(f"Layer - hint: {layer_type}")
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
                return GuessedLayer(json_data, *args, **kwargs)

        # switch based on type keyword
        if "type" in (really_json_data or {}):
            layer_type = really_json_data["type"]
            LOG.debug(f"Layer - json hint: {layer_type}")
            GuessedLayer = {
                "image": ImageLayer,
                "segmentation": SegmentationLayer,
                "mesh": SingleMeshLayer,
                "annotation": AnnotationLayer,
            }.get(layer_type, None)
            if GuessedLayer:
                return GuessedLayer(json_data, *args, **kwargs)

        # switch based on data source type
        if source:
            source = LayerDataSources(source)
            kwargs["source"] = source
            LOG.debug(f"Layer - source type: {type(source[0]).__name__}")
            if isinstance(source[0], VolumeDataSource):
                LOG.debug("Layer - guess ImageLayer")
                GuessedLayer = ImageLayer
            if isinstance(source[0], SkeletonDataSource):
                LOG.debug("Layer - guess SkeletonLayer")
                GuessedLayer = SkeletonLayer
            if isinstance(source[0], MeshDataSource):
                LOG.debug("Layer - guess MeshLayer")
                GuessedLayer = MeshLayer
            if GuessedLayer:
                return GuessedLayer(json_data, *args, **kwargs)

        # Fallback
        LOG.debug("Layer - fallback to simple Layer")
        return super().__call__(json_data, *args, **kwargs)


class _SourceMixin:

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
            except Exception:
                # The source may not be accessible, fail gracefully
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


class SkeletonFactory(LayerFactory):
    """Factory for skeletons."""

    def __call__(self, uri: URILike, *args, **kwargs) -> "Layer":  # noqa: D102
        if isinstance(uri, (str, PathLike)):
            layer_type = parse_protocols(uri)[0]
            layer = {
                "tracts": TractLayer,
                "skeleton": SkeletonLayer,
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


class ManagedLayer(Wraps(ng.ManagedLayer)):
    """Named layer."""

    def __getattr__(self, name: str) -> Layer:
        if name == "layer":
            return Layer(super().__getattr__(name))
        return super().__getattr__(name)


class Layers(Wraps(ng.Layers)):
    """List of named layers."""

    def __getitem__(self, *args, **kwargs) -> ManagedLayer:
        return ManagedLayer(super().__getitem__(*args, **kwargs))

    def __iter__(self) -> Iterator[ManagedLayer]:
        for layer in super().__iter__():
            yield ManagedLayer(layer)
