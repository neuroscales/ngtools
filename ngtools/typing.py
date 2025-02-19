"""Type aliases."""
# ruff: noqa
# flake8: noqa

# stdlib
from collections.abc import Mapping, Sequence
from numbers import Number
from threading import RLock
from typing import Any, Generic, Literal, ParamSpec, TypeVar

# externals
import neuroglancer as ng
import neuroglancer.json_wrappers as ngj
import numpy as np

T = TypeVar('T')
P = ParamSpec('P')

LayerDataSourceURL = (
    ng.LocalVolume |
    ng.skeleton.SkeletonSource |
    str
)
"""Type of `LayerDataSource.url`."""

LayerDataSourceLike = ng.LayerDataSource | LayerDataSourceURL | dict
"""Types that can be used to construct a `LayerDataSource`."""

LayerDataSourceTransform = ng.CoordinateSpaceTransform
"""Type of `LayerDataSource.transform`."""

LayerDataSubSources = dict[str, ng.LayerDataSubsource]
"""Type of `LayerDataSource.subsources`."""

SourceType = ng.LayerDataSource | ng.LocalVolume | ng.skeleton.SkeletonSource

LayerDataSourcesLike = (
    ng.LayerDataSources |
    LayerDataSourceLike |
    list[LayerDataSourceLike]
)
"""Types that can be used to construct a `LayerDataSourcesLike`."""

CoordinateSpaceTransformOutputDimensions = ng.CoordinateSpace
CoordinateSpaceTransformInputDimensions = ng.CoordinateSpace
CoordinateSpaceTransformSourceRank = int
CoordinateSpaceTransformMatrix = np.ndarray[np.float64]


# Less useful types

ToolType = str
ShaderControlToolControl = str
DimensionToolDimension = str
SidePanelLocationSide = str
SidePanelLocationVisible = bool
SidePanelLocationSize = int
SidePanelLocationFlex = float
SidePanelLocationRow = int
SidePanelLocationCol = int
SelectedLayerStateLayer = str
LayerSidePanelStateTab = str
LayerSidePanelStateTabs = set[str]
DimensionPlaybackVelocityVelocity = float
DimensionPlaybackVelocityAtBoundary = str
DimensionPlaybackVelocityPaused = bool
LayerType = str
LayerDimensions = ng.CoordinateSpace
LayerPosition = np.ndarray[np.float32]
LayerLocalVelocity = dict[str, ng.DimensionPlaybackVelocity]
LayerTab = str
LayerPanels = list[ng.LayerSidePanelState]
LayerPanel = ng.LayerSidePanelState
LayerPick = bool
LayerToolBindings = dict[str, ng.Tool]
LayerTool = ng.Tool
PointAnnotationLayerPoints = np.ndarray[np.float32]
ImageLayerSource = ng.LayerDataSources
ImageLayerShader = str
ShaderControlParameter = (
    ng.TransferFunctionParameters |
    ng.InvlerpParameters |
    Number |
    str
)
ShaderControls = dict[str, ShaderControlParameter]
ImageLayerShaderControls = ShaderControls
ImageLayerOpacity = float
ImageLayerBlend = str
ImageLayerVolumeRenderingMode = bool | str
ImageLayerVolumeRenderingGain = float
ImageLayerCrossSectionRenderScale = float
SegmentationLayerSource = ng.LayerDataSources
StarredSegmentsLike = ng.StarredSegments | dict[int, bool]
VisibleSegmentsLike = ng.VisibleSegments | set[int]
SegmentationLayerStarredSegments = ng.StarredSegments
SegmentationLayerVisibleSegments = ng.VisibleSegments
SegmentationLayerEquivalences = ng.EquivalenceMap


# ----------------------------------------------------------------------
# These types are not actually used.
# These are just my way to document the ng API.
# ----------------------------------------------------------------------

class Array(np.ndarray, Generic[T]):
    dtype: T


class SizedArray(tuple, Array):
    ...


class JsonObjectWrapper(ngj.JsonObjectWrapper):

    ArgType = ngj.JsonObjectWrapper | dict | None

    def __init__(self, arg: ArgType = None, **kwargs): ...

    _json_data: dict[str, object] = {}
    _cached_wrappers: dict[str, None | tuple[Any, Any]] = {}
    _lock: RLock = RLock()
    _readonly: bool = False

    # Class-specific properties:
    # property(
    #   fget: Gets attribute from cache and postprocess it using a type.
    #   fset: Pass value through validator and saves it in cache.
    # )

    def __setattr__(self, key: str, value: object) -> None:
        # When calling `setattr(key, value)`, the value to set is stored in
        # the cache, along with the corresponding value in `json_data``:
        # ```
        # _cached_wrappers[key] = (value, _json_data[key])
        # ```
        ...

    def __getattr__(self, key: str) -> object:
        # When calling `getattr(key)`, the cached value and json_data value
        # are retrieved.
        # * If the second value in the cache (i.e., the cached json_data
        #   value) and the value in json_data match (same address), it
        #   means that the json_data has not been updated since the set
        #   data was cached, and the latter can be returned.
        # * Otherwise, the data in json_data is returned, but first it
        #   is transformed into the correct type using the type wrapper
        #   and cached.
        # ```
        # current_json_data = _json_data[key]
        # (value, old_json_data) = _cached_wrappers[key]
        # if old_json_data is not current_json_data:
        #   value = wrapped_type(current_json_data)
        #   _cached_wrappers[key] = (value, current_json_data)
        # return value
        # ```
        ...


class Layers(ng.Layers):
    # Implements the `list[ng.ManagedLayer]` protocol.
    # Also implements the `dict[str, ng.ManagedLayer]` protocol.
    ArgType = (
        list[ng.ManagedLayer | dict] |
        dict[str, ng.Layer | ng.local_volume.LocalVolume | dict | None] |
        None
    )


class ManagedLayer(JsonObjectWrapper, ng.ManagedLayer):
    ArgNameType = str | ng.ManagedLayer
    ArgLayerType = ng.Layer | ng.local_volume.LocalVolume | dict | None
    # !! `layer` cannot be a `ManagedLayer``

    def __init__(
        self, name: ArgNameType, layer: ArgLayerType = None, **kwargs
    ) -> None: ...

    name: str
    layer: ng.Layer = ng.Layer()
    visible: bool | None = True
    archived: bool | None = False
    # All other attributes redirect to `layer`'s attributes.


class Layer(JsonObjectWrapper, ng.Layer):
    ArgType = ng.Layer | dict | None

    # kwargs / properties
    type: str | None = None
    layer_dimensions: ng.CoordinateSpace = ng.CoordinateSpace()
    layer_position: Array[np.float32] = np.ndarray([], np.float32)
    local_velocity: dict[str, ng.DimensionPlaybackVelocity]
    tab: str | None = None
    panels: list[ng.LayerSidePanelState] = []
    pick: bool = None
    tool_bindings: dict[str, ng.Tool] = {}
    tool: ng.Tool | None = None


ShaderControl = Number | ng.TransferFunctionParameters | ng.InvlerpParameters


class ImageLayer(Layer, ng.ImageLayer):
    ArgType = ng.ImageLayer | dict | None

    # kwargs / properties
    type: Literal["image"] = "image"
    source: ng.LayerDataSources = ng.LayerDataSources()
    shader: str = ""
    shader_controls: dict[str, ShaderControl] = {}
    opacity: float | None = 0.5
    blend: str | None = None
    volume_rendering_mode: bool | str | None = False
    volume_rendering_gain: float | None = 0.
    volume_rendering_depth_samples: float | None = 64.
    cross_section_render_scale: float | None = 1.
    annotation_color: str | None = None


class SegmentationLayer(Layer, ng.SegmentationLayer):
    ArgType = ng.SegmentationLayer | dict | None

    # kwargs / properties
    type: Literal["segmentation"] = "segmentation"
    source: ng.LayerDataSources = ng.LayerDataSources()
    starred_segments: ng.StarredSegments = ng.StarredSegments()
    visible_segments: ng.VisibleSegments = ng.VisibleSegments(starred_segments)
    segments: ng.VisibleSegments = visible_segments
    equivalences: ng.EquivalenceMap = ng.EquivalenceMap()
    hide_segment_zero: bool | None = True
    hover_highlight: bool | None = True
    base_segment_coloring: bool | None = False
    selected_alpha: float | None = 0.5
    not_selected_alpha: float | None = 0.
    object_alpha: float | None = 1.
    saturation: float | None = 1.
    ignore_null_visible_set: bool | None = True
    skeleton_rendering: ng.SkeletonRenderingOptions = ng.SkeletonRenderingOptions()
    skeleton_shader: str | None = skeleton_rendering.shader
    color_seed: int | None = 0
    cross_section_render_scale: float | None = 1.
    mesh_render_scale: float | None = 10.
    mesh_silhouette_rendering: float | None = 0.
    segment_query: str | None = None
    segment_colors: dict[np.uint64, str] = {}
    segment_default_color: str | None = None
    segment_html_color_dict: dict[np.uint64, str]
    linked_segmentation_group: str | None = None
    linked_segmentation_color_group: str | Literal[False] | None = None


class SkeletonRenderingOptions(JsonObjectWrapper, ng.SkeletonRenderingOptions):
    ArgType = ng.SkeletonRenderingOptions | dict | None

    # kwargs / properties
    shader: str | None = None
    shader_controls: dict[str, ShaderControl] = {}
    mode2d: str | None = None
    line_width2d: float | None = 2.
    mode3d: str | None = None
    line_width3d: float | None = 1.


class StarredSegments(JsonObjectWrapper, ng.StarredSegments, Mapping[np.uint64, bool]):
    # Implements the `dict[uint64, bool]` protocol

    ArgType = (
        None |
        ng.StarredSegments |
        Mapping[int | str, bool] |
        Sequence[tuple[int | str, bool]] |
        Sequence[int | str]
    )
    _data: dict[np.uint64, bool] = {}
    _visible: dict[np.uint64, bool] = {}


class AnnotationLayer(Layer, ng.AnnotationLayer):
    ArgType = ng.AnnotationLayer | dict | None

    type: Literal["annotation"] = "annotation"
    source: ng.LayerDataSources = ng.LayerDataSources()
    annotations: list[ng.AnnotationBase | dict] = []
    annotation_color: str | None = None
    annotation_properties: list[ng.AnnotationPropertySpec] = []
    annotation_relationships: list[str] = []
    linked_segmentation_layer: dict[str, str] = {}
    filter_by_segmentation: list[str] = []
    ignore_null_segment_filter: bool | None = True
    shader: str
    shader_controls: dict[str, ShaderControl]


class LocalAnnotationLayer(AnnotationLayer, ng.LocalAnnotationLayer):
    ArgType = ng.CoordinateSpace | dict  # dimensions
    Arg2Type = ng.LocalAnnotationLayer | dict | None

    source: ng.LayerDataSources = ng.LayerDataSources(url="local://annotations")


class AnnotationBase(JsonObjectWrapper, ng.AnnotationBase):
    ArgType = ng.viewer_state.AnnotationBase | dict | None

    id: str | None = None
    type: Literal["point", "line", "axis_aligned_bounding_box", "ellipsoid"]
    description: str | None = None
    segments: list[list[np.uint64]] | None = None
    props: list[Number | str] | None = None


class PointAnnotation(AnnotationBase, ng.PointAnnotation):
    ArgType = ng.PointAnnotation | dict | None
    type: Literal["point"] = "point"


class LineAnnotation(AnnotationBase, ng.LineAnnotation):
    ArgType = ng.LineAnnotation | dict | None
    type: Literal["line"] = "line"
    point_a: Array[np.float32]
    point_b: Array[np.float32]


class AxisAlignedBoundingBoxAnnotation(AnnotationBase, ng.AxisAlignedBoundingBoxAnnotation):
    ArgType = ng.AxisAlignedBoundingBoxAnnotation | dict | None
    type: Literal["axis_aligned_bounding_box"] = "axis_aligned_bounding_box"
    point_a: Array[np.float32]
    point_b: Array[np.float32]


class EllipsoidAnnotation(AnnotationBase, ng.EllipsoidAnnotation):
    ArgType = ng.EllipsoidAnnotation | dict | None
    type: Literal["ellipsoid"] = "ellipsoid"
    center: Array[np.float32]
    radii: Array[np.float32]


class AnnotationPropertySpec(JsonObjectWrapper, ng.AnnotationPropertySpec):
    ArgType = ng.AnnotationPropertySpec | dict | None
    id: str
    type: str
    description: str | None = None
    default: Number | str | None = None
    enum_values: list[Number | str] | None = None
    enum_values: list[str] | None = None


class InvlerpParameters(JsonObjectWrapper, ng.InvlerpParameters):
    ArgType = ng.InvlerpParameters | dict | None
    range: SizedArray[Number, Number] | None = None
    window: SizedArray[Number, Number] | None = None
    channel: list[int] | None = None


class TransferFunctionParameters(JsonObjectWrapper, ng.TransferFunctionParameters):
    ArgType = ng.TransferFunctionParameters | dict | None
    window: SizedArray[Number, Number] | None = None
    channel: list[int] | None = None
    controlPoints: list[list[Number | str]] | None = None
    defaultColor: str | None = None


class LayerDataSources(list[ng.LayerDataSource], ng.LayerDataSources):
    ArgType = (
        str |
        ng.local_volume.LocalVolume |
        ng.skeleton.SkeletonSource |
        ng.LayerDataSource |
        list[
            ng.local_volume.LocalVolume |
            ng.skeleton.SkeletonSource |
            ng.LayerDataSource |
            dict |
            None
        ] |
        ng.LayerDataSources |
        dict |
        None
    )

    # Implements the `list` protocol
    _data: list[ng.LayerDataSource] = []


class LayerDataSource(JsonObjectWrapper, ng.LayerDataSource):
    ArgType = (
        str |
        ng.local_volume.LocalVolume |
        ng.skeleton.SkeletonSource |
        ng.LayerDataSource |
        dict |
        None
    )

    url: (
        str |
        ng.local_volume.LocalVolume |
        ng.skeleton.SkeletonSource
    )
    transform: ng.CoordinateSpaceTransform | None = None
    subsources: dict[str, ng.LayerDataSubsource] = {}
    enable_default_subsources: bool | None = True


DataPanelLayoutType = Literal["xy", "yz", "xz", "xy-3d", "yz-3d", "xz-3d", "4panel", "3d"]
DataPanelLayoutLike = ng.DataPanelLayout | DataPanelLayoutType

StackLayoutType = Literal["row", "column"]
StackLayoutLike = ng.StackLayout | StackLayoutType

LayerGroupViewerType = Literal["viewer"]
LayerGroupViewerLike = LayerGroupViewerType | ng.LayerGroupViewer

LayoutSpecification = (
    DataPanelLayoutLike |
    StackLayoutLike |
    LayerGroupViewerLike |
    dict |
    None
)


class StackLayout(JsonObjectWrapper, ng.StackLayout):
    ArgType = ng.StackLayout | dict | None

    type: Literal["row", "column"]
    flex: float | None = 1.0
    children: list[LayoutSpecification]


class LayerGroupViewer(JsonObjectWrapper, ng.LayerGroupViewer):
    # Same view of multiple layers
    ArgType = ng.LayerGroupViewer | dict | None

    type: Literal["viewer"] = "viewer"
    flex: float | None = 1.0
    layers: list[str] = []
    layout: DataPanelLayoutLike = ng.DataPanelLayout("xy")
    position: Array[np.float32] = ng.LinkedPosition()
    velocity: dict[str, ng.DimensionPlaybackVelocity] = {}
    cross_section_orientation: Array[np.float32] = ng.LinkedOrientationState()
    cross_section_scale: float = ng.LinkedZoomFactor()
    cross_section_depth: float = ng.LinkedDepthRange()
    projection_orientation: Array[np.float32] = ng.LinkedOrientationState()
    projection_scale: float = ng.LinkedZoomFactor()
    projection_depth: float = ng.LinkedZoomFactor()
    tool_bindings: dict[str, ng.Tool] = {}


class DataPanelLayout(JsonObjectWrapper, ng.DataPanelLayout):
    ArgType = str | ng.DataPanelLayout | dict | None

    type: Literal["xy", "yz", "xz", "xy-3d", "yz-3d", "xz-3d", "4panel", "3d"]
    cross_sections : ng.CrossSectionMap
    orthographic_projection: bool | None = False


class CrossSectionMap(dict[str, ng.CrossSection], ng.CrossSectionMap):
    ArgType = dict[str, ng.CrossSection] | None


class CrossSection(JsonObjectWrapper, ng.CrossSection):
    ArgType = ng.CrossSection | dict | None

    width: int | None = 1000
    height: int | None = 1000
    position: Array[np.float32] = ng.LinkedPosition()
    orientation: Array[np.float32] = ng.LinkedOrientationState()
    scale: float = ng.LinkedZoomFactor()
