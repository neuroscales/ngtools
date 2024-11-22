"""Type aliases."""
# stdlib
from numbers import Number

# externals
import neuroglancer as ng
import numpy as np

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
