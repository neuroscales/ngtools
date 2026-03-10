"""Lists of known protocols."""
from collections import namedtuple

LAYERS = [
    'volume',           # Raster data (image or volume)
    'image',            # ^ alias
    'labels',           # Integer raster data, interpreted as labels
    'segmentation',     # ^ alias
    'surface',          # Triangular mesh
    'mesh',             # Other types of mesh ???
    'tracts',           # Set of piecewise curves
    'tractsv1',         # uses tracts as skeletons
    'tractsv2',         # converts tracts to precomputed annotations
    'roi',              # Region of interest ???
    'points',           # Pointcloud
    'transform',        # Spatial transform
    'affine',           # Affine transform
]

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
    'nibabel',          # All formats that ar eread by nibabel
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

FORMATS = NG_FORMATS + EXTRA_FORMATS

PROTOCOLS = [
    "file",          # Local file
    "http",          # HTTP
    "https",         # HTTPs
    "gs",            # Google Cloud
    "s3",            # Amazon S3
    "ftp",           # FTP
    "dandi",         # DANDI/LINC
    "local",         # protocol for LocalAnnotationLayer sources
    "python",        # protocol for LocalVolume/SkeletonSource
]

parsed_protocols = namedtuple(
    "parsed_protocols",
    ["layer", "format", "stream", "url"]
)


class parse_protocols(parsed_protocols):
    """Parse ngtools uri."""

    def __new__(cls, *args, **kwargs) -> "parse_protocols":  # noqa: D102
        layer = format = stream = url = None

        args = list(args)
        if args:
            url = str(args.pop(-1))
            # parse protocols
            *parts, url = str(url).split("://")
            # parse pipes
            url, *pipes = url.split("|")
            pipes = [p.strip().rstrip(":") for p in pipes]
            parts.extend(pipes)
            for part in parts:

                if part in PROTOCOLS:
                    if stream is not None:
                        raise ValueError("Too many streaming protocols:",
                                         stream, part)
                    stream = part

                elif part in LAYERS:
                    if layer is not None:
                        raise ValueError("Too many layer protocols:",
                                         layer, part)
                    layer = part

                elif part in FORMATS:
                    if format is not None:
                        raise ValueError("Too many format protocols:",
                                         format, part)
                    format = part

                else:
                    raise ValueError("Unknown protocol:", part)

        for i, arg in enumerate(reversed(args)):
            if i == 0:
                stream = arg
            elif i == 1:
                format = arg
            elif i == 2:
                layer = arg
            else:
                raise ValueError("Too many inputs")
        layer = kwargs.get("layer", layer)
        format = kwargs.get("format", format)
        stream = kwargs.get("stream", stream)
        url = kwargs.get("url", url)

        stream = stream or "file"
        if stream != "file":
            url = stream + "://" + url
        return super().__new__(cls, layer, format, stream, url)

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    @property
    def path(self) -> str:
        """Alias for url."""
        return self.url

    def __str__(self) -> str:
        out = self.url
        if self.stream and not self.url.startswith(self.stream):
            out = self.stream + "://" + out
        if self.format:
            out = self.format + "://" + out
        if self.layer:
            out = self.layer + "://" + out
        return out

    def with_part(self, **kwargs) -> "parse_protocols":
        """Replace parts."""
        if "stream" not in kwargs:
            url = kwargs.get("url", self.url)
            if "://" in url:
                kwargs["stream"] = url.split("://")[0]
        return parse_protocols(
            kwargs.get("layer", self.layer),
            kwargs.get("format", self.format),
            kwargs.get("stream", self.stream),
            kwargs.get("url", self.url),
        )

    def with_layer(self, layer: str | None) -> "parse_protocols":
        """Replace layer."""
        return self.with_part(layer=layer)

    def with_format(self, format: str | None) -> "parse_protocols":
        """Replace format."""
        return self.with_part(format=format)

    def with_stream(self, stream: str | None) -> "parse_protocols":
        """Replace stream."""
        return self.with_part(stream=stream)

    def with_url(self, url: str | None) -> "parse_protocols":
        """Replace url."""
        return self.with_part(url=url)

    def with_path(self, path: str | None) -> "parse_protocols":
        """Replace path."""
        return self.with_part(url=path)
