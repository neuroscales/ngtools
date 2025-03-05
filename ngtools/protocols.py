"""Lists of kown protocols."""

LAYERS = [
    'volume',           # Raster data (image or volume)
    'image',            # ^ alias
    'labels',           # Integer raster data, interpreted as labels
    'segmentation',     # ^ alias
    'surface',          # Triangular mesh
    'mesh',             # Other types of mesh ???
    'tracts',           # Set of piecewise curves
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
