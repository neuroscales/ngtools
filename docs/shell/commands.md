# All shell commands

## help

```shell
[1] help

usage: {help,load,unload,rename,...} ... [-h]

positional arguments:
  {help,load,unload,rename,...}
    help                Display help
    load                Load a file
    unload              Unload a file
    rename              Rename a file
    world_axes          Rename native axes
    rename_axes         Rename axes
    space               Cross-section orientation
    transform           Apply a transform
    save_transform      Save the current transform
    channel_mode        Change the way a dimension is interpreted
    shader              Apply a shader
    display             Dimensions to display
    layout              Layout
    state               Return the viewer's state
    move (position)     Move cursor
    zoom                Zoom by a factor [default: x2]
    unzoom              Unzoom by a factor [default: ÷2]
    zorder              Reorder layers
    cd                  Change directory
    ls                  List files
    ll                  List files (long form)
    pwd                 Path to working directory
    stdin               Set input stream
    stdout              Set output stream
    stderr              Set error stream
    exit (quit)         Exit neuroglancer

options:
  -h, --help            show this help message and exit
```

Display help

## load

```shell
[1] load

usage:  load FILENAME [FILENAME ...] [-h]
            [--name NAME [NAME ...]]
            [--transform TRANSFORM [TRANSFORM ...]]
            [--shader SHADER]

positional arguments:
  FILENAME              Filename(s) with protocols

options:
  -h, --help            show this help message and exit
  --name NAME [NAME ...], -n NAME [NAME ...]
                        A name for the image layer
  --transform TRANSFORM [TRANSFORM ...], -t TRANSFORM [TRANSFORM ...]
                        Apply a transform
  --shader SHADER, -s SHADER
                        Apply a shader
```

Load a file, which can be local or remote.

### Paths and URLs

Each path or url may be prepended by:

1) A layer type protocol that indicates the kind of object that the file
   contains.

   **Examples:** `volume://`, `labels://`, `tracts://`.

2) A format protocol that indicates the exact file format.

   **Examples:** `nifti://`, `zarr://`, `mgh://`.

3) An access protocol that indicates the protocol used to access the files.

   **Examples:** `https://`, `s3://`, `dandi://`.

All of these protocols are optional. If absent, a guess is made using the
file extension.

### Examples

* Absolute path to local file:  `/absolute/path/to/mri.nii.gz`
* Relative path to local file:  `relative/path/to/mri.nii.gz`
* Local file with format hint:  `mgh://relative/path/to/linkwithoutextension`
* Remote file:                  `https://url.to/mri.nii.gz`
* Remote file with format hint: `zarr://https://url.to/filewithoutextension`
* File on dandiarchive:         `dandi://dandi/<dandiset>/sub-<id>/path/to/file.ome.zarr`

### Layer names

Neuroglancer layers are named. The name of the layer can be specified with
the `--name` option. Otherwise, the base name of the file is used (that
is, without the folder hierarchy).

If multiple files are loaded and the `--name` option is used, then there
should be as many names as files.

### Transform

A spatial transform (common to all files) can be applied to the loaded
volume. The transform is specified with the `--transform` option, which
can be a flattened affine matrix (row major) or the path to a transform file.
Type help transform for more information.

### Shader

A shader (= colormap, common to all files) can be applied to the loaded
volume. The shader is specified with the `--shader` option, which
can be the name of a colormap, the path to a LUT file, or a snippet of
GLSL code. Type help shader for more information.
