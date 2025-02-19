# ngtools

Advanced tools for neuroglancer (conversion, tracts visualization, ...)

## Installation

```shell
pip install ngtools
```

## Description

`ngtools` contains a set of user-friendly utilties to accompany
[`neuroglancer`](https://github.com/google/neuroglancer) -- an
in-browser viewer for peta-scale volumetric data. Specifically, it
implements:

- a **local app** (`from ngtools.local.viewer import LocalNeuroglancer`)
  that runs a local `neuroglancer` instance, allows local files to be
  visualized, and implements additional file formats
  (`.trk`, `.tck`, `.tiff`, `'.mgh`).

  See: [**Local neuroglancer in python**](#Local-neuroglancer-in-python)

- a **shell console** (`nglocal --help`) for the local app with thorough
  documentation of each command, auto-completion and history.

  See: [**Local neuroglancer in the shell**](#Local-neuroglancer-in-the-shell)

- a **user-friendly python API** (`from ngtools.scene import Scene`) that
  simplifies the creation of neuroglancer scenes (and is used under
  the hood by `LocalNeuroglancer`).

  See: [**Scene building without running an instance**](#Scene-building-without-running-an-instance)

- **smart wrappers** (`ngtools.layers`, `ngtools.datasources`) around
  the neuroglancer python API, that can compute quantities that can
  normally only be accessed in the neuroglancer frontend (default
  transforms, voxel data, etc).

- **utilities** (`ngtools.shaders`, `ngtools.transforms`, `ngtools.spaces`)
  that greatly simplifies manipulating some of neuroglancer's most
  intricate features.

## Local neuroglancer in the shell

To run the app, simply type

```shell
nglocal
```

in your shell. It will open a neuroglancer window, and a shell-like
interface:
<pre><code>             _              _
 _ __   __ _| |_ ___   ___ | |___
| '_ \ / _` | __/ _ \ / _ \| / __|
| | | | (_| | || (_) | (_) | \__ \
|_| |_|\__, |\__\___/ \___/|_|___/
       |___/

fileserver:   http://127.0.0.1:9123/
neuroglancer: http://127.0.0.1:9321/v/1/

Type <b>help</b> to list available commands, or <b>help &lt;command&gt;</b> for specific help.
Type <b>Ctrl+C</b> to interrupt the current command and <b>Ctrl+D</b> to exit the app.
<b>[1]</b>
</code></pre>

Let's start with the list of commands
<pre><code><b>[1] <ins>help</ins></b>
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
    channel_mode        Change the way a dimension is interpreted
    shader              Apply a shader
    display             Dimensions to display
    layout              Layout
    state               Return the viewer's state
    move (position)     Move cursor
    zoom                Zoom
    unzoom              Zoom
    zorder              Reorder layers
    cd                  Change directory
    ls                  List files
    ll                  List files (long form)
    pwd                 Path to working directory
    exit (quit)         Exit neuroglancer

options:
  -h, --help            show this help message and exit
</code></pre>

We can load a bunch of files
<pre><code><b>[2] <ins>load</ins></b> /path/to/local_file.nii.gz
<b>[3] <ins>load</ins></b> tiff:///path/to/file_without_extension
<b>[4] <ins>load</ins></b> zarr://https://url.to/remote/zarr_asset <b>--name</b> my_image
</code></pre>
change their colormaps
<pre><code><b>[5] <ins>shader</ins></b> blackred   <b>--layer</b> local_file.nii.gz
<b>[6] <ins>shader</ins></b> blackgreen <b>--layer</b> file_without_extension
<b>[7] <ins>shader</ins></b> blackblue  <b>--layer</b> my_image
</code></pre>
and apply an affine transform
<pre><code><b>[8] <ins>transform</ins></b> /path/to/affine.lta <b>--layer</b> local_file.nii.gz
</code></pre>

Note that even though we are using our own neuroglancer instance,
its state can be transferred to a remote instance. Assuming that only
remote files were loaded (or that the local fileserver is still running),
states will be compatible, and the remote instance will display the exact
same scene. The state can be obtained in JSON form, or in URL form
(which is simply the quoted version of the JSON)
<pre><code><b>[10] <ins>state</ins></b></code></pre>
```json
{
    "dimensions": {
        "x": [
            0.001,
            "m"
        ],
        "y": [
            0.001,
            "m"
        ],
        "z": [
            0.001,
            "m"
        ]
    },
    "position": [
        2.5,
        1.5,
        22.5
    ],
    "crossSectionScale": 1,
    "projectionScale": 256,
    "layers": [
        {
            "type": "image",
            "source": "nifti://http://127.0.0.1:9123/root:///Users/yb947/Dropbox/data/niizarr/sub-control01_T1w.nii.gz",
            "tab": "source",
            "shaderControls": {
                "normalized": {
                    "range": [
                        0,
                        1140
                    ]
                }
            },
            "name": "sub-control01_T1w.nii.gz"
        }
    ],
    "layout": "4panel"
}
```
<pre><code><b>[11] <ins>state</ins></b> --url
https://neuroglancer-demo.appspot.com/#!%7B%22dimensions%22%3A%20%7B%22x%22%3A%20%5B0.001%2C%20%22m%22%5D%2C%20%22y%22%3A%20%5B0.001%2C%20%22m%22%5D%2C%20%22z%22%3A%20%5B0.001%2C%20%22m%22%5D%7D%2C%20%22displayDimensions%22%3A%20%5B%22x%22%2C%20%22y%22%2C%20%22z%22%5D%2C%20%22position%22%3A%20%5B2.5%2C%201.5%2C%2022.5%5D%2C%20%22crossSectionScale%22%3A%201%2C%20%22projectionScale%22%3A%20256%2C%20%22layers%22%3A%20%5B%7B%22type%22%3A%20%22image%22%2C%20%22source%22%3A%20%22nifti%3A//http%3A//127.0.0.1%3A9123/root%3A///Users/yb947/Dropbox/data/niizarr/sub-control01_T1w.nii.gz%22%2C%20%22tab%22%3A%20%22source%22%2C%20%22shaderControls%22%3A%20%7B%22normalized%22%3A%20%7B%22range%22%3A%20%5B0%2C%201140%5D%7D%7D%2C%20%22name%22%3A%20%22sub-control01_T1w.nii.gz%22%7D%5D%2C%20%22layout%22%3A%20%224panel%22%7D
</code></pre>

## Local neuroglancer in python

The example above can also be run entirely from python.

Let us first instantiate a local viewer and open it in the browser:

```python
from ngtools.local.viewer import LocalNeuroglancer
import webbrowser

viewer = LocalNeuroglancer()
print('fileserver:  ', viewer.get_fileserver_url())
print('neuroglancer:', neuroglancer.get_viewer_url())

webbrowser.open(viewer.get_viewer_url())
```

We may then apply the same set of commands as before:

```python
viewer.load("/path/to/local_file.nii.gz")
viewer.load("tiff:///path/to/file_without_extension")
viewer.load({"my_image": "zarr://https://url.to/remote/zarr_asset"})
viewer.shader("blackred", layer="local_file.nii.gz")
viewer.shader("blackgreen", layer="file_without_extension")
viewer.shader("blackblue", layer="my_image")
viewer.transform("/path/to/affine.lta", layer="local_file.nii.gz")
```

Note that the viewer gets refreshed after each command is called.
Alternatively, one may group such commands so that they are
applied to the viewer in a single step:

```python
with viewer.scene() as scene:
    scene.load("/path/to/local_file.nii.gz")
    scene.load("tiff:///path/to/file_without_extension")
    scene.load({"my_image": "zarr://https://url.to/remote/zarr_asset"})
    scene.shader("blackred", layer="local_file.nii.gz")
    scene.shader("blackgreen", layer="file_without_extension")
    scene.shader("blackblue", layer="my_image")
    scene.transform("/path/to/affine.lta", layer="local_file.nii.gz")
```

While these two procedures (through the viewer or through the scene)
should in most cases yield equivalent results, it is not assured, as
the neuroglancer instance may discard some of the requested changes, or
add changes of its own in-between calls.

If the scene is compatible with a remote instance of neuroglancer
(_i.e._, it does not contain local data, and does not use ngtools
extended features such as new file formats), a url to a remote
neuroglancer scene can be generated:

```python
webbrowser.open(viewer.state(url=True))
```

Or, the JSON state can be printed and pasted into a neuroglancer window:

```python
viewer.state(print=True)
```

## Scene building without running an instance

Alternatively, a scene state may be built from scratch without ever
running a neuroglancer instance, using the `Scene` class. The syntax is
very similar to that of the `LocalNeuroglancer` class:

```python
from ngtools.scene import Scene
import webbrowser

scene = Scene()

scene.load("/path/to/local_file.nii.gz")
scene.load("tiff:///path/to/file_without_extension")
scene.load({"my_image": "zarr://https://url.to/remote/zarr_asset"})
scene.shader("blackred", layer="local_file.nii.gz")
scene.shader("blackgreen", layer="file_without_extension")
scene.shader("blackblue", layer="my_image")
scene.transform("/path/to/affine.lta", layer="local_file.nii.gz")

webbrowser.open(scene.state(url=True))
```
