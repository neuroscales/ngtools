# Local Python

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