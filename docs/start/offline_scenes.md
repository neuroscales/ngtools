# Scene building without running a neuroglancer instance

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
