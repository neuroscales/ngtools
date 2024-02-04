# ngtools

Advanced tools for neuroglancer (conversion, tracts viusalization, ...)

# Installation

```shell
pip install git+https://github.com/neuroscales/ngtools
```

## Local neuroglancer app

This package implements an application built around a local neuroglancer
instance. It comes with a set of shell-like commands that simplifies
loading and manipulating layers, with a string focus on neuroimaging.

In particular, it knows how to display data in neuroimaging cardinal axes
(Right/Anterior/Superior and their permutations). To this end, it
leverages the proposed
[NIfTI-Zarr specification](https://github.com/neuroscales/nifti-zarr),
which embeds a NIfTI header in OME-Zarr files.

### Getting started

To run the app, simply type
```shell
neuroglancer
```
in your shell. It will open a neuroglancer window, and a shell-like
interface:
```shell
fileserver:   http://127.0.0.1:9123/
neuroglancer: http://127.0.0.1:9321/v/1/

Type help to list available commands, or help <command> for specific help.
[1]
```

Let's start with the list of commands
```shell
[1] help
usage: [-h] {help,load,unload,transform,shader,state,display,exit,quit} ...

positional arguments:
  {help,load,unload,transform,shader,state,display,exit,quit}
    help                Display help
    load                Load a file
    unload              Unload a file
    transform           Apply a transform
    shader              Apply a shader
    state               Return the viewer's state
    display             Dimensions to display
    exit (quit)         Exit neuroglancer

optional arguments:
  -h, --help            show this help message and exit
```
We can load a bunch of files
```shell
[2] load /path/to/local_file.nii.gz
[3] load tiff:///path/to/file_without_extension
[4] load zarr://https://url.to/remote/zarr_asset --name my_image
```
change their colormaps
```shell
[5] shader blackred   --layer local_file.nii.gz
[6] shader blackgreen --layer file_without_extension
[7] shader blackblue  --layer my_image
```
and apply an affine transform
```shell
[8] transform /path/to/affine.lta --layer local_file.nii.gz
```
Finally, we use an LIP frame (left, inferior, posterior)  to display
the data
```shell
[9] display LIP
```

Note that even though we are using our own neuroglancer instance,
its state can be transferred to a remote instance. Assuming that only
remote files were loaded (or that the local fileserver is still running),
states will be compatible, and the remote instance will display the exact
same scene. The state can be obtained in JSON form, or in URL form
(which is simply the quoted version of the JSON)
```shel
[10] state
state
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
[11] state --url
https://neuroglancer-demo.appspot.com/#!%7B%22dimensions%22%3A%20%7B%22x%22%3A%20%5B0.001%2C%20%22m%22%5D%2C%20%22y%22%3A%20%5B0.001%2C%20%22m%22%5D%2C%20%22z%22%3A%20%5B0.001%2C%20%22m%22%5D%7D%2C%20%22displayDimensions%22%3A%20%5B%22x%22%2C%20%22y%22%2C%20%22z%22%5D%2C%20%22position%22%3A%20%5B2.5%2C%201.5%2C%2022.5%5D%2C%20%22crossSectionScale%22%3A%201%2C%20%22projectionScale%22%3A%20256%2C%20%22layers%22%3A%20%5B%7B%22type%22%3A%20%22image%22%2C%20%22source%22%3A%20%22nifti%3A//http%3A//127.0.0.1%3A9123/root%3A///Users/yb947/Dropbox/data/niizarr/sub-control01_T1w.nii.gz%22%2C%20%22tab%22%3A%20%22source%22%2C%20%22shaderControls%22%3A%20%7B%22normalized%22%3A%20%7B%22range%22%3A%20%5B0%2C%201140%5D%7D%7D%2C%20%22name%22%3A%20%22sub-control01_T1w.nii.gz%22%7D%5D%2C%20%22layout%22%3A%20%224panel%22%7D
```
