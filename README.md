# ngtools

Advanced tools for neuroglancer (conversion, tracts visualization, ...)

# Installation

```shell
pip install git+https://github.com/neuroscales/ngtools
```

## Local neuroglancer app

This package implements an application built around a local
[neuroglancer](https://github.com/google/neuroglancer) instance.
It comes with a set of shell-like commands that simplifies loading and
manipulating layers, with a strong focus on neuroimaging.

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
<pre><code>fileserver:   http://127.0.0.1:9123/
neuroglancer: http://127.0.0.1:9321/v/1/

Type <b>help</b> to list available commands, or <b>help &lt;command&gt;</b> for specific help.
Type <b>Ctrl+C</b> to interrupt the current command and <b>Ctrl+D</b> to exit the app.
<b>[1]</b>
</code></pre>

Let's start with the list of commands
<pre><code><b>[1] <ins>help</ins></b>
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
Finally, we use an LIP frame (left, inferior, posterior)  to display
the data
<pre><code><b>[9] <ins>display</ins></b> LIP
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
