# All shell commands

##

### help

=== ":octicons-terminal-24:"
    ```shell
    help [command]
    ```

Display help.

#### Positional arguments

| Name      | Type                       | Description                       |
| --------- | -------------------------- | --------------------------------- |
| `command` | `{load,unload,rename,...}` | Command for which to display help |

### load

=== ":octicons-terminal-24:"
    ```shell
    load <filename +> [--name +] [--transform +] [--shader 1]
    ```

Load a file, which can be local or remote.

#### Positional arguments

| Name       | Type        | Description                |
| ---------- | ----------- | -------------------------- |
| `filename` | `list[uri]` | Filename(s) with protocols |

#### Options

| Flag                | Type                       | Description                                                                  | Default              |
| ------------------- | -------------------------- | ---------------------------------------------------------------------------- | -------------------- |
| `-n`, `--name`      | `list[str]`                | A name for the image layer(s).                                               | `basename(filename)` |
| `-t`, `--transform` | `uri` &vert; `list[float]` | Apply a transform . Can be a path to a transform file or a flattened matrix. |                      |
| `-s`, `--shader`    | `uri` &vert; `str`         | The name of a builtin shader, or the path to a lookup table.                 |                      |

#### Paths and URLs

Each path or url may be prepended by:

1. A **layer** protocol that indicates the kind of object that the file
   contains.

    * **Examples:** `volume://`, `labels://`, `tracts://`.

2. A **format** protocol that indicates the exact file format.

    * **Examples:** `nifti://`, `zarr://`, `mgh://`.

3. An **access** protocol that indicates the protocol used to access the files.

    * **Examples:** `https://`, `s3://`, `dandi://`.

All of these protocols are optional. If absent, a guess is made using the
file extension.

| Examples                     |                                                           |
| ---------------------------- | --------------------------------------------------------- |
| Absolute path to local file  | `/absolute/path/to/mri.nii.gz`                            |
| Relative path to local file  | `relative/path/to/mri.nii.gz`                             |
| Local file with format hint  | `mgh://relative/path/to/linkwithoutextension`             |
| Remote file                  | `https://url.to/mri.nii.gz`                               |
| Remote file with format hint | `zarr://https://url.to/filewithoutextension`              |
| File on dandiarchive         | `dandi://dandi/<dandiset>/sub-<id>/path/to/file.ome.zarr` |

#### Layer names

Neuroglancer layers are named. The name of the layer can be specified with
the `--name` option. Otherwise, the base name of the file is used (that
is, without the folder hierarchy).

If multiple files are loaded and the `--name` option is used, then there
should be as many names as files.

#### Transform

A spatial transform (common to all files) can be applied to the loaded
volume. The transform is specified with the `--transform` option, which
can be a flattened affine matrix (row major) or the path to a transform file.

Type `help transform` for more information.

#### Shader

A shader (= colormap, common to all files) can be applied to the loaded
volume. The shader is specified with the `--shader` option, which
can be the name of a colormap, the path to a LUT file, or a snippet of
GLSL code.

Type `help shader` for more information.

### unload

=== ":octicons-terminal-24:"
    ```shell
    unload <layer+>
    ```

Unload layers

#### Positional arguments

| Name    | Type        | Description                    |
| ------- | ----------- | ------------------------------ |
| `layer` | `list[str]` | Name(s) of layer(s) to unload. |

### rename

=== ":octicons-terminal-24:"
    ```shell
    unload <src> <dst>
    ```

Rename a layer.

#### Positional arguments

| Name  | Type  | Description              |
| ----- | ----- | ------------------------ |
| `src` | `str` | Name of layer to rename. |
| `dst` | `str` | New name for the layer.  |

### world_axes

=== ":octicons-terminal-24: Implicit"
    ```shell
    world_axes <dst *> [--source *] [--print]
    ```

=== ":octicons-terminal-24: Explicit"
    ```shell
    world_axes [--destination *] [--source *] [--print]
    ```

Neuroglancer is quite flexible in the sense that it does not have a
predefined "hard" coordinate frame in which the data lives. Instead,
arbitrary "model" axes can be defines, in terms of an affine transformation
of "native" axes.

By default, most formats use **(x, y, z)** as their
model axes, although they may not have predefined anatomical meaning.
NIfTI files do have an anatomical convention, under which axes
<font color="red">**x**</font>,
<font color="green">**y**</font>,
<font color="blue">**z**</font>
map to the
<font color="red">**right**</font>,
<font color="green">**anterior**</font>,
<font color="blue">**superior**</font>
sides of the brain.

This function allows native axes to replaces by more anatomically
meaningful names, by leveraging neuroglancer's transforms. In order
to allow these transforms to be undone, and new data to be loaded without
introducing conflicting coordinate frames, we store the mapping inside
local annotation layers named `__world_axes_native__` and
`__world_axes_current__`.

!!! note "See also"
    [`space`](#space), [`display`](#display)

| Examples                             |                                         |
| ------------------------------------ | --------------------------------------- |
| `world_axes right anterior superior` | `x > right, y > anterior, z > superior` |
| `world_axes ras`                     | `x > right, y > anterior, z > superior` |
| `world_axes u v w --src z y x`       | `x > w,     y > v,        z > u`        |

#### Positional arguments

| Name  | Type  | Description     |
| ----- | ----- | --------------- |
| `dst` | `str` | New axis names. |

#### Options

| Flag                           | Type        | Description        | Default     |
| ------------------------------ | ----------- | ------------------ | ----------- |
| `-d`, `--dst`, `--destination` | `list[str]` | New axis names.    |             |
| `-s`, `--src`, `--source`      | `list[str]` | Native axis names. | `[x, y, z]` |
| `-p`, `--print`                |             | Print result.      |             |

### rename_axes

=== ":octicons-terminal-24: Implicit"
    ```shell
    rename_axes <dst *> [--source *] [--print]
    ```

=== ":octicons-terminal-24: Explicit"
    ```shell
    rename_axes [--destination *] [--source *] [--print]
    ```

Rename axes.

#### Positional arguments

| Name  | Type  | Description     |
| ----- | ----- | --------------- |
| `dst` | `str` | New axis names. |

#### Options

| Flag                           | Type        | Description     | Default |
| ------------------------------ | ----------- | --------------- | ------- |
| `-d`, `--dst`, `--destination` | `list[str]` | New axis names. |         |
| `-s`, `--src`, `--source`      | `list[str]` | Old axis names. |         |
| `-p`, `--print`                |             | Print result.   |         |

### space

=== ":octicons-terminal-24:"
    ```shell
    space [{radio,neuro,default}] [--layer 1]
    ```

This function rotates and orients the cross-section plane such that

* Visual axes are pointed according to some defined convention
  (radio or neuro)
* The cross section is aligned with either the model coordinate frame,
  or one of the layer's voxel space.

!!! note
    When this function is used, the displayed axes are always reset
    to
    <font color="red">**x**</font>,
    <font color="green">**y**</font>,
    <font color="blue">**z**</font>
    (or their world names if `world_axes` has been used).

| Examples                      |                                                                                                                                                      |
| ----------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| `space radio`                 | Orients the cross section such that the <font color="red">**x**</font> axis points to the **left** of the first quadrant (radiological convention).  |
| `space neuro`                 | Orients the cross section such that the <font color="red">**x**</font> axis points to the **right** of the first quadrant (neurological convention). |
| `space --layer <LAYER>`       | Aligns the cross-section with the voxels of the designated layer, while keeping the existing radio or neuro convention.                              |
| `space --layer world`         | Aligns the cross-section with the canonical model space, while keeping the existing radio or neuro convention.                                       |
| `space radio --layer <LAYER>` | Aligns the cross-section with the voxels of the designated layer, and uses the radiological convention                                               |

!!! note "See also"
    [`display`](#display), [`world_axes`](#world_axes)

#### Positional arguments

| Name  | Type  | Description     |
| ----- | ----- | --------------- |
| `space` | `{radio,neuro,default}` | Space. |

#### Options

| Flag            | Type  | Description                                                                                                                       | Default |
| --------------- | ----- | --------------------------------------------------------------------------------------------------------------------------------- | ------- |
| `-l`, `--layer` | `str` | If the name of a layer, align the cross-section with its voxel grid. If "world", align the cross section with the canonical axes. |         |

### transform

=== ":octicons-terminal-24:"
    ```shell
    transform <transform *> [--layer +] [--inv] [--reset] [--mov] [--fix]
    ```

Apply a transform.

#### Positional arguments

| Name        | Type                       | Description                                                            |
| ----------- | -------------------------- | ---------------------------------------------------------------------- |
| `transform` | `uri` &vert; `list[float]` | Path to transform file or flattened transformation matrix (row major). |

#### Options

| Flag            | Type        | Description                                             | Default |
| --------------- | ----------- | ------------------------------------------------------- | ------- |
| `-l`, `--layer` | `list[str]` | Name(s) of layer(s) to transform                        | All layers     |
| `-i`, `--inv`   |             | Invert the transform before applying it                 |         |
| `-r`, `--reset` |             | Reset the layer's transform before applying the new one |         |
| `-m`, `--mov`   | `uri`       | Moving image (required by some formats)                 |         |
| `-f`, `--fix`   | `uri`       | Fixed image (required by some formats)                  |         |

### save_transform


=== ":octicons-terminal-24:"
    ```shell
    save_transform <output *> [--layer +] [--format 1]
    ```

Save the current transform

#### Positional arguments

| Name     | Type        | Description                  |
| -------- | ----------- | ---------------------------- |
| `output` | `list[uri]` | Path to output transform(s). |

#### Options

| Flag             | Type                 | Description                  | Default |
| ---------------- | -------------------- | ---------------------------- | ------- |
| `-l`, `--layer`  | `list[str]`          | Name(s) of layer(s) to save. |         |
| `-f`, `--format` | `{afni,fsl,itk,lta}` | Format to save into.         |         |

### channel_mode

=== ":octicons-terminal-24:"
    ```shell
    channel_mode <{local,channel,global}> [--layer +] [--dimension +]
    ```

|         | In neuroglancer, axes can be interpreted in three different ways                                             |
| ------- | ------------------------------------------------------------------------------------------------------------ |
| global  | The axis is common to all layers and can be navigated <br />(default for axes that have spatial or temporal units) |
| local   | The axis is specific to a layers and navigation is local <br />(dimension names end with `'`)                      |
| channel | The axis is specific to a layer; enables multi-channel shading <br />(dimension names end with `^`)                |

!!! note
    * When an axis is local (`'`), only one channel can be shown at once.

    * When an axis is a channel (`^`), all channels can be mixed into a single
      view using a multi-channel shader.

    * When an axis is global, navigation is linked across layers.

#### Positional arguments

| Name     | Type        | Description                  |
| -------- | ----------- | ---------------------------- |
| `mode` | `{local,channel,global}` | How to interpret the channel (or another) axis. |

#### Options

| Flag                | Type                 | Description                                          | Default |
| ------------------- | -------------------- | ---------------------------------------------------- | ------- |
| `-l`, `--layer`     | `list[str]`          | Name(s) of layer(s) to which the command is applied. |         |
| `-d`, `--dimension` | `str` | Name(s) of axes to transform.                        |         |


### shader

=== ":octicons-terminal-24:"
    ```shell
    shader <shader> [--layer +] [--layer-type +]
    ```

Applies a colormap, or a more advanced shading function to all or some of
the layers.

The input can also be the path to a (local or remote) freesurfer LUT file.

#### Positional arguments

| Name     | Type  | Description                      |
| -------- | ----- | -------------------------------- |
| `shader` | `str` | Shader name or GLSL shader code. |

#### Options

| Flag            | Type                                  | Description                                         | Default |
| --------------- | ------------------------------------- | --------------------------------------------------- | ------- |
| `-l`, `--layer` | `list[str]`                           | Name(s) of layer(s) to which the shader is applied. |         |
| `-t`, `--type`  | `{image,segmentation,annotation,...}` | Layer type(s) to which the shader is applied.       |         |

### display

=== ":octicons-terminal-24:"
    ```shell
    display <dimension *>
    ```

Neuroglancer is quite flexible in the sense that it does not have a
predefined "hard" coordinate frame in which the data is shown. Instead,
the
<font color="red">**red**</font>,
<font color="green">**green**</font>,
<font color="blue">**blue**</font>
"visual" axes can be
arbitrarily mapped to any of the existing "model" axes.

By default, most formats use **(x, y, z)** as their
model axes, although they may not have predefined anatomical meaning.
NIfTI files do have an anatomical convention, under which axes
<font color="red">**x**</font>,
<font color="green">**y**</font>,
<font color="blue">**z**</font>
map to the
<font color="red">**right**</font>,
<font color="green">**anterior**</font>,
<font color="blue">**superior**</font>
sides of the brain.

In order to show data in a frame that is standard in the neuroimaging
field, we therefore map the visual axes
<font color="red">**red**</font>,
<font color="green">**green**</font>,
<font color="blue">**blue**</font>
to the model axes
<font color="red">**x**</font>,
<font color="green">**y**</font>,
<font color="blue">**z**</font>
when loading data in an empty scene.
This mapping is also enforced every time the command [`space`](#space) is used.

The [`display`](#display) command can be used to assign a different set of
model axes to the visual axes.

!!! note "See also"
    [`space`](#space), [`world_axes`](#world_axes)

#### Positional arguments

| Name        | Type  | Description            |
| ----------- | ----- | ---------------------- |
| `dimension` | `str` | Dimensions to display. |


### layout

=== ":octicons-terminal-24:"
    ```shell
    layout [{xy,yz,xz,3d,xy-3d,yz-3d,xz-3d,4panel}]                       \
           [--stack {row,column}] [--layer +] [--flex] [--row] [--column] \
           [--append *] [--assign *] [--insert *] [--remove *]
    ```

Change the viewer's layout (i.e., the quadrants and their layers)

|        | Neuroglancer has 8 different window types |
| ------ | ----------------------------------------- |
| `xy`     | <font color="red">**X**</font><font color="green">**Y**</font> cross-section                          |
| `yz`     | <font color="green">**Y**</font><font color="blue">**Z**</font> cross-section                          |
| `xz`     | <font color="red">**X**</font><font color="blue">**Z**</font> cross-section                          |
| `xy-3d`  | <font color="red">**X**</font><font color="green">**Y**</font> cross-section in a **3D** window           |
| `yz-3d`  | <font color="green">**Y**</font><font color="blue">**Z**</font> cross-section in a **3D** window           |
| `xz-3d`  | <font color="red">**X**</font><font color="blue">**Z**</font> cross-section in a **3D** window           |
| `4panel` | Four quadrants (<font color="red">**X**</font><font color="green">**Y**</font>, <font color="red">**X**</font><font color="blue">**Z**</font>, **3D**, <font color="green">**Y**</font><font color="blue">**Z**</font>)           |
| `3d`     | **3D** window                                 |

It is possible to build a user-defined layout by stacking these basic
windows into a row or a column &mdash; or even nested rows and columns &mdash;
using the `--stack` option. The `--layer` option allows assigning
specific layers to a specific window. We also define `--append` and
`--insert` to add a new window into an existing stack of windows.

#### Positional arguments

| Name     | Type                                     | Description |
| -------- | ---------------------------------------- | ----------- |
| `layout` | `{xy,yz,xz,3d,xy-3d,yz-3d,xz-3d,4panel}` | Layout.     |

#### Options

| Flag                | Type                  | Description                             | Default |
| ------------------- | --------------------- | --------------------------------------- | ------- |
| `-s`, `--stack`     | `{row,column}`        | Stack direction                         |         |
| `-l`, `--layer`     | `list[str]`           | Layer(s) to include                     |         |
| `-f`, `--flex`      | `float`               | Flex                                    | `1`     |
| `-a`, `--append`    | `optional[list[int]]` | Append to existing (nested) layout      |         |
| `-x`, `--assign`    | `optional[list[int]]` | Assign into existing (nested) layout    |         |
| `-i`, `--insert`    | `optional[list[int]]` | Insert in existing (nested) layout      |         |
| `-r`, `--remove`    | `optional[list[int]]` | Remove from an existing (nested) layout |         |
| `--row`             |                       | Alias for `--stack row`                 |         |
| `--col`, `--column` |                       | Alias for `--stack column`              |         |

### state

=== ":octicons-terminal-24:"
    ```shell
    state [--no-print] [--save 1] [--load 1] [--url] [--open] [--instance {ng,linc}]
    ```

Return the viewer's state.

#### Options

| Flag               | Type               | Description                                                          | Default |
| ------------------ | ------------------ | -------------------------------------------------------------------- | ------- |
| `--no-print`       |                    | Do not print the state.                                              |         |
| `-s`, `--save`     | `uri`              | Save JSON state to this file.                                        |         |
| `-l`, `--load`     | `uri` &vert; `str` | Load JSON state from this file. Can also be a JSON string or a URL.. |         |
| `-u`, `--url`      |                    | Load (or print) the url form of the state.                           |         |
| `-o`, `--open`     |                    | Open the url (if `--url`) or viewer (otherwise)                      |         |
| `-i`, `--instance` | `{ng,linc}`        | Link to this neuroglancer instance                                   |         |

### move

=== ":octicons-terminal-24:"
    ```shell
    move <coord *> [--dimensions +] [--unit 1] [--absolute] [--reset]
    ```

Move cursor

#### Positional arguments

| Name    | Type          | Description                                     |
| ------- | ------------- | ----------------------------------------------- |
| `coord` | `list[float]` | Cursor coordinates. If None, print current one. |

#### Options

| Flag                        | Type        | Description                                                | Default |
| --------------------------- | ----------- | ---------------------------------------------------------- | ------- |
| `-d`, `--dimensions`        | `list[str]` | Axis name for each coordinate (can be compact)             |         |
| `-u`, `--unit`              | `str`       | Coordinates are expressed in this unit                     |         |
| `-a`, `--abs`, `--absolute` |             | Move to absolute position, rather than relative to current |         |
| `-r`, `--reset`             |             | Reset coordinates to zero                                  |         |

### zoom

=== ":octicons-terminal-24:"
    ```shell
    zoom [factor] [--reset]
    ```

Zoom by a factor [default: x2]

#### Positional arguments

| Name     | Type    | Description  | Default |
| -------- | ------- | ------------ | ------- |
| `factor` | `float` | Zoom factor. | `2`     |

#### Options

| Flag            | Type | Description                  | Default |
| --------------- | ---- | ---------------------------- | ------- |
| `-r`, `--reset` |      | Reset zoom level to default. |         |

### unzoom

=== ":octicons-terminal-24:"
    ```shell
    unzoom [factor] [--reset]
    ```

Unzoom by a factor [default: ÷2]

#### Positional arguments

| Name     | Type    | Description          | Default |
| -------- | ------- | -------------------- | ------- |
| `factor` | `float` | Inverse zoom factor. | `2`     |

#### Options

| Flag            | Type | Description                  | Default |
| --------------- | ---- | ---------------------------- | ------- |
| `-r`, `--reset` |      | Reset zoom level to default. |         |

### zorder

=== ":octicons-terminal-24:"
    ```shell
    zorder <layer *> [--up] [--down]
    ```

Modifies the z-order of the layers.

In neuroglancer, layers are listed in the order they are loaded, with the
latest layer appearing on top of the other ones in the scene.

Counter-intuitively, the latest/topmost layer is listed at the bottom of
the layer list, while the earliest/bottommost layer is listed at the top of
the layer list. In this command, layers should be listed in their expected
z-order, top to bottom.

There are two ways of using this command:

1. Provide the new order (top-to-bottom) of the layers

2. Provide a positive (`--up`) or negative (`--down`)
   number of steps by which to move the listed layers.
   In this case, more than one up or down step can be provided, using
   repeats of the option.

!!! example
      * `-vvvv` moves downward 4 times
      * `-^^^^` moves upwards 4 times

#### Positional arguments

| Name    | Type        | Description    |
| ------- | ----------- | -------------- |
| `layer` | `list[str]` | Layer name(s). |

#### Options

| Flag                 | Type | Description     | Default |
| -------------------- | ---- | --------------- | ------- |
| `-^`, `-u`, `--up`   |      | Move upwards.   |         |
| `-v`, `-d`, `--down` |      | Move downwards. |         |

### cd

=== ":octicons-terminal-24:"
    ```shell
    cd <path>
    ```

Change directory.

### ls

=== ":octicons-terminal-24:"
    ```shell
    ls <path> [-l,--long] [-h,--hidden]
    ```

List files.

### pwd

=== ":octicons-terminal-24:"
    ```shell
    pwd
    ```

Path to working directory.

### stdin

=== ":octicons-terminal-24:"
    ```shell
    stdin <file>
    ```

Set input stream.


### stdout

=== ":octicons-terminal-24:"
    ```shell
    stdout <file>
    ```

Set output stream.

### stderr

=== ":octicons-terminal-24:"
    ```shell
    stderr <file>
    ```

Set error stream.

### exit (quit)

=== ":octicons-terminal-24:"
    ```shell
    exit
    ```
=== ":octicons-terminal-24:"
    ```shell
    quit
    ```

Exit `nglocal`.
