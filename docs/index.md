---
icon: octicons/rocket-24
---

# Getting started

```text
 _ __   __ _| |_ ___   ___ | |___
| '_ \ / _` | __/ _ \ / _ \| / __|
| | | | (_| | || (_) | (_) | \__ \
|_| |_|\__, |\__\___/ \___/|_|___/
       |___/
```

## Installation

```shell
pip install ngtools
```

## Description

`ngtools` contains a set of user-friendly utilities to accompany
[`neuroglancer`](https://github.com/google/neuroglancer) -- an
in-browser viewer for peta-scale volumetric data. Specifically, it
implements:

- a **local app** (`from ngtools.local.viewer import LocalNeuroglancer`)
  that runs a local `neuroglancer` instance, allows local files to be
  visualized, and implements additional file formats
  (`.trk`, `.tck`, `.tiff`, `'.mgh`).

  See: [**Local neuroglancer in python**](/start/local_python/)

- a **shell console** (`nglocal --help`) for the local app with thorough
  documentation of each command, auto-completion and history.

  See: [**Local neuroglancer in the shell**](/start/local_shell/)

- a **user-friendly python API** (`from ngtools.scene import Scene`) that
  simplifies the creation of neuroglancer scenes (and is used under
  the hood by `LocalNeuroglancer`).

  See: [**Scene building without running an instance**](/start/offline_scenes/)

- **smart wrappers** (`ngtools.layers`, `ngtools.datasources`) around
  the neuroglancer python API, that can compute quantities that can
  normally only be accessed in the neuroglancer frontend (default
  transforms, voxel data, etc).

- **utilities** (`ngtools.shaders`, `ngtools.transforms`, `ngtools.spaces`)
  that greatly simplifies manipulating some of neuroglancer's most
  intricate features.
