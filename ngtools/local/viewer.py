"""Local neuroglancer instance with a shell-like interface."""
# stdlib
import argparse
import os.path
import textwrap

# externals
import neuroglancer as ng
import numpy as np
from neuroglancer.server import global_server_args

# internals
from ngtools.local.console import Console, _fixhelpformatter
from ngtools.local.fileserver import LocalFileServer, find_available_port
from ngtools.local.termcolors import bformat
from ngtools.scene import Scene
from ngtools.shaders import pretty_colormap_list

SourceType = ng.LayerDataSource | ng.LocalVolume | ng.skeleton.SkeletonSource


def action(needstate: bool = False) -> callable:
    """
    Decorate a neuroglancer action.

    These actions can thereby be triggered by argparse.
    """
    if callable(needstate):
        return action()(needstate)

    def decorator(func: callable) -> callable:

        def wrapper(
            self: "LocalNeuroglancer", *args: tuple, **kwargs: dict
        ) -> callable:
            args = list(args)
            if args and isinstance(args[0], argparse.Namespace):
                if len(args) > 1:
                    raise ValueError(
                        'Only one positional argument accepted when an '
                        'action is applied to an argparse object')
                parsedargs = vars(args.pop(0))
                parsedargs.update(kwargs)
                parsedargs.pop('func', None)
                kwargs = parsedargs

            if needstate and not kwargs.get('state', None):
                with self.viewer.txn(overwrite=True) as state:
                    kwargs['state'] = state
                    result = func(self, *args, **kwargs)
                return result
            else:
                result = func(self, *args, **kwargs)
                return result

        return wrapper

    return decorator


def state_action(name: str) -> callable:
    """Generate a state action that wraps a scene action."""

    @action(needstate=True)
    def func(
        self: "LocalNeuroglancer", *args, state: ng.ViewerState, **kwargs
    ) -> object | None:
        scene = Scene(state.to_json())
        out = getattr(scene, name)(*args, **kwargs)
        for key in scene.to_json().keys():
            setattr(state, key, getattr(scene, key))
        return out

    return func


def ensure_list(x: object) -> list:
    """Ensure that an object is a list. Make one if needed."""
    if isinstance(x, np.ndarray):
        x = x.tolist()
    if not isinstance(x, (list, tuple)):
        x = [x]
    return list(x)


class LocalNeuroglancer:
    """
    A local instance of neuroglancer that can launch its own local fileserver.

    It also comes with a shell-like interface that allows loading,
    unloading, applying transforms, etc.
    """

    def __init__(
        self,
        port: int = 0,
        ip: str = '',
        token: int = 1,
        fileserver: bool | int | LocalFileServer = True,
        debug: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        port : int
            Port to use.
        ip : str
            IP to use.
        token : str
            Unique id for the instance.
        filserver : LocalFileServerInBackground or bool
            A local file server.

            * If `True`, create a local file server.
            * If `False`, do not create one -- only remote files can be loaded.
        debug : bool
            Print full trace when an error is encountered.
        """
        if (
            fileserver is not False and
            not isinstance(fileserver, LocalFileServer)
        ):
            portf = fileserver if isinstance(fileserver, int) else 0
            fileserver = LocalFileServer(ip=ip, port=portf, interrupt=EOFError)
        self.fileserver = fileserver
        if self.fileserver:
            self.fileserver.start()

        port, ip = find_available_port(port, ip)
        global_server_args['bind_address'] = str(ip)
        global_server_args['bind_port'] = str(port)
        self.viewer = ng.Viewer(token=str(token))
        # self.viewer.shared_state.add_changed_callback(self.on_state_change)
        self.console = self._make_console(debug)

    # ==================================================================
    #
    #                   COMMANDLINE APPLICATION
    #
    # ==================================================================

    def await_input(self) -> None:
        """Launch shell-like interface."""
        # return self.console.await_input()
        # return self.console.await_input()
        try:
            return self.console.await_input()
        except SystemExit:
            # exit gracefully (cleanup the fileserver process, etc)
            if self.fileserver:
                self.fileserver.stop()
            raise

    def _make_console(self, debug: bool = False) -> Console:
        mainparser = Console('', debug=debug)
        parsers = mainparser.add_subparsers()
        formatter = _fixhelpformatter(argparse.RawDescriptionHelpFormatter)
        F = dict(formatter_class=formatter)

        def add_parser(cmd, *args, **kwargs):  # noqa: ANN001, ANN202
            # We define long descriptions in the _clihelp class at the
            # end of this file.
            kwargs.setdefault('description', getattr(_clihelp, cmd, None))
            return parsers.add_parser(cmd, *args, **kwargs, **F)

        # --------------------------------------------------------------
        #   HELP
        # --------------------------------------------------------------
        _ = add_parser('help', help='Display help')
        _.set_defaults(func=self.help)
        _.add_argument(
            dest='action', nargs='?', help='Command for which to display help')

        # --------------------------------------------------------------
        #   LOAD
        # --------------------------------------------------------------
        _ = add_parser('load', help='Load a file')
        _.set_defaults(func=self.load)
        _.add_argument(
            dest='filename', nargs='+', metavar='FILENAME',
            help='Filename(s) with protocols')
        _.add_argument(
            '--name', nargs='+', help='A name for the image layer')
        _.add_argument(
            '--transform', nargs='+', help='Apply a transform')

        # --------------------------------------------------------------
        #   UNLOAD
        # --------------------------------------------------------------
        _ = add_parser('unload', help='Unload a file')
        _.set_defaults(func=self.unload)
        _.add_argument(
            dest='layer', nargs='+', metavar='LAYER',
            help='Name(s) of layer(s) to unload')

        # --------------------------------------------------------------
        #   RENAME
        # --------------------------------------------------------------
        _ = add_parser('rename', help='Rename a file')
        _.set_defaults(func=self.rename)
        _.add_argument(
            dest='src', metavar='SOURCE', help='Current layer name')
        _.add_argument(
            dest='dst', metavar='DEST', help='New layer name')

        # --------------------------------------------------------------
        #   WORLD AXES
        # --------------------------------------------------------------
        _ = add_parser('world_axes', help='Rename native axes')
        _.set_defaults(func=self.world_axes)
        _.add_argument(
            dest='axes', metavar='DEST', nargs="*", help='New axis names')
        _.add_argument(
            "--dst", metavar='DEST', nargs="+", help='New axis names')
        _.add_argument(
            '--src', metavar='SOURCE', nargs="*", help='Native axis names')

        # --------------------------------------------------------------
        #   RENAME AXES
        # --------------------------------------------------------------
        _ = add_parser('rename_axes', help='Rename axes')
        _.set_defaults(func=self.rename_axes)
        _.add_argument(
            dest='axes', metavar='DEST', nargs="+", help='New axis names')
        _.add_argument(
            "--dst", metavar='DEST', nargs="+", help='New axis names')
        _.add_argument(
            '--src', metavar='SOURCE', nargs="*", help='Old axis names')
        _.add_argument(
            '--layer', nargs="*", help='Layer(s) to rename')

        # --------------------------------------------------------------
        #   SPACE
        # --------------------------------------------------------------
        MODES = ("radio", "neuro")
        _ = add_parser('space', help='Cross-section orientation')
        _.set_defaults(func=self.space)
        _.add_argument(
            dest='mode', nargs="?", help='New axis names', choices=MODES)
        _.add_argument(
            '--layer', nargs="*",
            help='If the name of a layer, align the cross-section with its '
                 'voxel grid. If "world", align the cross section with the '
                 'canonical axes.')

        # --------------------------------------------------------------
        #   TRANSFORM
        # --------------------------------------------------------------
        _ = add_parser('transform', help='Apply a transform')
        _.set_defaults(func=self.transform)
        _.add_argument(
            dest='transform', nargs='+', metavar='TRANSFORM',
            help='Path to transform file or flattened transformation '
                 'matrix (row major)')
        _.add_argument(
            '--layer', nargs='+', help='Name(s) of layer(s) to transform')
        _.add_argument(
            '--inv', action='store_true', default=False,
            help='Invert the transform before applying it')
        _.add_argument(
            '--mov', help='Moving image (required by some formats)')
        _.add_argument(
            '--fix', help='Fixed image (required by some formats)')

        # --------------------------------------------------------------
        #   AXIS MODE
        # --------------------------------------------------------------
        _ = add_parser(
            'channel_mode', help='Change the way a dimension is interpreted')
        _.set_defaults(func=self.channel_mode)
        _.add_argument(
            dest='mode', metavar='MODE',
            choices=('local', 'channel', 'global'),
            help='How to interpret the channel (or another) axis')
        _.add_argument(
            '--layer', nargs='+', default=None,
            help='Name(s) of layer(s) to transform')
        _.add_argument(
            '--dimension', nargs='+', default=['c'],
            help='Name(s) of axes to transform')

        # --------------------------------------------------------------
        #   SHADER
        # --------------------------------------------------------------
        _ = add_parser('shader', help='Apply a shader')
        _.set_defaults(func=self.shader)
        _.add_argument(
            dest='shader', metavar='SHADER',
            help='Shader name or GLSL shader code')
        _.add_argument(
            '--layer', nargs='+', help='Layer(s) to apply shader to')

        # --------------------------------------------------------------
        #   DISPLAY
        # --------------------------------------------------------------
        _ = add_parser('display', help='Dimensions to display')
        _.set_defaults(func=self.display)
        _.add_argument(
            dest='dimensions', nargs='*', metavar='DIMENSIONS',
            help='Dimensions to display')

        # --------------------------------------------------------------
        #   LAYOUT
        # --------------------------------------------------------------
        LAYOUTS = ["xy", "yz", "xz", "xy-3d", "yz-3d", "xz-3d", "4panel", "3d"]
        _ = add_parser('layout', help='Layout')
        _.set_defaults(func=self.layout)
        _.add_argument(
            dest='layout', nargs='*', choices=LAYOUTS, metavar='LAYOUT',
            help='Layout')
        _.add_argument(
            '--stack', choices=("row", "column"), help="Stack direction")
        _.add_argument(
            '--layer', nargs='*', help="Layer(s) to include")
        _.add_argument(
            '--flex', type=float, default=1, help="Flex")
        _.add_argument(
            '--append', type=int, nargs='*',
            help="Append to existing (nested) layout")
        _.add_argument(
            '--insert', type=int, nargs='+',
            help="Insert in existing (nested) layout")
        _.add_argument(
            '--remove', type=int, nargs='+',
            help="Remove from an existing (nested) layout")

        # --------------------------------------------------------------
        #   STATE
        # --------------------------------------------------------------
        _ = add_parser('state', help='Return the viewer\'s state')
        _.set_defaults(func=self.state)
        _.add_argument(
            '--no-print', action='store_false', default=True, dest='print',
            help='Do not print the state.')
        _.add_argument(
            '--save', help='Save JSON state to this file.')
        _.add_argument(
            '--load', help='Load JSON state from this file. '
                           'Can also be a JSON string or a URL.')
        _.add_argument(
            '--url', action='store_true', default=False,
            help='Load (or print) the url form of the state')

        # --------------------------------------------------------------
        #   POSITION
        # --------------------------------------------------------------
        # FIXME: comment out until fixed
        # def _add_common_args(parser):
        #     parser.add_argument(
        #         '--dimensions', '-d', nargs='+', default=None,
        #         help='Axis name for each coordinate (can be compact)')
        #     parser.add_argument(
        #         '--world', '-w', action='store_true', default=False,
        #         help='Coordinates are expressed in the world frame')
        #     parser.add_argument(
        #         '--layer', '-l', nargs='*', default=None,
        #         help='Coordinates are expressed in this frame')
        #     parser.add_argument(
        #         '--unit', '-u',
        #         help='Coordinates are expressed in this unit')
        #     parser.add_argument(
        #         '--reset', action='store_true', default=False,
        #         help='Reset coordinates to zero')

        # position = add_parser('position', help='Move cursor')
        # position.set_defaults(func=self.position)
        # position.add_argument(
        #     dest='coord', nargs='*', metavar='COORD', type=float,
        #     help='Cursor coordinates. If None, print current one.')
        # _add_common_args(position)

        # --------------------------------------------------------------
        #   ZORDER
        # --------------------------------------------------------------
        _ = add_parser('zorder', help='Reorder layers')
        _.set_defaults(func=self.zorder)
        _.add_argument(
            dest='layer', nargs='+', metavar='LAYER',
            help='Layer(s) name(s)')
        _.add_argument(
            '--up', '-u', '-^', action='count', default=0,
            help='Move upwards')
        _.add_argument(
            '--down', '-d', '-v', action='count', default=0,
            help='Move downwards')

        # --------------------------------------------------------------
        #   NAVIGATION
        # --------------------------------------------------------------
        _ = add_parser('cd', help='Change directory')
        _.set_defaults(func=self.cd)
        _.add_argument(dest='path', metavar='PATH')

        _ = add_parser('ls', help='List files')
        _.set_defaults(func=self.ls)
        _.add_argument(dest='path', nargs='?', default='.', metavar='PATH')

        _ = add_parser('pwd', help='Path to working directory')
        _.set_defaults(func=self.pwd)

        # --------------------------------------------------------------
        #   EXIT
        # --------------------------------------------------------------
        _ = add_parser('exit', aliases=['quit'], help='Exit neuroglancer')
        _.set_defaults(func=self.exit)
        return mainparser

    # ==================================================================
    #
    #                              ACTIONS
    #
    # ==================================================================

    load = state_action("load")
    unload = state_action("unload")
    rename = state_action("rename")
    world_axes = state_action("world_axes")
    rename_axes = state_action("rename_axes")
    space = state_action("space")
    display = state_action("display")
    redisplay = state_action("redisplay")
    transform = state_action("transform")
    channel_mode = state_action("channel_mode")
    position = state_action("position")
    shader = state_action("shader")
    layout = state_action("layout")
    zorder = state_action("zorder")
    state = state_action("state")

    @action
    def cd(self, path: str) -> str:
        """Change directory."""
        os.chdir(os.path.expanduser(path))
        return os.getcwd()

    @action
    def ls(self, path: str) -> list[str]:
        """List files."""
        files = os.listdir(os.path.expanduser(path))
        self.console.print(*files)
        return files

    @action
    def pwd(self) -> str:
        """Path to working directory."""
        self.console.print(os.getcwd())
        return os.getcwd()

    @action
    def help(self, action: str | None = None) -> None:
        """
        Display help.

        Parameters
        ----------
        action : str
            Action for which to display help
        """
        if action:
            self.console.parse_args([action, '--help'])
        else:
            self.console.parse_args(['--help'])

    @action
    def exit(self) -> None:
        """Exit gracefully."""
        raise SystemExit

    # ==================================================================
    #
    #                            CALLBACKS
    #
    # ==================================================================

    # def on_state_change(self) -> None:
    #     old_state = self.saved_state
    #     with self.viewer.txn() as state:
    #         self.saved_state = state
    #         for layer in state.layers:
    #             name = layer.name
    #             layer = layer.layer
    #             if isinstance(layer.source, ng.SkeletonSource):
    #                 if (layer.shaderControls.nbtracts
    #                         != old_state.shaderControls.nbtracts):
    #                     self.update_tracts_nb(name)

    # def update_tracts_nb(self, name: str) -> None:
    #     with self.viewer.txn() as state:
    #         layer = state.layers[name]
    #         print(type(layer))
    #         layer.source.max_tracts = layer.shaderControls.nbtracts
    #         layer.source._filter()
    #         layer.invalidate()


_clihelp = type('_clihelp', (object,), {})

bold = bformat.bold
red = bformat.fg.red
green = bformat.fg.green
blue = bformat.fg.blue

_clihelp.load = textwrap.dedent(
f"""
Load a file, which can be local or remote.

{bold("Paths and URLs")}
{bold("--------------")}
Each path or url may be prepended by:

1)  A layer type protocol, which indicates the kind of object that the file
    contains.
    Examples: {bold("volume://")}, {bold("labels://")}, {bold("tracts://")}.

2)  A format protocol, which indicates the exact file format.
    Examples: {bold("nifti://")}, {bold("zarr://")}, {bold("mgh://")}.

3)  An access protocol, which indices the protocol used to  access the files.
    Examples: {bold("https://")}, {bold("s3://")}, {bold("dandi://")}.

All of these protocols are optional. If absent, a guess is made using the
file extension.

{bold("Examples")}
{bold("--------")}

- Absolute path to local file:  {bold("/absolute/path/to/mri.nii.gz")}
- Relative path to local file:  {bold("relative/path/to/mri.nii.gz")}
- Local file with format hint:  {bold("mgh://relative/path/to/linkwithoutextension")}
- Remote file:                  {bold("https://url.to/mri.nii.gz")}
- Remote file with format hint: {bold("zarr://https://url.to/filewithoutextension")}
- File on dandiarchive:         {bold("dandi://dandi/<dandiset>/sub-<id>/path/to/file.ome.zarr")}

{bold("Layer names")}
{bold("-----------")}
Neuroglancer layers are named. The name of the layer can be specified with
the {bold("--name")} option. Otherwise, the base name of the file is used (that
is, without the folder hierarchy).

If multiple files are loaded _and_ the --name option is used, then there
should be as many names as files.

{bold("Transforms")}
{bold("----------")}
A spatial transform (common to all files) can be applied to the loaded
volume. The transform is specified with the {bold("--transform")} option, which
can be a flattened affine matrix (row major) or the path to a transform file.
Type {bold("help transform")} for more information.

{bold("Arguments")}
{bold("----------")}"""  # noqa: E122, E501
)

_clihelp.unload = "Unload layers"

_clihelp.rename = "Rename a layer"

_clihelp.shader = textwrap.dedent(
f"""
Applies a colormap, or a more advanced shading function to all or some of
the layers.

{bold("List of builtin colormaps")}
{bold("-------------------------")}
"""  # noqa: E122
) + textwrap.indent(pretty_colormap_list(), ' ')

_clihelp.zorder = textwrap.dedent(
f"""
Modifies the z-order of the layers.

In neuroglancer, layers are listed in the order they are loaded, with the
latest layer appearing on top of the other ones in the scene.

Counter-intuitively, the latest/topmost layer is listed at the bottom of
the layer list, while the earliest/bottommost layer is listed at the top of
the layer list. In this command, layers should be listed in their expected
z-order, top to bottom.

There are two ways of using this command:

1)  Provide the new order (top-to-bottom) of the layers

2)  Provide a positive ({bold("--up")}) or negative ({bold("--down")})
    number of steps by which to move the listed layers.
    In this case, more than one up or down step can be provided, using
    repeats of the option.
    Examples: {bold("-vvvv")} moves downward 4 times
              {bold("-^^^^")} moves upwards 4 times
"""  # noqa: E122
)

_clihelp.display = textwrap.dedent(
f"""
Display the data in a different space (world or canonical), or in a
different orientation (RAS, LPI, and permutations thereof)

By default, neuroglancer displays data in their "native" space, where
native means a "XYZ" coordinate frame, whose mapping from and to voxels
is format-spacific. In {bold("nifti://")} volumes, XYZ always corresponds to
the RAS+ world space, whereas in {bold("zarr://")}, XYZ correspond to canonical
axes ordered according to the OME metadata.

Because of our neuroimaging bias, we choose to interpret XYZ as RAS+ (which
is consistant with the nifti behavior).

The simplest usage of this command is therefore to switch between
different representations of the world coordinate system:
    . {bold("display RAS")}  (alias: {bold(" display right anterior superior")})
      maps the {red("red")}, {green("green")} and {bold("blue")} axes to {red("right")}, {green("anterior")}, {bold("superior")}.
    . {bold("display LPI")}  (alias: {bold(" display left posterior inferior")})
      maps the {red("red")}, {green("green")} and {bold("blue")} axes to {red("left")}, {green("posterior")}, {bold("inferior")}.

A second usage allows switching between displaying the data in the world
frame and displaying the data in the canonical frame of one of the layers
(that is, the frame axes are aligned with the voxel dimensions).
    . {bold("display --layer <name>")} maps the {red("red")}/{green("green")}/{bold("blue")}
      axes to the canonical axes of the layer <name>
    . {bold("display --world")} maps (back) the {red("red")}/{green("green")}/{bold("blue")}
      axes to the axes of world frame.

Of course, both usages can be combined:
    . {bold("display RAS --layer <name>")}
    . {bold("display LPI --world")}

"""  # noqa: E122, E501
)

_clihelp.layout = textwrap.dedent(
f"""
Change the viewer's layout (i.e., the quadrants and their layers)

Neuroglancer has 8 different window types:
. xy     : {red("X")}{green("Y")} cross-section
. yz     : {green("Y")}{bold("Z")} cross-section
. xz     : {red("X")}{bold("Z")} cross-section
. xy-3d  : {red("X")}{green("Y")} cross-section in a {bold("3D")} window
. yz-3d  : {green("Y")}{bold("Z")} cross-section in a {bold("3D")} window
. xz-3d  : {red("X")}{bold("Z")} cross-section in a {bold("3D")} window
. 4panel : Four quadrants ({red("X")}{green("Y")}, {red("X")}{bold("Z")}, {bold("3D")}, {green("Y")}{bold("Z")})
. 3d     : {bold("3D")} window

It is possible to build a user-defined layout by stacking these basic
windows into a row or a column -- or even nested rows and columns --
using the {bold("--stack")} option. The {bold("--layer")} option allows assigning
specific layers to a specific window. We also define {bold("--append")} and
{bold("--insert")} to add a new window into an existing stack of windows.

"""  # noqa: E122, E501
)
