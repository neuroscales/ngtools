"""Local neuroglancer instance with a shell-like interface."""
# stdlib
import argparse
import os.path
import socket
import sys
import textwrap

# externals
import neuroglancer as ng
import numpy as np
from neuroglancer.server import global_server_args

# internals
from ngtools.local.fileserver import LocalFileServerInBackground
from ngtools.local.parserapp import ParserApp, _fixhelpformatter
from ngtools.local.termcolors import bcolors
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
            self: LocalNeuroglancer, *args: tuple, **kwargs: dict
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
        self: LocalNeuroglancer, *args, state: ng.ViewerState, **kwargs
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
        fileserver: bool | LocalFileServerInBackground = True,
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
        if fileserver is True:
            fileserver = LocalFileServerInBackground(interrupt=EOFError)
        self.fileserver = fileserver
        if self.fileserver:
            self.fileserver.start_and_serve_forever()

        try:
            s = socket.socket()
            s.bind((ip, port))
            ip = s.getsockname()[0]
            port = s.getsockname()[1]
            s.close()
        except OSError:
            port0 = port
            s = socket.socket()
            s.bind((ip, 0))
            ip = s.getsockname()[0]
            port = s.getsockname()[1]
            s.close()
            print(f'Port {port0} already in use. Use port {port} instead.',
                  file=sys.stderr)

        global_server_args['bind_address'] = str(ip)
        global_server_args['bind_port'] = str(port)
        self.viewer = ng.Viewer(token=str(token))
        # self.viewer.shared_state.add_changed_callback(self.on_state_change)
        self.parser = self._make_parser(debug)

    # ==================================================================
    #
    #                   COMMANDLINE APPLICATION
    #
    # ==================================================================

    def await_input(self) -> None:
        """Launch shell-like interface."""
        try:
            return self.parser.await_input()
        except SystemExit:
            # exit gracefully (cleanup the fileserver process, etc)
            self.exit()

    def _make_parser(self, debug: bool = False) -> ParserApp:
        mainparser = ParserApp('', debug=debug)
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
        help = add_parser('help', help='Display help')
        help.set_defaults(func=self.help)
        help.add_argument(
            dest='action', nargs='?', help='Command for which to display help')

        # --------------------------------------------------------------
        #   LOAD
        # --------------------------------------------------------------
        load = add_parser('load', help='Load a file')
        load.set_defaults(func=self.load)
        load.add_argument(
            dest='filename', nargs='+', metavar='FILENAME',
            help='Filename(s) with protocols')
        load.add_argument(
            '--name', nargs='+', help='A name for the image layer')
        load.add_argument(
            '--transform', nargs='+', help='Apply a transform')

        # --------------------------------------------------------------
        #   UNLOAD
        # --------------------------------------------------------------
        unload = add_parser('unload', help='Unload a file')
        unload.set_defaults(func=self.unload)
        unload.add_argument(
            dest='layer', nargs='+', metavar='LAYER',
            help='Name(s) of layer(s) to unload')

        # --------------------------------------------------------------
        #   RENAME
        # --------------------------------------------------------------
        rename = add_parser('rename', help='Rename a file')
        rename.set_defaults(func=self.rename)
        rename.add_argument(
            dest='src', metavar='SOURCE', help='Current layer name')
        rename.add_argument(
            dest='dst', metavar='DEST', help='New layer name')

        # --------------------------------------------------------------
        #   TRANSFORM
        # --------------------------------------------------------------
        transform = add_parser('transform', help='Apply a transform')
        transform.set_defaults(func=self.transform)
        transform.add_argument(
            dest='transform', nargs='+', metavar='TRANSFORM',
            help='Path to transform file or flattened transformation '
                 'matrix (row major)')
        transform.add_argument(
            '--layer', nargs='+', help='Name(s) of layer(s) to transform')
        transform.add_argument(
            '--inv', action='store_true', default=False,
            help='Invert the transform before applying it')
        transform.add_argument(
            '--mov', help='Moving image (required by some formats)')
        transform.add_argument(
            '--fix', help='Fixed image (required by some formats)')

        # --------------------------------------------------------------
        #   AXIS MODE
        # --------------------------------------------------------------
        channelmode = add_parser(
            'channel_mode', help='Change the way a dimension is interpreted')
        channelmode.set_defaults(func=self.channel_mode)
        channelmode.add_argument(
            dest='mode', metavar='MODE',
            choices=('local', 'channel', 'spatial'),
            help='How to interpret the channel (or another) axis')
        channelmode.add_argument(
            '--layer', nargs='+', default=None,
            help='Name(s) of layer(s) to transform')
        channelmode.add_argument(
            '--dimension', nargs='+', default=['c'],
            help='Name(s) of axes to transform')

        # --------------------------------------------------------------
        #   SHADER
        # --------------------------------------------------------------
        shader = add_parser('shader', help='Apply a shader')
        shader.set_defaults(func=self.shader)
        shader.add_argument(
            dest='shader', metavar='SHADER',
            help='Shader name or GLSL shader code')
        shader.add_argument(
            '--layer', nargs='+', help='Layer(s) to apply shader to')

        # --------------------------------------------------------------
        #   DISPLAY
        # --------------------------------------------------------------
        display = add_parser('display', help='Dimensions to display')
        display.set_defaults(func=self.display)
        display.add_argument(
            dest='dimensions', nargs='*', metavar='DIMENSIONS',
            help='Dimensions to display')
        display.add_argument(
            '--layer', default=None,
            help='Show in this layer\'s canonical space')
        display.add_argument(
            '--world', dest='layer', action='store_const', const=self._WORLD,
            help='Show in world space', default=None)

        # --------------------------------------------------------------
        #   LAYOUT
        # --------------------------------------------------------------
        LAYOUTS = ["xy", "yz", "xz", "xy-3d", "yz-3d", "xz-3d", "4panel", "3d"]
        layout = add_parser('layout', help='Layout')
        layout.set_defaults(func=self.layout)
        layout.add_argument(
            dest='layout', nargs='*', choices=LAYOUTS, metavar='LAYOUT',
            help='Layout')
        layout.add_argument(
            '--stack', choices=("row", "column"), help="Stack direction")
        layout.add_argument(
            '--layer', nargs='*', help="Layer(s) to include")
        layout.add_argument(
            '--flex', type=float, default=1, help="Flex")
        layout.add_argument(
            '--append', type=int, nargs='*',
            help="Append to existing (nested) layout")
        layout.add_argument(
            '--insert', type=int, nargs='+',
            help="Insert in existing (nested) layout")
        layout.add_argument(
            '--remove', type=int, nargs='+',
            help="Remove from an existing (nested) layout")

        # --------------------------------------------------------------
        #   STATE
        # --------------------------------------------------------------
        state = add_parser('state', help='Return the viewer\'s state')
        state.set_defaults(func=self.state)
        state.add_argument(
            '--no-print', action='store_false', default=True, dest='print',
            help='Do not print the state.')
        state.add_argument(
            '--save', help='Save JSON state to this file.')
        state.add_argument(
            '--load', help='Load JSON state from this file. '
                           'Can also be a JSON string or a URL.')
        state.add_argument(
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

        # orient = add_parser(
        #     'orient', help='Reorient the cross-section')
        # orient.set_defaults(func=self.orient)
        # orient.add_argument(
        #     '--normal', nargs='*', type=float,
        #     help='Normal vector of the cross-section plane.')
        # _add_common_args(orient)

        # --------------------------------------------------------------
        #   ZORDER
        # --------------------------------------------------------------
        zorder = add_parser('zorder', help='Reorder layers')
        zorder.set_defaults(func=self.zorder)
        zorder.add_argument(
            dest='layer', nargs='+', metavar='LAYER',
            help='Layer(s) name(s)')
        zorder.add_argument(
            '--up', '-u', '-^', action='count', default=0,
            help='Move upwards')
        zorder.add_argument(
            '--down', '-d', '-v', action='count', default=0,
            help='Move downwards')

        # --------------------------------------------------------------
        #   NAVIGATION
        # --------------------------------------------------------------
        cd = add_parser('cd', help='Change directory')
        cd.set_defaults(func=self.cd)
        cd.add_argument(dest='path', metavar='PATH')

        ls = add_parser('ls', help='List files')
        ls.set_defaults(func=self.ls)
        ls.add_argument(dest='path', nargs='?', default='.', metavar='PATH')

        pwd = add_parser('pwd', help='Path to working directory')
        pwd.set_defaults(func=self.pwd)

        # --------------------------------------------------------------
        #   EXIT
        # --------------------------------------------------------------
        exit = add_parser('exit', aliases=['quit'], help='Exit neuroglancer')
        exit.set_defaults(func=self.exit)
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
    orient = state_action("orient")
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
        print(*files)
        return files

    @action
    def pwd(self) -> str:
        """Path to working directory."""
        print(os.getcwd())
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
            self.parser.parse_args([action, '--help'])
        else:
            self.parser.parse_args(['--help'])

    @action
    def exit(self) -> None:
        """Exit gracefully."""
        if hasattr(self, 'fileserver'):
            del self.fileserver
        sys.exit()

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
B = bcolors.bold
E = bcolors.endc
r, g, b = bcolors.fg.red, bcolors.fg.green, bcolors.fg.blue

_clihelp.load = textwrap.dedent(
f"""
Load a file, which can be local or remote.

{B}Paths and URLs{E}
{B}--------------{E}
Each path or url may be prepended by:

1)  A layer type protocol, which indicates the kind of object that the file
    contains.
    Examples: {B}volume://{E}, {B}labels://{E}, {B}tracts://{E}.

2)  A format protocol, which indicates the exact file format.
    Examples: {B}nifti://{E}, {B}zarr://{E}, {B}mgh://{E}.

3)  An access protocol, which indices the protocol used to  access the files.
    Examples: {B}https://{E}, {B}s3://{E}, {B}dandi://{E}.

All of these protocols are optional. If absent, a guess is made using the
file extension.

{B}Examples{E}
{B}--------{E}

- Absolute path to local file:  {B}/absolute/path/to/mri.nii.gz{E}
- Relative path to local file:  {B}relative/path/to/mri.nii.gz{E}
- Local file with format hint:  {B}mgh://relative/path/to/linkwithoutextension{E}
- Remote file:                  {B}https://url.to/mri.nii.gz{E}
- Remote file with format hint: {B}zarr://https://url.to/filewithoutextension{E}
- File on dandiarchive:         {B}dandi://dandi/<dandiset>/sub-<id>/path/to/file.ome.zarr{E}

{B}Layer names{E}
{B}-----------{E}
Neuroglancer layers are named. The name of the layer can be specified with
the {B}--name{E} option. Otherwise, the base name of the file is used (that
is, without the folder hierarchy).

If multiple files are loaded _and_ the --name option is used, then there
should be as many names as files.

{B}Transforms{E}
{B}----------{E}
A spatial transform (common to all files) can be applied to the loaded
volume. The transform is specified with the {B}--transform{E} option, which
can be a flattened affine matrix (row major) or the path to a transform file.
Type {B}help transform{E} for more information.

{B}Arguments{E}
{B}----------{E}"""  # noqa: E122, E501
)

_clihelp.unload = "Unload layers"

_clihelp.rename = "Rename a layer"

_clihelp.shader = textwrap.dedent(
f"""
Applies a colormap, or a more advanced shading function to all or some of
the layers.

{B}List of builtin colormaps{E}
{B}-------------------------{E}
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

2)  Provide a positive ({B}--up{E}) or negative ({B}--down{E}) number of
    steps by which to move the listed layers.
    In this case, more than one up or down step can be provided, using
    repeats of the option.
    Examples: {B}-vvvv{E} moves downward 4 times
              {B}-^^^^{E} moves upwards 4 times
"""  # noqa: E122
)

_clihelp.display = textwrap.dedent(
f"""
Display the data in a different space (world or canonical), or in a
different orientation (RAS, LPI, and permutations thereof)

By default, neuroglancer displays data in their "native" space, where
native means a "XYZ" coordinate frame, whose mapping from and to voxels
is format-spacific. In {B}nifti://{E} volumes, XYZ always corresponds to
the RAS+ world space, whereas in {B}zarr://{E}, XYZ correspond to canonical
axes ordered according to the OME metadata.

Because of our neuroimaging bias, we choose to interpret XYZ as RAS+ (which
is consistant with the nifti behavior).

The simplest usage of this command is therefore to switch between
different representations of the world coordinate system:
    . {B}display RAS{E}  (alias: {B} display right anterior superior{E})
      maps the {r}red{E}, {g}green{E} and {b}blue{E} axes to {r}right{E}, {g}anterior{E}, {b}superior{E}.
    . {B}display LPI{E}  (alias: {B} display left posterior inferior{E})
      maps the {r}red{E}, {g}green{E} and {b}blue{E} axes to {r}left{E}, {g}posterior{E}, {b}inferior{E}.

A second usage allows switching between displaying the data in the world
frame and displaying the data in the canonical frame of one of the layers
(that is, the frame axes are aligned with the voxel dimensions).
    . {B}display --layer <name>{E} maps the {r}red{E}/{g}green{E}/{b}blue{E}
      axes to the canonical axes of the layer <name>
    . {B}display --world{E} maps (back) the {r}red{E}/{g}green{E}/{b}blue{E}
      axes to the axes of world frame.

Of course, both usages can be combined:
    . {B}display RAS --layer <name>{E}
    . {B}display LPI --world{E}

"""  # noqa: E122, E501
)

_clihelp.layout = textwrap.dedent(
f"""
Change the viewer's layout (i.e., the quadrants and their layers)

Neuroglancer has 8 different window types:
. xy     : {r}X{E}{g}Y{E} cross-section
. yz     : {g}Y{E}{b}Z{E} cross-section
. xz     : {r}X{E}{b}Z{E} cross-section
. xy-3d  : {r}X{E}{g}Y{E} cross-section in a {B}3D{E} window
. yz-3d  : {g}Y{E}{b}Z{E} cross-section in a {B}3D{E} window
. xz-3d  : {r}X{E}{b}Z{E} cross-section in a {B}3D{E} window
. 4panel : Four quadrants ({r}X{E}{g}Y{E}, {r}X{E}{b}Z{E}, {B}3D{E}, {g}Y{E}{b}Z{E})
. 3d     : {B}3D{E} window

It is possible to build a user-defined layout by stacking these basic
windows into a row or a column -- or even nested rows and columns --
using the {B}--stack{E} option. The {B}--layer{E} option allows assigning
specific layers to a specific window. We also define {B}--append{E} and
{B}--insert{E} to add a new window into an existing stack of windows.

"""  # noqa: E122, E501
)
