"""Local neuroglancer instance with a shell-like interface."""
# stdlib
import argparse
import atexit
import contextlib
import datetime
import logging
import os.path
import stat
import sys
import textwrap
from functools import partial, wraps
from typing import Generator

# externals
import neuroglancer as ng
import numpy as np
from neuroglancer.server import set_server_bind_address as ng_bind_address
from neuroglancer.server import stop as ng_stop_server

# internals
from ngtools.local.console import ActionHelpFormatter, Console
from ngtools.local.fileserver import LocalFileServer, StaticFileHandler
from ngtools.local.handlers import LincHandler, LutHandler
from ngtools.local.termcolors import bformat
from ngtools.scene import Scene
from ngtools.shaders import pretty_colormap_list
from ngtools.utils import NG_URLS, find_available_port

# unix-specific imports
try:
    import grp
    import pwd
except ImportError:
    pwd = grp = None


LOG = logging.getLogger(__name__)


def action(needstate: bool = False) -> callable:
    """
    Decorate a neuroglancer action.

    These actions can thereby be triggered by argparse.
    """
    if callable(needstate):
        return action()(needstate)

    def decorator(func: callable) -> callable:

        @wraps(func)
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
    @wraps(getattr(Scene, name))
    def func(
        self: "LocalNeuroglancer", *args, state: ng.ViewerState, **kwargs
    ) -> object | None:
        # build scene
        scene = Scene(state.to_json(), stdio=self.console.stdio)
        # if loader, pass fileserver
        if name in ("load", "shader"):
            kwargs["fileserver"] = self.get_fileserver_url()
        # run action
        scene_fn = getattr(scene, name)
        out = scene_fn(*args, **kwargs)
        # save state
        for key in scene.to_json().keys():
            val = getattr(scene, key)
            setattr(state, key, val)
        return out

    return func


def ensure_list(x: object) -> list:
    """Ensure that an object is a list. Make one if needed."""
    if isinstance(x, np.ndarray):
        x = x.tolist()
    if not isinstance(x, (list, tuple)):
        x = [x]
    return list(x)


class OSMixin:
    """Operating system actions."""

    @action
    def cd(self, path: str) -> str:
        """Change directory."""
        os.chdir(os.path.expanduser(path))
        return os.getcwd()

    @action
    def ls(self, path: str, **kwargs) -> list[str]:
        """List files."""
        long = kwargs.pop("long", False)
        hidden = kwargs.pop("hidden", False)

        if long:
            files = os.scandir(os.path.expanduser(path))
            files = sorted(files, key=lambda x: x.name)
            keys = ("perms", "nlinks", "owners", "groups", "sizes",
                    "months", "days", "times", "names")
            mystat = dict([(key, []) for key in keys])
            for entry in files:
                if not hidden and entry.name[:1] == ".":
                    continue
                lstat = entry.stat(follow_symlinks=False)
                mystat["perms"].append(stat.filemode(lstat.st_mode))
                mystat["nlinks"].append(lstat.st_nlink)
                if pwd:
                    mystat["owners"].append(pwd.getpwuid(lstat.st_uid).pw_name)
                else:
                    mystat["owners"].append(lstat.st_uid)
                if grp:
                    mystat["groups"].append(grp.getgrgid(lstat.st_gid)[0])
                else:
                    mystat["groups"].append("")
                mystat["sizes"].append(lstat.st_size)
                date = datetime.datetime.fromtimestamp(lstat.st_mtime)
                mystat["months"].append(date.strftime("%b"))
                mystat["days"].append(date.strftime("%d"))
                if date.year != datetime.datetime.today().year:
                    mystat["times"].append(date.strftime("%Y"))
                else:
                    mystat["times"].append(date.strftime("%H:%M"))
                if entry.is_symlink():
                    fname = bformat.bold(bformat.fg.magenta(entry.name))
                    fname += " -> " + os.readlink(entry.path)
                elif entry.is_dir():
                    fname = bformat.bold(bformat.fg.blue(entry.name))
                elif "x" in mystat["perms"][-1]:
                    fname = bformat.bold(bformat.fg.green(entry.name))
                else:
                    fname = entry.name
                mystat["names"].append(fname)

            colsizes = [max(len(str(y)) for y in x) for x in mystat.values()]
            for values in zip(*mystat.values()):
                for key, value, colsize in zip(keys, values, colsizes):
                    align = (
                        ">" if key in ("nlinks", "sizes", "days", "times")
                        else ""
                    )
                    fmt = "{:" + align + str(colsize) + "s}"
                    print(fmt.format(str(value)), end=" ")
                print("")
        else:
            files = os.listdir(os.path.expanduser(path))
            if not hidden:
                files = [file for file in files if file[:1] != "."]
            print(*files)
        return files

    @action
    def pwd(self) -> str:
        """Path to working directory."""
        self.console.print(os.getcwd())
        return os.getcwd()

    @action
    def stdin(self, stdin: str | None = None) -> str:
        """Input stream."""
        if stdin:
            if stdin == "sys":
                stdin = sys.stdin
            elif stdin[:4] == "sys.":
                parts = stdin.split(".")[1:]
                stdin = sys
                for part in parts:
                    stdin = getattr(stdin, part)
            self.console.stdio.stdin = stdin
        return self.console.stdio.stdin

    @action
    def stdout(self, stdout: str | None = None) -> str:
        """Output stream."""
        if stdout:
            if stdout == "sys":
                stdout = sys.stdout
            elif stdout[:4] == "sys.":
                parts = stdout.split(".")[1:]
                stdout = sys
                for part in parts:
                    stdout = getattr(stdout, part)
            self.console.stdio.stdout = stdout
        return self.console.stdio.stdout

    @action
    def stderr(self, stderr: str | None = None) -> str:
        """Error stream."""
        if stderr:
            if stderr == "sys":
                stderr = sys.stderr
            elif stderr[:4] == "sys.":
                parts = stderr.split(".")[1:]
                stderr = sys
                for part in parts:
                    stderr = getattr(stderr, part)
            self.console.stdio.stderr = stderr
        return self.console.stdio.stderr


class LocalNeuroglancer(OSMixin):
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
        **console_kwargs,
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
        fileserver : bool | int | LocalFileServer
            Whether to run a local fileserver.
            If an int, should be the port to use.
        debug : bool
            Print full trace when an error is encountered.
        """
        # Setup neuroglancer instance
        port, ip = find_available_port(port, ip)
        ng_bind_address(str(ip), str(port))
        self.viewer = ng.Viewer(token=str(token))
        # self.viewer.shared_state.add_changed_callback(self.on_state_change)

        ip = self.get_viewer_url().split("://")[1].split(":")[0]

        # Add specific handlers
        if fileserver is not False:
            if not isinstance(fileserver, LocalFileServer):
                if fileserver is True:
                    fileserver_port = 0
                else:
                    fileserver_port = int(fileserver)
                fileserver = LocalFileServer(fileserver_port, ip)

            fileserver.app.handlers.extend([
                (r"^/local/(.*)", StaticFileHandler),
                (r"^/lut/([^/]+)/(.*)", LutHandler),
                (r"^/linc/(.*)", LincHandler),
            ])
        self.fileserver = fileserver

        # Setup console
        self.console = self._make_console(**console_kwargs)
        atexit.register(self._cleanup)

    def _cleanup(self) -> None:
        ng_stop_server()  # do I need this?
        atexit.unregister(self._cleanup)

    def __del__(self) -> None:
        self._cleanup()

    def txn(self) -> contextlib.AbstractContextManager[ng.ViewerState]:
        """
        Context manager that returns the underlying neuroglancer state.
        Transactions are written into the viewer at exit.
        """
        return self.viewer.txn()

    @contextlib.contextmanager
    def _scene(self) -> Generator[Scene, None, None]:
        with self.viewer.txn() as state:
            # Build and yield scene
            scene = Scene(state.to_json(), stdio=self.console.stdio)
            yield scene
            # Save scene's state into viewer's state
            for key in scene.to_json().keys():
                val = getattr(scene, key)
                setattr(state, key, val)

    def scene(self) -> contextlib.AbstractContextManager[Scene]:
        """
        Context manager that returns the underlying state as a `Scene`.
        It writes all changes to the scene back into the state at exit.
        """
        return self._scene()

    def get_server_url(self) -> str:
        """URL of the neuroglancer server."""
        return ng.server.get_server_url().rstrip("/") + "/"

    def get_viewer_url(self) -> str:
        """URL of the viewer."""
        return self.viewer.get_viewer_url().rstrip("/") + "/"

    def get_fileserver_url(self) -> str:
        """URL of the neuroglancer server."""
        return self.fileserver.get_url().rstrip("/") + "/"

    # ==================================================================
    #
    #                   COMMANDLINE APPLICATION
    #
    # ==================================================================

    def await_input(self) -> None:
        """Launch shell-like interface."""
        return self.console.await_input()
        # self._await_input = Thread(target=self.console.await_input)
        # self._await_input.start()

    def _make_console(self, **kwargs) -> Console:
        kwargs.setdefault('max_choices', 4)
        mainparser = Console('', **kwargs)
        parsers = mainparser.add_subparsers()
        F = dict(formatter_class=ActionHelpFormatter)

        def add_parser(cmd, *args, **kwargs):  # noqa: ANN001, ANN202
            # We define long descriptions in the _clihelp class at the
            # end of this file.
            description = kwargs.get("description", None)
            if description is None:
                description = getattr(_clihelp, cmd, None)
            if description is None:
                description = kwargs.get("help", None)
            if description:
                description = description.rstrip()
                description += "\n\n"
                description += _clihelp.header_attributes
            kwargs.setdefault('description', description)
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
            '--name', '-n', nargs='+', help='A name for the image layer')
        _.add_argument(
            '--transform', '-t', nargs='+', help='Apply a transform')
        _.add_argument(
            '--shader', '-s', help='Apply a shader')

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
            '--destination', '--dst', '-d', metavar='DEST', nargs="+",
            help='New axis names')
        _.add_argument(
            '--source', '--src', '-s', metavar='SOURCE', nargs="*",
            help='Native axis names')
        _.add_argument(
            '--print', '-p', action="store_true", help='Print result')

        # --------------------------------------------------------------
        #   RENAME AXES
        # --------------------------------------------------------------
        _ = add_parser('rename_axes', help='Rename axes')
        _.set_defaults(func=self.rename_axes)
        _.add_argument(
            dest='axes', metavar='DEST', nargs="+", help='New axis names')
        _.add_argument(
            '--destination', '--dst', '-d', metavar='DEST', nargs="+",
            help='New axis names')
        _.add_argument(
            '--source', '--src', '-s', metavar='SOURCE', nargs="*",
            help='Old axis names')
        _.add_argument(
            '--layer', '-l', nargs="*", help='Layer(s) to rename')

        # --------------------------------------------------------------
        #   SPACE
        # --------------------------------------------------------------
        MODES = ("radio", "neuro", "default")
        _ = add_parser('space', help='Cross-section orientation')
        _.set_defaults(func=self.space)
        _.add_argument(
            dest='mode', nargs="?", help='New axis names', choices=MODES)
        _.add_argument(
            '--layer', '-l',
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
            '--layer', '-l', nargs='+',
            help='Name(s) of layer(s) to transform')
        _.add_argument(
            '--inv', '-i', action='store_true', default=False,
            help='Invert the transform before applying it')
        _.add_argument(
            '--mov', '-m', help='Moving image (required by some formats)')
        _.add_argument(
            '--fix', '-f', help='Fixed image (required by some formats)')

        # --------------------------------------------------------------
        #   AXIS MODE
        # --------------------------------------------------------------
        _ = add_parser(
            'channel_mode', help='Change the way a dimension is interpreted')
        _.set_defaults(func=self.channel_mode)
        _.add_argument(
            dest='mode',
            choices=('local', 'channel', 'global'),
            help='How to interpret the channel (or another) axis')
        _.add_argument(
            '--layer', '-l', nargs='+', default=None,
            help='Name(s) of layer(s) to transform')
        _.add_argument(
            '--dimension', '-d', nargs='+', default=['c'],
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
            '--layer', '-l', nargs='+',
            help='Layer(s) to apply shader to')
        _.add_argument(
            '--layer-type', '-t', nargs='+',
            help='Layer type(s) to apply shader to')

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
            '--stack', '-s', choices=("row", "column"), help="Stack direction")
        _.add_argument(
            '--layer', '-l', nargs='*', help="Layer(s) to include")
        _.add_argument(
            '--flex', type=float, default=1, help="Flex")
        _.add_argument(
            '--append', '-a', type=int, nargs='*',
            help="Append to existing (nested) layout")
        _.add_argument(
            '--assign', '-x', type=int, nargs='*',
            help="Assign into existing (nested) layout")
        _.add_argument(
            '--insert', '-i', type=int, nargs='+',
            help="Insert in existing (nested) layout")
        _.add_argument(
            '--remove', '-r', type=int, nargs='+',
            help="Remove from an existing (nested) layout")
        _.add_argument(
            '--row', dest="stack", action="store_const", const="row",
            help="Alias for `--stack row`")
        _.add_argument(
            '--column', "--col", dest="stack",
            action="store_const", const="column",
            help="Alias for `--stack column`")

        # --------------------------------------------------------------
        #   STATE
        # --------------------------------------------------------------
        _ = add_parser('state', help='Return the viewer\'s state')
        _.set_defaults(func=self.state)
        _.add_argument(
            '--no-print', action='store_false', default=True, dest='print',
            help='Do not print the state.')
        _.add_argument(
            '--save', '-s', help='Save JSON state to this file.')
        _.add_argument(
            '--load', '-l',
            help='Load JSON state from this file. '
                 'Can also be a JSON string or a URL.')
        _.add_argument(
            '--url', '-u', action='store_true', default=False,
            help='Load (or print) the url form of the state')
        _.add_argument(
            '--open', '-o', action='store_true', default=False,
            help='Open the url (if `--url`) or viewer (otherwise)')
        _.add_argument(
            '--instance', '-i', choices=list(NG_URLS.keys()),
            help='Link to this neuroglancer instance')

        # --------------------------------------------------------------
        #   POSITION
        # --------------------------------------------------------------
        _ = add_parser('move', help='Move cursor', aliases=("position",))
        _.set_defaults(func=self.move)
        _.add_argument(
            dest='coord', nargs='*', metavar='COORD', type=float,
            help='Cursor coordinates. If None, print current one.')
        _.add_argument(
            '--dimensions', '-d', nargs='+', default=None,
            help='Axis name for each coordinate (can be compact)')
        _.add_argument(
            '--unit', '-u',
            help='Coordinates are expressed in this unit')
        _.add_argument(
            '--absolute', '--abs', '-a', action='store_true', default=False,
            help='Move to absolute position, rather than relative to current')
        _.add_argument(
            '--reset', '-r', action='store_true', default=False,
            help='Reset coordinates to zero')

        # --------------------------------------------------------------
        #   ZOOM
        # --------------------------------------------------------------
        _ = add_parser('zoom', help='Zoom by a factor [default: x2]')
        _.set_defaults(func=self.zoom)
        _.add_argument(
            dest='factor', nargs='?', type=float, default=2.0,
            help='Zoom factor.')
        _.add_argument(
            '--reset', '-r', action='store_true', default=False,
            help='Reset zoom level to default.')

        _ = add_parser('unzoom', help='Unzoom by a factor [default: ÷2]')
        _.set_defaults(func=self.unzoom)
        _.add_argument(
            dest='factor', nargs='?', type=float, default=2.0,
            help='Inverse zoom factor.')
        _.add_argument(
            '--reset', '-r', action='store_true', default=False,
            help='Reset zoom level to default.')

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
        _.add_argument('--long', '-l', action="store_true")
        _.add_argument('--hidden', '-a', action="store_true")

        _ = add_parser('ll', help='List files (long form)')
        _.set_defaults(func=partial(self.ls, long=True, hidden=True))
        _.add_argument(dest='path', nargs='?', default='.', metavar='PATH')

        _ = add_parser('pwd', help='Path to working directory')
        _.set_defaults(func=self.pwd)

        _ = add_parser('stdin', help='Set input stream')
        _.set_defaults(func=self.stdin)
        _.add_argument(dest='stdin', metavar='FILE')
        _ = add_parser('stdout', help='Set output stream')
        _.set_defaults(func=self.stdout)
        _.add_argument(dest='stdout', metavar='FILE')
        _ = add_parser('stderr', help='Set error stream')
        _.set_defaults(func=self.stderr)
        _.add_argument(dest='stderr', metavar='FILE')

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
    transform = state_action("transform")
    channel_mode = state_action("channel_mode")
    move = state_action("move")
    zoom = state_action("zoom")
    unzoom = state_action("unzoom")
    shader = state_action("shader")
    layout = state_action("change_layout")
    zorder = state_action("zorder")
    state = state_action("state")

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

b = bformat.bold
u = bformat.underline
i = bformat.italic
R = bformat.fg.red
G = bformat.fg.green
B = bformat.fg.blue

_clihelp.header_attributes = textwrap.dedent(
    f"{b('Arguments')}\n{b('----------')}\n"
)

_clihelp.load = textwrap.dedent(
f"""
Load a file, which can be local or remote.

{b("Paths and URLs")}
{b("--------------")}
Each path or url may be prepended by:

1)  A layer type protocol that indicates the kind of object that the file
    contains.
    Examples: {b("volume://")}, {b("labels://")}, {b("tracts://")}.

2)  A format protocol that indicates the exact file format.
    Examples: {b("nifti://")}, {b("zarr://")}, {b("mgh://")}.

3)  An access protocol that indicates the protocol used to access the files.
    Examples: {b("https://")}, {b("s3://")}, {b("dandi://")}.

All of these protocols are optional. If absent, a guess is made using the
file extension.

{b("Examples")}
{b("--------")}

. Absolute path to local file:  {b("/absolute/path/to/mri.nii.gz")}
. Relative path to local file:  {b("relative/path/to/mri.nii.gz")}
. Local file with format hint:  {b("mgh://relative/path/to/linkwithoutextension")}
. Remote file:                  {b("https://url.to/mri.nii.gz")}
. Remote file with format hint: {b("zarr://https://url.to/filewithoutextension")}
. File on dandiarchive:         {b("dandi://dandi/<dandiset>/sub-<id>/path/to/file.ome.zarr")}

{b("Layer names")}
{b("-----------")}
Neuroglancer layers are named. The name of the layer can be specified with
the {b("--name")} option. Otherwise, the base name of the file is used (that
is, without the folder hierarchy).

If multiple files are loaded {i("and")} the {b("--name")} option is used, then there
should be as many names as files.

{b("Transform")}
{b("---------")}
A spatial transform (common to all files) can be applied to the loaded
volume. The transform is specified with the {b("--transform")} option, which
can be a flattened affine matrix (row major) or the path to a transform file.
Type {b("help transform")} for more information.

{b("Shader")}
{b("------")}
A shader (= colormap, common to all files) can be applied to the loaded
volume. The shader is specified with the {b("--shader")} option, which
can be the name of a colormap, the path to a LUT file, or a snippet of
GLSL code. Type {b("help shader")} for more information.
"""  # noqa: E122, E501
)

_clihelp.unload = "Unload layers"

_clihelp.rename = "Rename a layer"

_clihelp.shader = textwrap.dedent(
f"""
Applies a colormap, or a more advanced shading function to all or some of
the layers.

The input can also be the path to a (local or remote) freesurfer LUT file.

{b("List of builtin colormaps")}
{b("-------------------------")}
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

2)  Provide a positive ({b("--up")}) or negative ({b("--down")})
    number of steps by which to move the listed layers.
    In this case, more than one up or down step can be provided, using
    repeats of the option.
    Examples: {b("-vvvv")} moves downward 4 times
              {b("-^^^^")} moves upwards 4 times
"""  # noqa: E122, E501
)

_clihelp.display = textwrap.dedent(
f"""
Neuroglancer is quite flexible in the sense that it does not have a
predefined "hard" coordinate frame in which the data is shown. Instead,
the {R("red")}, {G("green")} and {B("blue")} "visual" axes can be
arbitrarily mapped to any of the existing "model" axes.

By default, most formats use ({b("x")}, {b("y")}, {b("z")}) as their
model axes, although they may not have predefined anatomical meaning.
NIfTI files do have an anatomical convention, under which axes
{R("x")}, {G("y")}, {B("z")} map to the {R("right")}, {G("anterior")}, {B("superior")} sides of the brain.

In order to show data in a frame that is standard in the neuroimaging
field, we therefore map the visual axes {R("red")}, {G("green")}, {B("blue")}
to the model axes {R("x")}, {G("y")}, {B("z")} when loading data in an
empty scene. This mapping is also enforced every time the command
{b("space")} is used.

The {b("display")} command can be used to assign a different set of
model axes to the visual axes.

{u("See also")}:  {b("space")}, {b("world_axes")}
"""  # noqa: E122, E501
)

_clihelp.world_axes = textwrap.dedent(
f"""
Neuroglancer is quite flexible in the sense that it does not have a
predefined "hard" coordinate frame in which the data lives. Instead,
arbitrary "model" axes can be defines, in terms of an affine transformation
of "native" axes.

By default, most formats use ({b("x")}, {b("y")}, {b("z")}) as their
model axes, although they may not have predefined anatomical meaning.
NIfTI files do have an anatomical convention, under which axes
{R("x")}, {G("y")}, {B("z")} map to the {R("right")}, {G("anterior")}, {B("superior")} sides of the brain.

This function allows native axes to replaces by more anatomically
meaningful names, by leveraging neuroglancer's transforms. In order
to allow these transforms to be undone, and new data to be loaded without
introducing conflicting coordinate frames, we store the mapping inside
local annotation layers named "__world_axes_native__" and
"__world_axes_current__".

{u("See also")}:  {b("space")}, {b("display")}

{b("Examples")}
{b("--------")}

. `world_axes right anterior superior` : x -> right, y -> anterior, z -> superior
. `world_axes ras`                     : x -> right, y -> anterior, z -> superior
. `world_axes u v w --src z y x`       : z -> u, y -> v, z -> w
"""  # noqa: E122, E501
)

_clihelp.space = textwrap.dedent(
f"""
This function rotates and orients the cross-section plane such that

. Visual axes are pointed according to some defined convention
  (radio or neuro)
. The cross section is aligned with either the model coordinate frame,
  or one of the layer's voxel space.

{u("Note:")} when this function is used, the displayed axes are always reset
to {R("x")}, {G("y")}, {B("z")} (or their world names if {b("world_axes")}) has been used.

{b("Examples")}
{b("--------")}

. `space radio`                 : Orients the cross section such that the {R("x")}
                                  axis points to the {b("left")} of the first quadrant
                                  (radiological convention).
. `space neuro`                 : Orients the cross section such that the {R("x")}
                                  axis points to the {b("right")} of the first quadrant
                                  (neurological convention).
. `space --layer <LAYER>`       : Aligns the cross-section with the voxels of
                                  the designated layer, while keeping the
                                  existing radio or neuro convention.
. `space --layer world`         : Aligns the cross-section with the canonical
                                  model space, while keeping the existing radio
                                  or neuro convention.
. `space radio --layer <LAYER>` : Aligns the cross-section with the voxels of
                                  the designated layer, and uses the radiological
                                  convention.
"""  # noqa: E122, E501
)

_clihelp.layout = textwrap.dedent(
f"""
Change the viewer's layout (i.e., the quadrants and their layers)

Neuroglancer has 8 different window types:
. xy     : {R("X")}{G("Y")} cross-section
. yz     : {G("Y")}{B("Z")} cross-section
. xz     : {R("X")}{B("Z")} cross-section
. xy-3d  : {R("X")}{G("Y")} cross-section in a {b("3D")} window
. yz-3d  : {G("Y")}{B("Z")} cross-section in a {b("3D")} window
. xz-3d  : {R("X")}{B("Z")} cross-section in a {b("3D")} window
. 4panel : Four quadrants ({R("X")}{G("Y")}, {R("X")}{B("Z")}, {b("3D")}, {G("Y")}{B("Z")})
. 3d     : {b("3D")} window

It is possible to build a user-defined layout by stacking these basic
windows into a row or a column -- or even nested rows and columns --
using the {b("--stack")} option. The {b("--layer")} option allows assigning
specific layers to a specific window. We also define {b("--append")} and
{b("--insert")} to add a new window into an existing stack of windows.
"""  # noqa: E122, E501
)

_clihelp.channel_mode = textwrap.dedent(
f"""
In neuroglancer, axes can be interpreted in three different ways:

. global  : The axis is common to all layers and can be navigated
            (default for axes that have spatial or temporal units)

. local   : The axis is specific to a layers and navigation is local
            (dimension names end with ')

. channel : The axis is specific to a layer; enables multi-channel shading
            (dimension names end with ^)

{b("Notes")}
{b("-----")}

. When an axis is local ('), only one channel can be shown at once.

. When an axis is a channel (^), all channels can be mixed into a single
  view using a multi-channel shader.

. When an axis is global, navigation is linked across layers.
"""  # noqa: E122, E501
)
