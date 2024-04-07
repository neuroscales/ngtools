import sys
import json
import socket
import os.path
import argparse
import textwrap
import numpy as np
import neuroglancer as ng
from urllib.parse import urlparse, unquote as urlunquote, quote as urlquote
from neuroglancer.server import global_server_args
from typing import Sequence
from .fileserver import LocalFileServerInBackground
from .volume import LocalSource, RemoteSource
from .tracts import TractSource
from .spaces import (
    neurotransforms, letter2full, to_square, compose,
    ensure_same_scaling, subtransform, si_convert, compact2full)
from .shaders import shaders, colormaps, pretty_colormap_list
from .transforms import load_affine
from .opener import remote_protocols
from .parserapp import ParserApp, _fixhelpformatter
from .utils import bcolors
from .dandifs import RemoteDandiFileSystem

# hack-fix Layer state to expose channelDimensions
from neuroglancer.viewer_state import Layer, CoordinateSpace, wrapped_property
Layer.channel_dimensions = Layer.channelDimensions = wrapped_property(
    "channelDimensions", CoordinateSpace
)
Layer.local_dimensions = Layer.localDimensions = Layer.layerDimensions

_print = print


def action(needstate=False):
    """
    Decorator for neuroglancer actions (that can be triggered  by argparse)
    """
    if callable(needstate):
        return action()(needstate)

    def decorator(func):

        def wrapper(self, *args, **kwargs):
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


def ensure_list(x):
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

    def __init__(self, port=0, ip='', token=1, fileserver=True, debug=False):
        """
        Parameters
        ----------
        port : int
            Port to use
        token : str
            Unique id for the instance
        filserver : LocalFileServerInBackground or bool
            A local file server.

            * If `True`, create a local file server.
            * If `False`, do not create one -- only remote files can be loaded.
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
        self.to_world = None

    _WORLD = object()

    # ==================================================================
    #
    #                   COMMANDLINE APPLICATION
    #
    # ==================================================================

    def await_input(self):
        """Launch shell-like interface"""
        try:
            return self.parser.await_input()
        except SystemExit:
            # exit gracefully (cleanup the fileserver process, etc)
            self.exit()

    def _make_parser(self, debug=False):
        mainparser = ParserApp('', debug=debug)
        parsers = mainparser.add_subparsers()
        formatter = _fixhelpformatter(argparse.RawDescriptionHelpFormatter)
        F = dict(formatter_class=formatter)

        def add_parser(cmd, *args, **kwargs):
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

    @action
    def cd(self, path):
        """Change directory"""
        os.chdir(os.path.expanduser(path))
        return os.getcwd()

    @action
    def ls(self, path):
        """List files"""
        files = os.listdir(os.path.expanduser(path))
        print(*files)
        return files

    @action
    def pwd(self):
        """Path to working directory"""
        print(os.getcwd())
        return os.getcwd()

    @action(needstate=True)
    def load(self, filename, name=None, transform=None,
             *, state=None, **kwargs):
        """
        Load file(s)

        Parameters
        ----------
        filename : str or list[str]
            Paths or URL, eventually prepended with "type" and "format"
            protocols. Ex: `"labels://nifti://http://cloud.com/path/to/nii.gz"`
        name : str or list[str], optional
            Layer(s) name(s).
        transform : array_like or list[float] or str, optional
            Affine transform to apply to the loaded volume.
        """
        filenames = ensure_list(filename or [])
        names = ensure_list(name or [])
        if names and len(filenames) != len(names):
            raise ValueError(
                'The number of names should match the number of files')
        display_dimensions = state.display_dimensions
        if display_dimensions:
            display_dimensions = list(map(str, display_dimensions))
        self.redisplay(None, state=state)

        onames = []
        for n, filename in enumerate(filenames):

            while filename.endswith('/'):
                filename = filename[:-1]

            layertype, format, filename = self.parse_filename(filename)
            name = names[n] if names else os.path.basename(filename)
            onames.append(name)

            if format in self.NG_FORMATS:
                # Remote source
                filename = self.ensure_url(filename)
                source = RemoteSource.from_filename(format + '://' + filename)
                controls = None
                if hasattr(source, 'quantiles'):
                    mn, q0, q1, mx = source.quantiles([0.0, 0.01, 0.99, 1.0])
                    controls = {
                        "normalized": {
                            "range": np.stack([q0, q1]),
                            "window": np.stack([mn, mx]),
                        }
                    }
                layer = ng.ImageLayer(
                    source=ng.LayerDataSource(
                        url=format + '://' + filename,
                        transform=getattr(source, 'transform', None),
                    ),
                    shaderControls=controls,
                )

            elif layertype == 'tracts':
                source = TractSource(filename)
                source._load()
                maxtracts = len(source.tractfile.streamlines)
                nbtracts = min(maxtracts, kwargs.get('nb_tracts', 1000))
                source.max_tracts = nbtracts
                source._filter()
                controls = {
                    "nbtracts": {
                        "min": 0,
                        "max": maxtracts,
                        "default": nbtracts,
                    }
                }
                layer = ng.SegmentationLayer(
                    source=[source],
                    skeleton_shader=shaders.trkorient,
                    selected_alpha=0,
                    not_selected_alpha=0,
                    segments=[1],
                )

            elif format in self.EXTRA_FORMATS:
                # Local source
                if format:
                    filename = format + '://' + filename
                source = LocalSource.from_filename(
                    filename, layer_type=layertype)
                controls = None
                if hasattr(source, 'quantiles'):
                    mn, q0, q1, mx = source.quantiles([0.0, 0.01, 0.99, 1.0])
                    controls = {
                        "normalized": {
                            "range": np.stack([q0, q1]),
                            "window": np.stack([mn, mx]),
                        }
                    }
                layer = ng.ImageLayer(
                    source=source,
                    shaderControls=controls,
                )

            else:
                raise ValueError(
                    'Unrecognized format. Try specifying a format with the '
                    'protocol syntax.'
                )

            state.layers.append(name=name, layer=layer)

        if transform:
            self.transform(transform, name=onames, state=state)
        self.redisplay(display_dimensions, state=state)

    @action(needstate=True)
    def unload(self, layer=None, *, state=None):
        """
        Unload layers

        Parameters
        ----------
        layer : str or list[str]
            Layer(s) to unload
        """
        layers = layer or [layer.name for layer in state.layers]
        for name in ensure_list(layers):
            del state.layers[name]

    @action(needstate=True)
    def rename(self, src, dst, *, state=None):
        """
        Rename a layer

        Parameters
        ----------
        src : str
            Current name
        dst : str
            New name
        """
        for layer in state.layers:
            if layer.name == src:
                layer.name = dst
                return
        raise ValueError('No lauyer named', src)

    @action(needstate=True)
    def display(self, dimensions=None, layer=None, *, state=None):
        """
        Change displayed dimensions.

        Parameters
        ----------
        dimensions : str or list[str]
            The three dimensions to display.
            Each dimension can be one of:

            * `"left"` or `"right"`
            * `"posterior"` or `"anterior"`
            * `"inferior"` or `"superior"`

            Otherwise, dimensions can be native axis names of the loaded
            data, such as `"x"`, `"y"`, `"z"`.

            A compact representation (`"RAS"`, or `"zyx"`) can also be
            provided.

        layer : str
            If provided, display in this layer's canonical frame,
            instead of world space.
        """

        def getDimensions(source, _reentrant=False):
            if getattr(source, 'transform', None):
                transform = source.transform
                if transform.inputDimensions:
                    return transform.inputDimensions
            if getattr(source, 'dimensions', None):
                return source.dimensions
            if not _reentrant and not isinstance(source, ng.LocalVolume):
                mysource = RemoteSource.from_filename(source.url)
                return getDimensions(mysource, True)
            return None

        def getTransform(source):
            transform = None
            if getattr(source, 'transform', None):
                transform = source.transform
            elif not isinstance(source, ng.LocalVolume):
                mysource = RemoteSource.from_filename(source.url)
                if getattr(mysource, 'transform', None):
                    transform = mysource.transform
            if transform and not transform.input_dimensions:
                transform = ng.CoordinateSpaceTransform(
                    matrix=transform.matrix,
                    input_dimensions=getDimensions(source),
                    output_dimensions=transform.output_dimensions,
                )
            if transform and transform.matrix is None:
                matrix = np.eye(len(transform.output_dimensions.names))[:-1]
                transform = ng.CoordinateSpaceTransform(
                    matrix=matrix,
                    input_dimensions=transform.input_dimensions,
                    output_dimensions=transform.output_dimensions,
                )
            return transform

        def lin2aff(x):
            matrix = np.eye(len(x)+1)
            matrix[:-1, :-1] = x
            return matrix

        if layer is self._WORLD:
            # move to world axes
            if self.to_world is not None:
                self.transform(lin2aff(self.to_world), _mode='', state=state)
                self.to_world = None
        elif layer is not None:
            # move to canonical axes
            layer = state.layers[layer]
            transform = subtransform(getTransform(layer.source[0]), 'm')
            transform = ensure_same_scaling(transform)
            matrix = transform.matrix
            if matrix is None:
                matrix = np.eye(len(transform.output_dimensions.names))
            matrix = to_square(matrix)[:-1, :-1]
            # remove scales and shears
            u, _, vh = np.linalg.svd(matrix)
            rot = u @ vh
            # preserve permutations and flips
            orient = rot.round()  # may not work if exactly 45 deg rotation
            assert np.allclose(orient @ orient.T, np.eye(len(orient)))
            # apply transform
            rot = rot @ orient.T
            self.transform(lin2aff(rot.T), _mode='', state=state)
            if self.to_world is None:
                self.to_world = rot
            else:
                self.to_world = self.to_world @ rot

        if not dimensions:
            return

        dimensions = ensure_list(dimensions)
        if len(dimensions) == 1:
            dimensions = list(dimensions[0])
        if len(dimensions) > 3:
            raise ValueError('display takes at most three axis names')
        dimensions = [letter2full.get(letter.lower(), letter)
                      for letter in dimensions]
        display_dimensions = dimensions

        def compactNames(names):
            names = list(map(lambda x: x[0].lower(), names))
            names = ''.join([d for d in names if d not in 'ct'])
            return names

        names = compactNames(display_dimensions)

        def reorientUsingTransform(source):
            transform = getTransform(source)
            matrix = to_square(transform.matrix)
            if matrix is None:
                matrix = np.eye(4)
            odims = transform.outputDimensions
            onames = transform.outputDimensions.names
            onames = compactNames(onames)
            if all(name in onames for name in names):
                return False
            T0 = ng.CoordinateSpaceTransform(
                matrix=matrix[:-1],
                input_dimensions=getDimensions(source),
                output_dimensions=odims,
            )
            T = neurotransforms[(onames, names)]
            source.transform = compose(T, T0)
            return True

        def reorientUsingDimensions(source):
            idims = getDimensions(source)
            if not idims:
                return False
            inames = compactNames(idims.names)
            if all(name in inames for name in names):
                return True
            T0 = ng.CoordinateSpaceTransform(
                input_dimensions=idims,
                output_dimensions=idims,
            )
            T = neurotransforms[(inames, names)]
            source.transform = compose(T, T0)
            return True

        for layer in state.layers:
            layer = layer.layer
            if isinstance(layer, ng.ImageLayer):
                for source in layer.source:
                    if getTransform(source):
                        reorientUsingTransform(source)
                    else:
                        reorientUsingDimensions(source)

        self.redisplay(display_dimensions, state=state)

    def redisplay(self, display_dimensions, state=None):
        """
        Resets `displayDimensions` to its current value, or to a new value.
        This function does not transform the data accordingly. It only
        sets the state's `displayDimensions`.

        Parameters
        ----------
        dimensions : None or list[str], optional
        """
        if state is None:
            with self.viewer.txn() as state:
                return self.redisplay(display_dimensions, state=state)
        state.displayDimensions = display_dimensions

    @action(needstate=True)
    def transform(self, transform, layer=None, inv=False,
                  *, mov=None, fix=None, _mode='ras2ras', state=None):
        """
        Apply an affine transform

        Parameters
        ----------
        transform : list[float] or np.ndarray or fileobj
            Affine transform (RAS+)
        layer : str or list[str]
            Layer(s) to transform
        inv : bool
            Invert the transform

        Other Parameters
        ----------------
        mov : str
            Moving/Floating image (required by some affine formats)
        fix : str
            Fixed/Reference image (required by some affine formats)
        """
        layer_names = layer or []
        if not isinstance(layer_names, (list, tuple)):
            layer_names = [layer_names]

        if _mode == 'ras2ras':
            display_dimensions = state.display_dimensions
            to_world = self.to_world
            self.display('ras', self._WORLD, state=state)

        # prepare transformation matrix
        transform = ensure_list(transform)
        if len(transform) == 1:
            transform = transform[0]
        if isinstance(transform, str):
            transform = load_affine(transform, moving=mov, fixed=fix)
        transform = np.asarray(transform, dtype='float64')
        if transform.ndim == 1:
            if len(transform) == 12:
                transform = transform.reshape([3, 4])
            elif len(transform) == 16:
                transform = transform.reshape([4, 4])
            else:
                n = int(np.sqrt(1 + 4 * len(transform)).item()) // 2
                transform = transform.reshape([n, n+1])
        elif transform.ndim > 2:
            raise ValueError('Transforms must be matrices')
        transform = to_square(transform)
        if inv:
            transform = np.linalg.inv(transform)
        matrix1 = transform = transform[:-1]

        # make ng transform
        T = ng.CoordinateSpaceTransform(
            matrix=transform,
            input_dimensions=ng.CoordinateSpace(
                names=["right", "anterior", "superior"],
                units=["mm"] * 3,
                scales=[1] * 3
            ),
            output_dimensions=ng.CoordinateSpace(
                names=["right", "anterior", "superior"],
                units=["mm"] * 3,
                scales=[1] * 3
            ),
        )

        def getDimensions(source, _reentrant=False):
            if getattr(source, 'transform', None):
                transform = source.transform
                if transform.inputDimensions:
                    return transform.inputDimensions
            if getattr(source, 'dimensions', None):
                return source.dimensions
            if not _reentrant and not isinstance(source, ng.LocalVolume):
                mysource = RemoteSource.from_filename(source.url)
                return getDimensions(mysource, True)
            return None

        def getTransform(source):
            if getattr(source, 'transform', None):
                return source.transform
            if not isinstance(source, ng.LocalVolume):
                mysource = RemoteSource.from_filename(source.url)
                if getattr(mysource, 'transform', None):
                    return mysource.transform
            return None

        def composeTransform(source):
            transform = getTransform(source)
            idims = getDimensions(source)
            matrix = transform.matrix
            if matrix is None:
                matrix = np.eye(4)
            odims = transform.outputDimensions
            T0 = ng.CoordinateSpaceTransform(
                matrix=matrix,
                input_dimensions=idims,
                output_dimensions=odims,
            )
            if _mode != 'ras2ras':
                T1 = subtransform(ng.CoordinateSpaceTransform(
                    input_dimensions=odims,
                    output_dimensions=odims,
                ), unit='m')
                T1.matrix = matrix1
            else:
                T1 = T
            source.transform = compose(T1, T0)
            return True

        def applyTransform(source):
            idims = getDimensions(source)
            if not idims:
                return False
            T0 = ng.CoordinateSpaceTransform(
                input_dimensions=idims,
                output_dimensions=idims,
            )
            if _mode != 'ras2ras':
                T1 = subtransform(ng.CoordinateSpaceTransform(
                    matrix=matrix1,
                    input_dimensions=idims,
                    output_dimensions=idims,
                ), unit='m')
            else:
                T1 = T
            source.transform = compose(T1, T0)
            return True

        for layer in state.layers:
            if layer_names and layer.name not in layer_names:
                continue
            layer = layer.layer
            if isinstance(layer, ng.ImageLayer):
                for source in layer.source:
                    if getTransform(source):
                        composeTransform(source)
                    else:
                        applyTransform(source)

        if _mode == 'ras2ras':
            if to_world is not None:
                self.transform(to_world.T, _mode='', state=state)
            self.to_world = to_world
            self.display(display_dimensions, state=state)

    @action(needstate=True)
    def channel_mode(self, mode, layer=None, dimension='c', *, state=None):
        """
        Change the mode (local or intensity) of an axis.

        Parameters
        ----------
        mode : {"local", "channel", "spatial"}
            How to interpret this dimension:

            - "local": enables a switch/slider to change which channel
              is displayed (e.g., time).
            - "channel": enables the joint use of all channels to
              display a single intensity/color (e.g., RGB).
            - "spatial": enables the display of this axis in the
              cross-view.
        layer : [list of] str
            Names of layers to process
        dimension : [list of] str
            Names of dimensions to process
        """
        # NOTE
        #  localDimensions is named localDimensions in the neuroglancer
        #   python code
        dimensions = ensure_list(dimension)
        layers = ensure_list(layer or [])

        mode = mode[0].lower()
        if mode not in ('l', 'c', 's'):
            raise ValueError('Unknown channel mode. Should be one of '
                             '{local, channel, spatial}')

        def rename_key(space, src, dst):
            space = space.to_json()
            newspace = {}
            for key, val in space.items():
                if key == src:
                    key = dst
                newspace[key] = val
            return ng.CoordinateSpace(newspace)

        def update_transform(transform, olddim, newdim):
            if newdim.endswith(("'", "^")):
                sdim = newdim[:-1]
            else:
                sdim = newdim
            cdim = sdim + '^'
            transform.outputDimensions = rename_key(
                transform.outputDimensions, olddim, newdim)
            odims = list(transform.outputDimensions.to_json().keys())
            if newdim == cdim:
                # to channel dimension -> half unit shift
                shift = 0.5
            elif olddim == cdim:
                # from channel dimension -> remove half unit shift
                shift = -0.5
            else:
                shift = 0
            if shift:
                if transform.matrix is not None:
                    transform.matrix[odims.index(newdim), -1] += shift
                else:
                    matrix = np.eye(len(odims)+1)[:-1]
                    matrix[odims.index(newdim), -1] = shift
                    transform.matrix = matrix
            return transform

        def create_transform(scale, olddim, newdim):
            if newdim.endswith(("'", "^")):
                sdim = newdim[:-1]
            else:
                sdim = newdim
            cdim = sdim + '^'
            if newdim == cdim:
                # to channel dimension -> half unit shift
                shift = 0.5
            elif olddim == cdim:
                # from channel dimension -> remove half unit shift
                shift = -0.5
            else:
                shift = 0
            transform = ng.CoordinateSpaceTransform(
                matrix=np.asarray([[1, shift]]),
                inputDimensions=ng.CoordinateSpace(
                    names=[olddim],
                    scales=[scale[0]],
                    units=[scale[1]],
                ),
                outputDimensions=ng.CoordinateSpace(
                    names=[newdim],
                    scales=[scale[0]],
                    units=[scale[1]],
                )
            )
            return transform

        for layer in state.layers:
            if layers and layer.name not in layers:
                continue
            layer = layer.layer
            for dimension in dimensions:
                ldim = dimension + "'"
                cdim = dimension + "^"
                sdim = dimension
                localDimensions = layer.localDimensions.to_json()
                if layer.localPosition:
                    localPosition = layer.localPosition.tolist()
                else:
                    localPosition = []
                channelDimensions = layer.channelDimensions.to_json()
                transform = None
                for source in layer.source:
                    if getattr(source, 'transform', {}):
                        transform = source.transform
                        break
                else:
                    source = layer.source[0]

                if mode == 'l':     # LOCAL
                    if ldim in localDimensions:
                        continue
                    was_channel = False
                    if cdim in channelDimensions:
                        was_channel = True
                        scale = channelDimensions.pop(cdim)
                    else:
                        scale = [1, ""]
                    localDimensions[ldim] = scale
                    localPosition = [*(localPosition or []), 0]
                    if transform:
                        update_transform(
                            transform, cdim if was_channel else sdim, ldim)
                    else:
                        source.transform = create_transform(
                            scale, cdim if was_channel else sdim, ldim)

                elif mode == 'c':   # CHANNEL
                    if cdim in channelDimensions:
                        continue
                    was_local = False
                    if ldim in localDimensions:
                        was_local = True
                        for i, key in enumerate(localDimensions.keys()):
                            if key == ldim:
                                break
                        scale = localDimensions.pop(ldim)
                        if i < len(localPosition):
                            localPosition.pop(i)
                    else:
                        scale = [1, ""]
                    channelDimensions[cdim] = scale
                    if transform:
                        update_transform(
                            transform, ldim if was_local else sdim, cdim)
                    else:
                        source.transform = create_transform(
                            scale, ldim if was_local else sdim, cdim)

                elif mode == 's':   # SPATIAL
                    if cdim not in channelDimensions and \
                            ldim not in localDimensions:
                        continue
                    scale = [1, ""]
                    was_channel = False
                    if cdim in channelDimensions:
                        scale = channelDimensions.pop(cdim)
                        was_channel = True
                    if ldim in localDimensions:
                        for i, key in enumerate(localDimensions.keys()):
                            if key == ldim:
                                break
                        scale = localDimensions.pop(ldim)
                        if i < len(localPosition):
                            localPosition.pop(i)
                    if transform:
                        update_transform(
                            transform, cdim if was_channel else ldim, sdim)
                    else:
                        source.transform = create_transform(
                            scale, cdim if was_channel else ldim, sdim)
                    if sdim not in state.dimensions.to_json():
                        dimensions = state.dimensions.to_json()
                        dimensions[sdim] = scale
                        state.dimensions = CoordinateSpace(dimensions)
                layer.localDimensions = CoordinateSpace(localDimensions)
                layer.localPosition = np.asarray(localPosition)
                layer.channelDimensions = CoordinateSpace(channelDimensions)

    @action(needstate=True)
    def position(self, coord, dimensions=None, unit=None,
                 world=False, layer=False, *, state=None, **kwargs):
        """
        Change cursor position

        Parameters
        ----------
        coord : [list of] float
            New position
        dimensions : [list of] str
            Axis of each coordinate. Can be a compact name like 'RAS'.
            Default: Currently displayed axes.
        unit : str
            Units of the coordinates. Default: Unit of current axes.
        world : bool
            Coordinate is in world frame.
            Cannot be used at the same time as `layer`.
        layer : bool or str
            Coordinate is in this layer's canonical frame.
            Cannot be used at the same time as `world`.

        Returns
        -------
        coord : list[float]
            Current cursor position.
        """
        if kwargs.pop('reset', not isinstance(coord, Sequence) and coord == 0):
            return coord([0] * len(state.dimensions))

        if not state.dimensions:
            raise RuntimeError(
                'Dimensions not known. Are you running the app in windowless '
                'mode? If yes, you must open a neuroglancer window to access '
                'or modifiy the cursor position')

        dim = state.dimensions

        # No argument -> print current position
        if not coord:
            string = []
            position = list(map(float, state.position))
            for x, d, s, u in zip(position, dim.names, dim.scales, dim.units):
                x = float(x) * float(s)
                string += [f'{d}: {x:g} {u}']
            print(', '.join(string))
            return position

        # Preproc dimensions
        if isinstance(dimensions, str):
            dimensions = [dimensions]
        dimensions = dimensions or list(map(str, dim.names))
        if len(dimensions) == 1 and len(dimensions[0]) > 1:
            dimensions = compact2full(dimensions[0])
        dimensions = dimensions[:len(coord)]

        # Preproc layer
        if world and layer:
            raise ValueError('Cannot use both --world and --layer')
        if world:
            layer = self._WORLD
        if layer is True:
            layer = state.layers[0].name

        # Change space
        current_dimensions = list(map(str, dim.names))
        change_space = False
        to_world = None
        if layer or any(d not in current_dimensions for d in dimensions):
            to_world = self.to_world
            self.display(dimensions, layer, state=state)
            change_space = True

        # Convert unit
        unitmap = {n: u for u, n in zip(dim.units, dim.names)}
        current_units = [unitmap[d] for d in dimensions]
        coord = si_convert(coord, unit, current_units)

        # Sort coordinate in same order as dim
        coord = {n: x for x, n in zip(coord, dimensions)}
        for x, n in zip(state.position, dim.names):
            coord.setdefault(n, x)
        coord = [coord[n] for n in dim.names]

        # Assign new coord
        state.position = list(coord.values())

        # Change space back
        if change_space:
            self.display(current_dimensions, self._WORLD, state=state)

            def lin2aff(x):
                matrix = np.eye(len(x)+1)
                matrix[:-1, :-1] = x
                return matrix

            if to_world is not None:
                self.transform(lin2aff(to_world.T), _mode='', state=state)
                self.to_world = to_world

        return list(map(float, state.position))

    @action(needstate=True)
    def orient(self, position, dimensions=None, units=None,
               world=False, layer=False, reset=False, *, state=None):
        raise NotImplementedError('Not implemented yet (sorry!)')

    @action(needstate=True)
    def shader(self, shader, layer=None, *, state=None):
        """
        Apply a shader (that is, a colormap or lookup table)

        Parameters
        ----------
        shader : str
            A known shader name (from `ngtools.shaders`), or some
            user-defined shader code, or a LUT file.
        layer : str or list[str], optional
            Apply the shader to these layers. Default: all layers.
        """
        layer_names = ensure_list(layer or [])

        if shader.lower() == 'rgb':
            for layer in state.layers:
                layer_name = layer.name
                if layer_names and layer_name not in layer_names:
                    continue
                layer = layer.layer
                if not layer.channelDimensions.to_json():
                    self.channel_mode('channel', layer=layer_name, state=state)

        if hasattr(shaders, shader):
            shader = getattr(shaders, shader)
        elif hasattr(colormaps, shader):
            shader = shaders.colormap(shader)
        elif 'main()' not in shader:
            # assume it's a path
            shader = shaders.lut(shader)
        for layer in state.layers:
            if layer_names and layer.name not in layer_names:
                continue
            layer = layer.layer
            layer.shader = shader

    @action(needstate=True)
    def state(self, load=None, save=None, url=False, print=True,
              *, state=None):
        """
        Print or save or load the viewer's JSON state

        Parameters
        ----------
        load : str
            Load state from JSON file, or JSON string (or URL if `url=True`).
        save : str
            Save state to JSON file
        url : bool
            Print/load a JSON URL rather than a JSON object
        print : bool
            Print the JSON object or URL

        Returns
        -------
        state : dict
            JSON state
        """
        ngstate = state
        if load:
            if os.path.exists(load) or load.startswith(remote_protocols()):
                with open(load) as f:
                    state = json.load(f)
            elif url:
                if '://' in url:
                    url = urlparse(url).fragment
                    if url[0] != '!':
                        raise ValueError('Neuroglancer URL not recognized')
                    url = url[1:]
                state = json.loads(urlunquote(url))
            else:
                state = json.loads(url)
            ngstate.set_state(state)
        else:
            state = ngstate.to_json()

        if save:
            with open(save, 'wb') as f:
                json.dump(state, f, indent=4)

        if print:
            if url:
                state = urlquote(json.dumps(state))
                state = 'https://neuroglancer-demo.appspot.com/#!' + state
                _print(state)
            else:
                _print(json.dumps(state, indent=4))
        return state

    @action(needstate=True)
    def layout(self, layout=None, stack=None, layer=None, *,
               flex=1, append=None, insert=None, remove=None, state=None):
        """
        Change layout.

        Parameters
        ----------
        layout : [list of] {"xy", "yz", "xz", "xy-3d", "yz-3d", "xz-3d", "4panel", "3d"}
            Layout(s) to set or insert. If list, `stack` must be set.
        stack : {"row", "column"}, optional
            Insert a stack of layouts.
        layer : [list of] str
            Set of layers to include in the layout.
            By default, all layers are included (even future ones).

        Other Parameters
        ----------------
        flex : float, default=1
            ???
        append : bool or [list of] int or str
            Append the layout to an existing stack.
            If an integer or list of integer, they are used to navigate
            through the nested stacks of layouts.
            Only one of append or insert can be used.
        insert : int or [list of] int or str
            Insert the layout into an existing stack.
            If an integer or list of integer, they are used to navigate
            through the nested stacks of layouts.
            Only one of append or insert can be used.
        remove : int or [list of] int or str
            Remove the layout in an existing stack.
            If an integer or list of integer, they are used to navigate
            through the nested stacks of layouts.
            If `remove` is used, `layout` should be `None`.

        Returns
        -------
        layout : object
            Current JSON layout
        """  # noqa: E501
        if not layout and (remove is None):
            print(state.layout)
            return state.layout

        layout = ensure_list(layout or [])

        layer = ensure_list(layer or [])
        if (len(layout) > 1 or stack) and not layer:
            layer = [_.name for _ in state.layers]

        if layer:
            layout = [ng.LayerGroupViewer(
                layers=layer,
                layout=L,
                flex=flex,
            ) for L in layout]

        if len(layout) > 1 and not stack:
            stack = 'row'
        if not stack and len(layout) == 1:
            layout = layout[0]
        if stack:
            layout = ng.StackLayout(
                type=stack,
                children=layout,
                flex=flex,
            )

        indices = []
        do_append = append is not None
        if do_append:
            indices = ensure_list(append or [])
            append = do_append

        if insert:
            indices = ensure_list(insert or [])
            insert = indices.pop(-1)
        else:
            insert = False

        if remove:
            indices = ensure_list(remove or [])
            remove = indices.pop(-1)
        else:
            remove = False

        if bool(append) + bool(insert) + bool(remove) > 1:
            raise ValueError('Cannot use both append and insert')
        if layout and remove:
            raise ValueError('Do not set `layout` and `remove`')

        if append or (insert is not False) or (remove is not False):
            parent = state.layout
            while indices:
                parent = layout.children[indices.pop(0)]
            if layout and not isinstance(layout, ng.LayerGroupViewer):
                if not layer:
                    if len(parent.children):
                        layer = [L for L in parent.children[-1].layers]
                    else:
                        layer = [_.name for _ in state.layers]
                layout = ng.LayerGroupViewer(
                    layers=layer,
                    layout=layout,
                    flex=flex,
                )
            if append:
                parent.children.append(layout)
            elif insert:
                parent.children.insert(insert, layout)
            elif remove:
                del parent.children[remove]
        else:
            state.layout = layout
        return state.layout

    @action(needstate=True)
    def zorder(self, layer, steps=None, *, state=None, **kwargs):
        """
        Move or reorder layers.

        In neuroglancer, layers are listed in the order they are loaded,
        with the latest layer appearing on top of the other ones in the
        scene. Counter-intuitively, the latest/topmost layer is listed
        at the bottom of the layer list, while the earliest/bottommost
        layer is listed at the top of the layer list. In this function,
        we list layers in their expected z-order, top to bottom.

        Parameters
        ----------
        layer : str or list[str]
            Name of layers to move.
            If `steps=None`, layers are given the provided order.
            Else, the selected layers are moved by `steps` steps.
        steps : int
            Number of steps to take. Positive steps move layers towards
            the top and negative steps move layers towards the bottom.
        """
        up = kwargs.get('up', 0)
        down = kwargs.get('down', 0)
        if up or down:
            if steps is None:
                steps = 0
            steps += up
            steps -= down

        names = ensure_list(layer)
        if steps is None:
            # permutation
            names += list(reversed([layer.name for layer in state.layers
                                    if layer.name not in names]))
            layers = {layer.name: layer.layer for layer in state.layers}
            for name in names:
                del state.layers[name]
            for name in reversed(names):
                state.layers[name] = layers[name]
        elif steps == 0:
            return
        else:
            # move up/down
            layers = {layer.name: layer.layer for layer in state.layers}
            indices = {name: n for n, name in enumerate(layers)}
            for name in layers.keys():
                indices[name] += steps * (1 if name in names else -1)
            layers = {
                name: layers[name]
                for name in sorted(layers.keys(), key=lambda x: indices[x])
            }
            for name in layers.keys():
                del state.layers[name]
            for name, layer in layers.items():
                state.layers[name] = layer

    @action
    def help(self, action=None):
        """
        Display help

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
    def exit(self):
        """Exit gracefully"""
        if hasattr(self, 'fileserver'):
            del self.fileserver
        sys.exit()

    # ==================================================================
    #
    #                            CALLBACKS
    #
    # ==================================================================

    def on_state_change(self):
        old_state = self.saved_state
        with self.viewer.txn() as state:
            self.saved_state = state
            for layer in state.layers:
                name = layer.name
                layer = layer.layer
                if isinstance(layer.source, ng.SkeletonSource):
                    if (layer.shaderControls.nbtracts
                            != old_state.shaderControls.nbtracts):
                        self.update_tracts_nb(name)

    def update_tracts_nb(self, name):
        with self.viewer.txn() as state:
            layer = state.layers[name]
            print(type(layer))
            layer.source.max_tracts = layer.shaderControls.nbtracts
            layer.source._filter()
            layer.invalidate()

    # ==================================================================
    #
    #                            FILEUTILS
    #
    # ==================================================================

    def ensure_url(self, filename):
        DANDIHINTS = (
            'dandi://',
            'DANDI:',
            'https://identifiers.org/DANDI:',
            'https://dandiarchive.org/',
            'https://gui.dandiarchive.org/',
            'https://api.dandiarchive.org/',
        )
        if filename.startswith(DANDIHINTS):
            return RemoteDandiFileSystem().s3_url(filename)

        if not filename.startswith(remote_protocols()):
            filename = os.path.abspath(filename)
            while filename.endswith('/'):
                filename = filename[:-1]
            prefix = f'http://{self.fileserver.ip}:{self.fileserver.port}/'
            filename = prefix + filename
        return filename

    # Protocols that describe the type of data contained in the file
    LAYERTYPES = [
        'volume',           # Raster data (image or volume)
        'labels',           # Integer raster data, interpreted as labels
        'surface',          # Triangular mesh
        'mesh',             # Other types of mesh ???
        'tracts',           # Set of piecewise curves
        'roi',              # Region of interest ???
        'points',           # Pointcloud
        'transform',        # Spatial transform
        'affine',           # Affine transform
    ]

    # Native neuroglancer formats
    NG_FORMATS = [
        'boss',             # bossDB: Block & Object storage system
        'brainmap',         # Google Brain Maps
        'deepzoom',         # Deep Zoom file-backed data source
        'dvid',             # DVID
        'graphene',         # Graphene Zoom file-backed data source
        'local',            # Local in-memory
        'n5',               # N5 data source
        'nggraph',          # nggraph data source
        'nifti',            # Single NIfTI file
        'obj',              # Wavefront OBJ mesh file
        'precomputed',      # Precomputed file-backed data source
        'render',           # Render
        'vtk',              # VTK mesh file
        'zarr',             # Zarr data source
        'zarr2',            # Zarr v2 data source
        'zarr3',            # Zarr v3 data source
    ]

    # Extra local formats (if not specified, try to guess from file)
    EXTRA_FORMATS = [
        'mgh',              # Freesurfer volume format
        'mgz',              # Freesurfer volume format (compressed)
        'trk',              # Freesurfer streamlines
        'lta',              # Freesurfer affine transform
        'surf',             # Freesurfer surfaces
        'annot',            # Freesurfer surface annotation
        'tck',              # MRtrix streamlines
        'mif',              # MRtrix volume format
        'gii',              # Gifti
        'tiff',             # Tiff volume format
        'niftyreg',         # NiftyReg affine transform
    ]

    def parse_filename(self, filename):
        """
        Parse a filename that may contain protocol hints

        Parameters
        ----------
        filename : str
            A filename, that may be prepended by:

            1) A layer type protocol, which indicates the kind of object
               that the file contains.
               Examples: `volume://`, `labels://`, `tracts://`.
            2) A format protocol, which indicates the exact file format.
               Examples: `nifti://`, `zarr://`, `mgh://`.
            3) An access protocol, which indices the protocol used to
               access the files.
               Examples: `https://`, `s3://`, `dandi://`.

            All of these protocols are optional. If absent, a guess is
            made using the file extension.

        Returns
        -------
        layertype : str
            Data type protocol. Can be None.
        format : str
            File format protocol. Can be None.
        filename : str
            File path, eventually prepended with an access protocol.
        """
        layertype = None
        for dt in self.LAYERTYPES:
            if filename.startswith(dt + '://'):
                layertype = dt
                filename = filename[len(dt)+3:]
                break

        format = None
        for fmt in self.NG_FORMATS + self.EXTRA_FORMATS:
            if filename.startswith(fmt + '://'):
                format = fmt
                filename = filename[len(fmt)+3:]
                break

        if format is None:
            if filename.endswith('.mgh'):
                format = 'mgh'
            elif filename.endswith('.mgz'):
                format = 'mgz'
            elif filename.endswith(('.nii', '.nii.gz')):
                format = 'nifti'
            elif filename.endswith('.trk'):
                format = 'trk'
            elif filename.endswith('.tck'):
                format = 'tck'
            elif filename.endswith('.lta'):
                format = 'lta'
            elif filename.endswith('.mif'):
                format = 'mif'
            elif filename.endswith(('.tiff', '.tif')):
                format = 'tiff'
            elif filename.endswith('.gii'):
                format = 'gii'
            elif filename.endswith(('.zarr', '.zarr/')):
                format = 'zarr'
            elif filename.endswith('.vtk'):
                format = 'vtk'
            elif filename.endswith('.obj'):
                format = 'obj'
            elif filename.endswith(('.n5', '.n5/')):
                format = 'n5'

        if layertype is None:
            if format in ('tck', 'trk'):
                layertype = 'tracts'
            elif format in ('vtk', 'obj'):
                layertype = 'mesh'
            elif format in ('lta',):
                layertype = 'affine'

        return layertype, format, filename


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
