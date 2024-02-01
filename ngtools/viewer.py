import sys
import re
import os.path
import argparse
import shlex                    # parse user input as if a shell commandline
import readline                 # autocomplete/history in user input
import atexit                   # do stuff whe exiting (save history...)
import numpy as np
import neuroglancer as ng
from neuroglancer.server import global_server_args
from .fileserver import LocalFileServerInBackground
from .volume import LocalSource, RemoteSource
from .spaces import neurospaces, neurotransforms, to_square, letter2full


class LocalNeuroglancer:
    """
    A local instance of neuroglancer that can launch its own local fileserver.

    It also comes with a shell-like interfact that allows loading,
    unloading, applying transforms, etc.
    """

    def __init__(self, port=9321, token=1, fileserver=True):
        """
        Parameters
        ----------
        port : int
            Port to use
        token : str
            Unique id for the instance
        filserver : LocalFileServerInBackground or bool
            A local file server
        """
        if fileserver is True:
            fileserver = LocalFileServerInBackground()
        self.fileserver = fileserver
        if self.fileserver:
            self.fileserver.start_and_serve_forever()
        self.port = port
        global_server_args['bind_port'] = str(port)
        self.viewer = ng.Viewer(token=str(token))
        self.parser = self.make_parser()
        self.comp = UserInputCompleter([
            # Any chance we can get these from self.parser?
            'help',
            'load',
            'unload',
            'transform',
            'shader',
            'display',
            'exit',
        ])
        self.display_dimensions = ['x', 'y', 'z']

    def await_input(self):
        try:
            while True:
                args = input('[ng] ')
                try:
                    args = self.parser.parse_args(shlex.split(args))
                    try:
                        args.func(args)
                        self.display()
                    except KeyboardInterrupt as e:
                        raise e
                    except Exception as e:
                        print("(EXEC ERROR) ", e, file=sys.stderr)
                        raise e
                except KeyboardInterrupt as e:
                    raise e
                except Exception as e:
                    print("(PARSE ERROR)", e, file=sys.stderr)
                    raise e
        except KeyboardInterrupt:
            print('\n(EXIT)')

    def make_parser(self):
        mainparser = NoExitArgParse('')
        parsers = mainparser.add_subparsers()

        help = parsers.add_parser('help', help='Display help')
        help.set_defaults(func=self.help)
        help.add_argument(
            dest='action', nargs='?', help='Command for which to display help')

        load = parsers.add_parser('load', help='Load a file')
        load.set_defaults(func=self.load)
        load.add_argument(
            dest='filenames', nargs='+', help='Filename(s) with protocols')
        load.add_argument(
            '--name', help='A name for the image layer')
        load.add_argument(
            '--transform', help='Apply a transform')

        unload = parsers.add_parser('unload', help='Unload a file')
        unload.set_defaults(func=self.unload)
        load.add_argument(
            dest='names', nargs='+', help='Name(s) of layer(s) to unload')

        transform = parsers.add_parser('transform', help='Apply a transform')
        transform.set_defaults(func=self.transform)
        transform.add_argument(
            dest='transform', nargs='+',
            help='Path to transform file or flattened transformation '
                 'matrix (row major)')
        transform.add_argument(
            '--name', nargs='+', help='Name of layer(s) to transform')

        shader = parsers.add_parser('shader', help='Apply a shader')
        shader.set_defaults(func=self.shader)

        display = parsers.add_parser('display', help='Dimensions to display')
        display.set_defaults(func=self.display)
        display.add_argument(
            dest='dimensions', nargs='*', help='Dimensions to display')

        exit = parsers.add_parser('exit', help='Exit neuroglancer')
        exit.set_defaults(func=self.exit)
        return mainparser

    def display(self, args=None):
        if args and args.dimensions:
            self.display_dimensions = args.dimensions
        names = list(map(lambda x: x[0].lower(), self.display_dimensions))
        names = ''.join([d for d in names if d not in 'ct'])
        with self.viewer.txn() as state:
            for layer in state.layers:
                layer = layer.layer
                if isinstance(layer, ng.ImageLayer):
                    for source in layer.source:
                        print(source)
                        transform = source.transform
                        if not transform and not isinstance(source, ng.LocalVolume):
                            mysource = RemoteSource.from_filename(source.url)
                            transform = getattr(mysource, 'transform')
                        if transform:
                            print('hastransform', transform)
                            matrix = transform.matrix
                            if matrix is None:
                                matrix = np.eye(4)
                            idims = transform.inputDimensions
                            if not idims and not isinstance(source, ng.LocalVolume):
                                mysource = RemoteSource.from_filename(source.url)
                                idims = getattr(mysource, 'dimensions')
                            odims = transform.outputDimensions
                            onames = transform.outputDimensions.names
                            onames = list(map(lambda x: x[0].lower(), onames))
                            onames = ''.join([d for d in onames if d not in 'ct'])
                            if all(name in onames for name in names):
                                continue
                            T0 = to_square(matrix)
                            T = neurotransforms[(neurospaces[onames],
                                                neurospaces[names])].matrix
                            T = to_square(T)
                            T = T @ T0
                            T = ng.CoordinateSpaceTransform(
                                matrix=T[:3, :4],
                                inputDimensions=idims,
                                outputDimensions=ng.CoordinateSpace(
                                    names=[letter2full.get(x, x) for x in names],
                                    units=odims.units,
                                    scales=np.abs(T[:3, :3] @ odims.scales)
                                )
                            )
                            source.transform = T

                        elif isinstance(source, ng.LocalVolume):
                            print('local')
                            idims = getattr(source, 'dimensions', None)
                            if not idims:
                                continue
                            inames = idims.names
                            inames = list(map(lambda x: x[0].lower(), inames))
                            inames = ''.join([d for d in inames if d not in 'ct'])
                            if all(name in inames for name in names):
                                continue
                            T = neurotransforms[(neurospaces[inames],
                                                neurospaces[names])].matrix
                            T = ng.CoordinateSpaceTransform(
                                matrix=T[:3, :4],
                                inputDimensions=idims,
                                outputDimensions=ng.CoordinateSpace(
                                    names=[letter2full.get(x, x) for x in names],
                                    units=idims.units,
                                    scales=np.abs(T[:3, :3] @ idims.scales)
                                )
                            )
                            source.transform = T

                        else:
                            print('remote')
                            mysource = RemoteSource.from_filename(source.url)
                            odims = getattr(mysource, 'outputDimensions', None)
                            print(odims)
                            if not odims:
                                continue
                            onames = odims.names
                            onames = list(map(lambda x: x[0].lower(), onames))
                            onames = ''.join([d for d in onames if d not in 'ct'])
                            if all(name in onames for name in names):
                                continue
                            T = neurotransforms[(neurospaces[onames],
                                                neurospaces[names])].matrix
                            T = ng.CoordinateSpaceTransform(
                                matrix=T[:3, :4],
                                inputDimensions=odims,
                                outputDimensions=ng.CoordinateSpace(
                                    names=[letter2full.get(x, x) for x in names],
                                    units=odims.units,
                                    scales=np.abs(T[:3, :3] @ odims.scales)
                                )
                            )
                            source.transform = T
                            print(T)

        with self.viewer.txn() as state:
            state.displayDimensions = self.display_dimensions

    def load(self, args):
        if args.name and len(args.filenames or []) > 1:
            raise ValueError('Cannot give a single name to multiple layers. '
                             'Use separate `load` calls.')
        for filename in args.filenames or []:
            layertype, ngtype, format, filename = self.parse_filename(filename)
            name = args.name or os.path.basename(filename)

            if ngtype:
                # Remote source
                filename = self.ensure_url(filename)
                source = RemoteSource.from_filename(
                    ngtype + '://' + filename, layer_type=layertype)
                controls = None
                if hasattr(source, 'quantiles'):
                    q = source.quantiles([0.01, 0.99])
                    controls = {"normalized": {"range": q}}
                layer = ng.ImageLayer(
                    source=source,
                    shaderControls=controls,
                )

            else:
                # Local source
                if format:
                    filename = format + '://' + filename
                source = LocalSource.from_filename(
                    filename, layer_type=layertype)
                controls = None
                if hasattr(source, 'quantiles'):
                    q = source.quantiles([0.01, 0.99])
                    controls = {"normalized": {"range": q}}
                layer = ng.ImageLayer(
                    source=source,
                    shaderControls=controls,
                )

            with self.viewer.txn() as state:
                state.layers.append(name=name, layer=layer)

            if args.transform:
                xargs = self.parser.parse_args([
                    'transform', args.transform, '--name', name])
                xargs.func(xargs)

    def unload(self, args):
        for name in args.names or []:
            with self.viewer.txn() as state:
                del state[name]

    def transform(self, args):
        pass

    def shader(self, args):
        pass

    def help(self, args):
        if args.action:
            args = self.parser.parse_args([args.action, '--help'])
        else:
            args = self.parser.parse_args(['--help'])

    def exit(self, args):
        del self.fileserver
        sys.exit()

    def ensure_url(self, filename):
        remote_protocols = [p + '://' for p in self.REMOTE_PROTOCOLS]
        if not filename.startswith(tuple(remote_protocols)):
            if filename.startswith('/'):
                filename = 'root://' + filename
            prefix = f'http://{self.fileserver.ip}:{self.fileserver.port}/'
            filename = prefix + filename
        return filename

    # Protocols that describe the type of data contained in the file
    LAYERTYPES = [
        'volume',           # Raster data (image or volume)
        'labels',           # Integer raster data, interpreted as labels
        'surface',          # Triangular mesh
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

    REMOTE_PROTOCOLS = [
        'http',
        'https',
        's3',
    ]

    def parse_filename(self, filename):

        datatype = None
        for dt in self.LAYERTYPES:
            if filename.startswith(dt + '://'):
                datatype = dt
                filename = filename[len(dt)+3:]
                break

        ng_protocol = None
        for ngp in self.NG_FORMATS:
            if filename.startswith(ngp + '://'):
                ng_protocol = ngp
                filename = filename[len(ngp)+3:]
                break

        format = None
        for fmt in self.EXTRA_FORMATS:
            if filename.startswith(fmt + '://'):
                format = ngp
                filename = filename[len(fmt)+3:]
                break

        if format is None:
            if ng_protocol:
                format = ng_protocol
            elif filename.endswith('.mgh'):
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

        if not ng_protocol and format == 'nifti':
            ng_protocol = 'nifti'

        return datatype, ng_protocol, format, filename


class NoExitArgParse(argparse.ArgumentParser):
    def exit(self, status=0, message=None):
        pass

    def error(self, message):
        pass


class UserInputCompleter:

    RE_SPACE = re.compile(r'.*\s+$', re.M)
    HISTORY = os.path.expanduser('~/.neuroglancer_history')

    def __init__(self, commands=tuple(), ):
        self.commands = commands
        readline.set_completer_delims(' \t\n;')
        readline.parse_and_bind('tab: complete')
        readline.set_completer(self.complete)
        if not os.path.exists(self.HISTORY):
            with open(self.HISTORY, 'wt'):
                pass
        readline.read_history_file(self.HISTORY)
        atexit.register(self._save_history, self.HISTORY)

    def _save_history(self, histfile):
        readline.set_history_length(1000)
        readline.write_history_file(histfile)

    def _listdir(self, root):
        "List directory 'root' appending the path separator to subdirs."
        res = []
        for name in os.listdir(root):
            path = os.path.join(root, name)
            if os.path.isdir(path):
                name += os.sep
            res.append(name)
        return res

    def _complete_path(self, path=None):
        "Perform completion of filesystem path."
        if not path:
            return self._listdir('.')
        dirname, rest = os.path.split(path)
        tmp = dirname if dirname else '.'
        res = [os.path.join(dirname, p)
               for p in self._listdir(tmp) if p.startswith(rest)]
        # more than one match, or single match which does not exist (typo)
        if len(res) > 1 or not os.path.exists(path):
            return res
        # resolved to a single directory, so return list of files below it
        if os.path.isdir(path):
            return [os.path.join(path, p) for p in self._listdir(path)]
        # exact file match terminates this completion
        return [path + ' ']

    def complete_default(self, args):
        if not args:
            return self._complete_path('.')
        # treat the last arg as a path and complete it
        return self._complete_path(os.path.expanduser(args[-1]))

    def complete(self, text, state):
        "Generic readline completion entry point."
        buffer = readline.get_line_buffer()
        line = readline.get_line_buffer().split()
        # show all commands
        if not line:
            return [c + ' ' for c in self.commands][state]
        # account for last argument ending in a space
        if self.RE_SPACE.match(buffer):
            line.append('')
        # resolve command to the implementation function
        cmd = line[0].strip()
        if cmd in self.commands:
            impl = getattr(self, 'complete_%s' % cmd, self.complete_default)
            args = line[1:]
            if args:
                return (impl(args) + [None])[state]
            return [cmd + ' '][state]
        results = [c + ' ' for c in self.commands if c.startswith(cmd)]
        results += [None]
        return results[state]
