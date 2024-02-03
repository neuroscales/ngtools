import os
import sys
import argparse
import webbrowser
from .fileserver import LocalFileServerInBackground
from .viewer import LocalNeuroglancer


def cli(args=None):
    """
    A commandline launcher for local Neuroglancer instances
    """
    args = args or sys.argv[1:]

    parser = argparse.ArgumentParser('Run a local neuroglancer')
    parser.add_argument('--cwd', default=os.getcwd(), help='Working directory')
    parser.add_argument('--ip', default='127.0.0.1')
    parser.add_argument('--port-fileserver', type=int, default=9123)
    parser.add_argument('--port-viewer', type=int, default=9321)
    parser.add_argument('--token', type=str, default='1')
    parser.add_argument('--no-window', action='store_true', default=False)
    parser.add_argument(nargs='*', dest='filenames', help='Files to load')
    args = parser.parse_args(args)

    # instantiate file server
    if args.port_fileserver != 0:
        fileserver = LocalFileServerInBackground(
            cwd=args.cwd, port=args.port_fileserver, ip=args.ip)
        print('fileserver:  ', f'http://{fileserver.ip}:{fileserver.port}/')
    else:
        fileserver = False

    # instantiate neuroglancer
    neuroglancer = LocalNeuroglancer(
        port=args.port_viewer, token=args.token, fileserver=fileserver)
    print('neuroglancer:', neuroglancer.viewer.get_viewer_url())

    if not args.no_window:
        webbrowser.open(neuroglancer.viewer.get_viewer_url())

    # load files
    for filename in args.filenames or []:
        args = neuroglancer.parser.parse_args(['load', filename])
        args.func(args)
    neuroglancer.display()
    neuroglancer.await_input()
