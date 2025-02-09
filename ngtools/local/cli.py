"""Entrypoint to a local neuroglancer instance and associated shell."""
# stdlib
import argparse
import logging
import sys
import webbrowser
from functools import partial

# internals
from ngtools.local.fileserver import LocalFileServer
from ngtools.local.termcolors import bformat
from ngtools.local.viewer import LocalNeuroglancer

LOGO = r"""
             _              _
 _ __   __ _| |_ ___   ___ | |___
| '_ \ / _` | __/ _ \ / _ \| / __|
| | | | (_| | || (_) | (_) | \__ \
|_| |_|\__, |\__\___/ \___/|_|___/
       |___/
"""


def main(args: list[str] | None = None) -> None:
    """Commandline launcher for local Neuroglancer instances."""
    args = args or sys.argv[1:]

    help_fmt = partial(argparse.HelpFormatter, max_help_position=32)

    log_choices = ("any", "debug", "info", "warning", "error", "none")
    log_map = {
        "a": 0,
        "d": logging.DEBUG,
        "i": logging.INFO,
        "w": logging.WARNING,
        "e": logging.ERROR,
        "n": 1024
    }

    parser = argparse.ArgumentParser('Run a local neuroglancer',
                                     formatter_class=help_fmt)
    parser.add_argument('--token', type=str, default='1',
                        help="neuroglancer unique token")
    parser.add_argument('--ip', default='127.0.0.1',
                        help="local IP")
    parser.add_argument('--port-viewer', type=int, default=9321,
                        help="neuroglancer port", metavar="PORT")
    parser.add_argument('--port-fileserver', type=int, default=9123,
                        help="fileserver port", metavar="PORT")
    parser.add_argument('--no-fileserver', action='store_true', default=False,
                        help="do not run a local fileserver")
    parser.add_argument('--no-window', action='store_true', default=False,
                        help="do not open neuroglancer window")
    parser.add_argument('--debug', action='store_true', default=False,
                        help="run in debug mode")
    parser.add_argument('--log-level', choices=log_choices, default="none",
                        help="logging level")
    parser.add_argument(nargs='*', dest='filenames', help='Files to load')
    args = parser.parse_args(args)

    logging.basicConfig()
    logging.getLogger().setLevel(1000)
    logging.getLogger("ngtools").setLevel(log_map[args.log_level[0].lower()])

    if args.ip == 'auto':
        args.ip = ''

    # instantiate file server
    if not args.no_fileserver:
        fileserver = LocalFileServer(port=args.port_fileserver, ip=args.ip)
    else:
        fileserver = False

    # instantiate neuroglancer
    neuroglancer = LocalNeuroglancer(
        port=args.port_viewer, ip=args.ip, token=args.token,
        fileserver=fileserver, debug=args.debug)

    logo = bformat.bold(bformat.fg.blue(LOGO[1:-1]))
    neuroglancer.console.stdio.print(logo)

    if neuroglancer.fileserver:
        print('fileserver:  ', f'http://{fileserver.ip}:{fileserver.port}/')
    print('neuroglancer:', neuroglancer.viewer.get_viewer_url())

    if not args.no_window:
        webbrowser.open(neuroglancer.viewer.get_viewer_url())

    # load files
    for filename in args.filenames or []:
        args = neuroglancer.console.parse_args(['load', filename])
        args.func(args)
    neuroglancer.display()
    neuroglancer.await_input()
