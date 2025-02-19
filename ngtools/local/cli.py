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
    parser.add_argument('--port', type=int, default=9321,
                        help="viewer port", metavar="PORT")
    parser.add_argument('--port-fileserver', type=int, default=9123,
                        help="fileserver port", metavar="PORT")
    parser.add_argument('--no-window', action='store_true', default=False,
                        help="do not open neuroglancer window")
    parser.add_argument('--no-fileserver', action='store_true', default=False,
                        help="do not run a local fileserver")
    parser.add_argument('--debug', action='store_true', default=False,
                        help="run in debug mode")
    parser.add_argument('--log-level', choices=log_choices, default=None,
                        help="logging level")
    parser.add_argument('--stdin', help="input stream (default: stdin)")
    parser.add_argument('--stdout', help="output stream (default: stdout)")
    parser.add_argument('--stderr', help="error stream (default: stderr)")
    parser.add_argument(nargs='*', dest='filenames', help='Files to load')
    args = parser.parse_args(args)

    if args.ip == 'auto':
        args.ip = ''

    # -------
    # Logging
    # -------

    if args.log_level is None:
        args.log_level = "debug" if args.debug else "none"
    args.log_level = log_map[args.log_level[0].lower()]

    logging.basicConfig()
    logging.getLogger().setLevel(1000)
    logging.getLogger("dandi").setLevel(1000)
    logging.getLogger("ngtools").setLevel(args.log_level)

    # ----------
    # Fileserver
    # ----------

    fileserver = False
    if not args.no_fileserver:
        fileserver = LocalFileServer(args.port_fileserver, args.ip)

    # ------
    # Viewer
    # ------

    neuroglancer = LocalNeuroglancer(
        port=args.port,
        ip=args.ip,
        token=args.token,
        fileserver=fileserver,
        debug=args.debug,
        stdin=args.stdin or sys.stdin,
        stdout=args.stdout or sys.stdout,
        stderr=args.stderr or sys.stderr,
    )
    url = neuroglancer.get_viewer_url()

    logo = bformat.bold(bformat.fg.blue(LOGO[1:-1]))
    neuroglancer.console.stdio.print(logo)
    neuroglancer.console.stdio.print('neuroglancer:', url)
    if neuroglancer.viewer:
        fileserver_url = neuroglancer.get_fileserver_url()
        neuroglancer.console.stdio.print('fileserver:  ', fileserver_url)

    if not args.no_window:
        webbrowser.open(url)

    # ----------
    # Load files
    # ----------

    import time
    time.sleep(1)
    for filename in (args.filenames or []):
        neuroglancer.load(filename)

    neuroglancer.await_input()

    # with neuroglancer.console.stdio.stdin as f:
    #     f.writelines([f'load "{filename}"' for filename in args.filenames])
