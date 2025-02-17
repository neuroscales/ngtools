
"""
An ArgumentParser that can be used as a commandline app.
It handles history and autocomplete.

We use it to implement a shell-like UI for neuroglancer.
"""
# stdlib
import argparse
import logging
import os
import re
import readline  # autocomplete/history in user input
import shlex  # parse user input as if a shell commandline
import sys
import traceback
from functools import partial
from gettext import gettext  # to fix usage string

# internals
from ngtools.local.iostream import StandardIO
from ngtools.local.termcolors import bformat, iformat

LOG = logging.getLogger(__name__)


class Console(argparse.ArgumentParser):
    """
    An ArgumentParser that can be used as a commandline app.
    It handles history and autocomplete.
    """

    DEFAULT_HISTFILE = '~/.neuroglancer_history'
    DEFAULT_HISTSIZE = 1000

    def __init__(self, *args: tuple, **kwargs: dict) -> None:
        """
        Parameters
        ----------
        prog : str, default=os.path.basename(sys.argv[0])
            The name of the program.
        usage : str, optional
            A usage message. Default: auto-generated from arguments.
        description : str, default=""
            A description of what the program does.
        epilog : str, default=""
            Text following the argument descriptions
        prefix_chars : str, default="-"
            Characters that prefix optional arguments
        fromfile_prefix_chars: str | None, default=None
            Characters that prefix files containing additional arguments
        argument_default : object, default=None
            The default value for all arguments
        conflict_handler : {"error", "resolve"}, default="error"
            String indicating how to handle conflicts
        add_help : bool, default=True
            Add a -h/-help option
        allow_abbrev: bool, default=True
            Allow long options to be abbreviated unambiguously

        Other Parameters
        ----------------
        debug : bool, default=False
            Print traceback on error.
        history_file : str, default="~/.neuroglancer_history"
            Path to history file.
        history_size : int, default=1000
            History size.
        exit_on_error : bool, default=False
            Exit (non-gracefully) on error
        exit_on_help : bool, default=False
            Exist on help.
        stdin : TextIO | str, default=sys.stdin
            Input stream.
        stdout : TextIO | str, default=sys.stdout
            Output stream.
        stderr : TextIO | str, default=sys.stderr
            Error stream.
        max_choices : int | None, default=None
            Maximum numer of choices to show in usage string.
        """
        self._debug = kwargs.pop('debug', False)
        max_choices = kwargs.pop('max_choices', None)
        # Commandline behavior
        self.history_file = kwargs.pop('history_file', self.DEFAULT_HISTFILE)
        if self.history_file:
            self.history_file = os.path.expanduser(self.history_file)
        self.history_size = kwargs.pop('history_size', self.DEFAULT_HISTSIZE)
        # Exit behavior
        exit_on_error = kwargs.pop('exit_on_error', False)
        exit_on_help = kwargs.pop('exit_on_help', False)
        kwargs["exit_on_error"] = False
        # Input/output
        stdio = kwargs.pop("stdio", None)
        if stdio is None:
            stdin = kwargs.pop("stdin", sys.stdin)
            stdout = kwargs.pop("stdout", sys.stdout)
            stderr = kwargs.pop("stderr", sys.stderr)
            level = kwargs.pop("level", "info")
            if self._debug:
                level = "debug"
            stdio = StandardIO(
                stdin=stdin, stdout=stdout, stderr=stderr,
                level=level, logger=LOG
            )
        self.stdio = stdio
        # ArgumentParser.__init__
        kwargs["formatter_class"] = partial(
            MainHelpFormatter, max_choices=max_choices
        )
        super().__init__(*args, **kwargs)
        # Overwrite ArgumentParser attributes
        self.exit_on_error = exit_on_error
        self.exit_on_help = exit_on_help

    class InterruptParsing(Exception):
        """Exception raised when parsing gets interrupted."""

    def exit(self, status: int = 0, message: str | None = None) -> None:
        """Overload ArgumentParser.exit to disable it."""
        # We overload (and do not call) ArgumentParser.exit
        # This is called (e.g.) after running a help command.
        if message:
            self.print(message)
        raise self.InterruptParsing(status)

    def error(self, *args, **kwargs) -> None:
        """Error -- Print something to the stderr in red."""
        # We overload (and do not call) ArgumentParser.error
        if self.exit_on_error:
            raise SystemExit(*args)
        else:
            self.stdio.error(*args, **kwargs)

    def print(self, *args, **kwargs) -> None:  # noqa: D102
        self.stdio.print(*args, **kwargs)

    def debug(self, *args, **kwargs) -> None:  # noqa: D102
        self.stdio.debug(*args, **kwargs)

    def info(self, *args, **kwargs) -> None:  # noqa: D102
        self.stdio.info(*args, **kwargs)

    def warning(self, *args, **kwargs) -> None:  # noqa: D102
        self.stdio.warning(*args, **kwargs)

    def input(self, *args, **kwargs) -> str:  # noqa: D102
        return self.stdio.input(*args, **kwargs)

    def enter_console(self) -> None:
        """Set up history and auto-complete."""
        # NOTE key bindings
        #   ^[A : arrow up
        #   ^[B : arrow down
        #   ^[C : arrow right
        #   ^[D : arrow left
        # https://www.gnu.org/software/bash/manual/html_node/Commands-For-History.html
        readline.set_completer_delims(' \t\n;')
        readline.parse_and_bind("tab: complete")
        readline.parse_and_bind(r'"\e[A": previous-history')
        readline.parse_and_bind(r'"\e[B": next-history')
        readline.set_completer(self.complete)
        if self.history_file and self.history_size:
            if not os.path.exists(self.history_file):
                with open(self.history_file, 'wt'):
                    pass
            readline.read_history_file(self.history_file)
            readline.set_history_length(self.history_size)

    def exit_console(self) -> None:
        """Save history."""
        if self.history_file and self.history_size:
            readline.set_history_length(self.history_size)
            readline.write_history_file(self.history_file)

    def await_input(self) -> None:
        """Wait for next user input."""
        self.enter_console()

        self.print(
            f'\nType {bformat.bold("help")} to list available '
            f'commands, or {bformat.bold("help <command>")} '
            f'for specific help.\n'
            f'Type {bformat.bold("Ctrl+C")} to interrupt the '
            f'current command and {bformat.bold("Ctrl+D")} to '
            f'exit the app.'
        )

        def are_you_sure() -> bool:
            try:
                # Query input
                args = self.input("\nDo you really want to exit ([y]/n)?")
                args.strip()
                if args[:1].lower() == "n":
                    return False
                else:
                    raise EOFError
            except KeyboardInterrupt:
                # Ctrl+C -> do not exit
                return False

        count = 1
        try:
            while True:
                try:
                    # Query input
                    args = self.input(iformat.fg.green(f'[{count}] '))
                    args = args.strip()
                    if not args:
                        continue
                    if args[:1] == "#":
                        # Allow for "comments"
                        continue
                    count += 1
                except EOFError:
                    # Ctrl+D -> propagate and catch later
                    if not are_you_sure():
                        continue
                except KeyboardInterrupt:
                    # Ctrl+C -> generate new input
                    self.print("")
                    continue

                try:
                    # Parse
                    args = self.parse_args(shlex.split(args))
                    if not vars(args):
                        raise ValueError("Unknown command")
                except EOFError:
                    # Ctrl+D -> propagate and catch later
                    if not are_you_sure():
                        continue
                except self.InterruptParsing:
                    # Caught "exit" call in parser. Silent it.
                    # (This is triggered by --help)
                    continue
                except KeyboardInterrupt:
                    # Ctrl+C -> Stop parsing
                    count += 1
                    continue
                except Exception as e:
                    # Other exceptions -> print + new input field
                    if self._debug:
                        self.debug(traceback.print_tb(e.__traceback__))
                    self.error("(PARSE ERROR)", e)
                    continue

                try:
                    # Execute
                    func = getattr(args, 'func', lambda x: None)
                    func(args)
                except EOFError:
                    # Ctrl+D -> propagate and catch later
                    if not are_you_sure():
                        continue
                except self.InterruptParsing:
                    # Caught "exit" call in parser. Silent it.
                    # (This is triggered by --help)
                    continue
                except (Exception, KeyboardInterrupt) as e:
                    # Other exceptions -> print + new input field
                    if self._debug:
                        self.debug(traceback.print_tb(e.__traceback__))
                    self.error(f"(EXEC ERROR) [{type(e).__name__}]", e)
                    continue

        except EOFError:
            # Ctrl+D -> graceful exit
            self.print('exit')
            raise SystemExit
        finally:
            self.exit_console()

    @property
    def parsers(self) -> argparse._SubParsersAction | None:
        """Return registered sub parsers."""
        parsers = None
        for action in self._actions:
            if isinstance(action, argparse._SubParsersAction):
                parsers = action
                break
        return parsers

    @property
    def subcommands(self) -> list[str]:
        """Return the names of all subcommands."""
        if getattr(self.parsers, 'choices', None):
            return list(self.parsers.choices.keys())
        return []

    def _listdir(self, root: str) -> list[str]:
        """List directory 'root' appending the path separator to subdirs."""
        res = []
        for name in os.listdir(root):
            path = os.path.join(root, name)
            if os.path.isdir(path):
                name += os.sep
            res.append(name)
        return res

    def _complete_path(self, path: str | None = None) -> str:
        """Perform completion of filesystem path."""
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

    def complete_default(self, context: str) -> str:
        """Fallback autocompleter."""
        return self._complete_path(os.path.expanduser(context))

    RE_SPACE = re.compile(r'.*\s+$', re.M)

    def complete(self, context: str, state: str) -> str:
        """Entry point for `readline` completion."""
        line = readline.get_line_buffer()
        begidx, endidx = readline.get_begidx(), readline.get_endidx()
        args = shlex.split(line)

        # show matching commands
        if not args or begidx <= len(args[0]):
            addspace = len(line) <= endidx or line[endidx] != ' '
            addspace = ' ' if addspace else ''
            result = [c + addspace for c in self.subcommands
                      if c.startswith(context)]

        # resolve command to the implementation function
        else:
            cmd = args[0].strip()
            if cmd in self.subcommands:
                template = 'complete_' + cmd
                impl = getattr(self, template, self.complete_default)
            else:
                impl = self.complete_default
            result = impl(context)

        try:
            return result[state]
        except IndexError:
            return None


class FixOrderHelpFormatter(argparse.HelpFormatter):
    """
    Fix help to say that positionals must be given before options.

    argparse default usage string says that positional arguments
    should be given after optional arguments, whereas it's really the
    opposite.

    The ordering happens in the middle of the long-ish _format_usage
    method, so I see no other fix than copying and fixing the full
    method.
    """

    def _format_usage(  # noqa: ANN202
        self: argparse.HelpFormatter,
        usage: str | None,
        actions: list[argparse.Action],
        groups: list[argparse._ArgumentGroup],
        prefix: str | None,
    ) -> str:
        if prefix is None:
            prefix = gettext('usage: ')

        # if usage is specified, use that
        if usage is not None:
            usage = usage % dict(prog=self._prog)

        # if no optionals or positionals are available, usage is just prog
        elif usage is None and not actions:
            usage = '%(prog)s' % dict(prog=self._prog)

        # if optionals and positionals are available, calculate usage
        elif usage is None:
            prog = '%(prog)s' % dict(prog=self._prog)

            # split optionals from positionals
            optionals = []
            positionals = []
            for action in actions:
                if action.option_strings:
                    optionals.append(action)
                else:
                    positionals.append(action)

            # build full usage string
            format = self._format_actions_usage
            action_usage = format(positionals + optionals, groups)
            usage = ' '.join([s for s in [prog, action_usage] if s])

            # wrap the usage parts if it's too long
            text_width = self._width - self._current_indent
            if len(prefix) + len(usage) > text_width:

                # break usage into wrappable parts
                part_regexp = (
                    r'\(.*?\)+(?=\s|$)|'
                    r'\[.*?\]+(?=\s|$)|'
                    r'\S+'
                )
                opt_usage = format(optionals, groups)
                pos_usage = format(positionals, groups)
                opt_parts = re.findall(part_regexp, opt_usage)
                pos_parts = re.findall(part_regexp, pos_usage)
                assert ' '.join(opt_parts) == opt_usage
                assert ' '.join(pos_parts) == pos_usage

                # helper for wrapping lines
                def get_lines(
                        parts: list[str],
                        indent: str,
                        prefix: str | None = None
                ) -> list[str]:
                    lines = []
                    line = []
                    indent_length = len(indent)
                    if prefix is not None:
                        line_len = len(prefix) - 1
                    else:
                        line_len = indent_length - 1
                    for part in parts:
                        if line_len + 1 + len(part) > text_width and line:
                            lines.append(indent + ' '.join(line))
                            line = []
                            line_len = indent_length - 1
                        line.append(part)
                        line_len += len(part) + 1
                    if line:
                        lines.append(indent + ' '.join(line))
                    if prefix is not None:
                        lines[0] = lines[0][indent_length:]
                    return lines

                # if prog is short, follow it with optionals or positionals
                if len(prefix) + len(prog) <= 0.75 * text_width:
                    indent = ' ' * (len(prefix) + len(prog) + 1)
                    if pos_parts:
                        lines = get_lines([prog] + pos_parts, indent, prefix)
                    elif opt_parts:
                        lines = get_lines([prog] + opt_parts, indent, prefix)
                        lines.extend(get_lines(pos_parts, indent))
                    else:
                        lines = [prog]

                # if prog is long, put it on its own line
                else:
                    indent = ' ' * len(prefix)
                    parts = pos_parts + opt_parts
                    lines = get_lines(parts, indent)
                    if len(lines) > 1:
                        lines = []
                        lines.extend(get_lines(pos_parts, indent))
                        lines.extend(get_lines(opt_parts, indent))
                    lines = [prog] + lines

                # join lines into usage
                usage = '\n'.join(lines)

        # prefix with 'usage:'
        return '%s%s\n\n' % (prefix, usage)


class MaxChoiceHelpFormatter(argparse.HelpFormatter):
    """Only display up to `max_choices` choices in usage string."""

    def _metavar_formatter(
        self, action, default_metavar  # noqa: ANN001
    ) -> callable:
        if action.metavar is not None:
            result = action.metavar
        elif action.choices is not None:
            # --- begin patch ---
            choices = list(action.choices)
            max_choices = getattr(self, 'max_choices', None)
            if max_choices is not None and len(choices) > max_choices:
                choices = choices[:max_choices] + ['...']
            result = '{%s}' % ','.join(map(str, choices))
            # ---  end patch  ---
        else:
            result = default_metavar

        def format(tuple_size: int) -> tuple:
            if isinstance(result, tuple):
                return result
            else:
                return (result, ) * tuple_size
        return format


class ActionHelpFormatter(
    FixOrderHelpFormatter,
    argparse.RawTextHelpFormatter,
):
    """Formatter used for commands."""


class MainHelpFormatter(
    FixOrderHelpFormatter,
    MaxChoiceHelpFormatter,
    argparse.RawTextHelpFormatter,
):
    """Formatter used for the main console."""

    def __init__(self, *args, **kwargs) -> None:
        self.max_choices = kwargs.pop("max_choices", None)
        super().__init__(*args, **kwargs)
