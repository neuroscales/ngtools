import os
import sys
import re
import shlex                    # parse user input as if a shell commandline
import atexit                   # do stuff whe exiting (save history...)
import readline                 # autocomplete/history in user input
import argparse
import traceback
from gettext import gettext  # to fix usage string
from .utils import bcolors


class ParserApp(argparse.ArgumentParser):
    """
    An ArgumentParser that can be used as a commandline app.
    It handles history and autocomplete.
    """

    DEFAULT_HISTFILE = '~/.neuroglancer_history'
    DEFAULT_HISTSIZE = 1000

    def __init__(self, *args, **kwargs):
        self.debug = kwargs.pop('debug', False)
        # Commandline behavior
        self.history_file = kwargs.pop('history_file', self.DEFAULT_HISTFILE)
        if self.history_file:
            self.history_file = os.path.expanduser(self.history_file)
        self.history_size = kwargs.pop('history_size', self.DEFAULT_HISTSIZE)
        # Exit behavior
        kwargs.setdefault('exit_on_error', False)
        self.exit_on_help = kwargs.pop('exit_on_help', False)
        if sys.version_info < (3, 9):
            self.exit_on_error = kwargs.pop('exit_on_error')
        # ArgumentParser.__init__
        super().__init__(*args, **kwargs)
        # Fixed usage formatter (monkey patch)
        self.formatter_class = _fixhelpformatter(self.formatter_class)

    @property
    def parsers(self):
        parsers = None
        for action in self._actions:
            if isinstance(action, argparse._SubParsersAction):
                parsers = action
                break
        return parsers

    class InterruptParsing(Exception):
        pass

    def exit(self, status=0, message=None):
        """Overload ArgumentParser.exit to disable it"""
        pass
        raise self.InterruptParsing
        # if self.exit_on_help:
        #     return super().exit(status, message)
        # if message:
        #     print(message, file=sys.stderr)

    def error(self, message):
        """Overload ArgumentParser.error to disable it"""
        pass
        # if sys.version_info >= (3, 9) or self.exit_on_error:
        #     return super().error(message)
        # try:
        #     return super().error(message)
        # except SystemExit:
        #     pass

    def enter_console(self):
        """Setup history and auto-complete"""
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
            atexit.register(self.exit_console)

    def exit_console(self):
        """Save history"""
        if self.history_file and self.history_size:
            readline.set_history_length(self.history_size)
            readline.write_history_file(self.history_file)

    def await_input(self):
        self.enter_console()

        print(
            f'\nType {bcolors.bold}help{bcolors.endc} to list available '
            f'commands, or {bcolors.bold}help <command>{bcolors.endc} '
            f'for specific help.\n'
            f'Type {bcolors.bold}Ctrl+C{bcolors.endc} to interrupt the '
            f'current command and {bcolors.bold}Ctrl+D{bcolors.endc} to '
            f'exit the app.'
        )

        def green(s):
            return '\001' + bcolors.fg.green + str(s) + bcolors.endc + '\002'

        count = 1
        try:
            while True:
                try:
                    # Query input
                    args = input(green(f'[{count}] '))
                    if not args.strip():
                        print('')
                        continue
                    count += 1
                except KeyboardInterrupt:
                    # Ctrl+C -> generate new input
                    print('')
                    continue

                try:
                    # Parse
                    args = self.parse_args(shlex.split(args))
                    if not vars(args):
                        raise ValueError("Unknown command")
                except EOFError as e:
                    # Ctrl+D -> propagate anc catch later
                    raise e
                except self.InterruptParsing:
                    # Caught "exit" call in parser. Silent it.
                    continue
                except Exception as e:
                    # Other exceptions -> print + new input field
                    if self.debug:
                        print(traceback.print_tb(e.__traceback__),
                              file=sys.stderr)
                    print(f"{bcolors.fail}(PARSE ERROR)", e, bcolors.endc,
                          file=sys.stderr)
                    continue

                try:
                    # Execute
                    func = getattr(args, 'func', lambda x: None)
                    func(args)
                except EOFError as e:
                    # Ctrl+D -> propagate anc catch later
                    raise e
                except self.InterruptParsing:
                    # Caught "exit" call in parser. Silent it.
                    continue
                except Exception as e:
                    # Other exceptions -> print + new input field
                    if self.debug:
                        print(traceback.print_tb(e.__traceback__),
                              file=sys.stderr)
                    print(f"{bcolors.fail}(EXEC ERROR)", e, bcolors.endc,
                          file=sys.stderr)
                    continue

        except EOFError:
            # Ctrl+D -> graceful exit
            self.exit_console()
            print('exit')
            sys.exit()
        finally:
            self.exit_console()

    @property
    def subcommands(self):
        if getattr(self.parsers, 'choices', None):
            return list(self.parsers.choices.keys())
        return []

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

    def complete_default(self, context):
        return self._complete_path(os.path.expanduser(context))

    RE_SPACE = re.compile(r'.*\s+$', re.M)

    def complete(self, context, state):
        "Generic readline completion entry point."
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


def _fixhelpformatter(klass):
    """
    argparse default usage string says that positional arguments
    should be given after optional arguments, whereas it's really the
    opposite.

    The ordering happens in the middle of the long-ish _format_usage
    method, so I see no other fix than copying and fixing the full
    method.
    """

    def _format_usage(self, usage, actions, groups, prefix):
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
                def get_lines(parts, indent, prefix=None):
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
                    if opt_parts:
                        lines = get_lines([prog] + opt_parts, indent, prefix)
                        lines.extend(get_lines(pos_parts, indent))
                    elif pos_parts:
                        lines = get_lines([prog] + pos_parts, indent, prefix)
                    else:
                        lines = [prog]

                # if prog is long, put it on its own line
                else:
                    indent = ' ' * len(prefix)
                    parts = opt_parts + pos_parts
                    lines = get_lines(parts, indent)
                    if len(lines) > 1:
                        lines = []
                        lines.extend(get_lines(opt_parts, indent))
                        lines.extend(get_lines(pos_parts, indent))
                    lines = [prog] + lines

                # join lines into usage
                usage = '\n'.join(lines)

        # prefix with 'usage:'
        return '%s%s\n\n' % (prefix, usage)

    klass._format_usage = _format_usage
    return klass
