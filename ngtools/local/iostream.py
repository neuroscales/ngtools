"""Handling of standard IO + logging."""
import atexit
import logging
import sys
from io import StringIO
from os import PathLike
from typing import Literal, TextIO

from ngtools.local.termcolors import bformat

LogLevel = int | Literal["error", "warning", "info", "debug", "any"] | None
LOG = logging.getLogger(__name__)


class StandardIO:
    """An IO handling class."""

    def __init__(
        self,
        stdin: TextIO | PathLike | str = sys.stdin,
        stdout: TextIO | PathLike | str = sys.stdout,
        stderr: TextIO | PathLike | str = sys.stderr,
        level: LogLevel = "info",
        logger: logging.Logger = LOG,
    ) -> None:
        self.stdin = stdin
        self.stdout = stdout
        self.stderr = stderr
        self.logger = logger
        if isinstance(level, str):
            if level[0].lower() == "a":
                level = 0
            else:
                level = getattr(logging, level.upper())
        self.level = level
        atexit.register(self.__del__)

    @property
    def stdin(self) -> TextIO | PathLike | str:
        """Get input stream."""
        return self._stdin

    @stdin.setter
    def stdin(self, value: TextIO | PathLike | str) -> None:
        stdin_file = getattr(self, "_stdin_file", None)
        if stdin_file and not stdin_file.closed:
            stdin_file.close()
        self._stdin = value
        self._stdin_file = None

    def format(self, *args, **kwargs) -> str:
        """Format a message like `print` would."""
        with StringIO() as f:
            print(*args, **kwargs, file=f)
            msg = f.getvalue()
        return msg

    def print(self, *args, **kwargs) -> str:
        """Print something to the stdout (no logging)."""
        file = kwargs.pop("file", self.stdout)
        flush = kwargs.pop("flush", False)
        level = kwargs.pop("level", float('inf'))
        format = kwargs.pop("format", (lambda x: x))

        msg = self.format(*args, **kwargs)
        if file and self.level <= level:
            if isinstance(file, str):
                with open(file, "a") as f:
                    print(format(msg), file=f, flush=flush, end="")
            else:
                print(format(msg), file=file, flush=flush, end="")
        return msg

    def debug(self, *args, **kwargs) -> str:
        """Debug -- Print something to the stderr."""
        kwargs.setdefault("file", self.stderr)
        msg = self.print(*args, **kwargs, level=logging.DEBUG)
        if self.logger:
            self.logger.debug(msg)
        return msg

    def info(self, *args, **kwargs) -> str:
        """Info -- Print something to the stdout."""
        msg = self.print(*args, **kwargs, level=logging.INFO)
        if self.logger:
            self.logger.info(msg)
        return msg

    def warning(self, *args, **kwargs) -> str:
        """Warn -- Print something to the stdout in orange."""
        kwargs["format"] = bformat.warning
        msg = self.print(*args, **kwargs, level=logging.WARNING)
        if self.logger:
            self.logger.warning(msg)
        return msg

    def error(self, *args, **kwargs) -> str:
        """Warn -- Print something to the stderr in red."""
        kwargs["format"] = bformat.fail
        msg = self.print(*args, **kwargs, level=logging.ERROR)
        if self.logger:
            self.logger.error(msg)
        return msg

    def input(self, prompt: str = "") -> str:
        """Read input from stdin or from a file."""
        stdin = self.stdin
        if stdin is sys.stdin:
            return input(prompt)
        else:
            if not hasattr(stdin, "read"):
                stdin = getattr(self, "_stdin_file", None)
            if stdin is None:
                stdin = self._stdin_file = open(self.stdin)

            self.print(prompt, end="")
            while True:
                try:
                    inp = next(stdin).rstrip("\n").rstrip("\r").rstrip("\n")
                    self.print(inp)
                    return inp
                except StopIteration:
                    pass

    def __del__(self) -> None:
        if hasattr(self, "_stdin_file"):
            stdin = self._stdin_file
            if stdin and not stdin.closed:
                stdin.close()
        atexit.unregister(self.__del__)
