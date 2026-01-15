"""Color tags that can be used in a terminal prompt."""
import os
import sys

try:
    import colorama

    # Avoid initializing colorama in non-Windows platforms.
    colorama.just_fix_windows_console()
except (
    AttributeError,     # colorama <= 0.4.6.
    ImportError,        # colorama is not installed.
    # If just_fix_windows_console() accesses sys.stdout with
    # WSGIRestrictedStdout.
    OSError,
):
    HAS_COLORAMA = False
else:
    HAS_COLORAMA = True


def supports_color() -> bool:
    """
    Return True if the running system's terminal supports color,
    and False otherwise.
    """
    # Copied from Django - BSD-3
    # https://github.com/django/django/blob/main/django/core/management/color.py
    # https://github.com/django/django/blob/main/LICENSE

    def vt_codes_enabled_in_windows_registry() -> bool:
        """
        Check the Windows Registry to see if VT code handling has been enabled
        by default, see https://superuser.com/a/1300251/447564.
        """
        try:
            # winreg is only available on Windows.
            import winreg
        except ImportError:
            return False
        else:
            try:
                reg_key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Console")
                value, _ = winreg.QueryValueEx(reg_key, "VirtualTerminalLevel")
            except FileNotFoundError:
                return False
            else:
                return value == 1

    # isatty is not always implemented, #6223.
    is_a_tty = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()

    return is_a_tty and (
        sys.platform != "win32"
        or (HAS_COLORAMA and getattr(colorama, "fixed_windows_console", False))
        or "ANSICON" in os.environ
        or
        # Windows Terminal supports VT codes.
        "WT_SESSION" in os.environ
        or
        # Microsoft Visual Studio Code's built-in terminal supports colors.
        os.environ.get("TERM_PROGRAM") == "vscode"
        or vt_codes_enabled_in_windows_registry()
    )


SUPPORTS_COLOR = supports_color()


def rgb_to_ansi256(r: int, g: int, b: int) -> int:
    """Convert RGB [0-255] to ANSI256 code."""
    # we use the extended greyscale palette here, with the exception of
    # black and white. normal palette only has 4 greyscale shades.
    r, g, b = int(round(r)), int(round(g)), int(round(b))
    if (r == g and g == b):
        if (r < 8):
            return 16
        if (r > 248):
            return 231
        return round(((r - 8) / 247) * 24) + 232

    return int(
        16 +
        (36 * round(r / 255 * 5)) +
        (6 * round(g / 255 * 5)) +
        round(b / 255 * 5)
    )


class tcolors:
    """Namespace that maps colors to integers for use in terminals."""

    (endc, bold, faint, italic, underline, blink) = range(6)

    class fg:
        """Foreground colors."""

        (black, red, green, yellow, blue, magenta, cyan, white) \
            = range(30, 38)

        class bright:
            """Bright foreground colors."""

            (black, red, green, yellow, blue, magenta, cyan, white) \
                = range(90, 98)

    class bg:
        """Background colors."""

        (black, red, green, yellow, blue, magenta, cyan, white) \
            = range(40, 48)

        class bright:
            """Bright background colors."""

            (black, red, green, yellow, blue, magenta, cyan, white) \
                = range(100, 108)

    header = fg.bright.magenta
    okblue = fg.bright.blue
    okcyan = fg.bright.cyan
    okgreen = fg.bright.green
    warning = fg.bright.yellow
    fail = fg.bright.red


class bcolors:
    """Namespace that maps colors to tokens for use in bash terminals."""


class icolors:
    """Namespace that maps colors to tokens for use in bash input prompts."""


class bformat:
    """Namespace for formatting functions, for use in bash terminals."""


class iformat:
    """Namespace for formatting functions, for use in bash input prompts."""


# To add color in the input prompt, all delimiter must be wrapped
# in this. See:
# https://bugs.python.org/issue12972
# https://stackoverflow.com/questions/9468435/
L = '\001'
R = '\002'


def _copy_attrs(dst: type, src: type, fn: callable) -> None:
    """Copy attributes (and classes) from source class to dest class."""
    for key, val in src.__dict__.items():
        if key.startswith("_"):
            continue
        if isinstance(val, type):
            setattr(dst, key, type(key, (), {}))
            _copy_attrs(getattr(dst, key), getattr(src, key), fn)
        else:
            setattr(dst, key, fn(getattr(src, key)))


def _make_format(color: str, endc: str) -> classmethod:
    return classmethod(lambda _, x: f"{color}{x}{endc}")


_copy_attrs(bcolors, tcolors, lambda x: f'\033[{x}m')
_copy_attrs(icolors, bcolors, lambda x: L + x + R)
_copy_attrs(bformat, bcolors, lambda x: _make_format(x, bcolors.endc))
_copy_attrs(iformat, icolors, lambda x: _make_format(x, icolors.endc))


tcolors.fg.rgb256 = staticmethod(rgb_to_ansi256)
bcolors.fg.rgb256 = staticmethod(
    lambda *a: f'\033[38;5;{tcolors.fg.rgb256(*a)}m'
)
bformat.fg.rgb256 = staticmethod(
    lambda *a: (lambda x: f"{bcolors.fg.rgb256(*a)}{x}{bcolors.endc}")
)
iformat.fg.rgb256 = staticmethod(
    lambda *a: (lambda x: f"{icolors.fg.rgb256(*a)}{x}{icolors.endc}")
)

tcolors.bg.rgb256 = staticmethod(rgb_to_ansi256)
bcolors.bg.rgb256 = staticmethod(
    lambda *a: f'\033[48;5;{tcolors.bg.rgb256(*a)}m'
)
bformat.bg.rgb256 = staticmethod(
    lambda *a: (lambda x: f"{bcolors.bg.rgb256(*a)}{x}{bcolors.endc}")
)
iformat.bg.rgb256 = staticmethod(
    lambda *a: (lambda x: f"{icolors.bg.rgb256(*a)}{x}{icolors.endc}")
)
