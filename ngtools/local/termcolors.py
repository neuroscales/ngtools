"""Color tags that can be used in a terminal prompt."""


class bcolors:
    """Namespace for all colors."""

    # To add color in the input prompt, all delimiter must be wrapped
    # in  . See:
    # https://bugs.python.org/issue12972
    # https://stackoverflow.com/questions/9468435/
    RL_PROMPT_START_IGNORE = '\001'
    RL_PROMPT_END_IGNORE = '\002'

    # I should do this more programatically (only save the number, and
    # generate the full string automatically)

    endc = '\033[0m'
    bold = '\033[1m'
    faint = '\033[2m'
    underline = '\033[4m'
    blink = '\033[5m'

    class fg:
        """Foreground colors."""

        black = '\033[30m'
        red = '\033[31m'
        green = '\033[32m'
        yellow = '\033[33m'
        blue = '\033[34m'
        magenta = '\033[35m'
        cyan = '\033[36m'
        white = '\033[37m'

        class bright:
            """Bright foreground colors."""

            black = '\033[90m'
            red = '\033[91m'
            green = '\033[92m'
            yellow = '\033[93m'
            blue = '\033[94m'
            magenta = '\033[95m'
            cyan = '\033[96m'
            white = '\033[97m'

    class bg:
        """Background colors."""

        black = '\033[40m'
        red = '\033[41m'
        green = '\033[42m'
        yellow = '\033[43m'
        blue = '\033[44m'
        magenta = '\033[45m'
        cyan = '\033[46m'
        white = '\033[47m'

        class bright:
            """Bright background colors."""

            black = '\033[100m'
            red = '\033[101m'
            green = '\033[102m'
            yellow = '\033[103m'
            blue = '\033[104m'
            magenta = '\033[105m'
            cyan = '\033[106m'
            white = '\033[107m'

    header = fg.bright.magenta
    okblue = fg.bright.blue
    okcyan = fg.bright.cyan
    okgreen = fg.bright.green
    warning = fg.bright.yellow
    fail = fg.bright.red


def _make_format(name: str) -> classmethod:
    color = bcolors
    for name in name.split("."):
        color = getattr(color, name)
    return classmethod(lambda _, x: f"{color}{x}{bcolors.endc}")


class bformat:
    """Namespace for formatting functions."""

    bold = _make_format("bold")
    faint = _make_format("faint")
    underline = _make_format("underline")
    blink = _make_format("blink")

    class fg:
        """Foreground colors."""

        black = _make_format("fg.black")
        red = _make_format("fg.red")
        green = _make_format("fg.green")
        yellow = _make_format("fg.yellow")
        blue = _make_format("fg.blue")
        magenta = _make_format("fg.magenta")
        cyan = _make_format("fg.cyan")
        white = _make_format("fg.white")

        class bright:
            """Bright foreground colors."""

            black = _make_format("fg.bright.black")
            red = _make_format("fg.bright.red")
            green = _make_format("fg.bright.green")
            yellow = _make_format("fg.bright.yellow")
            blue = _make_format("fg.bright.blue")
            magenta = _make_format("fg.bright.magenta")
            cyan = _make_format("fg.bright.cyan")
            white = _make_format("fg.bright.white")

    class bg:
        """Background colors."""

        black = _make_format("bg.black")
        red = _make_format("bg.red")
        green = _make_format("bg.green")
        yellow = _make_format("bg.yellow")
        blue = _make_format("bg.blue")
        magenta = _make_format("bg.magenta")
        cyan = _make_format("bg.cyan")
        white = _make_format("bg.white")

        class bright:
            """Bright background colors."""

            black = _make_format("bg.bright.black")
            red = _make_format("bg.bright.red")
            green = _make_format("bg.bright.green")
            yellow = _make_format("bg.bright.yellow")
            blue = _make_format("bg.bright.blue")
            magenta = _make_format("bg.bright.magenta")
            cyan = _make_format("bg.bright.cyan")
            white = _make_format("bg.bright.white")

    header = fg.bright.magenta
    okblue = fg.bright.blue
    okcyan = fg.bright.cyan
    okgreen = fg.bright.green
    warning = fg.bright.yellow
    fail = fg.bright.red
