class bcolors:

    # To add color in the input prompt, all delimiter must be wrapped
    # in \001 \002. See:
    # https://bugs.python.org/issue12972
    # https://stackoverflow.com/questions/9468435/
    RL_PROMPT_START_IGNORE = '\001'
    RL_PROMPT_END_IGNORE = '\002'

    # I should do this more programatically (only save the number, and
    # generate the full string automatically)

    endc = '\001\033[0m\002'
    bold = '\001\033[1m\002'
    faint = '\001\033[2m\002'
    underline = '\001\033[4m\002'
    blink = '\001\033[5m\002'

    class fg:
        black = '\001\033[30m\002'
        red = '\001\033[31m\002'
        green = '\001\033[32m\002'
        yellow = '\001\033[33m\002'
        blue = '\001\033[34m\002'
        magenta = '\001\033[35m\002'
        cyan = '\001\033[36m\002'
        white = '\001\033[37m\002'

        class bright:
            black = '\001\033[90m\002'
            red = '\001\033[91m\002'
            green = '\001\033[92m\002'
            yellow = '\001\033[93m\002'
            blue = '\001\033[94m\002'
            magenta = '\001\033[95m\002'
            cyan = '\001\033[96m\002'
            white = '\001\033[97m\002'

    class bg:
        black = '\033[40m'
        red = '\033[41m'
        green = '\033[42m'
        yellow = '\033[43m'
        blue = '\033[44m'
        magenta = '\033[45m'
        cyan = '\033[46m'
        white = '\033[47m'

        class bright:
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
