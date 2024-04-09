import json
from urllib.parse import quote

class bcolors:

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
        black = '\033[30m'
        red = '\033[31m'
        green = '\033[32m'
        yellow = '\033[33m'
        blue = '\033[34m'
        magenta = '\033[35m'
        cyan = '\033[36m'
        white = '\033[37m'

        class bright:
            black = '\033[90m'
            red = '\033[91m'
            green = '\033[92m'
            yellow = '\033[93m'
            blue = '\033[94m'
            magenta = '\033[95m'
            cyan = '\033[96m'
            white = '\033[97m'

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


def neuroglancer_state_to_neuroglancer_url(
        state,
        base_url="https://neuroglancer.lincbrain.org/cloudfront/frontend/index.html"
):
    """
    Takes the current State of the Neuroglancer Viewer and converts it into a url-encoded value for the browser

    """
    ordered_dict = state.to_json()

    # Convert the dictionary to a JSON string
    json_str = json.dumps(ordered_dict)

    # URL-encode the JSON string
    encoded_json = quote(json_str)

    # Construct and return the full URL
    return f"{base_url}#!{encoded_json}"
