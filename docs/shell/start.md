# nglocal

```shell
>>> nglocal --help

Run a local neuroglancer

usage: nglocal [-h]
        [--token TOKEN]
        [--ip IP]
        [--port PORT]
        [--port-fileserver PORT]
        [--no-window]
        [--no-fileserver]
        [--debug]
        [--log-level {any,debug,info,warning,error,none}]
        [--stdin STDIN]
        [--stdout STDOUT]
        [--stderr STDERR]
        [filenames ...]

positional arguments:
  filenames                     Files to load

options:
  -h, --help                    show this help message and exit
  --token TOKEN                 neuroglancer unique token
  --ip IP                       local IP
  --port PORT                   viewer port
  --port-fileserver PORT        fileserver port
  --no-window                   do not open neuroglancer window
  --no-fileserver               do not run a local fileserver
  --debug                       run in debug mode
  --log-level {any,debug,info,warning,error,none}
                                logging level
  --stdin STDIN                 input stream (default: stdin)
  --stdout STDOUT               output stream (default: stdout)
  --stderr STDERR               error stream (default: stderr)
```

```shell
>>> nglocal
```

<pre><code>             _              _
 _ __   __ _| |_ ___   ___ | |___
| '_ \ / _` | __/ _ \ / _ \| / __|
| | | | (_| | || (_) | (_) | \__ \
|_| |_|\__, |\__\___/ \___/|_|___/
       |___/

fileserver:   http://127.0.0.1:9123/
neuroglancer: http://127.0.0.1:9321/v/1/

Type <b>help</b> to list available commands, or <b>help &lt;command&gt;</b> for specific help.
Type <b>Ctrl+C</b> to interrupt the current command and <b>Ctrl+D</b> to exit the app.
<b>[1]</b>
</code></pre>
