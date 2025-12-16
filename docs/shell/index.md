---
icon: octicons/terminal-24
---

# Shell API

=== ":octicons-terminal-24:"
    ```shell
    nglocal
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

### Positional arguments

| Name        | Type        | Description   |
| ----------- | ----------- | ------------- |
| `filenames` | `list[uri]` | Files to load |

### Options

| Flag                  | Type      | Description                       | Default   |
| --------------------- | --------- | --------------------------------- | --------- |
| `-h`, `--help`        |           | Show help message and exit        |           |
| `--token`             | `str`     | Neuroglancer unique token         | `1`       |
| `--ip`                | `str`     | local IP                          | `127.0.0.1` |
| `--port`              | `int`     | Viewer port                       | `9321`    |
| `--port-fileserver`   | `int`     | Fileserver port                   | `9123`    |
| `--no-window`         |           | Do not open neuroglancer window   |           |
| `--no-fileserver`     |           | Do not run a local fileserver     |           |
| `--debug`             |           | Run in debug mode                 |           |
| `--log-level`         | `{any,debug,info,warning,error,none}` | Logging level | `debug` if `--debug` else `none` |
| `--stdin`             | `str`     |  Input stream                     | `stdin`   |
| `--stdout`            | `str`     |  Output stream                    | `stdout`  |
| `--stderr`            | `str`     |  Error stream                     | `stderr`  |
