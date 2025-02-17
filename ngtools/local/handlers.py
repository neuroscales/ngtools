"""Handlers for the WSGI server."""
# stdlib
import json
import logging

# externals
import requests

# internals
from ngtools.local.fileserver import Handler
from ngtools.opener import exists, linc_auth_opt
from ngtools.shaders import load_fs_lut

LOG = logging.getLogger(__name__)


class LutHandler(Handler):
    """Handler that returns a segment_properties from a FS LUT."""

    def get(self, protocol: str, path: str) -> None:  # noqa: D102

        if not path.endswith("info"):
            return self.send_error(404, path)

        path = path[:-5]
        path = protocol + "://" + path

        if not exists(path):
            return self.send_error(404, path)

        try:
            lut = load_fs_lut(path)
        except Exception as e:
            return self.send_error(400, e)

        segment_properties = {
            "@type": "neuroglancer_segment_properties",
            "inline": {
                "ids": list(map(str, lut.keys())),
                "properties": [
                    {
                        "id": "name",
                        "type": "label",
                        "description": "Name",
                        "values": [name for name, _ in lut.values()]
                    }
                ]
            }
        }

        self.status = 200
        self.headers["Content-type"] = "application/json"
        self.body = json.dumps(segment_properties).encode()


class LincHandler(Handler):
    """Handler that redirects to linc data."""

    VALID_KEYS = (
        "Accept",
        "Accept-encoding",
        "Content-type",
        "Content-length",
        # "Connection",  # hop-by-hop
        "Last-modified",
        "Accept-ranges",
        "Range",
        "Date",
        "Etag",
        "Vary",
    )

    # NOTE: Connection is a hop-by-hop header and should therfore not
    # be forwarded to the next request.
    # https://0xn3va.gitbook.io/cheat-sheets/web-application/abusing-http-hop-by-hop-request-headers

    def prepare(self) -> None:  # noqa: D102
        if getattr(self.app, "_linc_cookies", None) is None:
            self.app._linc_cookies = linc_auth_opt()
        if getattr(self.app, "_linc_sessions", None) is None:
            self.app._linc_session = requests.Session()
        self._linc_session = self.app._linc_session
        self._linc_cookies = self.app._linc_cookies

    def _request(self, method: str, path: str) -> None:
        LOG.debug(
            f"LincHandler: {method} {path} << {dict(self.environ.headers)}"
        )
        session = self._linc_session
        cookies = self._linc_cookies

        # select input headers to forward
        headers = {
            key: value
            for key, value in self.environ.headers.items()
            if key.capitalize() in self.VALID_KEYS
        }

        # fetch
        url = "https://neuroglancer.lincbrain.org/" + str(path)
        try:
            fetch = getattr(session, method.lower())
            response = fetch(url, headers=headers, **cookies)

        except Exception as e:
            LOG.debug(f"LincHandler: {path} | {e}")
            return self.send_error(400, e)

        else:
            status = getattr(response, "status_code", 400)
            headers = dict(getattr(response, "headers", {}))
            content = getattr(response, "content", None)

            LOG.debug(
                f"LincHandler: {method} {path} >> {status} {headers}"
            )

        self.status = status

        # select output headers to forward
        for key, val in headers.items():
            if key.capitalize() in self.VALID_KEYS:
                self.headers.add_header(key, val)

        if method.upper() == "GET":
            self.body = content

        LOG.debug(f"{method} | {path} | done")

    def get(self, path: str) -> None:  # noqa: D102
        return self._request("GET", path)

    def head(self, path: str) -> None:  # noqa: D102
        return self._request("HEAD", path)
