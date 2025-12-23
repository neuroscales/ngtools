"""Handlers for the WSGI server."""
# stdlib
import json
import logging
from pathlib import Path
from typing import Tuple

import neuroglancer as ng

# externals
import numpy as np
import requests
import trk_to_annotation.preprocessing as tap
import trk_to_annotation.segmentation as tas
import trk_to_annotation.tract_sharding as tats
import trk_to_annotation.utils as tau

# internals
from ngtools.local.fileserver import Handler
from ngtools.opener import exists, filesystem, linc_auth_opt
from ngtools.shaders import load_fs_lut
from ngtools.utils import _SizedRefreshCache

LOG = logging.getLogger(__name__)

annotation_cache = _SizedRefreshCache(max_size=20)
spatial_cache = _SizedRefreshCache(max_size=20)
info_cache = _SizedRefreshCache(max_size=40)
tracts_cache = _SizedRefreshCache(max_size=20)


filter_layers = {}
filtered_layers = {}

class TractAnnotationHandler(Handler):
    """Handler that returns annotation data."""

    grid_densities = [1, 2, 4, 8]

    def get(self, protocol: str, path: str) -> None:  # noqa: D102
        
        path = protocol + "://" + path
        
        if path.endswith("info"):
            try:
                info = self._get_info(path)
            except Exception as e:
                return self.send_error(404, e)
            self.status = 200
            self.headers["Content-type"] = "application/json"
            self.body = json.dumps(info).encode()
            return None
        
        if self._parse_segmentation_path(path):
            try:
                buffer = self._get_segmentation(path)
            except Exception as e:
                return self.send_error(404, e)
            self.status = 200
            self.headers["Content-type"] = "application/octet-stream"
            self.body = buffer
            return None
        
        if path.split("/")[-2] == "by_tract":
            try:
                buffer = self._get_tract(path)
            except Exception as e:
                return self.send_error(404, e)
            self.status = 200
            self.headers["Content-type"] = "application/octet-stream"
            self.body = buffer
            return None
        
        if path.split("/")[-2] == "filter":
            try:
                buffer = self._get_filter(path)
            except Exception as e:
                return self.send_error(404, e)
            self.status = 200
            self.headers["Content-type"] = "application/octet-stream"
            self.body = buffer
            return None
        
        if self._parse_spatial_path(path) is not None:
            try:
                buffer = self._get_spatials(path)
            except Exception as e:
                return self.send_error(404, e)
            self.status = 200
            self.headers["Content-type"] = "application/octet-stream"
            self.body = buffer
            return None
        
        if path.endswith("transform.lta"):
            try:
                transform = self._get_transform(path)
            except Exception as e:
                return self.send_error(404, e)
            self.status = 200
            self.headers["Content-type"] = "text/plain"
            self.body = transform.encode("utf-8")
            return None
        
        return self.send_error(404, f"Not valid emulated path: {path}")
        
    def _read_from_file(self, path: str) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray
        ]:
        """Read annotation data from a given file. cache it as well as its segments."""
        if path in annotation_cache:
            return annotation_cache[path]
        file = filesystem(path)
        with file.open(path, "rb") as f:
            segments, bbox, offsets, affine = tap.load_from_file(f)
        segments, offsets = tap.split_along_grid(
            segments, bbox, [self.grid_densities[-1]]*3, offsets)
        annotation_cache[path] = (segments, bbox, offsets, affine)
        return segments, bbox, offsets, affine

    def _parse_spatial_path(self, path_str: str) -> Tuple[str, str, str, str] | None:
        """Parse the path which spatial section we are looking for."""
        p = Path(path_str)
        last = p.name
        parts = last.split("_")
        if len(parts) == 3:
            parent = p.parent.name
            a, b, c = parts
            return parent, a, b, c
        else:
            return None
        
    def _get_filter(self, path: str) -> bytes:
        trk_path = "/".join(path.split("/")[:-2])
        key = trk_path.split("://")[-1].replace("/", "$").replace(".", "$")
        for full_key in filter_layers.keys():
            if full_key.endswith(key):
                key = full_key
        
        segments, _, offsets, _ = self._read_from_file(trk_path)
        source = filtered_layers[key].source
        if isinstance(source, ng.viewer_state.LayerDataSources):
            transform = np.array(source[0].transform.to_json().get(
                "matrix", np.eye(4)[:-1]))
        else:
            transform = np.array(source.transform.to_json().get(
                "matrix", np.eye(4)[:-1]))
        
        transform_segments = segments.copy()
        transform_segments["start"] = np.hstack(
            [segments["start"], np.ones((segments["start"].shape[0], 1))])@transform.T
        transform_segments["end"] = np.hstack(
            [segments["end"], np.ones((segments["end"].shape[0], 1))])@transform.T
        tract_ids = self._get_annotations_inside(transform_segments, 
                                                 filter_layers[key])
        tract = tats.tract_bytes(tract_ids, offsets, segments)

        return tract
    
    def _get_tract(self, path: str) -> bytes:
        split_path = path.split("/")
        folder = "/".join(split_path[:-1])
        tract_id = int(split_path[-1])
        if folder in tracts_cache:
            if tract_id in tracts_cache[folder]:
                return tracts_cache[folder][tract_id]
        else:
            tracts_cache[folder] = _SizedRefreshCache(max_size=100_000)


        trk_path = "/".join(path.split("/")[:-2])
        segments, _, offsets, _ = self._read_from_file(trk_path)
        tract = tats.tract_bytes([int(path.split("/")[-1])], offsets, segments)
        tracts_cache[folder][tract_id] = tract

        return tract

    def _parse_segmentation_path(self, path_str: str) -> bool:
        split_path = path_str.split("/")
        if len(split_path) < 3:
            return False
        if split_path[-3] != "precomputed_segmentations":
            return False
        if len(split_path[-2].split("_")) != 3:
            return False
        return len(split_path[-1].split("_")) == 3

    def _get_spatials(self, path: str) -> bytes:
        """Get spatial data from a given path."""
        parent, a, b, c = self._parse_spatial_path(path)
        trk_path = "/".join(path.split("/")[:-2])

        if trk_path in spatial_cache:
            return spatial_cache[trk_path][int(parent)][f"{a}_{b}_{c}"]

        segments, bbox, offsets, _ = self._read_from_file(trk_path)
        spatial = tau.get_spatials(segments, bbox, offsets,
                                self.grid_densities)
        spatial_cache[trk_path] = spatial
        return spatial[int(parent)][f"{a}_{b}_{c}"]

    def _get_info(self, path: str) -> dict:
        """Get annotation info from a given path."""
        if info_cache.get(path) is not None:
            return info_cache[path]
        
        if path.endswith("/precomputed_segmentations/info"):
            return self._get_segmentation_info(path)

        if not path.endswith("/info"):
            raise FileNotFoundError(f"Info file not found: {path}")

        trk_path = path[:-5]
        segments, bbox, offsets, _ = self._read_from_file(trk_path)

        # Info file for Neuroglancer
        info = tau.generate_info_dict(
            segments, bbox, offsets, self.grid_densities, sharding=False)
        info["relationships"].append({
            "id": "filter",
            "key": "./filter"})

        info_cache[path] = info

        # Cache spatials as well so once the layer is loaded we have them
        if trk_path not in spatial_cache:
            spatial = tau.get_spatials(segments, bbox, offsets,
                                self.grid_densities)
            spatial_cache[trk_path] = spatial
            
        return info
    
    def _get_segmentation_info(self, path: str) -> dict:
        if info_cache.get(path) is not None:
            return info_cache[path]
        
        if not path.endswith("/precomputed_segmentations/info"):
            raise FileNotFoundError(f"Info file not found: {path}")
        
        trk_path = path[:-len("/precomputed_segmentations/info")]

        _, bbox, _, _ = self._read_from_file(trk_path)

        info = tas.generate_info(1, bbox, (bbox[1] - bbox[0]).tolist())
        info_cache[path] = info

        return info
    
    def _get_segmentation(self, path: str) -> bytes:
        trk_path = "/".join(path.split("/")[:-3])
        _, bbox, _, _ = self._read_from_file(trk_path)
        dimensions = np.append((bbox[1] - bbox[0]), [1])
        return np.asarray(np.zeros(dimensions), dtype='<u8').tobytes()
   
    def _get_transform(self, path: str) -> np.ndarray:
        """Get affine transform from a given path."""
        if not path.endswith("/transform.lta"):
            raise FileNotFoundError(f"Invalid transform path: {path}")
        trk_path = path[:-14]
        _, _, _, affine = self._read_from_file(trk_path)
        return tau.lta_data(affine)
    
    def _get_annotations_inside(
            self, lines: np.ndarray, filter_layer: ng.Layer
            ) -> np.ndarray:
        """
        Find all streamlines in a list of annotations that intersect 
        with the filter layer.
        """
        source = filter_layer.source
        if isinstance(source, ng.viewer_state.LayerDataSources):
            transform = np.array(source[0].transform.to_json().get(
                "matrix", np.eye(4)[:-1]))
        else:
            transform = np.array(source.transform.to_json().get(
                "matrix", np.eye(4)[:-1]))
        line_list = []
        for annotation in filter_layer.annotations:
            if annotation.type == "axis_aligned_bounding_box":
                points = np.vstack([annotation.pointA, annotation.pointB])
                min_point = points.min(axis=0)
                max_point = points.max(axis=0)

                mask = (lines["start"] > min_point).all(axis=1) & (lines["start"] < 
                                                                   max_point).all(axis=1)
                lines_in = lines[mask]

            elif annotation.type == "ellipsoid":
                center = np.array(annotation.center.tolist() + [1])@transform.T
                radii = np.array(annotation.radii)@transform[:, :3].T

                start_transformed = (lines["start"] - center)/radii
                end_transformed = (lines["end"] - center)/radii

                zero_vectors = np.all(end_transformed == start_transformed, axis=1)

                end_transformed[zero_vectors] = start_transformed[zero_vectors] + \
                    [10**-5]*3

                l2 = np.sum((start_transformed - end_transformed)**2, axis=1)

                t = np.clip(np.einsum("ij, ij->i", -start_transformed, \
                                      end_transformed-start_transformed)/l2, 0.0, 1.0)

                mask = np.linalg.norm(start_transformed + np.stack((t, t, t), axis=1)*
                                      (end_transformed-start_transformed), axis=1) <= 1
                lines_in = lines[mask]

            else:
                continue

            if len(lines_in) > 0:
                line_list.append(lines_in["streamline"].reshape((-1)))
        
        if len(line_list) > 0:
            return np.unique(np.concatenate(line_list))
        return np.array([], dtype=int)


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

    # NOTE: Connection is a hop-by-hop header and should therefore not
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
