from nibabel.streamlines.trk import TrkFile
import nibabel as nib
import numpy as np
import fsspec
import math
import random
import json


TRK = "https://dandiarchive.s3.amazonaws.com/blobs/d4a/c43/d4ac43bd-6896-4adf-a911-82edbea21f67"


with fsspec.open(TRK) as f:
    trk = TrkFile.load(f)


def make_sharding_info(nb_chunks: int) -> dict:
    preshift_bits = 12
    minishard_bits = int(math.ceil(math.log2(math.ceil(
        nb_chunks / 2**preshift_bits
    ))))
    return {
        "@type": "neuroglancer_uint64_sharded_v1",
        "hash": "identity",
        "preshift_bits": preshift_bits,
        "minishard_bits": minishard_bits,
        "shard_bits": 0,
        "minishard_index_encoding": "raw",
        "data_encoding": "raw",
    }


def make_info(verts, edges, scalars, spatial: list[dict]) -> dict:


    nb_streamlines = len(verts)
    lb = np.min(np.stack([np.min(x, axis=0) for x in verts]), axis=0)
    ub = np.max(np.stack([np.max(x, axis=0) for x in verts]), axis=0)

    return {
        "@type": "neuroglancer_annotations_v1",
        "dimensions": {"x": [1, "mm"], "y": [1, "mm"], "z": [1, "mm"]},
        "lower_bound": lb.tolist(),
        "upper_bound": ub.tolist(),
        "annotation_type": "LINE",
        "properties": [
            {
                "id": "orientation_color",
                "type": "rgb",
                "description": "Color-coding of the segment orientation",
            },
            {
                "id": "orientation_x",
                "type": "float32",
                "description": "Segment orientation (x)",
            },
            {
                "id": "orientation_y",
                "type": "float32",
                "description": "Segment orientation (y)",
            },
            {
                "id": "orientation_z",
                "type": "float32",
                "description": "Segment orientation (z)",
            },
            *[
                {
                    "id": key,
                    "type": "float32",
                }
                for key in scalars.keys()
            ]
        ],
        "relationships": [
            {
                "id": "tract",
                "key": "by_tract",
                "sharding": make_sharding_info(len(edges)),
            },
        ],
        "by_id": {
            "key": "by_id",
            "sharding": make_sharding_info(sum(map(len, edges))),
        },
        "spatial": spatial,
        # [
        #     # coarsest level
        #     {
        #     "key": "0",
        #     "sharding": { ... },
        #     "grid_shape": [1, 1, 1],
        #     "chunk_size": upper_bound - lower_bound,
        #     "limit": compute_limit(0, ...),

        #     },
        #     # finer level
        #     {
        #     "key": f"{nb_levels}",
        #     "sharding": { ... },
        #     "grid_shape": [ ... ],
        #     "chunk_size": [ ... ],
        #     "limit": compute_limit(nb_levels, ...),
        #     },
        # ],
        }


def make_spatial(edges, orient, scalars, mode="local") -> list[dict]:
    grid = [1, 1, 1]
    while True:



def shuffle(x: list) -> list:
    """Deterministic shuffling (makes a copy)"""
    x = list(x)
    random.seed(1234)
    random.shuffle(x)
    return x


def to_annot(inp: str, out: str, mode="local"):

    with fsspec.open(inp) as f:
        trk = TrkFile.load(f)

    verts = shuffle(trk.streamlines)
    scalars = {
        key: shuffle(val)
        for key, val in trk.tractogram.data_per_point.items()
    }

    edges = [np.stack([x[:-1:2], x[1::2]]) for x in verts]
    orient = [x[:, 1, :] - x[:, 0, :] for x in verts]
    orient = [x / (x*x).sum(-1, keepaxis=True)**0.5 for x in verts]
    scalars = {
        k: [(x[:-1:2] + x[1::2]) / 2 for x in v]
        for k, v in zip(scalars.items())
    }

    spatial_grids, spatial_info = make_spatial(edges, orient, scalars, mode)

    info = make_info(verts, edges, scalars, spatial_info)

    with open(f"{out}/info", "wb") as f:
        json.dump(info, f)

    with open(f"{out}/by_id/0.shard", "wb") as f:
        f.write(dump_by_id(verts, orient, scalars), f)

    for i, grid in enumerate(spatial_grids):
        with open(f"{out}/{i}/0.shard", "wb") as f:
            f.write(dump_spatial(grid, verts, orient, scalars), f)
