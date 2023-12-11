import numpy as np
from types import GeneratorType as generator
from neuroglancer import CoordinateSpace
from neuroglancer.skeleton import SkeletonSource, Skeleton, VertexAttributeInfo
from nibabel.streamlines.trk import TrkFile
from nibabel.streamlines.tck import TckFile
import random
import fsspec


DEFAULT_MAX_TRACTS = 10000


class TractSource(SkeletonSource):
    """
    This class reads a TRK stremalines file (using nibabel)
    and implements methods that allow serving and/or converting the
    streamlines data into neuroglancer's precomputed skeleton format.

    The skeleton format was originally implemented to render skeletonized
    volumetric segmentations, with a relatively small number of
    individual objects. Since tractography (especially high-resolution
    tractography) can generate millions of streamlines, saving each
    streamline as an individual skeleton is very ineficient -- both for
    querying and rendering.

    Instead, we combine all streamlines into a single skeleton. Still,
    the large number of streamlines slows down rendering quite a lot, so
    I currently sample `MAX_TRACTS` streamlines from the file to display
    (currently set to 10,000).

    There is no "multi-scale" representation of the tracts in neuroglancer
    (i.e., generate tracts with a smaller or larger number of edges
    based on the current rendering scale). Maybe this is something that
    we should ask be added to neuroglancer.

    If a segmentation of the tractogram is available, it would probably
    make sense to save the tracts belonging to each segment in a
    different skeleton. This would allow switching each segment on and
    off, and would allow segments to be displayed in different colors.

    I also save a unit-norm orientation vector for each vertex. Saving
    this information as `vertex_attribute` allows using it for rendering
    (e.g., it can be used to render orientation-coded tracts).

    The specification of the precomputed skeleton format is available here:
        https://github.com/google/neuroglancer/blob/master/
        src/neuroglancer/datasource/precomputed/skeletons.md
    """

    def __init__(self, fileobj, max_tracts=DEFAULT_MAX_TRACTS):
        """
        Parameters
        ----------
        fileobj : path or file-like
            TCK or TRK file
        max_tracts : int
            Maximum number of tracts to display
        """
        self.fileobj = fileobj
        self.max_tracts = max_tracts
        self.tractfile = None
        self.displayed_ids = None
        self.displayed_tracts = None
        self.displayed_orientations = None
        self._load(lazy=True)
        super().__init__(CoordinateSpace(
            names=["x", "y", "z"],
            units="mm",
            scales=[1, 1, 1],
        ))
        self.vertex_attributes["orientation"] = VertexAttributeInfo(
            data_type=np.float32,
            num_components=3,
        )

    def __getitem__(self, id):
        """Get a single tract"""
        if not self._is_fully_loaded():
            self._load()
        return self.tractfile.streamlines[id]

    def __len__(self):
        """Total number of tracts"""
        if not self._is_fully_loaded():
            self._load()
        return len(self.tractfile.streamlines)

    def _load(self, lazy=False):
        """Load tracts from file (if `lazy=True`, only load metadata)"""
        def load(f):
            trk_error = tck_error = None
            try:
                self.tractfile = TrkFile.load(f, lazy_load=lazy)
                return
            except Exception as e:
                trk_error = e
            try:
                self.tractfile = TckFile.load(f, lazy_load=lazy)
                return
            except Exception as e:
                tck_error = e
            raise RuntimeError(f'trk: {trk_error.message}\n'
                               f'tck: {tck_error.message}')

        if isinstance(self.fileobj, str):
            with fsspec.open(self.fileobj) as f:
                load(f)
        else:
            load(f)

    def _is_fully_loaded(self):
        return (self.tractfile and
                not isinstance(self.tractfile.streamlines, generator))

    def _filter(self):
        """Select `max_tracts` random tracts"""
        if not self._is_fully_loaded():
            self._load()
        num_tracts = len(self.tractfile.streamlines)
        ids = list(range(num_tracts))
        random.seed(1234)
        random.shuffle(ids)
        self.displayed_ids = ids[:self.max_tracts]

    def _make_skeleton(self):
        """
        Merge all (filtered) streamlines into a single skeleton

        Returns
        ----------
        vertices : (N, 3) array
        edges : (M, 2) array
        orientation : (N, 3) array
        """
        if not self.displayed_ids:
            self._filter()

        num_vertices = num_edges = 0
        vertices = []
        edges = []
        orientations = []

        for id in self.displayed_ids:
            tract = self[id]
            orient = self.compute_orientation(tract)

            # vertex_positions: [num_vertices, 3] float32le (C-order)
            vertices.append(tract)
            # edges: [num_edges, 2] uint32le (C-order)
            edges.append(np.stack([
                np.arange(len(tract) - 1, dtype='<u4') + num_vertices,
                np.arange(1, len(tract), dtype='<u4') + num_vertices,
            ], -1))
            # orientations: [num_vertices, 3] float32le (C-order)
            orientations.append(np.asarray(orient, dtype='<f4'))
            # increase counters
            num_vertices += len(tract)
            num_edges += len(tract) - 1

        vertices = np.concatenate(vertices)
        edges = np.concatenate(edges)
        orientations = np.concatenate(orientations)

        return vertices, edges, orientations

    @classmethod
    def compute_orientation(cls, tract):
        """Compute the orientation of a tract at each vertex"""
        # 1) compute directed orientation of each edge
        orient = tract[1:] - tract[:-1]
        # 2) compute directed orientation of each vertex as the
        #    length-weighted average of the orientations of its edges
        #    (we just add them, and normalize later)
        orient = np.concatenate([
            orient[:1],                 # first vertex:   only one edge
            orient[:-1] + orient[1:],   # other vertices: two edges
            orient[-1:],                # last vertex:    only one edge
        ], 0)
        # 3) make orientations unit-length
        length = np.sqrt((orient * orient).sum(-1, keepdims=True))
        length = np.clip(length, 1e-12, None)
        orient /= length
        return orient

    def get_skeleton(self, i):
        """
        Neuroglancer Python API

        Parameters
        ----------
        i : int
            Segment index (should be zero in our case)

        Returns
        -------
        skel : neuroglancer.Skeleton
        """
        if i != 1:
            raise ValueError('Unknown segment id')
        self._filter()
        vertices, edges, orients = self._make_skeleton()
        return Skeleton(vertices, edges, dict(orientation=orients))

    def precomputed_prop_info(self, combined=False):
        """
        "NG precomputed format" : property info

            https://github.com/google/neuroglancer/blob/master/
            src/neuroglancer/datasource/precomputed/segment_properties.md
        """
        self._filter()
        num_tracts = len(self.displayed_ids)

        # if not combined, each tract correspond to a different "segment"
        # if combined, all tracts are merged into a single skeleton
        ids = ["1"] if combined else [str(i) for i in range(num_tracts)]

        info = {
            "@type": "neuroglancer_segment_properties",
            "inline": {
                "ids": ids,
                "properties": [{
                    "id": "label",
                    "type": "label",
                    "values": ids,
                }]
            }
        }
        return info

    def precomputed_skel_info(self):
        """
        "NG precomputed format" : skeleton info

            https://github.com/google/neuroglancer/blob/master/
            src/neuroglancer/datasource/precomputed/skeletons.md
        """
        if not self.tractfile:
            self._load(lazy=True)

        # No need for a transform, as nibabel automatically converts
        # TRK coordinates to mm RAS+
        # All we need to do is convert mm to nm (we do this in the
        # track serving functions)

        info = {
            "@type": "neuroglancer_skeletons",
            "vertex_attributes": [
                {
                    "id": "orientation",
                    "data_type": "float32",
                    "num_components": 3,
                },
            ],
            "segment_properties": "prop",
        }

        return info

    def precomputed_skel_tract_combined(self, id=1):
        """
        "NG precomputed format" : skeleton data (combined)

        `id` should alway be `1` (since there is only one skeleton)

        TODO: combine tracts per segment if a tractogram segmentation
        is available

            https://github.com/google/neuroglancer/blob/master/
            src/neuroglancer/datasource/precomputed/skeletons.md
            #encoded-skeleton-file-format
        """
        if id != 1:
            return b''
        self._filter()

        num_vertices = num_edges = 0
        vertices = b''
        edges = b''
        orientations = b''

        for id in range(self.displayed_ids):
            tract = self.displayed_tracts[id] * 1E6  # nanometer
            v, e, o = self.get_skeleton(tract)
            # vertex_positions: [num_vertices, 3] float32le (C-order)
            vertices += np.asarray(v, dtype='<f4').tobytes()
            # edges: [num_edges, 2] uint32le (C-order)
            edges += np.asarray(e, dtype='<f4').tobytes()
            # orientations: [num_vertices, 3] float32le (C-order)
            orientations += np.asarray(o, dtype='<f4').tobytes()
            # increase counters
            num_vertices += len(tract)
            num_edges += len(tract) - 1

        bintract = b''
        # num_vertices: uint32le
        bintract += np.asarray(num_vertices, dtype='<u4').tobytes()
        # edges: uint32le
        bintract += np.asarray(num_edges, dtype='<u4').tobytes()
        # vertex_positions: [num_vertices, 3] float32le (C-order)
        bintract += vertices
        # edges: [num_edges, 2] uint32le (C-order)
        bintract += edges
        # attributes | orientation: [num_vertices, 3] float32le (C-order)
        bintract += orientations
        return bintract

    def precomputed_skel_tract(self, id):
        """
        Return a single tract

        This function is used in the case where each streamline is
        encoded by a single skeleton. Note that this is very inefficient.

        TODO: add orientation attribute
        """
        self._filter()
        tract = self.displayed_tracts[id] * 1E6

        bintract = b''
        # num_vertices: uint32le
        bintract += np.asarray(len(tract), dtype='<u4').tobytes()
        # edges: uint32le
        bintract += np.asarray(len(tract) - 1, dtype='<u4').tobytes()
        # vertex_positions: [num_vertices, 3] float32le (C-order)
        bintract += np.asarray(tract, dtype='<f4').tobytes()
        # edges: [num_edges, 2] uint32le (C-order)
        bintract += np.stack([
            np.arange(len(tract) - 1, dtype='<u4'),
            np.arange(1, len(tract), dtype='<u4')
        ], -1).tobytes()

        print('serve tract', id, '/', len(self.displayed_ids))
        return bintract
