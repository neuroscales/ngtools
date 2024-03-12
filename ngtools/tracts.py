import numpy as np
from types import GeneratorType as generator
from neuroglancer import CoordinateSpace
from neuroglancer.skeleton import SkeletonSource, Skeleton, VertexAttributeInfo
from nibabel.streamlines.trk import TrkFile
from nibabel.streamlines.tck import TckFile
import random
import fsspec


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

    DEFAULT_MAX_TRACTS = 1000

    def __init__(self, fileobj, max_tracts=DEFAULT_MAX_TRACTS, format=None,
                 **kwargs):
        """
        Parameters
        ----------
        fileobj : path or file-like
            TCK or TRK file
        max_tracts : int
            Maximum number of tracts to display
        format : [list of] {'tck', 'trk'}
            Format hint
        """
        self.fileobj = fileobj
        self.max_tracts = max_tracts

        format = format or ['trk', 'tck']
        if not isinstance(format, (list, tuple)):
            format = [format]
        format = list(format)
        if 'trk' not in format:
            format += ['trk']
        if 'tck' not in format:
            format += ['tck']
        self.format = format

        self.tractfile = None
        self.displayed_ids = None
        self.displayed_tracts = None
        self.displayed_orientations = None

        self._ensure_loaded(lazy=True)
        super().__init__(CoordinateSpace(
            names=["x", "y", "z"],
            units="mm",
            scales=[1, 1, 1],
        ), **kwargs)
        self.vertex_attributes["orientation"] = VertexAttributeInfo(
            data_type=np.float32,
            num_components=3,
        )

    def __getitem__(self, id):
        """Get a single tract"""
        self._ensure_loaded()
        return self.tractfile.streamlines[id]

    def __len__(self):
        """Total number of tracts"""
        self._ensure_loaded()
        return len(self.tractfile.streamlines)

    def _ensure_loaded(self, lazy=False):
        """Load tracts from file (if `lazy=True`, only load metadata)"""
        if self._is_loaded(lazy):
            return

        klasses = dict(tck=TckFile, trk=TrkFile)

        def load(f):
            for format in self.format:
                klass = klasses[format]
                errors = {}
                try:
                    self.tractfile = klass.load(f, lazy_load=lazy)
                    return
                except Exception as e:
                    errors[format] = e

            print('error', *errors.values())
            raise RuntimeError('\n'.join(
                [f'{format}: {errors[format]}' for format in self.format]))

        if isinstance(self.fileobj, str):
            with fsspec.open(self.fileobj) as f:
                load(f)
        else:
            load(f)

    def _is_loaded(self, lazy=False):
        """Check if file is loaded"""
        if not self.tractfile:
            # not loaded at all
            return False
        if isinstance(self.tractfile.streamlines, generator):
            # lazilty loaded
            return lazy
        return True

    def _filter(self):
        """Select `max_tracts` random tracts"""
        self._ensure_loaded()
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

            vertices.append(tract.astype('<f4'))
            edges.append(np.stack([
                np.arange(len(tract) - 1, dtype='<u4') + num_vertices,
                np.arange(1, len(tract), dtype='<u4') + num_vertices,
            ], -1))
            orientations.append(orient.astype('<f4'))
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

    def get_skeleton(self, i=1):
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
        # if i != 1:
        #     raise ValueError('Unknown segment id')
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
        self._ensure_loaded(lazy=True)

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

    def precomputed_skel_data(self, id=1, combined=True):
        if combined:
            return self._precomputed_skel_data_combined(id)
        else:
            return self._precomputed_skel_data_single(id)

    def _precomputed_skel_data_combined(self, id=1):
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

        # skeleton format:
        # num_vertices:     uint32le
        # edges:            uint32le
        # vertex_positions: float32le [num_vertices, 3]  (C-order)
        # edges:            uint32le  [num_edges,    2]  (C-order)
        # attr|orientation: float32le [num_vertices, 3]  (C-order)

        num_vertices = num_edges = 0
        vertices = b''
        edges = b''
        orientations = b''

        for id in self.displayed_ids:
            tract = self[id] * 1E6  # nanometer
            v, e, o = self.get_skeleton(tract)
            vertices += np.asarray(v, dtype='<f4').tobytes()
            edges += np.asarray(e, dtype='<f4').tobytes()
            orientations += np.asarray(o, dtype='<f4').tobytes()
            num_vertices += len(tract)
            num_edges += len(tract) - 1

        bintract = b''
        bintract += np.asarray(num_vertices, dtype='<u4').tobytes()
        bintract += np.asarray(num_edges, dtype='<u4').tobytes()
        bintract += vertices
        bintract += edges
        bintract += orientations
        return bintract

    def _precomputed_skel_data_single(self, id):
        """
        Return a single tract

        This function is used in the case where each streamline is
        encoded by a single skeleton. Note that this is very inefficient.

        TODO: add orientation attribute
        """
        self._filter()

        # skeleton format:
        # num_vertices:     uint32le
        # edges:            uint32le
        # vertex_positions: float32le [num_vertices, 3]  (C-order)
        # edges:            uint32le  [num_edges,    2]  (C-order)
        # attr|orientation: float32le [num_vertices, 3]  (C-order)

        tract = self.displayed_tracts[id] * 1E6
        v, e, o = self.get_skeleton(tract)

        bintract = b''
        bintract += np.asarray(len(v), dtype='<u4').tobytes()
        bintract += np.asarray(len(e), dtype='<u4').tobytes()
        bintract += np.asarray(v, dtype='<f4').tobytes()
        bintract += np.asarray(e, dtype='<u4').tobytes()
        bintract += np.asarray(o, dtype='<f4').tobytes()

        print('serve tract', id, '/', len(self.displayed_ids))
        return bintract
