import numpy as np
from types import GeneratorType as generator
from neuroglancer import CoordinateSpace
from neuroglancer.skeleton import SkeletonSource, Skeleton, VertexAttributeInfo


class Polygons(SkeletonSource):

    def __init__(self):
        super().__init__(CoordinateSpace(
            names=["x", "y", "z"],
            units="mm",
            scales=[1, 1, 1],
        ))
        self.paths = {}

    def add_polygon(self, positions, segment=1):
        self.paths.setdefault(segment, [])
        self.paths[segment].append(positions)

    def addto_polygon(self, position, segment=1, polygon=-1):
        self.paths.setdefault(segment, [])
        self.patths[segment][-1].append(position)
