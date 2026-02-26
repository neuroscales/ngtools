"""Handler for ROI intersections."""

import numpy as np


class ROIHandler():
    """Handle ROI filters."""

    def __init__(self, highest_density:int) -> None:
        self.highest_density = highest_density

    def _get_bbox_intersections(
        self,
        min_grid: np.ndarray,
        max_grid: np.ndarray,
        spatial: np.ndarray
    ) -> np.ndarray:
        """Find what segments are in the locations that collide with bbox."""
        min_grid = np.clip(min_grid, 0, self.highest_density)
        max_grid = np.clip(max_grid, -1, self.highest_density-1)

        grid_list = []
        for i in range(min_grid[0], max_grid[0] + 1):
            for j in range(min_grid[1], max_grid[1] + 1):
                for k in range(min_grid[2], max_grid[2] + 1):
                    if len(spatial[f"{i}_{j}_{k}"]) > 0:
                        grid_list.append(spatial[f"{i}_{j}_{k}"])

        seg_in_grid = np.array([])
        if len(grid_list) > 0:
            seg_in_grid = np.concatenate(grid_list)
        return seg_in_grid
    
    def axis_aligned_bounding_box(
            self, 
            bounding_box_filter:np.ndarray, 
            transform:np.ndarray, 
            bbox:np.ndarray, 
            dimensions:np.ndarray, 
            spatial:np.ndarray, 
            segments:np.ndarray
            ) -> np.ndarray:
        """Find streamlines that intersect with ellipsoid ROI."""
        # I have decided to treat each line segment as a bounding box.
        # This makes collision a lot easier to detect.
        # And because line segments are so small 
        # it will rarely lead to false positives
        box_points = np.vstack([
            (bounding_box_filter.pointA.tolist() + [1])@transform.T,
            (bounding_box_filter.pointB.tolist() + [1])@transform.T])
        min_box_point = box_points.min(axis=0)
        max_box_point = box_points.max(axis=0)

        min_grid = np.asarray(((min_box_point - bbox[0])*
                                self.highest_density
                                )//dimensions, dtype=int)
        max_grid = np.asarray(((max_box_point - bbox[0])*
                                self.highest_density
                                )//dimensions, dtype=int)
        
        seg_in_grid = self._get_bbox_intersections(min_grid, max_grid, spatial)

        if len(seg_in_grid) > 0:
            line_min = np.minimum(seg_in_grid["start"], seg_in_grid["end"])
            line_max = np.maximum(seg_in_grid["start"], seg_in_grid["end"])

            mask = np.all(line_min <= max_box_point, axis=1) & \
                np.all(line_max >= min_box_point, axis=1)

            return np.unique(segments[mask]["streamline"].reshape(-1))
        return np.array([], dtype=int)
        
    def ellipsoid(
            self, 
            elipsoid_filter:np.ndarray, 
            transform:np.ndarray, 
            bbox:np.ndarray, 
            dimensions:np.ndarray, 
            spatial:np.ndarray, 
            segments:np.ndarray
            ) -> np.ndarray:
        """Find streamlines that intersect with ellipsoid ROI."""
        center = np.array(elipsoid_filter.center.tolist() + [1])@transform.T
        radii = np.abs(np.array(elipsoid_filter.radii)@transform[:, :3].T)

        min_grid = np.asarray((((center - radii) - bbox[0])*
                    self.highest_density)//dimensions, dtype=int)
        max_grid = np.asarray((((center + radii) - bbox[0])*
                    self.highest_density)//dimensions, dtype=int)
        
        seg_in_grid = self._get_bbox_intersections(min_grid, max_grid, spatial)

        if len(seg_in_grid) > 0:
            start_transformed = (segments["start"] - center)/radii
            end_transformed = (segments["end"] - center)/radii

            zero_vectors = np.all(end_transformed == start_transformed, 
                                    axis=1)

            end_transformed[zero_vectors] = start_transformed[
                zero_vectors] + [10**-5]*3

            l2 = np.sum((start_transformed - end_transformed)**2, axis=1)

            t = np.clip(np.einsum("ij, ij->i", -start_transformed, 
                                end_transformed-start_transformed)/l2,
                                0.0, 1.0)

            mask = np.linalg.norm(start_transformed + 
                                np.stack((t, t, t), axis=1)*
                                (end_transformed-start_transformed),
                                axis=1) <= 1
            return np.unique(segments[mask]["streamline"].reshape(-1))
        return np.array([], dtype=int)