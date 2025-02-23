import numpy as np
import os
from dipy.io.streamline import load_trk
import nibabel as nib
import time
import json
from utils import generate_colors, convert_to_native, log_resource_usage


WORLD_SPACE_DIMENSION = 1

# Start timing
start_time = time.time()

# Load NIfTI file to get spatial properties
nifti_file = 'assets/sub-I58_sample-hemi_desc-preproc_dwi_FA.nii.gz'
nifti = nib.load(nifti_file)
voxel_size = nifti.header.get_zooms()[:3]  # Get voxel dimensions
volume_shape = nifti.shape[:3]  # Volume dimensions
affine = nifti.affine  # Affine transformation matrix
inverse_affine = np.linalg.inv(affine)  # Inverse affine
print(f"Voxel size: {voxel_size}, Volume shape: {volume_shape}, Affine: \n{affine}")

# Load streamlines
trk_file = 'assets/sub-I58_sample-hemi_desc-CSD_tractography.smalltest.trk'
print("Loading streamlines...")
sft = load_trk(trk_file, reference='same')
all_streamlines = sft.streamlines
streamline_colors = generate_colors(len(all_streamlines))
print(f"Total number of streamlines: {len(all_streamlines)}")
log_resource_usage("After Loading Streamlines")

# Output directory
output_dir = './precomputed_annotations_new'
os.makedirs(output_dir, exist_ok=True)
spatial_dir = os.path.join(output_dir, 'spatial0')
os.makedirs(spatial_dir, exist_ok=True)

# Define grid shape and chunk size dynamically
grid_density = 6  # Number of chunks along each axis
chunk_size = [dim // grid_density for dim in volume_shape]
# grid_shape = [grid_density] * 3
grid_shape = [1, 1, 1]

# Create binary spatial files
spatial_index = {f"{x}_{y}_{z}": [] for x in range(grid_shape[0]) for y in range(grid_shape[1]) for z in range(grid_shape[2])}

annotation_count = 0  # Track total annotation count

lb = np.min(np.stack([np.min(streamline, axis=0) for streamline in all_streamlines]), axis=0)
ub = np.max(np.stack([np.max(streamline, axis=0) for streamline in all_streamlines]), axis=0)

for streamline in all_streamlines:
    streamline = np.array(streamline)

    # Convert streamline coordinates to NIfTI voxel space
    # streamline_voxels = apply_affine(inverse_affine, streamline)
    # streamline_voxels = np.clip(streamline_voxels, 0, np.array(volume_shape) - 1)  # Clip to bounds

    # Divide streamline into line segments
    for i in range(len(streamline) - 1):

        start = streamline[i]
        end = streamline[i + 1]

        # Determine grid cell
        cell_x = int(int(grid_shape[0] * (start[0] - lb[0]) / (ub[0] - lb[0])))
        cell_y = int(int(grid_shape[1] * (start[1] - lb[1]) / (ub[1] - lb[1])))
        cell_z = int(int(grid_shape[2] * (start[2] - lb[2]) / (ub[2] - lb[2])))

        cell_x = min(grid_shape[0] - 1, cell_x)
        cell_y = min(grid_shape[1] - 1, cell_y)
        cell_z = min(grid_shape[2] - 1, cell_z)

        cell_key = f"{cell_x}_{cell_y}_{cell_z}"

        # Add annotation to spatial index
        spatial_index[cell_key].append((start, end))
        annotation_count += 1

# Save spatial index files in binary format
for cell_key, annotations in spatial_index.items():
    if len(annotations) > 0:
        cell_file = os.path.join(spatial_dir, cell_key)
        with open(cell_file, 'wb') as f:
            # Write number of annotations as countLow and countHigh
            # f.write(struct.pack('<II', len(annotations), 0))  # Little-endian uint32
            f.write(np.asarray(len(annotations), dtype='<u8').tobytes())

            for start, end in annotations:

                # Write start and end points as float32
                f.write(np.asarray(start, dtype='<f4').tobytes())
                f.write(np.asarray(end, dtype='<f4').tobytes())

            for annotation_id in range(len(annotations)):
                f.write(np.asarray(annotation_id, dtype='<u8').tobytes())  # Write ID as uint64le

        print(f"Saved spatial index for {cell_key} with {len(annotations)} annotations.")

# Save info file
info = {
    "@type": "neuroglancer_annotations_v1",
    "dimensions": {
        "x": [WORLD_SPACE_DIMENSION, "mm"],
        "y": [WORLD_SPACE_DIMENSION, "mm"],
        "z": [WORLD_SPACE_DIMENSION, "mm"]
    },
    "lower_bound": lb.tolist(),
    "upper_bound": ub.tolist(),
    "annotation_type": "LINE",
    "properties": [],
    "relationships": [],
    "by_id": {"key": "annotations_by_id"},
    "spatial": [
        {
            "key": "spatial0",
            "grid_shape": grid_shape,
            "chunk_size": chunk_size,
            "limit": 5000000000
        }
    ]
}
info_file_path = os.path.join(output_dir, 'info')
with open(info_file_path, 'w') as f:
    json.dump(convert_to_native(info), f)
print(f"Saved info file at {info_file_path}")

log_resource_usage("After Formatting Annotations")

# Final metrics
end_time = time.time()
print(f"Script completed in {end_time - start_time:.2f} seconds.")
log_resource_usage("Final Resource Utilization")