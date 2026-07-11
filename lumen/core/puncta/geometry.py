"""Shared geometry calculations and morphology helpers.

This module is the designated shared geometry layer for Project Lumen's 
puncta quantification pipeline. Future morphological features (such as 
equivalent diameter, area, perimeter, circularity, and bounding boxes) 
should be implemented here.
"""

import numpy as np
import scipy.ndimage as ndimage
from dataclasses import dataclass
from typing import Mapping, Tuple

@dataclass(frozen=True, slots=True)
class LabelGeometry:
    """Morphological and spatial properties of a labeled region."""
    label_id: int
    area: int
    perimeter: float
    equivalent_diameter: float
    bounding_box: Tuple[int, int, int, int]  # (min_row, min_col, height, width)
    centroid: Tuple[float, float]             # (centroid_y, centroid_x)
    aspect_ratio: float                       # width / height

def compute_label_centroids(
    labels: np.ndarray,
    object_ids: np.ndarray
) -> Mapping[int, Tuple[float, float]]:
    """Computes centroids for all given labeled regions.
    
    Parameters:
        labels: 2D integer label array.
        object_ids: 1D array of object IDs to calculate.
        
    Returns:
        A Mapping from object ID (int) -> (centroid_y, centroid_x)
    """
    if len(object_ids) == 0:
        return {}
    binary_mask = labels > 0
    if len(object_ids) == 1:
        centroid_val = ndimage.center_of_mass(binary_mask, labels, object_ids[0])
        return {int(object_ids[0]): (float(centroid_val[0]), float(centroid_val[1]))}
    else:
        centroids_vals = ndimage.center_of_mass(binary_mask, labels, object_ids)
        return {
            int(oid): (float(c[0]), float(c[1])) for oid, c in zip(object_ids, centroids_vals)
        }

def compute_region_geometry(
    labels: np.ndarray, 
    object_ids: np.ndarray
) -> Mapping[int, LabelGeometry]:
    """Computes all spatial and morphological features in a single pass.
    
    Why this works:
        - Centroids are computed using scipy.ndimage.center_of_mass on labels > 0.
        - Bounding box slices are computed using scipy.ndimage.find_objects.
        - Perimeter counts background-bordering pixel edges (discrete pixel perimeter).
        
    Parameters:
        labels: 2D integer label array.
        object_ids: 1D array of object IDs to calculate.
        
    Returns:
        A Mapping from object ID (int) -> LabelGeometry.
    """
    if len(object_ids) == 0:
        return {}

    # 1. Compute Centroids
    binary_mask = labels > 0
    if len(object_ids) == 1:
        centroid_val = ndimage.center_of_mass(binary_mask, labels, object_ids[0])
        centroids_dict = {int(object_ids[0]): (float(centroid_val[0]), float(centroid_val[1]))}
    else:
        centroids_vals = ndimage.center_of_mass(binary_mask, labels, object_ids)
        centroids_dict = {
            int(oid): (float(c[0]), float(c[1])) for oid, c in zip(object_ids, centroids_vals)
        }

    # 2. Compute Bounding Box slices, areas, perimeters, equivalent diameters
    slices = ndimage.find_objects(labels)
    results = {}

    for oid in object_ids:
        if oid <= 0 or oid > len(slices):
            continue
        slice_tup = slices[oid - 1]
        if slice_tup is None:
            continue

        # Extract local sub-image containing only the spot bounding box
        local_labels = labels[slice_tup]
        local_mask = local_labels == oid

        # Area
        area = int(np.sum(local_mask))

        # Perimeter (background-bordering edges)
        # Pad local mask to handle boundary cells correctly
        padded = np.pad(local_mask, 1, mode='constant', constant_values=False)
        perimeter = float(
            np.sum(padded[1:-1, 1:-1] & ~padded[2:, 1:-1]) +
            np.sum(padded[1:-1, 1:-1] & ~padded[:-2, 1:-1]) +
            np.sum(padded[1:-1, 1:-1] & ~padded[1:-1, 2:]) +
            np.sum(padded[1:-1, 1:-1] & ~padded[1:-1, :-2])
        )

        # Equivalent Diameter
        eq_diam = float(2.0 * np.sqrt(area / np.pi))

        # Bounding Box (min_row, min_col, height, width)
        min_row = int(slice_tup[0].start)
        max_row = int(slice_tup[0].stop - 1)
        min_col = int(slice_tup[1].start)
        max_col = int(slice_tup[1].stop - 1)
        height = max_row - min_row + 1
        width = max_col - min_col + 1
        bbox = (min_row, min_col, height, width)

        # Aspect Ratio
        aspect_ratio = float(width / height) if height > 0 else 0.0

        # Retrieve Centroid
        centroid = centroids_dict.get(oid, (0.0, 0.0))

        results[int(oid)] = LabelGeometry(
            label_id=int(oid),
            area=area,
            perimeter=perimeter,
            equivalent_diameter=eq_diam,
            bounding_box=bbox,
            centroid=centroid,
            aspect_ratio=aspect_ratio
        )

    return results
