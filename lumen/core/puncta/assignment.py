"""Cell assignment logic for associating detected puncta with parent cells.

This module resolves spatial relationships, mapping coordinates of detected 
puncta to the labeled regions of Cellpose cell segmentation masks.
"""

import numpy as np
from lumen.core.puncta.types import PunctaDetectionResult, PunctaAssignmentResult
from lumen.core.puncta.geometry import compute_label_centroids

class PunctaAssigner:
    """Orchestrates assigning localized spots to target segmented cell masks.
    
    Acts as a mapping engine between the spatial outputs of PunctaDetector 
    and the final cellular regions defined by cell segmentations.
    """

    def assign(self,
               cell_labels: np.ndarray,
               detection: PunctaDetectionResult) -> PunctaAssignmentResult:
        """Maps each detected punctum to its parent cell ID.
        
        This method operates as a pure function:
        1. Validates inputs, shapes, integer subdtypes, and non-negativity.
        2. Asserts unique labels in the detection result match detection.object_ids.
        3. Computes centroids of the labeled puncta via compute_label_centroids.
        4. Intersects coordinates (with deterministic integer truncation) with cell labels.
        5. Populates and sorts the output assignment mappings.
        
        Parameters:
            cell_labels: 2D integer array representing labeled cell masks (0 = background).
            detection: A PunctaDetectionResult object.
            
        Returns:
            A PunctaAssignmentResult containing cell_to_puncta, punctum_to_cell,
            and unassigned_puncta collections.
            
        Raises:
            TypeError: If input array or detection result types are invalid.
            ValueError: If shapes mismatch, labels are negative, or IDs are inconsistent.
        """
        # 1. Validation
        if not isinstance(cell_labels, np.ndarray):
            raise TypeError("cell_labels must be a numpy.ndarray")
        if not isinstance(detection, PunctaDetectionResult):
            raise TypeError("detection must be a PunctaDetectionResult instance")
            
        if cell_labels.ndim != 2:
            raise ValueError("cell_labels must be a 2D array")
        if cell_labels.shape != detection.labels.shape:
            raise ValueError(
                f"Dimension mismatch: cell_labels shape {cell_labels.shape} "
                f"does not match detection labels shape {detection.labels.shape}"
            )
        if not np.issubdtype(cell_labels.dtype, np.integer):
            raise TypeError("cell_labels must contain integer labels")
            
        # Verify cell_labels contains only non-negative integers
        if np.any(cell_labels < 0):
            raise ValueError("cell_labels must contain only non-negative integers")
            
        # Verify unique non-zero values in detection.labels match detection.object_ids
        unique_labels = np.unique(detection.labels)
        unique_labels = unique_labels[unique_labels > 0]
        if not np.array_equal(np.sort(unique_labels), np.sort(detection.object_ids)):
            raise ValueError(
                "detection.object_ids does not match the unique labels present in detection.labels"
            )

        # 2. Pre-populate cell_to_puncta dictionary with all positive Cell IDs from cell_labels
        unique_cells = np.unique(cell_labels)
        unique_cells = unique_cells[unique_cells > 0]
        cell_to_puncta = {int(c): [] for c in unique_cells}
        
        punctum_to_cell = {}
        unassigned_puncta = []

        # 3. Compute centroids (delegated to internal geometry helper)
        centroids_dict = compute_label_centroids(detection.labels, detection.object_ids)
        
        # 4. Perform assignment based on centroid coordinates
        height, width = cell_labels.shape
        for p_id in detection.object_ids:
            cy, cx = centroids_dict[p_id]
            
            # Truncate floats to integer pixel coordinates
            ry = min(max(int(cy), 0), height - 1)
            rx = min(max(int(cx), 0), width - 1)
            
            cell_id = int(cell_labels[ry, rx])
            if cell_id > 0:
                cell_to_puncta[cell_id].append(int(p_id))
                punctum_to_cell[int(p_id)] = cell_id
            else:
                unassigned_puncta.append(int(p_id))

        # 5. Sort lists to ensure deterministic output ordering
        for cell_id in cell_to_puncta:
            cell_to_puncta[cell_id].sort()
        unassigned_puncta.sort()

        return PunctaAssignmentResult(
            cell_to_puncta=cell_to_puncta,
            punctum_to_cell=punctum_to_cell,
            unassigned_puncta=unassigned_puncta
        )

