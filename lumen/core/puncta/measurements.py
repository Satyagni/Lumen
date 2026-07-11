"""Measurement engine for extracting quantitative features from puncta.

This module is responsible for analyzing raw signal values inside spot regions 
and aggregating metrics (counts, sizes, intensities) at the cell level.
"""

import numpy as np
import scipy.ndimage as ndimage
from lumen.core.puncta.types import PunctaDetectionResult, PunctaAssignmentResult
from lumen.core.puncta.results import PunctumMeasurement, PerCellPunctaSummary, PunctaResults
from lumen.core.puncta.geometry import compute_region_geometry

class PunctaMeasurer:
    """Measures morphological and intensity properties of detected and assigned puncta.
    
    This class extracts signal values from the raw microscopy image and matches 
    them with structural cell masks to form comprehensive analysis results.
    """

    def measure(self,
                image: np.ndarray,
                detection: PunctaDetectionResult,
                assignment: PunctaAssignmentResult,
                cell_labels: np.ndarray) -> PunctaResults:
        """Computes physical and signal metrics for all detected puncta.
        
        This method operates as a pure function:
        1. Validates inputs, shapes, integer subdtypes, and non-negativity.
        2. Computes geometries using the unified compute_region_geometry helper.
        3. Pre-computes cell mask areas in one pass using np.bincount.
        4. Extracts intensities strictly within the punctum mask (in float64 precision).
        5. Computes statistics and builds immutable PunctumMeasurement records.
        6. Aggregates and yields PerCellPunctaSummary statistics.
        
        Parameters:
            image: 2D numpy array representing the raw intensity channel image.
            detection: A PunctaDetectionResult object.
            assignment: A PunctaAssignmentResult object.
            cell_labels: 2D integer array representing labeled cell masks (0 = background).
            
        Returns:
            A PunctaResults object containing individual measurements and cell summaries.
            
        Raises:
            TypeError: If input array or result types are invalid.
            ValueError: If shapes mismatch, labels are negative, or IDs are inconsistent.
        """
        # 1. Validation
        if not isinstance(image, np.ndarray):
            raise TypeError("image must be a numpy.ndarray")
        if not isinstance(detection, PunctaDetectionResult):
            raise TypeError("detection must be a PunctaDetectionResult instance")
        if not isinstance(assignment, PunctaAssignmentResult):
            raise TypeError("assignment must be a PunctaAssignmentResult instance")
        if not isinstance(cell_labels, np.ndarray):
            raise TypeError("cell_labels must be a numpy.ndarray")
            
        if image.ndim != 2:
            raise ValueError("image must be a 2D array")
        if cell_labels.ndim != 2:
            raise ValueError("cell_labels must be a 2D array")
            
        if image.shape != detection.labels.shape or image.shape != cell_labels.shape:
            raise ValueError(
                f"Dimension mismatch: image shape {image.shape}, "
                f"detection shape {detection.labels.shape}, and "
                f"cell_labels shape {cell_labels.shape} must all match"
            )
            
        if not np.issubdtype(cell_labels.dtype, np.integer):
            raise TypeError("cell_labels must contain integer labels")
        if not np.issubdtype(detection.labels.dtype, np.integer):
            raise TypeError("detection labels must contain integer labels")
            
        # Verify cell_labels contains only non-negative integers
        if np.any(cell_labels < 0):
            raise ValueError("cell_labels must contain only non-negative integers")

        # 2. Geometry calculations
        geometry_dict = compute_region_geometry(detection.labels, detection.object_ids)
        slices = ndimage.find_objects(detection.labels)

        # 3. Pre-compute cell areas
        cell_areas = np.bincount(cell_labels.ravel())

        # 4. Puncta measurements computation
        puncta_list = []
        for p_id in detection.object_ids:
            geom = geometry_dict[p_id]
            cell_id = assignment.punctum_to_cell.get(p_id, 0)
            
            # Sub-image slice for intensity extraction
            bbox_slice = slices[p_id - 1]
            local_labels = detection.labels[bbox_slice]
            local_image = image[bbox_slice]
            
            # Exact mask extraction to prevent background contamination
            spot_mask = local_labels == p_id
            intensities = local_image[spot_mask]
            
            # Convert to float64 to ensure high-precision math accumulation
            intensities_f64 = intensities.astype(np.float64, copy=False)
            
            mean_intensity = float(np.mean(intensities_f64))
            median_intensity = float(np.median(intensities_f64))
            integrated_intensity = float(np.sum(intensities_f64))
            minimum_intensity = float(np.min(intensities_f64))
            maximum_intensity = float(np.max(intensities_f64))
            standard_deviation = float(np.std(intensities_f64)) if len(intensities_f64) > 1 else 0.0
            
            meas = PunctumMeasurement(
                punctum_id=int(p_id),
                cell_id=int(cell_id),
                area=float(geom.area),
                perimeter=float(geom.perimeter),
                equivalent_diameter=float(geom.equivalent_diameter),
                mean_intensity=mean_intensity,
                median_intensity=median_intensity,
                integrated_intensity=integrated_intensity,
                minimum_intensity=minimum_intensity,
                maximum_intensity=maximum_intensity,
                standard_deviation=standard_deviation,
                centroid_x=float(geom.centroid[1]),
                centroid_y=float(geom.centroid[0]),
                bounding_box=geom.bounding_box,
                aspect_ratio=float(geom.aspect_ratio)
            )
            puncta_list.append(meas)

        # 5. Cell summaries aggregation
        per_cell_summary = {}
        for cell_id, puncta_ids in assignment.cell_to_puncta.items():
            cell_area = float(cell_areas[cell_id]) if cell_id < len(cell_areas) else 0.0
            puncta_count = len(puncta_ids)
            
            if puncta_count == 0:
                summary = PerCellPunctaSummary(
                    cell_id=int(cell_id),
                    cell_area=cell_area,
                    puncta_count=0,
                    average_puncta_area=0.0,
                    average_puncta_intensity=0.0,
                    average_integrated_intensity=0.0,
                    largest_punctum_id=0,
                    smallest_punctum_id=0,
                    total_puncta_area=0.0,
                    total_puncta_integrated_intensity=0.0,
                    max_punctum_intensity=0.0,
                    max_punctum_area=0.0
                )
            else:
                cell_meas = [p for p in puncta_list if p.punctum_id in puncta_ids]
                
                total_area = sum(p.area for p in cell_meas)
                total_integrated = sum(p.integrated_intensity for p in cell_meas)
                
                avg_area = total_area / puncta_count
                avg_intensity = sum(p.mean_intensity for p in cell_meas) / puncta_count
                avg_integrated = total_integrated / puncta_count
                
                # Area-based limits
                sorted_by_area = sorted(cell_meas, key=lambda p: p.area)
                smallest_id = sorted_by_area[0].punctum_id
                largest_id = sorted_by_area[-1].punctum_id
                
                # Max intensity and max area features
                max_intensity = max(p.mean_intensity for p in cell_meas)
                max_area = max(p.area for p in cell_meas)
                
                summary = PerCellPunctaSummary(
                    cell_id=int(cell_id),
                    cell_area=cell_area,
                    puncta_count=puncta_count,
                    average_puncta_area=float(avg_area),
                    average_puncta_intensity=float(avg_intensity),
                    average_integrated_intensity=float(avg_integrated),
                    largest_punctum_id=int(largest_id),
                    smallest_punctum_id=int(smallest_id),
                    total_puncta_area=float(total_area),
                    total_puncta_integrated_intensity=float(total_integrated),
                    max_punctum_intensity=float(max_intensity),
                    max_punctum_area=float(max_area)
                )
                
            per_cell_summary[int(cell_id)] = summary

        # 6. Sort lists to ensure deterministic output
        puncta_list.sort(key=lambda p: p.punctum_id)

        return PunctaResults(
            puncta_list=puncta_list,
            per_cell_summary=per_cell_summary,
            metadata={}
        )

