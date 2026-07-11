"""User-facing results and measurements dataclasses for puncta quantification.

This module defines final tabular data models representing identified puncta 
attributes and per-cell aggregated statistics exported to CSV/reports.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple

@dataclass(frozen=True, slots=True)
class PunctumMeasurement:
    """Quantitative features extracted for a single assigned punctum.
    
    Attributes:
        punctum_id: Unique identifier for this punctum.
        cell_id: ID of the parent cell mask (0 if unassigned/background).
        area: Size of the spot in pixels.
        perimeter: Discrete boundary perimeter in pixels.
        equivalent_diameter: Equivalent circular diameter in pixels.
        mean_intensity: Mean intensity within the spot mask in the original image.
        median_intensity: Median intensity within the spot mask in the original image.
        integrated_intensity: Total sum of pixel intensities in the original image.
        minimum_intensity: Minimum pixel intensity inside the spot.
        maximum_intensity: Peak pixel intensity inside the spot.
        standard_deviation: Standard deviation of pixel intensities inside the spot.
        centroid_x: X-coordinate of spot centroid (float).
        centroid_y: Y-coordinate of spot centroid (float).
        bounding_box: Bounding box tuple (min_row, min_col, height, width).
        aspect_ratio: Aspect ratio (width / height).
    """
    punctum_id: int
    cell_id: int
    area: float
    perimeter: float
    equivalent_diameter: float
    mean_intensity: float
    median_intensity: float
    integrated_intensity: float
    minimum_intensity: float
    maximum_intensity: float
    standard_deviation: float
    centroid_x: float
    centroid_y: float
    bounding_box: Tuple[int, int, int, int]
    aspect_ratio: float

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the punctum measurement fields to a flat dictionary."""
        return {
            "punctum_id": self.punctum_id,
            "cell_id": self.cell_id,
            "area_px": self.area,
            "perimeter": self.perimeter,
            "equivalent_diameter": self.equivalent_diameter,
            "mean_intensity": self.mean_intensity,
            "median_intensity": self.median_intensity,
            "integrated_intensity": self.integrated_intensity,
            "minimum_intensity": self.minimum_intensity,
            "maximum_intensity": self.maximum_intensity,
            "standard_deviation": self.standard_deviation,
            "centroid_x": self.centroid_x,
            "centroid_y": self.centroid_y,
            "bbox_min_row": self.bounding_box[0],
            "bbox_min_col": self.bounding_box[1],
            "bbox_height": self.bounding_box[2],
            "bbox_width": self.bounding_box[3],
            "aspect_ratio": self.aspect_ratio
        }


@dataclass(frozen=True, slots=True)
class PerCellPunctaSummary:
    """Aggregated puncta quantification metrics for a single cell region.
    
    Attributes:
        cell_id: Target cell mask identifier.
        cell_area: Size of the cell mask in pixels.
        puncta_count: Total number of puncta localized within the cell.
        average_puncta_area: Mean pixel area of puncta inside the cell.
        average_puncta_intensity: Average of mean puncta intensities inside the cell.
        average_integrated_intensity: Average integrated intensity of puncta inside the cell.
        largest_punctum_id: ID of the largest punctum inside this cell (0 if none).
        smallest_punctum_id: ID of the smallest punctum inside this cell (0 if none).
        total_puncta_area: Summed area of all localized puncta in pixels.
        total_puncta_integrated_intensity: Summed integrated intensity of all puncta in the cell.
        max_punctum_intensity: Highest mean intensity of any punctum inside this cell (0.0 if none).
        max_punctum_area: Highest area of any punctum inside this cell (0.0 if none).
    """
    cell_id: int
    cell_area: float
    puncta_count: int
    average_puncta_area: float
    average_puncta_intensity: float
    average_integrated_intensity: float
    largest_punctum_id: int
    smallest_punctum_id: int
    total_puncta_area: float
    total_puncta_integrated_intensity: float
    max_punctum_intensity: float
    max_punctum_area: float

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the cell puncta summary fields to a flat dictionary."""
        return {
            "cell_id": self.cell_id,
            "cell_area": self.cell_area,
            "puncta_count": self.puncta_count,
            "average_puncta_area": self.average_puncta_area,
            "average_puncta_intensity": self.average_puncta_intensity,
            "average_integrated_intensity": self.average_integrated_intensity,
            "largest_punctum_id": self.largest_punctum_id,
            "smallest_punctum_id": self.smallest_punctum_id,
            "total_puncta_area": self.total_puncta_area,
            "total_puncta_integrated_intensity": self.total_puncta_integrated_intensity,
            "max_punctum_intensity": self.max_punctum_intensity,
            "max_punctum_area": self.max_punctum_area
        }


@dataclass
class PunctaResults:
    """Encapsulates the complete result set of a puncta analysis run.
    
    Attributes:
        puncta_list: Collection of individual PunctumMeasurement records.
        per_cell_summary: Map of cell_id -> PerCellPunctaSummary statistics.
        metadata: Execution parameters and run-specific descriptors.
    """
    puncta_list: List[PunctumMeasurement] = field(default_factory=list)
    per_cell_summary: Dict[int, PerCellPunctaSummary] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the full analysis results to a nested dictionary."""
        return {
            "puncta": [p.to_dict() for p in self.puncta_list],
            "cell_summaries": {cid: summary.to_dict() for cid, summary in self.per_cell_summary.items()},
            "metadata": self.metadata
        }
