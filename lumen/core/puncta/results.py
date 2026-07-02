"""User-facing results and measurements dataclasses for puncta quantification.

This module defines final tabular data models representing identified puncta 
attributes and per-cell aggregated statistics exported to CSV/reports.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple

@dataclass
class PunctumMeasurement:
    """Quantitative features extracted for a single assigned punctum.
    
    Attributes:
        punctum_id: Unique identifier for this punctum.
        cell_id: ID of the parent cell mask (0 if unassigned/background).
        centroid: Coordinate tuple (x, y) representing spot center.
        area: Size of the spot in pixels.
        mean_intensity: Average signal intensity within the spot mask.
        max_intensity: Peak signal intensity within the spot mask.
        integrated_intensity: Total sum of signal intensities within the spot mask.
        contrast: Estimated local signal-to-background contrast ratio.
    """
    punctum_id: int
    cell_id: int
    centroid: Tuple[float, float]
    area: float
    mean_intensity: float
    max_intensity: float
    integrated_intensity: float
    contrast: float

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the punctum measurement fields to a flat dictionary."""
        return {
            "punctum_id": self.punctum_id,
            "cell_id": self.cell_id,
            "centroid_x": self.centroid[0],
            "centroid_y": self.centroid[1],
            "area_px": self.area,
            "mean_intensity": self.mean_intensity,
            "max_intensity": self.max_intensity,
            "integrated_intensity": self.integrated_intensity,
            "contrast": self.contrast
        }


@dataclass
class PerCellPunctaSummary:
    """Aggregated puncta quantification metrics for a single cell region.
    
    Attributes:
        cell_id: Target cell mask identifier.
        puncta_count: Total number of puncta localized within the cell.
        average_punctum_size: Mean pixel area of puncta inside the cell.
        total_puncta_area: Summed area of all localized puncta in pixels.
        puncta_area_fraction: Proportion of cell area occupied by puncta (total_puncta_area / cell_area).
        average_punctum_intensity: Average signal intensity of the cell's puncta.
        puncta_density: Number of puncta per unit cell area (puncta_count / cell_area).
    """
    cell_id: int
    puncta_count: int
    average_punctum_size: float
    total_puncta_area: float
    puncta_area_fraction: float
    average_punctum_intensity: float
    puncta_density: float

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the cell puncta summary fields to a flat dictionary."""
        return {
            "cell_id": self.cell_id,
            "puncta_count": self.puncta_count,
            "average_punctum_size": self.average_punctum_size,
            "total_puncta_area": self.total_puncta_area,
            "puncta_area_fraction": self.puncta_area_fraction,
            "average_punctum_intensity": self.average_punctum_intensity,
            "puncta_density": self.puncta_density
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
