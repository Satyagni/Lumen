"""Intermediate data structures and typing definitions for puncta processing.

This module houses transient representations generated during the image-processing 
and spot-finding phase, decoupling raw detection from downstream cell-mask assignment.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List

@dataclass(slots=True)
class PunctaDetectionResult:
    """Strongly typed raw output from a puncta detection run before cell assignment.
    
    Attributes:
        labels: 2D integer array (H, W) where non-zero pixels index detected spot labels.
        object_ids: 1D integer array (N,) containing unique identifier IDs for each spot.
    """
    labels: np.ndarray
    object_ids: np.ndarray

@dataclass(slots=True)
class PunctaAssignmentResult:
    """Output from the puncta-to-cell assignment stage.
    
    Attributes:
        cell_to_puncta: Dictionary mapping Cell ID (int) -> sorted List of assigned puncta IDs (list of ints).
        punctum_to_cell: Dictionary mapping Punctum ID (int) -> assigned Cell ID (int).
        unassigned_puncta: Sorted list of puncta IDs (ints) that fell into the background.
    """
    cell_to_puncta: Dict[int, List[int]]
    punctum_to_cell: Dict[int, int]
    unassigned_puncta: List[int]


