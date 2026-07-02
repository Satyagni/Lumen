"""Intermediate data structures and typing definitions for puncta processing.

This module houses transient representations generated during the image-processing 
and spot-finding phase, decoupling raw detection from downstream cell-mask assignment.
"""

import numpy as np
from dataclasses import dataclass

@dataclass(slots=True)
class PunctaDetectionResult:
    """Strongly typed raw output from a puncta detection run before cell assignment.
    
    Attributes:
        labels: 2D integer array (H, W) where non-zero pixels index detected spot labels.
        object_ids: 1D integer array (N,) containing unique identifier IDs for each spot.
    """
    labels: np.ndarray
    object_ids: np.ndarray

