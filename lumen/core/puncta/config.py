"""Configuration classes and parameters for puncta quantification.

This module houses the parameter structures representing user configurations 
for spot detection algorithms, spot sizing constraints, and channels.
"""

from dataclasses import dataclass
from enum import StrEnum

class ThresholdMode(StrEnum):
    """Modes for thresholding Difference of Gaussian filtered images."""
    ADAPTIVE = "adaptive"
    ABSOLUTE = "absolute"

@dataclass(frozen=True, slots=True)
class PunctaParameters:
    """Parameters controlling the puncta detection and analysis pipeline.
    
    This configuration is immutable (frozen) to ensure safe, mutation-free
    state tracking, undo/redo handling, and parallel analysis execution.
    
    Attributes:
        sigma: Standard deviation representing the target spot scale (radius).
        dog_sigma_ratio: Ratio of the larger Gaussian standard deviation to the smaller one.
        threshold_mode: Thresholding algorithm mode (ADAPTIVE or ABSOLUTE).
        threshold_multiplier: Standard deviation multiplier used in ADAPTIVE threshold mode.
        absolute_threshold: Hard cutoff threshold value used in ABSOLUTE threshold mode.
        minimum_size: Minimum area of a valid punctum in pixels.
        maximum_size: Maximum area of a valid punctum in pixels.
        enabled: Toggle switch to enable/disable puncta analysis.
        channel: Selected fluorescence channel index for analysis.
    """
    sigma: float = 1.5
    dog_sigma_ratio: float = 1.6
    threshold_mode: ThresholdMode = ThresholdMode.ADAPTIVE
    threshold_multiplier: float = 3.0
    absolute_threshold: float = 3.0
    minimum_size: int = 2
    maximum_size: int = 100
    enabled: bool = False
    channel: int = 0

