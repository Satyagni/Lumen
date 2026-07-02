"""Puncta filtering and signal enhancement.

This module houses image filtering operations (like Difference of Gaussians) 
used to enhance spot signals and suppress non-uniform backgrounds.
"""

import numpy as np

# Difference of Gaussian detector (planned)
# Additional detectors reserved for future releases.

def apply_difference_of_gaussians(image: np.ndarray, sigma1: float, sigma2: float) -> np.ndarray:
    """Applies a Difference of Gaussians (DoG) filter to enhance spots.
    
    This acts as a bandpass filter that emphasizes signals of a specific size
    while eliminating low-frequency background gradients and high-frequency noise.
    
    Parameters:
        image: 2D numpy array representing raw channel intensities.
        sigma1: Standard deviation of the inner Gaussian filter (narrow).
        sigma2: Standard deviation of the outer Gaussian filter (wide).
        
    Returns:
        Filtered 2D image array of the same shape and type.
        
    TODO:
        - Implement subtraction of two blurred image channels using OpenCV or scipy.
    """
    return image.copy()
