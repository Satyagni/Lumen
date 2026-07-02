"""Puncta detection interfaces and algorithms.

This module houses the functional implementation of the Difference of Gaussian (DoG) 
spot detection algorithm. The detector acts as a pure mathematical function, 
independent of downstream measurements, cellular assignments, and application states.

Scientific Rationale:
--------------------
Difference of Gaussians (DoG) is an established bandpass filtering technique that 
approximates the Laplacian of Gaussian (LoG) operator. It operates by subtracting 
a heavily smoothed version of an image from a less smoothed version. The smaller 
blur scale (sigma) acts as a high-pass filter preserving spot features, while the 
larger blur scale (sigma * dog_sigma_ratio) acts as a low-pass filter to attenuate 
low-frequency background gradients, uneven illumination, and noise. 

This detector is optimized for small, bright, approximately circular puncta, 
relying on 8-connectivity component labeling to avoid diagonal pixel splits.
"""

import numpy as np
import scipy.ndimage as ndimage
from typing import Optional
from lumen.core.puncta.config import PunctaParameters, ThresholdMode
from lumen.core.puncta.types import PunctaDetectionResult

class PunctaDetector:
    """Orchestrates spot detection using the Difference of Gaussian (DoG) algorithm.
    
    This class operates as a stateless processor, ensuring identical inputs produce
    strictly identical output label maps across executions.
    """
    
    def __init__(self, parameters: Optional[PunctaParameters] = None):
        """Initializes the detector with optional default parameters.
        
        Parameters:
            parameters: An optional PunctaParameters instance.
        """
        self.parameters = parameters

    def detect(self, image: np.ndarray, parameters: PunctaParameters) -> PunctaDetectionResult:
        """Detects puncta spots on a single channel raw image.
        
        This method executes a pure, side-effect-free pipeline:
        1. Validate inputs and configuration parameters.
        2. Convert image to float32 without unnecessary memory copies.
        3. Apply dual Gaussian blurs and subtract them (DoG calculation).
        4. Select threshold dynamically (ADAPTIVE or ABSOLUTE) and binarize.
        5. Labeled components using 8-connectivity.
        6. Filter components by pixel size constraints.
        7. Relabel components sequentially starting at 1.
        
        Parameters:
            image: 2D numpy array representing the single channel image.
            parameters: A PunctaParameters instance.
            
        Returns:
            A PunctaDetectionResult object containing the 2D label map and object IDs.
            
        Raises:
            TypeError: If input array or parameter types are invalid.
            ValueError: If parameters are out of valid physical bounds or dimensions mismatch.
        """
        # 1. Validation
        if not isinstance(image, np.ndarray):
            raise TypeError("Input image must be a numpy.ndarray")
        if image.ndim != 2:
            raise ValueError("Input image must be a 2D array")
        if not isinstance(parameters, PunctaParameters):
            raise TypeError("Parameters must be a PunctaParameters instance")
            
        # Configuration bound validation
        if parameters.sigma <= 0:
            raise ValueError("sigma must be greater than 0")
        if parameters.dog_sigma_ratio <= 1.0:
            raise ValueError("dog_sigma_ratio must be greater than 1.0")
        if parameters.minimum_size < 1:
            raise ValueError("minimum_size must be >= 1")
        if parameters.maximum_size < parameters.minimum_size:
            raise ValueError("maximum_size must be >= minimum_size")
        if parameters.threshold_multiplier < 0.0:
            raise ValueError("threshold_multiplier must be >= 0")
        if parameters.absolute_threshold < 0.0:
            raise ValueError("absolute_threshold must be >= 0")

        # 2. Precision conversion
        img_float = image.astype(np.float32, copy=False)

        # 3. Difference of Gaussians
        sigma1 = parameters.sigma
        sigma2 = parameters.sigma * parameters.dog_sigma_ratio
        
        blur1 = ndimage.gaussian_filter(img_float, sigma=sigma1)
        blur2 = ndimage.gaussian_filter(img_float, sigma=sigma2)
        dog = blur1 - blur2

        # 4. Thresholding
        if parameters.threshold_mode == ThresholdMode.ADAPTIVE:
            dog_mean = float(np.mean(dog))
            dog_std = float(np.std(dog))
            threshold = dog_mean + parameters.threshold_multiplier * dog_std
        else:
            threshold = float(parameters.absolute_threshold)
            
        binary_mask = dog > threshold

        # 5. Connected Component Labeling (8-connectivity)
        # Using a fully connected 3x3 structuring element of all ones
        structure = np.ones((3, 3), dtype=bool)
        labeled, num_features = ndimage.label(binary_mask, structure=structure)
        
        if num_features == 0:
            return PunctaDetectionResult(
                labels=np.zeros_like(image, dtype=np.int32),
                object_ids=np.empty((0,), dtype=np.int32)
            )

        # 6. Size Filtering (vectorized)
        sizes = np.bincount(labeled.ravel())
        valid_mask = (sizes >= parameters.minimum_size) & (sizes <= parameters.maximum_size)
        valid_mask[0] = False  # Exclude background label
        
        valid_labels = np.where(valid_mask)[0]
        num_valid = len(valid_labels)
        
        if num_valid == 0:
            return PunctaDetectionResult(
                labels=np.zeros_like(image, dtype=np.int32),
                object_ids=np.empty((0,), dtype=np.int32)
            )

        # 7. Reindexing lookup table
        max_label = len(sizes) - 1
        lookup = np.zeros(max_label + 1, dtype=np.int32)
        for new_id, old_id in enumerate(valid_labels, start=1):
            lookup[old_id] = new_id
            
        labels_reindexed = lookup[labeled]
        object_ids = np.arange(1, num_valid + 1, dtype=np.int32)

        return PunctaDetectionResult(
            labels=labels_reindexed,
            object_ids=object_ids
        )

