import numpy as np
import cv2
from typing import List, Dict, Any

def calculate_perimeter(mask: np.ndarray) -> float:
    """Calculates the perimeter of a binary mask.
    
    Uses OpenCV's cv2.findContours and cv2.arcLength.
    For small/degenerate masks (area <= 2) where contour tracing may yield 0.0,
    it falls back to a deterministic pixel-edge boundary count (e.g. 4.0 for a single pixel).
    """
    area = np.sum(mask)
    if area == 0:
        return 0.0
    
    # Fallback for very small areas to avoid degenerate contours returning 0.0
    if area <= 2:
        padded = np.pad(mask, 1, mode="constant", constant_values=0)
        diff_y = np.abs(np.diff(padded, axis=0))
        diff_x = np.abs(np.diff(padded, axis=1))
        return float(np.sum(diff_y) + np.sum(diff_x))
        
    # Standard contour-based perimeter for larger masks
    mask_uint8 = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return 0.0
    
    perimeter = 0.0
    for contour in contours:
        perimeter += cv2.arcLength(contour, closed=True)
        
    # If the contour-based perimeter is 0, fall back to pixel-edge boundary count
    if perimeter == 0.0:
        padded = np.pad(mask, 1, mode="constant", constant_values=0)
        diff_y = np.abs(np.diff(padded, axis=0))
        diff_x = np.abs(np.diff(padded, axis=1))
        perimeter = float(np.sum(diff_y) + np.sum(diff_x))
        
    return float(perimeter)


def quantify_fluorescence(
    raw_channels: List[np.ndarray],
    masks: np.ndarray,
    channel_names: List[str]
) -> List[Dict[str, Any]]:
    """Quantifies fluorescence intensity metrics for each cell mask across multiple channels.
    
    Parameters:
        raw_channels: List of 2D numpy arrays representing raw intensity channels.
        masks: A 2D numpy array containing labeled cell regions (0 for background, positive integers for cell labels).
        channel_names: List of strings corresponding to the names of raw_channels.
        
    Returns:
        A list of dictionaries containing per-cell metrics, sorted by cell_id ascending.
        
    Raises:
        TypeError: If input types are incorrect.
        ValueError: If any input dimensions or shapes mismatch, or if channel names count doesn't match raw_channels count.
    """
    # 1. Type verification
    if not isinstance(raw_channels, list):
        raise TypeError("raw_channels must be a list of numpy arrays")
    if not isinstance(masks, np.ndarray):
        raise TypeError("masks must be a numpy.ndarray")
    if not isinstance(channel_names, list):
        raise TypeError("channel_names must be a list of strings")
        
    # 2. Match raw channels count with channel names count
    if len(raw_channels) != len(channel_names):
        raise ValueError(
            f"Mismatch: Number of raw channels ({len(raw_channels)}) does not match "
            f"number of channel names ({len(channel_names)})"
        )
        
    # 3. Shape verification and dtype safety conversion
    mask_shape = masks.shape
    if len(mask_shape) != 2:
        raise ValueError(f"Mask must be a 2D array, got shape {mask_shape}")
        
    float_channels = []
    for idx, channel in enumerate(raw_channels):
        if not isinstance(channel, np.ndarray):
            raise TypeError(f"Channel at index {idx} must be a numpy.ndarray")
        if channel.shape != mask_shape:
            raise ValueError(
                f"Shape mismatch: Mask shape {mask_shape} does not match "
                f"channel '{channel_names[idx]}' shape {channel.shape}. "
                "Mask resizing is not allowed."
            )
        # Convert to float64 to prevent overflows or precision issues
        float_channels.append(channel.astype(np.float64))
        
    # 4. Extract cell labels and sort ascending
    unique_labels = np.unique(masks)
    # Exclude background (0 or False)
    cell_labels = unique_labels[unique_labels != 0]
    
    # Sort labels to ensure deterministic output ordering
    cell_labels = np.sort(cell_labels)
    
    results = []
    for label in cell_labels:
        # Create boolean mask for the current cell label
        cell_mask = (masks == label)
        
        # Calculate cell geometry metrics
        area_val = int(np.sum(cell_mask))
        perimeter_val = calculate_perimeter(cell_mask)
        
        cell_metrics = {
            "cell_id": int(label),
            "area": area_val,
            "perimeter": perimeter_val
        }
        
        # Calculate channel-specific intensity metrics
        for name, channel_data in zip(channel_names, float_channels):
            # Extract raw signal values within the cell mask
            pixel_vals = channel_data[cell_mask]
            
            # Since area_val > 0 (as label exists), pixel_vals will not be empty.
            mean_val = float(np.mean(pixel_vals))
            median_val = float(np.median(pixel_vals))
            integrated_int = float(np.sum(pixel_vals))
            min_val = float(np.min(pixel_vals))
            max_val = float(np.max(pixel_vals))
            std_val = float(np.std(pixel_vals))
            
            # Flattened representation for compatibility/tabular export
            cell_metrics[f"{name}_mean"] = mean_val
            cell_metrics[f"{name}_median"] = median_val
            cell_metrics[f"{name}_integrated_intensity"] = integrated_int
            cell_metrics[f"{name}_min"] = min_val
            cell_metrics[f"{name}_max"] = max_val
            cell_metrics[f"{name}_std_deviation"] = std_val
            
        results.append(cell_metrics)
        
    return results
