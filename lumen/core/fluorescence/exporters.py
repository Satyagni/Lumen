import csv
import json
import numpy as np
from typing import List, Dict, Any, Optional

def infer_channels_from_output(quantifier_output: List[Dict[str, Any]]) -> List[str]:
    """Infers the channel names from the quantifier output while preserving their original order."""
    if not quantifier_output:
        return []
    
    first_cell = quantifier_output[0]
    channels = []
    for key in first_cell.keys():
        if key.endswith("_mean"):
            channel_name = key[:-5]  # Strip '_mean' suffix
            channels.append(channel_name)
    return channels


def export_cell_csv(
    quantifier_output: List[Dict[str, Any]],
    file_path: str,
    fallback_channel_names: Optional[List[str]] = None
) -> None:
    """Exports per-cell metrics to a CSV file in a flattened, tabular format.
    
    Headers are output in a stable deterministic order:
    cell_id, area, perimeter, followed by all metrics for each channel in the original channel order.
    
    Parameters:
        quantifier_output: List of dictionaries containing per-cell metrics.
        file_path: Destination file path for the CSV.
        fallback_channel_names: Channel names to use if the quantifier output is empty.
    """
    # Infer channels, falling back if empty
    channels = infer_channels_from_output(quantifier_output)
    if not channels and fallback_channel_names:
        channels = fallback_channel_names
        
    # Build deterministic headers list
    headers = ["cell_id", "area", "perimeter"]
    metrics = ["mean", "median", "integrated_intensity", "min", "max", "std_deviation"]
    for ch in channels:
        for metric in metrics:
            headers.append(f"{ch}_{metric}")
            
    # Write to CSV
    with open(file_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for row in quantifier_output:
            csv_row = [row.get(h, "") for h in headers]
            writer.writerow(csv_row)


def export_summary_csv(
    image_filename: str,
    quantifier_output: List[Dict[str, Any]],
    preprocessing_settings: Dict[str, Any],
    segmentation_settings: Dict[str, Any],
    timestamp: str,
    file_path: str,
    fallback_channel_names: Optional[List[str]] = None
) -> None:
    """Exports image-level summary metrics to a key-value style CSV file.
    
    Parameters:
        image_filename: Name of the processed image file.
        quantifier_output: List of dictionaries containing per-cell metrics.
        preprocessing_settings: Preprocessing parameters used.
        segmentation_settings: Segmentation parameters used.
        timestamp: Time when the analysis was executed.
        file_path: Destination file path for the CSV.
        fallback_channel_names: Channel names to use if the quantifier output is empty.
    """
    # Infer channels, falling back if empty
    channels = infer_channels_from_output(quantifier_output)
    if not channels and fallback_channel_names:
        channels = fallback_channel_names
        
    # Calculate image-level summary statistics
    total_cell_count = len(quantifier_output)
    if total_cell_count > 0:
        average_area = float(np.mean([row["area"] for row in quantifier_output]))
    else:
        average_area = 0.0
        
    # Build deterministic list of key-value tuples
    rows = [
        ("image_filename", image_filename),
        ("timestamp", timestamp),
        ("total_cell_count", total_cell_count),
        ("average_area", average_area)
    ]
    
    for ch in channels:
        mean_key = f"{ch}_mean"
        median_key = f"{ch}_median"
        
        if total_cell_count > 0:
            mean_vals = [row[mean_key] for row in quantifier_output if mean_key in row]
            median_vals = [row[median_key] for row in quantifier_output if median_key in row]
            
            mean_avg = float(np.mean(mean_vals)) if mean_vals else 0.0
            median_avg = float(np.mean(median_vals)) if median_vals else 0.0
        else:
            mean_avg = 0.0
            median_avg = 0.0
            
        rows.append((f"{ch}_mean_average", mean_avg))
        rows.append((f"{ch}_median_average", median_avg))
        
    # Serialize settings using JSON for robust key-value nesting
    rows.append(("preprocessing_settings", json.dumps(preprocessing_settings)))
    rows.append(("segmentation_settings", json.dumps(segmentation_settings)))
    
    # Write to CSV
    with open(file_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        for metric_name, value in rows:
            writer.writerow([metric_name, value])
