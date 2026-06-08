from typing import List
from lumen.core.logger import logger

def get_default_channel_names(channel_count: int, filename: str = "") -> List[str]:
    """Generates default descriptive channel names using filename hints or count heuristics."""
    if channel_count <= 1:
        return ["Grayscale"]

    fn_lower = filename.lower() if filename else ""

    # Common microscopy combinations
    if channel_count == 2:
        # Check if DAPI/Hoechst and GFP/FITC combination is likely
        if any(kw in fn_lower for kw in ["dapi", "hoechst", "blue", "nuc"]):
            return ["DAPI", "GFP"]
        return ["Channel 1", "Channel 2"]
    
    if channel_count == 3:
        return ["DAPI", "GFP", "RFP"]
    
    if channel_count == 4:
        return ["DAPI", "GFP", "RFP", "Cy5"]

    # General fallback
    return [f"Channel {i + 1}" for i in range(channel_count)]
