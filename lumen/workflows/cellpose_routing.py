import numpy as np
from typing import Dict, Any, List
from lumen.core.logger import logger

def determine_model_type(modality: str, filename: str = "") -> str:
    """Selects the Cellpose model type dynamically based on modality and filename keywords."""
    modality_lower = modality.lower()
    fn_lower = filename.lower() if filename else ""
    
    # Heuristic routing rules
    if "fluorescence" in modality_lower:
        # Check if the filename specifically indicates nucleus/DNA staining
        nuclei_keywords = ["dapi", "hoechst", "nuclei", "nucleus", "nuc"]
        if any(kw in fn_lower for kw in nuclei_keywords):
            model_type = "nuclei"
        else:
            # Safer default fallback for whole-cell fluorescence or general stains
            model_type = "cyto"
    elif "brightfield" in modality_lower:
        model_type = "cyto"
    elif "colony" in modality_lower:
        model_type = "cyto"
    else:
        model_type = "cyto"  # Safe default fallback
        
    logger.info("Cellpose Routing: Modality '%s' (file='%s') routed to model_type '%s'", modality, filename, model_type)
    return model_type

def determine_channels(modality: str, image_metadata: Dict[str, Any], raw_arr: np.ndarray) -> List[int]:
    """Selects the Cellpose channel mapping based on modality and image metadata.
    
    For MVP Phase 2A: Always returns safe [0, 0] grayscale/RGB fallbacks for high reliability.
    """
    # Modular structure preserved for future channel-specific routing logic
    channels = [0, 0]
    logger.info("Cellpose Routing: Modality '%s' routed to channels %s", modality, channels)
    return channels

def get_segmentation_config(mode: str) -> Dict[str, Any]:
    """Maps user-friendly presets to internal Cellpose parameters."""
    mode_lower = mode.lower()
    if mode_lower == "fast":
        return {
            "flow_threshold": 0.3,
            "cellprob_threshold": 1.5,
            "resample": False
        }
    elif mode_lower == "sensitive":
        return {
            "flow_threshold": 0.8,
            "cellprob_threshold": -2.0,
            "resample": True
        }
    elif mode_lower == "precise":
        return {
            "flow_threshold": 0.15,
            "cellprob_threshold": 1.5,
            "resample": True
        }
    else:  # balanced / default
        return {
            "flow_threshold": 0.4,
            "cellprob_threshold": 0.0,
            "resample": True
        }
