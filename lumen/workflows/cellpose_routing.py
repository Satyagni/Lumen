import numpy as np
from typing import Dict, Any, List
from lumen.core.logger import logger

def determine_model_type(modality: str, filename: str = "", metadata: dict = None) -> str:
    """Selects the Cellpose model type dynamically based on classifier, metadata, and filename hints."""
    modality_lower = modality.lower() if modality else ""
    fn_lower = filename.lower() if filename else ""
    
    # 1. Image classifier / morphology prediction
    if "nuclei" in modality_lower or "dapi" in modality_lower:
        logger.info("Cellpose Routing: Modality indicates nuclei/DAPI. Routed to model_type 'nuclei'")
        return "nuclei"
        
    # 2. Channel naming or metadata (DAPI/Hoechst/nuclei)
    if metadata:
        meta_class = str(metadata.get("classification", "")).lower()
        if "nuclei" in meta_class or "dapi" in meta_class:
            logger.info("Cellpose Routing: Metadata classification indicates nuclei/DAPI. Routed to model_type 'nuclei'")
            return "nuclei"
            
        channel_names = metadata.get("channel_names", [])
        for name in channel_names:
            name_lower = str(name).lower()
            if any(kw in name_lower for kw in ["dapi", "hoechst", "nuclei", "nuc"]):
                logger.info("Cellpose Routing: Channel name indicates nuclei/DAPI. Routed to model_type 'nuclei'")
                return "nuclei"
    
    # 3. Filename hints (as last fallback before generic fallback)
    nuclei_keywords = ["dapi", "hoechst", "nuclei", "nucleus", "nuc"]
    if any(kw in fn_lower for kw in nuclei_keywords):
        logger.info("Cellpose Routing: Filename hints indicate nuclei/DAPI. Routed to model_type 'nuclei'")
        return "nuclei"
        
    # 4. Fallback
    logger.info("Cellpose Routing: Defaulting to model_type 'cyto3'")
    return "cyto3"

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
