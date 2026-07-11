import os
from pathlib import Path
from typing import Dict, Any, List
from lumen.core.logger import logger

def classify_image(filename: str, channels: int, mode: str, format_str: str) -> Dict[str, Any]:
    """Applies explainable heuristics on file parameters to identify microscopy types.
    
    Returns:
        dict: {
            "type": str,          # "Fluorescence Microscopy" | "Brightfield Microscopy" | "Colony / Plate Imaging" | "Unknown"
            "confidence": str,    # "High" | "Moderate" | "Low"
            "workflows": list     # List of recommended workflows
        }
    """
    fn_lower = filename.lower()
    ext_lower = Path(filename).suffix.lower()

    # Define workflow descriptors
    cell_counting = {
        "id": "cell_counting",
        "name": "Cell Segmentation",
        "desc": "Detect and segment cells or nuclei.",
        "relevance": "High"
    }
    fluorescence = {
        "id": "fluorescence",
        "name": "Fluorescence Analysis",
        "desc": "Quantify signal intensity profiles across color channels.",
        "relevance": "High"
    }
    colony = {
        "id": "colony",
        "name": "Colony Analysis",
        "desc": "Measure colony counts and surface area on culture plates.",
        "relevance": "High"
    }
    custom = {
        "id": "custom",
        "name": "Custom Workflow",
        "desc": "Chain custom image processing pipeline blocks.",
        "relevance": "Moderate"
    }

    # 1. Rule: Fluorescence Microscopy
    fluor_keywords = ["dapi", "fitc", "gfp", "fluor", "nuclei", "stain"]
    has_fluor_keyword = any(kw in fn_lower for kw in fluor_keywords)
    is_czi_or_fluor_structure = ((mode == "grayscale" or channels == 1) and ext_lower in [".tiff", ".tif"]) or ext_lower == ".czi"

    if has_fluor_keyword or is_czi_or_fluor_structure:
        confidence = "High" if has_fluor_keyword else "Moderate"
        logger.info("Classifier: Fluorescence Microscopy detected. Rule source: keyword=%s, structure=%s", has_fluor_keyword, is_czi_or_fluor_structure)
        return {
            "type": "Fluorescence Microscopy",
            "confidence": confidence,
            "workflows": [cell_counting, fluorescence, custom]
        }

    # 2. Rule: Colony / Plate Imaging
    colony_keywords = ["plate", "agar", "well", "colony", "petri"]
    has_colony_keyword = any(kw in fn_lower for kw in colony_keywords)

    if has_colony_keyword:
        logger.info("Classifier: Colony / Plate Imaging detected via keyword rule.")
        return {
            "type": "Colony / Plate Imaging",
            "confidence": "High",
            "workflows": [colony, custom]
        }

    # 3. Rule: Brightfield Microscopy
    bf_keywords = ["brightfield", "bf"]
    has_bf_keyword = any(kw in fn_lower for kw in bf_keywords)
    is_rgb = (mode == "rgb" or channels >= 3)

    if has_bf_keyword or is_rgb:
        confidence = "High" if has_bf_keyword else "Moderate"
        logger.info("Classifier: Brightfield Microscopy detected. Rule source: keyword=%s, structure=%s", has_bf_keyword, is_rgb)
        
        # Adjust relevance for brightfield
        bf_counting = cell_counting.copy()
        bf_counting["relevance"] = "Moderate"
        bf_counting["desc"] = "Detect cells using thresholding and contour analysis."
        
        bf_morphology = {
            "id": "custom", # Treat custom workflow as morphology scaffold
            "name": "Cell Morphology",
            "desc": "Measure circularity, diameter, and shape factors.",
            "relevance": "High"
        }
        return {
            "type": "Brightfield Microscopy",
            "confidence": confidence,
            "workflows": [bf_morphology, bf_counting]
        }

    # 4. Fallback: Unknown
    logger.info("Classifier: Defaulting to Unknown Biological Imaging fallback.")
    # Fallback lists standard workflows with low/moderate relevance
    fallback_counting = cell_counting.copy()
    fallback_counting["relevance"] = "Low"
    fallback_fluorescence = fluorescence.copy()
    fallback_fluorescence["relevance"] = "Low"
    fallback_colony = colony.copy()
    fallback_colony["relevance"] = "Low"
    
    return {
        "type": "Unknown Biological Imaging",
        "confidence": "Low",
        "workflows": [fallback_counting, fallback_fluorescence, fallback_colony, custom]
    }
