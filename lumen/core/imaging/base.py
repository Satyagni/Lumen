from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, List, Dict, Any, Optional
import numpy as np

try:
    from enum import StrEnum
except ImportError:
    from enum import Enum
    class StrEnum(str, Enum):
        pass

class ProjectionMode(StrEnum):
    NONE = "none"
    MIP = "mip"
    MEAN = "mean"

@dataclass(frozen=True, slots=True)
class ImageMetadata:
    filename: str
    path: str
    dimensions: Tuple[int, ...]
    dimension_order: str
    dtype: np.dtype
    bit_depth: int
    mode: str
    scene_count: int
    z_planes: int
    timepoints: int
    channels: int
    channel_names: List[str]
    voxel_size: Tuple[float, float, float]
    physical_units: str
    raw_metadata: Dict[str, Any]

@dataclass(frozen=True)
class ImageData:
    image: np.ndarray
    metadata: ImageMetadata

class ImageReader(ABC):
    """Abstract base class for all Lumen image format readers.
    
    Memory Ownership Contract:
    --------------------------
    - Every subclass of ImageReader MUST return fully owned, contiguous NumPy
      arrays from read_slice().
    - The returned arrays must NOT point to native memory buffers managed by
      underlying C/C++ libraries (e.g. libCZI, libtiff).
    - Downstream components may cache arrays and destroy the reader object at
      any time. The returned arrays must remain valid and accessible after reader
      destruction.
    - If an underlying native library returns a view of C/C++ memory, the reader
      implementation MUST call array.copy() prior to returning the ImageData instance.
    """
    
    @abstractmethod
    def open(self, file_path: str) -> None:
        """Opens file handles and parses metadata headers."""
        pass

    @abstractmethod
    def get_metadata(self) -> ImageMetadata:
        """Returns the cached, immutable metadata representation."""
        pass

    @abstractmethod
    def read_slice(self, scene: int = 0, channel: int = 0, z: int = 0, t: int = 0) -> ImageData:
        """Reads a single 2D plane returning both array and metadata."""
        pass

    @abstractmethod
    def supports(self, feature: str) -> bool:
        """Returns True if the reader supports a given capability."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Safely closes file descriptors."""
        pass

class ProjectionEngine:
    """Stateless processor for collapsing dimensional stacks into 2D projections."""
    
    @staticmethod
    def project(image_stack: np.ndarray, mode: ProjectionMode) -> np.ndarray:
        """Projects a 3D Z-stack array (Z, H, W) or (Z, H, W, C) into a 2D array.
        
        Supports Maximum Intensity Projection (MIP) and Mean Projection.
        """
        # Support string comparison for robustness
        mode_str = str(mode).lower()
        if mode_str == "none" or mode_str == "":
            return image_stack
            
        if image_stack.ndim < 3:
            return image_stack
            
        if mode_str == "mip":
            return np.max(image_stack, axis=0)
        elif mode_str == "mean":
            return np.mean(image_stack, axis=0).astype(image_stack.dtype)
            
        return image_stack
