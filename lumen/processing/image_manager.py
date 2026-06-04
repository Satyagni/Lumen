import os
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
import PIL.Image
import tifffile
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt
from lumen.core.logger import logger
from lumen.core.constants import ALLOWED_EXTENSIONS
from lumen.workflows.state import state

class ImageManager:
    """Centralized loader, cache, metadata extractor, and validator for microscopy images."""

    def __init__(self):
        self._current_path: Optional[str] = None
        self._cached_qimage: Optional[QImage] = None
        self._cached_display_qimage: Optional[QImage] = None
        self._cached_metadata: Dict[str, Any] = {}
        self._thumbnail_cache: Dict[str, QPixmap] = {}
        self._raw_numpy_arr: Optional[np.ndarray] = None
        logger.info("ImageManager initialized.")

    def is_valid_file(self, file_path: str) -> bool:
        """Validates if a file extension is supported."""
        if not file_path:
            return False
        ext = Path(file_path).suffix.lower()
        return ext in ALLOWED_EXTENSIONS

    def load_image(self, file_path: str, set_state: bool = True) -> Tuple[bool, str]:
        """Validates, loads, extracts metadata, and updates central state for an image."""
        if not file_path:
            return False, "Empty file path provided."

        if not os.path.exists(file_path):
            return False, f"File does not exist: {file_path}"

        if not self.is_valid_file(file_path):
            return False, f"Unsupported file format. Supported: {', '.join(ALLOWED_EXTENSIONS)}"

        try:
            # Check if already loaded
            if self._current_path == file_path and self._cached_qimage:
                logger.debug("ImageManager: Cache hit for image: %s", file_path)
                return True, "Loaded from cache."

            # Clear old state
            self.clear_cache()

            # Determine extension
            ext = Path(file_path).suffix.lower()
            filename = os.path.basename(file_path)

            # Load raw image as NumPy array using tifffile for TIFFs, PIL for others
            if ext in [".tif", ".tiff"]:
                raw_arr = tifffile.imread(file_path)
                img_format = "TIFF"
            else:
                with PIL.Image.open(file_path) as pil_img:
                    raw_arr = np.asarray(pil_img)
                    img_format = pil_img.format if pil_img.format else ext[1:].upper()

            if not isinstance(raw_arr, np.ndarray) or raw_arr.size == 0:
                return False, "Failed to load image into a valid NumPy array."

            # Determine channels and mode from array shape
            shape = raw_arr.shape
            height, width = shape[0], shape[1]
            if len(shape) == 2:
                channels = 1
                mode = "grayscale"
            elif len(shape) == 3:
                channels = shape[2]
                if channels in [3, 4]:
                    mode = "rgb"
                elif channels == 1:
                    mode = "grayscale"
                    raw_arr = raw_arr[..., 0]  # squeeze to 2D
                    channels = 1
                else:
                    # multi-spectral / other formats: default to grayscale preview of first channel
                    mode = "rgb" if channels >= 3 else "grayscale"
            else:
                return False, f"Unsupported image array dimensions: {len(shape)}"

            bit_depth = raw_arr.dtype.itemsize * 8
            file_size_kb = os.path.getsize(file_path) / 1024

            # Heuristic Biological Classification
            from lumen.workflows.image_classifier import classify_image
            classification_data = classify_image(filename, channels, mode, img_format)

            # Convert raw array to raw QImage
            qimage_raw = self._numpy_to_qimage(raw_arr)
            if qimage_raw.isNull():
                return False, "Failed to convert raw NumPy array to QImage."

            # Apply display normalization for visualization
            display_arr = self._normalize_display_array(raw_arr)
            qimage_display = self._numpy_to_qimage(display_arr)
            if qimage_display.isNull():
                return False, "Failed to convert normalized NumPy array to QImage."

            self._current_path = file_path
            self._raw_numpy_arr = raw_arr
            self._cached_qimage = qimage_raw
            self._cached_display_qimage = qimage_display
            self._cached_metadata = {
                "filename": filename,
                "path": file_path,
                "width": width,
                "height": height,
                "format": img_format,
                "channels": channels,
                "bit_depth": bit_depth,
                "mode": mode,
                "size_kb": round(file_size_kb, 2),
                "classification": classification_data["type"],
                "confidence": classification_data["confidence"],
                "recommended_workflows": classification_data["workflows"]
            }

            logger.info(
                "ImageManager: Successfully loaded image: %s (%dx%d, %d channels, %d-bit, Mode: %s, Class: %s)",
                filename, width, height, channels, bit_depth, mode, classification_data["type"]
            )

            # Alert state manager
            if set_state:
                state.current_image_path = file_path
            return True, "Successfully loaded image."

        except Exception as e:
            msg = f"Image loading exception: {str(e)}"
            logger.error("ImageManager: %s", msg, exc_info=True)
            return False, msg

    def get_qimage(self) -> Optional[QImage]:
        """Returns the display-normalized QImage reference."""
        return self._cached_display_qimage

    def get_raw_qimage(self) -> Optional[QImage]:
        """Returns the unmodified raw scientific QImage reference."""
        return self._cached_qimage

    def get_qpixmap(self) -> Optional[QPixmap]:
        """Converts the display-normalized QImage to a QPixmap and returns it."""
        if self._cached_display_qimage:
            return QPixmap.fromImage(self._cached_display_qimage)
        return None

    def get_metadata(self) -> Dict[str, Any]:
        """Returns the loaded image's metadata."""
        return self._cached_metadata

    def get_thumbnail(self, max_width: int = 250, max_height: int = 250) -> Optional[QPixmap]:
        """Generates a high-quality scaled preview thumbnail from display image."""
        if not self._current_path or not self._cached_display_qimage:
            return None

        cache_key = f"{self._current_path}_{max_width}x{max_height}"
        if cache_key in self._thumbnail_cache:
            return self._thumbnail_cache[cache_key]

        try:
            scaled_pixmap = QPixmap.fromImage(self._cached_display_qimage).scaled(
                max_width, max_height,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self._thumbnail_cache[cache_key] = scaled_pixmap
            return scaled_pixmap
        except Exception as e:
            logger.error("ImageManager: Thumbnail scaling failed: %s", e)
            return None

    def clear_cache(self):
        """Flushes cached image reference and thumbnails."""
        self._current_path = None
        self._raw_numpy_arr = None
        self._cached_qimage = None
        self._cached_display_qimage = None
        self._cached_metadata = {}
        self._thumbnail_cache.clear()
        logger.debug("ImageManager cache cleared.")

    def _numpy_to_qimage(self, arr: np.ndarray) -> QImage:
        """Converts a numpy array to a QImage, making a deep copy to decouple memory buffers."""
        if arr is None:
            return QImage()

        arr = np.ascontiguousarray(arr)
        h, w = arr.shape[:2]

        if arr.ndim == 2:
            # Grayscale
            if arr.dtype == np.uint8:
                return QImage(arr.tobytes(), w, h, w, QImage.Format_Grayscale8).copy()
            elif arr.dtype == np.uint16:
                return QImage(arr.tobytes(), w, h, w * 2, QImage.Format_Grayscale16).copy()
            else:
                arr_u8 = arr.astype(np.uint8)
                return QImage(arr_u8.tobytes(), w, h, w, QImage.Format_Grayscale8).copy()

        elif arr.ndim == 3:
            c = arr.shape[2]
            if c == 1:
                return self._numpy_to_qimage(arr[..., 0])
            elif c == 3:
                if arr.dtype == np.uint8:
                    return QImage(arr.tobytes(), w, h, w * 3, QImage.Format_RGB888).copy()
                else:
                    arr_u8 = arr.astype(np.uint8)
                    return QImage(arr_u8.tobytes(), w, h, w * 3, QImage.Format_RGB888).copy()
            elif c == 4:
                if arr.dtype == np.uint8:
                    return QImage(arr.tobytes(), w, h, w * 4, QImage.Format_RGBA8888).copy()
                else:
                    arr_u8 = arr.astype(np.uint8)
                    return QImage(arr_u8.tobytes(), w, h, w * 4, QImage.Format_RGBA8888).copy()

        return QImage()

    def _normalize_display_array(self, arr: np.ndarray) -> np.ndarray:
        """Applies 1st and 99th percentile display normalization to map intensities to uint8."""
        arr = np.ascontiguousarray(arr)

        if arr.ndim == 2:
            # Grayscale
            p1 = np.percentile(arr, 1)
            p99 = np.percentile(arr, 99)

            if p99 <= p1:
                logger.warning(
                    "ImageManager: Percentile display range calculation warning: p99 (%s) <= p1 (%s). "
                    "Using min/max fallback.", p99, p1
                )
                p1 = np.min(arr)
                p99 = np.max(arr)

            if p99 <= p1:
                # Flat/uniform image: return zero array
                return np.zeros_like(arr, dtype=np.uint8)

            normalized = np.clip((arr - p1) / (p99 - p1), 0.0, 1.0) * 255.0
            return normalized.astype(np.uint8)

        elif arr.ndim == 3:
            # Multichannel (RGB/RGBA)
            channels = arr.shape[2]
            normalized = np.zeros_like(arr, dtype=np.uint8)
            for c in range(channels):
                channel = arr[..., c]
                p1 = np.percentile(channel, 1)
                p99 = np.percentile(channel, 99)

                if p99 <= p1:
                    p1 = np.min(channel)
                    p99 = np.max(channel)

                if p99 <= p1:
                    normalized[..., c] = 0
                else:
                    normalized[..., c] = (np.clip((channel - p1) / (p99 - p1), 0.0, 1.0) * 255.0).astype(np.uint8)
            return normalized

        return arr.astype(np.uint8)

# Instantiate global image manager
image_manager = ImageManager()
