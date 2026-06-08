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
        self._raw_channels = []
        self._active_channel_idx = -1
        self._channel_names = []
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

            # Transpose (C, H, W) to (H, W, C) if first dimension represents channels
            ndim = raw_arr.ndim
            if ndim == 3:
                # Heuristic: normally height/width are large, channels C <= 10
                if raw_arr.shape[0] <= 10 and raw_arr.shape[2] > 10:
                    raw_arr = np.transpose(raw_arr, (1, 2, 0))

            # Determine channels and mode from array shape
            shape = raw_arr.shape
            if ndim == 2:
                height, width = shape[0], shape[1]
                channels = 1
                mode = "grayscale"
                self._raw_channels = [raw_arr]
            elif ndim == 3:
                height, width = shape[0], shape[1]
                channels = shape[2]
                if channels in [3, 4]:
                    mode = "rgb"
                elif channels == 1:
                    mode = "grayscale"
                    raw_arr = raw_arr[..., 0]  # squeeze to 2D
                    channels = 1
                    self._raw_channels = [raw_arr]
                else:
                    mode = "rgb" if channels >= 3 else "grayscale"
                
                if channels > 1:
                    self._raw_channels = [raw_arr[..., c] for c in range(channels)]
            else:
                return False, f"Unsupported image array dimensions: {ndim}"

            bit_depth = raw_arr.dtype.itemsize * 8
            file_size_kb = os.path.getsize(file_path) / 1024

            # Heuristic Biological Classification
            from lumen.workflows.image_classifier import classify_image
            classification_data = classify_image(filename, channels, mode, img_format)

            # Initialize active channel index: composite (-1) if multi-channel, else 0
            self._active_channel_idx = -1 if channels > 1 else 0

            # Dynamic Channel Naming registry
            from lumen.core.fluorescence.channels import get_default_channel_names
            self._channel_names = get_default_channel_names(channels, filename)

            # Generate cached display images
            self._update_cached_images()

            if self._cached_qimage.isNull() or self._cached_display_qimage.isNull():
                return False, "Failed to build QImage from loaded image."

            self._current_path = file_path
            self._raw_numpy_arr = raw_arr
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
                state.channel_names = self._channel_names
                state.active_viewer_channel = self._active_channel_idx
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
        self._raw_channels = []
        self._active_channel_idx = -1
        self._channel_names = []
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

    def get_channel_data(self, channel_idx: int) -> Optional[np.ndarray]:
        """Returns the 2D raw NumPy array for the specified channel index."""
        if not self._raw_channels or channel_idx < 0 or channel_idx >= len(self._raw_channels):
            return None
        return self._raw_channels[channel_idx]

    def set_active_channel(self, channel_idx: int):
        """Sets the active display channel (-1 for composite) and updates cached display images."""
        if not self._raw_channels:
            return
        if channel_idx < -1 or channel_idx >= len(self._raw_channels):
            return
        self._active_channel_idx = channel_idx
        self._update_cached_images()

    def _update_cached_images(self):
        """Re-generates self._cached_qimage and self._cached_display_qimage based on active channel selection and preprocessing."""
        if not self._raw_channels:
            return

        # Determine active view mode
        if self._active_channel_idx == -1 and len(self._raw_channels) > 1:
            # Composite View
            composite_arr = self._generate_composite_array()
            self._cached_qimage = self._numpy_to_qimage(composite_arr)
            self._cached_display_qimage = self._numpy_to_qimage(composite_arr)
        else:
            # Single Channel View
            idx = max(0, self._active_channel_idx)
            chan_arr = self._raw_channels[idx]
            preprocessed_arr = self.preprocess_array(chan_arr)
            self._cached_qimage = self._numpy_to_qimage(chan_arr)
            self._cached_display_qimage = self._numpy_to_qimage(preprocessed_arr)

    def _generate_composite_array(self) -> np.ndarray:
        """Generates a composite RGB uint8 array from all available channels mapped to their active colors."""
        if not self._raw_channels:
            return np.zeros((100, 100, 3), dtype=np.uint8)

        h, w = self._raw_channels[0].shape[:2]
        composite = np.zeros((h, w, 3), dtype=np.float32)

        # Default fallback channel mapping colors (DAPI -> Blue, GFP -> Green, RFP -> Red, etc.)
        default_rgb_vectors = [
            [0.0, 0.0, 1.0],  # Blue
            [0.0, 1.0, 0.0],  # Green
            [1.0, 0.0, 0.0],  # Red
            [1.0, 1.0, 0.0],  # Yellow
            [0.0, 1.0, 1.0],  # Cyan
            [1.0, 0.0, 1.0],  # Magenta
        ]

        channel_names = []
        if hasattr(state, "current_image_path") and state.current_image_path == self._current_path:
            channel_names = state.channel_names
        if not channel_names and hasattr(self, "_channel_names"):
            channel_names = self._channel_names

        for idx, chan in enumerate(self._raw_channels):
            color_vec = default_rgb_vectors[idx % len(default_rgb_vectors)]
            if idx < len(channel_names):
                name = str(channel_names[idx]).lower()
                if any(kw in name for kw in ["dapi", "hoechst", "blue", "nuc"]):
                    color_vec = [0.0, 0.0, 1.0]
                elif any(kw in name for kw in ["gfp", "green", "fitc", "alexa488"]):
                    color_vec = [0.0, 1.0, 0.0]
                elif any(kw in name for kw in ["rfp", "red", "tritc", "cy5", "alexa594"]):
                    color_vec = [1.0, 0.0, 0.0]

            # Normalize channel to 0.0 - 1.0
            norm_chan = self._normalize_channel_to_float(chan)

            for c in range(3):
                composite[..., c] += norm_chan * color_vec[c]

        composite = np.clip(composite, 0.0, 1.0) * 255.0
        return composite.astype(np.uint8)

    def _normalize_channel_to_float(self, arr: np.ndarray) -> np.ndarray:
        """Normalizes raw channel numpy array to a 0.0 - 1.0 range float32 representation incorporating preprocessing."""
        preprocessed_uint8 = self.preprocess_array(arr)
        return preprocessed_uint8.astype(np.float32) / 255.0

    def preprocess_array(self, arr: np.ndarray) -> np.ndarray:
        """Applies non-destructive preprocessing pipeline (auto contrast, percentile stretch, brightness, contrast, gamma) to the array, returning a uint8 representation."""
        if arr is None:
            return None
            
        # Import state locally to avoid circular dependency
        from lumen.workflows.state import state
        
        auto_contrast = getattr(state, "preprocess_auto_contrast", True)
        p_low = getattr(state, "preprocess_percentile_low", 1.0)
        p_high = getattr(state, "preprocess_percentile_high", 99.0)
        brightness = getattr(state, "preprocess_brightness", 0.0)
        contrast = getattr(state, "preprocess_contrast", 1.0)
        gamma = getattr(state, "preprocess_gamma", 1.0)
        
        # 1. Convert to float32
        img = arr.astype(np.float32)
        
        if auto_contrast:
            # Percentile stretch
            p_l = np.percentile(img, p_low)
            p_h = np.percentile(img, p_high)
            if p_h > p_l:
                img = np.clip((img - p_l) / (p_h - p_l), 0.0, 1.0)
            else:
                # Fallback to min/max
                amin = np.min(img)
                amax = np.max(img)
                if amax > amin:
                    img = np.clip((img - amin) / (amax - amin), 0.0, 1.0)
                else:
                    img = np.zeros_like(img)
        else:
            # Determine maximum value based on data type to normalize to [0.0, 1.0] without stretching
            if np.issubdtype(arr.dtype, np.integer):
                max_val = float(np.iinfo(arr.dtype).max)
            else:
                max_val = 1.0
            img = np.clip(img / max_val, 0.0, 1.0)
                
        # 2. Contrast adjustment (midpoint 0.5)
        if contrast != 1.0:
            img = np.clip((img - 0.5) * contrast + 0.5, 0.0, 1.0)
            
        # 3. Brightness adjustment
        if brightness != 0.0:
            img = np.clip(img + brightness, 0.0, 1.0)
            
        # 4. Gamma correction
        if gamma != 1.0:
            img = np.clip(img, 1e-8, 1.0)
            img = np.power(img, gamma)
            
        return (img * 255.0).astype(np.uint8)

# Instantiate global image manager
image_manager = ImageManager()

