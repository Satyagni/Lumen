import time
from pathlib import Path
import numpy as np
from PySide6.QtCore import QThread, Signal, QObject, Qt
from lumen.core.logger import logger

class AnalysisWorker(QThread):
    """Asynchronous background worker executing local Cellpose model inference."""
    
    progress_updated = Signal(int)       # Reports progress percent (0 - 100)
    status_updated = Signal(str)         # Reports descriptive text status updates
    finished_successfully = Signal(dict) # Reports final analysis results_dict
    failed = Signal(str)                 # Reports error messages
 
    def __init__(self, image_path: str, parameters: dict):
        super().__init__()
        self.image_path = image_path
        self.parameters = parameters
        self._is_cancelled = False
 
    def run(self):
        """Asynchronously executes Cellpose model pipeline."""
        logger.info("AnalysisWorker: Starting Cellpose pipeline for image: %s", self.image_path)
        try:
            self.progress_updated.emit(10)
            self.status_updated.emit("Loading original raw image...")
            
            # Step 1: Retrieve raw numpy array
            from lumen.processing.image_manager import image_manager
            # Only reuse active manager cache if it matches the worker's target file path
            if image_manager._current_path == self.image_path:
                raw_arr = image_manager._raw_numpy_arr
            else:
                raw_arr = None

            if raw_arr is None:
                import tifffile
                import PIL.Image
                ext = Path(self.image_path).suffix.lower()
                if ext in [".tif", ".tiff"]:
                    raw_arr = tifffile.imread(self.image_path)
                else:
                    with PIL.Image.open(self.image_path) as pil_img:
                        raw_arr = np.asarray(pil_img)
            
            if raw_arr is None or raw_arr.size == 0:
                raise ValueError("Could not access or load raw image array.")

            self.progress_updated.emit(20)
            
            # Step 2: Heuristic routing
            # Retrieve metadata or compute it locally to avoid stale singleton cache reads
            if image_manager._current_path == self.image_path:
                meta = image_manager.get_metadata()
                modality = meta.get("classification", "Unknown Biological Imaging")
            else:
                from lumen.workflows.image_classifier import classify_image
                filename = Path(self.image_path).name
                shape = raw_arr.shape
                if len(shape) == 2:
                    channels = 1
                    mode = "grayscale"
                else:
                    channels = shape[2] if len(shape) == 3 else 1
                    mode = "rgb" if channels >= 3 else "grayscale"
                ext = Path(self.image_path).suffix.lower()
                img_format = "TIFF" if ext in [".tif", ".tiff"] else ext[1:].upper()
                
                classification_data = classify_image(filename, channels, mode, img_format)
                modality = classification_data["type"]
                meta = {
                    "channels": channels,
                    "mode": mode,
                    "format": img_format,
                    "classification": modality
                }
            
            segmentation_method = self.parameters.get("segmentation_method", "AI Segmentation")
            if segmentation_method != "AI Segmentation":
                raise ValueError(f"Unsupported segmentation method: {segmentation_method}")

            from lumen.workflows.cellpose_routing import determine_model_type, determine_channels, get_segmentation_config
            resolved_model_type = determine_model_type(modality, Path(self.image_path).name)
            resolved_channels = determine_channels(modality, meta, raw_arr)
            
            # Clean advanced parameter overrides injection (architectural foundation)
            model_type = self.parameters.get("model_type_override") or resolved_model_type
            channels = self.parameters.get("channel_override") or resolved_channels
            
            # Resolve quality mode configuration settings mapping
            quality_mode = self.parameters.get("quality_mode", "Balanced")
            quality_config = get_segmentation_config(quality_mode)
            
            flow_threshold = self.parameters.get("flow_threshold_override", quality_config["flow_threshold"])
            cellprob_threshold = self.parameters.get("cellprob_threshold_override", quality_config["cellprob_threshold"])
            resample = self.parameters.get("resample_override", quality_config["resample"])
            diameter = self.parameters.get("diameter_override", None)
            
            # Resolve execution backend based on preference
            from lumen.core.services.gpu_service import gpu_service
            from cellpose import models
            pref = self.parameters.get("backend_preference", "Use Global Setting")
            use_gpu, resolved_backend_name = gpu_service.resolve_execution_backend(pref)
            
            # Check if models exist in user home directory to log first-run setup
            cellpose_dir = Path.home() / '.cellpose' / 'models'
            model_download_needed = True
            if cellpose_dir.exists():
                model_files = list(cellpose_dir.glob(f"*{model_type}*"))
                if model_files:
                    model_download_needed = False
                    
            if model_download_needed:
                logger.info("AnalysisWorker: Local model weights not found. Preparing first-time download (~80MB)...")
                self.status_updated.emit("Downloading Cellpose model weights (first-time setup)...")
            else:
                self.status_updated.emit("Initializing Cellpose model...")
                
            model = models.Cellpose(gpu=use_gpu, model_type=model_type)
            
            if self._is_cancelled:
                return

            self.progress_updated.emit(40)
            self.status_updated.emit("Running image preprocessing...")
            
            # Step 4: Execute Cellpose safely
            self.status_updated.emit("Executing Cellpose segmentation (inference)...")
            logger.info("AnalysisWorker: Running model.eval on model_type='%s', gpu=%s, channels=%s, flow_threshold=%s, cellprob_threshold=%s, resample=%s", 
                        model_type, use_gpu, channels, flow_threshold, cellprob_threshold, resample)
            
            start_time = time.time()
            masks, flows, styles, diams = model.eval(
                raw_arr,
                channels=channels,
                flow_threshold=flow_threshold,
                cellprob_threshold=cellprob_threshold,
                resample=resample,
                diameter=diameter
            )
            elapsed = time.time() - start_time
            
            if self._is_cancelled:
                return

            self.progress_updated.emit(80)
            self.status_updated.emit("Processing and packaging results...")

            # Step 5: Pack expanded scientific results_dict
            unique_labels = np.unique(masks)
            valid_labels = [label for label in unique_labels if label != 0]
            cell_count = len(valid_labels)
            
            # Compute cell areas and individual cell metrics
            cell_metrics = {}
            cell_areas = []
            for label in valid_labels:
                indices = np.argwhere(masks == label)
                area = len(indices)
                cell_areas.append(area)
                
                if area > 0:
                    mean_y, mean_x = np.mean(indices, axis=0)
                    diameter = round(2 * np.sqrt(area / np.pi), 2)
                    cell_metrics[int(label)] = {
                        "area_px": int(area),
                        "centroid": (round(float(mean_x), 1), round(float(mean_y), 1)),
                        "diameter_px": float(diameter)
                    }
            
            if cell_count > 0:
                mean_cell_area_px = float(np.mean(cell_areas))
                median_cell_area_px = float(np.median(cell_areas))
                avg_diameter = float(np.mean(diams)) if diams is not None and len(np.atleast_1d(diams)) > 0 else 0.0
            else:
                mean_cell_area_px = 0.0
                median_cell_area_px = 0.0
                avg_diameter = 0.0
                
            # Cell density = cells / total image pixels
            h, w = raw_arr.shape[:2]
            image_area = h * w
            cell_density = float(cell_count / image_area) if image_area > 0 else 0.0
            
            results_dict = {
                "masks": masks,
                "cell_metrics": cell_metrics,
                "cell_count": cell_count,
                "average_diameter_px": round(avg_diameter, 2),
                "mean_cell_area_px": round(mean_cell_area_px, 2),
                "median_cell_area_px": round(median_cell_area_px, 2),
                "cell_density": cell_density,
                "processing_time_s": round(elapsed, 2),
                "used_gpu": use_gpu,
                "resolved_backend": resolved_backend_name,
                "model_type": model_type,
                "modality": modality
            }
            
            logger.info("AnalysisWorker: Successfully completed Cellpose segmentation. Count: %d, Time: %.2fs", cell_count, elapsed)
            self.progress_updated.emit(100)
            self.status_updated.emit("Analysis completed successfully.")
            self.finished_successfully.emit(results_dict)

        except Exception as e:
            logger.error("AnalysisWorker: Pipeline execution failed: %s", e, exc_info=True)
            self.failed.emit(f"Segmentation failed: {str(e)}")

    def cancel(self):
        """Signals worker to interrupt loop."""
        self._is_cancelled = True


class ProcessingManager(QObject):
    """Coordinates execution of background scientific computation workers."""

    def __init__(self):
        super().__init__()
        self.active_worker = None
        logger.info("ProcessingManager initialized.")

    def run_analysis(self, image_path: str, parameters: dict, callbacks: dict) -> bool:
        """Launches AnalysisWorker thread and hooks callbacks to worker signals."""
        if self.active_worker and self.active_worker.isRunning():
            logger.warning("ProcessingManager: Analysis already running.")
            return False

        self.active_worker = AnalysisWorker(image_path, parameters)
        
        # Connect callbacks with explicit QueuedConnection to run slots on the GUI thread
        if "progress" in callbacks:
            self.active_worker.progress_updated.connect(callbacks["progress"], Qt.QueuedConnection)
        if "status" in callbacks:
            self.active_worker.status_updated.connect(callbacks["status"], Qt.QueuedConnection)
        if "finished" in callbacks:
            self.active_worker.finished_successfully.connect(callbacks["finished"], Qt.QueuedConnection)
        if "failed" in callbacks:
            self.active_worker.failed.connect(callbacks["failed"], Qt.QueuedConnection)

        self.active_worker.start()
        return True

    def cancel_active_analysis(self):
        """Cancels currently running worker thread."""
        if self.active_worker and self.active_worker.isRunning():
            self.active_worker.cancel()
            self.active_worker.wait()
            logger.info("ProcessingManager: Running worker aborted.")

# Global processing manager
processing_manager = ProcessingManager()

