import time
from pathlib import Path
import numpy as np
from PySide6.QtCore import QThread, Signal, QObject, Qt, Slot
import threading
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

            if raw_arr.ndim == 3 and raw_arr.shape[0] <= 10 and raw_arr.shape[2] > 10:
                raw_arr = np.transpose(raw_arr, (1, 2, 0))

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
            resolved_model_type = determine_model_type(modality, Path(self.image_path).name, meta)
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
            
            # Select active channel slice if in fluorescence workflow
            from lumen.workflows.state import state
            input_arr = raw_arr
            eval_channels = channels
            if state.current_workflow == "fluorescence" and raw_arr.ndim == 3:
                seg_channel_idx = state.segmentation_channel
                if seg_channel_idx >= 0 and seg_channel_idx < raw_arr.shape[2]:
                    input_arr = raw_arr[..., seg_channel_idx]
                    eval_channels = [0, 0] # Grayscale eval for single 2D slice
                    logger.info("AnalysisWorker: Fluorescence mode active. Segmenting channel index %d as grayscale.", seg_channel_idx)

            # Apply non-destructive preprocessing pipeline to segmentation input
            from lumen.processing.image_manager import image_manager
            input_arr = image_manager.preprocess_array(input_arr)

            logger.info("AnalysisWorker: Running model.eval on model_type='%s', gpu=%s, channels=%s, flow_threshold=%s, cellprob_threshold=%s, resample=%s", 
                        model_type, use_gpu, eval_channels, flow_threshold, cellprob_threshold, resample)
            
            start_time = time.time()
            masks, flows, styles, diams = model.eval(
                input_arr,
                channels=eval_channels,
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
                        "diameter_px": float(diameter),
                        "diameter_estimate": float(diameter)
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


class BatchAnalysisWorker(QObject):
    """Asynchronous background worker executing sequential Cellpose model inference for a batch of images."""
    
    progress_updated = Signal(int, int)       # token, progress percent within current image (0 - 100)
    status_updated = Signal(int, str)         # token, status updates
    image_completed = Signal(int, str, dict)  # token, image path, results_dict
    image_failed = Signal(int, str, str)      # token, image path, error_msg
    finished = Signal(int)                    # token, finished
    
    def __init__(self, image_paths: list, parameters: dict, output_dir: str, start_idx: int = 0, batch_token: int = 0):
        super().__init__()
        self.image_paths = image_paths
        self.parameters = parameters
        self.output_dir = output_dir
        self.start_idx = start_idx
        self.batch_token = batch_token
        
        self._cancel_event = threading.Event()
        self._pause_event = threading.Event()
        self._pause_event.set()  # Initial state is unpaused
        
        self._output_done_event = threading.Event()
        self._output_done_event.set()  # Initial state is unblocked
        
    def cancel(self):
        self._cancel_event.set()
        self._pause_event.set()        # Unblock if paused
        self._output_done_event.set()  # Unblock if waiting on output
        
    def pause(self):
        self._pause_event.clear()
        
    def resume(self):
        self._pause_event.set()
        
    def mark_output_done(self):
        self._output_done_event.set()

    def _empty_gpu_cache(self):
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("BatchAnalysisWorker (token=%d): GPU cache cleared.", self.batch_token)
        except Exception:
            pass

    def _check_existing_outputs(self, image_path: str) -> bool:
        """Verifies if expected analysis output files already exist inside output folder."""
        image_name = Path(image_path).name
        img_folder = Path(self.output_dir) / image_name
        
        if not img_folder.exists():
            return False
            
        expected_files = [
            img_folder / f"{image_name}_overlay_preview.png",
            img_folder / f"{image_name}_labels_raw.tif",
            img_folder / f"{image_name}_cell_metrics.csv",
            img_folder / f"{image_name}_report.pdf"
        ]
        
        return all(f.exists() and f.stat().st_size > 0 for f in expected_files)

    def _parse_existing_outputs(self, image_path: str) -> dict:
        """Parses existing cell metrics CSV to reconstruct summary stats row for skipped image."""
        image_name = Path(image_path).name
        img_folder = Path(self.output_dir) / image_name
        cell_csv_path = img_folder / f"{image_name}_cell_metrics.csv"
        
        try:
            cell_count = 0
            areas = []
            diameters = []
            
            import csv
            with open(cell_csv_path, mode="r", newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                for row in reader:
                    if row:
                        cell_count += 1
                        areas.append(int(row[1]))
                        diameters.append(float(row[2]))
                        
            mean_area = float(np.mean(areas)) if areas else 0.0
            median_area = float(np.median(areas)) if areas else 0.0
            avg_diameter = float(np.mean(diameters)) if diameters else 0.0
            
            import PIL.Image
            import tifffile
            ext = Path(image_path).suffix.lower()
            if ext in [".tif", ".tiff"]:
                with tifffile.TiffFile(image_path) as tif:
                    shape = tif.series[0].shape
            else:
                with PIL.Image.open(image_path) as img:
                    size = img.size
                    shape = (size[1], size[0])
            
            h, w = shape[:2]
            density = cell_count / (h * w) if h * w > 0 else 0.0
            
            from lumen.core.services.gpu_service import gpu_service
            pref = self.parameters.get("backend_preference", "Use Global Setting")
            use_gpu, resolved_name = gpu_service.resolve_execution_backend(pref)
            
            # Check if this image was previously edited in run_manifest.json
            is_edited = False
            manifest_path = Path(self.output_dir) / "run_manifest.json"
            if manifest_path.exists():
                try:
                    import json
                    with open(manifest_path, mode="r", encoding="utf-8") as mf:
                        mdata = json.load(mf)
                    if "images" in mdata:
                        for img_rec in mdata["images"]:
                            if img_rec.get("image_name") == image_name:
                                is_edited = img_rec.get("edited", False) in [True, "True", "true", 1, "1"]
                                break
                except Exception:
                    pass
            
            from lumen.workflows.state import state
            from lumen.workflows.workflow_manager import workflow_manager
            wf = workflow_manager.get_workflow(state.current_workflow)
            wf_name = wf.name if wf else "Cell Segmentation"
            return {
                "image_name": image_name,
                "workflow": wf_name,
                "segmentation_mode": self.parameters.get("quality_mode", "Balanced"),
                "model_type": "cyto",
                "cell_count": cell_count,
                "mean_area_px": round(mean_area, 2),
                "median_area_px": round(median_area, 2),
                "average_diameter_px": round(avg_diameter, 2),
                "cell_density": f"{density:.2e}",
                "processing_time_s": 0.0,
                "used_gpu": "CUDA" if use_gpu else "CPU",
                "requested_backend": pref,
                "resolved_backend": resolved_name,
                "status": "SKIPPED_ALREADY_EXISTS",
                "edited": is_edited
            }
        except Exception as e:
            logger.error("BatchAnalysisWorker: Failed to parse existing CSV for %s: %s", image_name, e)
            return None

    @Slot()
    def run_batch(self):
        logger.info("BatchAnalysisWorker: Starting persistent loop for %d images with token %d.", 
                    len(self.image_paths), self.batch_token)
        
        model_cache = None
        model_cache_type = None
        model_cache_gpu = None
        
        for idx in range(self.start_idx, len(self.image_paths)):
            if self._cancel_event.is_set():
                break
                
            # Wait for pause/resume gate
            self._pause_event.wait()
            if self._cancel_event.is_set():
                break
                
            # Wait for output done gate (with 30 seconds timeout)
            success = self._output_done_event.wait(timeout=30.0)
            if not success:
                logger.warning("BatchAnalysisWorker (token=%d): Output generation timed out for previous image. Continuing...", self.batch_token)
                
            if self._cancel_event.is_set():
                break
                
            self._output_done_event.clear()
            
            image_path = self.image_paths[idx]
            logger.info("BatchAnalysisWorker (token=%d): Starting inference for image %d/%d: %s", 
                        self.batch_token, idx + 1, len(self.image_paths), Path(image_path).name)
            
            # Check resume safety skipped condition first
            if self._check_existing_outputs(image_path):
                logger.info("BatchAnalysisWorker (token=%d): Skipping %s - Outputs already exist.", 
                            self.batch_token, Path(image_path).name)
                parsed = self._parse_existing_outputs(image_path)
                if parsed:
                    self.image_completed.emit(self.batch_token, image_path, parsed)
                    # We continue to the next image immediately
                    continue
            
            try:
                self.progress_updated.emit(self.batch_token, 10)
                self.status_updated.emit(self.batch_token, "Loading original raw image...")
                
                # Step 1: Retrieve raw numpy array
                from lumen.processing.image_manager import image_manager
                if image_manager._current_path == image_path:
                    raw_arr = image_manager._raw_numpy_arr
                else:
                    raw_arr = None

                if raw_arr is None:
                    import tifffile
                    import PIL.Image
                    ext = Path(image_path).suffix.lower()
                    if ext in [".tif", ".tiff"]:
                        raw_arr = tifffile.imread(image_path)
                    else:
                        with PIL.Image.open(image_path) as pil_img:
                            raw_arr = np.asarray(pil_img)
                
                if raw_arr is None or raw_arr.size == 0:
                    raise ValueError("Could not access or load raw image array.")

                if raw_arr.ndim == 3 and raw_arr.shape[0] <= 10 and raw_arr.shape[2] > 10:
                    raw_arr = np.transpose(raw_arr, (1, 2, 0))

                self.progress_updated.emit(self.batch_token, 20)

                
                # Step 2: Heuristic routing
                if image_manager._current_path == image_path:
                    meta = image_manager.get_metadata()
                    modality = meta.get("classification", "Unknown Biological Imaging")
                else:
                    from lumen.workflows.image_classifier import classify_image
                    filename = Path(image_path).name
                    shape = raw_arr.shape
                    if len(shape) == 2:
                        channels = 1
                        mode = "grayscale"
                    else:
                        channels = shape[2] if len(shape) == 3 else 1
                        mode = "rgb" if channels >= 3 else "grayscale"
                    ext = Path(image_path).suffix.lower()
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
                resolved_model_type = determine_model_type(modality, Path(image_path).name, meta)
                resolved_channels = determine_channels(modality, meta, raw_arr)
                
                model_type = self.parameters.get("model_type_override") or resolved_model_type
                channels = self.parameters.get("channel_override") or resolved_channels
                
                quality_mode = self.parameters.get("quality_mode", "Balanced")
                quality_config = get_segmentation_config(quality_mode)
                
                flow_threshold = self.parameters.get("flow_threshold_override", quality_config["flow_threshold"])
                cellprob_threshold = self.parameters.get("cellprob_threshold_override", quality_config["cellprob_threshold"])
                resample = self.parameters.get("resample_override", quality_config["resample"])
                diameter = self.parameters.get("diameter_override", None)
                
                from lumen.core.services.gpu_service import gpu_service
                pref = self.parameters.get("backend_preference", "Use Global Setting")
                use_gpu, resolved_backend_name = gpu_service.resolve_execution_backend(pref)
                
                if self._cancel_event.is_set():
                    break
                    
                # Cache and reuse model
                if model_cache is None or model_cache_type != model_type or model_cache_gpu != use_gpu:
                    cellpose_dir = Path.home() / '.cellpose' / 'models'
                    model_download_needed = True
                    if cellpose_dir.exists():
                        model_files = list(cellpose_dir.glob(f"*{model_type}*"))
                        if model_files:
                            model_download_needed = False
                            
                    if model_download_needed:
                        logger.info("BatchAnalysisWorker: Local model weights not found. Preparing first-time download...")
                        self.status_updated.emit(self.batch_token, "Downloading Cellpose model weights...")
                    else:
                        self.status_updated.emit(self.batch_token, "Initializing Cellpose model...")
                        
                    from cellpose import models
                    model_cache = models.Cellpose(gpu=use_gpu, model_type=model_type)
                    model_cache_type = model_type
                    model_cache_gpu = use_gpu
                
                if self._cancel_event.is_set():
                    break

                self.progress_updated.emit(self.batch_token, 40)
                self.status_updated.emit(self.batch_token, "Running image preprocessing...")
                
                self.status_updated.emit(self.batch_token, "Executing Cellpose segmentation...")

                # Select active channel slice if in fluorescence workflow
                from lumen.workflows.state import state
                input_arr = raw_arr
                eval_channels = channels
                if state.current_workflow == "fluorescence" and raw_arr.ndim == 3:
                    seg_channel_idx = state.segmentation_channel
                    if seg_channel_idx >= 0 and seg_channel_idx < raw_arr.shape[2]:
                        input_arr = raw_arr[..., seg_channel_idx]
                        eval_channels = [0, 0] # Grayscale eval for single 2D slice
                        logger.info("BatchAnalysisWorker: Fluorescence mode active. Segmenting channel index %d as grayscale.", seg_channel_idx)

                # Apply non-destructive preprocessing pipeline to segmentation input
                from lumen.processing.image_manager import image_manager
                input_arr = image_manager.preprocess_array(input_arr)

                logger.info("BatchAnalysisWorker (token=%d): Running model.eval on model_type='%s', gpu=%s", 
                            self.batch_token, model_type, use_gpu)
                
                start_time = time.time()
                masks, flows, styles, diams = model_cache.eval(
                    input_arr,
                    channels=eval_channels,
                    flow_threshold=flow_threshold,
                    cellprob_threshold=cellprob_threshold,
                    resample=resample,
                    diameter=diameter
                )

                elapsed = time.time() - start_time
                
                if self._cancel_event.is_set():
                    break

                self.progress_updated.emit(self.batch_token, 80)
                self.status_updated.emit(self.batch_token, "Processing and packaging results...")

                unique_labels = np.unique(masks)
                valid_labels = [label for label in unique_labels if label != 0]
                cell_count = len(valid_labels)
                
                cell_metrics = {}
                cell_areas = []
                for label in valid_labels:
                    indices = np.argwhere(masks == label)
                    area = len(indices)
                    cell_areas.append(area)
                    
                    if area > 0:
                        mean_y, mean_x = np.mean(indices, axis=0)
                        diameter_val = round(2 * np.sqrt(area / np.pi), 2)
                        cell_metrics[int(label)] = {
                            "area_px": int(area),
                            "centroid": (round(float(mean_x), 1), round(float(mean_y), 1)),
                            "diameter_px": float(diameter_val),
                            "diameter_estimate": float(diameter_val)
                        }
                
                if cell_count > 0:
                    mean_cell_area_px = float(np.mean(cell_areas))
                    median_cell_area_px = float(np.median(cell_areas))
                    avg_diameter = float(np.mean(diams)) if diams is not None and len(np.atleast_1d(diams)) > 0 else 0.0
                else:
                    mean_cell_area_px = 0.0
                    median_cell_area_px = 0.0
                    avg_diameter = 0.0
                    
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
                
                self.progress_updated.emit(self.batch_token, 100)
                self.status_updated.emit(self.batch_token, "Analysis completed successfully.")
                self.image_completed.emit(self.batch_token, image_path, results_dict)

            except Exception as e:
                logger.error("BatchAnalysisWorker (token=%d): Pipeline execution failed for %s: %s", 
                             self.batch_token, image_path, e, exc_info=True)
                self.image_failed.emit(self.batch_token, image_path, f"Segmentation failed: {str(e)}")
                
            finally:
                # Explicit cleanup of temporary arrays per-image to prevent memory leak
                if 'raw_arr' in locals():
                    del raw_arr
                if 'masks' in locals():
                    del masks
                if 'flows' in locals():
                    del flows
                if 'styles' in locals():
                    del styles
                if 'results_dict' in locals():
                    del results_dict
                import gc
                gc.collect()

            # Periodically empty GPU CUDA cache to prevent VRAM runaway
            if idx > 0 and idx % 15 == 0:
                self._empty_gpu_cache()

        # Final cleanup
        model_cache = None
        self._empty_gpu_cache()
        logger.info("BatchAnalysisWorker (token=%d): Persistent loop finished.", self.batch_token)
        self.finished.emit(self.batch_token)


# Global processing manager
processing_manager = ProcessingManager()

