import os
import csv
import time
from pathlib import Path
import numpy as np
import tifffile
import PIL.Image
from PySide6.QtCore import QObject, Signal, QTimer, Qt, QThread, Slot
from lumen.core.logger import logger
from lumen.workflows.state import state
from lumen.processing.processing_manager import AnalysisWorker, BatchAnalysisWorker
from lumen.pages.results_page import (
    generate_overlay_image,
    export_cell_metrics_csv,
    export_pdf_report
)

class BatchProcessingManager(QObject):
    # Signals for UI connection
    batch_started = Signal(int)                          # total images
    batch_progress_updated = Signal(int, int, str)       # completed, failed, current_image_name
    batch_finished = Signal(int, int, str)               # completed, failed, results_dir
    batch_cancelled = Signal()
    batch_paused = Signal()
    batch_resumed = Signal()

    # Internal signal: relays image result to main thread for Qt-safe output generation.
    # Connected with Qt.QueuedConnection so the slot always runs on the main thread,
    # preventing the silent Qt crash caused by creating QTextDocument/QPdfWriter on worker threads.
    _image_result_ready = Signal(str, dict)              # image_path, results_dict

    def __init__(self):
        super().__init__()
        self.image_paths = []
        self.parameters = {}
        self.output_dir = ""
        
        self.current_idx = 0
        self.completed_count = 0
        self.failed_count = 0
        self.skipped_count = 0
        self.active_worker = None
        self.batch_thread = None
        self.batch_token = 0
        self._destroying_worker_ref = None
        self._on_cleanup_finished_callback = None
        self._is_cancelled = False
        self._is_paused_requested = False
        self._batch_ready_for_new_run = True
        
        self.start_time = 0.0
        self.image_start_time = 0.0
        self.image_runtimes = []
        self.summary_records = []
        self.resolved_backend = ""
        self.lifecycle_state = "IDLE"
        self.elapsed_time_accumulated = 0.0
        self.run_start_time = 0.0
        self.end_time = 0.0

        # Wire internal relay signal with QueuedConnection so output generation
        # always executes on the main thread, even when emitted from a worker thread.
        self._image_result_ready.connect(
            self._generate_outputs_on_main_thread,
            Qt.QueuedConnection
        )

    def prepare_batch(self, folder_path: str, parameters: dict, recursive: bool = False) -> int:
        """Scans folder for microscopy files, calculates estimated runtime, and prepares outputs."""
        if self.lifecycle_state == "RUNNING":
            logger.warning("BatchManager: Cannot prepare batch while running.")
            return 0

        if not getattr(self, "_batch_ready_for_new_run", True):
            logger.warning("BatchManager: Cannot prepare batch because previous run is not fully finalized.")
            state.batch_status = "Warning: Previous batch run is still finalizing. Please wait."
            return 0

        self._harden_lifecycle_reset()
        self.lifecycle_state = "IDLE"
        self.image_paths = []
        self.parameters = parameters
        self._is_cancelled = False
        
        # Supported extensions
        valid_exts = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
        
        folder = Path(folder_path)
        if not folder.exists() or not folder.is_dir():
            return 0
            
        pattern = "**/*" if recursive else "*"
        for path in folder.glob(pattern):
            if path.is_file() and path.suffix.lower() in valid_exts:
                # Exclude batch_results folder if it is nested inside folder_path
                if "batch_results" not in path.parts:
                    self.image_paths.append(str(path))
                    
        # Sort paths for deterministic sequential processing
        self.image_paths.sort()
        
        # Define default output directory: inside folder_path/batch_results
        self.output_dir = str(folder / "batch_results")
        
        return len(self.image_paths)

    def get_estimated_runtime_minutes(self) -> float:
        """Estimates batch processing time in minutes."""
        segmentation_method = self.parameters.get("segmentation_method", "AI Segmentation")
        if segmentation_method != "AI Segmentation":
            per_image_time = 5.0
            total_time_s = per_image_time * len(self.image_paths)
            return round(total_time_s / 60.0, 1)

        # Baseline per-image times based on backend mode (in seconds)
        from lumen.core.services.gpu_service import gpu_service
        pref = self.parameters.get("backend_preference", "Use Global Setting")
        use_gpu, resolved_name = gpu_service.resolve_execution_backend(pref)
        is_gpu = use_gpu
        
        # Add baseline overhead (in seconds) for disk I/O, overlay generation, and PDF report rendering
        io_overhead = 4.5 if is_gpu else 6.0
        
        q_mode = self.parameters.get("quality_mode", "Balanced").lower()
        if is_gpu:
            if q_mode == "fast":
                per_image_time = 1.2 + io_overhead
            elif q_mode == "precise":
                per_image_time = 3.5 + io_overhead
            elif q_mode == "sensitive":
                per_image_time = 3.8 + io_overhead
            else:
                per_image_time = 2.5 + io_overhead
        else:
            if q_mode == "fast":
                per_image_time = 8.5 + io_overhead
            elif q_mode == "precise":
                per_image_time = 17.5 + io_overhead
            elif q_mode == "sensitive":
                per_image_time = 21.0 + io_overhead
            else:
                per_image_time = 15.0 + io_overhead
                
        total_time_s = per_image_time * len(self.image_paths)
        return round(total_time_s / 60.0, 1)

    def start_batch(self):
        """Starts batch processing sequential loop."""
        if self.lifecycle_state in ("RUNNING", "PAUSED"):
            logger.warning("BatchManager: Start requested but batch is already active (state: %s).", self.lifecycle_state)
            return

        if not getattr(self, "_batch_ready_for_new_run", True):
            logger.warning("BatchManager: Cannot start batch because previous run is not fully finalized.")
            state.batch_status = "Error: Cannot start batch. Previous run is still finalizing."
            return

        if self.batch_thread and self.batch_thread.isRunning():
            logger.warning("BatchManager: Active worker thread is already running. Blocked execution.")
            return

        self._batch_ready_for_new_run = False
        logger.info("BatchManager: Starting isolated batch run.")

        # Resolve backend preference exactly once at batch start
        segmentation_method = self.parameters.get("segmentation_method", "AI Segmentation")
        if segmentation_method != "AI Segmentation":
            self.resolved_backend = "CPU (Alternative)"
        else:
            from lumen.core.services.gpu_service import gpu_service
            pref = self.parameters.get("backend_preference", "Use Global Setting")
            use_gpu, resolved_name = gpu_service.resolve_execution_backend(pref)
            self.resolved_backend = "GPU (CUDA)" if use_gpu else resolved_name

        if not self.image_paths:
            logger.warning("BatchManager: Start requested but image list is empty.")
            self.batch_finished.emit(0, 0, "")
            return
            
        self.current_idx = 0
        self.completed_count = 0
        self.failed_count = 0
        self.skipped_count = 0
        self.summary_records = []
        self.start_time = time.time()
        self.run_start_time = time.perf_counter()
        self.elapsed_time_accumulated = 0.0
        self.end_time = 0.0
        self.image_start_time = time.time()
        self.image_runtimes = []
        self._is_cancelled = False
        self._is_paused_requested = False
        self.lifecycle_state = "RUNNING"
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Save batch metadata for reproducibility
        self._save_batch_metadata()
        
        # Update AppState properties
        state.is_batch_active = True
        state.batch_progress = 0
        state.batch_status = f"Starting analysis: 0/{len(self.image_paths)}"
        
        self.batch_started.emit(len(self.image_paths))
        
        # Initialize token and persistent thread & worker
        if not hasattr(self, "batch_token"):
            self.batch_token = 0
        self.batch_token += 1

        self.batch_thread = QThread()
        self.active_worker = BatchAnalysisWorker(
            image_paths=self.image_paths,
            parameters=self.parameters,
            output_dir=self.output_dir,
            start_idx=0,
            batch_token=self.batch_token
        )
        self.active_worker.moveToThread(self.batch_thread)
        
        # Connect signals
        self.batch_thread.started.connect(self.active_worker.run_batch)
        self.active_worker.image_completed.connect(self._on_image_completed, Qt.QueuedConnection)
        self.active_worker.image_failed.connect(self._on_image_failed, Qt.QueuedConnection)
        self.active_worker.status_updated.connect(self._on_worker_status_updated, Qt.QueuedConnection)
        
        # Asynchronous cleanup chain (no wait blocking)
        self.active_worker.finished.connect(self.batch_thread.quit, Qt.QueuedConnection)
        self.batch_thread.finished.connect(self._on_thread_finished_cleanup, Qt.QueuedConnection)
        
        # Start execution
        self.batch_thread.start()

    def cancel_batch(self):
        """Cancels current batch analysis."""
        if self.lifecycle_state in ("CANCELLED", "COMPLETED", "CANCELLING"):
            logger.warning("BatchManager: Cancel requested but state is already %s.", self.lifecycle_state)
            return

        self._is_cancelled = True
        self.lifecycle_state = "CANCELLING"
        logger.info("BatchManager: Cancellation requested.")

        # Accumulate elapsed time up to cancellation point
        if self.run_start_time > 0.0:
            self.elapsed_time_accumulated += (time.perf_counter() - self.run_start_time)
            self.run_start_time = 0.0
        
        if self.active_worker:
            self.active_worker.cancel()
        else:
            self._on_cancel_finished()

    @Slot()
    def _on_thread_finished_cleanup(self):
        logger.info("BatchManager: Host QThread finished. Performing cleanup.")
        # Asynchronously clean up worker and thread references
        if self.active_worker:
            self.active_worker.deleteLater()
            self.active_worker = None
        if self.batch_thread:
            self.batch_thread.deleteLater()
            self.batch_thread = None
            
        self._destroying_worker_ref = None
        self._on_cleanup_finished_callback = None
        
        # Finalize the run state transition
        if self._is_cancelled:
            self._on_cancel_finished()
        else:
            self.finalize_batch()

    def _on_cancel_finished(self):
        self.lifecycle_state = "CANCELLED"
        state.is_batch_active = False
        self._write_master_csv()
        self._write_run_manifest()
        self.batch_cancelled.emit()
        self._cleanup_active_worker()
        self._batch_ready_for_new_run = True
        logger.info("BatchManager: Previous batch fully finalized. Ready for new batch.")

    @Slot(int, str, dict)
    def _on_image_completed(self, token: int, image_path: str, results: dict):
        if token != self.batch_token:
            logger.warning("BatchManager: Discarding stale completed signal (expected token %d, got %d).", 
                           self.batch_token, token)
            return
            
        logger.info("BatchManager (token=%d): Worker completed image: %s", token, Path(image_path).name)
        
        if results.get("status") == "SKIPPED_ALREADY_EXISTS":
            # Rehydrate record for skipped image without saving duplicate files
            self.summary_records.append(results)
            self.completed_count += 1
            self.skipped_count += 1
            self._advance_batch_after_image()
        else:
            # Save results on main thread
            self._generate_outputs_on_main_thread(image_path, results)

    def _generate_outputs_on_main_thread(self, image_path: str, results: dict):
        """Generates all file outputs (overlay PNG, label TIFF, CSV, PDF) on the main thread.
        
        Each output step runs independently so a failure in one step (e.g. PDF render)
        does not abort the other outputs (overlay, CSV) for that image.
        Qt render objects (QTextDocument, QPdfWriter, QImage) are safe to use here.
        """
        image_name = Path(image_path).name
        output_success = True  # track whether this image counts as completed
        
        # Step 1: Create output sub-folder
        img_folder = Path(self.output_dir) / image_name
        try:
            os.makedirs(img_folder, exist_ok=True)
        except Exception as e:
            logger.error("BatchManager: Failed to create output folder for %s: %s", image_name, e)
            self._record_output_failure(image_name)
            self._advance_batch_after_image()
            return

        # Step 2: Write visual overlay PNG
        try:
            overlay_path = img_folder / f"{image_name}_overlay_preview.png"
            overlay_pil = generate_overlay_image(image_path, results.get("masks"))
            overlay_pil.save(str(overlay_path))
        except Exception as e:
            logger.warning("BatchManager: Overlay PNG generation failed for %s: %s", image_name, e)
            output_success = False

        # Step 3: Save raw 16-bit label TIFF
        try:
            raw_labels_path = img_folder / f"{image_name}_labels_raw.tif"
            tifffile.imwrite(str(raw_labels_path), results["masks"].astype(np.uint16))
        except Exception as e:
            logger.warning("BatchManager: Label TIFF write failed for %s: %s", image_name, e)
            output_success = False

        # Step 4: Save per-cell CSV
        try:
            cell_csv_path = img_folder / f"{image_name}_cell_metrics.csv"
            export_cell_metrics_csv(str(cell_csv_path), results["cell_metrics"])
        except Exception as e:
            logger.warning("BatchManager: CSV export failed for %s: %s", image_name, e)
            output_success = False

        # Step 5: Save PDF report (requires main thread — Qt render objects used here)
        try:
            seg_method = self.parameters.get("segmentation_method", "AI Segmentation")
            if seg_method != "AI Segmentation":
                seg_mode_str = "Alternative"
            else:
                seg_mode_str = self.parameters.get("quality_mode", "Balanced")
            
            pdf_path = img_folder / f"{image_name}_report.pdf"
            export_pdf_report(
                str(pdf_path),
                image_path,
                seg_mode_str,
                state.current_workflow,
                results
            )
        except Exception as e:
            logger.warning("BatchManager: PDF report failed for %s: %s", image_name, e)
            # PDF failure is non-fatal — overlay, TIFF, and CSV may still have succeeded

        # Step 6: Build summary record and release numpy memory
        try:
            seg_method = self.parameters.get("segmentation_method", "AI Segmentation")
            if seg_method != "AI Segmentation":
                seg_mode_str = "Alternative"
            else:
                seg_mode_str = self.parameters.get("quality_mode", "Balanced")

            record = {
                "image_name": image_name,
                "workflow": state.current_workflow or "Cell Counting",
                "segmentation_mode": seg_mode_str,
                "model_type": results.get("model_type", "cyto"),
                "cell_count": results.get("cell_count", 0),
                "mean_area_px": round(results.get("mean_cell_area_px", 0.0), 2),
                "median_area_px": round(results.get("median_cell_area_px", 0.0), 2),
                "average_diameter_px": round(results.get("average_diameter_px", 0.0), 2),
                "cell_density": f"{results.get('cell_density', 0.0):.2e}",
                "processing_time_s": results.get("processing_time_s", 0.0),
                "used_gpu": "CUDA" if results.get("used_gpu") else "CPU",
                "requested_backend": self.resolved_backend if seg_method != "AI Segmentation" else self.parameters.get("backend_preference", "Use Global Setting"),
                "resolved_backend": results.get("resolved_backend", "CPU"),
                "status": "SUCCESS",
                "edited": False
            }
            self.summary_records.append(record)
            self.completed_count += 1
        except Exception as e:
            logger.error("BatchManager: Summary record build failed for %s: %s", image_name, e)
            self._record_output_failure(image_name)

        # Release numpy mask array to prevent memory accumulation across large batches
        if "masks" in results:
            del results["masks"]

        self._advance_batch_after_image()

    def _record_output_failure(self, image_name: str):
        """Appends a FAILED summary record and increments the failure counter."""
        record = self._build_failed_record(image_name, "OUTPUT_GENERATION_FAILED")
        self.summary_records.append(record)
        self.failed_count += 1

    def _advance_batch_after_image(self):
        duration = time.time() - self.image_start_time
        self.image_runtimes.append(duration)
        
        self.current_idx += 1
        
        # State transition guards based on state machine
        if self.lifecycle_state == "CANCELLING":
            # Wait for worker thread to exit and fire finished signal
            if self.active_worker:
                self.active_worker.mark_output_done()
        elif self.lifecycle_state == "PAUSED":
            # We don't advance the worker loop yet. The UI handles resume.
            pass
        elif self._is_paused_requested:
            self.lifecycle_state = "PAUSED"
            state.is_batch_active = False
            if self.run_start_time > 0.0:
                self.elapsed_time_accumulated += (time.perf_counter() - self.run_start_time)
                self.run_start_time = 0.0
            
            # Request worker to pause (clear its pause event)
            if self.active_worker:
                self.active_worker.pause()
                self.active_worker.mark_output_done()
            
            self.batch_paused.emit()
            logger.info("BatchManager: Batch paused on image boundary.")
        elif self.current_idx >= len(self.image_paths):
            # Normal end of batch. Unblock worker so it exits its run loop
            if self.active_worker:
                self.active_worker.mark_output_done()
        else:
            # Advance to next image
            image_path = self.image_paths[self.current_idx]
            image_name = Path(image_path).name
            state.batch_status = f"Processing {self.current_idx + 1}/{len(self.image_paths)}: {image_name}"
            progress_pct = int((self.current_idx / len(self.image_paths)) * 100)
            state.batch_progress = progress_pct
            self.batch_progress_updated.emit(self.completed_count, self.failed_count, image_name)
            
            self.image_start_time = time.time()
            if self.active_worker:
                self.active_worker.mark_output_done()

    def _on_worker_destroyed(self):
        logger.info("BatchManager: Worker thread QObject destroyed.")
        self._destroying_worker_ref = None
        callback = getattr(self, "_on_cleanup_finished_callback", None)
        if callback:
            self._on_cleanup_finished_callback = None
            QTimer.singleShot(0, callback)

    @Slot(int, str, str)
    def _on_image_failed(self, token: int, image_path: str, error_msg: str):
        if token != self.batch_token:
            logger.warning("BatchManager: Discarding stale failed signal (expected token %d, got %d).", 
                           self.batch_token, token)
            return
            
        image_name = Path(image_path).name
        logger.error("BatchManager (token=%d): Worker failed image %s: %s", token, image_name, error_msg)
        
        record = self._build_failed_record(image_name, f"FAILED: {error_msg}")
        self.summary_records.append(record)
        self.failed_count += 1
        
        self._advance_batch_after_image()

    @Slot(int, str)
    def _on_worker_status_updated(self, token: int, status: str):
        if token != self.batch_token:
            return
        image_name = Path(self.image_paths[self.current_idx]).name if self.current_idx < len(self.image_paths) else ""
        if image_name:
            state.batch_status = f"Processing {self.current_idx + 1}/{len(self.image_paths)} ({image_name}): {status}"
        else:
            state.batch_status = f"Processing: {status}"

    def _build_failed_record(self, image_name: str, status_msg: str) -> dict:
        return {
            "image_name": image_name,
            "workflow": state.current_workflow or "Cell Counting",
            "segmentation_mode": self.parameters.get("quality_mode", "Balanced"),
            "model_type": "-",
            "cell_count": 0,
            "mean_area_px": 0.0,
            "median_area_px": 0.0,
            "average_diameter_px": 0.0,
            "cell_density": "0.00e+00",
            "processing_time_s": 0.0,
            "used_gpu": "-",
            "requested_backend": self.parameters.get("backend_preference", "Use Global Setting"),
            "resolved_backend": "-",
            "status": status_msg,
            "edited": False
        }

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
            
            return {
                "image_name": image_name,
                "workflow": state.current_workflow or "Cell Counting",
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
            logger.error("BatchManager: Failed to parse existing CSV for %s: %s", image_name, e)
            return None

    def _save_batch_metadata(self):
        """Saves reproducibility foundation parameters."""
        meta_path = Path(self.output_dir) / "batch_metadata.txt"
        
        try:
            seg_method = self.parameters.get("segmentation_method", "AI Segmentation")
            if seg_method != "AI Segmentation":
                seg_mode_str = "Alternative"
            else:
                seg_mode_str = self.parameters.get("quality_mode", "Balanced")
            
            with open(meta_path, mode="w", encoding="utf-8") as f:
                f.write("====================================================\n")
                f.write("          LUMEN BATCH ANALYSIS REPRODUCIBILITY METADATA\n")
                f.write("====================================================\n")
                f.write(f"Lumen Version: 0.2.0\n")
                f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Active Workflow: {state.current_workflow or 'Cell Counting'}\n")
                f.write(f"Segmentation Method: {seg_method}\n")
                f.write(f"Segmentation Mode Preset: {seg_mode_str}\n")
                
                if seg_method != "AI Segmentation":
                    f.write(f"Resolved Model: {seg_method}\n")
                    f.write(f"Requested Backend: CPU (Alternative)\n")
                    f.write(f"Resolved Backend: CPU\n")
                else:
                    from lumen.workflows.cellpose_routing import determine_model_type
                    modality = "Fluorescence Microscopy"
                    sample_name = Path(self.image_paths[0]).name if self.image_paths else ""
                    model_used = determine_model_type(modality, sample_name)
                    f.write(f"Resolved Cellpose Model: {model_used}\n")
                    
                    from lumen.core.services.gpu_service import gpu_service
                    pref = self.parameters.get("backend_preference", "Use Global Setting")
                    use_gpu, resolved_name = gpu_service.resolve_execution_backend(pref)
                    f.write(f"Requested Backend: {pref}\n")
                    f.write(f"Resolved Backend: {resolved_name}\n")
                    
                f.write(f"Total Batch Image Files: {len(self.image_paths)}\n")
                f.write("====================================================\n")
            logger.info("BatchManager: Saved batch metadata.")
        except Exception as e:
            logger.error("BatchManager: Failed to write metadata: %s", e)

    def _write_master_csv(self):
        """Compiles self.summary_records into batch_summary.csv."""
        summary_path = Path(self.output_dir) / "batch_summary.csv"
        fields = [
            "image_name", "workflow", "segmentation_mode", "model_type",
            "cell_count", "mean_area_px", "median_area_px", "average_diameter_px",
            "cell_density", "processing_time_s", "used_gpu", "requested_backend", "resolved_backend", "status",
            "edited"
        ]
        
        try:
            with open(summary_path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                for rec in self.summary_records:
                    writer.writerow(rec)
            logger.info("BatchManager: Wrote master summary CSV: %s", summary_path)
        except Exception as e:
            logger.error("BatchManager: Failed to write master summary CSV: %s", e)

    def _write_run_manifest(self):
        """Compiles run parameters and image records into run_manifest.json."""
        manifest_path = Path(self.output_dir) / "run_manifest.json"
        import json
        
        manifest_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "workflow": state.current_workflow or "Cell Counting",
            "backend": self.resolved_backend,
            "segmentation_mode": self.parameters.get("quality_mode", "Balanced"),
            "images": self.summary_records
        }
        
        try:
            with open(manifest_path, mode="w", encoding="utf-8") as f:
                json.dump(manifest_data, f, indent=2)
            logger.info("BatchManager: Wrote run manifest JSON: %s", manifest_path)
        except Exception as e:
            logger.error("BatchManager: Failed to write run manifest JSON: %s", e)

    def finalize_batch(self):
        """Compiles outputs and notifies listeners."""
        self.lifecycle_state = "COMPLETED"
        if self.run_start_time > 0.0:
            self.elapsed_time_accumulated += (time.perf_counter() - self.run_start_time)
            self.run_start_time = 0.0
        self.end_time = time.perf_counter()

        self._write_master_csv()
        self._write_run_manifest()
        
        state.is_batch_active = False
        state.batch_progress = 100
        state.batch_status = f"Batch completed. Successful: {self.completed_count}, Failed: {self.failed_count}."
        
        logger.info("BatchManager: Batch completed. Total elapsed: %.2f minutes", self.get_elapsed_seconds() / 60.0)
        self._cleanup_active_worker()
        self._batch_ready_for_new_run = True
        logger.info("BatchManager: Previous batch fully finalized. Ready for new batch.")
        self.batch_finished.emit(self.completed_count, self.failed_count, self.output_dir)

    def pause_batch(self):
        """Requests batch pause at next iteration boundary."""
        if self.lifecycle_state != "RUNNING":
            return
        logger.info("BatchManager: Pause requested.")
        self._is_paused_requested = True

    def _on_pause_finished(self):
        self.lifecycle_state = "PAUSED"
        state.is_batch_active = False
        if self.run_start_time > 0.0:
            self.elapsed_time_accumulated += (time.perf_counter() - self.run_start_time)
            self.run_start_time = 0.0
        self.batch_paused.emit()

    def resume_batch(self):
        """Resumes batch processing from current index."""
        if self.lifecycle_state != "PAUSED":
            logger.warning("BatchManager: Resume requested but state is %s (expected PAUSED).", self.lifecycle_state)
            return
        logger.info("BatchManager: Resuming batch analysis.")
        self.lifecycle_state = "RUNNING"
        self._is_paused_requested = False
        self.run_start_time = time.perf_counter()
        
        state.is_batch_active = True
        state.batch_status = f"Processing {self.current_idx + 1}/{len(self.image_paths)}"
        
        self.batch_resumed.emit()
        self.image_start_time = time.time()
        
        # Resume the worker
        if self.active_worker:
            self.active_worker.resume()

    def get_elapsed_seconds(self) -> float:
        """Returns elapsed run time in seconds."""
        if self.lifecycle_state == "RUNNING" and self.run_start_time > 0.0:
            return self.elapsed_time_accumulated + (time.perf_counter() - self.run_start_time)
        return self.elapsed_time_accumulated

    def get_elapsed_time_hhmmss(self) -> str:
        """Returns elapsed run time formatted as HH:MM:SS."""
        elapsed = self.get_elapsed_seconds()
        hours = int(elapsed // 3600)
        mins = int((elapsed % 3600) // 60)
        secs = int(elapsed % 60)
        return f"{hours:02d}:{mins:02d}:{secs:02d}"

    def get_completed_time_str(self) -> str:
        """Returns completed time formatted as 'Xm Ys' or 'Zs'."""
        elapsed = self.get_elapsed_seconds()
        mins = int(elapsed // 60)
        secs = int(elapsed % 60)
        if mins > 0:
            return f"{mins}m {secs}s"
        return f"{secs}s"

    def _cleanup_active_worker(self, callback=None):
        """Ensures active worker thread is gracefully quit, waited on, and deleted."""
        if not self.active_worker:
            if callback:
                callback()
            return

        self._on_cleanup_finished_callback = callback
        
        logger.info("BatchManager: Performing safe asynchronous worker thread cleanup.")
        worker = self.active_worker
        thread = self.batch_thread
        
        self.active_worker = None
        self.batch_thread = None
        
        import weakref
        self._destroying_worker_ref = weakref.ref(worker) if worker else None
        
        try:
            if worker:
                try:
                    worker.image_completed.disconnect()
                except Exception:
                    pass
                try:
                    worker.image_failed.disconnect()
                except Exception:
                    pass
                try:
                    worker.status_updated.disconnect()
                except Exception:
                    pass
                try:
                    worker.finished.disconnect()
                except Exception:
                    pass
                
                # Connect worker destroyed to final clean callback trigger
                worker.destroyed.connect(self._on_worker_destroyed)
                worker.cancel()
                
            if thread:
                try:
                    thread.finished.disconnect()
                except Exception:
                    pass
                
                # Quit the thread asynchronously (no wait() on the GUI thread)
                if thread.isRunning():
                    thread.quit()
                    # Delete the thread when it exits
                    thread.finished.connect(thread.deleteLater)
                else:
                    thread.deleteLater()
                    
            if worker:
                worker.deleteLater()
            else:
                if callback:
                    self._on_cleanup_finished_callback = None
                    QTimer.singleShot(0, callback)
        except Exception as e:
            logger.error("BatchManager: Exception during worker cleanup: %s", e)
            self._destroying_worker_ref = None
            if callback:
                callback()

    def _harden_lifecycle_reset(self):
        logger.info("BatchManager: Performing lifecycle sanitation pass.")
        # 1. Clean up active worker if exists
        if self.active_worker or self.batch_thread:
            logger.info("BatchManager: Active worker exists during reset, cleaning up.")
            self._cleanup_active_worker()
            
        # 2. Wait for any destroying worker to be fully joined (thread finished)
        destroying_worker = getattr(self, "_destroying_worker_ref", None)
        if destroying_worker is not None and destroying_worker() is not None:
            worker = destroying_worker()
            logger.info("BatchManager: Lingering destroying worker found, waiting for destruction.")

        # 3. Use local event-loop flush only as a timeout/fallback safety path if destruction stalls
        destroying_worker = getattr(self, "_destroying_worker_ref", None)
        if destroying_worker is not None and destroying_worker() is not None:
            worker = destroying_worker()
            logger.info("BatchManager: Destructor stalled. Flushing event loop for deferred deletions...")
            from PySide6.QtCore import QEventLoop
            loop = QEventLoop()
            try:
                worker.destroyed.connect(loop.quit)
            except Exception:
                pass
            
            timeout_timer = QTimer()
            timeout_timer.setSingleShot(True)
            timeout_timer.timeout.connect(loop.quit)
            timeout_timer.start(500)  # 500ms safety timeout
            loop.exec()
            timeout_timer.stop()
                
        self._destroying_worker_ref = None
        self._on_cleanup_finished_callback = None
        self._is_cancelled = False
        self._is_paused_requested = False
        self._batch_ready_for_new_run = True
        logger.info("BatchManager: Lifecycle clean. Ready for new batch.")

    def reset_batch(self):
        """Resets the batch manager singleton to fresh idle state."""
        logger.info("BatchManager: Resetting batch state.")
        self._harden_lifecycle_reset()
        self.lifecycle_state = "IDLE"
        self._is_cancelled = True
        self.image_paths = []
        self.completed_count = 0
        self.failed_count = 0
        self.skipped_count = 0
        self.summary_records = []
        self.image_runtimes = []
        self.resolved_backend = ""
        self.elapsed_time_accumulated = 0.0
        self.run_start_time = 0.0
        state.is_batch_active = False

# Global singleton instance of BatchProcessingManager
batch_manager = BatchProcessingManager()
