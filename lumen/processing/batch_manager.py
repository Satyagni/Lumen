import os
import csv
import time
from pathlib import Path
import numpy as np
import tifffile
import PIL.Image
from PySide6.QtCore import QObject, Signal, QTimer, Qt
from lumen.core.logger import logger
from lumen.workflows.state import state
from lumen.processing.processing_manager import AnalysisWorker
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
        self._is_cancelled = False
        
        self.start_time = 0.0
        self.image_start_time = 0.0
        self.image_runtimes = []
        self.summary_records = []
        self.resolved_backend = ""
        self.lifecycle_state = ""

        # Wire internal relay signal with QueuedConnection so output generation
        # always executes on the main thread, even when emitted from a worker thread.
        self._image_result_ready.connect(
            self._generate_outputs_on_main_thread,
            Qt.QueuedConnection
        )

    def prepare_batch(self, folder_path: str, parameters: dict, recursive: bool = False) -> int:
        """Scans folder for microscopy files, calculates estimated runtime, and prepares outputs."""
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
        self.image_start_time = 0.0
        self.image_runtimes = []
        self._is_cancelled = False
        self.lifecycle_state = "RUNNING"
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Save batch metadata for reproducibility
        self._save_batch_metadata()
        
        # Update AppState properties
        state.is_batch_active = True
        state.batch_progress = 0
        state.batch_status = f"Starting analysis: 0/{len(self.image_paths)}"
        state.batch_results_dir = self.output_dir
        
        self.batch_started.emit(len(self.image_paths))
        
        # Start sequential run
        QTimer.singleShot(0, self.analyze_next_image)

    def cancel_batch(self):
        """Cancels current batch analysis."""
        self._is_cancelled = True
        self.lifecycle_state = "CANCELLING"
        logger.info("BatchManager: Cancellation requested.")
        
        if self.active_worker and self.active_worker.isRunning():
            # Signal cancellation — do NOT call .wait() here as it blocks the main thread
            # and causes a UI freeze. The worker will finish on its own and emit finished.
            self.active_worker.cancel()
        else:
            self._on_cancel_finished()

    def _on_worker_thread_finished(self):
        if self._is_cancelled:
            logger.info("BatchManager: Worker thread finished after cancellation request.")
            self._on_cancel_finished()

    def _on_cancel_finished(self):
        self.lifecycle_state = "CANCELLED"
        state.is_batch_active = False
        self._write_master_csv()
        self._write_run_manifest()
        self.batch_cancelled.emit()
        self.active_worker = None

    def analyze_next_image(self):
        if self._is_cancelled:
            return
            
        if self.current_idx >= len(self.image_paths):
            self.finalize_batch()
            return
            
        image_path = self.image_paths[self.current_idx]
        image_name = Path(image_path).name
        
        state.batch_status = f"Processing {self.current_idx + 1}/{len(self.image_paths)}: {image_name}"
        progress_pct = int((self.current_idx / len(self.image_paths)) * 100)
        state.batch_progress = progress_pct
        self.batch_progress_updated.emit(self.completed_count, self.failed_count, image_name)
        
        self.image_start_time = time.time()
        
        # Resume-safe check: Check if expected outputs already exist
        if self._check_existing_outputs(image_path):
            logger.info("BatchManager: Skipping %s - Outputs already exist.", image_name)
            parsed = self._parse_existing_outputs(image_path)
            if parsed:
                self.summary_records.append(parsed)
                self.completed_count += 1
                self.skipped_count += 1
                self.current_idx += 1
                
                # Record skip runtime (very short)
                duration = time.time() - self.image_start_time
                self.image_runtimes.append(duration)
                
                QTimer.singleShot(0, self.analyze_next_image)
                return
            else:
                logger.warning("BatchManager: Existing CSV parse failed for %s. Rerunning analysis.", image_name)

        # Start AnalysisWorker background thread
        worker_params = self.parameters.copy()
        
        self.active_worker = AnalysisWorker(image_path, worker_params)
        self.active_worker.finished_successfully.connect(self._on_image_completed)
        self.active_worker.failed.connect(self._on_image_failed, Qt.QueuedConnection)
        self.active_worker.finished.connect(self._on_worker_thread_finished, Qt.QueuedConnection)
        self.active_worker.start()

    def _on_image_completed(self, results: dict):
        """Called on the AnalysisWorker thread. Acts as a thin relay only.
        
        IMPORTANT: Do NOT perform any Qt GUI/render operations here (no QTextDocument,
        QPdfWriter, QImage, QPixmap). This slot fires on the worker thread and using
        Qt objects here causes a silent app crash.
        
        All output generation is delegated to _generate_outputs_on_main_thread() via
        the _image_result_ready signal (QueuedConnection), which executes on the main thread.
        """
        image_path = self.image_paths[self.current_idx]
        logger.info("BatchManager: Worker completed for %s — relaying to main thread for output generation.",
                    Path(image_path).name)
        self._image_result_ready.emit(image_path, results)

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
            self._advance_batch()
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

        self._advance_batch()

    def _record_output_failure(self, image_name: str):
        """Appends a FAILED summary record and increments the failure counter."""
        record = self._build_failed_record(image_name, "OUTPUT_GENERATION_FAILED")
        self.summary_records.append(record)
        self.failed_count += 1

    def _advance_batch(self):
        """Advances the batch loop index and schedules the next image or finalizes."""
        duration = time.time() - self.image_start_time
        self.image_runtimes.append(duration)

        self.current_idx += 1
        # Clear the worker reference before scheduling the next iteration
        self.active_worker = None

        if self._is_cancelled:
            self._on_cancel_finished()
        else:
            QTimer.singleShot(0, self.analyze_next_image)

    def _on_image_failed(self, error_msg: str):
        image_path = self.image_paths[self.current_idx]
        image_name = Path(image_path).name
        logger.error("BatchManager: Analysis worker reported failure on %s: %s", image_name, error_msg)
        
        record = self._build_failed_record(image_name, f"FAILED: {error_msg}")
        self.summary_records.append(record)
        
        self.failed_count += 1
        self._advance_batch()

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
                f.write(f"Lumen Version: 0.1.0\n")
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
        self._write_master_csv()
        self._write_run_manifest()
        
        state.is_batch_active = False
        state.batch_progress = 100
        state.batch_status = f"Batch completed. Successful: {self.completed_count}, Failed: {self.failed_count}."
        
        logger.info("BatchManager: Batch completed. Total elapsed: %.2f minutes", (time.time() - self.start_time) / 60.0)
        self.batch_finished.emit(self.completed_count, self.failed_count, self.output_dir)

# Global singleton instance of BatchProcessingManager
batch_manager = BatchProcessingManager()
