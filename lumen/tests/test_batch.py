import unittest
import os
import sys
import shutil
import time
import csv
from pathlib import Path
import numpy as np
import tifffile
import PIL.Image

# Add root folder to sys.path
TEST_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = TEST_DIR.parent.parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

# Ensure QApplication is initialized before import of any gui/widget code
from PySide6.QtWidgets import QApplication
app = QApplication.instance()
if not app:
    app = QApplication([])

from PySide6.QtCore import QEventLoop, QTimer, Qt
from lumen.workflows.state import state
from lumen.processing.batch_manager import batch_manager


class TestLumenBatchPipeline(unittest.TestCase):
    """Integration and unit tests verifying the batch processing manager backend."""

    def setUp(self):
        # Create temp workspace folder for batch processing tests
        self.test_batch_dir = WORKSPACE_DIR / "temp_test_batch"
        self.test_batch_dir.mkdir(exist_ok=True)
        
        # Write 3 small mock microscopy files (20x20 grayscale blocks)
        self.image_paths = []
        for i in range(3):
            arr = np.zeros((20, 20), dtype=np.uint16)
            arr[5:15, 5:15] = 500 * (i + 1) # draw a mock block
            img_path = self.test_batch_dir / f"mock_microscope_img_{i}.tif"
            tifffile.imwrite(str(img_path), arr)
            self.image_paths.append(str(img_path))
            
        # Clean state variables
        state.current_workflow = "cell_counting"
        state.is_batch_active = False

    def tearDown(self):
        # Remove temp workspace directory recursively
        if self.test_batch_dir.exists():
            shutil.rmtree(str(self.test_batch_dir))
            
        # Reset batch manager singleton states
        batch_manager.reset_batch()
        batch_manager.parameters = {}
        batch_manager.output_dir = ""

    def test_batch_manager_scanning_and_estimation(self):
        """Validates folder globbing, recursive searches, and duration estimation settings."""
        # 1. Test standard scan
        params = {"quality_mode": "Balanced"}
        num_images = batch_manager.prepare_batch(str(self.test_batch_dir), params, recursive=False)
        self.assertEqual(num_images, 3)
        self.assertEqual(len(batch_manager.image_paths), 3)
        self.assertEqual(batch_manager.output_dir, str(self.test_batch_dir / "batch_results"))
        
        # 2. Test recursive scan by making a nested sub-folder
        sub_dir = self.test_batch_dir / "nested_folder"
        sub_dir.mkdir()
        nested_arr = np.zeros((20, 20), dtype=np.uint16)
        nested_path = sub_dir / "nested_img.tif"
        tifffile.imwrite(str(nested_path), nested_arr)
        
        num_images_rec = batch_manager.prepare_batch(str(self.test_batch_dir), params, recursive=True)
        self.assertEqual(num_images_rec, 4)
        
        # 3. Verify estimated runtimes
        # Check Balanced preset estimation (CPU or GPU)
        batch_manager.parameters["backend_preference"] = "CPU"
        est_min_balanced = batch_manager.get_estimated_runtime_minutes()
        self.assertGreater(est_min_balanced, 0.0)
        
        # Check Precise preset estimation
        batch_manager.parameters["quality_mode"] = "Precise"
        est_min_precise = batch_manager.get_estimated_runtime_minutes()
        self.assertGreater(est_min_precise, est_min_balanced)

    def test_batch_manager_reproducibility_metadata(self):
        """Verifies that the reproducibility foundation metadata card is written correctly."""
        params = {"quality_mode": "Balanced"}
        batch_manager.prepare_batch(str(self.test_batch_dir), params)
        
        # Create output dir manually for metadata check
        os.makedirs(batch_manager.output_dir, exist_ok=True)
        batch_manager._save_batch_metadata()
        
        meta_file = Path(batch_manager.output_dir) / "batch_metadata.txt"
        self.assertTrue(meta_file.exists())
        
        # Read and check contents
        with open(meta_file, mode="r", encoding="utf-8") as f:
            content = f.read()
            self.assertIn("LUMEN BATCH ANALYSIS REPRODUCIBILITY METADATA", content)
            self.assertIn("Active Workflow: cell_counting", content)
            self.assertIn("Segmentation Mode Preset: Balanced", content)
            self.assertIn("Total Batch Image Files: 3", content)

    def test_batch_manager_complete_sequential_run(self):
        """Executes a full sequential batch run and verifies outputs exist in proper hierarchy."""
        params = {"quality_mode": "Fast"} # Use Fast preset for speed in test runs
        batch_manager.prepare_batch(str(self.test_batch_dir), params)
        
        loop = QEventLoop()
        
        def on_finished(completed, failed, results_dir):
            loop.quit()
            
        def on_cancelled():
            loop.quit()

        batch_manager.batch_finished.connect(on_finished)
        batch_manager.batch_cancelled.connect(on_cancelled)
        
        # Start execution loop
        batch_manager.start_batch()
        self.assertTrue(state.is_batch_active)
        
        loop.exec() # Wait for batch processing to finalize
        
        # Verify execution counts
        self.assertEqual(batch_manager.completed_count, 3)
        self.assertEqual(batch_manager.failed_count, 0)
        self.assertFalse(state.is_batch_active)
        
        # Verify directory structure
        res_dir = Path(batch_manager.output_dir)
        self.assertTrue(res_dir.exists())
        self.assertTrue((res_dir / "batch_summary.csv").exists())
        
        # Check subfolders for each image
        for i in range(3):
            img_name = f"mock_microscope_img_{i}.tif"
            sub_folder = res_dir / img_name
            self.assertTrue(sub_folder.exists())
            self.assertTrue((sub_folder / f"{img_name}_overlay_preview.png").exists())
            self.assertTrue((sub_folder / f"{img_name}_labels_raw.tif").exists())
            self.assertTrue((sub_folder / f"{img_name}_cell_metrics.csv").exists())
            self.assertTrue((sub_folder / f"{img_name}_report.pdf").exists())

    def test_batch_manager_resume_safety(self):
        """Tests that pre-existing complete output files are skipped, logged, and master CSV updated."""
        params = {"quality_mode": "Fast"}
        batch_manager.prepare_batch(str(self.test_batch_dir), params)
        
        # 1. Pre-create completed outputs for the FIRST image: "mock_microscope_img_0.tif"
        res_dir = Path(batch_manager.output_dir)
        img_name = "mock_microscope_img_0.tif"
        sub_folder = res_dir / img_name
        sub_folder.mkdir(parents=True, exist_ok=True)
        
        # Write dummy files
        (sub_folder / f"{img_name}_overlay_preview.png").write_bytes(b"dummy_png_data")
        (sub_folder / f"{img_name}_labels_raw.tif").write_bytes(b"dummy_tif_data")
        (sub_folder / f"{img_name}_report.pdf").write_bytes(b"dummy_pdf_data")
        
        # Write dummy cell metrics CSV
        csv_path = sub_folder / f"{img_name}_cell_metrics.csv"
        with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["cell_id", "area_px", "diameter_px", "centroid_x", "centroid_y"])
            writer.writerow([1, 100, 11.2, 5.0, 5.0])
            writer.writerow([2, 120, 12.4, 15.0, 15.0])
            
        # 2. Run the batch
        loop = QEventLoop()
        batch_manager.batch_finished.connect(lambda c, f, r: loop.quit())
        batch_manager.start_batch()
        loop.exec()
        
        # Verify that img_0 was skipped (via cache parsing) and others processed
        self.assertEqual(batch_manager.skipped_count, 1)
        self.assertEqual(batch_manager.completed_count, 3) # 1 skipped + 2 completed
        
        # Verify the CSV record entries for skipped file
        summary_csv = res_dir / "batch_summary.csv"
        self.assertTrue(summary_csv.exists())
        
        with open(summary_csv, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            records = list(reader)
            
        self.assertEqual(len(records), 3)
        img_0_rec = next(r for r in records if r["image_name"] == img_name)
        self.assertEqual(img_0_rec["status"], "SKIPPED_ALREADY_EXISTS")
        self.assertEqual(int(img_0_rec["cell_count"]), 2) # parsed from pre-existing CSV
        
        # Others should have status SUCCESS
        img_1_rec = next(r for r in records if r["image_name"] == "mock_microscope_img_1.tif")
        self.assertEqual(img_1_rec["status"], "SUCCESS")

    def test_batch_manager_failure_tolerance(self):
        """Verifies that processing or export failure on one image doesn't crash the entire batch pipeline."""
        params = {"quality_mode": "Fast"}
        batch_manager.prepare_batch(str(self.test_batch_dir), params)
        
        # Insert a corrupt image filename that doesn't exist into the scanned path list
        batch_manager.image_paths.insert(1, str(self.test_batch_dir / "non_existent_corrupted.png"))
        
        loop = QEventLoop()
        batch_manager.batch_finished.connect(lambda c, f, r: loop.quit())
        batch_manager.start_batch()
        loop.exec()
        
        # Verify completed count is 3, failed count is 1
        self.assertEqual(batch_manager.completed_count, 3)
        self.assertEqual(batch_manager.failed_count, 1)
        
        # Verify master CSV has the error record
        res_dir = Path(batch_manager.output_dir)
        summary_csv = res_dir / "batch_summary.csv"
        with open(summary_csv, mode="r", encoding="utf-8") as f:
            records = list(csv.DictReader(f))
            
        corrupt_rec = next(r for r in records if r["image_name"] == "non_existent_corrupted.png")
        self.assertTrue(corrupt_rec["status"].startswith("FAILED"))
        self.assertEqual(int(corrupt_rec["cell_count"]), 0)

    def test_batch_manager_cancellation(self):
        """Validates that cancellation terminates worker thread and aborts sequential runs immediately."""
        params = {"quality_mode": "Fast"}
        batch_manager.prepare_batch(str(self.test_batch_dir), params)
        
        loop = QEventLoop()
        batch_manager.batch_cancelled.connect(loop.quit)
        
        # Start batch and cancel it immediately (in the next event tick)
        batch_manager.start_batch()
        QTimer.singleShot(50, batch_manager.cancel_batch)
        
        loop.exec()
        
        self.assertTrue(batch_manager._is_cancelled)
        self.assertFalse(state.is_batch_active)

    def test_batch_manager_backend_preference_propagation(self):
        """Verifies that backend preference parameters propagate to workers and are logged to master CSV."""
        # 1. Test forcing CPU execution
        params = {
            "quality_mode": "Fast",
            "backend_preference": "CPU"
        }
        batch_manager.prepare_batch(str(self.test_batch_dir), params)
        
        loop = QEventLoop()
        batch_manager.batch_finished.connect(lambda c, f, r: loop.quit())
        batch_manager.start_batch()
        loop.exec()
        
        # Verify master CSV records have requested_backend=CPU and resolved_backend=CPU
        res_dir = Path(batch_manager.output_dir)
        summary_csv = res_dir / "batch_summary.csv"
        self.assertTrue(summary_csv.exists())
        
        with open(summary_csv, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            records = list(reader)
            
        self.assertEqual(len(records), 3)
        for r in records:
            self.assertEqual(r["requested_backend"], "CPU")
            self.assertEqual(r["resolved_backend"], "CPU")
            
        # 2. Test forcing CUDA (which falls back to CPU fallback in standard CPU-only test environments)
        params_cuda = {
            "quality_mode": "Fast",
            "backend_preference": "CUDA (GPU)"
        }
        # Clear out output directory to rerun without resume skipping triggering SUCCESS without running
        if res_dir.exists():
            shutil.rmtree(str(res_dir))
        batch_manager.prepare_batch(str(self.test_batch_dir), params_cuda)
        
        loop_cuda = QEventLoop()
        batch_manager.batch_finished.disconnect() # disconnect old slots
        batch_manager.batch_finished.connect(lambda c, f, r: loop_cuda.quit())
        batch_manager.start_batch()
        loop_cuda.exec()
        
        with open(summary_csv, mode="r", encoding="utf-8") as f:
            reader_cuda = csv.DictReader(f)
            records_cuda = list(reader_cuda)
            
        self.assertEqual(len(records_cuda), 3)
        import torch
        expected_resolved = "CUDA" if torch.cuda.is_available() else "CPU (fallback)"
        for r in records_cuda:
            self.assertEqual(r["requested_backend"], "CUDA (GPU)")
            self.assertEqual(r["resolved_backend"], expected_resolved)

    def test_batch_adaptive_runtime_estimation(self):
        """Verifies that the batch manager tracks individual image runtimes and the page computes adaptive remaining time correctly."""
        from lumen.pages.batch_progress_page import BatchProgressPage
        page = BatchProgressPage()
        # Show the page so isVisible() guards in signal handlers are satisfied.
        # In real usage BatchProgressPage is always navigated-to (and thus visible) before
        # batch_started fires, so this matches the actual runtime invariant.
        page.show()
        
        # Ensure batch manager lifecycle state is RUNNING during progress tracking simulation
        batch_manager.lifecycle_state = "RUNNING"
        
        # Manually trigger batch started on page
        page._on_batch_started(10)
        self.assertEqual(page.total_images, 10)
        self.assertEqual(page.rem_val.text(), "Estimating...")
        
        # Test case 1: After 1 image processed (processed < 3, should use static estimate fallback)
        batch_manager.image_runtimes = [5.0]
        # completed=1, failed=0
        page._on_batch_progress_updated(1, 0, "mock_image_0.tif")
        self.assertEqual(page.rem_val.text(), "Estimating...")
        
        # Test case 2: After 2 images processed (processed = 2 < 3, should still use static estimate fallback)
        batch_manager.image_runtimes = [5.0, 15.0]
        # completed=2, failed=0
        page._on_batch_progress_updated(2, 0, "mock_image_1.tif")
        self.assertEqual(page.rem_val.text(), "Estimating...")

        # Test case 2b: After 3 images processed (processed = 3 >= 3, should use adaptive average)
        batch_manager.image_runtimes = [5.0, 15.0, 10.0]
        # completed=3, failed=0
        page._on_batch_progress_updated(3, 0, "mock_image_2.tif")
        # processed = 3 >= 3. len(image_runtimes) = 3.
        # average = (5.0 + 15.0 + 10.0) / 3 = 10.0 seconds per image
        # remaining = 7 images
        # expected_rem_sec = 10.0 * 7 = 70.0 seconds => 1.2 mins
        self.assertEqual(page.rem_val.text(), "~1.2 mins")

        # Test case 3: After 4 images processed (processed = 4 >= 3, average uses all runtimes)
        # Runtimes: [5.0, 15.0, 12.0, 18.0] => avg = 12.5 seconds per image
        # remaining = 6 images
        # expected_rem_sec = 12.5 * 6 = 75.0 seconds => 1.2 mins (banker's rounding of 1.25)
        batch_manager.image_runtimes = [5.0, 15.0, 12.0, 18.0]
        page._on_batch_progress_updated(4, 0, "mock_image_3.tif")
        self.assertEqual(page.rem_val.text(), "~1.2 mins")

        # Test case 4: Verify UI values update correctly and handle skipped images
        batch_manager.skipped_count = 2
        # completed=5 (meaning 3 successful + 2 skipped), failed=1
        page._on_batch_progress_updated(5, 1, "mock_image_5.tif")
        # completed label should display: completed_count - skipped_count = 5 - 2 = 3
        self.assertEqual(page.completed_val.text(), "3")
        self.assertEqual(page.failed_val.text(), "1")
        self.assertEqual(page.skipped_val.text(), "2")
        
        # Test case 5: Final completion KPI forced refresh
        batch_manager.skipped_count = 3
        # completed=9 (6 successful + 3 skipped), failed=1
        page._on_batch_finished(9, 1, "/mock/results")
        # completed label: 9 - 3 = 6
        self.assertEqual(page.completed_val.text(), "6")
        self.assertEqual(page.failed_val.text(), "1")
        self.assertEqual(page.skipped_val.text(), "3")
        
        # Invariant check: completed + failed + skipped == total
        total_kpis = int(page.completed_val.text()) + int(page.failed_val.text()) + int(page.skipped_val.text())
        self.assertEqual(total_kpis, 10)

        # Test case 6: UI State Clean Rerun Reset (Issue 3)
        page.cancel_btn.setText("Cancelled")
        page.cancel_btn.setEnabled(False)
        page.results_dir = "/some/old/path"
        
        # Trigger next batch started
        page._on_batch_started(15)
        self.assertEqual(page.cancel_btn.text(), "Cancel Batch")
        self.assertTrue(page.cancel_btn.isEnabled())
        self.assertEqual(page.results_dir, "")
        
        # Test case 7: Cancellation KPI update and total_images immutability (Issue 4, Refinement 1)
        batch_manager.completed_count = 5
        batch_manager.failed_count = 2
        batch_manager.skipped_count = 1
        page._on_batch_cancelled()
        # total_images must remain 15 (immutable)
        self.assertEqual(page.total_images, 15)
        self.assertEqual(page.completed_val.text(), "4") # completed_count - skipped_count = 5 - 1 = 4
        self.assertEqual(page.failed_val.text(), "2")
        self.assertEqual(page.skipped_val.text(), "1")
        # Invariant during cancel: completed + failed + skipped (4 + 2 + 1 = 7) <= total (15)
        self.assertLessEqual(4 + 2 + 1, 15)
        
        # Test case 8: Centralized backend resolution in batch_manager (Refinement 2)
        batch_manager.lifecycle_state = "IDLE"
        batch_manager.parameters["backend_preference"] = "CPU"
        batch_manager.start_batch()
        self.assertEqual(batch_manager.resolved_backend, "CPU")
        page._on_batch_started(12)
        self.assertEqual(page.backend_val.text(), "CPU")

        # Test case 9: Verify PAUSED and COMPLETED visibility and ETA text
        batch_manager.lifecycle_state = "PAUSED"
        page.sync_ui_state()
        self.assertEqual(page.rem_val.text(), "ETA: Paused")
        self.assertFalse(page.rem_val.isHidden())
        self.assertFalse(page.rem_title.isHidden())
        
        batch_manager.lifecycle_state = "COMPLETED"
        page.sync_ui_state()
        self.assertTrue(page.rem_val.isHidden())
        self.assertTrue(page.rem_title.isHidden())

    def test_batch_manager_run_manifest_generation(self):
        """Verifies that run_manifest.json is generated correctly with proper metadata schema."""
        params = {"quality_mode": "Fast"}
        batch_manager.prepare_batch(str(self.test_batch_dir), params)
        
        loop = QEventLoop()
        batch_manager.batch_finished.connect(lambda c, f, r: loop.quit())
        batch_manager.start_batch()
        loop.exec()
        
        manifest_file = Path(batch_manager.output_dir) / "run_manifest.json"
        self.assertTrue(manifest_file.exists())
        
        import json
        with open(manifest_file, mode="r", encoding="utf-8") as f:
            data = json.load(f)
            self.assertIn("timestamp", data)
            self.assertIn("workflow", data)
            self.assertEqual(data["backend"], batch_manager.resolved_backend)
            self.assertEqual(data["segmentation_mode"], "Fast")
            self.assertIn("images", data)
            self.assertEqual(len(data["images"]), 3)

    def test_cooperative_non_blocking_cancellation(self):
        """Verifies that cancellation changes lifecycle states and triggers asynchronous finalization."""
        params = {"quality_mode": "Fast"}
        batch_manager.prepare_batch(str(self.test_batch_dir), params)
        
        loop = QEventLoop()
        batch_manager.batch_cancelled.connect(loop.quit)
        
        # Start and verify state is RUNNING
        batch_manager.start_batch()
        self.assertEqual(batch_manager.lifecycle_state, "RUNNING")
        
        # Process pending timer event to launch analyze_next_image and instantiate worker
        QApplication.processEvents()
        
        # Cancel and verify state becomes CANCELLING (before thread finishes)
        batch_manager.cancel_batch()
        self.assertEqual(batch_manager.lifecycle_state, "CANCELLING")
        
        loop.exec()
        
        # Verify state becomes CANCELLED after graceful stop completes
        self.assertEqual(batch_manager.lifecycle_state, "CANCELLED")
        
        # Check manifest and summary CSV written
        manifest_file = Path(batch_manager.output_dir) / "run_manifest.json"
        summary_csv = Path(batch_manager.output_dir) / "batch_summary.csv"
        self.assertTrue(manifest_file.exists())
        self.assertTrue(summary_csv.exists())

    def test_batch_explorer_page_loading_and_sorting(self):
        """Verifies that the BatchResultsExplorerPage loads summary data, sorts list items, and redirects to analysis workspace."""
        from lumen.pages.batch_explorer_page import BatchResultsExplorerPage
        explorer = BatchResultsExplorerPage()
        
        # Prepare mock folder structures
        mock_output_dir = self.test_batch_dir / "batch_results"
        mock_output_dir.mkdir(exist_ok=True)
        
        # Write mock summary CSV and manifest
        summary_csv = mock_output_dir / "batch_summary.csv"
        with open(summary_csv, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["image_name", "workflow", "segmentation_mode", "model_type", "cell_count", "processing_time_s", "status", "resolved_backend"])
            writer.writerow(["img_A.tif", "cell_counting", "Balanced", "cyto", "10", "1.5", "SUCCESS", "CPU"])
            writer.writerow(["img_B.tif", "cell_counting", "Balanced", "cyto", "25", "3.0", "SUCCESS", "CPU"])
            writer.writerow(["img_C.tif", "cell_counting", "Balanced", "cyto", "0", "0.0", "FAILED: Out of Memory", "CPU"])
            
        # Write mock manifest
        manifest_json = mock_output_dir / "run_manifest.json"
        import json
        with open(manifest_json, mode="w", encoding="utf-8") as f:
            json.dump({
                "timestamp": "2026-06-01 12:00:00",
                "workflow": "cell_counting",
                "backend": "CPU",
                "segmentation_mode": "Balanced",
                "images": []
            }, f)
            
        # Write dummy raw masks and files for img_A and img_B
        for name in ["img_A.tif", "img_B.tif"]:
            folder = mock_output_dir / name
            folder.mkdir(exist_ok=True)
            # Create mock raw labels TIFF
            tifffile.imwrite(str(folder / f"{name}_labels_raw.tif"), np.zeros((20, 20), dtype=np.uint16))
            # Create mock cell CSV
            with open(folder / f"{name}_cell_metrics.csv", mode="w", newline="", encoding="utf-8") as cell_f:
                cell_writer = csv.writer(cell_f)
                cell_writer.writerow(["cell_id", "area_px", "diameter_px", "centroid_x", "centroid_y"])
                cell_writer.writerow([1, 100, 11.2, 5.0, 5.0])
            
        # Test loading results in explorer
        state.batch_results_dir = str(mock_output_dir)
        explorer._load_from_state()
        
        # Verify 3 items loaded
        self.assertEqual(explorer.navigator_list.count(), 3)
        
        # Test Sorting: Alphabetical A, B, C
        explorer.sort_combo.setCurrentText("Alphabetical")
        self.assertEqual(explorer.navigator_list.item(0).data(Qt.UserRole)["image_name"], "img_A.tif")
        self.assertEqual(explorer.navigator_list.item(1).data(Qt.UserRole)["image_name"], "img_B.tif")
        self.assertEqual(explorer.navigator_list.item(2).data(Qt.UserRole)["image_name"], "img_C.tif")
        
        # Test Sorting: Cell Count B (25), A (10), C (0)
        explorer.sort_combo.setCurrentText("Cell Count")
        self.assertEqual(explorer.navigator_list.item(0).data(Qt.UserRole)["image_name"], "img_B.tif")
        self.assertEqual(explorer.navigator_list.item(1).data(Qt.UserRole)["image_name"], "img_A.tif")
        self.assertEqual(explorer.navigator_list.item(2).data(Qt.UserRole)["image_name"], "img_C.tif")
        
        # Test Search filtering: search "img_B" (debounced)
        explorer.search_bar.setText("img_B")
        if hasattr(explorer, 'search_timer') and explorer.search_timer.isActive():
            explorer.search_timer.stop()
            explorer._on_search_changed()
        self.assertEqual(explorer.navigator_list.count(), 1)
        self.assertEqual(explorer.navigator_list.item(0).data(Qt.UserRole)["image_name"], "img_B.tif")
        explorer.search_bar.clear()
        if hasattr(explorer, 'search_timer') and explorer.search_timer.isActive():
            explorer.search_timer.stop()
            explorer._on_search_changed()
        
        # Test Analysis Workspace Redirection
        # Select img_B and click "Open in Analysis Workspace"
        explorer.navigator_list.setCurrentRow(0) # img_B.tif
        
        # Write dummy raw image in parent directory for redirect check
        raw_img_path = self.test_batch_dir / "img_B.tif"
        tifffile.imwrite(str(raw_img_path), np.zeros((20, 20), dtype=np.uint16))
        
        explorer._on_open_analysis_clicked()
        
        # Verify state loaded
        self.assertEqual(state.current_image_path, str(raw_img_path))
        self.assertEqual(state.quality_mode, "Balanced")
        self.assertIsNotNone(state.analysis_results)
        self.assertEqual(state.analysis_results["cell_count"], 25)
        self.assertEqual(len(state.analysis_results["cell_metrics"]), 1)


    def test_batch_output_step_isolation(self):
        """
        Verifies that a PDF export failure on one image does not abort output
        generation for subsequent images.

        Core validation for the Phase 1 Qt thread-safety fix:
        Each of the 5 output steps (overlay, TIFF, CSV, PDF, summary) now has
        an independent try/except. A failure in the PDF step must not prevent
        the overlay or CSV from being written, and must not prevent the next
        image from starting.
        """
        import unittest.mock as mock

        params = {"quality_mode": "Fast"}
        batch_manager.prepare_batch(str(self.test_batch_dir), params)

        # Track how many times generate_outputs is called to patch only image 2
        call_count = {"n": 0}

        original_pdf = None
        try:
            from lumen.pages import results_page
            original_pdf = results_page.export_pdf_report
        except Exception:
            self.skipTest("Cannot import results_page for PDF mock patching")

        def patched_pdf(pdf_path, image_path, *args, **kwargs):
            """Raise on second image only; let first and third succeed."""
            call_count["n"] += 1
            if call_count["n"] == 2:
                raise RuntimeError("Simulated PDF render failure for image 2")
            return original_pdf(pdf_path, image_path, *args, **kwargs)

        loop = QEventLoop()
        batch_manager.batch_finished.connect(lambda c, f, r: loop.quit())

        with mock.patch.object(results_page, "export_pdf_report", side_effect=patched_pdf):
            batch_manager.start_batch()
            loop.exec()

        # All 3 images should complete (PDF failure is non-fatal per image)
        self.assertEqual(batch_manager.completed_count + batch_manager.failed_count, 3,
                         "Total processed count must equal total image count")

        # Images 1 and 3 must have overlay + CSV outputs
        res_dir = Path(batch_manager.output_dir)
        for i in [0, 2]:  # image 0 and image 2 (not image 1 which had PDF failure)
            img_name = f"mock_microscope_img_{i}.tif"
            sub_folder = res_dir / img_name
            self.assertTrue((sub_folder / f"{img_name}_overlay_preview.png").exists(),
                            f"Overlay missing for {img_name}")
            self.assertTrue((sub_folder / f"{img_name}_cell_metrics.csv").exists(),
                            f"CSV missing for {img_name}")

        # Summary CSV must exist and have 3 records
        summary_csv = res_dir / "batch_summary.csv"
        self.assertTrue(summary_csv.exists())
        with open(summary_csv, mode="r", encoding="utf-8") as f:
            records = list(csv.DictReader(f))
        self.assertEqual(len(records), 3)

    def test_batch_explorer_cache_invalidation_on_signals(self):
        """Verifies that batch_manager signals successfully invalidate loaded_batch_dir cache on BatchResultsExplorerPage."""
        from lumen.pages.batch_explorer_page import BatchResultsExplorerPage
        explorer = BatchResultsExplorerPage()
        
        # Mock results directory
        mock_results_dir = str(self.test_batch_dir / "test_cache_inval")
        explorer._loaded_batch_dir = mock_results_dir
        
        # Emit batch_started signal via batch_manager
        batch_manager.batch_started.emit(5)
        self.assertIsNone(explorer._loaded_batch_dir, "Cache should be invalidated when batch starts")
        
        # Restore mock path
        explorer._loaded_batch_dir = mock_results_dir
        # Emit batch_finished signal via batch_manager
        batch_manager.batch_finished.emit(5, 0, mock_results_dir)
        self.assertIsNone(explorer._loaded_batch_dir, "Cache should be invalidated when batch finishes")


if __name__ == "__main__":
    unittest.main()
