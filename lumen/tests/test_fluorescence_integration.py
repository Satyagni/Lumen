import unittest
import os
import sys
import tempfile
import numpy as np
from pathlib import Path

# Add root folder to sys.path
TEST_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = TEST_DIR.parent.parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

# Ensure QApp is initialized
from PySide6.QtWidgets import QApplication
app = QApplication.instance()
if not app:
    app = QApplication([])

from lumen.workflows.state import state
from lumen.processing.image_manager import image_manager
from lumen.pages.analysis_page import AnalysisPage
from lumen.pages.results_page import ResultsPage

class TestFluorescenceIntegration(unittest.TestCase):
    """Verifies Phase 2C integration of fluorescence quantification and exporters into the UI page workflows."""

    def setUp(self):
        # Reset state defaults
        state.workspace_manager.reset_analysis_session()
        state.reset_analysis_session()
        image_manager.clear_cache()
        
        # Setup temporary directories and files
        self.test_dir = tempfile.TemporaryDirectory()
        self.image_path = os.path.join(self.test_dir.name, "multi_channel.tif")
        
        # Save a mock 2-channel 15x15 TIFF
        import tifffile
        mock_img = np.random.randint(0, 255, (2, 15, 15), dtype=np.uint8)
        tifffile.imwrite(self.image_path, mock_img)
        
    def tearDown(self):
        self.test_dir.cleanup()
        image_manager.clear_cache()
        
        # Disconnect all receivers from state signals to prevent libshiboken lifetime errors in tests
        signals = [
            state.image_loaded, state.theme_changed, state.page_changed, state.workflow_selected,
            state.analysis_started, state.analysis_completed, state.manual_mask_saved,
            state.analysis_results_updated, state.sidebar_toggled, state.backend_changed,
            state.backend_preference_changed, state.mask_opacity_changed, state.show_original_changed,
            state.show_overlay_changed, state.quality_mode_changed, state.batch_started,
            state.batch_progress_updated, state.batch_finished, state.batch_cancelled,
            state.segmentation_method_changed, state.segmentation_model_changed, state.dirty_state_changed,
            state.channel_names_changed, state.segmentation_channel_changed,
            state.active_viewer_channel_changed, state.background_correction_changed,
            state.preprocessing_changed
        ]
        for signal in signals:
            try:
                signal.disconnect()
            except RuntimeError:
                pass # Signal had no connections

    def test_fluorescence_workflow_quantification_trigger(self):
        """Verifies quantification triggers on Cellpose completion and updates state for fluorescence workflow."""
        # 1. Load image and set workflow to fluorescence
        success, msg = image_manager.load_image(self.image_path, set_state=True)
        self.assertTrue(success, msg)
        state.current_workflow = "fluorescence"
        
        # Create Page instances
        analysis_page = AnalysisPage()
        results_page = ResultsPage()
        
        # 2. Simulate Cellpose finished callback with a mock mask
        mock_mask = np.zeros((15, 15), dtype=np.int32)
        mock_mask[2:5, 2:5] = 1  # Cell 1 (3x3 area = 9)
        mock_mask[8, 8] = 2      # Cell 2 (1x1 area = 1)
        
        mock_results = {
            "masks": mock_mask,
            "cell_count": 2,
            "mean_cell_area_px": 5.0,
            "median_cell_area_px": 5.0,
            "average_diameter_px": 2.0,
            "cell_density": 0.08,
            "processing_time_s": 0.5
        }
        
        # Call finished slot
        analysis_page._on_analysis_finished(mock_results)
        
        # Verify fluorescence results are populated
        fluor_results = state.fluorescence_results
        self.assertEqual(len(fluor_results), 2)
        
        # Verify sorted cell_id ascending
        self.assertEqual(fluor_results[0]["cell_id"], 1)
        self.assertEqual(fluor_results[1]["cell_id"], 2)
        
        # Verify area is integer
        self.assertEqual(fluor_results[0]["area"], 9)
        self.assertEqual(fluor_results[1]["area"], 1)
        self.assertIsInstance(fluor_results[0]["area"], int)
        
        # Verify summary statistics in state
        summary = state.fluorescence_summary
        self.assertEqual(summary["total_cell_count"], 2)
        self.assertAlmostEqual(summary["average_area"], 5.0)
        self.assertIn("Channel 1_mean_average", summary)
        self.assertIn("Channel 1_median_average", summary)
        
        # 3. Verify ResultsPage displays the metrics correctly
        results_page._sync_state()
        self.assertTrue(results_page.empty_state_card.isHidden())
        self.assertEqual(results_page.metric_cards[0].val_label.text(), "2") # Total cell count
        self.assertEqual(results_page.metric_cards[1].val_label.text(), "5.00") # Avg cell area
        
        # Verify preview table is shown (not hidden) for 2 rows and appropriate columns
        self.assertFalse(results_page.table_container.isHidden())
        self.assertEqual(results_page.table.rowCount(), 2)
        self.assertGreaterEqual(results_page.table.columnCount(), 4) # Cell ID, Area, Perimeter, Channel Mean
        
        # Verify export buttons visibility for fluorescence (using isHidden in headless test environment)
        self.assertTrue(results_page.csv_btn.isHidden())
        self.assertTrue(results_page.pdf_btn.isHidden())
        self.assertTrue(results_page.images_btn.isHidden())
        self.assertFalse(results_page.export_cell_csv_btn.isHidden())
        self.assertFalse(results_page.export_summary_csv_btn.isHidden())
        self.assertTrue(results_page.export_cell_csv_btn.isEnabled())
        self.assertTrue(results_page.export_summary_csv_btn.isEnabled())

    def test_non_fluorescence_workflow_does_not_quantify(self):
        """Verifies cell segmentation workflows remain untouched and do not trigger quantification."""
        success, msg = image_manager.load_image(self.image_path, set_state=True)
        self.assertTrue(success, msg)
        state.current_workflow = "cell_counting" # Non-fluorescence workflow
        
        analysis_page = AnalysisPage()
        results_page = ResultsPage()
        
        # Simulate Cellpose finished
        mock_mask = np.zeros((15, 15), dtype=np.int32)
        mock_mask[2:5, 2:5] = 1
        
        mock_results = {
            "masks": mock_mask,
            "cell_count": 1,
            "mean_cell_area_px": 9.0,
            "median_cell_area_px": 9.0,
            "average_diameter_px": 3.0,
            "cell_density": 0.04,
            "processing_time_s": 0.5
        }
        
        analysis_page._on_analysis_finished(mock_results)
        
        # Verify fluorescence results are EMPTY
        self.assertEqual(state.fluorescence_results, {})
        self.assertEqual(state.fluorescence_summary, {})
        
        # Verify ResultsPage displays original metrics
        results_page._sync_state()
        self.assertTrue(results_page.table_container.isHidden())
        self.assertFalse(results_page.csv_btn.isHidden())
        self.assertFalse(results_page.pdf_btn.isHidden())
        self.assertFalse(results_page.images_btn.isHidden())
        self.assertTrue(results_page.export_cell_csv_btn.isHidden())
        self.assertTrue(results_page.export_summary_csv_btn.isHidden())
        
        # Check standard metrics card
        self.assertEqual(results_page.metric_cards[0].val_label.text(), "1") # Total cells
        self.assertEqual(results_page.metric_cards[1].val_label.text(), "9.00") # Mean cell area
        self.assertEqual(results_page.metric_cards[2].val_label.text(), "9.00") # Median cell area

    def test_fluorescence_revert_discard_state(self):
        """Verifies that discarding/reverting restores the last committed fluorescence results and summary."""
        # 1. Load image and set workflow to fluorescence
        success, msg = image_manager.load_image(self.image_path, set_state=True)
        self.assertTrue(success, msg)
        state.current_workflow = "fluorescence"
        
        analysis_page = AnalysisPage()
        results_page = ResultsPage()
        
        # 2. Simulate Cellpose finished callback with a mock mask
        mock_mask = np.zeros((15, 15), dtype=np.int32)
        mock_mask[2:5, 2:5] = 1  # Cell 1 (3x3 area = 9)
        
        mock_results = {
            "masks": mock_mask,
            "cell_count": 1,
            "mean_cell_area_px": 9.0,
            "median_cell_area_px": 9.0,
            "average_diameter_px": 3.0,
            "cell_density": 0.04,
            "processing_time_s": 0.5
        }
        
        analysis_page._on_analysis_finished(mock_results)
        
        # Verify state is dirty initially after run
        self.assertTrue(state.is_dirty)
        initial_fluor = list(state.fluorescence_results)
        self.assertEqual(len(initial_fluor), 1)
        
        # Revert to last committed state (which was unanalysed/empty)
        state.revert_to_last_committed_state()
        
        # Verify state is clean and fluorescence results/summary are reverted to empty
        self.assertFalse(state.is_dirty)
        self.assertEqual(state.fluorescence_results, {})
        self.assertEqual(state.fluorescence_summary, {})
        
        # Verify ResultsPage displays empty state
        results_page._sync_state()
        self.assertFalse(results_page.empty_state_card.isHidden())
        
        # --- Now test reverting to a previously committed analysed state ---
        # Run analysis again
        analysis_page._on_analysis_finished(mock_results)
        self.assertTrue(state.is_dirty)
        
        # Commit / Save analysis
        save_success = analysis_page.save_analysis()
        self.assertTrue(save_success)
        self.assertFalse(state.is_dirty)
        
        # Modify and run again (simulating unsaved changed masks/results)
        mock_mask_2 = np.zeros((15, 15), dtype=np.int32)
        mock_mask_2[2:4, 2:4] = 1 # Cell 1 (2x2 area = 4)
        mock_mask_2[6:8, 6:8] = 2 # Cell 2 (2x2 area = 4)
        
        mock_results_2 = {
            "masks": mock_mask_2,
            "cell_count": 2,
            "mean_cell_area_px": 4.0,
            "median_cell_area_px": 4.0,
            "average_diameter_px": 2.0,
            "cell_density": 0.08,
            "processing_time_s": 0.5
        }
        
        analysis_page._on_analysis_finished(mock_results_2)
        self.assertTrue(state.is_dirty)
        self.assertEqual(len(state.fluorescence_results), 2)
        self.assertEqual(state.fluorescence_summary["total_cell_count"], 2)
        
        # Reset analysis changes manually via reset_analysis_changes
        reset_success = analysis_page.reset_analysis_changes()
        self.assertTrue(reset_success)
        self.assertFalse(state.is_dirty)
        
        # Verify reverted to the first run (1 cell)
        self.assertEqual(len(state.fluorescence_results), 1)
        self.assertEqual(state.fluorescence_summary["total_cell_count"], 1)
        
        # Run analysis again (so it's dirty)
        analysis_page._on_analysis_finished(mock_results_2)
        self.assertTrue(state.is_dirty)
        
        # Revert via revert_to_last_committed_state
        state.revert_to_last_committed_state()
        self.assertFalse(state.is_dirty)
        
        # Verify reverted to the first run (1 cell)
        self.assertEqual(len(state.fluorescence_results), 1)
        self.assertEqual(state.fluorescence_summary["total_cell_count"], 1)

    def test_fluorescence_grayscale_image_and_out_of_bounds_guard(self):
        """Verifies that running fluorescence analysis on a grayscale/single-channel image succeeds if channel is 0, but fails if channel is out of bounds."""
        from unittest.mock import patch
        
        # 1. Create a grayscale image (2D)
        temp_gray_path = os.path.join(self.test_dir.name, "grayscale.tif")
        import tifffile
        tifffile.imwrite(temp_gray_path, np.zeros((15, 15), dtype=np.uint8))
        
        # 2. Load it and stage
        success, msg = image_manager.load_image(temp_gray_path, set_state=True)
        self.assertTrue(success, msg)
        state.current_workflow = "fluorescence"
        
        analysis_page = AnalysisPage()
        
        # Verify segmentation_channel is 0 and it passes the guard to launch analysis
        state.segmentation_channel = 0
        with patch("lumen.pages.analysis_page.QMessageBox.critical") as mock_critical, \
             patch("lumen.processing.processing_manager.ProcessingManager.run_analysis", return_value=True) as mock_run:
            analysis_page._on_run_analysis_clicked()
            mock_critical.assert_not_called()
            mock_run.assert_called_once()
            
        # Verify if segmentation_channel is out of bounds (e.g. 1), it triggers critical message box and aborts
        state.segmentation_channel = 1
        with patch("lumen.pages.analysis_page.QMessageBox.critical") as mock_critical, \
             patch("lumen.processing.processing_manager.ProcessingManager.run_analysis", return_value=True) as mock_run:
            analysis_page._on_run_analysis_clicked()
            mock_critical.assert_called_once()
            mock_run.assert_not_called()
            self.assertIn("Selected segmentation channel 1 is out of bounds for the image with 1 channels.", mock_critical.call_args[0][2])
