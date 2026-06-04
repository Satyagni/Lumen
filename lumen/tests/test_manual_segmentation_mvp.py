import unittest
import numpy as np
from pathlib import Path
import sys

# Add root folder to sys.path
TEST_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = TEST_DIR.parent.parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

from PySide6.QtWidgets import QApplication
app = QApplication.instance()
if not app:
    app = QApplication([])

from lumen.ui.mask_editor_dialog import MaskEditorCanvas, MaskEditorDialog, update_results_mask

class TestManualSegmentationMVP(unittest.TestCase):
    """Verifies manual segmentation MVP logic: caching, undo/redo limits, and state updates."""

    def test_update_results_mask(self):
        """Checks update_results_mask recalculates basic statistics correctly."""
        original_results = {
            "masks": np.zeros((10, 10), dtype=np.uint16),
            "cell_count": 0,
            "average_diameter_px": 5.0,
            "mean_cell_area_px": 12.0
        }
        
        # Create a mock edited mask with 3 unique cells (labels 1, 2, 3), each 4 pixels (2x2)
        edited_mask = np.zeros((10, 10), dtype=np.uint16)
        edited_mask[0:2, 0:2] = 1
        edited_mask[3:5, 3:5] = 2
        edited_mask[6:8, 6:8] = 3
        
        updated = update_results_mask(original_results, edited_mask)
        
        # Verify masks are updated
        np.testing.assert_array_equal(updated["masks"], edited_mask)
        # Verify cell count is recalculated
        self.assertEqual(updated["cell_count"], 3)
        # Verify areas are recalculated to 4.0
        self.assertEqual(updated["mean_cell_area_px"], 4.0)
        self.assertEqual(updated["median_cell_area_px"], 4.0)
        # Verify density is cell_count / total_pixels
        self.assertEqual(updated["cell_density"], 3 / 100.0)
        # Verify cell metrics dictionary keys
        self.assertIn("cell_metrics", updated)
        self.assertEqual(len(updated["cell_metrics"]), 3)
        self.assertEqual(updated["cell_metrics"][1]["area_px"], 4)

    def test_undo_redo_stack_limits(self):
        """Verifies undo/redo stack limits are capped at 10 items to prevent memory growth."""
        canvas = MaskEditorCanvas()
        canvas.working_mask = np.zeros((10, 10), dtype=np.uint16)
        
        # Push 15 undo frames
        for i in range(1, 16):
            canvas.push_undo()
            # Mutate to distinguish snapshots
            canvas.working_mask[0, 0] = i
            
        # Verify undo stack is capped at 10
        self.assertEqual(len(canvas.undo_stack), 10)
        # Verify oldest frames (1-5) were discarded, and frames 5-14 remain in stack (meaning index 0 contains 5)
        self.assertEqual(canvas.undo_stack[0][0, 0], 5)
        self.assertEqual(canvas.undo_stack[-1][0, 0], 14)
        
        # Perform 5 undos
        for _ in range(5):
            canvas.undo()
            
        # Verify redo stack is populated and capped properly
        self.assertEqual(len(canvas.redo_stack), 5)
        # Verify current working mask matches frame 10 (which is index 4 in undo stack from start, i.e., 15 - 5 = 10)
        self.assertEqual(canvas.working_mask[0, 0], 10)
        
        # Push a new stroke (clears redo stack)
        canvas.push_undo()
        self.assertEqual(len(canvas.redo_stack), 0)

    def test_canvas_object_aware_selection(self):
        """Verifies object-aware selection, painting, and cell deletions."""
        canvas = MaskEditorCanvas()
        canvas.working_mask = np.zeros((10, 10), dtype=np.uint16)
        canvas.working_mask[0:2, 0:2] = 5 # cell 5
        canvas.working_mask[3:5, 3:5] = 8 # cell 8
        canvas.color_lut = np.zeros((100, 4), dtype=np.uint8)
        
        # Initially no selection
        self.assertIsNone(canvas.selected_label_id)
        
        # Select cell 5
        canvas.selected_label_id = 5
        canvas.update_selection_highlight()
        self.assertEqual(canvas.selected_label_id, 5)
        
        # Test add_new_cell
        canvas.add_new_cell()
        # new_id should be 9 (max is 8 + 1)
        self.assertEqual(canvas.selected_label_id, 9)
        
        # Test delete_selected_cells
        canvas.selected_label_id = 8
        canvas.delete_selected_cells()
        self.assertIsNone(canvas.selected_label_id)
        self.assertFalse(np.any(canvas.working_mask == 8))

    def test_path_normalization(self):
        """Verifies that WorkspaceManager normalizes backslashes to forward slashes."""
        from lumen.workflows.state import state
        # Reset batch sessions
        state.workspace_manager.reset_batch_session()
        
        session = state.workspace_manager.start_batch_session("C:\\some\\nested\\path")
        self.assertEqual(session.batch_results_dir, "C:\\some\\nested\\path")
        
        retrieved = state.workspace_manager.get_batch_session("C:/some/nested/path")
        self.assertEqual(retrieved, session)
        
        retrieved2 = state.workspace_manager.get_batch_session("C:\\some\\nested\\path")
        self.assertEqual(retrieved2, session)

    def test_canvas_brush_drawing_auto_new_cell(self):
        """Verifies that clicking empty space in Add brush mode when no cell is selected auto-creates a cell."""
        from PySide6.QtCore import Qt, QEvent, QPointF
        from PySide6.QtGui import QMouseEvent
        
        canvas = MaskEditorCanvas()
        canvas.working_mask = np.zeros((10, 10), dtype=np.uint16)
        canvas.color_lut = np.zeros((100, 4), dtype=np.uint8)
        canvas.mapToScene = lambda p: QPointF(p.x(), p.y())
        
        canvas.brush_mode = "add"
        canvas.selected_label_id = None
        
        # Simulate left-click on coordinates (2, 2) which is empty space (label 0)
        press_event = QMouseEvent(
            QEvent.MouseButtonPress,
            QPointF(2.0, 2.0),
            Qt.LeftButton,
            Qt.LeftButton,
            Qt.NoModifier
        )
        canvas.mousePressEvent(press_event)
        
        # Verify that a new cell was generated and selected
        self.assertEqual(canvas.selected_label_id, 1)
        self.assertTrue(canvas.drawing)
        self.assertEqual(canvas.working_mask[2, 2], 1)

    def test_dynamic_commit_button_text_and_origin_immutability(self):
        """Verifies dynamic button text and that session origin remains strictly immutable."""
        from lumen.workflows.state import state
        from lumen.pages.analysis_page import AnalysisPage
        
        state.workspace_manager.reset_analysis_session()
        state.reset_analysis_session()
        
        # Scenario 1: Open independently (single origin)
        img_path = "C:/test_images/study_A.tif"
        state.current_image_path = img_path
        
        page = AnalysisPage()
        page._sync_state()
        
        session = state.workspace_manager.get_analysis_session(img_path)
        self.assertIsNotNone(session)
        self.assertEqual(session.origin_type, "single")
        self.assertEqual(page.save_analysis_btn.text(), "💾 Save Analysis")
        
        # Verify single and batch sessions for the same image path can coexist safely
        sess_single = state.workspace_manager.get_analysis_session(img_path, "single")
        self.assertEqual(sess_single.origin_type, "single")
        self.assertIsNone(sess_single.batch_origin_context)

        sess_batch = state.workspace_manager.start_analysis_session(img_path, origin_type="batch", batch_origin_context="C:/some/batch")
        self.assertEqual(sess_batch.origin_type, "batch")
        self.assertEqual(sess_batch.batch_origin_context, "C:/some/batch")

        # Verify original single session is untouched
        sess_single_after = state.workspace_manager.get_analysis_session(img_path, "single")
        self.assertEqual(sess_single_after.origin_type, "single")
        self.assertIsNone(sess_single_after.batch_origin_context)
        
        # Scenario 2: Start new session with batch origin
        img_path_b = "C:/test_images/study_B.tif"
        # Set path first
        state.current_image_path = img_path_b
        # Set origin
        session_b = state.workspace_manager.start_analysis_session(img_path_b, origin_type="batch", batch_origin_context="C:/some/batch")
        self.assertEqual(session_b.origin_type, "batch")
        self.assertEqual(session_b.batch_origin_context, "C:/some/batch")
        
        page._sync_state()
        self.assertEqual(page.save_analysis_btn.text(), "💾 Save to Batch")

    def test_conservative_dirty_state(self):
        """Verifies dirty state is only set to True if actual changes occurred in editor."""
        from lumen.workflows.state import state
        from lumen.pages.analysis_page import AnalysisPage
        
        state.workspace_manager.reset_analysis_session()
        state.reset_analysis_session()
        state.is_dirty = False
        
        img_path = "C:/test_images/study_C.tif"
        state.current_image_path = img_path
        state.analysis_results = {"masks": np.zeros((10, 10), dtype=np.uint16)}
        
        page = AnalysisPage()
        
        # Simulate opening mask editor and applying WITHOUT changes
        from unittest.mock import MagicMock
        from PySide6.QtWidgets import QDialog
        
        # Mock MaskEditorDialog in mask_editor_dialog module since it is imported locally
        import lumen.ui.mask_editor_dialog as med_mod
        original_dialog_class = med_mod.MaskEditorDialog
        
        mock_editor = MagicMock()
        mock_editor.exec.return_value = QDialog.Accepted
        mock_editor.canvas.working_mask = np.zeros((10, 10), dtype=np.uint16)
        mock_editor.canvas.selected_label_id = None
        mock_editor.has_unsaved_changes.return_value = False # No actual changes!
        
        med_mod.MaskEditorDialog = MagicMock(return_value=mock_editor)
        
        try:
            page._on_edit_masks_clicked()
            # Verify dirty state is STILL False
            self.assertFalse(state.is_dirty)
            
            # Now simulate applying WITH changes
            mock_editor.has_unsaved_changes.return_value = True # Edits made!
            page._on_edit_masks_clicked()
            # Verify dirty state is now True
            self.assertTrue(state.is_dirty)
        finally:
            med_mod.MaskEditorDialog = original_dialog_class

    def test_comparative_study_protection(self):
        """Verifies that editing a single-origin copy does not mutate batch directories/records,

        while editing a batch-origin copy correctly updates batch outputs.
        """
        import os
        import tempfile
        import csv
        import json
        import shutil
        from pathlib import Path
        from lumen.workflows.state import state
        from lumen.pages.analysis_page import AnalysisPage
        
        state.workspace_manager.reset_analysis_session()
        state.workspace_manager.reset_batch_session()
        state.reset_analysis_session()
        state.is_dirty = False
        
        temp_dir = Path(tempfile.mkdtemp())
        try:
            # Setup dummy batch directory
            batch_dir = temp_dir / "batch_results"
            os.makedirs(batch_dir, exist_ok=True)
            
            summary_csv = batch_dir / "batch_summary.csv"
            with open(summary_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["image_name", "status", "cell_count", "mean_area_px"])
                writer.writerow(["image_A.tif", "SUCCESS", "10", "15.0"])
                
            manifest_json = batch_dir / "run_manifest.json"
            with open(manifest_json, "w", encoding="utf-8") as f:
                json.dump({"images": [{"image_name": "image_A.tif", "status": "SUCCESS", "cell_count": 10}]}, f)
                
            raw_img_path = temp_dir / "image_A.tif"
            # Create a mock 10x10 raw tif image
            import tifffile
            tifffile.imwrite(str(raw_img_path), np.zeros((10, 10), dtype=np.uint16))
            
            # Scenario A: User opens the image independently as "single" origin
            state.current_image_path = str(raw_img_path)
            
            page = AnalysisPage()
            page._sync_state()
            
            session_s = state.workspace_manager.get_analysis_session(str(raw_img_path))
            self.assertIsNotNone(session_s)
            self.assertEqual(session_s.origin_type, "single")
            
            state.analysis_results = {
                "masks": np.ones((10, 10), dtype=np.uint16),
                "cell_count": 1,
                "cell_metrics": {1: {"area_px": 100, "diameter_px": 10, "centroid": (5, 5)}},
                "mean_cell_area_px": 100.0,
                "median_cell_area_px": 100.0,
                "average_diameter_px": 10.0,
                "cell_density": 0.01
            }
            
            state.is_dirty = True
            
            # Save Analysis as single origin
            page.save_analysis()
            
            # Verify NO mutation occurred to batch directory or records
            self.assertFalse((batch_dir / "image_A.tif").exists())
            with open(summary_csv, "r", encoding="utf-8") as f:
                content = f.read()
                self.assertNotIn("edited", content)
                self.assertIn("10", content) # Cell count stays 10
                
            # Scenario B: User opens the image from Batch Explorer redirection
            # Reset and load as batch origin
            state.workspace_manager.reset_analysis_session()
            state.current_image_path = str(raw_img_path)
            
            session_b = state.workspace_manager.start_analysis_session(
                str(raw_img_path),
                origin_type="batch",
                batch_origin_context=str(batch_dir)
            )
            self.assertEqual(session_b.origin_type, "batch")
            
            page._sync_state()
            state.analysis_results = {
                "masks": np.ones((10, 10), dtype=np.uint16),
                "cell_count": 5,
                "cell_metrics": {1: {"area_px": 20, "diameter_px": 4, "centroid": (5, 5)}},
                "mean_cell_area_px": 20.0,
                "median_cell_area_px": 20.0,
                "average_diameter_px": 4.0,
                "cell_density": 0.05
            }
            state.is_dirty = True
            
            # Save to Batch
            page.save_analysis()
            
            # Verify batch directory and files ARE updated
            self.assertTrue((batch_dir / "image_A.tif" / "image_A.tif_labels_raw.tif").exists())
            self.assertTrue((batch_dir / "image_A.tif" / "image_A.tif_cell_metrics.csv").exists())
            
            # Check summary csv has edited column and cell count updated to 5
            with open(summary_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                self.assertEqual(rows[0]["cell_count"], "5")
                self.assertEqual(rows[0]["edited"], "True")
                
        finally:
            shutil.rmtree(temp_dir)

    def test_export_correctness(self):
        """Explicitly verifies that exporting results uses the latest committed/edited state
        instead of the original analysis/Cellpose output.
        """
        import tempfile
        import shutil
        import csv
        from unittest.mock import MagicMock
        from PySide6.QtWidgets import QDialog
        from lumen.workflows.state import state
        from lumen.pages.analysis_page import AnalysisPage
        from lumen.pages.results_page import export_cell_metrics_csv
        
        state.workspace_manager.reset_analysis_session()
        state.reset_analysis_session()
        state.is_dirty = False
        
        img_path = "C:/test_images/study_D.tif"
        state.current_image_path = img_path
        
        # Original Cellpose output: 10 cells, 100 mean area
        original_results = {
            "masks": np.zeros((10, 10), dtype=np.uint16),
            "cell_count": 10,
            "cell_metrics": {i: {"area_px": 100, "diameter_px": 10, "centroid": (5, 5)} for i in range(1, 11)},
            "mean_cell_area_px": 100.0,
            "median_cell_area_px": 100.0,
            "average_diameter_px": 10.0,
            "cell_density": 0.1
        }
        state.analysis_results = original_results
        
        page = AnalysisPage()
        
        # Create a mock edited mask (only 2 cells, labels 1 and 2)
        edited_mask = np.zeros((10, 10), dtype=np.uint16)
        edited_mask[0:2, 0:2] = 1
        edited_mask[3:5, 3:5] = 2
        
        import lumen.ui.mask_editor_dialog as med_mod
        original_dialog_class = med_mod.MaskEditorDialog
        
        mock_editor = MagicMock()
        mock_editor.exec.return_value = QDialog.Accepted
        mock_editor.canvas.working_mask = edited_mask
        mock_editor.canvas.selected_label_id = None
        mock_editor.has_unsaved_changes.return_value = True
        
        med_mod.MaskEditorDialog = MagicMock(return_value=mock_editor)
        
        temp_dir = Path(tempfile.mkdtemp())
        try:
            # 1. Edit Masks -> Apply Changes
            page._on_edit_masks_clicked()
            
            # Verify in-app state is updated to latest draft (Applied state)
            self.assertEqual(state.analysis_results["cell_count"], 2)
            self.assertEqual(state.analysis_results["mean_cell_area_px"], 4.0)
            self.assertTrue(state.is_dirty)
            
            # 2. Save Analysis
            saved = page.save_analysis()
            self.assertTrue(saved)
            self.assertFalse(state.is_dirty)
            
            # 3. Export to CSV and verify correctness of contents
            export_path = temp_dir / "exported_metrics.csv"
            success = export_cell_metrics_csv(str(export_path), state.analysis_results["cell_metrics"])
            self.assertTrue(success)
            self.assertTrue(export_path.exists())
            
            # Read back exported csv to ensure it has 2 records, not 10
            with open(export_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                self.assertEqual(len(rows), 2)
                # Verify cell IDs written are 1 and 2
                cell_ids = [int(row["cell_id"]) for row in rows]
                self.assertEqual(cell_ids, [1, 2])
                
        finally:
            med_mod.MaskEditorDialog = original_dialog_class
            shutil.rmtree(temp_dir)

    def test_batch_origin_restoration_sync(self):
        """Verifies that opening a batch image from Batch Explorer (both fresh and previously modified)
        correctly restores the analysis state and enables 'Edit Masks' button without rerunning Cellpose.
        """
        import os
        import tempfile
        import csv
        import json
        import shutil
        from pathlib import Path
        from unittest.mock import MagicMock
        from PySide6.QtWidgets import QDialog
        from lumen.workflows.state import state
        from lumen.pages.analysis_page import AnalysisPage
        from lumen.pages.batch_explorer_page import BatchResultsExplorerPage
        
        state.workspace_manager.reset_analysis_session()
        state.workspace_manager.reset_batch_session()
        state.reset_analysis_session()
        state.is_dirty = False
        
        temp_dir = Path(tempfile.mkdtemp())
        try:
            # 1. Create a dummy batch directory
            batch_dir = temp_dir / "batch_results"
            os.makedirs(batch_dir, exist_ok=True)
            
            # Create a mock raw image on disk
            raw_img_path = temp_dir / "sample_001.tif"
            import tifffile
            tifffile.imwrite(str(raw_img_path), np.zeros((10, 10), dtype=np.uint16))
            
            # Setup fresh batch outputs in the batch directory
            img_results_dir = batch_dir / "sample_001.tif"
            os.makedirs(img_results_dir, exist_ok=True)
            
            # Raw labels (10 cells)
            labels_mask = np.zeros((10, 10), dtype=np.uint16)
            for i in range(1, 11):
                labels_mask[0, i-1] = i  # 10 single-pixel cells
            tifffile.imwrite(str(img_results_dir / "sample_001.tif_labels_raw.tif"), labels_mask)
            
            # Cell metrics CSV
            metrics_csv = img_results_dir / "sample_001.tif_cell_metrics.csv"
            with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["cell_id", "area_px", "diameter_px", "centroid_x", "centroid_y"])
                for i in range(1, 11):
                    writer.writerow([i, 1, 1.0, float(0), float(i-1)])
                    
            # Summary CSV
            summary_csv = batch_dir / "batch_summary.csv"
            with open(summary_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["image_name", "status", "cell_count", "mean_area_px", "workflow", "segmentation_mode"])
                writer.writerow(["sample_001.tif", "SUCCESS", "10", "1.0", "cell_counting", "Balanced"])
                
            # Manifest
            manifest_json = batch_dir / "run_manifest.json"
            with open(manifest_json, "w", encoding="utf-8") as f:
                json.dump({"images": [{"image_name": "sample_001.tif", "status": "SUCCESS", "cell_count": 10}]}, f)
                
            # Construct BatchResultsExplorerPage and point it to the batch directory
            explorer = BatchResultsExplorerPage()
            explorer.batch_dir = batch_dir
            explorer._loaded_batch_dir = batch_dir
            explorer.records = [{
                "image_name": "sample_001.tif",
                "status": "SUCCESS",
                "cell_count": "10",
                "mean_area_px": "1.0",
                "workflow": "cell_counting",
                "segmentation_mode": "Balanced"
            }]
            
            # Populate explorer list to add item and select it
            explorer._populate_list(select_default=True)
            
            # Mock find_file in explorer to return our raw_img_path
            explorer.find_file = MagicMock(return_value=raw_img_path)
            
            # =================================================================
            # CASE 1: Fresh batch image restoration
            # =================================================================
            # Simulate user clicking open button
            explorer._on_open_analysis_clicked()
            
            # Verify global state values are staged
            self.assertEqual(state.current_image_path, str(raw_img_path))
            self.assertEqual(state.current_origin_type, "batch")
            self.assertEqual(state.current_batch_origin_context, str(batch_dir))
            
            # Verify AnalysisPage restores state completely and Edit Masks is immediately enabled
            page = AnalysisPage()
            # Restore is run automatically during __init__ calling _sync_state()
            self.assertIsNotNone(state.analysis_results)
            self.assertEqual(state.analysis_results["cell_count"], 10)
            self.assertTrue(page.edit_btn.isEnabled())
            self.assertEqual(page.save_analysis_btn.text(), "💾 Save to Batch")
            
            # =================================================================
            # CASE 2: Previously modified batch image restoration & modification
            # =================================================================
            # Simulate applying changes via editor
            edited_mask = np.zeros((10, 10), dtype=np.uint16)
            edited_mask[0:2, 0:2] = 5  # Cell 5 is edited to size 4
            
            import lumen.ui.mask_editor_dialog as med_mod
            original_dialog_class = med_mod.MaskEditorDialog
            
            mock_editor = MagicMock()
            mock_editor.exec.return_value = QDialog.Accepted
            mock_editor.canvas.working_mask = edited_mask
            mock_editor.canvas.selected_label_id = None
            mock_editor.has_unsaved_changes.return_value = True
            
            med_mod.MaskEditorDialog = MagicMock(return_value=mock_editor)
            
            try:
                page._on_edit_masks_clicked()
                self.assertTrue(state.is_dirty)
                
                # Save to Batch
                page.save_analysis()
                self.assertFalse(state.is_dirty)
                
                # Now close/reset the session and simulate reopening it
                state.workspace_manager.reset_analysis_session()
                state.reset_analysis_session()
                
                # Verify that reloading it via Batch Explorer loads the EDITED state,
                # enables Edit Masks immediately, and Save to Batch works.
                explorer.records = [{
                    "image_name": "sample_001.tif",
                    "status": "SUCCESS",
                    "cell_count": "1",
                    "mean_area_px": "4.0",
                    "workflow": "cell_counting",
                    "segmentation_mode": "Balanced",
                    "edited": "True"
                }]
                
                # Populate list with the updated record list
                explorer._populate_list(select_default=True)
                
                explorer._on_open_analysis_clicked()
                
                # Instantiate new analysis page
                page_reopened = AnalysisPage()
                
                self.assertIsNotNone(state.analysis_results)
                self.assertEqual(state.analysis_results["cell_count"], 1)
                self.assertEqual(state.analysis_results["mean_cell_area_px"], 4.0)
                self.assertTrue(page_reopened.edit_btn.isEnabled())
                self.assertEqual(page_reopened.save_analysis_btn.text(), "💾 Save to Batch")
                
            finally:
                med_mod.MaskEditorDialog = original_dialog_class
                
        finally:
            shutil.rmtree(temp_dir)

    def test_batch_pdf_report_regeneration(self):
        """Verifies that saving a batch-origin image regenerates the image-level PDF report
        reflecting the latest committed edited results and overlay visualization.
        """
        import os
        import tempfile
        import csv
        import json
        import shutil
        import time
        from pathlib import Path
        from unittest.mock import MagicMock
        from PySide6.QtWidgets import QDialog
        from lumen.workflows.state import state
        from lumen.pages.analysis_page import AnalysisPage
        
        state.workspace_manager.reset_analysis_session()
        state.workspace_manager.reset_batch_session()
        state.reset_analysis_session()
        state.is_dirty = False
        
        temp_dir = Path(tempfile.mkdtemp())
        try:
            # 1. Setup a dummy batch results folder
            batch_dir = temp_dir / "batch_results"
            os.makedirs(batch_dir, exist_ok=True)
            
            raw_img_path = temp_dir / "sample_001.tif"
            import tifffile
            tifffile.imwrite(str(raw_img_path), np.zeros((10, 10), dtype=np.uint16))
            
            img_results_dir = batch_dir / "sample_001.tif"
            os.makedirs(img_results_dir, exist_ok=True)
            
            # Stale / old PDF report on disk
            pdf_path = img_results_dir / "sample_001.tif_report.pdf"
            with open(pdf_path, "w", encoding="utf-8") as f:
                f.write("stale pdf data placeholder")
                
            # Raw labels (10 cells)
            labels_mask = np.zeros((10, 10), dtype=np.uint16)
            for i in range(1, 11):
                labels_mask[0, i-1] = i
            tifffile.imwrite(str(img_results_dir / "sample_001.tif_labels_raw.tif"), labels_mask)
            
            # Summary CSV
            summary_csv = batch_dir / "batch_summary.csv"
            with open(summary_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["image_name", "status", "cell_count", "mean_area_px", "workflow", "segmentation_mode"])
                writer.writerow(["sample_001.tif", "SUCCESS", "10", "1.0", "cell_counting", "Balanced"])
                
            # Stage the session with batch origin context
            state.current_image_path = str(raw_img_path)
            session = state.workspace_manager.start_analysis_session(
                str(raw_img_path),
                origin_type="batch",
                batch_origin_context=str(batch_dir)
            )
            
            state.current_workflow = "cell_counting"
            state.quality_mode = "Balanced"
            
            # Setup initial results
            original_results = {
                "masks": labels_mask,
                "cell_count": 10,
                "cell_metrics": {i: {"area_px": 1, "diameter_px": 1.0, "centroid": (0.0, float(i-1))} for i in range(1, 11)},
                "mean_cell_area_px": 1.0,
                "median_cell_area_px": 1.0,
                "average_diameter_px": 1.0,
                "cell_density": 0.1
            }
            state.analysis_results = original_results
            
            session.analysis_results = original_results
            session.current_workflow = state.current_workflow
            session.quality_mode = state.quality_mode
            session.segmentation_method = state.segmentation_method
            
            page = AnalysisPage()
            
            # Create a mock edited mask (only 1 cell, label 1)
            edited_mask = np.zeros((10, 10), dtype=np.uint16)
            edited_mask[0:2, 0:2] = 1
            
            import lumen.ui.mask_editor_dialog as med_mod
            original_dialog_class = med_mod.MaskEditorDialog
            
            mock_editor = MagicMock()
            mock_editor.exec.return_value = QDialog.Accepted
            mock_editor.canvas.working_mask = edited_mask
            mock_editor.canvas.selected_label_id = None
            mock_editor.has_unsaved_changes.return_value = True
            
            med_mod.MaskEditorDialog = MagicMock(return_value=mock_editor)
            
            try:
                # 2. Modify masks in editor -> Apply changes (draft stage)
                page._on_edit_masks_clicked()
                self.assertTrue(state.is_dirty)
                
                # Check stale PDF is still the old placeholder
                with open(pdf_path, "r", encoding="utf-8") as f:
                    self.assertEqual(f.read(), "stale pdf data placeholder")
                    
                # 3. Save to Batch (Commit stage)
                saved = page.save_analysis()
                self.assertTrue(saved)
                self.assertFalse(state.is_dirty)
                
                # Verify PDF report is regenerated (the file contents should change and not be the placeholder anymore)
                self.assertTrue(pdf_path.exists())
                # File size should be much larger than the placeholder string because it's a real PDF now
                self.assertGreater(pdf_path.stat().st_size, 100)
                
                # Confirm we don't have the stale placeholder content
                with open(pdf_path, "rb") as f:
                    pdf_bytes = f.read(10)
                    # Real PDF starts with %PDF header
                    self.assertTrue(pdf_bytes.startswith(b"%PDF"))
                    
            finally:
                med_mod.MaskEditorDialog = original_dialog_class
                
        finally:
            shutil.rmtree(temp_dir)

    def test_single_upload_staging(self):
        """Verifies that single image upload stages the image in UploadPage
        without immediately propagating to state.current_image_path,
        and propagates it only when Proceed to Analysis is clicked.
        """
        import os
        import tempfile
        import shutil
        from pathlib import Path
        from lumen.workflows.state import state
        from lumen.pages.upload_page import UploadPage
        from lumen.processing.image_manager import image_manager
        
        state.reset_session()
        self.assertIsNone(state.current_image_path)
        
        temp_dir = Path(tempfile.mkdtemp())
        try:
            raw_img_path = temp_dir / "sample_stage.png"
            from PIL import Image
            img = Image.new("L", (10, 10))
            img.save(str(raw_img_path))
            
            upload_page = UploadPage()
            upload_page.drop_zone.file_dropped.emit(str(raw_img_path))
            
            # Verify the image is staged inside the page
            self.assertEqual(upload_page.staged_image_path, str(raw_img_path))
            
            # Verify global state.current_image_path was NOT changed
            self.assertIsNone(state.current_image_path)
            
            # Verify Proceed button is enabled after recommended workflow is selected
            self.assertTrue(upload_page.proceed_btn.isEnabled())
            
            # Simulate clicking proceed
            upload_page._on_proceed_clicked()
            
            # Now it should be set
            self.assertEqual(state.current_image_path, str(raw_img_path))
            
        finally:
            shutil.rmtree(temp_dir)

    def test_draft_commit_revert_on_navigation(self):
        """Verifies that if user navigates away while dirty and chooses Discard,
        the in-memory analysis results revert to the last committed state.
        """
        from lumen.workflows.state import state
        from lumen.pages.analysis_page import AnalysisPage
        from PySide6.QtWidgets import QDialog, QMessageBox
        from unittest.mock import MagicMock
        
        state.workspace_manager.reset_analysis_session()
        state.reset_analysis_session()
        
        img_path = "dummy_path.png"
        state.current_image_path = img_path
        session = state.workspace_manager.start_analysis_session(img_path)
        
        original_results = {"masks": np.zeros((10, 10)), "cell_count": 5}
        state.analysis_results = original_results
        session.analysis_results = original_results
        session.committed_results = original_results
        
        page = AnalysisPage()
        
        # Modify masks to trigger dirty state
        edited_mask = np.zeros((10, 10))
        edited_mask[0, 0] = 1
        
        import lumen.ui.mask_editor_dialog as med_mod
        original_dialog = med_mod.MaskEditorDialog
        
        mock_editor = MagicMock()
        mock_editor.exec.return_value = QDialog.Accepted
        mock_editor.canvas.working_mask = edited_mask
        mock_editor.canvas.selected_label_id = None
        mock_editor.has_unsaved_changes.return_value = True
        
        med_mod.MaskEditorDialog = MagicMock(return_value=mock_editor)
        
        original_msgbox = QMessageBox.exec
        QMessageBox.exec = MagicMock()
        
        try:
            page._on_edit_masks_clicked()
            self.assertTrue(state.is_dirty)
            self.assertEqual(state.analysis_results["cell_count"], 1)
            
            # Set active page to analysis
            state.current_page = "analysis"
            # Navigate to results (triggers navigation_service.navigate_to which automatically reverts)
            from lumen.core.services.navigation_service import navigation_service
            success = navigation_service.navigate_to("results")
            self.assertTrue(success)
            self.assertFalse(state.is_dirty)
            self.assertEqual(state.analysis_results["cell_count"], 5)
            
        finally:
            med_mod.MaskEditorDialog = original_dialog
            QMessageBox.exec = original_msgbox

    def test_reset_changes_functionality(self):
        """Verifies that Reset Changes button is enabled when dirty,
        and clicking it discards edits, reverts analysis_results to
        committed_results, and clears the dirty flag.
        """
        from lumen.workflows.state import state
        from lumen.pages.analysis_page import AnalysisPage
        from PySide6.QtWidgets import QDialog
        from unittest.mock import MagicMock
        
        state.workspace_manager.reset_analysis_session()
        state.reset_analysis_session()
        
        img_path = "dummy_reset.png"
        state.current_image_path = img_path
        session = state.workspace_manager.start_analysis_session(img_path)
        
        original_results = {"masks": np.zeros((10, 10)), "cell_count": 8}
        state.analysis_results = original_results
        session.analysis_results = original_results
        session.committed_results = original_results
        
        page = AnalysisPage()
        self.assertFalse(page.reset_changes_btn.isEnabled())
        
        # Modify masks
        edited_mask = np.zeros((10, 10))
        edited_mask[0, 0] = 2
        
        import lumen.ui.mask_editor_dialog as med_mod
        original_dialog = med_mod.MaskEditorDialog
        
        mock_editor = MagicMock()
        mock_editor.exec.return_value = QDialog.Accepted
        mock_editor.canvas.working_mask = edited_mask
        mock_editor.canvas.selected_label_id = None
        mock_editor.has_unsaved_changes.return_value = True
        
        med_mod.MaskEditorDialog = MagicMock(return_value=mock_editor)
        
        try:
            page._on_edit_masks_clicked()
            self.assertTrue(state.is_dirty)
            self.assertTrue(page.reset_changes_btn.isEnabled())
            
            # Click reset changes
            page._on_reset_changes_clicked()
            
            self.assertFalse(state.is_dirty)
            self.assertFalse(page.reset_changes_btn.isEnabled())
            self.assertEqual(state.analysis_results["cell_count"], 8)
            
        finally:
            med_mod.MaskEditorDialog = original_dialog
            
    def test_has_unsaved_changes_after_accept_and_done(self):
        """Verifies that has_unsaved_changes remains True even after done() clears the canvas."""
        from lumen.ui.mask_editor_dialog import MaskEditorDialog
        from PIL import Image
        import tempfile
        import shutil
        
        # Setup dummy mask
        original_mask = np.zeros((10, 10), dtype=np.uint16)
        original_mask[0:2, 0:2] = 1
        
        # Create a mock raw image file on disk first
        temp_dir = tempfile.mkdtemp()
        img_path = Path(temp_dir) / "temp_edit_test_unsaved.png"
        img = Image.new("L", (20, 20))
        img.save(str(img_path))
        
        try:
            dialog = MaskEditorDialog(str(img_path), original_mask)
            
            # Initially no changes
            self.assertFalse(dialog.has_unsaved_changes())
            
            # Mutate mask in canvas
            dialog.canvas.working_mask[2, 2] = 5
            
            # Now has unsaved changes
            self.assertTrue(dialog.has_unsaved_changes())
            
            # Accept dialog (caches self.edited_mask)
            dialog.accept()
            
            # done() clears the canvas, mimicking dialog close
            dialog.done(1)
            
            # Verify that canvas is cleared
            self.assertIsNone(dialog.canvas.working_mask)
            
            # BUT because of our fix, has_unsaved_changes() must STILL return True (evaluating against edited_mask)
            self.assertTrue(dialog.has_unsaved_changes())
        finally:
            shutil.rmtree(temp_dir)

    def test_selection_outline_cleanup(self):
        """Verifies that highlight outline contours are cleared from the viewer
        on transitions, image sync, and reset changes.
        """
        from lumen.workflows.state import state
        from lumen.pages.analysis_page import AnalysisPage
        
        state.workspace_manager.reset_analysis_session()
        state.reset_analysis_session()
        
        img_path = "dummy_selection.png"
        state.current_image_path = img_path
        session = state.workspace_manager.start_analysis_session(img_path)
        
        original_results = {"masks": np.ones((10, 10)), "cell_count": 1}
        state.analysis_results = original_results
        session.analysis_results = original_results
        session.committed_results = original_results
        
        page = AnalysisPage()
        
        # Highlight cell
        page.image_viewer.set_analysis_results(original_results)
        page.image_viewer._highlight_cell(1, np.ones((10, 10)))
        self.assertIsNotNone(page.image_viewer.highlight_item)
        
        # Triggering clear_selection should clear it
        page.clear_selection()
        self.assertIsNone(page.image_viewer.highlight_item)
        
        # Redraw
        page.image_viewer._highlight_cell(1, np.ones((10, 10)))
        self.assertIsNotNone(page.image_viewer.highlight_item)
        
        # Syncing state (like changing image or tab) should clear selection
        page._sync_state()
        self.assertIsNone(page.image_viewer.highlight_item)

if __name__ == "__main__":
    unittest.main()
