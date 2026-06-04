import unittest
import os
import sys
import tempfile
import shutil
from pathlib import Path
from PIL import Image

# Ensure QApplication is initialized before importing widget code
from PySide6.QtWidgets import QApplication
app = QApplication.instance()
if not app:
    app = QApplication([])

from lumen.workflows.state import state
from lumen.pages.upload_page import UploadPage
from lumen.ui.main_window import MainWindow

class TestStabilityHardening(unittest.TestCase):
    """Verifies changes introduced during the Stability Hardening Pass."""

    def setUp(self):
        # Reset state managers
        state.workspace_manager.reset_analysis_session()
        state.workspace_manager.reset_batch_session()
        state.reset_analysis_session()
        state.current_image_path = None
        state.current_workflow = None

        # Create temp folder for files
        self.temp_dir = Path(tempfile.mkdtemp())
        self.raw_img_path = self.temp_dir / "sample_image.png"
        img = Image.new("L", (20, 20))
        img.save(str(self.raw_img_path))

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_composite_session_keys(self):
        """Verifies that WorkspaceManager distinguishes sessions using (path, origin_type)."""
        img_path = str(self.raw_img_path)

        # 1. Start single-origin session
        sess_single = state.workspace_manager.start_analysis_session(img_path, origin_type="single")
        sess_single.quality_mode = "Precise"

        # 2. Start batch-origin session for SAME image path
        sess_batch = state.workspace_manager.start_analysis_session(img_path, origin_type="batch", batch_origin_context="C:/some/batch")
        sess_batch.quality_mode = "Fast"

        # 3. Verify they are separate session instances
        self.assertNotEqual(sess_single, sess_batch)

        # 4. Verify properties are isolated
        self.assertEqual(sess_single.quality_mode, "Precise")
        self.assertEqual(sess_batch.quality_mode, "Fast")

        # 5. Verify lookup via active origin
        # With active origin = "batch"
        retrieved_sess = state.workspace_manager.get_analysis_session(img_path)
        self.assertEqual(retrieved_sess.quality_mode, "Fast")

        # Set active origin = "single" manually for check
        state.workspace_manager._active_analysis_origin = "single"
        retrieved_sess_single = state.workspace_manager.get_analysis_session(img_path)
        self.assertEqual(retrieved_sess_single.quality_mode, "Precise")

    def test_upload_page_local_staging(self):
        """Verifies that image upload on UploadPage is staged locally and does not pollute global state."""
        img_path = str(self.raw_img_path)

        upload_page = UploadPage()

        # 1. Drop a file on UploadPage
        upload_page.drop_zone.file_dropped.emit(img_path)

        # 2. Verify local staging
        self.assertEqual(upload_page.staged_image_path, img_path)

        # 3. Verify global state is NOT mutated
        self.assertIsNone(state.current_image_path)
        self.assertIsNone(state.current_workflow)

        # 4. Simulate selecting recommended card (updates staged_workflow_id, not global)
        upload_page._on_card_selected("cell_counting")
        self.assertEqual(upload_page.staged_workflow_id, "cell_counting")
        self.assertIsNone(state.current_workflow)

        # 5. Navigate away and back (staged upload remains in tact, global remains empty)
        state.current_page = "home"
        self.assertEqual(upload_page.staged_image_path, img_path)
        self.assertIsNone(state.current_image_path)

        state.current_page = "upload"
        self.assertEqual(upload_page.staged_image_path, img_path)

        # 6. Click Proceed (simulating commit)
        upload_page._on_proceed_clicked()

        # 7. Now verify global state has been mutated
        self.assertEqual(state.current_image_path, img_path)
        self.assertEqual(state.current_workflow, "cell_counting")

    def test_recursive_layout_updates(self):
        """Verifies thatMainWindow and pages can trigger layout updates without crashing."""
        win = MainWindow()
        # Verify page transitions trigger layout invalidations without exceptions
        state.page_changed.emit("upload")
        state.page_changed.emit("analysis")
        state.page_changed.emit("results")

        # Verify AnalysisPage force refresh
        win.analysis_page.force_layout_refresh()
        self.assertTrue(True)

    def test_force_layout_refresh_with_shadowed_layout(self):
        """Verifies that widgets with shadowed self.layout properties do not cause TypeError during layout updates."""
        from PySide6.QtWidgets import QWidget, QVBoxLayout
        win = MainWindow()

        # Create a custom widget that shadows self.layout with a layout instance
        class ShadowedWidget(QWidget):
            def __init__(self, parent=None):
                super().__init__(parent)
                self.layout = QVBoxLayout(self)

        # Attach it to AnalysisPage
        shadow_child = ShadowedWidget(win.analysis_page)
        
        # Verify force_layout_refresh on AnalysisPage and MainWindow does not crash
        try:
            win.analysis_page.force_layout_refresh()
            win._on_page_changed("analysis")
            success = True
        except TypeError as e:
            success = False
            self.fail(f"force_layout_refresh failed with TypeError due to shadowed layout attribute: {e}")
        
        self.assertTrue(success)

    def test_edited_mask_preservation_on_accept(self):
        """Verifies that MaskEditorDialog caches the working mask before canvas clearing in done()."""
        from lumen.ui.mask_editor_dialog import MaskEditorDialog
        import numpy as np
        
        # Setup dummy mask
        original_mask = np.zeros((10, 10), dtype=np.uint16)
        original_mask[0:2, 0:2] = 1
        
        # Create a mock raw image file on disk first
        img_path = self.temp_dir / "temp_edit_test.png"
        img = Image.new("L", (20, 20))
        img.save(str(img_path))
        
        dialog = MaskEditorDialog(str(img_path), original_mask)
        
        # Verify dialog.edited_mask is initially None
        self.assertIsNone(dialog.edited_mask)
        
        # Mutate mask in canvas
        dialog.canvas.working_mask[2, 2] = 5
        
        # Accept dialog
        dialog.accept()
        
        # Verify that dialog.edited_mask is cached
        self.assertIsNotNone(dialog.edited_mask)
        self.assertEqual(dialog.edited_mask[2, 2], 5)
        
        # Verify that after done() is called (which clears canvas), dialog.edited_mask remains cached
        dialog.done(1)
        self.assertIsNone(dialog.canvas.working_mask) # Canvas was cleared
        self.assertIsNotNone(dialog.edited_mask) # Cached reference is preserved
        self.assertEqual(dialog.edited_mask[2, 2], 5)


