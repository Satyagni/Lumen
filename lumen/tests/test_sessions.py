import unittest
import os
import sys
from pathlib import Path
from PySide6.QtGui import QTransform

# Add root folder to sys.path
TEST_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = TEST_DIR.parent.parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

# Ensure QApplication is initialized before importing widget code
from PySide6.QtWidgets import QApplication
app = QApplication.instance()
if not app:
    app = QApplication([])

from lumen.workflows.state import state, AnalysisSession, BatchResultSession
from lumen.pages.analysis_page import AnalysisPage
from lumen.pages.batch_explorer_page import BatchResultsExplorerPage

class TestWorkspaceSessions(unittest.TestCase):
    """Verifies state persistence, workspace sessions lifecycle, and invalidation rules."""

    def setUp(self):
        # Reset global state sessions
        state.workspace_manager.reset_analysis_session()
        state.workspace_manager.reset_batch_session()
        state.reset_analysis_session()
        state.batch_results_dir = ""
        
        # Create dummy nuclei_clean.tif for test_analysis_page_save_restore
        self.dummy_tiff = WORKSPACE_DIR / "nuclei_clean.tif"
        if not self.dummy_tiff.exists():
            import numpy as np
            import tifffile
            dummy_arr = np.zeros((100, 100), dtype=np.uint16)
            dummy_arr[20:80, 20:80] = 100
            tifffile.imwrite(str(self.dummy_tiff), dummy_arr)
            self.created_dummy = True
        else:
            self.created_dummy = False

    def tearDown(self):
        if hasattr(self, 'created_dummy') and self.created_dummy and self.dummy_tiff.exists():
            try:
                import os
                os.remove(self.dummy_tiff)
            except Exception:
                pass

    def test_analysis_session_lifecycle(self):
        """Validates that AnalysisSession correctly creates, stores, and invalidates context."""
        image_path = "C:/test_images/DAPI_stain.tif"
        
        # Test creation
        session = state.workspace_manager.start_analysis_session(image_path)
        self.assertIsNotNone(session)
        self.assertEqual(session.image_path, image_path)
        
        # Test retrieval with same path
        retrieved = state.workspace_manager.get_analysis_session(image_path)
        self.assertEqual(retrieved, session)
        
        # Test retrieval with mismatched path
        mismatched = state.workspace_manager.get_analysis_session("C:/test_images/other.tif")
        self.assertIsNone(mismatched)
        
        # Test explicit clearing
        state.workspace_manager.reset_analysis_session(image_path)
        self.assertIsNone(state.workspace_manager.get_analysis_session(image_path))

    def test_batch_session_lifecycle(self):
        """Validates that BatchResultSession correctly creates, stores, and invalidates context."""
        batch_dir = "C:/test_batches/run_01"
        
        # Test creation
        session = state.workspace_manager.start_batch_session(batch_dir)
        self.assertIsNotNone(session)
        self.assertEqual(session.batch_results_dir, batch_dir)
        
        # Test retrieval with same path
        retrieved = state.workspace_manager.get_batch_session(batch_dir)
        self.assertEqual(retrieved, session)
        
        # Test mismatched path
        mismatched = state.workspace_manager.get_batch_session("C:/test_batches/run_02")
        self.assertIsNone(mismatched)
        
        # Test explicit clearing
        state.workspace_manager.reset_batch_session(batch_dir)
        self.assertIsNone(state.workspace_manager.get_batch_session(batch_dir))

    def test_analysis_page_save_restore(self):
        """Verifies AnalysisPage can serialize to session and restore correctly on navigation."""
        page = AnalysisPage()
        
        # Setup mock active image path and analysis settings
        img_path = str(WORKSPACE_DIR / "nuclei_clean.tif")
        state.current_image_path = img_path
        state.quality_mode = "Precise"
        state.mask_opacity = 75
        state.show_original_image = False
        state.show_segmentation_overlay = True
        state.segmentation_method = "AI Segmentation"
        
        # Simulate loading the image to active view
        page._sync_state()
        
        # Set some zoom transform to simulate user interaction
        t = QTransform()
        t.scale(2.5, 2.5)
        page.image_viewer.setTransform(t)
        page.image_viewer._zoom_touched = True
        
        # Trigger save
        page._save_to_session()
        
        # Fetch created session
        session = state.workspace_manager.get_analysis_session(img_path)
        self.assertIsNotNone(session)
        self.assertEqual(session.quality_mode, "Precise")
        self.assertEqual(session.mask_opacity, 75)
        self.assertFalse(session.show_original_image)
        self.assertEqual(session.viewer_state["transform"].m11(), 2.5)
        self.assertTrue(session.viewer_state["zoom_touched"])
        
        # Clear global state variables to defaults
        state.quality_mode = "Balanced"
        state.mask_opacity = 40
        state.show_original_image = True
        
        # Trigger page restore
        page._sync_state()
        
        # Verify restored values in app state
        self.assertEqual(state.quality_mode, "Precise")
        self.assertEqual(state.mask_opacity, 75)
        self.assertFalse(state.show_original_image)
        
        # Verify restored values in page widgets & viewer
        self.assertEqual(page.quality_combo.currentText(), "Precise")
        self.assertEqual(page.opacity_slider.value(), 75)
        self.assertFalse(page.show_original_chk.isChecked())
        self.assertEqual(page.image_viewer.transform().m11(), 2.5)
        
    def test_batch_explorer_page_save_restore(self):
        """Verifies BatchResultsExplorerPage saves search, sort, and selection to session."""
        explorer = BatchResultsExplorerPage()
        
        # Setup mock active batch directory
        batch_dir = str(WORKSPACE_DIR)
        state.batch_results_dir = batch_dir
        explorer.batch_dir = Path(batch_dir)
        
        # Mock summary record data directly onto explorer
        explorer.records = [
            {"image_name": "nuclei_clean.tif", "status": "SUCCESS", "cell_count": "45"},
            {"image_name": "F01_1615w1.TIF", "status": "SUCCESS", "cell_count": "12"}
        ]
        
        # Set filters
        explorer.search_bar.setText("clean")
        explorer.sort_combo.setCurrentText("Cell Count")
        explorer.opacity_slider.setValue(60)
        
        # Simulate populate
        explorer._populate_list(select_default=False)
        
        # Select item
        item = explorer.navigator_list.item(0)
        explorer.navigator_list.setCurrentItem(item)
        
        # Save session
        explorer._save_to_session()
        
        # Check session
        session = state.workspace_manager.get_batch_session(batch_dir)
        self.assertIsNotNone(session)
        self.assertEqual(session.search_text, "clean")
        self.assertEqual(session.sort_by, "Cell Count")
        self.assertEqual(session.mask_opacity, 60)
        self.assertEqual(session.selected_filename, "nuclei_clean.tif")
        
        # Reset GUI settings (block signals to avoid triggering signals/saves)
        explorer.search_bar.blockSignals(True)
        explorer.search_bar.clear()
        explorer.search_bar.blockSignals(False)
        
        explorer.sort_combo.blockSignals(True)
        explorer.sort_combo.setCurrentText("Alphabetical")
        explorer.sort_combo.blockSignals(False)
        
        explorer.opacity_slider.blockSignals(True)
        explorer.opacity_slider.setValue(40)
        explorer.opacity_slider.blockSignals(False)
        
        explorer.records = []
        
        # Restore session
        explorer._load_from_state()
        
        # Verify restored search/sort/filter state
        self.assertEqual(explorer.search_bar.text(), "clean")
        self.assertEqual(explorer.sort_combo.currentText(), "Cell Count")
        self.assertEqual(explorer.opacity_slider.value(), 60)
        self.assertEqual(explorer.records[0]["image_name"], "nuclei_clean.tif")

    def test_results_navigation_redirection(self):
        """Validates that navigate_to('results') goes directly to 'results' (no batch redirection)."""
        from lumen.core.services.navigation_service import navigation_service
        
        # Scenario 1: No active batch session
        state.batch_results_dir = ""
        state.workspace_manager.reset_batch_session()
        
        success = navigation_service.navigate_to("results")
        self.assertTrue(success)
        self.assertEqual(state.current_page, "results")
        
        # Scenario 2: Active batch session exists (should still go to results, not batch_explorer!)
        batch_dir = str(WORKSPACE_DIR)
        state.batch_results_dir = batch_dir
        state.workspace_manager.start_batch_session(batch_dir)
        
        success = navigation_service.navigate_to("results")
        self.assertTrue(success)
        self.assertEqual(state.current_page, "results")
        
        # Scenario 3: Verify sidebar active highlighting for batch_explorer
        from lumen.ui.sidebar import SidebarWidget
        sidebar = SidebarWidget()
        sidebar._on_state_page_changed("batch_explorer")
        self.assertTrue(sidebar.nav_buttons["analysis"].isChecked())
        
        # Scenario 4: Verify sidebar active highlighting for analysis
        sidebar._on_state_page_changed("analysis")
        self.assertTrue(sidebar.nav_buttons["analysis"].isChecked())
