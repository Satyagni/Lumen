import unittest
import os
import sys
import numpy as np
import tempfile
import tifffile
from pathlib import Path
from PySide6.QtWidgets import QApplication

# Add root folder to sys.path
TEST_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = TEST_DIR.parent.parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

# Ensure QApplication is initialized
app = QApplication.instance()
if not app:
    app = QApplication([])

from lumen.workflows.state import state, AnalysisSession
from lumen.processing.image_manager import image_manager
from lumen.core.fluorescence.channels import get_default_channel_names

class TestFluorescencePipeline(unittest.TestCase):
    """Verifies image loading, multi-channel parsing, transposition, naming heuristics, and AppState properties."""

    def setUp(self):
        # Reset state defaults
        state.workspace_manager.reset_analysis_session()
        state.reset_analysis_session()
        image_manager.clear_cache()
        
        # Create temporary directory for test files
        self.test_dir = tempfile.TemporaryDirectory()
        
    def tearDown(self):
        self.test_dir.cleanup()
        image_manager.clear_cache()

    def test_single_channel_grayscale_image(self):
        """Verifies loading a standard single-channel grayscale image behaves correctly."""
        # Create a mock grayscale numpy array (20x20)
        img_arr = np.random.randint(0, 255, (20, 20), dtype=np.uint8)
        img_path = os.path.join(self.test_dir.name, "single_gray.png")
        
        import PIL.Image
        PIL.Image.fromarray(img_arr).save(img_path)
        
        success, msg = image_manager.load_image(img_path, set_state=True)
        self.assertTrue(success, msg)
        self.assertEqual(image_manager.get_metadata()["channels"], 1)
        self.assertEqual(image_manager.get_metadata()["width"], 20)
        self.assertEqual(image_manager.get_metadata()["height"], 20)
        self.assertEqual(state.channel_names, ["Grayscale"])
        self.assertEqual(state.active_viewer_channel, 0)
        self.assertEqual(image_manager._active_channel_idx, 0)

    def test_multichannel_chw_tiff_image(self):
        """Verifies loading a multi-channel (C, H, W) TIFF image gets correctly transposed to (H, W, C)."""
        # Create a mock multi-channel array: 3 channels, 15 height, 15 width
        img_arr = np.random.randint(0, 1000, (3, 15, 15), dtype=np.uint16)
        img_path = os.path.join(self.test_dir.name, "cell_DAPI_stain.tif")
        
        # Write CHW array
        tifffile.imwrite(img_path, img_arr)
        
        success, msg = image_manager.load_image(img_path, set_state=True)
        self.assertTrue(success, msg)
        
        meta = image_manager.get_metadata()
        self.assertEqual(meta["channels"], 3)
        self.assertEqual(meta["width"], 15)
        self.assertEqual(meta["height"], 15)
        self.assertEqual(image_manager._raw_numpy_arr.shape, (15, 15, 3)) # Transposed!
        self.assertEqual(len(image_manager._raw_channels), 3)
        self.assertEqual(image_manager._raw_channels[0].shape, (15, 15))
        
        # Check heuristic channel names: DAPI, GFP, RFP
        self.assertEqual(state.channel_names, ["DAPI", "GFP", "RFP"])
        
        # Multi-channel defaults viewer active channel to -1 (Composite)
        self.assertEqual(state.active_viewer_channel, -1)
        self.assertEqual(image_manager._active_channel_idx, -1)

    def test_active_viewer_channel_updates_cache(self):
        """Verifies changing the active display channel updates the cached display QImages properly."""
        # Setup multi-channel image
        img_arr = np.random.randint(0, 255, (2, 10, 10), dtype=np.uint8)
        img_path = os.path.join(self.test_dir.name, "multi_two.tif")
        tifffile.imwrite(img_path, img_arr)
        
        image_manager.load_image(img_path, set_state=True)
        
        # Initial display image is composite (-1)
        self.assertEqual(image_manager._active_channel_idx, -1)
        qimg_composite = image_manager.get_qimage()
        self.assertFalse(qimg_composite.isNull())
        
        # Switch to Channel 0 (DAPI)
        image_manager.set_active_channel(0)
        self.assertEqual(image_manager._active_channel_idx, 0)
        qimg_ch0 = image_manager.get_qimage()
        self.assertFalse(qimg_ch0.isNull())
        
        # Switch back to Composite
        image_manager.set_active_channel(-1)
        self.assertEqual(image_manager._active_channel_idx, -1)

    def test_app_state_fluorescence_session_properties(self):
        """Verifies fluorescence properties inside AppState correctly serialize and restore sessions."""
        img_path = os.path.join(self.test_dir.name, "dummy.tif")
        open(img_path, "w").close() # Dummy file touch
        
        # Set current image path first, then start analysis session
        state.current_image_path = img_path
        session = state.workspace_manager.start_analysis_session(img_path)
        
        # Write properties
        state.channel_names = ["Ch0_DAPI", "Ch1_GFP"]
        state.segmentation_channel = 1
        state.active_viewer_channel = 0
        state.background_mode = "Local Ring"
        state.background_params = {"offset": 3, "thickness": 6}
        state.active_metric = "median"
        
        # Verify stored in session
        self.assertEqual(session.channel_names, ["Ch0_DAPI", "Ch1_GFP"])
        self.assertEqual(session.segmentation_channel, 1)
        self.assertEqual(session.active_viewer_channel, 0)
        self.assertEqual(session.background_mode, "Local Ring")
        self.assertEqual(session.background_params, {"offset": 3, "thickness": 6})
        self.assertEqual(session.active_metric, "median")
        
        # Reset session
        state.reset_analysis_session()
        self.assertEqual(state.channel_names, [])
        self.assertEqual(state.segmentation_channel, 0)
        self.assertEqual(state.active_viewer_channel, -1)
        self.assertEqual(state.background_mode, "None")
        self.assertEqual(state.active_metric, "mean")

    def test_preprocessing_pipeline(self):
        """Verifies modular preprocessing calculations, state bindings, and defaults."""
        # 1. Check default state properties
        self.assertTrue(state.preprocess_auto_contrast)
        self.assertEqual(state.preprocess_percentile_low, 1.0)
        self.assertEqual(state.preprocess_percentile_high, 99.0)
        self.assertEqual(state.preprocess_brightness, 0.0)
        self.assertEqual(state.preprocess_contrast, 1.0)
        self.assertEqual(state.preprocess_gamma, 1.0)

        # Create mock 2D array
        arr = np.array([[10, 50, 100], [150, 200, 250]], dtype=np.uint8)
        
        # Test auto contrast stretching
        out_auto = image_manager.preprocess_array(arr)
        self.assertEqual(out_auto.shape, arr.shape)
        self.assertEqual(out_auto.dtype, np.uint8)
        self.assertEqual(out_auto[0, 0], 0) # stretched min
        self.assertEqual(out_auto[1, 2], 255) # stretched max

        # Disable auto contrast and test adjustments
        state.preprocess_auto_contrast = False
        out_no_auto = image_manager.preprocess_array(arr)
        self.assertNotEqual(out_no_auto[0, 0], 0) # not min stretched

        # Test brightness
        state.preprocess_brightness = 0.5
        out_bright = image_manager.preprocess_array(arr)
        self.assertTrue(np.all(out_bright >= out_no_auto))

        # Test reset
        state.reset_analysis_session()
        self.assertTrue(state.preprocess_auto_contrast)
        self.assertEqual(state.preprocess_brightness, 0.0)

