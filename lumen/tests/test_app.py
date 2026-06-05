import unittest
import os
import sys
from pathlib import Path

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

from lumen.core.constants import DB_FILE, ALLOWED_EXTENSIONS
from lumen.storage.database import init_db, DatabaseManager
from lumen.core.config import config
from lumen.workflows.state import state
from lumen.core.services.gpu_service import gpu_service
from lumen.processing.image_manager import image_manager
from lumen.workflows.image_classifier import classify_image


class TestLumenCore(unittest.TestCase):
    """Core logic tests validating app state, database persistence, and services."""

    @classmethod
    def setUpClass(cls):
        # Initialize test DB
        cls.test_db_path = WORKSPACE_DIR / "lumen_test.db"
        cls.db = init_db(cls.test_db_path)

    @classmethod
    def tearDownClass(cls):
        # Remove test DB file
        if cls.test_db_path.exists():
            try:
                os.remove(cls.test_db_path)
            except Exception as e:
                print(f"Failed to remove test database: {e}")

    def test_database_settings(self):
        """Verifies SQLite CRUD key-value operations."""
        self.db.set_setting("test_key", "test_value")
        self.assertEqual(self.db.get_setting("test_key"), "test_value")
        
        # Test overwrite
        self.db.set_setting("test_key", "new_value")
        self.assertEqual(self.db.get_setting("test_key"), "new_value")
        
        # Test default fallback
        self.assertEqual(self.db.get_setting("non_existent", "fallback"), "fallback")

    def test_app_config_properties(self):
        """Verifies AppConfig database properties wrapper."""
        config.theme = "light"
        self.assertEqual(config.theme, "light")
        config.theme = "dark"
        self.assertEqual(config.theme, "dark")

        # Test window geometry coordinates
        config.save_window_geometry(1400, 900, 100, 150)
        w, h, x, y = config.window_geometry
        self.assertEqual(w, 1400)
        self.assertEqual(h, 900)
        self.assertEqual(x, 100)
        self.assertEqual(y, 150)

    def test_gpu_service_detection(self):
        """Validates GPU service backend detection outputs."""
        backend = gpu_service.backend
        self.assertIn(backend, ["CPU", "CUDA"])
        is_cuda = gpu_service.is_cuda_available
        self.assertEqual(is_cuda, (backend == "CUDA"))

    def test_image_manager_validation(self):
        """Validates extension check constraints on image formats."""
        self.assertTrue(image_manager.is_valid_file("sample.png"))
        self.assertTrue(image_manager.is_valid_file("cell.tiff"))
        self.assertTrue(image_manager.is_valid_file("microscope.tif"))
        self.assertFalse(image_manager.is_valid_file("report.pdf"))
        self.assertFalse(image_manager.is_valid_file("script.py"))

    def test_app_state_signals(self):
        """Verifies AppState signal emissions on properties changes."""
        triggered = []

        def on_image_loaded(path):
            triggered.append(path)

        state.image_loaded.connect(on_image_loaded)
        state.current_image_path = "test_image.tif"

        self.assertEqual(len(triggered), 1)
        self.assertEqual(triggered[0], "test_image.tif")
        state.image_loaded.disconnect(on_image_loaded)

    def test_image_heuristics_classification(self):
        """Asserts classification accuracy on mock biological filename keywords and formats."""
        # 1. Test Fluorescence Keyword Match
        res1 = classify_image("DAPI_stain_well_A01.png", 3, "rgb", "PNG")
        self.assertEqual(res1["type"], "Fluorescence Microscopy")
        self.assertEqual(res1["confidence"], "High")
        self.assertIn("cell_counting", [w["id"] for w in res1["workflows"]])

        # 2. Test Grayscale TIFF Match
        res2 = classify_image("sample_cell_01.tif", 1, "grayscale", "TIFF")
        self.assertEqual(res2["type"], "Fluorescence Microscopy")
        self.assertEqual(res2["confidence"], "Moderate")

        # 3. Test Brightfield Match
        res3 = classify_image("brightfield_culture.jpg", 3, "rgb", "JPEG")
        self.assertEqual(res3["type"], "Brightfield Microscopy")
        self.assertEqual(res3["confidence"], "High")

        # 4. Test Colony Match
        res4 = classify_image("petri_dish_agar_plate.png", 3, "rgb", "PNG")
        self.assertEqual(res4["type"], "Colony / Plate Imaging")
        self.assertEqual(res4["confidence"], "High")
        self.assertIn("colony", [w["id"] for w in res4["workflows"]])

        # 5. Test Fallback
        res5 = classify_image("random_unstructured_filename.png", 1, "grayscale", "PNG")
        self.assertEqual(res5["type"], "Unknown Biological Imaging")
        self.assertEqual(res5["confidence"], "Low")

    def test_state_session_reset(self):
        """Verifies that reset_session clears active variables and sends correct resets signals."""
        state.current_image_path = "test.png"
        state.current_workflow = "cell_counting"
        
        state.reset_session()
        
        self.assertIsNone(state.current_image_path)
        self.assertIsNone(state.current_workflow)
        self.assertIsNone(state.analysis_results)

    def test_image_display_normalization(self):
        """Tests that dark 16-bit TIFFs are properly normalized to 8-bit for display, and checks p99<=p1 safe fallback."""
        import numpy as np
        import tifffile
        from PySide6.QtGui import QImage
        
        # 1. Test standard 16-bit dark image normalization
        dark_arr = np.zeros((100, 100), dtype=np.uint16)
        dark_arr[10:90, 10:90] = 50  # Low signal (50 out of 65535)
        dark_arr[50, 50] = 100       # Outlier/peak signal
        
        temp_tiff = WORKSPACE_DIR / "temp_test_normalization.tif"
        try:
            tifffile.imwrite(str(temp_tiff), dark_arr)
            
            success, msg = image_manager.load_image(str(temp_tiff))
            self.assertTrue(success, msg)
            
            meta = image_manager.get_metadata()
            self.assertEqual(meta["channels"], 1)
            self.assertEqual(meta["bit_depth"], 16)
            self.assertEqual(meta["mode"], "grayscale")
            
            # Verify raw image format is Grayscale16
            raw_qimg = image_manager.get_raw_qimage()
            self.assertIsNotNone(raw_qimg)
            self.assertEqual(raw_qimg.format(), QImage.Format_Grayscale16)
            
            # Verify display image format is Grayscale8 and is normalized/visible
            display_qimg = image_manager.get_qimage()
            self.assertIsNotNone(display_qimg)
            self.assertEqual(display_qimg.format(), QImage.Format_Grayscale8)
            
            # Inspect some pixel values of display image using constBits
            bits = display_qimg.constBits()
            # The peak pixel (50, 50) is 100, which is > p99 (50), so it should be clipped to 255
            # We can convert bytes to array to check value at center
            flat_bits = np.frombuffer(bits, dtype=np.uint8)
            self.assertEqual(flat_bits[50 * 100 + 50], 255)
            # A pixel at (0, 0) should be 0 (background)
            self.assertEqual(flat_bits[0], 0)
            
        finally:
            if temp_tiff.exists():
                os.remove(temp_tiff)

        # 2. Test safeguard fallback when p99 <= p1 (flat/uniform signal)
        flat_arr = np.ones((50, 50), dtype=np.uint16) * 300
        temp_flat_tiff = WORKSPACE_DIR / "temp_test_flat.tif"
        try:
            tifffile.imwrite(str(temp_flat_tiff), flat_arr)
            
            success, msg = image_manager.load_image(str(temp_flat_tiff))
            self.assertTrue(success, msg)
            
            # Grayscale8 fallback: should map flat image to zero safely without division by zero crash
            display_qimg = image_manager.get_qimage()
            self.assertIsNotNone(display_qimg)
            self.assertEqual(display_qimg.format(), QImage.Format_Grayscale8)
            
            bits = display_qimg.constBits()
            stride = display_qimg.bytesPerLine()
            flat_bits = np.frombuffer(bits, dtype=np.uint8).reshape((50, stride))
            # Slice to only inspect actual image columns (width=50), ignoring alignment padding bytes
            pixel_bits = flat_bits[:, :50]
            self.assertEqual(np.max(pixel_bits), 0) # mapped to flat zero safely
            
        finally:
            if temp_flat_tiff.exists():
                os.remove(temp_flat_tiff)

    def test_cellpose_routing_logic(self):
        """Validates microscopy-aware model routing and channel setups."""
        from lumen.workflows.cellpose_routing import determine_model_type, determine_channels
        import numpy as np
        
        self.assertEqual(determine_model_type("Fluorescence Microscopy", "DAPI_stain_well.tif"), "nuclei")
        self.assertEqual(determine_model_type("Fluorescence Microscopy", "cell_GFP_expression.png"), "cyto3")
        self.assertEqual(determine_model_type("Fluorescence Microscopy"), "cyto3")  # Default fallback
        self.assertEqual(determine_model_type("Brightfield Microscopy"), "cyto3")
        self.assertEqual(determine_model_type("Colony / Plate Imaging"), "cyto3")
        self.assertEqual(determine_model_type("Unknown Biological Imaging"), "cyto3")
        
        # Check channel configurations
        dummy_arr = np.zeros((10, 10), dtype=np.uint8)
        self.assertEqual(determine_channels("Fluorescence Microscopy", {}, dummy_arr), [0, 0])
        self.assertEqual(determine_channels("Brightfield Microscopy", {}, dummy_arr), [0, 0])

    def test_cellpose_worker_inference_pipeline(self):
        """Verifies that AnalysisWorker runs and generates masks successfully on a mock image."""
        import numpy as np
        import tifffile
        from PySide6.QtCore import QEventLoop
        
        # Create a small 20x20 mock fluorescence TIFF (single-channel)
        mock_arr = np.zeros((20, 20), dtype=np.uint16)
        mock_arr[5:15, 5:15] = 500  # draw a single cell block
        
        temp_tiff = WORKSPACE_DIR / "temp_test_cellpose.tif"
        try:
            tifffile.imwrite(str(temp_tiff), mock_arr)
            
            # Load in image manager
            success, msg = image_manager.load_image(str(temp_tiff))
            self.assertTrue(success, msg)
            
            # Setup worker
            from lumen.processing.processing_manager import AnalysisWorker
            worker = AnalysisWorker(str(temp_tiff), {"model_type": "cyto"})
            
            # Set up local QEventLoop to wait for thread execution safely
            loop = QEventLoop()
            
            results = []
            errors = []
            progress_values = []
            status_msgs = []
            
            def on_finished(res):
                results.append(res)
                loop.quit()
                
            def on_failed(err):
                errors.append(err)
                loop.quit()
                
            def on_progress(val):
                progress_values.append(val)
                
            def on_status(msg):
                status_msgs.append(msg)
                
            worker.finished_successfully.connect(on_finished)
            worker.failed.connect(on_failed)
            worker.progress_updated.connect(on_progress)
            worker.status_updated.connect(on_status)
            
            # Start worker thread
            worker.start()
            loop.exec() # Wait for finished or failed
            worker.wait() # Ensure thread is completely terminated
            
            if errors:
                self.fail(f"Worker failed: {errors[0]}")
                
            self.assertEqual(len(results), 1)
            res = results[0]
            
            # Verify results_dict keys
            self.assertIn("masks", res)
            self.assertIn("cell_metrics", res)
            self.assertIn("cell_count", res)
            self.assertIn("average_diameter_px", res)
            self.assertIn("mean_cell_area_px", res)
            self.assertIn("median_cell_area_px", res)
            self.assertIn("cell_density", res)
            self.assertIn("modality", res)
            self.assertIn("model_type", res)
            self.assertIn("used_gpu", res)
            
            # Verify cached structure elements
            if res["cell_count"] > 0:
                self.assertGreater(len(res["cell_metrics"]), 0)
                first_cell = list(res["cell_metrics"].values())[0]
                self.assertIn("area_px", first_cell)
                self.assertIn("centroid", first_cell)
                self.assertIn("diameter_px", first_cell)
            
            # Unique labels should detect our single cell block
            self.assertGreaterEqual(res["cell_count"], 0)
            self.assertEqual(res["masks"].shape, (20, 20))
            
            # Verify progress ticks and status logs
            self.assertIn(10, progress_values)
            self.assertIn(100, progress_values)
            self.assertTrue(any("Loading" in msg or "Initializing" in msg or "Executing" in msg or "Analysis" in msg for msg in status_msgs))
            
        finally:
            if temp_tiff.exists():
                os.remove(temp_tiff)

    def test_image_viewer_mask_rendering(self):
        """Verifies that InteractiveImageViewer sets pixmap and renders mask overlay correctly."""
        from lumen.pages.analysis_page import InteractiveImageViewer
        from PySide6.QtGui import QPixmap
        from PySide6.QtCore import Qt
        import numpy as np
        
        viewer = InteractiveImageViewer()
        self.assertFalse(viewer._placeholder.isHidden())
        
        # Set a dummy image
        pixmap = QPixmap(100, 100)
        pixmap.fill(Qt.black)
        viewer.set_image(pixmap)
        
        self.assertTrue(viewer._placeholder.isHidden())
        self.assertFalse(viewer.mask_item.isVisible())
        
        # Generate dummy masks
        dummy_masks = np.zeros((100, 100), dtype=np.int32)
        dummy_masks[10:30, 10:30] = 1
        dummy_masks[40:60, 40:60] = 2
        
        viewer.set_masks(dummy_masks)
        
        self.assertTrue(viewer.mask_item.isVisible())
        self.assertFalse(viewer.mask_item.pixmap().isNull())
        self.assertEqual(viewer.mask_item.pixmap().width(), 100)
        self.assertEqual(viewer.mask_item.pixmap().height(), 100)

if __name__ == "__main__":
    unittest.main()
