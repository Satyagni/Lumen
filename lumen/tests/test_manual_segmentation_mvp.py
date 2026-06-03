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

if __name__ == "__main__":
    unittest.main()
