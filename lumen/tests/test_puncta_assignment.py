import unittest
import numpy as np
from lumen.core.puncta import PunctaParameters, PunctaDetectionResult, PunctaAssigner

class TestPunctaAssignment(unittest.TestCase):
    
    def test_single_punctum_assignment(self):
        """Verifies that a single punctum inside a cell is correctly assigned."""
        # 30x30 image
        cell_labels = np.zeros((30, 30), dtype=np.int32)
        # Draw cell 1 in region [10:20, 10:20]
        cell_labels[10:20, 10:20] = 1
        
        puncta_labels = np.zeros((30, 30), dtype=np.int32)
        # Draw spot 1 in cell 1 center
        puncta_labels[14:17, 14:17] = 1
        
        detection = PunctaDetectionResult(
            labels=puncta_labels,
            object_ids=np.array([1], dtype=np.int32)
        )
        
        assigner = PunctaAssigner()
        res = assigner.assign(cell_labels, detection)
        
        self.assertEqual(res.cell_to_puncta, {1: [1]})
        self.assertEqual(res.punctum_to_cell, {1: 1})
        self.assertEqual(res.unassigned_puncta, [])

    def test_multiple_puncta_assignment(self):
        """Verifies multiple puncta mapped to the same cell and sorted correctly."""
        cell_labels = np.zeros((30, 30), dtype=np.int32)
        # Cell 5 in region [5:25, 5:25]
        cell_labels[5:25, 5:25] = 5
        
        puncta_labels = np.zeros((30, 30), dtype=np.int32)
        # Punctum 2
        puncta_labels[8:11, 8:11] = 2
        # Punctum 1
        puncta_labels[18:21, 18:21] = 1
        
        detection = PunctaDetectionResult(
            labels=puncta_labels,
            object_ids=np.array([1, 2], dtype=np.int32)
        )
        
        assigner = PunctaAssigner()
        res = assigner.assign(cell_labels, detection)
        
        # Verify cell list is sorted: [1, 2] instead of order of traversal
        self.assertEqual(res.cell_to_puncta, {5: [1, 2]})
        self.assertEqual(res.punctum_to_cell, {1: 5, 2: 5})
        self.assertEqual(res.unassigned_puncta, [])

    def test_multiple_cells_assignment(self):
        """Verifies discrete mapping of multiple cells and spots."""
        cell_labels = np.zeros((30, 30), dtype=np.int32)
        cell_labels[2:8, 2:8] = 1
        cell_labels[20:28, 20:28] = 2
        
        puncta_labels = np.zeros((30, 30), dtype=np.int32)
        puncta_labels[4:6, 4:6] = 2   # lands in Cell 1
        puncta_labels[24:26, 24:26] = 1 # lands in Cell 2
        
        detection = PunctaDetectionResult(
            labels=puncta_labels,
            object_ids=np.array([1, 2], dtype=np.int32)
        )
        
        assigner = PunctaAssigner()
        res = assigner.assign(cell_labels, detection)
        
        self.assertEqual(res.cell_to_puncta, {1: [2], 2: [1]})
        self.assertEqual(res.punctum_to_cell, {2: 1, 1: 2})
        self.assertEqual(res.unassigned_puncta, [])

    def test_unassigned_puncta(self):
        """Verifies punctum falling in background is added to unassigned_puncta."""
        cell_labels = np.zeros((30, 30), dtype=np.int32)
        cell_labels[2:8, 2:8] = 3
        
        puncta_labels = np.zeros((30, 30), dtype=np.int32)
        puncta_labels[15:18, 15:18] = 2 # lands in background
        puncta_labels[4:6, 4:6] = 1     # lands in Cell 3
        
        detection = PunctaDetectionResult(
            labels=puncta_labels,
            object_ids=np.array([1, 2], dtype=np.int32)
        )
        
        assigner = PunctaAssigner()
        res = assigner.assign(cell_labels, detection)
        
        self.assertEqual(res.cell_to_puncta, {3: [1]})
        self.assertEqual(res.punctum_to_cell, {1: 3})
        self.assertEqual(res.unassigned_puncta, [2])

    def test_no_puncta(self):
        """Verifies assignment when no puncta are detected."""
        cell_labels = np.zeros((10, 10), dtype=np.int32)
        cell_labels[2:5, 2:5] = 1
        
        detection = PunctaDetectionResult(
            labels=np.zeros((10, 10), dtype=np.int32),
            object_ids=np.empty((0,), dtype=np.int32)
        )
        
        assigner = PunctaAssigner()
        res = assigner.assign(cell_labels, detection)
        
        # Cell 1 is pre-populated with empty list
        self.assertEqual(res.cell_to_puncta, {1: []})
        self.assertEqual(res.punctum_to_cell, {})
        self.assertEqual(res.unassigned_puncta, [])

    def test_no_cells(self):
        """Verifies assignment when no cells are in the image."""
        cell_labels = np.zeros((10, 10), dtype=np.int32)
        
        puncta_labels = np.zeros((10, 10), dtype=np.int32)
        puncta_labels[2:4, 2:4] = 2
        puncta_labels[6:8, 6:8] = 1
        
        detection = PunctaDetectionResult(
            labels=puncta_labels,
            object_ids=np.array([1, 2], dtype=np.int32)
        )
        
        assigner = PunctaAssigner()
        res = assigner.assign(cell_labels, detection)
        
        self.assertEqual(res.cell_to_puncta, {})
        self.assertEqual(res.punctum_to_cell, {})
        self.assertEqual(res.unassigned_puncta, [1, 2])

    def test_border_spots(self):
        """Verifies spots positioned on boundary borders are resolved without crash."""
        cell_labels = np.zeros((10, 10), dtype=np.int32)
        cell_labels[0, 0] = 10
        cell_labels[9, 9] = 20
        
        puncta_labels = np.zeros((10, 10), dtype=np.int32)
        puncta_labels[0, 0] = 1
        puncta_labels[9, 9] = 2
        
        detection = PunctaDetectionResult(
            labels=puncta_labels,
            object_ids=np.array([1, 2], dtype=np.int32)
        )
        
        assigner = PunctaAssigner()
        res = assigner.assign(cell_labels, detection)
        
        self.assertEqual(res.cell_to_puncta, {10: [1], 20: [2]})
        self.assertEqual(res.punctum_to_cell, {1: 10, 2: 20})
        self.assertEqual(res.unassigned_puncta, [])

    def test_non_contiguous_label_ids(self):
        """Verifies that non-contiguous label IDs (e.g., 1, 7, 15) map correctly."""
        cell_labels = np.zeros((30, 30), dtype=np.int32)
        cell_labels[10:20, 10:20] = 4
        
        puncta_labels = np.zeros((30, 30), dtype=np.int32)
        puncta_labels[12:15, 12:15] = 7  # lands inside Cell 4
        puncta_labels[2:5, 2:5] = 15     # lands in background
        puncta_labels[16:19, 16:19] = 1  # lands inside Cell 4
        
        detection = PunctaDetectionResult(
            labels=puncta_labels,
            object_ids=np.array([1, 7, 15], dtype=np.int32)
        )
        
        assigner = PunctaAssigner()
        res = assigner.assign(cell_labels, detection)
        
        # Verify sorted mapped list: [1, 7]
        self.assertEqual(res.cell_to_puncta, {4: [1, 7]})
        self.assertEqual(res.punctum_to_cell, {1: 4, 7: 4})
        self.assertEqual(res.unassigned_puncta, [15])

    def test_boundary_centroid_determinism(self):
        """Verifies deterministic int truncation for spots on boundary lines."""
        cell_labels = np.zeros((30, 30), dtype=np.int32)
        # Border boundary between Cell 1 and Cell 2 is at col 15
        cell_labels[:, :15] = 1
        cell_labels[:, 15:] = 2
        
        # Punctum whose centroid evaluates to exactly 14.5 in x coordinate
        # Draw spot on columns 13, 14, 15, 16 (centered at x = 14.5)
        puncta_labels = np.zeros((30, 30), dtype=np.int32)
        puncta_labels[10, 13:17] = 1
        
        # Verify centroid is indeed 14.5
        y_indices, x_indices = np.where(puncta_labels == 1)
        self.assertEqual(x_indices.mean(), 14.5)
        
        detection = PunctaDetectionResult(
            labels=puncta_labels,
            object_ids=np.array([1], dtype=np.int32)
        )
        
        assigner = PunctaAssigner()
        res = assigner.assign(cell_labels, detection)
        
        # int(14.5) -> 14. cell_labels[:, 14] is Cell 1.
        self.assertEqual(res.cell_to_puncta, {1: [1], 2: []})
        self.assertEqual(res.punctum_to_cell, {1: 1})

    def test_validation_handling(self):
        """Verifies that invalid dimensions, types, or negative labels raise exceptions."""
        cell_labels = np.zeros((10, 10), dtype=np.int32)
        detection = PunctaDetectionResult(
            labels=np.zeros((10, 10), dtype=np.int32),
            object_ids=np.empty((0,), dtype=np.int32)
        )
        
        assigner = PunctaAssigner()
        
        # Shape mismatch
        with self.assertRaises(ValueError):
            assigner.assign(np.zeros((5, 5), dtype=np.int32), detection)
            
        # 3D cell labels
        with self.assertRaises(ValueError):
            assigner.assign(np.zeros((10, 10, 3), dtype=np.int32), detection)
            
        # Float dtype cell labels
        with self.assertRaises(TypeError):
            assigner.assign(np.zeros((10, 10), dtype=np.float32), detection)
            
        # Negative cell labels
        neg_labels = np.zeros((10, 10), dtype=np.int32)
        neg_labels[2, 2] = -1
        with self.assertRaises(ValueError):
            assigner.assign(neg_labels, detection)
            
        # Object ID mismatch (labels has label 1, but object_ids is empty)
        bad_labels = np.zeros((10, 10), dtype=np.int32)
        bad_labels[2, 2] = 1
        bad_detection = PunctaDetectionResult(
            labels=bad_labels,
            object_ids=np.empty((0,), dtype=np.int32)
        )
        with self.assertRaises(ValueError):
            assigner.assign(cell_labels, bad_detection)

    def test_preserve_inputs(self):
        """Verifies that inputs are not modified in-place."""
        cell_labels = np.zeros((10, 10), dtype=np.int32)
        cell_labels[2:5, 2:5] = 1
        
        puncta_labels = np.zeros((10, 10), dtype=np.int32)
        puncta_labels[3, 3] = 1
        
        detection = PunctaDetectionResult(
            labels=puncta_labels,
            object_ids=np.array([1], dtype=np.int32)
        )
        
        cell_labels_copy = cell_labels.copy()
        puncta_labels_copy = puncta_labels.copy()
        object_ids_copy = detection.object_ids.copy()
        
        assigner = PunctaAssigner()
        assigner.assign(cell_labels, detection)
        
        self.assertTrue(np.array_equal(cell_labels, cell_labels_copy))
        self.assertTrue(np.array_equal(detection.labels, puncta_labels_copy))
        self.assertTrue(np.array_equal(detection.object_ids, object_ids_copy))

    def test_determinism(self):
        """Verifies identical outputs across repeated runs."""
        cell_labels = np.zeros((30, 30), dtype=np.int32)
        cell_labels[2:10, 2:10] = 1
        cell_labels[15:25, 15:25] = 2
        
        puncta_labels = np.zeros((30, 30), dtype=np.int32)
        puncta_labels[4:6, 4:6] = 2
        puncta_labels[18:20, 18:20] = 1
        
        detection = PunctaDetectionResult(
            labels=puncta_labels,
            object_ids=np.array([1, 2], dtype=np.int32)
        )
        
        assigner = PunctaAssigner()
        res1 = assigner.assign(cell_labels, detection)
        res2 = assigner.assign(cell_labels, detection)
        
        self.assertEqual(res1.cell_to_puncta, res2.cell_to_puncta)
        self.assertEqual(res1.punctum_to_cell, res2.punctum_to_cell)
        self.assertEqual(res1.unassigned_puncta, res2.unassigned_puncta)
