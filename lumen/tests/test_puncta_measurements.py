import unittest
import numpy as np
from lumen.core.puncta import (
    PunctaDetectionResult,
    PunctaAssignmentResult,
    PunctaMeasurer,
    PunctumMeasurement,
    PerCellPunctaSummary,
    PunctaResults
)

class TestPunctaMeasurements(unittest.TestCase):

    def test_constant_intensity_punctum(self):
        """Verifies measurements of a 5x5 constant intensity punctum."""
        # 10x10 image, float32
        image = np.zeros((10, 10), dtype=np.float32)
        # Spot 1 in region [2:7, 2:7]
        image[2:7, 2:7] = 10.0
        
        puncta_labels = np.zeros((10, 10), dtype=np.int32)
        puncta_labels[2:7, 2:7] = 1
        
        detection = PunctaDetectionResult(
            labels=puncta_labels,
            object_ids=np.array([1], dtype=np.int32)
        )
        
        assignment = PunctaAssignmentResult(
            cell_to_puncta={},
            punctum_to_cell={},
            unassigned_puncta=[1]
        )
        
        cell_labels = np.zeros((10, 10), dtype=np.int32)
        
        measurer = PunctaMeasurer()
        res = measurer.measure(image, detection, assignment, cell_labels)
        
        self.assertEqual(len(res.puncta_list), 1)
        m = res.puncta_list[0]
        
        self.assertEqual(m.punctum_id, 1)
        self.assertEqual(m.cell_id, 0)
        self.assertEqual(m.area, 25.0)
        self.assertEqual(m.perimeter, 20.0) # 5+5+5+5 edges
        self.assertAlmostEqual(m.equivalent_diameter, 2.0 * np.sqrt(25.0 / np.pi))
        self.assertEqual(m.centroid_y, 4.0) # index 4 is center of [2, 3, 4, 5, 6]
        self.assertEqual(m.centroid_x, 4.0)
        self.assertEqual(m.bounding_box, (2, 2, 5, 5))
        self.assertEqual(m.aspect_ratio, 1.0)
        
        # Intensity checks
        self.assertEqual(m.mean_intensity, 10.0)
        self.assertEqual(m.median_intensity, 10.0)
        self.assertEqual(m.integrated_intensity, 250.0)
        self.assertEqual(m.minimum_intensity, 10.0)
        self.assertEqual(m.maximum_intensity, 10.0)
        self.assertEqual(m.standard_deviation, 0.0)

    def test_single_pixel_punctum(self):
        """Verifies measurements of a 1x1 single pixel punctum."""
        image = np.zeros((10, 10), dtype=np.float32)
        image[5, 5] = 15.0
        
        puncta_labels = np.zeros((10, 10), dtype=np.int32)
        puncta_labels[5, 5] = 1
        
        detection = PunctaDetectionResult(
            labels=puncta_labels,
            object_ids=np.array([1], dtype=np.int32)
        )
        
        assignment = PunctaAssignmentResult(
            cell_to_puncta={},
            punctum_to_cell={},
            unassigned_puncta=[1]
        )
        
        cell_labels = np.zeros((10, 10), dtype=np.int32)
        
        measurer = PunctaMeasurer()
        res = measurer.measure(image, detection, assignment, cell_labels)
        
        self.assertEqual(len(res.puncta_list), 1)
        m = res.puncta_list[0]
        
        self.assertEqual(m.area, 1.0)
        self.assertEqual(m.perimeter, 4.0)
        self.assertAlmostEqual(m.equivalent_diameter, 2.0 * np.sqrt(1.0 / np.pi))
        self.assertEqual(m.centroid_y, 5.0)
        self.assertEqual(m.centroid_x, 5.0)
        self.assertEqual(m.bounding_box, (5, 5, 1, 1))
        self.assertEqual(m.aspect_ratio, 1.0)
        
        self.assertEqual(m.mean_intensity, 15.0)
        self.assertEqual(m.median_intensity, 15.0)
        self.assertEqual(m.integrated_intensity, 15.0)
        self.assertEqual(m.minimum_intensity, 15.0)
        self.assertEqual(m.maximum_intensity, 15.0)
        self.assertEqual(m.standard_deviation, 0.0)

    def test_float_and_random_intensity(self):
        """Verifies float32 compatibility and random intensity statistics match numpy exactly."""
        np.random.seed(42)
        random_values = np.random.normal(loc=100.0, scale=15.0, size=(5, 5)).astype(np.float32)
        
        image = np.zeros((10, 10), dtype=np.float32)
        image[2:7, 2:7] = random_values
        
        puncta_labels = np.zeros((10, 10), dtype=np.int32)
        puncta_labels[2:7, 2:7] = 2
        
        detection = PunctaDetectionResult(
            labels=puncta_labels,
            object_ids=np.array([2], dtype=np.int32)
        )
        
        assignment = PunctaAssignmentResult(
            cell_to_puncta={},
            punctum_to_cell={},
            unassigned_puncta=[2]
        )
        
        cell_labels = np.zeros((10, 10), dtype=np.int32)
        
        measurer = PunctaMeasurer()
        res = measurer.measure(image, detection, assignment, cell_labels)
        
        self.assertEqual(len(res.puncta_list), 1)
        m = res.puncta_list[0]
        
        # Verify precision calculations match numpy float64 conversions
        ref_values = random_values.astype(np.float64)
        self.assertAlmostEqual(m.mean_intensity, np.mean(ref_values), places=5)
        self.assertAlmostEqual(m.median_intensity, np.median(ref_values), places=5)
        self.assertAlmostEqual(m.integrated_intensity, np.sum(ref_values), places=4)
        self.assertAlmostEqual(m.minimum_intensity, np.min(ref_values), places=5)
        self.assertAlmostEqual(m.maximum_intensity, np.max(ref_values), places=5)
        self.assertAlmostEqual(m.standard_deviation, np.std(ref_values), places=5)

    def test_rectangular_punctum_2x8(self):
        """Verifies morphological geometry calculations for a 2x8 rectangular punctum."""
        image = np.zeros((15, 15), dtype=np.float32)
        image[3:5, 2:10] = 5.0
        
        puncta_labels = np.zeros((15, 15), dtype=np.int32)
        puncta_labels[3:5, 2:10] = 1
        
        detection = PunctaDetectionResult(
            labels=puncta_labels,
            object_ids=np.array([1], dtype=np.int32)
        )
        
        assignment = PunctaAssignmentResult(
            cell_to_puncta={},
            punctum_to_cell={},
            unassigned_puncta=[1]
        )
        
        cell_labels = np.zeros((15, 15), dtype=np.int32)
        
        measurer = PunctaMeasurer()
        res = measurer.measure(image, detection, assignment, cell_labels)
        
        self.assertEqual(len(res.puncta_list), 1)
        m = res.puncta_list[0]
        
        self.assertEqual(m.area, 16.0)
        # Perimeter: top (8) + bottom (8) + left (2) + right (2) = 20
        self.assertEqual(m.perimeter, 20.0)
        self.assertEqual(m.bounding_box, (3, 2, 2, 8)) # (min_row, min_col, height, width)
        self.assertEqual(m.aspect_ratio, 8.0 / 2.0)
        self.assertEqual(m.centroid_y, 3.5) # center of [3, 4]
        self.assertEqual(m.centroid_x, 5.5) # center of [2, 3, 4, 5, 6, 7, 8, 9]

    def test_multiple_cells_and_puncta(self):
        """Verifies cell summaries and correct mappings for multiple cells and spots."""
        image = np.ones((20, 20), dtype=np.float32) * 5.0
        
        puncta_labels = np.zeros((20, 20), dtype=np.int32)
        # Spot 1: Area=4, integrated=20
        puncta_labels[2:4, 2:4] = 1
        # Spot 2: Area=1, integrated=5
        puncta_labels[8, 8] = 2
        
        detection = PunctaDetectionResult(
            labels=puncta_labels,
            object_ids=np.array([1, 2], dtype=np.int32)
        )
        
        # Cell labels: Cell 1 (area 16), Cell 2 (area 25)
        cell_labels = np.zeros((20, 20), dtype=np.int32)
        cell_labels[1:5, 1:5] = 1
        cell_labels[7:12, 7:12] = 2
        
        assignment = PunctaAssignmentResult(
            cell_to_puncta={1: [1], 2: [2], 3: []}, # Cell 3 has no puncta
            punctum_to_cell={1: 1, 2: 2},
            unassigned_puncta=[]
        )
        
        measurer = PunctaMeasurer()
        res = measurer.measure(image, detection, assignment, cell_labels)
        
        self.assertEqual(len(res.puncta_list), 2)
        self.assertEqual(len(res.per_cell_summary), 3)
        
        # Summary for Cell 1 (puncta: [1])
        c1 = res.per_cell_summary[1]
        self.assertEqual(c1.cell_id, 1)
        self.assertEqual(c1.cell_area, 16.0)
        self.assertEqual(c1.puncta_count, 1)
        self.assertEqual(c1.average_puncta_area, 4.0)
        self.assertEqual(c1.average_puncta_intensity, 5.0)
        self.assertEqual(c1.average_integrated_intensity, 20.0)
        self.assertEqual(c1.largest_punctum_id, 1)
        self.assertEqual(c1.smallest_punctum_id, 1)
        self.assertEqual(c1.total_puncta_area, 4.0)
        self.assertEqual(c1.total_puncta_integrated_intensity, 20.0)
        self.assertEqual(c1.max_punctum_intensity, 5.0)
        self.assertEqual(c1.max_punctum_area, 4.0)
        
        # Summary for Cell 2 (puncta: [2])
        c2 = res.per_cell_summary[2]
        self.assertEqual(c2.cell_id, 2)
        self.assertEqual(c2.cell_area, 25.0)
        self.assertEqual(c2.puncta_count, 1)
        self.assertEqual(c2.average_puncta_area, 1.0)
        self.assertEqual(c2.average_puncta_intensity, 5.0)
        self.assertEqual(c2.average_integrated_intensity, 5.0)
        
        # Summary for Cell 3 (empty cell)
        c3 = res.per_cell_summary[3]
        self.assertEqual(c3.cell_id, 3)
        self.assertEqual(c3.cell_area, 0.0) # Not present in cell_labels image
        self.assertEqual(c3.puncta_count, 0)
        self.assertEqual(c3.average_puncta_area, 0.0)
        self.assertEqual(c3.average_puncta_intensity, 0.0)
        self.assertEqual(c3.average_integrated_intensity, 0.0)
        self.assertEqual(c3.largest_punctum_id, 0)
        self.assertEqual(c3.smallest_punctum_id, 0)
        self.assertEqual(c3.total_puncta_area, 0.0)
        self.assertEqual(c3.total_puncta_integrated_intensity, 0.0)
        self.assertEqual(c3.max_punctum_intensity, 0.0)
        self.assertEqual(c3.max_punctum_area, 0.0)

    def test_validation_handling(self):
        """Verifies shape mismatches, bad types, and negative label assertions."""
        image = np.zeros((10, 10), dtype=np.float32)
        detection = PunctaDetectionResult(
            labels=np.zeros((10, 10), dtype=np.int32),
            object_ids=np.empty((0,), dtype=np.int32)
        )
        assignment = PunctaAssignmentResult(
            cell_to_puncta={},
            punctum_to_cell={},
            unassigned_puncta=[]
        )
        cell_labels = np.zeros((10, 10), dtype=np.int32)
        
        measurer = PunctaMeasurer()
        
        # Dimension Mismatch
        with self.assertRaises(ValueError):
            measurer.measure(np.zeros((5, 5), dtype=np.float32), detection, assignment, cell_labels)
            
        # Float Dtype cell_labels
        with self.assertRaises(TypeError):
            measurer.measure(image, detection, assignment, np.zeros((10, 10), dtype=np.float32))
            
        # Negative cell labels
        neg_cells = np.zeros((10, 10), dtype=np.int32)
        neg_cells[2, 2] = -1
        with self.assertRaises(ValueError):
            measurer.measure(image, detection, assignment, neg_cells)
