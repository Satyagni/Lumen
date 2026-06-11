import unittest
import numpy as np
from lumen.core.fluorescence.quantifier import quantify_fluorescence, calculate_perimeter

class TestFluorescenceQuantifier(unittest.TestCase):
    """Unit tests for the standalone fluorescence quantification engine."""

    def test_shape_and_type_mismatches(self):
        """Verifies that the quantifier raises errors on incorrect types or shape mismatches."""
        # 1. Type mismatches
        with self.assertRaises(TypeError):
            quantify_fluorescence("not a list", np.zeros((5, 5)), ["Ch1"])
            
        with self.assertRaises(TypeError):
            quantify_fluorescence([np.zeros((5, 5))], "not an array", ["Ch1"])
            
        with self.assertRaises(TypeError):
            quantify_fluorescence([np.zeros((5, 5))], np.zeros((5, 5)), "not a list")
            
        # 2. Mismatched channel count and channel names count
        with self.assertRaises(ValueError):
            quantify_fluorescence(
                [np.zeros((5, 5)), np.zeros((5, 5))],
                np.zeros((5, 5)),
                ["Ch1"]
            )
            
        # 3. Shape mismatch between mask and raw channels
        raw_channel = np.zeros((5, 5))
        mismatched_mask = np.zeros((6, 5))
        with self.assertRaises(ValueError) as ctx:
            quantify_fluorescence([raw_channel], mismatched_mask, ["Ch1"])
        self.assertIn("Shape mismatch", str(ctx.exception))
        
        # 4. Dimension check (mask must be 2D)
        with self.assertRaises(ValueError) as ctx:
            quantify_fluorescence([np.zeros((5, 5, 2))], np.zeros((5, 5, 2)), ["Ch1"])
        self.assertIn("Mask must be a 2D array", str(ctx.exception))

    def test_single_cell_deterministic(self):
        """Verifies quantification for a single cell mask with both uniform and gradient signals."""
        # Setup a 5x5 image
        masks = np.zeros((5, 5), dtype=np.int32)
        # 3x3 square at the center as Cell 1
        masks[1:4, 1:4] = 1
        
        # Channel 1: Uniform intensity of 10.0
        ch1 = np.zeros((5, 5), dtype=np.uint16)
        ch1[1:4, 1:4] = 10
        
        # Channel 2: Gradient from 1.0 to 9.0 in the cell region
        ch2 = np.zeros((5, 5), dtype=np.float32)
        ch2[1:4, 1:4] = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ])
        
        results = quantify_fluorescence([ch1, ch2], masks, ["Uniform", "Gradient"])
        
        self.assertEqual(len(results), 1)
        res = results[0]
        
        # Geometric metrics verification
        self.assertEqual(res["cell_id"], 1)
        self.assertEqual(res["area"], 9.0)
        # 3x3 square contour perimeter through pixel centers is 8.0
        self.assertEqual(res["perimeter"], 8.0)
        
        # Uniform channel verification (ch1)
        ch1_metrics = res["channels"]["Uniform"]
        self.assertAlmostEqual(ch1_metrics["mean"], 10.0)
        self.assertAlmostEqual(ch1_metrics["median"], 10.0)
        self.assertAlmostEqual(ch1_metrics["integrated intensity"], 90.0)
        self.assertAlmostEqual(ch1_metrics["integrated_intensity"], 90.0)
        self.assertAlmostEqual(ch1_metrics["min"], 10.0)
        self.assertAlmostEqual(ch1_metrics["max"], 10.0)
        self.assertAlmostEqual(ch1_metrics["std deviation"], 0.0)
        self.assertAlmostEqual(ch1_metrics["std_deviation"], 0.0)
        
        # Flat format verification
        self.assertAlmostEqual(res["Uniform_mean"], 10.0)
        self.assertAlmostEqual(res["Uniform_integrated_intensity"], 90.0)
        
        # Gradient channel verification (ch2)
        ch2_metrics = res["channels"]["Gradient"]
        self.assertAlmostEqual(ch2_metrics["mean"], 5.0)
        self.assertAlmostEqual(ch2_metrics["median"], 5.0)
        self.assertAlmostEqual(ch2_metrics["integrated intensity"], 45.0)
        self.assertAlmostEqual(ch2_metrics["integrated_intensity"], 45.0)
        self.assertAlmostEqual(ch2_metrics["min"], 1.0)
        self.assertAlmostEqual(ch2_metrics["max"], 9.0)
        # Expected std deviation of [1, 2, 3, 4, 5, 6, 7, 8, 9] is sqrt(20/3) approx 2.581988897
        expected_std = np.std(np.arange(1, 10))
        self.assertAlmostEqual(ch2_metrics["std deviation"], expected_std)
        self.assertAlmostEqual(ch2_metrics["std_deviation"], expected_std)

    def test_multi_cell_isolation(self):
        """Verifies multi-cell isolation: separation, no leakage, and sorted outputs."""
        masks = np.zeros((5, 5), dtype=np.int32)
        # Cell 1: 1x3 line segment at row 1
        masks[1, 1:4] = 1
        # Cell 2: 1x2 line segment at row 3
        masks[3, 3:5] = 2
        
        # Raw channel with distinct signals
        # Cell 1 pixels: [10.0, 20.0, 30.0]
        # Cell 2 pixels: [100.0, 200.0]
        ch = np.zeros((5, 5), dtype=np.uint16)
        ch[1, 1] = 10
        ch[1, 2] = 20
        ch[1, 3] = 30
        ch[3, 3] = 100
        ch[3, 4] = 200
        
        results = quantify_fluorescence([ch], masks, ["Ch1"])
        
        # Check sorted by cell_id ascending
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["cell_id"], 1)
        self.assertEqual(results[1]["cell_id"], 2)
        
        # Verify Cell 1 metrics
        cell1 = results[0]
        self.assertEqual(cell1["area"], 3.0)
        # Area = 3 (> 2), uses cv2.findContours. 1x3 segment contour length is 4.0
        self.assertEqual(cell1["perimeter"], 4.0)
        
        ch1_metrics_c1 = cell1["channels"]["Ch1"]
        self.assertAlmostEqual(ch1_metrics_c1["mean"], 20.0)
        self.assertAlmostEqual(ch1_metrics_c1["median"], 20.0)
        self.assertAlmostEqual(ch1_metrics_c1["integrated intensity"], 60.0)
        self.assertAlmostEqual(ch1_metrics_c1["min"], 10.0)
        self.assertAlmostEqual(ch1_metrics_c1["max"], 30.0)
        self.assertAlmostEqual(ch1_metrics_c1["std deviation"], np.std([10.0, 20.0, 30.0]))
        
        # Verify Cell 2 metrics
        cell2 = results[1]
        self.assertEqual(cell2["area"], 2.0)
        # Area = 2 (<= 2), falls back to pixel-edge perimeter. A 1x2 segment has 6 outer edges.
        self.assertEqual(cell2["perimeter"], 6.0)
        
        ch1_metrics_c2 = cell2["channels"]["Ch1"]
        self.assertAlmostEqual(ch1_metrics_c2["mean"], 150.0)
        self.assertAlmostEqual(ch1_metrics_c2["median"], 150.0)
        self.assertAlmostEqual(ch1_metrics_c2["integrated intensity"], 300.0)
        self.assertAlmostEqual(ch1_metrics_c2["min"], 100.0)
        self.assertAlmostEqual(ch1_metrics_c2["max"], 200.0)
        self.assertAlmostEqual(ch1_metrics_c2["std deviation"], np.std([100.0, 200.0]))

    def test_single_pixel_perimeter_fallback(self):
        """Verifies fallback behavior for 1x1 pixel mask."""
        masks = np.zeros((3, 3), dtype=np.int32)
        masks[1, 1] = 1 # Area 1
        
        # Perimeter of a single pixel should fall back to 4.0
        p = calculate_perimeter(masks == 1)
        self.assertEqual(p, 4.0)
        
        # Perimeter of a 2-pixel mask should fall back to 6.0
        masks[1, 2] = 1
        p2 = calculate_perimeter(masks == 1)
        self.assertEqual(p2, 6.0)
