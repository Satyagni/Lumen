import unittest
import numpy as np
import scipy.ndimage as ndimage
from lumen.core.puncta import PunctaParameters, ThresholdMode, PunctaDetector

class TestPunctaDetector(unittest.TestCase):
    
    def setUp(self):
        # Set random seed for reproducibility
        np.random.seed(42)

    def _create_spot(self, shape, cy, cx, intensity=100.0, sigma=1.5):
        """Helper to create a Gaussian spot on a grid."""
        y, x = np.ogrid[:shape[0], :shape[1]]
        dist_sq = (y - cy) ** 2 + (x - cx) ** 2
        return intensity * np.exp(-dist_sq / (2.0 * sigma ** 2))

    def test_empty_image(self):
        """Verifies behavior for an empty (zeros) image."""
        img = np.zeros((64, 64), dtype=np.float32)
        detector = PunctaDetector()
        params = PunctaParameters(threshold_mode=ThresholdMode.ADAPTIVE)
        res = detector.detect(img, params)
        
        self.assertEqual(res.labels.max(), 0)
        self.assertEqual(len(res.object_ids), 0)
        self.assertEqual(res.labels.dtype, np.int32)

    def test_single_punctum(self):
        """Verifies single spot localization and label indexing."""
        img = np.zeros((64, 64), dtype=np.float32)
        # Place a single spot at (32, 32)
        img += self._create_spot(img.shape, 32, 32, intensity=10.0, sigma=1.0)
        
        detector = PunctaDetector()
        params = PunctaParameters(
            threshold_mode=ThresholdMode.ADAPTIVE,
            threshold_multiplier=3.0,
            minimum_size=2,
            maximum_size=100
        )
        res = detector.detect(img, params)
        
        self.assertEqual(len(res.object_ids), 1)
        self.assertEqual(res.object_ids[0], 1)
        
        # Verify the labeled pixels are clustered around (32, 32)
        coords = np.argwhere(res.labels == 1)
        cy, cx = coords.mean(axis=0)
        self.assertAlmostEqual(cy, 32.0, delta=1.0)
        self.assertAlmostEqual(cx, 32.0, delta=1.0)

    def test_multiple_puncta(self):
        """Verifies detection of multiple isolated puncta."""
        img = np.zeros((100, 100), dtype=np.float32)
        # Draw 3 spots
        img += self._create_spot(img.shape, 20, 20, intensity=20.0, sigma=1.0)
        img += self._create_spot(img.shape, 80, 20, intensity=25.0, sigma=1.2)
        img += self._create_spot(img.shape, 50, 70, intensity=30.0, sigma=0.8)
        
        detector = PunctaDetector()
        params = PunctaParameters(
            threshold_mode=ThresholdMode.ADAPTIVE,
            threshold_multiplier=3.0,
            minimum_size=2,
            maximum_size=100
        )
        res = detector.detect(img, params)
        
        self.assertEqual(len(res.object_ids), 3)
        self.assertTrue(np.array_equal(res.object_ids, np.array([1, 2, 3], dtype=np.int32)))

    def test_border_puncta(self):
        """Verifies that spots on the image boundary are detected and do not crash."""
        img = np.zeros((64, 64), dtype=np.float32)
        # Spot at top-left corner
        img += self._create_spot(img.shape, 0, 0, intensity=15.0, sigma=1.0)
        # Spot at bottom-right corner
        img += self._create_spot(img.shape, 63, 63, intensity=15.0, sigma=1.0)
        
        detector = PunctaDetector()
        params = PunctaParameters(
            threshold_mode=ThresholdMode.ADAPTIVE,
            threshold_multiplier=2.5,
            minimum_size=1
        )
        res = detector.detect(img, params)
        self.assertEqual(len(res.object_ids), 2)
        
        # Verify corner components exist in the label map
        self.assertGreater(res.labels[0, 0], 0)
        self.assertGreater(res.labels[63, 63], 0)

    def test_different_image_sizes(self):
        """Verifies that the detector performs consistently on different image dimensions."""
        detector = PunctaDetector()
        params = PunctaParameters(threshold_mode=ThresholdMode.ADAPTIVE, threshold_multiplier=3.0)
        
        for size in [128, 512]:
            img = np.zeros((size, size), dtype=np.float32)
            img += self._create_spot(img.shape, size // 2, size // 2, intensity=20.0, sigma=1.0)
            res = detector.detect(img, params)
            self.assertEqual(len(res.object_ids), 1)

    def test_size_filtering_boundaries(self):
        """Verifies that minimum_size and maximum_size filter out spots correctly."""
        img = np.zeros((100, 100), dtype=np.float32)
        # Draw 3 spots of different widths (yielding different areas)
        img += self._create_spot(img.shape, 25, 25, intensity=20.0, sigma=0.5)   # Small area
        img += self._create_spot(img.shape, 50, 50, intensity=20.0, sigma=1.5)   # Medium area
        img += self._create_spot(img.shape, 75, 75, intensity=20.0, sigma=4.0)   # Large area
        
        detector = PunctaDetector()
        
        # 1. Broad parameters: should detect all 3
        params_all = PunctaParameters(
            threshold_mode=ThresholdMode.ABSOLUTE,
            absolute_threshold=0.5,
            minimum_size=1,
            maximum_size=1000
        )
        res_all = detector.detect(img, params_all)
        self.assertEqual(len(res_all.object_ids), 3)
        
        # Determine original labeled component sizes
        sizes = np.bincount(res_all.labels.ravel())
        spot_sizes = sorted([sizes[i] for i in range(1, 4)])
        
        # 2. Strict minimum size: should filter out the smallest spot
        params_min = PunctaParameters(
            threshold_mode=ThresholdMode.ABSOLUTE,
            absolute_threshold=0.5,
            minimum_size=spot_sizes[0] + 1,
            maximum_size=1000
        )
        res_min = detector.detect(img, params_min)
        self.assertEqual(len(res_min.object_ids), 2)
        
        # 3. Strict maximum size: should filter out the largest spot
        params_max = PunctaParameters(
            threshold_mode=ThresholdMode.ABSOLUTE,
            absolute_threshold=0.5,
            minimum_size=1,
            maximum_size=spot_sizes[2] - 1
        )
        res_max = detector.detect(img, params_max)
        self.assertEqual(len(res_max.object_ids), 2)

    def test_type_and_intensity_invariance(self):
        """Verifies that the adaptive threshold mode is scale invariant and works across dtypes."""
        img = np.zeros((64, 64), dtype=np.float32)
        img += self._create_spot(img.shape, 32, 32, intensity=10.0, sigma=1.2)
        
        detector = PunctaDetector()
        params = PunctaParameters(threshold_mode=ThresholdMode.ADAPTIVE, threshold_multiplier=3.0)
        
        # Base float32 result
        res_f32 = detector.detect(img, params)
        
        # 1. Scale invariant check (10x intensity)
        res_scaled = detector.detect(img * 10.0, params)
        self.assertTrue(np.array_equal(res_f32.labels, res_scaled.labels))
        
        # 2. uint8 dtype consistency
        img_u8 = (img * (255.0 / img.max())).astype(np.uint8)
        res_u8 = detector.detect(img_u8, params)
        self.assertTrue(np.array_equal(res_f32.labels, res_u8.labels))

        # 3. uint16 dtype consistency
        img_u16 = (img * (65535.0 / img.max())).astype(np.uint16)
        res_u16 = detector.detect(img_u16, params)
        self.assertTrue(np.array_equal(res_f32.labels, res_u16.labels))

    def test_background_gradient(self):
        """Verifies adaptive DoG detector is robust against high background gradients."""
        img = np.zeros((64, 64), dtype=np.float32)
        # Place spot at (16, 16) and (48, 48)
        img += self._create_spot(img.shape, 16, 16, intensity=15.0, sigma=1.0)
        img += self._create_spot(img.shape, 48, 48, intensity=15.0, sigma=1.0)
        
        # Add a severe background gradient (optical uneven illumination)
        y_gradient, x_gradient = np.mgrid[:64, :64]
        gradient = (y_gradient + x_gradient) * 0.15
        img_with_grad = img + gradient
        
        detector = PunctaDetector()
        params = PunctaParameters(threshold_mode=ThresholdMode.ADAPTIVE, threshold_multiplier=3.0)
        
        res = detector.detect(img_with_grad, params)
        self.assertEqual(len(res.object_ids), 2)
        
        # Check centroids are preserved
        coords1 = np.argwhere(res.labels == 1)
        coords2 = np.argwhere(res.labels == 2)
        
        cy1, cx1 = coords1.mean(axis=0)
        cy2, cx2 = coords2.mean(axis=0)
        
        self.assertAlmostEqual(cy1, 16.0, delta=1.0)
        self.assertAlmostEqual(cx1, 16.0, delta=1.0)
        self.assertAlmostEqual(cy2, 48.0, delta=1.0)
        self.assertAlmostEqual(cx2, 48.0, delta=1.0)

    def test_determinism(self):
        """Verifies that the detector produces strictly identical outputs across repeated runs."""
        img = np.random.rand(64, 64).astype(np.float32) * 5.0
        # Add a couple of spots
        img += self._create_spot(img.shape, 20, 20, intensity=50.0, sigma=1.0)
        img += self._create_spot(img.shape, 40, 40, intensity=50.0, sigma=1.0)
        
        detector = PunctaDetector()
        params = PunctaParameters(threshold_mode=ThresholdMode.ADAPTIVE, threshold_multiplier=3.0)
        
        res1 = detector.detect(img, params)
        res2 = detector.detect(img, params)
        res3 = detector.detect(img, params)
        
        self.assertTrue(np.array_equal(res1.labels, res2.labels))
        self.assertTrue(np.array_equal(res1.labels, res3.labels))
        self.assertTrue(np.array_equal(res1.object_ids, res2.object_ids))

    def test_invalid_parameters(self):
        """Verifies that out of bounds parameters raise ValueError."""
        img = np.zeros((64, 64), dtype=np.float32)
        detector = PunctaDetector()
        
        with self.assertRaises(ValueError):
            detector.detect(img, PunctaParameters(sigma=0.0))
            
        with self.assertRaises(ValueError):
            detector.detect(img, PunctaParameters(sigma=-1.0))
            
        with self.assertRaises(ValueError):
            detector.detect(img, PunctaParameters(dog_sigma_ratio=0.9))
            
        with self.assertRaises(ValueError):
            detector.detect(img, PunctaParameters(minimum_size=0))
            
        with self.assertRaises(ValueError):
            detector.detect(img, PunctaParameters(minimum_size=10, maximum_size=5))
            
        with self.assertRaises(ValueError):
            detector.detect(img, PunctaParameters(threshold_multiplier=-0.5))
            
        with self.assertRaises(ValueError):
            detector.detect(img, PunctaParameters(absolute_threshold=-1.0))
            
        # Verify ValueError for wrong dimensions
        with self.assertRaises(ValueError):
            detector.detect(np.zeros((64, 64, 3), dtype=np.float32), PunctaParameters())

        # Verify TypeError for wrong types
        with self.assertRaises(TypeError):
            detector.detect([[1, 2], [3, 4]], PunctaParameters())  # type: ignore
            
        with self.assertRaises(TypeError):
            detector.detect(img, {})  # type: ignore
