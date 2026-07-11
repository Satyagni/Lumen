import unittest
import numpy as np
from dataclasses import FrozenInstanceError
from lumen.core.puncta import (
    PunctaParameters,
    ThresholdMode,
    PunctaDetectionResult,
    PunctaAssignmentResult,
    PunctaDetector,
    PunctaAssigner,
    PunctaMeasurer,
    PunctumMeasurement,
    PerCellPunctaSummary,
    PunctaResults,
    PunctaExporter,
)
from lumen.workflows.state import state

class TestPunctaFoundation(unittest.TestCase):
    
    def test_parameters_defaults(self):
        """Verifies default values of PunctaParameters."""
        params = PunctaParameters()
        self.assertEqual(params.sigma, 1.5)
        self.assertEqual(params.dog_sigma_ratio, 1.6)
        self.assertEqual(params.threshold_mode, ThresholdMode.ADAPTIVE)
        self.assertEqual(params.threshold_multiplier, 3.0)
        self.assertEqual(params.absolute_threshold, 3.0)
        self.assertEqual(params.minimum_size, 2)
        self.assertEqual(params.maximum_size, 100)
        self.assertFalse(params.enabled)
        self.assertEqual(params.channel, 0)

    def test_parameters_immutability(self):
        """Verifies that PunctaParameters is frozen (immutable)."""
        params = PunctaParameters()
        with self.assertRaises(FrozenInstanceError):
            params.sigma = 2.0  # type: ignore

    def test_detection_result_slots(self):
        """Verifies PunctaDetectionResult attributes and structure."""
        labels = np.zeros((10, 10), dtype=np.int32)
        object_ids = np.empty((0,), dtype=np.int32)
        
        res = PunctaDetectionResult(
            labels=labels,
            object_ids=object_ids
        )
        self.assertIs(res.labels, labels)
        self.assertIs(res.object_ids, object_ids)

    def test_detector_api(self):
        """Verifies detector initialization and detect() output structure."""
        params = PunctaParameters(sigma=2.5, threshold_multiplier=4.0)
        detector = PunctaDetector(params)
        self.assertEqual(detector.parameters.sigma, 2.5)
        
        image = np.ones((10, 10), dtype=np.float32)
        res = detector.detect(image, params)
        self.assertEqual(res.labels.shape, (10, 10))
        self.assertEqual(res.object_ids.shape, (0,))

    def test_assigner_placeholder(self):
        """Verifies assigner initialization and output mapping."""
        assigner = PunctaAssigner()
        cell_labels = np.zeros((10, 10), dtype=np.int32)
        detection = PunctaDetectionResult(
            labels=np.zeros((10, 10), dtype=np.int32),
            object_ids=np.empty((0,), dtype=np.int32)
        )
        res = assigner.assign(cell_labels, detection)
        self.assertEqual(res.cell_to_puncta, {})
        self.assertEqual(res.punctum_to_cell, {})
        self.assertEqual(res.unassigned_puncta, [])

    def test_measurer_placeholder(self):
        """Verifies measurer initialization and output results."""
        measurer = PunctaMeasurer()
        image = np.ones((10, 10), dtype=np.float32)
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
        
        res = measurer.measure(image, detection, assignment, cell_labels)
        self.assertIsInstance(res, PunctaResults)
        self.assertEqual(len(res.puncta_list), 0)
        self.assertEqual(len(res.per_cell_summary), 0)

    def test_exporter_not_implemented(self):
        """Verifies that exporter interfaces raise NotImplementedError."""
        exporter = PunctaExporter()
        results = PunctaResults()
        
        with self.assertRaises(NotImplementedError):
            exporter.export_csv(results, "dummy_path.csv")
            
        with self.assertRaises(NotImplementedError):
            exporter.export_summary(results, "dummy_path.csv")

    def test_state_integration(self):
        """Verifies state integration of puncta_settings."""
        # 1. Default state settings
        self.assertIsInstance(state.puncta_settings, PunctaParameters)
        self.assertFalse(state.puncta_settings.enabled)
        
        # 2. Modify state settings when no image path is loaded (transient state)
        new_params = PunctaParameters(enabled=True, sigma=3.0)
        state.puncta_settings = new_params
        self.assertEqual(state.puncta_settings.sigma, 3.0)
        self.assertTrue(state.puncta_settings.enabled)
        
        # 3. Modify state settings with session active
        state.current_image_path = "c:/test_image_puncta.png"
        session = state.workspace_manager.start_analysis_session("c:/test_image_puncta.png")
        
        state.puncta_settings = PunctaParameters(enabled=True, threshold_multiplier=5.0)
        self.assertEqual(session.puncta_settings.threshold_multiplier, 5.0)
        self.assertEqual(state.puncta_settings.threshold_multiplier, 5.0)
        
        # 4. Resetting session should restore default PunctaParameters
        state.reset_analysis_session()
        self.assertFalse(state.puncta_settings.enabled)
        self.assertEqual(state.puncta_settings.sigma, 1.5)

