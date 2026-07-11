import unittest
import os
import csv
import json
import tempfile
from lumen.core.fluorescence.exporters import export_cell_csv, export_summary_csv

class TestFluorescenceExporters(unittest.TestCase):
    """Unit tests for the standalone CSV exporters."""

    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.test_dir.cleanup()

    def test_cell_level_export_headers_and_values(self):
        """Verifies cell-level CSV headers, ordering, values, and channel-order preservation."""
        # Setup mock quantifier output where channels are specifically ordered as GFP then DAPI
        quantifier_output = [
            {
                "cell_id": 1,
                "area": 250,
                "perimeter": 60.5,
                "GFP_mean": 10.0,
                "GFP_median": 9.5,
                "GFP_integrated_intensity": 2500.0,
                "GFP_min": 5.0,
                "GFP_max": 15.0,
                "GFP_std_deviation": 2.1,
                "DAPI_mean": 100.0,
                "DAPI_median": 99.0,
                "DAPI_integrated_intensity": 25000.0,
                "DAPI_min": 80.0,
                "DAPI_max": 120.0,
                "DAPI_std_deviation": 10.5
            },
            {
                "cell_id": 2,
                "area": 150,
                "perimeter": 45.0,
                "GFP_mean": 20.0,
                "GFP_median": 19.0,
                "GFP_integrated_intensity": 3000.0,
                "GFP_min": 10.0,
                "GFP_max": 30.0,
                "GFP_std_deviation": 5.2,
                "DAPI_mean": 200.0,
                "DAPI_median": 198.0,
                "DAPI_integrated_intensity": 30000.0,
                "DAPI_min": 150.0,
                "DAPI_max": 250.0,
                "DAPI_std_deviation": 25.4
            }
        ]
        
        file_path = os.path.join(self.test_dir.name, "cells.csv")
        export_cell_csv(quantifier_output, file_path)
        
        # Read the file back and verify
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)
            
        # 1. Verify headers list (must preserve GFP before DAPI)
        expected_headers = [
            "cell_id", "area", "perimeter",
            "GFP_mean", "GFP_median", "GFP_integrated_intensity", "GFP_min", "GFP_max", "GFP_std_deviation",
            "DAPI_mean", "DAPI_median", "DAPI_integrated_intensity", "DAPI_min", "DAPI_max", "DAPI_std_deviation"
        ]
        self.assertEqual(rows[0], expected_headers)
        
        # 2. Verify row ordering and values
        self.assertEqual(len(rows), 3) # Header + 2 data rows
        
        # Row 1 (Cell 1)
        self.assertEqual(int(rows[1][0]), 1)       # cell_id
        self.assertEqual(int(rows[1][1]), 250)     # area
        self.assertEqual(float(rows[1][2]), 60.5)  # perimeter
        self.assertEqual(float(rows[1][3]), 10.0)  # GFP_mean
        self.assertEqual(float(rows[1][9]), 100.0) # DAPI_mean
        
        # Row 2 (Cell 2)
        self.assertEqual(int(rows[2][0]), 2)       # cell_id
        self.assertEqual(int(rows[2][1]), 150)     # area
        self.assertEqual(float(rows[2][2]), 45.0)  # perimeter
        self.assertEqual(float(rows[2][3]), 20.0)  # GFP_mean
        self.assertEqual(float(rows[2][9]), 200.0) # DAPI_mean

    def test_summary_level_export(self):
        """Verifies summary CSV key-value pair output, calculations, and settings serialization."""
        quantifier_output = [
            {
                "cell_id": 1,
                "area": 200,
                "GFP_mean": 10.0,
                "GFP_median": 8.0,
                "DAPI_mean": 100.0,
                "DAPI_median": 90.0
            },
            {
                "cell_id": 2,
                "area": 300,
                "GFP_mean": 20.0,
                "GFP_median": 18.0,
                "DAPI_mean": 200.0,
                "DAPI_median": 190.0
            }
        ]
        
        pre_settings = {"auto_contrast": True}
        seg_settings = {"threshold": 0.35}
        timestamp = "2026-06-11 12:00:00"
        file_path = os.path.join(self.test_dir.name, "summary.csv")
        
        export_summary_csv(
            image_filename="sample.png",
            quantifier_output=quantifier_output,
            preprocessing_settings=pre_settings,
            segmentation_settings=seg_settings,
            timestamp=timestamp,
            file_path=file_path
        )
        
        # Read the file back and verify
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)
            
        self.assertEqual(rows[0], ["Metric", "Value"])
        
        # Convert rows to a dictionary for easy validation
        metrics = {r[0]: r[1] for r in rows[1:]}
        
        self.assertEqual(metrics["image_filename"], "sample.png")
        self.assertEqual(metrics["timestamp"], timestamp)
        self.assertEqual(int(metrics["total_cell_count"]), 2)
        self.assertAlmostEqual(float(metrics["average_area"]), 250.0)
        
        # Verify GFP channel averages
        self.assertAlmostEqual(float(metrics["GFP_mean_average"]), 15.0)
        self.assertAlmostEqual(float(metrics["GFP_median_average"]), 13.0)
        
        # Verify DAPI channel averages
        self.assertAlmostEqual(float(metrics["DAPI_mean_average"]), 150.0)
        self.assertAlmostEqual(float(metrics["DAPI_median_average"]), 140.0)
        
        # Verify settings serialization
        self.assertEqual(json.loads(metrics["preprocessing_settings"]), pre_settings)
        self.assertEqual(json.loads(metrics["segmentation_settings"]), seg_settings)

    def test_empty_case_handling(self):
        """Verifies both exporters handle empty quantifier output gracefully."""
        # 1. Cell-level empty export
        cell_file_path = os.path.join(self.test_dir.name, "empty_cells.csv")
        export_cell_csv([], cell_file_path, fallback_channel_names=["GFP", "DAPI"])
        
        with open(cell_file_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)
            
        self.assertEqual(len(rows), 1) # Only header row
        expected_headers = [
            "cell_id", "area", "perimeter",
            "GFP_mean", "GFP_median", "GFP_integrated_intensity", "GFP_min", "GFP_max", "GFP_std_deviation",
            "DAPI_mean", "DAPI_median", "DAPI_integrated_intensity", "DAPI_min", "DAPI_max", "DAPI_std_deviation"
        ]
        self.assertEqual(rows[0], expected_headers)
        
        # 2. Summary-level empty export
        sum_file_path = os.path.join(self.test_dir.name, "empty_summary.csv")
        export_summary_csv(
            image_filename="sample.png",
            quantifier_output=[],
            preprocessing_settings={},
            segmentation_settings={},
            timestamp="2026-06-11",
            file_path=sum_file_path,
            fallback_channel_names=["GFP"]
        )
        
        with open(sum_file_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            sum_rows = list(reader)
            
        metrics = {r[0]: r[1] for r in sum_rows[1:]}
        self.assertEqual(int(metrics["total_cell_count"]), 0)
        self.assertEqual(float(metrics["average_area"]), 0.0)
        self.assertEqual(float(metrics["GFP_mean_average"]), 0.0)
        self.assertEqual(float(metrics["GFP_median_average"]), 0.0)

    def test_round_trip_validation(self):
        """Verifies round-trip validation (write CSV -> reload -> verify values)."""
        quantifier_output = [
            {
                "cell_id": 1,
                "area": 100,
                "perimeter": 40.0,
                "Ch1_mean": 5.5,
                "Ch1_median": 5.0,
                "Ch1_integrated_intensity": 550.0,
                "Ch1_min": 1.0,
                "Ch1_max": 10.0,
                "Ch1_std_deviation": 2.5
            }
        ]
        
        file_path = os.path.join(self.test_dir.name, "round_trip.csv")
        export_cell_csv(quantifier_output, file_path)
        
        # Reload and verify structure & data type correctness
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
        self.assertEqual(len(rows), 1)
        reloaded_row = rows[0]
        
        # Verify types and value matching
        self.assertEqual(int(reloaded_row["cell_id"]), 1)
        self.assertEqual(int(reloaded_row["area"]), 100)
        self.assertEqual(float(reloaded_row["perimeter"]), 40.0)
        self.assertEqual(float(reloaded_row["Ch1_mean"]), 5.5)
        self.assertEqual(float(reloaded_row["Ch1_median"]), 5.0)
        self.assertEqual(float(reloaded_row["Ch1_integrated_intensity"]), 550.0)
        self.assertEqual(float(reloaded_row["Ch1_min"]), 1.0)
        self.assertEqual(float(reloaded_row["Ch1_max"]), 10.0)
        self.assertEqual(float(reloaded_row["Ch1_std_deviation"]), 2.5)

    def test_micron_mode_export(self):
        """Verifies cell-level and summary CSV exports in micron mode."""
        quantifier_output = [
            {
                "cell_id": 1,
                "area": 25.0,  # physical area (e.g. um2)
                "perimeter": 15.2, # physical perimeter (e.g. um)
                "GFP_mean": 10.0,
                "GFP_median": 9.5,
                "GFP_integrated_intensity": 250.0,
                "GFP_min": 5.0,
                "GFP_max": 15.0,
                "GFP_std_deviation": 2.1
            }
        ]
        
        # 1. Test cell export in micron mode
        cell_path = os.path.join(self.test_dir.name, "cells_micron.csv")
        export_cell_csv(quantifier_output, cell_path, calibration_mode="micron")
        
        with open(cell_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
            rows = list(reader)
            
        self.assertEqual(headers[:3], ["cell_id", "area_um2", "perimeter_um"])
        self.assertEqual(float(rows[0]["area_um2"]), 25.0)
        self.assertEqual(float(rows[0]["perimeter_um"]), 15.2)

        # 2. Test summary export in micron mode
        summary_path = os.path.join(self.test_dir.name, "summary_micron.csv")
        export_summary_csv(
            image_filename="test_image.czi",
            quantifier_output=quantifier_output,
            preprocessing_settings={},
            segmentation_settings={},
            timestamp="2026-07-06 12:00:00",
            file_path=summary_path,
            calibration_mode="micron"
        )
        
        with open(summary_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)
            
        summary_dict = dict(rows)
        self.assertIn("average_area_um2", summary_dict)
        self.assertEqual(float(summary_dict["average_area_um2"]), 25.0)
