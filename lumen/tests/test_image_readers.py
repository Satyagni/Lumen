import unittest
import numpy as np
import tempfile
import os
from pathlib import Path
import PIL.Image
import tifffile

from lumen.core.imaging import (
    ImageReaderFactory,
    ImageMetadata,
    ImageData,
    ProjectionMode,
    ProjectionEngine,
    TiffReader,
    PilReader
)

class TestImageReaders(unittest.TestCase):
    """Regression test suite for the unified ImageReader and metadata pipeline."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

        # 1. Create a mock 2D grayscale TIFF file
        self.tiff_2d_gray_path = self.temp_path / "test_2d_gray.tif"
        self.tiff_2d_gray_data = np.arange(100, dtype=np.uint16).reshape((10, 10))
        tifffile.imwrite(str(self.tiff_2d_gray_path), self.tiff_2d_gray_data)

        # 2. Create a mock 3D multi-channel TIFF file (e.g. 3 channels)
        self.tiff_3d_path = self.temp_path / "test_3d_channels.tif"
        # shape is (10, 10, 3) after our transposition logic, so we save as (10, 10, 3)
        self.tiff_3d_data = np.arange(300, dtype=np.uint16).reshape((10, 10, 3))
        tifffile.imwrite(str(self.tiff_3d_path), self.tiff_3d_data)

        # 3. Create a mock PNG image
        self.png_path = self.temp_path / "test_image.png"
        self.png_data = np.zeros((10, 10, 3), dtype=np.uint8)
        self.png_data[3:7, 3:7, 0] = 255  # Red box
        pil_img = PIL.Image.fromarray(self.png_data)
        pil_img.save(str(self.png_path))

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_factory_registry(self):
        """Verifies that the factory returns the correct reader instance based on file extension."""
        tiff_reader = ImageReaderFactory.get_reader(str(self.tiff_2d_gray_path))
        self.assertIsInstance(tiff_reader, TiffReader)

        png_reader = ImageReaderFactory.get_reader(str(self.png_path))
        self.assertIsInstance(png_reader, PilReader)

        with self.assertRaises(ValueError):
            ImageReaderFactory.get_reader("invalid_file.unsupported_ext")

    def test_tiff_reader_2d_gray(self):
        """Verifies metadata and data loading for a 2D grayscale TIFF."""
        reader = ImageReaderFactory.get_reader(str(self.tiff_2d_gray_path))
        reader.open(str(self.tiff_2d_gray_path))

        meta = reader.get_metadata()
        self.assertEqual(meta.filename, "test_2d_gray.tif")
        self.assertEqual(meta.channels, 1)
        self.assertEqual(meta.z_planes, 1)
        self.assertEqual(meta.timepoints, 1)
        self.assertEqual(meta.dimensions, (10, 10))
        self.assertEqual(meta.dimension_order, "YX")
        self.assertEqual(meta.bit_depth, 16)
        self.assertEqual(meta.mode, "grayscale")
        self.assertFalse(reader.supports("z_stack"))

        # Read slice
        slice_data = reader.read_slice(scene=0, channel=0)
        self.assertIsInstance(slice_data, ImageData)
        np.testing.assert_array_equal(slice_data.image, self.tiff_2d_gray_data)
        self.assertEqual(slice_data.metadata, meta)

        reader.close()

    def test_tiff_reader_3d_channels(self):
        """Verifies metadata and slice slicing for a 3D multi-channel TIFF."""
        reader = ImageReaderFactory.get_reader(str(self.tiff_3d_path))
        reader.open(str(self.tiff_3d_path))

        meta = reader.get_metadata()
        self.assertEqual(meta.channels, 3)
        self.assertEqual(meta.dimensions, (10, 10, 3))
        self.assertEqual(meta.dimension_order, "YXC")
        self.assertEqual(meta.mode, "rgb")

        # Read channel slices
        for c in range(3):
            slice_data = reader.read_slice(scene=0, channel=c)
            np.testing.assert_array_equal(slice_data.image, self.tiff_3d_data[..., c])

        reader.close()

    def test_pil_reader_png(self):
        """Verifies PIL reader metadata and array loading for PNG."""
        reader = ImageReaderFactory.get_reader(str(self.png_path))
        reader.open(str(self.png_path))

        meta = reader.get_metadata()
        self.assertEqual(meta.filename, "test_image.png")
        self.assertEqual(meta.channels, 3)
        self.assertEqual(meta.dimension_order, "YXC")
        self.assertEqual(meta.bit_depth, 8)
        self.assertEqual(meta.mode, "rgb")

        slice_data = reader.read_slice(scene=0, channel=0)
        np.testing.assert_array_equal(slice_data.image, self.png_data[..., 0])

        reader.close()

    def test_projection_engine(self):
        """Verifies that the stateless ProjectionEngine performs MIP and MEAN projections correctly."""
        # Create a mock 3D volume of shape (3, 4, 4)
        # Plane 0: all 1s
        # Plane 1: all 5s
        # Plane 2: all 3s
        stack = np.zeros((3, 4, 4), dtype=np.uint8)
        stack[0, ...] = 1
        stack[1, ...] = 5
        stack[2, ...] = 3

        # MIP should return 5 everywhere
        mip_proj = ProjectionEngine.project(stack, ProjectionMode.MIP)
        np.testing.assert_array_equal(mip_proj, np.ones((4, 4), dtype=np.uint8) * 5)

        # MEAN should return (1+5+3)/3 = 3 everywhere
        mean_proj = ProjectionEngine.project(stack, ProjectionMode.MEAN)
        np.testing.assert_array_equal(mean_proj, np.ones((4, 4), dtype=np.uint8) * 3)

        # NONE mode should return the original stack unmodified
        none_proj = ProjectionEngine.project(stack, ProjectionMode.NONE)
        np.testing.assert_array_equal(none_proj, stack)

        # 2D stack projection should return the 2D array unchanged
        flat_2d = np.ones((4, 4), dtype=np.uint8) * 10
        proj_2d = ProjectionEngine.project(flat_2d, ProjectionMode.MIP)
        np.testing.assert_array_equal(proj_2d, flat_2d)

    def test_czi_reader_mocked(self):
        """Verifies CziReader parsing and loading behavior using mocked aicspylibczi CziFile."""
        from unittest.mock import patch, MagicMock
        from lumen.core.imaging import CziReader

        mock_xml = """<?xml version="1.0" encoding="utf-8"?>
        <ImageDocument>
            <Metadata>
                <Scaling>
                    <Items>
                        <Distance Id="X">
                            <Value>0.25e-6</Value>
                        </Distance>
                        <Distance Id="Y">
                            <Value>0.25e-6</Value>
                        </Distance>
                        <Distance Id="Z">
                            <Value>1.0e-6</Value>
                        </Distance>
                    </Items>
                </Scaling>
                <DisplaySetting>
                    <Channels>
                        <Channel Name="DAPI" />
                        <Channel Name="GFP" />
                    </Channels>
                </DisplaySetting>
            </Metadata>
        </ImageDocument>
        """

        mock_image_data = np.arange(100, dtype=np.uint16).reshape((1, 1, 1, 10, 10)) # S=1, Z=1, C=1, Y=10, X=10

        with patch("aicspylibczi.CziFile") as MockCziFile:
            mock_instance = MagicMock()
            mock_instance.dims = "SZCYX"
            mock_instance.size = (1, 1, 2, 10, 10)
            mock_instance.read_image.return_value = (mock_image_data, None)
            mock_instance.raw_metadata.return_value = mock_xml
            MockCziFile.return_value = mock_instance

            reader = CziReader()
            reader.open("dummy_file.czi")

            meta = reader.get_metadata()
            self.assertEqual(meta.filename, "dummy_file.czi")
            self.assertEqual(meta.channels, 2)
            self.assertEqual(meta.scene_count, 1)
            self.assertEqual(meta.z_planes, 1)
            self.assertEqual(meta.timepoints, 1)
            self.assertEqual(meta.dimension_order, "SZCYX")
            self.assertEqual(meta.mode, "grayscale")
            self.assertEqual(meta.channel_names, ["DAPI", "GFP"])
            self.assertEqual(meta.physical_units, "µm")
            self.assertAlmostEqual(meta.voxel_size[0], 0.25)  # 0.25e-6 * 1e6
            self.assertAlmostEqual(meta.voxel_size[1], 0.25)
            self.assertAlmostEqual(meta.voxel_size[2], 1.0)
            self.assertTrue(reader.supports("physical_units"))
            self.assertFalse(reader.supports("z_stack"))  # z_planes = 1

            # Read slice
            slice_data = reader.read_slice(scene=0, channel=0)
            self.assertEqual(slice_data.image.shape, (10, 10))
            self.assertEqual(slice_data.metadata, meta)

            reader.close()

if __name__ == "__main__":
    unittest.main()
