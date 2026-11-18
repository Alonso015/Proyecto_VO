from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from calibration.calibrate import calibrate, chessboard_points
from calibration.undistort import Calibration, Undistorter


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        self.matrix = np.array([[300.0, 0, 32], [0, 300, 24], [0, 0, 1]])
        self.calibration = Calibration(self.matrix, np.zeros(5), (64, 48), 0.2)

    def test_saves_loads_and_protects_existing_calibration(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "webcam.npz"
            self.calibration.save(path)
            restored = Calibration.load(path)
            with self.assertRaises(FileExistsError):
                self.calibration.save(path)
        np.testing.assert_array_equal(restored.camera_matrix, self.matrix)
        np.testing.assert_array_equal(restored.dist_coeffs, np.zeros(5))
        self.assertEqual(restored.image_size, (64, 48))
        self.assertAlmostEqual(restored.rms, 0.2)

    def test_rejects_corrupt_and_incomplete_files(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "broken.npz"
            path.write_bytes(b"not a numpy archive")
            with self.assertRaisesRegex(ValueError, "No se pudo leer"):
                Calibration.load(path)
            np.savez(path, camera_matrix=self.matrix)
            with self.assertRaisesRegex(ValueError, "No se pudo leer"):
                Calibration.load(path)

    def test_rejects_a_nonfinite_matrix(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "broken.npz"
            matrix = self.matrix.copy()
            matrix[0, 0] = np.nan
            Calibration(matrix, np.zeros(5), (64, 48), 0.2).save(path)
            with self.assertRaisesRegex(ValueError, "intrínseca"):
                Calibration.load(path)

    def test_undistortion_preserves_an_image_with_zero_distortion(self):
        image = np.random.default_rng(0).integers(0, 256, (48, 64, 3), dtype=np.uint8)
        undistorter = Undistorter(self.calibration)
        np.testing.assert_array_equal(undistorter.apply(image), image)
        np.testing.assert_array_equal(undistorter.apply(image), image)
        with self.assertRaisesRegex(ValueError, "64x48"):
            undistorter.apply(image[:, :32])

    def test_chessboard_points_use_columns_first(self):
        points = chessboard_points((8, 6), 25)
        self.assertEqual(points.shape, (48, 3))
        np.testing.assert_array_equal(points[[0, 7, 8, -1]],
                                      [[0, 0, 0], [175, 0, 0], [0, 25, 0], [175, 125, 0]])

    def test_empty_folder_reports_insufficient_views(self):
        with TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, "0 de 0"):
                calibrate(Path(folder), (8, 6), 25)
