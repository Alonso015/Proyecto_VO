import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from vision.camera import open_camera


class CameraTests(unittest.TestCase):
    def test_falls_back_when_backend_opens_but_cannot_read(self):
        failing = Mock()
        failing.read.return_value = (False, None)
        working = Mock()
        frame = np.zeros((48, 64, 3), np.uint8)
        working.read.return_value = (True, frame)
        with patch("vision.camera.sys.platform", "win32"), patch(
            "vision.camera.cv2.VideoCapture", side_effect=[failing, working]
        ):
            with open_camera(1, (64, 48)) as (capture, first_frame):
                self.assertIs(capture, working)
                self.assertIs(first_frame, frame)
                failing.release.assert_called_once()
                working.release.assert_not_called()
        failing.open.assert_called_once_with(1, cv2.CAP_DSHOW)
        working.open.assert_called_once_with(1, cv2.CAP_MSMF)
        working.release.assert_called_once()

    def test_reports_unavailable_camera_and_releases_it(self):
        capture = Mock()
        capture.open.return_value = False
        with patch("vision.camera.sys.platform", "linux"), patch(
            "vision.camera.cv2.VideoCapture", return_value=capture
        ):
            with self.assertRaisesRegex(RuntimeError, "webcam 0"), open_camera(0, (64, 48)):
                self.fail("La cámara simulada no se podía abrir")
        capture.release.assert_called_once()
