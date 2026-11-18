import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

import config
from main import draw_overlay, run_live
from vision.marker import MarkerDetector
from vision.tracker import MarkerTracker
from vision.trajectory import StrokeRecorder


def marker_frame(*circles):
    frame = np.zeros((240, 320, 3), np.uint8)
    for center, radius in circles:
        cv2.circle(frame, center, radius, (0, 255, 0), -1)
    return frame


class MarkerTests(unittest.TestCase):
    def test_detects_centroid_and_box_while_ignoring_small_blobs(self):
        frame = marker_frame(((100, 120), 18), ((200, 30), 4), ((230, 90), 1))
        detections, mask = MarkerDetector().detect(frame)
        self.assertEqual(len(detections), 1)
        detection = detections[0]
        np.testing.assert_allclose(detection.center, (100, 120), atol=0.5)
        x, y, width, height = detection.box
        self.assertTrue(x < 100 < x + width and y < 120 < y + height)
        self.assertEqual(mask[120, 100], 255)
        self.assertEqual(mask[90, 230], 0)

    def test_rejects_an_entire_background_of_marker_color(self):
        frame = marker_frame()
        frame[:] = (0, 255, 0)
        self.assertEqual(MarkerDetector().detect(frame)[0], [])

    def test_hue_range_can_cross_zero(self):
        hsv = np.zeros((240, 320, 3), np.uint8)
        cv2.circle(hsv, (60, 100), 15, (175, 255, 255), -1)
        cv2.circle(hsv, (130, 100), 15, (3, 255, 255), -1)
        cv2.circle(hsv, (200, 100), 15, (60, 255, 255), -1)
        detector = MarkerDetector((170, 90, 70), (8, 255, 255))
        detections, _ = detector.detect(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR))
        self.assertEqual(sorted(round(item.center[0]) for item in detections), [60, 130])

    def test_rejects_invalid_thresholds(self):
        for lower, upper in (
            ((180, 90, 70), (85, 255, 255)),
            ((40, 200, 70), (85, 100, 255)),
        ):
            with self.subTest(lower=lower, upper=upper), self.assertRaises(ValueError):
                MarkerDetector(lower, upper)


class TrackerTests(unittest.TestCase):
    def setUp(self):
        self.detector = MarkerDetector()
        self.tracker = MarkerTracker()

    def update(self, *circles):
        frame = marker_frame(*circles)
        detections, _ = self.detector.detect(frame)
        return self.tracker.update(detections, frame.shape)

    def test_prefers_nearby_candidate_and_smooths_position(self):
        first = self.update(((55, 100), 15))
        second = self.update(((75, 100), 15), ((240, 100), 24))
        self.assertEqual(first.center, (55.0, 100.0))
        self.assertAlmostEqual(second.center[0], 55 + 20 * config.NEW_POSITION_WEIGHT)
        self.assertGreater(second.box[0], first.box[0])
        self.assertLess(second.box[0], 75)

    def test_short_occlusion_recovers_near_prior_center(self):
        self.update(((55, 100), 15))
        self.assertIsNone(self.update())
        self.assertTrue(self.tracker.recovering)
        resumed = self.update(((75, 100), 15))
        self.assertEqual(resumed.center, (75.0, 100.0))
        self.assertFalse(self.tracker.recovering)

    def test_reinitializes_after_a_persistent_loss(self):
        self.update(((55, 100), 15))
        for _ in range(config.MISSING_FRAMES_BEFORE_RESET):
            self.assertIsNone(self.update(((250, 100), 15)))
        self.assertFalse(self.tracker.recovering)
        self.assertEqual(self.update(((250, 100), 15)).center, (250.0, 100.0))


class StrokeTests(unittest.TestCase):
    def test_records_only_after_start_and_discards_jitter(self):
        recorder = StrokeRecorder()
        recorder.add((0, 0))
        self.assertFalse(recorder.start(None))
        self.assertTrue(recorder.start((10, 10)))
        recorder.add((11, 10))
        recorder.add(None)
        recorder.add((25, 15))
        recorder.add((40, 10))
        self.assertEqual(recorder.finish(), ((10, 10), (25, 15), (40, 10)))
        recorder.clear()
        self.assertEqual(recorder.points, ())

    def test_rejects_short_stroke_and_preserves_closed_shape(self):
        recorder = StrokeRecorder()
        recorder.start((10, 10))
        recorder.add((15, 10))
        recorder.add((20, 10))
        self.assertEqual(recorder.finish(), ())
        recorder.start((10, 10))
        for point in ((50, 10), (30, 40), (10, 10)):
            recorder.add(point)
        stroke = recorder.finish()
        self.assertEqual(len(stroke), 4)
        self.assertEqual(stroke[0], stroke[-1])

    def test_overlay_draws_box_and_completed_stroke(self):
        frame = np.zeros((240, 320, 3), np.uint8)
        marker = MarkerTracker()
        detections, _ = MarkerDetector().detect(marker_frame(((100, 120), 16)))
        track = marker.update(detections, frame.shape)
        recorder = StrokeRecorder()
        recorder.start((170, 150))
        recorder.add((210, 150))
        recorder.add((250, 150))
        recorder.finish()
        draw_overlay(frame, track, recorder, marker, 30, False, "Trazo listo.")
        x, y, _, _ = track.box
        self.assertTrue(frame[y, x].any())
        self.assertTrue(frame[150, 200].any())

    def test_live_loop_cancels_lost_stroke_and_accepts_another(self):
        first = marker_frame(((80, 130), 15))
        visible = [marker_frame(((x, 130), 15)) for x in (100, 120, 140, 140)]
        missing = [marker_frame()] * config.MISSING_FRAMES_BEFORE_RESET
        capture = Mock()
        capture.read.side_effect = [(True, frame) for frame in missing + visible]
        keys = [ord(" ")] + [-1] * len(missing) + [ord(" "), -1, ord(" "), ord("q")]
        states = []

        def inspect_overlay(frame, track, recorder, tracker, fps, calibrated, message):
            states.append((recorder.recording, tuple(recorder.points), message))

        with patch.multiple("main.cv2", namedWindow=Mock(), imshow=Mock(),
                            destroyAllWindows=Mock(), getWindowProperty=Mock(return_value=1),
                            waitKey=Mock(side_effect=keys)), patch(
            "main.draw_overlay", side_effect=inspect_overlay
        ):
            run_live(capture, first, MarkerDetector(), None, show_mask=True)
            cv2.destroyAllWindows.assert_called_once()

        self.assertEqual(states[len(missing)], (False, (), "Trazo cancelado: marcador perdido."))
        recording, points, message = states[-1]
        self.assertFalse(recording)
        self.assertEqual(len(points), 3)
        self.assertEqual(message, "Trazo listo: 3 puntos.")
