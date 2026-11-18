"""Asocia contornos en el tiempo y suaviza el centro del marcador."""

from dataclasses import dataclass
from math import dist, hypot

import config
from vision.marker import Box, Point


@dataclass(frozen=True)
class Track:
    center: Point
    box: Box


class MarkerTracker:
    def __init__(self):
        self._last_detection = None
        self._center = None
        self._missing = 0

    @property
    def recovering(self):
        return self._last_detection is not None and self._missing > 0

    def update(self, detections, frame_shape):
        if self._last_detection is None:
            detection = max(detections, key=lambda item: item.area, default=None)
        else:
            height, width = frame_shape[:2]
            max_jump = hypot(width, height) * config.MAX_JUMP_FRACTION
            last_center = self._last_detection.center
            detection = min(detections, key=lambda item: dist(item.center, last_center), default=None)
            if detection is not None and dist(detection.center, last_center) > max_jump:
                detection = None

        if detection is None:
            if self._last_detection is not None:
                self._missing += 1
                if self._missing >= config.MISSING_FRAMES_BEFORE_RESET:
                    self._last_detection = self._center = None
                    self._missing = 0
            return None

        if self._center is None or self._missing:
            self._center = detection.center
        else:
            weight = config.NEW_POSITION_WEIGHT
            self._center = tuple(
                (1 - weight) * previous + weight * current
                for previous, current in zip(self._center, detection.center)
            )
        self._last_detection = detection
        self._missing = 0
        return Track(self._center, detection.box)
