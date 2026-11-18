"""Registra un trazo explícito, conservando sus puntos en orden temporal."""

from math import dist, hypot

import config
from vision.marker import Point


class StrokeRecorder:
    def __init__(self):
        self._active = None
        self.last: tuple[Point, ...] = ()

    @property
    def recording(self):
        return self._active is not None

    @property
    def points(self):
        return self._active if self.recording else self.last

    def start(self, point):
        if self.recording or point is None:
            return False
        self.last = ()
        self._active = [point]
        return True

    def add(self, point):
        if self.recording and point is not None:
            if dist(self._active[-1], point) >= config.MIN_POINT_DISTANCE:
                self._active.append(point)

    def finish(self):
        if not self.recording:
            return ()
        points = self._active
        self._active = None
        if len(points) >= 3:
            xs, ys = zip(*points)
            extent = hypot(max(xs) - min(xs), max(ys) - min(ys))
            if extent >= config.MIN_STROKE_EXTENT:
                self.last = tuple(points)
        return self.last

    def clear(self):
        self._active = None
        self.last = ()
