"""Segmentación HSV de la práctica 1 y morfología de la práctica 3."""

from dataclasses import dataclass

import cv2
import numpy as np

import config

Point = tuple[float, float]
Box = tuple[int, int, int, int]


@dataclass(frozen=True)
class Detection:
    center: Point
    box: Box  # x, y, ancho, alto
    area: float


class MarkerDetector:
    def __init__(self, lower=config.HSV_LOWER, upper=config.HSV_UPPER,
                 min_area=config.MIN_MARKER_AREA):
        if len(lower) != 3 or len(upper) != 3:
            raise ValueError("Cada límite HSV debe contener H, S y V.")
        if any(not 0 <= value <= limit
               for limits in (lower, upper)
               for value, limit in zip(limits, (179, 255, 255))):
            raise ValueError("H debe estar entre 0 y 179. S y V entre 0 y 255.")
        if lower[1] > upper[1] or lower[2] > upper[2]:
            raise ValueError("Los mínimos de S y V deben ser menores o iguales que sus máximos.")
        if not np.isfinite(min_area) or min_area <= 0:
            raise ValueError("El área mínima debe ser positiva y finita.")

        self.min_area = min_area
        self.ranges = [(tuple(lower), tuple(upper))]
        if lower[0] > upper[0]:
            # El tono rojo atraviesa la discontinuidad entre H=179 y H=0.
            self.ranges = [
                (tuple(lower), (179, upper[1], upper[2])),
                ((0, lower[1], lower[2]), tuple(upper)),
            ]
        kernel_size = config.MORPHOLOGY_KERNEL_SIZE
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    def detect(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, *self.ranges[0])
        if len(self.ranges) == 2:
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, *self.ranges[1]))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        max_area = mask.size * config.MAX_MARKER_AREA_FRACTION
        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if not self.min_area <= area <= max_area:
                continue
            moments = cv2.moments(contour)
            if moments["m00"] == 0:
                continue
            center = (moments["m10"] / moments["m00"], moments["m01"] / moments["m00"])
            detections.append(Detection(center, cv2.boundingRect(contour), area))
        return detections, mask
