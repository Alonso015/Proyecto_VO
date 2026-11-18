"""Carga parámetros pinhole y precalcula los mapas de corrección."""

from dataclasses import dataclass
from pathlib import Path
from zipfile import BadZipFile

import cv2
import numpy as np


@dataclass(frozen=True)
class Calibration:
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    image_size: tuple[int, int]
    rms: float

    @classmethod
    def load(cls, path):
        try:
            with np.load(path, allow_pickle=False) as data:
                matrix = np.asarray(data["camera_matrix"], dtype=np.float64)
                distortion = np.asarray(data["dist_coeffs"], dtype=np.float64).reshape(-1)
                size = np.asarray(data["image_size"])
                rms = np.asarray(data["rms"], dtype=np.float64)
        except (OSError, TypeError, ValueError, KeyError, EOFError, BadZipFile) as error:
            raise ValueError(f"No se pudo leer la calibración {path}: {error}") from error

        if (matrix.shape != (3, 3) or not np.isfinite(matrix).all()
                or matrix[0, 0] <= 0 or matrix[1, 1] <= 0
                or not np.allclose(matrix[2], [0, 0, 1])):
            raise ValueError("La matriz intrínseca de la calibración es inválida.")
        if distortion.size not in (4, 5, 8, 12, 14) or not np.isfinite(distortion).all():
            raise ValueError("Los coeficientes de distorsión pinhole son inválidos.")
        if size.shape != (2,) or size.dtype.kind not in "iu" or not (size > 0).all():
            raise ValueError("La resolución de calibración debe indicar ancho y alto enteros.")
        if rms.shape != () or not np.isfinite(rms) or rms < 0:
            raise ValueError("El error RMS de la calibración es inválido.")
        return cls(matrix, distortion, (int(size[0]), int(size[1])), float(rms))

    def save(self, path):
        path = Path(path)
        if path.suffix.lower() != ".npz":
            raise ValueError("La ruta de salida debe terminar en .npz.")
        path.parent.mkdir(parents=True, exist_ok=True)
        # El modo exclusivo protege calibraciones anteriores.
        with path.open("xb") as output:
            np.savez(output, camera_matrix=self.camera_matrix, dist_coeffs=self.dist_coeffs,
                     image_size=self.image_size, rms=self.rms)


class Undistorter:
    def __init__(self, calibration):
        self.image_size = calibration.image_size
        self._maps = None
        self._calibration = calibration

    def apply(self, frame):
        actual_size = (frame.shape[1], frame.shape[0])
        if actual_size != self.image_size:
            width, height = self.image_size
            actual_width, actual_height = actual_size
            raise ValueError(
                f"La calibración es de {width}x{height}, pero la webcam entrega "
                f"{actual_width}x{actual_height}. Usa una resolución compatible "
                "o calibra la cámara con esa resolución."
            )
        if self._maps is None:
            matrix = self._calibration.camera_matrix
            self._maps = cv2.initUndistortRectifyMap(
                matrix, self._calibration.dist_coeffs, None, matrix, self.image_size, cv2.CV_16SC2
            )
        return cv2.remap(frame, *self._maps, cv2.INTER_LINEAR)
