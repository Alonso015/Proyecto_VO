"""Calibración offline con tablero, adaptada de la práctica 2."""

import argparse
from math import isfinite
from pathlib import Path
import sys

import cv2
import numpy as np

from calibration.undistort import Calibration

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
MIN_VIEWS = 5


def chessboard_points(pattern, square_size):
    columns, rows = pattern
    return np.array(
        [(column * square_size, row * square_size, 0)
         for row in range(rows) for column in range(columns)],
        dtype=np.float32,
    )


def calibrate(images, pattern, square_size):
    if min(pattern) < 3 or not isfinite(square_size) or square_size <= 0:
        raise ValueError("Indica al menos 3 esquinas por eje y un tamaño de casilla positivo.")
    paths = sorted(
        path for path in images.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    board = chessboard_points(pattern, square_size)
    corners_per_view = []
    image_size = None
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)

    for path in paths:
        # imdecode con Path admite rutas Unicode en Windows.
        contents = path.read_bytes()
        gray = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_GRAYSCALE) if contents else None
        if gray is None:
            print(f"Imagen ilegible: {path.name}", file=sys.stderr)
            continue
        found, corners = cv2.findChessboardCorners(gray, pattern, cv2.CALIB_CB_FAST_CHECK)
        if not found:
            continue
        size = (gray.shape[1], gray.shape[0])
        if image_size is not None and size != image_size:
            raise ValueError(f"Las vistas útiles deben tener la misma resolución ({path.name}).")
        image_size = size
        corners_per_view.append(cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria))

    used = len(corners_per_view)
    if used < MIN_VIEWS:
        raise ValueError(f"Tablero detectado en {used} de {len(paths)} imágenes. Se necesitan {MIN_VIEWS} vistas como mínimo.")
    rms, matrix, distortion, _, _ = cv2.calibrateCamera(
        [board] * used, corners_per_view, image_size, None, None
    )
    if not np.isfinite(rms) or not np.isfinite(matrix).all() or not np.isfinite(distortion).all():
        raise ValueError("La calibración no convergió. Prueba con vistas más variadas.")
    return Calibration(matrix, distortion, image_size, rms), used, len(paths)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True, type=Path, help="Carpeta de fotos del tablero")
    parser.add_argument("--pattern", type=int, nargs=2, metavar=("COLUMNAS", "FILAS"), default=(8, 6),
                        help="Número de esquinas interiores (por defecto, 8 6)")
    parser.add_argument("--square-size", required=True, type=float, help="Lado de una casilla en mm")
    parser.add_argument("--output", type=Path, default=Path("assets/calibration/webcam.npz"))
    args = parser.parse_args()
    try:
        calibration, used, total = calibrate(args.images, tuple(args.pattern), args.square_size)
        calibration.save(args.output)
    except (OSError, ValueError, cv2.error) as error:
        print(f"Error de calibración: {error}", file=sys.stderr)
        return 1
    width, height = calibration.image_size
    print(f"{used}/{total} vistas a {width}x{height}. RMS: {calibration.rms:.4f} px")
    print("Matriz intrínseca:")
    print(calibration.camera_matrix)
    print("Distorsión:", calibration.dist_coeffs.ravel())
    print(f"Guardado en {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
