"""Corrección de distorsión en tiempo real con mapas precalculados.

Uso como módulo:
    from undistort import Undistorter
    und = Undistorter("data/calibration/calibration.npz")
    frame = und(frame)

Uso como demo (vista original | corregida, con FPS):
    python src/undistort.py --camera 1
"""

import argparse
import time

import cv2
import numpy as np


class Undistorter:
    def __init__(self, calib_path="data/calibration/calibration.npz", alpha=0.0):
        data = np.load(calib_path)
        self.K = data["K"]
        self.dist = data["dist"]
        self.calib_size = tuple(int(v) for v in data["image_size"])
        self.alpha = alpha
        self._maps = {}

    def _get_maps(self, size):
        if size not in self._maps:
            K = self.K.copy()
            if size != self.calib_size:
                # Mismo sensor a otra resolución con igual relación de aspecto: escalar intrínsecos
                sx, sy = size[0] / self.calib_size[0], size[1] / self.calib_size[1]
                if abs(sx - sy) > 1e-3:
                    raise ValueError(f"Relación de aspecto {size} distinta de la calibrada {self.calib_size}")
                K[0, :] *= sx
                K[1, :] *= sy
            new_K, _ = cv2.getOptimalNewCameraMatrix(K, self.dist, size, self.alpha)
            self._maps[size] = cv2.initUndistortRectifyMap(K, self.dist, None, new_K, size, cv2.CV_16SC2)
        return self._maps[size]

    def __call__(self, frame):
        size = (frame.shape[1], frame.shape[0])
        map1, map2 = self._get_maps(size)
        return cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--calib", default="data/calibration/calibration.npz")
    parser.add_argument("--size", type=int, nargs=2, default=(1280, 720))
    args = parser.parse_args()

    und = Undistorter(args.calib)
    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.size[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.size[1])
    if not cap.isOpened():
        raise SystemExit(f"No se pudo abrir la cámara {args.camera}")

    win = "Original | Corregida - 'q' para salir"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    prev, fps = time.perf_counter(), 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        fixed = und(frame)
        now = time.perf_counter()
        fps = 0.9 * fps + 0.1 / max(now - prev, 1e-6)
        prev = now
        view = np.hstack([frame, fixed])
        cv2.putText(view, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        cv2.imshow(win, view)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
