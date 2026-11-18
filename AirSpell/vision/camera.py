"""Captura de webcam adaptada de Lab_Project/src/test.py."""

from contextlib import contextmanager
import sys

import cv2


def read_frame(capture):
    success, frame = capture.read()
    if success and frame is not None and frame.size:
        return frame
    return None


@contextmanager
def open_camera(index, size):
    """Cede la captura y su primer fotograma válido. Libera cada backend al salir."""
    backends = (cv2.CAP_DSHOW, cv2.CAP_MSMF) if sys.platform == "win32" else (cv2.CAP_ANY,)
    for backend in backends:
        capture = cv2.VideoCapture()
        try:
            if not capture.open(index, backend):
                continue
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
            # Algunos dispositivos necesitan varias lecturas para arrancar.
            for _ in range(5):
                frame = read_frame(capture)
                if frame is not None:
                    yield capture, frame
                    return
        finally:
            capture.release()
    raise RuntimeError(
        f"No se pudo leer la webcam {index}. Comprueba su conexión, permisos y uso "
        "por otras aplicaciones, o prueba otro índice con --camera."
    )
