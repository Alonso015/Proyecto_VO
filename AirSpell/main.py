"""AirSpell, captura y registro de trazos con un marcador de color."""

import argparse
from pathlib import Path
import sys
from time import perf_counter

import cv2
import numpy as np

from calibration.undistort import Calibration, Undistorter
import config
from vision.camera import open_camera, read_frame
from vision.marker import MarkerDetector
from vision.tracker import MarkerTracker
from vision.trajectory import StrokeRecorder

WINDOW = "AirSpell"
MASK_WINDOW = "Mascara HSV"


def draw_overlay(frame, track, recorder, tracker, fps, calibrated, message):
    if track is not None:
        x, y, width, height = track.box
        cv2.rectangle(frame, (x, y), (x + width - 1, y + height - 1), (50, 235, 80), 2)
        cv2.circle(frame, tuple(round(value) for value in track.center), 4, (255, 255, 255), -1)

    color = (40, 190, 255) if recorder.recording else (255, 200, 70)
    if recorder.points:
        points = np.rint(recorder.points).astype(np.int32)
        if len(points) > 1:
            cv2.polylines(frame, [points], False, color, 2, cv2.LINE_AA)
        else:
            cv2.circle(frame, tuple(points[0]), 2, color, -1)

    marker_status = "Buscando marcador"
    if track is not None:
        marker_status = "Marcador detectado"
    elif tracker.recovering:
        marker_status = "Recuperando marcador"
    recording_status = "Grabando" if recorder.recording else "En espera"
    calibration_status = "Calibrada" if calibrated else "Sin calibrar"
    lines = (
        f"{fps:.1f} FPS | {calibration_status}",
        f"{marker_status} | {recording_status}",
        message,
        "Espacio: iniciar/terminar | R: borrar | Q: salir",
    )
    for row, label in enumerate(lines, start=1):
        position = (9, row * 20)
        cv2.putText(frame, label, position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, label, position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def run_live(capture, frame, detector, undistorter, show_mask):
    tracker = MarkerTracker()
    recorder = StrokeRecorder()
    message = "Coloca el marcador a la vista."
    fps = 0.0
    previous_time = None
    mask_shown = False
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    try:
        if show_mask:
            cv2.namedWindow(MASK_WINDOW, cv2.WINDOW_NORMAL)
        while frame is not None:
            if undistorter is not None:
                frame = undistorter.apply(frame)
            view = cv2.flip(frame, 1)
            detections, mask = detector.detect(view)
            track = tracker.update(detections, view.shape)
            if track is not None:
                recorder.add(track.center)
            elif recorder.recording and not tracker.recovering:
                recorder.clear()
                message = "Trazo cancelado: marcador perdido."

            now = perf_counter()
            if previous_time is not None:
                instantaneous = 1 / max(now - previous_time, 1e-6)
                fps = instantaneous if fps == 0 else 0.9 * fps + 0.1 * instantaneous
            previous_time = now
            draw_overlay(view, track, recorder, tracker, fps, undistorter is not None, message)
            cv2.imshow(WINDOW, view)
            if show_mask:
                if not mask_shown or cv2.getWindowProperty(MASK_WINDOW, cv2.WND_PROP_VISIBLE) >= 1:
                    cv2.imshow(MASK_WINDOW, mask)
                    mask_shown = True
                else:
                    show_mask = False
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                return
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                return
            if key == ord(" "):
                if recorder.recording:
                    points = recorder.finish()
                    message = f"Trazo listo: {len(points)} puntos." if points else "Trazo demasiado corto."
                elif recorder.start(track.center if track is not None else None):
                    message = "Dibuja y pulsa Espacio para terminar."
                else:
                    message = "Muestra el marcador antes de empezar."
            elif key in (ord("r"), ord("R")):
                recorder.clear()
                message = "Trazo borrado."
            frame = read_frame(capture)
        raise RuntimeError("La webcam dejó de entregar fotogramas.")
    finally:
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0, help="Índice de webcam (por defecto, 0)")
    parser.add_argument("--size", type=int, nargs=2, metavar=("ANCHO", "ALTO"), help="Resolución solicitada")
    parser.add_argument("--calibration", type=Path, help="Archivo .npz de esta webcam")
    parser.add_argument("--hsv-lower", type=int, nargs=3, metavar=("H", "S", "V"), default=config.HSV_LOWER)
    parser.add_argument("--hsv-upper", type=int, nargs=3, metavar=("H", "S", "V"), default=config.HSV_UPPER)
    parser.add_argument("--min-area", type=float, default=config.MIN_MARKER_AREA, help="Área mínima en px²")
    parser.add_argument("--show-mask", action="store_true", help="Muestra la máscara segmentada")
    parser.add_argument("--check-camera", action="store_true", help="Lee un fotograma y termina sin abrir ventanas")
    args = parser.parse_args()
    if args.camera < 0 or (args.size is not None and min(args.size) <= 0):
        parser.error("El índice de webcam debe ser no negativo y la resolución positiva.")

    try:
        detector = MarkerDetector(args.hsv_lower, args.hsv_upper, args.min_area)
        calibration = Calibration.load(args.calibration) if args.calibration else None
        default_size = calibration.image_size if calibration else config.CAMERA_SIZE
        size = tuple(args.size) if args.size else default_size
        undistorter = Undistorter(calibration) if calibration else None
        with open_camera(args.camera, size) as (capture, frame):
            if args.check_camera:
                if undistorter is not None:
                    undistorter.apply(frame)
                height, width = frame.shape[:2]
                print(f"Webcam {args.camera}: {width}x{height} ({capture.getBackendName()})")
                return 0
            run_live(capture, frame, detector, undistorter, args.show_mask)
    except KeyboardInterrupt:
        return 0
    except (ValueError, RuntimeError, cv2.error) as error:
        print(f"AirSpell: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
