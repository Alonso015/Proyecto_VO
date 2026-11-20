"""Calibración offline de la cámara con un tablero de ajedrez.

Basado en la Práctica 2 (calibración de cámara), con estas diferencias:
  * Se calibra con las esquinas refinadas a nivel subpíxel (cornerSubPix).
  * Las imágenes se redimensionan a la resolución de trabajo del sistema
    (por defecto 1280x720, la misma que usa la captura en tiempo real).
  * Se guardan los parámetros (K, distorsión, RMS...) para que el sistema
    en tiempo real los cargue y corrija la distorsión con mapas precalculados.

Uso (desde la raíz del repositorio):
    python src/calibration.py --images fotos_damero --square-mm 30
"""

import argparse
import glob
import json
import os

import cv2
import numpy as np

SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def get_chessboard_points(pattern_size, square_size):
    """Coordenadas 3D (Z=0) de las esquinas interiores del tablero, en mm."""
    cols, rows = pattern_size
    points = np.zeros((cols * rows, 3), np.float32)
    points[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size
    return points


def load_images(folder, target_size):
    paths = sorted(glob.glob(os.path.join(folder, "*.jpg")) + glob.glob(os.path.join(folder, "*.png")))
    images = []
    for path in paths:
        img = cv2.imread(path)
        if img is None:
            print(f"[!] No se pudo leer {path}")
            continue
        if target_size and (img.shape[1], img.shape[0]) != tuple(target_size):
            img = cv2.resize(img, tuple(target_size), interpolation=cv2.INTER_AREA)
        images.append((path, img))
    return images


def detect_corners(images, pattern_size):
    """Detecta y refina las esquinas. Devuelve [(path, img, esquinas_refinadas)]."""
    detections = []
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    for path, img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
        if not found:
            print(f"[!] Tablero no encontrado en {os.path.basename(path)}")
            continue
        refined = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), SUBPIX_CRITERIA)
        detections.append((path, img, refined))
    return detections


def per_view_errors(obj_points, img_points, rvecs, tvecs, K, dist):
    errors = []
    for objp, imgp, r, t in zip(obj_points, img_points, rvecs, tvecs):
        proj, _ = cv2.projectPoints(objp, r, t, K, dist)
        errors.append(float(np.sqrt(np.mean(np.sum((proj - imgp) ** 2, axis=2)))))
    return errors


def rms_vs_num_images(obj_points, img_points, image_size, trials=10, seed=0):
    """RMS medio (y desviación) usando n imágenes elegidas al azar, para n = 3..N."""
    rng = np.random.default_rng(seed)
    n_total = len(img_points)
    results = []
    for n in range(3, n_total + 1):
        rms_list = []
        for _ in range(trials if n < n_total else 1):
            idx = rng.choice(n_total, n, replace=False)
            rms, *_ = cv2.calibrateCamera([obj_points[i] for i in idx], [img_points[i] for i in idx],
                                          image_size, None, None)
            rms_list.append(rms)
        results.append((n, float(np.mean(rms_list)), float(np.std(rms_list))))
    return results


def save_figures(detections, pattern_size, K, dist, image_size, rms_curve, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # 1. Mosaico con las esquinas detectadas
    thumbs = []
    for _, img, corners in detections:
        vis = cv2.drawChessboardCorners(img.copy(), pattern_size, corners, True)
        thumbs.append(cv2.resize(vis, (image_size[0] // 4, image_size[1] // 4)))
    cols = 5
    while len(thumbs) % cols:
        thumbs.append(np.zeros_like(thumbs[0]))
    mosaic = np.vstack([np.hstack(thumbs[i:i + cols]) for i in range(0, len(thumbs), cols)])
    cv2.imwrite(os.path.join(out_dir, "detecciones.jpg"), mosaic)

    # 2. Original vs. corregida (primera imagen)
    img = detections[0][1]
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, dist, image_size, 0)
    map1, map2 = cv2.initUndistortRectifyMap(K, dist, None, new_K, image_size, cv2.CV_16SC2)
    und = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)
    cv2.putText(img := img.copy(), "Original", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
    cv2.putText(und, "Corregida", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    cv2.imwrite(os.path.join(out_dir, "original_vs_corregida.jpg"), np.hstack([img, und]))

    # 3. RMS frente a número de imágenes
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        n, mean, std = map(np.array, zip(*rms_curve))
        plt.figure(figsize=(6, 4))
        plt.errorbar(n, mean, yerr=std, marker="o", capsize=3)
        plt.xlabel("Número de imágenes")
        plt.ylabel("RMS de reproyección [px]")
        plt.title("Error de calibración frente al número de imágenes")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "rms_vs_imagenes.png"), dpi=150)
        plt.close()
    except ImportError:
        print("[!] matplotlib no disponible: no se genera la gráfica RMS vs nº de imágenes")


def main():
    parser = argparse.ArgumentParser(description="Calibración offline de la cámara")
    parser.add_argument("--images", default="fotos_damero", help="Carpeta con las fotos del tablero")
    parser.add_argument("--pattern", type=int, nargs=2, default=(8, 6), metavar=("COLS", "ROWS"),
                        help="Esquinas interiores del tablero")
    parser.add_argument("--square-mm", type=float, default=30.0, help="Lado de la casilla en mm")
    parser.add_argument("--size", type=int, nargs=2, default=(1280, 720), metavar=("W", "H"),
                        help="Resolución de trabajo (la de la captura en tiempo real)")
    parser.add_argument("--out", default="data/calibration", help="Carpeta de salida de los parámetros")
    parser.add_argument("--figures", default="results/calibration", help="Carpeta de salida de las figuras")
    args = parser.parse_args()

    pattern_size = tuple(args.pattern)
    image_size = tuple(args.size)

    images = load_images(args.images, image_size)
    if not images:
        raise SystemExit(f"No hay imágenes en {args.images}")
    detections = detect_corners(images, pattern_size)
    print(f"Tablero detectado en {len(detections)}/{len(images)} imágenes")
    if len(detections) < 3:
        raise SystemExit("Se necesitan al menos 3 detecciones válidas")

    objp = get_chessboard_points(pattern_size, args.square_mm)
    obj_points = [objp] * len(detections)
    img_points = [d[2] for d in detections]

    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, image_size, None, None)
    errors = per_view_errors(obj_points, img_points, rvecs, tvecs, K, dist)
    new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist, image_size, 0)

    np.set_printoptions(precision=4, suppress=True)
    print("\nMatriz intrínseca K:\n", K)
    print("Coeficientes de distorsión [k1 k2 p1 p2 k3]:\n", dist.ravel())
    print(f"RMS de reproyección: {rms:.4f} px")
    print("Error por imagen:")
    for (path, _, _), e in zip(detections, errors):
        print(f"  {os.path.basename(path)}: {e:.4f} px")

    os.makedirs(args.out, exist_ok=True)
    np.savez(os.path.join(args.out, "calibration.npz"), K=K, dist=dist, new_K=new_K, roi=np.array(roi),
             image_size=np.array(image_size), rms=rms)
    with open(os.path.join(args.out, "calibration.json"), "w", encoding="utf-8") as f:
        json.dump({
            "image_size": list(image_size),
            "pattern_size": list(pattern_size),
            "square_mm": args.square_mm,
            "num_images": len(detections),
            "rms_px": rms,
            "K": K.tolist(),
            "dist": dist.ravel().tolist(),
            "new_K": new_K.tolist(),
            "roi": list(roi),
            "per_view_error_px": {os.path.basename(p): e for (p, _, _), e in zip(detections, errors)},
        }, f, indent=2)
    print(f"\nParámetros guardados en {args.out}/calibration.npz y calibration.json")

    curve = rms_vs_num_images(obj_points, img_points, image_size)
    save_figures(detections, pattern_size, K, dist, image_size, curve, args.figures)
    print(f"Figuras guardadas en {args.figures}/")


if __name__ == "__main__":
    main()
