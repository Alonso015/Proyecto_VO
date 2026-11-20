import cv2
from typing import List
import numpy as np
import imageio.v3 as imageio
import copy
import os
import glob


def load_images(filenames: List) -> List:
    return [imageio.imread(filename) for filename in filenames]


def get_chessboard_points(chessboard_shape, dx, dy):
    rows, cols = chessboard_shape
    points = []
    for i in range(cols):
        for j in range(rows):
            points.append([j * dx, i * dy, 0.0])
    return np.array(points, dtype=np.float32)


imgs_path = [filename for filename in glob.glob(os.path.join("fotos_damero", "*.jpg"))]
imgs = load_images(imgs_path)

corners = [cv2.findChessboardCorners(img, (8, 6)) for img in imgs]

corners_copy = copy.deepcopy(corners)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)

imgs_gray = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in imgs]

corners_refined = [cv2.cornerSubPix(img_grey, corner[1], (5, 5), (-1, -1), criteria) if corner[0] else [] for img_grey, corner in zip(imgs_gray, corners_copy)]

# Filter data and get only those with adequate detections
valid_corners = [corner_refined for corner_refined, corner in zip(corners_refined, corners) if corner[0]]
# Convert list to numpy array with np.float32 elements
valid_corners = np.asarray(valid_corners, dtype=np.float32)

chessboard_points = [get_chessboard_points((8, 6), 30.0, 30.0) for _ in range(len(valid_corners))]

chessboard_points = np.array(chessboard_points, dtype=np.float32)

rms, intrinsics, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(chessboard_points, valid_corners, (imgs[0].shape[1], imgs[0].shape[0]), None, None)

# Obtain extrinsics
extrinsics = list(map(lambda rvec, tvec: np.hstack((cv2.Rodrigues(rvec)[0], tvec)), rvecs, tvecs))

# Print outputs
print("Intrinsics:\n", intrinsics)
print("Distortion coefficients:\n", dist_coeffs)
print("Root mean squared reprojection error:\n", rms)

# Save the parameters so the real-time system can correct the distortion
os.makedirs("data/calibration", exist_ok=True)
np.savez("data/calibration/calibration.npz", intrinsics=intrinsics, dist_coeffs=dist_coeffs, rms=rms,
         image_size=(imgs[0].shape[1], imgs[0].shape[0]))
