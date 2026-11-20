import cv2
import numpy as np

calibration = np.load("data/calibration/calibration.npz")
intrinsics = calibration["intrinsics"]
dist_coeffs = calibration["dist_coeffs"]
calibration_size = calibration["image_size"]


def get_undistort_maps(dim):
    # The photos were taken at a higher resolution than the video: scale the intrinsics
    scaled_intrinsics = intrinsics.copy()
    scaled_intrinsics[:2] *= dim[0] / calibration_size[0]
    return cv2.initUndistortRectifyMap(scaled_intrinsics, dist_coeffs, np.eye(3), scaled_intrinsics, dim, cv2.CV_16SC2)


def main(camera_index=0, width=1280, height=720):
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"Could not open the camera (index {camera_index}).")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    map1, map2 = get_undistort_maps((width, height))

    window_name = "Undistorted Camera - press 'q' to quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("No frame received (the camera may have been disconnected).")
                break

            undistorted = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)

            cv2.imshow(window_name, undistorted)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main(camera_index=0, width=1280, height=720)
