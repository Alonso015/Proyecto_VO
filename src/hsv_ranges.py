import cv2
import numpy as np
from undistort import get_undistort_maps


def nothing(x):
    pass


def get_hsv_color_ranges(image: np.array):

    # Create a window (resizable, so the whole frame is scaled into it)
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow("image", 960, 800)

    # Create trackbars for color change
    cv2.createTrackbar('HMin', 'image', 0, 255, nothing)
    cv2.createTrackbar('SMin', 'image', 0, 255, nothing)
    cv2.createTrackbar('VMin', 'image', 0, 255, nothing)
    cv2.createTrackbar('HMax', 'image', 0, 255, nothing)
    cv2.createTrackbar('SMax', 'image', 0, 255, nothing)
    cv2.createTrackbar('VMax', 'image', 0, 255, nothing)

    # Set default value for MAX HSV trackbars.
    cv2.setTrackbarPos('HMax', 'image', 255)
    cv2.setTrackbarPos('SMax', 'image', 255)
    cv2.setTrackbarPos('VMax', 'image', 255)

    # Initialize to check if HSV min/max value changes
    hMin = sMin = vMin = hMax = sMax = vMax = 0
    phMin = psMin = pvMin = phMax = psMax = pvMax = 0

    output = image
    wait_time = 33

    while(1):

        if cv2.getWindowProperty('image', cv2.WND_PROP_VISIBLE) < 1:
            break
        # get current positions of all trackbars
        hMin = cv2.getTrackbarPos('HMin','image')
        sMin = cv2.getTrackbarPos('SMin','image')
        vMin = cv2.getTrackbarPos('VMin','image')

        hMax = cv2.getTrackbarPos('HMax','image')
        sMax = cv2.getTrackbarPos('SMax','image')
        vMax = cv2.getTrackbarPos('VMax','image')

        # Set minimum and max HSV values to display
        lower = np.array([hMin, sMin, vMin])
        upper = np.array([hMax, sMax, vMax])

        # Create HSV Image and threshold into a range.
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        output = cv2.bitwise_and(image,image, mask= mask)

        # Print if there is a change in HSV value
        if( (phMin != hMin) | (psMin != sMin) | (pvMin != vMin) | (phMax != hMax) | (psMax != sMax) | (pvMax != vMax) ):
            print("(hMin = %d , sMin = %d, vMin = %d), (hMax = %d , sMax = %d, vMax = %d)" % (hMin , sMin , vMin, hMax, sMax , vMax))
            phMin = hMin
            psMin = sMin
            pvMin = vMin
            phMax = hMax
            psMax = sMax
            pvMax = vMax

        # Display output image
        cv2.imshow('image',output)

        # Wait longer to prevent freeze for videos.
        if cv2.waitKey(wait_time) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


def main(camera_index=0, width=1280, height=720):
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"Could not open the camera (index {camera_index}).")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    map1, map2 = get_undistort_maps((width, height))

    window_name = "Press SPACE to capture a frame - 'q' to quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    frame = None
    while True:
        ret, frame = cap.read()
        if not ret:
            print("No frame received (the camera may have been disconnected).")
            break

        frame = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            break
        if key == ord('q') or key == 27:
            frame = None
            break

    cap.release()
    cv2.destroyAllWindows()

    # Tune the HSV range of the marker on the captured frame
    if frame is not None:
        get_hsv_color_ranges(frame)


if __name__ == "__main__":
    main(camera_index=0, width=1280, height=720)
