"""Fixed camera settings for the background subtraction of phase 2.

The webcam raises its exposure when a face appears (automatic exposure and
backlight compensation), which changes the whole image. While phase 2 runs:
  * exposure fixed at -5 (2^-5 s ~ 1/32 s: longer exposures drop below 30 FPS),
  * backlight compensation and automatic white balance off,
  * the image is darker, so it is multiplied by a fixed gain (computed once)
    to get the brightness of the automatic mode. A gain keeps the contrast.
The original settings are restored with unlock().
"""

import cv2

EXPOSURE = -5

SETTINGS = [cv2.CAP_PROP_BACKLIGHT, cv2.CAP_PROP_AUTO_WB]


def mean_brightness(cap, frames=5):
    # Mean gray level after letting the camera settle a few frames
    for _ in range(frames + 5):
        ret, frame = cap.read()
    return float(frame.mean()) if ret else 0.0


class CameraLock:
    def __init__(self, cap):
        self.cap = cap
        self.saved = None     # original settings (None while unlocked)
        self.gain = 1.0

    def lock(self):
        if self.saved is not None:
            return
        self.saved = {p: self.cap.get(p) for p in SETTINGS}
        target = mean_brightness(self.cap, 20)   # brightness of the automatic mode

        self.cap.set(cv2.CAP_PROP_BACKLIGHT, 0)
        self.cap.set(cv2.CAP_PROP_AUTO_WB, 0)
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)   # manual
        self.cap.set(cv2.CAP_PROP_EXPOSURE, EXPOSURE)

        self.gain = target / max(mean_brightness(self.cap, 10), 1.0)
        print(f"Camera locked: exposure {EXPOSURE}, gain {self.gain:.2f}")

    def apply(self, frame):
        # Fixed digital gain while locked
        if self.saved is None:
            return frame
        return cv2.convertScaleAbs(frame, alpha=self.gain)

    def unlock(self):
        if self.saved is None:
            return
        for p, value in self.saved.items():
            self.cap.set(p, value)
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)   # automatic
        self.saved = None
        self.gain = 1.0
        print("Camera settings restored")
