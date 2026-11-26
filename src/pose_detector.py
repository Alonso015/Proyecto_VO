"""Phase 2: body poses with a star skeleton (I, Y, T, L, X), without color.

  * Silhouette: background subtraction (MOG2). The background is learned with
    nobody in the frame (key 'b') and then frozen, so only the person is
    foreground. The camera exposure is fixed meanwhile (camera_settings.py).
  * Star skeleton (Fujiyoshi and Lipton, 1998): the extremities (head, hands,
    feet) are the local maxima of the distance from the silhouette centroid
    to its contour. A stick goes from the centroid to each extremity.
  * Pose: direction of each stick and distances between the extremities
    (hand-hand, foot-foot), normalized by the head stick of the neutral pose.
  * A pose is confirmed when it is held for HOLD_FRAMES frames, and the arms
    must go back down (neutral pose) before the next one.
"""

import math
from collections import deque
import cv2
import numpy as np

# Processing scale (half resolution is faster)
SCALE = 0.5

# Frames used to learn the empty background
BG_FRAMES = 30

# MOG2 threshold (squared Mahalanobis distance): higher = less noise (default 16)
VAR_THRESHOLD = 25

# Minimum silhouette area (pixels, at processing scale)
MIN_BODY_AREA = 3000

# A pose is confirmed when it is the pose of at least HOLD_VOTES of the last
# HOLD_FRAMES frames (a vote, so one noisy frame does not reset it); the arms
# count as down again with NEUTRAL_VOTES of the last HOLD_FRAMES frames
HOLD_FRAMES = 10
HOLD_VOTES = 7
NEUTRAL_VOTES = 5

# Thresholds relative to the head stick length in the neutral pose
RAISED = 1.2      # an arm is raised above the head if its stick is longer than this
ARM_MIN = 1.0      # minimum stick length of an outstretched arm
ARMS_OPEN = 1.0   # minimum hand-hand distance for Y and T
LEGS_OPEN = 0.6    # minimum foot-foot distance for X

kernel = np.ones((5, 5), np.uint8)
vertical_kernel = np.ones((31, 5), np.uint8)


def segment_body(bg, small):
    # Foreground with the frozen background (255 = foreground, 127 = shadow)
    small = cv2.GaussianBlur(small, (5, 5), 0)
    fg = bg.apply(small, learningRate=0)
    _, mask = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)

    # Morphology (Lab 3): opening removes noise, closing fills holes
    mask = cv2.erode(mask, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=4)
    mask = cv2.erode(mask, kernel, iterations=2)

    # Vertical closing: joins the legs to the torso when the waist is missed
    mask = cv2.dilate(mask, vertical_kernel)
    mask = cv2.erode(mask, vertical_kernel)

    # Keep only the biggest blob (the person), filled
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    silhouette = np.zeros_like(mask)
    if not contours:
        return silhouette, None
    body = max(contours, key=cv2.contourArea)
    if cv2.contourArea(body) < MIN_BODY_AREA:
        return silhouette, None
    cv2.drawContours(silhouette, [body], -1, 255, -1)
    return silhouette, body


def find_extremities(contour, centroid, shape):
    # Star skeleton: local maxima of the centroid-contour distance
    points = contour[:, 0, :]
    n = len(points)
    d = np.linalg.norm(points - centroid, axis=1)

    # Smooth the distance signal (it is circular, so pad it with its own ends)
    k = max(n // 100, 1)
    d = np.convolve(np.concatenate([d[-k:], d, d[:k]]), np.ones(2 * k + 1) / (2 * k + 1), mode="valid")

    # Points on the bottom border are not extremities (legs cut by the frame);
    # the top and side borders are kept (hands raised out of the frame are still arms)
    h, w = shape
    d[points[:, 1] >= h - 2] = 0
    d = d.astype(np.float32)

    # Local maximum = equal to the maximum of its neighbourhood (1D dilation)
    win = max(n // 16, 1)
    padded = np.concatenate([d[-win:], d, d[:win]]).reshape(1, -1)
    local_max = cv2.dilate(padded, np.ones((1, 2 * win + 1), np.uint8))[0, win:-win]
    peaks = np.where((d >= local_max) & (d > d.mean()))[0]

    # A flat maximum gives several consecutive peaks: keep one
    extremities = []
    last = -2 * win
    for i in peaks:
        if i - last > win:
            extremities.append(i)
        last = i
    if len(extremities) > 1 and extremities[0] + n - extremities[-1] <= win:
        extremities.pop()
    return [(int(points[i][0]), int(points[i][1])) for i in extremities]


def make_stick(centroid, point):
    dx = point[0] - centroid[0]
    dy = centroid[1] - point[1]
    # Elevation: 90 = straight up, 0 = horizontal, -90 = straight down (side does not matter)
    elevation = math.degrees(math.atan2(dy, abs(dx)))
    if elevation > 65:
        direction = "UP"          # head or an arm above the head
    elif elevation > 25:
        direction = "DIAG_UP"     # arm up and open
    elif elevation > -25:
        direction = "SIDE"        # horizontal arm
    elif elevation > -50:
        direction = None          # arm down and open (not used)
    else:
        direction = "LEG"
    return {"point": point, "length": math.hypot(dx, dy), "dx": dx, "direction": direction}


def pair(sticks):
    # Longest stick on the left and on the right, or None if a side is missing
    left = [s for s in sticks if s["dx"] < 0]
    right = [s for s in sticks if s["dx"] >= 0]
    if not left or not right:
        return None
    return max(left, key=lambda s: s["length"]), max(right, key=lambda s: s["length"])


def classify_pose(sticks, ref):
    """Return (pose, hands pair, feet pair, hand distance, foot distance)."""
    groups = {"UP": [], "DIAG_UP": [], "SIDE": [], "LEG": []}
    for s in sticks:
        if s["direction"] in groups:
            groups[s["direction"]].append(s)

    # Scale of the body: head stick of the neutral pose (or the current top stick
    # until it is known: the user starts with the arms down)
    top = max(groups["UP"], key=lambda s: s["length"]) if groups["UP"] else None
    scale = ref or (top["length"] if top else None)

    # An outstretched arm is longer than the head stick; shorter sticks are the
    # shoulders or the hands hanging at the hips, not arms
    for direction in ("DIAG_UP", "SIDE"):
        groups[direction] = [s for s in groups[direction] if scale and s["length"] >= ARM_MIN * scale]
        for s in sticks:
            if s["direction"] == direction and s not in groups[direction]:
                s["direction"] = None

    diag = pair(groups["DIAG_UP"])
    side = pair(groups["SIDE"])
    hands = diag or side
    feet = pair(groups["LEG"])

    # Distances between extremities, relative to the head stick of the neutral pose
    hands_dist = math.dist(hands[0]["point"], hands[1]["point"]) / ref if hands and ref else None
    feet_dist = math.dist(feet[0]["point"], feet[1]["point"]) / ref if feet and ref else None

    # Raised arm: an upward stick (vertical or a bit open) clearly longer than the head
    upward = groups["UP"] + groups["DIAG_UP"]
    # (or cut by the top border: with the arms down the whole body fits in the frame)
    raised = ref is not None and any(s["length"] > RAISED * ref or s["point"][1] <= 2 for s in upward)
    arms_out = groups["SIDE"] or groups["DIAG_UP"]

    pose = None
    if diag and hands_dist and hands_dist > ARMS_OPEN:
        pose = "X" if feet_dist and feet_dist > LEGS_OPEN else "Y"
    elif side and hands_dist and hands_dist > ARMS_OPEN:
        pose = "T"
    elif raised and len(groups["SIDE"]) == 1:
        pose = "L"
    elif raised and not groups["SIDE"]:
        pose = "I"         # arms up, vertical or not open enough for a Y
    elif not raised and not arms_out:
        pose = "NEUTRAL"
    return pose, hands, feet, hands_dist, feet_dist


def full(p):
    # Point at processing scale -> point in the full frame
    return int(p[0] / SCALE), int(p[1] / SCALE)


class PoseDetector:
    def __init__(self):
        self.bg = None            # background model (None until 'b' is pressed)
        self.learn_count = 0      # frames left to learn the background
        self.mask = None          # last silhouette (to show it)
        self.ref = None           # head stick length in the neutral pose (scale of the body)
        self.history = deque(maxlen=HOLD_FRAMES)   # poses of the last frames
        self.candidate = None     # most voted pose of the last frames
        self.armed = True         # False until the arms go back to NEUTRAL
        self.last_pose = None     # last confirmed pose

    def learn_background(self):
        # Learn a new background (nobody has to be in the frame)
        self.bg = cv2.createBackgroundSubtractorMOG2(varThreshold=VAR_THRESHOLD, detectShadows=True)
        self.learn_count = BG_FRAMES
        self.ref = None

    def process(self, frame):
        """Detect the pose in this frame, draw it and return the confirmed pose (or None)."""
        small = cv2.resize(frame, None, fx=SCALE, fy=SCALE)

        if self.bg is None or self.learn_count > 0:
            self.mask = np.zeros(small.shape[:2], np.uint8)
            if self.bg is None:
                text = "Press 'b' with nobody in the frame"
            else:
                self.bg.apply(cv2.GaussianBlur(small, (5, 5), 0), learningRate=-1)
                self.learn_count -= 1
                text = f"Learning background... {self.learn_count}"
            cv2.putText(frame, text, (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            return None

        self.mask, body = segment_body(self.bg, small)

        pose = None
        if body is not None:
            M = cv2.moments(body)
            centroid = np.array([M["m10"] / M["m00"], M["m01"] / M["m00"]])
            sticks = [make_stick(centroid, p) for p in find_extremities(body, centroid, small.shape[:2])]
            pose, hands, feet, hands_dist, feet_dist = classify_pose(sticks, self.ref)

            # Scale of the body: head stick in the neutral pose (smoothed), only measured
            # with the whole body inside the frame (not while walking in)
            x, y, w, h = cv2.boundingRect(body)
            inside = x > 2 and y > 2 and x + w < small.shape[1] - 2
            ups = [s["length"] for s in sticks if s["direction"] == "UP"]
            head = max(ups) if ups else None
            upward = [s["length"] for s in sticks if s["direction"] in ("UP", "DIAG_UP")]
            if pose == "NEUTRAL" and inside and head:
                self.ref = head if self.ref is None else 0.9 * self.ref + 0.1 * head

            # Debug: head stick relative to the reference (> RAISED means arms above the head)
            if upward and self.ref:
                cv2.putText(frame, f"Top stick: {max(upward) / self.ref:.2f} x ref (raised > {RAISED})", (20, 345),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Star skeleton: a stick from the centroid to each extremity
            c = full(centroid)
            for s in sticks:
                # Legs magenta, arms/head cyan, ignored sticks (shoulders, hands down) gray
                color = {"LEG": (255, 0, 255), None: (128, 128, 128)}.get(s["direction"], (255, 255, 0))
                p = full(s["point"])
                cv2.line(frame, c, p, color, 3)
                cv2.circle(frame, p, 10, (0, 0, 255), -1)
                cv2.putText(frame, s["direction"] or "-", (p[0] + 12, p[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.circle(frame, c, 10, (0, 255, 0), -1)

            # Distances between the hands and between the feet
            for ends, dist in ((hands, hands_dist), (feet, feet_dist)):
                if ends and dist:
                    p, q = full(ends[0]["point"]), full(ends[1]["point"])
                    cv2.line(frame, p, q, (0, 255, 255), 2)
                    mid = ((p[0] + q[0]) // 2, (p[1] + q[1]) // 2 - 10)
                    cv2.putText(frame, f"{dist:.2f}", mid, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

            # Tracker: bounding box of the body
            cv2.rectangle(frame, full((x, y)), full((x + w, y + h)), (0, 255, 0), 2)

        # Vote over the last HOLD_FRAMES frames
        self.history.append(pose)
        votes = [p for p in self.history if p is not None]
        self.candidate = max(set(votes), key=votes.count) if votes else None

        confirmed = None
        if self.history.count("NEUTRAL") >= NEUTRAL_VOTES:
            self.armed = True
        elif self.candidate and self.armed and self.history.count(self.candidate) >= HOLD_VOTES:
            confirmed = self.candidate
            self.last_pose = self.candidate
            self.armed = False
            self.history.clear()

        cv2.putText(frame, f"Pose: {pose or '-'}", (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
        if self.last_pose:
            cv2.putText(frame, f"Spell: {self.last_pose}", (20, 225), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        if not self.armed:
            cv2.putText(frame, "Arms down to cast again", (20, 265), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        if self.ref is None:
            cv2.putText(frame, "Stand with arms down", (20, 305), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        return confirmed
