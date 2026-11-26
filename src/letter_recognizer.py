"""Letter recognition from a single stroke drawn in the air.

Template matching based on the $1 Unistroke Recognizer
(Wobbrock, Wilson and Li, UIST 2007), without the rotation step,
because the orientation of a letter matters.

    stroke -> smooth -> resample to N points -> scale -> center -> compare with templates
"""

import json
import os
import numpy as np

N_POINTS = 64        # points after resampling
SQUARE_SIZE = 250    # size of the reference square
THRESHOLD = 40       # maximum mean distance to accept a letter


def smooth(points, k=3):
    # Moving average to remove the jitter of the marker detection
    points = np.asarray(points, dtype=np.float32)
    if len(points) <= k:
        return points
    kernel = np.ones(k) / k
    x = np.convolve(points[:, 0], kernel, mode="valid")
    y = np.convolve(points[:, 1], kernel, mode="valid")
    return np.stack([x, y], axis=1)


def resample(points, n=N_POINTS):
    # n points equally spaced along the stroke (independent of the drawing speed)
    segments = np.linalg.norm(np.diff(points, axis=0), axis=1)
    distance = np.concatenate([[0], np.cumsum(segments)])
    targets = np.linspace(0, distance[-1], n)
    x = np.interp(targets, distance, points[:, 0])
    y = np.interp(targets, distance, points[:, 1])
    return np.stack([x, y], axis=1)


def scale_to_square(points, size=SQUARE_SIZE):
    # Fit the bounding box into a size x size square (independent of the letter size)
    width, height = points.max(axis=0) - points.min(axis=0)
    return points * [size / max(width, 1e-6), size / max(height, 1e-6)]


def translate_to_origin(points):
    # Move the centroid to (0, 0) (independent of where the letter was drawn)
    return points - points.mean(axis=0)


def normalize(points):
    points = smooth(points)
    points = resample(points)
    points = scale_to_square(points)
    points = translate_to_origin(points)
    return points


def path_distance(a, b):
    # Mean distance between corresponding points
    return float(np.mean(np.linalg.norm(a - b, axis=1)))


class LetterRecognizer:
    def __init__(self, templates_path="data/templates.json", threshold=THRESHOLD):
        self.templates_path = templates_path
        self.threshold = threshold
        self.templates = {}  # letter -> list of normalized strokes
        if os.path.exists(templates_path):
            with open(templates_path, encoding="utf-8") as f:
                data = json.load(f)
            self.templates = {letter: [np.array(t) for t in strokes] for letter, strokes in data.items()}

    def add_template(self, letter, points):
        self.templates.setdefault(letter, []).append(normalize(points))
        self.save()

    def remove_last_template(self, letter):
        # Delete the last template saved for a letter (to undo a bad stroke)
        if not self.templates.get(letter):
            return False
        self.templates[letter].pop()
        if not self.templates[letter]:
            del self.templates[letter]
        self.save()
        return True

    def save(self):
        os.makedirs(os.path.dirname(self.templates_path), exist_ok=True)
        data = {letter: [t.round(2).tolist() for t in strokes] for letter, strokes in self.templates.items()}
        with open(self.templates_path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    def recognize(self, points):
        """Return (letter, distance). letter is None if no template is close enough."""
        if not self.templates:
            return None, float("inf")
        candidate = normalize(points)
        best_letter, best_distance = None, float("inf")
        for letter, strokes in self.templates.items():
            for template in strokes:
                d = path_distance(candidate, template)
                if d < best_distance:
                    best_letter, best_distance = letter, d
        if best_distance > self.threshold:
            return None, best_distance
        return best_letter, best_distance
