import time
import cv2
import numpy as np
from undistort import get_undistort_maps
from letter_recognizer import LetterRecognizer
from decoder import SequenceDecoder

# HSV range of the finger marker (find it with src/hsv_ranges.py)
LOWER_MARKER = (31, 32, 140)
UPPER_MARKER = (90, 255, 255)



# Minimum contour area (in pixels) to accept a detection
MIN_AREA = 150

kernel = np.ones((5, 5), np.uint8)

# Letters of the alphabet (keys 1..5 select the letter in template mode)
LETTERS = ["L", "V", "Z", "C", "S"]

# Minimum number of points of a valid stroke
MIN_POINTS = 10

# Password of the security system (sequence of 4 letters)
PASSWORD = ["L", "Z", "V", "C"]


def segment_marker(frame):
    # Color segmentation in HSV (Lab 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_MARKER, UPPER_MARKER)

    # Morphology (Lab 3): erosion removes small noise, dilation fills holes
    mask = cv2.erode(mask, kernel)
    mask = cv2.dilate(mask, kernel, iterations=2)
    return mask


def find_marker(mask):
    # Keep the biggest blob and return its centroid
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < MIN_AREA:
        return None
    M = cv2.moments(contour)
    return int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])


def main(camera_index=0, width=1280, height=720):
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"Could not open the camera (index {camera_index}).")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    map1, map2 = get_undistort_maps((width, height))

    window_name = "Marker tracker - 's' start, 'e' end, 't' templates, 'u' undo, 'r' reset, 'c' clear, 'q' quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.namedWindow("Mask", cv2.WINDOW_NORMAL)

    trajectory = []      # stroke being drawn
    strokes = []         # finished strokes
    recording = False
    prev_time = time.time()

    recognizer = LetterRecognizer("data/templates.json")
    template_mode = False   # True: finished strokes are saved as templates
    template_letter = LETTERS[0]
    message = ""            # result shown on screen
    decoder = SequenceDecoder(PASSWORD)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("No frame received (the camera may have been disconnected).")
                break

            # Undistort and mirror the image
            frame = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
            frame = cv2.flip(frame, 1)

            mask = segment_marker(frame)
            center = find_marker(mask)

            if center is not None:
                # Only store the points while recording
                if recording:
                    trajectory.append(center)
                cv2.circle(frame, center, 10, (0, 0, 255), -1)

            # Draw the finished strokes (blue) and the current one (green)
            for stroke in strokes:
                for p, q in zip(stroke, stroke[1:]):
                    cv2.line(frame, p, q, (255, 0, 0), 4)
            for p, q in zip(trajectory, trajectory[1:]):
                cv2.line(frame, p, q, (0, 255, 0), 4)

            if recording:
                cv2.putText(frame, "REC", (width - 120, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

            # Security system: password state
            decoder.update()
            if not template_mode:
                color = (0, 255, 0) if decoder.unlocked else (0, 0, 255)
                state = "UNLOCKED" if decoder.unlocked else "LOCKED"
                cv2.putText(frame, f"{state}   {decoder.display()}", (20, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
                cv2.putText(frame, decoder.message, (20, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            # Mode and last result
            if template_mode:
                n = len(recognizer.templates.get(template_letter, []))
                info = f"TEMPLATES: letter {template_letter} ({n} saved) - keys 1-{len(LETTERS)} letter, u undo"
            else:
                info = "RECOGNITION"
            cv2.putText(frame, info, (20, height - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            cv2.putText(frame, message, (20, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

            # FPS
            now = time.time()
            fps = 1 / (now - prev_time)
            prev_time = now
            cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

            cv2.imshow(window_name, frame)
            cv2.imshow("Mask", mask)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('s') and not recording:
                # Start a new stroke
                recording = True
                trajectory = []
            if key == ord('e') and recording:
                # End the stroke and keep it
                recording = False
                if len(trajectory) >= MIN_POINTS:
                    strokes.append(trajectory)
                    if template_mode:
                        recognizer.add_template(template_letter, trajectory)
                        message = f"Template saved for {template_letter}"
                    else:
                        letter, distance = recognizer.recognize(trajectory)
                        if letter is None:
                            message = f"Not recognized (distance {distance:.1f})"
                        else:
                            message = f"Letter: {letter} (distance {distance:.1f})"
                        decoder.add(letter)
                        print(decoder.message)
                    print(message)
                else:
                    message = "Stroke too short"
                trajectory = []
            if key == ord('t'):
                template_mode = not template_mode
                message = ""
            if key == ord('u') and template_mode:
                # Undo: remove the last template of the selected letter
                if recognizer.remove_last_template(template_letter):
                    message = f"Last template of {template_letter} removed"
                else:
                    message = f"No templates of {template_letter} to remove"
                print(message)
            if template_mode and ord('1') <= key < ord('1') + len(LETTERS):
                template_letter = LETTERS[key - ord('1')]
            if key == ord('r'):
                # Forget the sequence (and lock again if it was unlocked)
                decoder.lock()
                strokes = []
            if key == ord('c'):
                recording = False
                trajectory = []
                strokes = []
                message = ""
            if key == ord('q') or key == 27:
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main(camera_index=0, width=1280, height=720)
