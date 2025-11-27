import cv2
import numpy as np
import os

# === Config ===
VIDEO_PATH = '/Users/apple/Desktop/FYP/My Eye Tracking Model/your_video.mp4'
OUTPUT_PATH = '/Users/apple/Desktop/FYP/My Eye Tracking Model/scanpath.png'
MAX_FRAMES = 300

# === Load pre-trained classifiers ===
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# === Open the video ===
cap = cv2.VideoCapture(VIDEO_PATH)
eye_centers = []
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret or frame_count >= MAX_FRAMES:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)
    print(f"[Frame {frame_count}] Faces found: {len(faces)}")

    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(face_roi)
        print(f"[Frame {frame_count}] Eyes found: {len(eyes)}")
        for (ex, ey, ew, eh) in eyes:
            cx = x + ex + ew // 2
            cy = y + ey + eh // 2
            eye_centers.append((cx, cy))
            break  # one eye only
        break  # one face only

    frame_count += 1

cap.release()

# === Create the scanpath image ===
if eye_centers:
    canvas_size = 1000
    scanpath = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)  # black background

    xs = [pt[0] for pt in eye_centers]
    ys = [pt[1] for pt in eye_centers]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    scale_x = (canvas_size - 100) / (max_x - min_x + 1)
    scale_y = (canvas_size - 100) / (max_y - min_y + 1)

    norm_points = [(
        int((x - min_x) * scale_x + 50),
        int((y - min_y) * scale_y + 50)
    ) for x, y in eye_centers]

    total = len(norm_points)
    third = total // 3

    for i in range(1, total):
        # Using the same colors as the original dataset: early (blue), middle (yellow), late (red)
        if i < third:
            color = (255, 0, 0)  # blue in BGR
        elif i < 2 * third:
            color = (0, 255, 255)  # yellow in BGR
        else:
            color = (0, 0, 255)  # red in BGR

        cv2.line(scanpath, norm_points[i - 1], norm_points[i], color, 1)
        cv2.circle(scanpath, norm_points[i], 2, color, -1)

    cv2.imwrite(OUTPUT_PATH, scanpath)
    print(f"[✔] Scanpath image saved as {OUTPUT_PATH}")
else:
    print("[✖] No eye data found.")