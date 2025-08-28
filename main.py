import cv2
import time
import numpy as np
from deepface import DeepFace
import os

# -------- Settings ----------
AUTHORIZED_IMAGE = r"D:\Vision_Hazard_Detector_Starter\vision_Hazard\authorized_me.jpg"  # your face image
ALERT_COOLDOWN_SECS = 2.0      # avoid continuous beeping
TOLERANCE = 0.05               # extra flexibility on threshold
TEMP_FACE = "temp_face.jpg"    # static temp file for cropped face
# ----------------------------

# --- Simple cross-platform beep helpers ---
def beep():
    try:
        import winsound
        winsound.Beep(1000, 250)  # Windows
    except Exception:
        print('\a', end='', flush=True)  # Fallback for Linux/Mac

# --- Mouse interaction to draw restricted rectangle ---
drawing = False
rx1 = ry1 = rx2 = ry2 = -1
restricted_rect = None  # (x1, y1, x2, y2)

def on_mouse(event, x, y, flags, param):
    global drawing, rx1, ry1, rx2, ry2, restricted_rect
    if event == cv2.EVENT_RBUTTONDOWN:
        drawing = True
        rx1, ry1 = x, y
        rx2, ry2 = x, y
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        rx2, ry2 = x, y
    elif event == cv2.EVENT_RBUTTONUP:
        drawing = False
        rx2, ry2 = x, y
        x1, x2 = sorted([rx1, rx2])
        y1, y2 = sorted([ry1, ry2])
        restricted_rect = (x1, y1, x2, y2)

# --- Utility: check if point is inside restricted rectangle ---
def point_in_rect(pt, rect):
    if rect is None:
        return False
    (x, y) = pt
    (x1, y1, x2, y2) = rect
    return x1 <= x <= x2 and y1 <= y <= y2

# --- Video loop ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

cv2.namedWindow("Restricted Area Monitor")
cv2.setMouseCallback("Restricted Area Monitor", on_mouse)

last_alert_time = 0.0
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()

        # Draw restricted rectangle
        if restricted_rect:
            x1, y1, x2, y2 = restricted_rect
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 165, 255), 2)
            cv2.putText(display, "Restricted Zone", (x1, max(0, y1-10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

        if drawing:
            x1, y1 = min(rx1, rx2), min(ry1, ry2)
            x2, y2 = max(rx1, rx2), max(ry1, ry2)
            cv2.rectangle(display, (x1, y1), (x2, y2), (50, 200, 255), 1)

        # Face detection with Haar cascade
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_crop = frame[y:y+h, x:x+w]
            cx, cy = x + w//2, y + h//2

            # Save cropped face to project folder instead of temp
            cv2.imwrite(TEMP_FACE, face_crop)

            try:
                result = DeepFace.verify(
                    img1_path=AUTHORIZED_IMAGE,
                    img2_path=TEMP_FACE,
                    model_name="Facenet",
                    enforce_detection=False
                )

                distance = result["distance"]
                threshold = result["threshold"]

                # Debug info in terminal
                print(f"Distance={distance:.4f}, Threshold={threshold:.4f}")

                # Apply tolerance
                verified = distance <= (threshold + TOLERANCE)

            except Exception as e:
                print("Verification error:", e)
                verified = False

            if verified:
                color = (0, 200, 0)  # green
                label = "AUTHORIZED"
            else:
                color = (0, 0, 255)  # red
                label = "UNAUTHORIZED"

            cv2.rectangle(display, (x, y), (x+w, y+h), color, 2)
            cv2.putText(display, label, (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.circle(display, (cx, cy), 4, color, -1)

            # Alert: unauthorized + inside restricted area
            if not verified and point_in_rect((cx, cy), restricted_rect):
                now = time.time()
                if now - last_alert_time > ALERT_COOLDOWN_SECS:
                    beep()
                    last_alert_time = now
                    cv2.putText(display, "ALERT: UNAUTHORIZED IN RESTRICTED ZONE",
                                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

        cv2.putText(display, "Right-click & drag to set Restricted Zone | 'q' to quit",
                    (15, display.shape[0]-12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        cv2.imshow("Restricted Area Monitor", display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    if os.path.exists(TEMP_FACE):
        os.remove(TEMP_FACE)  # clean up temp file