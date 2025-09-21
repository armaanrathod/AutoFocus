import cv2
import mediapipe as mp
import numpy as np
import time
import platform
import os
from threading import Thread
import pandas as pd
from ultralytics import YOLO  # pip install ultralytics

# -----------------------------
# Cross-platform beep
def beep():
    if platform.system() == "Windows":
        import winsound
        winsound.Beep(1000, 500)
    else:
        os.system("echo -e '\a'")

# -----------------------------
# Mediapipe initialization
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(max_num_faces=1, refine_landmarks=True)

# -----------------------------
# YOLOv8 phone detection
yolo_model = YOLO("yolov8n.pt")  # nano model for speed

# -----------------------------
# Data storage
columns = ["Date", "Time", "Event", "Value"]
event_log = pd.DataFrame(columns=columns)

# -----------------------------
# Parameters
EYE_AR_THRESH = 0.25   # Eye aspect ratio threshold
HEAD_TILT_THRESH = 15  # degrees threshold for alerts
eye_close_start = None

# -----------------------------
# Eye aspect ratio function
def eye_aspect_ratio(eye):
    A = np.linalg.norm(np.array(eye[1]) - np.array(eye[5]))
    B = np.linalg.norm(np.array(eye[2]) - np.array(eye[4]))
    C = np.linalg.norm(np.array(eye[0]) - np.array(eye[3]))
    return (A + B) / (2.0 * C)

# -----------------------------
# Webcam capture
cap = cv2.VideoCapture(0)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame.shape

        # -----------------------------
        # Face landmarks
        face_results = face_mesh.process(frame_rgb)

        # -----------------------------
        # YOLOv8 phone detection
        yolo_results = yolo_model(frame)
        phone_detected_flag = False
        for r in yolo_results:
            boxes = r.boxes.xyxy.cpu().numpy() if hasattr(r.boxes, 'xyxy') else []
            classes = r.boxes.cls.cpu().numpy() if hasattr(r.boxes, 'cls') else []
            for box, cls in zip(boxes, classes):
                if int(cls) == 67:  # COCO class 67 = cell phone
                    phone_detected_flag = True
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 2)
                    cv2.putText(frame, "PHONE DETECTED", (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        # -----------------------------
        # Current date and time
        now = time.localtime()
        date_str = time.strftime("%Y-%m-%d", now)
        time_str = time.strftime("%H:%M:%S", now)

        # -----------------------------
        if face_results.multi_face_landmarks:
            landmarks = face_results.multi_face_landmarks[0].landmark

            # Eye landmarks
            left_eye_idx = [33, 160, 158, 133, 153, 144]
            right_eye_idx = [362, 385, 387, 263, 373, 380]
            left_eye = [(landmarks[i].x*w, landmarks[i].y*h) for i in left_eye_idx]
            right_eye = [(landmarks[i].x*w, landmarks[i].y*h) for i in right_eye_idx]

            ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0

            # -----------------------------
            # Drowsiness detection
            if ear < EYE_AR_THRESH:
                if eye_close_start is None:
                    eye_close_start = time.time()
                elapsed = time.time() - eye_close_start
                if elapsed > 2:
                    Thread(target=beep).start()
                    cv2.putText(frame, f"DROWSINESS {round(elapsed,1)}s!", (50,50),
                                cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
                    if len(event_log) == 0 or event_log.iloc[-1]['Event'] != 'Drowsiness':
                        event_log = pd.concat([event_log, pd.DataFrame([{
                            "Date": date_str,
                            "Time": time_str,
                            "Event": "Drowsiness",
                            "Value": round(elapsed,2)
                        }])], ignore_index=True)
            else:
                eye_close_start = None

            # -----------------------------
            # Head tilt detection
            # Horizontal (side to side)
            left_corner = landmarks[33]
            right_corner = landmarks[263]
            dx_lr = right_corner.x - left_corner.x
            dy_lr = right_corner.y - left_corner.y
            tilt_lr = np.degrees(np.arctan2(dy_lr, dx_lr))

            # Vertical (forward/back)
            nose_tip = landmarks[1]
            eye_center_y = (landmarks[159].y + landmarks[386].y)/2
            tilt_fb = (nose_tip.y - eye_center_y) * 100  # scaled to degrees/percentage

            if abs(tilt_lr) > HEAD_TILT_THRESH or abs(tilt_fb) > HEAD_TILT_THRESH:
                Thread(target=beep).start()
                cv2.putText(frame, f"HEAD TILT ALERT LR:{round(tilt_lr,1)} FB:{round(tilt_fb,1)}", (50,80),
                            cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
                if len(event_log) == 0 or event_log.iloc[-1]['Event'] != 'Head Tilt':
                    event_log = pd.concat([event_log, pd.DataFrame([{
                        "Date": date_str,
                        "Time": time_str,
                        "Event": "Head Tilt",
                        "Value": f"LR:{round(tilt_lr,2)}, FB:{round(tilt_fb,2)}"
                    }])], ignore_index=True)

        # -----------------------------
        # Phone logging
        if phone_detected_flag:
            Thread(target=beep).start()
            if len(event_log) == 0 or event_log.iloc[-1]['Event'] != 'Phone':
                event_log = pd.concat([event_log, pd.DataFrame([{
                    "Date": date_str,
                    "Time": time_str,
                    "Event": "Phone",
                    "Value": "Detected"
                }])], ignore_index=True)

        # -----------------------------
        cv2.imshow("Distraction Monitor", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    event_log.to_csv("distraction_events.csv", index=False)
    print("Saved distraction_events.csv")
