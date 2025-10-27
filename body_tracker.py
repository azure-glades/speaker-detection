# new file: body_tracker.py
import mediapipe as mp
import numpy as np
import cv2

class BodyTracker:
    def __init__(self):
        self.pose = mp.solutions.pose.Pose(
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def __call__(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)
        h, w = frame.shape[:2]
        boxes = []
        if results.pose_landmarks:
            # extract 2-D keypoints
            pts = [(lm.x * w, lm.y * h) for lm in results.pose_landmarks.landmark]
            # shoulder, hip, ankle landmarks
            shoulder_l = pts[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value]
            shoulder_r = pts[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value]
            hip_l      = pts[mp.solutions.pose.PoseLandmark.LEFT_HIP.value]
            ankle_l    = pts[mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value]
            ankle_r    = pts[mp.solutions.pose.PoseLandmark.RIGHT_ANKLE.value]

            x_min = min(shoulder_l[0], shoulder_r[0], hip_l[0], ankle_l[0], ankle_r[0])
            x_max = max(shoulder_l[0], shoulder_r[0], hip_l[0], ankle_l[0], ankle_r[0])
            y_min = min(shoulder_l[1], shoulder_r[1])
            y_max = max(ankle_l[1], ankle_r[1])

            # 20 % padding
            pad_x = int((x_max - x_min) * 0.2)
            pad_y = int((y_max - y_min) * 0.2)
            x1 = max(0, int(x_min) - pad_x)
            y1 = max(0, int(y_min) - pad_y)
            x2 = min(w, int(x_max) + pad_x)
            y2 = min(h, int(y_max) + pad_y)
            boxes.append((x1, y1, x2, y2))
        return boxes