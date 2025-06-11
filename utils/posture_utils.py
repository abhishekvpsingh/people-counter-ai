import mediapipe as mp
import cv2

mp_pose = mp.solutions.pose

def detect_posture(frame, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    person_crop = frame[y1:y2, x1:x2]

    with mp_pose.Pose(static_image_mode=True) as pose:
        results = pose.process(cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB))
        if not results.pose_landmarks:
            return "unknown"

        landmarks = results.pose_landmarks.landmark
        shoulder_y = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y
        hip_y = landmarks[mp_pose.PoseLandmark.LEFT_HIP].y

        return "sitting" if abs(shoulder_y - hip_y) < 0.1 else "standing"
