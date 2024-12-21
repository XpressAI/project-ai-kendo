import os
import cv2
import math
import numpy as np
from utils import calculate_shinai_endpoints, draw_shinai

def compute_vertical_angle(x1, y1, x2, y2):
    """
    Compute the angle of the line formed by points (x1, y1) -> (x2, y2) relative to the vertical.
    """
    dx = x2 - x1
    dy = y2 - y1
    length = math.sqrt(dx * dx + dy * dy)
    if length < 1e-6:
        return 0.0
    dx /= length
    dy /= length
    angle = math.degrees(math.acos(dy))
    return angle


def process_frames(input_frames_dir, pose_output_dir, seg_output_dir, pose, mp_drawing, mp_pose, calculate_palm_center):
    frame_files = sorted([f for f in os.listdir(input_frames_dir) if f.endswith(".png")])
    first_frame_data = None
    last_frame_data = None

    for i, frame_file in enumerate(frame_files):
        frame_path = os.path.join(input_frames_dir, frame_file)
        frame = cv2.imread(frame_path)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        height, width, _ = frame.shape
        blank_image = 255 * np.ones((height, width, 3), dtype=np.uint8)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            def lm(l): return np.array([landmarks[l].x * width, landmarks[l].y * height])
            
            nose = lm(mp_pose.PoseLandmark.NOSE.value)
            left_shoulder = lm(mp_pose.PoseLandmark.LEFT_SHOULDER.value)
            right_shoulder = lm(mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
            left_hip = lm(mp_pose.PoseLandmark.LEFT_HIP.value)
            right_hip = lm(mp_pose.PoseLandmark.RIGHT_HIP.value)

            # Palms
            left_index = lm(mp_pose.PoseLandmark.LEFT_INDEX.value)
            left_thumb = lm(mp_pose.PoseLandmark.LEFT_THUMB.value)
            left_pinky = lm(mp_pose.PoseLandmark.LEFT_PINKY.value)
            left_palm_center = calculate_palm_center(left_index, left_thumb, left_pinky)

            right_index = lm(mp_pose.PoseLandmark.RIGHT_INDEX.value)
            right_thumb = lm(mp_pose.PoseLandmark.RIGHT_THUMB.value)
            right_pinky = lm(mp_pose.PoseLandmark.RIGHT_PINKY.value)
            right_palm_center = calculate_palm_center(right_index, right_thumb, right_pinky)

            # Body vertical axis (hip_mid to shoulder_mid)
            shoulder_mid = (left_shoulder + right_shoulder) / 2.0
            hip_mid = (left_hip + right_hip) / 2.0
            body_vx = shoulder_mid[0] - hip_mid[0]
            body_vy = shoulder_mid[1] - hip_mid[1]

            # Draw pose
            mp_drawing.draw_landmarks(blank_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2))

            # Draw shinai
            shinai_start, shinai_end = calculate_shinai_endpoints(left_palm_center, right_palm_center)
            draw_shinai(blank_image, shinai_start, shinai_end)

            # Determine orientation based on shinai midpoint position relative to body midpoint
            shinai_mid = (np.array(shinai_start) + np.array(shinai_end)) / 2.0

            orientation = "facing_left" if shinai_mid[0] < hip_mid[0] else "facing_right"

            frame_data = {
                "frame_number": i,
                "nose": nose.tolist(),
                "shoulder_mid": shoulder_mid.tolist(),
                "hip_mid": hip_mid.tolist(),
                "body_vx": float(body_vx),
                "body_vy": float(body_vy),
                "orientation": orientation,
                "left_palm": left_palm_center,
                "right_palm": right_palm_center,
                "shinai_start": shinai_start,
                "shinai_end": shinai_end
            }

            if first_frame_data is None:
                first_frame_data = frame_data
            last_frame_data = frame_data

        # Save pose frame
        pose_frame_path = os.path.join(pose_output_dir, frame_file)
        cv2.imwrite(pose_frame_path, blank_image)

    return first_frame_data, last_frame_data
