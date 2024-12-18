import os
import cv2
import math
import numpy as np
from utils import calculate_shinai_endpoints, draw_shinai
# from segmentation import run_segmentation_placeholder

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

def determine_body_orientation_from_arms(left_palm_center, right_palm_center):
    """
    Determine body orientation based on the relative positions of the palms.
    """
    if right_palm_center[0] > left_palm_center[0]:
        return "facing_right"
    else:
        return "facing_left"

def process_frames(input_frames_dir, pose_output_dir, seg_output_dir, pose, mp_drawing, mp_pose, calculate_palm_center):
    """
    Process frames for pose estimation and segmentation.

    Parameters:
    - input_frames_dir: Directory containing input frames.
    - pose_output_dir: Directory to save pose estimation results.
    - seg_output_dir: Directory to save segmentation results.
    - pose: Mediapipe Pose object.
    - mp_drawing: Mediapipe drawing utilities.
    - mp_pose: Mediapipe Pose enum.
    - calculate_palm_center: Function to compute palm center coordinates.

    Returns:
    - first_frame_data: Data from the first frame with valid pose results.
    - last_frame_data: Data from the last frame with valid pose results.
    """
    frame_files = sorted([f for f in os.listdir(input_frames_dir) if f.endswith(".png")])
    first_frame_data = None
    last_frame_data = None

    for i, frame_file in enumerate(frame_files):
        frame_path = os.path.join(input_frames_dir, frame_file)
        frame = cv2.imread(frame_path)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        # Run segmentation (placeholder)
        # seg_mask = run_segmentation_placeholder(frame)

        # Create blank image for pose
        height, width, _ = frame.shape
        blank_image = 255 * np.ones((height, width, 3), dtype=np.uint8)  # White background

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Extract key points
            left_index = [landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value].x * width,
                          landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value].y * height]
            left_thumb = [landmarks[mp_pose.PoseLandmark.LEFT_THUMB.value].x * width,
                          landmarks[mp_pose.PoseLandmark.LEFT_THUMB.value].y * height]
            left_pinky = [landmarks[mp_pose.PoseLandmark.LEFT_PINKY.value].x * width,
                          landmarks[mp_pose.PoseLandmark.LEFT_PINKY.value].y * height]
            left_palm_center = calculate_palm_center(left_index, left_thumb, left_pinky)

            right_index = [landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value].x * width,
                           landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value].y * height]
            right_thumb = [landmarks[mp_pose.PoseLandmark.RIGHT_THUMB.value].x * width,
                           landmarks[mp_pose.PoseLandmark.RIGHT_THUMB.value].y * height]
            right_pinky = [landmarks[mp_pose.PoseLandmark.RIGHT_PINKY.value].x * width,
                           landmarks[mp_pose.PoseLandmark.RIGHT_PINKY.value].y * height]
            right_palm_center = calculate_palm_center(right_index, right_thumb, right_pinky)

            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * width,
                             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * height]
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * width,
                              landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * height]

            # Draw pose skeleton
            mp_drawing.draw_landmarks(blank_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                       mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                                       mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2))

            # Draw shinai
            shinai_start, shinai_end = calculate_shinai_endpoints(left_palm_center, right_palm_center)
            draw_shinai(blank_image, shinai_start, shinai_end)

            body_angle = compute_vertical_angle(left_shoulder[0], left_shoulder[1], right_shoulder[0], right_shoulder[1])
            shinai_angle = compute_vertical_angle(shinai_start[0], shinai_start[1], shinai_end[0], shinai_end[1])
            orientation = determine_body_orientation_from_arms(left_palm_center, right_palm_center)

            frame_data = {
                "frame_number": i,
                "body_angle": body_angle,
                "shinai_angle": shinai_angle,
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

        # Save segmentation result (for placeholder, save binary mask)
        seg_frame_path = os.path.join(seg_output_dir, frame_file)
        # cv2.imwrite(seg_frame_path, (seg_mask * 255).astype('uint8'))

    return first_frame_data, last_frame_data
