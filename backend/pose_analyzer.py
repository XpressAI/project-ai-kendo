import cv2
import mediapipe as mp
from utils import calculate_shinai_endpoints, draw_shinai

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

def calculate_palm_center(index, thumb, pinky):
    """
    Calculate the center of the palm based on index, thumb, and pinky landmarks.
    """
    x = (index[0] + thumb[0] + pinky[0]) / 3
    y = (index[1] + thumb[1] + pinky[1]) / 3
    return [x, y]

def process_frame(frame, results):
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        height, width, _ = frame.shape

        # Extract and scale key points for the left hand
        left_index = [landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value].x * width,
                      landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value].y * height]
        left_thumb = [landmarks[mp_pose.PoseLandmark.LEFT_THUMB.value].x * width,
                      landmarks[mp_pose.PoseLandmark.LEFT_THUMB.value].y * height]
        left_pinky = [landmarks[mp_pose.PoseLandmark.LEFT_PINKY.value].x * width,
                      landmarks[mp_pose.PoseLandmark.LEFT_PINKY.value].y * height]
        left_palm_center = calculate_palm_center(left_index, left_thumb, left_pinky)

        # Extract and scale key points for the right hand
        right_index = [landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value].x * width,
                       landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value].y * height]
        right_thumb = [landmarks[mp_pose.PoseLandmark.RIGHT_THUMB.value].x * width,
                       landmarks[mp_pose.PoseLandmark.RIGHT_THUMB.value].y * height]
        right_pinky = [landmarks[mp_pose.PoseLandmark.RIGHT_PINKY.value].x * width,
                       landmarks[mp_pose.PoseLandmark.RIGHT_PINKY.value].y * height]
        right_palm_center = calculate_palm_center(right_index, right_thumb, right_pinky)

        # Calculate shinai endpoints
        shinai_start, shinai_end = calculate_shinai_endpoints(left_palm_center, right_palm_center)

        # Draw landmarks and shinai
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        draw_shinai(frame, shinai_start, shinai_end)

    return frame

def process_video_with_shinai(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        processed_frame = process_frame(frame, results)

        if out is None:
            height, width, _ = frame.shape
            out = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))

        out.write(processed_frame)

    cap.release()
    if out:
        out.release()
    print("Video with shinai processed and saved!")
