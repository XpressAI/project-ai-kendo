import cv2
import mediapipe as mp
import json
import math
import os
from utils import (calculate_shinai_endpoints, draw_shinai, 
                   save_frame_with_overlay, draw_transparent_line, draw_angle_text)

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

def compute_vertical_angle(x1, y1, x2, y2):
    """
    Compute the angle of the line formed by points (x1, y1) -> (x2, y2) relative to the vertical line going down.
    Vertical line down is considered 0 degrees.
    The angle returned is between 0 and 180 degrees.
    """
    dx = x2 - x1
    dy = y2 - y1
    length = math.sqrt(dx*dx + dy*dy)
    if length < 1e-6:
        return 0.0
    dx /= length
    dy /= length
    # Angle from vertical (0,1)
    # cos(theta) = dy, so theta = arccos(dy)
    angle = math.degrees(math.acos(dy))
    return angle

def process_frame(frame, results):
    """
    Process a single frame and return necessary data.
    """
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        height, width, _ = frame.shape

        # Extract and scale key points for hands
        left_index = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_INDEX.value].x * width,
                      landmarks[mp.solutions.pose.PoseLandmark.LEFT_INDEX.value].y * height]
        left_thumb = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_THUMB.value].x * width,
                      landmarks[mp.solutions.pose.PoseLandmark.LEFT_THUMB.value].y * height]
        left_pinky = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_PINKY.value].x * width,
                      landmarks[mp.solutions.pose.PoseLandmark.LEFT_PINKY.value].y * height]
        left_palm_center = calculate_palm_center(left_index, left_thumb, left_pinky)

        right_index = [landmarks[mp.solutions.pose.PoseLandmark.RIGHT_INDEX.value].x * width,
                       landmarks[mp.solutions.pose.PoseLandmark.RIGHT_INDEX.value].y * height]
        right_thumb = [landmarks[mp.solutions.pose.PoseLandmark.RIGHT_THUMB.value].x * width,
                       landmarks[mp.solutions.pose.PoseLandmark.RIGHT_THUMB.value].y * height]
        right_pinky = [landmarks[mp.solutions.pose.PoseLandmark.RIGHT_PINKY.value].x * width,
                       landmarks[mp.solutions.pose.PoseLandmark.RIGHT_PINKY.value].y * height]
        right_palm_center = calculate_palm_center(right_index, right_thumb, right_pinky)

        # Shoulder points
        left_shoulder = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value].x * width,
                         landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value].y * height]
        right_shoulder = [landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value].x * width,
                          landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value].y * height]

        # Calculate shinai endpoints
        shinai_start, shinai_end = calculate_shinai_endpoints(left_palm_center, right_palm_center)

        # Draw pose landmarks
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        # Draw shinai
        draw_shinai(frame, shinai_start, shinai_end)

        return frame, left_palm_center, right_palm_center, left_shoulder, right_shoulder, shinai_start, shinai_end
    return frame, None, None, None, None, None, None

def determine_body_orientation_from_arms(left_palm_center, right_palm_center):
    """
    Determine if body is facing left or right based on where the arms (shinai) are pointing.
    If the right palm center is to the right of the left palm center, facing right; else facing left.
    """
    if right_palm_center[0] > left_palm_center[0]:
        return "facing_right"
    else:
        return "facing_left"

def classify_cut(initial_angle, final_angle):
    """
    Classify cut as 'big' or 'small' based on angle difference.
    Adjust threshold as needed.
    """
    angle_diff = abs(final_angle - initial_angle)
    if angle_diff > 40:
        return "big_cut"
    else:
        return "small_cut"
def process_single_video(input_path, output_path):
    """
    Process a single video to perform kendo analysis.

    Parameters:
    - input_path: path to the input video
    - output_path: directory where results will be saved (analysis json, pngs, processed video)
    """

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {input_path}")
        return None

    # Prepare output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30.0

    output_video_path = os.path.join(output_path, "processed.mp4")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    first_frame_data = None
    last_frame_data = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        processed_frame, left_palm, right_palm, l_shoulder, r_shoulder, shinai_start, shinai_end = process_frame(frame, results)
        out.write(processed_frame)

        if left_palm and right_palm and l_shoulder and r_shoulder and shinai_start and shinai_end:
            body_angle = compute_vertical_angle(l_shoulder[0], l_shoulder[1], r_shoulder[0], r_shoulder[1])
            shinai_angle = compute_vertical_angle(shinai_start[0], shinai_start[1], shinai_end[0], shinai_end[1])
            orientation = determine_body_orientation_from_arms(left_palm, right_palm)

            frame_data = {
                "frame_number": frame_count,
                "body_angle": body_angle,
                "shinai_angle": shinai_angle,
                "orientation": orientation,
                "left_palm": left_palm,
                "right_palm": right_palm,
                "shinai_start": shinai_start,
                "shinai_end": shinai_end
            }

            if first_frame_data is None:
                first_frame_data = frame_data
            last_frame_data = frame_data

        frame_count += 1

    cap.release()
    out.release()

    # Perform final analysis
    analysis_results = {}
    if first_frame_data and last_frame_data:
        initial_shinai_angle = first_frame_data["shinai_angle"]
        final_shinai_angle = last_frame_data["shinai_angle"]
        cut_classification = classify_cut(initial_shinai_angle, final_shinai_angle)
        initial_relative_angle = initial_shinai_angle - first_frame_data["body_angle"]
        final_relative_angle = final_shinai_angle - last_frame_data["body_angle"]

        analysis_results = {
            "initial_frame": first_frame_data["frame_number"],
            "final_frame": last_frame_data["frame_number"],
            "initial_shinai_angle": initial_shinai_angle,
            "final_shinai_angle": final_shinai_angle,
            "initial_relative_angle": initial_relative_angle,
            "final_relative_angle": final_relative_angle,
            "cut_classification": cut_classification
        }

        # Save first and last frame images with overlay
        cap = cv2.VideoCapture(input_path)

        # Save first frame image
        cap.set(cv2.CAP_PROP_POS_FRAMES, first_frame_data["frame_number"])
        ret, first_frame = cap.read()
        if ret:
            first_frame_path = os.path.join(output_path, "kamae_analysis.png")
            save_frame_with_overlay(
                first_frame,
                first_frame_data["shinai_start"],
                first_frame_data["shinai_end"],
                first_frame_data["left_palm"],
                first_frame_data["right_palm"],
                first_frame_data["body_angle"],
                first_frame_data["shinai_angle"],
                first_frame_path,
                draw_vertical_line=True,
                draw_arc=True
            )

        # Save last frame image
        cap.set(cv2.CAP_PROP_POS_FRAMES, last_frame_data["frame_number"])
        ret, last_frame = cap.read()
        if ret:
            final_frame_path = os.path.join(output_path, "final_cut_analysis.png")
            save_frame_with_overlay(
                last_frame,
                last_frame_data["shinai_start"],
                last_frame_data["shinai_end"],
                last_frame_data["left_palm"],
                last_frame_data["right_palm"],
                last_frame_data["body_angle"],
                last_frame_data["shinai_angle"],
                final_frame_path,
                draw_vertical_line=True,
                draw_arc=True
            )

        cap.release()

        # Save results to JSON
        json_path = os.path.join(output_path, "analysis_results.json")
        with open(json_path, "w") as f:
            json.dump(analysis_results, f, indent=4)

    return analysis_results

# TODO: Implement a swing detector that takes a long video and splits it into individual swing videos.
# For now, we assume input_dir already contains the individual cut videos.
# def detect_swings(long_video_path, temp_dir):
#     # TODO: implement swing detection
#     pass

def process_kendo_analysis_on_dir(input_dir, output_dir, temp_dir=None):
    """
    Process all videos in input_dir (each representing a single swing),
    perform analysis, and output results to output_dir.
    """

    if temp_dir is None:
        temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
            input_path = os.path.join(input_dir, filename)
            video_name = os.path.splitext(filename)[0]
            video_output_dir = os.path.join(output_dir, video_name)
            os.makedirs(video_output_dir, exist_ok=True)

            print(f"Processing video: {input_path}")
            analysis_results = process_single_video(input_path, video_output_dir)
            if analysis_results is not None:
                print(f"Analysis results saved in {video_output_dir}")
