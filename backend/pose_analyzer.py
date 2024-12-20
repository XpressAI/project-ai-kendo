import os
import json
import cv2
import mediapipe as mp
import math
from frame_extractor import extract_frames
from frame_processor import process_frames
from video_combiner import (combine_frames_into_video, create_processed_video, 
                            create_masked_video, create_original_pose_video,
                            create_original_segmented_video, create_final_combined_video)
from utils import (save_frame_with_overlay, fit_principal_line, 
                    compute_angle_from_line, line_points)

from evf_sam2_inference import run_evf_sam2_inference

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

def calculate_palm_center(index, thumb, pinky):
    x = (index[0] + thumb[0] + pinky[0]) / 3
    y = (index[1] + thumb[1] + pinky[1]) / 3
    return [x, y]

def compute_vertical_angle(x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    length = math.sqrt(dx*dx + dy*dy)
    if length < 1e-6:
        return 0.0
    dx /= length
    dy /= length
    angle = math.degrees(math.acos(dy))
    return angle

def determine_body_orientation_from_arms(left_palm_center, right_palm_center):
    if right_palm_center[0] > left_palm_center[0]:
        return "facing_right"
    else:
        return "facing_left"

def classify_cut(initial_angle, final_angle):
    angle_diff = abs(final_angle - initial_angle)
    if angle_diff > 40:
        return "big_cut"
    else:
        return "small_cut"

def process_single_video(input_path, output_dir, temp_dir):
    video_name = os.path.splitext(os.path.basename(input_path))[0]
    video_results_dir = os.path.join(output_dir, video_name)
    os.makedirs(video_results_dir, exist_ok=True)

    input_frames_dir = os.path.join(video_results_dir, "input_frames")
    os.makedirs(input_frames_dir, exist_ok=True)

    # Extract frames
    fps, width, height, frame_count = extract_frames(input_path, input_frames_dir)

    # Temp dirs
    pose_output_dir = os.path.join(video_results_dir, "temp_pose")
    seg_output_dir = os.path.join(video_results_dir, "temp_seg")
    os.makedirs(pose_output_dir, exist_ok=True)
    os.makedirs(seg_output_dir, exist_ok=True)

    # Pose estimation
    first_frame_data, last_frame_data = process_frames(
        input_frames_dir,
        pose_output_dir,
        seg_output_dir,
        pose,
        mp_drawing,
        mp_pose,
        calculate_palm_center
    )

    # Run segmentation using EVF-SAM2
    run_evf_sam2_inference(
        version="EVF-SAM/checkpoints/evf_sam2",
        input_folder=input_frames_dir,
        output_folder=seg_output_dir,
        prompt="A shinai (竹刀) is a Japanese sword...",
        model_type="sam2",
        precision="fp16"
    )

    # Create videos
    final_video_path = os.path.join(video_results_dir, "processed.mp4")
    combine_frames_into_video(input_frames_dir, pose_output_dir, final_video_path, fps)
    pose_video_path = os.path.join(video_results_dir, "pose_only.mp4")
    create_processed_video(pose_output_dir, pose_video_path, fps)
    masked_video_path = os.path.join(video_results_dir, "masked_only.mp4")
    create_masked_video(input_frames_dir, seg_output_dir, masked_video_path, fps)
    original_pose_path = os.path.join(video_results_dir, "original_pose.mp4")
    create_original_pose_video(input_frames_dir, pose_output_dir, original_pose_path, fps)
    original_segmented_path = os.path.join(video_results_dir, "original_segmented.mp4")
    create_original_segmented_video(input_frames_dir, seg_output_dir, original_segmented_path, fps)
    final_combined_path = os.path.join(video_results_dir, "final_combined.mp4")
    create_final_combined_video(input_frames_dir, pose_output_dir, seg_output_dir, final_combined_path, fps)

    # Final analysis using principal axis of the mask
    if first_frame_data and last_frame_data:
        # Identify the mask paths for the first and last frames
        first_mask_path = os.path.join(seg_output_dir, f"frame_{first_frame_data['frame_number']:06d}_mask.png")
        final_mask_path = os.path.join(seg_output_dir, f"frame_{last_frame_data['frame_number']:06d}_mask.png")

        first_line = fit_principal_line(first_mask_path)
        final_line = fit_principal_line(final_mask_path)

        if first_line is not None and final_line is not None:
            vx1, vy1, x1, y1 = first_line
            vx2, vy2, x2, y2 = final_line

            initial_shinai_angle = compute_angle_from_line(vx1, vy1)
            final_shinai_angle = compute_angle_from_line(vx2, vy2)

            # Compute relative angles
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

            # Save analysis results
            json_path = os.path.join(video_results_dir, "analysis_results.json")
            with open(json_path, "w") as f:
                json.dump(analysis_results, f, indent=4)

            # Function to get two points from line parameters for drawing
            def line_points(x, y, vx, vy, length=200):
                x1 = int(x - length * vx)
                y1 = int(y - length * vy)
                x2 = int(x + length * vx)
                y2 = int(y + length * vy)
                return (x1, y1), (x2, y2)

            first_shinai_start, first_shinai_end = line_points(x1, y1, vx1, vy1)
            last_shinai_start, last_shinai_end = line_points(x2, y2, vx2, vy2)

            # Save overlay images for first and last frames
            cap = cv2.VideoCapture(input_path)

            # First frame overlay
            cap.set(cv2.CAP_PROP_POS_FRAMES, first_frame_data["frame_number"])
            ret, first_frame = cap.read()
            if ret:
                first_frame_path = os.path.join(video_results_dir, "kamae_analysis.png")
                save_frame_with_overlay(
                    first_frame,
                    first_shinai_start,
                    first_shinai_end,
                    first_frame_data["left_palm"],
                    first_frame_data["right_palm"],
                    first_frame_data["body_angle"],
                    initial_shinai_angle,
                    first_frame_path,
                    draw_vertical_line=True,
                    draw_arc=True
                )

            # Last frame overlay
            cap.set(cv2.CAP_PROP_POS_FRAMES, last_frame_data["frame_number"])
            ret, last_frame = cap.read()
            if ret:
                final_frame_path = os.path.join(video_results_dir, "final_cut_analysis.png")
                save_frame_with_overlay(
                    last_frame,
                    last_shinai_start,
                    last_shinai_end,
                    last_frame_data["left_palm"],
                    last_frame_data["right_palm"],
                    last_frame_data["body_angle"],
                    final_shinai_angle,
                    final_frame_path,
                    draw_vertical_line=True,
                    draw_arc=True
                )
            cap.release()

            return analysis_results
    return None

def process_kendo_analysis_on_dir(input_dir, output_dir, temp_dir=None):
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
            input_path = os.path.join(input_dir, filename)
            print(f"Processing video: {input_path}")
            analysis_results = process_single_video(input_path, output_dir, temp_dir)
            if analysis_results is not None:
                print(f"Analysis results saved in output directory for {filename}")
