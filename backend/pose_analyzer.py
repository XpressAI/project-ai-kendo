import os
import json
import cv2
import math
from frame_extractor import extract_frames
from frame_processor import process_frames
from video_combiner import (combine_frames_into_video, create_processed_video, 
                            create_masked_video, create_original_pose_video,
                            create_original_segmented_video, create_final_combined_video)
from utils import (save_frame_with_overlay_perp, fit_principal_line, 
                   classify_cut, angle_from_vector, angle_difference)
from evf_sam2_inference import run_evf_sam2_inference
from mediapipe import solutions as mp

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

    pose_inst = mp.pose.Pose()
    mp_drawing = mp.drawing_utils
    mp_pose = mp.pose
    def calculate_palm_center(index, thumb, pinky):
        x = (index[0] + thumb[0] + pinky[0]) / 3
        y = (index[1] + thumb[1] + pinky[1]) / 3
        return [x, y]

    first_frame_data, last_frame_data = process_frames(
        input_frames_dir,
        pose_output_dir,
        seg_output_dir,
        pose_inst,
        mp_drawing,
        mp_pose,
        calculate_palm_center
    )

    # Run segmentation
    run_evf_sam2_inference(
        version="EVF-SAM/checkpoints/evf_sam2",
        input_folder=input_frames_dir,
        output_folder=seg_output_dir,
        prompt="A shinai (竹刀) is a Japanese sword typically made of bamboo used for practice and competition in kendō",
        model_type="sam2",
        precision="fp16"
    )

    # Create videos
    final_video_path = os.path.join(video_results_dir, "processed.mp4")
    combine_frames_into_video(input_frames_dir, pose_output_dir, final_video_path, fps)
    create_processed_video(pose_output_dir, os.path.join(video_results_dir, "pose_only.mp4"), fps)
    create_masked_video(input_frames_dir, seg_output_dir, os.path.join(video_results_dir, "masked_only.mp4"), fps)
    create_original_pose_video(input_frames_dir, pose_output_dir, os.path.join(video_results_dir, "original_pose.mp4"), fps)
    create_original_segmented_video(input_frames_dir, seg_output_dir, os.path.join(video_results_dir, "original_segmented.mp4"), fps)
    create_final_combined_video(input_frames_dir, pose_output_dir, seg_output_dir, os.path.join(video_results_dir, "final_combined.mp4"), fps)

    if first_frame_data and last_frame_data:
        first_mask_path = os.path.join(seg_output_dir, f"frame_{first_frame_data['frame_number']:06d}_mask.png")
        final_mask_path = os.path.join(seg_output_dir, f"frame_{last_frame_data['frame_number']:06d}_mask.png")

        first_line = fit_principal_line(first_mask_path)
        final_line = fit_principal_line(final_mask_path)

        if first_line is not None and final_line is not None:
            vx1, vy1, x1, y1 = first_line
            vx2, vy2, x2, y2 = final_line

            # Compute body angles for first and last frames
            def get_perp_angle(body_vx, body_vy, orientation):
                body_angle = angle_from_vector(body_vx, body_vy, orientation)
                perp_angle = body_angle + 90.0
                # Normalize to (-180,180)
                if perp_angle > 180:
                    perp_angle -= 360
                elif perp_angle <= -180:
                    perp_angle += 360
                return perp_angle

            # First frame angles
            first_perp = get_perp_angle(
                first_frame_data["body_vx"], 
                first_frame_data["body_vy"],
                first_frame_data["orientation"]
            )
            first_shinai_angle = angle_from_vector(
                vx1, vy1, 
                first_frame_data["orientation"]
            )
            first_relative_angle = angle_difference(first_perp, first_shinai_angle)

            # Final frame angles
            final_perp = get_perp_angle(
                last_frame_data["body_vx"], 
                last_frame_data["body_vy"],
                last_frame_data["orientation"]
            )
            final_shinai_angle = angle_from_vector(
                vx2, vy2, 
                last_frame_data["orientation"]
            )
            final_relative_angle = angle_difference(final_perp, final_shinai_angle)

            # Classify cut based on final_relative_angle
            cut_classification = classify_cut(final_relative_angle)

            analysis_results = {
                "initial_frame": first_frame_data["frame_number"],
                "initial_relative_angle": first_relative_angle,
                "final_frame": last_frame_data["frame_number"],
                "final_relative_angle": final_relative_angle,
                "cut_classification": cut_classification,
                "orientation_first": first_frame_data["orientation"],
                "orientation_last": last_frame_data["orientation"]
            }

            # Save results
            json_path = os.path.join(video_results_dir, "analysis_results.json")
            with open(json_path, "w") as f:
                json.dump(analysis_results, f, indent=4)

            # Overlays
            cap = cv2.VideoCapture(input_path)

            # First frame overlay (kamae)
            cap.set(cv2.CAP_PROP_POS_FRAMES, first_frame_data["frame_number"])
            ret, first_frame = cap.read()
            if ret:
                first_frame_path = os.path.join(video_results_dir, "kamae_analysis.png")
                save_frame_with_overlay_perp(
                    frame=first_frame,
                    baseline_angle=first_perp,
                    current_angle=first_shinai_angle,
                    x=x1, y=y1,
                    save_path=first_frame_path,
                    draw_arc=True,
                    orientation=first_frame_data["orientation"]
                )

            # Last frame overlay (final cut)
            cap.set(cv2.CAP_PROP_POS_FRAMES, last_frame_data["frame_number"])
            ret, last_frame = cap.read()
            if ret:
                final_frame_path = os.path.join(video_results_dir, "final_cut_analysis.png")
                save_frame_with_overlay_perp(
                    frame=last_frame,
                    baseline_angle=final_perp,
                    current_angle=final_shinai_angle,
                    x=x2, y=y2,
                    save_path=final_frame_path,
                    draw_arc=True,
                    orientation=last_frame_data["orientation"]
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
                print(f"Analysis results saved for {filename}")
