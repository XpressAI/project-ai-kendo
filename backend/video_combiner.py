# backend/video_combiner.py
import cv2
import os
import numpy as np

def combine_frames_into_video(input_frames_dir, processed_frames_dir, output_video_path, fps):
    frame_files = sorted([f for f in os.listdir(input_frames_dir) if f.endswith(".png")])
    if not frame_files:
        return
    first_input_frame = cv2.imread(os.path.join(input_frames_dir, frame_files[0]))
    first_processed_frame = cv2.imread(os.path.join(processed_frames_dir, frame_files[0]))
    height, width, _ = first_input_frame.shape

    if first_processed_frame is None:
        first_processed_frame = np.zeros_like(first_input_frame)
    else:
        if first_processed_frame.shape != first_input_frame.shape:
            first_processed_frame = cv2.resize(first_processed_frame, (width, height))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width*2, height))

    for f in frame_files:
        input_frame = cv2.imread(os.path.join(input_frames_dir, f))
        processed_frame = cv2.imread(os.path.join(processed_frames_dir, f))
        if processed_frame is None:
            processed_frame = np.zeros_like(input_frame)
        if processed_frame.shape != input_frame.shape:
            processed_frame = cv2.resize(processed_frame, (input_frame.shape[1], input_frame.shape[0]))

        combined = np.hstack((input_frame, processed_frame))
        out.write(combined)

    out.release()

def create_processed_video(processed_frames_dir, output_video_path, fps):
    frame_files = sorted([f for f in os.listdir(processed_frames_dir) if f.endswith(".png")])
    if not frame_files:
        return
    first_frame = cv2.imread(os.path.join(processed_frames_dir, frame_files[0]))
    height, width, _ = first_frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    for f in frame_files:
        frame = cv2.imread(os.path.join(processed_frames_dir, f))
        if frame is not None:
            out.write(frame)
    out.release()

def create_masked_video(input_frames_dir, seg_output_dir, output_video_path, fps):
    """
    Create a video showing the original frame and the segmentation mask overlay.
    We'll overlay the mask visualization (_vis.png) onto the original frame side-by-side.
    """
    frame_files = sorted([f for f in os.listdir(input_frames_dir) if f.endswith(".png")])
    if not frame_files:
        return

    # We have mask visualization in seg_output_dir as {frame_basename}_vis.png
    first_frame = cv2.imread(os.path.join(input_frames_dir, frame_files[0]))
    height, width, _ = first_frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width*2, height))

    for f in frame_files:
        base_name = os.path.splitext(f)[0]
        input_frame = cv2.imread(os.path.join(input_frames_dir, f))
        vis_path = os.path.join(seg_output_dir, f"{base_name}_vis.png")
        vis_frame = cv2.imread(vis_path)
        if vis_frame is None:
            # If no vis, create a blank or just use the input_frame
            vis_frame = np.zeros_like(input_frame)
        # Ensure same size
        if vis_frame.shape != input_frame.shape:
            vis_frame = cv2.resize(vis_frame, (width, height))

        combined = np.hstack((input_frame, vis_frame))
        out.write(combined)

    out.release()

def create_original_pose_video(input_frames_dir, pose_frames_dir, output_video_path, fps):
    """
    Create a video combining original frames and pose estimation frames side by side.
    """
    frame_files = sorted([f for f in os.listdir(input_frames_dir) if f.endswith(".png")])
    if not frame_files:
        return

    first_input_frame = cv2.imread(os.path.join(input_frames_dir, frame_files[0]))
    height, width, _ = first_input_frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width*2, height))

    for f in frame_files:
        input_frame = cv2.imread(os.path.join(input_frames_dir, f))
        pose_frame = cv2.imread(os.path.join(pose_frames_dir, f))
        if pose_frame is None:
            pose_frame = np.zeros_like(input_frame)
        if pose_frame.shape != input_frame.shape:
            pose_frame = cv2.resize(pose_frame, (width, height))

        combined = np.hstack((input_frame, pose_frame))
        out.write(combined)

    out.release()


def create_original_segmented_video(input_frames_dir, seg_output_dir, output_video_path, fps):
    """
    Create a video combining original frames and segmented frames side by side.
    """
    frame_files = sorted([f for f in os.listdir(input_frames_dir) if f.endswith(".png")])
    if not frame_files:
        return

    first_input_frame = cv2.imread(os.path.join(input_frames_dir, frame_files[0]))
    height, width, _ = first_input_frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width*2, height))

    for f in frame_files:
        input_frame = cv2.imread(os.path.join(input_frames_dir, f))
        seg_vis_path = os.path.join(seg_output_dir, f"{os.path.splitext(f)[0]}_vis.png")
        seg_frame = cv2.imread(seg_vis_path)
        if seg_frame is None:
            seg_frame = np.zeros_like(input_frame)
        if seg_frame.shape != input_frame.shape:
            seg_frame = cv2.resize(seg_frame, (width, height))

        combined = np.hstack((input_frame, seg_frame))
        out.write(combined)

    out.release()

def create_final_combined_video(input_frames_dir, pose_frames_dir, seg_output_dir, output_video_path, fps):
    """
    Create a 2x2 grid video:
    Top-left: original
    Top-right: pose estimation
    Bottom-left: segmentation visualization
    Bottom-right: combined (original + pose + segmentation mask overlay)
    """
    frame_files = sorted([f for f in os.listdir(input_frames_dir) if f.endswith(".png")])
    if not frame_files:
        return

    # Check frame sizes
    first_input = cv2.imread(os.path.join(input_frames_dir, frame_files[0]))
    height, width, _ = first_input.shape

    # We'll read corresponding pose and segmentation vis frames
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width*2, height*2))

    for f in frame_files:
        base_name = os.path.splitext(f)[0]
        input_frame = cv2.imread(os.path.join(input_frames_dir, f))
        pose_frame = cv2.imread(os.path.join(pose_frames_dir, f))
        if pose_frame is None:
            pose_frame = np.zeros_like(input_frame)
        seg_vis_path = os.path.join(seg_output_dir, f"{base_name}_vis.png")
        seg_frame = cv2.imread(seg_vis_path)
        if seg_frame is None:
            seg_frame = np.zeros_like(input_frame)

        # Create combined (original + pose + segmentation overlay)
        # We can try a simple overlay: combine pose and seg over original.
        # Example: combined_frame = original * 0.5 + pose * 0.25 + seg * 0.25 (just as an example)
        combined_frame = input_frame.astype(float)*0.5 + pose_frame.astype(float)*0.25 + seg_frame.astype(float)*0.25
        combined_frame = combined_frame.astype(np.uint8)

        # Resize if needed (all should match input_frame size)
        if pose_frame.shape != input_frame.shape:
            pose_frame = cv2.resize(pose_frame, (width, height))
        if seg_frame.shape != input_frame.shape:
            seg_frame = cv2.resize(seg_frame, (width, height))
        if combined_frame.shape != input_frame.shape:
            combined_frame = cv2.resize(combined_frame, (width, height))

        # Construct grid
        top_row = np.hstack((input_frame, pose_frame))
        bottom_row = np.hstack((seg_frame, combined_frame))
        grid = np.vstack((top_row, bottom_row))

        out.write(grid)

    out.release()
