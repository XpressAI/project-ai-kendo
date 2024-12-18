import cv2
import os
import numpy as np

def combine_frames_into_video(input_frames_dir, processed_frames_dir, output_video_path, fps):
    """
    Combine input frames and processed frames into one video.
    For simplicity, let's just overlay them side by side.
    """
    frame_files = sorted([f for f in os.listdir(input_frames_dir) if f.endswith(".png")])
    if not frame_files:
        return

    # Check size
    first_input_frame = cv2.imread(os.path.join(input_frames_dir, frame_files[0]))
    first_processed_frame = cv2.imread(os.path.join(processed_frames_dir, frame_files[0]))
    height, width, _ = first_input_frame.shape

    # Ensure processed frame matches size; if not, resize
    processed_height, processed_width, _ = first_processed_frame.shape
    if (processed_height != height) or (processed_width != width):
        # Resize processed to match
        first_processed_frame = cv2.resize(first_processed_frame, (width, height))

    # Output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width*2, height))

    for f in frame_files:
        input_frame = cv2.imread(os.path.join(input_frames_dir, f))
        processed_frame = cv2.imread(os.path.join(processed_frames_dir, f))
        if processed_frame is None:
            # If no processed frame, use a blank frame
            processed_frame = np.zeros_like(input_frame)
        if processed_frame.shape != input_frame.shape:
            processed_frame = cv2.resize(processed_frame, (input_frame.shape[1], input_frame.shape[0]))

        combined = np.hstack((input_frame, processed_frame))
        out.write(combined)

    out.release()

def create_processed_video(processed_frames_dir, output_video_path, fps):
    """
    Save the processed frames (pose + shinai only) as a standalone video.
    """
    frame_files = sorted([f for f in os.listdir(processed_frames_dir) if f.endswith(".png")])
    if not frame_files:
        return

    # Check frame size
    first_frame = cv2.imread(os.path.join(processed_frames_dir, frame_files[0]))
    height, width, _ = first_frame.shape

    # Output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for f in frame_files:
        frame_path = os.path.join(processed_frames_dir, f)
        frame = cv2.imread(frame_path)
        if frame is not None:
            out.write(frame)

    out.release()
