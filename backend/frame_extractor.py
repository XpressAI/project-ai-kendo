import cv2
import os

def extract_frames(video_path, output_dir):
    """
    Extracts all frames from the video and saves them in output_dir.
    Returns fps, width, height, frame_count.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(output_dir, f"frame_{frame_idx:06d}.png")
        cv2.imwrite(frame_path, frame)
        frame_idx += 1

    cap.release()
    return fps, width, height, frame_count
