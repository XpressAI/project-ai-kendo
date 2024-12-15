import cv2
import mediapipe as mp

# Initialize BlazePose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Video processing function
def process_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Output video codec
    out = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to RGB for BlazePose
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        # Draw landmarks on frame
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

        # Initialize output video writer
        if out is None:
            height, width, _ = frame.shape
            out = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))

        # Write the processed frame to output
        out.write(frame)

    cap.release()
    if out:
        out.release()
    print("Video processing complete!")

# Example usage
process_video("input_video.mp4", "output_video.mp4")
