import cv2
import mediapipe as mp

# Initialize MediaPipe pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=True)

# Load or capture the video source
cap = cv2.VideoCapture(0)  # Replace with 0 for webcam input

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the image color (MediaPipe uses RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    # Draw pose landmarks
    if results.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
        )

    # Display frame
    cv2.imshow('Kendo Pose Tracking', frame)
    if cv2.waitKey(5) & 0xFF == 27:
        break

pose.close()
cap.release()
cv2.destroyAllWindows()
