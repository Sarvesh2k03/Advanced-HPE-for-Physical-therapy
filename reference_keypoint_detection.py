import cv2
from ultralytics import YOLO
import numpy as np

# Initialize YOLOv8 Pose Model using the correct path
model = YOLO(r'C:\Users\TheOl\Desktop\PhysioPose\scripts\yolov8n-pose.pt')

# Load the reference video
reference_video_path = r'C:\Users\TheOl\Desktop\PhysioPose\reference_videos\flex.mp4'  # Update to the path of your video
reference_video = cv2.VideoCapture(reference_video_path)

# Check if the reference video opened successfully
if not reference_video.isOpened():
    print(f"Error: Could not open the video file at {reference_video_path}")
    exit()

# Debug information about the video
frame_width = int(reference_video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(reference_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Video Resolution: {frame_width}x{frame_height}")

# Loop through each frame in the video
while reference_video.isOpened():
    ret, frame = reference_video.read()
    if not ret or frame is None:
        print("End of video or failed to capture frame.")
        break

    # Use YOLO to predict keypoints in the current frame
    results = model(frame, show=False)

    # Check if keypoints are detected and draw them on the frame
    if results[0].keypoints is not None:
        print(f"Number of persons detected: {len(results[0].keypoints)}")  # Debug: Number of persons detected

        # Use the `xy` attribute to access absolute (x, y) coordinates of keypoints
        keypoints = results[0].keypoints.xy  # This gives an array of shape (1, 17, 2) for (x, y)

        # Debug: Print the shape and contents of keypoints
        print(f"Keypoints Array Shape: {keypoints.shape}")
        print(f"Keypoints Array Contents:\n{keypoints}")

        # Draw each keypoint on the frame
        for person_keypoints in keypoints:
            for (x, y) in person_keypoints:  # Unpack (x, y) values
                # Ensure x, y are within the frame bounds
                if 0 <= x <= frame.shape[1] and 0 <= y <= frame.shape[0]:
                    cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)  # Draw green circle at keypoint

    # Show the frame with keypoints
    cv2.imshow('Reference Video with Keypoints', frame)

    # Add a delay to ensure the window is rendered correctly
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video and destroy all windows
reference_video.release()
cv2.destroyAllWindows()
