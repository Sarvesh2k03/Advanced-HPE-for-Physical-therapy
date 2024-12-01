import cv2
import numpy as np
from ultralytics import YOLO

# Initialize YOLOv8 Pose Model for keypoint detection
model = YOLO('yolov8n-pose.pt')  # Using the small pre-trained YOLOv8 model

# Load the reference video (flex.mp4)
reference_video_path = "../reference_videos/flex.mp4"
reference_video = cv2.VideoCapture(reference_video_path)

# Initialize webcam for live feed
live_video = cv2.VideoCapture(0)

# Function to extract keypoints from YOLOv8 results and convert to numpy arrays
def extract_keypoints_from_yolo(results):
    keypoints = []
    if results[0].keypoints is not None:
        keypoints = results[0].keypoints.cpu().numpy()  # Convert tensors to numpy arrays
        keypoints = keypoints[0][:, :2]  # Get only the (x, y) coordinates for the first person
    return keypoints

# Function to calculate similarity between keypoints
def calculate_similarity(ref_keypoints, live_keypoints):
    # Ensure both keypoints arrays are non-empty and have the same shape
    if len(ref_keypoints) > 0 and len(live_keypoints) > 0:
        if ref_keypoints.shape == live_keypoints.shape:
            distances = np.linalg.norm(ref_keypoints - live_keypoints, axis=1)
            accuracy = 100 - np.mean(distances) * 100  # Scale to percentage
            return max(0, round(accuracy, 2))  # Ensure minimum accuracy is 0%
        else:
            print("Keypoint shape mismatch between reference and live frames.")
            return 0  # Return 0 accuracy if shapes don't match
    else:
        print("One or both keypoint sets are empty.")
        return 0  # Return 0 accuracy if any keypoints are missing

# Function to draw an accuracy bar on the side of the video
def draw_accuracy_bar(frame, accuracy, max_height=300):
    bar_width = 50
    bar_height = int((accuracy / 100) * max_height)
    bar_color = (0, 255, 0)  # Green for accuracy
    background_color = (50, 50, 50)  # Dark gray background

    # Create a blank image for the bar
    bar_frame = np.ones((max_height, bar_width, 3), dtype=np.uint8) * 50
    cv2.rectangle(bar_frame, (0, max_height - bar_height), (bar_width, max_height), bar_color, -1)

    # Add the accuracy percentage text
    cv2.putText(bar_frame, f'{accuracy}%', (5, max_height - bar_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Place the bar next to the frame
    return cv2.hconcat([frame, bar_frame])

# Real-time loop: Read frames from the live webcam
while live_video.isOpened():
    # Read frames from the live webcam
    ret_live, live_frame = live_video.read()

    # Check if frames are successfully read
    if not ret_live or live_frame is None:
        print("Error reading live feed.")
        break

    # If reference video ends, reset it to the beginning
    ret_ref, ref_frame = reference_video.read()
    if not ret_ref or ref_frame is None:
        reference_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret_ref, ref_frame = reference_video.read()

    # Use YOLO to predict keypoints in both the reference and live frames
    ref_results = model(ref_frame, show=False)
    live_results = model(live_frame, show=False)

    # Extract keypoints from YOLO results
    ref_keypoints = extract_keypoints_from_yolo(ref_results)
    live_keypoints = extract_keypoints_from_yolo(live_results)

    # Calculate the real-time similarity (accuracy) between reference and live keypoints
    accuracy = calculate_similarity(ref_keypoints, live_keypoints)

    # Draw keypoints on the live frame
    live_display_frame = live_frame.copy()
    for (x, y) in live_keypoints:
        cv2.circle(live_display_frame, (int(x * live_display_frame.shape[1]), int(y * live_display_frame.shape[0])), 5, (0, 255, 0), -1)

    # Display the real-time accuracy on the live video feed
    cv2.putText(live_display_frame, f'Accuracy: {accuracy}%', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Add the accuracy bar to the right side of the live frame
    live_frame_with_bar = draw_accuracy_bar(live_display_frame, accuracy)

    # Display the final frame with the accuracy bar
    cv2.imshow('Live Feed with Real-Time Accuracy', live_frame_with_bar)

    # Break if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
reference_video.release()
live_video.release()
cv2.destroyAllWindows()
