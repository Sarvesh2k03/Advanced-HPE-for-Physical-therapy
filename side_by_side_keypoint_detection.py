import cv2
import numpy as np
from ultralytics import YOLO

# Initialize YOLOv8 Pose Model using the correct path
model = YOLO(r'C:\Users\TheOl\Desktop\PhysioPose\scripts\yolov8n-pose.pt')

# Load the reference video (flex.mp4)
reference_video_path = r'C:\Users\TheOl\Desktop\PhysioPose\reference_videos\flex.mp4'
reference_video = cv2.VideoCapture(reference_video_path)

# Initialize webcam for live feed
live_video = cv2.VideoCapture(0)

# Check if the reference video opened successfully
if not reference_video.isOpened():
    print(f"Error: Could not open the reference video file at {reference_video_path}")
    exit()

# Check if the live video feed is opened successfully
if not live_video.isOpened():
    print("Error: Could not open the live video feed.")
    exit()

# Debug: Print video properties
print(f"Reference Video Resolution: {int(reference_video.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(reference_video.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
print(f"Live Video Resolution: {int(live_video.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(live_video.get(cv2.CAP_PROP_FRAME_HEIGHT))}")

# Confidence threshold for keypoints
confidence_threshold = 0.5

# Define the keypoints of interest for angle calculation (e.g., shoulder, elbow, wrist)
KEYPOINT_INDICES = {
    'left_shoulder': 5,
    'left_elbow': 7,
    'left_wrist': 9,
    'right_shoulder': 6,
    'right_elbow': 8,
    'right_wrist': 10
}

# Function to calculate the angle between three points (P1 -> P2 -> P3)
def calculate_angle(p1, p2, p3):
    v1 = p1 - p2
    v2 = p3 - p2
    cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))  # Clip to avoid floating point errors
    return np.degrees(angle)

# Function to extract keypoints from YOLO results
def extract_keypoints(results):
    keypoints = []
    if results[0].keypoints is not None:
        keypoints = results[0].keypoints.xy.cpu().numpy()  # Convert tensors to numpy arrays
    return np.array(keypoints)

# Function to calculate accuracy based on joint angles
def calculate_angle_accuracy(ref_keypoints, live_keypoints):
    if ref_keypoints.shape == live_keypoints.shape and len(ref_keypoints) > 0:
        ref_angles = []
        live_angles = []
        
        # Calculate angles for both arms
        for side in ['left', 'right']:
            try:
                ref_angle = calculate_angle(
                    ref_keypoints[KEYPOINT_INDICES[f'{side}_shoulder']],
                    ref_keypoints[KEYPOINT_INDICES[f'{side}_elbow']],
                    ref_keypoints[KEYPOINT_INDICES[f'{side}_wrist']]
                )
                live_angle = calculate_angle(
                    live_keypoints[KEYPOINT_INDICES[f'{side}_shoulder']],
                    live_keypoints[KEYPOINT_INDICES[f'{side}_elbow']],
                    live_keypoints[KEYPOINT_INDICES[f'{side}_wrist']]
                )

                ref_angles.append(ref_angle)
                live_angles.append(live_angle)

            except IndexError:
                # Handle the case where keypoints might not be available
                continue

        if ref_angles and live_angles:
            angle_diff = np.abs(np.array(ref_angles) - np.array(live_angles))
            normalized_diff = np.clip(angle_diff / 180, 0, 1)  # Normalize the difference
            accuracy = 100 - np.mean(normalized_diff) * 100  # Scale to percentage
            return max(0, round(accuracy, 2))  # Ensure accuracy is at least 0%
    
    return 50  # Return baseline accuracy (50%) if we don't have valid keypoints for comparison

# Function to draw keypoints, ensuring valid keypoints are present
def draw_keypoints(frame, results, color):
    """ Draw keypoints on the frame, only if valid keypoints are detected """
    if results[0].keypoints is not None:
        keypoints = results[0].keypoints.xy  # Absolute (x, y) coordinates
        confidences = results[0].keypoints.conf  # Confidence values

        # Check if keypoints and confidences are valid before drawing
        if keypoints is not None and confidences is not None:
            for person_keypoints, person_confidences in zip(keypoints, confidences):
                for (x, y), conf in zip(person_keypoints, person_confidences):
                    # Filter by confidence and ensure keypoints are within the frame
                    if conf >= confidence_threshold and 0 <= x <= frame.shape[1] and 0 <= y <= frame.shape[0]:
                        cv2.circle(frame, (int(x), int(y)), 5, color, -1)  # Draw keypoints with specified color

# Loop through each frame in the reference and live video
while live_video.isOpened():
    # Read frames from both videos
    ret_ref, ref_frame = reference_video.read()
    ret_live, live_frame = live_video.read()

    # If reference video ends, loop it back to the beginning
    if not ret_ref or ref_frame is None:
        reference_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret_ref, ref_frame = reference_video.read()

    # Check if live frame was successfully read
    if not ret_live or live_frame is None:
        print("Error: Failed to capture frames from the live video.")
        break

    # Resize both frames to fit within the screen resolution while maintaining aspect ratio
    ref_frame_resized = cv2.resize(ref_frame, (640, 480))  # Fixed size for simplicity
    live_frame_resized = cv2.resize(live_frame, (640, 480))

    # Use YOLO to predict keypoints in both resized frames
    ref_results = model(ref_frame_resized, show=False)
    live_results = model(live_frame_resized, show=False)

    # Extract keypoints from the results
    ref_keypoints = extract_keypoints(ref_results)
    live_keypoints = extract_keypoints(live_results)

    # Calculate the accuracy based on angles between keypoints
    accuracy = calculate_angle_accuracy(ref_keypoints, live_keypoints)

    # Draw keypoints on the reference frame (red)
    draw_keypoints(ref_frame_resized, ref_results, (0, 0, 255))

    # Draw keypoints on the live frame (green)
    draw_keypoints(live_frame_resized, live_results, (0, 255, 0))

    # Combine the frames side-by-side for comparison
    combined_frame = cv2.hconcat([ref_frame_resized, live_frame_resized])

    # Display the combined frame
    cv2.imshow('Reference vs Live Video with Keypoints', combined_frame)

    # Print the accuracy in the console
    print(f"Accuracy: {accuracy}%")

    # Press 'q' to quit the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video resources
reference_video.release()
live_video.release()
cv2.destroyAllWindows()
