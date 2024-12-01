import cv2
import numpy as np
from ultralytics import YOLO
import time

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

# Get the fps of the reference video
fps = reference_video.get(cv2.CAP_PROP_FPS)
frame_time = 1 / fps  # Time per frame in seconds

# Variables to track distance, keypoint errors, and frame processing
min_distance = float('inf')
start_time = time.time()
total_frames = 0  # To count the total number of frames processed
processing_times = []  # List to store processing times for each frame
keypoint_errors_list = []  # List to store keypoint errors per frame

# Function to extract keypoints from YOLO results
def extract_keypoints(results):
    keypoints = []
    if results[0].keypoints is not None:
        keypoints = results[0].keypoints.xy.cpu().numpy()  # Convert tensors to numpy arrays
    return np.array(keypoints)

# Updated function to draw keypoints on a frame
def draw_keypoints(frame, keypoints, color=(0, 255, 0)):
    for keypoint_set in keypoints:
        for keypoint in keypoint_set:
            if len(keypoint) >= 2:  # Ensure the keypoint has at least (x, y) coordinates
                x, y = keypoint[:2]  # Extract only the (x, y) coordinates
                if x > 0 and y > 0:  # Draw only if valid keypoints exist
                    cv2.circle(frame, (int(x), int(y)), 5, color, -1)

# Function to calculate mean error per keypoint
def calculate_mean_error(ref_keypoints, live_keypoints):
    if ref_keypoints.size == 0 or live_keypoints.size == 0:
        return np.array([])  # Return empty if no keypoints detected

    # Ensure both have the same number of keypoints
    num_keypoints = min(ref_keypoints.shape[0], live_keypoints.shape[0])
    if num_keypoints == 0:
        return np.array([])

    # Filter out undetected keypoints
    ref_keypoints_filtered = ref_keypoints[:num_keypoints]
    live_keypoints_filtered = live_keypoints[:num_keypoints]

    # Calculate individual keypoint errors
    keypoint_errors = np.linalg.norm(ref_keypoints_filtered - live_keypoints_filtered, axis=1)
    return keypoint_errors

# Function to calculate weighted Euclidean distance between two sets of keypoints
def calculate_distance(ref_keypoints, live_keypoints):
    if ref_keypoints.size == 0 or live_keypoints.size == 0:
        return float('inf')  # Return infinite distance if no keypoints detected

    # Ensure both have the same number of keypoints for comparison
    num_keypoints = min(ref_keypoints.shape[0], live_keypoints.shape[0])  # Find the minimum number of detected keypoints
    if num_keypoints == 0:
        return float('inf')

    # Filter out undetected keypoints
    ref_keypoints_filtered = ref_keypoints[:num_keypoints]
    live_keypoints_filtered = live_keypoints[:num_keypoints]

    # Define weights, prioritizing wrist, elbow, and shoulder
    keypoint_weights = np.ones(num_keypoints)  # Start with equal weights
    keypoint_indices = [2, 5, 8]  # Indices for wrist, elbow, shoulder; adjust if needed
    for idx in keypoint_indices:
        if idx < num_keypoints and np.any(ref_keypoints_filtered[idx]) and np.any(live_keypoints_filtered[idx]):
            keypoint_weights[idx] = 2  # Give higher weight to these keypoints

    distances = np.linalg.norm(ref_keypoints_filtered - live_keypoints_filtered, axis=1)  # Calculate distances for each keypoint
    weighted_distances = distances * keypoint_weights  # Apply weights
    return np.mean(weighted_distances)  # Return the average weighted distance

# Function to calculate accuracy based on distance
def calculate_accuracy(distance):
    # Define accuracy thresholds based on distance ranges
    if distance < 900:  # Close match
        return 100.0
    elif distance < 1300:  # Moderate match
        return max(60.0, 100.0 - (distance - 900) * (40.0 / 400))  # Linear decrease from 100 to 60
    else:  # Far match
        return max(0.0, 100.0 - (distance - 1300) * (100.0 / 200))  # Linearly decrease accuracy

# Function to provide improvement suggestions
def suggest_improvement(accuracy):
    if accuracy < 80:
        return "Posture or exercise not done properly."
    else:
        return "Good job! To improve accuracy, focus on:\n- Maintaining proper shoulder alignment.\n- Ensuring your arms are fully extended during the exercise.\n- Avoiding excessive leaning."

while True:
    # Read frames from both videos
    ret_ref, ref_frame = reference_video.read()
    ret_live, live_frame = live_video.read()

    # Check if live frame was successfully read
    if not ret_live:
        print("Error: Failed to capture frames from the live video.")
        break

    # If reference video ends, break the loop after 20 seconds
    if not ret_ref or time.time() - start_time > 20:
        break

    # Measure the time taken to process each frame
    frame_start_time = time.time()

    # Use YOLO to predict keypoints in both frames
    ref_results = model(ref_frame, show=False)
    live_results = model(live_frame, show=False)

    # Extract keypoints from the results
    ref_keypoints = extract_keypoints(ref_results)
    live_keypoints = extract_keypoints(live_results)

    # Calculate distance for real-time accuracy
    distance = calculate_distance(ref_keypoints, live_keypoints)

    # Update minimum distance if current distance is lower
    if distance < min_distance:
        min_distance = distance

    # Calculate accuracy based on distance
    final_accuracy = calculate_accuracy(min_distance)

    # Draw keypoints on reference and live frames
    if ref_keypoints.size > 0:
        draw_keypoints(ref_frame, [ref_keypoints], color=(0, 0, 255))  # Red keypoints for reference frame
    if live_keypoints.size > 0:
        draw_keypoints(live_frame, [live_keypoints], color=(0, 255, 0))  # Green keypoints for live frame

    # Display distance on the live frame
    cv2.putText(live_frame, f'Distance: {distance:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Suggest improvement based on accuracy
    suggestion = suggest_improvement(final_accuracy)

    # Display suggestion on the live frame
    cv2.putText(live_frame, suggestion, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2 if final_accuracy < 80 else 1)

    # Display total keypoints tracked
    print(f'Number of Keypoints Tracked per Frame: {ref_keypoints.shape[0]}')

    # Frame processing time
    frame_processing_time = (time.time() - frame_start_time) * 1000  # In milliseconds
    processing_times.append(frame_processing_time)
    total_frames += 1

    # Display the frames
    cv2.imshow('Reference Frame', ref_frame)
    cv2.imshow('Live Frame', live_frame)

    # Press 'q' to quit the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the session ends, calculate performance metrics
if keypoint_errors_list:
    all_errors = np.vstack(keypoint_errors_list)  # Combine all frame keypoint errors
    mean_error_per_keypoint = np.mean(all_errors, axis=0)  # Mean error for each keypoint
    print(f'Mean Error per Keypoint: {mean_error_per_keypoint}')

# Calculate average FPS and processing time
average_fps = total_frames / (time.time() - start_time)
average_processing_time = np.mean(processing_times)

# Print the model performance metrics
print(f'Minimum Distance Achieved: {min_distance:.2f}')
print(f'Final Accuracy: {final_accuracy:.2f}%')
print(f'Total Frames Processed: {total_frames}')
print(f'Average FPS: {average_fps:.2f}')
print(f'Average Frame Processing Time: {average_processing_time:.2f} ms')

# Release video resources
reference_video.release()
live_video.release()
cv2.destroyAllWindows()
