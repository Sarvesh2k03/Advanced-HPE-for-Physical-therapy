import cv2
import numpy as np
from ultralytics import YOLO
from scipy.spatial.distance import cdist
from fastdtw import fastdtw

# Initialize YOLOv8 Pose Model for keypoint detection
model = YOLO(r'C:\Users\TheOl\Desktop\PhysioPose\scripts\yolo11n.pt')  # Update with your trained model path

# Load reference video
reference_video_path = r'C:\Users\TheOl\Desktop\PhysioPose\reference_videos\flex.mp4'  # Update with your video path
reference_video = cv2.VideoCapture(reference_video_path)

# Initialize webcam/live feed for real-time analysis
live_video = cv2.VideoCapture(0)

# Function to extract keypoints from YOLO results
def extract_keypoints(results):
    keypoints = []
    if results[0].keypoints is not None:
        keypoints = results[0].keypoints.xy.cpu().numpy()  # Convert tensors to numpy arrays
    return np.array(keypoints)

# Function to compute Dynamic Time Warping (DTW) distance
def compute_dtw(ref_keypoints, live_keypoints):
    distance, _ = fastdtw(ref_keypoints, live_keypoints, dist=cdist)  # Compute DTW distance
    return distance

# Function to visualize keypoints on the frame
def draw_keypoints(frame, keypoints):
    for point in keypoints:
        if point[0] > 0 and point[1] > 0:  # Only draw valid keypoints
            cv2.circle(frame, (int(point[0]), int(point[1])), 5, (0, 255, 0), -1)  # Draw green keypoints

while reference_video.isOpened() and live_video.isOpened():
    # Read frames from both reference video and live video
    ret_ref, ref_frame = reference_video.read()
    ret_live, live_frame = live_video.read()

    if not ret_ref or not ret_live:
        # Loop the reference video when it ends
        reference_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    # Get keypoints from YOLO pose estimation for both reference and live frames
    ref_results = model(ref_frame, show=False)
    live_results = model(live_frame, show=False)
    
    ref_keypoints = extract_keypoints(ref_results)
    live_keypoints = extract_keypoints(live_results)
    
    # Print detected keypoints for debugging
    print("Reference Keypoints:", ref_keypoints)
    print("Live Keypoints:", live_keypoints)

    # Check for keypoints detection
    if ref_keypoints.size == 0:
        print("No keypoints detected in reference video")
        continue
    if live_keypoints.size == 0:
        print("No keypoints detected in live video")
        continue
    
    # Compute DTW distance to measure similarity
    dtw_distance = compute_dtw(ref_keypoints, live_keypoints)
    print("DTW Distance:", dtw_distance)

    # Draw keypoints on the frames for visualization
    draw_keypoints(ref_frame, ref_keypoints)
    draw_keypoints(live_frame, live_keypoints)

    # Display the reference and live frames
    cv2.imshow('Reference Frame', ref_frame)
    cv2.imshow('Live Frame', live_frame)

    # Check for 'q' key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video resources and close windows
reference_video.release()
live_video.release()
cv2.destroyAllWindows()
