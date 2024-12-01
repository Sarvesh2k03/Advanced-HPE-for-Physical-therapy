import cv2

# Load the reference video
reference_video_path = "../reference_videos/flex.mp4"  # Change to Pendulum.mp4 as needed
reference_video = cv2.VideoCapture(reference_video_path)

# Initialize webcam for live feed
live_video = cv2.VideoCapture(0)

while reference_video.isOpened() and live_video.isOpened():
    # Read frames from both videos
    ret_ref, ref_frame = reference_video.read()
    ret_live, live_frame = live_video.read()

    if not ret_ref or not ret_live:
        break

    # Display side-by-side
    combined_frame = cv2.hconcat([ref_frame, live_frame])
    cv2.imshow('Reference vs Live Feed', combined_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

reference_video.release()
live_video.release()
cv2.destroyAllWindows()

import cv2
import mediapipe as mp

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Load the reference video
reference_video_path = "../reference_videos/flex.mp4"  # Change to Pendulum.mp4 as needed
reference_video = cv2.VideoCapture(reference_video_path)

# Initialize webcam for live feed
live_video = cv2.VideoCapture(0)

# Check if reference video and webcam are opened successfully
if not reference_video.isOpened():
    print(f"Error: Cannot open reference video {reference_video_path}")
    exit()
if not live_video.isOpened():
    print("Error: Cannot open webcam")
    exit()

while reference_video.isOpened() and live_video.isOpened():
    # Read frames from both videos
    ret_ref, ref_frame = reference_video.read()
    ret_live, live_frame = live_video.read()

    # Check if frames are valid
    if not ret_ref or not ret_live or ref_frame is None or live_frame is None:
        print("One of the frames is empty or failed to capture. Exiting...")
        break

    # Check if both frames have the same number of rows (height)
    if ref_frame.shape[0] != live_frame.shape[0]:
        # Resize live frame to match reference frame's height
        live_frame = cv2.resize(live_frame, (ref_frame.shape[1], ref_frame.shape[0]))

    # Ensure both frames have the same number of columns (width)
    if ref_frame.shape != live_frame.shape:
        print(f"Resized live frame to match reference dimensions: {ref_frame.shape}")
        
    # Ensure both frames have the same data type
    if ref_frame.dtype != live_frame.dtype:
        live_frame = live_frame.astype(ref_frame.dtype)

    # Perform pose estimation on both frames
    ref_rgb = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2RGB)
    live_rgb = cv2.cvtColor(live_frame, cv2.COLOR_BGR2RGB)
    ref_results = pose.process(ref_rgb)
    live_results = pose.process(live_rgb)

    # Draw landmarks on frames
    if ref_results.pose_landmarks:
        mp_drawing.draw_landmarks(ref_frame, ref_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    if live_results.pose_landmarks:
        mp_drawing.draw_landmarks(live_frame, live_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Display the side-by-side comparison
    combined_frame = cv2.hconcat([ref_frame, live_frame])
    cv2.imshow('Reference vs Live Feed with Pose Estimation', combined_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
reference_video.release()
live_video.release()
cv2.destroyAllWindows()



