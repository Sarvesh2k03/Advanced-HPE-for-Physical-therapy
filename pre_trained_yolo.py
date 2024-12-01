from ultralytics import YOLO
import cv2
import os

# Load the pre-trained YOLOv8 model for pose detection
model = YOLO('yolov8n-pose.pt')  # Use 'yolov8s-pose.pt' for a larger model if needed

# Predict keypoints on the image
results = model.predict(source=r'C:\Users\TheOl\Desktop\PhysioPose\reference_videos\image.png')

# Create the output directory if it doesn't exist
output_dir = r'C:\Users\TheOl\Desktop\PhysioPose\output_images'
os.makedirs(output_dir, exist_ok=True)

# Iterate over the results and save each one
for i, result in enumerate(results):
    # Plot the results onto the image
    img_with_keypoints = result.plot()  # This draws the keypoints on the image

    # Define the output file path
    output_file = os.path.join(output_dir, f"result_{i+1}.png")

    # Save the image using OpenCV
    cv2.imwrite(output_file, img_with_keypoints)
    print(f"Saved result {i+1} to '{output_file}'")
