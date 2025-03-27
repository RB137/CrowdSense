import cv2
import torch
from ultralytics import YOLO
import os
import numpy as np
from glob import glob
import argparse

# Parse arguments
parser = argparse.ArgumentParser(description='Test head detection model')
parser.add_argument('--model', type=str, default='head_detection/scut_head/weights/best.pt', help='Path to model weights')
parser.add_argument('--img-dir', type=str, default='head_detection_dataset/images/val', help='Directory with test images')
parser.add_argument('--conf', type=float, default=0.3, help='Confidence threshold')
parser.add_argument('--output-dir', type=str, default='test_results', help='Output directory')
args = parser.parse_args()

# Load the trained model
print(f"Loading model from {args.model}")
model = YOLO(args.model)

# Create output directory
os.makedirs(args.output_dir, exist_ok=True)

# Get test images
test_images = glob(f"{args.img_dir}/*.jpg")
print(f"Found {len(test_images)} test images")

# Process each test image
total_heads = 0
for img_path in test_images:
    # Read the image
    img = cv2.imread(img_path)
    if img is None:
        print(f"Warning: Could not read image {img_path}")
        continue
    
    # Run inference
    results = model(img, conf=args.conf)
    
    # Create a copy for visualization
    img_result = img.copy()
    
    # Get detected heads
    head_count = 0
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Get box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            
            # Calculate head center
            head_x = (x1 + x2) // 2
            head_y = (y1 + y2) // 2
            radius = max(5, min(x2-x1, y2-y1) // 2)
            
            # Draw circle on head
            cv2.circle(img_result, (head_x, head_y), radius, (0, 255, 0), 2)
            
            # Draw confidence (optional)
            cv2.putText(img_result, f"{conf:.2f}", (x1, y1-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            head_count += 1
    
    total_heads += head_count
    
    # Add head count
    cv2.putText(img_result, f"Head Count: {head_count}", (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Save result
    filename = os.path.basename(img_path)
    cv2.imwrite(f"{args.output_dir}/{filename}", img_result)
    
    print(f"Processed {filename} - Found {head_count} heads")

print(f"Testing complete. Total heads detected: {total_heads}")
print(f"Results saved in '{args.output_dir}' folder")