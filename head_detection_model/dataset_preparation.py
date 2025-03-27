import os
import shutil
import random
import xml.etree.ElementTree as ET
from glob import glob
import cv2

# Define paths - adjust these to your actual paths
dataset_root = r"C:\Users\RAMESWAR BISOYI\Documents\DEV\CrowdSense\ml\CrowdSense\head_detection_model\SCUT_HEAD_Part_A"  # Change this to your dataset location
yolo_dataset_dir = "head_detection_dataset"




# Folder structure in SCUT-HEAD
annotations_dir = os.path.join(dataset_root, "Annotations")
images_dir = os.path.join(dataset_root, "JPEGImages")
imagesets_dir = os.path.join(dataset_root, "ImageSets", "Main")

# Create directory structure for YOLO format
os.makedirs(f"{yolo_dataset_dir}/images/train", exist_ok=True)
os.makedirs(f"{yolo_dataset_dir}/images/val", exist_ok=True)
os.makedirs(f"{yolo_dataset_dir}/labels/train", exist_ok=True)
os.makedirs(f"{yolo_dataset_dir}/labels/val", exist_ok=True)

# Function to convert XML annotation to YOLO format
def convert_annotation(xml_file, image_width, image_height):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    yolo_annotations = []
    
    # Process each object (person/head) in the XML
    for obj in root.findall('object'):
        name = obj.find('name').text
        
        # We only care about the 'person' class (which is actually head in this dataset)
        if name != 'person':
            continue
            
        # Get bounding box coordinates
        bbox = obj.find('bndbox')
        xmin = float(bbox.find('xmin').text)
        ymin = float(bbox.find('ymin').text)
        xmax = float(bbox.find('xmax').text)
        ymax = float(bbox.find('ymax').text)
        
        # Convert to YOLO format: class_id, x_center, y_center, width, height (all normalized)
        width = xmax - xmin
        height = ymax - ymin
        x_center = (xmin + width/2) / image_width
        y_center = (ymin + height/2) / image_height
        norm_width = width / image_width
        norm_height = height / image_height
        
        # Class ID 0 for head
        yolo_annotations.append(f"0 {x_center} {y_center} {norm_width} {norm_height}")
    
    return yolo_annotations

# Read train and val split from ImageSets
def read_image_set(file_path):
    with open(file_path, 'r') as f:
        return [line.strip() for line in f.readlines()]

# Get the train and validation splits
train_ids = read_image_set(os.path.join(imagesets_dir, "train.txt"))
val_ids = read_image_set(os.path.join(imagesets_dir, "val.txt"))

print(f"Found {len(train_ids)} training images and {len(val_ids)} validation images")

# Process training images
for img_id in train_ids:
    # Image path
    img_path = os.path.join(images_dir, f"{img_id}.jpg")
    if not os.path.exists(img_path):
        print(f"Warning: Image {img_path} not found")
        continue
        
    # XML annotation path
    xml_path = os.path.join(annotations_dir, f"{img_id}.xml")
    if not os.path.exists(xml_path):
        print(f"Warning: Annotation {xml_path} not found")
        continue
    
    # Get image dimensions
    img = cv2.imread(img_path)
    if img is None:
        print(f"Warning: Could not read image {img_path}")
        continue
        
    h, w = img.shape[:2]
    
    # Convert and save annotation
    yolo_annotations = convert_annotation(xml_path, w, h)
    
    # Only process if annotations were found
    if yolo_annotations:
        # Copy image
        shutil.copy(img_path, f"{yolo_dataset_dir}/images/train/{img_id}.jpg")
        
        # Save YOLO annotation
        with open(f"{yolo_dataset_dir}/labels/train/{img_id}.txt", 'w') as f:
            f.write('\n'.join(yolo_annotations))

# Process validation images
for img_id in val_ids:
    # Image path
    img_path = os.path.join(images_dir, f"{img_id}.jpg")
    if not os.path.exists(img_path):
        print(f"Warning: Image {img_path} not found")
        continue
        
    # XML annotation path
    xml_path = os.path.join(annotations_dir, f"{img_id}.xml")
    if not os.path.exists(xml_path):
        print(f"Warning: Annotation {xml_path} not found")
        continue
    
    # Get image dimensions
    img = cv2.imread(img_path)
    if img is None:
        print(f"Warning: Could not read image {img_path}")
        continue
        
    h, w = img.shape[:2]
    
    # Convert and save annotation
    yolo_annotations = convert_annotation(xml_path, w, h)
    
    # Only process if annotations were found
    if yolo_annotations:
        # Copy image
        shutil.copy(img_path, f"{yolo_dataset_dir}/images/val/{img_id}.jpg")
        
        # Save YOLO annotation
        with open(f"{yolo_dataset_dir}/labels/val/{img_id}.txt", 'w') as f:
            f.write('\n'.join(yolo_annotations))

# Count how many images were processed
train_count = len(glob(f"{yolo_dataset_dir}/images/train/*.jpg"))
val_count = len(glob(f"{yolo_dataset_dir}/images/val/*.jpg"))

print(f"Successfully processed {train_count} training images and {val_count} validation images")
print(f"Dataset prepared in {yolo_dataset_dir}")