from ultralytics import YOLO
import torch
import os
import torch.multiprocessing
torch.multiprocessing.set_start_method('spawn', force=True)


# Create a directory to save results
os.makedirs('head_detection', exist_ok=True)

# Check if CUDA is available
device = 0 if torch.cuda.is_available() else 'cpu'
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Load a pre-trained YOLOv8 model
model = YOLO('yolov8s.pt')  

# Train the model with customized parameters
results = model.train(
    data='head_dataset.yaml',
    epochs=100,
    imgsz=640,
    batch=4,             # Adjust based on available GPU memory
    patience=20,
    save=True,
    device=device,
    workers=0,            # Adjust for better data loading speed
    project='head_detection',
    name='scut_head',
    exist_ok=True,
    pretrained=True,
    optimizer='AdamW',    # AdamW often works better for object detection
    lr0=0.001,
    weight_decay=0.0005,
    cos_lr=True,
    augment=True,
    mixup=0.0,            # Disable MixUp unless specifically beneficial
    mosaic=0.75,          # Use moderate mosaic augmentation
    verbose=True,
    seed=42
)

# Validate the model
metrics = model.val()
print(f"Validation results: {metrics}")

# Path to best model
model_path = os.path.join('head_detection', 'scut_head', 'weights', 'best.pt')
if os.path.exists(model_path):
    # Export the best model to ONNX format for deployment
    model.export(format='onnx')
    print(f"Model exported to ONNX format: {model_path}")
else:
    print("Best model not found, skipping export.")
