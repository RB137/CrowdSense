import torch
from torch.utils.tensorboard import SummaryWriter
from ultralytics import YOLO

# Load YOLO model from best.pt
model = YOLO(r"C:\Users\RAMESWAR BISOYI\Documents\DEV\CrowdSense\ml\CrowdSense\head_detection_model\head_detection\scut_head\weights\best.pt").model

# Move model to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Verify model is loaded
print(model)

# Initialize TensorBoard writer
writer = SummaryWriter("runs/yolo_model")

# Dummy input for logging
dummy_input = torch.randn(1, 3, 640, 640).to(device)

# Make a forward pass (ensures graph is properly built)
output = model(dummy_input)

# Log model graph
writer.add_graph(model, dummy_input)
writer.flush()
writer.close()

print("TensorBoard data saved! Run 'tensorboard --logdir=runs' to visualize.")


print(model)

