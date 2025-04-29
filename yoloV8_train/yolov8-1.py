# Ultralytics YOLO 🚀, GPL-3.0 license
from ultralytics import YOLO

# Load a COCO-pretrained YOLOv8n model
model = YOLO("956PT_0428_best.pt")

# Display model information (optional)
model.info()

# Train the model on the COCO8 example dataset for 100 epochs
results = model.train(data="coco8.yaml", epochs=10, imgsz=640)

