# Ultralytics YOLO 🚀, GPL-3.0 license
from ultralytics import YOLO

# Load a model
model = YOLO(r"C:\Users\Gillion-BennyWinNB\Desktop\2025HIWIN_poolball\yoloV8_train\956PT_0428_best.pt")  # pretrained YOLO11n model

#Run batched inference on a list of images
results = model(r"C:\Users\Gillion-BennyWinNB\Desktop\2025HIWIN_poolball\take_images\photo_000.png")  # return a list of Results objects

# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()  # display to screen
