from ultralytics import YOLO

# 1. Load your YOLO-Pose model (.pt file)
model = YOLO("models/yolo11n-pose-pruned-75.pt") 

# 2. Export to INT8 TFLite
# The 'data' argument is used for calibration to maintain accuracy
model.export(
    format="tflite", 
    int8=True, 
    data="coco8-pose.yaml",
    device=0 # Use a small dataset to calibrate weights
)