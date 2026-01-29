from ultralytics import YOLO

# 1. Load your existing model
model = YOLO("yolo11n-pose.pt")

# 2. Export to NCNN (FP16)
# This converts weights to 16-bit floats. It's the best for Raspberry Pi 4 CPU.
model.export(format="ncnn", imgsz=640, half=True)

# 3. Export to TFLite (INT8)
# This converts weights to 8-bit integers. 
# NOTE: 'data' is required for INT8 to "calibrate" the weights using sample images.
model.export(format="tflite", imgsz=640, int8=True, data="coco8-pose.yaml")

print("✅ Conversion Complete!")