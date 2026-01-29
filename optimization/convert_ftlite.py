from ultralytics import YOLO

# 1. Load your trained PyTorch model
model = YOLO("models/yolo11n-pose.pt")

# 2. Export the model to TFLite format
# Use 'int8=True' for maximum compression/speed on edge devices
# Use 'half=True' for FP16 quantization
model.export(format="tflite", imgsz=640, int8=False)

# The exported file will typically be saved in a subfolder named:
# 'path/to/your_model_saved_model/your_model_float32.tflite'