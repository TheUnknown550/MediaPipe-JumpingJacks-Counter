from ultralytics import YOLO, settings
settings.update({"sync": False})

# 1. Load your YOLO-Pose model
# Make sure your model is on the CPU during export to avoid conversion errors
model = YOLO("models/yolo11n-pose-pruned-25.pt") 


# 2. Export to FP16 TFLite
# 'half=True' triggers the 16-bit float quantization
model.export(
    format="tflite", 
    half=True,
    device=0
)

print("Export complete: Your FP16 model is ready!")