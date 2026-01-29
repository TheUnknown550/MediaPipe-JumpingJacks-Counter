"""
Model Quantization & Conversion Script
======================================
Converts YOLO11n-Pose model to optimized formats for edge deployment.

Formats supported:
- NCNN (FP16) - For Raspberry Pi 4 CPU
- TFLite (INT8) - For devices with Edge TPU (Google Coral)
"""

from ultralytics import YOLO
import os

# Model path
MODEL_PT = "../yolo11n-pose.pt"

def convert_to_ncnn():
    """Export to NCNN format (FP16) for Raspberry Pi"""
    print("Converting to NCNN (FP16)...")
    try:
        model = YOLO(MODEL_PT)
        model.export(format="ncnn", imgsz=640, half=True)
        print("✅ NCNN conversion complete!")
        return True
    except Exception as e:
        print(f"❌ NCNN conversion failed: {e}")
        return False

def convert_to_tflite():
    """Export to TFLite format (INT8) for Edge TPU"""
    print("\nConverting to TFLite (INT8)...")
    try:
        model = YOLO(MODEL_PT)
        # Note: INT8 requires calibration data - use coco8-pose.yaml as example
        model.export(format="tflite", imgsz=640, int8=True, data="coco8-pose.yaml")
        print("✅ TFLite conversion complete!")
        return True
    except Exception as e:
        print(f"⚠️  TFLite conversion skipped (requires calibration data): {e}")
        return False

def convert_to_onnx():
    """Export to ONNX format for cross-platform compatibility"""
    print("\nConverting to ONNX...")
    try:
        model = YOLO(MODEL_PT)
        model.export(format="onnx", imgsz=640)
        print("✅ ONNX conversion complete!")
        return True
    except Exception as e:
        print(f"❌ ONNX conversion failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("YOLO11n-Pose Model Quantization & Conversion")
    print("=" * 60)
    
    results = []
    results.append(("NCNN (FP16)", convert_to_ncnn()))
    results.append(("TFLite (INT8)", convert_to_tflite()))
    results.append(("ONNX", convert_to_onnx()))
    
    print("\n" + "=" * 60)
    print("Conversion Summary:")
    print("=" * 60)
    for format_name, success in results:
        status = "✅ Success" if success else "❌ Failed/Skipped"
        print(f"{format_name:<20} {status}")
    
    print("\nOptimized models ready for deployment!")
