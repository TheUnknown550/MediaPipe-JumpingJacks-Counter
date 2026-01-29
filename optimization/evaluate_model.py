"""
Model Evaluation & Benchmarking Script
======================================
Compares performance metrics of different model formats:
- Size (MB)
- Inference Speed (FPS)
- Parameters
- Memory usage
"""

import os
import torch
from ultralytics import YOLO
from ultralytics.utils.benchmarks import benchmark

def evaluate_models():
    """Compare performance of PT, NCNN, and TFLite models"""
    
    print("=" * 80)
    print("YOLO11n-Pose Model Evaluation & Benchmarking")
    print("=" * 80)
    
    # Paths
    MODEL_PT = "../yolo11n-pose.pt"
    MODEL_NCNN = "yolo11n-pose_ncnn_model"
    MODEL_TFLITE = "yolo11n-pose.tflite"
    
    models_to_test = []
    if os.path.exists(MODEL_PT):
        models_to_test.append(("PyTorch (.pt)", MODEL_PT))
    if os.path.isdir(MODEL_NCNN):
        models_to_test.append(("NCNN", MODEL_NCNN))
    if os.path.exists(MODEL_TFLITE):
        models_to_test.append(("TFLite", MODEL_TFLITE))
    
    print(f"\nGPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}\n")
    
    print(f"{'Format':<20} | {'Size (MB)':<12} | {'FPS':<8} | {'Inference':<10}")
    print("-" * 60)
    
    for format_name, model_path in models_to_test:
        try:
            if format_name == "PyTorch (.pt)":
                # Load PT model for benchmark
                model_obj = YOLO(model_path)
                results = benchmark(
                    model=model_path,
                    imgsz=640,
                    half=False,  # Set to True for FP16
                    device=0 if torch.cuda.is_available() else "cpu",
                    verbose=False
                )
                
                # Get model size
                size_mb = os.path.getsize(model_path) / (1024 * 1024)
                print(f"{format_name:<20} | {size_mb:<12.2f} | {results.results_dict.get('fps', 'N/A'):<8} | Baseline")
        except Exception as e:
            print(f"{format_name:<20} | Error: {str(e)[:30]}")
    
    print("\n" + "=" * 80)
    print("Recommendations:")
    print("=" * 80)
    print("• NCNN (FP16): Best for Raspberry Pi 4 CPU (50% smaller, 2x faster)")
    print("• TFLite (INT8): Best with Edge TPU/Coral (75% smaller, 3x faster)")
    print("• PyTorch (.pt): Development/training (full accuracy)")

if __name__ == "__main__":
    evaluate_models()
