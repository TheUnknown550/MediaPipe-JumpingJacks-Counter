import os
import torch
from ultralytics import YOLO
from ultralytics.utils.benchmarks import benchmark

# 1. Verification of GPU for your logs
print(f"CUDA status: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Running on: {torch.cuda.get_device_name(0)}")

# Paths to your models
MODEL_PT = "yolo11n-pose.pt"
MODEL_NCNN = "yolo11n-pose_ncnn_model"

models = [MODEL_PT, MODEL_NCNN]

print(f"\n{'Method':<25} | {'mAP':<8} | {'FPS':<8} | {'Size (MB)':<10} | {'Inference':<10} | {'Params'}")
print("-" * 85)

for path in models:
    # --- Part A: Physical Metrics ---
    # Load model to count parameters (only works for .pt models)
    try:
        model_obj = YOLO(path)
        n_params = f"{sum(p.numel() for p in model_obj.model.parameters()) / 1e6:.2f}M"
    except:
        n_params = "N/A" # NCNN doesn't expose params the same way in Python

    # Get Model Size (MB)
    if os.path.isdir(path):
        size_mb = sum(os.path.getsize(os.path.join(path, f)) for f in os.listdir(path)) / (1024 * 1024)
    else:
        size_mb = os.path.getsize(path) / (1024 * 1024)

    # --- Part B: Performance Metrics ---
    # Run benchmark on GPU (device 0)
    # verbose=False keeps the console clean; imgsz=640 is your target resolution
    results = benchmark(model=path, data="coco8-pose.yaml", imgsz=640, half=True, device=0, verbose=False)

    # Extracting results from the benchmark data object
    if results and len(results) > 0:
        res_data = results[0]
        map_val = f"{res_data.get('metrics/mAP50-95(B)', 0):.3f}"
        inf_time = res_data.get('inference', 0) # in milliseconds
        fps = f"{1000 / inf_time:.1f}" if inf_time > 0 else "N/A"
        inf_str = f"{inf_time:.2f}ms"
    else:
        map_val = fps = inf_str = "Error"

    print(f"{path:<25} | {map_val:<8} | {fps:<8} | {size_mb:>8.2f} | {inf_str:<10} | {n_params}")

print("\n⚠️ Note: FPS on your RTX 4050 will be much higher than on the Raspberry Pi.")