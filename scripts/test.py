import os
from pathlib import Path

# Search your models directory for any tflite files
model_path = Path("models")
tflite_files = list(model_path.rglob("*.tflite"))

print("--- Found TFLite Files ---")
for f in tflite_files:
    size_mb = os.path.getsize(f) / (1024 * 1024)
    print(f"Location: {f}")
    print(f"Physical Size: {size_mb:.2f} MB\n")