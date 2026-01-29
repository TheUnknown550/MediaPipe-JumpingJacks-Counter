import numpy as np
import tensorflow as tf
import os

# Update these paths to match your tflite_exports folder exactly
models = {
    "Original": "models/tflite_models/no-pruning/yolo11n-pose_int8.tflite",
    "Pruned 25%": "models/tflite_models/25pruning/yolo11n-pose-pruned-25_int8.tflite",
    "Pruned 50%": "models/tflite_models/50pruning/yolo11n-pose-pruned-50_int8.tflite",
    "Pruned 75%": "models/tflite_models/75pruning/yolo11n-pose-pruned-75_int8.tflite"
}

def check_tflite_stats(name, path):
    if not os.path.exists(path):
        print(f"⚠️  File not found: {path}")
        return

    size_mb = os.path.getsize(path) / (1024 * 1024)
    
    try:
        # Load the TFLite model
        interpreter = tf.lite.Interpreter(model_path=path)
        interpreter.allocate_tensors()
        
        details = interpreter.get_tensor_details()
        total_params = 0
        zero_params = 0
        
        for detail in details:
            idx = detail['index']
            shape = detail['shape']
            
            # 1. Only look at tensors with 2+ dimensions (Weights/Filters)
            # 2. Skip 'null' tensors by checking if they are constant weights
            try:
                tensor_data = interpreter.get_tensor(idx)
                
                # We only care about numeric arrays that aren't empty
                if tensor_data is not None and len(shape) >= 2:
                    total_params += tensor_data.size
                    zero_params += np.sum(tensor_data == 0)
            except ValueError:
                # This skips intermediate tensors that have no data allocated
                continue
        
        sparsity = (zero_params / total_params * 100) if total_params > 0 else 0
        
        print(f"{'='*10} {name} (TFLite) {'='*10}")
        print(f"Path: {path}")
        print(f"File Size: {size_mb:.2f} MB")
        print(f"Total Weight Params: {total_params:,}")
        print(f"Zeroed Weights:      {zero_params:,}")
        print(f"Calculated Sparsity: {sparsity:.2f}%")
        print("-" * (30 + len(name)))
        print("")

    except Exception as e:
        print(f"❌ Error loading {name}: {e}")

if __name__ == "__main__":
    print("Starting TFLite Model Verification...\n")
    for name, path in models.items():
        check_tflite_stats(name, path)