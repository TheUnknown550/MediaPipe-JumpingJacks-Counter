import torch
import torch.nn.utils.prune as prune
from ultralytics import YOLO
import os

# 1. Define the targets
prune_targets = {
    0.25: "models/yolo11n-pose-pruned-25.pt",
    0.50: "models/yolo11n-pose-pruned-50.pt",
    0.75: "models/yolo11n-pose-pruned-75.pt"
}

# Ensure the models directory exists
os.makedirs("models", exist_ok=True)

for amount, save_path in prune_targets.items():
    print(f"\n--- Pruning Model to {amount*100}% Sparsity ---")
    
    # 2. IMPORTANT: Load a FRESH model for every pruning level
    # If you don't reload, the pruning percentages will stack incorrectly
    model = YOLO("models/yolo11n-pose.pt")

    # 3. Apply pruning to Conv2d layers
    for name, module in model.model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            # Apply L1 Structured pruning
            prune.ln_structured(module, name='weight', amount=amount, n=1, dim=0)
            # Remove the pruning hooks to make it a standard .pt file again
            prune.remove(module, 'weight')

    # 4. Save with the correct name
    model.save(save_path)
    print(f"✅ Saved to {save_path}")

print("\nAll models pruned successfully!")