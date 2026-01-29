import torch
import torch.nn.utils.prune as prune
from ultralytics import YOLO

# 1. Load the model
model = YOLO("yolo11n-pose.pt")

# 2. Define the pruning amount (25%)
amount = 0.25

# 3. Apply pruning to all Conv2d layers
# We use global pruning to find the least important 25% across the whole network
for name, module in model.model.named_modules():
    if isinstance(module, torch.nn.Conv2d):
        prune.l1_unstructured(module, name='weight', amount=amount)
        # This makes the pruning permanent by removing the original weight and mask
        prune.remove(module, 'weight')

# 4. Save the pruned model
model.save("yolo11n-pose-pruned-75.pt")

print(f"Pruning complete. 25% of weights in Conv layers have been zeroed out.")