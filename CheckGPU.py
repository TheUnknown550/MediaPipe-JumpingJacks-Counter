import torch
print(f"Is CUDA available? {torch.cuda.is_available()}")
print(f"Using GPU: {torch.cuda.current_device()}")
print(f"GPU Name: {torch.cuda.get_device_name(0)}")