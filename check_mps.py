# check_mps.py
import torch

if torch.backends.mps.is_available():
    print("MPS is available! You can use the GPU.")
    device = torch.device("mps")
else:
    print("MPS not available, using CPU instead.")
    device = torch.device("cpu")

# Create a tensor on the MPS device
x = torch.rand(5, 3, device=device)
print(f"Tensor created on device: {x.device}")