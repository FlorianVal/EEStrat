import sys
import os
import torch
import json

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Add the project root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from src.model import BranchyResNet
from src.utils import build_path_count_dict, build_probs_dict

# Load the best model
model = BranchyResNet(num_classes=10, base_model="resnet18")
model.load_state_dict(torch.load("best_model.pth"))
model.eval()  # Set the model to evaluation mode

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)

print("Best model loaded successfully.")


# Define the transform for CIFAR-10
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load CIFAR-10 dataset
cifar10_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
cifar10_loader = DataLoader(cifar10_dataset, batch_size=100, shuffle=False)

# Build the path count dictionary using CIFAR-10 dataset
path_count_dict = build_path_count_dict(model, cifar10_loader, device, discretization_steps=4)
print(path_count_dict.keys())
probs_dict = build_probs_dict(path_count_dict)

# Before dumping to JSON, convert tuple keys to strings
probs_dict_str_keys = {str(k): v for k, v in probs_dict.items()}
for key in probs_dict_str_keys:
    print(key)
    print(type(key))
with open('probs_dict.json', 'w') as f:
    json.dump(probs_dict_str_keys, f, indent=4)

print("Probabilities dictionary has been written to 'probs_dict.json'")
# print("Path count dictionary built using CIFAR-10 dataset:")
# for key, value in path_count_dict.items():
#     print(f"Path {key}: {value}")

