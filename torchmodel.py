"""
Torch model configuration and MNIST dataset setup
"""
import torch
from torchvision import transforms, datasets

# Image normalization for MNIST
IMAGE_NORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Training dataset
TRAIN_DS = datasets.MNIST(
    root='../data/mnist',
    train=True,
    download=True,
    transform=IMAGE_NORM
)

# Test dataset
TEST_DS = datasets.MNIST(
    root='../data/mnist',
    train=False,
    download=True,
    transform=IMAGE_NORM
)
