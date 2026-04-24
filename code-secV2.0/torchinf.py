"""
Model summary via torchinfo
"""
from torchvision import models
from torchinfo import summary

_arch = models.alexnet()
summary(_arch, (1, 3, 224, 224))  # batch=1, channels=3, height=224, width=224
