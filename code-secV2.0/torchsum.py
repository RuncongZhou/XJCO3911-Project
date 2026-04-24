"""
Model summary via torchsummary
"""
import torch
from torchvision import models
from torchsummary import summary

_dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_model = models.mobilenet_v2().to(_dev)
summary(_model, (3, 224, 224))
