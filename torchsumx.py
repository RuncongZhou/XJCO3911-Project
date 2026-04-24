"""
Model summary via torchsummaryX
"""
import torch
from torchsummaryX import summary
from torchvision import models

_net = models.vgg16()
summary(_net, torch.zeros((1, 3, 224, 224)))
