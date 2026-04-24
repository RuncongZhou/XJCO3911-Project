"""
Model statistics via torchstat
"""
from torchstat import stat
import torchvision.models as models

# Instantiate and profile VGG16
_net = models.vgg16()
stat(_net, (3, 224, 224))
