"""
MACs/FLOPs profiler for PyTorch models
"""
import torch
from torchprofile import profile_macs
from torchvision import models

def _main():
    for name, builder in models.__dict__.items():
        if not name.islower() or name.startswith('__') or not callable(builder):
            continue
        try:
            net = builder().eval()
            inp = torch.randn(1, 3, 299, 299) if 'inception' in name else torch.randn(1, 3, 224, 224)
            macs = profile_macs(net, inp)
            print(f'{name}: {macs / 1e9:.4g} G')
        except Exception:
            pass

if __name__ == '__main__':
    _main()
