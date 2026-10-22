"""Model architectures for FEDEATA FedACG training."""

from .mobilenetv2 import mobilenetv2
from .resnet import resnet18, resnet34

__all__ = [
    "mobilenetv2",
    "resnet18",
    "resnet34",
]
