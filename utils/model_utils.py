"""Shared model construction and checkpoint-inference helpers."""

from models.mobilenetv2 import mobilenetv2
from models.resnet import resnet18, resnet34


SUPPORTED_BACKBONES = ("resnet18", "resnet34", "mobilenetv2")


def build_model(
    backbone: str,
    num_classes: int,
    norm_type: str = "bn",
    num_groups: int = 32,
    imagenet_stem: bool = False,
):
    builders = {
        "resnet18": resnet18,
        "resnet34": resnet34,
        "mobilenetv2": mobilenetv2,
    }
    try:
        builder = builders[backbone]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported backbone: {backbone}. "
            f"Choose one of {SUPPORTED_BACKBONES}."
        ) from exc
    return builder(
        num_classes=num_classes,
        norm_type=norm_type,
        num_groups=num_groups,
        imagenet_stem=imagenet_stem,
    )


def infer_backbone_from_state_dict(state_dict):
    """Infer one of the supported architectures from checkpoint parameter keys."""
    if "fc.weight" not in state_dict:
        raise KeyError("state_dict is missing fc.weight")

    if any(key.startswith("features.") for key in state_dict):
        return "mobilenetv2"

    if "conv1.weight" not in state_dict or "layer4.0.conv1.weight" not in state_dict:
        raise ValueError("Checkpoint does not match a supported ResNet or MobileNetV2")

    # ResNet34 has 3/4/6/3 basic blocks; ResNet18 has 2/2/2/2.
    if "layer3.5.conv2.weight" in state_dict:
        return "resnet34"
    return "resnet18"


def infer_imagenet_stem_from_state_dict(state_dict) -> bool:
    """Infer the ResNet input stem. MobileNetV2 stem stride is not serialized."""
    if "conv1.weight" not in state_dict:
        if any(key.startswith("features.") for key in state_dict):
            return False
        raise KeyError("state_dict is missing conv1.weight")
    kernel = tuple(state_dict["conv1.weight"].shape[-2:])
    if kernel == (7, 7):
        return True
    if kernel == (3, 3):
        return False
    raise ValueError(f"Unknown conv1 kernel size: {kernel}")
