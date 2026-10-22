"""MobileNetV2 for CIFAR-style federated training and ImageNet inputs."""

import torch
import torch.nn as nn


def _make_divisible(value, divisor=8):
    return max(divisor, int(value + divisor / 2) // divisor * divisor)


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, stride=1, groups=1):
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True),
        )


class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super().__init__()
        hidden_channels = int(round(in_channels * expand_ratio))
        self.use_residual = stride == 1 and in_channels == out_channels

        layers = []
        if expand_ratio != 1:
            layers.extend(
                [
                    nn.Conv2d(in_channels, hidden_channels, 1, bias=False),
                    nn.BatchNorm2d(hidden_channels),
                    nn.ReLU6(inplace=True),
                ]
            )
        layers.extend(
            [
                ConvBNReLU(
                    hidden_channels,
                    hidden_channels,
                    stride=stride,
                    groups=hidden_channels,
                ),
                nn.Conv2d(hidden_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
            ]
        )
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        return x + out if self.use_residual else out


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=10, width_mult=1.0, imagenet_stem=False):
        super().__init__()
        first_channels = _make_divisible(32 * width_mult)
        last_channels = _make_divisible(1280 * max(1.0, width_mult))
        first_stride = 2 if imagenet_stem else 1

        # expansion, output channels, repeats, first-block stride
        settings = (
            (1, 16, 1, 1),
            (6, 24, 2, 1 if not imagenet_stem else 2),
            (6, 32, 3, 2),
            (6, 64, 4, 2),
            (6, 96, 3, 1),
            (6, 160, 3, 2),
            (6, 320, 1, 1),
        )

        features = [ConvBNReLU(3, first_channels, stride=first_stride)]
        in_channels = first_channels
        for expand_ratio, channels, repeats, stride in settings:
            out_channels = _make_divisible(channels * width_mult)
            for index in range(repeats):
                block_stride = stride if index == 0 else 1
                features.append(
                    InvertedResidual(
                        in_channels,
                        out_channels,
                        block_stride,
                        expand_ratio,
                    )
                )
                in_channels = out_channels
        features.extend(
            [
                nn.Conv2d(in_channels, last_channels, 1, bias=False),
                nn.BatchNorm2d(last_channels),
                nn.ReLU6(inplace=True),
            ]
        )

        self.features = nn.Sequential(*features)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # Keep the common ``fc`` name so existing checkpoint/evaluation code works.
        self.fc = nn.Linear(last_channels, num_classes)
        self.imagenet_stem = imagenet_stem
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def mobilenetv2(num_classes=10, norm_type="bn", num_groups=32, imagenet_stem=False):
    if norm_type != "bn":
        raise ValueError("mobilenetv2 requires norm_type='bn'")
    return MobileNetV2(num_classes=num_classes, imagenet_stem=imagenet_stem)
