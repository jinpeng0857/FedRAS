"""
EATACResNet-ImageNet: ResNet for ImageNet (224x224) with sub-network dropout.

Standard ImageNet ResNet stem (7x7 conv + maxpool) with stochastic depth
for EATA-C test-time adaptation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, norm_layer=nn.BatchNorm2d):
        super().__init__()
        width = planes
        self.conv1 = nn.Conv2d(in_planes, width, kernel_size=1, bias=False)
        self.bn1 = norm_layer(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = norm_layer(width)
        self.conv3 = nn.Conv2d(width, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * self.expansion)
        self.has_identity_shortcut = (stride == 1 and in_planes == planes * self.expansion)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, kernel_size=1,
                         stride=stride, bias=False),
                norm_layer(planes * self.expansion),
            )

    def forward(self, x, drop_block=False, scale=1.0):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if drop_block:
            if self.has_identity_shortcut:
                return F.relu(identity)
            else:
                return F.relu(self.shortcut(x))

        out = out * scale
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class EATACImageNetResNet(nn.Module):
    """ResNet for ImageNet (224x224) with EATA-C sub-network capability."""

    def __init__(self, block, layers, num_classes=1000,
                 subnet_drop_rate=0.05, min_active_blocks=2):
        super().__init__()
        self.in_planes = 64
        self.num_layers = 4
        norm_layer = nn.BatchNorm2d

        # ImageNet stem: 7x7 conv + maxpool
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.subnet_drop_rate = subnet_drop_rate
        self.min_active_blocks = min_active_blocks
        self.force_sub_network = False

        self._all_blocks = (
            list(self.layer1) + list(self.layer2)
            + list(self.layer3) + list(self.layer4)
        )

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, return_sub_network=False, no_relu=False):
        results = {}

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)

        if (return_sub_network or self.force_sub_network) and (self.training or self.force_sub_network):
            out, results = self._forward_with_subnet(out, results, no_relu)
        else:
            layers = [self.layer1, self.layer2, self.layer3, self.layer4]
            for i, layer in enumerate(layers):
                out = layer(out)
                if no_relu:
                    results[f"layer{i+1}"] = out

        feature = self.avgpool(out).view(out.size(0), -1)
        logit = self.fc(feature)

        results["logit"] = logit
        results["feature"] = feature
        return results

    def _forward_with_subnet(self, out, results, no_relu):
        layers = [self.layer1, self.layer2, self.layer3, self.layer4]
        scale = 1.0 / (1.0 - self.subnet_drop_rate) if self.subnet_drop_rate < 1.0 else 1.0

        for i, layer in enumerate(layers):
            for block in layer:
                drop = torch.rand(1).item() < self.subnet_drop_rate
                out = block(out, drop_block=drop, scale=1.0 if drop else scale)
            if no_relu:
                results[f"layer{i+1}"] = out

        return out, results


def eatac_resnet50(num_classes=1000, subnet_drop_rate=0.05, min_active_blocks=2):
    """EATAC ResNet-50 for ImageNet (224x224 input)."""
    return EATACImageNetResNet(
        Bottleneck, [3, 4, 6, 3], num_classes,
        subnet_drop_rate=subnet_drop_rate,
        min_active_blocks=min_active_blocks,
    )


def eatac_resnet101(num_classes=1000, subnet_drop_rate=0.05, min_active_blocks=2):
    """EATAC ResNet-101 for ImageNet (224x224 input)."""
    return EATACImageNetResNet(
        Bottleneck, [3, 4, 23, 3], num_classes,
        subnet_drop_rate=subnet_drop_rate,
        min_active_blocks=min_active_blocks,
    )
