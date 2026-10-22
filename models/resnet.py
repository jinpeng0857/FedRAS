"""Standard ResNet model for CIFAR-style federated training.

The forward path is intentionally simple: ``model(x)`` returns logits only.
This keeps the training checkpoint compatible with the main continual test,
where only BatchNorm affine parameters are adapted.
"""

import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = norm_layer(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = norm_layer(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                norm_layer(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block,
        num_blocks,
        num_classes=10,
        norm_type="bn",
        num_groups=32,
        imagenet_stem=False,
    ):
        super().__init__()
        self.in_planes = 64

        if norm_type == "bn":
            norm_layer = nn.BatchNorm2d
        elif norm_type == "gn":
            norm_layer = lambda c: nn.GroupNorm(num_groups, c)
        else:
            raise ValueError(f"Unknown norm type: {norm_type}")

        self.norm_type = norm_type
        self.imagenet_stem = imagenet_stem
        if imagenet_stem:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.maxpool = nn.Identity()
        self.bn1 = norm_layer(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, norm_layer=norm_layer)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, norm_layer=norm_layer)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, norm_layer=norm_layer)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, norm_layer=norm_layer)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, norm_layer):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s, norm_layer))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out).view(out.size(0), -1)
        logits = self.fc(out)
        return logits


def resnet18(num_classes=10, norm_type="bn", num_groups=32, imagenet_stem=False):
    return ResNet(
        BasicBlock,
        [2, 2, 2, 2],
        num_classes,
        norm_type,
        num_groups,
        imagenet_stem=imagenet_stem,
    )


def resnet34(num_classes=10, norm_type="bn", num_groups=32, imagenet_stem=False):
    return ResNet(
        BasicBlock,
        [3, 4, 6, 3],
        num_classes,
        norm_type,
        num_groups,
        imagenet_stem=imagenet_stem,
    )
