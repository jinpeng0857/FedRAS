"""
EATACResNet: ResNet with sub-network dropout for EATA-C TTA.

Extends standard ResNet with:
  - Dict output {'logit', 'feature', 'layer0'..'layer5'}
  - Sub-network generation via stochastic depth dropout
  - Used for FL training with EATA-C test-time adaptation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.resnet import BasicBlock as ResNetBasicBlock


class BasicBlock(ResNetBasicBlock):
    """Extends ResNet BasicBlock with stochastic depth (drop_block) support."""

    def __init__(self, in_planes, planes, stride=1, norm_layer=nn.BatchNorm2d):
        super().__init__(in_planes, planes, stride, norm_layer)
        self.has_identity_shortcut = (stride == 1 and in_planes == self.expansion * planes)

    def forward(self, x, drop_block=False, scale=1.0):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if drop_block:
            # Drop this residual block entirely
            if self.has_identity_shortcut:
                return F.relu(identity)
            else:
                return F.relu(self.shortcut(x))

        # Stochastic Depth: scale kept block by 1/(1-p) to maintain expected magnitude
        out = out * scale
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class EATACResNet(nn.Module):
    """
    ResNet with EATA-C sub-network capability.

    When return_sub_network=True, randomly drops residual blocks
    with probability subnet_drop_rate to create a stochastic sub-network.
    """

    def __init__(self, block, num_blocks, num_classes=10,
                 subnet_drop_rate=0.05, min_active_blocks=2):
        super().__init__()
        self.in_planes = 64
        self.num_layers = 4  # layer1..layer4
        norm_layer = nn.BatchNorm2d

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # Sub-network params
        self.subnet_drop_rate = subnet_drop_rate
        self.min_active_blocks = min_active_blocks
        # Global flag to force sub-network mode from outside (circumvent FL wrapper barriers)
        self.force_sub_network = False

        # Track all blocks for dropout
        self._all_blocks = list(self.layer1) + list(self.layer2) + \
                           list(self.layer3) + list(self.layer4)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, return_sub_network=False, no_relu=False):
        """
        Args:
            x: input tensor
            return_sub_network: if True, randomly drops blocks
            no_relu: if True, skip final ReLU on intermediate outputs

        Returns:
            dict with keys: logit, feature, (optionally layer0..layer5)
        """
        results = {}

        out = self.bn1(self.conv1(x))
        if no_relu:
            results["layer0"] = out
        out = F.relu(out)

        if (return_sub_network or self.force_sub_network) and (self.training or self.force_sub_network):
            out, results = self._forward_with_subnet(out, results, no_relu)
        else:
            out, results = self._forward_standard(out, results, no_relu)

        feature = self.avgpool(out).view(out.size(0), -1)
        logit = self.fc(feature)

        results["logit"] = logit
        results["feature"] = feature
        return results

    def _forward_standard(self, out, results, no_relu):
        layers = [self.layer1, self.layer2, self.layer3, self.layer4]
        for i, layer in enumerate(layers):
            out = layer(out)
            if no_relu:
                results[f"layer{i+1}"] = out
        return out, results

    def _forward_with_subnet(self, out, results, no_relu):
        """Forward with random block dropping and Stochastic Depth scaling."""
        layers = [self.layer1, self.layer2, self.layer3, self.layer4]
        # Stochastic Depth: scale kept blocks by 1/(1-p) to preserve expected output magnitude
        scale = 1.0 / (1.0 - self.subnet_drop_rate) if self.subnet_drop_rate < 1.0 else 1.0

        for i, layer in enumerate(layers):
            for j, block in enumerate(layer):
                drop = torch.rand(1).item() < self.subnet_drop_rate
                out = block(out, drop_block=drop, scale=1.0 if drop else scale)
            if no_relu:
                results[f"layer{i+1}"] = out

        return out, results

    def use_bn_layer(self):
        """Ensure all norm layers are BatchNorm."""
        for m in self.modules():
            if isinstance(m, nn.GroupNorm):
                # Already using BatchNorm by default via __init__
                pass


def eatac_resnet18(num_classes=10, subnet_drop_rate=0.05, min_active_blocks=2):
    return EATACResNet(
        BasicBlock, [2, 2, 2, 2], num_classes,
        subnet_drop_rate=subnet_drop_rate,
        min_active_blocks=min_active_blocks,
    )


def eatac_resnet34(num_classes=10, subnet_drop_rate=0.05, min_active_blocks=2):
    return EATACResNet(
        BasicBlock, [3, 4, 6, 3], num_classes,
        subnet_drop_rate=subnet_drop_rate,
        min_active_blocks=min_active_blocks,
    )
