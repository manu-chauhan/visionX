"""
ResNet model implementation in PyTorch.

Deep Residual Learning for Image Recognition:
Link: https://arxiv.org/abs/1512.03385

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

# to reduce unnecessary typing and repeated function lookups
conv = partial(nn.Conv2d, bias=False)
bn = nn.BatchNorm2d
relu = F.relu


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, *, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv(
            in_planes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn1 = bn(planes)
        self.conv2 = conv(planes, planes, kernel_size=3,
                          stride=1, padding=1)
        self.bn2 = bn(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                conv(in_planes, self.expansion * planes,
                     kernel_size=1, stride=stride),
                bn(self.expansion * planes)
            )

    def forward(self, x):
        out = relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out += self.shortcut(x)  # <-- pay attention

        out = relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, *, block, num_blocks, num_classes=10, before_linear_dim=4):
        super().__init__()
        self.in_planes = 64
        self.before_linear_dim = before_linear_dim

        self.conv1 = conv(3, 64, kernel_size=3,
                          stride=1, padding=1)
        self.bn1 = bn(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)

        layers = []

        for stride in strides:
            layers.append(block(in_planes=self.in_planes,
                                planes=planes, stride=stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, self.before_linear_dim)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18():
    return ResNet(block=BasicBlock, num_blocks=[2, 2, 2, 2])


def ResNet34():
    return ResNet(block=BasicBlock, num_blocks=[3, 4, 6, 3])


def test():
    net = ResNet34()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())


# test()
