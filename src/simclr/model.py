"""ResNet-18 with CIFAR stem, 2-layer MLP projection head, and linear classifier."""
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """Standard ResNet BasicBlock with bias=False convs and zero-init gamma option."""

    expansion: int = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes, eps=1e-5, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-5, momentum=0.1)

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
                nn.BatchNorm2d(self.expansion * planes, eps=1e-5, momentum=0.1),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResNet18CIFAR(nn.Module):
    """ResNet-18 base encoder adapted for 32x32 CIFAR images.

    Follows paper Appendix B.9 and official repo (tf2/resnet.py cifar_stem=True):
    - Stem: 3x3 Conv (stride 1, 64 channels) + BatchNorm + ReLU (NO MAX POOLING).
    - Stages: [2, 2, 2, 2] BasicBlocks with channels [64, 128, 256, 512].
    - Global average pooling output representation h in R^512.
    """

    def __init__(self, num_blocks: Tuple[int, int, int, int] = (2, 2, 2, 2)):
        super().__init__()
        self.in_planes = 64

        # CIFAR stem: 3x3 conv, stride 1, no max-pool
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, eps=1e-5, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(512, num_blocks[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.feature_dim = 512 * BasicBlock.expansion

        self._initialize_weights()

    def _make_layer(self, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, stride=s))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

        # Matching official repo detail: initialize last BN in each residual block with gamma = 0
        for m in self.modules():
            if isinstance(m, BasicBlock):
                nn.init.constant_(m.bn2.weight, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        h = torch.flatten(out, 1)
        return h


class ProjectionHead(nn.Module):
    """2-layer MLP projection head g(.) for SimCLR v1.

    Structure per paper Section 2.1 & official repo (tf2/model.py ProjectionHead):
    Linear(in_dim, hidden_dim) -> BN -> ReLU -> Linear(hidden_dim, out_dim) -> BN.
    No ReLU on the final output.
    """

    def __init__(
        self,
        in_dim: int = 512,
        hidden_dim: int = 512,
        out_dim: int = 128,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim, eps=1e-5, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim, bias=False),
            nn.BatchNorm1d(out_dim, eps=1e-5, momentum=0.1),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


class SimCLRModel(nn.Module):
    """Full SimCLR model combining base encoder and projection head."""

    def __init__(
        self,
        encoder: Optional[nn.Module] = None,
        projector: Optional[nn.Module] = None,
        in_dim: int = 512,
        proj_dim: int = 128,
    ):
        super().__init__()
        self.encoder = encoder if encoder is not None else ResNet18CIFAR()
        self.projector = (
            projector
            if projector is not None
            else ProjectionHead(in_dim=in_dim, hidden_dim=in_dim, out_dim=proj_dim)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns representation h and projection z."""
        h = self.encoder(x)
        z = self.projector(h)
        return h, z

    def get_representation(self, x: torch.Tensor) -> torch.Tensor:
        """Returns representation h without passing through the projection head."""
        return self.encoder(x)


class LinearClassifier(nn.Module):
    """Single linear layer for downstream classification on frozen representations."""

    def __init__(self, feature_dim: int = 512, num_classes: int = 10):
        super().__init__()
        self.fc = nn.Linear(feature_dim, num_classes)
        # Official CIFAR script: zero-init logits layer
        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.fc(h)
