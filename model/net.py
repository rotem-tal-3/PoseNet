from typing import Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SqueezeExciteBlock(nn.Module):
    """
    Implements a Squeeze-and-Excitation block for 1D convolutions.

    This acts as a channel-wise attention mechanism, allowing the model
    to emphasize specific landmarks that are more relevant to a given movement.

    Args:
        channels (int): Number of input channels.
        reduction (int): Reduction ratio for the bottleneck.
    """

    def __init__(self, channels: int, reduction: int = 16):
        super(SqueezeExciteBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies attention weights to the input tensor.

        Args:
            x (torch.Tensor): Input of shape (Batch, Channels, Time).
        """
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class ResidualBlock1D(nn.Module):
    """
    A Residual block with two 1D convolutions and a Squeeze-Excite layer.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        stride (int): Stride for the first convolution.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(ResidualBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.se = SqueezeExciteBlock(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with skip connection.
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += self.shortcut(x)
        return F.relu(out)


class AdvancedPoseResNet(nn.Module):
    """
    A deeper Residual 1D-CNN for complex exercise classification (14-30+ classes).

    This architecture uses skip connections and attention to handle inter-class
    similarity and provide a robust embedding for OOD detection.

    Args:
        input_channels (int): landmarks * coordinates (e.g., 66).
        num_classes (int): Number of target exercises.
        hidden_dim (int): Dimension of the OOD embedding layer.
    """

    def __init__(self, input_channels: int = 66, num_classes: int = 14, hidden_dim: int = 256):
        super(AdvancedPoseResNet, self).__init__()

        self.prep = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        self.layer1 = ResidualBlock1D(64, 128, stride=1)
        self.layer2 = ResidualBlock1D(128, 256, stride=1)
        self.layer3 = ResidualBlock1D(256, 512, stride=1)

        self.avg_pool = nn.AdaptiveAvgPool1d(1)

        self.fc_embedding = nn.Sequential(
            nn.Linear(512, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim)
        )

        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts high-dimensional embeddings for OOD logic.
        """
        x = self.prep(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avg_pool(x).view(x.size(0), -1)
        return self.fc_embedding(x)

    def forward(self, x: torch.Tensor) -> tuple[Any, Tensor]:
        """
        Returns classification logits.
        """
        feat = self.forward_features(x)
        return self.classifier(feat), feat