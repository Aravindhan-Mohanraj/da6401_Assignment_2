"""Classification components
"""

import torch
import torch.nn as nn

from .vgg11 import VGG11Encoder
from .layers import CustomDropout


class VGG11Classifier(nn.Module):
    """Full classifier = VGG11Encoder + ClassificationHead.

    The classification head mirrors the original VGG paper's FC layers:
        FC(25088 → 4096) – ReLU – Dropout
        FC(4096  → 4096) – ReLU – Dropout
        FC(4096  → num_classes)

    BatchNorm1d is added after the first two FC layers to stabilise training.
    CustomDropout is placed after each FC+BN+ReLU block to reduce overfitting.
    Dropout in the FC layers is especially important because fully-connected
    layers have many parameters and are prone to co-adaptation; placing
    Dropout here yields better generalisation than placing it in the conv blocks.
    """

    def __init__(self, num_classes: int = 37, in_channels: int = 3, dropout_p: float = 0.5):
        """
        Initialize the VGG11Classifier model.
        Args:
            num_classes: Number of output classes.
            in_channels: Number of input channels.
            dropout_p: Dropout probability for the classifier head.
        """
        super().__init__()
        self.encoder = VGG11Encoder(in_channels=in_channels)

        # Adaptive pool so the classifier works with any spatial resolution
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        # FC head: 512 * 7 * 7 = 25088
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for classification model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].
        Returns:
            Classification logits [B, num_classes].
        """
        x = self.encoder(x)           # [B, 512, 7, 7]
        x = self.avgpool(x)           # [B, 512, 7, 7]
        x = torch.flatten(x, 1)       # [B, 25088]
        x = self.classifier(x)        # [B, num_classes]
        return x