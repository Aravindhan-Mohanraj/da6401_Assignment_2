"""VGG11 encoder
"""

from typing import Dict, Tuple, Union

import torch
import torch.nn as nn

from .layers import CustomDropout


class VGG11Encoder(nn.Module):
    """VGG11-style encoder with optional intermediate feature returns.

    Architecture follows the original VGG11 paper (Simonyan & Zisserman, 2014):
    https://arxiv.org/abs/1409.1556

    Conv blocks (with BatchNorm after each Conv):
        Block 1 : 1 x Conv(3→64),   MaxPool → 112×112
        Block 2 : 1 x Conv(64→128),  MaxPool →  56×56
        Block 3 : 2 x Conv(128→256), MaxPool →  28×28
        Block 4 : 2 x Conv(256→512), MaxPool →  14×14
        Block 5 : 2 x Conv(512→512), MaxPool →   7×7

    Input assumed to be 224×224 (standard VGG input size per paper).

    BatchNorm is placed after every Conv and before ReLU. This is the
    common modern convention: it stabilises training, reduces covariate
    shift and allows higher learning rates without the network diverging.
    """

    def __init__(self, in_channels: int = 3):
        """Initialize the VGG11Encoder model."""
        super().__init__()

        # ----- Block 1: 1 conv, 64 filters -----
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 224→112

        # ----- Block 2: 1 conv, 128 filters -----
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 112→56

        # ----- Block 3: 2 convs, 256 filters -----
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 56→28

        # ----- Block 4: 2 convs, 512 filters -----
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)  # 28→14

        # ----- Block 5: 2 convs, 512 filters -----
        self.block5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)  # 14→7

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Forward pass.

        Args:
            x: input image tensor [B, 3, H, W].
            return_features: if True, also return skip maps for U-Net decoder.

        Returns:
            - if return_features=False: bottleneck feature tensor [B, 512, 7, 7].
            - if return_features=True: (bottleneck, feature_dict) where
              feature_dict contains pre-pool feature maps keyed by block name.
        """
        # Block 1
        f1 = self.block1(x)          # [B,  64, 224, 224]
        x = self.pool1(f1)           # [B,  64, 112, 112]

        # Block 2
        f2 = self.block2(x)          # [B, 128, 112, 112]
        x = self.pool2(f2)           # [B, 128,  56,  56]

        # Block 3
        f3 = self.block3(x)          # [B, 256,  56,  56]
        x = self.pool3(f3)           # [B, 256,  28,  28]

        # Block 4
        f4 = self.block4(x)          # [B, 512,  28,  28]
        x = self.pool4(f4)           # [B, 512,  14,  14]

        # Block 5
        f5 = self.block5(x)          # [B, 512,  14,  14]
        x = self.pool5(f5)           # [B, 512,   7,   7]

        if return_features:
            features = {
                "block1": f1,  # [B,  64, 224, 224]
                "block2": f2,  # [B, 128, 112, 112]
                "block3": f3,  # [B, 256,  56,  56]
                "block4": f4,  # [B, 512,  28,  28]
                "block5": f5,  # [B, 512,  14,  14]
            }
            return x, features

        return x


# Alias used by the autograder: `from models.vgg11 import VGG11`
VGG11 = VGG11Encoder