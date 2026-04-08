"""Localization modules
"""

import torch
import torch.nn as nn

from .vgg11 import VGG11Encoder
from .layers import CustomDropout


class VGG11Localizer(nn.Module):
    """VGG11-based localizer.

    Uses the VGG11 convolutional backbone as a feature extractor and attaches
    a regression head that predicts [x_center, y_center, width, height] in
    pixel coordinates of the original (224×224) image space.

    The encoder weights are fine-tuned (not frozen) during localization
    training. Bounding box regression benefits from task-specific feature
    adaptation — frozen features trained for classification may not capture
    the spatial extent information needed for accurate localisation.

    The regression head uses a sigmoid activation scaled to image dimensions
    (224) so that outputs are always positive and within a plausible range.
    Sigmoid ensures outputs are bounded in (0, 1) before scaling, preventing
    runaway predictions and stabilising MSE + IoU loss optimisation.
    """

    def __init__(self, in_channels: int = 3, dropout_p: float = 0.5):
        """
        Initialize the VGG11Localizer model.

        Args:
            in_channels: Number of input channels.
            dropout_p: Dropout probability for the localization head.
        """
        super().__init__()
        self.encoder = VGG11Encoder(in_channels=in_channels)

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        # Regression head
        self.regressor = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(4096, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(1024, 4),
        )

        # Image size hardcoded per assignment (VGG11 input = 224×224)
        self.img_size = 224.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for localization model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].

        Returns:
            Bounding box coordinates [B, 4] in (x_center, y_center, width, height)
            format in original image pixel space (not normalised values).
        """
        feat = self.encoder(x)           # [B, 512, 7, 7]
        feat = self.avgpool(feat)         # [B, 512, 7, 7]
        feat = torch.flatten(feat, 1)     # [B, 25088]
        out = self.regressor(feat)        # [B, 4]  — raw logits

        # Sigmoid → scale to pixel space [0, img_size]
        # This keeps outputs bounded and meaningful w.r.t. image dimensions.
        out = torch.sigmoid(out) * self.img_size
        return out