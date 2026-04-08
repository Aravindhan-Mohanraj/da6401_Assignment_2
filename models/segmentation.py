"""Segmentation model
"""

import torch
import torch.nn as nn

from .vgg11 import VGG11Encoder
from .layers import CustomDropout


def _decoder_block(in_ch: int, skip_ch: int, out_ch: int, dropout_p: float = 0.0) -> nn.Sequential:
    """Create a single decoder block: ConvTranspose (upsample) + concat handled outside,
    then two Conv-BN-ReLU layers.

    Args:
        in_ch:    channels of the upsampled feature map (before concat with skip).
        skip_ch:  channels of the skip-connection feature map.
        out_ch:   output channels after the two conv layers.
        dropout_p: dropout probability (0 = no dropout).
    """
    layers = [
        nn.Conv2d(in_ch + skip_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    ]
    if dropout_p > 0.0:
        layers.append(CustomDropout(p=dropout_p))
    return nn.Sequential(*layers)


class VGG11UNet(nn.Module):
    """U-Net style segmentation network.

    Encoder: VGG11 convolutional backbone (5 blocks with max-pooling).
    Decoder: Symmetric expansive path using ConvTranspose2d for upsampling
             (bilinear/nearest interpolation is NOT used per the assignment).
             At each decoder stage, the upsampled feature map is concatenated
             with the corresponding encoder skip connection (feature fusion).

    Architecture (encoder → decoder channel progression):
        Encoder: 3 → 64 → 128 → 256 → 512 → 512 (bottleneck 7×7)
        Decoder: 512 → 512 → 256 → 128 → 64 → num_classes

    Loss: Cross-entropy (standard for multi-class semantic segmentation).
    CE loss is appropriate here because each pixel is assigned to one of
    3 mutually exclusive classes (pet foreground, background, boundary).

    BatchNorm after every conv and CustomDropout in the deeper decoder blocks
    help regularise the large number of parameters.
    """

    def __init__(self, num_classes: int = 3, in_channels: int = 3, dropout_p: float = 0.5):
        """
        Initialize the VGG11UNet model.

        Args:
            num_classes: Number of output classes.
            in_channels: Number of input channels.
            dropout_p: Dropout probability for the segmentation head.
        """
        super().__init__()
        self.encoder = VGG11Encoder(in_channels=in_channels)

        # ---- Decoder upsampling (ConvTranspose2d) ----
        # Each ConvTranspose2d doubles spatial resolution.
        # up5: 7×7 → 14×14  (bottleneck → before pool5)
        self.up5 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        # after concat with f5 (512): in=512+512=1024
        self.dec5 = _decoder_block(512, 512, 512, dropout_p=dropout_p)

        # up4: 14×14 → 28×28
        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        # after concat with f4 (512): in=256+512=768
        self.dec4 = _decoder_block(256, 512, 256, dropout_p=dropout_p)

        # up3: 28×28 → 56×56
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        # after concat with f3 (256): in=128+256=384
        self.dec3 = _decoder_block(128, 256, 128, dropout_p=dropout_p)

        # up2: 56×56 → 112×112
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        # after concat with f2 (128): in=64+128=192
        self.dec2 = _decoder_block(64, 128, 64, dropout_p=0.0)

        # up1: 112×112 → 224×224
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        # after concat with f1 (64): in=32+64=96
        self.dec1 = _decoder_block(32, 64, 32, dropout_p=0.0)

        # Final 1×1 conv to produce class logits
        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for segmentation model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].

        Returns:
            Segmentation logits [B, num_classes, H, W].
        """
        # Encoder with skip connections
        bottleneck, features = self.encoder(x, return_features=True)
        # bottleneck: [B, 512, 7, 7]
        # features['block1']: [B,  64, 224, 224]
        # features['block2']: [B, 128, 112, 112]
        # features['block3']: [B, 256,  56,  56]
        # features['block4']: [B, 512,  28,  28]
        # features['block5']: [B, 512,  14,  14]

        # Decoder stage 5: 7→14, concat with block5
        d = self.up5(bottleneck)                          # [B, 512, 14, 14]
        d = torch.cat([d, features['block5']], dim=1)    # [B, 1024, 14, 14]
        d = self.dec5(d)                                  # [B, 512, 14, 14]

        # Decoder stage 4: 14→28, concat with block4
        d = self.up4(d)                                   # [B, 256, 28, 28]
        d = torch.cat([d, features['block4']], dim=1)    # [B, 768, 28, 28]
        d = self.dec4(d)                                  # [B, 256, 28, 28]

        # Decoder stage 3: 28→56, concat with block3
        d = self.up3(d)                                   # [B, 128, 56, 56]
        d = torch.cat([d, features['block3']], dim=1)    # [B, 384, 56, 56]
        d = self.dec3(d)                                  # [B, 128, 56, 56]

        # Decoder stage 2: 56→112, concat with block2
        d = self.up2(d)                                   # [B, 64, 112, 112]
        d = torch.cat([d, features['block2']], dim=1)    # [B, 192, 112, 112]
        d = self.dec2(d)                                  # [B, 64, 112, 112]

        # Decoder stage 1: 112→224, concat with block1
        d = self.up1(d)                                   # [B, 32, 224, 224]
        d = torch.cat([d, features['block1']], dim=1)    # [B, 96, 224, 224]
        d = self.dec1(d)                                  # [B, 32, 224, 224]

        out = self.final_conv(d)                          # [B, num_classes, 224, 224]
        return out