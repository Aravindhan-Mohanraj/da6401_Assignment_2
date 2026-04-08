"""Unified multi-task model
"""

import torch
import torch.nn as nn

from .vgg11 import VGG11Encoder
from .layers import CustomDropout


class MultiTaskPerceptionModel(nn.Module):
    """Shared-backbone multi-task model.

    Loads pre-trained weights from classifier.pth, localizer.pth and
    unet.pth, then extracts the shared VGG11 encoder and the three
    task-specific heads. A single forward pass produces:
        - classification logits  [B, num_breeds]
        - bounding box coords    [B, 4]
        - segmentation logits    [B, seg_classes, H, W]
    """

    def __init__(
        self,
        num_breeds: int = 37,
        seg_classes: int = 3,
        in_channels: int = 3,
        classifier_path: str = "checkpoints/classifier.pth",
        localizer_path: str = "checkpoints/localizer.pth",
        unet_path: str = "checkpoints/unet.pth",
    ):
        """
        Initialize the shared backbone/heads using these trained weights.
        Args:
            num_breeds: Number of output classes for classification head.
            seg_classes: Number of output classes for segmentation head.
            in_channels: Number of input channels.
            classifier_path: Path to trained classifier weights.
            localizer_path: Path to trained localizer weights.
            unet_path: Path to trained unet weights.
        """
        import gdown
        gdown.download(id="<classifier.pth drive id>", output=classifier_path, quiet=False)
        gdown.download(id="<localizer.pth drive id>", output=localizer_path, quiet=False)
        gdown.download(id="<unet.pth drive id>", output=unet_path, quiet=False)

        super().__init__()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ------------------------------------------------------------------ #
        # Build sub-networks and load their trained checkpoints              #
        # ------------------------------------------------------------------ #
        from .classification import VGG11Classifier
        from .localization import VGG11Localizer
        from .segmentation import VGG11UNet

        classifier = VGG11Classifier(num_classes=num_breeds, in_channels=in_channels)
        localizer = VGG11Localizer(in_channels=in_channels)
        unet = VGG11UNet(num_classes=seg_classes, in_channels=in_channels)

        def _load(model, path):
            ckpt = torch.load(path, map_location=device)
            state = ckpt.get("state_dict", ckpt)
            model.load_state_dict(state, strict=True)

        _load(classifier, classifier_path)
        _load(localizer, localizer_path)
        _load(unet, unet_path)

        # ------------------------------------------------------------------ #
        # Shared backbone: use the encoder from the classifier               #
        # (all three were trained with the same VGG11Encoder architecture)   #
        # ------------------------------------------------------------------ #
        self.encoder = classifier.encoder  # VGG11Encoder

        # ------------------------------------------------------------------ #
        # Classification head                                                 #
        # ------------------------------------------------------------------ #
        self.avgpool = classifier.avgpool
        self.cls_head = classifier.classifier   # FC layers → num_breeds

        # ------------------------------------------------------------------ #
        # Localization head (regression)                                      #
        # ------------------------------------------------------------------ #
        self.loc_avgpool = localizer.avgpool
        self.loc_head = localizer.regressor
        self.img_size = localizer.img_size

        # ------------------------------------------------------------------ #
        # Segmentation decoder + heads from U-Net                            #
        # ------------------------------------------------------------------ #
        self.up5 = unet.up5
        self.dec5 = unet.dec5
        self.up4 = unet.up4
        self.dec4 = unet.dec4
        self.up3 = unet.up3
        self.dec3 = unet.dec3
        self.up2 = unet.up2
        self.dec2 = unet.dec2
        self.up1 = unet.up1
        self.dec1 = unet.dec1
        self.seg_final = unet.final_conv

    def forward(self, x: torch.Tensor):
        """Forward pass for multi-task model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].
        Returns:
            A dict with keys:
            - 'classification': [B, num_breeds] logits tensor.
            - 'localization': [B, 4] bounding box tensor.
            - 'segmentation': [B, seg_classes, H, W] segmentation logits tensor
        """
        # Shared encoder — run ONCE with skip features for segmentation
        bottleneck, features = self.encoder(x, return_features=True)
        # bottleneck: [B, 512, 7, 7]

        # ---- Classification ----
        cls_feat = self.avgpool(bottleneck)          # [B, 512, 7, 7]
        cls_feat = torch.flatten(cls_feat, 1)        # [B, 25088]
        cls_out = self.cls_head(cls_feat)            # [B, num_breeds]

        # ---- Localization ----
        loc_feat = self.loc_avgpool(bottleneck)      # [B, 512, 7, 7]
        loc_feat = torch.flatten(loc_feat, 1)        # [B, 25088]
        loc_out = self.loc_head(loc_feat)            # [B, 4]
        loc_out = torch.sigmoid(loc_out) * self.img_size  # pixel space

        # ---- Segmentation ----
        d = self.up5(bottleneck)
        d = torch.cat([d, features['block5']], dim=1)
        d = self.dec5(d)

        d = self.up4(d)
        d = torch.cat([d, features['block4']], dim=1)
        d = self.dec4(d)

        d = self.up3(d)
        d = torch.cat([d, features['block3']], dim=1)
        d = self.dec3(d)

        d = self.up2(d)
        d = torch.cat([d, features['block2']], dim=1)
        d = self.dec2(d)

        d = self.up1(d)
        d = torch.cat([d, features['block1']], dim=1)
        d = self.dec1(d)

        seg_out = self.seg_final(d)                  # [B, seg_classes, H, W]

        return {
            "classification": cls_out,
            "localization": loc_out,
            "segmentation": seg_out,
        }