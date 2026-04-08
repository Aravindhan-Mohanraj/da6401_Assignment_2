"""Dataset skeleton for Oxford-IIIT Pet.
"""

import os
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class OxfordIIITPetDataset(Dataset):
    """Oxford-IIIT Pet multi-task dataset loader.

    Loads images, breed class labels, bounding boxes and trimaps from the
    Oxford-IIIT Pet dataset directory structure.

    Expected directory layout (standard dataset download):
        root/
          images/           *.jpg  (images)
          annotations/
            list.txt        (filename, class_id, species, breed_id)
            xmls/           *.xml  (bounding boxes)
            trimaps/        *.png  (pixel-wise segmentation masks)

    The dataset supports three tasks simultaneously:
        - Classification : 37 breed labels (0-indexed).
        - Localization   : bounding box [x_center, y_center, w, h] in pixel space.
        - Segmentation   : trimap mask (1=pet, 2=background, 3=boundary → remapped to 0,1,2).

    Args:
        root        : Path to the dataset root directory.
        split       : 'train' or 'val' (or 'test').
        transform   : Optional albumentations/torchvision transform for images.
        img_size    : Resize target (default 224 per VGG11 paper).
        task        : One of 'all', 'classification', 'localization', 'segmentation'.
    """

    # 37 breed names (alphabetical, matching list.txt class ids 1-37)
    BREEDS = [
        "Abyssinian", "Bengal", "Birman", "Bombay", "British_Shorthair",
        "Egyptian_Mau", "Maine_Coon", "Persian", "Ragdoll", "Russian_Blue",
        "Siamese", "Sphynx", "american_bulldog", "american_pit_bull_terrier",
        "basset_hound", "beagle", "boxer", "chihuahua", "english_cocker_spaniel",
        "english_setter", "german_shorthaired", "great_pyrenees", "havanese",
        "japanese_chin", "keeshond", "leonberger", "miniature_pinscher",
        "newfoundland", "pomeranian", "pug", "saint_bernard", "samoyed",
        "scottish_terrier", "shiba_inu", "staffordshire_bull_terrier",
        "wheaten_terrier", "yorkshire_terrier",
    ]

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        img_size: int = 224,
        task: str = "all",
    ):
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.img_size = img_size
        self.task = task

        self.img_dir = self.root / "images"
        self.ann_dir = self.root / "annotations"
        self.xml_dir = self.ann_dir / "xmls"
        self.mask_dir = self.ann_dir / "trimaps"

        self.samples = self._load_split()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_split(self):
        """Parse list.txt and return list of (filename_stem, class_id_0idx)."""
        list_file = self.ann_dir / "list.txt"
        samples = []
        if not list_file.exists():
            raise FileNotFoundError(f"list.txt not found at {list_file}")

        with open(list_file) as f:
            lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]

        # Split deterministically: first 90% train, last 10% val
        n = len(lines)
        if self.split == "train":
            lines = lines[: int(0.9 * n)]
        elif self.split in ("val", "test"):
            lines = lines[int(0.9 * n) :]
        # else use all lines

        for line in lines:
            parts = line.split()
            if len(parts) < 2:
                continue
            stem = parts[0]          # e.g. "Abyssinian_100"
            class_id = int(parts[1]) - 1  # 1-indexed → 0-indexed
            samples.append((stem, class_id))

        return samples

    def _parse_bbox_xml(self, xml_path: Path) -> Optional[Tuple[float, float, float, float]]:
        """Parse Pascal VOC XML and return (x_center, y_center, w, h) in pixel space."""
        try:
            import xml.etree.ElementTree as ET
            tree = ET.parse(xml_path)
            root = tree.getroot()
            obj = root.find("object")
            if obj is None:
                return None
            bndbox = obj.find("bndbox")
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)
            w = xmax - xmin
            h = ymax - ymin
            cx = xmin + w / 2.0
            cy = ymin + h / 2.0
            return cx, cy, w, h
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        stem, class_id = self.samples[idx]

        # ---- Image ----
        img_path = self.img_dir / f"{stem}.jpg"
        image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size  # PIL gives (W, H)

        # ---- Segmentation mask (trimap) ----
        mask_path = self.mask_dir / f"{stem}.png"
        mask = None
        if mask_path.exists():
            mask = np.array(Image.open(mask_path))
            # Trimap values: 1=pet, 2=background, 3=boundary → 0,1,2
            mask = (mask - 1).clip(0, 2).astype(np.int64)

        # ---- Bounding box ----
        bbox = None
        xml_path = self.xml_dir / f"{stem}.xml"
        if xml_path.exists():
            bbox = self._parse_bbox_xml(xml_path)

        # ---- Resize image (and scale bbox) ----
        image = image.resize((self.img_size, self.img_size), Image.BILINEAR)
        scale_x = self.img_size / orig_w
        scale_y = self.img_size / orig_h

        if bbox is not None:
            cx, cy, bw, bh = bbox
            bbox = (
                cx * scale_x,
                cy * scale_y,
                bw * scale_x,
                bh * scale_y,
            )

        if mask is not None:
            mask = np.array(
                Image.fromarray(mask.astype(np.uint8)).resize(
                    (self.img_size, self.img_size), Image.NEAREST
                )
            ).astype(np.int64)

        # ---- Apply transforms ----
        img_np = np.array(image)  # H×W×3, uint8
        if self.transform is not None:
            if mask is not None:
                augmented = self.transform(image=img_np, mask=mask)
                img_np = augmented["image"]
                mask = augmented["mask"]
            else:
                augmented = self.transform(image=img_np)
                img_np = augmented["image"]

        # Convert to tensor: normalise to [0,1] and reorder to C×H×W
        if isinstance(img_np, np.ndarray):
            img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
        else:
            img_tensor = img_np  # already a tensor (albumentations ToTensorV2)

        # Normalise with ImageNet stats (standard for VGG11)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std

        # ---- Build output dict ----
        sample = {
            "image": img_tensor,
            "label": torch.tensor(class_id, dtype=torch.long),
        }

        if bbox is not None:
            sample["bbox"] = torch.tensor(list(bbox), dtype=torch.float32)
        else:
            sample["bbox"] = torch.zeros(4, dtype=torch.float32)

        if mask is not None:
            if isinstance(mask, np.ndarray):
                sample["mask"] = torch.from_numpy(mask).long()
            else:
                sample["mask"] = mask.long()
        else:
            sample["mask"] = torch.zeros(self.img_size, self.img_size, dtype=torch.long)

        return sample