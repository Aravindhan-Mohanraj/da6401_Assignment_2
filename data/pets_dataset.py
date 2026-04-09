"""Oxford-IIIT Pet dataset — augmentation generator, splitter, and PyTorch Dataset.

This single file handles:
  1) Offline augmentation (run as script or call `build_augmented_set()`)
  2) Stratified train/val splitting
  3) PyTorch Dataset that returns (image, class_label, bbox, segmentation_mask)

Usage as script (to pre-generate augmented data):
    python pets_dataset.py augment --data_dir data/oxford-iiit-pet --num_copies 4

Usage in training:
    from data.pets_dataset import OxfordIIITPetDataset, stratified_train_val_split
    train_recs, val_recs = stratified_train_val_split("data/oxford-iiit-pet/annotations/trainval.txt")
    train_ds = OxfordIIITPetDataset("data/oxford-iiit-pet", entries=train_recs, ...)
"""

import argparse
import os
import pathlib
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Dataset


# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════
INPUT_DIM = 224
MEAN_RGB = (0.485, 0.456, 0.406)
STD_RGB = (0.229, 0.224, 0.225)
_BORDER_REFLECT = 4  # cv2.BORDER_REFLECT_101

TRIMAP_MAP: Dict[int, int] = {1: 0, 2: 1, 3: 2}  # foreground, background, boundary
NUM_BREEDS = 37
NUM_SEG_CLS = 3


# ═══════════════════════════════════════════════════════════════════════════════
# Augmentation policies (for offline generation)
# ═══════════════════════════════════════════════════════════════════════════════

def _aug_spatial(sz=INPUT_DIM):
    """Geometric-only: flip, crop, rotation."""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomResizedCrop(size=(sz, sz), scale=(0.60, 1.0), ratio=(0.85, 1.18), p=1.0),
        A.Rotate(limit=35, border_mode=_BORDER_REFLECT, p=0.8),
    ])


def _aug_color(sz=INPUT_DIM):
    """Color/exposure only, no geometry."""
    return A.Compose([
        A.Resize(sz, sz),
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.06, p=0.9),
        A.RandomGamma(gamma_limit=(70, 130), p=0.5),
        A.CLAHE(clip_limit=3.0, p=0.4),
        A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.3),
    ])


def _aug_full(sz=INPUT_DIM):
    """Combined spatial + color + dropout."""
    return A.Compose([
        A.RandomResizedCrop(size=(sz, sz), scale=(0.6, 1.0), ratio=(0.9, 1.1), p=1.0),
        A.HorizontalFlip(p=0.5),
        A.Affine(scale=(0.9, 1.1), translate_percent=(0.05, 0.1), rotate=(-15, 15), p=0.6),
        A.OneOf([A.GaussianBlur(blur_limit=7), A.MotionBlur(blur_limit=7)], p=0.4),
        A.CoarseDropout(max_holes=6, max_height=32, max_width=32, p=0.5),
        A.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.05, p=0.6),
    ])


def _aug_quality(sz=INPUT_DIM):
    """Simulates real-world degradation: blur, noise, compression."""
    return A.Compose([
        A.Resize(sz, sz),
        A.HorizontalFlip(p=0.5),
        A.OneOf([
            A.GaussianBlur(blur_limit=(5, 9)),
            A.MotionBlur(blur_limit=(5, 9)),
            A.MedianBlur(blur_limit=(5, 7)),
        ], p=0.7),
        A.GaussNoise(p=0.5),
        A.ImageCompression(quality_range=(80, 95), p=0.4),
    ])


_OFFLINE_POLICIES = [_aug_spatial, _aug_color, _aug_full, _aug_quality]


# ═══════════════════════════════════════════════════════════════════════════════
# Annotation file reader
# ═══════════════════════════════════════════════════════════════════════════════

def _read_annotation_file(filepath):
    """Read list.txt / trainval.txt → list of [image_id, cls_id_str, species_str, breed_str]."""
    rows = []
    with open(filepath, "r") as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            rows.append(ln.split())
    return rows


# ═══════════════════════════════════════════════════════════════════════════════
# Bounding box XML parser
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_bbox_from_xml(xml_file, width, height):
    """Parse PASCAL-VOC XML → normalized (cx, cy, w, h) or None."""
    p = pathlib.Path(xml_file)
    if not p.exists():
        return None
    try:
        tree = ET.parse(p)
        box_node = tree.getroot().find(".//bndbox")
        if box_node is None:
            return None
        x1 = max(0.0, min(float(box_node.find("xmin").text), width))
        y1 = max(0.0, min(float(box_node.find("ymin").text), height))
        x2 = max(0.0, min(float(box_node.find("xmax").text), width))
        y2 = max(0.0, min(float(box_node.find("ymax").text), height))
        cw = (x2 - x1) / width
        ch = (y2 - y1) / height
        if cw < 0.01 or ch < 0.01:
            return None
        return ((x1 + x2) / 2.0 / width, (y1 + y2) / 2.0 / height, cw, ch)
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# Stratified train / val split
# ═══════════════════════════════════════════════════════════════════════════════

def stratified_train_val_split(ann_path, val_ratio=0.1, rng_seed=42):
    """Split annotation file into train and val record lists (stratified by class).

    Args:
        ann_path: path to trainval.txt or trainval_aug.txt
        val_ratio: fraction held out for validation
        rng_seed: reproducibility seed

    Returns:
        (train_records, val_records) where each record is [image_id, cls_str, species_str, breed_str]
    """
    all_rows = _read_annotation_file(ann_path)
    class_ids = np.array([int(r[1]) for r in all_rows])

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=rng_seed)
    idx_train, idx_val = next(splitter.split(class_ids, class_ids))

    recs_train = [all_rows[i] for i in idx_train]
    recs_val = [all_rows[i] for i in idx_val]

    # sanity: no overlap
    ids_t = {r[0] for r in recs_train}
    ids_v = {r[0] for r in recs_val}
    assert len(ids_t & ids_v) == 0, "Train/val overlap detected!"

    # sanity: every class represented in both
    t_cls = np.array([int(r[1]) for r in recs_train])
    v_cls = np.array([int(r[1]) for r in recs_val])
    n_cls = class_ids.max()
    assert np.all(np.bincount(t_cls - 1, minlength=n_cls) > 0), "Missing classes in train"
    assert np.all(np.bincount(v_cls - 1, minlength=n_cls) > 0), "Missing classes in val"

    return recs_train, recs_val


# ═══════════════════════════════════════════════════════════════════════════════
# Online transforms (used during training / validation)
# ═══════════════════════════════════════════════════════════════════════════════

def build_train_transform(sz=INPUT_DIM):
    """Training transform with augmentation + normalization. Includes bbox support."""
    bp = A.BboxParams(format="yolo", label_fields=["bbox_labels"], clip=True, min_visibility=0.3, min_area=100)
    return A.Compose([
        A.RandomResizedCrop(size=(sz, sz), scale=(0.7, 1.0), ratio=(0.85, 1.18), p=1.0),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, border_mode=_BORDER_REFLECT, p=0.3),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5),
        A.GaussianBlur(blur_limit=(3, 5), p=0.1),
        A.Normalize(mean=MEAN_RGB, std=STD_RGB),
        ToTensorV2(),
    ], bbox_params=bp)


def build_eval_transform(sz=INPUT_DIM):
    """Eval/val transform: deterministic resize + normalize. Includes bbox support."""
    bp = A.BboxParams(format="yolo", label_fields=["bbox_labels"], clip=True, min_visibility=0.3)
    return A.Compose([
        A.Resize(sz, sz),
        A.Normalize(mean=MEAN_RGB, std=STD_RGB),
        ToTensorV2(),
    ], bbox_params=bp)


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════════════════

class OxfordIIITPetDataset(Dataset):
    """Unified dataset returning (image, class_label, bbox, seg_mask) per sample.

    Two ways to construct:
      1) From a split name: OxfordIIITPetDataset(root, split_name="trainval")
      2) From pre-split records: OxfordIIITPetDataset(root, entries=records, img_dir=..., mask_dir=...)
    """

    def __init__(self, root, split_name=None, entries=None, img_dir=None, mask_dir=None, transform=None, img_size=INPUT_DIM):
        self.base = pathlib.Path(root)
        self.sz = img_size
        self.xml_folder = self.base / "annotations" / "xmls"

        if entries is not None:
            # Construct from pre-split record list
            assert img_dir is not None and mask_dir is not None
            self.img_folder = pathlib.Path(img_dir)
            self.mask_folder = pathlib.Path(mask_dir)
            self.ids = [e[0] for e in entries]
            self.cls_labels = [int(e[1]) - 1 for e in entries]
            self.tfm = transform if transform is not None else build_eval_transform(img_size)

        elif split_name is not None:
            assert split_name in ("trainval", "trainval_aug", "test")
            use_aug = (split_name == "trainval_aug")
            self.img_folder = self.base / ("images_aug" if use_aug else "images")
            self.mask_folder = self.base / "annotations" / ("trimaps_aug" if use_aug else "trimaps")
            ann = self.base / "annotations" / f"{split_name}.txt"
            if not ann.exists():
                raise FileNotFoundError(f"Missing annotation file: {ann}")
            rows = _read_annotation_file(str(ann))
            self.ids = [r[0] for r in rows]
            self.cls_labels = [int(r[1]) - 1 for r in rows]
            self.tfm = transform if transform is not None else (
                build_train_transform(img_size) if "trainval" in split_name else build_eval_transform(img_size)
            )
        else:
            raise ValueError("Provide either split_name or entries")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        sample_id = self.ids[index]
        cls_lbl = self.cls_labels[index]

        # --- image ---
        rgb = np.array(Image.open(self.img_folder / f"{sample_id}.jpg").convert("RGB"))
        h_orig, w_orig = rgb.shape[:2]

        # --- segmentation mask ---
        mpath = self.mask_folder / f"{sample_id}.png"
        if mpath.exists():
            raw_mask = np.array(Image.open(mpath).convert("L"))
        else:
            raw_mask = np.ones((h_orig, w_orig), dtype=np.uint8)

        seg = np.zeros_like(raw_mask, dtype=np.uint8)
        for src_val, tgt_val in TRIMAP_MAP.items():
            seg[raw_mask == src_val] = tgt_val

        # --- bounding box (normalized yolo: cx, cy, w, h) ---
        # strip aug suffix to find original xml
        base_id = sample_id
        for suf in ("_aug1", "_aug2", "_aug3", "_aug4"):
            base_id = base_id.replace(suf, "")
        box = _extract_bbox_from_xml(self.xml_folder / f"{base_id}.xml", w_orig, h_orig)
        if box is None:
            box = (0.5, 0.5, 1.0, 1.0)

        # --- apply transforms ---
        result = self.tfm(image=rgb, mask=seg, bboxes=[box], bbox_labels=[cls_lbl])

        img_tensor = result["image"]
        mask_tensor = result["mask"].long()

        if len(result["bboxes"]) > 0:
            box_tensor = torch.tensor(result["bboxes"][0], dtype=torch.float32)
        else:
            box_tensor = torch.tensor([0.5, 0.5, 1.0, 1.0], dtype=torch.float32)

        return img_tensor, torch.tensor(cls_lbl, dtype=torch.long), box_tensor, mask_tensor


# ═══════════════════════════════════════════════════════════════════════════════
# Offline augmentation builder (run once before training)
# ═══════════════════════════════════════════════════════════════════════════════

def build_augmented_set(data_dir, num_copies=4, sz=INPUT_DIM, rng_seed=42):
    """Generate offline augmented images + masks and write trainval_aug.txt.

    Creates:
        <data_dir>/images_aug/           — resized originals + augmented copies
        <data_dir>/annotations/trimaps_aug/ — corresponding masks
        <data_dir>/annotations/trainval_aug.txt — combined annotation list
    """
    np.random.seed(rng_seed)
    base = pathlib.Path(data_dir)
    src_images = base / "images"
    src_masks = base / "annotations" / "trimaps"
    ann_file = base / "annotations" / "trainval.txt"

    dst_images = base / "images_aug"
    dst_masks = base / "annotations" / "trimaps_aug"
    dst_images.mkdir(exist_ok=True)
    dst_masks.mkdir(exist_ok=True)

    rows = _read_annotation_file(str(ann_file))
    combined_lines = []
    simple_resize = A.Compose([A.Resize(sz, sz)])

    # pass 1: save resized originals
    for r in rows:
        sid, cid, sp, br = r
        img_path = src_images / f"{sid}.jpg"
        if not img_path.exists():
            continue
        pic = np.array(Image.open(img_path).convert("RGB"))
        oh, ow = pic.shape[:2]
        msk_path = src_masks / f"{sid}.png"
        msk = np.array(Image.open(msk_path).convert("L")) if msk_path.exists() else np.ones((oh, ow), dtype=np.uint8)

        out_ip = dst_images / f"{sid}.jpg"
        out_mp = dst_masks / f"{sid}.png"
        if not out_ip.exists():
            t = simple_resize(image=pic, mask=msk)
            Image.fromarray(t["image"]).save(out_ip, quality=95)
            Image.fromarray(t["mask"]).save(out_mp)
        combined_lines.append(f"{sid} {cid} {sp} {br}")

    print(f"Originals written: {len(rows)}")

    # pass 2: generate augmented copies
    active_policies = _OFFLINE_POLICIES[:num_copies]
    n_aug = 0
    for ri, r in enumerate(rows):
        sid, cid, sp, br = r
        img_path = src_images / f"{sid}.jpg"
        if not img_path.exists():
            continue
        pic = np.array(Image.open(img_path).convert("RGB"))
        oh, ow = pic.shape[:2]
        msk_path = src_masks / f"{sid}.png"
        msk = np.array(Image.open(msk_path).convert("L")) if msk_path.exists() else np.ones((oh, ow), dtype=np.uint8)

        for ai, pfn in enumerate(active_policies, start=1):
            aug_name = f"{sid}_aug{ai}"
            o_img = dst_images / f"{aug_name}.jpg"
            o_msk = dst_masks / f"{aug_name}.png"

            if o_img.exists() and o_msk.exists():
                combined_lines.append(f"{aug_name} {cid} {sp} {br}")
                n_aug += 1
                continue

            pipe = pfn(sz)
            res = pipe(image=pic, mask=msk)
            a_img, a_msk = res["image"], res["mask"]
            if a_img.shape[:2] != (sz, sz):
                fix = A.Compose([A.Resize(sz, sz)])(image=a_img, mask=a_msk)
                a_img, a_msk = fix["image"], fix["mask"]

            Image.fromarray(a_img).save(o_img, quality=95)
            Image.fromarray(a_msk.astype(np.uint8)).save(o_msk)
            combined_lines.append(f"{aug_name} {cid} {sp} {br}")
            n_aug += 1

        if (ri + 1) % 500 == 0:
            print(f"  processed {ri + 1}/{len(rows)} ...")

    out_ann = base / "annotations" / "trainval_aug.txt"
    with open(out_ann, "w") as fh:
        fh.write("\n".join(combined_lines) + "\n")

    print(f"\nFinished augmentation.")
    print(f"  Originals : {len(rows)}")
    print(f"  Augmented : {n_aug}")
    print(f"  Total     : {len(combined_lines)}")
    print(f"  Saved to  : {dst_images}")
    print(f"  Annotation: {out_ann}")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Oxford-IIIT Pet dataset preparation")
    sub = parser.add_subparsers(dest="command")

    # sub-command: augment
    aug_p = sub.add_parser("augment", help="Generate offline augmented dataset")
    aug_p.add_argument("--data_dir", default="data/oxford-iiit-pet")
    aug_p.add_argument("--num_copies", type=int, default=4)
    aug_p.add_argument("--img_size", type=int, default=224)
    aug_p.add_argument("--seed", type=int, default=42)

    # sub-command: split
    sp_p = sub.add_parser("split", help="Print stratified split stats")
    sp_p.add_argument("--ann_file", default="data/oxford-iiit-pet/annotations/trainval.txt")
    sp_p.add_argument("--val_ratio", type=float, default=0.1)
    sp_p.add_argument("--seed", type=int, default=42)

    # sub-command: test
    tst_p = sub.add_parser("test", help="Smoke-test the dataset")
    tst_p.add_argument("--data_dir", default="data/oxford-iiit-pet")

    args = parser.parse_args()

    if args.command == "augment":
        build_augmented_set(args.data_dir, args.num_copies, args.img_size, args.seed)

    elif args.command == "split":
        tr, vl = stratified_train_val_split(args.ann_file, args.val_ratio, args.seed)
        tc = np.array([int(r[1]) for r in tr])
        vc = np.array([int(r[1]) for r in vl])
        print(f"Train: {len(tr)} | per-class min={np.bincount(tc - 1).min()} max={np.bincount(tc - 1).max()}")
        print(f"Val:   {len(vl)} | per-class min={np.bincount(vc - 1).min()} max={np.bincount(vc - 1).max()}")
        print("No overlap ✓")

    elif args.command == "test":
        from torch.utils.data import DataLoader
        ds = OxfordIIITPetDataset(args.data_dir, split_name="trainval")
        print(f"Dataset length: {len(ds)}")
        im, lb, bx, mk = ds[0]
        print(f"  image: {im.shape} {im.dtype}")
        print(f"  label: {lb} {lb.dtype}")
        print(f"  bbox:  {bx} {bx.dtype}")
        print(f"  mask:  {mk.shape} {mk.dtype} unique={mk.unique().tolist()}")
        assert im.shape == (3, 224, 224)
        assert bx.shape == (4,)
        assert mk.shape == (224, 224)
        assert set(mk.unique().tolist()).issubset({0, 1, 2})
        batch = next(iter(DataLoader(ds, batch_size=8, shuffle=True, num_workers=0)))
        print(f"\nBatch shapes: img={batch[0].shape} lbl={batch[1].shape} box={batch[2].shape} msk={batch[3].shape}")
        print("All checks passed ✓")

    else:
        parser.print_help()