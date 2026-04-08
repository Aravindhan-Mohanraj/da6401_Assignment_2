"""Inference and evaluation

Usage:
    python inference.py --data_dir /path/to/pets --device cuda

Runs the MultiTaskPerceptionModel on the validation split and prints:
    - Classification macro F1-score
    - Localization mean IoU
    - Segmentation mean Dice coefficient
"""

import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.pets_dataset import OxfordIIITPetDataset
from models.multitask import MultiTaskPerceptionModel
from losses.iou_loss import IoULoss


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────

def compute_iou_single(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
    """Compute IoU for a single pair of (cx,cy,w,h) boxes (pixel space)."""
    px1 = pred[0] - pred[2] / 2;  py1 = pred[1] - pred[3] / 2
    px2 = pred[0] + pred[2] / 2;  py2 = pred[1] + pred[3] / 2
    tx1 = target[0] - target[2] / 2;  ty1 = target[1] - target[3] / 2
    tx2 = target[0] + target[2] / 2;  ty2 = target[1] + target[3] / 2

    ix1 = max(px1, tx1);  iy1 = max(py1, ty1)
    ix2 = min(px2, tx2);  iy2 = min(py2, ty2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    pred_area  = max(0, px2 - px1) * max(0, py2 - py1)
    tgt_area   = max(0, tx2 - tx1) * max(0, ty2 - ty1)
    union = pred_area + tgt_area - inter + eps
    return float(inter / union)


def dice_score_batch(pred_masks: torch.Tensor, true_masks: torch.Tensor,
                     num_classes: int = 3, eps: float = 1e-6) -> float:
    """Mean Dice over classes for a batch."""
    dice = 0.0
    for cls in range(num_classes):
        p = (pred_masks == cls).float()
        t = (true_masks == cls).float()
        dice += (2 * (p * t).sum() + eps) / (p.sum() + t.sum() + eps)
    return (dice / num_classes).item()


# ────────────────────────────────────────────────────────────────────────────
# Main evaluation
# ────────────────────────────────────────────────────────────────────────────

def evaluate(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Build dataset / dataloader
    val_ds = OxfordIIITPetDataset(root=args.data_dir, split="val")
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    # Load model
    model = MultiTaskPerceptionModel(
        num_breeds=37,
        seg_classes=3,
        in_channels=3,
        classifier_path=args.classifier_path,
        localizer_path=args.localizer_path,
        unet_path=args.unet_path,
    ).to(device)
    model.eval()

    all_cls_preds, all_cls_labels = [], []
    all_iou = []
    all_dice = []

    with torch.no_grad():
        for batch in val_loader:
            imgs   = batch["image"].to(device)
            labels = batch["label"].to(device)
            bboxes = batch["bbox"].to(device)
            masks  = batch["mask"].to(device)

            outputs = model(imgs)

            # Classification
            cls_preds = outputs["classification"].argmax(dim=1)
            all_cls_preds.extend(cls_preds.cpu().numpy())
            all_cls_labels.extend(labels.cpu().numpy())

            # Localization IoU
            loc_preds = outputs["localization"].cpu()
            loc_tgts  = bboxes.cpu()
            for i in range(loc_preds.size(0)):
                all_iou.append(compute_iou_single(
                    loc_preds[i].numpy(), loc_tgts[i].numpy()
                ))

            # Segmentation Dice
            seg_preds = outputs["segmentation"].argmax(dim=1)
            all_dice.append(dice_score_batch(seg_preds, masks))

    # Metrics
    from sklearn.metrics import f1_score
    macro_f1 = f1_score(all_cls_labels, all_cls_preds, average="macro", zero_division=0)
    mean_iou = float(np.mean(all_iou))
    mean_dice = float(np.mean(all_dice))

    print("=" * 50)
    print(f"  Classification Macro F1 : {macro_f1:.4f}")
    print(f"  Localization Mean IoU   : {mean_iou:.4f}")
    print(f"  Segmentation Mean Dice  : {mean_dice:.4f}")
    print("=" * 50)

    return {"macro_f1": macro_f1, "mean_iou": mean_iou, "mean_dice": mean_dice}


def get_args():
    parser = argparse.ArgumentParser(description="DA6401 Assignment 2 Inference")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--classifier_path", type=str, default="checkpoints/classifier.pth")
    parser.add_argument("--localizer_path",  type=str, default="checkpoints/localizer.pth")
    parser.add_argument("--unet_path",       type=str, default="checkpoints/unet.pth")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    evaluate(args)