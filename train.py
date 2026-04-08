"""Training entrypoint

Usage examples:
    # Task 1 – Classification
    python train.py --task classification --data_dir /path/to/pets --epochs 20

    # Task 2 – Localization
    python train.py --task localization --data_dir /path/to/pets --epochs 20 \
                    --classifier_ckpt checkpoints/classifier.pth

    # Task 3 – Segmentation
    python train.py --task segmentation --data_dir /path/to/pets --epochs 20

All checkpoints are saved to the checkpoints/ directory.
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import wandb

from data.pets_dataset import OxfordIIITPetDataset
from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from models.segmentation import VGG11UNet
from losses.iou_loss import IoULoss

# ────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ────────────────────────────────────────────────────────────────────────────

def get_args():
    parser = argparse.ArgumentParser(description="DA6401 Assignment 2 Training")
    parser.add_argument("--task", type=str, required=True,
                        choices=["classification", "localization", "segmentation"],
                        help="Which task to train")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to Oxford-IIIT Pet dataset root")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout_p", type=float, default=0.5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--classifier_ckpt", type=str, default=None,
                        help="Path to pretrained classifier checkpoint (for localization)")
    parser.add_argument("--wandb_project", type=str, default="da6401-assignment2")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_dataloaders(data_dir: str, batch_size: int, num_workers: int):
    """Build train and val dataloaders with albumentations augmentation."""
    try:
        import albumentations as A
        from albumentations.pytorch import ToTensorV2

        train_transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
            A.Rotate(limit=15, p=0.3),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
        val_transform = A.Compose([
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
        use_albumentations = True
    except ImportError:
        train_transform = None
        val_transform = None
        use_albumentations = False

    train_ds = OxfordIIITPetDataset(
        root=data_dir, split="train",
        transform=train_transform if use_albumentations else None,
    )
    val_ds = OxfordIIITPetDataset(
        root=data_dir, split="val",
        transform=val_transform if use_albumentations else None,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader


# ────────────────────────────────────────────────────────────────────────────
# Task-specific training loops
# ────────────────────────────────────────────────────────────────────────────

def train_classification(args, device):
    """Train VGG11Classifier for 37-breed classification."""
    model = VGG11Classifier(num_classes=37, in_channels=3, dropout_p=args.dropout_p).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    train_loader, val_loader = get_dataloaders(args.data_dir, args.batch_size, args.num_workers)

    best_acc = 0.0
    ckpt_path = os.path.join(args.checkpoint_dir, "classifier.pth")

    for epoch in range(1, args.epochs + 1):
        # ---- Train ----
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for batch in train_loader:
            imgs = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += imgs.size(0)

        train_loss = total_loss / total
        train_acc = correct / total

        # ---- Validate ----
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["image"].to(device)
                labels = batch["label"].to(device)
                logits = model(imgs)
                loss = criterion(logits, labels)
                val_loss += loss.item() * imgs.size(0)
                preds = logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += imgs.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss /= val_total
        val_acc = val_correct / val_total

        # Macro F1
        from sklearn.metrics import f1_score
        macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

        scheduler.step()

        wandb.log({
            "epoch": epoch,
            "train/loss": train_loss,
            "train/acc": train_acc,
            "val/loss": val_loss,
            "val/acc": val_acc,
            "val/macro_f1": macro_f1,
            "lr": scheduler.get_last_lr()[0],
        })
        print(f"[Cls] Epoch {epoch:03d} | "
              f"Train loss {train_loss:.4f} acc {train_acc:.4f} | "
              f"Val loss {val_loss:.4f} acc {val_acc:.4f} F1 {macro_f1:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "best_metric": best_acc,
            }, ckpt_path)
            print(f"  ✔ Saved best classifier checkpoint (acc={best_acc:.4f})")

    return model


def train_localization(args, device):
    """Train VGG11Localizer for bounding-box regression."""
    model = VGG11Localizer(in_channels=3, dropout_p=args.dropout_p).to(device)

    # Optionally initialise encoder from a pretrained classifier
    if args.classifier_ckpt and os.path.exists(args.classifier_ckpt):
        from models.classification import VGG11Classifier
        classifier = VGG11Classifier(num_classes=37, in_channels=3)
        ckpt = torch.load(args.classifier_ckpt, map_location=device)
        classifier.load_state_dict(ckpt.get("state_dict", ckpt))
        model.encoder.load_state_dict(classifier.encoder.state_dict())
        print("  ✔ Loaded encoder weights from classifier checkpoint.")

    mse_loss = nn.MSELoss()
    iou_loss = IoULoss(reduction="mean")
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    train_loader, val_loader = get_dataloaders(args.data_dir, args.batch_size, args.num_workers)

    best_val_loss = float("inf")
    ckpt_path = os.path.join(args.checkpoint_dir, "localizer.pth")

    for epoch in range(1, args.epochs + 1):
        # ---- Train ----
        model.train()
        total_loss, total = 0.0, 0
        for batch in train_loader:
            imgs = batch["image"].to(device)
            bboxes = batch["bbox"].to(device)   # [B, 4]

            optimizer.zero_grad()
            pred = model(imgs)                  # [B, 4]

            loss_mse = mse_loss(pred, bboxes)
            loss_iou = iou_loss(pred, bboxes)
            loss = loss_mse + loss_iou
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            total += imgs.size(0)

        train_loss = total_loss / total

        # ---- Validate ----
        model.eval()
        val_loss_total, val_total = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["image"].to(device)
                bboxes = batch["bbox"].to(device)
                pred = model(imgs)
                loss_mse = mse_loss(pred, bboxes)
                loss_iou = iou_loss(pred, bboxes)
                loss = loss_mse + loss_iou
                val_loss_total += loss.item() * imgs.size(0)
                val_total += imgs.size(0)

        val_loss = val_loss_total / val_total
        scheduler.step()

        wandb.log({
            "epoch": epoch,
            "train/loss": train_loss,
            "val/loss": val_loss,
            "lr": scheduler.get_last_lr()[0],
        })
        print(f"[Loc] Epoch {epoch:03d} | Train loss {train_loss:.4f} | Val loss {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "best_metric": best_val_loss,
            }, ckpt_path)
            print(f"  ✔ Saved best localizer checkpoint (val_loss={best_val_loss:.4f})")

    return model


def train_segmentation(args, device):
    """Train VGG11UNet for semantic segmentation (3 classes)."""
    model = VGG11UNet(num_classes=3, in_channels=3, dropout_p=args.dropout_p).to(device)
    # Cross-entropy is appropriate for 3 mutually exclusive pixel classes
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    train_loader, val_loader = get_dataloaders(args.data_dir, args.batch_size, args.num_workers)

    best_dice = 0.0
    ckpt_path = os.path.join(args.checkpoint_dir, "unet.pth")

    def dice_score(pred_mask, true_mask, num_classes=3, eps=1e-6):
        """Compute mean Dice over classes (excluding class 0 background optionally)."""
        dice = 0.0
        for cls in range(num_classes):
            p = (pred_mask == cls).float()
            t = (true_mask == cls).float()
            dice += (2 * (p * t).sum() + eps) / (p.sum() + t.sum() + eps)
        return (dice / num_classes).item()

    for epoch in range(1, args.epochs + 1):
        # ---- Train ----
        model.train()
        total_loss, total = 0.0, 0
        for batch in train_loader:
            imgs = batch["image"].to(device)
            masks = batch["mask"].to(device)  # [B, H, W] long

            optimizer.zero_grad()
            logits = model(imgs)             # [B, 3, H, W]
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            total += imgs.size(0)

        train_loss = total_loss / total

        # ---- Validate ----
        model.eval()
        val_loss_total, val_total = 0.0, 0
        val_dice_total = 0.0
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["image"].to(device)
                masks = batch["mask"].to(device)
                logits = model(imgs)
                loss = criterion(logits, masks)
                val_loss_total += loss.item() * imgs.size(0)
                val_total += imgs.size(0)

                pred_masks = logits.argmax(dim=1)  # [B, H, W]
                val_dice_total += dice_score(pred_masks, masks) * imgs.size(0)

        val_loss = val_loss_total / val_total
        val_dice = val_dice_total / val_total
        scheduler.step()

        wandb.log({
            "epoch": epoch,
            "train/loss": train_loss,
            "val/loss": val_loss,
            "val/dice": val_dice,
            "lr": scheduler.get_last_lr()[0],
        })
        print(f"[Seg] Epoch {epoch:03d} | Train loss {train_loss:.4f} | "
              f"Val loss {val_loss:.4f} Dice {val_dice:.4f}")

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save({
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "best_metric": best_dice,
            }, ckpt_path)
            print(f"  ✔ Saved best UNet checkpoint (dice={best_dice:.4f})")

    return model


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def main():
    args = get_args()
    set_seed(args.seed)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialise W&B
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        config=vars(args),
        name=f"{args.task}_seed{args.seed}",
    )

    if args.task == "classification":
        train_classification(args, device)
    elif args.task == "localization":
        train_localization(args, device)
    elif args.task == "segmentation":
        train_segmentation(args, device)

    wandb.finish()


if __name__ == "__main__":
    main()