"""Custom IoU loss 
"""

import torch
import torch.nn as nn


class IoULoss(nn.Module):
    """IoU loss for bounding box regression.

    Computes 1 - IoU for boxes in (x_center, y_center, width, height) format.
    The loss is in range [0, 1]: 0 means perfect overlap, 1 means no overlap.

    Two reduction modes are supported:
        'mean' (default): average the per-sample losses.
        'sum'           : sum the per-sample losses.

    Gradient viability: the IoU is computed entirely via differentiable
    torch operations so gradients flow back to pred_boxes.
    Numerical stability: eps is added to the union denominator to avoid
    division by zero when a predicted box has zero area.
    """

    def __init__(self, eps: float = 1e-6, reduction: str = "mean"):
        """
        Initialize the IoULoss module.
        Args:
            eps: Small value to avoid division by zero.
            reduction: Specifies the reduction to apply to the output: 'mean' | 'sum'.
        """
        super().__init__()
        self.eps = eps
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError(
                f"Invalid reduction '{reduction}'. Choose from 'mean', 'sum', 'none'."
            )
        self.reduction = reduction

    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """Compute IoU loss between predicted and target bounding boxes.

        Args:
            pred_boxes:   [B, 4] predicted boxes in (x_center, y_center, width, height) format.
            target_boxes: [B, 4] target boxes  in (x_center, y_center, width, height) format.

        Returns:
            Scalar loss (or per-sample tensor if reduction='none').
        """
        # ---- Convert (cx, cy, w, h) → (x1, y1, x2, y2) ----
        # Predicted
        pred_x1 = pred_boxes[:, 0] - pred_boxes[:, 2] / 2.0
        pred_y1 = pred_boxes[:, 1] - pred_boxes[:, 3] / 2.0
        pred_x2 = pred_boxes[:, 0] + pred_boxes[:, 2] / 2.0
        pred_y2 = pred_boxes[:, 1] + pred_boxes[:, 3] / 2.0

        # Target
        tgt_x1 = target_boxes[:, 0] - target_boxes[:, 2] / 2.0
        tgt_y1 = target_boxes[:, 1] - target_boxes[:, 3] / 2.0
        tgt_x2 = target_boxes[:, 0] + target_boxes[:, 2] / 2.0
        tgt_y2 = target_boxes[:, 1] + target_boxes[:, 3] / 2.0

        # ---- Intersection ----
        inter_x1 = torch.max(pred_x1, tgt_x1)
        inter_y1 = torch.max(pred_y1, tgt_y1)
        inter_x2 = torch.min(pred_x2, tgt_x2)
        inter_y2 = torch.min(pred_y2, tgt_y2)

        inter_w = (inter_x2 - inter_x1).clamp(min=0.0)
        inter_h = (inter_y2 - inter_y1).clamp(min=0.0)
        inter_area = inter_w * inter_h  # [B]

        # ---- Union ----
        pred_area = (pred_x2 - pred_x1).clamp(min=0.0) * (pred_y2 - pred_y1).clamp(min=0.0)
        tgt_area  = (tgt_x2  - tgt_x1).clamp(min=0.0) * (tgt_y2  - tgt_y1).clamp(min=0.0)
        union_area = pred_area + tgt_area - inter_area + self.eps  # [B]

        # ---- IoU ∈ [0, 1] ----
        iou = inter_area / union_area          # [B]

        # ---- Loss = 1 - IoU  (also ∈ [0, 1]) ----
        loss = 1.0 - iou                       # [B]

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss