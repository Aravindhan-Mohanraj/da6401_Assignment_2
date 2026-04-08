"""Reusable custom layers 
"""

import torch
import torch.nn as nn


class CustomDropout(nn.Module):
    """Custom Dropout layer.
    
    Implements inverted dropout from scratch without using nn.Dropout or
    F.dropout. During training, each element is zeroed with probability p
    and the remaining elements are scaled by 1/(1-p) so that the expected
    value is preserved (inverted/scaled dropout). During eval, the layer
    is a no-op (identity).
    
    Design choice: Inverted dropout is preferred over vanilla dropout
    because the scaling happens at train time, so the model can be used
    at inference without any modification — the output magnitudes are
    already correct.
    """

    def __init__(self, p: float = 0.5):
        """
        Initialize the CustomDropout layer.

        Args:
            p: Dropout probability. Must be in [0, 1).
        """
        super().__init__()
        if not (0.0 <= p < 1.0):
            raise ValueError(f"Dropout probability must be in [0, 1), got {p}")
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the CustomDropout layer.

        Args:
            x: Input tensor of any shape.

        Returns:
            Output tensor (same shape as input).
        """
        if not self.training or self.p == 0.0:
            # At eval time or p==0, identity — no dropout applied.
            return x

        # Create a binary mask: each element is kept with prob (1 - p).
        # torch.bernoulli samples 1 with the given probability.
        keep_prob = 1.0 - self.p
        # Build a tensor of keep_prob values matching x's shape and device
        mask = torch.bernoulli(torch.full(x.shape, keep_prob, dtype=x.dtype, device=x.device))
        # Inverted dropout: scale kept values by 1/(1-p) so expected value = x
        return x * mask / keep_prob