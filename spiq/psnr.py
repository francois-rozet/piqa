r"""Peak Signal-to-Noise Ratio (PSNR)

This module implements the PSNR in PyTorch.

Wikipedia:
    https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
"""

###########
# Imports #
###########

import torch
import torch.nn as nn

from typing import Tuple


#############
# Functions #
#############

def psnr(x: torch.Tensor, y: torch.Tensor, dim: Tuple[int, ...] = (), keepdim: bool = False, value_range: float = 1., epsilon: float = 1e-8) -> torch.Tensor:
    r"""Returns the PSNR between `x` and `y`.

    Args:
        x: input tensor
        y: target tensor
        dim: dimension(s) along which to average
        keepdim: whether the output tensor has `dim` retained or not
        value_range: value range of the inputs (usually 1. or 255)
        epsilon: numerical stability term
    """

    mse = ((x - y) ** 2).mean(dim=dim, keepdim=keepdim) + epsilon
    return 10 * torch.log10(value_range ** 2 / mse)


###########
# Classes #
###########

class PSNR(nn.Module):
    r"""Creates a criterion that measures the PSNR between an input and a target.

    Args:
        value_range: value range of the inputs (usually 1. or 255)
        reduction: reduction type (`'mean'`, `'sum'` or `'none'`)

    Call:
        The input and target tensors should be of shape (N, ...).
    """

    def __init__(self, value_range: float = 1., reduction: str = 'mean'):
        super().__init__()

        self.value_range = value_range
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        l = psnr(
            input, target,
            dim=tuple(range(1, input.ndimension())),
            value_range=self.value_range
        )

        if self.reduction == 'mean':
            l = l.mean()
        elif self.reduction == 'sum':
            l = l.sum()

        return l
