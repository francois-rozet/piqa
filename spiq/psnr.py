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


#############
# Functions #
#############

def psnr(x: torch.Tensor, y: torch.Tensor, dim: tuple=(), value_range: float=1., epsilon: float=1e-8) -> torch.Tensor:
    r"""Returns the PSNR between `x` and `y`.

    Args:
        x: input tensor
        y: target tensor
        dim: dimension(s) to reduce
        value_range: value range of the inputs (usually 1. or 255)
        epsilon: numerical stability
    """

    mse = ((x - y) ** 2).mean(dim=dim) + epsilon
    return 10 * torch.log10(value_range ** 2 / mse)


###########
# Classes #
###########

class PSNR(nn.Module):
    r"""Creates a criterion that measures the PSNR between an input and a target.
    """

    def __init__(self, value_range: float=1., reduction='mean'):
        super().__init__()

        self.value_range = value_range
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            input: input tensor, (N, ...)
            target: target tensor, (N, ...)
        """

        l = psnr(
            input, target,
            dim=tuple(range(1, input.ndimension())),
            value_range=self.value_range
        )

        if self.reduction == 'mean':
            return l.mean()
        elif self.reduction == 'sum':
            return l.sum()

        return l
