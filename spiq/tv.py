r"""Total Variation (TV)

This module implements the TV in PyTorch.

Wikipedia:
    https://en.wikipedia.org/wiki/Total_variation
"""

###########
# Imports #
###########

import torch
import torch.nn as nn


#############
# Functions #
#############

def tv(x: torch.Tensor, norm='L1') -> torch.Tensor:
    r"""Returns the TV of `x`.

    Args:
        x: input tensor, (..., C, H, W)
        norm: norm to use ('L1', 'L2' or 'L2_squared')
    """

    w_var = x[..., :, 1:] - x[..., :, :-1]
    h_var = x[..., 1:, :] - x[..., :-1, :]

    if norm in ['L2', 'L2_squared']:
        w_var = w_var ** 2
        h_var = h_var ** 2
    else: # norm == 'L1'
        w_var = w_var.abs()
        h_var = h_var.abs()

    score = w_var.sum(dim=(-1, -2, -3)) + h_var.sum(dim=(-1, -2, -3))

    if norm == 'L2':
        score = torch.sqrt(score)

    return score


###########
# Classes #
###########

class TV(nn.Module):
    r"""Creates a criterion that measures the TV of an input.
    """

    def __init__(self, norm='L1', reduction='mean'):
        super().__init__()

        self.norm = norm
        self.reduction = reduction

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            input: input tensor, (N, C, H, W)
        """

        l = tv(input, norm=self.norm)

        if self.reduction == 'mean':
            return l.mean()
        elif self.reduction == 'sum':
            return l.sum()

        return l
