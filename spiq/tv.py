r"""Total Variation (TV)

This module implements the TV in PyTorch.

Wikipedia:
    https://en.wikipedia.org/wiki/Total_variation
"""

import torch
import torch.nn as nn

from spiq.utils import build_reduce


def tv(x: torch.Tensor, norm: str = 'L2') -> torch.Tensor:
    r"""Returns the TV of `x`.

    Args:
        x: An input tensor, (..., C, H, W).
        norm: A norm function name (`'L1'`, `'L2'` or `'L2_squared'`).
    """

    w_var = x[..., :, 1:] - x[..., :, :-1]
    h_var = x[..., 1:, :] - x[..., :-1, :]

    if norm in ['L2', 'L2_squared']:
        w_var = w_var ** 2
        h_var = h_var ** 2
    else:  # norm == 'L1'
        w_var = w_var.abs()
        h_var = h_var.abs()

    score = w_var.sum(dim=(-1, -2, -3)) + h_var.sum(dim=(-1, -2, -3))

    if norm == 'L2':
        score = torch.sqrt(score)

    return score


class TV(nn.Module):
    r"""Creates a criterion that measures the TV of an input.

    Args:
        norm: A norm function name (`'L1'`, `'L2'` or `'L2_squared'`).
        reduction: A reduction type (`'mean'`, `'sum'` or `'none'`).

    Call:
        The input tensor should be of shape (N, C, H, W).
    """

    def __init__(self, norm: str = 'L2', reduction: str = 'mean'):
        super().__init__()

        self.norm = norm
        self.reduce = build_reduce(reduction)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        l = tv(input, norm=self.norm)

        return self.reduce(l)
