r"""Total Variation (TV)

This module implements the TV in PyTorch.

Wikipedia:
    https://en.wikipedia.org/wiki/Total_variation
"""

import torch
import torch.nn as nn

from spiq.utils import build_reduce, tensor_norm


def tv(x: torch.Tensor, norm: str = 'L2') -> torch.Tensor:
    r"""Returns the TV of `x`.

    Args:
        x: An input tensor, (..., C, H, W).
        norm: A norm function name (`'L1'`, `'L2'` or `'L2_squared'`).
    """

    variation = torch.cat([
        x[..., :, 1:] - x[..., :, :-1],
        x[..., 1:, :] - x[..., :-1, :],
    ], dim=-2)

    tv = tensor_norm(variation, dim=(-1, -2, -3), norm=norm)

    return tv


class TV(nn.Module):
    r"""Creates a criterion that measures the TV of an input.

    Args:
        reduction: A reduction type (`'mean'`, `'sum'` or `'none'`).

        `**kwargs` are transmitted to `tv`.

    Call:
        The input tensor should be of shape (N, C, H, W).
    """

    def __init__(self, reduction: str = 'mean', **kwargs):
        super().__init__()

        self.reduce = build_reduce(reduction)
        self.kwargs = kwargs

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        l = tv(input, **self.kwargs)

        return self.reduce(l)
