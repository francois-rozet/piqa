r"""Total Variation (TV)

This module implements the TV in PyTorch.

Wikipedia:
    https://en.wikipedia.org/wiki/Total_variation
"""

import torch
import torch.nn as nn

from piqa.utils import build_reduce, tensor_norm


def tv(x: torch.Tensor, norm: str = 'L2') -> torch.Tensor:
    r"""Returns the TV of `x`.

    Args:
        x: An input tensor, (*, C, H, W).
        norm: Specifies the norm funcion to apply:
            `'L1'` | `'L2'` | `'L2_squared'`.
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
        reduction: Specifies the reduction to apply to the output:
            `'none'` | `'mean'` | `'sum'`.

        `**kwargs` are transmitted to `tv`.

    Shape:
        * Input: (N, C, H, W)
        * Output: (N,) or (1,) depending on `reduction`
    """

    def __init__(self, reduction: str = 'mean', **kwargs):
        r""""""
        super().__init__()

        self.reduce = build_reduce(reduction)
        self.kwargs = kwargs

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        r"""Defines the computation performed at every call.
        """

        l = tv(input, **self.kwargs)

        return self.reduce(l)
