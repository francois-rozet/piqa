r"""Peak Signal-to-Noise Ratio (PSNR)

This module implements the PSNR in PyTorch.

Wikipedia:
    https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
"""

import torch
import torch.nn as nn

from piqa.utils import build_reduce

from typing import Tuple


def psnr(
    x: torch.Tensor,
    y: torch.Tensor,
    dim: Tuple[int, ...] = (),
    keepdim: bool = False,
    value_range: float = 1.,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    r"""Returns the PSNR between `x` and `y`.

    Args:
        x: An input tensor.
        y: A target tensor.
        dim: The dimension(s) along which to average.
        keepdim: Whether the output tensor has `dim` retained or not.
        value_range: The value range of the inputs (usually 1. or 255).
        epsilon: A numerical stability term.

    Example:
        >>> x = torch.rand(5, 3, 256, 256)
        >>> y = torch.rand(5, 3, 256, 256)
        >>> l = psnr(x, y)
        >>> l.size()
        torch.Size([])
    """

    mse = ((x - y) ** 2).mean(dim=dim, keepdim=keepdim) + epsilon
    return 10 * torch.log10(value_range ** 2 / mse)


class PSNR(nn.Module):
    r"""Creates a criterion that measures the PSNR
    between an input and a target.

    Args:
        reduction: Specifies the reduction to apply to the output:
            `'none'` | `'mean'` | `'sum'`.

        `**kwargs` are transmitted to `psnr`, with
        the exception of `dim` and `keepdim`.

    Shape:
        * Input: (N, *), where * means any number of additional dimensions
        * Target: (N, *), same shape as the input
        * Output: (N,) or (1,) depending on `reduction`

    Example:
        >>> criterion = PSNR()
        >>> x = torch.rand(5, 3, 256, 256).cuda()
        >>> y = torch.rand(5, 3, 256, 256).cuda()
        >>> l = criterion(x, y)
        >>> l.size()
        torch.Size([])
    """

    def __init__(self, reduction: str = 'mean', **kwargs):
        r""""""
        super().__init__()

        self.reduce = build_reduce(reduction)
        self.kwargs = {
            k: v for k, v in kwargs.items() if k not in ['dim', 'keepdim']
        }

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        r"""Defines the computation performed at every call.
        """

        l = psnr(
            input.unsqueeze(1).flatten(1),
            target.unsqueeze(1).flatten(1),
            dim=1,
            **self.kwargs,
        )

        return self.reduce(l)
