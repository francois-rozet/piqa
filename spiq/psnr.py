r"""Peak Signal-to-Noise Ratio (PSNR)

This module implements the PSNR in PyTorch.

Wikipedia:
    https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
"""

import torch
import torch.nn as nn

from spiq.utils import build_reduce

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
    """

    mse = ((x - y) ** 2).mean(dim=dim, keepdim=keepdim) + epsilon
    return 10 * torch.log10(value_range ** 2 / mse)


class PSNR(nn.Module):
    r"""Creates a criterion that measures the PSNR
    between an input and a target.

    Args:
        reduction: A reduction type (`'mean'`, `'sum'` or `'none'`).

        `**kwargs` are transmitted to `psnr`, with
        the exception of `dim` and `keepdim`.

    Call:
        The input and target tensors should be of shape (N, ...).
    """

    def __init__(self, reduction: str = 'mean', **kwargs):
        super().__init__()

        self.reduce = build_reduce(reduction)
        self.kwargs = {
            k: v for k, v in kwargs.items()
            if k not in ['dim', 'keepdim']
        }

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        l = psnr(
            input,
            target,
            dim=tuple(range(1, input.ndimension())),
            **self.kwargs,
        )

        return self.reduce(l)
