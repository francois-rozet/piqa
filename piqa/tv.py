r"""Total Variation (TV)

This module implements the TV in PyTorch.

Wikipedia:
    https://en.wikipedia.org/wiki/Total_variation
"""

import torch
import torch.nn as nn

from piqa.utils import _jit, build_reduce, tensor_norm


@_jit
def tv(x: torch.Tensor, norm: str = 'L2') -> torch.Tensor:
    r"""Returns the TV of `x`.

    Args:
        x: An input tensor, (*, C, H, W).
        norm: Specifies the norm funcion to apply:
            `'L1'` | `'L2'` | `'L2_squared'`.

    Example:
        >>> x = torch.rand(5, 3, 256, 256)
        >>> l = tv(x)
        >>> l.size()
        torch.Size([5])
    """

    w_var = x[..., :, 1:] - x[..., :, :-1]
    h_var = x[..., 1:, :] - x[..., :-1, :]

    if norm == 'L1':
        w_var = w_var.abs()
        h_var = h_var.abs()
    else:  # norm in ['L2', 'L2_squared']
        w_var = w_var ** 2
        h_var = h_var ** 2

    var = w_var.sum(dim=(-1, -2, -3)) + h_var.sum(dim=(-1, -2, -3))

    if norm == 'L2':
        var = torch.sqrt(var)

    return var


class TV(nn.Module):
    r"""Creates a criterion that measures the TV of an input.

    Args:
        reduction: Specifies the reduction to apply to the output:
            `'none'` | `'mean'` | `'sum'`.

        `**kwargs` are transmitted to `tv`.

    Shape:
        * Input: (N, C, H, W)
        * Output: (N,) or (1,) depending on `reduction`

    Example:
        >>> criterion = TV()
        >>> x = torch.rand(5, 3, 256, 256, requires_grad=True).cuda()
        >>> l = criterion(x)
        >>> l.size()
        torch.Size([])
        >>> l.backward()
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
