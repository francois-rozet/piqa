r"""Total Variation (TV)

This module implements the TV in PyTorch.

Wikipedia:
    https://wikipedia.org/wiki/Total_variation
"""

import torch
import torch.nn as nn

from torch import Tensor

from .utils import assert_type
from .utils.functional import reduce_tensor


@torch.jit.script_if_tracing
def tv(x: Tensor, norm: str = 'L1') -> Tensor:
    r"""Returns the TV of :math:`x`.

    With `'L1'`,

    .. math::
        \text{TV}(x) = \sum_{i, j}
            \left| x_{i+1, j} - x_{i, j} \right| +
            \left| x_{i, j+1} - x_{i, j} \right|

    Alternatively, with `'L2'`,

    .. math::
        \text{TV}(x) = \left( \sum_{c, i, j}
            (x_{c, i+1, j} - x_{c, i, j})^2 +
            (x_{c, i, j+1} - x_{c, i, j})^2 \right)^{\frac{1}{2}}

    Args:
        x: An input tensor, :math:`(*, C, H, W)`.
        norm: Specifies the norm funcion to apply:
            `'L1'`, `'L2'` or `'L2_squared'`.

    Returns:
        The TV tensor, :math:`(*,)`.

    Example:
        >>> x = torch.rand(5, 3, 256, 256)
        >>> l = tv(x)
        >>> l.shape
        torch.Size([5])
    """

    w_var = torch.diff(x, dim=-1)
    h_var = torch.diff(x, dim=-2)

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
    r"""Measures the TV of an input.

    Args:
        reduction: Specifies the reduction to apply to the output:
            `'none'`, `'mean'` or `'sum'`.
        kwargs: Keyword arguments passed to :func:`tv`.

    Example:
        >>> criterion = TV()
        >>> x = torch.rand(5, 3, 256, 256, requires_grad=True)
        >>> l = criterion(x)
        >>> l.shape
        torch.Size([])
        >>> l.backward()
    """

    def __init__(self, reduction: str = 'mean', **kwargs):
        super().__init__()

        self.reduction = reduction
        self.kwargs = kwargs

    def forward(self, x: Tensor) -> Tensor:
        r"""
        Args:
            x: An input tensor, :math:`(N, C, H, W)`.

        Returns:
            The TV vector, :math:`(N,)` or :math:`()` depending on `reduction`.
        """

        assert_type(x, dim_range=(4, 4))

        l = tv(x, **self.kwargs)

        return reduce_tensor(l, self.reduction)
