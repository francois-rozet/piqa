r"""Peak Signal-to-Noise Ratio (PSNR)

This module implements the PSNR in PyTorch.

Wikipedia:
    https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
"""

import torch
import torch.nn as nn

from torch import Tensor

from .utils import _jit, assert_type, reduce_tensor


@_jit
def mse(x: Tensor, y: Tensor) -> Tensor:
    r"""Returns the Mean Squared Error (MSE) between :math:`x` and :math:`y`.

    .. math::
        \text{MSE}(x, y) = \frac{1}{\text{size}(x)} \sum_i (x_i - y_i)^2

    Args:
        x: An input tensor, :math:`(N, *)`.
        y: A target tensor, :math:`(N, *)`.

    Returns:
        The MSE vector, :math:`(N,)`.

    Example:
        >>> x = torch.rand(5, 3, 256, 256)
        >>> y = torch.rand(5, 3, 256, 256)
        >>> l = mse(x, y)
        >>> l.size()
        torch.Size([5])
    """

    return ((x - y) ** 2).reshape(x.size(0), -1).mean(dim=-1)


@_jit
def psnr(
    x: Tensor,
    y: Tensor,
    epsilon: float = 1e-8,
    value_range: float = 1.,
) -> Tensor:
    r"""Returns the PSNR between :math:`x` and :math:`y`.

    .. math::
        \text{PSNR}(x, y) = 10 \log_{10} \left( \frac{L^2}{\text{MSE}(x, y)} \right)

    Args:
        x: An input tensor, :math:`(N, *)`.
        y: A target tensor, :math:`(N, *)`.
        epsilon: A numerical stability term.
        value_range: The value range :math:`L` of the inputs (usually `1.` or `255`).

    Returns:
        The PSNR vector, :math:`(N,)`.

    Example:
        >>> x = torch.rand(5, 3, 256, 256)
        >>> y = torch.rand(5, 3, 256, 256)
        >>> l = psnr(x, y)
        >>> l.size()
        torch.Size([5])
    """

    return 10 * torch.log10(value_range ** 2 / (mse(x, y) + epsilon))


class PSNR(nn.Module):
    r"""Creates a criterion that measures the PSNR
    between an input and a target.

    Args:
        reduction: Specifies the reduction to apply to the output:
            `'none'` | `'mean'` | `'sum'`.

    Note:
        `**kwargs` are passed to :func:`psnr`.

    Shapes:
        input: :math:`(N, *)`
        target: :math:`(N, *)`
        output: :math:`(N,)` or :math:`()` depending on `reduction`

    Example:
        >>> criterion = PSNR()
        >>> x = torch.rand(5, 3, 256, 256, requires_grad=True).cuda()
        >>> y = torch.rand(5, 3, 256, 256).cuda()
        >>> l = -criterion(x, y)
        >>> l.size()
        torch.Size([])
        >>> l.backward()
    """

    def __init__(self, reduction: str = 'mean', **kwargs):
        super().__init__()

        self.reduction = reduction
        self.value_range = kwargs.get('value_range', 1.)
        self.kwargs = kwargs

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        assert_type(
            input, target,
            dim_range=(1, -1),
            value_range=(0., self.value_range),
        )

        l = psnr(input, target, **self.kwargs)

        return reduce_tensor(l, self.reduction)
