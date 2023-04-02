r"""Gradient Magnitude Similarity Deviation (GMSD)
and Multi-Scale Gradient Magnitude Similarity Deviation (MS-GMSD)

This module implements the GMSD and MS-GMSD in PyTorch.

Original:
    https://www4.comp.polyu.edu.hk/~cslzhang/IQA/GMSD/GMSD.htm

References:
    | Gradient Magnitude Similarity Deviation: An Highly Efficient Perceptual Image Quality Index (Xue et al., 2013)
    | https://arxiv.org/abs/1308.3052

    | Gradient Magnitude Similarity Deviation on multiple scales for color image quality assessment (Zhang et al., 2017)
    | https://ieeexplore.ieee.org/document/7952357
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

from .utils import assert_type
from .utils.color import ColorConv
from .utils.functional import (
    prewitt_kernel,
    gradient_kernel,
    channel_conv,
    l2_norm,
    reduce_tensor,
)


@torch.jit.script_if_tracing
def gmsd(
    x: Tensor,
    y: Tensor,
    kernel: Tensor,
    value_range: float = 1.0,
    c: float = 170 / 255 ** 2,
    alpha: float = 0.0,
) -> Tensor:
    r"""Returns the GMSD between :math:`x` and :math:`y`, without color space
    conversion and downsampling.

    :math:`\text{GMSD}(x, y)` is the standard deviation of the Gradient Magnitude
    Similarity (GMS).

    .. math::
        \text{GMS}(x, y) &= \frac{(2 - \alpha) \text{GM}(x) \text{GM}(y) + C}
            {\text{GM}(x)^2 + \text{GM}(y)^2 - \alpha \text{GM}(x) \text{GM}(y) + C} \\
        \text{GM}(z) &= \left\| \nabla z \right\|_2

    where :math:`\nabla z` is the result of a gradient convolution over :math:`z`.

    Args:
        x: An input tensor, :math:`(N, 1, H, W)`.
        y: A target tensor, :math:`(N, 1, H, W)`.
        kernel: A gradient kernel, :math:`(2, 1, K, K)`.
        value_range: The value range :math:`L` of the inputs (usually 1 or 255).

    Note:
        For the remaining arguments, refer to Xue et al. (2013).

    Returns:
        The GMSD vector, :math:`(N,)`.

    Example:
        >>> x = torch.rand(5, 1, 256, 256)
        >>> y = torch.rand(5, 1, 256, 256)
        >>> kernel = gradient_kernel(prewitt_kernel())
        >>> l = gmsd(x, y, kernel)
        >>> l.shape
        torch.Size([5])
    """

    c *= value_range ** 2

    # Gradient magnitude
    pad = kernel.shape[-1] // 2

    gm_x = l2_norm(channel_conv(x, kernel, padding=pad), dim=1)
    gm_y = l2_norm(channel_conv(y, kernel, padding=pad), dim=1)

    gm_xy = gm_x * gm_y

    # Gradient magnitude similarity
    gms_num = (2 - alpha) * gm_xy + c
    gms_den = gm_x ** 2 + gm_y ** 2 - alpha * gm_xy + c
    gms = gms_num / gms_den

    # Gradient magnitude similarity deviation
    gmsd = torch.std(gms, dim=(-1, -2))

    return gmsd


@torch.jit.script_if_tracing
def ms_gmsd(
    x: Tensor,
    y: Tensor,
    kernel: Tensor,
    weights: Tensor,
    value_range: float = 1.0,
    c: float = 170 / 255 ** 2,
    alpha: float = 0.5,
) -> Tensor:
    r"""Returns the MS-GMSD between :math:`x` and :math:`y`, without color space
    conversion.

    .. math::
        \text{MS-GMSD}(x, y) = \sqrt{\sum^{M}_{i = 1} w_i \text{GMSD}(x^i, y^i)^2}

    where :math:`x^i` and :math:`y^i` are obtained by downsampling the initial tensors
    by a factor :math:`2^{i - 1}`.

    Args:
        x: An input tensor, :math:`(N, 1, H, W)`.
        y: A target tensor, :math:`(N, 1, H, W)`.
        kernel: A gradient kernel, :math:`(2, 1, K, K)`.
        weights: The weights :math:`w_i` of the scales, :math:`(M,)`.
        value_range: The value range :math:`L` of the inputs (usually 1 or 255).

    Note:
        For the remaining arguments, refer to Zhang et al. (2017).

    Returns:
        The MS-GMSD vector, :math:`(N,)`.

    Example:
        >>> x = torch.rand(5, 1, 256, 256)
        >>> y = torch.rand(5, 1, 256, 256)
        >>> kernel = gradient_kernel(prewitt_kernel())
        >>> weights = torch.rand(4)
        >>> l = ms_gmsd(x, y, kernel, weights)
        >>> l.shape
        torch.Size([5])
    """

    gmsds = []

    for i in range(weights.numel()):
        if i > 0:
            x = F.avg_pool2d(x, kernel_size=2, ceil_mode=True)
            y = F.avg_pool2d(y, kernel_size=2, ceil_mode=True)

        gmsds.append(
            gmsd(
                x, y,
                kernel=kernel,
                value_range=value_range,
                c=c,
                alpha=alpha,
            )
        )

    msgmsd = weights * torch.stack(gmsds, dim=-1).square()
    msgmsd = msgmsd.sum(dim=-1).sqrt()

    return msgmsd


class GMSD(nn.Module):
    r"""Measures the GMSD between an input and a target.

    Before applying :func:`gmsd`, the input and target are converted from RBG to Y, the
    luminance color space, and downsampled by a factor 2.

    Args:
        downsample: Whether downsampling is enabled or not.
        kernel: A gradient kernel, :math:`(2, 1, K, K)`.
            If :py:`None`, use the Prewitt kernel instead.
        reduction: Specifies the reduction to apply to the output:
            `'none'`, `'mean'` or `'sum'`.
        kwargs: Keyword arguments passed to :func:`gmsd`.

    Example:
        >>> criterion = GMSD()
        >>> x = torch.rand(5, 3, 256, 256, requires_grad=True)
        >>> y = torch.rand(5, 3, 256, 256)
        >>> l = criterion(x, y)
        >>> l.shape
        torch.Size([])
        >>> l.backward()
    """

    def __init__(
        self,
        downsample: bool = True,
        kernel: Tensor = None,
        reduction: str = 'mean',
        **kwargs,
    ):
        super().__init__()

        if kernel is None:
            kernel = gradient_kernel(prewitt_kernel())

        self.register_buffer('kernel', kernel)

        self.convert = ColorConv('RGB', 'Y')
        self.downsample = downsample
        self.reduction = reduction
        self.value_range = kwargs.get('value_range', 1.0)
        self.kwargs = kwargs

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        r"""
        Args:
            x: An input tensor, :math:`(N, 3, H, W)`.
            y: A target tensor, :math:`(N, 3, H, W)`.

        Returns:
            The GMSD vector, :math:`(N,)` or :math:`()` depending on `reduction`.
        """

        assert_type(
            x, y,
            device=self.kernel.device,
            dim_range=(4, 4),
            n_channels=3,
            value_range=(0.0, self.value_range),
        )

        # Downsample
        if self.downsample:
            x = F.avg_pool2d(x, 2, ceil_mode=True)
            y = F.avg_pool2d(y, 2, ceil_mode=True)

        # RGB to Y
        x = self.convert(x)
        y = self.convert(y)

        # GMSD
        l = gmsd(x, y, kernel=self.kernel, **self.kwargs)

        return reduce_tensor(l, self.reduction)


class MS_GMSD(nn.Module):
    r"""Measures the MS-GMSD between an input and a target.

    Before applying :func:`ms_gmsd`, the input and target are converted from
    RBG to Y, the luminance color space.

    Args:
        kernel: A gradient kernel, :math:`(2, 1, K, K)`.
            If :py:`None`, use the Prewitt kernel instead.
        weights: The weights of the scales, :math:`(M,)`.
            If :py:`None`, use the :const:`MS_GMSD.WEIGHTS` instead.
        reduction: Specifies the reduction to apply to the output:
            `'none'`, `'mean'` or `'sum'`.
        kwargs: Keyword arguments passed to :func:`ms_gmsd`.

    Example:
        >>> criterion = MS_GMSD()
        >>> x = torch.rand(5, 3, 256, 256, requires_grad=True)
        >>> y = torch.rand(5, 3, 256, 256)
        >>> l = criterion(x, y)
        >>> l.shape
        torch.Size([])
        >>> l.backward()
    """

    WEIGHTS: Tensor = torch.tensor([0.096, 0.596, 0.289, 0.019])
    r"""Scale weights of Zhang et al. (2017)."""

    def __init__(
        self,
        kernel: Tensor = None,
        weights: Tensor = None,
        reduction: str = 'mean',
        **kwargs,
    ):
        super().__init__()

        if kernel is None:
            kernel = gradient_kernel(prewitt_kernel())

        if weights is None:
            weights = self.WEIGHTS

        self.register_buffer('kernel', kernel)
        self.register_buffer('weights', weights)

        self.convert = ColorConv('RGB', 'Y')
        self.reduction = reduction
        self.value_range = kwargs.get('value_range', 1.0)
        self.kwargs = kwargs

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        r"""
        Args:
            x: An input tensor, :math:`(N, 3, H, W)`.
            y: A target tensor, :math:`(N, 3, H, W)`.

        Returns:
            The MS-GMSD vector, :math:`(N,)` or :math:`()` depending on `reduction`.
        """

        assert_type(
            x, y,
            device=self.kernel.device,
            dim_range=(4, 4),
            n_channels=3,
            value_range=(0.0, self.value_range),
        )

        # RGB to Y
        x = self.convert(x)
        y = self.convert(y)

        # MS-GMSD
        l = ms_gmsd(
            x, y,
            kernel=self.kernel,
            weights=self.weights,
            **self.kwargs,
        )

        return reduce_tensor(l, self.reduction)
