r"""Gradient Magnitude Similarity Deviation (GMSD)
and Multi-Scale Gradient Magnitude Similiarity Deviation (MS-GMSD)

This module implements the GMSD and MS-GMSD in PyTorch.

References:
    [1] Gradient Magnitude Similarity Deviation:
    An Highly Efficient Perceptual Image Quality Index
    (Xue et al., 2013)
    https://arxiv.org/abs/1308.3052

    [2] Gradient Magnitude Similarity Deviation on
    multiple scales for color image quality assessment
    (Zhang et al., 2017)
    https://ieeexplore.ieee.org/document/7952357
"""

__pdoc__ = {'_gmsd': True, '_ms_gmsd': True}

import torch
import torch.nn as nn
import torch.nn.functional as F

from piqa.utils import (
    _jit,
    build_reduce,
    prewitt_kernel,
    gradient_kernel,
    channel_conv,
    tensor_norm,
)

_Y_WEIGHTS = torch.FloatTensor([0.299, 0.587, 0.114])
_MS_WEIGHTS = torch.FloatTensor([0.096, 0.596, 0.289, 0.019])


@_jit
def _gmsd(
    x: torch.Tensor,
    y: torch.Tensor,
    kernel: torch.Tensor,
    value_range: float = 1.,
    c: float = 0.00261,  # 170. / (255. ** 2)
    alpha: float = 0.,
) -> torch.Tensor:
    r"""Returns the GMSD between \(x\) and \(y\),
    without color space conversion and downsampling.

    \(\text{GMSD}(x, y)\) is the standard deviation of the Gradient
    Magnitude Similarity (GMS).

    $$ \text{GMS}(x, y) = \frac{(2 - \alpha) \text{GM}(x) \text{GM}(y)
        + C}{\text{GM}(x)^2 + \text{GM}(y)^2 - \alpha \text{GM}(x)
        \text{GM}(y) + C} $$

    $$ \text{GM}(z) = \left\| \nabla z \right\|_2 $$

    where \(\nabla z\) is the result of a gradient convolution over \(z\).

    Args:
        x: An input tensor, \((N, 1, H, W)\).
        y: A target tensor, \((N, 1, H, W)\).
        kernel: A gradient kernel, \((2, 1, K, K)\).
        value_range: The value range \(L\) of the inputs (usually 1. or 255).

        For the remaining arguments, refer to [1].

    Returns:
        The GMSD vector, \((N,)\).

    Example:
        >>> x = torch.rand(5, 1, 256, 256)
        >>> y = torch.rand(5, 1, 256, 256)
        >>> kernel = gradient_kernel(prewitt_kernel())
        >>> l = _gmsd(x, y, kernel)
        >>> l.size()
        torch.Size([5])
    """

    c *= value_range ** 2

    # Gradient magnitude
    pad = kernel.size(-1) // 2

    gm_x = tensor_norm(channel_conv(x, kernel, padding=pad), dim=[1])
    gm_y = tensor_norm(channel_conv(y, kernel, padding=pad), dim=[1])

    gm_xy = gm_x * gm_y

    # Gradient magnitude similarity
    gms_num = (2. - alpha) * gm_xy + c
    gms_den = gm_x ** 2 + gm_y ** 2 + c

    if alpha > 0.:
        gms_den = gms_den - alpha * gm_xy

    gms = gms_num / gms_den

    # Gradient magnitude similarity deviation
    gmsd = torch.std(gms, dim=(-1, -2))

    return gmsd


@_jit
def _ms_gmsd(
    x: torch.Tensor,
    y: torch.Tensor,
    kernel: torch.Tensor,
    weights: torch.Tensor,
    value_range: float = 1.,
    c: float = 0.00261,
    alpha: float = 0.5,
) -> torch.Tensor:
    r"""Returns the MS-GMSD between \(x\) and \(y\),
    without color space conversion.

    $$ \text{MS-GMSD}(x, y) = \sum^{M}_{i = 1}
        w_i \text{GMSD}(x^i, y^i) $$

    where \(x^i\) and \(y^i\) are obtained by downsampling
    the original tensors by a factor \(2^{i - 1}\).

    Args:
        x: An input tensor, \((N, 1, H, W)\).
        y: A target tensor, \((N, 1, H, W)\).
        kernel: A gradient kernel, \((2, 1, K, K)\).
        weights: The weights \(w_i\) of the scales, \((M,)\).
        value_range: The value range \(L\) of the inputs (usually 1. or 255).

        For the remaining arguments, refer to [2].

    Returns:
        The MS-GMSD vector, \((N,)\).

    Example:
        >>> x = torch.rand(5, 1, 256, 256)
        >>> y = torch.rand(5, 1, 256, 256)
        >>> kernel = gradient_kernel(prewitt_kernel())
        >>> weights = torch.rand(4)
        >>> l = _ms_gmsd(x, y, kernel, weights)
        >>> l.size()
        torch.Size([5])
    """

    gmsds = []

    for i in range(weights.numel()):
        if i > 0:
            x = F.avg_pool2d(x, kernel_size=2, ceil_mode=True)
            y = F.avg_pool2d(y, kernel_size=2, ceil_mode=True)

        gmsds.append(_gmsd(
            x, y, kernel,
            value_range=value_range,
            c=c, alpha=alpha,
        ))

    msgmsd = torch.stack(gmsds, dim=-1) ** 2
    msgmsd = torch.sqrt((msgmsd * weights).sum(dim=-1))

    return msgmsd


def gmsd(
    x: torch.Tensor,
    y: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    r"""Returns the GMSD between \(x\) and \(y\).

    Args:
        x: An input tensor, \((N, 3, H, W)\).
        y: A target tensor, \((N, 3, H, W)\).

        `**kwargs` are transmitted to `GMSD`.

    Returns:
        The GMSD vector, \((N,)\).

    Example:
        >>> x = torch.rand(5, 3, 256, 256)
        >>> y = torch.rand(5, 3, 256, 256)
        >>> l = gmsd(x, y)
        >>> l.size()
        torch.Size([5])
    """

    kwargs['reduction'] = 'none'

    return GMSD(**kwargs).to(x.device)(x, y)


def ms_gmsd(
    x: torch.Tensor,
    y: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    r"""Returns the MS-GMSD between \(x\) and \(y\).

    Args:
        x: An input tensor, \((N, 3, H, W)\).
        y: A target tensor, \((N, 3, H, W)\).

        `**kwargs` are transmitted to `MS_GMSD`.

    Returns:
        The MS-GMSD vector, \((N,)\).

    Example:
        >>> x = torch.rand(5, 3, 256, 256)
        >>> y = torch.rand(5, 3, 256, 256)
        >>> l = ms_gmsd(x, y)
        >>> l.size()
        torch.Size([5])
    """

    kwargs['reduction'] = 'none'

    return MS_GMSD(**kwargs).to(x.device)(x, y)


class GMSD(nn.Module):
    r"""Creates a criterion that measures the GMSD
    between an input and a target.

    Before applying `_gmsd`, the input and target are converted from
    RBG to Y, the luminance color space, and downsampled by a factor 2.

    Args:
        kernel: A gradient kernel, \((2, 1, K, K)\).
            If `None`, use the Prewitt kernel instead.
        reduction: Specifies the reduction to apply to the output:
            `'none'` | `'mean'` | `'sum'`.

        `**kwargs` are transmitted to `_gmsd`.

    Shapes:
        * Input: \((N, 3, H, W)\)
        * Target: \((N, 3, H, W)\)
        * Output: \((N,)\) or \(()\) depending on `reduction`

    Example:
        >>> criterion = GMSD().cuda()
        >>> x = torch.rand(5, 3, 256, 256, requires_grad=True).cuda()
        >>> y = torch.rand(5, 3, 256, 256, requires_grad=True).cuda()
        >>> l = criterion(x, y)
        >>> l.size()
        torch.Size([])
        >>> l.backward()
    """

    def __init__(
        self,
        kernel: torch.Tensor = None,
        reduction: str = 'mean',
        **kwargs,
    ):
        r""""""
        super().__init__()

        if kernel is None:
            kernel = gradient_kernel(prewitt_kernel())

        self.register_buffer('kernel', kernel)
        self.register_buffer('y_weights', _Y_WEIGHTS.view(1, 3, 1, 1))

        self.reduce = build_reduce(reduction)
        self.kwargs = kwargs

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        r"""Defines the computation performed at every call.
        """

        # Downsample
        input = F.avg_pool2d(input, 2, ceil_mode=True)
        target = F.avg_pool2d(target, 2, ceil_mode=True)

        # RGB to Y
        input = F.conv2d(input, self.y_weights)
        target = F.conv2d(target, self.y_weights)

        # GMSD
        l = _gmsd(input, target, self.kernel, **self.kwargs)

        return self.reduce(l)


class MS_GMSD(nn.Module):
    r"""Creates a criterion that measures the MS-GMSD
    between an input and a target.

    Before applying `_ms_gmsd`, the input and target are converted from
    RBG to Y, the luminance color space.

    Args:
        kernel: A gradient kernel, \((2, 1, K, K)\).
            If `None`, use the Prewitt kernel instead.
        weights: The weights of the scales, \((M,)\).
            If `None`, use the official weights instead.
        reduction: Specifies the reduction to apply to the output:
            `'none'` | `'mean'` | `'sum'`.

        `**kwargs` are transmitted to `_ms_gmsd`.

    Shapes:
        * Input: \((N, 3, H, W)\)
        * Target: \((N, 3, H, W)\)
        * Output: \((N,)\) or \(()\) depending on `reduction`

    Example:
        >>> criterion = MS_GMSD().cuda()
        >>> x = torch.rand(5, 3, 256, 256, requires_grad=True).cuda()
        >>> y = torch.rand(5, 3, 256, 256, requires_grad=True).cuda()
        >>> l = criterion(x, y)
        >>> l.size()
        torch.Size([])
        >>> l.backward()
    """

    def __init__(
        self,
        kernel: torch.Tensor = None,
        weights: torch.Tensor = None,
        reduction: str = 'mean',
        **kwargs,
    ):
        r""""""
        super().__init__()

        if kernel is None:
            kernel = gradient_kernel(prewitt_kernel())

        if weights is None:
            weights = _MS_WEIGHTS

        self.register_buffer('kernel', kernel)
        self.register_buffer('weights', weights)
        self.register_buffer('y_weights', _Y_WEIGHTS.view(1, 3, 1, 1))

        self.reduce = build_reduce(reduction)
        self.kwargs = kwargs

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        r"""Defines the computation performed at every call.
        """

        # RGB to Y
        input = F.conv2d(input, self.y_weights)
        target = F.conv2d(target, self.y_weights)

        # MS-GMSD
        l = _ms_gmsd(input, target, self.kernel, self.weights, **self.kwargs)

        return self.reduce(l)
