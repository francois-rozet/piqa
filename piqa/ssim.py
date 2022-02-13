r"""Structural Similarity (SSIM) and Multi-Scale Structural Similarity (MS-SSIM)

This module implements the SSIM and MS-SSIM in PyTorch.

Original:
    https://ece.uwaterloo.ca/~z70wang/research/ssim/

Wikipedia:
    https://en.wikipedia.org/wiki/Structural_similarity

References:
    .. [Wang2004a] Image quality assessment: From error visibility to structural similarity (Wang et al., 2004)

    .. [Wang2004b] Multiscale structural similarity for image quality assessment (Wang et al., 2004)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Tuple

from .utils import _jit, assert_type, reduce_tensor
from .utils.functional import (
    gaussian_kernel,
    kernel_views,
    channel_convs,
)


@_jit
def ssim(
    x: Tensor,
    y: Tensor,
    kernel: Tensor,
    channel_avg: bool = True,
    padding: bool = False,
    value_range: float = 1.,
    k1: float = 0.01,
    k2: float = 0.03,
) -> Tuple[Tensor, Tensor]:
    r"""Returns the SSIM and Contrast Sensitivity (CS) between
    :math:`x` and :math:`y`.

    .. math::
        \text{SSIM}(x, y) &=
            \frac{2 \mu_x \mu_y + C_1}{\mu^2_x + \mu^2_y + C_1} \text{CS}(x, y) \\
        \text{CS}(x, y) &=
            \frac{2 \sigma_{xy} + C_2}{\sigma^2_x + \sigma^2_y + C_2}

    where :math:`\mu_x`, :math:`\mu_y`, :math:`\sigma^2_x`, :math:`\sigma^2_y` and
    :math:`\sigma_{xy}` are the results of a smoothing convolution over
    :math:`x`, :math:`y`, :math:`(x - \mu_x)^2`, :math:`(y - \mu_y)^2` and
    :math:`(x - \mu_x)(y - \mu_y)`, respectively.

    In practice, SSIM and CS are averaged over the spatial dimensions.
    If `channel_avg` is `True`, they are also averaged over the channels.

    Note:
        The number of spatial dimensions is not limited to 2. You can also apply
        :func:`ssim` (and :class:`SSIM`) on images with 3, or even more, dimensions.

    Args:
        x: An input tensor, :math:`(N, C, H, *)`.
        y: A target tensor, :math:`(N, C, H, *)`.
        kernel: A smoothing kernel, :math:`(C, 1, K)`.
        channel_avg: Whether to average over the channels or not.
        padding: Whether to pad with :math:`\frac{K}{2}` zeros the spatial
            dimensions or not.
        value_range: The value range :math:`L` of the inputs (usually `1.` or `255`).

    Note:
        For the remaining arguments, refer to [Wang2004a]_.

    Returns:
        The SSIM and CS tensors, both :math:`(N, C)` or :math:`(N,)`
        depending on `channel_avg`.

    Example:
        >>> x = torch.rand(5, 3, 64, 64, 64)
        >>> y = torch.rand(5, 3, 64, 64, 64)
        >>> kernel = gaussian_kernel(7).repeat(3, 1, 1)
        >>> ss, cs = ssim(x, y, kernel)
        >>> ss.size(), cs.size()
        (torch.Size([5]), torch.Size([5]))
    """

    c1 = (k1 * value_range) ** 2
    c2 = (k2 * value_range) ** 2

    window = kernel_views(kernel, x.dim() - 2)

    if padding:
        pad = kernel.size(-1) // 2
    else:
        pad = 0

    # Mean (mu)
    mu_x = channel_convs(x, window, pad)
    mu_y = channel_convs(y, window, pad)

    mu_xx = mu_x ** 2
    mu_yy = mu_y ** 2
    mu_xy = mu_x * mu_y

    # Variance (sigma)
    sigma_xx = channel_convs(x ** 2, window, pad) - mu_xx
    sigma_yy = channel_convs(y ** 2, window, pad) - mu_yy
    sigma_xy = channel_convs(x * y, window, pad) - mu_xy

    # Contrast sensitivity (CS)
    cs = (2. * sigma_xy + c2) / (sigma_xx + sigma_yy + c2)

    # Structural similarity (SSIM)
    ss = (2. * mu_xy + c1) / (mu_xx + mu_yy + c1) * cs

    # Average
    if channel_avg:
        ss, cs = ss.flatten(1), cs.flatten(1)
    else:
        ss, cs = ss.flatten(2), cs.flatten(2)

    return ss.mean(dim=-1), cs.mean(dim=-1)


@_jit
def ms_ssim(
    x: Tensor,
    y: Tensor,
    kernel: Tensor,
    weights: Tensor,
    padding: bool = False,
    value_range: float = 1.,
    k1: float = 0.01,
    k2: float = 0.03,
) -> Tensor:
    r"""Returns the MS-SSIM between :math:`x` and :math:`y`.

    .. math::
        \text{MS-SSIM}(x, y) = \text{SSIM}(x^M, y^M)^{\gamma_M}
            \prod^{M - 1}_{i = 1} \text{CS}(x^i, y^i)^{\gamma_i}

    where :math:`x^i` and :math:`y^i` are obtained by downsampling
    the initial tensors by a factor :math:`2^{i - 1}`.

    Args:
        x: An input tensor, :math:`(N, C, H, W)`.
        y: A target tensor, :math:`(N, C, H, W)`.
        kernel: A smoothing kernel, :math:`(C, 1, K)`.
        weights: The weights :math:`\gamma_i` of the scales, :math:`(M,)`.
        padding: Whether to pad with :math:`\frac{K}{2}` zeros the spatial
            dimensions or not.
        value_range: The value range :math:`L` of the inputs (usually `1.` or `255`).

    Note:
        For the remaining arguments, refer to [Wang2004b]_.

    Returns:
        The MS-SSIM vector, :math:`(N,)`.

    Example:
        >>> x = torch.rand(5, 3, 256, 256)
        >>> y = torch.rand(5, 3, 256, 256)
        >>> kernel = gaussian_kernel(7).repeat(3, 1, 1)
        >>> weights = torch.rand(5)
        >>> l = ms_ssim(x, y, kernel, weights)
        >>> l.size()
        torch.Size([5])
    """

    css = []

    m = weights.numel()
    for i in range(m):
        if i > 0:
            x = F.avg_pool2d(x, kernel_size=2, ceil_mode=True)
            y = F.avg_pool2d(y, kernel_size=2, ceil_mode=True)

        ss, cs = ssim(
            x, y, kernel,
            channel_avg=False,
            padding=padding,
            value_range=value_range,
            k1=k1, k2=k2,
        )

        css.append(torch.relu(cs) if i + 1 < m else torch.relu(ss))

    msss = torch.stack(css, dim=-1) ** weights
    msss = msss.prod(dim=-1).mean(dim=-1)

    return msss


class SSIM(nn.Module):
    r"""Creates a criterion that measures the SSIM
    between an input and a target.

    Args:
        window_size: The size of the window.
        sigma: The standard deviation of the window.
        n_channels: The number of channels :math:`C`.
        reduction: Specifies the reduction to apply to the output:
            `'none'` | `'mean'` | `'sum'`.

    Note:
        `**kwargs` are passed to :func:`ssim`.

    Shapes:
        input: :math:`(N, C, H, *)`
        target: :math:`(N, C, H, *)`
        output: :math:`(N,)` or :math:`()` depending on `reduction`

    Example:
        >>> criterion = SSIM().cuda()
        >>> x = torch.rand(5, 3, 256, 256, requires_grad=True).cuda()
        >>> y = torch.rand(5, 3, 256, 256).cuda()
        >>> l = 1 - criterion(x, y)
        >>> l.size()
        torch.Size([])
        >>> l.backward()
    """

    def __init__(
        self,
        window_size: int = 11,
        sigma: float = 1.5,
        n_channels: int = 3,
        reduction: str = 'mean',
        **kwargs,
    ):
        super().__init__()

        kernel = gaussian_kernel(window_size, sigma)

        self.register_buffer('kernel', kernel.repeat(n_channels, 1, 1))

        self.reduction = reduction
        self.value_range = kwargs.get('value_range', 1.)
        self.kwargs = kwargs

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        assert_type(
            input, target,
            device=self.kernel.device,
            dim_range=(3, -1),
            n_channels=self.kernel.size(0),
            value_range=(0., self.value_range),
        )

        l = ssim(input, target, kernel=self.kernel, **self.kwargs)[0]

        return reduce_tensor(l, self.reduction)


class MS_SSIM(nn.Module):
    r"""Creates a criterion that measures the MS-SSIM
    between an input and a target.

    Args:
        window_size: The size of the window.
        sigma: The standard deviation of the window.
        n_channels: The number of channels :math:`C`.
        weights: The weights of the scales, :math:`(M,)`.
            If `None`, use :const:`MS_SSIM.WEIGHTS` instead.
        reduction: Specifies the reduction to apply to the output:
            `'none'` | `'mean'` | `'sum'`.

    Note:
        `**kwargs` are passed to :func:`ms_ssim`.

    Shapes:
        input: :math:`(N, C, H, W)`
        target: :math:`(N, C, H, W)`
        output: :math:`(N,)` or :math:`()` depending on `reduction`

    Example:
        >>> criterion = MS_SSIM().cuda()
        >>> x = torch.rand(5, 3, 256, 256, requires_grad=True).cuda()
        >>> y = torch.rand(5, 3, 256, 256).cuda()
        >>> l = 1 - criterion(x, y)
        >>> l.size()
        torch.Size([])
        >>> l.backward()
    """

    WEIGHTS: Tensor = torch.tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    r"""Scale weights of [Wang2004b]_."""

    def __init__(
        self,
        window_size: int = 11,
        sigma: float = 1.5,
        n_channels: int = 3,
        weights: Tensor = None,
        reduction: str = 'mean',
        **kwargs,
    ):
        super().__init__()

        kernel = gaussian_kernel(window_size, sigma)

        self.register_buffer('kernel', kernel.repeat(n_channels, 1, 1))

        if weights is None:
            weights = self.WEIGHTS

        self.register_buffer('weights', weights)

        self.reduction = reduction
        self.value_range = kwargs.get('value_range', 1.)
        self.kwargs = kwargs

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        assert_type(
            input, target,
            device=self.kernel.device,
            dim_range=(4, 4),
            n_channels=self.kernel.size(0),
            value_range=(0., self.value_range),
        )

        l = ms_ssim(
            input, target,
            kernel=self.kernel,
            weights=self.weights,
            **self.kwargs,
        )

        return reduce_tensor(l, self.reduction)
