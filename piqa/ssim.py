r"""Structural Similarity (SSIM) and Multi-Scale Structural Similarity (MS-SSIM)

This module implements the SSIM and MS-SSIM in PyTorch.

Wikipedia:
    https://en.wikipedia.org/wiki/Structural_similarity

Credits:
    Inspired by [pytorch-msssim](https://github.com/VainF/pytorch-msssim)

References:
    [1] Image quality assessment: From error visibility to structural similarity
    (Wang et al., 2004)
    https://ieeexplore.ieee.org/document/1284395/

    [2] Multiscale structural similarity for image quality assessment
    (Wang et al., 2004)
    https://ieeexplore.ieee.org/document/1292216/
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from piqa.utils import _jit, assert_type, reduce_tensor
from piqa.utils.functional import (
    gaussian_kernel,
    kernel_views,
    channel_convs,
)

from typing import Tuple


@_jit
def ssim(
    x: torch.Tensor,
    y: torch.Tensor,
    kernel: torch.Tensor,
    channel_avg: bool = True,
    padding: bool = False,
    value_range: float = 1.,
    k1: float = 0.01,
    k2: float = 0.03,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Returns the SSIM and Contrast Sensitivity (CS) between
    \(x\) and \(y\).

    $$ \text{SSIM}(x, y) =
        \frac{2 \mu_x \mu_y + C_1}{\mu^2_x + \mu^2_y + C_1} \text{CS}(x, y) $$

    $$ \text{CS}(x, y) =
        \frac{2 \sigma_{xy} + C_2}{\sigma^2_x + \sigma^2_y + C_2} $$

    where \(\mu_x\), \(\mu_y\), \(\sigma^2_x\), \(\sigma^2_y\) and
    \(\sigma_{xy}\) are the results of a smoothing convolution over
    \(x\), \(y\), \((x - \mu_x)^2\), \((y - \mu_y)^2\) and
    \((x - \mu_x)(y - \mu_y)\), respectively.

    In practice, SSIM and CS are averaged over the spatial dimensions.
    If `channel_avg` is `True`, they are also averaged over the channels.

    Note:
        The number of spatial dimensions is not limited to 2. You can apply
        `ssim` (and `SSIM`) on images with 3, or even more, dimensions.

    Args:
        x: An input tensor, \((N, C, H, *)\).
        y: A target tensor, \((N, C, H, *)\).
        kernel: A smoothing kernel, \((C, 1, K)\).
            E.g. `piqa.utils.functional.gaussian_kernel`.
        channel_avg: Whether to average over the channels or not.
        padding: Whether to pad with \(\frac{K}{2}\) zeros the spatial
            dimensions or not.
        value_range: The value range \(L\) of the inputs (usually 1. or 255).

        For the remaining arguments, refer to [1].

    Returns:
        The SSIM and CS tensors, both \((N, C)\) or \((N,)\)
        depending on `channel_avg`

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
    x: torch.Tensor,
    y: torch.Tensor,
    kernel: torch.Tensor,
    weights: torch.Tensor,
    padding: bool = False,
    value_range: float = 1.,
    k1: float = 0.01,
    k2: float = 0.03,
) -> torch.Tensor:
    r"""Returns the MS-SSIM between \(x\) and \(y\).

    $$ \text{MS-SSIM}(x, y) =
        \text{SSIM}(x^M, y^M)^{\gamma_M} \prod^{M - 1}_{i = 1}
        \text{CS}(x^i, y^i)^{\gamma_i} $$

    where \(x^i\) and \(y^i\) are obtained by downsampling
    the original tensors by a factor \(2^{i - 1}\).

    Args:
        x: An input tensor, \((N, C, H, W)\).
        y: A target tensor, \((N, C, H, W)\).
        kernel: A smoothing kernel, \((C, 1, K)\).
            E.g. `piqa.utils.functional.gaussian_kernel`.
        weights: The weights \(\gamma_i\) of the scales, \((M,)\).
        padding: Whether to pad with \(\frac{K}{2}\) zeros the spatial
            dimensions or not.
        value_range: The value range \(L\) of the inputs (usually 1. or 255).

        For the remaining arguments, refer to [2].

    Returns:
        The MS-SSIM vector, \((N,)\).

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

    msss = torch.stack(css, dim=-1)
    msss = (msss ** weights).prod(dim=-1)

    return msss.mean(dim=-1)


class SSIM(nn.Module):
    r"""Creates a criterion that measures the SSIM
    between an input and a target.

    Args:
        window_size: The size of the window.
        sigma: The standard deviation of the window.
        n_channels: The number of channels \(C\).
        reduction: Specifies the reduction to apply to the output:
            `'none'` | `'mean'` | `'sum'`.

        `**kwargs` are transmitted to `ssim`.

    Shapes:
        * Input: \((N, C, H, *)\)
        * Target: \((N, C, H, *)\)
        * Output: \((N,)\) or \(()\) depending on `reduction`

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
        r""""""
        super().__init__()

        kernel = gaussian_kernel(window_size, sigma)

        self.register_buffer('kernel', kernel.repeat(n_channels, 1, 1))

        self.reduction = reduction
        self.value_range = kwargs.get('value_range', 1.)
        self.kwargs = kwargs

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        r"""Defines the computation performed at every call.
        """

        assert_type(
            [input, target],
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
        n_channels: The number of channels \(C\).
        weights: The weights of the scales, \((M,)\).
            If `None`, use the `MS_SSIM.OFFICIAL_WEIGHTS` instead.
        reduction: Specifies the reduction to apply to the output:
            `'none'` | `'mean'` | `'sum'`.

        `**kwargs` are transmitted to `ms_ssim`.

    Shapes:
        * Input: \((N, C, H, W)\)
        * Target: \((N, C, H, W)\)
        * Output: \((N,)\) or \(()\) depending on `reduction`

    Example:
        >>> criterion = MS_SSIM().cuda()
        >>> x = torch.rand(5, 3, 256, 256, requires_grad=True).cuda()
        >>> y = torch.rand(5, 3, 256, 256).cuda()
        >>> l = 1 - criterion(x, y)
        >>> l.size()
        torch.Size([])
        >>> l.backward()
    """

    OFFICIAL_WEIGHTS: torch.Tensor = torch.tensor(
        [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
    )

    def __init__(
        self,
        window_size: int = 11,
        sigma: float = 1.5,
        n_channels: int = 3,
        weights: torch.Tensor = None,
        reduction: str = 'mean',
        **kwargs,
    ):
        r""""""
        super().__init__()

        kernel = gaussian_kernel(window_size, sigma)

        self.register_buffer('kernel', kernel.repeat(n_channels, 1, 1))

        if weights is None:
            weights = self.OFFICIAL_WEIGHTS

        self.register_buffer('weights', weights)

        self.reduction = reduction
        self.value_range = kwargs.get('value_range', 1.)
        self.kwargs = kwargs

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        r"""Defines the computation performed at every call.
        """

        assert_type(
            [input, target],
            device=self.kernel.device,
            dim_range=(4, 4),
            n_channels=self.kernel.size(0),
            value_range=(0., self.value_range),
        )

        l = ms_ssim(
            input,
            target,
            kernel=self.kernel,
            weights=self.weights,
            **self.kwargs,
        )

        return reduce_tensor(l, self.reduction)
