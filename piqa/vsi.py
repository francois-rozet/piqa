r"""Visual Saliency-based Index (VSI)

This module implements the VSI in PyTorch.

Original:
    https://www.putianjian.net/linzhang/IQA/VSI/VSI.html

Wikipedia:
    https://wikipedia.org/wiki/Salience_(neuroscience)#Visual_saliency_modeling

References:
    | VSI: A Visual Saliency-Induced Index for Perceptual Image Quality Assessment (Zhang et al., 2014)
    | https://ieeexplore.ieee.org/document/6873260

    | SDSP: A novel saliency detection method by combining simple priors (Zhang et al., 2013)
    | https://ieeexplore.ieee.org/document/6738036
"""

import torch
import torch.fft as fft
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

from .utils import assert_type
from .utils.color import ColorConv, rgb_to_xyz, xyz_to_lab
from .utils.functional import (
    scharr_kernel,
    gradient_kernel,
    filter_grid,
    log_gabor,
    channel_conv,
    l2_norm,
    downsample,
    reduce_tensor,
)


@torch.jit.script_if_tracing
def vsi(
    x: Tensor,
    y: Tensor,
    vs_x: Tensor,
    vs_y: Tensor,
    kernel: Tensor,
    value_range: float = 1.0,
    c1: float = 1.27,
    c2: float = 386 / 255 ** 2,
    c3: float = 130 / 255 ** 2,
    alpha: float = 0.4,
    beta: float = 0.02,
) -> Tensor:
    r"""Returns the VSI between :math:`x` and :math:`y`, without color space
    conversion and downsampling.

    Args:
        x: An input tensor, :math:`(N, 3 \text{ or } 1, H, W)`.
        y: A target tensor, :math:`(N, 3 \text{ or } 1, H, W)`.
        vs_x: The input visual saliency, :math:`(N, H, W)`.
        vs_y: The target visual saliency, :math:`(N, H, W)`.
        kernel: A gradient kernel, :math:`(2, 1, K, K)`.
        value_range: The value range :math:`L` of the inputs (usually 1 or 255).

    Note:
        For the remaining arguments, refer to Zhang et al. (2014).

    Returns:
        The VSI vector, :math:`(N,)`.

    Example:
        >>> x = torch.rand(5, 3, 256, 256)
        >>> y = torch.rand(5, 3, 256, 256)
        >>> filtr = sdsp_filter(x)
        >>> vs_x, vs_y = sdsp(x, filtr), sdsp(y, filtr)
        >>> kernel = gradient_kernel(scharr_kernel())
        >>> l = vsi(x, y, vs_x, vs_y, kernel)
        >>> l.shape
        torch.Size([5])
    """

    c2 *= value_range ** 2
    c3 *= value_range ** 2

    l_x, l_y = x[:, :1], y[:, :1]

    # Visual saliency similarity
    vs_m = torch.max(vs_x, vs_y)
    s_vs = (2 * vs_x * vs_y + c1) / (vs_x ** 2 + vs_y ** 2 + c1)

    # Gradient magnitude similarity
    pad = kernel.shape[-1] // 2

    g_x = l2_norm(channel_conv(l_x, kernel, padding=pad), dim=1)
    g_y = l2_norm(channel_conv(l_y, kernel, padding=pad), dim=1)

    s_g = (2 * g_x * g_y + c2) / (g_x ** 2 + g_y ** 2 + c2)

    # Chorminance similarity
    if x.shape[1] == 3:
        mn_x, mn_y = x[:, 1:], y[:, 1:]

        s_c = (2 * mn_x * mn_y + c3) / (mn_x ** 2 + mn_y ** 2 + c3)
        s_c = s_c.prod(dim=1)

        s_c = torch.complex(s_c, torch.zeros_like(s_c))
        s_c_beta = (s_c ** beta).real

        s_vs = s_vs * s_c_beta

    # Visual Saliency-based Index
    s = s_vs * s_g ** alpha
    vsi = (s * vs_m).sum(dim=(-1, -2)) / vs_m.sum(dim=(-1, -2))

    return vsi


@torch.jit.script_if_tracing
def sdsp_filter(
    x: Tensor,
    omega_0: float = 0.021,
    sigma_f: float = 1.34,
) -> Tensor:
    r"""Returns the log-Gabor filter for :func:`sdsp`.

    Args:
        x: An input tensor, :math:`(*, H, W)`.

    Note:
        For the remaining arguments, refer to Zhang et al. (2013).

    Returns:
        The filter tensor, :math:`(H, W)`.
    """

    r, _ = filter_grid(x)
    filtr = log_gabor(r, omega_0, sigma_f)
    filtr = filtr * (r <= 0.5)  # low-pass filter

    return filtr


@torch.jit.script_if_tracing
def sdsp(
    x: Tensor,
    filtr: Tensor,
    value_range: float = 1.0,
    sigma_c: float = 0.001,
    sigma_d: float = 145.0,
) -> Tensor:
    r"""Detects salient regions from :math:`x`.

    Args:
        x: An input tensor, :math:`(N, 3, H, W)`.
        filtr: The frequency domain filter, :math:`(H, W)`.
        value_range: The value range :math:`L` of the input (usually 1 or 255).

    Note:
        For the remaining arguments, refer to Zhang et al. (2013).

    Returns:
        The visual saliency tensor, :math:`(N, H, W)`.

    Example:
        >>> x = torch.rand(5, 3, 256, 256)
        >>> filtr = sdsp_filter(x)
        >>> vs = sdsp(x, filtr)
        >>> vs.shape
        torch.Size([5, 256, 256])
    """

    x_lab = xyz_to_lab(rgb_to_xyz(x, value_range))

    # Frequency prior
    x_f = fft.ifft2(fft.fft2(x_lab) * filtr).real
    s_f = l2_norm(x_f, dim=1)

    # Color prior
    x_ab = x_lab[:, 1:]

    lo = x_ab.flatten(-2).min(dim=-1).values
    up = x_ab.flatten(-2).max(dim=-1).values

    lo = lo.reshape(lo.shape + (1, 1))
    up = up.reshape(lo.shape)

    x_ab = (x_ab - lo) / (up - lo + 1e-8)

    s_c = 1 - torch.exp(-x_ab.square().sum(dim=1) / sigma_c ** 2)

    # Location prior
    a, b = [
        torch.arange(n).to(x) - (n - 1) / 2
        for n in x.shape[-2:]
    ]

    s_d = torch.exp(-(a[None, :] ** 2 + b[:, None] ** 2) / sigma_d ** 2)

    # Visual saliency
    vs = s_f * s_c * s_d

    return vs


class VSI(nn.Module):
    r"""Measures the VSI between an input and a target.

    Before applying :func:`vsi`, the input and target are converted from RBG to L(MN)
    and downsampled to a 256-ish resolution.

    The visual saliency maps of the input and target are determined by :func:`sdsp`.

    Args:
        chromatic: Whether to use the chromatic channels (MN) or not.
        downsample: Whether downsampling is enabled or not.
        kernel: A gradient kernel, :math:`(2, 1, K, K)`.
            If :py:`None`, use the Scharr kernel instead.
        reduction: Specifies the reduction to apply to the output:
            `'none'`, `'mean'` or `'sum'`.
        kwargs: Keyword arguments passed to :func:`vsi`.

    Example:
        >>> criterion = VSI()
        >>> x = torch.rand(5, 3, 256, 256, requires_grad=True)
        >>> y = torch.rand(5, 3, 256, 256)
        >>> l = 1 - criterion(x, y)
        >>> l.shape
        torch.Size([])
        >>> l.backward()
    """

    def __init__(
        self,
        chromatic: bool = True,
        downsample: bool = True,
        kernel: Tensor = None,
        reduction: str = 'mean',
        **kwargs,
    ):
        super().__init__()

        if kernel is None:
            kernel = gradient_kernel(scharr_kernel())

        self.register_buffer('kernel', kernel)

        self.convert = ColorConv('RGB', 'LMN' if chromatic else 'L')
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
            The VSI vector, :math:`(N,)` or :math:`()` depending on `reduction`.
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
            x = downsample(x, 256)
            y = downsample(y, 256)

        # Visual saliency
        filtr = sdsp_filter(x)

        vs_x = sdsp(x, filtr, self.value_range)
        vs_y = sdsp(y, filtr, self.value_range)

        # RGB to L(MN)
        x = self.convert(x)
        y = self.convert(y)

        # VSI
        l = vsi(x, y, vs_x, vs_y, kernel=self.kernel, **self.kwargs)

        return reduce_tensor(l, self.reduction)
