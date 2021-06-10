r"""Visual Saliency-based Index (VSI)

This module implements the VSI in PyTorch.

Wikipedia:
    https://en.wikipedia.org/wiki/Salience_(neuroscience)#Visual_saliency_modeling

Credits:
    Inspired by the [official implementation](https://sse.tongji.edu.cn/linzhang/IQA/VSI/VSI.htm)

References:
    [1] VSI: A Visual Saliency-Induced Index for Perceptual Image Quality Assessment
    (Zhang et al., 2014)
    https://ieeexplore.ieee.org/document/6873260

    [2] SDSP: A novel saliency detection method by combining simple priors
    (Zhang et al., 2013)
    https://ieeexplore.ieee.org/document/6738036
"""

import torch
import torch.fft as fft
import torch.nn as nn
import torch.nn.functional as F

from piqa.utils import _jit, _assert_type, _reduce
from piqa.utils.color import ColorConv, rgb_to_xyz, xyz_to_lab
from piqa.utils.functional import (
    scharr_kernel,
    gradient_kernel,
    filter_grid,
    log_gabor,
    channel_conv,
)

import piqa.utils.complex as cx


@_jit
def vsi(
    x: torch.Tensor,
    y: torch.Tensor,
    vs_x: torch.Tensor,
    vs_y: torch.Tensor,
    kernel: torch.Tensor,
    value_range: float = 1.,
    c1: float = 1.27,
    c2: float = 386. / (255. ** 2),
    c3: float = 130. / (255. ** 2),
    alpha: float = 0.4,
    beta: float = 0.02,
) -> torch.Tensor:
    r"""Returns the VSI between \(x\) and \(y\),
    without downsampling and color space conversion.

    Args:
        x: An input tensor, \((N, 3, H, W)\).
        y: A target tensor, \((N, 3, H, W)\).
        vs_x: The input visual saliency, \((N, H, W)\).
        vs_y: The target visual saliency, \((N, H, W)\).
        kernel: A gradient kernel, \((2, 1, K, K)\).
        value_range: The value range \(L\) of the inputs (usually 1. or 255).

        For the remaining arguments, refer to [1].

    Returns:
        The VSI vector, \((N,)\).

    Example:
        >>> x = torch.rand(5, 3, 256, 256)
        >>> y = torch.rand(5, 3, 256, 256)
        >>> filtr = sdsp_filter(x)
        >>> vs_x, vs_y = sdsp(x, filtr), sdsp(y, filtr)
        >>> kernel = gradient_kernel(scharr_kernel())
        >>> l = vsi(x, y, vs_x, vs_y, kernel)
        >>> l.size()
        torch.Size([5])
    """

    c2 *= value_range ** 2
    c3 *= value_range ** 2

    l_x, mn_x = x[:, :1], x[:, 1:]
    l_y, mn_y = y[:, :1], y[:, 1:]

    # Visual saliency similarity
    vs_m = torch.max(vs_x, vs_y)
    s_vs = (2 * vs_x * vs_y + c1) / (vs_x ** 2 + vs_y ** 2 + c1)

    # Gradient magnitude similarity
    pad = kernel.size(-1) // 2

    g_x = torch.linalg.norm(channel_conv(l_x, kernel, padding=pad), dim=1)
    g_y = torch.linalg.norm(channel_conv(l_y, kernel, padding=pad), dim=1)

    s_g = (2 * g_x * g_y + c2) / (g_x ** 2 + g_y ** 2 + c2)

    # Chorminance similarity
    s_c = (2 * mn_x * mn_y + c3) / (mn_x ** 2 + mn_y ** 2 + c3)
    s_c = s_c.prod(dim=1)

    s_c = cx.complex(s_c, torch.zeros_like(s_c))
    s_c_beta = cx.real(cx.pow(s_c, beta))

    # Visual Saliency-based Index
    s = s_vs * s_g ** alpha * s_c_beta
    vsi = (s * vs_m).sum(dim=(-1, -2)) / vs_m.sum(dim=(-1, -2))

    return vsi


@_jit
def sdsp_filter(
    x: torch.Tensor,
    omega_0: float = 0.021,
    sigma_f: float = 1.34,
) -> torch.Tensor:
    r"""Returns the log-Gabor filter for `sdsp`.

    Args:
        x: An input tensor, \((*, H, W)\).

        For the remaining arguments, refer to [2].

    Returns:
        The filter tensor, \((H, W)\).
    """

    r, _ = filter_grid(x)
    filtr = log_gabor(r, omega_0, sigma_f)
    filtr = filtr * (r <= 0.5)  # low-pass filter

    return filtr


@_jit
def sdsp(
    x: torch.Tensor,
    filtr: torch.Tensor,
    value_range: float = 1.,
    sigma_c: float = 0.001,
    sigma_d: float = 145.,
) -> torch.Tensor:
    r"""Detects salient regions from \(x\).

    Args:
        x: An input tensor, \((N, 3, H, W)\).
        filtr: The frequency domain filter, \((H, W)\).
        value_range: The value range \(L\) of the input (usually 1. or 255).

        For the remaining arguments, refer to [2].

    Returns:
        The visual saliency tensor, \((N, H, W)\).

    Example:
        >>> x = torch.rand(5, 3, 256, 256)
        >>> filtr = sdsp_filter(x)
        >>> vs = sdsp(x, filtr)
        >>> vs.size()
        torch.Size([5, 256, 256])
    """

    x_lab = xyz_to_lab(rgb_to_xyz(x, value_range))

    # Frequency prior
    x_f = fft.ifft2(fft.fft2(x_lab) * filtr)
    x_f = cx.real(torch.view_as_real(x_f))

    s_f = torch.linalg.norm(x_f, dim=1)

    # Color prior
    x_ab = x_lab[:, 1:]

    lo, _ = x_ab.flatten(-2).min(dim=-1)
    up, _ = x_ab.flatten(-2).max(dim=-1)

    lo = lo.view(lo.shape + (1, 1))
    up = up.view(lo.shape)
    span = torch.where(up > lo, up - lo, torch.tensor(1.).to(lo))

    x_ab = (x_ab - lo) / span

    s_c = 1. - torch.exp(-torch.sum(x_ab ** 2, dim=1) / sigma_c ** 2)

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
    r"""Creates a criterion that measures the VSI
    between an input and a target.

    Before applying `vsi`, the input and target are converted from
    RBG to LMN and downsampled by a factor \( \frac{\min(H, W)}{256} \).

    The visual saliency maps of the input and target are determined by `sdsp`.

    Args:
        kernel: A gradient kernel, \((2, 1, K, K)\).
            If `None`, use the Scharr kernel instead.
        reduction: Specifies the reduction to apply to the output:
            `'none'` | `'mean'` | `'sum'`.

        `**kwargs` are transmitted to `vsi`.

    Example:
        >>> criterion = VSI().cuda()
        >>> x = torch.rand(5, 3, 256, 256, requires_grad=True).cuda()
        >>> y = torch.rand(5, 3, 256, 256).cuda()
        >>> l = 1 - criterion(x, y)
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
            kernel = gradient_kernel(scharr_kernel())

        self.register_buffer('kernel', kernel)
        self.register_buffer('filter', torch.zeros((0, 0)))

        self.convert = ColorConv('RGB', 'LMN')
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

        _assert_type(
            [input, target],
            device=self.kernel.device,
            dim_range=(4, 4),
            n_channels=3,
            value_range=(0., self.value_range),
        )

        # Downsample
        _, _, h, w = input.size()
        M = round(min(h, w) / 256)

        if M > 1:
            input = F.avg_pool2d(input, kernel_size=M, ceil_mode=True)
            target = F.avg_pool2d(target, kernel_size=M, ceil_mode=True)

        # Visual saliancy
        if self.filter.shape != (h, w):
            self.filter = sdsp_filter(input)

        vs_input = sdsp(input, self.filter, self.value_range)
        vs_target = sdsp(target, self.filter, self.value_range)

        # RGB to LMN
        input = self.convert(input)
        target = self.convert(target)

        # VSI
        l = vsi(input, target, vs_input, vs_target, kernel=self.kernel, **self.kwargs)

        return _reduce(l, self.reduction)
