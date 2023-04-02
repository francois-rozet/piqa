r"""Feature Similarity (FSIM)

This module implements the FSIM in PyTorch.

Original:
    https://www4.comp.polyu.edu.hk/~cslzhang/IQA/FSIM/FSIM.htm

References:
    | FSIM: A Feature Similarity Index for Image Quality Assessment (Zhang et al., 2011)
    | https://ieeexplore.ieee.org/document/5705575

    | Image Features From Phase Congruency (Kovesi, 1999)
"""

import math
import torch
import torch.fft as fft
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

from .utils import assert_type
from .utils.color import ColorConv
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
def fsim(
    x: Tensor,
    y: Tensor,
    pc_x: Tensor,
    pc_y: Tensor,
    kernel: Tensor,
    value_range: float = 1.0,
    t1: float = 0.85,
    t2: float = 160 / 255 ** 2,
    t3: float = 200 / 255 ** 2,
    t4: float = 200 / 255 ** 2,
    lmbda: float = 0.03,
) -> Tensor:
    r"""Returns the FSIM between :math:`x` and :math:`y`, without color space
    conversion and downsampling.

    Args:
        x: An input tensor, :math:`(N, 3 \text{ or } 1, H, W)`.
        y: A target tensor, :math:`(N, 3 \text{ or } 1, H, W)`.
        pc_x: The input phase congruency, :math:`(N, H, W)`.
        pc_y: The target phase congruency, :math:`(N, H, W)`.
        kernel: A gradient kernel, :math:`(2, 1, K, K)`.
        value_range: The value range :math:`L` of the inputs (usually 1 or 255).

    Note:
        For the remaining arguments, refer to Zhang et al. (2011).

    Returns:
        The FSIM vector, :math:`(N,)`.

    Example:
        >>> x = torch.rand(5, 3, 256, 256)
        >>> y = torch.rand(5, 3, 256, 256)
        >>> filters = pc_filters(x)
        >>> pc_x = phase_congruency(x[:, :1], filters)
        >>> pc_y = phase_congruency(y[:, :1], filters)
        >>> kernel = gradient_kernel(scharr_kernel())
        >>> l = fsim(x, y, pc_x, pc_y, kernel)
        >>> l.shape
        torch.Size([5])
    """

    t2 *= value_range ** 2
    t3 *= value_range ** 2
    t4 *= value_range ** 2

    y_x, y_y = x[:, :1], y[:, :1]

    # Phase congruency similarity
    pc_m = torch.max(pc_x, pc_y)
    s_pc = (2 * pc_x * pc_y + t1) / (pc_x ** 2 + pc_y ** 2 + t1)

    # Gradient magnitude similarity
    pad = kernel.shape[-1] // 2

    g_x = l2_norm(channel_conv(y_x, kernel, padding=pad), dim=1)
    g_y = l2_norm(channel_conv(y_y, kernel, padding=pad), dim=1)

    s_g = (2 * g_x * g_y + t2) / (g_x ** 2 + g_y ** 2 + t2)

    # Chrominance similarity
    s_l = s_pc * s_g

    if x.shape[1] == 3:
        i_x, i_y = x[:, 1], y[:, 1]
        q_x, q_y = x[:, 2], y[:, 2]

        s_i = (2 * i_x * i_y + t3) / (i_x ** 2 + i_y ** 2 + t3)
        s_q = (2 * q_x * q_y + t4) / (q_x ** 2 + q_y ** 2 + t4)

        s_iq = s_i * s_q
        s_iq = torch.complex(s_iq, torch.zeros_like(s_iq))
        s_iq_lambda = (s_iq ** lmbda).real

        s_l = s_l * s_iq_lambda

    # Feature similarity
    fs = (s_l * pc_m).sum(dim=(-1, -2)) / pc_m.sum(dim=(-1, -2))

    return fs


@torch.jit.script_if_tracing
def pc_filters(
    x: Tensor,
    scales: int = 4,
    orientations: int = 4,
    wavelength: float = 6.0,
    factor: float = 2.0,
    sigma_f: float = 0.5978,  # -log(0.55)
    sigma_theta: float = 0.6545,  # pi / (4 * 1.2)
) -> Tensor:
    r"""Returns the log-Gabor filters for :func:`phase_congruency`.

    Args:
        x: An input tensor, :math:`(*, H, W)`.
        scales: The number of scales, :math:`S_1`.
        orientations: The number of orientations, :math:`S_2`.

    Note:
        For the remaining arguments, refer to Kovesi (1999).

    Returns:
        The filters tensor, :math:`(S_1, S_2, H, W)`.
    """

    r, theta = filter_grid(x)

    # Low-pass filter
    lowpass = 1 / (1 + (r / 0.45) ** (2 * 15))

    # Radial
    radial = []

    for i in range(scales):
        f_0 = 1 / (wavelength * factor ** i)
        lg = log_gabor(r, f_0, sigma_f)
        radial.append(lg)

    radial = torch.stack(radial)

    # Angular
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)

    theta_j = math.pi * torch.arange(orientations).to(x) / orientations
    theta_j = theta_j.reshape(orientations, 1, 1)

    ## Measure (theta - theta_j) in the sine/cosine domains
    ## to prevent wrap-around errors
    delta_sin = sin_theta * theta_j.cos() - cos_theta * theta_j.sin()
    delta_cos = cos_theta * theta_j.cos() + sin_theta * theta_j.sin()
    delta_theta = torch.atan2(delta_sin, delta_cos)

    angular = torch.exp(-delta_theta ** 2 / (2 * sigma_theta ** 2))

    # Combination
    filters = lowpass * radial[:, None] * angular[None, :]

    return filters


@torch.jit.script_if_tracing
def phase_congruency(
    x: Tensor,
    filters: Tensor,
    value_range: float = 1.0,
    k: float = 2.0,
    rescale: float = 1.7,
    eps: float = 1e-8,
) -> Tensor:
    r"""Returns the Phase Congruency (PC) of :math:`x`.

    Args:
        x: An input tensor, :math:`(N, 1, H, W)`.
        filters: The frequency domain filters, :math:`(S_1, S_2, H, W)`.
        value_range: The value range :math:`L` of the input (usually 1 or 255).

    Note:
        For the remaining arguments, refer to Kovesi (1999).

    Returns:
        The PC tensor, :math:`(N, H, W)`.

    Example:
        >>> x = torch.rand(5, 1, 256, 256)
        >>> filters = pc_filters(x)
        >>> pc = phase_congruency(x, filters)
        >>> pc.shape
        torch.Size([5, 256, 256])
    """

    x = x * (255 / value_range)

    # Filters
    M_hat = filters
    M = fft.ifft2(M_hat).real

    # Even & odd (real and imaginary) responses
    eo = fft.ifft2(fft.fft2(x)[:, None] * M_hat)

    # Expected E^2
    A = eo.abs()
    A2 = A[:, 0].square()
    median_A2 = A2.flatten(-2).median(dim=-1).values
    expect_A2 = median_A2 / math.log(2)

    expect_M2_hat = M_hat[0].square().mean(dim=(-1, -2))
    expect_MiMj = (M[:, None] * M[None, :]).sum(dim=(0, 1, 3, 4))

    expect_E2 = expect_A2 * expect_MiMj / expect_M2_hat

    # Threshold
    sigma_G = expect_E2.sqrt()
    mu_R = sigma_G * math.sqrt(math.pi / 2)
    sigma_R = sigma_G * math.sqrt(2 - math.pi / 2)

    T = mu_R + k * sigma_R
    T = T / rescale  # empirical rescaling
    T = T[..., None, None]

    # Phase deviation
    fh = eo.sum(dim=1, keepdim=True)
    fh = fh / (fh.abs() + eps)

    dot = eo.real * fh.real + eo.imag * fh.imag
    cross = eo.real * fh.imag - eo.imag * fh.real

    E = (dot - cross.abs()).sum(dim=1)

    # Phase congruency
    pc = (E - T).relu().sum(dim=1) / (A.sum(dim=(1, 2)) + eps)

    return pc


class FSIM(nn.Module):
    r"""Measures the FSIM between an input and a target.

    Before applying :func:`fsim`, the input and target are converted from RBG to Y(IQ)
    and downsampled to a 256-ish resolution.

    Args:
        chromatic: Whether to use the chromatic channels (IQ) or not.
        downsample: Whether downsampling is enabled or not.
        kernel: A gradient kernel, :math:`(2, 1, K, K)`.
            If :py:`None`, use the Scharr kernel instead.
        reduction: Specifies the reduction to apply to the output:
            `'none'`, `'mean'` or `'sum'`.
        kwargs: Keyword arguments passed to :func:`fsim`.

    Example:
        >>> criterion = FSIM()
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

        self.convert = ColorConv('RGB', 'YIQ' if chromatic else 'Y')
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
            The FSIM vector, :math:`(N,)` or :math:`()` depending on `reduction`.
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

        # RGB to Y(IQ)
        x = self.convert(x)
        y = self.convert(y)

        # Phase congruency
        filters = pc_filters(x)

        pc_x = phase_congruency(x[:, :1], filters, self.value_range)
        pc_y = phase_congruency(y[:, :1], filters, self.value_range)

        # FSIM
        l = fsim(x, y, pc_x, pc_y, kernel=self.kernel, **self.kwargs)

        return reduce_tensor(l, self.reduction)
