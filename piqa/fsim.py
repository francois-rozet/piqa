"""Feature Similarity (FSIM)

This module implements the FSIM in PyTorch.

Credits:
    Inspired by the [official implementation](https://www4.comp.polyu.edu.hk/~cslzhang/IQA/FSIM/FSIM.htm)

References:
    [1] FSIM: A Feature Similarity Index for Image Quality Assessment
    (Zhang et al., 2011)
    https://ieeexplore.ieee.org/document/5705575

    [2] Image Features From Phase Congruency
    (Kovesi, 1999)
    https://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.4.1641
"""

import math
import torch
import torch.fft as fft
import torch.nn as nn
import torch.nn.functional as F

from piqa.utils import _jit, _assert_type, _reduce
from piqa.utils.color import ColorConv
from piqa.utils.functional import (
    scharr_kernel,
    gradient_kernel,
    filter_grid,
    log_gabor,
    channel_conv,
)

import piqa.utils.complex as cx


@_jit
def phase_congruency(
    x: torch.Tensor,
    value_range: float = 1.,
    scales: int = 4,
    orientations: int = 4,
    wavelength: float = 6.,
    factor: float = 2.,
    sigma_f: float = 0.5978,  # -log(0.55)
    sigma_theta: float = 0.6545,  # pi / (4 * 1.2)
    k: float = 2.,
    rescale: float = 1.7,
    eps: float = 1e-8,
) -> torch.Tensor:
    r"""Returns the phase congruency of \(x\).

    Args:
        x: An input tensor, \((N, 1, H, W)\).

        For the remaining arguments, refer to [2].

    Returns:
        The PC tensor, \((N, H, W)\).

    Example:
        >>> x = torch.rand(5, 1, 256, 256)
        >>> l = phase_congruency(x)
        >>> l.size()
        torch.Size([5, 256, 256])
    """

    x = x * (255. / value_range)

    # log-Gabor filters
    r, theta = filter_grid(x)  # (H, W)

    ## Radial
    lowpass = 1 / (1 + (r / 0.45) ** (2 * 15))

    a = torch.stack([
        log_gabor(r, 1 / (wavelength * factor ** i), sigma_f) * lowpass
        for i in range(scales)
    ])

    ## Angular
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)

    theta_j = math.pi * torch.arange(orientations).to(x) / orientations
    theta_j = theta_j.view(orientations, 1, 1)

    # Measure (theta - theta_j) in the sine/cosine domains
    # to prevent wrap-around errors
    delta_sin = sin_theta * theta_j.cos() - cos_theta * theta_j.sin()
    delta_cos = cos_theta * theta_j.cos() + sin_theta * theta_j.sin()
    delta_theta = torch.atan2(delta_sin, delta_cos)

    b = torch.exp(-delta_theta ** 2 / (2 * sigma_theta ** 2))

    ## Combine
    filters = a[:, None] * b[None, :]

    # Even & odd (real and imaginary) filter responses
    eo = fft.ifft2(fft.fft2(x[:, None]) * filters)
    eo = torch.view_as_real(eo)  # (N, scales, orientations, H, W, 2)

    ## Amplitude
    a = cx.mod(eo)

    ## Energy
    sum_eo = eo.sum(dim=1, keepdim=True)
    mean_eo = sum_eo / (cx.mod(sum_eo)[..., None] + eps)

    rot90_eo = cx.complex(-cx.imag(eo), cx.real(eo))

    energy = cx.dot(eo, mean_eo) - cx.dot(rot90_eo, mean_eo).abs()
    energy = energy.sum(dim=1, keepdim=True)
    energy = energy.squeeze(1)  # (N, orientations, H, W)

    # Noise
    e2 = a[:, 0] ** 2
    median_e2, _ = torch.median(e2.flatten(-2), dim=-1)
    mean_e2 = -median_e2 / math.log(0.5)

    em = (filters[0] ** 2).sum(dim=(-1, -2))
    noise_power = mean_e2 / em

    ## Total energy^2 due to noise
    ifft_filters = fft.ifft2(filters)
    ifft_filters = cx.real(torch.view_as_real(ifft_filters))

    sum_aiaj = (ifft_filters[None, :] * ifft_filters[:, None]).sum(dim=(0, 1, 3, 4))
    sum_aiaj = sum_aiaj * r.numel()

    noise_energy2 = noise_power * sum_aiaj  # (N, orientations)
    noise_energy2 = noise_energy2[..., None, None]

    ## Noise threshold
    tau = noise_energy2.sqrt()  # Rayleigh parameter

    c, d = (math.pi / 2) ** 0.5, (2 - math.pi / 2) ** 0.5
    noise_threshold = tau * (c + k * d)
    noise_threshold = noise_threshold / rescale  # emprirical rescaling

    energy = (energy - noise_threshold).relu()

    # Phase congruency
    pc = energy.sum(dim=1) / (a.sum(dim=(1, 2)) + eps)  # (N, H, W)

    return pc


@_jit
def fsim(
    x: torch.Tensor,
    y: torch.Tensor,
    kernel: torch.Tensor,
    value_range: float = 1.,
    t1: float = 0.85,
    t2: float = 160. / (255. ** 2),
    t3: float = 200. / (255. ** 2),
    t4: float = 200. / (255. ** 2),
    lmbda: float = 0.03,
) -> torch.Tensor:
    r"""Returns the FSIM between \(x\) and \(y\),
    without color space conversion and downsampling.

    Args:
        x: An input tensor, \((N, 3 \text{ or } 1, H, W)\).
        y: A target tensor, \((N, 3 \text{ or } 1, H, W)\).
        kernel: A gradient kernel, \((2, 1, K, K)\).
        value_range: The value range \(L\) of the inputs (usually 1. or 255).

        For the remaining arguments, refer to [1].

    Returns:
        The FSIM vector, \((N,)\).

    Example:
        >>> x = torch.rand(5, 3, 256, 256)
        >>> y = torch.rand(5, 3, 256, 256)
        >>> kernel = gradient_kernel(scharr_kernel())
        >>> l = fsim(x, y, kernel)
        >>> l.size()
        torch.Size([5])
    """

    t2 *= value_range ** 2
    t3 *= value_range ** 2
    t4 *= value_range ** 2

    y_x, y_y = x[:, :1], y[:, :1]

    # Phase congruency similarity
    pc_x = phase_congruency(y_x, value_range)
    pc_y = phase_congruency(y_y, value_range)
    pc_m = torch.max(pc_x, pc_y)

    s_pc = (2 * pc_x * pc_y + t1) / (pc_x ** 2 + pc_y ** 2 + t1)

    # Gradient magnitude similarity
    pad = kernel.size(-1) // 2

    g_x = torch.linalg.norm(channel_conv(y_x, kernel, padding=pad), dim=1)
    g_y = torch.linalg.norm(channel_conv(y_y, kernel, padding=pad), dim=1)

    s_g = (2 * g_x * g_y + t2) / (g_x ** 2 + g_y ** 2 + t2)

    # Chrominance similarity
    s_l = s_pc * s_g

    if x.size(1) == 3:
        i_x, i_y = x[:, 1], y[:, 1]
        q_x, q_y = x[:, 2], y[:, 2]

        s_i = (2 * i_x * i_y + t3) / (i_x ** 2 + i_y ** 2 + t3)
        s_q = (2 * q_x * q_y + t4) / (q_x ** 2 + q_y ** 2 + t4)

        s_iq = s_i * s_q
        s_iq = cx.complex(s_iq, torch.zeros_like(s_iq))
        s_iq_lambda =  cx.real(cx.pow(s_iq, lmbda))

        s_l = s_l * s_iq_lambda

    # Feature similarity
    fs = (s_l * pc_m).sum(dim=(-1, -2)) / pc_m.sum(dim=(-1, -2))

    return fs


class FSIM(nn.Module):
    r"""Creates a criterion that measures the FSIM
    between an input and a target.

    Before applying `fsim`, the input and target are converted from
    RBG to Y(IQ) and downsampled by a factor \( \frac{\min(H, W)}{256} \).

    Args:
        chromatic: Whether to use the chromatic channels (IQ) or not.
        kernel: A gradient kernel, \((2, 1, K, K)\).
            If `None`, use the Scharr kernel instead.
        reduction: Specifies the reduction to apply to the output:
            `'none'` | `'mean'` | `'sum'`.

        `**kwargs` are transmitted to `fsim`.

    Example:
        >>> criterion = FSIM().cuda()
        >>> x = torch.rand(5, 3, 256, 256, requires_grad=True).cuda()
        >>> y = torch.rand(5, 3, 256, 256).cuda()
        >>> l = 1 - criterion(x, y)
        >>> l.size()
        torch.Size([])
        >>> l.backward()
    """

    def __init__(
        self,
        chromatic: bool = True,
        kernel: torch.Tensor = None,
        reduction: str = 'mean',
        **kwargs,
    ):
        r""""""
        super().__init__()

        if kernel is None:
            kernel = gradient_kernel(scharr_kernel())

        self.register_buffer('kernel', kernel)

        self.convert = ColorConv('RGB', 'YIQ' if chromatic else 'Y')
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

        # RGB to Y(IQ)
        input = self.convert(input)
        target = self.convert(target)

        # FSIM
        l = fsim(input, target, kernel=self.kernel, **self.kwargs)

        return _reduce(l, self.reduction)
