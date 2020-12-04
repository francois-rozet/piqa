r"""Structural Similarity (SSIM) and Multi-Scale Structural Similarity (MS-SSIM)

This module implements the SSIM and MS-SSIM in PyTorch.

Wikipedia:
    https://en.wikipedia.org/wiki/Structural_similarity

Credits:
    Inspired by [pytorch-msssim](https://github.com/VainF/pytorch-msssim)

References:
    [1] Multiscale structural similarity for image quality assessment
    (Wang et al., 2003)
    https://ieeexplore.ieee.org/abstract/document/1292216/

    [2] Image quality assessment: From error visibility to structural similarity
    (Wang et al., 2004)
    https://ieeexplore.ieee.org/abstract/document/1284395/
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from spiq.utils import build_reduce, gaussian_kernel

_WEIGHTS = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])


def create_window(window_size: int, n_channels: int) -> torch.Tensor:
    r"""Returns the SSIM convolution window (kernel) of size `window_size`.

    Args:
        window_size: The size of the window.
        n_channels: A number of channels.
    """

    kernel = gaussian_kernel(window_size, 1.5)

    window = kernel.unsqueeze(0).unsqueeze(0)
    window = window.expand(n_channels, 1, window_size, window_size)

    return window


def ssim_per_channel(
    x: torch.Tensor,
    y: torch.Tensor,
    window: torch.Tensor,
    value_range: float = 1.,
    k1: float = 0.01,
    k2: float = 0.03,
) -> torch.Tensor:
    r"""Returns the SSIM and the contrast sensitivity per channel
    between `x` and `y`.

    Args:
        x: An input tensor, (N, C, H, W).
        y: A target tensor, (N, C, H, W).
        window: A convolution window.
        value_range: The value range of the inputs (usually 1. or 255).

        For the remaining arguments, refer to [1].
    """

    n_channels, _, window_size, _ = window.size()

    c1 = (k1 * value_range) ** 2
    c2 = (k2 * value_range) ** 2

    # Mean (mu)
    mu_x = F.conv2d(x, window, padding=0, groups=n_channels)
    mu_y = F.conv2d(y, window, padding=0, groups=n_channels)

    mu_x_sq = mu_x ** 2
    mu_y_sq = mu_y ** 2
    mu_xy = mu_x * mu_y

    # Variance (sigma)
    sigma_x_sq = F.conv2d(x ** 2, window, padding=0, groups=n_channels)
    sigma_x_sq -= mu_x_sq
    sigma_y_sq = F.conv2d(y ** 2, window, padding=0, groups=n_channels)
    sigma_y_sq -= mu_y_sq
    sigma_xy = F.conv2d(x * y, window, padding=0, groups=n_channels)
    sigma_xy -= mu_xy

    # Contrast sensitivity
    cs = (2. * sigma_xy + c2) / (sigma_x_sq + sigma_y_sq + c2)

    # Structural similarity
    ss = (2. * mu_x * mu_y + c1) / (mu_x_sq + mu_y_sq + c1) * cs

    return ss.mean((-1, -2)), cs.mean((-1, -2))


def ssim(
    x: torch.Tensor,
    y: torch.Tensor,
    window_size: int = 11,
    **kwargs,
) -> torch.Tensor:
    r"""Returns the SSIM between `x` and `y`.

    Args:
        x: An input tensor, (N, C, H, W).
        y: A target tensor, (N, C, H, W).
        window_size: The size of the window.

        `**kwargs` are transmitted to `ssim_per_channel`.
    """

    n_channels = x.size(1)
    window = create_window(window_size, n_channels).to(x.device)

    return ssim_per_channel(x, y, window, **kwargs)[0].mean(-1)


def msssim_per_channel(
    x: torch.Tensor,
    y: torch.Tensor,
    window: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    """Returns the MS-SSIM per channel between `x` and `y`.

    Args:
        x: An input tensor, (N, C, H, W).
        y: A target tensor, (N, C, H, W).
        window: A convolution window.

        `**kwargs` are transmitted to `ssim_per_channel`.
    """

    weights = _WEIGHTS.to(x.device)

    mcs = []

    for i in range(weights.numel()):
        if i > 0:
            padding = (x.shape[-2] % 2, x.shape[-1] % 2)
            x = F.avg_pool2d(x, kernel_size=2, padding=padding)
            y = F.avg_pool2d(y, kernel_size=2, padding=padding)

        ss, cs = ssim_per_channel(x, y, window, **kwargs)
        mcs.append(torch.relu(cs))

    msss = torch.stack(mcs[:-1] + [ss], dim=0)
    msss = msss ** weights.view(-1, 1, 1)
    msss = msss.prod(dim=0)

    return msss


def msssim(
    x: torch.Tensor,
    y: torch.Tensor,
    window_size: int = 11,
    **kwargs,
) -> torch.Tensor:
    r"""Returns the MS-SSIM between `x` and `y`.

    Args:
        x: An input tensor, (N, C, H, W).
        y: A target tensor, (N, C, H, W).
        window_size: The size of the window.

        `**kwargs` are transmitted to `msssim_per_channel`.
    """

    n_channels = x.size(1)
    window = create_window(window_size, n_channels).to(x.device)

    return msssim_per_channel(x, y, window, **kwargs).mean(-1)


class SSIM(nn.Module):
    r"""Creates a criterion that measures the SSIM
    between an input and a target.

    Args:
        window_size: The size of the window.
        n_channels: A number of channels.
        reduction: A reduction type (`'mean'`, `'sum'` or `'none'`).

        `**kwargs` are transmitted to `ssim_per_channel`.

    Call:
        The input and target tensors should be of shape (N, C, H, W).
    """

    def __init__(
        self,
        window_size: int = 11,
        n_channels: int = 3,
        reduction: str = 'mean',
        **kwargs,
    ):
        super().__init__()

        self.register_buffer('window', create_window(window_size, n_channels))

        self.reduce = build_reduce(reduction)
        self.kwargs = kwargs

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        l = ssim_per_channel(
            input,
            target,
            window=self.window,
            **self.kwargs,
        )[0].mean(-1)

        return self.reduce(l)


class MSSSIM(SSIM):
    r"""Creates a criterion that measures the MS-SSIM
    between an input and a target.

    Args:
        All arguments are inherited from `SSIM`.

        `**kwargs` are transmitted to `msssim_per_channel`.

    Call:
        The input and target tensors should be of shape (N, C, H, W).
    """

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        l = msssim_per_channel(
            input,
            target,
            window=self.window,
            **self.kwargs,
        ).mean(-1)

        return self.reduce(l)
