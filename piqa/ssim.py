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

from piqa.utils import build_reduce, gaussian_kernel, filter2d

from typing import Tuple

_WEIGHTS = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])


def create_window(window_size: int, n_channels: int) -> torch.Tensor:
    r"""Returns the SSIM convolution window (kernel) of size `window_size`.

    Args:
        window_size: The size of the window.
        n_channels: A number of channels.

    Example:
        >>> win = create_window(5, n_channels=3)
        >>> win.size()
        torch.Size([3, 1, 5, 5])
        >>> win[0]
        tensor([[[0.0144, 0.0281, 0.0351, 0.0281, 0.0144],
                 [0.0281, 0.0547, 0.0683, 0.0547, 0.0281],
                 [0.0351, 0.0683, 0.0853, 0.0683, 0.0351],
                 [0.0281, 0.0547, 0.0683, 0.0547, 0.0281],
                 [0.0144, 0.0281, 0.0351, 0.0281, 0.0144]]])
    """

    kernel = gaussian_kernel(window_size, 1.5)
    window = kernel.repeat(n_channels, 1, 1, 1)

    return window


def ssim_per_channel(
    x: torch.Tensor,
    y: torch.Tensor,
    window: torch.Tensor,
    value_range: float = 1.,
    k1: float = 0.01,
    k2: float = 0.03,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Returns the SSIM and the contrast sensitivity per channel
    between `x` and `y`.

    Args:
        x: An input tensor, (N, C, H, W).
        y: A target tensor, (N, C, H, W).
        window: A convolution window, (C, 1, K, K).
        value_range: The value range of the inputs (usually 1. or 255).

        For the remaining arguments, refer to [1].

    Example:
        >>> x = torch.rand(5, 3, 256, 256)
        >>> y = torch.rand(5, 3, 256, 256)
        >>> window = create_window(7, 3)
        >>> ss, cs = ssim_per_channel(x, y, window)
        >>> ss.size(), cs.size()
        (torch.Size([5, 3]), torch.Size([5, 3]))
    """

    c1 = (k1 * value_range) ** 2
    c2 = (k2 * value_range) ** 2

    # Mean (mu)
    mu_x = filter2d(x, window)
    mu_y = filter2d(y, window)

    mu_x_sq = mu_x ** 2
    mu_y_sq = mu_y ** 2
    mu_xy = mu_x * mu_y

    # Variance (sigma)
    sigma_x_sq = filter2d(x ** 2, window) - mu_x_sq
    sigma_y_sq = filter2d(y ** 2, window) - mu_y_sq
    sigma_xy = filter2d(x * y, window) - mu_xy

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

    Example:
        >>> x = torch.rand(5, 3, 256, 256)
        >>> y = torch.rand(5, 3, 256, 256)
        >>> l = ssim(x, y)
        >>> l.size()
        torch.Size([5])
    """

    n_channels = x.size(1)
    window = create_window(window_size, n_channels).to(x.device)

    return ssim_per_channel(x, y, window, **kwargs)[0].mean(-1)


def msssim_per_channel(
    x: torch.Tensor,
    y: torch.Tensor,
    window: torch.Tensor,
    weights: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    """Returns the MS-SSIM per channel between `x` and `y`.

    Args:
        x: An input tensor, (N, C, H, W).
        y: A target tensor, (N, C, H, W).
        window: A convolution window, (C, 1, K, K).
        weights: The weights of the scales, (M,).

        `**kwargs` are transmitted to `ssim_per_channel`.

    Example:
        >>> x = torch.rand(5, 3, 256, 256)
        >>> y = torch.rand(5, 3, 256, 256)
        >>> window = create_window(7, 3)
        >>> weights = torch.rand(5)
        >>> l = msssim_per_channel(x, y, window, weights)
        >>> l.size()
        torch.Size([5, 3])
    """

    css = []

    for i in range(weights.numel()):
        if i > 0:
            x = F.avg_pool2d(x, kernel_size=2, ceil_mode=True)
            y = F.avg_pool2d(y, kernel_size=2, ceil_mode=True)

        ss, cs = ssim_per_channel(x, y, window, **kwargs)
        css.append(torch.relu(cs))

    msss = torch.stack(css[:-1] + [ss], dim=-1)
    msss = (msss ** weights).prod(dim=-1)

    return msss


def msssim(
    x: torch.Tensor,
    y: torch.Tensor,
    window_size: int = 11,
    weights: torch.Tensor = None,
    **kwargs,
) -> torch.Tensor:
    r"""Returns the MS-SSIM between `x` and `y`.

    Args:
        x: An input tensor, (N, C, H, W).
        y: A target tensor, (N, C, H, W).
        window_size: The size of the window.
        weights: The weights of the scales, (M,).
            If `None`, use the official weights instead.

        `**kwargs` are transmitted to `msssim_per_channel`.

    Example:
        >>> x = torch.rand(5, 3, 256, 256)
        >>> y = torch.rand(5, 3, 256, 256)
        >>> l = msssim(x, y)
        >>> l.size()
        torch.Size([5])
    """

    n_channels = x.size(1)
    window = create_window(window_size, n_channels).to(x.device)

    if weights is None:
        weights = _WEIGHTS.to(x.device)

    return msssim_per_channel(x, y, window, weights, **kwargs).mean(-1)


class SSIM(nn.Module):
    r"""Creates a criterion that measures the SSIM
    between an input and a target.

    Args:
        window_size: The size of the window.
        n_channels: A number of channels.
        reduction: Specifies the reduction to apply to the output:
            `'none'` | `'mean'` | `'sum'`.

        `**kwargs` are transmitted to `ssim_per_channel`.

    Shape:
        * Input: (N, C, H, W)
        * Target: (N, C, H, W), same shape as the input
        * Output: (N,) or (1,) depending on `reduction`

    Example:
        >>> criterion = SSIM().cuda()
        >>> x = torch.rand(5, 3, 256, 256).cuda()
        >>> y = torch.rand(5, 3, 256, 256).cuda()
        >>> l = criterion(x, y)
        >>> l.size()
        torch.Size([])
    """

    def __init__(
        self,
        window_size: int = 11,
        n_channels: int = 3,
        reduction: str = 'mean',
        **kwargs,
    ):
        r""""""
        super().__init__()

        self.register_buffer('window', create_window(window_size, n_channels))

        self.reduce = build_reduce(reduction)
        self.kwargs = kwargs

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        r"""Defines the computation performed at every call.
        """

        l = ssim_per_channel(
            input,
            target,
            window=self.window,
            **self.kwargs,
        )[0].mean(-1)

        return self.reduce(l)


class MSSSIM(nn.Module):
    r"""Creates a criterion that measures the MS-SSIM
    between an input and a target.

    Args:
        window_size: The size of the window.
        n_channels: A number of channels.
        weights: The weights of the scales, (M,).
            If `None`, use the official weights instead.
        reduction: Specifies the reduction to apply to the output:
            `'none'` | `'mean'` | `'sum'`.

        `**kwargs` are transmitted to `msssim_per_channel`.

    Shape:
        * Input: (N, C, H, W)
        * Target: (N, C, H, W), same shape as the input
        * Output: (N,) or (1,) depending on `reduction`

    Example:
        >>> criterion = MSSSIM().cuda()
        >>> x = torch.rand(5, 3, 256, 256).cuda()
        >>> y = torch.rand(5, 3, 256, 256).cuda()
        >>> l = criterion(x, y)
        >>> l.size()
        torch.Size([])
    """

    def __init__(
        self,
        window_size: int = 11,
        n_channels: int = 3,
        weights: torch.Tensor = None,
        reduction: str = 'mean',
        **kwargs,
    ):
        r""""""
        super().__init__()

        if weights is None:
            weights = _WEIGHTS

        self.register_buffer('window', create_window(window_size, n_channels))
        self.register_buffer('weights', weights)

        self.reduce = build_reduce(reduction)
        self.kwargs = kwargs

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        r"""Defines the computation performed at every call.
        """

        l = msssim_per_channel(
            input,
            target,
            window=self.window,
            weights=self.weights,
            **self.kwargs,
        ).mean(-1)

        return self.reduce(l)
