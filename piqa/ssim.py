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

from piqa.utils import _jit, build_reduce, gaussian_kernel, channel_sep_conv

from typing import List, Tuple

_MS_WEIGHTS = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])


@_jit
def ssim_per_channel(
    x: torch.Tensor,
    y: torch.Tensor,
    window: List[torch.Tensor],
    value_range: float = 1.,
    non_negative: bool = False,
    k1: float = 0.01,
    k2: float = 0.03,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Returns the SSIM and the contrast sensitivity per channel
    between `x` and `y`.

    Args:
        x: An input tensor, (N, C, H, W).
        y: A target tensor, (N, C, H, W).
        window: A separated kernel, [(C, 1, K, 1), (C, 1, 1, K)].
        value_range: The value range of the inputs (usually 1. or 255).
        non_negative: Whether negative values are clipped or not.

        For the remaining arguments, refer to [2].

    Example:
        >>> x = torch.rand(5, 3, 256, 256)
        >>> y = torch.rand(5, 3, 256, 256)
        >>> window = gaussian_kernel(7, n_channels=3)
        >>> ss, cs = ssim_per_channel(x, y, window)
        >>> ss.size(), cs.size()
        (torch.Size([5, 3]), torch.Size([5, 3]))
    """

    c1 = (k1 * value_range) ** 2
    c2 = (k2 * value_range) ** 2

    # Mean (mu)
    mu_x = channel_sep_conv(x, window)
    mu_y = channel_sep_conv(y, window)

    mu_xx = mu_x ** 2
    mu_yy = mu_y ** 2
    mu_xy = mu_x * mu_y

    # Variance (sigma)
    sigma_xx = channel_sep_conv(x ** 2, window) - mu_xx
    sigma_yy = channel_sep_conv(y ** 2, window) - mu_yy
    sigma_xy = channel_sep_conv(x * y, window) - mu_xy

    # Contrast sensitivity
    cs = (2. * sigma_xy + c2) / (sigma_xx + sigma_yy + c2)

    # Structural similarity
    ss = (2. * mu_xy + c1) / (mu_xx + mu_yy + c1) * cs

    # Average
    ss, cs = ss.mean((-1, -2)), cs.mean((-1, -2))

    if non_negative:
        ss, cs = torch.relu(ss), torch.relu(cs)

    return ss, cs


def ssim(
    x: torch.Tensor,
    y: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    **kwargs,
) -> torch.Tensor:
    r"""Returns the SSIM between `x` and `y`.

    Args:
        x: An input tensor, (N, C, H, W).
        y: A target tensor, (N, C, H, W).
        window_size: The size of the window.
        sigma: The standard deviation of the window.

        `**kwargs` are transmitted to `ssim_per_channel`.

    Example:
        >>> x = torch.rand(5, 3, 256, 256)
        >>> y = torch.rand(5, 3, 256, 256)
        >>> l = ssim(x, y)
        >>> l.size()
        torch.Size([5])
    """

    window = gaussian_kernel(
        window_size, sigma,
        n_channels=x.size(1), device=x.device,
    )

    return ssim_per_channel(x, y, window, **kwargs)[0].mean(-1)


@_jit
def msssim_per_channel(
    x: torch.Tensor,
    y: torch.Tensor,
    window: List[torch.Tensor],
    weights: torch.Tensor,
    value_range: float = 1.,
    k1: float = 0.01,
    k2: float = 0.03,
) -> torch.Tensor:
    """Returns the MS-SSIM per channel between `x` and `y`.

    Args:
        x: An input tensor, (N, C, H, W).
        y: A target tensor, (N, C, H, W).
        window: A separated kernel, [(C, 1, K, 1), (C, 1, 1, K)].
        weights: The weights of the scales, (M,).
        value_range: The value range of the inputs (usually 1. or 255).

        For the remaining arguments, refer to [2].

    Example:
        >>> x = torch.rand(5, 3, 256, 256)
        >>> y = torch.rand(5, 3, 256, 256)
        >>> window = gaussian_kernel(7, n_channels=3)
        >>> weights = torch.rand(5)
        >>> l = msssim_per_channel(x, y, window, weights)
        >>> l.size()
        torch.Size([5, 3])
    """

    css = []

    m = weights.numel()
    for i in range(m):
        if i > 0:
            x = F.avg_pool2d(x, kernel_size=2, ceil_mode=True)
            y = F.avg_pool2d(y, kernel_size=2, ceil_mode=True)

        ss, cs = ssim_per_channel(
            x, y, window,
            value_range=value_range,
            non_negative=True,
            k1=k1, k2=k2,
        )

        css.append(cs if i + 1 < m else ss)

    msss = torch.stack(css, dim=-1)
    msss = (msss ** weights).prod(dim=-1)

    return msss


def msssim(
    x: torch.Tensor,
    y: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    weights: torch.Tensor = None,
    **kwargs,
) -> torch.Tensor:
    r"""Returns the MS-SSIM between `x` and `y`.

    Args:
        x: An input tensor, (N, C, H, W).
        y: A target tensor, (N, C, H, W).
        window_size: The size of the window.
        sigma: The standard deviation of the window.
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

    window = gaussian_kernel(
        window_size, sigma,
        n_channels=x.size(1), device=x.device,
    )

    if weights is None:
        weights = _MS_WEIGHTS.to(x.device)

    return msssim_per_channel(x, y, window, weights, **kwargs).mean(-1)


class SSIM(nn.Module):
    r"""Creates a criterion that measures the SSIM
    between an input and a target.

    Args:
        window_size: The size of the window.
        sigma: The standard deviation of the window.
        n_channels: The number of channels.
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
        sigma: float = 1.5,
        n_channels: int = 3,
        reduction: str = 'mean',
        **kwargs,
    ):
        r""""""
        super().__init__()

        window = gaussian_kernel(window_size, sigma, n_channels=n_channels)

        for i, w in enumerate(window):
            self.register_buffer(f'window{i}', w)

        self.reduce = build_reduce(reduction)
        self.kwargs = kwargs

    @property
    def window(self) -> List[torch.Tensor]:
        return [self.window0, self.window1]

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
        sigma: The standard deviation of the window.
        n_channels: The number of channels.
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
        sigma: float = 1.5,
        n_channels: int = 3,
        weights: torch.Tensor = None,
        reduction: str = 'mean',
        **kwargs,
    ):
        r""""""
        super().__init__()

        window = gaussian_kernel(window_size, sigma, n_channels=n_channels)

        for i, w in enumerate(window):
            self.register_buffer(f'window{i}', w)

        if weights is None:
            weights = _MS_WEIGHTS

        self.register_buffer('weights', weights)

        self.reduce = build_reduce(reduction)
        self.kwargs = kwargs

    @property
    def window(self) -> List[torch.Tensor]:
        return [self.window0, self.window1]

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
