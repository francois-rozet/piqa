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

###########
# Imports #
###########

import torch
import torch.nn as nn
import torch.nn.functional as F

from spiq.utils import gaussian_kernel


#############
# Constants #
#############

_SIGMA = 1.5
_K1, _K2 = 0.01, 0.03
_WEIGHTS = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])


#############
# Functions #
#############

def create_window(window_size: int, n_channels: int) -> torch.Tensor:
    r"""Returns the SSIM convolution window (kernel) of size `window_size`.

    Args:
        window_size: size of the window
        n_channels: number of channels
    """

    kernel = gaussian_kernel(window_size, _SIGMA)

    window = kernel.unsqueeze(0).unsqueeze(0)
    window = window.expand(n_channels, 1, window_size, window_size)

    return window


def ssim_per_channel(x: torch.Tensor, y: torch.Tensor, window: torch.Tensor, value_range: float = 1.) -> torch.Tensor:
    r"""Returns the SSIM and the contrast sensitivity (CS) per channel between `x` and `y`.

    Args:
        x: input tensor, (N, C, H, W)
        y: target tensor, (N, C, H, W)
        window: convolution window
        value_range: value range of the inputs (usually 1. or 255)
    """

    n_channels, _, window_size, _ = window.size()

    mu_x = F.conv2d(x, window, padding=0, groups=n_channels)
    mu_y = F.conv2d(y, window, padding=0, groups=n_channels)

    mu_x_sq = mu_x ** 2
    mu_y_sq = mu_y ** 2
    mu_xy = mu_x * mu_y

    sigma_x_sq = F.conv2d(x ** 2, window, padding=0, groups=n_channels) - mu_x_sq
    sigma_y_sq = F.conv2d(y ** 2, window, padding=0, groups=n_channels) - mu_y_sq
    sigma_xy = F.conv2d(x * y, window, padding=0, groups=n_channels) - mu_xy

    c1 = (_K1 * value_range) ** 2
    c2 = (_K2 * value_range) ** 2

    cs_map = (2. * sigma_xy + c2) / (sigma_x_sq + sigma_y_sq + c2)
    ssim_map = (2. * mu_x * mu_y + c1) / (mu_x_sq + mu_y_sq + c1) * cs_map

    return ssim_map.mean((-1, -2)), cs_map.mean((-1, -2))


def ssim(x: torch.Tensor, y: torch.Tensor, window_size: int = 11, value_range: float = 1.) -> torch.Tensor:
    r"""Returns the SSIM between `x` and `y`.

    Args:
        x: input tensor, (N, C, H, W)
        y: target tensor, (N, C, H, W)
        window_size: size of the window
        value_range: value range of the inputs (usually 1. or 255)
    """

    n_channels = x.size(1)
    window = create_window(window_size, n_channels).to(x.device)

    return ssim_per_channel(x, y, window, value_range)[0].mean(-1)


def msssim_per_channel(x: torch.Tensor, y: torch.Tensor, window: torch.Tensor, value_range: float = 1., weights: torch.Tensor = _WEIGHTS) -> torch.Tensor:
    """Returns the MS-SSIM per channel between `x` and `y`.

    Args:
        x: input tensor, (N, C, H, W)
        y: target tensor, (N, C, H, W)
        window: convolution window
        value_range: value range of the inputs (usually 1. or 255)
        weights: weights of the scales, (M,)
    """

    mcs = []

    for i in range(weights.numel()):
        if i > 0:
            padding = (x.shape[-2] % 2, x.shape[-1] % 2)
            x = F.avg_pool2d(x, kernel_size=2, padding=padding)
            y = F.avg_pool2d(y, kernel_size=2, padding=padding)

        ssim, cs = ssim_per_channel(x, y, window, value_range)
        mcs.append(torch.relu(cs))

    msssim = torch.stack(mcs[:-1] + [ssim], dim=0)
    msssim = msssim ** weights.view(-1, 1, 1)

    return msssim.prod(dim=0)


def msssim(x: torch.Tensor, y: torch.Tensor, window_size: int = 11, value_range: float = 1., weights: torch.Tensor = _WEIGHTS) -> torch.Tensor:
    r"""Returns the MS-SSIM between `x` and `y`.

    Args:
        x: input tensor, (N, C, H, W)
        y: target tensor, (N, C, H, W)
        window_size: size of the window
        value_range: value range of the inputs (usually 1. or 255)
        weights: weights of the scales, (M,)
    """

    n_channels = x.size(1)
    window = create_window(window_size, n_channels).to(x.device)
    weights = weights.to(x.device)

    return msssim_per_channel(x, y, window, value_range, weights).mean(-1)


###########
# Classes #
###########

class SSIM(nn.Module):
    r"""Creates a criterion that measures the SSIM between an input and a target.

    Args:
        window_size: size of the window
        n_channels: number of channels
        value_range: value range of the inputs (usually 1. or 255)
        reduction: reduction type (`'mean'`, `'sum'` or `'none'`)

    Call:
        The input and target tensors should be of shape (N, C, H, W).
    """

    def __init__(self, window_size: int = 11, n_channels: int = 3, value_range: float = 1., reduction: str = 'mean'):
        super().__init__()

        self.register_buffer(
            'window',
            create_window(window_size, n_channels)
        )

        self.value_range = value_range
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        l = ssim_per_channel(
            input, target,
            window=self.window,
            value_range=self.value_range
        )[0].mean(-1)

        if self.reduction == 'mean':
            l = l.mean()
        elif self.reduction == 'sum':
            l = l.sum()

        return l


class MSSSIM(SSIM):
    r"""Creates a criterion that measures the MS-SSIM between an input and a target.

    Args:
        window_size: size of the window
        n_channels: number of channels
        value_range: value range of the inputs (usually 1. or 255)
        weights: weights of the scales, (M,)
        reduction: reduction type (`'mean'`, `'sum'` or `'none'`)

    Call:
        The input and target tensors should be of shape (N, C, H, W).
    """

    def __init__(self, window_size: int = 11, n_channels: int = 3, value_range: float = 1., weights: torch.Tensor = _WEIGHTS, reduction: str = 'mean'):
        super().__init__(window_size, n_channels, value_range, reduction)

        self.register_buffer('weights', weights)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        l = msssim_per_channel(
            input, target,
            window=self.window,
            value_range=self.value_range,
            weights=self.weights
        ).mean(-1)

        if self.reduction == 'mean':
            l = l.mean()
        elif self.reduction == 'sum':
            l = l.sum()

        return l
