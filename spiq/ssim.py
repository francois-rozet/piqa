r"""Structural Similarity (SSIM)

This module implements the SSIM in PyTorch.

Wikipedia:
    https://en.wikipedia.org/wiki/Structural_similarity

Credits:
    Inspired by pytorch-msssim
    https://github.com/VainF/pytorch-msssim

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


#############
# Constants #
#############

_SIGMA = 1.5
_K1, _K2 = 0.01, 0.03


#############
# Functions #
#############

def gaussian_kernel(kernel_size: int, sigma: float=1., n: int=2) -> torch.Tensor:
    r"""Returns the `n`-dimensional Gaussian kernel of size `kernel_size`.

    The distribution is centered around the kernel's center and the standard deviation is `sigma`.

    Args:
        kernel_size: size of the kernel
        sigma: standard deviation of the distribution
        n: number of dimensions of the kernel

    Wikipedia:
        https://en.wikipedia.org/wiki/Normal_distribution
    """

    distrib = torch.arange(kernel_size).float()
    distrib -= (kernel_size - 1) / 2
    distrib = distrib ** 2

    kernel = distrib.clone()

    for i in range(1, n):
        distrib = distrib.unsqueeze(0)
        kernel = kernel.unsqueeze(i)
        kernel = kernel + distrib

    kernel = torch.exp(-kernel / (2 * sigma ** 2))
    kernel /= kernel.sum()

    return kernel


def create_window(window_size: int, n_channels: int) -> torch.Tensor:
    r"""Returns the SSIM convolution window of size `window_size`.

    Args:
        window_size: size of the window
        n_channels: number of channels
    """

    kernel = gaussian_kernel(window_size, _SIGMA)

    window = kernel.unsqueeze(0).unsqueeze(0)
    window = window.expand(n_channels, 1, window_size, window_size)

    return window


def ssim_per_channel(x: torch.Tensor, y: torch.Tensor, window: torch.Tensor, value_range: float=1.) -> torch.Tensor:
    r"""Returns the SSIM and the contrast sensitivity (CS) per channel between `x` and `y`.

    Args:
        x: input tensor, (N, C, H, W)
        y: target tensor, (N, C, H, W)
        window: convolution window
        value_range: value range of the inputs (usually 1. or 255)
    """

    n_channels, _, window_size, _ = window.size()
    padding = window_size // 2

    mu_x = F.conv2d(x, window, padding=padding, groups=n_channels)
    mu_y = F.conv2d(y, window, padding=padding, groups=n_channels)

    mu_x_sq = mu_x ** 2
    mu_y_sq = mu_y ** 2
    mu_xy = mu_x * mu_y

    sigma_x_sq = F.conv2d(x ** 2, window, padding=padding, groups=n_channels) - mu_x_sq
    sigma_y_sq = F.conv2d(y ** 2, window, padding=padding, groups=n_channels) - mu_y_sq
    sigma_xy = F.conv2d(x * y, window, padding=padding, groups=n_channels) - mu_xy

    c1 = (_K1 * value_range) ** 2
    c2 = (_K2 * value_range) ** 2

    cs_map = (2. * sigma_xy + c2) / (sigma_x_sq + sigma_y_sq + c2)
    ssim_map = (2. * mu_x * mu_y + c1) / (mu_x_sq + mu_y_sq + c1) * cs_map

    return ssim_map.mean((-1, -2)), cs_map.mean((-1, -2))


def ssim(x: torch.Tensor, y: torch.Tensor, window_size: int=11, value_range: float=1.) -> torch.Tensor:
    r"""Returns the SSIM between `x` and `y`.

    Args:
        x: input tensor of shape, (N, C, H, W)
        y: target tensor of shape, (N, C, H, W)
        window_size: size of the window
        value_range: value range of the inputs (usually 1. or 255)
    """

    n_channels = x.size(1)
    window = create_window(window_size, n_channels).to(x.device)

    return ssim_per_channel(x, y, window, value_range)[0].mean(-1)


###########
# Classes #
###########

class SSIM(nn.Module):
    r"""Creates a criterion that measures the SSIM between an input and a target.
    """

    def __init__(self, window_size: int=11, n_channels: int=3, value_range: float=1., reduction='mean'):
        super().__init__()

        self.register_buffer(
            'window',
            create_window(window_size, n_channels)
        )

        self.value_range = value_range
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            input: input tensor, (N, C, H, W)
            target: target tensor, (N, C, H, W)
        """

        l = ssim_per_channel(
            input, target,
            window=self.window,
            value_range=self.value_range
        )[0].mean(-1)

        if self.reduction == 'mean':
            return l.mean()
        elif self.reduction == 'sum':
            return l.sum()

        return l
