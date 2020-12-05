r"""Gradient Magnitude Similarity Deviation (GMSD)

This module implements the GMSD in PyTorch.

References:
    [1] Gradient Magnitude Similarity Deviation:
    An Highly Efficient Perceptual Image Quality Index
    (Xue et al., 2013)
    https://arxiv.org/abs/1308.3052
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from spiq.utils import build_reduce, prewitt_kernel, gradient2d, tensor_norm

_L_WEIGHTS = torch.FloatTensor([0.2989, 0.587, 0.114])


def gmsd(
    x: torch.Tensor,
    y: torch.Tensor,
    value_range: float = 1.,
    c: float = 0.00261,  # 170. / (255. ** 2)
) -> torch.Tensor:
    r"""Returns the GMSD between `x` and `y`.

    Args:
        x: An input tensor, (N, 3, H, W).
        y: A target tensor, (N, 3, H, W).
        value_range: The value range of the inputs (usually 1. or 255).

        For the remaining arguments, refer to [1].
    """

    _, _, h, w = x.size()

    # Downsample
    padding = (0, w % 2, 0, h % 2)

    if sum(padding) > 0:
        x = F.pad(x, pad=padding)
        y = F.pad(y, pad=padding)

    x = F.avg_pool2d(x, kernel_size=2)
    y = F.avg_pool2d(y, kernel_size=2)

    # RGB to luminance
    l_weights = _L_WEIGHTS.to(x.device).view(1, 3, 1, 1)
    l_weights /= value_range

    x = F.conv2d(x, l_weights)
    y = F.conv2d(y, l_weights)

    # Gradient magnitude
    kernel = prewitt_kernel()
    kernel = torch.stack([kernel, kernel.t()]).unsqueeze(1).to(x.device)

    gm_x = tensor_norm(gradient2d(x, kernel), dim=1)
    gm_y = tensor_norm(gradient2d(y, kernel), dim=1)

    # Gradient magnitude similarity
    gms = (2. * gm_x * gm_y + c) / (gm_x ** 2 + gm_y ** 2 + c)

    # Gradient magnitude similarity diviation
    gmsd = (gms - gms.mean((-1, -2), keepdim=True)) ** 2
    gmsd = torch.sqrt(gmsd.mean((-1, -2)))

    return gmsd


class GMSD(nn.Module):
    r"""Creates a criterion that measures the GMSD
    between an input and a target.

    Args:
        reduction: Specifies the reduction to apply to the output:
            `'none'` | `'mean'` | `'sum'`.

        `**kwargs` are transmitted to `gmsd`.

    Shape:
        * Input: (N, 3, H, W)
        * Target: (N, 3, H, W)
        * Output: (N,) or (1,) depending on `reduction`
    """

    def __init__(self, reduction: str = 'mean', **kwargs):
        super().__init__()

        self.reduce = build_reduce(reduction)
        self.kwargs = kwargs

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        l = gmsd(input, target, **self.kwargs)

        return self.reduce(l)
