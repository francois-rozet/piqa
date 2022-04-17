r"""Color space conversion tools"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Optional, Tuple


def spatial(x: Tensor) -> int:
    r"""Returns the number of spatial dimensions of :math:`x`.

    Args:
        x: A tensor, :math:`(N, C, *)`.
    """

    return len(x.shape) - 2


def color_conv(
    x: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
) -> Tensor:
    r"""Returns the color convolution of :math:`x` with the kernel `weight`.

    Args:
        x: A tensor, :math:`(N, C, *)`.
        weight: A weight kernel, :math:`(C', C)`.
        bias: A bias vector, :math:`(C',)`.
    """

    return F.linear(x.transpose(1, -1), weight, bias).transpose(1, -1)


RGB_TO_YIQ = torch.tensor([
    [0.299, 0.587, 0.114],
    [0.5969, -0.2746, -0.3213],
    [0.2115, -0.5227, 0.3112],
])

RGB_TO_LHM = torch.tensor([
    [0.2989, 0.5870, 0.1140],
    [0.3, 0.04, -0.35],
    [0.34, -0.6, 0.17],
])

RGB_TO_LMN = torch.tensor([
    [0.06, 0.63, 0.27],
    [0.30, 0.04, -0.35],
    [0.34, -0.6, 0.17],
])


_WEIGHTS = {
    ('RGB', 'YIQ'): RGB_TO_YIQ,  # HaarPSI
    ('RGB', 'Y'): RGB_TO_YIQ[:1],  # GMSD
    ('RGB', 'LHM'): RGB_TO_LHM,  # MDSI
    ('RGB', 'LMN'): RGB_TO_LMN,  # VSI
    ('RGB', 'L'): RGB_TO_LMN[:1],  # VSI
}


class ColorConv(nn.Module):
    r"""Color convolution module.

    Args:
        src: The source color space (e.g. `'RGB'`).
        dst: The destination color space (e.g. `'YIQ'`).

    Example:
        >>> x = torch.rand(5, 3, 256, 256)
        >>> conv = ColorConv('RGB', 'YIQ')
        >>> y = conv(x)
        >>> y.size()
        torch.Size([5, 3, 256, 256])
    """

    def __init__(self, src: str, dst: str):
        super().__init__()

        assert (src, dst) in _WEIGHTS, f"Unknown {src} to {dst} conversion"

        self.register_buffer('weight', _WEIGHTS[(src, dst)])

    @property
    def device(self) -> torch.device:
        return self.weight.device

    def forward(self, x: Tensor) -> Tensor:
        return color_conv(x, self.weight)


def rgb_to_xyz(x: Tensor, value_range: float = 1.) -> Tensor:
    r"""Converts from sRGB to (CIE) XYZ.

    Args:
        value_range: The value range :math:`L` of the inputs (usually `1.` or `255`).

    Wikipedia:
        https://en.wikipedia.org/wiki/SRGB
    """

    x = x / value_range
    x = torch.where(
        x <= 0.04045,
        x / 12.92,
        ((x + 0.055) / 1.055) ** 2.4,
    )

    weight = torch.tensor([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ])

    return color_conv(x, weight.to(x))


def xyz_to_lab(x: Tensor) -> Tensor:
    r"""Converts from (CIE) XYZ to (CIE) LAB.

    Wikipedia:
        https://en.wikipedia.org/wiki/CIELAB_color_space
    """

    illuminants = torch.tensor([0.964212, 1., 0.825188])  # D50
    delta = 6 / 29

    x = color_conv(x, torch.diag(illuminants).to(x))
    x = torch.where(
        x > delta ** 3,
        x ** (1 / 3),
        x / (3 * delta ** 2) + 4 / 29,
    )

    weight = torch.tensor([
        [0., 116., 0.],
        [500., -500., 0.],
        [0., 200., -200.],
    ])

    bias = torch.tensor([-16., 0., 0.])

    return color_conv(x, weight.to(x), bias.to(x))
