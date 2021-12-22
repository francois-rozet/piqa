r"""Color space conversion tools
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple


def spatial(x: torch.Tensor) -> int:
    r"""Returns the number of spatial dimensions of \(x\)."""

    return len(x.shape) - 2


def color_conv(
    x: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    r"""Returns the color convolution of \(x\) with the kernel `weight`.

    Args:
        x: A tensor, \((N, C, *)\).
        weight: A weight kernel, \((C', C)\).
    """

    return F.conv1d(x, weight.view(weight.shape + (1,) * spatial(x)))


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
        src: The source color space. E.g. `RGB`.
        dst: The destination color space. E.g. `YIQ`.

    Example:
        >>> x = torch.rand(5, 3, 256, 256)
        >>> conv = ColorConv('RGB', 'YIQ')
        >>> y = conv(x)
        >>> y.size()
        torch.Size([5, 3, 256, 256])
    """

    def __init__(self, src: str, dst: str):
        super().__init__()

        assert (src, dst) in _WEIGHTS, f'Unknown {src} to {dst} conversion'

        self.register_buffer('weight', _WEIGHTS[(src, dst)])

    @property
    def device(self) -> torch.device:
        return self.weight.device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Defines the computation performed at every call.
        """

        return color_conv(x, self.weight)


def rgb_to_xyz(x: torch.Tensor, value_range: float = 1.) -> torch.Tensor:
    r"""Converts from sRGB to (CIE) XYZ.

    References:
        https://en.wikipedia.org/wiki/SRGB
    """

    x = x / value_range

    mask = x <= 0.04045
    left = x / 12.92
    right = ((x + 0.055) / 1.055) ** 2.4

    x = torch.where(mask, left, right)

    weight = torch.tensor([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ])

    return color_conv(x, weight.to(x))


def xyz_to_lab(
    x: torch.Tensor,
    illuminants: Tuple[float, float, float] = (0.956797052643698, 1., 0.9214805860173273),
) -> torch.Tensor:
    r"""Converts from (CIE) XYZ to (CIE) LAB.

    References:
        https://en.wikipedia.org/wiki/CIELAB_color_space
    """

    scale = torch.tensor(illuminants).view((3,) + (1,) * spatial(x))
    x = x / scale.to(x)

    delta = 6 / 29

    mask = x > delta ** 3
    left = x ** (1 / 3)
    right = x / (3 * delta ** 2) + 4 / 29

    x = torch.where(mask, left, right)

    weight = torch.tensor([
        [0., 116., 0.],
        [500., -500., 0.],
        [0., 200., -200.],
    ])

    bias = torch.tensor([-16., 0., 0.]).view(scale.shape)

    return color_conv(x, weight.to(x)) + bias.to(x)
