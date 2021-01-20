r"""Color space conversion tools
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


_WEIGHTS = {
    ('RGB', 'YIQ'): torch.tensor([  # HaarPSI
        [0.299, 0.587, 0.114],
        [0.5969, -0.2746, -0.3213],
        [0.2115, -0.5227, 0.3112],
    ]),
    ('RGB', 'Y'): torch.tensor([  # GMSD
        [0.299, 0.587, 0.114],
    ]),
    ('RGB', 'LHM'): torch.tensor([  # MDSI
        [0.2989, 0.5870, 0.1140],
        [0.3, 0.04, -0.35],
        [0.34, -0.6, 0.17],
    ]),
}


def get_conv(src: str, dst: str, dim: int = 2) -> nn.Module:
    r"""Returns a color conversion module.

    Args:
        src: The source color space. E.g. `RGB`.
        dst: The destination color space. E.g. `YIQ`.
        dim: The number of space dimensions. E.g. 2 for 2D images.

    Returns:
        The color conversion module.

    Example:
        >>> x = torch.rand(5, 3, 256, 256)
        >>> conv = get_conv('RGB', 'YIQ', dim=2)
        >>> y = conv(x)
        >>> y.size()
        torch.Size([5, 3, 256, 256])
    """

    assert (src, dst) in _WEIGHTS, f'Unknown {src} to {dst} conversion'

    weight = _WEIGHTS[(src, dst)]
    weight = weight.view(weight.size() + (1,) * dim)

    return _ColorConv(weight)


class _ColorConv(nn.Module):
    r"""Color Conversion/Convolution module"""

    def __init__(self, weight: torch.Tensor):
        super().__init__()

        self.register_buffer('weight', weight)

    @property
    def device(self):
        return self.weight.device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv1d(x, self.weight)
