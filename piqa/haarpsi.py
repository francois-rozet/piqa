r"""Haar Perceptual Similarity Index (HaarPSI)

This module implements the HaarPSI in PyTorch.

Wikipedia:
    https://en.wikipedia.org/wiki/Haar_wavelet

Credits:
    Inspired by [haarpsi](https://github.com/rgcda/haarpsi)

References:
    [1] A Haar Wavelet-Based Perceptual Similarity Index for
    Image Quality Assessment
    (Reisenhofer et al., 2018)
    https://arxiv.org/abs/1607.06140
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from piqa.utils import _jit, _assert_type, _reduce
from piqa.utils.color import get_conv
from piqa.utils.functional import (
    haar_kernel,
    gradient_kernel,
    channel_conv,
)


@_jit
def haarpsi(
    x: torch.Tensor,
    y: torch.Tensor,
    n_kernels: int = 3,
    value_range: float = 1.,
    c: float = 30. / (255. ** 2),
    alpha: float = 4.2,
) -> torch.Tensor:
    r"""Returns the HaarPSI between \(x\) and \(y\),
    without color space conversion.

    Args:
        x: An input tensor, \((N, 3 \text{ or } 1, H, W)\).
        y: A target tensor, \((N, 3 \text{ or } 1, H, W)\).
        n_kernels: The number of Haar wavelet kernels to use.
        value_range: The value range \(L\) of the inputs (usually 1. or 255).

        For the remaining arguments, refer to [1].

    Returns:
        The HaarPSI vector, \((N,)\).

    Example:
        >>> x = torch.rand(5, 3, 256, 256)
        >>> y = torch.rand(5, 3, 256, 256)
        >>> l = haarpsi(x, y)
        >>> l.size()
        torch.Size([5])
    """

    c *= value_range ** 2

    # Y
    y_x, y_y = x[:, :1], y[:, :1]

    ## Gradient(s)
    g_xy: List[Tuple[Tensor, Tensor]] = []

    for j in range(1, n_kernels + 1):
        kernel_size = int(2 ** j)

        ### Haar wavelet kernel
        kernel = gradient_kernel(haar_kernel(kernel_size)).to(x.device)

        ### Haar filter (gradient)
        pad = kernel_size // 2

        g_x = channel_conv(y_x, kernel, padding=pad)[..., 1:, 1:].abs()
        g_y = channel_conv(y_y, kernel, padding=pad)[..., 1:, 1:].abs()

        g_xy.append((g_x, g_y))

    ## Gradient similarity(ies)
    gs = []
    for g_x, g_y in g_xy[:-1]:
        gs.append((2. * g_x * g_y + c) / (g_x ** 2 + g_y ** 2 + c))

    ## Local similarity(ies)
    ls = torch.stack(gs, dim=-1).sum(dim=-1) / 2.  # (N, 2, H, W)

    ## Weight(s)
    w = torch.stack(g_xy[-1], dim=-1).max(dim=-1)[0]  # (N, 2, H, W)

    # IQ
    if x.size(1) == 3:
        iq_x, iq_y = x[:, 1:], y[:, 1:]

        ## Mean filter
        m_x = F.avg_pool2d(iq_x, 2, stride=1, padding=1)[..., 1:, 1:].abs()
        m_y = F.avg_pool2d(iq_y, 2, stride=1, padding=1)[..., 1:, 1:].abs()

        ## Chromatic similarity(ies)
        cs = (2. * m_x * m_y + c) / (m_x ** 2 + m_y ** 2 + c)

        ## Local similarity(ies)
        ls = torch.cat([ls, cs.mean(1, True)], dim=1)  # (N, 3, H, W)

        ## Weight(s)
        w = torch.cat([w, w.mean(1, True)], dim=1)  # (N, 3, H, W)

    # HaarPSI
    hs = torch.sigmoid(ls * alpha)
    hpsi = (hs * w).sum(dim=(-1, -2, -3)) / w.sum(dim=(-1, -2, -3))
    hpsi = (torch.logit(hpsi) / alpha) ** 2

    return hpsi


class HaarPSI(nn.Module):
    r"""Creates a criterion that measures the HaarPSI
    between an input and a target.

    Before applying `haarpsi`, the input and target are converted from
    RBG to Y(IQ) and downsampled by a factor 2.

    Args:
        chromatic: Whether to use the chromatic channels (IQ) or not.
        reduction: Specifies the reduction to apply to the output:
            `'none'` | `'mean'` | `'sum'`.

        `**kwargs` are transmitted to `haarpsi`.

    Shapes:
        * Input: \((N, 3, H, W)\)
        * Target: \((N, 3, H, W)\)
        * Output: \((N,)\) or \(()\) depending on `reduction`

    Example:
        >>> criterion = HaarPSI().cuda()
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
        reduction: str = 'mean',
        **kwargs,
    ):
        r""""""
        super().__init__()

        self.convert = get_conv('RGB', 'YIQ' if chromatic else 'Y')
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
            device=self.convert.device,
            dim_range=(4, 4),
            n_channels=3,
            value_range=(0., self.value_range),
        )

        # Downsample
        input = F.avg_pool2d(input, 2, ceil_mode=True)
        target = F.avg_pool2d(target, 2, ceil_mode=True)

        # RGB to Y(IQ)
        input = self.convert(input)
        target = self.convert(target)

        # HaarPSI
        l = haarpsi(input, target, **self.kwargs)

        return _reduce(l, self.reduction)
