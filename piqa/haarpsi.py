r"""Haar Perceptual Similarity Index (HaarPSI)

This module implements the HaarPSI in PyTorch.

Original:
    https://github.com/rgcda/haarpsi

Wikipedia:
    https://wikipedia.org/wiki/Haar_wavelet

References:
    | A Haar Wavelet-Based Perceptual Similarity Index for Image Quality Assessment (Reisenhofer et al., 2018)
    | https://arxiv.org/abs/1607.06140
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

from .utils import assert_type
from .utils.color import ColorConv
from .utils.functional import (
    haar_kernel,
    gradient_kernel,
    channel_conv,
    reduce_tensor,
)


@torch.jit.script_if_tracing
def haarpsi(
    x: Tensor,
    y: Tensor,
    n_kernels: int = 3,
    value_range: float = 1.0,
    c: float = 30 / 255 ** 2,
    alpha: float = 4.2,
) -> Tensor:
    r"""Returns the HaarPSI between :math:`x` and :math:`y`, without color space
    conversion.

    Args:
        x: An input tensor, :math:`(N, 3 \text{ or } 1, H, W)`.
        y: A target tensor, :math:`(N, 3 \text{ or } 1, H, W)`.
        n_kernels: The number of Haar wavelet kernels to use.
        value_range: The value range :math:`L` of the inputs (usually 1 or 255).

    Note:
        For the remaining arguments, refer to Reisenhofer et al. (2018).

    Returns:
        The HaarPSI vector, :math:`(N,)`.

    Example:
        >>> x = torch.rand(5, 3, 256, 256)
        >>> y = torch.rand(5, 3, 256, 256)
        >>> l = haarpsi(x, y)
        >>> l.shape
        torch.Size([5])
    """

    c *= value_range ** 2

    # Y
    y_x, y_y = x[:, :1], y[:, :1]

    ## Gradient similarity(ies)
    gs = []

    for j in range(1, n_kernels + 1):
        kernel_size = int(2 ** j)

        ### Haar wavelet kernel
        kernel = gradient_kernel(haar_kernel(kernel_size)).to(x.device)

        ### Haar filter (gradient)
        pad = kernel_size // 2

        g_x = channel_conv(y_x, kernel, padding=pad)[..., 1:, 1:].abs()
        g_y = channel_conv(y_y, kernel, padding=pad)[..., 1:, 1:].abs()

        if j < n_kernels:
            gs.append((2 * g_x * g_y + c) / (g_x ** 2 + g_y ** 2 + c))
        else:
            gs.append(g_x)
            gs.append(g_y)

    ## Local similarity(ies)
    ls = torch.stack(gs[:-2], dim=-1).mean(dim=-1)  # (N, 2, H, W)

    ## Weight(s)
    w = torch.maximum(gs[-2], gs[-1])  # (N, 2, H, W)

    # IQ
    if x.shape[1] == 3:
        iq_x, iq_y = x[:, 1:], y[:, 1:]

        ## Mean filter
        m_x = F.avg_pool2d(iq_x, 2, stride=1, padding=1)[..., 1:, 1:].abs()
        m_y = F.avg_pool2d(iq_y, 2, stride=1, padding=1)[..., 1:, 1:].abs()

        ## Chromatic similarity(ies)
        cs = (2 * m_x * m_y + c) / (m_x ** 2 + m_y ** 2 + c)

        ## Local similarity(ies)
        ls = torch.cat((ls, cs.mean(dim=1, keepdim=True)), dim=1)  # (N, 3, H, W)

        ## Weight(s)
        w = torch.cat((w, w.mean(dim=1, keepdim=True)), dim=1)  # (N, 3, H, W)

    # HaarPSI
    hs = torch.sigmoid(ls * alpha)
    hpsi = (hs * w).sum(dim=(-1, -2, -3)) / w.sum(dim=(-1, -2, -3))
    hpsi = (torch.logit(hpsi) / alpha) ** 2

    return hpsi


class HaarPSI(nn.Module):
    r"""Measures the HaarPSI between an input and a target.

    Before applying :func:`haarpsi`, the input and target are converted from
    RBG to Y(IQ) and downsampled by a factor 2.

    Args:
        chromatic: Whether to use the chromatic channels (IQ) or not.
        downsample: Whether downsampling is enabled or not.
        reduction: Specifies the reduction to apply to the output:
            `'none'`, `'mean'` or `'sum'`.
        kwargs: Keyword arguments passed to :func:`haarpsi`.

    Example:
        >>> criterion = HaarPSI()
        >>> x = torch.rand(5, 3, 256, 256, requires_grad=True)
        >>> y = torch.rand(5, 3, 256, 256)
        >>> l = 1 - criterion(x, y)
        >>> l.shape
        torch.Size([])
        >>> l.backward()
    """

    def __init__(
        self,
        chromatic: bool = True,
        downsample: bool = True,
        reduction: str = 'mean',
        **kwargs,
    ):
        super().__init__()

        self.convert = ColorConv('RGB', 'YIQ' if chromatic else 'Y')
        self.downsample = downsample
        self.reduction = reduction
        self.value_range = kwargs.get('value_range', 1.0)
        self.kwargs = kwargs

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        r"""
        Args:
            x: An input tensor, :math:`(N, 3, H, W)`.
            y: A target tensor, :math:`(N, 3, H, W)`.

        Returns:
            The HaarPSI vector, :math:`(N,)` or :math:`()` depending on `reduction`.
        """

        assert_type(
            x, y,
            device=self.convert.weight.device,
            dim_range=(4, 4),
            n_channels=3,
            value_range=(0.0, self.value_range),
        )

        # Downsample
        if self.downsample:
            x = F.avg_pool2d(x, 2, ceil_mode=True)
            y = F.avg_pool2d(y, 2, ceil_mode=True)

        # RGB to Y(IQ)
        x = self.convert(x)
        y = self.convert(y)

        # HaarPSI
        l = haarpsi(x, y, **self.kwargs)

        return reduce_tensor(l, self.reduction)
