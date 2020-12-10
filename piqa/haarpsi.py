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

from piqa.utils import build_reduce, haar_kernel, filter2d

_YIQ_WEIGHTS = torch.FloatTensor([
    [0.299, 0.587, 0.114],
    [0.596, -0.274, -0.322],
    [0.211, -0.523, 0.312],
])


def _haarpsi(
    x: torch.Tensor,
    y: torch.Tensor,
    value_range: float = 1.,
    n_kernels: int = 3,
    c: float = 0.00046,  # 30. / (255. ** 2)
    alpha: float = 4.2,
) -> torch.Tensor:
    r"""Returns the HaarPSI between `x` and `y`,
    without color space conversion.

    `_haarpsi` is an auxiliary function for `haarpsi` and `HaarPSI`.

    Args:
        x: An input tensor, (N, 3 or 1, H, W).
        y: A target tensor, (N, 3 or 1, H, W).
        value_range: The number of value range of the inputs (usually 1. or 255).
        n_kernels: The number of Haar wavelet kernels to use.

        For the remaining arguments, refer to [1].

    Example:
        >>> x = torch.rand(5, 3, 256, 256)
        >>> y = torch.rand(5, 3, 256, 256)
        >>> l = _haarpsi(x, y)
        >>> l.size()
        torch.Size([5])
    """

    c *= value_range ** 2

    # Y

    ## Gradient(s)
    g_xy = []

    for j in range(1, n_kernels + 1):
        kernel_size = 2 ** j
        half_size = kernel_size // 2

        ### Haar wavelet kernel
        kernel = haar_kernel(kernel_size)
        kernel = torch.stack([kernel, kernel.t()]).unsqueeze(1)
        kernel = kernel.to(x.device)

        ### Haar filter (gradient)
        pad = (half_size - 1, half_size, half_size - 1, half_size)
        g_x = filter2d(F.pad(x[:, :1], pad=pad), kernel).abs()
        g_y = filter2d(F.pad(y[:, :1], pad=pad), kernel).abs()

        g_xy.append((g_x, g_y))

    ## Gradient similarity(ies)
    gs = []
    for g_x, g_y in g_xy[:-1]:
        gs.append((2. * g_x * g_y + c) / (g_x ** 2 + g_y ** 2 + c))

    ## Local similarity(ies)
    ls = sum(gs) / 2.  # (N, 2, H, W)

    ## Weight(s)
    w = torch.stack(g_xy[-1], dim=-1).max(dim=-1)[0]  # (N, 2, H, W)

    # IQ
    if x.size(1) == 3:
        ## Mean filter
        pad = (0, 1, 0, 1)
        m_x = F.avg_pool2d(F.pad(x[:, 1:], pad=pad), 2, stride=1).abs()
        m_y = F.avg_pool2d(F.pad(y[:, 1:], pad=pad), 2, stride=1).abs()

        ## Chromatic similarity(ies)
        cs = (2. * m_x * m_y + c) / (m_x ** 2 + m_y ** 2 + c)

        ## Local similarity(ies)
        ls = torch.cat([ls, cs.mean(1, True)], dim=1)  # (N, 3, H, W)

        ## Weight(s)
        w = torch.cat([w, w.mean(1, True)], dim=1)  # (N, 3, H, W)

    # HaarPSI
    hs = torch.sigmoid(ls * alpha)
    hpsi = (hs * w).sum((-1, -2, -3)) / w.sum((-1, -2, -3))
    hpsi = (torch.logit(hpsi) / alpha) ** 2

    return hpsi


def haarpsi(
    x: torch.Tensor,
    y: torch.Tensor,
    chromatic: bool = True,
    **kwargs,
):
    r"""Returns the HaarPSI between `x` and `y`.

    Args:
        x: An input tensor, (N, 3, H, W).
        y: A target tensor, (N, 3, H, W).
        chromatic: Whether to use the chromatic channels of not.

        `**kwargs` are transmitted to `_haarpsi`.

    Example:
        >>> x = torch.rand(5, 3, 256, 256)
        >>> y = torch.rand(5, 3, 256, 256)
        >>> l = haarpsi(x, y)
        >>> l.size()
        torch.Size([5])
    """

    # Downsample
    x = F.avg_pool2d(x, kernel_size=2, ceil_mode=True)
    y = F.avg_pool2d(y, kernel_size=2, ceil_mode=True)

    # RBG to YIQ
    if chromatic:
        yiq_weights = _YIQ_WEIGHTS.view(3, 3, 1, 1)
    else:
        yiq_weights = _YIQ_WEIGHTS[:1].view(1, 3, 1, 1)

    yiq_weights = yiq_weights.to(x.device)

    x = F.conv2d(x, yiq_weights)
    y = F.conv2d(y, yiq_weights)

    return _haarpsi(x, y, **kwargs)


class HaarPSI(nn.Module):
    r"""Creates a criterion that measures the HaarPSI
    between an input and a target.

    Args:
        reduction: Specifies the reduction to apply to the output:
            `'none'` | `'mean'` | `'sum'`.

        `**kwargs` are transmitted to `_haarpsi`.

    Shape:
        * Input: (N, 3, H, W)
        * Target: (N, 3, H, W)
        * Output: (N,) or (1,) depending on `reduction`

    Example:
        >>> criterion = HaarPSI().cuda()
        >>> x = torch.rand(5, 3, 256, 256).cuda()
        >>> y = torch.rand(5, 3, 256, 256).cuda()
        >>> l = criterion(x, y)
        >>> l.size()
        torch.Size([])
    """

    def __init__(
        self,
        chromatic: bool = True,
        reduction: str = 'mean',
        **kwargs,
    ):
        r""""""
        super().__init__()

        if chromatic:
            yiq_weights = _YIQ_WEIGHTS.view(3, 3, 1, 1)
        else:
            yiq_weights = _YIQ_WEIGHTS[:1].view(1, 3, 1, 1)

        self.register_buffer('yiq_weights', yiq_weights)

        self.reduce = build_reduce(reduction)
        self.kwargs = kwargs

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        r"""Defines the computation performed at every call.
        """

        # Downsample
        input = F.avg_pool2d(input, 2, ceil_mode=True)
        target = F.avg_pool2d(target, 2, ceil_mode=True)

        # RGB to YIQ
        input = F.conv2d(input, self.yiq_weights)
        target = F.conv2d(target, self.yiq_weights)

        # HaarPSI
        l = _haarpsi(input, target, **self.kwargs)

        return self.reduce(l)
