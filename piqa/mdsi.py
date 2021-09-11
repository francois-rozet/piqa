r"""Mean Deviation Similarity Index (MDSI)

This module implements the MDSI in PyTorch.

Credits:
    Inspired by the [official implementation](https://www.mathworks.com/matlabcentral/fileexchange/59809-mdsi-ref-dist-combmethod)

References:
    [1] Mean Deviation Similarity Index:
    Efficient and Reliable Full-Reference Image Quality Evaluator
    (Nafchi et al., 2016)
    https://arxiv.org/abs/1608.07433
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from piqa.utils import _jit, _assert_type, _reduce
from piqa.utils.color import ColorConv
from piqa.utils.functional import (
    prewitt_kernel,
    gradient_kernel,
    channel_conv,
)

import piqa.utils.complex as cx


@_jit
def mdsi(
    x: torch.Tensor,
    y: torch.Tensor,
    kernel: torch.Tensor,
    combination: str = 'sum',
    value_range: float = 1.,
    c1: float = 140. / (255. ** 2),
    c2: float = 55. / (255. ** 2),
    c3: float = 550. / (255. ** 2),
    alpha: float = 0.6,  # 'sum'
    beta: float = 0.1,  # 'prod'
    gamma: float = 0.2,  # 'prod'
    rho: float = 1.,
    q: float = 0.25,
    o: float = 0.25,
) -> torch.Tensor:
    r"""Returns the MDSI between \(x\) and \(y\),
    without downsampling and color space conversion.

    Args:
        x: An input tensor, \((N, 3, H, W)\).
        y: A target tensor, \((N, 3, H, W)\).
        kernel: A gradient kernel, \((2, 1, K, K)\).
        combination: Specifies the scheme to combine the gradient
            and chromaticity similarities (GS, CS): `'sum'` | `'prod'`.
        value_range: The value range \(L\) of the inputs (usually 1. or 255).

        For the remaining arguments, refer to [1].

    Returns:
        The MDSI vector, \((N,)\).

    Example:
        >>> x = torch.rand(5, 1, 256, 256)
        >>> y = torch.rand(5, 1, 256, 256)
        >>> kernel = gradient_kernel(prewitt_kernel())
        >>> l = mdsi(x, y, kernel)
        >>> l.size()
        torch.Size([5])
    """

    c1 *= value_range ** 2
    c2 *= value_range ** 2
    c3 *= value_range ** 2

    l_x, hm_x = x[:, :1], x[:, 1:]
    l_y, hm_y = y[:, :1], y[:, 1:]

    # Gradient magnitude
    pad = kernel.size(-1) // 2

    gm_x = torch.linalg.norm(channel_conv(l_x, kernel, padding=pad), dim=1)
    gm_y = torch.linalg.norm(channel_conv(l_y, kernel, padding=pad), dim=1)
    gm_avg = torch.linalg.norm(
        channel_conv((l_x + l_y) / 2., kernel, padding=pad),
        dim=1,
    )

    gm_x_sq, gm_y_sq, gm_avg_sq = gm_x ** 2, gm_y ** 2, gm_avg ** 2

    # Gradient similarity
    gs_x_y = (2. * gm_x * gm_y + c1) / (gm_x_sq + gm_y_sq + c1)
    gs_x_avg = (2. * gm_x * gm_avg + c2) / (gm_x_sq + gm_avg_sq + c2)
    gs_y_avg = (2. * gm_y * gm_avg + c2) / (gm_y_sq + gm_avg_sq + c2)

    gs = gs_x_y + gs_x_avg - gs_y_avg

    # Chromaticity similarity
    cs_num = 2. * (hm_x * hm_y).sum(dim=1) + c3
    cs_den = (hm_x ** 2 + hm_y ** 2).sum(dim=1) + c3
    cs = cs_num / cs_den

    # Gradient-chromaticity similarity
    gs = cx.complx(gs, torch.zeros_like(gs))
    cs = cx.complx(cs, torch.zeros_like(cs))

    if combination == 'prod':
        gcs = cx.prod(cx.pow(gs, gamma), cx.pow(cs, beta))
    else:  # combination == 'sum'
        gcs = alpha * gs + (1. - alpha) * cs

    # Mean deviation similarity
    gcs_q = cx.pow(gcs, q)
    gcs_q_avg = gcs_q.mean(dim=(-2, -3), keepdim=True)
    score = cx.mod(gcs_q - gcs_q_avg, squared=True) ** (rho / 2)
    mds = score.mean(dim=(-1, -2)) ** (o / rho)

    return mds


class MDSI(nn.Module):
    r"""Creates a criterion that measures the MDSI
    between an input and a target.

    Before applying `mdsi`, the input and target are converted from
    RBG to LHM and downsampled by a factor \( \frac{\min(H, W)}{256} \).

    Args:
        downsample: Whether downsampling is enabled or not.
        kernel: A gradient kernel, \((2, 1, K, K)\).
            If `None`, use the Prewitt kernel instead.
        reduction: Specifies the reduction to apply to the output:
            `'none'` | `'mean'` | `'sum'`.

        `**kwargs` are transmitted to `mdsi`.

    Shapes:
        * Input: \((N, 3, H, W)\)
        * Target: \((N, 3, H, W)\)
        * Output: \((N,)\) or \(()\) depending on `reduction`

    Example:
        >>> criterion = MDSI().cuda()
        >>> x = torch.rand(5, 3, 256, 256, requires_grad=True).cuda()
        >>> y = torch.rand(5, 3, 256, 256).cuda()
        >>> l = criterion(x, y)
        >>> l.size()
        torch.Size([])
        >>> l.backward()
    """

    def __init__(
        self,
        downsample: bool = True,
        kernel: torch.Tensor = None,
        reduction: str = 'mean',
        **kwargs,
    ):
        r""""""
        super().__init__()

        if kernel is None:
            kernel = gradient_kernel(prewitt_kernel())

        self.register_buffer('kernel', kernel)

        self.convert = ColorConv('RGB', 'LHM')
        self.downsample = downsample
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
            device=self.kernel.device,
            dim_range=(4, 4),
            n_channels=3,
            value_range=(0., self.value_range),
        )

        # Downsample
        if self.downsample:
            _, _, h, w = input.size()
            M = round(min(h, w) / 256)

            if M > 1:
                input = F.avg_pool2d(input, kernel_size=M, ceil_mode=True)
                target = F.avg_pool2d(target, kernel_size=M, ceil_mode=True)

        # RGB to LHM
        input = self.convert(input)
        target = self.convert(target)

        # MDSI
        l = mdsi(input, target, kernel=self.kernel, **self.kwargs)

        return _reduce(l, self.reduction)
