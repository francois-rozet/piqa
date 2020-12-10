r"""Mean Deviation Similarity Index (MDSI)

This module implements the MDSI in PyTorch.

References:
    [1] Mean Deviation Similarity Index:
    Efficient and Reliable Full-Reference Image Quality Evaluator
    (Nafchi et al., 2016)
    https://arxiv.org/abs/1608.07433
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from piqa.utils import build_reduce, prewitt_kernel, filter2d, tensor_norm

_LHM_WEIGHTS = torch.FloatTensor([
    [0.299, 0.587, 0.114],
    [0.3, 0.04, -0.35],
    [0.34, -0.6, 0.17],
])


def _mdsi(
    x: torch.Tensor,
    y: torch.Tensor,
    kernel: torch.Tensor,
    value_range: float = 1.,
    combination: str = 'sum',
    c1: float = 0.00215,  # 140. / (255. ** 2)
    c2: float = 0.00085,  # 55. / (255. ** 2)
    c3: float = 0.00846,  # 550. / (255. ** 2)
    alpha: float = 0.6,  # 'sum'
    beta: float = 0.1,  # 'prod'
    gamma: float = 0.2,  # 'prod'
    rho: float = 1.,
    q: float = 0.25,
    o: float = 0.25,
) -> torch.Tensor:
    r"""Returns the MDSI between `x` and `y`,
    without downsampling and color space conversion.

    `_mdsi` is an auxiliary function for `mdsi` and `MDSI`.

    Args:
        x: An input tensor, (N, 3, H, W).
        y: A target tensor, (N, 3, H, W).
        kernel: A 2D gradient kernel, (2, 1, K, K).
        value_range: The value range of the inputs (usually 1. or 255).
        combination: Specifies the scheme to combine the gradient
            and chromaticity similarities (GS, CS):
            `'sum'` | `'prod'`.

        For the remaining arguments, refer to [1].

    Example:
        >>> x = torch.rand(5, 1, 256, 256)
        >>> y = torch.rand(5, 1, 256, 256)
        >>> kernel = torch.rand(2, 1, 3, 3)
        >>> l = _mdsi(x, y, kernel)
        >>> l.size()
        torch.Size([5])
    """

    c1 *= value_range ** 2
    c2 *= value_range ** 2
    c3 *= value_range ** 2

    # Gradient magnitude
    pad = kernel.size(-1) // 2

    gm_x = tensor_norm(filter2d(x[:, :1], kernel, padding=pad), dim=1)
    gm_y = tensor_norm(filter2d(y[:, :1], kernel, padding=pad), dim=1)
    gm_avg = tensor_norm(
        filter2d((x + y)[:, :1] / 2., kernel, padding=pad),
        dim=1,
    )

    gm_x_sq, gm_y_sq, gm_avg_sq = gm_x ** 2, gm_y ** 2, gm_avg ** 2

    # Gradient similarity
    gs_x_y = (2. * gm_x * gm_y + c1) / (gm_x_sq + gm_y_sq + c1)
    gs_x_avg = (2. * gm_x * gm_avg + c2) / (gm_x_sq + gm_avg_sq + c2)
    gs_y_avg = (2. * gm_y * gm_avg + c2) / (gm_y_sq + gm_avg_sq + c2)

    gs = gs_x_y + gs_x_avg - gs_y_avg

    # Chromaticity similarity
    cs_num = 2. * (x[:, 1:] * y[:, 1:]).sum(1) + c3
    cs_den = (x[:, 1:] ** 2 + y[:, 1:] ** 2).sum(1) + c3
    cs = cs_num / cs_den

    # Gradient-chromaticity similarity
    gs, cs = gs.type(torch.cfloat), cs.type(torch.cfloat)

    if combination == 'prod':
        gcs = (gs ** gamma) * (cs ** beta)
    else:  # combination == 'sum'
        gcs = alpha * gs + (1. - alpha) * cs

    # Mean deviation similarity
    gcs_q = gcs ** q
    gcs_q_avg = torch.view_as_real(gcs_q).mean((-2, -3), True)
    gcs_q_avg = torch.view_as_complex(gcs_q_avg)
    score = (gcs_q - gcs_q_avg).abs()
    mds = (score ** rho).mean((-1, -2)) ** (o / rho)

    return mds


def mdsi(
    x: torch.Tensor,
    y: torch.Tensor,
    kernel: torch.Tensor = None,
    **kwargs,
) -> torch.Tensor:
    r"""Returns the MDSI between `x` and `y`.

    Args:
        x: An input tensor, (N, 3, H, W).
        y: A target tensor, (N, 3, H, W).
        kernel: A 2D gradient kernel, (2, 1, K, K).
            If `None`, use the Prewitt kernel instead.

        `**kwargs` are transmitted to `_mdsi`.

    Example:
        >>> x = torch.rand(5, 3, 256, 256)
        >>> y = torch.rand(5, 3, 256, 256)
        >>> l = mdsi(x, y)
        >>> l.size()
        torch.Size([5])
    """

    # Downsample
    _, _, h, w = x.size()
    M = max(1, min(h, w) // 256)

    x = F.avg_pool2d(x, kernel_size=M, ceil_mode=True)
    y = F.avg_pool2d(y, kernel_size=M, ceil_mode=True)

    # RGB to LHM
    lhm_weights = _LHM_WEIGHTS.to(x.device).view(3, 3, 1, 1)

    x = F.conv2d(x, lhm_weights)
    y = F.conv2d(y, lhm_weights)

    # Kernel
    if kernel is None:
        kernel = prewitt_kernel()
        kernel = torch.stack([kernel, kernel.t()]).unsqueeze(1)
        kernel = kernel.to(x.device)

    return _mdsi(x, y, kernel, **kwargs)


class MDSI(nn.Module):
    r"""Creates a criterion that measures the MDSI
    between an input and a target.

    Args:
        kernel: A 2D gradient kernel, (2, 1, K, K).
            If `None`, use the Prewitt kernel instead.
        reduction: Specifies the reduction to apply to the output:
            `'none'` | `'mean'` | `'sum'`.

        `**kwargs` are transmitted to `_mdsi`.

    Shape:
        * Input: (N, 3, H, W)
        * Target: (N, 3, H, W)
        * Output: (N,) or (1,) depending on `reduction`

    Example:
        >>> criterion = MDSI().cuda()
        >>> x = torch.rand(5, 3, 256, 256).cuda()
        >>> y = torch.rand(5, 3, 256, 256).cuda()
        >>> l = criterion(x, y)
        >>> l.size()
        torch.Size([])
    """

    def __init__(
        self,
        kernel: torch.Tensor = None,
        reduction: str = 'mean',
        **kwargs,
    ):
        r""""""
        super().__init__()

        if kernel is None:
            kernel = prewitt_kernel()
            kernel = torch.stack([kernel, kernel.t()]).unsqueeze(1)

        self.register_buffer('kernel', kernel)
        self.register_buffer('lhm_weights', _LHM_WEIGHTS.view(3, 3, 1, 1))

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
        _, _, h, w = input.size()
        M = max(1, min(h, w) // 256)

        input = F.avg_pool2d(input, kernel_size=M, ceil_mode=True)
        target = F.avg_pool2d(target, kernel_size=M, ceil_mode=True)

        # RGB to LHM
        input = F.conv2d(input, self.lhm_weights)
        target = F.conv2d(target, self.lhm_weights)

        # MDSI
        l = _mdsi(input, target, self.kernel, **self.kwargs)

        return self.reduce(l)
