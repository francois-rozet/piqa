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

from spiq.utils import build_reduce, prewitt_kernel, gradient2d, tensor_norm

_LHM_WEIGHTS = torch.FloatTensor([
    [0.2989, 0.587, 0.114],
    [0.3, 0.04, -0.35],
    [0.34, -0.6, 0.17],
])


def mdsi(
    x: torch.Tensor,
    y: torch.Tensor,
    value_range: float = 1.,
    combine: str = 'sum',
    c1: float = 0.00215,  # 140. / (255. ** 2)
    c2: float = 0.00085,  # 55. / (255. ** 2)
    c3: float = 0.00846,  # 550. / (255. ** 2)
    alpha: float = 0.6,
    beta: float = 0.1,
    gamma: float = 0.2,
    rho: float = 1.,
    q: float = 0.25,
    o: float = 0.25,
) -> torch.Tensor:
    r"""Returns the MDSI between `x` and `y`.

    Args:
        x: An input tensor, (N, 3, H, W).
        y: A target tensor, (N, 3, H, W).
        value_range: The value range of the inputs (usually 1. or 255).
        combine: The combination scheme of the gradient similarity (GS) and
            the chromaticity similarity (CS) (`'sum'` or `'prod'`).

        For the remaining arguments, refer to [1].
    """

    _, _, h, w = x.size()

    # Downsample
    M = max(1, min(h, w) // 256)
    padding = (0, M - (w - 1 % M) + 1, 0, M - (h - 1 % M) + 1)

    if sum(padding) > 0:
        x = F.pad(x, pad=padding)
        y = F.pad(y, pad=padding)

    x = F.avg_pool2d(x, kernel_size=M)
    y = F.avg_pool2d(y, kernel_size=M)

    # RGB to LHM
    lhm_weights = _LHM_WEIGHTS.to(x.device).view(3, 3, 1, 1)
    lhm_weights /= value_range

    x = F.conv2d(x, lhm_weights)
    y = F.conv2d(y, lhm_weights)

    # Gradient magnitude
    kernel = prewitt_kernel()
    kernel = torch.stack([kernel, kernel.t()]).unsqueeze(1).to(x.device)

    gm_x = tensor_norm(gradient2d(x[:, :1], kernel), dim=1)
    gm_y = tensor_norm(gradient2d(y[:, :1], kernel), dim=1)
    gm_avg = tensor_norm(gradient2d((x + y)[:, :1] / 2., kernel), dim=1)

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

    if combine == 'prod':
        gcs = (gs ** gamma) * (cs ** beta)
    else:  # combine == 'sum'
        gcs = alpha * gs + (1. - alpha) * cs

    # Mean deviation similarity
    gcs_q = gcs ** q
    score = (gcs_q - gcs_q.mean((-1, -2), keepdim=True)).abs()
    mds = (score ** rho).mean((-1, -2)) ** (o / rho)

    return mds


class MDSI(nn.Module):
    r"""Creates a criterion that measures the MDSI
    between an input and a target.

    Args:
        reduction: A reduction type (`'mean'`, `'sum'` or `'none'`).

        `**kwargs` are transmitted to `mdsi`.

    Call:
        The input and target tensors should be of shape (N, 3, H, W).
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
        l = mdsi(input, target, **self.kwargs)

        return self.reduce(l)
