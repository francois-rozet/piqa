r"""Structural Similarity (SSIM) and Multi-Scale Structural Similarity (MS-SSIM)

This module implements the SSIM and MS-SSIM in PyTorch.

Wikipedia:
    https://en.wikipedia.org/wiki/Structural_similarity

Credits:
    Inspired by [pytorch-msssim](https://github.com/VainF/pytorch-msssim)

References:
    [1] Image quality assessment: From error visibility to structural similarity
    (Wang et al., 2004)
    https://ieeexplore.ieee.org/abstract/document/1284395/

    [2] Multiscale structural similarity for image quality assessment
    (Wang et al., 2004)
    https://ieeexplore.ieee.org/abstract/document/1292216/
"""

__pdoc__ = {'_ssim': True, '_ms_ssim': True}

import torch
import torch.nn as nn
import torch.nn.functional as F

from piqa.utils import _jit, build_reduce, gaussian_kernel, channel_convs

from typing import List, Tuple

_MS_WEIGHTS = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])


@_jit
def _ssim(
    x: torch.Tensor,
    y: torch.Tensor,
    kernel: torch.Tensor,
    value_range: float = 1.,
    non_negative: bool = False,
    k1: float = 0.01,
    k2: float = 0.03,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Returns the channel-wise SSIM and Contrast Sensitivity (CS)
    between \(x\) and \(y\).

    $$ \text{SSIM}(x, y) =
        \frac{2 \mu_x \mu_y + C_1}{\mu^2_x + \mu^2_y + C_1} \text{CS}(x, y) $$

    $$ \text{CS}(x, y) =
        \frac{2 \sigma_{xy} + C_2}{\sigma^2_x + \sigma^2_y + C_2} $$

    where \(\mu_x\), \(\mu_y\), \(\sigma^2_x\), \(\sigma^2_y\) and
    \(\sigma_{xy}\) are the results of a smoothing convolution over
    \(x\), \(y\), \((x - \mu_x)^2\), \((y - \mu_y)^2\) and
    \((x - \mu_x)(y - \mu_y)\), respectively.

    In practice, SSIM and CS are averaged over the image width and height.

    Args:
        x: An input tensor, \((N, C, H, W)\).
        y: A target tensor, \((N, C, H, W)\).
        kernel: A smoothing kernel, \((C, 1, K)\).
            E.g. `piqa.utils.gaussian_kernel`.
        value_range: The value range \(L\) of the inputs (usually 1. or 255).
        non_negative: Whether negative values are clipped or not.

        For the remaining arguments, refer to [1].

    Returns:
        The channel-wise SSIM and CS tensors, both \((N, C)\).

    Example:
        >>> x = torch.rand(5, 3, 256, 256)
        >>> y = torch.rand(5, 3, 256, 256)
        >>> kernel = gaussian_kernel(7).repeat(3, 1, 1)
        >>> ss, cs = _ssim(x, y, kernel)
        >>> ss.size(), cs.size()
        (torch.Size([5, 3]), torch.Size([5, 3]))
    """

    c1 = (k1 * value_range) ** 2
    c2 = (k2 * value_range) ** 2

    window = [kernel.unsqueeze(-1), kernel.unsqueeze(-2)]

    # Mean (mu)
    mu_x = channel_convs(x, window)
    mu_y = channel_convs(y, window)

    mu_xx = mu_x ** 2
    mu_yy = mu_y ** 2
    mu_xy = mu_x * mu_y

    # Variance (sigma)
    sigma_xx = channel_convs(x ** 2, window) - mu_xx
    sigma_yy = channel_convs(y ** 2, window) - mu_yy
    sigma_xy = channel_convs(x * y, window) - mu_xy

    # Contrast sensitivity (CS)
    cs = (2. * sigma_xy + c2) / (sigma_xx + sigma_yy + c2)

    # Structural similarity (SSIM)
    ss = (2. * mu_xy + c1) / (mu_xx + mu_yy + c1) * cs

    # Average
    ss, cs = ss.flatten(2).mean(dim=-1), cs.flatten(2).mean(dim=-1)

    if non_negative:
        ss, cs = torch.relu(ss), torch.relu(cs)

    return ss, cs


@_jit
def _ms_ssim(
    x: torch.Tensor,
    y: torch.Tensor,
    kernel: torch.Tensor,
    weights: torch.Tensor,
    value_range: float = 1.,
    k1: float = 0.01,
    k2: float = 0.03,
) -> torch.Tensor:
    r"""Returns the channel-wise MS-SSIM between \(x\) and \(y\).

    $$ \text{MS-SSIM}(x, y) =
        \text{SSIM}(x^M, y^M)^{\gamma_M} \prod^{M - 1}_{i = 1}
        \text{CS}(x^i, y^i)^{\gamma_i} $$

    where \(x^i\) and \(y^i\) are obtained by downsampling
    the original tensors by a factor \(2^{i - 1}\).

    Args:
        x: An input tensor, \((N, C, H, W)\).
        y: A target tensor, \((N, C, H, W)\).
        kernel: A smoothing kernel, \((C, 1, K)\).
            E.g. `piqa.utils.gaussian_kernel`.
        weights: The weights \(\gamma_i\) of the scales, \((M,)\).
        value_range: The value range \(L\) of the inputs (usually 1. or 255).

        For the remaining arguments, refer to [2].

    Returns:
        The channel-wise MS-SSIM tensor, \((N, C)\).

    Example:
        >>> x = torch.rand(5, 3, 256, 256)
        >>> y = torch.rand(5, 3, 256, 256)
        >>> kernel = gaussian_kernel(7).repeat(3, 1, 1)
        >>> weights = torch.rand(5)
        >>> l = _ms_ssim(x, y, kernel, weights)
        >>> l.size()
        torch.Size([5, 3])
    """

    css = []

    m = weights.numel()
    for i in range(m):
        if i > 0:
            x = F.avg_pool2d(x, kernel_size=2, ceil_mode=True)
            y = F.avg_pool2d(y, kernel_size=2, ceil_mode=True)

        ss, cs = _ssim(
            x, y, kernel,
            value_range=value_range,
            non_negative=True,
            k1=k1, k2=k2,
        )

        css.append(cs if i + 1 < m else ss)

    msss = torch.stack(css, dim=-1)
    msss = (msss ** weights).prod(dim=-1)

    return msss


def ssim(
    x: torch.Tensor,
    y: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    r"""Returns the SSIM between \(x\) and \(y\).

    Args:
        x: An input tensor, \((N, C, H, W)\).
        y: A target tensor, \((N, C, H, W)\).

        `**kwargs` are transmitted to `SSIM`.

    Returns:
        The SSIM vector, \((N,)\).

    Example:
        >>> x = torch.rand(5, 3, 256, 256)
        >>> y = torch.rand(5, 3, 256, 256)
        >>> l = ssim(x, y)
        >>> l.size()
        torch.Size([5])
    """

    kwargs['reduction'] = 'none'

    return SSIM(**kwargs).to(x.device)(x, y)


def ms_ssim(
    x: torch.Tensor,
    y: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    r"""Returns the MS-SSIM between \(x\) and \(y\).

    Args:
        x: An input tensor, \((N, C, H, W)\).
        y: A target tensor, \((N, C, H, W)\).

        `**kwargs` are transmitted to `MS_SSIM`.

    Returns:
        The MS-SSIM vector, \((N,)\).

    Example:
        >>> x = torch.rand(5, 3, 256, 256)
        >>> y = torch.rand(5, 3, 256, 256)
        >>> l = ms_ssim(x, y)
        >>> l.size()
        torch.Size([5])
    """

    kwargs['reduction'] = 'none'

    return MS_SSIM(**kwargs).to(x.device)(x, y)


class SSIM(nn.Module):
    r"""Creates a criterion that measures the SSIM
    between an input and a target.

    Args:
        window_size: The size of the window.
        sigma: The standard deviation of the window.
        n_channels: The number of channels \(C\).
        reduction: Specifies the reduction to apply to the output:
            `'none'` | `'mean'` | `'sum'`.

        `**kwargs` are transmitted to `_ssim`.

    Shapes:
        * Input: \((N, C, H, W)\)
        * Target: \((N, C, H, W)\)
        * Output: \((N,)\) or \(()\) depending on `reduction`

    Example:
        >>> criterion = SSIM().cuda()
        >>> x = torch.rand(5, 3, 256, 256, requires_grad=True).cuda()
        >>> y = torch.rand(5, 3, 256, 256, requires_grad=True).cuda()
        >>> l = criterion(x, y)
        >>> l.size()
        torch.Size([])
    """

    def __init__(
        self,
        window_size: int = 11,
        sigma: float = 1.5,
        n_channels: int = 3,
        reduction: str = 'mean',
        **kwargs,
    ):
        r""""""
        super().__init__()

        kernel = gaussian_kernel(window_size, sigma)

        self.register_buffer('kernel', kernel.repeat(n_channels, 1, 1))

        self.reduce = build_reduce(reduction)
        self.kwargs = kwargs

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        r"""Defines the computation performed at every call.
        """

        l = _ssim(
            input,
            target,
            kernel=self.kernel,
            **self.kwargs,
        )[0].mean(dim=-1)

        return self.reduce(l)


class MS_SSIM(nn.Module):
    r"""Creates a criterion that measures the MS-SSIM
    between an input and a target.

    Args:
        window_size: The size of the window.
        sigma: The standard deviation of the window.
        n_channels: The number of channels \(C\).
        weights: The weights of the scales, \((M,)\).
            If `None`, use the official weights instead.
        reduction: Specifies the reduction to apply to the output:
            `'none'` | `'mean'` | `'sum'`.

        `**kwargs` are transmitted to `_ms_ssim`.

    Shapes:
        * Input: \((N, C, H, W)\)
        * Target: \((N, C, H, W)\)
        * Output: \((N,)\) or \(()\) depending on `reduction`

    Example:
        >>> criterion = MS_SSIM().cuda()
        >>> x = torch.rand(5, 3, 256, 256, requires_grad=True).cuda()
        >>> y = torch.rand(5, 3, 256, 256, requires_grad=True).cuda()
        >>> l = criterion(x, y)
        >>> l.size()
        torch.Size([])
        >>> l.backward()
    """

    def __init__(
        self,
        window_size: int = 11,
        sigma: float = 1.5,
        n_channels: int = 3,
        weights: torch.Tensor = None,
        reduction: str = 'mean',
        **kwargs,
    ):
        r""""""
        super().__init__()

        kernel = gaussian_kernel(window_size, sigma)

        self.register_buffer('kernel', kernel.repeat(n_channels, 1, 1))

        if weights is None:
            weights = _MS_WEIGHTS

        self.register_buffer('weights', weights)

        self.reduce = build_reduce(reduction)
        self.kwargs = kwargs

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        r"""Defines the computation performed at every call.
        """

        l = _ms_ssim(
            input,
            target,
            kernel=self.kernel,
            weights=self.weights,
            **self.kwargs,
        ).mean(dim=-1)

        return self.reduce(l)
