r"""General purpose tensor functionals"""

import math
import torch
import torch.fft as fft
import torch.nn.functional as F

from torch import Tensor
from typing import *


def channel_conv(
    x: Tensor,
    kernel: Tensor,
    padding: int = 0,
) -> Tensor:
    r"""Returns the channel-wise convolution of :math:`x` with the kernel `kernel`.

    Args:
        x: A tensor, :math:`(N, C, *)`.
        kernel: A kernel, :math:`(C', 1, *)`.
        padding: The implicit paddings on both sides of the input dimensions.

    Example:
        >>> x = torch.arange(25).float().reshape(1, 1, 5, 5)
        >>> x
        tensor([[[[ 0.,  1.,  2.,  3.,  4.],
                  [ 5.,  6.,  7.,  8.,  9.],
                  [10., 11., 12., 13., 14.],
                  [15., 16., 17., 18., 19.],
                  [20., 21., 22., 23., 24.]]]])
        >>> kernel = torch.ones((1, 1, 3, 3))
        >>> channel_conv(x, kernel)
        tensor([[[[ 54.,  63.,  72.],
                  [ 99., 108., 117.],
                  [144., 153., 162.]]]])
    """

    D = kernel.dim() - 2

    assert D <= 3, "PyTorch only supports 1D, 2D or 3D convolutions."

    if D == 3:
        return F.conv3d(x, kernel, padding=padding, groups=x.shape[-4])
    elif D == 2:
        return F.conv2d(x, kernel, padding=padding, groups=x.shape[-3])
    elif D == 1:
        return F.conv1d(x, kernel, padding=padding, groups=x.shape[-2])
    else:
        return F.linear(x, kernel.expand(x.shape[-1]))


def channel_convs(
    x: Tensor,
    kernels: List[Tensor],
    padding: int = 0,
) -> Tensor:
    r"""Returns the channel-wise convolution of :math:`x` with the series of
    kernel `kernels`.

    Args:
        x: A tensor, :math:`(N, C, *)`.
        kernels: A list of kernels, each :math:`(C', 1, *)`.
        padding: The implicit paddings on both sides of the input dimensions.

    Example:
        >>> x = torch.arange(25).float().reshape(1, 1, 5, 5)
        >>> x
        tensor([[[[ 0.,  1.,  2.,  3.,  4.],
                  [ 5.,  6.,  7.,  8.,  9.],
                  [10., 11., 12., 13., 14.],
                  [15., 16., 17., 18., 19.],
                  [20., 21., 22., 23., 24.]]]])
        >>> kernels = [torch.ones((1, 1, 3, 1)), torch.ones((1, 1, 1, 3))]
        >>> channel_convs(x, kernels)
        tensor([[[[ 54.,  63.,  72.],
                  [ 99., 108., 117.],
                  [144., 153., 162.]]]])
    """

    if padding > 0:
        pad = (padding,) * (2 * x.dim() - 4)
        x = F.pad(x, pad=pad)

    for k in kernels:
        x = channel_conv(x, k)

    return x


def gaussian_kernel(
    size: int,
    sigma: float = 1.0,
) -> Tensor:
    r"""Returns the 1-dimensional Gaussian kernel of size :math:`K`.

    .. math::
        G(x) = \gamma \exp \left(\frac{(x - \mu)^2}{2 \sigma^2}\right)

    where :math:`\gamma` is such that

    .. math:: \sum_{x = 1}^{K} G(x) = 1

    and :math:`\mu = \frac{1 + K}{2}`.

    Wikipedia:
        https://wikipedia.org/wiki/Gaussian_blur

    Note:
        An :math:`N`-dimensional Gaussian kernel is separable, meaning that
        applying it is equivalent to applying a series of :math:`N` 1-dimensional
        Gaussian kernels, which has a lower computational complexity.

    Args:
        size: The kernel size :math:`K`.
        sigma: The standard deviation :math:`\sigma` of the distribution.

    Returns:
        The kernel vector, :math:`(K,)`.

    Example:
        >>> gaussian_kernel(5, sigma=1.5)
        tensor([0.1201, 0.2339, 0.2921, 0.2339, 0.1201])
    """

    kernel = torch.arange(size, dtype=torch.float)
    kernel -= (size - 1) / 2
    kernel = kernel ** 2 / (2 * sigma ** 2)
    kernel = torch.exp(-kernel)
    kernel /= kernel.sum()

    return kernel


def kernel_views(kernel: Tensor, n: int = 2) -> List[Tensor]:
    r"""Returns the :math:`N`-dimensional views of the 1-dimensional kernel `kernel`.

    Args:
        kernel: A kernel, :math:`(C, 1, K)`.
        n: The number of dimensions :math:`N`.

    Returns:
        The list of views, each :math:`(C, 1, \underbrace{1, \dots, 1}_{i}, K, \underbrace{1, \dots, 1}_{N - i - 1})`.

    Example:
        >>> kernel = gaussian_kernel(5, sigma=1.5).repeat(3, 1, 1)
        >>> kernel.shape
        torch.Size([3, 1, 5])
        >>> views = kernel_views(kernel, n=2)
        >>> views[0].shape, views[1].shape
        (torch.Size([3, 1, 5, 1]), torch.Size([3, 1, 1, 5]))
    """

    if n == 1:
        return [kernel]
    elif n == 2:
        return [kernel.unsqueeze(-1), kernel.unsqueeze(-2)]

    # elif n > 2:
    c, _, k = kernel.shape

    shape: List[int] = [c, 1] + [1] * n
    views = []

    for i in range(2, n + 2):
        shape[i] = k
        views.append(kernel.reshape(shape))
        shape[i] = 1

    return views


def haar_kernel(size: int) -> Tensor:
    r"""Returns the horizontal Haar kernel.

    Wikipedia:
        https://wikipedia.org/wiki/Haar_wavelet

    Args:
        size: The kernel (even) size :math:`K`.

    Returns:
        The kernel, :math:`(K, K)`.

    Example:
        >>> haar_kernel(2)
        tensor([[ 0.5000, -0.5000],
                [ 0.5000, -0.5000]])
    """

    return torch.outer(
        torch.ones(size) / size,
        torch.tensor([1.0, -1.0]).repeat_interleave(size // 2),
    )


def prewitt_kernel() -> Tensor:
    r"""Returns the Prewitt kernel.

    Wikipedia:
        https://wikipedia.org/wiki/Prewitt_operator

    Returns:
        The kernel, :math:`(3, 3)`.

    Example:
        >>> prewitt_kernel()
        tensor([[ 0.3333,  0.0000, -0.3333],
                [ 0.3333,  0.0000, -0.3333],
                [ 0.3333,  0.0000, -0.3333]])
    """

    return torch.outer(
        torch.tensor([1.0, 1.0, 1.0]) / 3,
        torch.tensor([1.0, 0.0, -1.0]),
    )


def sobel_kernel() -> Tensor:
    r"""Returns the Sobel kernel.

    Wikipedia:
        https://wikipedia.org/wiki/Sobel_operator

    Returns:
        The kernel, :math:`(3, 3)`.

    Example:
        >>> sobel_kernel()
        tensor([[ 0.2500,  0.0000, -0.2500],
                [ 0.5000,  0.0000, -0.5000],
                [ 0.2500,  0.0000, -0.2500]])
    """

    return torch.outer(
        torch.tensor([1.0, 2.0, 1.0]) / 4,
        torch.tensor([1.0, 0.0, -1.0]),
    )


def scharr_kernel() -> Tensor:
    r"""Returns the Scharr kernel.

    Wikipedia:
        https://wikipedia.org/wiki/Scharr_operator

    Returns:
        The kernel, :math:`(3, 3)`.

    Example:
        >>> scharr_kernel()
        tensor([[ 0.1875,  0.0000, -0.1875],
                [ 0.6250,  0.0000, -0.6250],
                [ 0.1875,  0.0000, -0.1875]])
    """

    return torch.outer(
        torch.tensor([3.0, 10.0, 3.0]) / 16,
        torch.tensor([1.0, 0.0, -1.0]),
    )


def gradient_kernel(kernel: Tensor) -> Tensor:
    r"""Returns `kernel` transformed into a gradient.

    Args:
        kernel: A convolution kernel, :math:`(K, K)`.

    Returns:
        The gradient kernel, :math:`(2, 1, K, K)`.

    Example:
        >>> g = gradient_kernel(prewitt_kernel())
        >>> g.shape
        torch.Size([2, 1, 3, 3])
    """

    return torch.stack((kernel, kernel.t())).unsqueeze(1)


def filter_grid(x: Tensor) -> Tuple[Tensor, Tensor]:
    r"""Returns the (quadrant-shifted) frequency grid for :math:`x`.

    Args:
        x: A tensor, :math:`(*, H, W)`.

    Returns:
        The radius and phase tensors, both :math:`(H, W)`.

    Example:
        >>> x = torch.rand(5, 5)
        >>> r, phi = filter_grid(x)
        >>> r
        tensor([[0.0000, 0.2500, 0.5000, 0.5000, 0.2500],
                [0.2500, 0.3536, 0.5590, 0.5590, 0.3536],
                [0.5000, 0.5590, 0.7071, 0.7071, 0.5590],
                [0.5000, 0.5590, 0.7071, 0.7071, 0.5590],
                [0.2500, 0.3536, 0.5590, 0.5590, 0.3536]])
        >>> phi
        tensor([[-0.0000, -1.5708, -1.5708,  1.5708,  1.5708],
                [-0.0000, -0.7854, -1.1071,  1.1071,  0.7854],
                [-0.0000, -0.4636, -0.7854,  0.7854,  0.4636],
                [-3.1416, -2.6779, -2.3562,  2.3562,  2.6779],
                [-3.1416, -2.3562, -2.0344,  2.0344,  2.3562]])
    """

    H, W = x.shape[-2:]

    u = (torch.arange(H).to(x) - H // 2) / (H - H % 2)
    v = (torch.arange(W).to(x) - W // 2) / (W - W % 2)

    u = fft.ifftshift(u)[:, None]
    v = fft.ifftshift(v)[None, :]

    r = torch.sqrt(u ** 2 + v ** 2)
    phi = torch.atan2(-v, u)

    return r, phi


def log_gabor(f: Tensor, f_0: float, sigma_f: float) -> Tensor:
    r"""Returns the log-Gabor filter of :math:`f`.

    .. math::
        G(f) = \exp \left( - \frac{\log(f / f_0)^2}{2 \sigma_f^2} \right)

    Wikipedia:
        https://wikipedia.org/wiki/Log_Gabor_filter

    Args:
        f: A frequency tensor, :math:`(*,)`.
        f_0: The center frequency :math:`f_0`.
        sigma_f: The bandwidth (log-)deviation :math:`\sigma_f`.

    Returns:
        The filter tensor, :math:`(*,)`.

    Example:
        >>> x = torch.rand(5, 5)
        >>> r, phi = filter_grid(x)
        >>> log_gabor(r, 1.0, 1.0)
        tensor([[0.0000, 0.3825, 0.7864, 0.7864, 0.3825],
                [0.3825, 0.5825, 0.8444, 0.8444, 0.5825],
                [0.7864, 0.8444, 0.9417, 0.9417, 0.8444],
                [0.7864, 0.8444, 0.9417, 0.9417, 0.8444],
                [0.3825, 0.5825, 0.8444, 0.8444, 0.5825]])
    """

    return torch.exp(-(f / f_0).log() ** 2 / (2 * sigma_f ** 2))


def l2_norm(
    x: torch.Tensor,
    dim: int,
    keepdim: bool = False,
) -> torch.Tensor:
    r"""Returns the :math:`L_2` norm of :math:`x`.

    .. math:
        L_2(x) = \left\| x \right\|_2 = \sqrt{\sum_i x^2_i}

    Wikipedia:
        https://wikipedia.org/wiki/Norm_(mathematics)

    Args:
        x: A tensor, :math:`(*,)`.
        dim: The dimension along which to calculate the norm.
        keepdim: Whether the output tensor has `dim` retained or not.

    Example:
        >>> x = torch.arange(9).float().reshape(3, 3)
        >>> x
        tensor([[0., 1., 2.],
                [3., 4., 5.],
                [6., 7., 8.]])
        >>> l2_norm(x, dim=0)
        tensor([6.7082, 8.1240, 9.6437])
    """

    x = x.square()
    x = x.sum(dim=dim, keepdim=keepdim)
    x = x.sqrt()

    return x


@torch.jit.script_if_tracing
def downsample(x: Tensor, resolution: int = 256):
    r"""Downsamples :math:`x` to a target resolution.

    Args:
        x: A tensor, :math:`(N, C, H, W)`.
        resolution: The target resolution :math:`R`.

    Returns:
        The downsampled tensor, :math:`(N, C, H / f, W / f)` where :math:`f = \lfloor
        \frac{\min(H, W)}{R} \rfloor`.

    Example:
        >>> x = torch.rand(5, 3, 1920, 1080)
        >>> x = downsample(x, resolution=256)
        >>> x.shape
        torch.Size([5, 3, 480, 270])
    """

    N, C, H, W = x.shape

    factor = math.floor(min(H, W) / resolution)

    if factor > 1:
        return F.avg_pool2d(x, kernel_size=factor, ceil_mode=True)
    else:
        return x


@torch.jit.script_if_tracing
def reduce_tensor(x: Tensor, reduction: str = 'mean') -> Tensor:
    r"""Returns the reduction of :math:`x`.

    Args:
        x: A tensor, :math:`(*,)`.
        reduction: Specifies the reduction type:
            `'none'`, `'mean'` or `'sum'`.

    Example:
        >>> x = torch.arange(5)
        >>> reduce_tensor(x, reduction='sum')
        tensor(10)
    """

    if reduction == 'mean':
        return x.mean()
    elif reduction == 'sum':
        return x.sum()

    return x
