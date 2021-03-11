r"""General purpose tensor functionals
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Tuple  #, Union


def channel_conv(
    x: torch.Tensor,
    kernel: torch.Tensor,
    padding: int = 0,  # Union[int, Tuple[int, ...]]
) -> torch.Tensor:
    r"""Returns the channel-wise convolution of \(x\) with the kernel `kernel`.

    Args:
        x: A tensor, \((N, C, *)\).
        kernel: A kernel, \((C', 1, *)\).
        padding: The implicit paddings on both sides of the input dimensions.

    Example:
        >>> x = torch.arange(25, dtype=torch.float).view(1, 1, 5, 5)
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

    return F.conv1d(x, kernel, padding=padding, groups=x.size(1))


def channel_convs(
    x: torch.Tensor,
    kernels: List[torch.Tensor],
    padding: int = 0,  # Union[int, Tuple[int, ...]]
) -> torch.Tensor:
    r"""Returns the channel-wise convolution of \(x\) with
    the series of kernel `kernels`.

    Args:
        x: A tensor, \((N, C, *)\).
        kernels: A list of kernels, each \((C', 1, *)\).
        padding: The implicit paddings on both sides of the input dimensions.

    Example:
        >>> x = torch.arange(25, dtype=torch.float).view(1, 1, 5, 5)
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
    sigma: float = 1.
) -> torch.Tensor:
    r"""Returns the 1-dimensional Gaussian kernel of size \(K\).

    $$ G(x) = \gamma \exp
        \left(\frac{(x - \mu)^2}{2 \sigma^2}\right) $$

    where \(\gamma\) is such that

    $$ \sum_{x = 1}^{K} G(x) = 1 $$

    and \(\mu = \frac{1 + K}{2}\).

    Args:
        size: The kernel size \(K\).
        sigma: The standard deviation \(\sigma\) of the distribution.

    Returns:
        The kernel vector, \((K,)\).

    Note:
        An \(N\)-dimensional Gaussian kernel is separable, meaning that
        applying it is equivalent to applying a series of \(N\) 1-dimensional
        Gaussian kernels, which has a lower computational complexity.

    Wikipedia:
        https://en.wikipedia.org/wiki/Gaussian_blur

    Example:
        >>> gaussian_kernel(5, sigma=1.5)
        tensor([0.1201, 0.2339, 0.2921, 0.2339, 0.1201])
    """

    kernel = torch.arange(size, dtype=torch.float)
    kernel -= (size - 1) / 2
    kernel = kernel ** 2 / (2. * sigma ** 2)
    kernel = torch.exp(-kernel)
    kernel /= kernel.sum()

    return kernel


def kernel_views(kernel: torch.Tensor, n: int = 2) -> List[torch.Tensor]:
    r"""Returns the \(N\)-dimensional views of the 1-dimensional
    kernel `kernel`.

    Args:
        kernel: A kernel, \((C, 1, K)\).
        n: The number of dimensions \(N\).

    Returns:
        The list of views, each \((C, 1, \underbrace{1, \dots, 1}_{i}, K,
        \underbrace{1, \dots, 1}_{N - i - 1})\).

    Example:
        >>> kernel = gaussian_kernel(5, sigma=1.5).repeat(3, 1, 1)
        >>> kernel.size()
        torch.Size([3, 1, 5])
        >>> views = kernel_views(kernel, n=2)
        >>> views[0].size(), views[1].size()
        (torch.Size([3, 1, 5, 1]), torch.Size([3, 1, 1, 5]))
    """

    if n == 1:
        return [kernel]
    elif n == 2:
        return [kernel.unsqueeze(-1), kernel.unsqueeze(-2)]

    # elif n > 2:
    c, _, k = kernel.size()

    shape: List[int] = [c, 1] + [1] * n
    views = []

    for i in range(2, n + 2):
        shape[i] = k
        views.append(kernel.view(shape))
        shape[i] = 1

    return views


def haar_kernel(size: int) -> torch.Tensor:
    r"""Returns the horizontal Haar kernel.

    Args:
        size: The kernel (even) size \(K\).

    Returns:
        The kernel, \((K, K)\).

    Wikipedia:
        https://en.wikipedia.org/wiki/Haar_wavelet

    Example:
        >>> haar_kernel(2)
        tensor([[ 0.5000, -0.5000],
                [ 0.5000, -0.5000]])
    """

    return torch.outer(
        torch.ones(size) / size,
        torch.tensor([1., -1.]).repeat_interleave(size // 2),
    )


def prewitt_kernel() -> torch.Tensor:
    r"""Returns the Prewitt kernel.

    Returns:
        The kernel, \((3, 3)\).

    Wikipedia:
        https://en.wikipedia.org/wiki/Prewitt_operator

    Example:
        >>> prewitt_kernel()
        tensor([[ 0.3333,  0.0000, -0.3333],
                [ 0.3333,  0.0000, -0.3333],
                [ 0.3333,  0.0000, -0.3333]])
    """

    return torch.outer(
        torch.tensor([1., 1., 1.]) / 3,
        torch.tensor([1., 0., -1.]),
    )


def sobel_kernel() -> torch.Tensor:
    r"""Returns the Sobel kernel.

    Returns:
        The kernel, \((3, 3)\).

    Wikipedia:
        https://en.wikipedia.org/wiki/Sobel_operator

    Example:
        >>> sobel_kernel()
        tensor([[ 0.2500,  0.0000, -0.2500],
                [ 0.5000,  0.0000, -0.5000],
                [ 0.2500,  0.0000, -0.2500]])
    """

    return torch.outer(
        torch.tensor([1., 2., 1.]) / 4,
        torch.tensor([1., 0., -1.]),
    )


def scharr_kernel() -> torch.Tensor:
    r"""Returns the Scharr kernel.

    Returns:
        The kernel, \((3, 3)\).

    Wikipedia:
        https://en.wikipedia.org/wiki/Scharr_operator

    Example:
        >>> scharr_kernel()
        tensor([[ 0.1875,  0.0000, -0.1875],
                [ 0.6250,  0.0000, -0.6250],
                [ 0.1875,  0.0000, -0.1875]])
    """

    return torch.outer(
        torch.tensor([3., 10., 3.]) / 16,
        torch.tensor([1., 0., -1.]),
    )


def gradient_kernel(kernel: torch.Tensor) -> torch.Tensor:
    r"""Returns `kernel` transformed into a gradient.

    Args:
        kernel: A convolution kernel, \((K, K)\).

    Returns:
        The gradient kernel, \((2, 1, K, K)\).

    Example:
        >>> g = gradient_kernel(prewitt_kernel())
        >>> g.size()
        torch.Size([2, 1, 3, 3])
    """

    return torch.stack([kernel, kernel.t()]).unsqueeze(1)
