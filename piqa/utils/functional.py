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

    $$ G(x) = \frac{1}{\sum_{y = 1}^{K} G(y)} \exp
        \left(\frac{(x - \mu)^2}{2 \sigma^2}\right) $$

    where \(x \in [1; K]\) is a position in the kernel
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


def tensor_norm(
    x: torch.Tensor,
    dim: List[int],  # Union[int, Tuple[int, ...]] = ()
    keepdim: bool = False,
    norm: str = 'L2',
) -> torch.Tensor:
    r"""Returns the norm of \(x\).

    $$ L_1(x) = \left\| x \right\|_1 = \sum_i \left| x_i \right| $$

    $$ L_2(x) = \left\| x \right\|_2 = \left( \sum_i x^2_i \right)^\frac{1}{2} $$

    Args:
        x: A tensor, \((*,)\).
        dim: The dimension(s) along which to calculate the norm.
        keepdim: Whether the output tensor has `dim` retained or not.
        norm: Specifies the norm funcion to apply:
            `'L1'` | `'L2'` | `'L2_squared'`.

    Wikipedia:
        https://en.wikipedia.org/wiki/Norm_(mathematics)

    Example:
        >>> x = torch.arange(9).float().view(3, 3)
        >>> x
        tensor([[0., 1., 2.],
                [3., 4., 5.],
                [6., 7., 8.]])
        >>> tensor_norm(x, dim=0)
        tensor([6.7082, 8.1240, 9.6437])
    """

    if norm == 'L1':
        x = x.abs()
    else:  # norm in ['L2', 'L2_squared']
        x = x ** 2

    x = x.sum(dim=dim, keepdim=keepdim)

    if norm == 'L2':
        x = x.sqrt()

    return x


def normalize_tensor(
    x: torch.Tensor,
    dim: List[int],  # Union[int, Tuple[int, ...]] = ()
    norm: str = 'L2',
    epsilon: float = 1e-8,
) -> torch.Tensor:
    r"""Returns \(x\) normalized.

    $$ \hat{x} = \frac{x}{\left\|x\right\|} $$

    Args:
        x: A tensor, \((*,)\).
        dim: The dimension(s) along which to normalize.
        norm: Specifies the norm funcion to use:
            `'L1'` | `'L2'` | `'L2_squared'`.
        epsilon: A numerical stability term.

    Returns:
        The normalized tensor, \((*,)\).

    Example:
        >>> x = torch.arange(9, dtype=torch.float).view(3, 3)
        >>> x
        tensor([[0., 1., 2.],
                [3., 4., 5.],
                [6., 7., 8.]])
        >>> normalize_tensor(x, dim=0)
        tensor([[0.0000, 0.1231, 0.2074],
                [0.4472, 0.4924, 0.5185],
                [0.8944, 0.8616, 0.8296]])
    """

    norm = tensor_norm(x, dim=dim, keepdim=True, norm=norm)

    return x / (norm + epsilon)


def unravel_index(
    indices: torch.LongTensor,
    shape: List[int],
) -> torch.LongTensor:
    r"""Converts flat indices into unraveled coordinates in a target shape.

    This is a `torch` implementation of `numpy.unravel_index`.

    Args:
        indices: A tensor of (flat) indices, \((*, N)\).
        shape: The targeted shape, \((D,)\).

    Returns:
        The unraveled coordinates, \((*, N, D)\).

    Example:
        >>> unravel_index(torch.arange(9), shape=(3, 3))
        tensor([[0, 0],
                [0, 1],
                [0, 2],
                [1, 0],
                [1, 1],
                [1, 2],
                [2, 0],
                [2, 1],
                [2, 2]])
    """

    coord = []

    for dim in reversed(shape):
        coord.append(indices % dim)
        indices = indices // dim

    coord = torch.stack(coord[::-1], dim=-1)

    return coord
