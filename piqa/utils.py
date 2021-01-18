r"""Miscellaneous tools such as modules, functionals and more.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Tuple, Union


_jit = torch.jit.script


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
        The list of views, each \((C, 1, 1^i, K, 1^{N - i - 1})\).

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


def cstack(real: torch.Tensor, imag: torch.Tensor) -> torch.Tensor:
    r"""Returns a complex tensor with its real part equal to \(\Re\) and
    its imaginary part equal to \(\Im\).

    $$ c = \Re + i \Im $$

    Args:
        real: A tensor \(\Re\), \((*,)\).
        imag: A tensor \(\Im\), \((*,)\).

    Returns:
        The complex tensor, \((*, 2)\).

    Example:
        >>> x = torch.tensor([1., 0.707])
        >>> y = torch.tensor([0., 0.707])
        >>> cstack(x, y)
        tensor([[1.0000, 0.0000],
                [0.7070, 0.7070]])
    """

    return torch.stack([real, imag], dim=-1)


def cabs(x: torch.Tensor, squared: bool = False) -> torch.Tensor:
    r"""Returns the absolute value (modulus) of \(x\).

    $$ \left| x \right| = \sqrt{ \Re(x)^2 + \Im(x)^2 } $$

    Args:
        x: A complex tensor, \((*, 2)\).
        squared: Whether the output is squared or not.

    Returns:
        The absolute value tensor, \((*,)\).

    Example:
        >>> x = torch.tensor([[1., 0.], [0.707, 0.707]])
        >>> cabs(x)
        tensor([1.0000, 0.9998])
    """

    x = (x ** 2).sum(dim=-1)

    if not squared:
        x = torch.sqrt(x)

    return x


def cangle(x: torch.Tensor) -> torch.Tensor:
    r"""Returns the angle (phase) of \(x\).

    $$ \phi(x) = \operatorname{atan2}(\Im(x), \Re(x)) $$

    Args:
        x: A complex tensor, \((*, 2)\).

    Returns:
        The angle tensor, \((*,)\).

    Example:
        >>> x = torch.tensor([[1., 0.], [0.707, 0.707]])
        >>> cangle(x)
        tensor([0.0000, 0.7854])
    """

    return torch.atan2(x[..., 1], x[..., 0])


def cprod(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    r"""Returns the product of \(x\) and \(y\).

    $$ x y = \Re(x) \Re(y) - \Im(x) \Im(y)
        + i \left( \Re(x) \Im(y) - \Im(x) \Re(y) \right) $$

    Args:
        x: A complex tensor, \((*, 2)\).
        y: A complex tensor, \((*, 2)\).

    Returns:
        The product tensor, \((*, 2)\).

    Example:
        >>> x = torch.tensor([[1.,  0.], [0.707,  0.707]])
        >>> y = torch.tensor([[1., -0.], [0.707, -0.707]])
        >>> cprod(x, y)
        tensor([[1.0000, 0.0000],
                [0.9997, 0.0000]])
    """

    x_r, x_i = x[..., 0], x[..., 1]
    y_r, y_i = y[..., 0], y[..., 1]

    return cstack(
        x_r * y_r - x_i * y_i,
        x_i * y_r + x_r * y_i,
    )


def cpow(x: torch.Tensor, exponent: float) -> torch.Tensor:
    r"""Returns the power of \(x\) with `exponent`.

    $$ x^p = \left| x \right|^p \exp(i \phi(x))^p $$

    Args:
        x: A complex tensor, \((*, 2)\).
        exponent: The exponent \(p\).

    Returns:
        The power tensor, \((*, 2)\).

    Example:
        >>> x = torch.tensor([[1., 0.], [0.707, 0.707]])
        >>> cpow(x, 2.)
        tensor([[ 1.0000e+00,  0.0000e+00],
                [-4.3698e-08,  9.9970e-01]])
    """

    r = cabs(x, squared=True) ** (exponent / 2)
    phi = cangle(x) * exponent

    return cstack(r * torch.cos(phi), r * torch.sin(phi))


class Intermediary(nn.Module):
    r"""Module that catches and returns the outputs of indermediate
    target layers of a sequential module during its forward pass.

    Args:
        layers: A sequential module.
        targets: A list of target layer indexes.
    """

    def __init__(self, layers: nn.Sequential, targets: List[int]):
        r""""""
        super().__init__()

        self.layers = nn.ModuleList()
        j = 0

        seq: List[nn.Module] = []

        for i, layer in enumerate(layers):
            seq.append(layer)

            if i == targets[j]:
                self.layers.append(nn.Sequential(*seq))
                seq.clear()

                j += 1
                if j == len(targets):
                    break

    def forward(self, input: torch.Tensor) -> List[torch.Tensor]:
        r"""Defines the computation performed at every call.
        """

        output = []

        for layer in self.layers:
            input = layer(input)
            output.append(input)

        return output


def build_reduce(reduction: str = 'mean') -> nn.Module:
    r"""Returns a reducing module.

    Args:
        reduction: Specifies the reduce type:
            `'none'` | `'mean'` | `'sum'`.

    Example:
        >>> r = build_reduce(reduction='sum')
        >>> r(torch.arange(5))
        tensor(10)
    """

    if reduction == 'mean':
        return _Mean()
    elif reduction == 'sum':
        return _Sum()

    return nn.Identity()


class _Mean(nn.Module):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input.mean()


class _Sum(nn.Module):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input.sum()
