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
    r"""Returns the channel-wise convolution of `x` with respect to `kernel`.

    Args:
        x: An input tensor, (N, C, *).
        kernel: A kernel, (C', 1, *).
        padding: The implicit paddings on both sides of the input dimensions.

    Example:
        >>> x = torch.arange(25).float().view(1, 1, 5, 5)
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


def channel_sep_conv(
    x: torch.Tensor,
    kernel: List[torch.Tensor],
    padding: int = 0,  # Union[int, Tuple[int, ...]]
) -> torch.Tensor:
    r"""Returns the channel-wise convolution of `x` with respect to the
    separated `kernel`.

    Args:
        x: An input tensor, (N, C, *).
        kernel: A separated kernel, (C', 1, *).
        padding: The implicit paddings on both sides of the input dimensions.

    Example:
        >>> x = torch.arange(25).float().view(1, 1, 5, 5)
        >>> x
        tensor([[[[ 0.,  1.,  2.,  3.,  4.],
                  [ 5.,  6.,  7.,  8.,  9.],
                  [10., 11., 12., 13., 14.],
                  [15., 16., 17., 18., 19.],
                  [20., 21., 22., 23., 24.]]]])
        >>> kernels = [torch.ones((1, 1, 3, 1)), torch.ones((1, 1, 1, 3))]
        >>> channel_sep_conv(x, kernels)
        tensor([[[[ 54.,  63.,  72.],
                  [ 99., 108., 117.],
                  [144., 153., 162.]]]])
    """

    if padding > 0:
        pad = (padding,) * (2 * len(kernel))
        x = F.pad(x, pad=pad)

    for k in kernel:
        x = channel_conv(x, k)

    return x


def unravel_index(
    indices: torch.LongTensor,
    shape: Tuple[int, ...],
) -> torch.LongTensor:
    r"""Converts flat indices into unraveled coordinates in a target shape.

    This is a `torch` implementation of `numpy.unravel_index`.

    Args:
        indices: A tensor of (flat) indices, (*, N).
        shape: The targeted shape, (D,).

    Returns:
        coord: The unraveled coordinates, (*, N, D).

    Example:
        >>> unravel_index(torch.arange(9), (3, 3))
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
    sigma: float = 1.,
    n: int = 2,
    n_channels: int = 1,
    device: torch.device = torch.device('cpu'),
) -> List[torch.Tensor]:
    r"""Returns the `n`-dimensional separated Gaussian kernel of size `size`.

    The distribution is centered around the kernel's center
    and the standard deviation is `sigma`.

    Args:
        size: The size of the kernel.
        sigma: The standard deviation of the distribution.
        n: The number of dimensions.
        n_channels: The number of channels.
        device: Specifies the return device.

    Wikipedia:
        https://en.wikipedia.org/wiki/Normal_distribution

    Example:
        >>> k = gaussian_kernel(5, sigma=1.5)[0]
        >>> k.size()
        torch.Size([1, 1, 5, 1])
        >>> k.squeeze()
        tensor([0.1201, 0.2339, 0.2921, 0.2339, 0.1201])
    """

    kernel = torch.arange(size, dtype=torch.float, device=device)
    kernel -= (size - 1) / 2
    kernel = kernel ** 2 / (2. * sigma ** 2)
    kernel = torch.exp(-kernel)
    kernel /= kernel.sum()

    kernels = []
    for i in range(n):
        shape = (n_channels, 1) + (1,) * i + (-1,) + (1,) * (n - i - 1)
        kernels.append(
            kernel.repeat(n_channels, 1).view(shape)
        )

    return kernels


def haar_kernel(size: int) -> torch.Tensor:
    r"""Returns the horizontal Haar kernel.

    Args:
        size: The kernel (even) size.

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
    r"""Returns the horizontal 3x3 Prewitt kernel.

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
    r"""Returns the horizontal 3x3 Sobel kernel.

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
    r"""Returns the horizontal 3x3 Scharr kernel.

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


def gradient_kernel(
    kernel: torch.Tensor,
    device: torch.device = torch.device('cpu'),
) -> torch.Tensor:
    r"""Returns `kernel` transformed into a gradient.

    Args:
        kernel: A convolution kernel, (K, K).
        device: Specifies the return device.

    Example:
        >>> g = gradient_kernel(prewitt_kernel())
        >>> g.size()
        torch.Size([2, 1, 3, 3])
    """

    return torch.stack([kernel, kernel.t()]).unsqueeze(1).to(device)


def tensor_norm(
    x: torch.Tensor,
    dim: List[int],  # Union[int, Tuple[int, ...]] = ()
    keepdim: bool = False,
    norm: str = 'L2',
) -> torch.Tensor:
    r"""Returns the norm of `x`.

    Args:
        x: An input tensor.
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

    if norm in ['L2', 'L2_squared']:
        x = x ** 2
    else:  # norm == 'L1'
        x = x.abs()

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
    r"""Returns `x` normalized.

    Args:
        x: An input tensor.
        dim: The dimension(s) along which to normalize.
        norm: Specifies the norm funcion to use:
            `'L1'` | `'L2'` | `'L2_squared'`.
        epsilon: A numerical stability term.

    Example:
        >>> x = torch.arange(9).float().view(3, 3)
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


def cpow(
    x: torch.cfloat,
    exponent: Union[int, float, torch.Tensor],
) -> torch.cfloat:
    r"""Returns the power of `x` with `exponent`.

    Args:
        x: A complex input tensor.
        exponent: The exponent value or tensor.

    Example:
        >>> x = torch.tensor([1. + 0.j, 0.707 + 0.707j])
        >>> cpow(x, 2)
        tensor([ 1.0000e+00+0.0000j, -4.3698e-08+0.9997j])
    """

    r = x.abs() ** exponent
    phi = torch.atan2(x.imag, x.real) * exponent

    return torch.complex(r * torch.cos(phi), r * torch.sin(phi))


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
        >>> red = build_reduce(reduction='sum')
        >>> red(torch.arange(5))
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
